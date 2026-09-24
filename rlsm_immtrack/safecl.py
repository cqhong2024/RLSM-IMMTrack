"""Safe closed-loop IMM baseline; preserved from the experiment implementation."""
from __future__ import annotations
import math
import time
import cv2
import numpy as np
import torch
from .imm import BoxIMM, box_from_state, measurement_from_box

def clip_box(box, image_width: int, image_height: int, margin: float = 2.0):
    x, y, w, h = map(float, box)
    w = min(max(w, margin), image_width)
    h = min(max(h, margin), image_height)
    x = min(max(x, 0.0), max(0.0, image_width - w))
    y = min(max(y, 0.0), max(0.0, image_height - h))
    return np.array([x, y, w, h], dtype=np.float64)


def fuse_covariance(states, covariances, probabilities):
    mean = sum(probabilities[index] * states[index] for index in range(3))
    covariance = np.zeros((8, 8), dtype=np.float64)
    for index in range(3):
        delta = states[index] - mean
        covariance += probabilities[index] * (
            covariances[index] + np.outer(delta, delta)
        )
    return mean, covariance


class AdaptiveBoxIMM(BoxIMM):
    """Split predict/correct IMM with confidence-controlled measurement noise."""

    def predict(self, camera_affine=None, camera_quality: float = 0.0):
        normalization, mixed_states, mixed_covariances = self._mix()
        normalization /= normalization.sum()
        self.states = []
        self.covariances = []
        for model in range(3):
            transition_matrix = self.transitions[model]
            self.states.append(transition_matrix @ mixed_states[model])
            self.covariances.append(
                transition_matrix
                @ mixed_covariances[model]
                @ transition_matrix.T
                + self.process_noises[model]
            )
        self.probabilities = normalization
        if camera_affine is not None:
            self._apply_camera(camera_affine, camera_quality)
        fused, covariance = fuse_covariance(
            self.states, self.covariances, self.probabilities
        )
        return (
            box_from_state(fused, self.image_width, self.image_height),
            covariance,
        )

    def _apply_camera(self, affine: np.ndarray, quality: float):
        linear = affine[:, :2]
        translation = affine[:, 2]
        scale = math.sqrt(max(abs(np.linalg.det(linear)), 1e-8))
        for state in self.states:
            center_pixels = np.array(
                [state[0] * self.image_width, state[1] * self.image_height]
            )
            velocity_pixels = np.array(
                [state[2] * self.image_width, state[3] * self.image_height]
            )
            center_pixels = linear @ center_pixels + translation
            velocity_pixels = linear @ velocity_pixels
            state[0] = center_pixels[0] / self.image_width
            state[1] = center_pixels[1] / self.image_height
            state[2] = velocity_pixels[0] / self.image_width
            state[3] = velocity_pixels[1] / self.image_height
            state[4] += math.log(scale)
            state[5] += math.log(scale)
        camera_sigma = 0.0015 + (1.0 - quality) * 0.004
        for covariance in self.covariances:
            covariance[0, 0] += camera_sigma**2
            covariance[1, 1] += camera_sigma**2

    def correct(
        self,
        measurement_box: np.ndarray,
        quality: float,
        fixed_measurement_noise: bool = False,
    ):
        if quality < 0.10:
            fused, covariance = fuse_covariance(
                self.states, self.covariances, self.probabilities
            )
            return (
                box_from_state(fused, self.image_width, self.image_height),
                covariance,
                False,
            )

        measurement = measurement_from_box(
            measurement_box, self.image_width, self.image_height
        )
        if fixed_measurement_noise:
            center_sigma, size_sigma = 0.012, 0.080
        else:
            # A confident modern tracker is more accurate than a motion model.
            # The fourth-power schedule makes high-confidence corrections
            # effectively measurement-dominated while retaining strong
            # smoothing/gating when confidence collapses.
            center_sigma = 0.00020 + (1.0 - quality) ** 4 * 0.030
            size_sigma = 0.0030 + (1.0 - quality) ** 4 * 0.180
        measurement_noise = np.diag(
            [center_sigma**2, center_sigma**2, size_sigma**2, size_sigma**2]
        )

        updated_states = []
        updated_covariances = []
        log_probabilities = []
        identity = np.eye(8)
        for model in range(3):
            state = self.states[model]
            covariance = self.covariances[model]
            residual = measurement - self.measurement_matrix @ state
            innovation_covariance = (
                self.measurement_matrix
                @ covariance
                @ self.measurement_matrix.T
                + measurement_noise
            )
            gain = (
                covariance
                @ self.measurement_matrix.T
                @ np.linalg.inv(innovation_covariance)
            )
            updated_state = state + gain @ residual
            correction = identity - gain @ self.measurement_matrix
            updated_covariance = (
                correction @ covariance @ correction.T
                + gain @ measurement_noise @ gain.T
            )
            updated_states.append(updated_state)
            updated_covariances.append(updated_covariance)
            log_probabilities.append(
                math.log(max(self.probabilities[model], 1e-300))
                + self._log_likelihood(residual, innovation_covariance)
            )

        log_probabilities = np.asarray(log_probabilities)
        log_probabilities -= log_probabilities.max()
        probabilities = np.exp(log_probabilities)
        self.probabilities = probabilities / probabilities.sum()
        self.states = updated_states
        self.covariances = updated_covariances
        fused, covariance = fuse_covariance(
            self.states, self.covariances, self.probabilities
        )
        return (
            box_from_state(fused, self.image_width, self.image_height),
            covariance,
            True,
        )


class CameraMotionEstimator:
    def __init__(self, max_side: int = 640):
        self.max_side = max_side

    def _gray_small(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, self.max_side / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        return gray, scale

    def estimate(self, previous: np.ndarray, current: np.ndarray):
        previous_gray, scale = self._gray_small(previous)
        current_gray, _ = self._gray_small(current)
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=350,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if points is None or len(points) < 12:
            return np.eye(2, 3, dtype=np.float64), 0.0, 0
        tracked, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
        )
        valid = status.reshape(-1).astype(bool)
        if error is not None:
            valid &= error.reshape(-1) < 30.0
        source = points.reshape(-1, 2)[valid]
        destination = tracked.reshape(-1, 2)[valid]
        if len(source) < 10:
            return np.eye(2, 3, dtype=np.float64), 0.0, len(source)
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1000,
            confidence=0.99,
        )
        if affine is None or inliers is None:
            return np.eye(2, 3, dtype=np.float64), 0.0, len(source)
        inlier_count = int(inliers.sum())
        quality = inlier_count / max(len(source), 1)
        affine = affine.astype(np.float64)
        affine[:, 2] /= scale
        determinant = np.linalg.det(affine[:, :2])
        scale_factor = math.sqrt(max(abs(determinant), 1e-8))
        if (
            quality < 0.25
            or not 0.85 <= scale_factor <= 1.18
            or np.linalg.norm(affine[:, 2]) > max(current.shape[:2]) * 0.25
        ):
            return np.eye(2, 3, dtype=np.float64), 0.0, inlier_count
        return affine, float(quality), inlier_count


class AppearanceMemory:
    def __init__(self, image: np.ndarray, box: np.ndarray):
        patch = self.extract(image, box)
        self.initial = patch
        self.recent = patch.copy()

    @staticmethod
    def extract(image: np.ndarray, box: np.ndarray, output_size: int = 64):
        height, width = image.shape[:2]
        x, y, w, h = map(float, box)
        x1 = int(np.clip(math.floor(x), 0, width - 1))
        y1 = int(np.clip(math.floor(y), 0, height - 1))
        x2 = int(np.clip(math.ceil(x + w), x1 + 1, width))
        y2 = int(np.clip(math.ceil(y + h), y1 + 1, height))
        patch = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        patch = cv2.resize(
            patch, (output_size, output_size), interpolation=cv2.INTER_AREA
        )
        return patch.astype(np.float32)

    @staticmethod
    def similarity(first: np.ndarray, second: np.ndarray):
        first = first - first.mean()
        second = second - second.mean()
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1e-6:
            return 0.0
        return float(np.clip((first * second).sum() / denominator, -1.0, 1.0))

    def score(self, image: np.ndarray, box: np.ndarray):
        patch = self.extract(image, box)
        score = max(
            self.similarity(self.initial, patch),
            self.similarity(self.recent, patch),
        )
        return float(np.clip(0.5 * (score + 1.0), 0.0, 1.0))

    def update(self, image: np.ndarray, box: np.ndarray):
        self.recent = self.extract(image, box)


class ClosedLoopTracker:
    def __init__(
        self,
        tracker,
        initial_image: np.ndarray,
        initial_box: np.ndarray,
        disable_camera: bool,
        disable_redetection: bool,
        fixed_measurement_noise: bool,
    ):
        height, width = initial_image.shape[:2]
        self.tracker = tracker
        self.width = width
        self.height = height
        self.imm = AdaptiveBoxIMM(initial_box, width, height, robust=False)
        self.camera = CameraMotionEstimator()
        self.appearance = AppearanceMemory(initial_image, initial_box)
        self.previous_image = initial_image
        self.last_box = np.asarray(initial_box, dtype=np.float64)
        self.last_measurement = np.asarray(initial_box, dtype=np.float64)
        self.safety_measurement = np.asarray(initial_box, dtype=np.float64)
        self.last_quality = 1.0
        self.lost_count = 0
        self.frame_id = 0
        self.disable_camera = disable_camera
        self.disable_redetection = disable_redetection
        self.fixed_measurement_noise = fixed_measurement_noise
        self.redetection_count = 0
        self.expanded_search_count = 0
        self.skipped_update_count = 0
        self.camera_valid_count = 0

    def visual_probe(self, image: np.ndarray, anchor: np.ndarray, search_factor: float):
        from lib.test.tracker.utils import sample_target
        from lib.utils.box_ops import clip_box as sutrack_clip_box

        search_patch, resize_factor = sample_target(
            image,
            anchor.tolist(),
            search_factor,
            output_sz=self.tracker.params.search_size,
        )
        search = self.tracker.preprocessor.process(search_patch)
        if self.tracker.multi_modal_vision and search.size(1) == 3:
            search = torch.cat((search, search), axis=1)
        with torch.no_grad():
            encoded = self.tracker.network.forward_encoder(
                self.tracker.template_list,
                [search],
                self.tracker.template_anno_list,
                self.tracker.text_src,
                self.tracker.task_index_batch,
            )
            output = self.tracker.network.forward_decoder(feature=encoded)
        score_map = output["score_map"]
        response = (
            self.tracker.output_window * score_map
            if self.tracker.cfg.TEST.WINDOW
            else score_map
        )
        if "size_map" in output:
            boxes, confidence = self.tracker.network.decoder.cal_bbox(
                response,
                output["size_map"],
                output["offset_map"],
                return_score=True,
            )
        else:
            boxes, confidence = self.tracker.network.decoder.cal_bbox(
                response, output["offset_map"], return_score=True
            )
        boxes = boxes.view(-1, 4)
        predicted = (
            boxes.mean(dim=0)
            * self.tracker.params.search_size
            / resize_factor
        ).tolist()
        anchor_center_x = anchor[0] + 0.5 * anchor[2]
        anchor_center_y = anchor[1] + 0.5 * anchor[3]
        half_side = 0.5 * self.tracker.params.search_size / resize_factor
        center_x, center_y, box_width, box_height = predicted
        mapped = [
            center_x + anchor_center_x - half_side - 0.5 * box_width,
            center_y + anchor_center_y - half_side - 0.5 * box_height,
            box_width,
            box_height,
        ]
        mapped = np.asarray(
            sutrack_clip_box(mapped, self.height, self.width, margin=10),
            dtype=np.float64,
        )

        response_array = response.detach().float().cpu().numpy().reshape(
            response.shape[-2], response.shape[-1]
        )
        peak_index = np.unravel_index(np.argmax(response_array), response_array.shape)
        mask = np.ones_like(response_array, dtype=bool)
        row, column = peak_index
        mask[
            max(0, row - 1) : min(mask.shape[0], row + 2),
            max(0, column - 1) : min(mask.shape[1], column + 2),
        ] = False
        sidelobe = response_array[mask]
        peak = float(response_array[peak_index])
        psr = (peak - float(sidelobe.mean())) / (float(sidelobe.std()) + 1e-6)
        network_score = float(confidence.detach().cpu().item())
        appearance_score = self.appearance.score(image, mapped)
        psr_score = 1.0 / (
            1.0
            + math.exp(
                -(math.log(max(psr, 1e-6)) - math.log(50.0)) / 0.55
            )
        )
        quality = float(
            np.clip(
                0.75 * np.clip(network_score, 0.0, 1.0)
                + 0.15 * psr_score
                + 0.10 * appearance_score,
                0.0,
                1.0,
            )
        )
        return {
            "box": mapped,
            "quality": quality,
            "network_score": network_score,
            "psr": psr,
            "appearance_score": appearance_score,
            "search_factor": search_factor,
        }

    def global_proposals(self, image: np.ndarray, predicted_box: np.ndarray, limit=3):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        downscale = min(1.0, 960.0 / max(gray.shape))
        if downscale < 1.0:
            gray_small = cv2.resize(
                gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA
            )
        else:
            gray_small = gray
        templates = [self.appearance.initial, self.appearance.recent]
        candidates = []
        predicted_w = max(4.0, predicted_box[2] * downscale)
        predicted_h = max(4.0, predicted_box[3] * downscale)
        for template_index, template in enumerate(templates):
            for scale_factor in (0.60, 0.80, 1.00, 1.25, 1.55):
                width = int(round(predicted_w * scale_factor))
                height = int(round(predicted_h * scale_factor))
                if (
                    width < 4
                    or height < 4
                    or width >= gray_small.shape[1]
                    or height >= gray_small.shape[0]
                ):
                    continue
                resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
                response = cv2.matchTemplate(
                    gray_small.astype(np.float32),
                    resized.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
                for _ in range(2):
                    _, maximum, _, location = cv2.minMaxLoc(response)
                    candidates.append(
                        (
                            float(maximum),
                            np.array(
                                [
                                    location[0] / downscale,
                                    location[1] / downscale,
                                    width / downscale,
                                    height / downscale,
                                ],
                                dtype=np.float64,
                            ),
                            template_index,
                        )
                    )
                    x1 = max(0, location[0] - width)
                    y1 = max(0, location[1] - height)
                    x2 = min(response.shape[1], location[0] + width)
                    y2 = min(response.shape[0], location[1] + height)
                    response[y1:y2, x1:x2] = -1.0
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for ncc, box, template_index in candidates:
            center = box[:2] + 0.5 * box[2:]
            if any(
                np.linalg.norm(center - (other[1][:2] + 0.5 * other[1][2:]))
                < 0.5 * math.sqrt(box[2] * box[3])
                for other in selected
            ):
                continue
            selected.append((ncc, box, template_index))
            if len(selected) >= limit:
                break
        return selected

    def update_template(self, image: np.ndarray, box: np.ndarray, quality: float):
        if (
            self.tracker.num_template <= 1
            or self.frame_id % self.tracker.update_intervals != 0
            or quality < float(self.tracker.update_threshold)
        ):
            return
        from lib.test.tracker.utils import sample_target, transform_image_to_crop

        patch, resize_factor = sample_target(
            image,
            box.tolist(),
            self.tracker.params.template_factor,
            output_sz=self.tracker.params.template_size,
        )
        template = self.tracker.preprocessor.process(patch)
        if self.tracker.multi_modal_vision and template.size(1) == 3:
            template = torch.cat((template, template), axis=1)
        self.tracker.template_list.append(template)
        if len(self.tracker.template_list) > self.tracker.num_template:
            self.tracker.template_list.pop(1)
        annotation = transform_image_to_crop(
            torch.tensor(box, dtype=torch.float32),
            torch.tensor(box, dtype=torch.float32),
            resize_factor,
            torch.tensor(
                [self.tracker.params.template_size, self.tracker.params.template_size],
                dtype=torch.float32,
            ),
            normalize=True,
        )
        self.tracker.template_anno_list.append(
            annotation.to(template.device).unsqueeze(0)
        )
        if len(self.tracker.template_anno_list) > self.tracker.num_template:
            self.tracker.template_anno_list.pop(1)
        self.appearance.update(image, box)

    def track(self, image: np.ndarray):
        self.frame_id += 1
        start_camera = time.perf_counter()
        if self.disable_camera:
            affine, camera_quality, camera_inliers = (
                np.eye(2, 3, dtype=np.float64),
                0.0,
                0,
            )
        else:
            affine, camera_quality, camera_inliers = self.camera.estimate(
                self.previous_image, image
            )
            if camera_quality > 0:
                self.camera_valid_count += 1
        camera_seconds = time.perf_counter() - start_camera

        predicted_box, prior_covariance = self.imm.predict(
            None if self.disable_camera else affine,
            camera_quality,
        )
        predicted_box = clip_box(predicted_box, self.width, self.height, margin=10)
        target_scale = max(math.sqrt(predicted_box[2] * predicted_box[3]), 10.0)
        center_sigma = math.sqrt(
            max(
                prior_covariance[0, 0] * self.width**2
                + prior_covariance[1, 1] * self.height**2,
                0.0,
            )
        )
        uncertainty_extra = min(2.0, center_sigma / target_scale)
        imm_search_factor = float(
            np.clip(4.0 + uncertainty_extra + 0.75 * self.lost_count, 4.0, 8.0)
        )

        torch.cuda.synchronize()
        start_visual = time.perf_counter()
        # Guarded dual hypothesis: the ordinary SUTrack recursion is always
        # retained as the safety branch.  An IMM-guided search is evaluated
        # only when confidence falls and must win by a margin to take over.
        safety = self.visual_probe(image, self.safety_measurement, 4.0)
        self.safety_measurement = safety["box"].copy()
        best = safety
        probes = 1
        used_redetection = False
        used_imm_search = False
        if best["quality"] < 0.35:
            safety_quality = best["quality"]
            safety_network_score = best["network_score"]
            imm_candidate = self.visual_probe(
                image, predicted_box, imm_search_factor
            )
            probes += 1
            if (
                imm_candidate["quality"] > safety_quality + 0.15
                and imm_candidate["network_score"] > safety_network_score + 0.12
                and imm_candidate["appearance_score"]
                > safety["appearance_score"] + 0.05
            ):
                best = imm_candidate
                used_imm_search = True
        if best["quality"] < 0.15:
            expanded_factor = min(12.0, max(7.0, imm_search_factor * 1.65))
            expanded = self.visual_probe(image, predicted_box, expanded_factor)
            probes += 1
            self.expanded_search_count += 1
            if expanded["quality"] > best["quality"] + 0.03:
                best = expanded
                used_imm_search = True

        if (
            not self.disable_redetection
            and (best["quality"] < 0.08 or self.lost_count >= 6)
        ):
            for ncc, proposal, template_index in self.global_proposals(
                image, predicted_box, limit=3
            ):
                candidate = self.visual_probe(image, proposal, 4.5)
                candidate["template_ncc"] = ncc
                candidate["template_index"] = template_index
                probes += 1
                if candidate["quality"] > best["quality"] + 0.03:
                    best = candidate
                    used_redetection = True
            if used_redetection:
                self.redetection_count += 1
        torch.cuda.synchronize()
        visual_seconds = time.perf_counter() - start_visual

        fused_box, posterior_covariance, updated = self.imm.correct(
            best["box"],
            best["quality"],
            fixed_measurement_noise=self.fixed_measurement_noise,
        )
        if not updated:
            self.skipped_update_count += 1
        fused_box = clip_box(fused_box, self.width, self.height, margin=10)
        measurement_weight = float(
            np.clip((best["quality"] - 0.05) / 0.10, 0.0, 1.0)
        )
        output_box = (
            measurement_weight * best["box"]
            + (1.0 - measurement_weight) * fused_box
        )
        output_box = clip_box(output_box, self.width, self.height, margin=10)
        if best["quality"] >= 0.22:
            self.lost_count = 0
        else:
            self.lost_count += 1
        # Only the safety branch may update shared appearance templates, so a
        # mistaken recovery hypothesis cannot poison future baseline searches.
        self.update_template(
            image, safety["box"], safety["network_score"]
        )
        self.last_box = output_box
        self.last_measurement = best["box"].copy()
        self.last_quality = best["quality"]
        self.previous_image = image
        self.tracker.state = output_box.tolist()
        diagnostic = {
            **best,
            "camera_quality": camera_quality,
            "camera_inliers": camera_inliers,
            "camera_seconds": camera_seconds,
            "visual_seconds": visual_seconds,
            "probes": probes,
            "used_redetection": used_redetection,
            "used_imm_search": used_imm_search,
            "measurement_updated": updated,
            "low_dynamic_probability": float(self.imm.probabilities[0]),
            "constant_velocity_probability": float(self.imm.probabilities[1]),
            "high_maneuver_probability": float(self.imm.probabilities[2]),
            "center_sigma_px": center_sigma,
        }
        diagnostic.pop("box", None)
        diagnostic.setdefault("template_ncc", "")
        diagnostic.setdefault("template_index", "")
        return output_box, diagnostic
