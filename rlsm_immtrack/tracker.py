"""Future-validated dual memory and an output-only speculative shadow branch."""
from __future__ import annotations
import math
import time
from collections import deque
import numpy as np
import torch
from .safecl import AppearanceMemory, ClosedLoopTracker

class RiskLimitedDualMemory:
    """Provisional writes plus conservative, calibrated two-layer scoring."""

    def __init__(
        self, image, box, short_capacity=6, long_capacity=4,
        memory_supports=3, long_weight=0.70, disagreement_penalty=0.25,
    ):
        patch = AppearanceMemory.extract(image, box)
        initial = {"patch": patch, "quality": 1.0, "frame": 0, "initial": True}
        self.short = deque([initial.copy()], maxlen=max(2, short_capacity))
        self.long = [initial]
        self.long_capacity = max(2, long_capacity)
        self.memory_supports = max(1, int(memory_supports))
        self.long_weight = float(np.clip(long_weight, 0.0, 1.0))
        self.disagreement_penalty = max(0.0, float(disagreement_penalty))
        self.pending = None
        self.accepted_support = deque([0.90], maxlen=64)
        self.last_proposal_frame = 0
        self.last_long_frame = 0
        self.proposals = 0
        self.provisional_commits = 0
        self.long_commits = 0
        self.rejections = 0

    @staticmethod
    def similarity(first, second):
        return AppearanceMemory.similarity(first, second)

    @staticmethod
    def normalized(raw):
        return float(np.clip(0.5 * (raw + 1.0), 0.0, 1.0))

    def _layer_score(self, patch, entries):
        raw = np.asarray(
            [self.similarity(entry["patch"], patch) for entry in entries],
            dtype=np.float64,
        )
        order = np.argsort(raw)[::-1]
        best = float(raw[order[0]])
        if len(order) > 1:
            best = 0.80 * best + 0.20 * float(raw[order[1]])
        return self.normalized(best), float(raw[order[0]])

    def score_patch(self, patch):
        long_score, long_raw = self._layer_score(patch, self.long)
        short_score, short_raw = self._layer_score(patch, list(self.short))
        disagreement = abs(long_score - short_score)
        combined = float(np.clip(
            self.long_weight * long_score
            + (1.0 - self.long_weight) * short_score
            - self.disagreement_penalty * disagreement,
            0.0, 1.0,
        ))
        return combined, {
            "long_score": long_score,
            "short_score": short_score,
            "agreement": float(np.clip(1.0 - disagreement, 0.0, 1.0)),
            "long_raw": long_raw,
            "short_raw": short_raw,
        }

    def score(self, image, box):
        return self.score_patch(AppearanceMemory.extract(image, box))

    def _support_threshold(self):
        # Calibrate from confirmed positive transitions instead of a fixed NCC.
        return float(np.clip(np.percentile(self.accepted_support, 15) - 0.10, 0.48, 0.72))

    def _commit(self, patch, quality, frame, components):
        entry = {"patch": patch, "quality": float(quality), "frame": frame, "initial": False}
        self.short.append(entry)
        self.provisional_commits += 1
        long_floor = max(0.42, self._support_threshold() - 0.08)
        novelty = max(self.similarity(item["patch"], patch) for item in self.long)
        if (
            quality >= 0.78
            and components["long_score"] >= long_floor
            and frame - self.last_long_frame >= 25
            and novelty < 0.985
        ):
            if len(self.long) < self.long_capacity:
                self.long.append(entry.copy())
            else:
                replace = min(
                    range(1, len(self.long)),
                    key=lambda index: self.long[index]["quality"]
                    - 0.0002 * max(frame - self.long[index]["frame"], 0),
                )
                self.long[replace] = entry.copy()
            self.long_commits += 1
            self.last_long_frame = frame

    def observe(self, image, box, quality, frame, motion_consistent):
        patch = AppearanceMemory.extract(image, box)
        _, components = self.score_patch(patch)
        committed = False
        if self.pending is not None:
            support = self.normalized(self.similarity(self.pending["patch"], patch))
            if quality >= 0.62 and motion_consistent and support >= self._support_threshold():
                self.pending["supports"] += 1
                self.pending["support_scores"].append(support)
            elif quality < 0.45 or support < 0.35:
                self.pending["failures"] += 1
            age = frame - self.pending["frame"]
            if self.pending["supports"] >= self.memory_supports:
                mean_support = float(np.mean(self.pending["support_scores"]))
                self.accepted_support.append(mean_support)
                self._commit(patch, quality, frame, components)
                self.pending = None
                committed = True
            elif age > 10 or self.pending["failures"] >= 3:
                self.pending = None
                self.rejections += 1
        if (
            self.pending is None
            and not committed
            and quality >= 0.68
            and motion_consistent
            and frame - self.last_proposal_frame >= 5
        ):
            self.pending = {
                "patch": patch, "quality": float(quality), "frame": frame,
                "supports": 0, "failures": 0, "support_scores": [],
            }
            self.last_proposal_frame = frame
            self.proposals += 1
        return components

    def quarantine(self):
        if self.pending is not None:
            self.pending = None
            self.rejections += 1


class RiskLimitedShadowTracker(ClosedLoopTracker):
    """Exact SafeCL recursion plus a non-contaminating shadow trajectory."""

    def __init__(
        self, tracker, initial_image, initial_box, disable_camera,
        disable_redetection, fixed_measurement_noise, short_capacity,
        long_capacity, memory_supports=3, long_weight=0.70,
        disagreement_penalty=0.25, entry_streak=3,
        entry_evidence=0.045, exit_patience=2,
    ):
        super().__init__(
            tracker, initial_image, initial_box, disable_camera,
            disable_redetection, fixed_measurement_noise,
        )
        self.shadow_memory = RiskLimitedDualMemory(
            initial_image, initial_box, short_capacity, long_capacity,
            memory_supports, long_weight, disagreement_penalty,
        )
        self.entry_streak = max(1, int(entry_streak))
        self.entry_evidence = float(entry_evidence)
        self.exit_patience = max(1, int(exit_patience))
        self.shadow_anchor = np.asarray(initial_box, dtype=np.float64).copy()
        self.shadow_velocity = np.zeros(2, dtype=np.float64)
        self.shadow_active = False
        self.shadow_win_streak = 0
        self.shadow_loss_streak = 0
        self.shadow_switches = 0
        self.shadow_rollbacks = 0
        self.shadow_probe_count = 0
        self.shadow_selected_frames = 0
        self.last_memory_components = {
            "long_score": 1.0, "short_score": 1.0, "agreement": 1.0
        }

    @staticmethod
    def _center(box):
        return box[:2] + 0.5 * box[2:]

    def _shadow_prediction(self):
        predicted = self.shadow_anchor.copy()
        predicted[:2] += self.shadow_velocity
        return predicted

    def _motion_score(self, box, predicted):
        scale = max(math.sqrt(predicted[2] * predicted[3]), 10.0)
        center_error = np.linalg.norm(self._center(box) - self._center(predicted))
        scale_error = abs(math.log(max(box[2], 1.0) / max(predicted[2], 1.0)))
        scale_error += abs(math.log(max(box[3], 1.0) / max(predicted[3], 1.0)))
        return float(np.exp(-0.5 * (center_error / (1.5 * scale)) ** 2 - 0.20 * scale_error))

    @staticmethod
    def _psr_value(value):
        return float(np.tanh(math.log(max(float(value), 1e-6)) / 6.0))

    def _advance_shadow(self, box):
        old_center = self._center(self.shadow_anchor)
        new_center = self._center(box)
        instantaneous = new_center - old_center
        self.shadow_velocity = 0.65 * self.shadow_velocity + 0.35 * instantaneous
        self.shadow_anchor = np.asarray(box, dtype=np.float64).copy()

    def _reset_shadow(self, baseline_box):
        previous = self._center(self.shadow_anchor)
        current = self._center(baseline_box)
        self.shadow_velocity = 0.5 * (current - previous)
        self.shadow_anchor = np.asarray(baseline_box, dtype=np.float64).copy()
        self.shadow_win_streak = 0
        self.shadow_loss_streak = 0

    def track(self, image):
        # Preserve the exact templates used by the SafeCL probe on this frame.
        # The parent may update templates at the end of its recursion; the
        # shadow probe must not use a current-frame template to track itself.
        pre_templates = list(self.tracker.template_list)
        pre_annotations = list(self.tracker.template_anno_list)
        baseline_box, diagnostic = super().track(image)
        baseline_box = np.asarray(baseline_box, dtype=np.float64)

        baseline_memory, baseline_components = self.shadow_memory.score(
            image, baseline_box
        )
        target_scale = max(math.sqrt(baseline_box[2] * baseline_box[3]), 10.0)
        uncertainty_ratio = diagnostic["center_sigma_px"] / target_scale
        should_probe = (
            self.shadow_active
            or self.shadow_win_streak > 0
            or diagnostic["quality"] < 0.62
            or uncertainty_ratio > 0.90
        )
        shadow = None
        evidence = 0.0
        divergence = 0.0
        selected_shadow = False
        rollback = False
        shadow_memory_score = baseline_memory
        shadow_components = baseline_components
        shadow_motion = 1.0

        if should_probe:
            predicted = self._shadow_prediction()
            search_factor = float(np.clip(
                4.0 + 0.50 * uncertainty_ratio + 0.40 * self.shadow_loss_streak,
                4.0, 6.5,
            ))
            post_templates = self.tracker.template_list
            post_annotations = self.tracker.template_anno_list
            self.tracker.template_list = pre_templates
            self.tracker.template_anno_list = pre_annotations
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                shadow = self.visual_probe(image, predicted, search_factor)
            finally:
                torch.cuda.synchronize()
                diagnostic["visual_seconds"] += time.perf_counter() - start
                diagnostic["probes"] += 1
                self.shadow_probe_count += 1
                self.tracker.template_list = post_templates
                self.tracker.template_anno_list = post_annotations

            shadow_memory_score, shadow_components = self.shadow_memory.score(
                image, shadow["box"]
            )
            shadow_motion = self._motion_score(shadow["box"], predicted)
            baseline_motion = self._motion_score(baseline_box, predicted)
            network_delta = shadow["network_score"] - diagnostic["network_score"]
            quality_delta = shadow["quality"] - diagnostic["quality"]
            appearance_delta = shadow_memory_score - baseline_memory
            psr_delta = self._psr_value(shadow["psr"]) - self._psr_value(diagnostic["psr"])
            evidence = float(
                0.50 * network_delta + 0.15 * quality_delta
                + 0.17 * appearance_delta + 0.08 * psr_delta
                + 0.10 * (shadow_motion - baseline_motion)
            )
            divergence = float(
                np.linalg.norm(self._center(shadow["box"]) - self._center(baseline_box))
                / target_scale
            )
            strong_win = (
                evidence > self.entry_evidence
                and shadow["network_score"] > diagnostic["network_score"] + 0.025
                and shadow_memory_score >= baseline_memory - 0.025
                and shadow_motion >= 0.35
            )
            keep_shadow = (
                evidence > -0.025
                and shadow["network_score"] >= diagnostic["network_score"] - 0.035
                and shadow_memory_score >= baseline_memory - 0.070
                and shadow_motion >= 0.18
            )
            immediate_rollback = (
                shadow["network_score"] < diagnostic["network_score"] - 0.10
                or (divergence > 3.0 and evidence <= 0.0)
            )

            if strong_win:
                self.shadow_win_streak += 1
                self.shadow_loss_streak = 0
                self._advance_shadow(shadow["box"])
                if (
                    not self.shadow_active
                    and self.shadow_win_streak >= self.entry_streak
                    and divergence >= 0.10
                ):
                    self.shadow_active = True
                    self.shadow_switches += 1
            elif self.shadow_active and keep_shadow and not immediate_rollback:
                self.shadow_loss_streak = 0
                self._advance_shadow(shadow["box"])
            elif self.shadow_active:
                self.shadow_loss_streak += 1
                if immediate_rollback or self.shadow_loss_streak >= self.exit_patience:
                    self.shadow_active = False
                    self.shadow_rollbacks += 1
                    self.shadow_memory.quarantine()
                    self._reset_shadow(baseline_box)
                    rollback = True
            else:
                self.shadow_win_streak = max(0, self.shadow_win_streak - 1)
                self._reset_shadow(baseline_box)

            if self.shadow_active and not rollback and (strong_win or keep_shadow):
                selected_shadow = True
                self.shadow_selected_frames += 1
        else:
            self._reset_shadow(baseline_box)

        motion_consistent = bool(
            diagnostic["measurement_updated"]
            and diagnostic["quality"] >= 0.50
            and uncertainty_ratio < 2.5
        )
        self.last_memory_components = self.shadow_memory.observe(
            image, baseline_box, diagnostic["network_score"], self.frame_id,
            motion_consistent,
        )
        output_box = np.asarray(
            shadow["box"] if selected_shadow else baseline_box,
            dtype=np.float64,
        )
        diagnostic.update({
            "baseline_x": baseline_box[0], "baseline_y": baseline_box[1],
            "baseline_w": baseline_box[2], "baseline_h": baseline_box[3],
            "selected_shadow": selected_shadow,
            "shadow_active": self.shadow_active,
            "shadow_evidence": evidence,
            "shadow_divergence": divergence,
            "shadow_network_score": "" if shadow is None else shadow["network_score"],
            "shadow_quality": "" if shadow is None else shadow["quality"],
            "shadow_motion_score": shadow_motion,
            "baseline_memory_score": baseline_memory,
            "shadow_memory_score": shadow_memory_score,
            "long_memory_score": self.last_memory_components["long_score"],
            "short_memory_score": self.last_memory_components["short_score"],
            "memory_agreement": self.last_memory_components["agreement"],
            "memory_pending": self.shadow_memory.pending is not None,
            "memory_proposals": self.shadow_memory.proposals,
            "memory_commits": self.shadow_memory.provisional_commits,
            "long_memory_commits": self.shadow_memory.long_commits,
            "memory_rejections": self.shadow_memory.rejections,
            "shadow_switches": self.shadow_switches,
            "shadow_rollbacks": self.shadow_rollbacks,
        })
        return output_box, diagnostic

RLSMIMMTrack = RiskLimitedShadowTracker
