"""Original eight-state, three-mode IMM implementation (algorithm unchanged)."""
from __future__ import annotations
import math
import numpy as np

def state_from_box(box: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    x, y, w, h = map(float, box)
    w_norm = max(w / image_width, 1e-6)
    h_norm = max(h / image_height, 1e-6)
    return np.array(
        [
            (x + 0.5 * w) / image_width,
            (y + 0.5 * h) / image_height,
            0.0,
            0.0,
            math.log(w_norm),
            math.log(h_norm),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def measurement_from_box(
    box: np.ndarray, image_width: int, image_height: int
) -> np.ndarray:
    state = state_from_box(box, image_width, image_height)
    return state[[0, 1, 4, 5]]


def box_from_state(state: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    cx = float(state[0] * image_width)
    cy = float(state[1] * image_height)
    w = float(np.clip(math.exp(np.clip(state[4], -13.0, 0.5)), 1e-6, 1.5) * image_width)
    h = float(np.clip(math.exp(np.clip(state[5], -13.0, 0.5)), 1e-6, 1.5) * image_height)
    return np.array([cx - 0.5 * w, cy - 0.5 * h, w, h], dtype=np.float64)


def transition(velocity_retention: float) -> np.ndarray:
    matrix = np.eye(8, dtype=np.float64)
    for position, velocity in ((0, 2), (1, 3), (4, 6), (5, 7)):
        matrix[position, velocity] = 1.0
        matrix[velocity, velocity] = velocity_retention
    return matrix


def process_noise(center_acceleration: float, size_acceleration: float) -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=np.float64)
    base = np.array([[0.25, 0.5], [0.5, 1.0]], dtype=np.float64)
    for position, velocity, sigma in (
        (0, 2, center_acceleration),
        (1, 3, center_acceleration),
        (4, 6, size_acceleration),
        (5, 7, size_acceleration),
    ):
        indices = np.ix_([position, velocity], [position, velocity])
        matrix[indices] = base * sigma**2
    return matrix + np.eye(8) * 1e-12


class BoxIMM:
    """Three-model interacting multiple-model Kalman filter."""

    def __init__(self, initial_box, image_width: int, image_height: int, robust: bool):
        self.image_width = image_width
        self.image_height = image_height
        self.robust = robust
        initial = state_from_box(initial_box, image_width, image_height)
        self.states = [initial.copy() for _ in range(3)]
        observed_variance = np.array([0.012**2, 0.012**2, 0.08**2, 0.08**2])
        initial_variance = np.array(
            [
                observed_variance[0],
                observed_variance[1],
                0.025**2,
                0.025**2,
                observed_variance[2],
                observed_variance[3],
                0.10**2,
                0.10**2,
            ]
        )
        self.covariances = [np.diag(initial_variance) for _ in range(3)]
        self.probabilities = np.array([0.45, 0.40, 0.15], dtype=np.float64)
        self.model_transition = np.array(
            [[0.92, 0.07, 0.01], [0.06, 0.90, 0.04], [0.03, 0.12, 0.85]],
            dtype=np.float64,
        )
        self.transitions = [
            transition(0.15),
            transition(1.00),
            transition(0.92),
        ]
        self.process_noises = [
            process_noise(0.0008, 0.006),
            process_noise(0.0030, 0.020),
            process_noise(0.0150, 0.080),
        ]
        self.measurement_matrix = np.zeros((4, 8), dtype=np.float64)
        self.measurement_matrix[0, 0] = 1.0
        self.measurement_matrix[1, 1] = 1.0
        self.measurement_matrix[2, 4] = 1.0
        self.measurement_matrix[3, 5] = 1.0
        self.measurement_noise = np.diag(observed_variance)

    def _mix(self):
        normalization = self.probabilities @ self.model_transition
        normalization = np.maximum(normalization, 1e-300)
        mixing = (
            self.probabilities[:, None] * self.model_transition
        ) / normalization[None, :]
        mixed_states = []
        mixed_covariances = []
        for destination in range(3):
            weights = mixing[:, destination]
            mean = sum(weights[source] * self.states[source] for source in range(3))
            covariance = np.zeros((8, 8), dtype=np.float64)
            for source in range(3):
                delta = self.states[source] - mean
                covariance += weights[source] * (
                    self.covariances[source] + np.outer(delta, delta)
                )
            mixed_states.append(mean)
            mixed_covariances.append(covariance)
        return normalization, mixed_states, mixed_covariances

    @staticmethod
    def _log_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            covariance = covariance + np.eye(len(residual)) * 1e-9
            _, log_determinant = np.linalg.slogdet(covariance)
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        return -0.5 * (
            len(residual) * math.log(2.0 * math.pi) + log_determinant + quadratic
        )

    def update(self, measurement_box: np.ndarray):
        measurement = measurement_from_box(
            measurement_box, self.image_width, self.image_height
        )
        normalization, mixed_states, mixed_covariances = self._mix()
        predicted_states = []
        predicted_covariances = []
        for model in range(3):
            transition_matrix = self.transitions[model]
            predicted_states.append(transition_matrix @ mixed_states[model])
            predicted_covariances.append(
                transition_matrix
                @ mixed_covariances[model]
                @ transition_matrix.T
                + self.process_noises[model]
            )

        measurement_noise = self.measurement_noise.copy()
        inflation = 1.0
        if self.robust:
            prior_mean = sum(
                normalization[model] * predicted_states[model] for model in range(3)
            ) / normalization.sum()
            prior_covariance = sum(
                normalization[model] * predicted_covariances[model]
                for model in range(3)
            ) / normalization.sum()
            center_residual = measurement[:2] - prior_mean[:2]
            center_covariance = prior_covariance[:2, :2] + measurement_noise[:2, :2]
            distance = math.sqrt(
                max(0.0, float(center_residual @ np.linalg.solve(center_covariance, center_residual)))
            )
            if distance > 4.0:
                inflation = min(100.0, (distance / 4.0) ** 2)
                measurement_noise[:2, :2] *= inflation
                measurement_noise[2:, 2:] *= math.sqrt(inflation)

        updated_states = []
        updated_covariances = []
        log_probabilities = []
        identity = np.eye(8)
        for model in range(3):
            state = predicted_states[model]
            covariance = predicted_covariances[model]
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
                math.log(normalization[model])
                + self._log_likelihood(residual, innovation_covariance)
            )

        log_probabilities = np.asarray(log_probabilities)
        log_probabilities -= log_probabilities.max()
        probabilities = np.exp(log_probabilities)
        self.probabilities = probabilities / probabilities.sum()
        self.states = updated_states
        self.covariances = updated_covariances
        fused = sum(
            self.probabilities[model] * self.states[model] for model in range(3)
        )
        return (
            box_from_state(fused, self.image_width, self.image_height),
            self.probabilities.copy(),
            inflation,
        )
