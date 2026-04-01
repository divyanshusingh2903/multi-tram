"""
Kalman Filter for Bounding Box Tracking

Provides temporal prediction and measurement update for tracking objects across frames.
Handles occlusions and estimates velocity automatically.
"""
import numpy as np


class KalmanFilter:
    """
    Kalman Filter for tracking bounding boxes

    State: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): bbox center
    - (w, h): bbox width/height
    - (vx, vy): velocity in x/y
    - (vw, vh): velocity in w/h

    Uses constant velocity model with process and measurement noise.
    """

    def __init__(self, bbox: np.ndarray, dt: float = 1.0):
        """
        Initialize Kalman filter with initial detection.

        Args:
            bbox: Initial bounding box [x, y, w, h]
            dt: Time step (default: 1 frame)
        """
        self.dt = dt
        self.state = np.zeros(8)  # [x, y, w, h, vx, vy, vw, vh]

        # Initialize position
        self.state[:4] = bbox

        # Covariance matrix (uncertainty in state)
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] = np.eye(4) * 1000.0  # High uncertainty in velocity initially

        # Process noise (uncertainty in motion model)
        self.Q = np.eye(8) * 0.1
        self.Q[4:, 4:] = np.eye(4) * 0.1

        # Measurement noise (uncertainty in detections)
        self.R = np.eye(4) * 10.0

        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = dt  # x += vx * dt
        self.F[1, 5] = dt  # y += vy * dt
        self.F[2, 6] = dt  # w += vw * dt
        self.F[3, 7] = dt  # h += vh * dt

        # Measurement matrix (we only measure position/size, not velocity)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

    def predict(self) -> np.ndarray:
        """
        Predict next state.

        Returns:
            Predicted bounding box [x, y, w, h]
        """
        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Clip to reasonable values
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)  # width/height >= 1

        return self.state[:4].copy()

    def update(self, bbox: np.ndarray) -> None:
        """
        Update with new detection.

        Args:
            bbox: Detected bounding box [x, y, w, h]
        """
        # Innovation (measurement residual)
        z = bbox
        y = z - (self.H @ self.state)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S + 1e-6)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.P = (np.eye(8) - K @ self.H) @ self.P

        # Clip to reasonable values
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)

    def get_confidence(self) -> float:
        """
        Get confidence in current state based on covariance.

        Returns:
            Confidence score [0, 1]
        """
        # Confidence inversely proportional to uncertainty in position
        position_uncertainty = np.trace(self.P[:2, :2])
        confidence = 1.0 / (1.0 + position_uncertainty)
        return float(np.clip(confidence, 0.0, 1.0))
