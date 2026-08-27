from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass
class ShooterConfig:
    target_width: int = 595
    target_height: int = 842
    inner_radius: int = 28
    outer_radius: int = 293
    white_threshold: int = 160
    max_shots: int = 10
    min_interval: float = 0.3
    min_distance: float = 0.0
    scores: tuple[int, ...] = (10, 8, 6, 4, 2)

    def validate(self) -> None:
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("Target width and height must be positive.")
        if self.inner_radius <= 0:
            raise ValueError("Inner radius must be positive.")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("Outer radius must be larger than inner radius.")
        if not 0 <= self.white_threshold <= 255:
            raise ValueError("White threshold must be between 0 and 255.")
        if self.max_shots <= 0:
            raise ValueError("Maximum shots must be positive.")

    @property
    def ring_radii(self) -> list[int]:
        return np.linspace(
            self.inner_radius, self.outer_radius, len(self.scores)
        ).astype(int).tolist()


@dataclass(frozen=True)
class Shot:
    number: int
    x: int
    y: int
    score: int


@dataclass
class FrameResult:
    preview: np.ndarray
    target: np.ndarray | None
    threshold_mask: np.ndarray
    marker_count: int
    new_shots: list[Shot]
    round_complete: bool
    total_score: int
    shot_count: int
    max_shots: int


class LaserShooterEngine:
    def __init__(self, config: ShooterConfig):
        config.validate()
        self.config = config
        self.shots: list[Shot] = []
        self.hits: list[tuple[int, int, float]] = []

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    @property
    def total_score(self) -> int:
        return sum(shot.score for shot in self.shots)

    def reset(self) -> None:
        self.shots.clear()
        self.hits.clear()

    def update_threshold(self, threshold: int) -> None:
        self.config.white_threshold = max(0, min(255, int(threshold)))

    def calculate_score(self, x: int, y: int) -> int:
        center_x = self.config.target_width // 2
        center_y = self.config.target_height // 2
        distance = np.hypot(x - center_x, y - center_y)

        for radius, score in zip(self.config.ring_radii, self.config.scores):
            if distance <= radius:
                return score
        return 0

    def is_new_shot(self, point: tuple[int, int], timestamp: float) -> bool:
        for hit_x, hit_y, hit_time in self.hits:
            distance = np.hypot(point[0] - hit_x, point[1] - hit_y)
            if self.config.min_distance > 0 and distance < self.config.min_distance:
                return False
            if timestamp - hit_time < self.config.min_interval:
                return False
        return True

    def process_frame(self, frame: np.ndarray, player_name: str) -> FrameResult:
        preview = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        detected_ids: set[int] = set()
        if ids is not None:
            detected_ids = {int(marker_id) for marker_id in ids.flatten()}
            cv2.aruco.drawDetectedMarkers(preview, corners, ids)

        required_ids = {0, 1, 2, 3}
        marker_count = len(required_ids.intersection(detected_ids))
        marker_color = (30, 200, 30) if marker_count == 4 else (30, 180, 255)
        cv2.putText(
            preview,
            f"Target markers: {marker_count}/4",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            marker_color,
            2,
        )

        raw_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        _, raw_cr, _ = cv2.split(raw_ycrcb)
        _, threshold_mask = cv2.threshold(
            raw_cr,
            self.config.white_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        target = None
        new_shots: list[Shot] = []
        if ids is not None and required_ids.issubset(detected_ids):
            marker_corners = {
                marker_id: corners[np.where(ids == marker_id)[0][0]][0]
                for marker_id in required_ids
            }
            source_points = np.array(
                [
                    marker_corners[0][0],
                    marker_corners[1][1],
                    marker_corners[2][2],
                    marker_corners[3][3],
                ],
                dtype=np.float32,
            )
            destination_points = np.array(
                [
                    [0, 0],
                    [self.config.target_width, 0],
                    [self.config.target_width, self.config.target_height],
                    [0, self.config.target_height],
                ],
                dtype=np.float32,
            )

            transform = cv2.getPerspectiveTransform(
                source_points, destination_points
            )
            target = cv2.warpPerspective(
                frame,
                transform,
                (self.config.target_width, self.config.target_height),
            )
            new_shots = self._detect_shots(target)
            self._draw_target_overlay(target, player_name)

        return FrameResult(
            preview=preview,
            target=target,
            threshold_mask=threshold_mask,
            marker_count=marker_count,
            new_shots=new_shots,
            round_complete=len(self.shots) >= self.config.max_shots,
            total_score=self.total_score,
            shot_count=len(self.shots),
            max_shots=self.config.max_shots,
        )

    def _detect_shots(self, target: np.ndarray) -> list[Shot]:
        if len(self.shots) >= self.config.max_shots:
            return []

        ycrcb = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
        _, red_chroma, _ = cv2.split(ycrcb)
        _, binary = cv2.threshold(
            red_chroma,
            self.config.white_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        new_shots: list[Shot] = []
        for contour in contours:
            if len(self.shots) >= self.config.max_shots:
                break
            if cv2.contourArea(contour) < 5:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            timestamp = time.time()
            if not self.is_new_shot((center_x, center_y), timestamp):
                continue

            shot = Shot(
                number=len(self.shots) + 1,
                x=center_x,
                y=center_y,
                score=self.calculate_score(center_x, center_y),
            )
            self.shots.append(shot)
            self.hits.append((center_x, center_y, timestamp))
            new_shots.append(shot)

        return new_shots

    def _draw_target_overlay(self, target: np.ndarray, player_name: str) -> None:
        center = (
            self.config.target_width // 2,
            self.config.target_height // 2,
        )
        for radius, score in zip(self.config.ring_radii, self.config.scores):
            cv2.circle(target, center, radius, (255, 255, 255), 2)
            cv2.putText(
                target,
                str(score),
                (center[0] + radius - 30, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        for index, shot in enumerate(self.shots):
            is_latest = index == len(self.shots) - 1
            color = (0, 255, 0) if is_latest else (0, 0, 255)
            radius = 7 if is_latest else 5
            cv2.circle(target, (shot.x, shot.y), radius, color, -1)

        title = f"{player_name} - Total: {self.total_score}"
        if self.shots:
            title += f" - Last: {self.shots[-1].score}"
        cv2.putText(
            target,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
