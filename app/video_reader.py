import os
import logging
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReaderError(Exception):
    """Raised when a video file is missing, unreadable, or corrupt."""


class VideoReader:
    def __init__(self, video_path: str, config: dict):
        if not os.path.exists(video_path):
            raise VideoReaderError(f"Video file does not exist: {video_path}")
        if os.path.getsize(video_path) == 0:
            raise VideoReaderError(f"Video file is empty (0 bytes): {video_path}")

        self.video_path = video_path
        self.video_cfg = config.get("video", {})
        self.mode = self.video_cfg.get("mode", "sampled")
        self.sample_fps = float(self.video_cfg.get("sample_fps", 5.0))
        self.camera_mode = self.video_cfg.get("camera_mode", "auto")
        self.adaptive_sampling = bool(self.video_cfg.get("adaptive_sampling", True))
        self.motion_thresh = float(self.video_cfg.get("motion_pixel_threshold", 12.0))

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise VideoReaderError(f"OpenCV could not open video file (corrupt or unsupported codec): {video_path}")

        self.native_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.native_fps <= 0 or np.isnan(self.native_fps):
            logger.warning("Invalid native FPS reported by container; defaulting to 30.0 FPS.")
            self.native_fps = 30.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.width <= 0 or self.height <= 0:
            self.release()
            raise VideoReaderError(f"Video has invalid frame dimensions ({self.width}x{self.height}): {video_path}")

        self._prev_gray: Optional[np.ndarray] = None
        self._last_processed_ts: float = -999.0
        self._boost_until_ts: float = -1.0

    def estimate_camera_motion(self, frame: np.ndarray) -> Tuple[str, float]:
        if self.camera_mode in ("fixed", "moving"):
            return self.camera_mode, 0.0

        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return "fixed", 0.0

        pts = cv2.goodFeaturesToTrack(self._prev_gray, maxCorners=80, qualityLevel=0.05, minDistance=12)
        if pts is None or len(pts) < 8:
            self._prev_gray = gray
            return "uncertain_motion", 0.0

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts, None)
        self._prev_gray = gray

        if next_pts is None or status is None:
            return "uncertain_motion", 0.0

        good_old = pts[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]
        if len(good_old) < 6:
            return "uncertain_motion", 0.0

        displacements = np.linalg.norm(good_new - good_old, axis=2).flatten()
        median_disp = float(np.median(displacements))
        scale_factor = self.width / 320.0
        scaled_disp = median_disp * scale_factor

        state = "moving" if scaled_disp >= self.motion_thresh else "fixed"
        return state, scaled_disp

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, str, float], None, None]:
        frame_idx = -1
        consecutive_failures = 0
        max_failures = 30
        read_any = False

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures or (self.total_frames > 0 and frame_idx >= self.total_frames - 1):
                        break
                    frame_idx += 1
                    continue

                consecutive_failures = 0
                read_any = True
                frame_idx += 1

                pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec > 0:
                    timestamp = float(pos_msec) / 1000.0
                else:
                    timestamp = float(frame_idx) / self.native_fps

                if timestamp <= self._last_processed_ts and frame_idx > 0:
                    timestamp = float(frame_idx) / self.native_fps

                if self.mode == "sampled":
                    target_fps = self.sample_fps
                    if self.adaptive_sampling and timestamp < self._boost_until_ts:
                        target_fps = min(self.native_fps, self.sample_fps * 2.0)

                    min_interval = (1.0 / target_fps) - (0.35 / self.native_fps)
                    if (timestamp - self._last_processed_ts) < min_interval:
                        continue

                cam_state, motion_px = self.estimate_camera_motion(frame)
                if self.adaptive_sampling and motion_px >= self.motion_thresh:
                    self._boost_until_ts = timestamp + 1.5

                self._last_processed_ts = timestamp
                yield frame_idx, timestamp, frame, cam_state, motion_px

            if not read_any:
                raise VideoReaderError(f"No valid video frames could be decoded from: {self.video_path}")
        finally:
            self.release()

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
