import os
import tempfile
import yaml
from typing import List
from app.models import Detection
from app.detector import PersonDetector


class MultiObjectTracker:
    def __init__(self, config: dict, detector: PersonDetector):
        self.t_cfg = config.get("tracker", {})
        self.v_cfg = config.get("video", {})
        self.detector = detector
        self._temp_dir = tempfile.TemporaryDirectory()
        self.fixed_yaml = self._write_tracker_yaml(moving=False)
        self.moving_yaml = self._write_tracker_yaml(moving=True)

    def _write_tracker_yaml(self, moving: bool) -> str:
        tracker_type = self.t_cfg.get("type", "botsort")
        proximity = 0.35 if moving else float(self.t_cfg.get("proximity_thresh", 0.50))
        appearance = 0.30 if moving else float(self.t_cfg.get("appearance_thresh", 0.25))
        gmc_method = "sparseOptFlow"

        data = {
            "tracker_type": tracker_type,
            "track_high_thresh": float(self.t_cfg.get("track_high_thresh", 0.45)),
            "track_low_thresh": float(self.t_cfg.get("track_low_thresh", 0.15)),
            "new_track_thresh": float(self.t_cfg.get("new_track_thresh", 0.55)),
            "track_buffer": int(self.t_cfg.get("track_buffer", 45)),
            "match_thresh": float(self.t_cfg.get("match_thresh", 0.80)),
            "fuse_score": True,
            "gmc_method": gmc_method,
            "proximity_thresh": proximity,
            "appearance_thresh": appearance,
            "with_reid": bool(self.t_cfg.get("with_reid", True)),
            "model": "auto",
        }
        filename = "botsort_moving.yaml" if moving else "botsort_fixed.yaml"
        path = os.path.join(self._temp_dir.name, filename)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        return path

    def step(
        self,
        frame,
        frame_idx: int,
        timestamp: float,
        camera_state: str,
    ) -> List[Detection]:
        yaml_path = self.moving_yaml if camera_state == "moving" else self.fixed_yaml
        return self.detector.detect_and_track(frame, frame_idx, timestamp, yaml_path)

    def cleanup(self) -> None:
        self._temp_dir.cleanup()
