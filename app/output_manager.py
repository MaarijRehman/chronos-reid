import csv
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional
import cv2
import numpy as np
from app.models import FinalizedScreenshotEvent, FramePacket, IdentityRecord


class OutputManager:
    def __init__(self, video_path: str, config: dict, custom_out_dir: Optional[str] = None):
        out_cfg = config.get("output", {})
        base_dir = custom_out_dir if custom_out_dir else out_cfg.get("base_dir", "output")
        video_stem = os.path.splitext(os.path.basename(video_path))[0]

        self.root_dir = os.path.join(base_dir, video_stem)
        self.screenshots_dir = os.path.join(self.root_dir, "screenshots")
        self.crops_dir = os.path.join(self.root_dir, "person_crops")
        self.metadata_dir = os.path.join(self.root_dir, "metadata")
        self.debug_dir = os.path.join(self.root_dir, "debug")

        for d in (self.screenshots_dir, self.crops_dir, self.metadata_dir, self.debug_dir):
            os.makedirs(d, exist_ok=True)

        self.jpeg_quality = int(out_cfg.get("jpeg_quality", 92))
        self.save_crops = bool(out_cfg.get("save_person_crops", True))
        self.save_debug_frames = bool(out_cfg.get("save_annotated_debug", False))
        self.save_debug_video = bool(out_cfg.get("save_debug_video", False))

        self.saved_events: List[FinalizedScreenshotEvent] = []
        self._written_filenames: set = set()
        self._video_writer: Optional[cv2.VideoWriter] = None

    def save_event_screenshot(self, event: FinalizedScreenshotEvent, packet: FramePacket) -> bool:
        if event.screenshot_filename in self._written_filenames:
            return False

        out_path = os.path.join(self.screenshots_dir, event.screenshot_filename)
        cv2.imwrite(out_path, packet.frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        self._written_filenames.add(event.screenshot_filename)
        self.saved_events.append(event)

        if self.save_crops:
            for det in packet.detections:
                if det.global_id in event.triggering_global_ids and det.crop is not None and det.crop.size > 0:
                    crop_name = f"{event.event_id}_{det.global_id}_f{packet.frame_idx:06d}.jpg"
                    cv2.imwrite(
                        os.path.join(self.crops_dir, crop_name),
                        det.crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )

        if self.save_debug_frames:
            annotated = self.draw_debug_annotations(packet, highlight_ids=set(event.triggering_global_ids))
            dbg_path = os.path.join(self.debug_dir, f"debug_{event.screenshot_filename}")
            cv2.imwrite(dbg_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        return True

    def write_debug_video_frame(self, packet: FramePacket, fps: float) -> None:
        if not self.save_debug_video:
            return
        annotated = self.draw_debug_annotations(packet)
        if self._video_writer is None:
            h, w = annotated.shape[:2]
            vid_path = os.path.join(self.debug_dir, "debug_tracked.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))
        self._video_writer.write(annotated)

    @staticmethod
    def draw_debug_annotations(packet: FramePacket, highlight_ids: Optional[set] = None) -> np.ndarray:
        canvas = packet.frame.copy()
        highlights = highlight_ids or set()
        for det in packet.detections:
            x1, y1, x2, y2 = det.bbox
            is_trig = det.global_id in highlights
            color = (0, 255, 0) if is_trig else (255, 180, 0)
            if det.global_id is None:
                color = (0, 165, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.global_id or 'CAND'} (t:{det.track_id} c:{det.confidence:.2f} q:{det.quality_score:.2f})"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        header = f"Frame: {packet.frame_idx} | TS: {packet.timestamp:.2f}s | Cam: {packet.camera_mode_state} ({packet.camera_motion_px:.1f}px)"
        cv2.putText(canvas, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        return canvas

    def export_metadata(self, identities: Dict[str, IdentityRecord], run_stats: Dict) -> None:
        events_json_path = os.path.join(self.metadata_dir, "events.json")
        events_csv_path = os.path.join(self.metadata_dir, "events_summary.csv")
        gallery_json_path = os.path.join(self.metadata_dir, "identity_gallery.json")

        events_payload = {
            "run_summary": run_stats,
            "events": [asdict(e) for e in self.saved_events],
        }
        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump(events_payload, f, indent=2)

        csv_fields = [
            "event_id",
            "screenshot_filename",
            "frame_number",
            "timestamp",
            "primary_event_type",
            "triggering_global_ids",
            "all_visible_global_ids",
            "tracker_ids",
            "frame_quality_score",
            "camera_motion_state",
            "uncertainty_flags",
        ]
        with open(events_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for ev in self.saved_events:
                writer.writerow({
                    "event_id": ev.event_id,
                    "screenshot_filename": ev.screenshot_filename,
                    "frame_number": ev.frame_number,
                    "timestamp": ev.timestamp,
                    "primary_event_type": ev.primary_event_type,
                    "triggering_global_ids": ";".join(ev.triggering_global_ids),
                    "all_visible_global_ids": ";".join(ev.all_visible_global_ids),
                    "tracker_ids": ";".join(map(str, ev.tracker_ids)),
                    "frame_quality_score": ev.frame_quality_score,
                    "camera_motion_state": ev.camera_motion_state,
                    "uncertainty_flags": ";".join(ev.uncertainty_flags),
                })

        gallery_payload = {}
        for gid, rec in identities.items():
            gallery_payload[gid] = {
                "global_id": rec.global_id,
                "final_state": rec.state.value,
                "first_seen_frame": rec.first_seen_frame,
                "first_seen_timestamp": round(rec.first_seen_ts, 3),
                "last_seen_frame": rec.last_seen_frame,
                "last_seen_timestamp": round(rec.last_seen_ts, 3),
                "last_confirmed_visible_timestamp": round(rec.last_confirmed_visible_ts, 3),
                "associated_tracker_ids": sorted(rec.associated_track_ids),
                "num_observations": rec.num_observations,
                "appearance_episodes": rec.appearance_episodes,
                "stored_embeddings_count": len(rec.embeddings),
                "mean_confidence": round(float(np.mean(rec.confidence_history)), 4) if rec.confidence_history else 0.0,
                "mean_crop_quality": round(float(np.mean(rec.quality_history)), 4) if rec.quality_history else 0.0,
                "mean_match_score": round(float(np.mean(rec.match_score_history)), 4) if rec.match_score_history else 0.0,
                "event_history": rec.event_history,
                "disappearance_intervals": [
                    {"left_ts": round(s, 3), "returned_ts": round(e, 3) if e is not None else None}
                    for s, e in rec.disappearance_history
                ],
                "uncertainty_flags": sorted(rec.uncertainty_flags),
            }

        with open(gallery_json_path, "w", encoding="utf-8") as f:
            json.dump(gallery_payload, f, indent=2)

        if self._video_writer is not None:
            self._video_writer.release()
