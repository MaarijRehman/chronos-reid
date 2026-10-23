from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np
from app.models import (
    FinalizedScreenshotEvent,
    FramePacket,
    PendingEvent,
)


class CaptureEngine:
    def __init__(self, config: dict):
        ev_cfg = config.get("events", {})
        self.grouping_window_sec = float(ev_cfg.get("grouping_window_sec", 1.0))
        self.max_selection_delay_sec = float(ev_cfg.get("max_selection_delay_sec", 1.5))
        self.buffer_size = int(ev_cfg.get("frame_buffer_size", 25))

        self.frame_buffer: Deque[FramePacket] = deque(maxlen=self.buffer_size)
        self.pending_group: List[PendingEvent] = []
        self.group_opened_ts: float = -1.0
        self.finalized_event_keys: Set[str] = set()
        self._event_counter = 1

    def score_frame_for_group(self, packet: FramePacket, target_gids: Set[str]) -> float:
        det_by_gid = {d.global_id: d for d in packet.detections if d.global_id is not None}
        present_targets = [det_by_gid[gid] for gid in target_gids if gid in det_by_gid]

        if not present_targets:
            return -1.0

        coverage_ratio = len(present_targets) / float(max(1, len(target_gids)))
        mean_target_quality = float(np.mean([d.quality_score for d in present_targets]))
        boundary_penalty = sum(0.08 for d in present_targets if d.touches_boundary)
        occlusion_penalty = sum(0.06 for d in present_targets if d.is_occluded)

        score = (
            0.55 * coverage_ratio
            + 0.30 * mean_target_quality
            + 0.15 * packet.frame_quality_score
            - boundary_penalty
            - occlusion_penalty
        )
        return float(score)

    def step(
        self,
        packet: FramePacket,
        new_events: List[PendingEvent],
    ) -> List[Tuple[FinalizedScreenshotEvent, FramePacket]]:
        self.frame_buffer.append(packet)
        ready_outputs: List[Tuple[FinalizedScreenshotEvent, FramePacket]] = []

        for ev in new_events:
            if ev.event_key in self.finalized_event_keys:
                continue
            if any(p.event_key == ev.event_key for p in self.pending_group):
                continue

            if self.pending_group and (ev.trigger_ts - self.group_opened_ts) > self.grouping_window_sec:
                flushed = self._finalize_current_group()
                if flushed is not None:
                    ready_outputs.append(flushed)

            if not self.pending_group:
                self.group_opened_ts = ev.trigger_ts
            self.pending_group.append(ev)

        if self.pending_group:
            elapsed_since_open = packet.timestamp - self.group_opened_ts
            last_event_ts = max(e.trigger_ts for e in self.pending_group)
            elapsed_since_last = packet.timestamp - last_event_ts

            if elapsed_since_open >= max(self.grouping_window_sec, self.max_selection_delay_sec) or elapsed_since_last >= self.max_selection_delay_sec:
                flushed = self._finalize_current_group()
                if flushed is not None:
                    ready_outputs.append(flushed)

        return ready_outputs

    def flush(self) -> List[Tuple[FinalizedScreenshotEvent, FramePacket]]:
        if not self.pending_group:
            return []
        flushed = self._finalize_current_group()
        return [flushed] if flushed is not None else []

    def _finalize_current_group(self) -> Optional[Tuple[FinalizedScreenshotEvent, FramePacket]]:
        if not self.pending_group or not self.frame_buffer:
            self.pending_group.clear()
            return None

        events = list(self.pending_group)
        self.pending_group.clear()

        target_gids = {e.global_id for e in events}
        min_ts = min(e.trigger_ts for e in events) - 0.25

        best_packet: Optional[FramePacket] = None
        best_score = -999.0

        for pkt in self.frame_buffer:
            if pkt.timestamp < min_ts:
                continue
            s = self.score_frame_for_group(pkt, target_gids)
            if s > best_score:
                best_score = s
                best_packet = pkt

        if best_packet is None:
            best_packet = self.frame_buffer[-1]
            best_score = best_packet.frame_quality_score

        for e in events:
            self.finalized_event_keys.add(e.event_key)

        ev_id = f"event_{self._event_counter:04d}"
        self._event_counter += 1

        event_types = sorted({e.event_type.value for e in events})
        primary_type = "new_person" if "new_person" in event_types else "return"
        trig_ids = sorted(target_gids)
        filename = f"{ev_id}_f{best_packet.frame_idx:06d}_{primary_type}_{'-'.join(trig_ids)}.jpg"

        bboxes: Dict[str, List[int]] = {}
        confidences: Dict[str, float] = {}
        tracker_ids: List[int] = []
        uncertainty: Set[str] = set()

        for det in best_packet.detections:
            if det.track_id is not None:
                tracker_ids.append(det.track_id)
            uncertainty.update(det.uncertainty_flags)
            key = det.global_id if det.global_id else f"unconfirmed_track_{det.track_id}"
            bboxes[key] = list(det.bbox)
            confidences[key] = round(det.confidence, 4)

        reid_diag = {e.global_id: e.match_diagnostics for e in events}
        conf_info = {e.global_id: e.confirmation_info for e in events}
        all_visible = sorted(best_packet.active_global_ids)

        final_ev = FinalizedScreenshotEvent(
            event_id=ev_id,
            event_keys=[e.event_key for e in events],
            screenshot_filename=filename,
            frame_number=best_packet.frame_idx,
            timestamp=round(best_packet.timestamp, 3),
            event_types=event_types,
            primary_event_type=primary_type,
            triggering_global_ids=trig_ids,
            all_visible_global_ids=all_visible,
            tracker_ids=sorted(set(tracker_ids)),
            bboxes=bboxes,
            confidences=confidences,
            reid_diagnostics=reid_diag,
            confirmation_info=conf_info,
            frame_quality_score=round(float(max(0.0, best_score)), 4),
            uncertainty_flags=sorted(uncertainty),
            camera_motion_state=best_packet.camera_mode_state,
        )
        return final_ev, best_packet
