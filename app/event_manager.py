from typing import Dict, List, Set, Tuple
from app.models import (
    Detection,
    EventType,
    IdentityRecord,
    IdentityState,
    PendingEvent,
)


class AppearanceEventManager:
    def __init__(self, config: dict):
        ev_cfg = config.get("events", {})
        self.return_absence_thresh = float(ev_cfg.get("return_absence_threshold_sec", 60.0))
        self.enable_return_events = bool(ev_cfg.get("enable_return_events", True))
        self.processed_event_keys: Set[str] = set()
        self.new_person_count = 0
        self.return_event_count = 0

    @staticmethod
    def build_event_key(global_id: str, event_type: EventType, episode_id: int) -> str:
        return f"{global_id}:{event_type.value}:ep{episode_id}"

    def step(
        self,
        identities: Dict[str, IdentityRecord],
        newly_confirmed: List[Tuple[IdentityRecord, Detection, Dict]],
        confirmed_returns: List[Tuple[IdentityRecord, Detection, Dict]],
        frame_idx: int,
        timestamp: float,
    ) -> List[PendingEvent]:
        emitted_events: List[PendingEvent] = []
        visible_ids_this_frame: Set[str] = set()

        for rec, det, diag in newly_confirmed:
            visible_ids_this_frame.add(rec.global_id)
            ev_key = self.build_event_key(rec.global_id, EventType.NEW_PERSON, rec.appearance_episodes)
            if ev_key in self.processed_event_keys or ("new_person", rec.appearance_episodes) in rec.captured_episodes:
                continue

            self.processed_event_keys.add(ev_key)
            rec.captured_episodes.add(("new_person", rec.appearance_episodes))
            self.new_person_count += 1

            ev = PendingEvent(
                event_key=ev_key,
                event_type=EventType.NEW_PERSON,
                global_id=rec.global_id,
                episode_id=rec.appearance_episodes,
                trigger_frame=frame_idx,
                trigger_ts=timestamp,
                confirmation_info=diag,
                match_diagnostics={"initial_gallery_max_sim": diag.get("best_gallery_similarity", 0.0)},
            )
            rec.event_history.append({
                "event_key": ev_key,
                "event_type": EventType.NEW_PERSON.value,
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "episode_id": rec.appearance_episodes,
            })
            emitted_events.append(ev)

        for rec, det, diag in confirmed_returns:
            visible_ids_this_frame.add(rec.global_id)
            if not self.enable_return_events:
                continue

            ev_key = self.build_event_key(rec.global_id, EventType.RETURN, rec.appearance_episodes)
            if ev_key in self.processed_event_keys or ("return", rec.appearance_episodes) in rec.captured_episodes:
                continue

            self.processed_event_keys.add(ev_key)
            rec.captured_episodes.add(("return", rec.appearance_episodes))
            self.return_event_count += 1

            ev = PendingEvent(
                event_key=ev_key,
                event_type=EventType.RETURN,
                global_id=rec.global_id,
                episode_id=rec.appearance_episodes,
                trigger_frame=frame_idx,
                trigger_ts=timestamp,
                confirmation_info={"episode_id": rec.appearance_episodes, "confirmed_return": True},
                match_diagnostics=diag,
            )
            rec.event_history.append({
                "event_key": ev_key,
                "event_type": EventType.RETURN.value,
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "episode_id": rec.appearance_episodes,
                "absence_duration_sec": diag.get("absence_duration_sec", 0.0),
            })
            emitted_events.append(ev)

        for gid, rec in identities.items():
            if rec.last_seen_frame == frame_idx and rec.last_seen_ts == timestamp:
                visible_ids_this_frame.add(gid)
                continue

            absence_sec = timestamp - rec.last_confirmed_visible_ts
            if rec.state == IdentityState.ACTIVE and absence_sec > 0.0:
                rec.state = IdentityState.TEMPORARILY_MISSING
                if not rec.disappearance_history or rec.disappearance_history[-1][1] is not None:
                    rec.disappearance_history.append((rec.last_confirmed_visible_ts, None))

            if rec.state in (IdentityState.TEMPORARILY_MISSING, IdentityState.RETURN_CANDIDATE):
                if absence_sec >= self.return_absence_thresh:
                    if (timestamp - rec.last_seen_ts) > 1.5:
                        rec.state = IdentityState.ABSENT
                        rec.return_obs_frames.clear()

        return emitted_events
