from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from app.models import (
    CandidateRecord,
    Detection,
    IdentityRecord,
    IdentityState,
)
from app.reidentifier import AppearanceEmbedder
from app.frame_quality import compute_iou


class GlobalIdentityManager:
    def __init__(self, config: dict, embedder: Optional[AppearanceEmbedder] = None):
        self.reid_cfg = config.get("reid", {})
        self.conf_cfg = config.get("confirmation", {})
        self.ev_cfg = config.get("events", {})
        self.embedder = embedder

        self.sim_threshold = float(self.reid_cfg.get("similarity_threshold", 0.68))
        self.ambiguous_margin = float(self.reid_cfg.get("ambiguous_margin", 0.05))
        self.uncertain_lower = float(self.reid_cfg.get("uncertain_lower_bound", 0.52))
        self.max_embeddings = int(self.reid_cfg.get("max_embeddings_per_identity", 12))
        self.max_crops = int(self.reid_cfg.get("max_crops_per_identity", 5))
        self.min_emb_interval = float(self.reid_cfg.get("min_embedding_interval_sec", 0.4))

        self.min_obs = int(self.conf_cfg.get("min_observations", 2))
        self.window_frames = int(self.conf_cfg.get("window_frames", 3))
        self.max_confirm_sec = float(self.conf_cfg.get("max_confirmation_window_sec", 6.0))
        self.cand_timeout_sec = float(self.conf_cfg.get("candidate_timeout_sec", 4.0))
        self.single_obs_exc = bool(self.conf_cfg.get("single_obs_exception", True))
        self.single_obs_conf = float(self.conf_cfg.get("single_obs_min_conf", 0.86))
        self.single_obs_qual = float(self.conf_cfg.get("single_obs_min_quality", 0.72))
        self.crop_qual_thresh = float(self.conf_cfg.get("crop_quality_threshold", 0.38))

        self.return_absence_thresh = float(self.ev_cfg.get("return_absence_threshold_sec", 60.0))

        self.identities: Dict[str, IdentityRecord] = {}
        self.candidates: Dict[str, CandidateRecord] = {}
        self.track_to_global: Dict[int, str] = {}
        self.track_to_candidate: Dict[int, str] = {}
        self._last_emb_ts: Dict[str, float] = {}
        self._next_person_int = 1
        self._next_cand_int = 1
        self._processed_step = 0

    def _allocate_global_id(self) -> str:
        gid = f"person_{self._next_person_int:04d}"
        self._next_person_int += 1
        return gid

    def _allocate_cand_id(self) -> str:
        cid = f"cand_{self._next_cand_int:04d}"
        self._next_cand_int += 1
        return cid

    def compute_gallery_similarity(
        self,
        embedding: np.ndarray,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        exclude = exclude_ids or set()
        scores: List[Tuple[str, float]] = []

        for gid, rec in self.identities.items():
            if gid in exclude or not rec.embeddings:
                continue
            sims = sorted([float(np.dot(embedding, emb)) for emb in rec.embeddings], reverse=True)
            if len(sims) >= 2:
                score = 0.65 * sims[0] + 0.35 * sims[1]
            else:
                score = sims[0]
            scores.append((gid, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _update_identity_gallery(self, rec: IdentityRecord, det: Detection) -> None:
        if det.embedding is None or det.quality_score < self.crop_qual_thresh:
            return

        last_ts = self._last_emb_ts.get(rec.global_id, -999.0)
        if (det.timestamp - last_ts) < self.min_emb_interval and len(rec.embeddings) >= 3:
            return

        if rec.embeddings:
            sims = [float(np.dot(det.embedding, e)) for e in rec.embeddings]
            max_sim = max(sims)
            max_idx = int(np.argmax(sims))
            if max_sim > 0.96:
                if det.quality_score > rec.embedding_qualities[max_idx]:
                    rec.embeddings[max_idx] = det.embedding
                    rec.embedding_qualities[max_idx] = det.quality_score
                return

        if len(rec.embeddings) < self.max_embeddings:
            rec.embeddings.append(det.embedding)
            rec.embedding_qualities.append(det.quality_score)
        else:
            min_idx = int(np.argmin(rec.embedding_qualities[1:])) + 1
            if det.quality_score >= rec.embedding_qualities[min_idx] * 0.9:
                rec.embeddings[min_idx] = det.embedding
                rec.embedding_qualities[min_idx] = det.quality_score

        self._last_emb_ts[rec.global_id] = det.timestamp

        if det.crop is not None and det.quality_score >= self.crop_qual_thresh:
            rec.crops.append((det.quality_score, det.crop))
            rec.crops.sort(key=lambda x: x[0], reverse=True)
            if len(rec.crops) > self.max_crops:
                rec.crops = rec.crops[: self.max_crops]

    def process_frame_detections(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp: float,
        camera_state: str,
    ) -> Tuple[List[Tuple[IdentityRecord, Detection, Dict]], List[Tuple[IdentityRecord, Detection, Dict]]]:
        self._processed_step += 1
        newly_confirmed: List[Tuple[IdentityRecord, Detection, Dict]] = []
        confirmed_returns: List[Tuple[IdentityRecord, Detection, Dict]] = []
        assigned_global_ids_this_frame: Set[str] = set()

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        for det in sorted_dets:
            if det.embedding is None and self.embedder is not None and det.crop is not None:
                if det.quality_score >= (self.crop_qual_thresh * 0.75):
                    det.embedding = self.embedder.extract(det.crop)

            if det.track_id is not None and det.track_id in self.track_to_global:
                gid = self.track_to_global[det.track_id]
                if gid not in assigned_global_ids_this_frame and gid in self.identities:
                    rec = self.identities[gid]
                    hijacked = False
                    if det.embedding is not None and rec.embeddings and det.quality_score >= self.crop_qual_thresh:
                        sim = max(float(np.dot(det.embedding, e)) for e in rec.embeddings)
                        det.match_score = sim
                        if sim < (self.uncertain_lower - 0.12):
                            hijacked = True
                            det.uncertainty_flags.append("tracker_id_switch_suspected")

                    if not hijacked:
                        assigned_global_ids_this_frame.add(gid)
                        ret_event = self._record_identity_observation(rec, det, self._processed_step)
                        if ret_event is not None:
                            confirmed_returns.append(ret_event)
                        continue
                    else:
                        del self.track_to_global[det.track_id]

            if det.embedding is not None:
                matches = self.compute_gallery_similarity(
                    det.embedding, exclude_ids=assigned_global_ids_this_frame
                )
            else:
                matches = []

            best_gid, best_score = matches[0] if matches else (None, 0.0)
            second_score = matches[1][1] if len(matches) > 1 else 0.0
            det.match_score = best_score

            effective_thresh = self.sim_threshold
            if best_gid is not None:
                target_rec = self.identities[best_gid]
                time_gap = timestamp - target_rec.last_seen_ts
                if time_gap <= 2.5 and target_rec.state in (IdentityState.ACTIVE, IdentityState.TEMPORARILY_MISSING):
                    if camera_state == "moving" or len(self.identities) == 1:
                        effective_thresh = max(self.uncertain_lower + 0.04, self.sim_threshold - 0.08)

            is_ambiguous = (
                len(matches) >= 2
                and best_score >= effective_thresh
                and (best_score - second_score) < self.ambiguous_margin
            )
            if is_ambiguous:
                det.uncertainty_flags.append("ambiguous_gallery_match")

            if best_gid is not None and best_score >= effective_thresh and not is_ambiguous:
                rec = self.identities[best_gid]
                assigned_global_ids_this_frame.add(best_gid)
                if det.track_id is not None:
                    self.track_to_global[det.track_id] = best_gid
                ret_event = self._record_identity_observation(rec, det, self._processed_step)
                if ret_event is not None:
                    confirmed_returns.append(ret_event)
                continue

            new_ev, ret_ev = self._process_candidate_observation(
                det=det,
                step_idx=self._processed_step,
                best_gid=best_gid,
                best_score=best_score,
                assigned_gids=assigned_global_ids_this_frame,
            )
            if new_ev is not None:
                newly_confirmed.append(new_ev)
            if ret_ev is not None:
                confirmed_returns.append(ret_ev)

        self._prune_expired_candidates(timestamp)
        return newly_confirmed, confirmed_returns

    def _record_identity_observation(
        self,
        rec: IdentityRecord,
        det: Detection,
        step_idx: int,
    ) -> Optional[Tuple[IdentityRecord, Detection, Dict]]:
        det.global_id = rec.global_id
        if det.track_id is not None:
            rec.associated_track_ids.add(det.track_id)

        absence_duration = det.timestamp - rec.last_confirmed_visible_ts
        rec.last_seen_frame = det.frame_idx
        rec.last_seen_ts = det.timestamp
        rec.num_observations += 1
        rec.confidence_history.append(det.confidence)
        rec.quality_history.append(det.quality_score)
        if det.match_score > 0:
            rec.match_score_history.append(det.match_score)
        rec.uncertainty_flags.update(det.uncertainty_flags)
        self._update_identity_gallery(rec, det)

        if rec.state in (IdentityState.ABSENT, IdentityState.RETURN_CANDIDATE):
            rec.state = IdentityState.RETURN_CANDIDATE
            if det.quality_score >= self.crop_qual_thresh:
                rec.return_obs_frames.append(step_idx)
            rec.return_obs_frames = [
                s for s in rec.return_obs_frames if (step_idx - s) < self.window_frames
            ]

            single_ok = (
                self.single_obs_exc
                and det.confidence >= self.single_obs_conf
                and det.quality_score >= self.single_obs_qual
            )
            if len(rec.return_obs_frames) >= self.min_obs or single_ok:
                rec.state = IdentityState.ACTIVE
                rec.appearance_episodes += 1
                rec.return_obs_frames.clear()
                if rec.disappearance_history and rec.disappearance_history[-1][1] is None:
                    start_abs = rec.disappearance_history[-1][0]
                    rec.disappearance_history[-1] = (start_abs, det.timestamp)
                rec.last_confirmed_visible_ts = det.timestamp
                diag = {
                    "match_score": round(det.match_score, 4),
                    "absence_duration_sec": round(absence_duration, 2),
                    "episode_id": rec.appearance_episodes,
                }
                return rec, det, diag
            return None

        if rec.state == IdentityState.TEMPORARILY_MISSING:
            if absence_duration >= self.return_absence_thresh:
                rec.state = IdentityState.RETURN_CANDIDATE
                if det.quality_score >= self.crop_qual_thresh:
                    rec.return_obs_frames = [step_idx]
                return None
            rec.state = IdentityState.ACTIVE

        rec.last_confirmed_visible_ts = det.timestamp
        return None

    def _process_candidate_observation(
        self,
        det: Detection,
        step_idx: int,
        best_gid: Optional[str],
        best_score: float,
        assigned_gids: Set[str],
    ) -> Tuple[Optional[Tuple[IdentityRecord, Detection, Dict]], Optional[Tuple[IdentityRecord, Detection, Dict]]]:
        cand: Optional[CandidateRecord] = None

        if det.track_id is not None and det.track_id in self.track_to_candidate:
            cid = self.track_to_candidate[det.track_id]
            cand = self.candidates.get(cid)

        if cand is None:
            for existing_cand in self.candidates.values():
                if existing_cand.best_bbox is not None and compute_iou(det.bbox, existing_cand.best_bbox) >= 0.45:
                    cand = existing_cand
                    break
                if det.embedding is not None and existing_cand.embeddings:
                    sim = max(float(np.dot(det.embedding, e)) for e in existing_cand.embeddings)
                    if sim >= self.sim_threshold:
                        cand = existing_cand
                        break

        if cand is None:
            cid = self._allocate_cand_id()
            cand = CandidateRecord(
                candidate_id=cid,
                first_seen_frame=det.frame_idx,
                first_seen_ts=det.timestamp,
                last_seen_frame=det.frame_idx,
                last_seen_ts=det.timestamp,
            )
            self.candidates[cid] = cand

        if det.track_id is not None:
            cand.track_ids.add(det.track_id)
            self.track_to_candidate[det.track_id] = cand.candidate_id

        cand.last_seen_frame = det.frame_idx
        cand.last_seen_ts = det.timestamp
        cand.observation_frames.append(step_idx)
        cand.confidences.append(det.confidence)
        cand.qualities.append(det.quality_score)
        cand.best_bbox = det.bbox
        cand.uncertainty_flags.update(det.uncertainty_flags)

        is_valid_obs = (
            det.quality_score >= self.crop_qual_thresh
            and det.embedding is not None
            and "below_min_dimensions" not in det.uncertainty_flags
        )
        if is_valid_obs:
            cand.valid_observation_frames.append(step_idx)
            cand.embeddings.append(det.embedding)
            if det.quality_score >= cand.best_quality and det.crop is not None:
                cand.best_quality = det.quality_score
                cand.best_crop = det.crop

        if best_gid is not None and self.uncertain_lower <= best_score < self.sim_threshold:
            cand.provisional_target_id = best_gid
            cand.uncertainty_flags.add("uncertain_reid_score")
            det.uncertainty_flags.append("provisional_candidate")
            if len(cand.embeddings) >= (self.min_obs + 1) and best_gid not in assigned_gids:
                mean_emb = AppearanceEmbedder.l2_normalize(np.mean(cand.embeddings, axis=0))
                re_matches = self.compute_gallery_similarity(mean_emb, exclude_ids=assigned_gids)
                if re_matches and re_matches[0][1] >= (self.sim_threshold - 0.04):
                    matched_id = re_matches[0][1]
                    rec = self.identities[re_matches[0][0]]
                    assigned_gids.add(rec.global_id)
                    self._remove_candidate(cand.candidate_id)
                    det.match_score = float(matched_id)
                    ret_ev = self._record_identity_observation(rec, det, step_idx)
                    return None, ret_ev
            return None, None

        recent_valid = [s for s in cand.valid_observation_frames if (step_idx - s) < self.window_frames]
        single_exception_met = (
            self.single_obs_exc
            and is_valid_obs
            and det.confidence >= self.single_obs_conf
            and det.quality_score >= self.single_obs_qual
            and not det.touches_boundary
            and best_score < self.uncertain_lower
        )

        if len(recent_valid) >= self.min_obs or single_exception_met:
            gid = self._allocate_global_id()
            rec = IdentityRecord(
                global_id=gid,
                state=IdentityState.ACTIVE,
                first_seen_frame=cand.first_seen_frame,
                first_seen_ts=cand.first_seen_ts,
                last_seen_frame=det.frame_idx,
                last_seen_ts=det.timestamp,
                last_confirmed_visible_ts=det.timestamp,
                associated_track_ids=set(cand.track_ids),
                embeddings=list(cand.embeddings[: self.max_embeddings]),
                embedding_qualities=list(cand.qualities[-len(cand.embeddings) :]),
                confidence_history=list(cand.confidences),
                quality_history=list(cand.qualities),
                num_observations=len(cand.observation_frames),
                uncertainty_flags=set(cand.uncertainty_flags),
            )
            if cand.best_crop is not None:
                rec.crops.append((cand.best_quality, cand.best_crop))

            self.identities[gid] = rec
            assigned_gids.add(gid)
            det.global_id = gid
            for tid in cand.track_ids:
                self.track_to_global[tid] = gid
            self._remove_candidate(cand.candidate_id)

            confirm_diag = {
                "observations_used": len(recent_valid),
                "single_frame_exception": bool(single_exception_met and len(recent_valid) < self.min_obs),
                "best_gallery_similarity": round(best_score, 4),
            }
            return (rec, det, confirm_diag), None

        return None, None

    def _remove_candidate(self, cid: str) -> None:
        cand = self.candidates.pop(cid, None)
        if cand is not None:
            for tid in cand.track_ids:
                self.track_to_candidate.pop(tid, None)

    def _prune_expired_candidates(self, current_ts: float) -> None:
        to_delete = []
        for cid, cand in self.candidates.items():
            unseen_sec = current_ts - cand.last_seen_ts
            age_sec = current_ts - cand.first_seen_ts
            if unseen_sec > self.cand_timeout_sec or age_sec > self.max_confirm_sec:
                to_delete.append(cid)
        for cid in to_delete:
            self._remove_candidate(cid)
