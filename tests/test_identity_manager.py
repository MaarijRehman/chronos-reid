import numpy as np
from app.models import Detection
from app.identity_manager import GlobalIdentityManager
from app.reidentifier import AppearanceEmbedder


def make_base_config() -> dict:
    return {
        "reid": {
            "similarity_threshold": 0.70,
            "ambiguous_margin": 0.05,
            "uncertain_lower_bound": 0.52,
            "max_embeddings_per_identity": 10,
            "max_crops_per_identity": 3,
            "min_embedding_interval_sec": 0.0,
        },
        "confirmation": {
            "min_observations": 2,
            "window_frames": 3,
            "max_confirmation_window_sec": 6.0,
            "candidate_timeout_sec": 4.0,
            "single_obs_exception": True,
            "single_obs_min_conf": 0.88,
            "single_obs_min_quality": 0.75,
            "min_crop_width": 32,
            "min_crop_height": 64,
            "crop_quality_threshold": 0.40,
        },
        "events": {"return_absence_threshold_sec": 60.0},
    }


def make_orthogonal_emb(index: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index % dim] = 1.0
    return v


def make_det(
    frame_idx: int,
    ts: float,
    emb: np.ndarray,
    track_id: int = 1,
    conf: float = 0.80,
    qual: float = 0.65,
    bbox=(100, 100, 200, 300),
    touches_boundary: bool = False,
) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=conf,
        frame_idx=frame_idx,
        timestamp=ts,
        track_id=track_id,
        crop=np.zeros((200, 100, 3), dtype=np.uint8),
        quality_score=qual,
        sharpness_score=0.7,
        touches_boundary=touches_boundary,
        embedding=emb,
    )


def test_5_new_tracker_id_does_not_duplicate_identity():
    mgr = GlobalIdentityManager(make_base_config())
    emb_a = make_orthogonal_emb(0)
    mgr.process_frame_detections([make_det(1, 0.2, emb_a, track_id=1)], 1, 0.2, "fixed")
    new_ids, _ = mgr.process_frame_detections([make_det(2, 0.4, emb_a, track_id=1)], 2, 0.4, "fixed")
    assert len(new_ids) == 1
    assert len(mgr.identities) == 1

    new_ids_switch, _ = mgr.process_frame_detections([make_det(3, 0.6, emb_a, track_id=99)], 3, 0.6, "fixed")
    assert len(new_ids_switch) == 0
    assert len(mgr.identities) == 1
    assert 99 in mgr.identities["person_0001"].associated_track_ids


def test_10_partial_person_confirmed_across_multiple_observations():
    mgr = GlobalIdentityManager(make_base_config())
    emb_edge = make_orthogonal_emb(1)
    d1 = make_det(1, 0.2, emb_edge, track_id=5, conf=0.92, qual=0.80, bbox=(0, 80, 70, 260), touches_boundary=True)
    new_1, _ = mgr.process_frame_detections([d1], 1, 0.2, "fixed")
    assert len(new_1) == 0

    d2 = make_det(2, 0.4, emb_edge, track_id=5, conf=0.82, qual=0.55, bbox=(2, 80, 75, 260), touches_boundary=True)
    new_2, _ = mgr.process_frame_detections([d2], 2, 0.4, "fixed")
    assert len(new_2) == 1
    assert new_2[0][0].global_id == "person_0001"


def test_11_uncertain_low_quality_detection_not_confirmed():
    mgr = GlobalIdentityManager(make_base_config())
    emb = make_orthogonal_emb(2)
    for f in range(1, 10):
        low_q_det = make_det(f, f * 0.2, emb, track_id=7, conf=0.35, qual=0.20)
        new_ids, _ = mgr.process_frame_detections([low_q_det], f, f * 0.2, "fixed")
        assert len(new_ids) == 0
    assert len(mgr.identities) == 0


def test_12_temporary_occlusion_does_not_create_new_identity():
    mgr = GlobalIdentityManager(make_base_config())
    emb = make_orthogonal_emb(3)
    mgr.process_frame_detections([make_det(1, 0.2, emb, track_id=1)], 1, 0.2, "fixed")
    mgr.process_frame_detections([make_det(2, 0.4, emb, track_id=1)], 2, 0.4, "fixed")
    assert len(mgr.identities) == 1

    mgr.process_frame_detections([], 15, 3.0, "fixed")
    new_ids, ret_ids = mgr.process_frame_detections([make_det(27, 5.4, emb, track_id=12)], 27, 5.4, "fixed")
    assert len(new_ids) == 0
    assert len(ret_ids) == 0
    assert len(mgr.identities) == 1


def test_13_similar_looking_people_not_merged():
    mgr = GlobalIdentityManager(make_base_config())
    emb_a = make_orthogonal_emb(0)
    emb_b = AppearanceEmbedder.l2_normalize(0.60 * make_orthogonal_emb(0) + 0.80 * make_orthogonal_emb(1))

    for f in (1, 2):
        mgr.process_frame_detections(
            [
                make_det(f, f * 0.2, emb_a, track_id=1, bbox=(50, 50, 150, 250)),
                make_det(f, f * 0.2, emb_b, track_id=2, bbox=(300, 50, 400, 250)),
            ],
            f,
            f * 0.2,
            "fixed",
        )
    assert len(mgr.identities) == 2

    emb_c = AppearanceEmbedder.l2_normalize(0.50 * emb_a + 0.50 * emb_b + 0.45 * make_orthogonal_emb(2))
    det_c = make_det(3, 0.6, emb_c, track_id=3, bbox=(500, 50, 600, 250))
    mgr.process_frame_detections([det_c], 3, 0.6, "fixed")
    assert det_c.global_id is None


def test_14_camera_movement_does_not_trigger_new_person():
    mgr = GlobalIdentityManager(make_base_config())
    emb = make_orthogonal_emb(4)
    mgr.process_frame_detections([make_det(1, 0.2, emb, track_id=1, bbox=(50, 100, 150, 300))], 1, 0.2, "moving")
    mgr.process_frame_detections([make_det(2, 0.4, emb, track_id=1, bbox=(80, 100, 180, 300))], 2, 0.4, "moving")
    assert len(mgr.identities) == 1

    slightly_shifted_emb = AppearanceEmbedder.l2_normalize(0.82 * emb + 0.57 * make_orthogonal_emb(5))
    new_ids, _ = mgr.process_frame_detections(
        [make_det(3, 0.6, slightly_shifted_emb, track_id=44, bbox=(480, 120, 580, 320))],
        3,
        0.6,
        "moving",
    )
    assert len(new_ids) == 0
    assert len(mgr.identities) == 1


def test_15_single_frame_appearance_exception_policy():
    cfg = make_base_config()
    mgr = GlobalIdentityManager(cfg)
    emb_strong = make_orthogonal_emb(6)
    emb_weak = make_orthogonal_emb(7)

    new_strong, _ = mgr.process_frame_detections(
        [make_det(1, 0.2, emb_strong, track_id=10, conf=0.93, qual=0.82)], 1, 0.2, "fixed"
    )
    assert len(new_strong) == 1

    new_weak, _ = mgr.process_frame_detections(
        [make_det(2, 0.4, emb_weak, track_id=11, conf=0.70, qual=0.65)], 2, 0.4, "fixed"
    )
    assert len(new_weak) == 0
