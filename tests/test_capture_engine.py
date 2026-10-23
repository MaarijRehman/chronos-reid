import os
import pytest
import numpy as np
from app.video_reader import VideoReader, VideoReaderError
from app.identity_manager import GlobalIdentityManager
from app.event_manager import AppearanceEventManager
from app.capture_engine import CaptureEngine
from app.output_manager import OutputManager
from app.models import FramePacket, PendingEvent, EventType
from tests.test_identity_manager import make_base_config, make_det, make_orthogonal_emb


def test_9_two_new_people_arriving_together_grouped_into_one_screenshot(tmp_path):
    cfg = make_base_config()
    cfg["events"]["grouping_window_sec"] = 1.0
    cfg["events"]["max_selection_delay_sec"] = 1.0
    mgr = GlobalIdentityManager(cfg)
    ev_mgr = AppearanceEventManager(cfg)
    cap_eng = CaptureEngine(cfg)

    emb_a = make_orthogonal_emb(0)
    emb_b = make_orthogonal_emb(1)
    dummy_frame = np.full((360, 640, 3), 128, dtype=np.uint8)

    finalized = []
    timeline = [
        (1, 1.0, [make_det(1, 1.0, emb_a, track_id=1, bbox=(50, 50, 150, 250))]),
        (2, 1.2, [make_det(2, 1.2, emb_a, track_id=1, bbox=(50, 50, 150, 250))]),
        (3, 1.4, [
            make_det(3, 1.4, emb_a, track_id=1, bbox=(50, 50, 150, 250)),
            make_det(3, 1.4, emb_b, track_id=2, bbox=(300, 50, 400, 250)),
        ]),
        (4, 1.6, [
            make_det(4, 1.6, emb_a, track_id=1, bbox=(50, 50, 150, 250)),
            make_det(4, 1.6, emb_b, track_id=2, bbox=(300, 50, 400, 250)),
        ]),
        (5, 3.0, [
            make_det(5, 3.0, emb_a, track_id=1, bbox=(50, 50, 150, 250)),
            make_det(5, 3.0, emb_b, track_id=2, bbox=(300, 50, 400, 250)),
        ]),
    ]

    for f_idx, ts, dets in timeline:
        new_c, ret_c = mgr.process_frame_detections(dets, f_idx, ts, "fixed")
        pending = ev_mgr.step(mgr.identities, new_c, ret_c, f_idx, ts)
        pkt = FramePacket(
            frame_idx=f_idx,
            timestamp=ts,
            frame=dummy_frame,
            detections=dets,
            active_global_ids={d.global_id for d in dets if d.global_id},
            camera_mode_state="fixed",
            camera_motion_px=0.0,
            frame_quality_score=0.8,
        )
        finalized.extend(cap_eng.step(pkt, pending))

    assert len(finalized) == 1
    ev, _ = finalized[0]
    assert ev.triggering_global_ids == ["person_0001", "person_0002"]


def test_16_video_with_no_people_produces_no_screenshots(tmp_path):
    cfg = make_base_config()
    mgr = GlobalIdentityManager(cfg)
    ev_mgr = AppearanceEventManager(cfg)
    cap_eng = CaptureEngine(cfg)
    out_mgr = OutputManager("empty_scene.mp4", cfg, custom_out_dir=str(tmp_path))
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    for f in range(1, 50):
        ts = f * 0.2
        new_c, ret_c = mgr.process_frame_detections([], f, ts, "fixed")
        pending = ev_mgr.step(mgr.identities, new_c, ret_c, f, ts)
        pkt = FramePacket(f, ts, dummy_frame, [], set(), "fixed", 0.0, 0.5)
        for fin_ev, best_pkt in cap_eng.step(pkt, pending):
            out_mgr.save_event_screenshot(fin_ev, best_pkt)

    assert len(out_mgr.saved_events) == 0
    assert len(os.listdir(out_mgr.screenshots_dir)) == 0


def test_17_corrupt_input_files_produce_clear_error(tmp_path):
    cfg = make_base_config()
    with pytest.raises(VideoReaderError, match="does not exist"):
        VideoReader(str(tmp_path / "nonexistent.mp4"), cfg)

    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"NOT_A_VALID_VIDEO_HEADER_12345")
    with pytest.raises(VideoReaderError):
        reader = VideoReader(str(corrupt_file), cfg)
        list(reader)


def test_18_idempotent_event_reprocessing_no_duplicate_files(tmp_path):
    cfg = make_base_config()
    cap_eng = CaptureEngine(cfg)
    out_mgr = OutputManager("test_vid.mp4", cfg, custom_out_dir=str(tmp_path))
    dummy_frame = np.full((360, 640, 3), 200, dtype=np.uint8)

    ev = PendingEvent(
        event_key="person_0001:new_person:ep1",
        event_type=EventType.NEW_PERSON,
        global_id="person_0001",
        episode_id=1,
        trigger_frame=1,
        trigger_ts=0.2,
        confirmation_info={},
        match_diagnostics={},
    )
    pkt1 = FramePacket(1, 0.2, dummy_frame, [], {"person_0001"}, "fixed", 0.0, 0.8)
    pkt2 = FramePacket(10, 3.0, dummy_frame, [], {"person_0001"}, "fixed", 0.0, 0.8)

    cap_eng.step(pkt1, [ev])
    res1 = cap_eng.step(pkt2, [ev])
    for fin_ev, best_pkt in res1:
        out_mgr.save_event_screenshot(fin_ev, best_pkt)
        saved_again = out_mgr.save_event_screenshot(fin_ev, best_pkt)
        assert saved_again is False

    res2 = cap_eng.step(pkt2, [ev])
    assert len(res2) == 0
    assert len(os.listdir(out_mgr.screenshots_dir)) == 1
