from app.identity_manager import GlobalIdentityManager
from app.event_manager import AppearanceEventManager
from app.models import EventType
from tests.test_identity_manager import make_base_config, make_det, make_orthogonal_emb


def run_step(mgr: GlobalIdentityManager, ev_mgr: AppearanceEventManager, dets, frame_idx: int, ts: float):
    new_c, ret_c = mgr.process_frame_detections(dets, frame_idx, ts, "fixed")
    return ev_mgr.step(mgr.identities, new_c, ret_c, frame_idx, ts)


def test_1_and_2_one_new_person_and_thousands_of_frames_no_repeat():
    cfg = make_base_config()
    mgr = GlobalIdentityManager(cfg)
    ev_mgr = AppearanceEventManager(cfg)
    emb = make_orthogonal_emb(0)

    total_events = []
    for f in range(1, 2005):
        ts = f * 0.2
        tid = 1 if f < 1000 else 2
        bbox = (100 + (f % 50), 100, 200 + (f % 50), 300)
        evs = run_step(mgr, ev_mgr, [make_det(f, ts, emb, track_id=tid, bbox=bbox)], f, ts)
        total_events.extend(evs)

    assert len(total_events) == 1
    assert total_events[0].event_type == EventType.NEW_PERSON
    assert total_events[0].global_id == "person_0001"


def test_3_and_4_three_people_visible_then_fourth_person_enters():
    cfg = make_base_config()
    mgr = GlobalIdentityManager(cfg)
    ev_mgr = AppearanceEventManager(cfg)
    embs = [make_orthogonal_emb(i) for i in range(4)]

    events_phase1 = []
    for f in range(1, 901):
        ts = f * 0.2
        dets = [
            make_det(f, ts, embs[0], track_id=1, bbox=(50, 100, 130, 280)),
            make_det(f, ts, embs[1], track_id=2, bbox=(200, 100, 280, 280)),
            make_det(f, ts, embs[2], track_id=3, bbox=(350, 100, 430, 280)),
        ]
        events_phase1.extend(run_step(mgr, ev_mgr, dets, f, ts))

    assert len(events_phase1) == 3

    events_phase2 = []
    for f in (901, 902):
        ts = f * 0.2
        dets = [
            make_det(f, ts, embs[0], track_id=1, bbox=(50, 100, 130, 280)),
            make_det(f, ts, embs[1], track_id=2, bbox=(200, 100, 280, 280)),
            make_det(f, ts, embs[2], track_id=3, bbox=(350, 100, 430, 280)),
            make_det(f, ts, embs[3], track_id=4, bbox=(500, 100, 580, 280)),
        ]
        events_phase2.extend(run_step(mgr, ev_mgr, dets, f, ts))

    assert len(events_phase2) == 1
    assert events_phase2[0].global_id == "person_0004"
    assert events_phase2[0].event_type == EventType.NEW_PERSON


def test_6_7_8_return_threshold_and_single_return_episode_suppression():
    cfg = make_base_config()
    cfg["events"]["return_absence_threshold_sec"] = 60.0
    mgr = GlobalIdentityManager(cfg)
    ev_mgr = AppearanceEventManager(cfg)
    emb = make_orthogonal_emb(0)

    run_step(mgr, ev_mgr, [make_det(1, 1.0, emb, track_id=1)], 1, 1.0)
    new_evs = run_step(mgr, ev_mgr, [make_det(2, 1.2, emb, track_id=1)], 2, 1.2)
    assert len(new_evs) == 1

    run_step(mgr, ev_mgr, [], 3, 2.0)
    early_ret_1 = run_step(mgr, ev_mgr, [make_det(4, 35.0, emb, track_id=2)], 4, 35.0)
    early_ret_2 = run_step(mgr, ev_mgr, [make_det(5, 35.2, emb, track_id=2)], 5, 35.2)
    assert len(early_ret_1) == 0
    assert len(early_ret_2) == 0

    run_step(mgr, ev_mgr, [], 6, 40.0)
    run_step(mgr, ev_mgr, [], 7, 98.0)

    return_events = []
    for i in range(50):
        f = 10 + i
        ts = 100.0 + i * 0.2
        return_events.extend(run_step(mgr, ev_mgr, [make_det(f, ts, emb, track_id=3)], f, ts))

    assert len(return_events) == 1
    assert return_events[0].event_type == EventType.RETURN
    assert return_events[0].global_id == "person_0001"
