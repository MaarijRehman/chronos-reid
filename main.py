import argparse
import logging
import os
import sys
import time
import yaml

from app.video_reader import VideoReader, VideoReaderError
from app.frame_quality import FrameQualityAssessor
from app.detector import PersonDetector
from app.tracker import MultiObjectTracker
from app.reidentifier import AppearanceEmbedder
from app.identity_manager import GlobalIdentityManager
from app.event_manager import AppearanceEventManager
from app.capture_engine import CaptureEngine
from app.output_manager import OutputManager
from app.models import FramePacket


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Person-Aware Video Event Extractor: Captures new person arrivals and confirmed returns."
    )
    parser.add_argument("--video", required=True, help="Path to local input video file.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument(
        "--camera-mode",
        choices=["fixed", "moving", "auto"],
        help="Camera motion mode override (fixed, moving, auto).",
    )
    parser.add_argument(
        "--mode",
        choices=["sampled", "full"],
        help="Frame analysis mode: 'sampled' (default 5 FPS) or 'full' (every frame).",
    )
    parser.add_argument("--sample-fps", type=float, help="Target processed frames per second in sampled mode.")
    parser.add_argument(
        "--return-threshold",
        type=float,
        help="Confirmed absence duration in seconds required before triggering a return screenshot.",
    )
    parser.add_argument("--grouping-window", type=float, help="Event grouping window in seconds.")
    parser.add_argument("--output-dir", default=None, help="Custom base directory for output files.")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save annotated debug screenshots and tracked debug video.",
    )
    return parser


def load_and_merge_config(args: argparse.Namespace) -> dict:
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.camera_mode:
        cfg.setdefault("video", {})["camera_mode"] = args.camera_mode
    if args.mode:
        cfg.setdefault("video", {})["mode"] = args.mode
    if args.sample_fps is not None:
        if args.sample_fps <= 0:
            raise ValueError("--sample-fps must be > 0")
        cfg.setdefault("video", {})["sample_fps"] = args.sample_fps
    if args.return_threshold is not None:
        if args.return_threshold <= 0:
            raise ValueError("--return-threshold must be > 0 seconds")
        cfg.setdefault("events", {})["return_absence_threshold_sec"] = args.return_threshold
    if args.grouping_window is not None:
        if args.grouping_window < 0:
            raise ValueError("--grouping-window must be >= 0 seconds")
        cfg.setdefault("events", {})["grouping_window_sec"] = args.grouping_window
    if args.save_debug:
        cfg.setdefault("output", {})["save_annotated_debug"] = True
        cfg.setdefault("output", {})["save_debug_video"] = True

    return cfg


def run_pipeline(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("main")

    try:
        cfg = load_and_merge_config(args)
        reader = VideoReader(args.video, cfg)
    except (FileNotFoundError, ValueError, VideoReaderError) as exc:
        logger.error("Initialization error: %s", exc)
        return 1

    quality_assessor = FrameQualityAssessor(cfg)
    detector = PersonDetector(cfg)
    tracker = MultiObjectTracker(cfg, detector)
    embedder = AppearanceEmbedder(cfg)
    identity_mgr = GlobalIdentityManager(cfg, embedder=embedder)
    event_mgr = AppearanceEventManager(cfg)
    capture_eng = CaptureEngine(cfg)
    output_mgr = OutputManager(args.video, cfg, custom_out_dir=args.output_dir)

    start_wall = time.time()
    processed_frames = 0
    screenshots_saved = 0

    logger.info(
        "Starting extraction on '%s' (Native FPS=%.2f, Total Frames=%d, Mode=%s, Camera=%s)",
        args.video,
        reader.native_fps,
        reader.total_frames,
        cfg["video"]["mode"],
        cfg["video"]["camera_mode"],
    )

    try:
        for frame_idx, timestamp, frame, cam_state, motion_px in reader:
            processed_frames += 1

            detections = tracker.step(frame, frame_idx, timestamp, cam_state)
            frame_q = quality_assessor.assess_detections(frame, detections)

            newly_confirmed, confirmed_returns = identity_mgr.process_frame_detections(
                detections, frame_idx, timestamp, cam_state
            )

            pending_events = event_mgr.step(
                identity_mgr.identities,
                newly_confirmed,
                confirmed_returns,
                frame_idx,
                timestamp,
            )

            active_gids = {
                gid
                for gid, rec in identity_mgr.identities.items()
                if rec.last_seen_frame == frame_idx
            }

            packet = FramePacket(
                frame_idx=frame_idx,
                timestamp=timestamp,
                frame=frame,
                detections=detections,
                active_global_ids=active_gids,
                camera_mode_state=cam_state,
                camera_motion_px=motion_px,
                frame_quality_score=frame_q,
            )

            finalized_list = capture_eng.step(packet, pending_events)
            for fin_ev, best_pkt in finalized_list:
                if output_mgr.save_event_screenshot(fin_ev, best_pkt):
                    screenshots_saved += 1
                    logger.info(
                        "Saved [%s] -> %s (Triggers=%s, Frame=%d, TS=%.2fs)",
                        fin_ev.primary_event_type,
                        fin_ev.screenshot_filename,
                        fin_ev.triggering_global_ids,
                        fin_ev.frame_number,
                        fin_ev.timestamp,
                    )

            debug_fps = (
                reader.sample_fps
                if cfg["video"]["mode"] == "sampled"
                else reader.native_fps
            )
            output_mgr.write_debug_video_frame(packet, debug_fps)

            if processed_frames % 25 == 0:
                elapsed = max(1e-3, time.time() - start_wall)
                proc_fps = processed_frames / elapsed
                if reader.total_frames > 0 and frame_idx > 0:
                    progress = frame_idx / float(reader.total_frames)
                    eta_sec = (elapsed / max(1e-4, progress)) - elapsed
                else:
                    eta_sec = 0.0
                logger.info(
                    "Progress: Frame %d/%d | Proc: %d (%.1f FPS) | Elapsed: %.1fs | ETA: %.1fs | "
                    "IDs: %d | NewEv: %d | RetEv: %d | UncertainCand: %d | Saved: %d",
                    frame_idx,
                    reader.total_frames,
                    processed_frames,
                    proc_fps,
                    elapsed,
                    max(0.0, eta_sec),
                    len(identity_mgr.identities),
                    event_mgr.new_person_count,
                    event_mgr.return_event_count,
                    len(identity_mgr.candidates),
                    screenshots_saved,
                )

        for fin_ev, best_pkt in capture_eng.flush():
            if output_mgr.save_event_screenshot(fin_ev, best_pkt):
                screenshots_saved += 1

    except VideoReaderError as exc:
        logger.error("Video decoding error: %s", exc)
        return 1
    finally:
        tracker.cleanup()

    elapsed_total = max(1e-3, time.time() - start_wall)
    run_stats = {
        "video_path": args.video,
        "processed_frames": processed_frames,
        "processing_fps": round(processed_frames / elapsed_total, 2),
        "elapsed_seconds": round(elapsed_total, 2),
        "total_confirmed_identities": len(identity_mgr.identities),
        "new_person_events": event_mgr.new_person_count,
        "return_events": event_mgr.return_event_count,
        "unresolved_uncertain_candidates": len(identity_mgr.candidates),
        "screenshots_saved": screenshots_saved,
    }
    output_mgr.export_metadata(identity_mgr.identities, run_stats)
    logger.info("Completed extraction. Summary: %s", run_stats)
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline(build_arg_parser().parse_args()))
