import argparse
import json
import os
import time
import tracemalloc
import cv2
import numpy as np
import torch

try:
    import psutil
except ImportError:
    psutil = None

from main import build_arg_parser, run_pipeline


def generate_synthetic_video(path: str, duration_sec: int = 12, fps: int = 30) -> None:
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    total_frames = duration_sec * fps

    rng = np.random.default_rng(42)
    bg_points = rng.integers(40, [w - 40, h - 40], size=(120, 2))

    for f in range(total_frames):
        canvas = np.full((h, w, 3), 210, dtype=np.uint8)
        for px, py in bg_points:
            cv2.rectangle(canvas, (int(px), int(py)), (int(px + 8), int(py + 8)), (90, 90, 90), -1)

        x = int(100 + (f * 4) % (w - 250))
        y = 220
        cv2.rectangle(canvas, (x, y), (x + 90, y + 260), (40, 80, 160), -1)
        cv2.circle(canvas, (x + 45, y - 30), 28, (180, 160, 140), -1)
        writer.write(canvas)

    writer.release()


def get_ram_mb() -> float:
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def profile_mode(video_path: str, mode: str, sample_fps: float, return_thresh: float) -> dict:
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / native_fps
    cap.release()

    tracemalloc.start()
    ram_before = get_ram_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cmd_args = [
        "--video", video_path,
        "--mode", mode,
        "--sample-fps", str(sample_fps),
        "--return-threshold", str(return_thresh),
        "--output-dir", f"output_bench_{mode}",
    ]
    parsed_args = build_arg_parser().parse_args(cmd_args)

    t0 = time.perf_counter()
    exit_code = run_pipeline(parsed_args)
    wall_time = max(1e-4, time.perf_counter() - t0)

    ram_after = get_ram_mb()
    tracemalloc.stop()

    vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    events_json = os.path.join(f"output_bench_{mode}", video_stem, "metadata", "events.json")
    processed_frames = 0
    screenshots_saved = 0
    confirmed_ids = 0

    if os.path.exists(events_json):
        with open(events_json, "r", encoding="utf-8") as f:
            data = json.load(f).get("run_summary", {})
            processed_frames = data.get("processed_frames", 0)
            screenshots_saved = data.get("screenshots_saved", 0)
            confirmed_ids = data.get("total_confirmed_identities", 0)

    return {
        "mode": f"{mode} ({sample_fps} FPS)" if mode == "sampled" else "full (native FPS)",
        "exit_code": exit_code,
        "video_duration_sec": round(video_duration_sec, 2),
        "wall_time_sec": round(wall_time, 2),
        "realtime_speedup": round(video_duration_sec / wall_time, 2),
        "processed_frames": processed_frames,
        "effective_proc_fps": round(processed_frames / wall_time, 2),
        "peak_ram_mb": round(ram_after, 1),
        "ram_delta_mb": round(max(0.0, ram_after - ram_before), 1),
        "peak_vram_mb": round(vram_mb, 1),
        "confirmed_identities": confirmed_ids,
        "screenshots_saved": screenshots_saved,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark speed, RAM/VRAM, and throughput.")
    parser.add_argument("--video", default="smoke_test.mp4", help="Path to test video (auto-created if missing).")
    parser.add_argument("--return-threshold", type=float, default=15.0, help="Return threshold in seconds for test.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[*] Generating 12-second 720p test video at '{args.video}'...")
        generate_synthetic_video(args.video, duration_sec=12, fps=30)

    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Running Efficiency Benchmark on Device: {device_str}\n")

    res_sampled = profile_mode(args.video, mode="sampled", sample_fps=5.0, return_thresh=args.return_threshold)
    res_full = profile_mode(args.video, mode="full", sample_fps=30.0, return_thresh=args.return_threshold)

    print("\n" + "=" * 78)
    print(f"{'Metric':<30} | {'Sampled (5 FPS)':<20} | {'Full Mode (Every Frame)':<20}")
    print("-" * 78)
    for key, label in [
        ("video_duration_sec", "Video Duration (s)"),
        ("wall_time_sec", "Wall-Clock Time (s)"),
        ("realtime_speedup", "Real-Time Speed Multiplier"),
        ("processed_frames", "Frames Analyzed"),
        ("effective_proc_fps", "Processing Throughput (FPS)"),
        ("peak_ram_mb", "Peak Process RAM (MB)"),
        ("peak_vram_mb", "Peak GPU VRAM (MB)"),
        ("confirmed_identities", "Confirmed Global IDs"),
        ("screenshots_saved", "Screenshots Saved"),
    ]:
        print(f"{label:<30} | {str(res_sampled[key]):<20} | {str(res_full[key]):<20}")
    print("=" * 78)
