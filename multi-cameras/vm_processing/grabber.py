import argparse
import csv
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
CACHE_DIR = BASE_DIR / ".cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import cv2


CAMERAS = [
    {
        "camera_id": "times_square",
        "page_url": "https://www.earthcam.com/usa/newyork/timessquare/?cam=tsstreet",
        "earthcam_camera_key": "tsstreet",
    },
    {
        "camera_id": "abbey_road",
        "page_url": "https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk",
        "earthcam_camera_key": "abbeyroad_uk",
    },
    {
        "camera_id": "liberty_bell",
        "page_url": "https://www.earthcam.com/usa/pennsylvania/philadelphia/independencehall/?cam=libertybell",
        "earthcam_camera_key": "libertybell",
    },
]
ARCHITECTURE = "vm_processing"
USER_AGENT = "Mozilla/5.0"

DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_SAVE_EVERY = 5
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IMGSZ = 640
DEFAULT_VM_URL = "http://127.0.0.1:8000/detect"
DEFAULT_VM_TIMEOUT_SECONDS = 120
DEFAULT_METRICS_INTERVAL_SECONDS = 1.0

RUNS_DIR = BASE_DIR / "runs"
CURRENT_RUN_FILENAME = BASE_DIR / "current_run.json"
RUNS_INDEX_FILENAME = BASE_DIR / "runs_index.json"

RESULT_FIELDNAMES = [
    "run_id",
    "camera_count",
    "schedule_mode",
    "scheduled_at_utc",
    "captured_at_utc",
    "completed_at_utc",
    "camera_id",
    "architecture",
    "frame_number",
    "person_count",
    "vehicle_count",
    "capture_ms",
    "schedule_lag_ms",
    "vm_roundtrip_ms",
    "vm_decode_ms",
    "vm_queue_ms",
    "vm_inference_ms",
    "vm_total_ms",
    "network_overhead_ms",
    "end_to_end_ms",
    "image_size_bytes",
    "response_json_size_bytes",
    "within_interval",
    "status",
    "error",
]

SYSTEM_METRIC_FIELDNAMES = [
    "timestamp_utc",
    "cpu_percent",
    "memory_used_mb",
    "memory_total_mb",
    "memory_percent",
    "detector_process_memory_mb",
    "network_received_bytes",
    "network_sent_bytes",
    "load_average_1m",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Part 4 multi-camera experiment against the VM detector."
    )
    parser.add_argument("--vm-url", default=DEFAULT_VM_URL)
    parser.add_argument("--vm-timeout-seconds", type=int, default=DEFAULT_VM_TIMEOUT_SECONDS)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--camera-count",
        type=int,
        choices=range(1, len(CAMERAS) + 1),
        default=1,
        help="Number of cameras to use from the predefined camera list.",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=("synchronized", "staggered"),
        default="synchronized",
        help="Start cameras together or distribute their starts across each interval.",
    )
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc)


def fetch_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def get_earthcam_stream_url(camera):
    page_text = fetch_text(camera["page_url"])
    camera_pattern = re.escape(camera["earthcam_camera_key"])
    match = re.search(
        rf'"{camera_pattern}"\s*:\s*\{{.*?"stream"\s*:\s*"([^"]+)"',
        page_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find stream URL for {camera['camera_id']}")
    return match.group(1).replace("\\/", "/")


def capture_frame_with_ffmpeg(camera, stream_url, output_path):
    headers = (
        f"Referer: {camera['page_url']}\r\n"
        "Origin: https://www.earthcam.com\r\n"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-user_agent",
        USER_AGENT,
        "-headers",
        headers,
        "-i",
        stream_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(command, check=True, timeout=45)


def request_json(url, timeout_seconds):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def send_frame_to_vm(image_bytes, vm_url, confidence_threshold, image_size, timeout_seconds):
    query = urllib.parse.urlencode(
        {"confidence": confidence_threshold, "imgsz": image_size}
    )
    separator = "&" if "?" in vm_url else "?"
    request_url = f"{vm_url}{separator}{query}"
    request = urllib.request.Request(
        request_url,
        data=image_bytes,
        headers={"Content-Type": "image/jpeg"},
        method="POST",
    )

    request_start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_body = response.read()
    vm_roundtrip_ms = int((time.perf_counter() - request_start) * 1000)
    return json.loads(response_body), vm_roundtrip_ms, len(response_body)


def metrics_url_from_vm_url(vm_url):
    parsed = urllib.parse.urlsplit(vm_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def draw_detections(frame, boxes):
    for box in boxes:
        label = box["label"]
        confidence = float(box["confidence"])
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        color = (0, 255, 0) if label == "person" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def update_runs_index(run_entry):
    if RUNS_INDEX_FILENAME.exists():
        with open(RUNS_INDEX_FILENAME, encoding="utf-8") as index_file:
            index = json.load(index_file)
    else:
        index = {"runs": []}

    runs = [
        item for item in index.get("runs", [])
        if item.get("run_id") != run_entry["run_id"]
    ]
    runs.append(run_entry)
    runs.sort(key=lambda item: item.get("started_at_utc", ""), reverse=True)
    write_json(RUNS_INDEX_FILENAME, {"runs": runs})


def average(values):
    return round(sum(values) / len(values), 2) if values else None


def run_experiment():
    args = parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        raise ValueError("duration-seconds and interval-seconds must be positive")

    selected_cameras = CAMERAS[: args.camera_count]
    started_at = utc_now()
    run_id = (
        f"vm_{args.camera_count}cam_{args.schedule_mode}_"
        f"{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = RUNS_DIR / run_id
    frames_dir = run_dir / "frames"
    latest_dir = run_dir / "latest"
    work_dir = run_dir / "work"
    for directory in (frames_dir, latest_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "database.csv"
    metrics_path = run_dir / "vm_system_metrics.csv"
    parameters_path = run_dir / "run_parameters.json"
    summary_path = run_dir / "run_summary.json"
    expected_frames_per_camera = math.ceil(args.duration_seconds / args.interval_seconds)

    latest_images = {
        camera["camera_id"]: str(
            (latest_dir / f"{camera['camera_id']}.jpg").relative_to(BASE_DIR)
        )
        for camera in selected_cameras
    }
    current_run = {
        "architecture": ARCHITECTURE,
        "run_id": run_id,
        "camera_count": len(selected_cameras),
        "schedule_mode": args.schedule_mode,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "status": "running",
        "results_csv": str(results_path.relative_to(BASE_DIR)),
        "system_metrics_csv": str(metrics_path.relative_to(BASE_DIR)),
        "summary_json": str(summary_path.relative_to(BASE_DIR)),
        "latest_images": latest_images,
    }
    write_json(CURRENT_RUN_FILENAME, current_run)
    update_runs_index(current_run)

    parameters = {
        "assignment_part": "Part 4 - Multi-camera VM Processing",
        "run_id": run_id,
        "camera_count": len(selected_cameras),
        "cameras": selected_cameras,
        "schedule_mode": args.schedule_mode,
        "capture_method": "Independent EarthCam HLS workers using ffmpeg",
        "processing_method": "YOLOv8 inference served by VM over HTTP POST /detect",
        "architecture": ARCHITECTURE,
        "vm_url": args.vm_url,
        "confidence_threshold": args.confidence,
        "inference_image_size": args.imgsz,
        "interval_seconds_per_camera": args.interval_seconds,
        "scheduling_window_seconds": args.duration_seconds,
        "expected_frames_per_camera": expected_frames_per_camera,
        "expected_total_requests": expected_frames_per_camera * len(selected_cameras),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "successful_requests": 0,
        "failed_requests": 0,
    }
    write_json(parameters_path, parameters)

    print("Fetching EarthCam stream URLs...")
    stream_urls = {}
    for camera in selected_cameras:
        stream_urls[camera["camera_id"]] = get_earthcam_stream_url(camera)
        print(f"Using {camera['camera_id']}: {camera['earthcam_camera_key']}")
    print(f"Run ID: {run_id}")
    print(f"Sending frames to VM detector: {args.vm_url}")

    results_lock = threading.Lock()
    result_rows = []
    system_metric_rows = []
    stop_metrics = threading.Event()

    results_file = open(results_path, "w", newline="", encoding="utf-8")
    results_writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDNAMES)
    results_writer.writeheader()
    results_file.flush()

    metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=SYSTEM_METRIC_FIELDNAMES)
    metrics_writer.writeheader()
    metrics_file.flush()

    def record_result(row):
        with results_lock:
            result_rows.append(row)
            results_writer.writerow(row)
            results_file.flush()

    def collect_metrics():
        metrics_url = metrics_url_from_vm_url(args.vm_url)
        while not stop_metrics.is_set():
            try:
                row = request_json(metrics_url, timeout_seconds=10)
                row["error"] = ""
            except Exception as error:
                row = {name: "" for name in SYSTEM_METRIC_FIELDNAMES}
                row["timestamp_utc"] = utc_now().isoformat()
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"VM metrics error: {row['error']}")
            system_metric_rows.append(row)
            metrics_writer.writerow(row)
            metrics_file.flush()
            stop_metrics.wait(DEFAULT_METRICS_INTERVAL_SECONDS)

    experiment_start_monotonic = time.perf_counter() + 1.0
    experiment_end_monotonic = experiment_start_monotonic + args.duration_seconds
    experiment_start_utc = started_at + timedelta(seconds=1)

    def camera_worker(camera, camera_index):
        camera_id = camera["camera_id"]
        offset_seconds = 0.0
        if args.schedule_mode == "staggered":
            offset_seconds = camera_index * args.interval_seconds / len(selected_cameras)

        frame_number = 1
        while True:
            scheduled_offset = offset_seconds + (frame_number - 1) * args.interval_seconds
            if scheduled_offset >= args.duration_seconds:
                break

            scheduled_monotonic = experiment_start_monotonic + scheduled_offset
            sleep_seconds = scheduled_monotonic - time.perf_counter()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if time.perf_counter() >= experiment_end_monotonic:
                break

            scheduled_at = experiment_start_utc + timedelta(seconds=scheduled_offset)
            frame_start = time.perf_counter()
            captured_at = utc_now()
            schedule_lag_ms = max(
                0,
                int((time.perf_counter() - scheduled_monotonic) * 1000),
            )
            raw_path = work_dir / f"{camera_id}.jpg"

            base_row = {
                "run_id": run_id,
                "camera_count": len(selected_cameras),
                "schedule_mode": args.schedule_mode,
                "scheduled_at_utc": scheduled_at.isoformat(),
                "captured_at_utc": captured_at.isoformat(),
                "completed_at_utc": "",
                "camera_id": camera_id,
                "architecture": ARCHITECTURE,
                "frame_number": frame_number,
                "person_count": "",
                "vehicle_count": "",
                "capture_ms": "",
                "schedule_lag_ms": schedule_lag_ms,
                "vm_roundtrip_ms": "",
                "vm_decode_ms": "",
                "vm_queue_ms": "",
                "vm_inference_ms": "",
                "vm_total_ms": "",
                "network_overhead_ms": "",
                "end_to_end_ms": "",
                "image_size_bytes": "",
                "response_json_size_bytes": "",
                "within_interval": False,
                "status": "error",
                "error": "",
            }

            try:
                capture_start = time.perf_counter()
                capture_frame_with_ffmpeg(camera, stream_urls[camera_id], raw_path)
                capture_ms = int((time.perf_counter() - capture_start) * 1000)
                image_bytes = raw_path.read_bytes()

                vm_result, vm_roundtrip_ms, response_size = send_frame_to_vm(
                    image_bytes,
                    args.vm_url,
                    args.confidence,
                    args.imgsz,
                    args.vm_timeout_seconds,
                )
                frame = cv2.imread(str(raw_path))
                if frame is None:
                    raise RuntimeError("OpenCV could not read the captured frame")
                processed_frame = draw_detections(frame, vm_result["boxes"])
                latest_path = latest_dir / f"{camera_id}.jpg"
                cv2.imwrite(str(latest_path), processed_frame)
                if frame_number == 1 or frame_number % DEFAULT_SAVE_EVERY == 0:
                    cv2.imwrite(
                        str(frames_dir / f"{camera_id}_{frame_number:03d}.jpg"),
                        processed_frame,
                    )

                end_to_end_ms = int((time.perf_counter() - frame_start) * 1000)
                vm_total_ms = int(vm_result["vm_total_ms"])
                base_row.update(
                    {
                        "completed_at_utc": utc_now().isoformat(),
                        "person_count": vm_result["person_count"],
                        "vehicle_count": vm_result["vehicle_count"],
                        "capture_ms": capture_ms,
                        "vm_roundtrip_ms": vm_roundtrip_ms,
                        "vm_decode_ms": vm_result["vm_decode_ms"],
                        "vm_queue_ms": vm_result["vm_queue_ms"],
                        "vm_inference_ms": vm_result["vm_inference_ms"],
                        "vm_total_ms": vm_total_ms,
                        "network_overhead_ms": max(0, vm_roundtrip_ms - vm_total_ms),
                        "end_to_end_ms": end_to_end_ms,
                        "image_size_bytes": len(image_bytes),
                        "response_json_size_bytes": response_size,
                        "within_interval": end_to_end_ms <= args.interval_seconds * 1000,
                        "status": "ok",
                    }
                )
                print(
                    f"[{camera_id} {frame_number:03d}] "
                    f"people={base_row['person_count']} vehicles={base_row['vehicle_count']} "
                    f"lag={schedule_lag_ms} ms roundtrip={vm_roundtrip_ms} ms "
                    f"queue={base_row['vm_queue_ms']} ms"
                )
            except Exception as error:
                base_row["completed_at_utc"] = utc_now().isoformat()
                base_row["end_to_end_ms"] = int((time.perf_counter() - frame_start) * 1000)
                base_row["error"] = f"{type(error).__name__}: {error}"
                print(f"[{camera_id} {frame_number:03d}] ERROR {base_row['error']}")

            record_result(base_row)
            frame_number += 1

    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=len(selected_cameras)) as executor:
            futures = [
                executor.submit(camera_worker, camera, index)
                for index, camera in enumerate(selected_cameras)
            ]
            for future in futures:
                future.result()
    finally:
        stop_metrics.set()
        metrics_thread.join(timeout=15)
        results_file.close()
        metrics_file.close()

    finished_at = utc_now()
    successful_rows = [row for row in result_rows if row["status"] == "ok"]
    failed_rows = [row for row in result_rows if row["status"] != "ok"]
    response_times = [int(row["vm_roundtrip_ms"]) for row in successful_rows]

    valid_cpu = [
        float(row["cpu_percent"])
        for row in system_metric_rows
        if row.get("cpu_percent") not in (None, "")
    ]
    valid_memory = [
        float(row["memory_percent"])
        for row in system_metric_rows
        if row.get("memory_percent") not in (None, "")
    ]
    valid_network = [
        row
        for row in system_metric_rows
        if row.get("network_received_bytes") not in (None, "")
    ]
    received_delta = 0
    sent_delta = 0
    if len(valid_network) >= 2:
        received_delta = int(valid_network[-1]["network_received_bytes"]) - int(
            valid_network[0]["network_received_bytes"]
        )
        sent_delta = int(valid_network[-1]["network_sent_bytes"]) - int(
            valid_network[0]["network_sent_bytes"]
        )

    elapsed_seconds = (finished_at - started_at).total_seconds()
    expected_total = expected_frames_per_camera * len(selected_cameras)
    summary = {
        "architecture": ARCHITECTURE,
        "run_id": run_id,
        "camera_count": len(selected_cameras),
        "schedule_mode": args.schedule_mode,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "scheduling_window_seconds": args.duration_seconds,
        "actual_elapsed_seconds": round(elapsed_seconds, 2),
        "interval_seconds_per_camera": args.interval_seconds,
        "expected_frames_per_camera": expected_frames_per_camera,
        "expected_total_requests": expected_total,
        "attempted_requests": len(result_rows),
        "successful_requests": len(successful_rows),
        "failed_requests": len(failed_rows),
        "missed_requests": max(0, expected_total - len(result_rows)),
        "average_response_ms": average(response_times),
        "average_cpu_percent": average(valid_cpu),
        "max_cpu_percent": max(valid_cpu) if valid_cpu else None,
        "average_memory_percent": average(valid_memory),
        "max_memory_percent": max(valid_memory) if valid_memory else None,
        "network_received_bytes": max(0, received_delta),
        "network_sent_bytes": max(0, sent_delta),
    }
    write_json(summary_path, summary)

    parameters["finished_at_utc"] = finished_at.isoformat()
    parameters["successful_requests"] = len(successful_rows)
    parameters["failed_requests"] = len(failed_rows)
    write_json(parameters_path, parameters)

    current_run["status"] = "finished"
    current_run["finished_at_utc"] = finished_at.isoformat()
    write_json(CURRENT_RUN_FILENAME, current_run)
    update_runs_index(current_run)
    print(f"Results saved to: {results_path}")
    print(f"VM system metrics saved to: {metrics_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
