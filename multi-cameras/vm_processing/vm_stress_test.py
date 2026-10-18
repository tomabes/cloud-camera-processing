import argparse
import csv
import json
import math
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = (
    BASE_DIR
    / "runs"
    / "vm_3cam_synchronized_20261017T170652Z"
    / "work"
)
STRESS_RUNS_DIR = BASE_DIR / "stress_runs"
COMBINED_SUMMARY_PATH = STRESS_RUNS_DIR / "stress_test_summary.csv"

DEFAULT_VM_URL = "http://127.0.0.1:8000/detect"
DEFAULT_VM_TIMEOUT_SECONDS = 30
DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IMGSZ = 640
DEFAULT_METRICS_INTERVAL_SECONDS = 1.0
DEFAULT_WARMUP_REQUESTS = 3

REQUEST_FIELDNAMES = [
    "run_id",
    "virtual_camera_count",
    "virtual_camera_id",
    "source_image",
    "scheduled_at_utc",
    "completed_at_utc",
    "frame_number",
    "person_count",
    "vehicle_count",
    "schedule_lag_ms",
    "vm_roundtrip_ms",
    "vm_decode_ms",
    "vm_queue_ms",
    "vm_inference_ms",
    "vm_total_ms",
    "network_overhead_ms",
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

CAMERA_SUMMARY_FIELDNAMES = [
    "run_id",
    "virtual_camera_count",
    "virtual_camera_id",
    "source_image",
    "expected_requests",
    "attempted_requests",
    "successful_requests",
    "failed_requests",
    "missed_requests",
    "late_requests",
    "average_person_count",
    "average_vehicle_count",
    "average_vm_inference_ms",
    "max_vm_inference_ms",
    "average_vm_queue_ms",
    "max_vm_queue_ms",
    "average_vm_roundtrip_ms",
    "max_vm_roundtrip_ms",
]

RUN_SUMMARY_FIELDNAMES = [
    "run_id",
    "virtual_camera_count",
    "source_image_count",
    "duration_seconds",
    "interval_seconds",
    "imgsz",
    "confidence",
    "expected_requests",
    "attempted_requests",
    "successful_requests",
    "failed_requests",
    "missed_requests",
    "late_requests",
    "average_person_count",
    "average_vehicle_count",
    "average_vm_inference_ms",
    "max_vm_inference_ms",
    "average_vm_queue_ms",
    "max_vm_queue_ms",
    "average_vm_roundtrip_ms",
    "max_vm_roundtrip_ms",
    "average_cpu_percent",
    "max_cpu_percent",
    "average_memory_percent",
    "max_memory_percent",
    "network_received_bytes",
    "network_sent_bytes",
    "started_at_utc",
    "finished_at_utc",
    "actual_elapsed_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test the VM detector with synchronized virtual cameras that "
            "replay saved JPEG frames."
        )
    )
    parser.add_argument(
        "--virtual-camera-count",
        type=int,
        required=True,
        help="Number of simultaneous virtual cameras.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing raw JPEG frames to replay round-robin.",
    )
    parser.add_argument("--vm-url", default=DEFAULT_VM_URL)
    parser.add_argument(
        "--vm-timeout-seconds",
        type=float,
        default=DEFAULT_VM_TIMEOUT_SECONDS,
    )
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=DEFAULT_WARMUP_REQUESTS,
        help="Sequential warm-up requests before the measured stress test.",
    )
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc)


def average(values):
    return round(sum(values) / len(values), 2) if values else ""


def maximum(values):
    return max(values) if values else ""


def request_json(url, timeout_seconds):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def endpoint_url(vm_url, path):
    parsed = urllib.parse.urlsplit(vm_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def send_frame_to_vm(image_bytes, vm_url, confidence, imgsz, timeout_seconds):
    query = urllib.parse.urlencode({"confidence": confidence, "imgsz": imgsz})
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
    roundtrip_ms = int((time.perf_counter() - request_start) * 1000)
    return json.loads(response_body), roundtrip_ms, len(response_body)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def numeric_values(rows, fieldname):
    return [
        float(row[fieldname])
        for row in rows
        if row.get(fieldname) not in (None, "")
    ]


def load_source_images(source_dir):
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError(f"No JPEG source images found in: {source_dir}")

    return [
        {
            "name": path.name,
            "path": str(path),
            "bytes": path.read_bytes(),
        }
        for path in paths
    ]


def build_camera_summary(
    run_id,
    virtual_cameras,
    result_rows,
    expected_frames_per_camera,
):
    summary_rows = []
    for camera in virtual_cameras:
        camera_rows = [
            row
            for row in result_rows
            if row["virtual_camera_id"] == camera["virtual_camera_id"]
        ]
        successful_rows = [row for row in camera_rows if row["status"] == "ok"]
        failed_rows = [row for row in camera_rows if row["status"] != "ok"]
        late_rows = [
            row
            for row in successful_rows
            if str(row["within_interval"]).lower() != "true"
        ]

        inference_values = numeric_values(successful_rows, "vm_inference_ms")
        queue_values = numeric_values(successful_rows, "vm_queue_ms")
        roundtrip_values = numeric_values(successful_rows, "vm_roundtrip_ms")

        summary_rows.append(
            {
                "run_id": run_id,
                "virtual_camera_count": len(virtual_cameras),
                "virtual_camera_id": camera["virtual_camera_id"],
                "source_image": camera["source_image"]["name"],
                "expected_requests": expected_frames_per_camera,
                "attempted_requests": len(camera_rows),
                "successful_requests": len(successful_rows),
                "failed_requests": len(failed_rows),
                "missed_requests": max(
                    0,
                    expected_frames_per_camera - len(camera_rows),
                ),
                "late_requests": len(late_rows),
                "average_person_count": average(
                    numeric_values(successful_rows, "person_count")
                ),
                "average_vehicle_count": average(
                    numeric_values(successful_rows, "vehicle_count")
                ),
                "average_vm_inference_ms": average(inference_values),
                "max_vm_inference_ms": maximum(inference_values),
                "average_vm_queue_ms": average(queue_values),
                "max_vm_queue_ms": maximum(queue_values),
                "average_vm_roundtrip_ms": average(roundtrip_values),
                "max_vm_roundtrip_ms": maximum(roundtrip_values),
            }
        )
    return summary_rows


def run_stress_test():
    args = parse_args()
    if args.virtual_camera_count <= 0:
        raise ValueError("virtual-camera-count must be positive")
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        raise ValueError("duration-seconds and interval-seconds must be positive")
    if args.vm_timeout_seconds <= 0:
        raise ValueError("vm-timeout-seconds must be positive")
    if args.warmup_requests <= 0:
        raise ValueError("warmup-requests must be positive")

    source_images = load_source_images(args.source_dir)
    virtual_cameras = [
        {
            "virtual_camera_id": f"virtual_{index + 1:03d}",
            "source_image": source_images[index % len(source_images)],
        }
        for index in range(args.virtual_camera_count)
    ]

    health = request_json(endpoint_url(args.vm_url, "/health"), timeout_seconds=5)
    if health.get("status") != "ok":
        raise RuntimeError(f"VM health check failed: {health}")

    print(f"VM health: {health['status']} ({health.get('model', 'unknown model')})")
    print(f"Loaded {len(source_images)} source JPEG files")
    print(f"Virtual cameras: {len(virtual_cameras)}")

    warmup_result = None
    warmup_roundtrip_ms = None
    for warmup_number in range(1, args.warmup_requests + 1):
        warmup_result, warmup_roundtrip_ms, _ = send_frame_to_vm(
            source_images[0]["bytes"],
            args.vm_url,
            args.confidence,
            args.imgsz,
            args.vm_timeout_seconds,
        )
        print(
            f"Warm-up {warmup_number}/{args.warmup_requests}: "
            f"inference={warmup_result['vm_inference_ms']} ms, "
            f"roundtrip={warmup_roundtrip_ms} ms"
        )

    deadline_ms = args.interval_seconds * 1000
    if warmup_roundtrip_ms > deadline_ms:
        raise RuntimeError(
            "VM is not ready for a measured run: the final sequential warm-up "
            f"took {warmup_roundtrip_ms} ms, above the {deadline_ms:.0f} ms interval"
        )

    started_at = utc_now()
    run_id = (
        f"vm_stress_{args.virtual_camera_count}vcam_"
        f"{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = STRESS_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    requests_path = run_dir / "requests.csv"
    camera_summary_path = run_dir / "camera_summary.csv"
    metrics_path = run_dir / "vm_system_metrics.csv"
    run_summary_path = run_dir / "run_summary.csv"
    parameters_path = run_dir / "run_parameters.json"

    expected_frames_per_camera = math.ceil(
        args.duration_seconds / args.interval_seconds
    )
    expected_total_requests = expected_frames_per_camera * len(virtual_cameras)

    parameters = {
        "experiment": "VM synthetic multi-camera stress test",
        "run_id": run_id,
        "virtual_camera_count": len(virtual_cameras),
        "virtual_cameras": [
            {
                "virtual_camera_id": camera["virtual_camera_id"],
                "source_image": camera["source_image"]["path"],
            }
            for camera in virtual_cameras
        ],
        "source_image_count": len(source_images),
        "source_directory": str(args.source_dir.expanduser().resolve()),
        "vm_url": args.vm_url,
        "confidence": args.confidence,
        "imgsz": args.imgsz,
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "expected_frames_per_camera": expected_frames_per_camera,
        "expected_total_requests": expected_total_requests,
        "warmup_requests_before_measurement": args.warmup_requests,
        "final_warmup_inference_ms": warmup_result["vm_inference_ms"],
        "final_warmup_roundtrip_ms": warmup_roundtrip_ms,
        "started_at_utc": started_at.isoformat(),
    }
    write_json(parameters_path, parameters)

    result_rows = []
    system_metric_rows = []
    results_lock = threading.Lock()
    stop_metrics = threading.Event()

    requests_file = open(requests_path, "w", newline="", encoding="utf-8")
    requests_writer = csv.DictWriter(
        requests_file,
        fieldnames=REQUEST_FIELDNAMES,
    )
    requests_writer.writeheader()
    requests_file.flush()

    metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=SYSTEM_METRIC_FIELDNAMES,
    )
    metrics_writer.writeheader()
    metrics_file.flush()

    def record_result(row):
        with results_lock:
            result_rows.append(row)
            requests_writer.writerow(row)
            requests_file.flush()

    def collect_metrics():
        metrics_url = endpoint_url(args.vm_url, "/metrics")
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

    def virtual_camera_worker(camera):
        camera_id = camera["virtual_camera_id"]
        source_image = camera["source_image"]

        for frame_number in range(1, expected_frames_per_camera + 1):
            scheduled_offset = (frame_number - 1) * args.interval_seconds
            scheduled_monotonic = experiment_start_monotonic + scheduled_offset
            sleep_seconds = scheduled_monotonic - time.perf_counter()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if time.perf_counter() >= experiment_end_monotonic:
                break

            scheduled_at = experiment_start_utc + timedelta(seconds=scheduled_offset)
            schedule_lag_ms = max(
                0,
                int((time.perf_counter() - scheduled_monotonic) * 1000),
            )
            request_started = time.perf_counter()
            row = {
                "run_id": run_id,
                "virtual_camera_count": len(virtual_cameras),
                "virtual_camera_id": camera_id,
                "source_image": source_image["name"],
                "scheduled_at_utc": scheduled_at.isoformat(),
                "completed_at_utc": "",
                "frame_number": frame_number,
                "person_count": "",
                "vehicle_count": "",
                "schedule_lag_ms": schedule_lag_ms,
                "vm_roundtrip_ms": "",
                "vm_decode_ms": "",
                "vm_queue_ms": "",
                "vm_inference_ms": "",
                "vm_total_ms": "",
                "network_overhead_ms": "",
                "image_size_bytes": len(source_image["bytes"]),
                "response_json_size_bytes": "",
                "within_interval": False,
                "status": "error",
                "error": "",
            }

            try:
                vm_result, roundtrip_ms, response_size = send_frame_to_vm(
                    source_image["bytes"],
                    args.vm_url,
                    args.confidence,
                    args.imgsz,
                    args.vm_timeout_seconds,
                )
                vm_total_ms = int(vm_result["vm_total_ms"])
                row.update(
                    {
                        "completed_at_utc": utc_now().isoformat(),
                        "person_count": vm_result["person_count"],
                        "vehicle_count": vm_result["vehicle_count"],
                        "vm_roundtrip_ms": roundtrip_ms,
                        "vm_decode_ms": vm_result["vm_decode_ms"],
                        "vm_queue_ms": vm_result["vm_queue_ms"],
                        "vm_inference_ms": vm_result["vm_inference_ms"],
                        "vm_total_ms": vm_total_ms,
                        "network_overhead_ms": max(0, roundtrip_ms - vm_total_ms),
                        "response_json_size_bytes": response_size,
                        "within_interval": (
                            roundtrip_ms <= args.interval_seconds * 1000
                        ),
                        "status": "ok",
                    }
                )
                if frame_number in (1, expected_frames_per_camera):
                    print(
                        f"[{camera_id} {frame_number:03d}] "
                        f"people={row['person_count']} "
                        f"vehicles={row['vehicle_count']} "
                        f"roundtrip={roundtrip_ms} ms "
                        f"queue={row['vm_queue_ms']} ms"
                    )
            except Exception as error:
                row["completed_at_utc"] = utc_now().isoformat()
                row["vm_roundtrip_ms"] = int(
                    (time.perf_counter() - request_started) * 1000
                )
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"[{camera_id} {frame_number:03d}] ERROR {row['error']}")

            record_result(row)

    print(f"Run ID: {run_id}")
    print(f"Expected requests: {expected_total_requests}")
    print(f"Sending saved frames to: {args.vm_url}")

    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=len(virtual_cameras)) as executor:
            futures = [
                executor.submit(virtual_camera_worker, camera)
                for camera in virtual_cameras
            ]
            for future in futures:
                future.result()
    finally:
        stop_metrics.set()
        metrics_thread.join(timeout=15)
        requests_file.close()
        metrics_file.close()

    finished_at = utc_now()
    result_rows.sort(
        key=lambda row: (row["virtual_camera_id"], int(row["frame_number"]))
    )
    write_csv(requests_path, REQUEST_FIELDNAMES, result_rows)

    camera_summary_rows = build_camera_summary(
        run_id,
        virtual_cameras,
        result_rows,
        expected_frames_per_camera,
    )
    write_csv(
        camera_summary_path,
        CAMERA_SUMMARY_FIELDNAMES,
        camera_summary_rows,
    )

    successful_rows = [row for row in result_rows if row["status"] == "ok"]
    failed_rows = [row for row in result_rows if row["status"] != "ok"]
    late_rows = [
        row
        for row in successful_rows
        if str(row["within_interval"]).lower() != "true"
    ]

    inference_values = numeric_values(successful_rows, "vm_inference_ms")
    queue_values = numeric_values(successful_rows, "vm_queue_ms")
    roundtrip_values = numeric_values(successful_rows, "vm_roundtrip_ms")
    cpu_values = numeric_values(system_metric_rows, "cpu_percent")
    memory_values = numeric_values(system_metric_rows, "memory_percent")
    network_rows = [
        row
        for row in system_metric_rows
        if row.get("network_received_bytes") not in (None, "")
    ]

    received_delta = 0
    sent_delta = 0
    if len(network_rows) >= 2:
        received_delta = int(network_rows[-1]["network_received_bytes"]) - int(
            network_rows[0]["network_received_bytes"]
        )
        sent_delta = int(network_rows[-1]["network_sent_bytes"]) - int(
            network_rows[0]["network_sent_bytes"]
        )

    run_summary = {
        "run_id": run_id,
        "virtual_camera_count": len(virtual_cameras),
        "source_image_count": len(source_images),
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "expected_requests": expected_total_requests,
        "attempted_requests": len(result_rows),
        "successful_requests": len(successful_rows),
        "failed_requests": len(failed_rows),
        "missed_requests": max(0, expected_total_requests - len(result_rows)),
        "late_requests": len(late_rows),
        "average_person_count": average(
            numeric_values(successful_rows, "person_count")
        ),
        "average_vehicle_count": average(
            numeric_values(successful_rows, "vehicle_count")
        ),
        "average_vm_inference_ms": average(inference_values),
        "max_vm_inference_ms": maximum(inference_values),
        "average_vm_queue_ms": average(queue_values),
        "max_vm_queue_ms": maximum(queue_values),
        "average_vm_roundtrip_ms": average(roundtrip_values),
        "max_vm_roundtrip_ms": maximum(roundtrip_values),
        "average_cpu_percent": average(cpu_values),
        "max_cpu_percent": maximum(cpu_values),
        "average_memory_percent": average(memory_values),
        "max_memory_percent": maximum(memory_values),
        "network_received_bytes": max(0, received_delta),
        "network_sent_bytes": max(0, sent_delta),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "actual_elapsed_seconds": round(
            (finished_at - started_at).total_seconds(),
            2,
        ),
    }

    write_csv(run_summary_path, RUN_SUMMARY_FIELDNAMES, [run_summary])
    append_csv(COMBINED_SUMMARY_PATH, RUN_SUMMARY_FIELDNAMES, run_summary)

    parameters["finished_at_utc"] = finished_at.isoformat()
    write_json(parameters_path, parameters)

    print()
    print("Stress test finished")
    print(
        f"Requests: {len(successful_rows)}/{expected_total_requests} successful, "
        f"{len(failed_rows)} failed, "
        f"{run_summary['missed_requests']} missed, "
        f"{len(late_rows)} late"
    )
    print(
        "VM inference avg/max: "
        f"{run_summary['average_vm_inference_ms']} / "
        f"{run_summary['max_vm_inference_ms']} ms"
    )
    print(
        "VM round-trip avg/max: "
        f"{run_summary['average_vm_roundtrip_ms']} / "
        f"{run_summary['max_vm_roundtrip_ms']} ms"
    )
    print(f"All requests: {requests_path}")
    print(f"Per-camera averages: {camera_summary_path}")
    print(f"VM metrics: {metrics_path}")
    print(f"Run summary: {run_summary_path}")
    print(f"Combined experiment table: {COMBINED_SUMMARY_PATH}")


if __name__ == "__main__":
    run_stress_test()
