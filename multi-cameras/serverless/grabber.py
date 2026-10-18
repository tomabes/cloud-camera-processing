import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings


BASE_DIR = Path(__file__).resolve().parent
LOCAL_SETTINGS_FILE = BASE_DIR.parent.parent / "_secrets" / "local.settings.json"

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
ARCHITECTURE = "serverless"
USER_AGENT = "Mozilla/5.0"

DEFAULT_STORAGE_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
DEFAULT_RAW_CONTAINER = "raw-images"
DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_RESULTS_URL = "https://tomastorage.z43.web.core.windows.net/runs/results.csv"
DEFAULT_RESULT_WAIT_SECONDS = 120.0

RUNS_DIR = BASE_DIR / "runs"
CURRENT_RUN_FILENAME = BASE_DIR / "current_run.json"

UPLOAD_LOG_FIELDNAMES = [
    "run_id",
    "camera_count",
    "schedule_mode",
    "scheduled_at_utc",
    "capture_started_at_utc",
    "captured_at_utc",
    "uploaded_at_utc",
    "camera_id",
    "architecture",
    "frame_number",
    "blob_name",
    "capture_ms",
    "schedule_lag_ms",
    "upload_ms",
    "end_to_end_upload_ms",
    "image_size_bytes",
    "interval_seconds",
    "status",
    "error",
]

FUNCTION_RESULT_FIELDNAMES = [
    "processed_at_utc",
    "scheduled_at_utc",
    "captured_at_utc",
    "run_id",
    "camera_count",
    "expected_frames_per_camera",
    "schedule_mode",
    "camera_id",
    "architecture",
    "frame_number",
    "raw_blob",
    "processed_image",
    "person_count",
    "vehicle_count",
    "response_time_ms",
    "vision_ms",
    "queue_delay_ms",
    "end_to_end_ms",
    "image_size_bytes",
    "vision_model_version",
    "status",
    "error",
]


def load_local_settings_env():
    if not LOCAL_SETTINGS_FILE.exists():
        return
    with open(LOCAL_SETTINGS_FILE, encoding="utf-8") as settings_file:
        settings = json.load(settings_file)
    for name, value in settings.get("Values", {}).items():
        if value is not None:
            os.environ.setdefault(name, str(value))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the multi-camera upload experiment for Azure Functions."
    )
    parser.add_argument(
        "--storage-connection-string-env",
        default=DEFAULT_STORAGE_CONNECTION_STRING_ENV,
        help="Environment variable that stores the raw input Storage connection string.",
    )
    parser.add_argument("--container-name", default=DEFAULT_RAW_CONTAINER)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
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
    parser.add_argument("--results-url", default=DEFAULT_RESULTS_URL)
    parser.add_argument(
        "--result-wait-seconds",
        type=float,
        default=DEFAULT_RESULT_WAIT_SECONDS,
        help="How long to wait for Azure Function results after uploads finish.",
    )
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc)


def compact_utc(value):
    return value.strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"


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


def require_connection_string(env_name):
    connection_string = os.environ.get(env_name)
    if not connection_string and env_name == DEFAULT_STORAGE_CONNECTION_STRING_ENV:
        connection_string = os.environ.get("AzureWebJobsStorage")
    if not connection_string:
        raise RuntimeError(
            f"Set {env_name}, or add AzureWebJobsStorage to {LOCAL_SETTINGS_FILE}. "
            "Use the connection string of the Storage Account that contains raw-images."
        )
    return connection_string


def build_blob_name(
    camera_id,
    run_id,
    camera_count,
    schedule_mode,
    expected_frames_per_camera,
    frame_number,
    scheduled_at,
    captured_at,
):
    filename = (
        f"frame_{frame_number:04d}__cameras_{camera_count}__"
        f"mode_{schedule_mode}__expected_{expected_frames_per_camera}__"
        f"scheduled_{compact_utc(scheduled_at)}__"
        f"captured_{compact_utc(captured_at)}.jpg"
    )
    return f"runs/{run_id}/{camera_id}/raw-images/{filename}"


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def average(values):
    return round(sum(values) / len(values), 2) if values else None


def numeric_values(rows, fieldname):
    values = []
    for row in rows:
        value = row.get(fieldname)
        if value not in (None, ""):
            values.append(float(value))
    return values


def fetch_run_results(results_url, run_id):
    request = urllib.request.Request(
        f"{results_url}?v={time.time_ns()}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8-sig")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise

    by_blob = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("run_id") != run_id:
            continue
        key = row.get("raw_blob") or (
            f"{row.get('camera_id', '')}-{row.get('frame_number', '')}"
        )
        previous = by_blob.get(key)
        if previous is None or previous.get("status") != "ok" or row.get("status") == "ok":
            by_blob[key] = row
    return sorted(by_blob.values(), key=lambda row: row.get("processed_at_utc", ""))


def wait_for_run_results(results_url, run_id, expected_results, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    latest_rows = []
    while True:
        try:
            latest_rows = fetch_run_results(results_url, run_id)
        except Exception as error:
            print(f"Result check failed: {type(error).__name__}: {error}")

        successful = [row for row in latest_rows if row.get("status") == "ok"]
        print(
            f"Function results: {len(successful)}/{expected_results} successful "
            f"({len(latest_rows)} total result rows)"
        )
        if len(successful) >= expected_results or time.monotonic() >= deadline:
            return latest_rows
        time.sleep(2)


def write_csv(path, rows):
    fieldnames = list(rows[0]) if rows else FUNCTION_RESULT_FIELDNAMES
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_uploader():
    load_local_settings_env()
    args = parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        raise ValueError("duration-seconds and interval-seconds must be positive")
    if args.result_wait_seconds <= 0:
        raise ValueError("result-wait-seconds must be positive")

    selected_cameras = CAMERAS[: args.camera_count]
    started_at = utc_now()
    run_id = (
        f"serverless_{args.camera_count}cam_{args.schedule_mode}_"
        f"{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = RUNS_DIR / run_id
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_log_path = run_dir / "upload_log.csv"
    parameters_path = run_dir / "run_parameters.json"
    summary_path = run_dir / "run_summary.json"
    function_results_path = run_dir / "function_results.csv"
    expected_frames_per_camera = math.ceil(args.duration_seconds / args.interval_seconds)

    connection_string = require_connection_string(args.storage_connection_string_env)
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    raw_container = blob_service.get_container_client(args.container_name)

    current_run = {
        "architecture": ARCHITECTURE,
        "run_id": run_id,
        "status": "running",
        "upload_log_csv": str(upload_log_path.relative_to(BASE_DIR)),
        "summary_json": str(summary_path.relative_to(BASE_DIR)),
        "function_results_csv": str(function_results_path.relative_to(BASE_DIR)),
    }
    write_json(CURRENT_RUN_FILENAME, current_run)

    parameters = {
        "experiment": "Multi-camera Serverless Processing",
        "run_id": run_id,
        "camera_count": len(selected_cameras),
        "cameras": selected_cameras,
        "schedule_mode": args.schedule_mode,
        "capture_method": "Independent EarthCam HLS workers using ffmpeg",
        "processing_method": "Azure Blob trigger Function + Azure AI Vision",
        "architecture": ARCHITECTURE,
        "raw_container": args.container_name,
        "interval_seconds_per_camera": args.interval_seconds,
        "scheduling_window_seconds": args.duration_seconds,
        "expected_frames_per_camera": expected_frames_per_camera,
        "expected_total_uploads": expected_frames_per_camera * len(selected_cameras),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "successful_uploads": 0,
        "failed_uploads": 0,
        "results_url": args.results_url,
    }
    write_json(parameters_path, parameters)

    print("Fetching EarthCam stream URLs...")
    stream_urls = {}
    for camera in selected_cameras:
        stream_urls[camera["camera_id"]] = get_earthcam_stream_url(camera)
        print(f"Using {camera['camera_id']}: {camera['earthcam_camera_key']}")
    print(f"Run ID: {run_id}")
    print(f"Uploading raw frames to container: {args.container_name}")

    rows = []
    log_lock = threading.Lock()
    log_file = open(upload_log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=UPLOAD_LOG_FIELDNAMES)
    log_writer.writeheader()
    log_file.flush()

    def record_row(row):
        with log_lock:
            rows.append(row)
            log_writer.writerow(row)
            log_file.flush()

    experiment_start_monotonic = time.perf_counter() + 1.0
    experiment_end_monotonic = experiment_start_monotonic + args.duration_seconds
    experiment_start_utc = utc_now() + timedelta(seconds=1)
    parameters["measurement_started_at_utc"] = experiment_start_utc.isoformat()
    write_json(parameters_path, parameters)

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
            capture_started_at = utc_now()
            frame_start = time.perf_counter()
            schedule_lag_ms = max(
                0,
                int((time.perf_counter() - scheduled_monotonic) * 1000),
            )
            raw_path = work_dir / f"{camera_id}.jpg"
            row = {
                "run_id": run_id,
                "camera_count": len(selected_cameras),
                "schedule_mode": args.schedule_mode,
                "scheduled_at_utc": scheduled_at.isoformat(),
                "capture_started_at_utc": capture_started_at.isoformat(),
                "captured_at_utc": "",
                "uploaded_at_utc": "",
                "camera_id": camera_id,
                "architecture": ARCHITECTURE,
                "frame_number": frame_number,
                "blob_name": "",
                "capture_ms": "",
                "schedule_lag_ms": schedule_lag_ms,
                "upload_ms": "",
                "end_to_end_upload_ms": "",
                "image_size_bytes": "",
                "interval_seconds": args.interval_seconds,
                "status": "error",
                "error": "",
            }

            try:
                capture_start = time.perf_counter()
                capture_frame_with_ffmpeg(camera, stream_urls[camera_id], raw_path)
                capture_ms = int((time.perf_counter() - capture_start) * 1000)
                captured_at = utc_now()
                image_bytes = raw_path.read_bytes()
                blob_name = build_blob_name(
                    camera_id,
                    run_id,
                    len(selected_cameras),
                    args.schedule_mode,
                    expected_frames_per_camera,
                    frame_number,
                    scheduled_at,
                    captured_at,
                )

                upload_start = time.perf_counter()
                raw_container.upload_blob(
                    name=blob_name,
                    data=image_bytes,
                    overwrite=True,
                    content_settings=ContentSettings(
                        content_type="image/jpeg",
                        cache_control="no-cache",
                    ),
                    metadata={
                        "run_id": run_id,
                        "camera_id": camera_id,
                        "camera_count": str(len(selected_cameras)),
                        "schedule_mode": args.schedule_mode,
                        "frame_number": str(frame_number),
                        "scheduled_at_utc": scheduled_at.isoformat(),
                        "captured_at_utc": captured_at.isoformat(),
                    },
                )
                upload_ms = int((time.perf_counter() - upload_start) * 1000)
                row.update(
                    {
                        "captured_at_utc": captured_at.isoformat(),
                        "uploaded_at_utc": utc_now().isoformat(),
                        "blob_name": blob_name,
                        "capture_ms": capture_ms,
                        "upload_ms": upload_ms,
                        "end_to_end_upload_ms": int(
                            (time.perf_counter() - frame_start) * 1000
                        ),
                        "image_size_bytes": len(image_bytes),
                        "status": "ok",
                    }
                )
                print(
                    f"[{camera_id} {frame_number:03d}] uploaded={blob_name} "
                    f"lag={schedule_lag_ms} ms upload={upload_ms} ms"
                )
            except Exception as error:
                row["uploaded_at_utc"] = utc_now().isoformat()
                row["end_to_end_upload_ms"] = int(
                    (time.perf_counter() - frame_start) * 1000
                )
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"[{camera_id} {frame_number:03d}] ERROR {row['error']}")

            record_row(row)
            frame_number += 1

    try:
        with ThreadPoolExecutor(max_workers=len(selected_cameras)) as executor:
            futures = [
                executor.submit(camera_worker, camera, index)
                for index, camera in enumerate(selected_cameras)
            ]
            for future in futures:
                future.result()
    finally:
        log_file.close()

    uploads_finished_at = utc_now()
    successful_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    expected_total = expected_frames_per_camera * len(selected_cameras)
    summary = {
        "architecture": ARCHITECTURE,
        "run_id": run_id,
        "camera_count": len(selected_cameras),
        "schedule_mode": args.schedule_mode,
        "started_at_utc": started_at.isoformat(),
        "uploads_finished_at_utc": uploads_finished_at.isoformat(),
        "scheduling_window_seconds": args.duration_seconds,
        "upload_phase_elapsed_seconds": round(
            (uploads_finished_at - experiment_start_utc).total_seconds(), 2
        ),
        "interval_seconds_per_camera": args.interval_seconds,
        "expected_frames_per_camera": expected_frames_per_camera,
        "expected_total_uploads": expected_total,
        "attempted_uploads": len(rows),
        "successful_uploads": len(successful_rows),
        "failed_uploads": len(failed_rows),
        "upload_success_percent": round(
            len(successful_rows) / expected_total * 100, 2
        ) if expected_total else 0,
        "average_capture_ms": average(
            [int(row["capture_ms"]) for row in successful_rows]
        ),
        "average_upload_ms": average(
            [int(row["upload_ms"]) for row in successful_rows]
        ),
        "average_schedule_lag_ms": average(
            [int(row["schedule_lag_ms"]) for row in rows]
        ),
        "uploaded_image_bytes": sum(
            int(row["image_size_bytes"]) for row in successful_rows
        ),
    }
    current_run["status"] = "waiting_for_function"
    write_json(CURRENT_RUN_FILENAME, current_run)
    print("Waiting for Azure Function results...")
    function_rows = wait_for_run_results(
        args.results_url,
        run_id,
        len(successful_rows),
        args.result_wait_seconds,
    )
    write_csv(function_results_path, function_rows)

    processed_rows = [row for row in function_rows if row.get("status") == "ok"]
    processing_error_rows = [row for row in function_rows if row.get("status") != "ok"]
    end_to_end_values = numeric_values(processed_rows, "end_to_end_ms")
    vision_values = numeric_values(processed_rows, "vision_ms")
    late_rows = [
        row
        for row in processed_rows
        if row.get("end_to_end_ms") not in (None, "")
        and float(row["end_to_end_ms"]) > args.interval_seconds * 1000
    ]
    finished_at = utc_now()
    summary.update(
        {
            "finished_at_utc": finished_at.isoformat(),
            "processed_successfully": len(processed_rows),
            "processing_errors": len(processing_error_rows),
            "missing_function_results": max(
                0, len(successful_rows) - len(function_rows)
            ),
            "late_results": len(late_rows),
            "average_vision_ms": average(vision_values),
            "max_vision_ms": max(vision_values) if vision_values else None,
            "average_end_to_end_ms": average(end_to_end_values),
            "max_end_to_end_ms": max(end_to_end_values) if end_to_end_values else None,
        }
    )
    write_json(summary_path, summary)
    parameters["finished_at_utc"] = finished_at.isoformat()
    parameters["successful_uploads"] = len(successful_rows)
    parameters["failed_uploads"] = len(failed_rows)
    write_json(parameters_path, parameters)
    current_run["status"] = "finished"
    write_json(CURRENT_RUN_FILENAME, current_run)
    print(f"Upload log saved to: {upload_log_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_uploader()
