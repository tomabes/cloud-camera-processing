import csv
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import azure.functions as func
import cv2
import numpy as np
import requests
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings


VEHICLE_LABELS = {
    "automobile",
    "bicycle",
    "bike",
    "bus",
    "car",
    "land vehicle",
    "minibus",
    "motorbike",
    "motorcycle",
    "pickup truck",
    "taxi",
    "truck",
    "van",
    "vehicle",
}

CSV_FIELDNAMES = [
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

LOCAL_SETTINGS_FILE = Path(__file__).resolve().parents[3] / "_secrets" / "local.settings.json"


def load_local_settings_env():
    if not LOCAL_SETTINGS_FILE.exists():
        return

    with open(LOCAL_SETTINGS_FILE, encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

    for name, value in settings.get("Values", {}).items():
        if value is not None:
            os.environ.setdefault(name, str(value))


load_local_settings_env()


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required application setting: {name}")
    return value


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def vision_analyze_url(endpoint, api_version):
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/computervision"):
        base = endpoint
    else:
        base = f"{endpoint}/computervision"
    return f"{base}/imageanalysis:analyze?api-version={api_version}&features=people,objects&language=en"


def normalized_label(value):
    return value.strip().lower().replace("_", " ").replace("-", " ")


def analyze_image(image_bytes):
    endpoint = require_env("AZURE_VISION_ENDPOINT")
    key = require_env("AZURE_VISION_KEY")
    api_version = os.environ.get("AZURE_VISION_API_VERSION", "2024-02-01")
    person_confidence_threshold = float(os.environ.get("PERSON_CONFIDENCE_THRESHOLD", "0.2"))
    vehicle_confidence_threshold = float(os.environ.get("VEHICLE_CONFIDENCE_THRESHOLD", "0.2"))

    request_start = time.perf_counter()
    response = requests.post(
        vision_analyze_url(endpoint, api_version),
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/octet-stream",
        },
        data=image_bytes,
        timeout=30,
    )
    response.raise_for_status()
    vision_ms = int((time.perf_counter() - request_start) * 1000)
    payload = response.json()

    people = []
    for item in payload.get("peopleResult", {}).get("values", []):
        confidence = float(item.get("confidence", 0))
        box = item.get("boundingBox", {})
        width = int(box.get("w", 0))
        height = int(box.get("h", 0))
        if confidence < person_confidence_threshold or width <= 0 or height <= 0:
            continue
        people.append(
            {
                "x": int(box.get("x", 0)),
                "y": int(box.get("y", 0)),
                "w": width,
                "h": height,
                "confidence": round(confidence, 4),
            }
        )

    vehicles = []
    for item in payload.get("objectsResult", {}).get("values", []):
        tags = item.get("tags", [])
        matching_tag = None
        for tag in tags:
            tag_name = normalized_label(str(tag.get("name", "")))
            tag_confidence = float(tag.get("confidence", 0))
            if tag_name in VEHICLE_LABELS and tag_confidence >= vehicle_confidence_threshold:
                matching_tag = {
                    "label": tag_name,
                    "confidence": round(tag_confidence, 4),
                }
                break

        if matching_tag is None:
            continue

        box = item.get("boundingBox", {})
        width = int(box.get("w", 0))
        height = int(box.get("h", 0))
        if width <= 0 or height <= 0:
            continue

        vehicles.append(
            {
                "x": int(box.get("x", 0)),
                "y": int(box.get("y", 0)),
                "w": width,
                "h": height,
                **matching_tag,
            }
        )

    return people, vehicles, vision_ms, payload.get("modelVersion", "")


def draw_detection_boxes(image_bytes, people, vehicles):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode JPEG image for bbox drawing")

    for person in people:
        x1 = int(person["x"])
        y1 = int(person["y"])
        x2 = x1 + int(person["w"])
        y2 = y1 + int(person["h"])
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"person {float(person['confidence']):.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    for vehicle in vehicles:
        label = vehicle["label"]
        x1 = int(vehicle["x"])
        y1 = int(vehicle["y"])
        x2 = x1 + int(vehicle["w"])
        y2 = y1 + int(vehicle["h"])
        color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {float(vehicle['confidence']):.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    success, encoded_image = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not success:
        raise RuntimeError("Could not encode processed JPEG image")

    return encoded_image.tobytes()


def parse_compact_utc(value):
    return datetime.strptime(value, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)


def parse_blob_context(raw_blob_name):
    parts = raw_blob_name.split("/")
    raw_container = os.environ.get("RAW_IMAGES_CONTAINER", "raw-images")
    if parts and parts[0] == raw_container:
        parts = parts[1:]

    context = {
        "camera_id": "unknown",
        "run_id": "legacy",
        "camera_count": 1,
        "expected_frames_per_camera": "",
        "schedule_mode": "legacy",
        "frame_number": "",
        "scheduled_at": None,
        "captured_at": None,
    }

    if len(parts) == 5 and parts[0] == "runs" and parts[3] == "raw-images":
        filename_match = re.fullmatch(
            r"frame_(\d+)__cameras_(\d+)__mode_(.+?)__expected_(\d+)__"
            r"scheduled_(.+?)__captured_(.+?)\.jpg",
            parts[4],
        )
        if filename_match is None:
            return context
        context.update(
            {
                "run_id": parts[1],
                "camera_id": parts[2],
                "frame_number": int(filename_match.group(1)),
                "camera_count": int(filename_match.group(2)),
                "schedule_mode": filename_match.group(3),
                "expected_frames_per_camera": int(filename_match.group(4)),
                "scheduled_at": parse_compact_utc(filename_match.group(5)),
                "captured_at": parse_compact_utc(filename_match.group(6)),
            }
        )
        return context

    if len(parts) >= 8 and parts[0] == "runs" and parts[2] == "raw":
        run_id = parts[1]
        camera_id = parts[3]
        camera_count_part = parts[4]
        schedule_mode = parts[5]
        expected_frames_part = parts[6]
        filename = parts[7]
    elif len(parts) >= 6:
        camera_id = parts[0]
        run_id = parts[1]
        camera_count_part = parts[2]
        schedule_mode = parts[3]
        expected_frames_part = parts[4]
        filename = parts[5]
    else:
        return context

    context["camera_id"] = camera_id
    context["run_id"] = run_id
    camera_count_match = re.fullmatch(r"(\d+)-cameras", camera_count_part)
    if camera_count_match:
        context["camera_count"] = int(camera_count_match.group(1))
    context["schedule_mode"] = schedule_mode
    expected_frames_match = re.fullmatch(r"(\d+)-frames", expected_frames_part)
    if expected_frames_match:
        context["expected_frames_per_camera"] = int(expected_frames_match.group(1))

    filename_match = re.fullmatch(
        r"scheduled_(.+?)__captured_(.+?)__frame_(\d+)\.jpg",
        filename,
    )
    if filename_match:
        context["scheduled_at"] = parse_compact_utc(filename_match.group(1))
        context["captured_at"] = parse_compact_utc(filename_match.group(2))
        context["frame_number"] = int(filename_match.group(3))
    return context


def web_container_client():
    connection_string = os.environ.get("OUTPUT_STORAGE_CONNECTION_STRING") or require_env("AzureWebJobsStorage")
    container_name = os.environ.get("WEB_CONTAINER_NAME", "$web")
    return BlobServiceClient.from_connection_string(connection_string).get_container_client(container_name)


def upload_web_blob(container, name, data, content_type):
    container.upload_blob(
        name=name,
        data=data,
        overwrite=True,
        content_settings=ContentSettings(
            content_type=content_type,
            cache_control="no-cache",
        ),
    )


def append_csv_row(container, blob_name, row):
    csv_blob = container.get_blob_client(blob_name)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDNAMES)
    writer.writerow(row)
    row_bytes = output.getvalue().encode("utf-8")

    if not csv_blob.exists():
        try:
            csv_blob.create_append_blob(
                content_settings=ContentSettings(
                    content_type="text/csv; charset=utf-8",
                    cache_control="no-cache",
                )
            )
            header = io.StringIO()
            csv.DictWriter(header, fieldnames=CSV_FIELDNAMES).writeheader()
            csv_blob.append_block(header.getvalue().encode("utf-8"))
        except ResourceExistsError:
            pass

    csv_blob.append_block(row_bytes)


def process_raw_image(inputblob: func.InputStream):
    start = time.perf_counter()
    raw_blob_name = inputblob.name
    image_bytes = inputblob.read()
    context = parse_blob_context(raw_blob_name)
    processed_at = datetime.now(timezone.utc)
    web_container = web_container_client()
    run_root = f"runs/{context['run_id']}"
    camera_root = f"{run_root}/{context['camera_id']}"
    frame_stem = f"frame_{context['frame_number']:04d}"
    raw_image_name = f"{camera_root}/raw-images/{frame_stem}.jpg"
    processed_image_name = f"{camera_root}/processed-images/{frame_stem}.jpg"
    result_blob_name = f"{run_root}/json/{context['camera_id']}/{frame_stem}.json"
    run_results_name = f"{run_root}/results.csv"
    latest_image_name = f"{run_root}/latest/{context['camera_id']}.jpg"
    latest_camera_name = f"{run_root}/latest/{context['camera_id']}.json"
    scheduled_at = context["scheduled_at"]
    captured_at = context["captured_at"]

    base_row = {
        "processed_at_utc": processed_at.isoformat(),
        "scheduled_at_utc": scheduled_at.isoformat() if scheduled_at else "",
        "captured_at_utc": captured_at.isoformat() if captured_at else "",
        "run_id": context["run_id"],
        "camera_count": context["camera_count"],
        "expected_frames_per_camera": context["expected_frames_per_camera"],
        "schedule_mode": context["schedule_mode"],
        "camera_id": context["camera_id"],
        "architecture": "serverless",
        "frame_number": context["frame_number"],
        "raw_blob": raw_image_name,
        "processed_image": processed_image_name,
        "person_count": "",
        "vehicle_count": "",
        "response_time_ms": "",
        "vision_ms": "",
        "queue_delay_ms": "",
        "end_to_end_ms": "",
        "image_size_bytes": len(image_bytes),
        "vision_model_version": "",
        "status": "error",
        "error": "",
    }

    try:
        upload_web_blob(web_container, raw_image_name, image_bytes, "image/jpeg")
        people, vehicles, vision_ms, model_version = analyze_image(image_bytes)
        processed_image = draw_detection_boxes(image_bytes, people, vehicles)
        response_time_ms = int((time.perf_counter() - start) * 1000)
        processed_at = datetime.now(timezone.utc)
        end_to_end_ms = ""
        queue_delay_ms = ""
        if captured_at is not None:
            end_to_end_ms = max(0, int((processed_at - captured_at).total_seconds() * 1000))
            queue_delay_ms = max(0, end_to_end_ms - response_time_ms)

        row = {
            **base_row,
            "processed_at_utc": processed_at.isoformat(),
            "person_count": len(people),
            "vehicle_count": len(vehicles),
            "response_time_ms": response_time_ms,
            "vision_ms": vision_ms,
            "queue_delay_ms": queue_delay_ms,
            "end_to_end_ms": end_to_end_ms,
            "vision_model_version": model_version,
            "status": "ok",
        }
        latest_payload = {
            **row,
            "people": people,
            "vehicles": vehicles,
            "result_blob": result_blob_name,
            "latest_image": latest_image_name,
            "trigger_blob": raw_blob_name,
        }

        upload_web_blob(web_container, processed_image_name, processed_image, "image/jpeg")
        upload_web_blob(web_container, latest_image_name, processed_image, "image/jpeg")
        payload_bytes = json.dumps(latest_payload, indent=2).encode("utf-8")
        upload_web_blob(
            web_container,
            latest_camera_name,
            payload_bytes,
            "application/json; charset=utf-8",
        )
        upload_web_blob(
            web_container,
            result_blob_name,
            payload_bytes,
            "application/json; charset=utf-8",
        )
        append_csv_row(web_container, "runs/results.csv", row)
        append_csv_row(web_container, run_results_name, row)

        logging.info(
            "Processed %s: people=%s vehicles=%s response_time_ms=%s vision_ms=%s",
            raw_blob_name,
            len(people),
            len(vehicles),
            response_time_ms,
            vision_ms,
        )
    except Exception as error:
        error_row = {
            **base_row,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "processed_image": "",
            "response_time_ms": int((time.perf_counter() - start) * 1000),
            "error": f"{type(error).__name__}: {error}",
        }
        error_payload = {
            **error_row,
            "result_blob": result_blob_name,
            "trigger_blob": raw_blob_name,
        }
        upload_web_blob(
            web_container,
            result_blob_name,
            json.dumps(error_payload, indent=2).encode("utf-8"),
            "application/json; charset=utf-8",
        )
        append_csv_row(web_container, "runs/results.csv", error_row)
        append_csv_row(web_container, run_results_name, error_row)
        logging.exception("Failed to process %s", raw_blob_name)
        raise
