import argparse
import json
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IMGSZ = 640

MODEL = None
MODEL_NAME = DEFAULT_MODEL
INFERENCE_LOCK = threading.Lock()
METRICS_LOCK = threading.Lock()
PREVIOUS_CPU_TOTAL = None
PREVIOUS_CPU_IDLE = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection server on the VM.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_cpu_times():
    with open("/proc/stat", encoding="utf-8") as stat_file:
        values = stat_file.readline().split()[1:]
    numbers = [int(value) for value in values]
    idle = numbers[3] + numbers[4]
    return sum(numbers), idle


def read_memory_metrics():
    values = {}
    with open("/proc/meminfo", encoding="utf-8") as memory_file:
        for line in memory_file:
            name, raw_value = line.split(":", 1)
            values[name] = int(raw_value.strip().split()[0])

    total_kb = values["MemTotal"]
    available_kb = values["MemAvailable"]
    used_kb = total_kb - available_kb
    return {
        "memory_used_mb": round(used_kb / 1024, 2),
        "memory_total_mb": round(total_kb / 1024, 2),
        "memory_percent": round(used_kb / total_kb * 100, 2),
    }


def read_network_metrics():
    received_bytes = 0
    sent_bytes = 0
    with open("/proc/net/dev", encoding="utf-8") as network_file:
        for line in network_file.readlines()[2:]:
            interface, raw_values = line.split(":", 1)
            if interface.strip() == "lo":
                continue
            values = raw_values.split()
            received_bytes += int(values[0])
            sent_bytes += int(values[8])
    return {
        "network_received_bytes": received_bytes,
        "network_sent_bytes": sent_bytes,
    }


def read_process_memory_mb():
    with open("/proc/self/status", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 2)
    return 0.0


def collect_system_metrics():
    global PREVIOUS_CPU_TOTAL, PREVIOUS_CPU_IDLE

    with METRICS_LOCK:
        cpu_total, cpu_idle = read_cpu_times()
        cpu_percent = None
        if PREVIOUS_CPU_TOTAL is not None:
            total_delta = cpu_total - PREVIOUS_CPU_TOTAL
            idle_delta = cpu_idle - PREVIOUS_CPU_IDLE
            if total_delta > 0:
                cpu_percent = round((1 - idle_delta / total_delta) * 100, 2)
        PREVIOUS_CPU_TOTAL = cpu_total
        PREVIOUS_CPU_IDLE = cpu_idle

    return {
        "timestamp_utc": utc_now_iso(),
        "cpu_percent": cpu_percent,
        **read_memory_metrics(),
        "detector_process_memory_mb": read_process_memory_mb(),
        **read_network_metrics(),
        "load_average_1m": round(os.getloadavg()[0], 2),
    }


def run_detection(frame, confidence_threshold, image_size):
    queue_start = time.perf_counter()
    with INFERENCE_LOCK:
        queue_ms = int((time.perf_counter() - queue_start) * 1000)
        detection_start = time.perf_counter()
        detections = MODEL.predict(
            source=frame,
            conf=confidence_threshold,
            imgsz=image_size,
            verbose=False,
        )
        inference_ms = int((time.perf_counter() - detection_start) * 1000)

    person_count = 0
    vehicle_count = 0
    boxes = []

    if detections and detections[0].boxes is not None:
        for box in detections[0].boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            if confidence < confidence_threshold:
                continue

            if class_id == PERSON_CLASS_ID:
                label = "person"
                person_count += 1
            elif class_id in VEHICLE_CLASS_IDS:
                label = VEHICLE_CLASS_IDS[class_id]
                vehicle_count += 1
            else:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            boxes.append(
                {
                    "label": label,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return person_count, vehicle_count, boxes, queue_ms, inference_ms


class DetectionHandler(BaseHTTPRequestHandler):
    server_version = "VMYOLODetector/2.0"

    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "model": MODEL_NAME,
                    "default_confidence": DEFAULT_CONFIDENCE,
                    "default_imgsz": DEFAULT_IMGSZ,
                },
            )
            return

        if parsed_url.path == "/metrics":
            self.send_json(200, collect_system_metrics())
            return

        self.send_json(404, {"error": "Use POST /detect, GET /health, or GET /metrics"})

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path != "/detect":
            self.send_json(404, {"error": "Use POST /detect"})
            return

        query = urllib.parse.parse_qs(parsed_url.query)
        confidence_threshold = float(
            query.get("confidence", [DEFAULT_CONFIDENCE])[0]
        )
        image_size = int(query.get("imgsz", [DEFAULT_IMGSZ])[0])

        request_start = time.perf_counter()
        content_length = int(self.headers["Content-Length"])
        image_bytes = self.rfile.read(content_length)

        decode_start = time.perf_counter()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        decode_ms = int((time.perf_counter() - decode_start) * 1000)
        if frame is None:
            raise RuntimeError("OpenCV could not decode the uploaded frame")

        person_count, vehicle_count, boxes, queue_ms, inference_ms = run_detection(
            frame,
            confidence_threshold,
            image_size,
        )
        server_total_ms = int((time.perf_counter() - request_start) * 1000)

        self.send_json(
            200,
            {
                "person_count": person_count,
                "vehicle_count": vehicle_count,
                "boxes": boxes,
                "request_image_size_bytes": len(image_bytes),
                "vm_decode_ms": decode_ms,
                "vm_queue_ms": queue_ms,
                "vm_inference_ms": inference_ms,
                "vm_total_ms": server_total_ms,
                "confidence_threshold": confidence_threshold,
                "image_size": image_size,
            },
        )


def main():
    global MODEL, MODEL_NAME

    args = parse_args()
    MODEL_NAME = args.model
    print("Loading YOLOv8 model on VM...")
    MODEL = YOLO(args.model)
    collect_system_metrics()

    server = ThreadingHTTPServer((args.host, args.port), DetectionHandler)
    print(f"VM detector listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVM detector stopped.")


if __name__ == "__main__":
    main()
