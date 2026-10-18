import argparse
import csv
import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
CACHE_DIR = BASE_DIR / ".cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ultralytics import YOLO

CAMERA_ID = "times_square"
CAMERA_PAGE_URL = "https://www.earthcam.com/usa/newyork/timessquare/?cam=tsstreet"
EARTHCAM_CAMERA_KEY = "tsstreet"
ARCHITECTURE = "local"
USER_AGENT = "Mozilla/5.0"
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_DURATION_SECONDS = 5 * 60
DEFAULT_INTERVAL_SECONDS = 5
DEFAULT_SAVE_EVERY = 5
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IMGSZ = 1280

CSV_FILENAME = BASE_DIR / "database.csv"
LATEST_RAW_FRAME = BASE_DIR / "raw_frame.jpg"
LATEST_PROCESSED_FRAME = BASE_DIR / "processed_frame.jpg"
PLOT_FILENAME = BASE_DIR / "people_count_plot.png"
PARAMETERS_FILENAME = BASE_DIR / "run_parameters.json"
RAW_FRAMES_DIR = BASE_DIR / "frames" / "raw"
PROCESSED_FRAMES_DIR = BASE_DIR / "frames" / "processed"

CSV_FIELDNAMES = [
    "timestamp",
    "camera_id",
    "architecture",
    "frame_number",
    "person_count",
    "vehicle_count",
    "response_time_ms",
    "image_size_bytes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the local Part 1 person-counting experiment."
    )
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def fetch_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def get_times_square_stream_url():
    page_text = fetch_text(CAMERA_PAGE_URL)
    camera_pattern = re.escape(EARTHCAM_CAMERA_KEY)
    match = re.search(
        rf'"{camera_pattern}"\s*:\s*\{{.*?"stream"\s*:\s*"([^"]+)"',
        page_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find stream URL for {EARTHCAM_CAMERA_KEY}")

    return match.group(1).replace("\\/", "/")


def capture_frame_with_ffmpeg(stream_url, output_path):
    headers = (
        f"Referer: {CAMERA_PAGE_URL}\r\n"
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


def detect_objects(frame, model, confidence_threshold, image_size):
    detections = model.predict(
        source=frame,
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False,
    )
    person_count = 0
    vehicle_count = 0

    if not detections or detections[0].boxes is None:
        return person_count, vehicle_count, frame

    for box in detections[0].boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        if confidence < confidence_threshold:
            continue

        if class_id == PERSON_CLASS_ID:
            label = "person"
            color = (0, 255, 0)
            person_count += 1
        elif class_id in VEHICLE_CLASS_IDS:
            label = VEHICLE_CLASS_IDS[class_id]
            color = (0, 165, 255)
            vehicle_count += 1
        else:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
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

    return person_count, vehicle_count, frame


def should_save_frame(frame_number, total_frames):
    return (
        frame_number == 1
        or frame_number == total_frames
        or frame_number % DEFAULT_SAVE_EVERY == 0
    )


def save_parameters(parameters):
    with open(PARAMETERS_FILENAME, "w", encoding="utf-8") as parameters_file:
        json.dump(parameters, parameters_file, indent=2)


def draw_people_count_plot(rows, output_path):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No frames were captured",
            ha="center",
            va="center",
            fontsize=16,
            color="#9b1c1c",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    timestamps = [
        datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        for row in rows
    ]
    people_counts = [int(row["person_count"]) for row in rows]
    vehicle_counts = [int(row["vehicle_count"]) for row in rows]
    average_people = sum(people_counts) / len(people_counts)
    average_vehicles = sum(vehicle_counts) / len(vehicle_counts)
    max_count = max(people_counts + vehicle_counts)

    ax.plot(
        timestamps,
        people_counts,
        color="#2563eb",
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.6,
        label="Detected people",
    )
    ax.plot(
        timestamps,
        vehicle_counts,
        color="#16a34a",
        linewidth=2.2,
        marker="s",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Detected vehicles",
    )
    ax.axhline(
        average_people,
        color="#2563eb",
        linewidth=1.8,
        linestyle="--",
        alpha=0.55,
        label=f"People avg: {average_people:.1f}",
    )
    ax.axhline(
        average_vehicles,
        color="#16a34a",
        linewidth=1.8,
        linestyle="--",
        alpha=0.55,
        label=f"Vehicles avg: {average_vehicles:.1f}",
    )

    ax.set_title(
        "People and Vehicles Detected Over Time",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )
    ax.set_xlabel("Time", fontsize=12, labelpad=10)
    ax.set_ylabel("Detected objects", fontsize=12, labelpad=10)
    ax.set_ylim(bottom=0, top=max(max_count + 2, 5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))

    ax.grid(True, axis="y", color="#d8dee9", linewidth=0.9)
    ax.grid(True, axis="x", color="#edf0f5", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4b2")
    ax.spines["bottom"].set_color("#9aa4b2")
    ax.tick_params(axis="both", labelsize=10, colors="#475569")
    fig.autofmt_xdate(rotation=30, ha="right")

    summary = (
        f"Frames: {len(people_counts)}   "
        f"People avg: {average_people:.1f}   "
        f"Vehicles avg: {average_vehicles:.1f}   "
        f"Max objects/frame: {max_count}"
    )
    ax.text(
        0,
        1.02,
        summary,
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
        va="bottom",
    )
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_experiment():
    args = parse_args()
    total_frames = args.frames or max(1, args.duration_seconds // args.interval_seconds)

    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    parameters = {
        "assignment_part": "Part 1 - Local Work",
        "camera_id": CAMERA_ID,
        "camera_page_url": CAMERA_PAGE_URL,
        "earthcam_camera_key": EARTHCAM_CAMERA_KEY,
        "capture_method": "EarthCam HLS stream captured with ffmpeg",
        "architecture": ARCHITECTURE,
        "model": args.model,
        "detected_classes": {
            "people": ["person"],
            "vehicles": list(VEHICLE_CLASS_IDS.values()),
        },
        "confidence_threshold": args.confidence,
        "inference_image_size": args.imgsz,
        "interval_seconds": args.interval_seconds,
        "duration_seconds": args.duration_seconds,
        "total_frames_requested": total_frames,
        "save_every_n_frames": DEFAULT_SAVE_EVERY,
        "csv_file": CSV_FILENAME.name,
        "latest_raw_frame": LATEST_RAW_FRAME.name,
        "latest_processed_frame": LATEST_PROCESSED_FRAME.name,
        "plot_file": PLOT_FILENAME.name,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "successful_frames": 0,
        "failed_frames": 0,
    }
    save_parameters(parameters)

    print("Loading YOLOv8 model...")
    yolo_net = YOLO(args.model)
    print("Fetching Times Square stream URL...")
    stream_url = get_times_square_stream_url()
    print(f"Using EarthCam Times Square stream: {EARTHCAM_CAMERA_KEY}")

    rows = []

    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for frame_number in range(1, total_frames + 1):
            iteration_start = time.perf_counter()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            capture_frame_with_ffmpeg(stream_url, LATEST_RAW_FRAME)
            frame = cv2.imread(str(LATEST_RAW_FRAME))
            if frame is None:
                raise RuntimeError("OpenCV could not read the captured frame")

            raw_frame = frame.copy()
            image_size_bytes = LATEST_RAW_FRAME.stat().st_size

            detection_start = time.perf_counter()
            person_count, vehicle_count, processed_frame = detect_objects(
                frame,
                yolo_net,
                args.confidence,
                args.imgsz,
            )
            response_time_ms = int((time.perf_counter() - detection_start) * 1000)

            cv2.imwrite(str(LATEST_PROCESSED_FRAME), processed_frame)

            if should_save_frame(frame_number, total_frames):
                raw_path = RAW_FRAMES_DIR / f"{frame_number:03d}_raw.jpg"
                processed_path = PROCESSED_FRAMES_DIR / f"{frame_number:03d}_processed.jpg"
                cv2.imwrite(str(raw_path), raw_frame)
                cv2.imwrite(str(processed_path), processed_frame)

            row = {
                "timestamp": timestamp,
                "camera_id": CAMERA_ID,
                "architecture": ARCHITECTURE,
                "frame_number": frame_number,
                "person_count": person_count,
                "vehicle_count": vehicle_count,
                "response_time_ms": response_time_ms,
                "image_size_bytes": image_size_bytes,
            }
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            print(
                f"[{frame_number:03d}/{total_frames:03d}] "
                f"{timestamp} people={person_count} "
                f"vehicles={vehicle_count} detection_ms={response_time_ms}"
            )

            elapsed = time.perf_counter() - iteration_start
            sleep_seconds = args.interval_seconds - elapsed
            if frame_number < total_frames and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    parameters["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parameters["successful_frames"] = len(rows)
    save_parameters(parameters)
    draw_people_count_plot(rows, PLOT_FILENAME)
    print(f"CSV saved to: {CSV_FILENAME}")
    print(f"Plot saved to: {PLOT_FILENAME}")
    print(f"Run parameters saved to: {PARAMETERS_FILENAME}")


if __name__ == "__main__":
    run_experiment()
