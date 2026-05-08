import os
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


# Set your video path here (example: r"D:\videos\sample.mp4")
VIDEO_PATH = r"src\face_detect.mp4"
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# Official MediaPipe model assets.
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe hand skeleton edges (21 points).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def download_if_missing(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    print(f"Downloading model: {output_path.name}")
    urllib.request.urlretrieve(url, str(output_path))


def draw_points(image, landmarks, color=(255, 255, 0), radius=1):
    height, width = image.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), radius, color, -1)


def draw_connections(image, landmarks, connections, color=(0, 255, 0), thickness=2):
    height, width = image.shape[:2]
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))
        cv2.line(image, p1, p2, color, thickness)


def build_face_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def build_hand_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    # Python 3.14 + current mediapipe package path: use Tasks API models.
    try:
        download_if_missing(FACE_MODEL_URL, FACE_MODEL_PATH)
        download_if_missing(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as exc:
        print("Failed to download model files automatically.")
        print(f"Error: {exc}")
        raise SystemExit(1)

    vid = cv2.VideoCapture(VIDEO_PATH)
    if not vid.isOpened():
        print(f"Cannot open video source: {VIDEO_PATH}")
        raise SystemExit(1)

    face_landmarker = build_face_landmarker()
    hand_landmarker = build_hand_landmarker()

    previous_time = time.time()
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret or frame is None:
            break

        frame = cv2.resize(frame, (800, 600))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                draw_points(frame, face_landmarks, color=(255, 0, 255), radius=1)

        if hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                if idx == 0:
                    line_color = (0, 255, 255)
                    point_color = (0, 255, 0)
                else:
                    line_color = (255, 255, 0)
                    point_color = (255, 0, 0)
                draw_connections(
                    frame,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    color=line_color,
                    thickness=2,
                )
                draw_points(frame, hand_landmarks, color=point_color, radius=3)

        current_time = time.time()
        fps = 1 / (current_time - previous_time) if current_time != previous_time else 0
        previous_time = current_time
        cv2.putText(
            frame,
            f"{int(fps)} FPS",
            (10, 70),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Facial and Hand Landmarks", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    face_landmarker.close()
    hand_landmarker.close()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()