from pathlib import Path

import cv2
import mediapipe as mp
import pyttsx3
from scipy.spatial import distance

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
_VIDEO_NAME = "face_detect.mp4"


def _iter_video_paths():
    """Try script dir, repo root, and cwd — `src/face_detect.mp4` is often at repo root."""
    seen = set()
    for path in (
        SCRIPT_DIR / "src" / _VIDEO_NAME,
        SCRIPT_DIR / _VIDEO_NAME,
        REPO_ROOT / "src" / _VIDEO_NAME,
        Path.cwd() / "src" / _VIDEO_NAME,
        Path.cwd() / "08_Drowsiness_Detection" / "src" / _VIDEO_NAME,
        Path.cwd() / "08_Drowsiness_Detection" / _VIDEO_NAME,
    ):
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        yield path


def open_capture() -> cv2.VideoCapture:
    for path in _iter_video_paths():
        if path.is_file():
            resolved = str(path.resolve())
            cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
            if cap.isOpened():
                print(f"Using video file: {resolved}")
                return cap
            cap.release()
            print(f"Warning: found video file but OpenCV could not open it: {resolved}")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Using default webcam (0).")
        return cap
    tried = "\n  ".join(str(p) for p in _iter_video_paths())
    raise SystemExit(
        "No video source found. Tried:\n  "
        + tried
        + "\n\nIf the file exists, OpenCV may still fail to open it (codec). "
        "Try re-encoding to H.264 in an .mp4 container, or use a webcam."
    )


# Six landmarks per eye in the order expected by `detect_eye`.
LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)


try:
    engine = pyttsx3.init()
except Exception as exc:
    print(f"pyttsx3 init failed ({exc}); continuing without voice alerts.")
    engine = None

cap = open_capture()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def detect_eye(eye):
    """Eye aspect ratio; `eye` is six (x, y) points: corners at 0 and 3, vertical pairs (1,5) and (2,4)."""
    poi_a = distance.euclidean(eye[1], eye[5])
    poi_b = distance.euclidean(eye[2], eye[4])
    poi_c = distance.euclidean(eye[0], eye[3])
    if poi_c == 0:
        return 1.0
    return (poi_a + poi_b) / (2.0 * poi_c)


def landmarks_to_eye_points(face_landmarks, indices, w, h):
    pts = []
    for i in indices:
        lm = face_landmarks.landmark[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("End of stream or read failed; exiting.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_eye = landmarks_to_eye_points(face_landmarks, LEFT_EYE_IDX, w, h)
            right_eye = landmarks_to_eye_points(face_landmarks, RIGHT_EYE_IDX, w, h)

            for eye, color in ((left_eye, (255, 255, 0)), (right_eye, (0, 255, 0))):
                for i in range(6):
                    j = (i + 1) % 6
                    cv2.line(frame, eye[i], eye[j], color, 1)

            right_ratio = detect_eye(right_eye)
            left_ratio = detect_eye(left_eye)
            eye_rat = round((left_ratio + right_ratio) / 2.0, 2)

            if eye_rat < 0.25:
                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED",
                    (50, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (21, 56, 210),
                    3,
                )
                cv2.putText(
                    frame,
                    "Alert!!!! WAKE UP DUDE",
                    (50, 450),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (21, 56, 212),
                    3,
                )
                if engine is not None:
                    engine.say("Alert!!!! WAKE UP DUDE")
                    engine.runAndWait()

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
