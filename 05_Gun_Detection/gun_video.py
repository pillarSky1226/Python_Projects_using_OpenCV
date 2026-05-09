import cv2
import imutils
import datetime
from pathlib import Path

# Build paths relative to this script so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "cascade.xml"
VIDEO_PATH = BASE_DIR / "src" / "face_detect.mp4"

gun_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if gun_cascade.empty():
    raise FileNotFoundError(f"Could not load cascade file: {CASCADE_PATH}")

vid = cv2.VideoCapture(str(VIDEO_PATH))
if not vid.isOpened():
    raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

gun_exist = False

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    guns = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))

    if len(guns) > 0:
        gun_exist = True

    for (x, y, w, h) in guns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(
        frame,
        datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 255),
        1,
    )

    cv2.imshow("Security Feed", frame)

    if gun_exist:
        print("Gun detected")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()