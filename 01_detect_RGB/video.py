import cv2
import numpy as np

# Set your video path here (example: r"D:\videos\sample.mp4")
VIDEO_PATH = r"src\football.mp4"
vid = cv2.VideoCapture(VIDEO_PATH)

if not vid.isOpened():
    print(f"Cannot open video source: {VIDEO_PATH}")
    raise SystemExit(1)

try:
    while True:

        ret, frame = vid.read()

        if not ret:
            print("No more frames or failed to read source.")
            break

        cv2.imshow("frame", frame)

        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]

        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)

        if b_mean > g_mean and b_mean > r_mean:
            print("Blue")

        elif g_mean > r_mean and g_mean > b_mean:
            print("Green")

        else:
            print("Red")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Error occurred:", e)

finally:
    vid.release()
    cv2.destroyAllWindows()