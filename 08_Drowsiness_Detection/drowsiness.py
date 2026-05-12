import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

# INITIALIZING THE pyttsx3 SO THAT 
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU
# CAN EVEN CHOOSE 0 IN PLACE OF 1
video_path = "src/face_detect.mp4"
cap = cv2.VideoCapture(video_path)

# MAIN LOOP IT WILL RUN ALL THE UNLESS 
# AND UNTIL THE PROGRAM IS BEING KILLED 
# BY THE USER
while True:
    null, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(9)
    if key == 20:
        break
cap.release()
cv2.destroyAllWindows()

