import cv2
import numpy as np
import os
# from google.colab.patches import cv2_imshow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_proto = os.path.join(BASE_DIR, "model", "opencv_face_detector.pbtxt")
face_model = os.path.join(BASE_DIR, "model", "opencv_face_detector_uint8.pb")

age_proto = os.path.join(BASE_DIR, "model", "age_deploy.prototxt")
age_model = os.path.join(BASE_DIR, "model", "age_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]

face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

def detect_faces(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame, face_boxes

def predict_age(face, net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    age_preds = net.forward()
    age = age_list[age_preds[0].argmax()]
    return age

def imread_file(path):
    """Load BGR image; works with Unicode paths on Windows (unlike cv2.imread)."""
    if not os.path.isfile(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def process_image(image_path):
    frame = imread_file(image_path)

    if frame is None:
        src_dir = os.path.join(BASE_DIR, "src")
        print(f"Error: Could not open image: {image_path}")
        if not os.path.isdir(src_dir):
            print(f"Missing folder: {src_dir}")
        print(f"Create that folder (if needed) and put your photo there as age.jpg, age.jpeg, or age.png.")
        return

    frame, face_boxes = detect_faces(face_net, frame)
    
    for (x1, y1, x2, y2) in face_boxes:
        face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1), 
                     max(0, x1-20):min(x2+20, frame.shape[1]-1)]
        age = predict_age(face, age_net)
        cv2.putText(frame, f"Age: {age}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
 
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "src/age.jpg"
process_image(image_path)