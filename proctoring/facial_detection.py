import dlib
import cv2
from imutils import face_utils
import os
# Get the absolute path to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "shape_predictor_model", "shape_predictor_68_face_landmarks.dat")

# Load the predictor
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at path: {model_path}")

shapePredictor = dlib.shape_predictor(model_path)

# Initialize face detector only once
faceDetector = dlib.get_frontal_face_detector()

def detectFace(frame):
    """
    Input: A video frame from the front camera
    Output: Tuple (faceCount, faces) - number of faces and detected face rectangles
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Increase upsampling for better detection of small/far faces
    faces = faceDetector(gray, 1)
    faceCount = len(faces)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Draw fancy corners
        cv2.line(frame, (x, y), (x + 20, y), (0, 255, 255), 2)
        cv2.line(frame, (x, y), (x, y + 20), (0, 255, 255), 2)
        cv2.line(frame, (x + w, y), (x + w - 20, y), (0, 255, 255), 2)
        cv2.line(frame, (x + w, y), (x + w, y + 20), (0, 255, 255), 2)
        cv2.line(frame, (x, y + h), (x + 20, y + h), (0, 255, 255), 2)
        cv2.line(frame, (x, y + h), (x, y + h - 20), (0, 255, 255), 2)
        cv2.line(frame, (x + w, y + h), (x + w - 20, y + h), (0, 255, 255), 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - 20), (0, 255, 255), 2)

        # Facial landmarks
        landmarks = shapePredictor(gray, face)
        landmarks_np = face_utils.shape_to_np(landmarks)
        for (a, b) in landmarks_np:
            cv2.circle(frame, (a, b), 2, (255, 255, 0), -1)

    return (faceCount, faces)
