import cv2
import numpy as np
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO Model Files
weights_path = os.path.join(BASE_DIR, "object_detection_model", "weights", "yolov3-tiny.weights")
config_path = os.path.join(BASE_DIR, "object_detection_model", "config", "yolov3-tiny.cfg")
labels_path = os.path.join(BASE_DIR, "object_detection_model", "objectLabels", "coco.names")

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Load class labels
with open(labels_path, "r") as file:
    label_classes = [line.strip() for line in file.readlines()]

# Get output layers (handles both old & new OpenCV versions)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Random colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(label_classes), 3))

font = cv2.FONT_HERSHEY_SIMPLEX

def detectObject(frame):
    labels_this_frame = []
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            label = label_classes[class_ids[i]]
            conf = confidences[i]
            labels_this_frame.append((label, conf))

    return labels_this_frame
