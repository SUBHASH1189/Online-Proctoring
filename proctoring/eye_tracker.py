import dlib
import cv2
import numpy as np
import os

# Dynamically resolve the full path to the shape predictor model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
shapePredictorModel = os.path.join(BASE_DIR, "shape_predictor_model", "shape_predictor_68_face_landmarks.dat")
shapePredictor = dlib.shape_predictor(shapePredictorModel)


def createMask(frame):
    height, width, channels = frame.shape
    mask = np.zeros((height, width), np.uint8)
    return mask

def extractEye(mask, region, frame):
    cv2.polylines(mask, region, True, 255, 2)
    cv2.fillPoly(mask, region, 255)
    eyes = cv2.bitwise_and(frame, frame, mask=mask)
    return eyes

def eyeSegmentationAndReturnWhite(img, side):
    height, width = img.shape
    if side == 'left':
        img = img[0:height, 0:int(width/2)]
        return cv2.countNonZero(img)
    else:
        img = img[0:height, int(width/2):width]
        return cv2.countNonZero(img)

def gazeDetection(faces, frame):
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2
    TrialRation = 1.2
    result = ""

    leftEye = [36, 37, 38, 39, 40, 41]
    rightEye = [42, 43, 44, 45, 46, 47]

    for face in faces:
        facialLandmarks = shapePredictor(frame, face)

        leftEyeRegion = np.array([(facialLandmarks.part(i).x, facialLandmarks.part(i).y) for i in leftEye], np.int32)
        rightEyeRegion = np.array([(facialLandmarks.part(i).x, facialLandmarks.part(i).y) for i in rightEye], np.int32)

        mask = createMask(frame)
        eyes = extractEye(mask, [leftEyeRegion, rightEyeRegion], frame)

        lmin_x, lmax_x = np.min(leftEyeRegion[:, 0]), np.max(leftEyeRegion[:, 0])
        lmin_y, lmax_y = np.min(leftEyeRegion[:, 1]), np.max(leftEyeRegion[:, 1])
        rmin_x, rmax_x = np.min(rightEyeRegion[:, 0]), np.max(rightEyeRegion[:, 0])
        rmin_y, rmax_y = np.min(rightEyeRegion[:, 1]), np.max(rightEyeRegion[:, 1])

        left = eyes[lmin_y:lmax_y, lmin_x:lmax_x]
        right = eyes[rmin_y:rmax_y, rmin_x:rmax_x]

        leftGrayEye = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightGrayEye = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        leftTh = cv2.adaptiveThreshold(leftGrayEye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        rightTh = cv2.adaptiveThreshold(rightGrayEye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        leftSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, 'right')
        rightSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, 'left')

        leftSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, 'right')
        rightSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, 'left')

        if rightSideOfRightEye >= TrialRation * leftSideOfRightEye:
            result += 'left'
        elif leftSideOfLeftEye >= TrialRation * rightSideOfLeftEye:
            result += 'right'
        else:
            result += 'center'

        cv2.putText(frame, result, (50, 110), font, 1, (255, 0, 255), thickness)

    return {
        "direction": result,
        "left_offset": leftSideOfLeftEye,
        "right_offset": rightSideOfRightEye,
    }
