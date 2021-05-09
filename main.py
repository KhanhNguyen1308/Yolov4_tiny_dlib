from ctypes import *
import random
import os
import cv2
import dlib
import time
import darknet
import argparse
from ham import midpoint, mid, e_dist, nose_point, eye_ratio, play_sound, convert2dlib
from imutils import face_utils
from threading import Thread, enumerate
from queue import Queue

config_file = "cfg/yolov4-tiny-224.cfg",
weights_file = "backup/yolov4-tiny-224.weights",
data_file = "yolo.data"
thresh = .5
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nose_start, nose_end) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
#############################################################################
class VideoStream:
    def __init__(self, resolution=(480, 320), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


#############################################################################


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted
network, class_names, class_colors = darknet.load_network(
            "cfg/yolov4-tiny-224.cfg",
            "yolo.data",
            "backup/yolov4-tiny-224.weights",
            batch_size=1
        )
darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)
videostream = VideoStream(resolution=(480, 320), framerate=30).start()
while True:
    start_time = time.time()
    frame = videostream.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
    darknet_image = img_for_detect
    prev_time = time.time()
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    bbox_adjusted = []
    if frame is not None:
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            #############################################################################
            startX = convert2dlib(bbox_adjusted)[0]
            endX = convert2dlib(bbox_adjusted)[1]
            rect = dlib.rectangle(int(startX[0]), int(startX[1]), int(endX[0]), int(endX[1]))
            landmark = landmark_detect(gray, rect)
            landmark = face_utils.shape_to_np(landmark)
            #############################################################################
            leftEye = landmark[left_eye_start:left_eye_end]
            rightEye = landmark[right_eye_start:right_eye_end]
            nose = landmark[nose_start:nose_end]
            nosepoint1 = nose_point(nose)
            #############################################################################
            left_eye_ratio = eye_ratio(leftEye)
            right_eye_ratio = eye_ratio(rightEye)
            left_eye_midpoint = midpoint(leftEye)
            right_eye_midpoint = midpoint(rightEye)
            #####################################################################################
            left_eye_midpoint1 = (int(left_eye_midpoint[0]), int(left_eye_midpoint[1]))
            right_eye_midpoint1 = (int(right_eye_midpoint[0]), int(right_eye_midpoint[1]))
            midpoint_eye = mid(left_eye_midpoint, right_eye_midpoint)
            startY = (int(midpoint_eye[0]), 0)
            endY = (int(midpoint_eye[0]), 400)
            left_y_doixung = (int(midpoint_eye[0]), int(left_eye_midpoint[1]))
            right_y_doixung = (int(midpoint_eye[0]), int(right_eye_midpoint[1]))
            left_x_doixung = (int(left_eye_midpoint[0]), int(nosepoint1[1]))
            right_x_doixung = (int(right_eye_midpoint[0]), int(nosepoint1[1]))
            left_y = [midpoint_eye[0], left_eye_midpoint[1]]
            right_y = [midpoint_eye[0], right_eye_midpoint[1]]
            left_x = [left_eye_midpoint[0], nosepoint1[1]]
            right_x = [right_eye_midpoint[0], nosepoint1[1]]
            midpoint_eye = (int(midpoint_eye[0]), int(midpoint_eye[1]))
            phai_ratio = round(((e_dist(right_eye_midpoint, right_x)) / (e_dist(right_eye_midpoint, right_y))), 4)
            trai_ratio = round(((e_dist(left_eye_midpoint, left_x)) / (e_dist(left_eye_midpoint, left_y))), 4)
            eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
            print(eye_avg_ratio)
            mid_eye = mid(left_eye_midpoint, right_eye_midpoint)
            #####################################################################################
            cv2.line(frame, startY, endY, (0, 255, 0), 2)
            cv2.line(frame, (0, nosepoint1[1]), (640, nosepoint1[1]), (0, 255, 0), 2)
            cv2.line(frame, left_eye_midpoint1, left_y_doixung, (255, 215, 0), 2)
            cv2.line(frame, right_eye_midpoint1, right_y_doixung, (255, 215, 0), 2)
            cv2.line(frame, left_eye_midpoint1, left_x_doixung, (255, 215, 0), 2)
            cv2.line(frame, right_eye_midpoint1, right_x_doixung, (255, 215, 0), 2)
            cv2.circle(frame, midpoint_eye, 3, (255, 0, 0), -1)
            #############################################################################
    fps = int(1/(time.time() - prev_time))
    print("FPS: {}".format(fps))
    darknet.free_image(darknet_image)
    detections_adjusted = []
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
    image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
    cv2.imshow('Inference', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if cv2.waitKey(fps) == 27:
        videostream.stop()
        break

videostream.stop()

cv2.destroyAllWindows()
    