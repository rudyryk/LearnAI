# import the necessary packages
import sys
import time
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

VIDEO_PATH = '../data/youtube_IjLCq8wfpUc.mp4'
 
# Capture source
camera = cv2.VideoCapture(VIDEO_PATH)

# Exit if video not opened.
if not camera.isOpened():
    print("Could not open video input")
    sys.exit()

# Read first frame.
grabbed, frame = camera.read()
if not grabbed:
    print("Cannot read video input")
    sys.exit()
 
# Define an initial bounding box
# bbox = (80, 150, 7, 7)
bbox = (270, 225, 20, 15)

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
tracker = cv2.Tracker_create("MIL")

frame_counter = 0

while True:
    # grab the current frame
    # if frame_counter < 48:
    if frame_counter < 70:
        (grabbed, frame) = camera.read()
    else:
        (grabbed, _) = False, None

    if not grabbed:
        tracker.init(frame, bbox)
        break
        # time.sleep(0.1)
        # continue

    frame_counter += 1
    print(frame_counter)

    # resize frame
    frame = imutils.resize(frame, width=600)

    # capture box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (0, 0, 255))

    # Display result
    cv2.imshow("Tracking", frame)

    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

while True:
    # Read a new frame
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=600)
     
    # Update tracker
    grabbed, bbox = tracker.update(frame)

    # Draw bounding box
    if grabbed:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))

    # Display result
    cv2.imshow("Tracking", frame)

    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
