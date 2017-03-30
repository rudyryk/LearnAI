# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
args = vars(ap.parse_args())
 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
tracker = cv2.Tracker_create("KCF")

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
bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

# define the lower and upper boundaries of the "orange"
# ball in the HSV color space, then initialize the
# list of tracked points
colorLower = (14, 50, 50)
colorUpper = (22, 255, 255)

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # construct a mask for the color "orange", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        bbox = cv2.boundingRect(c)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))

    # Display result
    cv2.imshow("Tracking", frame)

    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Initialize tracker with first frame and bounding box
grabbed = tracker.init(frame, bbox)

while True:
    # Read a new frame
    grabbed, frame = camera.read()
    if not grabbed:
        break
     
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
