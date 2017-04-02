import argparse
import numpy as np
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
    cap = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    cap = cv2.VideoCapture(args["video"])

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    mask = fgbg.apply(frame)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("No background", mask)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
