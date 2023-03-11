import os
import sys
import cv2
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

camera_url = os.getenv('CAMERA_URL')
cap = cv2.VideoCapture(camera_url)
detections = 0
timestamp = datetime.now().isoformat()
print("Timestamp: ", timestamp)
newpath = f'saved-images/{timestamp}'
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = frame[300:600, 1100:1600]
    # initialize the HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # detect humans in input image
    (humans, _) = hog.detectMultiScale(frame, winStride=(10, 10),
                                padding=(32, 32), scale=1.1)
    
    # getting no. of human detected
    
    # loop over all detected humans
    for (x, y, w, h) in humans:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

    if len(humans) > 0:
        print("Humans detected")
        cv2.imwrite(f"saved-images-{timestamp}/detection_{detections}.jpg", frame)
        detections += 1

    # display the output image
    cv2.imshow("Image", frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()

