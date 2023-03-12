import os
import sys
import cv2
import argparse
import base64
import mimetypes
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from requests import HTTPError
from googleapiclient.errors import HttpError
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.message import EmailMessage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase


# ----------------------------------------------------------
# Define global variables
# ----------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*'XVID')

# ----------------------------------------------------------
# Define functions 
# ----------------------------------------------------------

def send_message(image, creds):
    try:
        service = build('gmail', 'v1', credentials=creds)
        service = build('gmail', 'v1', credentials=creds)
        mime_message = EmailMessage()
        mime_message['To'] = 'jure.taslak@gmail.com'
        mime_message['From'] = 'kamera.hrasce@gmail.com'
        mime_message['Subject'] = 'Surveilance camera'
        mime_message.set_content('Detection')

        # guessing the MIME type
        type_subtype, _ = mimetypes.guess_type('image.jpg')
        maintype, subtype = type_subtype.split('/')

        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        byte_im = im_buf_arr.tobytes()

        mime_message.add_attachment(byte_im, maintype, subtype)

        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

        create_message = {
            'raw': encoded_message
        }
        send_message = (service.users().messages().send
                        (userId="me", body=create_message).execute())
        print(F'Sent email, message Id: {send_message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        send_message = None
    return send_message

# function to get the output layer names
# in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_detection(image):
    date = datetime.now().isoformat()[0:10]
    folder_name = f'saved-images/{date}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(folder_name + '/videos')
        cv2.imwrite(f'{folder_name}/detection_0.jpg', image)
    else:
        contents = os.listdir(folder_name)
        detections = len(contents) - 1
        cv2.imwrite(f'{folder_name}/detection_{detections}.jpg', image)

def create_videorecorder(recording_start):
    date = datetime.now().isoformat()[0:10]
    folder_name = f'saved-images/{date}'
    recorder = cv2.VideoWriter(folder_name + '/videos/ ' + recording_start.isoformat() + '.avi', fourcc, 1.0, (1920, 1080))
    return recorder

# -----------------------------------------------------------
#   Start of program
# -----------------------------------------------------------

def main(args):
    # Set up gmail API
    if args.email:
        SCOPES = [
                "https://www.googleapis.com/auth/gmail.send"
            ]
        flow = InstalledAppFlow.from_client_secrets_file('client_secret_765936832111-d34cv8g721c4spr879dh3jb13kmglo0c.apps.googleusercontent.com.json', SCOPES)
        creds = flow.run_local_server(port=0)

    # Set up camera
    camera_url = os.getenv('CAMERA_URL')
    cap = cv2.VideoCapture(camera_url)

    # Set up neural network parameters
    classes = None
    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # Set up program variables
    SKIP = 25
    skipped = 0
    detection_times = []
    recording = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        if skipped < SKIP:
            skipped += 1
            continue
        else:
            skipped = 0

        if recording:
            now = datetime.now()
            total_recording_time = (now - recording_start).seconds
            recorder.write(frame)
            if total_recording_time >= 300:
                # We recorded for 5 minutes, now stop recording and go back to detecting
                recording = False
                recorder.release()
                detection_times = []
                continue

        image = frame[300:600, 1100:1600]

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        # set input blob for the network
        net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        detected_person_or_car = False

        # Only detect things inside this area
        pts = np.array([[260, 160], [444, 200], [450, 300], [0, 300], [0, 78], [260, 85]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Uncomment this if you want to draw the are
        # image = cv2.polylines(image, [pts], True, (255, 0, 0), 2)

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (class_id in [0,2]) and confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    point_in_area = cv2.pointPolygonTest(pts, (int(x), int(y+h)), False)
                    if point_in_area < 0:
                        continue
                    detected_person_or_car = True
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    if class_id == 0:
                        detection_time = datetime.now()
                        detection_times.append(detection_time)

                        while (detection_time - detection_times[0]).seconds > 300:
                            detection_times.pop(0)

        if detected_person_or_car:
            # apply non-max suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # go through the detections remaining
            # after nms and draw bounding box
            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_bounding_box(image, classes, COLORS, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
                save_detection(image)

        if not recording and len(detection_times) > 3:
            if args.email:
                send_message(image, creds)
            recording = True
            recording_start = datetime.now()
            recorder = create_videorecorder(recording_start)
            detection_times = []

        # display the output image
        if args.video:
            cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env.

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=False, action='store_true', help = 'Show the video stream')
    ap.add_argument('-e', '--email', required=False, action='store_true', help = 'Send email of detections')
    args = ap.parse_args()
    main(args)

