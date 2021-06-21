from cv2 import cv2
import numpy as np
import os
# from util import get_frame_label
import tensorflow as tf
from collections import deque

cwd = os.getcwd(), os.chdir("..")
video_dir = os.path.join(os.getcwd(), 'input', 'data', 'normal')

index = 80
video_path = os.path.join(video_dir, os.listdir(video_dir)[index])

cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video  file")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
labels = [0]*(frame_count+1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = tf.keras.models.load_model(
    os.path.join('saved_model', 'firstModel.tf'))

frame_no = 0
slide_window_length = 5
frame_stack = deque()
while(frame_no < frame_count):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    copy_frame = frame.copy()
    copy_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copy_frame = cv2.resize(copy_frame, (112, 112))
    normalized_frame = copy_frame/255.0
    if(frame_no < slide_window_length):
        frame_stack.append(normalized_frame)
        frame_no += 1
        continue
    else:
        input = np.array(frame_stack)
        input = np.expand_dims(input, axis=0)
        prediction = np.argmax(model.predict(input))
        if frame_no == slide_window_length:
            for i in range(frame_no):
                labels[i] = prediction
        labels[frame_no] = prediction
        frame_stack.popleft()
        frame_stack.append(normalized_frame)
    frame_no += 1


cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video  file")
frame_no = 0

while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    if(labels[frame_no] == 0):
        label = 'HARASSMENT'
        color = (0, 0, 255)
    else:
        label = 'NORMAL'
        color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                label,
                (60, 80),
                font, 2,
                color,
                3,
                cv2.LINE_4)
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_no += 1

cap.release()
cv2.destroyAllWindows()
