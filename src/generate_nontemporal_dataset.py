from cv2 import cv2
import numpy as np
import os
from util import get_frame_label
from tqdm import tqdm




cwd = os.getcwd(), os.chdir("..")
video_dir = os.path.join(os.getcwd(), 'input', 'videos')
annotation_dir = os.path.join(os.getcwd(), 'input', 'annotations')



if(not os.path.isdir('input/NonTemporalDataset')):
    os.mkdir('input/NonTemporalDataset')
    
dataset_path = os.path.join('input', 'NonTemporalDataset')   

train_path = os.path.join(dataset_path,'train')
test_path = os.path.join(dataset_path,'test')

if(not os.path.isdir(train_path)):
    os.mkdir(train_path)
    os.mkdir(os.path.join(train_path,'harassment'))
    os.mkdir(os.path.join(train_path,'normal'))

if(not os.path.isdir(test_path)):
    os.mkdir(test_path)
    os.mkdir(os.path.join(test_path, 'harassment'))
    os.mkdir(os.path.join(test_path, 'normal'))

video_list = os.listdir(video_dir)[:5]
number_of_videos = len(video_list)
test_size = 0.20

for index, video in tqdm(enumerate(video_list[:])):
    video_path = os.path.join(video_dir, video)
    annotation_path = os.path.join(annotation_dir, video.replace('.mp4', '.txt'))
    # print(video_path)
    # print(annotation_path)
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video  file")
        break
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_label = get_frame_label(frame_count, annotation_path)
    if(index <int((1-test_size)*number_of_videos)):
        save_path = train_path
    else:
        save_path = test_path
    frame_no = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame_name = '{}_{}.jpg'.format(video.split('.')[0], str(frame_no).zfill(5))
            if(frame_label[frame_no]):
                cv2.imwrite(os.path.join(save_path, 'harassment', frame_name), frame)
            else:
                cv2.imwrite(os.path.join(save_path, 'normal', frame_name), frame)
        else:
            break
        frame_no += 1
    cap.release()
    cv2.destroyAllWindows()








