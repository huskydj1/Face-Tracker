import sys
sys.path.append("D:/Python/Face Tracker")

import cv2
import time
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

import drawframe
from VGGFace2_matching import Matching
import organizefiles


def track(inputFileFolder, inputFileName, device, fontScale = 0.3, thresh = 0.5):
    #Open Input
    cap = organizefiles.openInputVideo(inputFileFolder, inputFileName)
    
    #Open MP4 Output
    out = organizefiles.openOutputVideo(folder = "outputVideos", name = ("MTCNN_" + inputFileName), 
        fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));

    #//////////////////////////////////

    #MTCNNex
    mtcnn = MTCNN(keep_all = True, device = device)

    for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if ret:
            boxes, probs, landmarks = mtcnn.detect(frame, landmarks = True)

            numFacesDetected = 0 if boxes is None else len(boxes)
            if numFacesDetected!=0: 
                drawframe.notate(frame, boxes, landmarks = landmarks, probs = probs)

            infoTemp = "FRAME #: " + str(frame_num) + " Faces Detected: " + str(numFacesDetected)
            cv2.putText(frame, infoTemp, (30, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
            out.write(frame)
        else:
            break
    
    #//////////////////////////////////

    #Close Input
    cap.release()

    #Close MP4 Output 
    out.release()

# CUDA DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print('Running on device: {}'.format(device))


# RUN
track("sourceVideos", "walkinghallway-pexels", device = device, fontScale = 1.3)

'''
track("sourceVideos", "oneman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onewoman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onemanonewoman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onemantwowomen_face-demographics-walking-and-pause", device = device)

track("sourceVideos", "walkinghallway-pexels", device = device, fontScale = 1.3)
track("sourceVideos", "dogrunning", device = device, fontScale = 1.3)
'''