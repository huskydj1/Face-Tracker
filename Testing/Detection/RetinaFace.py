#https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/

import sys
sys.path.append("D:/Python/Face Tracker")

import cv2
import time
import torch
import numpy as np
from retinaface import RetinaFace
from PIL import Image

import drawframe
from VGGFace2_matching import Matching
import organizefiles


def track(inputFileFolder, inputFileName, device, fontScale = 0.3, thresh = 0.5):
    #Open Input
    cap = organizefiles.openInputVideo(inputFileFolder, inputFileName)
    
    #Open MP4 Output
    out = organizefiles.openOutputVideo(folder = "outputVideos", name = ("RETINA_" + inputFileName), 
        fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));

    #//////////////////////////////////

    #RetinaFace

    for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame_filename = "RetinaFace_FRAME.jpg"
        ret, frame = cap.read()
        cv2.imwrite(frame_filename, frame)

        if ret:
            faces = RetinaFace.detect_faces(frame_filename)

            if frame_num%100 == 0:
                print("Frame Num: " + str(frame_num))
            # print(len(faces))
            # print(type(faces))

            numFacesDetected = 0 if faces is None or not(type(faces) is dict) else len(faces)
            print(numFacesDetected)

            if numFacesDetected > 0:
                for key, identity in faces.items():
                    facial_area = identity["facial_area"]
                    landmarks = identity["landmarks"]
                    
                    #highlight facial area
                    cv2.rectangle(frame, (facial_area[2], facial_area[3])
                    , (facial_area[0], facial_area[1]), (255, 255, 255), 1)
                    
                    #highlight the landmarks
                    cv2.circle(frame, tuple(landmarks["left_eye"]), 1, (0, 0, 255), -1)
                    cv2.circle(frame, tuple(landmarks["right_eye"]), 1, (0, 0, 255), -1)
                    cv2.circle(frame, tuple(landmarks["nose"]), 1, (0, 0, 255), -1)
                    cv2.circle(frame, tuple(landmarks["mouth_left"]), 1, (0, 0, 255), -1)
                    cv2.circle(frame, tuple(landmarks["mouth_right"]), 1, (0, 0, 255), -1)

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


track("sourceVideos", "oneman_face-demographics-walking-and-pause", device = device)



'''
track("sourceVideos", "oneman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onewoman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onemanonewoman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onemantwowomen_face-demographics-walking-and-pause", device = device)

track("sourceVideos", "walkinghallway-pexels", device = device, fontScale = 1.3)
track("sourceVideos", "dogrunning", device = device, fontScale = 1.3)
'''