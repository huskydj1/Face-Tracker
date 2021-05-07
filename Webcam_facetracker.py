import cv2
import time
import torch
import numpy as np
import os.path
from facenet_pytorch import MTCNN

from greedymatching import face_movements
from greedymatching import face

import drawframe

def find_available_file(name):
        file_name = name + '{}.mp4'
        file_num = 0

        while os.path.isfile(file_name.format(file_num)):
            file_num += 1

        print(file_name.format(file_num))
        return file_name.format(file_num)

class FaceTracker(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn
        self.time_elapsed_sum = 0.0
        self.time_elapsed_cnt = 0.0
    

    def run_webcam(self):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        frame_rate = 30 #LOGITECH C20 MAX FPS=30
        prev = 0

        print('Desired Time Between Frames: {}'
            .format(1.0/frame_rate))
        print('Desired FPS: {}'
            .format(frame_rate))

        while True:
            time_elapsed = time.time() - prev
            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if time_elapsed > 1.0/frame_rate:
                if prev != 0:
                    self.time_elapsed_sum += time_elapsed
                    self.time_elapsed_cnt += 1
                    # print(time_elapsed, self.time_elapsed_sum/self.time_elapsed_cnt)

                prev = time.time()

                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks = True)
                
                if not(boxes is None):
                    drawframe.notate(frame, boxes, landmarks)

                cv2.imshow('Face Detection', frame)

        avg = fct.time_elapsed_sum/fct.time_elapsed_cnt
        print('Average Time Between Frames: {}'
            .format(avg))
        print('Average FPS: {}'
            .format(1.0/avg))
        
        cap.release()
        cv2.destroyAllWindows()

    def test_Webcam(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

    def check_res(self):
        cap = cv2.VideoCapture(1)

        ret, frame = cap.read()
        height, width, channels = frame.shape
        print(height, width, channels)


    def run_video(self):
        cap = cv2.VideoCapture('sourceVideos/crowd_resized.mp4')
        cap.set(cv2.CAP_PROP_FPS, 24)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(find_available_file(name = 'outputVideos/crowdwalking_output'), fourcc, 24, (640, 480))

        while True:
            ret, frame = cap.read()
            if ret:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks = True)

                if not(boxes is None):
                    drawframe.notate(frame, boxes, landmarks = landmarks, probs = probs)

                out.write(frame)
                # cv2.imshow('Crowd Faces', frame)
            else:
                break
            

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def crowdTracking_greedy(self, inputFileFolder, inputFileName):
        inputFile = inputFileFolder + "/" + inputFileName + ".mp4"
        cap = cv2.VideoCapture(inputFile)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps = 8 # cap.get(cv2.CAP_PROP_FPS)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(find_available_file(name=("outputVideos/OUTPUT_" + inputFileName)), 
            fourcc1, fps, (frame_width, frame_height))

        frame_num = 0

        trackList = face_movements()
        
        debugFile = open("FACEINFO_{}.txt".format(inputFileName), "w")

        while True:
            ret, frame = cap.read()
            if ret:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks = True)
                numFacesDetected = 0 if boxes is None else len(boxes)

                # Frame by Frame Info (Coupled with outputVideos\onepersonwalking_output12.mp4)
                # Frame #, Face Detected (bool), trackList string representation (face class identified by frame #)
                infoTemp = "FRAME #: " + str(frame_num) + " Faces Detected: " + str(numFacesDetected)
                debugFile.write(infoTemp + " TRACKLIST: "+ str(trackList) + '\n')

                if not(boxes is None):
                    face_nums = [0] * numFacesDetected
                    for i, landi in enumerate(landmarks):
                        new_face = face(landmarks=landi, frame_num=frame_num)
                        face_nums[i] = trackList.addFace(new_face)
                    
                    drawframe.notate(frame, boxes, landmarks = landmarks, probs = probs, faceids=face_nums)
                
                cv2.putText(frame, infoTemp, (30, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
                out.write(frame)
                
                frame_num += 1
            else:
                break
            
        debugFile.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()


torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all = True, device = device)

fct = FaceTracker(mtcnn)
fct.crowdTracking_greedy("sourceVideos", "oneman_face-demographics-walking-and-pause")
fct.crowdTracking_greedy("sourceVideos", "onemanonewoman_face-demographics-walking-and-pause")
fct.crowdTracking_greedy("sourceVideos", "onemantwowomen_face-demographics-walking-and-pause")
fct.crowdTracking_greedy("sourceVideos", "onewoman_face-demographics-walking-and-pause")