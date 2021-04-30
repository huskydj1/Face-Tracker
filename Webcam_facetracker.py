import cv2
import time
import torch
import numpy as np
import os.path
from facenet_pytorch import MTCNN

from greedymatching import face_movements
from greedymatching import face

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

    def drawRect(self, frame, boxes, probs, landmarks, facenums, show_rect = True, show_prob = False, show_land = True):
        # print(type(boxes), boxes.shape)
        # print(type(probs), probs.shape)
        # print(type(landmarks), landmarks.shape)

        for box, prob, ld, numi in zip(boxes, np.asarray(probs), landmarks, facenums):
            # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
            
            if show_rect:
                cv2.rectangle(img = frame, pt1 = (box[0], box[1]), pt2 = (box[2], box[3]), 
                    color = (0, 0, 255), thickness = 2)
            if show_prob:
                cv2.putText(img = frame, text = str(prob), org = (box[2], box[3]), 
                    fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 2, lineType = cv2.LINE_AA)
            if numi != -1:
                cv2.putText(img = frame, text = str(numi), org = (box[2], box[3]), 
                    fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 1, lineType = cv2.LINE_AA)
            
            if show_land:
                for land_x, land_y in ld:
                    cv2.circle(img = frame, center=(land_x, land_y), radius = 2, color = (255, 0, 0), thickness=-1)

        return frame

    def run_webcam(self):
        cap = cv2.VideoCapture(1)

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
                    self.drawRect(frame, boxes, probs, landmarks)

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
        cap = cv2.VideoCapture('crowd_resized.mp4')
        cap.set(cv2.CAP_PROP_FPS, 24)
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(find_available_file(name = 'crowdwalking_output'), fourcc, 24, (640, 480))

        while True:
            ret, frame = cap.read()
            if ret:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks = True)

                if not(boxes is None):
                    self.drawRect(frame, boxes, probs, landmarks)

                out.write(frame)
                # cv2.imshow('Crowd Faces', frame)
            else:
                break
            

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def crowdTracking_greedy(self):
        cap = cv2.VideoCapture('crowd_resized.mp4')
        cap.set(cv2.CAP_PROP_FPS, 24)
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(find_available_file(name='crowdwalking_output'), 
            fourcc, 24, (640, 480))

        frame_num = 0

        trackList = face_movements()
        
        while True:
            ret, frame = cap.read()
            if ret:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks = True)

                if not(boxes is None):
                    face_nums = []
                    for landi in landmarks:
                        new_face = face(landmarks=landi, frame_num=frame_num)
                        face_nums.append(trackList.addFace(new_face))
                    
                    self.drawRect(frame, boxes, probs, landmarks, face_nums)

                out.write(frame)
                # cv2.imshow('Crowd Faces', frame)
                
                frame_num += 1
            else:
                break
            

        cap.release()
        out.release()
        cv2.destroyAllWindows()


torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all = True, device = device)

fct = FaceTracker(mtcnn)
fct.crowdTracking_greedy()