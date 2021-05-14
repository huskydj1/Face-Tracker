import cv2
import time
import torch
import numpy as np
import os.path
from facenet_pytorch import MTCNN
from PIL import Image

import drawframe
from VGGFace2_matching import Matching

def find_available_file(name):
    file_name = name + '{}.mp4'
    file_num = 0

    while os.path.isfile(file_name.format(file_num)):
        file_num += 1

    print(file_name.format(file_num))
    return file_name.format(file_num)

def extract_faces(boxes, pixels, required_size = (224, 224)):
    print(boxes)
    face_array = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        print(x1, y1, x2, y2)
        face = np.array(pixels)[int(y1):int(y2), int(x1):int(x2)]
        image = Image.fromarray(face)
        image = image.resize(required_size)

        #Pre-whiten
        face_array.append(np.asarray(image))
    
    return face_array

def track(inputFileFolder, inputFileName, mtcnn):
    #Input File Path
    inputFile = inputFileFolder + "/" + inputFileName + ".mp4"

    #Open File
    cap = cv2.VideoCapture(inputFile)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    #Output, processed video
    fps = 8 # cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(find_available_file(name=("outputVideos/OUTPUT_" + inputFileName)), 
        fourcc1, fps, (frame_width, frame_height))

    #Matching Class
    matching = Matching()

    #Read frames
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if ret:
            # print(str(frame_num) + ": " + str(matching))
            face_array, prob = mtcnn(frame, return_prob = True)
            numFacesDetected = 0 if face_array is None else len(face_array)

            if not(face_array is None):
                boxes, probs, landmarks = mtcnn.detect(frame, landmarks = True)
                id_list = matching.update_batch(face_array, frame_num)
                drawframe.notate(frame, boxes, landmarks = landmarks, probs = probs, faceids=id_list)

            # Frame by Frame Info (Coupled with outputVideos\onepersonwalking_output12.mp4)
            # Frame #, Face Detected (bool), trackList string representation (face class identified by frame #)
            infoTemp = "FRAME #: " + str(frame_num) + " Faces Detected: " + str(numFacesDetected)

            cv2.putText(frame, infoTemp, (30, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
            out.write(frame)
            
            frame_num += 1
        else:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# CUDA DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all = True, device = device)
'''img = cv2.imread("sourceVideos/muskbezos.jpg")

face_array, prob = mtcnn(img, return_prob = True)

for i, face in enumerate(face_array):
    cv2.imshow("image" + str(i), face.permute(1, 2, 0).numpy()); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
'''
track("sourceVideos", "onemanonewoman_face-demographics-walking-and-pause", mtcnn)