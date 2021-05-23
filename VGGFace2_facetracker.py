import cv2
import time
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

import drawframe
from VGGFace2_matching import Matching
import organizefiles


def track(inputFileFolder, inputFileName, device):
    #Open Input
    cap = organizefiles.openInputVideo(inputFileFolder, inputFileName)
    
    #Open MP4 Output
    out = organizefiles.openOutputVideo(folder = "outputVideos", name = ("OUT_" + inputFileName), 
        fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));

    #Open TXT Output
    debugFile = organizefiles.openOutputText(folder = "outputTexts", name = ("INFO_" + inputFileName))

    #//////////////////////////////////

    #MTCNN
    mtcnn = MTCNN(keep_all = True, device = device)

    #Matching Class (VGGFace2)
    matching = Matching()

    for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if ret:
            #Get Cropped Faces (MTCNN)
            face_array = mtcnn(frame)
            
            numFacesDetected = 0 if face_array is None else len(face_array)
            if numFacesDetected > 0:
                boxes, probs, landmarks = mtcnn.detect(frame, landmarks = True)
                id_list, scores = matching.update_batch(face_array, frame_num, thresh = 1.0) #TODO: FIX LATER

                drawframe.draw_boxes(frame, boxes)
                drawframe.draw_land(frame, landmarks)
                drawframe.draw_id(frame, id_list, boxes, fontScale = 1)
                
                debugFile.write(str(frame_num) + ": " + str(id_list) + " " + str(scores) + '\n')

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

    #Close TXT Output
    debugFile.close()

# CUDA DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print('Running on device: {}'.format(device))   

track("sourceVideos", "walkinghallway-pexels", device = device)

'''track("sourceVideos", "dogrunning", mtcnn)
track("sourceVideos", "oneman_face-demographics-walking-and-pause", mtcnn)
track("sourceVideos", "onemanonewoman_face-demographics-walking-and-pause", mtcnn)
track("sourceVideos", "onemantwowomen_face-demographics-walking-and-pause", mtcnn)
track("sourceVideos", "onewoman_face-demographics-walking-and-pause", mtcnn)'''