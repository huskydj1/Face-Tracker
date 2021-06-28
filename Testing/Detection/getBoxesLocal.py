#https://github.com/hukkelas/DSFD-Pytorch-Inference

# Working directory: "Face Tracker" (not needed for colab)
import sys
sys.path.append("D:/Python/Face Tracker")

# Necessary Packages
import numpy as np  
import cv2
import torch
import time

# Necessary Repos
from retinaface import RetinaFace # https://github.com/serengil/retinaface

# Necessary Files
import drawframe
import organizefiles


def track(conf_threshes):

    inputFileFolder = "sourceVideos"
    input_videos = {
        "fourhallway" :  "walkinghallway-pexels",
        "oneman" : "oneman_face-demographics-walking-and-pause",
        "onewoman" : "onewoman_face-demographics-walking-and-pause",
        "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",
        "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause",
        "pannning" : "dogrunning"
    }
    
    
    for conf_thresh in conf_threshes: # For each confidence threshold

        detector_name = "RetinaFace"

        for input_short, input_name in input_videos.items(): # For each input video
            print(input_name)
            
            # Input:
            # Open MP4 Input
            cap = organizefiles.openInputVideo(inputFileFolder, input_name)

            # Output:
            output_long = detector_name + "_" + input_short + "_" + "C" + str(conf_thresh) + "_ " +  "I" + str(iou_thresh) + "v"
            print(output_long)

            # Open TXT Output
            debugFile = organizefiles.openOutputText(folder = "outputTexts/" + detector_name, name = output_long)

            # Run
            #//////////////////////////////////

            debugFile.write("STARTED: confidence_threshold = {} \n".format(conf_thresh))

            #Start Timer
            start = time.time()

            for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                frame_filename = "RetinaFace_FRAME.jpg"
                ret, frame = cap.read()
                cv2.imwrite(frame_filename, frame)

                if ret:
                    if frame_num%100 == 0:
                        print("Frame Num: " + str(frame_num))
                    else:
                        continue
                    '''
                    elif frame_num==100:
                        break
                    else:
                        continue
                    '''

                    
                    faces = RetinaFace.detect_faces(frame_filename, threshold = conf_thresh)

                    numFacesDetected = 0 if faces is None or not(type(faces) is dict) else len(faces)
                    # print(numFacesDetected)
                    if frame_num%100 == 0:
                        print(numFacesDetected)
                        
                    # Store Frame Information
                    debugFile.write("FRAME#: " + str(frame_num) + " FACESDETECTED: " + str(numFacesDetected) + '\n')

                    if numFacesDetected > 0:
                        for key, identity in faces.items():
                            facial_area = identity["facial_area"]
                            
                            debugFile.write(str(facial_area[0]) + " " + str(facial_area[1]) + 
                            " " + str(facial_area[2]) + " " +  str(facial_area[3]) + " " + str(identity["score"]) + '\n')

                else:
                    break

            #End Timer
            end = time.time()

            debugFile.write("COMPLETED Runtime of the program is {} seconds \n".format(end - start))
            
            #//////////////////////////////////

            #Close Input
            cap.release()

            #Close TXT Output
            debugFile.close()


# CUDA DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print('Running on device: {}'.format(device))

track(conf_threshes = [0.3]) #doesn't allow nms_threshold modifications line 58 https://github.com/serengil/retinaface/blob/master/retinaface/RetinaFace.py