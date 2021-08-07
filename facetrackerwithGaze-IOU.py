# Working directory: "Face Tracker" (not needed for colab)
import sys
sys.path.append("D:/Python/Face Tracker")

# Necessary Packages
import numpy as np  
import time
import cv2
import random
import colorsys
import re

# Necessary Repos

# Necessary Files
import drawframe
import organizefiles

# copied directly from gaze360 colab notebook
def compute_iou(bb1,bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    bb2_area = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    eps = 1e-8

    if iou <= 0.0  or iou > 1.0 + eps: return 0.0

    return iou

def find_id(bbox,id_dict):
    id_final = None
    max_iou = 0.5
    for k in id_dict.keys():
        if(compute_iou(bbox,id_dict[k][0])>max_iou): 
            id_final = k
            max_iou = compute_iou(bbox,id_dict[k][0])
    return id_final



class FaceTracker(object):
    def __init__(self):
        self.dummy_id = 1e3

    
    # Helper Function for Reading Boxes from File (Outputted by RetinaFace, run by Colab)
    def readBoxes(self, boxFile, manual_conf, num_faces):

        box_info = np.empty(shape = (num_faces, 5))
        landmarks = np.empty(shape = (num_faces, 5, 2))
        for i in range(num_faces):
            box_info[i] = np.asarray(boxFile.readline().split())
            for j in range(5):
                landmarks[i][j] = np.asarray(boxFile.readline().split())

        boxes = box_info[:, 0:4]
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        probs = box_info[:, 4]

        boxes = (boxes.round()).astype(np.int)
        landmarks = (landmarks.round().astype(np.int))

        boxes = boxes[probs>=manual_conf]
        landmarks = landmarks[probs>=manual_conf]
        probs = probs[probs>=manual_conf]
        
        return boxes, probs, landmarks, len(boxes)
    
    def track(self, inputFileFolder, input_short, input_name, file_input_path, detector_name, manual_conf = 0.5):
        # Input:
        # Open MP4 Input
        cap = organizefiles.openInputVideo(inputFileFolder, input_name)
        fps_original =  cap.get(cv2.CAP_PROP_FPS)
        
        # Output: 
        
        # Matching Structures (from gaze360 notebook https://colab.research.google.com/drive/1MSIgZwREdFhrTbtEP-sixdkRhBn2S3FP#scrollTo=WP8T_pPbXchh)
        id_num = 0
        identity_last = dict()

        # Gaze Estimation Infastructure

        for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()

             # Open Txt Input    
            boxFile = open(file_input_path + "/{}.txt".format(str(frame_num)), "r")

            # Get Video Info
            assert(frame_num==int(boxFile.readline().strip()))
            num_faces = int(boxFile.readline().strip())

            #if not(0<=frame_num and frame_num<=10):
            #    continue

            if ret:
                # Read Bounding Box Information
                boxes, probs, landmarks, num_faces = self.readBoxes(boxFile, manual_conf, num_faces)

                print("FRAME: {num1} FACES DETECTED: {num2}".format(
                    num1 = frame_num,
                    num2 = num_faces,
                ))
                
                identity_next = dict()
                if num_faces > 0:

                    translatedLandmarks = np.zeros(shape = landmarks.shape)
                    
                    for i in range(len(translatedLandmarks)):
                        for j in range(len(translatedLandmarks[i])):
                            for k in range(len(translatedLandmarks[i, j])):
                                translatedLandmarks[i, j, k] = landmarks[i, j, k] - boxes[i, k]
                                
                    # Tracking     
                    tracking_id_frame = open("D:/Python/Face Tracker/exportTrackingResults/{}/{}.txt".format(input_short, frame_num), "w")
                    tracking_id_frame.write(str(num_faces) + '\n')
                    for j in range(num_faces):
                        bbox_head = boxes[j]
                        assert(not(bbox_head is None))
                        id_val = find_id(bbox_head, identity_last)
                        if id_val is None: 
                            id_num+=1
                            id_val = id_num

                        # Choices for Eye Location
                        # eyes = [(bbox_head[0]+bbox_head[2])/2.0, (0.65*bbox_head[1]+0.35*bbox_head[3])]
                        eyes = [(landmarks[j, 0, 0] + landmarks[j, 1, 0])/2, (landmarks[j, 0, 1] + landmarks[j, 1, 1])/2]
                    
                        identity_next[id_val] = (bbox_head, eyes)

                        #Print to file
                        tracking_id_frame.write(str(id_val) + '\n')
                        tracking_id_frame.write(re.sub("[^0-9^.^ ^-]", "", str(bbox_head)) + '\n')
                        tracking_id_frame.write(str(eyes[0]) + " "  + str(eyes[1]) + '\n')

                    tracking_id_frame.close()
                    

                else:
                    # EXPORT "NO FACES" IN FILE
                    tracking_id_frame = open("D:/Python/Face Tracker/exportTrackingResults/{}/{}.txt".format(input_short, frame_num), "w")
                    tracking_id_frame.write("0\n")
                    tracking_id_frame.close()
                identity_last = identity_next

            else:
                break

        # Close MP4 Input
        cap.release()

        # Close Txt Input
        boxFile.close()

inputFileFolder = "sourceVideos"
input_videos = {   
    "bigcrowd" : "skywalkmahanakhon-videvo",
    "rainpedestrians" : "crowdedstreetundertherain-pexels",
    "fourhallway" :  "walkinghallway-pexels",
}
'''
Up Next:
    "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause", 
    "voccamp" : "voccamp",
'''

input_videos_file = {
    "voccamp" : "D:/Python/Face Tracker/outputTexts/8--voccamp",
    "onemantwowoman" : "D:/Python/Face Tracker/outputTexts/0--onemantwowoman/",
    "bigcrowd" : "D:/Python/Face Tracker/outputTexts/6--bigcrowd",
    "rainpedestrians" : "D:/Python/Face Tracker/outputTexts/7--rainpedestrians",
    "fourhallway" :  "D:/Python/Face Tracker/outputTexts/5--fourhallway/",
}

'''
Inactive Videos:
    "panning" : "dogrunning",
    "oneman" : "oneman_face-demographics-walking-and-pause",
    "onewoman" : "onewoman_face-demographics-walking-and-pause",
    "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",

'''


for manual_conf in [0.95]: #
    for input_short, input_name in input_videos.items():
        trackVideo = FaceTracker()
        print(input_short, input_name)
        trackVideo.track(
            inputFileFolder = inputFileFolder, 
            input_short = input_short,
            input_name = input_name,
            file_input_path = input_videos_file[input_short],
            detector_name = "RetinaFace",
            manual_conf = manual_conf,
        )


