
# Working directory: "Face Tracker" (not needed for colab)
import sys
sys.path.append("D:/Python/Face Tracker")

# Necessary Packages
import numpy as np  
import time
import cv2

# Necessary Repos

# Necessary Files
import drawframe
import organizefiles


from facenet_pytorch import MTCNN
from facematcher import Matching

def readBoxes(boxFile, manual_conf):
    frame_info = boxFile.readline().strip()
    frame_info_arr = frame_info.split()
    frame_num = int(frame_info_arr[1])
    num_faces = int(frame_info_arr[3])

    faces = np.empty([num_faces, 5])
    for i in range(num_faces):
        faces[i] = np.asarray(boxFile.readline().split())

    boxes = faces[:, 0:4]
    boxes = (boxes.round()).astype(np.int)
    probs = faces[:, 4].round(decimals = 3)

    boxes = boxes[probs>=manual_conf]
    probs = probs[probs>=manual_conf]
    
    return boxes, probs
    
def track(inputFileFolder, input_short, input_name, detector_name, conf_thresh, iou_thresh, manual_conf = 0.5):
    # Input:
    # Open MP4 Input
    cap = organizefiles.openInputVideo(inputFileFolder, input_name)
    
    # Open Txt Input    
    input_long = detector_name + "_" + input_short + "_" + "C" + str(conf_thresh) + "_ " +  "I" + str(iou_thresh) + "v"
    print("outputTexts/" + detector_name + "/" + input_long + "0.txt")

    boxFile = open("outputTexts/" + detector_name + "/" + input_long + "0.txt", "r")

    # Output: 
    # Open MP4 Output
    output_long = detector_name + "_" + input_short + "_" + "C" + str(manual_conf) + "_ " +  "I" + str(iou_thresh) + "v"
    out = organizefiles.openOutputVideo(folder = "outputInterpretedVideos/"  + detector_name, name = ("OUT_" + output_long), 
        fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));

    # Get Video Info
    description = boxFile.readline().strip()
    device = boxFile.readline().strip()

    # Get Frame Info 
    inputVideo_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Scale Attributes
    scaled_thickness = int((4*inputVideo_height)//1440)
    scaled_box_fontScale = (1.0*inputVideo_height)/1440
    scaled_title_fontScale = (1.5*inputVideo_height)/1440
    scaled_title_offset = int((50*inputVideo_height)/1440)
    adjusted_fontColor = (240, 125, 2) if "fourhallway" in input_short else (0, 241, 245)
    
    # Matching Class (VGGFace2)
    matching = Matching()
    
    last_update_cnt = -1
    last_update_frame = -1

    for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()

        if ret:
            # Read Bounding Box Information
            boxes, probs = readBoxes(boxFile, manual_conf) # #TODO: Add landmark retrieval here
            landmarks_dummy = boxes

            print(frame_num, len(boxes))

            # Cropped Faces
            face_array = []
            if len(boxes) > 0:
                for xmin, ymin, xmax, ymax in boxes:
                    xmin = max(xmin, 0)
                    ymin = max(ymin, 0)
                    xmax = max(xmax, 0)
                    ymax = max(ymax, 0)
                    face_array.append(frame[ymin:ymax, xmin:xmax, :])

            numFacesDetected = 0 if face_array is None else len(face_array)

            '''
            # Old-Face Translating Updates
            if numFacesDetected > 0:
                matching.updateBatch(
                    face_array = face_array, 
                    boxes = boxes, 
                    landmarks = landmarks_dummy, 
                    frame_num = frame_num, 
                    thresh = 0.75, # For matching
                )
            
            matching.drawData(frame, fontScale = scaled_box_fontScale, color = adjusted_fontColor)
            '''

            # Primitive On the Spot
            id_list = scores = None
            if numFacesDetected > 0:
                id_list, scores = matching.updateBatch_directNewcentric(
                    face_array = face_array,
                    frame_num = frame_num,
                    thresh = 0.75, # For Matching
                )

            drawframe.notate(img=frame, boxes = boxes, faceids = id_list, 
                thickness = scaled_thickness, fontScale = scaled_box_fontScale, fontColor = adjusted_fontColor)

            if numFacesDetected != last_update_cnt:
                last_update_frame = frame_num
                last_update_cnt = numFacesDetected

            frame_info = "FRAME #: " + str(frame_num) + " Faces Detected: " + str(numFacesDetected) + " Last Update On: " + str(last_update_frame)
            cv2.putText(frame, frame_info, (scaled_title_offset, scaled_title_offset), cv2.FONT_HERSHEY_DUPLEX, scaled_title_fontScale, (0, 0, 0))
            out.write(frame)

        else:
            break

    # Close MP4 Input
    cap.release()

    # Close Txt Input
    boxFile.close()

    #Close MP4 Output 
    out.release()

inputFileFolder = "sourceVideos"
input_videos = {
    "panning" : "dogrunning",
    "oneman" : "oneman_face-demographics-walking-and-pause",
    "onewoman" : "onewoman_face-demographics-walking-and-pause",
    "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",
    "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause",    
    "fourhallway" :  "walkinghallway-pexels",
}

for input_short, input_name in input_videos.items():
    track(
        inputFileFolder = inputFileFolder, 
        input_short = input_short,
        input_name = input_name,
        detector_name = "RetinaFace", 
        conf_thresh = 0.3,
        iou_thresh = 0.3,
        manual_conf = 0.6,
    )
