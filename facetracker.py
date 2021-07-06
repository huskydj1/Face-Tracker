
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

from facematcher import Matching

class FaceTracker(object):
    def __init__(self):
        self.dummy_id = 1e3

        self.dot_bank = []

        self.color_map = {}
        self.color_map[self.dummy_id] = (0, 0, 255)

    
    # Drawing people's paths
    def randomColorGenerator(self):
        return list(np.random.random(size=3) * 256)
    def getColor(self, id):
        if not(id in self.color_map.keys()):
            color = self.randomColorGenerator()
            while color in self.color_map.values():
                color = self.randomColorGenerator()
            self.color_map[id] = color

        return self.color_map[id]

    # Helper Function for Reading Boxes from File (Outputted by RetinaFace, run by Colab)
    def readBoxes(self, boxFile, manual_conf):
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
    
    def track(self, inputFileFolder, input_short, input_name, detector_name, conf_thresh, iou_thresh, manual_conf = 0.5):
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

            if frame_num>20:
                continue

            if ret:
                # Read Bounding Box Information
                boxes, probs = self.readBoxes(boxFile, manual_conf) # #TODO: Add landmark retrieval here
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
                id_mp = None
                
                if numFacesDetected > 0:
                    id_mp = matching.updateBatch_directNewcentric(
                        face_array = face_array,
                        frame_num = frame_num,
                        thresh = 0.7, # For Matching
                    )
                    
                    # Fill First with Dummy Ids
                    id_list = np.full(shape = (numFacesDetected), fill_value = self.dummy_id, dtype = np.int16)

                    # Correct faces with matched ids
                    for i, id in id_mp.items():
                        assert(id != self.dummy_id)
                        id_list[i] = id
                    
                    # Assign Colors
                    boxColors = []
                    for i, id in enumerate(id_list):
                        boxColors.append(self.getColor(id))

                        center = (
                            int(round((boxes[i][0] + boxes[i][2])/2)), 
                            int(round((boxes[i][1] + boxes[i][3])/2)),                        
                        )

                        self.dot_bank.append((center, boxColors[i]))

                    drawframe.notate(
                        img=frame, 
                        boxes = boxes,
                        boxColors = boxColors,
                        faceids = id_list,
                        thickness = scaled_thickness,
                        fontScale = scaled_box_fontScale,
                        fontColor = adjusted_fontColor,
                        dummyId = self.dummy_id,
                    )

                drawframe.drawDots(img=frame, dot_bank = self.dot_bank)

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
    "rainpedestrians" : "crowdedstreetundertherain-pexels",
    "bigcrowd" : "skywalkmahanakhon-videvo",
}
'''
Inactive Videos:
    
    "fourhallway" :  "walkinghallway-pexels",
    "panning" : "dogrunning",
    "oneman" : "oneman_face-demographics-walking-and-pause",
    "onewoman" : "onewoman_face-demographics-walking-and-pause",
    "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",
    "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause",  
'''

for input_short, input_name in input_videos.items():
    trackVideo = FaceTracker()
    trackVideo.track(
        inputFileFolder = inputFileFolder, 
        input_short = input_short,
        input_name = input_name,
        detector_name = "RetinaFace", 
        conf_thresh = 0.3,
        iou_thresh = 0.3,
        manual_conf = 0.6,
    )
