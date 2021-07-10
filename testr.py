
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
        num_faces = int(boxFile.readline().strip())

        box_info = np.empty([num_faces, 5])
        landmarks = np.empty((num_faces, 5, 2))
        for i in range(num_faces):
            box_info[i] = np.asarray(boxFile.readline().split())
            for j in range(5):
                landmarks[i][j] = np.asarray(boxFile.readline().split())

        boxes = box_info[:, 0:4]
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        probs = box_info[:, 4].round(decimals = 3)

        boxes = (boxes.round()).astype(np.int)
        landmarks = (landmarks.round().astype(np.int))

        boxes = boxes[probs>=manual_conf]
        landmarks = landmarks[probs>=manual_conf]
        probs = probs[probs>=manual_conf]
        
        return boxes, probs, landmarks
    
    def track(self, inputFileFolder, input_short, input_name, file_input_path, detector_name, manual_conf = 0.5, manual_match = 0.7):
        # Input:
        # Open MP4 Input
        cap = organizefiles.openInputVideo(inputFileFolder, input_name)
        
        # Output: 
        # Open MP4 Output
        output_long = detector_name + "_" + input_short + "_" + "C" + str(manual_conf) + "_ " + "M" + str(manual_match) + "_" + "v"
        out = organizefiles.openOutputVideo(folder = "outputInterpretedVideos/"  + detector_name, name = ("OUT_" + output_long), 
            fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));

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

             # Open Txt Input    
            boxFile = open(file_input_path + "/{}.txt".format(str(frame_num)), "r")

            # Get Video Info
            filename = boxFile.readline().strip()

            
            if frame_num>1: 
                continue

            if ret:
                # Read Bounding Box Information
                boxes, probs, landmarks = self.readBoxes(boxFile, manual_conf)

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

                # Primitive On the Spot
                id_mp = None
                
                if numFacesDetected > 0:
                    translatedLandmarks = np.empty(landmarks.shape)
                    
                    for i in range(len(translatedLandmarks)):
                        for j in range(len(translatedLandmarks[i])):
                            for k in range(len(translatedLandmarks[i, j])):
                                translatedLandmarks[i][j][k] = landmarks[i, j, k] - boxes[i, k]

                    '''
                    for i in range(len(face_array)):
                        face = drawframe.draw_land(
                            img = face_array[i], 
                            landmarks = translatedLandmarks[i][np.newaxis, :, :],
                        )
                        cv2.imshow("Face{}".format(i), face)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    '''

                    id_mp = matching.updateBatch_direct(
                        face_array = face_array,
                        landmark_array = translatedLandmarks,
                        frame_num = frame_num,
                        thresh = manual_match, # For Matching
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
                        landmarks = landmarks,
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
    "bigcrowd" : "skywalkmahanakhon-videvo",
}
'''
Up Next:

    "fourhallway" :  "walkinghallway-pexels",
    "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause",  
    "rainpedestrians" : "crowdedstreetundertherain-pexels",
'''

input_videos_file = {
    "bigcrowd" : "D:/Python/Face Tracker/outputTexts/6--bigcrowd",
}
'''
Up Next:

    "fourhallway" :  "D:/Python/Face Tracker/outputTexts/5--fourhallway/",
    "rainpedestrians" : "D:/Python/Face Tracker/outputTexts/7--rainpedestrians",
    "onemantwowoman" : "D:/Python/Face Tracker/outputTexts/0--onemantwowoman/",
'''

'''
Inactive Videos:
    "panning" : "dogrunning",
    "oneman" : "oneman_face-demographics-walking-and-pause",
    "onewoman" : "onewoman_face-demographics-walking-and-pause",
    "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",
'''
for manual_conf in [0.99]: #
    for manual_match in [0.75]: #[0.65, 0.75, 0.85]
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
                manual_match = manual_match,
            )
