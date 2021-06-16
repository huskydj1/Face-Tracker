
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

def interpretRetina(conf_threshes):

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

            #Open TXT Input
            input_long = detector_name + "_" + input_short + "_" + "C" + str(conf_thresh) + "v"
            print("outputTexts/" + detector_name + "/" + input_long + "0.txt")

            '''
            boxFile = open("outputTexts/" + detector_name + "/" + input_long + "0.txt", "r")


            #Open MP4 Output

            # Run
            #//////////////////////////////////

            #Start Timer
            start = time.time()

            for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                frame_filename = "RetinaFace_FRAME.jpg"
                ret, frame = cap.read()
                cv2.imwrite(frame_filename, frame)

                if ret:
                    if frame_num%100 == 0:
                        print("Frame Num: " + str(frame_num))

                    elif frame_num==100:
                        break
                    else:
                        continue

                else:
                    break
            
            #//////////////////////////////////

            #Close Input
            cap.release()

            #Close TXT Output
            debugFile.close()
            '''

def interpretDSFD(conf_threshes, iou_threshes):

    inputFileFolder = "sourceVideos"
    input_videos = {
        "fourhallway" :  "walkinghallway-pexels",
        "panning" : "dogrunning",
        "oneman" : "oneman_face-demographics-walking-and-pause",
        "onewoman" : "onewoman_face-demographics-walking-and-pause",
        "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",
        "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause"
    }
    
    
    for conf_thresh in conf_threshes: # For each confidence threshold
        for iou_thresh in iou_threshes: # For each IOU (interesection over union) threshold

            detector_name = "DSFD"

            for input_short, input_name in input_videos.items(): # For each input video
                # Input:
                # Open MP4 Input
                cap = organizefiles.openInputVideo(inputFileFolder, input_name)

                # Open Txt Input
                input_long = detector_name + "_" + input_short + "_" + "C" + str(conf_thresh) + "_ " +  "I" + str(iou_thresh) + "v"
                print("outputTexts/" + detector_name + "/" + input_long + "0.txt")

                boxFile = open("outputTexts/" + detector_name + "/" + input_long + "0.txt", "r")

                # Output: 
                # Open MP4 Output
                out = organizefiles.openOutputVideo(folder = "outputInterpretedVideos", name = ("OUT_" + input_long), 
                    fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));


                # Start Reading
                description = boxFile.readline()
                device = boxFile.readline()

                for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    ret, frame = cap.read()

                    if ret:
                        frame_info = boxFile.readline().split()
                        frame_num = int(frame_info[1])
                        num_faces = int(frame_info[3])

                        faces = np.empty([num_faces, 5])
                        for i in range(num_faces):
                            faces[i] = np.asarray(boxFile.readline().split())
                        boxes = faces[:, 0:4]
                        probs = faces[:, 4]

                        drawframe.notate(img=frame, boxes = boxes, probs = probs)
                        cv2.putText(frame, frame_info, (30, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
                        out.write(frame)                    
                    else:
                        break

                # Close MP4 Input
                cap.release()

                # Close Txt Input
                boxFile.close()

                #Close MP4 Output 
                out.release()

# interpretRetina(conf_threshes = [0.3])

interpretDSFD(conf_threshes = [0.3], iou_threshes = [0.2, 0.3, 0.4])