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
# from facenet_pytorch import MTCNN
# from retinaface import RetinaFace # https://github.com/serengil/retinaface
import face_detection # https://github.com/hukkelas/DSFD-Pytorch-Inference

# Necessary Files
import drawframe
import organizefiles


def track(conf_threshes, iou_threshes):

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
        for iou_thresh in iou_threshes: # For each IOU (interesection over union) threshold

            detectors = {
                "RetinaFace" : face_detection.build_detector("RetinaNetResNet50", confidence_threshold=conf_thresh, nms_iou_threshold=iou_thresh),
                "DSFD" : face_detection.build_detector("DSFDDetector", confidence_threshold=conf_thresh, nms_iou_threshold=iou_thresh)
            }

            for detector_name, detector in detectors.items(): # For each detector

                for input_short, input_name in input_videos.items(): # For each input video
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

                    debugFile.write("STARTED: confidence_threshold = {} nms_iou_threshold = {} \n".format(conf_thresh, iou_thresh))
                    debugFile.write(str(face_detection.torch_utils.get_device()) + '\n')

                    #Start Timer
                    start = time.time()

                    for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                        ret, im = cap.read()
                        # Expects BGR to RGB
                        frame = np.copy(im)[:, :, ::-1]

                        if ret:
                            if frame_num%100 == 0:
                                print("Frame Num: " + str(frame_num))
                            '''
                            elif frame_num==100:
                                break
                            else:
                                continue
                            '''
                            print(frame)
                            print(frame.shape, type(frame), type(frame[0, 0, 0]))
                            faces = detector.detect(frame)

                            numFacesDetected = 0 if faces is None else len(faces)
                            if frame_num%100 == 0:
                                print(numFacesDetected)
                                
                            # Store Frame Information
                            debugFile.write("FRAME#: " + str(frame_num) + " FACESDETECTED: " + str(numFacesDetected) + '\n')

                            if numFacesDetected > 0:
                                for xmin, ymin, xmax, ymax, detection_confidence in faces:
                                    debugFile.write(str(xmin) + " " + str(ymin) + " " + str(xmax) + " " +  str(ymax) + " " + str(detection_confidence) + '\n')
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

track(conf_threshes = [0.35, 0.5, 0.65], iou_threshes = [0.2, 0.3, 0.4])