#https://github.com/hukkelas/DSFD-Pytorch-Inference

import sys
sys.path.append("D:/Python/Face Tracker")

import numpy as np  
import cv2
import time
import torch
import face_detection

import drawframe
import organizefiles


def track(inputFileFolder, inputFileName, device, fontScale = 0.3, thresh = 0.5):
    #Open Input
    cap = organizefiles.openInputVideo(inputFileFolder, inputFileName)
    
    #Open MP4 Output
    out = organizefiles.openOutputVideo(folder = "outputVideos", name = ("DFSD_" + inputFileName), 
        fps = 8, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)));

    #Open TXT Output
    debugFile = organizefiles.openOutputText(folder = "outputTexts", name = ("DFSDINFO_" + inputFileName))

    #//////////////////////////////////

    #DFSD
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device = torch.device("cpu"))

    debugFile.write("STARTED: confidence_threshold = {} nms_iou_threshold = {} \n".format("0.5", "0.3"))
    print(face_detection.torch_utils.get_device())
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
            # else:
            #    continue
            
            faces = detector.detect(frame)

            numFacesDetected = 0 if faces is None else len(faces)

            if frame_num%100 == 0:
                print(numFacesDetected)

            if numFacesDetected > 0:
                im = drawframe.draw_boxes(img = im, boxes = faces)

            debugFile.write(str(frame_num) + ": " + str(numFacesDetected) + " " 
                + str(faces) + '\n')

            infoTemp = "FRAME #: " + str(frame_num) + " Faces Detected: " + str(numFacesDetected)
            cv2.putText(im, infoTemp, (30, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
            out.write(im)
        else:
            break

    #End Timer
    end = time.time()

    debugFile.write("COMPLETED Runtime of the program is {} seconds \n".format(end - start))
    
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

# RUN
'''
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# BGR to RGB
im = cv2.imread("sourceVideos/muskbezos.jpg")
frame = np.copy(im)[:, :, ::-1]

detections = detector.detect(frame)

im = drawframe.draw_boxes(img = im, boxes = detections)

cv2.imshow("Display window", np.array(im, dtype = np.uint8))
k = cv2.waitKey(0)
cv2.destroyAllWindows()
'''

track("sourceVideos", "walkinghallway-pexels", device = device, fontScale = 1.3)
track("sourceVideos", "dogrunning", device = device, fontScale = 1.3)

'''
track("sourceVideos", "oneman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onewoman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onemanonewoman_face-demographics-walking-and-pause", device = device)
track("sourceVideos", "onemantwowomen_face-demographics-walking-and-pause", device = device)

track("sourceVideos", "walkinghallway-pexels", device = device, fontScale = 1.3)
track("sourceVideos", "dogrunning", device = device, fontScale = 1.3)
'''