# Working directory: "Face Tracker" (not needed for colab)
import sys
sys.path.append("D:/Python/Face Tracker")

# Necessary Packages
import numpy as np 
import cv2

# Necessary Repos

# Necessary Files
import organizefiles

def combine(retinaFileName, dsfdFileName, mtcnnFileName, outputFileName):
    
    # Input:
    # Open MP4 Inputs
    retinaCap = organizefiles.openInputVideo("outputInterpretedVideos/RetinaFace", retinaFileName)
    retinaCap_height = int(retinaCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    retinaCap_width = int(retinaCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    retinaCap_length = int(retinaCap.get(cv2.CAP_PROP_FRAME_COUNT))

    dsfdCap = organizefiles.openInputVideo("outputInterpretedVideos/DSFD", dsfdFileName)
    dsfdCap_height = int(dsfdCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dsfdCap_width = int(dsfdCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dsfdCap_length = int(dsfdCap.get(cv2.CAP_PROP_FRAME_COUNT))

    mtcnnCap = organizefiles.openInputVideo("outputVideos", mtcnnFileName)
    mtcnnCap_height = int(mtcnnCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mtcnnCap_width = int(mtcnnCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mtcnnCap_length = int(mtcnnCap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert retinaCap_height==dsfdCap_height and retinaCap_width==dsfdCap_width and retinaCap_length==dsfdCap_length
    assert mtcnnCap_height==dsfdCap_height and mtcnnCap_width==dsfdCap_width and mtcnnCap_length==dsfdCap_length

    # Output:
    # Open MP4 Output

    out = organizefiles.openOutputVideo(folder = "outputVideos", name = outputFileName, 
                fps = 8, frame_height = 2 * retinaCap_height, frame_width = 2 * retinaCap_width)

    for frame_num in range(retinaCap_length):
        retA, retinaCap_frame = retinaCap.read()
        retB, dsfdCap_frame = dsfdCap.read()
        retC, mtcnnCap_frame = mtcnnCap.read()

        if retA:
            cv2.putText(retinaCap_frame, "RetinaFace, Conf Thres: 0.9", (25, retinaCap_height-25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
            cv2.putText(dsfdCap_frame, "DSFD, Conf Thres: 0.6, IOU Thresh: 0.3", (25, dsfdCap_height-25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
            mtcnnCap_frame = cv2.putText(mtcnnCap_frame, "Stock MTCNN", (25, mtcnnCap_height-25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))


            topRow = np.hstack((mtcnnCap_frame, mtcnnCap_frame))
            bottomRow = np.hstack((retinaCap_frame, dsfdCap_frame))
            out.write(np.vstack((topRow, bottomRow)))
        else:
            break

    # Close MP4 Inputs
    retinaCap.release()
    dsfdCap.release()
    mtcnnCap.release()

    #Close MP4 Output 
    out.release()

combine(retinaFileName = "OUT_RetinaFace_fourhallway_C0.3v0", dsfdFileName = "OUT_DSFD_fourhallway_C0.3_ I0.3v0", 
    mtcnnFileName = "MTCNN_walkinghallway-pexels0", outputFileName = "COMBINED_fourhallway_v")

combine(retinaFileName = "OUT_RetinaFace_oneman_C0.3v0", dsfdFileName = "OUT_DSFD_oneman_C0.3_ I0.3v0", 
    mtcnnFileName = "MTCNN_oneman_face-demographics-walking-and-pause0", outputFileName = "COMBINED_oneman_v")

combine(retinaFileName = "OUT_RetinaFace_onemanonewoman_C0.3v0", dsfdFileName = "OUT_DSFD_onemanonewoman_C0.3_ I0.3v0", 
    mtcnnFileName = "MTCNN_onemanonewoman_face-demographics-walking-and-pause0", outputFileName = "COMBINED_onemanonewoman_v")

combine(retinaFileName = "OUT_RetinaFace_onemantwowoman_C0.3v0", dsfdFileName = "OUT_DSFD_onemantwowoman_C0.3_ I0.3v0", 
    mtcnnFileName = "MTCNN_onemantwowomen_face-demographics-walking-and-pause0", outputFileName = "COMBINED_onemantwowoman_v")

combine(retinaFileName = "OUT_RetinaFace_onewoman_C0.3v0", dsfdFileName = "OUT_DSFD_onewoman_C0.3_ I0.3v0", 
    mtcnnFileName = "MTCNN_onewoman_face-demographics-walking-and-pause0", outputFileName = "COMBINED_onewoman_v")

combine(retinaFileName = "OUT_RetinaFace_panning_C0.3v0", dsfdFileName = "OUT_DSFD_panning_C0.3_ I0.3v0", 
    mtcnnFileName = "MTCNN_dogrunning0", outputFileName = "COMBINED_panning_v")