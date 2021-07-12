
# Working directory: "Face Tracker" (not needed for colab)
import sys
sys.path.append("D:/Python/Face Tracker")

import os

# Necessary Packages
import numpy as np  
import time
import cv2

# Necessary Repos

# Necessary Files
import drawframe
import organizefiles

from facematcher import Matching

    
def breakdown(inputFileFolder, input_short, input_name):
    # Input:
    # Open MP4 Input
    cap = organizefiles.openInputVideo(inputFileFolder, input_name)

    
    videoFolderName = "D:/Python/Face Tracker/VideosFrameByFrame" + "/" + input_short
    os.mkdir(videoFolderName)


    for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()

        if ret:
            frameFileName = videoFolderName + "/" + str(frame_num) + ".jpg"
            cv2.imwrite(filename = frameFileName, img = frame)
        else:
            break

    # Close MP4 Input
    cap.release()

def generateWiderValTxt():
    #Open TXT Output
    out = open("D:/Python/Face Tracker/wider_val.txt", "w")

    '''
    for i in range(478):
        out.write("/0--onemantwowoman/{}.jpg".format(str(i)) + '\n')
    for i in range(391):
        out.write("/5--fourhallway/{}.jpg".format(str(i)) + '\n')
    for i in range(705):
        out.write("/6--bigcrowd/{}.jpg".format(str(i)) + '\n')
    for i in range(821):
        out.write("/7--rainpedestrians/{}.jpg".format(str(i)) + '\n')
    '''

    for i in range(1525):
        out.write("/8--voccamp/{}.jpg".format(str(i)) + '\n')
    
    out.close()


inputFileFolder = "sourceVideos"
input_videos = {  
    "voccamp" : "voccamp",
}
'''
Inactive Videos:
    Have been run before, used with up-to-date retinaface:
    "rainpedestrians" : "crowdedstreetundertherain-pexels",
    "bigcrowd" : "skywalkmahanakhon-videvo",
    "fourhallway" :  "walkinghallway-pexels",
    "onemantwowoman" : "onemantwowomen_face-demographics-walking-and-pause",  

    Have never been run yet:
    "panning" : "dogrunning",
    "oneman" : "oneman_face-demographics-walking-and-pause",
    "onewoman" : "onewoman_face-demographics-walking-and-pause",
    "onemanonewoman" : "onemanonewoman_face-demographics-walking-and-pause",
'''
'''
for input_short, input_name in input_videos.items():
    breakdown(
        inputFileFolder = inputFileFolder, 
        input_short = input_short,
        input_name = input_name,
    )

'''

generateWiderValTxt()