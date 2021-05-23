import cv2
import os.path

def find_nextavailable_file(folder, name, type = "mp4"):
    file_name = folder + "/" + name + '{}' + "." + type;

    file_num = 0
    while os.path.isfile(file_name.format(file_num)):
        file_num += 1
    res = file_name.format(file_num)

    print(res)
    return res

def openInputVideo(folder, name, type = "mp4", fps = None):
    inputFile = folder + "/" + name + "." + type

    cap = cv2.VideoCapture(inputFile)

    if not (fps is None):
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap

def openOutputVideo(folder, name, fps, frame_height, frame_width):
    #TODO: ACCEPT OUTPUTVIDEOS OTHER THAN MP4s
    fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(find_nextavailable_file(folder = folder,
        name= name, type = "mp4"), 
        fourcc1, fps, (frame_width, frame_height))
    
    return out

def openOutputText(folder, name):
    file = find_nextavailable_file(folder = folder, name = name, type = "txt")

    return open(file, "w")