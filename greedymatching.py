import numpy as np
from collections import deque

'''
Naive Face-Tracking Implementation

Assumptions:
- high frame-rate video
- accurate face-recognition (with MTCNN)

Algorithm:
- For each new face, compare with faces from previous frame and match with closest one (using L2 distance)

Problems:
- MTCNN is not always accurate, and the gaps in face detection results in incorrect tracking results

Next-Steps:
- New model:
   1) Initialize new faces using facenet-based face identification
   2) If an old face, match with a) same face from previous frame b) consider each face as an update (try to update previous frame)
   3) End based on velocity of face-tracking
'''

class face(object):
    def __init__(self, landmarks, frame_num):
        print(landmarks.shape)
        self.landmarks = landmarks
        self.frame_num = frame_num
    
    def __str__(self):
        return str(self.frame_num)

    def L2Dist(self, other):
        return np.sum(np.linalg.norm(self.landmarks-other.landmarks, axis = 1))

class sustained(object):
    def __init__(self):
        self.dq = deque()
    
    def __str__(self):
        res = ""
        for i, a in enumerate(self.dq):
            res += str(a) + " "
        return res

    def getFront(self):
        return self.dq[0]

    def getEnd(self):
        return self.dq[-1]

    def addEnd(self, new_face):
        return self.dq.append(new_face)

class face_movements(object):
    def __init__(self):
        self.list = []
    
    def __str__(self):
        res = ":"
        for i, a in enumerate(self.list):
            res += "[" + str(a) + "] "
        return res

    def addFace(self, new_face):
        if not self.list:
            new_sustained = sustained()
            new_sustained.addEnd(new_face)
            self.list.append(new_sustained)
            return 0
        else:
            minIndex = -1
            minDist = pow(2, 30)
            
            for i, susti in enumerate(self.list):
                if susti.getEnd().frame_num == new_face.frame_num - 1:
                    curDist = susti.getEnd().L2Dist(new_face)

                    if curDist < minDist:
                        minDist = curDist
                        minIndex = i

            if minIndex == -1:
                new_sustained = sustained()
                new_sustained.addEnd(new_face)
                self.list.append(new_sustained)
                return len(self.list) - 1
            else:
                self.list[minIndex].addEnd(new_face)
                return minIndex
