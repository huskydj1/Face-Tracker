import sys
sys.path.insert(1, 'D:/Python/face.evoLVe.PyTorch/')

from backbone import model_irse as mi
from util import extract_feature_v1 as ef

from applications.align import face_align_import as fa

from scipy import spatial
import os

from PIL import Image
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize


import drawframe

class Face(object):
    def __init__(self, embedding, box, landmarks, frame_num, id):
        self.avg_embedding = embedding

        self.recent_embedding = embedding
        self.recent_box = box
        self.recent_landmarks = landmarks
        self.recent_frame_num = frame_num

        self.static_count = 1
        self.id = id
    
    def update(self, embedding, box, landmarks, frame_num, alpha = 0.3):
        self.avg_embedding = alpha * embedding + (1.0-alpha) * self.avg_embedding

        self.recent_embedding = embedding
        self.recent_box = box
        self.recent_landmarks = landmarks
        self.recent_frame_num = frame_num
        
        self.static_count += 1

    def __str__(self):
        str = "ID: {fid}, REC_FRAME: {fframe}"
        return str.format(fid = self.id, fframe = self.recent_frame_num)

class Matching(object):
    def __init__(self):  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone = mi.Backbone([112, 112], 50, 'ir')
        self.prev_data = []

        #Reset Data Folder
        import shutil

        root = "D:/Python/face.evoLVe.PyTorch/data/FaceTrackerData"
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.mkdir(root)
    
    def __str__(self):
        res = ""
        for facei in self.prev_data:
            res += "[{frep}] ".format(frep = str(facei))
        return res

    def getBoxes(self):
        boxes = []
        for facei in self.prev_data:
            boxes.append(facei.recent_box)
        return boxes
    
    def getLandmarks(self):
        landmarks = []
        for facei in self.prev_data:
            landmarks.append(facei.recent_landmarks)
        return landmarks
    
    def getIds(self):
        id_list = []
        for facei in self.prev_data:
            id_list.append(facei.id)
        return id_list

    def getRecentFrames(self):
        frame_list = []
        for facei in self.prev_data:
            frame_list.append(facei.recent_frame_num)
        return frame_list
    
    def getCounts(self):
        count_list = []
        for facei in self.prev_data:
            count_list.append(facei.static_count)
        return count_list
    
    def drawData(self, frame, fontScale = 0.3, color = (0, 241, 245)):
        boxes = self.getBoxes()
        landmarks = self.getLandmarks()
        id_list = self.getIds()
        frame_list = self.getRecentFrames()

        drawframe.draw_boxes(frame, boxes)
        # drawframe.draw_land(frame, landmarks) TODO: RETRIEVE AND DRAW LANDMARKS
        drawframe.draw_id(frame, id_list, boxes, fontScale = fontScale, color = color)
        drawframe.draw_prob(frame, frame_list, boxes, fontScale = fontScale, color = color)

    def get_embeddings(self, face_array, frame_num):

        # Write in Face Crops from Current Frame
        directory = f"D:/Python/face.evoLVe.PyTorch/data/FaceTrackerData/{frame_num}"

        os.mkdir(directory)

        image_path = directory +  "/id1"
        os.mkdir(image_path)
        for i, face in enumerate(face_array):
            cv2.imwrite(image_path + f"/{i}.jpg", face)

        # Perform Face Alignment First
        os.mkdir(directory + "_align")
        os.mkdir(directory + "_align/id1")

        #python face_align.py -source_root D:/Python/face.evoLVe.PyTorch/data/FaceTrackerData/0 -dest_root D:/Python/face.evoLVe.PyTorch/data/FaceTrackerData/0_align -crop_size 112
        print(directory)
        foundLandmarks = fa.face_align(
            source_root = directory,
            dest_root = directory + '_align',
            crop_size = 112, # I'm not confident if this is correct tbh
        )

        print(foundLandmarks)

        if len(foundLandmarks)==0:
            return {}

        # Get Embeddings
        embeddings = ef.extract_feature(
            data_root = directory + "_align",
            backbone = self.backbone,
            model_root = "D:/Python/face.evoLVe.PyTorch/data/checkpoint/backbone_ir50_ms1m_epoch120.pth",
            input_size = [112, 112], 
            batch_size = 1,
            device = 'cpu', 
        )

        foundEmbeddings = {}
        for i, embedding in zip(foundLandmarks, embeddings):
            index = int(i[0:-4]) # Index in Face Array
            foundEmbeddings[index] = embedding

        return foundEmbeddings
    
    def match_score(self, known_embedding, new_embedding):
        '''known_embedding = normalize(known_embedding.reshape(1, -1))
        new_embedding = normalize(new_embedding.reshape(1, -1))'''
        score = cosine(known_embedding, new_embedding)
        return score

    '''
    def updateBatch(self, face_array, boxes, landmarks, frame_num, thresh = 0.75):
        # Update-based, inferring between lapses in detection
        embeddings = self.get_embeddings(face_array, frame_num)

        N_new = len(face_array)
        N_old = len(self.prev_data)

        matched_new = np.full((N_new,), -1)
        matched_old = np.full((N_old,), -1)

        if N_old > 0:
            # Get all Scores and Distances between new and old faces
            scores = np.zeros((N_new, N_old))
            distances = np.zeros((N_new, N_old))

            for i in range(N_new):
                for j in range(N_old):
                    scores[i, j] = self.match_score(embeddings[i], self.prev_data[j].avg_embedding)
                    # distances[i, j] = np.sum(np.linalg.norm(landmarks[i]-self.prev_data[j].recent_landmarks,
                    #                                    axis = 1))
                    distances[i, j] = np.linalg.norm(landmarks[i]-self.prev_data[j].recent_landmarks)
            
            # Update Existing Faces
            # Match in Order of Increasing Distance (update prev_faces who have a stronger match first)
            values = []
            for i in range(N_new):
                for j in range(N_old):
                    if scores[i, j] <= thresh:
                        values.append((j, i, distances[i, j]));
            values = np.array(values, dtype = [('j', int), ('i', int), ('matched-distance', np.float64)])

            for j, i, match_dist in np.sort(values, order = 'matched-distance'):
                if matched_old[j]==-1 and matched_new[i]==-1:
                    matched_old[j] = i
                    matched_new[i] = j

                    self.prev_data[j].update(embeddings[i], boxes[i], landmarks[i], frame_num)

        #Update for Not Taken New Faces
        for i in range(N_new):
            if matched_new[i] == -1:
                matched_new[i] = len(self.prev_data)
                new_face = Face(embeddings[i], boxes[i], landmarks[i], frame_num, matched_new[i])
                self.prev_data.append(new_face)

        #Update for Not Taken Old Faces
        for j in range(N_old):
            if matched_old[j] == -1:
                self.prev_data[j].static_count += 1
    '''

    def updateBatch_directNewcentric(self, face_array, frame_num, thresh = 0.5):

        embeddings = self.get_embeddings(face_array, frame_num)

        id_mp = {}

        for cur_face, embedding in embeddings.items():
            # Get Best Pairing 
            i_best = -1
            score_best = 100

            for i, person in enumerate(self.prev_data):
                if person.recent_frame_num == frame_num:
                    continue
                elif person.recent_frame_num > frame_num:
                    print("FUTURE FACE FOUND IN PREV_DATA (facematcher.py)")

                score_CurWithPrevI = self.match_score(person.recent_embedding, embedding)
                
                if score_CurWithPrevI < score_best:
                    i_best = i
                    score_best = score_CurWithPrevI
            
            if score_best <= thresh: # If best similarity passes threshold
                # Assign id to cur_face
                id_mp[cur_face] = i_best
                
                # Update Bank of Previous Faces
                self.prev_data[id_mp[cur_face]].recent_embedding = embedding
                self.prev_data[id_mp[cur_face]].recent_frame_num = frame_num
                self.prev_data[id_mp[cur_face]].static_count = 1
            else:
                # Cur_face has not been seen before
                id_mp[cur_face] = len(self.prev_data)
                new_face = Face(
                    embedding = embedding,
                    box = None, 
                    landmarks = None,
                    frame_num = frame_num,
                    id = id_mp[cur_face], 
                )
                self.prev_data.append(new_face)

        return id_mp