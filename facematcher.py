import sys
sys.path.insert(1, 'D:/Python/face.evoLVe.PyTorch/')

from backbone.model_irse import Backbone
from util import extract_feature_v1 as ef

from applications.align import face_align_import as fa

from scipy import spatial
import os

from PIL import Image
import math
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize


import drawframe

class Face(object):
    def __init__(self, embedding, landmarks, img, frame_num, id):
        self.recent_embedding = embedding
        self.recent_landmarks = landmarks
        self.recent_img = img
        self.recent_frame_num = frame_num

        self.static_count = 1
        self.id = id
    
    '''
    def update(self, embedding, frame_num, alpha = 0.3):
        self.avg_embedding = alpha * embedding + (1.0-alpha) * self.avg_embedding

        self.recent_embedding = embedding
        self.recent_frame_num = frame_num
        
        self.static_count += 1
    '''

class Matching(object):
    def __init__(self):  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.prev_data = []

        #Reset Data Folder
        import shutil

        root = "D:/Python/face.evoLVe.PyTorch/data/FaceTrackerData"
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.mkdir(root)
    
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
    
    def get_embeddings(self, face_array, landmark_array, frame_num):

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
        #print(directory)
        fa.face_align(
            source_root = directory,
            dest_root = directory + '_align',
            crop_size = 112, # I'm not confident if this is correct tbh
            landmark_array = landmark_array,
        )

        # Get Embeddings
        # print(directory + "_align")
        feature_mp = ef.extract_feature(
            data_root = directory + "_align",
            backbone = Backbone(input_size = [112, 112], num_layers = 50),
            model_root = "D:/Python/face.evoLVe.PyTorch/data/checkpoint/backbone_ir50_ms1m_epoch120.pth",
            input_size = [112, 112], 
            batch_size = 1,
            device = 'cpu', 
            tta = False, # Maybe not, dunno yet
        ) # Map of features (img_path, feature embedding shape = (512, ))

        embeddings = []
        for i in range(len(face_array)):
            path_i = directory + "_align\\id1\\" + str(i) + ".jpg"
            # print(i, path_i)
            embeddings.append(feature_mp[path_i])
        
        return embeddings
    
    def match_score(self, x, y):
        '''known_embedding = normalize(known_embedding.reshape(1, -1))
        new_embedding = normalize(new_embedding.reshape(1, -1))'''
        # score=cosine_similarity(x.reshape(1,-1),y.reshape(1,-1))
        score = cosine(u = x, v = y)
        return 1 - score

    def landmarkDist(self, landmarks_a, landmarks_b):
        avg_a = np.mean(a = landmarks_a, axis = 0)
        avg_b = np.mean(a = landmarks_b, axis = 0)
        return np.linalg.norm(avg_a - avg_b) 

    def updateBatch_direct(self, face_array, landmark_array, actuallandmark_array, frame_num, thresh = 0.5):
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        new_embeddings = self.get_embeddings(face_array, landmark_array, frame_num)
        
        num_new = len(new_embeddings)
        num_old = len(self.prev_data)

        '''
        if frame_num == 1:
            for new_i in range(num_new): 
                print(new_i, new_embeddings[new_i])
            for old_i in range(num_old):
                print(old_i, self.prev_data[old_i].recent_embedding)
        '''
        
        # Initialize here  so that we can use it if numOld==0
        new_used = np.zeros(shape = (num_new,))
        old_used = np.zeros(shape = (num_old,))
        id_mp = {}

        # print(num_new, num_old)

        if num_old > 0:

            # Goal of all of this is to find the cos_similarity between all pairs of new and old faces
            
            cos_similarity = np.zeros(shape = (num_new, num_old))
            for new_i in range(0, num_new):
                for old_i in range(0, num_old):
                    cos_similarity[new_i, old_i] = self.match_score(x = new_embeddings[new_i], y = self.prev_data[old_i].recent_embedding)

            # print(cos_similarity)

            score_list = []
            for new_i in range(num_new):
                for old_i in range(num_old):
                    if cos_similarity[new_i, old_i] > thresh:
                        score_list.append((new_i, old_i, cos_similarity[new_i, old_i]))
                    # score_list.append((new_i, old_i, cos_similarity[new_i, old_i]))
            score_list = np.array(score_list, dtype = [('new_i', int), ('old_i', int), ('dist', np.float64)])
            score_list = np.sort(score_list, order = 'dist')
            score_list = score_list[::-1]
            
            # print(score_list)

            # Match

            for new_i, old_i, dist in score_list:
                #TODO: ADDED FOR VOC USE, CONSIDER DELETING OR REFINING FOR REAL RUNS
                # print(new_i, old_i, self.landmarkDist(landmarks_a = actuallandmark_array[new_i], landmarks_b = self.prev_data[old_i].recent_landmarks))
                if self.landmarkDist(landmarks_a = actuallandmark_array[new_i], landmarks_b = self.prev_data[old_i].recent_landmarks) > 50:
                    continue

                if new_used[new_i] == 1:
                    continue
                elif old_used[old_i] == 1:
                    continue
                else:
                    # Assign id to cur_face
                    id_mp[new_i] = old_i
                    
                    # Update Bank of Previous Faces
                    self.prev_data[old_i].recent_embedding = new_embeddings[new_i]
                    self.prev_data[old_i].recent_landmarks = actuallandmark_array[new_i]
                    self.prev_data[old_i].recent_img = face_array[new_i]
                    self.prev_data[old_i].recent_frame_num = frame_num
                    self.prev_data[old_i].static_count = 1

                    new_used[new_i] = 1
                    old_used[old_i] = 1

        for new_i, status in enumerate(new_used):
            if status == 0:
                # Cur_face has not been seen before
                id_mp[new_i] = len(self.prev_data)
                new_face = Face(
                    embedding = new_embeddings[new_i],
                    landmarks = actuallandmark_array[new_i],
                    img = face_array[new_i],
                    frame_num = frame_num,
                    id = id_mp[new_i], 
                )
                self.prev_data.append(new_face)

        # print(id_mp)

        return id_mp


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