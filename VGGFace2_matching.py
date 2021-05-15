from PIL import Image
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

from facenet_pytorch import InceptionResnetV1

class Face(object):
    def __init__(self, embedding, frame_num, id):
        self.embedding = embedding  
        self.frame_num = frame_num
        self.id = id

class Matching(object):
    def __init__(self):  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.prev_data = []

        
    def __str__(self):
        res = ""
        for person in self.prev_data:
            res += "["
            for face in person:
                res += str(face.frame_num) + " "
            res += "]"
        return res

    def get_embeddings(self, face_array):       
        aligned = face_array.to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()
        return embeddings
    
    def match_score(self, known_embedding, new_embedding):
        '''known_embedding = normalize(known_embedding.reshape(1, -1))
        new_embedding = normalize(new_embedding.reshape(1, -1))'''
        score = cosine(known_embedding, new_embedding)
        return score
    
   #def is_match(self, known_embedding, new_embedding, thresh):
   #     return self.match_score(known_embedding, new_embedding) <= thresh
    
    def update_batch(self, face_array, frame_num, thresh = 0.5):
        embeddings = self.get_embeddings(face_array)

        id_list = []
        scores = []

        for embedding in embeddings:
            scores.append([])

            best_i = -1
            best_score = 100
            for i, person in enumerate(self.prev_data):
                known_face = person[-1]
                cur_score = self.match_score(known_face.embedding, embedding)
                scores[-1].append(cur_score)
                # print(cur_score)
                if cur_score < best_score:
                    best_score = cur_score
                    best_i = i
            
            if best_score <= thresh:
                id_list.append(best_i)
                self.prev_data[best_i].append(Face(embedding, frame_num, best_i))
            else:
                id_list.append(len(self.prev_data))
                new_face = Face(embedding, frame_num, len(self.prev_data))
                self.prev_data.append([new_face])

        return [id_list, scores]
