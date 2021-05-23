from PIL import Image
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

from facenet_pytorch import InceptionResnetV1


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

class Matching(object):
    def __init__(self):  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.prev_data = []
    
    def drawData(self, frame):
        boxes = []
        landmarks = []
        id_list = []
        count_list = []

        for facei in self.prev_data:
            boxes.append(facei.recent_box)
            landmarks.append(facei.recent_landmarks)
            id_list.append(facei.id)
            count_list.append(facei.static_count)

        drawframe.draw_boxes(frame, boxes)
        drawframe.draw_land(frame, landmarks)
        drawframe.draw_id(frame, id_list, boxes, fontScale = 1.0)
        drawframe.draw_prob(frame, count_list, boxes, fontScale = 1.0)

    def get_embeddings(self, face_array):       
        aligned = face_array.to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()
        return embeddings
    
    def match_score(self, known_embedding, new_embedding):
        '''known_embedding = normalize(known_embedding.reshape(1, -1))
        new_embedding = normalize(new_embedding.reshape(1, -1))'''
        score = cosine(known_embedding, new_embedding)
        return score

    def updateBatch(self, face_array, boxes, landmarks, frame_num, thresh = 0.75):
        embeddings = self.get_embeddings(face_array)
        taken = np.full((len(face_array),), -1)

        if len(self.prev_data) > 0:
            # Get all Scores and Distances between new and old faces
            scores = np.zeros((len(face_array), len(self.prev_data)))
            distances = np.zeros((len(face_array), len(self.prev_data)))

            for i in range(len(face_array)):
                for j in range(len(self.prev_data)):
                    scores[i][j] = self.match_score(self.prev_data[j].avg_embedding,
                                                        embeddings[i])
                    distances[i][j] = np.sum(np.linalg.norm(landmarks[i]-self.prev_data[j].recent_landmarks,
                                                        axis = 1))
            
            # Update Existing Faces
            MAX_DIST = pow(10, 32)
            distances[scores>thresh] = MAX_DIST
            matches = np.argmin(distances, axis = 0)

            # print(str(frame_num) + ": " + str(distances))

            for j, old_face in enumerate(self.prev_data):
                if distances[matches[j]][j] == MAX_DIST or taken[matches[j]] != -1:
                    old_face.static_count += 1
                else:
                    taken[matches[j]] = j
                    old_face.update(embeddings[matches[j]], boxes[matches[j]], landmarks[matches[j]], frame_num)

        # Add Non-Taken Faces
        for i in range(len(face_array)):
            if taken[i] == -1:
                taken[i] = len(self.prev_data)
                new_face = Face(embeddings[i], boxes[i], landmarks[i], frame_num, taken[i])
                self.prev_data.append(new_face)

    def updateBatch_directNewcentric(self, face_array, frame_num, thresh = 0.5):
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
