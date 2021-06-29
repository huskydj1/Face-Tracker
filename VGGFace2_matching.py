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

    def __str__(self):
        str = "ID: {fid}, REC_FRAME: {fframe}"
        return str.format(fid = self.id, fframe = self.recent_frame_num)

class Matching(object):
    def __init__(self):  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.prev_data = []
    
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
        drawframe.draw_land(frame, landmarks)
        drawframe.draw_id(frame, id_list, boxes, fontScale = fontScale, color = color)
        drawframe.draw_prob(frame, frame_list, boxes, fontScale = fontScale, color = color)

    def get_embeddings(self, face_array):       
        # aligned = face_array.to(self.device)
        # embeddings = self.resnet(aligned).detach().cpu()
        embeddings = self.resnet(face_array)
        return embeddings
    
    def match_score(self, known_embedding, new_embedding):
        '''known_embedding = normalize(known_embedding.reshape(1, -1))
        new_embedding = normalize(new_embedding.reshape(1, -1))'''
        score = cosine(known_embedding, new_embedding)
        return score

    def updateBatch(self, face_array, boxes, landmarks, frame_num, thresh = 0.75):
        embeddings = self.get_embeddings(face_array)

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
                    distances[i, j] = np.sum(np.linalg.norm(landmarks[i]-self.prev_data[j].recent_landmarks,
                                                        axis = 1))
            
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
