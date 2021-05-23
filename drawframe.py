import cv2
import numpy as np

def draw_boxes(img, boxes, color = (0, 0, 255), thickness = 2):
    for box in boxes:
            cv2.rectangle(img = img, pt1 = (box[0], box[1]), pt2 = (box[2], box[3]), 
                color = color, thickness = thickness)
    return img

def draw_prob(img, probs, boxes, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (217, 255, 0)):
    for prob, box in zip(probs, boxes):
        cv2.putText(img = img, text = str(format(prob, '.3f')), org = (box[2], box[3]), 
                fontFace = fontFace, fontScale = fontScale, 
                    color = color)
    return img

def draw_land(img, landmarks, radius = 2, color = (255, 0, 0)):
    for ld in landmarks:
        for land_x, land_y in ld:
                cv2.circle(img = img, center=(land_x, land_y), radius = radius, 
                    color = color)
    return img

def draw_id(img, faceids, boxes, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (217, 255, 0)):
    for idi, box in zip(faceids, boxes):
        cv2.putText(img = img, text = str(idi), org = (box[2], box[1]), 
                fontFace = fontFace, fontScale = fontScale, 
                    color = color)
    return img

def notate(img, boxes, landmarks = None, probs = None, faceids = None):
    img = draw_boxes(img, boxes)
    if not landmarks is None:
        img = draw_land(img, landmarks)
    if not probs is None:
        img = draw_prob(img, probs, boxes)
    if not faceids is None:
        img = draw_id(img, faceids, boxes)
        
    return img