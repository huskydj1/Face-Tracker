import cv2
import numpy as np


def draw_boxes(img, boxes, boxColors = [], thickness = 2):
    for i, box in enumerate(boxes):
            cv2.rectangle(img = img, pt1 = (box[0], box[1]), pt2 = (box[2], box[3]), 
                color = (0, 0, 255) if boxColors is None else boxColors[i], thickness = thickness)
    return img

def draw_prob(img, probs, boxes, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (240, 125, 2)):
    for prob, box in zip(probs, boxes):
        cv2.putText(img = img, text = str(prob), org = (box[2], box[3]), 
                fontFace = fontFace, fontScale = fontScale, 
                    color = color)
    return img

def draw_land(img, landmarks, radius = 2, color = (255, 0, 0)):
    for ld in landmarks:
        for land_x, land_y in ld:
                cv2.circle(img = img, center=(land_x, land_y), radius = radius, 
                    color = color)
    return img

def draw_id(img, faceids, boxes, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (0, 241, 245), dummyId = 1e3):
    for idi, box in zip(faceids, boxes):
        cv2.putText(
            img = img,
            text = str(-1) if dummyId == idi else str(idi),
            org = (box[2], box[1]), 
            fontFace = fontFace,
            fontScale = fontScale, 
            color = color
        )
    return img

def notate(img, boxes, boxColors = [], landmarks = None, probs = None, faceids = None, thickness = 2, fontScale = 0.3, fontColor = (0, 241, 245), dummyId = 1e3):
    img = draw_boxes(img, boxes, boxColors = boxColors, thickness = thickness)
    if not landmarks is None:
        img = draw_land(img, landmarks)
    if not probs is None:
        img = draw_prob(img, probs, boxes, fontScale = fontScale, color = fontColor)
    if not faceids is None:
        img = draw_id(img, faceids, boxes, fontScale = fontScale, color = fontColor, dummyId = dummyId)
        
    return img

def drawDots(img, dot_bank):
    for point, color in dot_bank:
        assert(len(point)==2)
        cv2.circle(
            img = img, 
            center = point, 
            radius = 3, 
            color = color, 
            thickness = -1, 
        )
'''
def draw_boxes(img, boxes, color = (0, 0, 255), thickness = 2):
    for box in boxes:
            cv2.rectangle(img = img, pt1 = (box[0], box[1]), pt2 = (box[2], box[3]), 
                color = color, thickness = thickness)
    return img

def draw_prob(img, probs, boxes, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (240, 125, 2)):
    for prob, box in zip(probs, boxes):
        cv2.putText(img = img, text = str(prob), org = (box[2], box[3]), 
                fontFace = fontFace, fontScale = fontScale, 
                    color = color)
    return img

def draw_land(img, landmarks, radius = 2, color = (255, 0, 0)):
    for ld in landmarks:
        for land_x, land_y in ld:
                cv2.circle(img = img, center=(land_x, land_y), radius = radius, 
                    color = color)
    return img

def draw_id(img, faceids, boxes, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (0, 241, 245)):
    for idi, box in zip(faceids, boxes):
        cv2.putText(img = img, text = str(idi), org = (box[2], box[1]), 
                fontFace = fontFace, fontScale = fontScale, 
                    color = color)
    return img

def notate(img, boxes, landmarks = None, probs = None, faceids = None, thickness = 2, fontScale = 0.3, fontColor = (0, 241, 245)):
    img = draw_boxes(img, boxes, thickness = thickness)
    if not landmarks is None:
        img = draw_land(img, landmarks)
    if not probs is None:
        img = draw_prob(img, probs, boxes, fontScale = fontScale, color = fontColor)
    if not faceids is None:
        img = draw_id(img, faceids, boxes, fontScale = fontScale, color = fontColor)
        
    return img
'''