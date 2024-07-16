import tensorflow as tf
from typing import List, Tuple
import cv2 as cv
from math import sqrt, atan2, degrees

NUM_COCO_KPS = 17
POSENET_SHAPE = 257
MOVENET_SHAPE = 192
POSENET_DTYPE = tf.float32
MOVENET_DTYPE = tf.uint8

KP_DICT_17 = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2, 
    'left_ear': 3, 
    'right_ear': 4, 
    'left_shoulder': 5, 
    'right_shoulder': 6, 
    'left_elbow': 7, 
    'right_elbow': 8, 
    'left_wrist': 9, 
    'right_wrist': 10, 
    'left_hip': 11, 
    'right_hip': 12, 
    'left_knee': 13, 
    'right_knee': 14, 
    'left_ankle': 15, 
    'right_ankle': 16 
}

SKELETON_17_KPS = [
    (0, 1),  # nose to left_eye
    (0, 2),  # nose to right_eye
    (1, 3),  # left_eye to left_ear
    (2, 4),  # right_eye to right_ear
    (5, 6),  # left_shoulder to right_shoulder
    (5, 7),  # left_shoulder to left_elbow
    (7, 9),  # left_elbow to left_wrist
    (6, 8),  # right_shoulder to right_elbow
    (8, 10), # right_elbow to right_wrist
    (5, 11), # left_shoulder to left_hip
    (6, 12), # right_shoulder to right_hip
    (11, 12),# left_hip to right_hip
    (11, 13),# left_hip to left_knee
    (13, 15),# left_knee to left_ankle
    (12, 14),# right_hip to right_knee
    (14, 16) # right_knee to right_ankle
]

KP_DICT_33 = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3, 
    "right_eye_inner": 4, 
    "right_eye": 5, 
    "right_eye_outer": 6, 
    "left_ear": 7, 
    "right_ear": 8, 
    "mouth_left": 9, 
    "mouth_right": 10,
    "left_shoulder": 11, 
    "right_shoulder": 12, 
    "left_elbow": 13, 
    "right_elbow": 14, 
    "left_wrist": 15, 
    "right_wrist": 16, 
    "left_pinky": 17, 
    "right_pinky": 18, 
    "left_index": 19, 
    "right_index": 20, 
    "left_thumb": 21, 
    "right_thumb": 22, 
    "left_hip": 23, 
    "right_hip": 24, 
    "left_knee": 25, 
    "right_knee": 26, 
    "left_ankle": 27, 
    "right_ankle": 28, 
    "left_heel": 29, 
    "right_heel": 30, 
    "left_foot_index": 31, 
    "right_foot_index": 32
}

SKELETON_33_KPS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), 
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), 
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

class KP2D:
    def __init__(self, x, y, score, name):
        self.x = x
        self.y = y
        self.prob = score
        self.name = name

class KP3D:
    def __init__(self, x, y, z, score, name):
        self.x = x
        self.y = y
        self.z = z
        self.prob = score
        self.name = name

class Pose:
    def __init__(self, score: float, kps: dict[str, KP2D]):
        self.score = score
        self.kps = kps
    
    def draw_on_frame(self, frame):
        for kp in self.kps:
            cv.circle(frame, (int(kp.x), int(kp.y)), 5, (0, 255, 0), -1)
        return frame
    
    def get_dist_btw(self, kp1_name, kp2_name):
        kp1 = self.kps[kp1_name]
        kp2 = self.kps[kp2_name]
        return sqrt((kp1.x - kp2.x)^2 + (kp1.y - kp2.y)^2)
    
    def get_key_dists(self, cxns: List[Tuple[str, str]]):
        dists = []
        for c in cxns:
            dists.append(self.get_dist_btw(c[0], c[1]))
        return dists
    
    def get_angle_between(self, kp1_name: str, kp2_name: str) -> float:
        kp1 = self.kps[kp1_name]
        kp2 = self.kps[kp2_name]
        delta_x = kp2.x - kp1.x
        delta_y = kp2.y - kp1.y
        angle = atan2(delta_y, delta_x)
        return degrees(angle)

class Pose3D(Pose):
    def __init__(self, score: float, kps: dict[str, KP2D], kps3d: dict[str, KP3D]):
        self.kps3d = kps3d
        super().__init__(score, kps) 