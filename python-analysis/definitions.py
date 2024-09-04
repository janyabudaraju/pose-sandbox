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

SKELETON_33_KPS = [
    (0, 1),     # nose to left eye inner
    (1, 2),     # left eye inner to left eye
    (2, 3),     # left eye to left eye outer
    (3, 7),     # left eye outer to left ear
    (0, 4),     # nose to right eye inner
    (4, 5),     # right eye inner to right eye
    (5, 6),     # right eye to right eye outer
    (6, 8),     # right eye outer to right ear
    (9, 10),    # left shoulder to right shoulder
    (11, 12),   # left hip to right hip
    (11, 13),   # left hip to left knee
    (13, 15),   # left knee to left ankle
    (15, 17),   # left ankle to left heel
    (15, 19),   # left ankle to left foot index
    (15, 21),   # left ankle to left foot small toe
    (17, 19),   # left heel to left foot index
    (12, 14),   # right hip to right knee
    (14, 16),   # right knee to right ankle
    (16, 18),   # right ankle to right heel
    (16, 20),   # right ankle to right foot index
    (16, 22),   # right ankle to right foot small toe
    (18, 20),   # right heel to right foot index
    (11, 23),   # left hip to left shoulder
    (12, 24),   # right hip to right shoulder
    (23, 24),   # left shoulder to right shoulder
    (23, 25),   # left shoulder to left elbow
    (24, 26),   # right shoulder to right elbow
    (25, 27),   # left elbow to left wrist
    (26, 28),   # right elbow to right wrist
    (27, 29),   # left wrist to left hand index
    (28, 30),   # right wrist to right hand index
    (29, 31),   # left hand index to left hand pinky
    (30, 32),   # right hand index to right hand pinky
    (27, 31),   # left wrist to left hand pinky
    (28, 32)    # right wrist to right hand pinky
]

class KP2D:
    """
    2D keypoint object representation for a pose model keypoint prediction.

    attributes:
        coords (list[float]): the (x, y) coordinates of the keypoint.
        prob (float): confidence score of the keypoint.
        name (str): name of the keypoint.
    """

    def __init__(self, x, y, score, name):
        self.coords = [x, y]
        self.prob = score
        self.name = name

class KP3D(KP2D):
    """
    3D keypoint object representation for a pose model keypoint prediction.
    extends KP2D to avoid attribute replication.

    attributes:
        coords (list[float]): the (x, y, z) coordinates of the keypoint.
    """
    def __init__(self, x, y, z, score, name):
        super().__init__(x, y, score, name)
        self.coords.append(z)

class Pose:
    """
    class representing a 2D prediction derived from a pose model.

    attributes:
        score (float): overall confidence score of the pose.
        kps (dict[str, KP2D]): dictionary of keypoint names to KP2D objects.
    """

    def __init__(self, score: float, kps: dict[str, KP2D]):
        self.score = score
        self.kps = kps
    
    def draw_on_frame(self, frame):
        """
        function to draw keypoints on a given video frame. draws only in 2 dimensions.

        params:
            frame: the video frame on which to draw the keypoints.

        returns:
            frame: the frame with keypoints drawn on it.
        """
        for kp in self.kps:
            cv.circle(frame, (int(kp.coords[0]), int(kp.coords[1])), 5, (0, 255, 0), -1)
        return frame
    
    def get_dist_btw(self, kp1_name, kp2_name):
        """
        function to calculate the euclidean distance between two KP objects.
        Both must be either KP2D or KP3D (same number of coordinate indices)

        params:
            kp1_name (str): name of the first keypoint.
            kp2_name (str): name of the second keypoint.

        returns:
            float: the euclidean distance between the two keypoints.
        """

        kp1 = self.kps[kp1_name]
        kp2 = self.kps[kp2_name]

        assert len(kp1.coords) == len(kp2.coords), "KPs must have the same dimension"
        sum = 0
        for i in range(len(kp1.coords)):
            sum += (kp1.coords[i] - kp2.coords[i])**2
        return sqrt(sum)
    
    def get_key_dists(self, cxns: List[Tuple[str, str]]):
        """
        function to calculate the euclidean distances between pairs of keypoints.

        params:
            cxns (List[tuple[str, str]]): list of tuples, each containing the names of two keypoints.

        returns:
            List[float]: list of euclidean distances between each pair of keypoints.
        """

        dists = []
        for c in cxns:
            dists.append(self.get_dist_btw(c[0], c[1]))
        return dists
    
    def get_angle_between(self, kp1_name: str, kp2_name: str) -> float:
        """
        calculate the angle between two keypoints with respect to the horizontal axis.
        note that for 3D keypoints, only the first two coordinate indices are used to determine
        the angle with the horizontal axis.

        params:
            kp1_name (str): name of the first keypoint.
            kp2_name (str): name of the second keypoint.

        returns:
            float: angle in degrees between the line connecting the keypoints and the horizontal axis.
        """

        kp1 = self.kps[kp1_name]
        kp2 = self.kps[kp2_name]
        
        delta_x = kp2.coords[0] - kp1.coords[0]
        delta_y = kp2.coords[1] - kp1.coords[1]

        angle = atan2(delta_y, delta_x)
        return degrees(angle)

class Pose3D(Pose):
    """
    class representing a 3D prediction derived from a pose model.
    inherits from Pose2D to avoid attribute duplication.

    attributes:
        score (float): overall confidence score of the pose.
        kps (dict[str, KP2D]): dictionary of keypoint names to KP2D objects.
        kps3d (dict[str, KP3D]): dictionary of keypoint names to KP3D objects.
    """

    def __init__(self, score: float, kps: dict[str, KP2D], kps3d: dict[str, KP3D]):
        self.kps3d = kps3d
        super().__init__(score, kps) 