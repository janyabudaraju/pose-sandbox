import parser
import definitions as defs
import numpy as np
from typing import Tuple

LENGTH_CHECKS = [('left_shoulder', 'left_hip'), 
                ('right_shoulder', 'right_hip'),
                ('left_shoulder', 'right_shoulder'), 
                ('left_hip', 'right_hip'),
                ('left_shoulder', 'right_knee'), 
                ('right_shoulder', 'left_knee'),
                ('left_shoulder', 'right_knee'), 
                ('right_shoulder', 'left_knee'),
                ('left_shoulder', 'left_knee'), 
                ('right_shoulder', 'right_knee')]
ANGLE_CHECKS = [('left_hip', 'left_shoulder', 'right_shoulder'), 
                ('left_shoulder', 'right_shoulder', 'right_hip'),
                ('left_shoulder', 'left_hip', 'right_hip'),
                ('left_hip', 'right_hip', 'right_shoulder')]
PRESENCE_CHECKS = ['left_eye', 'right_eye', 'nose', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

def get_length(p1: defs.KP2D, p2: defs.KP2D):
    """
    function to calculate the Euclidean distance between two 2D keypoints.

    params:
        p1 (defs.KP2D): first keypoint, with coordinates stored in `coords`.
        p2 (defs.KP2D): second keypoint, with coordinates stored in `coords`.

    returns:
        float: Euclidean distance between the two keypoints.
    """
    assert len(p1.coords) == len(p2.coords)
    dist_sum = 0
    for i in range(len(p1.coords)):
        dist_sum += (p1.coords[i] - p2.coords[i])**2
    return dist_sum**.5

def get_angle(seg1: Tuple[defs.KP2D, defs.KP2D], seg2: Tuple[defs.KP2D, defs.KP2D]):
    """
    calculate angle between two segments, defined by pairs of 2D keypoints.

    Args:
        seg1 (Tuple[defs.KP2D, defs.KP2D]): first segment, defined by two keypoints.
        seg2 (Tuple[defs.KP2D, defs.KP2D]): second segment, defined by two keypoints.

    Returns:
        Optional[float]: angle between the two segments in radians. `None` if either segment has 0 length.
    """

    assert len(seg1[0].coords) == len(seg1[1].coords)
    assert len(seg1[0].coords) == len(seg2[0].coords)
    assert len(seg2[0].coords) == len(seg2[1].coords)

    vec1 = np.array([seg1[1].coords[i] - seg1[0].coords[i] for i in range(len(seg1[0].coords))])
    vec2 = np.array([seg2[1].coords[i] - seg2[0].coords[i] for i in range(len(seg2[0].coords))])

    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return None
    
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

def get_all_lengths(kps):
    """
    calculates the lengths of predefined segments (defined in LENGTH_CHECKS) between keypoints.

    params:
        kps (Dict[str, defs.KP2D]): dict of keypoints, with labels as keys.

    returns:
        List[float]: list of lengths for each segment defined in `LENGTH_CHECKS`.
                    if keypoint is missing, corresponding length for segment is set to -1.
    """

    if not kps:
        return [-1] * len(LENGTH_CHECKS)
    lengths = []
    for seg in LENGTH_CHECKS:
        key1, key2 = seg
        if key1 not in kps or key2 not in kps:
            lengths.append(-1)
        else:
            p1 = kps[key1]
            p2 = kps[key2]
            lengths.append(get_length(p1, p2))
    return lengths

def get_all_angles(kps):
    """
    calculates the angles between predefined keypoint triplets (defined in ANGLE_CHECKS).

    params:
        kps (Dict[str, defs.KP2D]): dict of keypoints, with labels as keys.

    returns:
        List[float]: list of angles for each triplet defined in `ANGLE_CHECKS`.
                    if keypoint is missing, corresponding angle for triplet is set to -1.
    """
    if not kps:
        return [-1] * len(ANGLE_CHECKS)
    angles = []
    for triad in ANGLE_CHECKS:
        key1, key2, key3 = triad
        if key1 not in kps or key2 not in kps or key3 not in kps:
            angles.append(-1)
        else:
            beg = kps[key1]
            mid = kps[key2]
            end = kps[key3]
            angles.append(get_angle((beg, mid), (mid, end)))
    return angles

def check_presences(kps, conf_thresh=0.6):
    """
    check 'presence' of keypoints based on their confidence scores.

    Args:
        kps (Dict[str, defs.KP2D]): A dictionary of keypoints with their labels as keys.
        conf_thresh (float, optional): The confidence threshold for considering a keypoint as present. Defaults to 0.6.

    Returns:
        List[int]: 1 for present keypoint, 0 for absent keypoint. order defined in `PRESENCE_CHECKS`.
                if a keypoint is missing or its confidence is below the threshold, presence set to 0.
    """
    if not kps:
        return [-1] * len(PRESENCE_CHECKS)
    presences = []
    for key in PRESENCE_CHECKS:
        if key not in kps:
            presences.append(0)
            continue
        if(kps[key].prob > conf_thresh):
            presences.append(1)
        presences.append(0)
    return presences

if __name__ == '__main__': 
    json_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/inference_data_2024-07-16T17-35-11-669Z.json'
    data, _ = parser.clean_dict_from_JSON(json_path)
    pose = parser.get_data_at_time(data, timestamp=4.5, model_id='movenet')
    angles = get_all_angles(pose)
    print("ANGLES")
    print(angles)
    print("LENGTHS")
    lengths = get_all_lengths(pose)
    print(lengths)
    print("VISIBILITY")
    presences = check_presences(pose)
    print(presences)


