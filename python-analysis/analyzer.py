from definitions import Pose, Pose3D, KP2D, KP3D
import json

def convert_JSON_to_numpy(filepath: str):
    with open(filepath, 'r') as f:
        parsed = json.load(f)

    formatted = {}

    for entry in parsed:
        for pred in entry:
            timestamp = pred['timestamp']        
            model_id = pred['modelId']
            pose_data = pred['poseData']
            pose_items = []
            cur_pose = None

            for pose in pose_data:
                kp_items = []
                overall_score = pose['score']
                for kp in pose['keypoints']:
                    kp_items.append(KP2D(x = kp['x'], y = kp['y'], score = kp['score'], name = kp['name']))
                
                if 'keypoints3D' in pose:
                    kp3d_items = []
                    for kp3d in pose['keypoints3D']:
                        kp3d_items.append(KP3D(x = kp3d['x'], y = kp3d['y'], z = kp3d['z'], score = kp3d['score'], name = kp3d['name']))
                    cur_pose = Pose3D(overall_score, kp_items, kp3d_items)
                else:
                    cur_pose = Pose(overall_score, kp_items)
                pose_items.append(cur_pose)

            if not timestamp in formatted:
                formatted[timestamp] = {}
            formatted[timestamp][model_id] = pose_items


convert_JSON_to_numpy('python-analysis/data/raw/inference_data.json')
