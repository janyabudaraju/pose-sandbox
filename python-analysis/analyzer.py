from definitions import Pose, Pose3D, KP2D, KP3D
import json
import cv2 as cv
import subprocess
from pathlib import Path

def convert_webm_to_mp4(webm_path, mp4_path):
    command = [
        'ffmpeg',
        '-i', webm_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-c:a', 'aac',
        mp4_path
    ]
    subprocess.run(command, check=True)

def clean_dict_from_JSON(filepath: str):
    with open(filepath, 'r') as f:
        parsed = json.load(f)

    formatted = {}

    for entry in parsed:
        for pred in entry:
            timestamp = pred['frameIdx']
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
    return formatted

def get_timestamp_from_frame(vidpath):
    cap = cv.VideoCapture(vidpath)

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"total duration: {duration}s")
    print(f"FPS: {fps}")
    print(f"total number of frames: {frame_count}")


    frame_no = 0
    while(cap.isOpened()):
        frame_exists, _ = cap.read()
        if frame_exists:
            print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv.CAP_PROP_POS_MSEC)))
            frame_no += 1
        else:
            break
    frame_no += 1

    cap.release()

def get_conversion_factor(vidpath, max_json_frame):
    cap = cv.VideoCapture(vidpath)
    max_vid_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return max_vid_frame / max_json_frame

if __name__ == '__main__': 
    vid_path = ''
    json_path = ''

    if not vid_path.endswith('.mp4'):
        new_path = Path(vid_path).stem + '.mp4'
        convert_webm_to_mp4(vid_path, new_path)
        vid_path = new_path
    
    data, max_jf = clean_dict_from_JSON(json_path)
    frame_cf = get_conversion_factor(vid_path, max_json_frame=max_jf)
    
    

# convert_webm_to_mp4('python-analysis/data/raw/video.webm', 'python-analysis/data/raw/output.mp4')
# data = clean_dict_from_JSON('python-analysis/data/raw/data.json')
# print(data.keys())
# get_timestamp_from_frame('python-analysis/data/raw/output.mp4')

# TODO: apply scaling to frames IN THE json read function
# TODO: auto determine whether conversion needs to occur by checking the extension
# TODO: validate predictions
# TODO: if this doesn't work try converting from seconds to frames using fps, maybe?