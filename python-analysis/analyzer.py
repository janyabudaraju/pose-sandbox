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
    zeroed = False
    max_fn = -1

    for entry in parsed:
        for pred in entry:
            frame_num = pred['frameIdx']
            
            if frame_num == 0:
                zeroed = True
            if not zeroed:
                continue
            
            if frame_num > max_fn:
                max_fn = frame_num

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

            if not frame_num in formatted: # create empty dict for first instance of frame
                formatted[frame_num] = {}
            if not model_id in formatted[frame_num]:
                formatted[frame_num][model_id] = pose_items
    return formatted, max_fn

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

    cap.release()

def get_conversion_factor(vidpath, max_json_frame):
    cap = cv.VideoCapture(vidpath)
    max_vid_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return max_vid_frame / max_json_frame

def get_data_for_frame(data, fnum, model_id):
    all_fnums = list(data.keys())
    closest_fnum = min(all_fnums, key=lambda x: abs(x - fnum))
    print(f'[LOGGING: get_data_for_frame] requested: {fnum} closest: {closest_fnum}')
    if model_id not in data[closest_fnum]:
        return None
    poses = data[closest_fnum][model_id]
    return poses[0].kps
    
def get_data_for_adj_frame(data, fnum, model_id, cf):
    fnum_adj = fnum / cf
    # print(f'[LOGGING: get_data_for_adj_frame] original: {fnum} adjusted: {fnum_adj}')
    return get_data_for_frame(data, fnum_adj, model_id)

def get_frame_from_fnum(vidpath, fnum):
    cap = cv.VideoCapture(vidpath)
    cap.set(cv.CAP_PROP_POS_FRAMES, fnum-1)
    _, frame = cap.read()
    return frame

def draw_pose_on_frame(frame, kps):
    for kp in kps:
        # print(f'[LOGGING: draw_pose_on_frame] xpos: {kp.x} ypos: {kp.y}')
        cv.circle(frame, (int(kp.x), int(kp.y)), 1, (0, 0, 255), -1)
    return frame

if __name__ == '__main__': 
    vid_path = 'python-analysis/data/raw/video.mp4'
    json_path = 'python-analysis/data/raw/data.json'

    if not vid_path.endswith('.mp4'):
        new_path = Path(vid_path).with_suffix('.mp4')
        convert_webm_to_mp4(vid_path, new_path)
        vid_path = new_path
    
    data, max_jf = clean_dict_from_JSON(json_path)
    frame_cf = get_conversion_factor(vid_path, max_json_frame=max_jf)

    fn = 55
    model = 'movenet'
    kps = get_data_for_adj_frame(data, fnum = fn, model_id=model, cf=frame_cf)

    if kps is None:
        print("no points found")
    else:
        frame = get_frame_from_fnum(vid_path, fn)
        augmntd = draw_pose_on_frame(frame, kps)

        cv.imshow('with points', augmntd)
        cv.waitKey(0)

# TODO: validate predictions (if this doesn't work try converting from seconds to frames using fps, maybe)
# TODO: add connections between points