from definitions import Pose, Pose3D, KP2D, KP3D
import definitions as defs
import json
import cv2 as cv
import subprocess
from pathlib import Path
import subprocess

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
    prior_fn = -1
    max_ts = -1
    min_ts = -1

    for entry in parsed:
        for pred in entry:
            frame_num = pred['frameIdx']
            timestamp = pred['timeStamp']

            # if frame_num == 0:
            #     zeroed = True
            # if not zeroed:
            #     continue
                        
            # if frame_num > prior_fn + 1:
            #     continue
            
            if frame_num > max_fn:
                max_fn = frame_num
            if timestamp > max_ts:
                max_ts = timestamp
            if timestamp < min_ts:
                min_ts = timestamp

            prior_fn = frame_num

            model_id = pred['modelId']
            pose_data = pred['poseData']
            pose_items = []
            cur_pose = None

            for pose in pose_data:
                kp_items = {}
                overall_score = pose['score']
                for kp in pose['keypoints']:
                    name = kp['name']
                    kp_items[name] = KP2D(x = kp['x'], y = kp['y'], score = kp['score'], name = kp['name'])
                
                if 'keypoints3D' in pose:
                    kp3d_items = {}
                    for kp3d in pose['keypoints3D']:
                        name = kp3d['name']
                        kp3d_items[name] = KP3D(x = kp3d['x'], y = kp3d['y'], z = kp3d['z'], score = kp3d['score'], name = kp3d['name'])
                    cur_pose = Pose3D(overall_score, kp_items, kp3d_items)
                else:
                    cur_pose = Pose(overall_score, kp_items)
                pose_items.append(cur_pose)

            if not frame_num in formatted: # create empty dict for first instance of frame
                formatted[frame_num] = {}
            if not model_id in formatted[frame_num]:
                formatted[frame_num][model_id] = pose_items
    
    # print(json.dumps(parsed, indent=4))
    # with open('output.json', 'w') as f:
    #     json.dump(parsed, f, indent=4)
    return formatted, max_fn, max_ts, min_ts

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
    print(f'[LOGGING: get_conversion_factor] max vid frame: {max_vid_frame} max json frame: {max_json_frame}')
    return max_json_frame / max_vid_frame

def get_data_for_frame(data, fnum, model_id):
    all_fnums = list(data.keys())
    closest_fnum = min(all_fnums, key=lambda x: abs(x - fnum))
    print(f'[LOGGING: get_data_for_frame] requested: {fnum} closest: {closest_fnum}')
    if model_id not in data[closest_fnum]:
        return None
    poses = data[closest_fnum][model_id]

    if abs(closest_fnum - fnum) > 10:
        print(f'[FLAG] diff between requested and returned frames is unexpectedly large. requested: {fnum} returned: {closest_fnum}')

    return poses[0].kps

def get_data_for_adj_frame(data, fnum, model_id, cf):
    fnum_adj = fnum * cf
    # print(f'[LOGGING: get_data_for_adj_frame] original: {fnum} adjusted: {fnum_adj}')
    return get_data_for_frame(data, fnum_adj, model_id)

def get_frame_from_fnum(vidpath, fnum):
    cap = cv.VideoCapture(vidpath)
    cap.set(cv.CAP_PROP_POS_FRAMES, fnum-1)
    _, frame = cap.read()
    return frame

def draw_pose_on_frame(frame, kps, kp_mapping=None, skeleton_list=None):
    for kp in kps.values():
        # print(f'[LOGGING: draw_pose_on_frame] xpos: {kp.x} ypos: {kp.y}')
        cv.circle(frame, (int(kp.x), int(kp.y)), 1, (0, 0, 255), -1)

    if not skeleton_list or not kp_mapping:
        return None
    
    for start, end in skeleton_list:
        if start in kps and end in kps:
            start_kp = kps[start]
            end_kp = kps[end]
            start_pos = (int(start_kp.x), int(start_kp.y))
            end_pos = (int(end_kp.x), int(end_kp.y))
            cv.line(frame, start_pos, end_pos, (0, 255, 0), 2)
            
    return frame

def frame_count(video_path, manual=True):
    def manual_count(handler):
        frames = 0
        while True:
            status, _ = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv.VideoCapture(video_path)
    if manual:
        frames = manual_count(cap)
    else:
        try:
            frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

if __name__ == '__main__': 
    vid_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/test_video.MOV'
    json_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/inference_data_2024-07-16T15-40-05-791Z.json'

    if vid_path.endswith('.webm'):
        new_path = Path(vid_path).with_suffix('.mp4')
        convert_webm_to_mp4(vid_path, new_path)
        vid_path = new_path
    
    data, max_jf, max_jts, min_jts = clean_dict_from_JSON(json_path)
    js_dur = max_jts - min_jts
    vid_dur = get_length(vid_path)

    print(f"json recorded duration: {js_dur} | video_duration: {vid_dur}")
    # max_vf = frame_count(vid_path)
    # print(f"max json frames: {max_jf} | max vid frames: {max_vf}")
    #frame_cf = get_conversion_factor(vid_path, max_json_frame=max_jf)
    #print(frame_cf)

    # fn = 500
    # model = 'blazepose'
    # kp_mapping = defs.KP_DICT_33
    # skeleton = defs.SKELETON_33_KPS

    # kps = get_data_for_adj_frame(data, fnum = fn, model_id=model, cf=frame_cf)

    # if kps is None:
    #     print("no points found")
    # else:
    #     frame = get_frame_from_fnum(vid_path, fn)
    #     augmntd = draw_pose_on_frame(frame, kps, kp_mapping, skeleton)

    #     cv.imshow('with points', augmntd)
    #     cv.waitKey(0)

# TODO: compute relevant metrics for each pose (functions are there, make tuple lists)
# TODO: video is still slightly out of sync, it seems. likely worth tracking down. need to ensure that videos are actually at 60fps

# might want to figure out how much actual memory is needed to perform algorithms for error/flagging
    # might not need the whole history. figure out how much is actually necessary
    # take videos for multiple labels (too far, too close, multiple people)-- * separately *
    # too far : ~15 feet
    # too close: full body not captured
# want to look at covariance table
# all-point euclidean distance and angles
# excel has a covariance feature-- "sensitivity report"
# top 10 measure: time series, angles, points, etc. at a 2D level
# support vector machines maybe?
# probably a voting system between some different measurements