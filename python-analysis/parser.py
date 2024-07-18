from definitions import Pose, Pose3D, KP2D, KP3D
import json
import cv2 as cv
from pathlib import Path
import video_utils as vidutils

def clean_dict_from_JSON(filepath: str):
    with open(filepath, 'r') as f:
        parsed = json.load(f)

    formatted = {}
    # max_fn = -1
    max_ts = -1

    for entry in parsed:
        for pred in entry:
            # frame_num = pred['frameIdx']
            timestamp = pred['timeStamp']
            
            # if frame_num > max_fn:
            #     max_fn = frame_num
            if timestamp > max_ts:
                max_ts = timestamp

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

            if not timestamp in formatted: # create empty dict for first instance of frame
                formatted[timestamp] = {}
            if not model_id in formatted[timestamp]:
                formatted[timestamp][model_id] = pose_items
    
    # print(json.dumps(parsed, indent=4))
    # with open('output.json', 'w') as f:
    #     json.dump(parsed, f, indent=4)
    return formatted, max_ts

def get_data_for_frame(data, timestamp, model_id):
    all_ts = list(data.keys())
    closest_timestamp = min(all_ts, key=lambda x: abs(x - timestamp))
    print(f'[LOGGING]: get_data_for_frame] requested: {timestamp} closest: {closest_timestamp}')
    if model_id not in data[closest_timestamp]:
        return None
    poses = data[closest_timestamp][model_id]
    if abs(closest_timestamp - timestamp) > 3:
        print(f'[FLAG] diff between requested and returned frames is unexpectedly large.')
    return poses[0].kps

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
            # start_pos = (int(start_kp.x), int(start_kp.y))
            # end_pos = (int(end_kp.x), int(end_kp.y))
            # cv.line(frame, start_pos, end_pos, (0, 255, 0), 2)
            
    return frame


if __name__ == '__main__': 
    vid_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/recorded_video_0701.mp4'
    json_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/inference_data_2024-07-16T17-35-11-669Z.json'

    if vid_path.endswith('.webm'):
        new_path = Path(vid_path).with_suffix('.mp4')
        vidutils.convert_webm_to_mp4(vid_path, new_path)
        vid_path = new_path
    
    data, max_jf, max_jts = clean_dict_from_JSON(json_path)
    vid_dur = vidutils.get_length(vid_path)
    print(f"[LOGGING] json recorded duration: {max_jts} | video_duration: {vid_dur}")


    # max_vf = vidutils.frame_count(vid_path)
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