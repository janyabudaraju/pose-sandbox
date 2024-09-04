from definitions import Pose, Pose3D, KP2D, KP3D
import json
import cv2 as cv
from pathlib import Path
import video_utils as vidutils

def clean_dict_from_JSON(filepath: str):
    """
    load and clean a json file, converting it into a dictionary of poses indexed by timestamp.

    params:
        filepath (str): path to the json file to be processed.

    returns:
        tuple: a tuple containing:
            - formatted (dict[int, dict[str, list[Pose]]]): dictionary where keys are timestamps and values are dictionaries
            mapping model ids to lists of pose objects.
            - max_ts (int): the maximum timestamp found in the json data.
    """

    # open json for parsing and processing
    with open(filepath, 'r') as f:
        parsed = json.load(f)

    formatted = {}
    max_ts = -1

    for entry in parsed:
        for pred in entry:
            timestamp = pred['timeStamp']
            
            # update max_ts if current timestamp is greater
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
                    # create KP2D object for each keypoint
                    kp_items[name] = KP2D(x = kp['x'], y = kp['y'], score = kp['score'], name = kp['name'])
                
                if 'keypoints3D' in pose:
                    kp3d_items = {}
                    for kp3d in pose['keypoints3D']:
                        name = kp3d['name']
                        # create KP3D object for each 3D keypoint
                        kp3d_items[name] = KP3D(x = kp3d['x'], y = kp3d['y'], z = kp3d['z'], score = kp3d['score'], name = kp3d['name'])
                    cur_pose = Pose3D(overall_score, kp_items, kp3d_items)
                else:
                    cur_pose = Pose(overall_score, kp_items)
                pose_items.append(cur_pose)

            # initialize empty dict for new timestamp
            if not timestamp in formatted:
                formatted[timestamp] = {}
            if not model_id in formatted[timestamp]:
                formatted[timestamp][model_id] = pose_items
    
    # optional dump into json
    # print(json.dumps(parsed, indent=4))
    # with open('output.json', 'w') as f:
    #     json.dump(parsed, f, indent=4)
    return formatted, max_ts

def get_data_at_time(data, timestamp, model_id):
    """
    retrieve keypoint data at the closest timestamp to the requested one.

    params:
        data (dict[int, dict[str, list[Pose]]]): the dictionary of poses indexed by timestamp.
        timestamp (int): the timestamp for which data is requested.
        model_id (str): the model id for which data is requested.

    returns:
        dict[str, KP2D] or None: a dictionary of keypoints if data is available at the closest timestamp,
        or none if the model id is not found.
    """

    all_ts = list(data.keys())
    # find closest timestamp to the requested one
    closest_timestamp = min(all_ts, key=lambda x: abs(x - timestamp))
    # print(f'[LOGGING]: get_data_for_frame] requested: {timestamp} closest: {closest_timestamp}')
    if model_id not in data[closest_timestamp]:
        return None
    poses = data[closest_timestamp][model_id]
    if abs(closest_timestamp - timestamp) > 3:
        print(f'[FLAG] diff between requested and returned frames is unexpectedly large.')
    return poses[0].kps

def draw_pose_on_frame(frame, kps, kp_mapping=None, skeleton_list=None):
    """
    draw keypoints and skeleton connections on a frame.

    params:
        frame (any): the frame (image) on which keypoints and skeletons will be drawn.
        kps (dict[str, KP2D]): dictionary of keypoints, where the key is the keypoint name 
        and the value is the KP2D object.
        kp_mapping (dict[str, str], optional): optional mapping of keypoint names to their 
        display names. Defaults to None.
        skeleton_list (list[tuple[str, str]], optional): list of tuples, each containing two 
        keypoint names to draw a line between. Defaults to None.

    returns:
        any: the frame with keypoints and skeleton connections drawn on it.
    """

    for kp in kps.values():
        # print(f'[LOGGING: draw_pose_on_frame] xpos: {kp.x} ypos: {kp.y}')
        cv.circle(frame, (int(kp.x), int(kp.y)), 1, (0, 0, 255), -1)

    if not skeleton_list or not kp_mapping:
        return None
    
    for start, end in skeleton_list:
        if start in kps and end in kps:
            start_kp = kps[start]
            end_kp = kps[end]
            # draw a line between connected keypoints
            start_pos = (int(start_kp.coords[0]), int(start_kp.coords[1]))
            end_pos = (int(end_kp.coords[0]), int(end_kp.coords[1]))
            cv.line(frame, start_pos, end_pos, (0, 255, 0), 2)
    return frame


if __name__ == '__main__': 
    vid_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/recorded_video_0701.mp4'
    json_path = '/Users/janyabudaraju/Desktop/curveassure/pose-sandbox/python-analysis/data/raw/inference_data_2024-07-16T17-35-11-669Z.json'

    if vid_path.endswith('.webm'):
        new_path = Path(vid_path).with_suffix('.mp4')
        vidutils.convert_webm_to_mp4(vid_path, new_path)
        vid_path = new_path
    
    data, max_jts = clean_dict_from_JSON(json_path)
    vid_dur = vidutils.get_duration(vid_path)
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