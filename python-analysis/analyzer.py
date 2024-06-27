from definitions import Pose, Pose3D, KP2D, KP3D
import json
import cv2 as cv
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
    return formatted

def get_timestamp_from_frame(vidpath):
    cap = cv.VideoCapture(vidpath)

    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"total duration: {duration}s")
    print(f"FPS: {fps}")
    print(f"total number of frames: {frame_count}")

    # frame_index = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break  # Break the loop if there are no frames to read
        
    #     # Calculate the timestamp for the current frame
    #     timestamp = frame_index / fps
    #     print(f"Timestamp for frame {frame_index}: {timestamp}s")
        
    #     # Here you can add code to process the frame if necessary
    #     # For example, displaying the frame:
    #     # cv2.imshow('Frame', frame)
    #     # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     #     break

    #     frame_index += 1

    # cap.release()
    # cv2.destroyAllWindows()

# convert_webm_to_mp4('python-analysis/data/raw/newvid.webm', 'python-analysis/data/raw/output.mp4')
data = clean_dict_from_JSON('python-analysis/data/raw/inference_data_0627.json')
print(data.keys())
get_timestamp_from_frame('python-analysis/data/raw/output.mp4')