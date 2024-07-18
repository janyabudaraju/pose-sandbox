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

def get_frame_from_fnum(vidpath, fnum):
    cap = cv.VideoCapture(vidpath)
    cap.set(cv.CAP_PROP_POS_FRAMES, fnum-1)
    _, frame = cap.read()
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

def get_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_conversion_factor(vidpath, max_json_frame):
    cap = cv.VideoCapture(vidpath)
    max_vid_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'[LOGGING: get_conversion_factor] max vid frame: {max_vid_frame} max json frame: {max_json_frame}')
    return max_json_frame / max_vid_frame