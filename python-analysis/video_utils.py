import cv2 as cv
import subprocess

def convert_webm_to_mp4(webm_path, mp4_path):
    """
    convert a video file from webm format to mp4 format using ffmpeg.

    params:
        webm_path (str): path to the input webm file.
        mp4_path (str): path where the output mp4 file will be saved.

    """

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
    """
    retrieve a specific frame from a video file.

    params:
        vidpath (str): path to the video file.
        fnum (int): the frame number to retrieve (1-based index).

    returns:
        frame: the frame corresponding to the specified frame number.
    """

    cap = cv.VideoCapture(vidpath)
    cap.set(cv.CAP_PROP_POS_FRAMES, fnum-1)
    _, frame = cap.read()
    return frame

def frame_count(video_path, manual=True):
    """
    count the total number of frames in a video file.

    params:
        video_path (str): path to the video file.
        manual (bool, optional): whether to count frames manually (default is True). If False, 
        uses built-in property.

    returns:
        int: the total number of frames in the video.
    """
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
    """
    get the duration of a video file in seconds using ffprobe.

    params:
        filename (str): path to the video file.

    returns:
        float: the duration of the video in seconds.
    """

    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_conversion_factor(vidpath, max_json_frame):
    """
    calculate the conversion factor between JSON frame numbers and video frame numbers.

    args:
        vidpath (str): path to the video file.
        max_json_frame (int): maximum frame number from the JSON data.

    returns:
        float: the conversion factor from video frames to JSON frames.
    """
    cap = cv.VideoCapture(vidpath)
    max_vid_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'[LOGGING: get_conversion_factor] max vid frame: {max_vid_frame} max json frame: {max_json_frame}')
    return max_json_frame / max_vid_frame