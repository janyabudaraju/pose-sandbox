import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp
from mediapipe.tasks import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core import BaseOptions

def load_posenet():
    model = hub.load("https://tfhub.dev/google/posenet/1")
    return model

def load_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model

def load_blazepose(model_path = 'python-analysis/models/pose_landmarker_full.task'):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE)
    landmarker = PoseLandmarker.create_from_options(options)
    return landmarker

