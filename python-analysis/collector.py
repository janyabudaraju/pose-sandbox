import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

def load_posenet():
    interpreter = tf.lite.Interpreter(model_path="python-analysis/models/posenet.tflite")
    interpreter.allocate_tensors()
    return interpreter

def load_movenet():
    interpreter = tf.lite.Interpreter(model_path='python-analysis/models/movenet-lightning.tflite')
    interpreter.allocate_tensors()
    return interpreter

def load_blazepose(model_path = 'python-analysis/models/blazepose.task'):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE)
    model = PoseLandmarker.create_from_options(options)
    return model

def run_inference_interpreter(interpreter, img, dtype=tf.uint8, shape=257):
    input_image = cv.resize(img, (shape, shape))
    input_image = tf.cast(input_image, dtype=dtype)
    input_image = tf.expand_dims(input_image, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    kps = interpreter.get_tensor(output_details[0]['index'])
    # TODO: convert to standard form
    return kps

def run_inference_mp(model, img):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    output = model.detect(mp_img)
    # TODO: convert to standard form
    return output

POSENET_SHAPE = 257
MOVENET_SHAPE = 192
POSENET_DTYPE = tf.float32
MOVENET_DTYPE = tf.uint8

model = load_posenet()
im = cv.imread('python-analysis/data/raw/test.jpg', cv.IMREAD_COLOR)
op = run_inference_interpreter(model, im, dtype=tf.float32)