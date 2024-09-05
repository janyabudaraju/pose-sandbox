import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import definitions as defs

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

def run_inference_tflite(interpreter, img, dtype, shape=257):
    input_image = cv.resize(img, (shape, shape))
    input_image = tf.cast(input_image, dtype=dtype)
    input_image = tf.expand_dims(input_image, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    kps = interpreter.get_tensor(output_details[0]['index'])
    return kps

def process_posenet(map, threshold=0.5):
    keypoints = []
    for idx in range(map.shape[-1]):
        y, x = np.unravel_index(np.argmax(map[..., idx]), map.shape[:2])
        conf = map[y, x, idx]
        if conf > threshold:
            kp = defs.KP2D(idx, y, x, conf, defs.KP_DICT_33[idx])
            keypoints.append(kp)
    return keypoints


def movenet(interpreter, img):
    output = run_inference_tflite(interpreter, img, dtype = defs.MOVENET_DTYPE, shape=defs.MOVENET_SHAPE)
    return output

def posenet(interpreter, img):    
    output = run_inference_tflite(interpreter, img, dtype = defs.POSENET_DTYPE, shape=defs.POSENET_SHAPE)
    return output

def blazepose(model, img):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    output = model.detect(mp_img)
    return output.pose_landmarks


model = load_blazepose()
im = cv.imread('python-analysis/data/raw/test.jpg', cv.IMREAD_COLOR)
output = blazepose(model, im)
print(output.shape)