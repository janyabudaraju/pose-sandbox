# pose sandbox

## overview

this project aims to allow for real-time pose detection using various models and video input. primary functionality includes webcam integration, video recording, and inference of poses using different models.

the webcam component triggers a download of the `.webm` recorded video and the associated json, with timestamps and inferences for each frame. the video upload component prints the json with timestamps & inferences to console, which can be copied and pasted to a document of choice.

## Features

- **webcam integration**: can capture live video from the webcam and perform pose detection in real-time.
- **video upload**: allows for upload of video files on which to perform pose detection analysis.
- **model selection**: currently supports inference with 1+ of the BlazePose, PoseNet, and MoveNet models.
- **data visualization**: along with the recorded inference data, poses can be tracked visually through plotted points, color coded per model. 
- **FPS display**: webcam and video both display the frames per second (FPS) of the video feed.

## dependencies

- **TypeScript**
- **npm**
- **Node.js** 
- **Chakra UI**
- **TensorFlow.js**
