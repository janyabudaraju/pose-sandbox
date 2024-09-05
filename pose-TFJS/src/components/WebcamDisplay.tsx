import { useRef, useEffect, useCallback, useState } from "react";
import Webcam from "react-webcam";
import { Box, Button } from "@chakra-ui/react";
import { PoseModel, BasePose, Inference, modelOptions } from "../utils/ModelDefinitions";
import { drawKeypoints2D } from "../utils/utilities";

// props type for webcamdisplay, containing pose models to load
type Props = {
  models: PoseModel<BasePose>[];
};

function WebcamDisplay({models}: Props){

    // state to check if the model selection has changed
    const [isModelChanged, setIsModelChanged] = useState(true);

    // ref to keep track of currently loaded models
    const curModelRefs = useRef<PoseModel<BasePose>[]>([]);

    // state to store inference data for the current recording
    const [inferenceData, setInferenceData] = useState<Inference[][]>([]);

    // refs for the webcam, canvas, download functionality, and media recorder
    const camRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [recordedChunks, setRecordedChunks] = useState<BlobPart[]>([]);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);

    // ref and state for controlling the request animation frame loop and fps calculation
    const requestRef =  useRef<number | undefined>(undefined);
    const [fps, setFps] = useState(0);
    const frameCountFps = useRef(0);
    const frameCountIdx = useRef(0);
    const prevTime = useRef(Date.now());

    // function to run inference on a video frame and draw the corresponding keypoints.
    const estimatePose = useCallback(async () => {
      if (camRef.current?.video?.readyState === 4) {
        const video = camRef.current.video;
        const canvas = canvasRef.current;

        // update the video and canvas dimensions to match the webcam feed
        if(video.videoWidth != video.width || video.videoHeight != video.height) {
          video.width = video.videoWidth;
          video.height = video.videoHeight;
        }
        if (canvas && video && video.width > 0 && video.height > 0) {     
          if (canvas.width != video.width || canvas.height != video.height) {
            canvas.width = video.width;
            canvas.height = video.height;
          }

          // run inference across all models
          const modelInferences = await Promise.all(models.map(async (model) => {
            const poseData = await model.runInference(video);
            return { timeStamp: video.currentTime, frameIdx: frameCountIdx.current, modelId: model.id, poseData };
          }));

          // clear and draw keypoints for each model on the canvas
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, video.width, video.height);
            modelInferences.forEach(({ poseData, modelId }) => {

              const color = modelOptions.find(model => model.id === modelId)?.color
              if(color) {
                poseData.forEach(pose => {
                  drawKeypoints2D(pose.keypoints, 0.1, color, ctx);
                });
              }
            });
          }

          // update the inference data state to append the new frame of data
          setInferenceData(prevData => [...prevData, modelInferences]);
        }
      }
      // increment frame counters to update fps and request next animation frame
      frameCountFps.current += 1;
      frameCountIdx.current += 1;
      requestRef.current = requestAnimationFrame(estimatePose);
    }, [models]);

      // function to load and initialize models when the selection changes
      const runModel = useCallback(async () => {
        curModelRefs.current.forEach(model => model.dispose());
        curModelRefs.current = [];

        // load all models asynchronously
        await Promise.all(models.map(async (model) => {
          await model.load();
        }));
        curModelRefs.current = models;
        console.log("Models loaded.");

        // start pose estimation loop
        requestRef.current = requestAnimationFrame(() => estimatePose());
      }, [models, estimatePose]);

      // set the model changed flag when new models are passed
      useEffect(() => {
        setIsModelChanged(true);
      }, [models]);
      
      // hook to start the loop and clean up as necessary.
      useEffect(() => {
        if (isModelChanged) {
          setIsModelChanged(false);
    
          if (requestRef.current) {
            window.cancelAnimationFrame(requestRef.current);
          }
          runModel();
        }
      }, [isModelChanged, runModel]);

      // function to calculate and display fps every second
      useEffect(() => {
        const calculateFps = () => {
          const now = Date.now();
          const deltaTime = (now - prevTime.current) / 1000;
          const resultFps = frameCountFps.current / deltaTime;
          setFps(resultFps);
          frameCountFps.current = 0;
          prevTime.current = now;
        };
        // recalculate fps every second
        const interval = setInterval(calculateFps, 1000);
        return () => clearInterval(interval);
      }, []);

      const handleDataAvailable = useCallback(({ data }: BlobEvent) => {
        if (data.size > 0) {
            setRecordedChunks(prev => [...prev, data]);
        }
      }, []);

      // handle starting video recording and inference collection
      const handleStartRecording = useCallback(() => {
        if (camRef.current && camRef.current.video) {
          //reset fps counters / inference data list
          setInferenceData([]);
          frameCountIdx.current = 0;

          // handle video recording
          const videoStream: MediaStream = camRef.current.video.srcObject as MediaStream;
          mediaRecorderRef.current = new MediaRecorder(videoStream, { mimeType: 'video/webm' });
          mediaRecorderRef.current.addEventListener('dataavailable', handleDataAvailable);
          mediaRecorderRef.current.start();
        }
      }, [handleDataAvailable]);

      // trigger download of a desired file (inference data in this case)
      const downloadFile = (url: string, filename: string) => {
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    };

    // handle stopping video recording and saving the video and inference data
    const handleStopRecording = useCallback(() => {
      mediaRecorderRef.current?.stop();
      mediaRecorderRef.current?.addEventListener('stop', () => {
          if (recordedChunks.length) {
              const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
              const videoUrl = URL.createObjectURL(videoBlob);
              downloadFile(videoUrl, "recorded_video.webm");
  
              const dataBlob = new Blob([JSON.stringify(inferenceData)], { type: 'application/json' });
              const dataUrl = URL.createObjectURL(dataBlob);
              downloadFile(dataUrl, "inference_data.json");
  
              setRecordedChunks([]);
              setInferenceData([]);
          }
      });
    }, [recordedChunks, inferenceData]);

    // render webcam, canvas, and start/stop buttons
    return (
      <div style={{ position: 'relative'}}>
        <div style={{ position: 'relative', width: '1280px', height: '720px' }} className="mirrored-container">
          <Webcam 
            ref={camRef} 
            style={{ 
              position: 'absolute',
              width: '100%',
              height: '100%',
            }} 
            videoConstraints={{ width: 1280, height: 720, facingMode: "user"}}

            // optional code to manually set frame rate. testing inconclusive for whether this
            // actually works?
            // videoConstraints={{ width: 1280, height: 720, facingMode: "user", frameRate: 15 }}
          />
          <canvas 
            ref={canvasRef} 
            style={{ 
              position: 'absolute',
              width: '100%',
              height: '100%',
              zIndex: 10,
              border: '4px solid white'
            }}
          />
        </div>
        <Box position="absolute" top="0" left="0" p="2" bgColor="rgba(0,0,0,0.5)" color="white" fontSize="sm">
          {fps.toFixed(1)} FPS
        </Box>
        <Button onClick={handleStartRecording}>Start Recording</Button>
        <Button onClick={handleStopRecording}>Stop Recording</Button>
      </div>
    );
}

export default WebcamDisplay;
