import { useRef, useEffect, useCallback, useState } from "react";
import Webcam from "react-webcam";
import { Box } from "@chakra-ui/react";
import { PoseModel, BasePose, Inference } from "../utils/ModelDefinitions";
import { drawKeypoints2D } from "../utils/utilities";
import RecordRTC from 'recordrtc';

type Props = {
  model: PoseModel<BasePose>;
};

function WebcamDisplay({model}: Props){

    const [isModelChanged, setIsModelChanged] = useState(true);
    const curModelRef = useRef<PoseModel<BasePose> | null>(null);
    const [inferenceData, setInferenceData] = useState<Inference[]>([]);

    const camRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const recorderRef = useRef<RecordRTC | null>(null);

    const requestRef =  useRef<number | undefined>(undefined);
    const [fps, setFps] = useState(0);
    const frameCount = useRef(0);
    const prevTime = useRef(Date.now());

    // function to run inference on a video frame and draw the corresponding keypoints.
    const estimatePose = useCallback(async () => {
      if (camRef.current?.video?.readyState === 4) {
        const video = camRef.current.video;
        const canvas = canvasRef.current;
        if(video.videoWidth != video.width || video.videoHeight != video.height) {
          video.width = video.videoWidth;
          video.height = video.videoHeight;
        }
        if (canvas && video && video.videoWidth > 0 && video.videoHeight > 0) {     
          if (canvas.width != video.width || canvas.height != video.height) {
            canvas.width = video.width;
            canvas.height = video.height;
          }
          const poseData = await model.runInference(video);

          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, video.width, video.height);
            poseData.forEach(pose => { 
              drawKeypoints2D(pose.keypoints, 0.1, ctx); 
            });
          }
          const timestamp = video.currentTime;
          const modelName = model.name;
          setInferenceData(prevData => [...prevData, { timestamp, modelName, poseData }]);
        }
      }
      frameCount.current += 1;
      requestRef.current = requestAnimationFrame(estimatePose);
    }, [model]);

      // function to loop the runInference function for each frame of the video.
      const runModel = useCallback(async () => {
        if (curModelRef.current) {
          curModelRef.current.dispose();
        }

        await model.load();
        curModelRef.current = model;
        console.log("model loaded. ", model.name);

        requestRef.current = requestAnimationFrame(() => estimatePose());
      }, [model, estimatePose]);

      useEffect(() => {
        setIsModelChanged(true);
      }, [model]);
      
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

      useEffect(() => {
        const calculateFps = () => {
          const now = Date.now();
          const deltaTime = (now - prevTime.current) / 1000;
          const resultFps = frameCount.current / deltaTime;
          setFps(resultFps);
          frameCount.current = 0;
          prevTime.current = now;
          console.log(model.name + ' fps ' + resultFps);
        };
        // recalculate fps every second
        const interval = setInterval(calculateFps, 1000);
        return () => clearInterval(interval);
      }, [model.name]);

      const startRecording = () => {
        if (camRef.current?.video) {
          setInferenceData([]);
          const newRecorder = new RecordRTC(camRef.current.video, { type: "video" });
          recorderRef.current = newRecorder;
          recorderRef.current?.startRecording();
        }
      };

      const stopRecording = () => {
        if (recorderRef.current) {
          recorderRef.current.stopRecording(() => {
            const blob = recorderRef.current?.getBlob();
            if(blob) {
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.style.display = "none";
              a.href = url;
              a.download = "recorded_video.mp4";
              document.body.appendChild(a);
              a.click();
              window.URL.revokeObjectURL(url);
            }            
            const blobData = new Blob([JSON.stringify(inferenceData)], { type: "application/json" });
            const urlData = URL.createObjectURL(blobData);
            const aData = document.createElement("a");
            aData.style.display = "none";
            aData.href = urlData;
            aData.download = "inference_data.json";
            document.body.appendChild(aData);
            aData.click();
            window.URL.revokeObjectURL(urlData);
          });
          recorderRef.current.destroy();
        }
      };

      return (
        <div style={{ position: 'relative'}}>
          <button onClick={startRecording}>Start Recording</button>
          <button onClick={stopRecording}>Stop Recording</button>
          <div style={{ position: 'relative', width: '1280px', height: '720px' }} className="mirrored-container">
            <Webcam 
              ref={camRef} 
              style={{ 
                position: 'absolute',
                width: '100%',
                height: '100%',
              }} 
              videoConstraints={{ width: 1280, height: 720, facingMode: "user"}}
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
        </div>
      );
}

export default WebcamDisplay;
