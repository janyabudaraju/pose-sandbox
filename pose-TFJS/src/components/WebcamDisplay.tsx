import { useRef, useEffect, useCallback, useState } from "react";
import Webcam from "react-webcam";
// import { drawKeypoints2D } from "../utils/utilities";
import { Box } from "@chakra-ui/react";
import { PoseModel, BasePose } from "../utils/ModelDefinitions";

type Props = {
  model: PoseModel<BasePose>;
};

function WebcamDisplay({model}: Props){

    const camRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const requestRef =  useRef<number | undefined>(undefined);

    const [fps, setFps] = useState(0);
    const frameCount = useRef(0);
    const prevTime = useRef(Date.now());

    // function to run inference on a video frame and draw the corresponding keypoints.
    const estimatePose = useCallback(async () => {
      if (camRef.current?.video?.readyState === 4) {
          const video = camRef.current.video;
          const canvas = canvasRef.current;
          video.width = video.videoWidth;
          video.height = video.videoHeight;

          if (canvas && video && video.videoWidth > 0 && video.videoHeight > 0) {      
              const poseData = await model.runInference(video);
              const ctx = canvas.getContext("2d");
              if (ctx) {
                  ctx.clearRect(0, 0, video.videoWidth, video.videoHeight);
                  console.log(poseData);
              }
          }
      }

      frameCount.current += 1;
      requestRef.current = requestAnimationFrame(estimatePose);
    }, [model]);

      // function to loop the runInference function for each frame of the video.
      const runPosenet = useCallback(async () => {
        await model.load();
        console.log("model loaded.");
        requestRef.current = requestAnimationFrame(() => estimatePose());
      }, [model, estimatePose]);

      // hook to start the loop and clean up as necessary.
      useEffect(() => {
        runPosenet();
        return () => {
          if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
          }
        };
      }, [runPosenet]);

      useEffect(() => {
        const calculateFps = () => {
          const now = Date.now();
          const deltaTime = (now - prevTime.current) / 1000;
          const resultFps = frameCount.current / deltaTime;
          setFps(resultFps);
          frameCount.current = 0;
          prevTime.current = now;
        };
        // recalculate fps every second
        const interval = setInterval(calculateFps, 1000);
        return () => clearInterval(interval);
      }, []);

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
              videoConstraints={{ width: 1280, height: 720, facingMode: "user", frameRate: { ideal: 120, max: 120 } }}
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
