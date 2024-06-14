import { useRef, useEffect, useCallback, useState } from "react";
import Webcam from "react-webcam";
import * as posenet from "@tensorflow-models/posenet";
import { drawKeypoints2D } from "../utils/utilities";
import { Box } from "@chakra-ui/react";

// TODO: add prop for model

function WebcamDisplay(){

    const camRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const requestRef =  useRef<number | undefined>(undefined);

    const [fps, setFps] = useState(0);
    const frameCount = useRef(0);
    const prevTime = useRef(Date.now());

    // function to run inference on a video frame and draw the corresponding keypoints.
    const runInference = useCallback(async (net: posenet.PoseNet) => {
        if (camRef.current?.video?.readyState === 4) {
          const vid = camRef.current.video;
          const canvas = canvasRef.current;
          if(canvas) {
            // obtain intrinsic video dims and set html vid and canvas dims
            const vidW = vid.videoWidth;
            const vidH = vid.videoHeight;
            vid.width = vidW;
            vid.height = vidH;

            if (canvas.width !== vidW || canvas.height !== vidH) {
              canvas.width = vidW;
              canvas.height = vidH;
            }
            // get prediction
            const pose = await net.estimateSinglePose(vid);

            const ctx = canvas.getContext("2d");
            if(ctx) {
              ctx.clearRect(0, 0, vidW, vidH);
              drawKeypoints2D(pose["keypoints"], 0.1, ctx);
            }
          }
        }
        frameCount.current += 1;
        requestRef.current = requestAnimationFrame(() => runInference(net));
      }, []);

      // function to loop the runInference function for each frame of the video.
      const runPosenet = useCallback(async () => {
        const net = await posenet.load();
        console.log("posenet model loaded.");
        requestRef.current = requestAnimationFrame(() => runInference(net));
      }, [runInference]);

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
      }, [fps]);

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
