import { useRef, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as posenet from "@tensorflow-models/posenet";
import { drawKeypoints } from "../utils/utilities";

// TODO: add prop for model

function WebcamDisplay(){
    const camRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const requestRef =  useRef<number | undefined>(undefined);

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

            const pose = await net.estimateSinglePose(vid);

            const ctx = canvas.getContext("2d");
            if(ctx) {
              ctx.clearRect(0, 0, vidW, vidH);
              drawKeypoints(pose["keypoints"], 0.1, ctx);
            }
          }
        }
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

      return (
        <div style={{ position: 'relative', width: '1280px', height: '720px' }}>
          {/* <Webcam
          ref={camRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            width: 640,
            height: 480,
          }}
        /> */}

          <Webcam 
            ref={camRef} 
            style={{ 
              position: 'absolute',
              width: '100%',
              height: '100%',
              top: 0,
              left: 0,
            }} 
            videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
          />
          <canvas 
            ref={canvasRef} 
            style={{ 
              position: 'absolute',
              width: '100%',
              height: '100%',
              top: 0,
              left: 0,
              zIndex: 10,
              border: '4px solid black'
            }}
          />
        </div>
      );
}

export default WebcamDisplay;
