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
    const runInference = useCallback(async (net: posenet) => {
        if (camRef.current?.video?.readyState === 4) {
          const vid = camRef.current.video;
          const canvas = canvasRef.current;
          if(canvas) {
            // obtain intrinsic video dims and set html vid and canvas dims
            const vidW = vid.videoWidth;
            const vidH = vid.videoHeight;
            vid.width = vidW;
            vid.height = vidH;
            canvas.width = vidW;
            canvas.height = vidH;

            const pose = await net.estimateSinglePose(vid);
            console.log(pose);

            const ctx = canvas.getContext("2d");
            if(ctx) {
              drawKeypoints(pose["keypoints"], 0.6, ctx);
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
        <div>
          <video ref={camRef} style={{ display: 'none' }} autoPlay playsInline muted></video>
          <canvas ref={canvasRef}></canvas>
        </div>
      );
}

export default WebcamDisplay;
