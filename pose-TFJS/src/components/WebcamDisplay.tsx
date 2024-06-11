import React, { useRef } from "react";
import Webcam from "react-webcam";

// TODO: add prop for model
function WebcamDisplay(){
    const camRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const runInference = async (net) => {
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

            const pred = await net.detect(vid);
            // Draw mesh
            const ctx = canvasRef.current.getContext("2d");
            drawPred(pred, ctx);  // TODO: implement
          }
        }
      };

    // const runModel = async () => {
    //     // TODO: load model into net

    // };
}

export default WebcamDisplay;
