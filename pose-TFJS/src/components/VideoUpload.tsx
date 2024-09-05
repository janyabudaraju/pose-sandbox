
import { useRef, useEffect, useCallback, useState } from "react";
import { Box, Input } from "@chakra-ui/react";
import { PoseModel, BasePose, Inference, modelOptions } from "../utils/ModelDefinitions";
import { drawKeypoints2D } from "../utils/utilities";

type Props = {
    models: PoseModel<BasePose>[];
}

function VideoUpload({ models }: Props) {

    // state to track model changes from selections
    const [isModelChanged, setIsModelChanged] = useState(true);
    // references to track current models to run
    const curModelRefs = useRef<PoseModel<BasePose>[]>([]);
    // state to store inference data for the current video
    const [inferenceData, setInferenceData] = useState<Inference[][]>([]);

    // refs for the video play component and canvas
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // ref and state for fps calculation
    const [fps, setFps] = useState(0);
    const frameCountFps = useRef(0);
    const frameCountIdx = useRef(0);
    const prevTime = useRef(Date.now());

    /**
     * core function responsible for running inference on
     * each video frame & drawing keypoints onto canvas.
     * handles:
     * - running the selected pose detection models for the current frame
     * - drawing keypoints on the canvas
     * - storing the inference results
     */
    const estimatePose = useCallback(async () => {
        if (videoRef.current?.readyState === 4) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            // update the video and canvas dimensions to match video width and height
            if (video.videoWidth !== video.width || video.videoHeight !== video.height) {
                video.width = video.videoWidth;
                video.height = video.videoHeight;
            }
            if (canvas && video && video.width > 0 && video.height > 0) {
                if (canvas.width !== video.width || canvas.height !== video.height) {
                    canvas.width = video.width;
                    canvas.height = video.height;
                }
                
                // run pose detection models on the current frame
                const modelInferences = await Promise.all(models.map(async (model) => {
                    const poseData = await model.runInference(video);
                    return { timeStamp: video.currentTime, frameIdx: frameCountIdx.current, modelId: model.id, poseData };
                }));

                // draw keypoints on the canvas
                const ctx = canvas.getContext("2d");
                if (ctx) {
                    ctx.clearRect(0, 0, video.width, video.height);
                    modelInferences.forEach(({ poseData, modelId }) => {
                        const color = modelOptions.find(model => model.id === modelId)?.color;
                        if (color) {
                            poseData.forEach(pose => {
                                drawKeypoints2D(pose.keypoints, 0.1, color, ctx);
                            });
                        }
                    });
                }
                // update the inference data state to append the new frame of data
                setInferenceData(prevData => {
                    const updatedData = [...prevData, modelInferences];
                    return updatedData;
                });
            }
        }
        // increment frame counters to update fps
        frameCountFps.current++;
        frameCountIdx.current++;

        // check if video still has time on the clock (still playing)
        if (videoRef.current?.currentTime && videoRef.current?.currentTime < videoRef.current?.duration) {
            requestAnimationFrame(estimatePose);
        }
        // move to printing all collected data once video is complete
        else {
            if (inferenceData.length > 0) {
                try{
                    // print data to console as json
                    const jsonData = JSON.stringify(inferenceData, null, 2);
                    console.log('stringified Data:', jsonData);
                }
                catch (error) {
                    console.error('Failed to serialize inference data:', error);
                }
            } else {
                console.log('no inference data to log.');
            }
        }
    }, [models, inferenceData]);

    // load all models and begin animation loop.
    const runModel = useCallback(async () => {
        curModelRefs.current.forEach(model => model.dispose());
        curModelRefs.current = [];
        await Promise.all(models.map(async (model) => {
            await model.load();
        }));
        curModelRefs.current = models;
        console.log("models loaded.");

        if (videoRef.current) {
            videoRef.current.play();
            requestAnimationFrame(() => estimatePose());
        }
    }, [models, estimatePose]);

    // set the model changed flag when new models are passed
    useEffect(() => {
        setIsModelChanged(true);
    }, [models]);

    // re-run model inference when models change
    useEffect(() => {
        if (isModelChanged) {
            setIsModelChanged(false);
            runModel();
        }
    }, [isModelChanged, runModel]);

    // fps calculation, running once per second
    useEffect(() => {
        const calculateFps = () => {
            const now = Date.now();
            const deltaTime = (now - prevTime.current) / 1000;
            const resultFps = frameCountFps.current / deltaTime;
            setFps(resultFps);
            frameCountFps.current = 0;
            prevTime.current = now;
        };
        const interval = setInterval(calculateFps, 1000);
        return () => clearInterval(interval);
    }, []);

    // handle file upload by resetting inference data and loading reference
    const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            if (videoRef.current) {
                videoRef.current.src = url;
                videoRef.current.load();
                setInferenceData([]);
                frameCountIdx.current = 0;
            }
        }
    }, []);

    // render video and canvas, using chakra input type
    return (
        <div style={{ position: 'relative' }}>
            <Input type="file" accept="video/*" onChange={handleFileUpload} />
            <div style={{ position: 'relative', width: '1280px', height: '720px' }}>
                <video
                    ref={videoRef}
                    style={{
                        position: 'absolute',
                        width: '100%',
                        height: '100%',
                    }}
                    controls
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

export default VideoUpload;
