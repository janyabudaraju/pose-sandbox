
import { useRef, useEffect, useCallback, useState } from "react";
import { Box, Input } from "@chakra-ui/react";
import { PoseModel, BasePose, Inference, modelOptions } from "../utils/ModelDefinitions";
import { drawKeypoints2D } from "../utils/utilities";

type Props = {
    models: PoseModel<BasePose>[];
}

function WebcamDisplay({ models }: Props) {
    const [isModelChanged, setIsModelChanged] = useState(true);
    const curModelRefs = useRef<PoseModel<BasePose>[]>([]);
    const [inferenceData, setInferenceData] = useState<Inference[][]>([]);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [fps, setFps] = useState(0);
    const frameCountFps = useRef(0);
    const frameCountIdx = useRef(0);
    const prevTime = useRef(Date.now());

    const downloadFile = useCallback((url: string, filename: string) => {
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }, []);

    const downloadInferenceData = () => {
        const dataBlob = new Blob([JSON.stringify(inferenceData)], { type: 'application/json' });
        const dataUrl = URL.createObjectURL(dataBlob);
        const curTime = new Date().getTime();
        const fileName = 'inference_data' + curTime + '.json';
        downloadFile(dataUrl, fileName);
    };

    const estimatePose = useCallback(async () => {
        if (videoRef.current?.readyState === 4) {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            if (video.videoWidth !== video.width || video.videoHeight !== video.height) {
                video.width = video.videoWidth;
                video.height = video.videoHeight;
            }
            if (canvas && video && video.width > 0 && video.height > 0) {
                if (canvas.width !== video.width || canvas.height !== video.height) {
                    canvas.width = video.width;
                    canvas.height = video.height;
                }
        
                const modelInferences = await Promise.all(models.map(async (model) => {
                    const poseData = await model.runInference(video);
                    return { timeStamp: video.currentTime, frameIdx: frameCountIdx.current, modelId: model.id, poseData };
                }));

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
                setInferenceData(prevData => [...prevData, modelInferences]);
            }
        }
        frameCountFps.current++;
        frameCountIdx.current++;

        if (videoRef.current?.currentTime && videoRef.current?.currentTime < videoRef.current?.duration) {
            requestAnimationFrame(estimatePose);
        }
        else {
            downloadInferenceData();
        }
    }, [models, downloadInferenceData]);

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

    useEffect(() => {
        setIsModelChanged(true);
    }, [models]);

    useEffect(() => {
        if (isModelChanged) {
            setIsModelChanged(false);
            runModel();
        }
    }, [isModelChanged, runModel]);

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

    return (
        <div style={{ position: 'relative' }}>
            <Input type="file" accept="video/*" onChange={handleFileUpload} />
            <div style={{ position: 'relative', width: '1280px', height: '720px' }} className="mirrored-container">
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

export default WebcamDisplay;
