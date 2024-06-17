import * as poseDetection from '@tensorflow-models/pose-detection';
import { PoseDetector, Pose } from '@tensorflow-models/pose-detection';

export interface KP2D {
    x: number | undefined;
    y: number | undefined;
    score: number | undefined;
    name: string | undefined;
}
export interface KP3D extends KP2D{
    z: number | undefined;
}

export interface BasePose {
    score: number | undefined;
    keypoints: KP2D[];
}
export interface Pose3D extends BasePose {
    keypoints3D: KP3D[];
    // optionally can add segmentation info
}
export interface PoseModel <T> {
    id: string;
    name: string;
    load: () => Promise<void>;
    runInference: (video: HTMLVideoElement) => Promise<T[]>;
}

export function poseNetModel(): PoseModel<BasePose> {
    let model: PoseDetector | null = null;
    const id = 'posenet';
    const name = 'PoseNet';
    const load = async (): Promise<void> => {
        model = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet);
    };

    const runInference = async (video: HTMLVideoElement): Promise<BasePose[]> => {
        if (!model) {
            throw new Error('posenet not loaded.');
        }
        const poses: Pose[] = await model.estimatePoses(video);
        const inference = poses.map(pose => ({
            score: pose.score,
            keypoints: pose.keypoints.map(kp => ({
                x: kp.x,
                y: kp.y,
                score: kp.score,
                name: kp.name
            }))
        }));
        return inference;
    };
    return { id, name, load, runInference };
}

export function moveNetModel(): PoseModel<BasePose> {
    let model: PoseDetector | null = null;
    const id = 'movenet';
    const name = 'MoveNet';
    const load = async (): Promise<void> => {
        model = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet);
    };

    const runInference = async (video: HTMLVideoElement): Promise<BasePose[]> => {
        if (!model) {
            throw new Error('movenet not loaded.');
        }
        const poses: Pose[] = await model.estimatePoses(video);
        const inference = poses.map(pose => ({
            score: pose.score,
            keypoints: pose.keypoints.map(kp => ({
                x: kp.x,
                y: kp.y,
                score: kp.score,
                name: kp.name
            }))
        }));
        return inference;
    };
    return { id, name, load, runInference };
}

export function blazeNetModel(): PoseModel<Pose3D> {
    let model: PoseDetector | null = null;
    const id = 'blazepose';
    const name = 'BlazePose';
    const load = async (): Promise<void> => {
        const detectorConfig = {
            runtime: 'tfjs',
            enableSmoothing: true,
            modelType: 'full'
        };
        model = await poseDetection.createDetector(poseDetection.SupportedModels.BlazePose, detectorConfig);
    };

    const runInference = async (video: HTMLVideoElement): Promise<Pose3D[]> => {
        if (!model) {
            throw new Error('blazepose not loaded.');
        }
        const poses: Pose[] = await model.estimatePoses(video);
        const inference = poses.map(pose => ({
            score: pose.score,
            keypoints: pose.keypoints.map(kp => ({
                x: kp.x,
                y: kp.y,
                z: kp.z || 0,
                score: kp.score,
                name: kp.name
            })),
            keypoints3D: pose.keypoints3D?.map(kp => ({ 
                x: kp.x,
                y: kp.y,
                z: kp.z,
                score: kp.score,
                name: kp.name
            })) || []
        }));
        return inference;
    };

    return { id, name, load, runInference };
}

export const modelOptions: (PoseModel<BasePose>)[] = [ poseNetModel(), moveNetModel(), blazeNetModel()];
