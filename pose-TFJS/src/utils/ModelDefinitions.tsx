import * as poseDetection from '@tensorflow-models/pose-detection';
import { PoseDetector, Pose } from '@tensorflow-models/pose-detection';

export interface KP {
    x: number | undefined;
    y: number | undefined;
    score: number | undefined;
    name: string | undefined;
}
export interface Pose2D {
    score: number | undefined;
    keypoints: KP[];
}
export interface PoseModel <T> {
    id: string;
    name: string;
    load: () => Promise<void>;
    runInference: (video: HTMLVideoElement) => Promise<T[]>;
}

export function poseNetModel(): PoseModel<Pose2D> {
    let model: PoseDetector | null = null;
    const id = 'posenet';
    const name = 'PoseNet';
    const load = async (): Promise<void> => {
        model = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet);
    };

    const runInference = async (video: HTMLVideoElement): Promise<Pose2D[]> => {
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


