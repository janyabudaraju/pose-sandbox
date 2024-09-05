import * as poseDetection from '@tensorflow-models/pose-detection';
import { PoseDetector, Pose } from '@tensorflow-models/pose-detection';

/**
 * interface representing form of a single inference result, which includes
 * the timestamp, frame index, model ID, and pose data.
 * this is a convenience representation for logging purposes, not the raw
 * model output, which is of type BasePose.
 */
export interface Inference {
    timeStamp: number;
    frameIdx: number;
    modelId: string;
    poseData: BasePose[];
}

/**
 * interface representing expected form of a 2D keypoint.
 */
export interface KP2D {
    x: number | undefined;
    y: number | undefined;
    score: number | undefined;
    name: string | undefined;
}

/**
 * interface representing expected form of a 2D keypoint.
 * inherits from KP2D to avoid duplicate fields.
 */
export interface KP3D extends KP2D {
    z: number | undefined;
}

/**
 * interface representing a single pose estimate inference.
 * expected raw output of a pose estimation model.
 */
export interface BasePose {
    score: number | undefined;
    keypoints: KP2D[];
}

/**
 * interface representing a single pose estimate inference with 3D keypoints (blazepose).
 * expected raw output of a pose estimation model.
 */
export interface Pose3D extends BasePose {
    keypoints3D: KP3D[];
}

/**
 * interface detailing expected implementation of pose detection models.
 * methods for loading, running inference, and disposing of the model.
 * allows for models to be stored and run in a single data structure, with the ability
 * to select and unselect cleanly & add more models as desired.
 */
export interface PoseModel <T> {
    id: string;
    name: string;
    color: string;
    load: () => Promise<void>;
    runInference: (video: HTMLVideoElement) => Promise<T[]>;
    dispose: () => void;
}

/**
 * creates and implements all functionality for a PoseModel instance  
 * for PoseNet.
 * @returns PoseModel<BasePose> - PoseNet model instance
 */
export function poseNetModel(): PoseModel<BasePose> {
    let model: PoseDetector | null = null;
    const id = 'posenet';
    const name = 'PoseNet';
    const color = 'orange';

    const load = async (): Promise<void> => {
        model = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet);
    };

    const runInference = async (video: HTMLVideoElement): Promise<BasePose[]> => {
        // console.log('posenet inference');

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
    const dispose = (): void => {
        if (model) {
            model.dispose();
            model = null;
        }
    };
    return { id, name, color, load, runInference, dispose };
}

/**
 * creates and implements all functionality for a PoseModel instance  
 * for MoveNet.
 * @returns PoseModel<BasePose> - MoveNet model instance
 */
export function moveNetModel(): PoseModel<BasePose> {
    let model: PoseDetector | null = null;
    const id = 'movenet';
    const name = 'MoveNet';
    const color = 'blue';

    const load = async (): Promise<void> => {
        model = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet);
    };

    const runInference = async (video: HTMLVideoElement): Promise<BasePose[]> => {
        if (!model) {
            throw new Error('movenet not loaded.');
        }
        // console.log('movenet inference');
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
    const dispose = (): void => {
        if (model) {
            model.dispose();
            model = null;
        }
    };
    return { id, name, color, load, runInference, dispose };
}

/**
 * creates and implements all functionality for a PoseModel instance  
 * for BlazePose.
 * @returns PoseModel<BasePose> - BlazePose model instance
 */
export function blazeNetModel(): PoseModel<Pose3D> {
    let model: PoseDetector | null = null;
    const id = 'blazepose';
    const name = 'BlazePose';
    const color = 'red';

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
        // console.log('blazepose inference');
        const poses: Pose[] = await model.estimatePoses(video);
        const inference = poses.map(pose => ({
            score: pose.score,
            keypoints: pose.keypoints.map(kp => ({
                x: kp.x,
                y: kp.y,
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
    const dispose = (): void => {
        if (model) {
            model.dispose();
            model = null;
        }
    };
    return { id, name, color, load, runInference, dispose };
}

/**
 *  placeholder PoseModel that does not perform any actual inference.
 * 
 * @returns PoseModel<BasePose> - no-op model instance
 */
export function noModel(): PoseModel<BasePose> {
    const id = 'none';
    const name = 'None';
    const color = '';
    const load = async(): Promise<void> => { };

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const runInference = async (video: HTMLVideoElement): Promise<BasePose[]> => {
        // console.log('no inference ', video.width);
        return [];
    }
    const dispose = (): void => { };
    return { id, name, color, load, runInference, dispose };
}

export const modelOptions: (PoseModel<BasePose>)[] = [ blazeNetModel(), poseNetModel(), moveNetModel() ];
