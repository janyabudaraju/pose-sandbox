// import * as posenet from '@tensorflow-models/posenet';

import { KP } from "./ModelDefinitions";

export function drawKeypoints(keypoints: KP[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1) {
    const color = "aqua"
    const rad = 3

    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }
        const { y, x } = keypoint.position;
        drawPoint(ctx, x * scale, y * scale, rad, color);
        // console.log('drew kp at %d %d' , x, y)
    }
}

export function drawPoint(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}  

export function drawSegment([ay, ax]: [number, number], [by, bx]: [number, number], color: string, scale: number, ctx: CanvasRenderingContext2D) {
    const lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ax * scale, ay * scale);
    ctx.lineTo(bx * scale, by * scale);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
}

// export function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
//     const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);
//     adjacentKeyPoints.forEach((keypoints) => {
//         drawSegment(toTuple(keypoints[0].position), toTuple(keypoints[1].position), color, scale, ctx);
//     });
// }