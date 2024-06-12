interface Keypoint {
    x: number;
    y: number;
    position: {y: number, x: number}
    score: number;
}

export function drawKeypoints(keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1) {
    const color = "aqua"
    const rad = 3

    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }

        const { y, x } = keypoint.position;
        drawPoint(ctx, x * scale, y * scale, rad, color);
    }
}

export function drawPoint(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}  