# pose-sandbox

project aimed to allow for testing and development of real-time pose estimation in the browser.

typescript project contained in directory `pose-TFJS`; contains separate `README` detailing functionality. the typescript project contains all the code required to locally host a site to experiment, visualize, and analyze different forms of video with different pose estimation models.

python code is contained in directory `python-analysis`. pairs with `pose-TFJS` directory by providing tools to parse, find, and analyze JSON data (and corresponding video, if applicable). intended to function alongside a JSON file downloaded/copied from the typescript sandbox.

# JSON formatting

the JSON structure shared between the typescript and python project is as follows:

```
[
    [
        {
            "timeStamp": float,
            "modelId": str,
            "poseData": [
                {
                    "score": float,
                    "keypoints": [
                        {
                            "x": float,
                            "y": float,
                            "score": float,
                            "name": str
                        },
                        ...
                    ],
                    "keypoints3D": [
                        {
                            "x": float,
                            "y": float,
                            "z": float,
                            "score": float,
                            "name": str
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ],
    ...
]
```