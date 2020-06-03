# edgeai-computer-pointer-controller
Computer Pointer Controller app controls the mouse pointer by using eye and head position.

### Introduction

Computer Pointer Controller app controls the mouse pointer by rolling of the eyes and head pose estimation. The app takes video or webcam stream as input and uses Intel OpenVino toolket to run the interference on image frames and move the mouse pointer accordingly. 

## Project Set Up and Installation

### Setup
1. Install the openvino toolkit following the instructions here. Here are the instructions for [macos](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html).

2. Clone the repo.

3. Run jupyter notebook: computer-pointer-controller-workflow.ipynb. This will download the needed models in the `./models` directory. Model details are provided in later section.

4. Now source the OpenVino environmnet. 
```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
```

Python version can be any python 3 version installed on the computer system.

5. Install all the python requirements from `./requirements.txt` into the conda or venv environment.
                        
## Demo

Now that the setup is done, we are ready to run the workflow. 

1. To run the job either using `queue_job.sh` or python3 code. The shell script is wrapper around following call:
```bash
python3 ./src/main.py -fm "$FACE_DETECTION_MODEL_PATH" \
                -hm "$HEAD_POSE_ESTIMATION_MODEL_PATH" \
                -lm "$FACIAL_LANDMARKS_DETECTION_MODEL_PATH" \
                -gm "$GAZE_ESTIMATION_MODEL_PATH" \
                -i "$INPUT" \
                -o "$OUTPUT" \
                -d "$DEVICE" \
                -t "$THRESHOLD"
```
To get the detailed help type:
```bash
python3 ./src/main.py -h
```

Here is the help output:
```bash
  -h, --help            show this help message and exit
  -fm FACE_DETECTION_MODEL, --face-detection-model FACE_DETECTION_MODEL
                        Path to Face Detection model without extension
  -hm HEAD_POSE_MODEL, --head-pose-model HEAD_POSE_MODEL
                        Path to Head Pose Estimation model without extension
  -lm FACIAL_LANDMARKS_MODEL, --facial-landmarks-model FACIAL_LANDMARKS_MODEL
                        Path to Facial Landmarks Detection model without
                        extension
  -gm GAZE_ESTIMATION_MODEL, --gaze-estimation-model GAZE_ESTIMATION_MODEL
                        Path to Gaze Estimation model without extension
  -i INPUT, --input INPUT
                        Path to input video. Use 'cam' for capturing video
                        stream from camera
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        shared lib with the kernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; Can be: CPU,
                        GPU, FPGA or MYRIAD
  -t THRESHOLD, --threshold THRESHOLD
                        Probability threshold for detections
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to output directory
  -v SHOW_INTERMEDIATE_VISUALIZATION, --show-intermediate-visualization SHOW_INTERMEDIATE_VISUALIZATION
                        Shows intermediate step visualization
```

As a sample input, `demo.mp4` is provided as sample video in `./original_videos` directory. For intermediate visualization, please move the demo video as sometimes the intermediate visualization may be below the actual demo video.


## Benchmarks
Here are the benchmarks for CPU on my local system:

| Precision | Load Time    | Inference Time | Effective FPS    |
| --------- | -----------: | -------------- | ---------------- |  
| FP16      | 497 ms       | 23.2 s         |2.5               |
| FP32      | 543 ms       | 23.1 s         | 2.4              |  
| FP16-INT8      | 693 ms       | 23.1 s         | 2.55             |  

Across devices:
| Precision | Load Time    | Inference Time | Effective FPS    |
| --------- | -----------: | -------------- | ---------------- |  
| FP16      | 497 ms       | 23.2 s         |2.5               |
| FP32      | 543 ms       | 23.1 s         | 2.4              |  
| FP16-INT8      | 693 ms       | 23.1 s         | 2.55             |  

## Results

Here are the results:
- Decreasing the precision of model decreases accuracy. It should in general decrease inference time, but not always.
- With higher precision, model takes slightly higher time in inference, but accuracy drop from FP32 to FP16 is not significant. This may be due to models being trained and simplified in such a way that they work nicely at low precision rates

### Edge Cases

- Face detection currently occurs for 1 face. Not sure how the model will react in case of multiple people in frame
- Lighting condition expected may result in gaze prediction in case eye vector is not properly recognized.