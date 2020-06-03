#!/bin/bash

#exec 1>/tmp/stdout.log 2>/tmp/stderr.log

#DEVICE=$1
#PRECISION=$2
#INPUT=$3
#THRESHOLD=$4
FP32_PRECISION="FP32"
FP16_PRECISION="FP16"
FP8_PRECISION="FP8"

DEVICE="CPU"
PRECISION=$FP16_PRECISION
INPUT="./original_videos/demo.mp4"
THRESHOLD=0.5
OUTPUT="./results/$PRECISION/$DEVICE/"

FACE_DETECTION_MODEL_NAME="face-detection-adas-binary-0001"
FACIAL_LANDMARKS_DETECTION_MODEL_NAME="face-detection-adas-binary-0001"
GAZE_ESTIMATION_MODEL_NAME="face-detection-adas-binary-0001"
HEAD_POSE_ESTIMATION_MODEL_NAME="face-detection-adas-binary-0001"




function getModelPath() {
    model_name=$1
    precision=$2
    echo $model_name
    echo $precision
    if [ "$1" == "$FACE_DETECTION_MODEL_NAME" ]
    then
      precision="FP32-INT1"
    fi

    local model_path="./models/intel/$model_name/$precision/"
    echo "$model_path"
}




#FACE_DETECTION_MODEL_PATH=$(getModelPath $FACE_DETECTION_MODEL_NAME $PRECISION)
#HEAD_POSE_ESTIMATION_MODEL_PATH=$(getModelPath $FACIAL_LANDMARKS_DETECTION_MODEL_NAME $PRECISION)
#FACIAL_LANDMARKS_DETECTION_MODEL_PATH=$(getModelPath $GAZE_ESTIMATION_MODEL_NAME $PRECISION)
#GAZE_ESTIMATION_MODEL_PATH=$(getModelPath $HEAD_POSE_ESTIMATION_MODEL_NAME $PRECISION)

if [ $PRECISION == $FP16_PRECISION ];
then
  FACE_DETECTION_MODEL_PATH=./models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
  HEAD_POSE_ESTIMATION_MODEL_PATH=./models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001
  FACIAL_LANDMARKS_DETECTION_MODEL_PATH=./models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009
  GAZE_ESTIMATION_MODEL_PATH=./models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002
else if [ $PRECISION == $FP32_PRECISION ];
then
  FACE_DETECTION_MODEL_PATH=./models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
  HEAD_POSE_ESTIMATION_MODEL_PATH=./models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
  FACIAL_LANDMARKS_DETECTION_MODEL_PATH=./models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009
  GAZE_ESTIMATION_MODEL_PATH=./models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
else
  FACE_DETECTION_MODEL_PATH=./models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
  HEAD_POSE_ESTIMATION_MODEL_PATH=./models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001
  FACIAL_LANDMARKS_DETECTION_MODEL_PATH=./models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009
  GAZE_ESTIMATION_MODEL_PATH=./models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002
fi



mkdir -p $OUTPUT

echo "DEVICE: $DEVICE, PRECISION: $PRECISION, VIDEO: $INPUT, OUTPUT: $OUTPUT THRESHOLD: $THRESHOLD"
echo "MODELS: $FACE_DETECTION_MODEL_PATH; $HEAD_POSE_ESTIMATION_MODEL_PATH; $FACIAL_LANDMARKS_DETECTION_MODEL_PATH; $GAZE_ESTIMATION_MODEL_PATH"

python3 ./src/main.py -fm "$FACE_DETECTION_MODEL_PATH" \
                -hm "$HEAD_POSE_ESTIMATION_MODEL_PATH" \
                -lm "$FACIAL_LANDMARKS_DETECTION_MODEL_PATH" \
                -gm "$GAZE_ESTIMATION_MODEL_PATH" \
                -i "$INPUT" \
                -o "$OUTPUT" \
                -d "$DEVICE" \
                -t "$THRESHOLD"



#cd /output
#
#tar zcvf output.tgz *