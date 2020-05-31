#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

DEVICE=$1
PRECISION=$2
INPUT=$3
THRESHOLD=$5

OUTPUT="results/$PRECISION/$DEVICE"

FACE_DETECTION_MODEL_NAME="face-detection-adas-binary-0001"
FACIAL_LANDMARKS_DETECTION_MODEL_NAME="face-detection-adas-binary-0001"
GAZE_ESTIMATION_MODEL_NAME="face-detection-adas-binary-0001"
HEAD_POSE_ESTIMATION_MODEL_NAME="face-detection-adas-binary-0001"


FP32_PRECISION="FP32"
FP16_PRECISION="FP16"

function getModelPath() {
    model_name=$1
    precision=$2

    if [ "$1" == "$FACE_DETECTION_MODEL_NAME" ]
    then
      precision="FP32-INT1"
    fi

    local model_path="./models/intel/$model_name/$precision/"
    echo "$model_path"
}




FACE_DETECTION_MODEL_PATH=getModelPath $FACE_DETECTION_MODEL_NAME $PRECISION
HEAD_POSE_ESTIMATION_MODEL_PATH=getModelPath $FACIAL_LANDMARKS_DETECTION_MODEL_NAME $PRECISION
FACIAL_LANDMARKS_DETECTION_MODEL_PATH=getModelPath $GAZE_ESTIMATION_MODEL_NAME $PRECISION
GAZE_ESTIMATION_MODEL_PATH=getModelPath $HEAD_POSE_ESTIMATION_MODEL_NAME $PRECISION

mkdir -p $OUTPUT

if echo "$DEVICE" | grep -q "FPGA"; then # if device passed in is FPGA, load bitstream to program FPGA
    #Environment variables and compilation for edge compute nodes with FPGAs
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2

    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx

    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

echo "DEVICE: $DEVICE, PRECISION: $PRECISION, VIDEO: $INPUT, OUTPUT: $OUTPUT THRESHOLD: $THRESHOLD"

python3 src/main.py -fm "$FACE_DETECTION_MODEL_PATH" \
                -pm "$HEAD_POSE_ESTIMATION_MODEL_PATH" \
                -lm "$FACIAL_LANDMARKS_DETECTION_MODEL_PATH" \
                -gm "$GAZE_ESTIMATION_MODEL_PATH" \
                -i "$INPUT" \
                -o "$OUTPUT" \
                -d "$DEVICE" \
                -t "$THRESHOLD"



cd /output

tar zcvf output.tgz *