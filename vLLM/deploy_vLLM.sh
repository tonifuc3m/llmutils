#!/bin/bash

# Default values for arguments
IS_LOCAL_MODEL="${1:-1}"
VOLUME_HOST_PATH="${2:-/data/models/vLLM_models/}"
VOLUME_CONTAINER_PATH="${3:-/mnt/model/}"
HOST_PORT="${4:-8000}"
IMAGE_NAME="${5:-vllm/vllm-openai:v0.4.2}"
MODEL_NAME="${6:-NousResearch--Meta-Llama-3-8B-Instruct--own-awq-w_bit_4--GEMM}"
DTYPE="${7:-auto}"
GPU_MEMORY_UTILIZATION="${8:-0.5}"
SERVED_MODEL_NAME="${9:-model1}"
MAX_MODEL_LEN="${10:-4096}"

# Create final model name
if [ "$IS_LOCAL_MODEL" -eq 1 ]; then
    FINAL_MODEL_NAME="${VOLUME_CONTAINER_PATH}${MODEL_NAME}"
else
    FINAL_MODEL_NAME=$MODEL_NAME
fi

# Function to echo and run Docker command
run_docker() {
    echo "Executing Docker Command:"
    echo "$1"
    eval "$1"
}

# Common options for both local and HF models
COMMON_OPTS="--privileged --gpus all -d \
        --restart unless-stopped \
        -v \"$VOLUME_HOST_PATH:$VOLUME_CONTAINER_PATH\" \
        -p $HOST_PORT:$HOST_PORT \
        --ipc=host"

COMMON_IMAGE_OPTS="$IMAGE_NAME \
        --model=$FINAL_MODEL_NAME \
        --dtype=$DTYPE \
        --port $HOST_PORT \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --served-model-name $SERVED_MODEL_NAME"

if [ "$MAX_MODEL_LEN" -eq 4096 ]; then
    COMMON_IMAGE_OPTS="$COMMON_IMAGE_OPTS"
else
    COMMON_IMAGE_OPTS="$COMMON_IMAGE_OPTS --max-model-len $MAX_MODEL_LEN"
fi

# Determine Docker command based on model location
if [ "$IS_LOCAL_MODEL" -eq 1 ]; then
    CMD="docker run $COMMON_OPTS \
        --env \"TRANSFORMERS_OFFLINE=1\" \
        --env \"HF_DATASET_OFFLINE=1\" \
        $COMMON_IMAGE_OPTS"
else
    CMD="docker run $COMMON_OPTS \
        --env \"HF_HOME=/data/models/transformers/\" \
        $COMMON_IMAGE_OPTS"
fi

run_docker "$CMD"