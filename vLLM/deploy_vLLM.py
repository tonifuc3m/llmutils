#!/usr/bin/env python3

import argparse
import subprocess
import sys
import logging
import json
import os



def main():
    parser = argparse.ArgumentParser(
        description='Run Docker command with specified arguments.',
        epilog='''Example usage:
  python3 script.py --IS_LOCAL_MODEL 1 --VOLUME_HOST_PATH /data/models/ --VOLUME_CONTAINER_PATH /mnt/model/ \
--HOST_PORT 8000 --IMAGE_NAME my_image:latest --MODEL_NAME my_model --DTYPE auto --GPU_MEMORY_UTILIZATION 0.5''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # All arguments are optional and must be provided with their respective flags
    parser.add_argument('--IS_LOCAL_MODEL', type=int, choices=[0, 1], help='Is local model (1 or 0)')
    parser.add_argument('--VOLUME_HOST_PATH', help='Volume host path (must exist)')
    parser.add_argument('--VOLUME_CONTAINER_PATH', help='Volume container path')
    parser.add_argument('--HOST_PORT', type=int, help='Host port')
    parser.add_argument('--IMAGE_NAME', help='vLLM Docker image name (e.g., vllm/vllm-openai:v0.4.2)')
    parser.add_argument('--MODEL_NAME', help='Model name')
    parser.add_argument('--DTYPE', choices=['auto', 'half', 'float', 'float32', 'float16', 'bfloat16'], help='Data type')
    parser.add_argument('--GPU_MEMORY_UTILIZATION', type=float, help='GPU memory utilization (0.0 - 1.0)')
    parser.add_argument('--SERVED_MODEL_NAME', help='Served model name. It is recommended to use the same name as the model name. (optional)')
    parser.add_argument('--MAX_MODEL_LEN', type=int, help='Max model length (optional)')
    parser.add_argument('--tensor_parallel_size', type=int, help='Number of GPUs')
    parser.add_argument('--dry-run', action='store_true', help='Display the Docker command without executing it')
    parser.add_argument('--config', help='Path to a configuration JSON file')
    parser.add_argument('--log', help='Path to a log file')

    args = parser.parse_args()

    # Default values for arguments
    defaults = {
        'IS_LOCAL_MODEL': 1,
        'VOLUME_HOST_PATH': '/data/models/vLLM_models/',
        'VOLUME_CONTAINER_PATH': '/mnt/model/',
        'HOST_PORT': 8000,
        'IMAGE_NAME': 'vllm/vllm-openai:v0.4.2',
        'MODEL_NAME': 'NousResearch--Meta-Llama-3-8B-Instruct--own-awq-w_bit_4--GEMM',
        'DTYPE': 'auto',
        'GPU_MEMORY_UTILIZATION': 0.5,
        'SERVED_MODEL_NAME': None,
        'MAX_MODEL_LEN': None,
        'tensor_parallel_size': None,
    }

    # Initialize logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    logging.info("Script started.")


    # Load configuration from file if provided
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    if getattr(args, key, None) is None:
                        setattr(args, key, value)
                        logging.info(f"Loaded {key} from config file: {value}")
        else:
            logging.error(f"Configuration file '{args.config}' does not exist.")
            print(f"Error: Configuration file '{args.config}' does not exist.")
            sys.exit(1)
    
    # Assign default values and print messages for unspecified arguments
    for arg, default_value in defaults.items():
        if getattr(args, arg) is None:
            setattr(args, arg, default_value)
            if default_value is not None:
                print(f"{arg} not specified. Using default value: {default_value}")
                logging.info(f"{arg} not specified. Using default value: {default_value}")
            else:
                print(f"{arg} not specified. It will not be added to the Docker command.")
                logging.info(f"{arg} not specified. It will not be added to the Docker command.")



    # Extract arguments after defaults have been applied
    IS_LOCAL_MODEL = args.IS_LOCAL_MODEL
    VOLUME_HOST_PATH = args.VOLUME_HOST_PATH
    VOLUME_CONTAINER_PATH = args.VOLUME_CONTAINER_PATH
    HOST_PORT = args.HOST_PORT
    IMAGE_NAME = args.IMAGE_NAME
    MODEL_NAME = args.MODEL_NAME
    DTYPE = args.DTYPE
    GPU_MEMORY_UTILIZATION = args.GPU_MEMORY_UTILIZATION
    SERVED_MODEL_NAME = args.SERVED_MODEL_NAME
    MAX_MODEL_LEN = args.MAX_MODEL_LEN
    tensor_parallel_size = args.tensor_parallel_size
    DRY_RUN = args.dry_run

    # Validate IS_LOCAL_MODEL
    if IS_LOCAL_MODEL not in [0, 1]:
        logging.error("IS_LOCAL_MODEL must be either 0 or 1.")
        print("Error: --IS_LOCAL_MODEL must be either 0 or 1.")
        sys.exit(1)

    # Validate HOST_PORT
    if not (1 <= HOST_PORT <= 65535):
        logging.error("HOST_PORT must be between 1 and 65535.")
        print("Error: --HOST_PORT must be between 1 and 65535.")
        sys.exit(1)

    # Validate GPU_MEMORY_UTILIZATION
    if not (0.0 < GPU_MEMORY_UTILIZATION <= 1.0):
        logging.error("GPU_MEMORY_UTILIZATION must be between 0.0 (exclusive) and 1.0 (inclusive).")
        print("Error: --GPU_MEMORY_UTILIZATION must be between 0.0 (exclusive) and 1.0 (inclusive).")
        sys.exit(1)

    # Validate VOLUME_HOST_PATH existence
    if not os.path.exists(VOLUME_HOST_PATH):
        logging.error(f"The host path '{VOLUME_HOST_PATH}' does not exist.")
        print(f"Error: The host path '{VOLUME_HOST_PATH}' does not exist.")
        sys.exit(1)

    # Create final model name
    if IS_LOCAL_MODEL == 1:
        FINAL_MODEL_NAME = os.path.join(VOLUME_CONTAINER_PATH, MODEL_NAME)

    else:
        FINAL_MODEL_NAME = MODEL_NAME

    # Common options for both local and HF models
    COMMON_OPTS = (
        f'--privileged --gpus all -d '
        f'--restart unless-stopped '
        f'-v "{VOLUME_HOST_PATH}:{VOLUME_CONTAINER_PATH}" '
        f'-p {HOST_PORT}:{HOST_PORT} '
        f'--ipc=host'
    )

    # Build COMMON_IMAGE_OPTS
    COMMON_IMAGE_OPTS = (
        f'{IMAGE_NAME} '
        f'--model={FINAL_MODEL_NAME} '
        f'--dtype={DTYPE} '
        f'--port {HOST_PORT} '
        f'--gpu-memory-utilization {GPU_MEMORY_UTILIZATION}'
    )

    if tensor_parallel_size is not None:
        COMMON_IMAGE_OPTS += f' --tensor-parallel-size {tensor_parallel_size}'

    if SERVED_MODEL_NAME is not None:
        COMMON_IMAGE_OPTS += f' --served-model-name {SERVED_MODEL_NAME}'

    if MAX_MODEL_LEN is not None:
        COMMON_IMAGE_OPTS += f' --max-model-len {MAX_MODEL_LEN}'

    # Determine Docker command based on model location
    if IS_LOCAL_MODEL == 1:
        CMD = (
            f'docker run {COMMON_OPTS} '
            f'--env "TRANSFORMERS_OFFLINE=1" '
            f'--env "HF_DATASET_OFFLINE=1" '
            f'{COMMON_IMAGE_OPTS}'
        )
    else:
        CMD = (
            f'docker run {COMMON_OPTS} '
            f'--env "HF_HOME=/data/models/transformers/" '
            f'{COMMON_IMAGE_OPTS}'
        )

    run_docker(CMD, DRY_RUN)

def run_docker(cmd, dry_run):
    print("\nDocker Command:")
    print(cmd)
    logging.info(f"Docker Command: {cmd}")
    if dry_run:
        print("\nDry run enabled. Command not executed.")
        logging.info("Dry run enabled. Command not executed.")
    else:
        try:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("\nDocker command executed successfully.")
            logging.info("Docker command executed successfully.")
            logging.info(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print("\nError executing Docker command:")
            print(e.stderr)
            logging.error("Error executing Docker command:")
            logging.error(e.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main()
