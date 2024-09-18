# Docker deployment script for vLLM

This script facilitates the deployment of LLMs with vLLM using Docker. It supports deploying both local and HuggingFace-hosted models, adjusting resource usage, and setting environmental configurations to ensure optimal performance and security.

## Prerequisites

- **Docker**: Ensure Docker is installed on your system. Visit the [Docker installation guide](https://docs.docker.com/get-docker/) for instructions.
- **GPU Availability**: The script requires an NVIDIA GPU with CUDA support. Ensure NVIDIA drivers and NVIDIA Docker toolkit are installed. cuda>12.1
- **Bash Shell**: A Unix-like operating system with Bash shell is required to run the script.
- **Python 3.6 or higher**: Required if you choose to use the new Python deployment script.

## Bash Deployment Script

### Configuration

#### Environment Variables

Set the following environment variables according to your setup or pass them as arguments to the script:

- `IS_LOCAL_MODEL`: Set to `1` for local models and `0` for repository models (default: `1`)
- `VOLUME_HOST_PATH`: Host machine path to model data (default: `/data/models/vLLM_models/`)
- `VOLUME_CONTAINER_PATH`: Container path for mounting model data (default: `/mnt/model/`)
- `HOST_PORT`: The port on which the Docker container will serve (default: `8000`)
- `IMAGE_NAME`: Docker image to be used (default: `vllm/vllm-openai:v0.4.0.post1`)
- `MODEL_NAME`: Name or path of the model (default: `NousResearch--Meta-Llama-3-8B-Instruct--own-awq-w_bit_4--GEMM`)
- `DTYPE`: Data type for model operations (default: `auto`)
- `GPU_MEMORY_UTILIZATION`: Fraction of GPU memory to be utilized (default: `0.5`)
- `SERVED_MODEL_NAME`: Name under which the model will be served (default: `model1`)
- `MAX_MODEL_LEN`: Maximum model length (default: `4096`)

### Configuration File

Alternatively, create a `config.sh` file with the above variables set to override the defaults. Source this file before running the script.

## Usage

To run the script, use the following command format:

```bash
./deploy_vLLM.sh [IS_LOCAL_MODEL] [VOLUME_HOST_PATH] [VOLUME_CONTAINER_PATH] [HOST_PORT] [IMAGE_NAME] [MODEL_NAME] [DTYPE] [GPU_MEMORY_UTILIZATION] [SERVED_MODEL_NAME] [MAX_MODEL_LEN]
```

### Example execution

```bash
./deploy_vLLM.sh 0 ~/.cache/huggingface /root/.cache/huggingface 8000 vllm/vllm-openai:v0.4.2 NousResearch/Meta-Llama-3-8B-Instruct

./deploy_vLLM.sh 1 /data/models/transformers/ /mnt/model/ 8080 vllm/vllm-openai:v0.4.2 Meta-Llama-3-8B-Instruct auto 0.44 Meta-Llama-3-8B-Instruct 4097
./deploy_vLLM.sh 1 /data/models/transformers/ /mnt/model/ 8000 vllm/vllm-openai:v0.4.2 Qwen2-7B-Instruct auto 0.44 Qwen2-7B-Instruct 12288
```

### Diagnosis

```bash
curl http://${URL}:8080/v1/models
curl http://${URL}:8080/v1/completions -H "Content-Type: application/json" -d '{"model": "Meta-Llama-3-8B-Instruct", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'

curl http://${URL}:8000/v1/models
curl http://${URL}:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen2-7B-Instruct", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'
```

## Python Deployment Script for vLLM

In addition to the Bash script, we provide a Python version of the deployment script that offers enhanced usability and additional features such as argument validation, configuration file support, logging, and a dry-run mode.

### Features
- **All arguments are optional**: The script uses default values if arguments are not specified.
- **Named arguments**: All arguments must be provided with their respective flags (e.g., --HOST_PORT 8000).
- **Default value notifications**: If an argument is not specified, the script notifies you and uses the default value.
- **Configuration file support**: You can specify a JSON configuration file containing the arguments.
- **Logging**: Option to log script activities to a file.
- **Dry-run mode**: Option to display the Docker command without executing it.
- **Argument validation**: Ensures that provided arguments are within expected ranges and formats.

### Prerequisites
**Python 3.6 or higher**: Ensure Python is installed on your system. You can check your Python version with python3 --version.


### Usage
To run the Python script, use the following command format:

```bash
python3 deploy_vLLM.py [OPTIONS]
```

Available Options
- --IS_LOCAL_MODEL: Is local model (1 or 0). Default: 1
- --VOLUME_HOST_PATH: Volume host path. Default: /data/models/vLLM_models/
- --VOLUME_CONTAINER_PATH: Volume container path. Default: /mnt/model/
- --HOST_PORT: Host port (1-65535). Default: 8000
- --IMAGE_NAME: Docker image name. Default: vllm/vllm-openai:v0.4.2
- --MODEL_NAME: Model name. Default: NousResearch--Meta-Llama-3-8B-Instruct--own-awq-w_bit_4--GEMM
- --DTYPE: Data type (auto, float32, float16, bfloat16, int8). Default: auto
- --GPU_MEMORY_UTILIZATION: GPU memory utilization (0.0 - 1.0). Default: 0.5
- --SERVED_MODEL_NAME: Served model name (optional)
- --MAX_MODEL_LEN: Max model length (optional)
- --dry-run: Display the Docker command without executing it
- --config: Path to a configuration JSON file
- --log: Path to a log file

### Example Execution
**Using default values:**
```bash
python3 deploy_vLLM.py
```

This will run the script using all default values. The script will inform you about the default values being used.

**Overriding Some Arguments**
```bash
python3 deploy_vLLM.py --HOST_PORT 9000 --MODEL_NAME "meta-llama/Meta-Llama-3.1-8B-Instruct"
```
This will use 9000 as the host port and `meta-llama/Meta-Llama-3.1-8B-Instruct` as the model name. Other arguments will use their default values. The model will be downloaded from HF.

**Dry-Run Mode**
To display the Docker command without executing it:
```bash
python3 deploy_vLLM.py --dry-run
```

### Diagnosis
After deploying the model, you can test the deployment using curl commands:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "CustomModel", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'
```

### Argument Descriptions
- --IS_LOCAL_MODEL: Specifies whether the model is local (1) or needs to be downloaded (0). If set to 1, the script constructs the model path by combining VOLUME_CONTAINER_PATH and MODEL_NAME.

- --VOLUME_HOST_PATH: The path on the host machine where the model data is stored.

- --VOLUME_CONTAINER_PATH: The path inside the Docker container where the model data will be mounted.

- --HOST_PORT: The port on which the Docker container will serve. Ensure that this port is open and not in use.

- --IMAGE_NAME: The Docker image to be used for deployment.

- --MODEL_NAME: The name or path of the model to be deployed.

- --DTYPE: The data type for model operations. Options include auto, float32, float16, bfloat16, int8.

- --GPU_MEMORY_UTILIZATION: Fraction of GPU memory to be utilized by the model. Must be a value between 0.0 (exclusive) and 1.0 (inclusive).

- --SERVED_MODEL_NAME: The name under which the model will be served. If not specified, this argument is omitted.

- --MAX_MODEL_LEN: The maximum model length. If not specified, this argument is omitted.

- --dry-run: When specified, the script displays the Docker command without executing it. Useful for verification.

- --config: Path to a JSON configuration file containing arguments. Command-line arguments override values in the configuration file.

- --log: Path to a log file where script activities will be recorded.

### Notes
- Validation: The script includes validation for arguments to ensure they are within expected ranges and formats.

- Default Values: If an argument is not specified, the script will use the default value and notify you.

- Configuration File Precedence: Command-line arguments override values provided in the configuration file.

- Logging: If a log file is specified, the script will record activities and errors to the file.

- Dry-Run Mode: Use the --dry-run option to preview the Docker command without executing it.

