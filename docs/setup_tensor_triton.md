# **Instructions to set up TensorRT inference for Llava**
## **Repository Structure**
```
llava_inference_trrt
├── docs
│   └── setup_tensor_triton.md
├── engines   
├── llava-1.5-7b-hf   
├── scripts
│   ├── llava_engine_build.py
│   └── test_triton_server.py
└── tensorrtllm_backend
    ├── LICENSE
    ├── README.md
    ├── dockerfile
    ├── docs
    ├── images
    ├── multimodal_ifb
    ├── scripts
    └── tensorrt_llm
```
Carry out all CLI instructions from base directory.

---
## **Setup Swap file for extra RAM if system RAM <2-3x model weights in gb**
```
sudo fallocate -l 32G /mnt/data/swapfile
sudo chmod 600 /mnt/data/swapfile
sudo mkswap /mnt/data/swapfile
sudo swapon /mnt/data/swapfile
```

## **Setup for building TensorRT and TensorRT-LLM Engine**

- Downloading relevant repositories into base directory.
```
sudo apt-get update && sudo apt-get install git-lfs -y
git lfs install

git clone https://github.com/Eros483/tensorrtllm_backend.git
cd tensorrtllm_backend
git submodule update --init --recursive
cd ..

git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf

mkdir -p engines
```
- It is assumed that the relevant scripts are already present in `/scripts` directory.

- Launching Triton Container and engine build
  - Make sure to be present in base directory.
  - Make sure to verify file and check configuration as per requirements.
```
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v $(pwd)/tensorrtllm_backend:/tensorrtllm_backend \
    -v $(pwd)/llava-1.5-7b-hf:/llava-1.5-7b-hf \
    -v $(pwd)/engines:/engines \
    -v $(pwd)/scripts:/workspace \
    nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

# from inside the container image
pip install psutil gputil tritonclient[http] pillow requests

cd /workspace
python3 llava_engine_build.py
```
This concludes engine build, exit container, and return to base directory.

---

## **Building Triton Inference server**
````
docker run --rm -it --net=host \
  --shm-size=2g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus all \
  -v $(pwd)/tensorrtllm_backend:/workspace/tensorrtllm_backend \
  -v $(pwd)/engines:/engines \
  -v $(pwd)/llava-1.5-7b-hf:/llava-1.5-7b-hf \
  -v $(pwd)/scripts:/workspace/tensorrtllm_backend/scripts \
  -w /workspace/tensorrtllm_backend \
  nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 bash

rm -rf multimodal_ifb
cp -r tensorrt_llm/triton_backend/all_models/inflight_batcher_llm/ multimodal_ifb
cp -r tensorrt_llm/triton_backend/all_models/multimodal/ensemble multimodal_ifb/
cp -r tensorrt_llm/triton_backend/all_models/multimodal/multimodal_encoders multimodal_ifb/

export MODEL_NAME="llava-1.5-7b-hf"
export HF_MODEL_PATH="/llava-1.5-7b-hf"
export ENGINE_PATH="/engines/llava1.5/llm"
export MULTIMODAL_ENGINE_PATH="/engines/llava1.5/vision"
export ENCODER_INPUT_FEATURES_DTYPE="TYPE_FP16"
export MAX_BATCH_SIZE=4
export MAX_QUEUE_DELAY=20000
export GPU_FRACTION_USED=0.9

# configuring tensorRT llm backend
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/tensorrt_llm/config.pbtxt \
  triton_backend:tensorrtllm,\
triton_max_batch_size:${MAX_BATCH_SIZE},\
decoupled_mode:False,\
max_beam_width:1,\
engine_dir:${ENGINE_PATH},\
enable_kv_cache_reuse:False,\
batching_strategy:inflight_fused_batching,\
max_queue_delay_microseconds:${MAX_QUEUE_DELAY},\
enable_chunked_context:False,\
encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},\
logits_datatype:TYPE_FP32,\
prompt_embedding_table_data_type:TYPE_FP16,\
kv_cache_free_gpu_mem_fraction:${GPU_FRACTION_USED}

# configuring preprocessing
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/preprocessing/config.pbtxt \
  tokenizer_dir:${HF_MODEL_PATH},\
triton_max_batch_size:${MAX_BATCH_SIZE},\
preprocessing_instance_count:1,\
multimodal_model_path:${MULTIMODAL_ENGINE_PATH},\
engine_dir:${ENGINE_PATH},\
max_num_images:1,\
max_queue_delay_microseconds:${MAX_QUEUE_DELAY}

# configure postprocessing
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/postprocessing/config.pbtxt \
  tokenizer_dir:${HF_MODEL_PATH},\
triton_max_batch_size:${MAX_BATCH_SIZE},\
postprocessing_instance_count:1

# configure ensemble
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/ensemble/config.pbtxt \
  triton_max_batch_size:${MAX_BATCH_SIZE},\
logits_datatype:TYPE_FP32

# configure BLS
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
  triton_max_batch_size:${MAX_BATCH_SIZE},\
decoupled_mode:False,\
bls_instance_count:1,\
accumulate_tokens:False,\
tensorrt_llm_model_name:tensorrt_llm,\
multimodal_encoders_name:multimodal_encoders,\
logits_datatype:TYPE_FP32,\
prompt_embedding_table_data_type:TYPE_FP16

# configure multimodal encoders
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/multimodal_encoders/config.pbtxt \
  triton_max_batch_size:${MAX_BATCH_SIZE},\
multimodal_model_path:${MULTIMODAL_ENGINE_PATH},\
encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},\
hf_model_path:${HF_MODEL_PATH},\
max_queue_delay_microseconds:${MAX_QUEUE_DELAY},\
prompt_embedding_table_data_type:TYPE_FP16
````
Configuration setup is done, can exit container if inference isn't required immediately.

---
## **Sequence of Operations for Inference**
- ### Launching server
    - If outside container:
    ```
    docker run --rm -it --net=host \
      --shm-size=2g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --gpus all \
      -v $(pwd)/tensorrtllm_backend:/workspace/tensorrtllm_backend \
      -v $(pwd)/engines:/engines \
      -v $(pwd)/llava-1.5-7b-hf:/llava-1.5-7b-hf \
      -v $(pwd)/scripts:/workspace/tensorrtllm_backend/scripts \
      -w /workspace/tensorrtllm_backend \
      nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 bash
    ```

    ```
    export PMIX_MCA_gds=hash

    python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py \
    --world_size 1 \
    --model_repo=multimodal_ifb/ \
    --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders \
    --multimodal_gpu0_cuda_mem_pool_bytes 2000000000
    ```

- ### **Inference setup from server**
    - Initial Environment set-up.
    ```
    # replace with your own docker container id using docker ps
    docker exec -it e6ea3c587db5 bash
    ```
    ```
    pip3 install tritonclient[all]
    pip install tabulate
    ```

    - Verification Request to test if setup is working
    ```
    python3 tensorrt_llm/triton_backend/tools/multimodal/client.py \
    --text 'Describe this image in detail.' \
    --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' \
    --request-output-len 50 \
    --model_type llava
    ```

    - Batched Request
    ```
    cd scripts
    python3 test_triton_server.py --batch-size 4 #or vary accordingly
    ```

- **Server metrics**: `curl localhost:8002/metrics`
- **Server Shutdown**: `pkill tritonserver`
