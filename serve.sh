#!/usr/bin/env bash
set -euo pipefail

source_url=https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark.git
revision=b7c5da1a9e7448a14b480deee66eda3799b1eb43
image=vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
target=nvidia/Qwen3.8-Flash-Next-NVFP4
target_revision=fc694b54fb0174e0913e6adf86691ef85a4ead47
project="$HOME/.cache/spark-flash-next-vllm"
source_dir="$project/upstream"
name=spark-catalog-flash-vllm
export HF_HOME="$project/huggingface" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
snapshot="$HF_HOME/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/$target_revision"

case "${1:-}" in
  download)
    if [[ ! -d "$source_dir/.git" ]]; then
      mkdir -p "$source_dir"
      git -C "$source_dir" init -q
    fi
    if ! git -C "$source_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
      git -C "$source_dir" fetch --depth=1 "$source_url" "$revision"
    fi
    git -C "$source_dir" checkout --detach "$revision"
    docker pull --platform linux/arm64 "$image"
    HF_HUB_DISABLE_XET=0 hf download "$target" --revision "$target_revision" --local-dir "$snapshot"
    mkdir -p "$snapshot/../../refs"
    printf '%s' "$target_revision" > "$snapshot/../../refs/main"
    (
      umask 077
      cat > "$source_dir/.env.recipe.tmp" <<'CONFIG'
IMAGE=vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
TP1_MODEL_ID=nvidia/Qwen3.8-Flash-Next-NVFP4
TP1_CONTAINER_NAME=spark-catalog-flash-vllm
SERVED_MODEL_NAME=Qwen3.8-Flash-Next-vLLM
BIND=127.0.0.1
PORT=18897
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=5
MTP_NUM_SPECULATIVE_TOKENS=3
HOST_RESERVE_GIB=30
PLE_OFFLOAD=true
PLE_GIB=47.68
MTP_WEIGHTS_GIB=2.34
KV_CACHE_DTYPE=fp8
READY_TIMEOUT_S=1800
ABLIT=0
CONFIG
      mv -f "$source_dir/.env.recipe.tmp" "$source_dir/.env"
    )
    ;;
  start)
    test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
    test -f "$snapshot/model-fp8-mtp-ple.safetensors"
    test ! -L "$snapshot/model-fp8-mtp-ple.safetensors"
    cd "$source_dir"
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u API_KEY -u EXTRA_DOCKER_ARGS -u EXTRA_VLLM_ARGS \
      HF_HUB_OFFLINE=1 ABLIT=0 BIND=127.0.0.1 PORT=18897 \
      MAX_MODEL_LEN=262144 MAX_NUM_SEQS=5 MTP_NUM_SPECULATIVE_TOKENS=3 \
      HOST_RESERVE_GIB=30 PLE_GIB=47.68 MTP_WEIGHTS_GIB=2.34 bash start.sh
    ;;
  stop)
    docker stop "$name"
    docker rm "$name"
    ;;
  *)
    echo "Usage: $0 {download|start|stop}" >&2
    exit 2
    ;;
esac
