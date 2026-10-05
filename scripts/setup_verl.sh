#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.

# Install VERL training dependencies into a managed Python environment.
set -euo pipefail

FLASH_ATTN_VERSION="2.8.3"

# Intel XPU stack. torch pins follow SGLang's XPU guide, which is what the
# rollout engine is built against; verl 0.9.1 is the floor for the platform
# plugin layer that verl-hardware-plugin registers PlatformXPU with.
XPU_VERL_VERSION="0.9.1"
XPU_TORCH_SPEC="torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu"
XPU_TORCH_INDEX="https://download.pytorch.org/whl/xpu"
XPU_HARDWARE_PLUGIN="git+https://github.com/verl-project/verl-hardware-plugin.git"
# Published wheels are CUDA-only, so the XPU backend is built from source.
XPU_MEMORY_SAVER="git+https://github.com/fzyzcjy/torch_memory_saver.git@a5c99f11b18ebb8e9fda71a68812e476ae49e417"

usage() {
    echo "Usage: bash scripts/setup_verl.sh <0.7.1|0.8.0> <cu129|cu130> [venv_path]"
    echo "       bash scripts/setup_verl.sh xpu [venv_path]"
}

if [ "${1:-}" = "xpu" ]; then
    if [ "$#" -gt 2 ]; then
        usage
        exit 1
    fi
    PYTHON_BIN="${2:-.venv}/bin/python"
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: expected Python executable not found: $PYTHON_BIN"
        echo "Run 'uv sync' from the project root first, or pass a venv path."
        exit 1
    fi

    echo "Using Python executable: $PYTHON_BIN"
    echo "Using VERL version: $XPU_VERL_VERSION (Intel XPU)"

    uv pip install --python "$PYTHON_BIN" pip

    # Pin torch first and from the XPU index only: resolving it alongside verl
    # lets the PyPI CPU build win, which silently drops torch.xpu.
    # shellcheck disable=SC2086
    uv pip install --python "$PYTHON_BIN" $XPU_TORCH_SPEC --index-url "$XPU_TORCH_INDEX"

    # dpctl is not optional: Ray counts Intel GPUs through it
    # (ray/_private/accelerators/intel_gpu.py), and reports zero devices without
    # it, which leaves verl's resource pool with nothing to place workers on.
    uv pip install --python "$PYTHON_BIN" \
        "verl==$XPU_VERL_VERSION" \
        "$XPU_HARDWARE_PLUGIN" \
        dpctl \
        --extra-index-url "$XPU_TORCH_INDEX" \
        --index-strategy unsafe-best-match

    # Backs rollout sleep/wake. verl asks SGLang for enable_memory_saver
    # unconditionally, and SGLang raises if the package is missing.
    TMS_PLATFORM=xpu uv pip install --python "$PYTHON_BIN" --no-build-isolation "$XPU_MEMORY_SAVER"

    # No flash-attn: it builds against the CUDA runtime. verl imports it only for
    # the Qwen2-VL/GLM4V paths, so text models are unaffected.
    echo
    echo "Installed the XPU training stack. Two steps remain, both documented in"
    echo "docs/00-installation.md:"
    echo "  1. Build SGLang for XPU from source -- there is no published wheel,"
    echo "     and it must link the local oneAPI runtime."
    echo "  2. Export VERL_PLATFORM=intel and source the oneAPI environment"
    echo "     before training, so verl selects PlatformXPU and oneCCL loads."
    exit 0
fi

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
    exit 1
fi

VERL_VERSION="$1"
CUDA_VARIANT="$2"
VENV_PATH="${3:-.venv}"
PYTHON_BIN="$VENV_PATH/bin/python"

if [ "$VERL_VERSION" != "0.7.1" ] && [ "$VERL_VERSION" != "0.8.0" ]; then
    usage
    exit 1
fi

if [ "$CUDA_VARIANT" != "cu129" ] && [ "$CUDA_VARIANT" != "cu130" ]; then
    usage
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: expected Python executable not found: $PYTHON_BIN"
    echo "Run 'uv sync' from the project root first, or pass a venv path."
    exit 1
fi

if [ "$VERL_VERSION" = "0.7.1" ]; then
    VLLM_VERSION="0.12.0"
else
    VLLM_VERSION="0.22.0"
fi

echo "Using Python executable: $PYTHON_BIN"
echo "Using VERL version: $VERL_VERSION"
echo "Using CUDA wheel variant: $CUDA_VARIANT"

uv pip install --python "$PYTHON_BIN" pip

if [ "$VERL_VERSION" = "0.7.1" ]; then
    uv pip install --python "$PYTHON_BIN" \
        "vllm==$VLLM_VERSION" "verl==$VERL_VERSION" \
        --torch-backend="$CUDA_VARIANT" \
        --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$CUDA_VARIANT" \
        --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
        --index-strategy unsafe-best-match
else
    uv pip install --python "$PYTHON_BIN" \
        "vllm==$VLLM_VERSION" \
        --torch-backend="$CUDA_VARIANT" \
        --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$CUDA_VARIANT" \
        --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
        --index-strategy unsafe-best-match

    uv pip install --python "$PYTHON_BIN" "verl[mcore]==$VERL_VERSION"
fi

# flash-attn is built from source against torch's CUDA runtime. For cu130 the system
# CUDA toolkit (often 12.x) is too old, so install a matching CUDA 13.0 pip toolchain
# and point the build at it. Versions are pinned to 13.0.x so nvcc's version matches
# torch's CUDART (13000); a mismatched minor (e.g. 13.3) trips cccl's compatibility
# check ("CUDA compiler and CUDA toolkit headers are incompatible").
if [ "$CUDA_VARIANT" = "cu130" ]; then
    uv pip install --python "$PYTHON_BIN" \
        "nvidia-cuda-nvcc>=13.0,<13.1" \
        "nvidia-cuda-crt>=13.0,<13.1" \
        "nvidia-nvvm>=13.0,<13.1" \
        "nvidia-cuda-cccl>=13.0,<13.1" \
        "nvidia-cuda-runtime>=13.0,<13.1"

    CUDA_HOME="$("$PYTHON_BIN" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
    ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0}"
fi

FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PYTHON_BIN" \
    "flash-attn==$FLASH_ATTN_VERSION" \
    --force-reinstall \
    --no-cache \
    --no-binary flash-attn \
    --no-build-isolation \
    --no-deps
