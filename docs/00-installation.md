# Installation

This guide sets up a single-node environment for Agent Lightning v1.0. After completing it, you can run single-machine training jobs.

Before getting started, install `uv` and a GPU runtime: NVIDIA CUDA `12.9` or `13.0`, or Intel oneAPI for Intel GPUs (see [Intel XPU](#intel-xpu)).

#### Step 1: UV Sync

From the project root, run:

```bash
cd <this-repo>
uv sync
```

This installs the base Python environment into `.venv` under the project root.

#### Step 2: Install `verl` and FlashAttention

Agent Lightning uses `verl` as its training backend. The compatible versions of `verl`, `vllm`, and `torch` are tightly coupled, and installing `flash-attn` can also be error-prone. We recommend using `scripts/setup_verl.sh` to install the tested, pinned GPU stack and build `flash-attn` from source.

Pass the `verl` version and CUDA wheel variant explicitly. The script supports `verl==0.7.1` or `verl==0.8.0`, and CUDA wheel variant `cu129` or `cu130`. We recommend CUDA `13.0` with `verl==0.8.0`:

```bash
source .venv/bin/activate
bash scripts/setup_verl.sh 0.8.0 cu130

# or

bash scripts/setup_verl.sh 0.7.1 cu129
```

For `verl==0.7.1`, the script installs `vllm==0.12.0`. For `verl==0.8.0`, it installs `vllm==0.20.2` first, then installs `verl==0.8.0`. Both paths build `flash-attn==2.8.3` locally against the selected environment. Depending on the number of CPU cores available, the script can take 10-30 minutes to complete.

#### Step 3: W&B Login

By default, all tasks upload logs and trajectories to Weights & Biases. Log in to W&B before running a task:

```bash
uv run wandb login
```

## Intel XPU

Training also runs on Intel GPUs (Arc, Arc Pro, Data Center GPU Max). The
differences from the CUDA path above are the device plugin, the rollout engine,
and the absence of FlashAttention.

#### Step 1: UV Sync

Unchanged -- `uv sync` from the project root.

#### Step 2: Install `verl` and the Intel XPU plugin

```bash
source .venv/bin/activate
bash scripts/setup_verl.sh xpu
```

This installs torch from PyTorch's XPU index, `verl==0.9.1`,
[verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin)
(which registers verl's `PlatformXPU` -- device `xpu`, vendor `intel`, `xccl`
collectives, `ZE_AFFINITY_MASK` visibility), `dpctl`, and `torch_memory_saver`
built for XPU. It does not install FlashAttention, which only builds against the
CUDA runtime; verl imports it for the Qwen2-VL/GLM4V paths only.

`dpctl` is not optional. Ray counts Intel GPUs through it, and without it
advertises zero devices, which leaves verl's resource pool unable to place any
worker.

#### Step 3: Build SGLang for XPU

The rollout engine is SGLang rather than vLLM, and there is no published XPU
wheel -- it has to be built against your local oneAPI. Follow
[SGLang's Intel GPU guide](https://docs.sglang.io/hardware-platforms/xpu),
which also covers the `--enable-memory-saver` backend that rollout sleep/wake
relies on.

Weight sync needs an SGLang build that can serialize XPU tensors between
processes (`sgl-project/sglang#40720`); without it,
`update_weights_from_tensor` fails with `_share_fd_: only available on CPU`,
since PyTorch has no XPU tensor IPC.

#### Step 4: Environment

```bash
source /opt/intel/oneapi/setvars.sh   # oneCCL; xccl init fails without it
export VERL_PLATFORM=intel            # or let verl auto-detect
```

#### Step 5: Training config

Start from any example under `examples/` and change three keys:

| Key | Value | Why |
|---|---|---|
| `actor_rollout_ref.rollout.name` | `sglang` | vLLM's XPU backend is not wired into verl's RL rollout path |
| `actor_rollout_ref.actor.strategy` | `fsdp` or `fsdp2` | Megatron has no XPU backend, so the `*_megatron` and `*_moe` examples do not apply |
| `trainer.n_gpus_per_node` | your XPU count | Ray reports Intel GPUs as `GPU`, so this key is unchanged in meaning |

SGLang picks its own attention backend on XPU (triton). To use Intel's
optimized kernels instead, set
`actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=intel_xpu`
together with a `page_size` of 32, 64 or 128.

Peak host memory matters more than on CUDA: with no device-to-device IPC, each
weight-sync payload is staged through `/dev/shm`, one copy per rollout TP rank.
Size `/dev/shm` (`--shm-size` in containers) for `tp_size * bucket_bytes`.
