# Qwen3.8-27B on RTX 3090s — one-command sglang container

Fastest validated recipe for running **Qwen/Qwen3.8-27B** (hybrid GDN vision-language model) on
Ampere consumer GPUs with [SGLang](https://github.com/sgl-project/sglang):

- **Target**: [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) — W4A16 INT4 (Marlin kernels; FP8/NVFP4 have no fast path on Ampere)
- **Speculative decoding**: [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) 1.36B DSpark draft — accept length up to ~4 on coding/greedy content
- **Everything pinned**: sglang `2e7c85da6` (git), torch `2.13.0+cu130`, triton `3.7.1`, flashinfer `0.6.17`, pip nvcc `13.3.73` — the exact stack benchmarked below
- Image + video input supported (multimodal)

## Run (single command)

```bash
docker run --gpus all --shm-size 32g -p 8000:8000 \
  -v $PWD/models:/models \
  ghcr.io/0xsero/qwen38-3090-sglang:latest        # TP=1 by default
```

Weights (~26 GB) download automatically into `./models` on first start. Server comes up in
~5 min (kernel JIT on first boot).

Pick your GPU count:

```bash
# 2x 3090
docker run --gpus all --shm-size 32g -p 8000:8000 -v $PWD/models:/models \
  -e TP=2 ghcr.io/0xsero/qwen38-3090-sglang:latest

# 4x 3090
docker run --gpus all --shm-size 32g -p 8000:8000 -v $PWD/models:/models \
  -e TP=4 ghcr.io/0xsero/qwen38-3090-sglang:latest
```

Use it:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"hello /no_think"}],"max_tokens":100}'
```

## Environment

| var | default | values | notes |
|---|---|---|---|
| `TP` | `1` | `1 \| 2 \| 4` | tensor parallel GPUs; presets tuned per count |
| `DSPARK` | `1` | `1 \| 0` | DSpark speculative decoding (recommended: greedy/coding workloads gain most) |
| `PORT` | `8000` | | |
| `MODELS_DIR` | `/models` | | mount a volume here to cache weights |

## Measured on 4× RTX 3090 (96 GB, PCIe, no NVLink, TP=4)

| workload | speed |
|---|---|
| coding, greedy | **91–128 tok/s** single stream (DSpark accept len 2.8–4.1) |
| prose, greedy | 71 tok/s |
| 8 concurrent streams | 226 tok/s aggregate |
| prefill | ~1.3–1.6k tok/s |
| KV pool | 275k tokens (bf16), max 9 concurrent requests (GDN state bound) |

1× / 2× numbers: see the dashboard section below (TP2 ≈ half TP4 bandwidth; TP1 fits in 24 GB
with a reduced mamba state pool).

## Why these choices (Ampere specifics)

- **AWQ INT4 W4A16** beats FP8 on 3090s: Ampere has no FP8 tensor cores, so FP8 only saves
  memory while paying dequant; INT4 Marlin kernels cut weight bandwidth ~4× → ~2× decode.
- **DSpark acceptance depends on target quant + content**: measured best with the AWQ target
  on greedy coding content; sampled prose drafts worse (~1.7 accept len).
- **GDN hybrid**: the model is 48 Gated-DeltaNet layers + 16 gated-attention layers;
  `--mamba-radix-cache-strategy extra_buffer` is required (wrong strategies deadlock graph capture),
  and the mamba state pool sizes concurrency, not just KV.
- flashinfer's bundled cccl version guard is relaxed for nvcc 13.3 (patched in image).

## Optional: PCIe P2P (more speed)

On multi-3090 PCIe rigs, GPU P2P is driver-gated. The [aikitoria fork](https://github.com/aikitoria/open-gpu-kernel-modules)
of the open kernel modules re-enables BAR1 P2P (`610.57.04-p2p` branch). Requires BIOS
Resizable-BAR + Above-4G enabled and `iommu=pt`. Not part of the container (host kernel level).

## License

MIT
