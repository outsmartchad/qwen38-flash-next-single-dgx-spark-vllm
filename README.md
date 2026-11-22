# Qwen3.8 Flash Next with vLLM on one DGX Spark

vLLM serves the public NVIDIA NVFP4 checkpoint with CPU-offloaded PLE and MTP ×3. The upstream Mia checkpoint returned HTTP 401 during staging. The model API binds to `127.0.0.1:18897`.

[How the model works](docs/how-it-works.md) · [Quick start](#quick-start) · [Runtime](#runtime) · [Raw results](results/benchmark-2026-11-21.json)

## Benchmarks

Measured November 21, 2026 on one NVIDIA GB10 DGX Spark. The benchmark sent direct streaming requests to the localhost model API. Other model servers were paused.

### Decode, prose

SparkDash hash-map prompt with thinking on, temperature 0.7, top-p 1, top-k 20 and 400 generated tokens per request. Each row shows the fastest aggregate decode wave across three repetitions. Per-request decode and TTFT are medians from the same wave. TTFT counts the first streamed reasoning or answer token.

| Streams | Best aggregate tok/s | Per request tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 29.8 | 29.8 | 403 ms |
| 2 | 43.7 | 23.0 | 435 ms |
| 3 | 59.5 | 21.0 | 465 ms |
| 4 | 76.5 | 19.3 | 510 ms |
| 5 | 78.8 | 17.1 | 836 ms |

### Fresh prompt prefill

The input rate divides uncached input tokens by client TTFT. The server reported zero cached tokens on both prompts. The benchmark uses an independent synthetic key for each context size and compares the response with that key.

| Prompt tokens | Input tok/s to first token | TTFT | Retrieval |
| ---: | ---: | ---: | --- |
| 32,740 | 1,763 | 18.58 s | Exact key |
| 199,952 | 1,744 | 114.63 s | Exact key |

## Runtime

- Pinned ARM64 vLLM image and NVIDIA target snapshot `fc694b54fb0174e0913e6adf86691ef85a4ead47` (123.57 GiB).
- MTP ×3, FP8 KV, five maximum sequences and a configured 262,144-token model window.
- The pinned recipe sets a 30 GiB host reserve. The CPU offload worker mmaped the 47.68 GiB PLE table. vLLM reported 8.12 GiB and 488,656 token positions for its FP8 KV cache.
- Initial server startup took 752 seconds after the one-time PLE packing. QSA rebuilt attention metadata between MTP draft steps on this backend.
- Peak GPU memory was 87.6 GiB across sampled inference-container processes. The Docker memory limit was 101.0 GiB.

## Quick start

Use a single DGX Spark with Docker, NVIDIA Container Toolkit, the Hugging Face CLI, Python 3 and curl. Allow roughly 175 GiB for checkpoints and the packed PLE table for model files plus room for the runtime image. Install the Hugging Face CLI with Xet support for the packed 53.7 GB shard. The launcher materializes the pinned snapshot and records its offline ref so the model-only container mount can resolve every shard. Stop other model servers before running this recipe.

```bash
./serve.sh download
./serve.sh start
# start waits for localhost readiness
```

When the localhost API is ready, run the direct benchmark on the Spark.

```bash
curl -fsS http://127.0.0.1:18897/v1/models
export SPARK_BENCHMARK_URL=http://127.0.0.1:18897
export SPARK_BENCHMARK_MODEL=Qwen3.8-Flash-Next-vLLM
python3 benchmark.py > decode-local.json
python3 benchmark.py 32768 2026144868 > prefill-32768-local.json
python3 benchmark.py 200000 2026312100 > prefill-200000-local.json
./serve.sh stop
```

This recipe builds on [Mia AI Lab's pinned deployment](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/b7c5da1a9e7448a14b480deee66eda3799b1eb43). Our scripts, documentation and synthetic fixtures use the [MIT license](LICENSE). Upstream models and runtime components retain their own terms.
