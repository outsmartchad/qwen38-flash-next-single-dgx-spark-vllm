# How Qwen3.8 Flash Next works with vLLM

## How the model works

This recipe serves NVIDIA’s pinned `Qwen3.8-Flash-Next-NVFP4` snapshot at `fc694b54fb0174e0913e6adf86691ef85a4ead47`. Qwen3.8-Flash-Next is a 125B sparse MoE with 6B activated per token, plus 51B of n-gram embedding tables and a 4B MTP module. NVFP4 stores the model weights compactly, FP8 stores the KV cache, and the launcher prepares the packed n-gram PLE table for the runtime. [Qwen model overview](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [pinned checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47)

The 48-layer trunk repeats twelve groups of three Gated DeltaNet layers and one Qwen Sparse Attention layer. Each mixer is followed by a 512-expert MoE with 10 routed experts and one shared expert. QSA selects 512 micro-blocks or 2,048 tokens through its compressed indexer, while GDN carries a fixed-size recurrent state. [Qwen architecture report, §2](https://arxiv.org/html/2608.30320v1#S2) · [vLLM documentation](https://docs.vllm.ai/) · [upstream source revision](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/b7c5da1a9e7448a14b480deee66eda3799b1eb43)

Gated Residual widens the residual stream to four branches with gated reads and writes. The 51B n-gram tables sit outside the accelerator’s hot trunk and are prefetched through the PLE path. The 4B MTP module drafts future tokens, vLLM verifies them, and this backend rebuilds QSA metadata between draft steps. [Qwen model overview](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [Qwen architecture report, §2](https://arxiv.org/html/2608.30320v1#S2) · [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html)

## How this recipe runs it on one DGX Spark

```text
pinned HF snapshot -> PLE packing + mmap
                  -> vLLM OpenAI server
                  -> prefill and FP8 KV allocation
                  -> MTP draft/verify loop -> streamed tokens
```

`serve.sh download` pins the Mia source commit, pulls the digest-pinned vLLM image, downloads the NVIDIA snapshot, and materializes the PLE file. `serve.sh start` runs the upstream launcher with five maximum sequences, a 262,144-token window, MTP ×3, FP8 KV, and a 30 GiB host reserve. vLLM definitely mmaps the PLE table through its CPU offload worker, keeping the n-gram embedding table in host-mapped memory while active MoE and attention work runs on the GPU. The PLE table is 47.68 GiB in the measured profile, the MTP weights are 2.34 GiB, and vLLM reported 8.12 GiB of FP8 KV capacity across 488,656 token positions. [Launcher](../serve.sh) · [running measurements](../results/benchmark-2026-11-21.json)

The decode benchmark used one through five concurrent streaming requests, 400 output tokens, thinking enabled, temperature 0.7, top-p 1, and top-k 20. Best aggregate rates were 29.8, 43.7, 59.5, 76.5, and 78.8 tokens per second. Fresh retrieval prompts measured 1,763 tokens per second at 32,740 input tokens and 1,744 tokens per second at 199,952 input tokens, with exact key recovery. These figures describe this pinned inference profile. [Raw results](../results/benchmark-2026-11-21.json)

PLE offload keeps a large table in host-mapped storage, while GPU memory carries active weights, KV, and runtime work. Five sequences raise aggregate throughput through batching but increase per-request contention and TTFT. A 262,144-token window remains bounded by available unified memory, reserve policy, and active request count. The measured scope is localhost streaming decode and two independent fresh-prompt retrieval runs, with multimodal input and production network serving outside this run. [README runtime notes](../README.md#runtime)
