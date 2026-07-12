# Qwen3.6 local inference lab — RTX 3090 × 4

> Originally forked from [`noonghunna/qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) (single-3090 vLLM recipe for the dense 27B). This repo vendors that recipe under `qwen36-27b-single-3090/`, fixes its post-2026-04-24 breakage, and extends it with multi-GPU 27B variants (`tp2`, `tp4`, `tp4-2`) and a full Qwen3.6-35B-A3B MoE stack (vLLM AWQ + llama.cpp GGUF + MTP-3 speculative-decode). The original repo's history is dropped because the local edits diverge significantly; commits, attributions, and rationale for every change are documented in [`FINDINGS.md`](./FINDINGS.md).

Local OpenAI-compatible endpoints for **Qwen3.6-27B (dense)** and **Qwen3.6-35B-A3B (MoE)** on a workstation with 4× RTX 3090. Documents which inference engine and quantization is the right choice for each GPU count and workload, with measured TPS and GPU-saturation numbers backing every recommendation. Full technical writeup in [`FINDINGS.md`](./FINDINGS.md).

Three launcher scripts, multiple endpoints (run independently or side-by-side, GPU layout permitting):

| Model | Launcher | Endpoint | Model name (OpenAI API `model`) |
|---|---|---|---|
| Qwen3.6-27B (dense) | [`qwen.sh`](./qwen.sh) | `http://localhost:8020/v1` | `qwen3.6-27b-autoround` |
| Qwen3.6-35B-A3B (MoE) | [`qwen-moe.sh`](./qwen-moe.sh) | `http://localhost:8021/v1` (vLLM) or `:8022/v1` (llama.cpp) | `qwen3.6-35b-a3b-awq` or `qwen3.6-35b-a3b-gguf` |
| Gemma-4-26B-A4B (MoE) + Gemma-4-E4B | [`gemma.sh`](./gemma.sh) | `:8023/v1` (vLLM AWQ+DFlash) / `:8024/v1` (llama.cpp GGUF) / `:8025/v1` (E4B) | `gemma-4-26b-a4b-awq` / `gemma-4-26b-a4b-gguf` / `gemma-4-e4b` |
| DiffusionGemma-26B-A4B (MoE, text diffusion) | [`diffusion-gemma.sh`](./diffusion-gemma.sh) | `http://localhost:8031/v1` | `diffusiongemma-26b-a4b-nvfp4` |

The Gemma stack mirrors the recipe from [this LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1t796qe/gemma_4_26b_hits_600_toks_on_one_rtx_5090/) (vLLM AWQ + DFlash speculative decoding, ~578 tok/s on a 5090) plus the 3090-specific `llama.cpp + ngram-mod` recipe surfaced in the same thread (~130 tok/s, full 262K ctx). Full benchmarks, replication notes, and the four compose-file fixes needed to make the recipe run on a 3090 are documented in [`GEMMA_FINDINGS.md`](./GEMMA_FINDINGS.md).

The DiffusionGemma stack runs Google's [block-diffusion 26B-A4B](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) (released 2026-06-10) on a single 3090 at ~200 TPS decode — the fastest decode on this rig — via the unmerged [vLLM PR #45163](https://github.com/vllm-project/vllm/pull/45163) CI image and the RedHatAI NVFP4 quant. Day-zero benchmarks and the Ampere quant/parallelism failure modes (Marlin dim alignment kills TP=2; PP unimplemented) are documented in [`DIFFUSIONGEMMA_FINDINGS.md`](./DIFFUSIONGEMMA_FINDINGS.md).

## Headline numbers on this rig

| Mode | Engine / quant | GPUs | Max ctx | Decode TPS | Vision | When to use |
|---|---|---|---|---|---|---|
| `qwen.sh start tp2` | vLLM / Lorbus AutoRound INT4 | 2× 3090 | 100K | **110–112 narr / 141–147 code** | ✅ | **27B sweet spot** — best dense quality on this rig, prefix cache ON |
| `qwen.sh start tp2-mtp` | vLLM / AutoRound + MTP-4 + no-prefix-cache | 2× 3090 | 100K | **117 narr / 155–170 code** | ✅ | 27B max throughput (no prefix cache) |
| `qwen.sh start tp4` | vLLM / AutoRound INT4 | 4× 3090 | **500K** | 88–97 / 117 | ✅ | only when you need >100K ctx — single session | | `qwen.sh start tp4-2` | vLLM / AutoRound INT4 | 4× 3090 | **500K** | ~90 per session (114 combined) | ✅ | **two concurrent sessions** — zero-overhead parallel decode |
| `qwen-moe.sh start gguf` | llama.cpp / Unsloth IQ4_XS GGUF | 1× 3090 | 128K | **115–133** | ❌ | **MoE single-GPU sweet spot** — 96% SM util |
| `qwen-moe.sh start tp2` | vLLM / cyankiwi AWQ-INT4 | 2× 3090 | 200K | **149** | ✅ | MoE with vision + tools |
| `qwen-moe.sh start tp2-mtp` | vLLM / AWQ + MTP-3 + no-prefix-cache | 2× 3090 | 200K | **165–191 narr / 221–226 code** | ✅ | MoE max throughput (no prefix cache) |
| `gemma.sh start gguf` | llama.cpp / Unsloth UD-Q4_K_XL GGUF + ngram-mod spec | 1× 3090 | **262K** | **127–131 flat** | ✅ | **Gemma-4 26B-A4B sweet spot** — 95% SM util, 130 TPS at 0K → 120 at 17K → 104 at 65K |
| `gemma.sh start tp1` | vLLM / cyankiwi AWQ + z-lab DFlash-13 | 1× 3090 | 20K | 95 narr / **179 code** / 117 explain | ✅ | Gemma-4 26B-A4B short-ctx code generation only — DFlash collapses past ~16K (≈50 TPS at 17K) |
| `diffusion-gemma.sh start` | vLLM PR #45163 / RedHatAI NVFP4 (block diffusion) | 1× 3090 | 16K | **195–221 narr / 159–189 code** | ❓ | **fastest decode on this rig** — quality below autoregressive Gemma-4; speed-critical interactive work |

Picking between dense 27B and MoE 35B-A3B at the same TP=2: the **27B wins every Qwen-published quality benchmark** by 1–10 pts (MMLU-Pro 86.2 vs 85.2, GPQA 87.8 vs 86.0, SWE-bench 77.2 vs 73.4, AIME 94.1 vs 92.7, Terminal-Bench 59.3 vs 51.5, …). The MoE wins on **speed** (~2× the TPS at the same TP=2 budget). For coding tasks where quality matters, 27B is the default; for fast/cheap calls, the MoE.

## Quick start

```fish
cd ~/projects/qwen3.6

# 27B (dense) on 2 GPUs — recommended default
./qwen.sh start tp2

# MoE on 1 GPU — recommended single-GPU mode
./qwen-moe.sh start gguf

# MoE on 2 GPUs with MTP — fastest dense throughput
./qwen-moe.sh start tp2-mtp

# 27B on 4 GPUs with 2 concurrent sessions (pi multi-session setup)
./qwen.sh start tp4-2

# Gemma-4 26B-A4B on 1 GPU — llama.cpp ngram-mod, 130 TPS, full 262K ctx
./gemma.sh start gguf

# DiffusionGemma 26B-A4B on 1 GPU — block-diffusion decode, ~200 TPS
./diffusion-gemma.sh start

./qwen.sh status            # 27B status
./qwen-moe.sh status        # MoE status
./gemma.sh status           # Gemma status
./qwen.sh modes             # list all 27B variants with descriptions
./qwen-moe.sh modes         # list all MoE variants
./gemma.sh modes            # list all Gemma variants
./qwen.sh logs              # follow logs
./qwen.sh stop
```

Both endpoints are OpenAI-compatible; any non-empty API key is accepted. The MoE on `gguf` mode pins to GPU 0 by default; the 2-GPU modes share GPUs 0,1 by default (and they conflict, so they don't run simultaneously without re-pinning `CUDA_VISIBLE_DEVICES`).

## Variant matrix

### 27B (dense), via `qwen.sh`

| Variant   | GPUs | Max ctx | TPS (narr / code) | Vision | Tools | KV dtype           | When to use |
|-----------|------|---------|-------------------|--------|-------|--------------------|-------------|
| `default` | 1    | 20 K    | 70–72 / 91        | yes    | yes   | `fp8_e5m2`         | quick chat / vision, frees 3 GPUs |
| `text`    | 1    | 75 K    | 68–72 / 92        | no     | yes   | `fp8_e5m2`         | long documents, code review, no images |
| `longctx` | 1    | 125 K   | 30–33 / 42        | yes    | yes   | `turboquant_3bit_nc` | full-repo + vision on a single card |
| **`tp2`** | 2    | 100 K   | **110–112 / 141–147** | yes | yes   | `fp16` (FA2)       | **TPS sweet spot — recommended default** |
| `tp2-mtp` | 2    | 100 K   | **117 / 155–170** | yes    | yes   | `fp16` (FA2)       | max single-shot throughput — prefix cache OFF, MTP k=4 |
| `tp4`     | 4    | 500 K   | 88–97 / 117       | yes    | yes   | `fp8_e5m2`         | only when you need >100K ctx — needle verified at **470K** (475 s) | | `tp4-2`   | 4    | 500 K   | ~90 per session (114 combined) | yes    | yes   | `fp8_e5m2`         | **two concurrent sessions** — parallel decode with ~1–2% per-session slowdown |

### 35B-A3B (MoE), via `qwen-moe.sh`

| Variant   | Engine / quant            | GPUs | Max ctx | TPS                              | Vision | When to use |
|-----------|---------------------------|------|---------|----------------------------------|--------|-------------|
| **`gguf`** | llama.cpp / IQ4_XS GGUF   | 1    | 128 K   | **115–133**                      | no     | **single-GPU MoE — recommended** |
| `tp1`     | vLLM / AWQ-INT4           | 1    | 20 K    | 18 (launch-bound, see below)     | no     | kept for posterity, do not use |
| `tp2`     | vLLM / AWQ-INT4           | 2    | 200 K   | **149**                          | yes    | MoE with vision/tools and prefix cache |
| **`tp2-mtp`** | vLLM / AWQ + MTP-3 + no-prefix-cache | 2 | 200 K | **165–191 narr / 221–226 code** | yes | **MoE max throughput, no prefix cache** |

### Gemma-4-26B-A4B (MoE) + Gemma-4-E4B, via `gemma.sh`

Recipe sourced from [this LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1t796qe/gemma_4_26b_hits_600_toks_on_one_rtx_5090/) (vLLM AWQ + DFlash on a 5090) and the 3090-specific reply by `coder543` in the same thread (llama.cpp + ngram-mod). Both measured on this rig — see findings below.

| Variant   | Engine / quant                                 | GPUs | Max ctx | TPS (narr / code / explain) | Vision | When to use |
|-----------|------------------------------------------------|------|---------|-----------------------------|--------|-------------|
| **`gguf`** | llama.cpp / Unsloth UD-Q4_K_XL + ngram-mod    | 1    | **262 K** | **131 / 129 / 129**       | yes (mmproj BF16) | **Gemma-4 sweet spot on this rig** — 95% SM util, flat across workloads, 120 TPS at 17K ctx |
| `tp1`     | vLLM / cyankiwi AWQ + z-lab DFlash-13          | 1    | 20 K    | 95 / **179** / 117          | yes    | short-ctx code generation only — DFlash collapses to ~50 TPS past 16K input, narrative is *slower* than gguf |
| `e4b`     | vLLM / bf16 Gemma-4-E4B-it                     | 1    | 128 K   | (small/fast variant, not benchmarked yet) | yes | quick chat / agent helper |

DFlash on a 3090 is **workload-dependent**:
- 26B-A4B AWQ weights leave only ~2.6 GB for KV after the DFlash draft + cudagraphs → had to cap `--max-model-len` at 20 K (Reddit ran 32 K on a 32 GB 5090).
- DFlash drafts get accepted heavily on structured tokens (code: 179 TPS, +40% vs gguf) but rejected on prose (narrative: 95 TPS, -27% vs gguf). Net loss on most workloads.
- Acceptance also drops with context length — by 17 K input, decode is roughly 50 TPS, vs gguf's 120 TPS.
- `gguf` is the safe default. Use `tp1` only if you know you're generating structured output at short context.

## Tensor-parallel scaling on this rig

Doubling cards 1→2 gives ~30% decode speedup; doubling 2→4 gives essentially nothing. Decode is bandwidth-bound — every transformer layer all-reduces, and on consumer 3090s without NVLink the all-reduce traverses host memory. TP=2 is one round-trip; TP=4 is a tree-reduce with two hops and four-card PCIe contention. The compute savings from sharding wider don't beat the comm overhead. **`tp2` is the right default for TPS; `tp4` exists only for the +400K context.**

This depends on PCIe topology — a rig split across NUMA nodes would pay an extra QPI/UPI hop on every all-reduce and the curve would tilt further toward "lower TP is better." All 4 cards here are on the same NUMA node (`nvidia-smi topo -m` → `NODE` everywhere).

## Why single-GPU MoE = llama.cpp, not vLLM

The vLLM/AWQ MoE tp1 mode is included as a documented **anti-pattern**: 18 TPS on the same hardware where llama.cpp/GGUF gets 115–133. Direct GPU profiling explains:

| Stack (1× 3090, MoE) | SM util | Mem BW util | Power | TPS |
|---|---|---|---|---|
| vLLM AWQ-INT4 (eager forced) | **22%** | 12% | 142 W | **18.6** |
| **llama.cpp IQ4_XS GGUF** | **96%** | 52% | **317 W** | **115–133** |

vLLM's AWQ-INT4 takes 21.56 GB on a 24 GB 3090 — leaves only ~0.7 GB for activations + KV. Both CUDA graphs (~0.7 GB) and torch.compile cache (~0.7 GB) are blocked → `--enforce-eager` is forced → kernel-launch overhead dominates (78% of decode time waiting for the next launch). MoE expert routing makes this dramatically worse than the dense 27B because each layer dispatches many small expert kernels.

The fix isn't tuning vLLM further — it's using a smaller-footprint quant. llama.cpp + Unsloth's IQ4_XS GGUF (~17 GB resident) leaves ~5 GB headroom for KV pool and compute graphs and the GPU saturates. Same model, same hardware, ~7× more throughput. Per the LocalLLaMA recipe thread that motivated this: *"vllm is not going to run on a single 3090 with this model"* — confirmed.

## 2026-07-12: the "onegraph" fixes — +19–39 % on every TP=2 mode

Porting the lossless lesson from the [HF/Google Gemma Challenge](https://huggingface.co/spaces/agent-collaborations/gemma-collab-lessons) ("onegraph": kill kernel-launch overhead in the MTP spec-decode path via full CUDA-graph capture) surfaced three stacked fixes, all shipped in the compose files:

1. **`--attention-backend FLASH_ATTN` + fp16 KV** — with MTP enabled, the old fp8_e5m2 KV forced the FlashInfer backend, which silently downgraded the *whole model* to piecewise cudagraphs. FA2 keeps full decode graphs alive; full-graph capture alone is ~2× decode TPS at fixed kernels on these launch-bound PCIe 3090s.
2. **`PYTORCH_CUDA_ALLOC_CONF` without `expandable_segments:True`** — expandable segments break custom-allreduce IPC ([vllm #42609](https://github.com/vllm-project/vllm/issues/42609)); since the BAR1 P2P driver, this crashed *every* TP≥2 mode at boot (`custom_all_reduce.cuh:455`). With the flag removed, vLLM's CUSTOM allreduce works over P2P: +10 % code TPS.
3. **MTP k=4 on the dense `tp2-mtp`** — with the verify pass this cheap, k=4 beats the old k=3 on code (+8 %) at flat narrative.

Result: dense `tp2` 91–96/114–120 → **110–112/141–147**, dense `tp2-mtp` 96/124 → **117/155–170**, MoE `tp2-mtp` → **179/224** (narr/code TPS). Full A/B ladder and attribution controls in [`FINDINGS.md`](./FINDINGS.md) § 2026-07-12. *Caveat: measured while the host was under load from other services — relative deltas are solid (consistent across all A/B pairs), absolute numbers need re-validation on an idle box.*

## Why `tp2-mtp` works (and `tp2` is still the safe default)

vLLM's MTP speculative decoding (the "method=mtp" path that re-uses the target model's hidden states for drafting) is a +20–77% decode-rate win on **2× 3090** when paired with `--no-enable-prefix-caching`. K-sweep on this rig:

| Config       | narrative-story | code-mergesort | narrative-explain |
|--------------|-----------------|----------------|-------------------|
| baseline (no MTP, prefix cache ON) | 149 | 149 | 149 |
| MTP k=1 | 162 (+8%) | 178 (+19%) | 165 (+11%) |
| MTP k=2 | 182 (+22%) | 229 (+54%) | 191 (+28%) |
| **MTP k=3** ← `tp2-mtp` | **179 (+20%)** | **264 (+77%)** | **200 (+34%)** |
| MTP k=4 | 171 (+15%) | 276 (+85%) | 190 (+27%) |

`k=3` is the cross-prompt sweet spot. `k=4` keeps gaining on code (276) but starts regressing on narrative — verify-cost dominates beyond k=3 for natural-language acceptance patterns.

**Tradeoff:** prefix caching is OFF in this mode. Per [vllm #38182](https://github.com/vllm-project/vllm/issues/38182), MTP drops cache hit rate ≈92% → ≈71%, so cache-loss penalty masks the compute speedup when prefix caching is ON. For pure single-shot throughput, use `tp2-mtp`. For long-system-prompt agentic workflows that genuinely benefit from prefix cache hits, stay on plain `tp2`.

This **does not contradict** the negative finding for **llama.cpp draft-then-verify** spec-decode on A3B reported at [`thc1006/qwen3.6-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) — that's a different mechanism (separate draft model forward pass) and the expert-saturation pathology shapes differently. Don't enable llama.cpp's `--spec-type ngram-cache`, `--spec-type ngram-mod`, or `--model-draft` on `gguf` mode; every variant is net-negative on consumer Ampere.

## Layout

```
qwen3.6/
├── README.md                   this file
├── FINDINGS.md                 full technical writeup with bench data
├── qwen.sh                     27B launcher (modes: default text longctx tp2 tp4 tp4-2 bf16-tp4)
├── qwen-moe.sh                 MoE launcher (modes: gguf tp1 tp2 tp2-mtp)
├── diffusion-gemma.sh          DiffusionGemma launcher (vLLM PR #45163 CI image, port 8031)
├── DIFFUSIONGEMMA_FINDINGS.md  DiffusionGemma day-zero bench + Ampere quant/TP failure modes
└── qwen36-27b-single-3090/     vendored fork of the noonghunna recipe
    ├── compose/
    │   ├── docker-compose.yml                       27B 20K + vision (1 GPU)
    │   ├── docker-compose.tools-text.yml            27B 75K text-only (1 GPU)
    │   ├── docker-compose.longctx-experimental.yml  27B 125K + vision (1 GPU, eager)
    │   ├── docker-compose.tp2.yml                   27B 100K + vision (2 GPUs)
    │   ├── docker-compose.tp4.yml                   27B 500K + vision (4 GPUs)
    │   ├── docker-compose.tp4-2.yml                 27B 500K + vision, 2 sessions (4 GPUs)
    │   ├── docker-compose.35b-a3b-awq-tp1.yml       MoE vLLM AWQ 1 GPU (anti-pattern)
    │   ├── docker-compose.35b-a3b-awq.yml           MoE vLLM AWQ 200K (2 GPUs)
    │   ├── docker-compose.35b-a3b-awq-mtp.yml       MoE vLLM AWQ + MTP-3 (2 GPUs)
    │   └── docker-compose.35b-a3b-gguf.yml          MoE llama.cpp IQ4_XS (1 GPU)
    ├── patches/
    │   ├── patch_tolist_cudagraph.py
    │   └── genesis/                                 cloned by setup.sh (gitignored)
    ├── models/                                      gitignored — fetch via setup.sh / hf download
    └── scripts/
        ├── setup.sh           one-shot: clone Genesis + download 27B AutoRound weights
        ├── bench.sh           TPS bench against the running endpoint
        ├── probe_moe_ctx.sh   probe max stable max-model-len for MoE tp1/tp2
        ├── verify.sh          functional smoke tests
        └── verify-full.sh     full functional matrix
```

`models/` is excluded from version control — total ~58 GB across the three model sets:
- `qwen3.6-27b-autoround-int4/` (~18 GB) — `Lorbus/Qwen3.6-27B-int4-AutoRound`
- `qwen3.6-35b-a3b-awq-int4/` (~23 GB) — `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
- `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (~17 GB) — `unsloth/Qwen3.6-35B-A3B-GGUF`

Fetch with `bash qwen36-27b-single-3090/scripts/setup.sh` for the 27B and `hf download` for the others (commands documented in `FINDINGS.md`).

## Using the endpoints

Anywhere that talks OpenAI:

| Client | Configuration |
|---|---|
| **Open WebUI** | Settings → Connections → OpenAI API → `http://host.docker.internal:8020/v1` (27B) or `:8021/v1` (MoE-vLLM) or `:8022/v1` (MoE-llama.cpp) |
| **OpenCode** | `~/.config/opencode/config.json` provider entry, `base_url` per the URL above |
| **Aider** | `aider --openai-api-base http://localhost:8020/v1 --openai-api-key sk-x --model openai/qwen3.6-27b-autoround` |
| **Continue / Cline (VS Code)** | "OpenAI-compatible" provider, same URL |
| **curl** | `curl http://localhost:8021/v1/chat/completions -d '{"model":"qwen3.6-35b-a3b-awq", "messages":[…]}'` |
| **Claude Code** | not directly — needs an Anthropic↔OpenAI proxy (e.g. `claude-code-router`) |

Tool calling is enabled (`qwen3_coder` parser on the vLLM modes; llama.cpp uses `--jinja` template tools) — agentic CLIs that make function calls work end-to-end. To suppress Qwen3.6's verbose chain-of-thought (which can eat your `max_tokens` before any user-visible content appears), pass `"chat_template_kwargs": {"enable_thinking": false}` in the request body.

## Hardware

- 4× RTX 3090 (24 GB each, Ampere SM_86), all on NUMA node 0, no NVLink
- Driver 610.43.02 (aikitoria BAR1-P2P patched, apt-held — see `p2p-bench/INSTALL_LOG.md`) / CUDA 13.2
- ~75 GB total disk for all three model sets + the vLLM and llama.cpp images

## Why the local edits

The upstream noonghunna recipe was published when Sandermage's `genesis-vllm-patches` shipped a monolithic `patch_genesis_unified.py`. That file was removed on **2026-04-24** (one day before the initial replication). The compose entrypoints reference the missing file. We worked around it differently per variant:

- `default` and `text`: no Genesis needed (fp8_e5m2 KV bypasses the TurboQuant hybrid gate that Genesis was patching). Volume mount commented out, entrypoint gated on file existence.
- `longctx`: wired the new modular Genesis (`vllm/_genesis/` package + `genesis_vllm_plugin/` entry-point); also `--enforce-eager` (Genesis's static-shape preallocs are incompatible with `torch.compile` Dynamo), `GENESIS_GDN_MAX_BATCHED_TOKENS=4128` (P28 prealloc must match `--max-num-batched-tokens`), and `--gpu-memory-utilization 0.95` (Genesis adds extra preallocs on top of the KV pool).

The MoE compose files don't need Genesis (different model architecture entirely; the hybrid Mamba/attention is handled by mainline vLLM). The MoE-specific tunings are documented inline in each compose file and in `FINDINGS.md`.

Full breakdown including reproduction error messages and resolutions: [`FINDINGS.md`](./FINDINGS.md).

## Troubleshooting

```fish
./qwen.sh status            # check 27B
./qwen-moe.sh status        # check MoE
docker stats vllm-qwen36-27b llamacpp-qwen36-35b-a3b vllm-qwen36-35b-a3b   # live GPU/CPU usage

# port conflict — something else owns the port
lsof -i :8020   # 27B
lsof -i :8021   # MoE vLLM
lsof -i :8022   # MoE llama.cpp

# "out of memory" on startup:
#   - default 27B uses ~22 GB on GPU 0; check no other process: nvidia-smi
#   - longctx 27B is at 24/24.5 GB; lower --gpu-memory-utilization in the compose
#   - MoE tp1 (vLLM single-GPU) WILL OOM on CUDA graph capture — use the gguf mode instead
#   - MoE tp2 / tp2-mtp need 2 idle 3090s; check no other model is up

# coexistence: 27B and MoE share GPUs 0,1 by default
#   - to run both, re-pin one of them: edit CUDA_VISIBLE_DEVICES in the active compose file
#   - e.g. MoE tp2-mtp on GPUs 1,2 + 27B tp2 on GPUs 3,0 keeps both up

# model recall returning gibberish at long context:
#   - drop max_tokens in the request, or
#   - switch 27B to longctx variant if you're hitting the 75K cap on text mode
```

Bigger issues: see [`FINDINGS.md`](./FINDINGS.md) § Things not done + § Pointers (upstream issue links).

## Credits

- Recipe origin: [`noonghunna/qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) (the 27B replication path; vendored here with deltas)
- Genesis vLLM patches: [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) (used by the longctx variant only)
- llama.cpp single-3090 MoE bench: [`thc1006/qwen3.6-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) (motivated the `gguf` mode + the spec-decode anti-pattern)
- vLLM TP=2 MTP A/B retest: [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090) (motivated the `tp2-mtp` mode)
- Quant: [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound), [`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit), [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- DiffusionGemma: [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it), [`RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4) quant, [vLLM PR #45163](https://github.com/vllm-project/vllm/pull/45163) by LucasWilkinson
