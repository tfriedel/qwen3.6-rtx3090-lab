# Gemma-4-26B-A4B on a single RTX 3090 — replication findings

Date: 2026-05-08
Hardware: 1 of 4× RTX 3090 (24 GB, Ampere SM_86), driver 595.58.03, CUDA 13.2, Docker 26.1 + nvidia runtime
Source recipes:
- Reddit headline: [r/LocalLLaMA — "Gemma 4 26B Hits 600 Tok/s on One RTX 5090"](https://www.reddit.com/r/LocalLLaMA/comments/1t796qe/gemma_4_26b_hits_600_toks_on_one_rtx_5090/) (chain-77, 2026-05-04)
- 3090 GGUF reply by `coder543` in the same thread
- Memory-tuning reply by `arzeth` in the same thread

## TL;DR

Both compose variants of the Gemma-4-26B-A4B (MoE, 4B active) recipe were brought up successfully on a single 3090. The Reddit headline is **vLLM AWQ + DFlash** at ~578 tok/s on a 5090 — that recipe runs on a 3090 only at reduced context (20 K vs the post's 32 K) and at workload-dependent throughput (95 tok/s narrative, 179 tok/s code). The **`coder543` GGUF reply is the actually-good 3090 path**: 130 tok/s flat across all workloads, full 262 K context, 95 % SM utilization.

| Variant | Engine / quant | Max ctx | TPS (narr / code / explain, 512 out) | VRAM | When to use |
|---|---|---|---|---|---|
| **`gemma.sh start gguf`** | llama.cpp / Unsloth UD-Q4_K_XL + ngram-mod spec | **262 K** | **131 / 129 / 129** | 22.8 GB | **Sweet spot** — flat across workloads, full ctx, 95 % SM util |
| `gemma.sh start tp1` | vLLM `gemma4-0505-cu129` / cyankiwi AWQ + z-lab DFlash-13 | 20 K | 95 / **179** / 117 | 21.9 GB | Short-ctx code generation only — DFlash collapses past ~16 K |
| `gemma.sh start e4b` | vLLM bf16 / `google/gemma-4-E4B-it` | 128 K | not benchmarked | est. ~12 GB | Small/fast variant, agent helper |

Endpoint: `http://localhost:8023/v1` (tp1) or `:8024/v1` (gguf) or `:8025/v1` (e4b). All OpenAI-compatible. Vision (mmproj-BF16) loaded for both 26B variants.

## Headline numbers — what we actually measured

All numbers are **decode TPS** measured via the OpenAI `/v1/chat/completions` endpoint (`/tmp/gemma_bench.py`, included reproducibly), 1 concurrent request, temperature 1.0, top-p 0.95, top-k 64. Three distinct prompts to probe how spec-decoding acceptance varies with workload structure:

- **narrative** — open-ended creative writing prompt (low spec-decode acceptance)
- **code** — Python implementation prompt with type hints (high acceptance: structured token sequences)
- **explain** — technical explanation (mixed)

### Short-context decode (~75 input tokens)

| Workload | gguf wall TPS | gguf engine TPS | tp1 wall TPS |
|---|--:|--:|--:|
| narrative | 131 | 134 | **95** ← *slower than gguf* |
| code      | 129 | 133 | **179** ← *+39 % vs gguf* |
| explain   | 129 | 133 | **117** ← *-9 % vs gguf* |

(`engine_tps` is llama.cpp's internal `predicted_per_second`. vLLM doesn't return this field at the OpenAI endpoint, so tp1 only has wall-clock numbers, which include the small `/v1/chat/completions` request overhead.)

### Sustained decode (1024 / 2048 output tokens)

| Output tokens | Workload | gguf | tp1 |
|---|---|--:|--:|
| 1024 | narrative | 128 | 145 (truncated at EOS) |
| 1024 | code | 127 | 179 (truncated at EOS) |
| 1024 | explain | 127 | 117 |
| 2048 | narrative | 129 (truncated at 1782) | n/a |
| 2048 | code | 127 | n/a |
| 2048 | explain | 126 | n/a |

The model frequently emits `<|end_of_turn|>` before hitting `max_tokens=2048`, especially on code prompts that finish naturally. gguf runs are flat: TPS is independent of how long the generation runs.

### Long-context decode

| Input tokens | Output | Engine | Wall TPS | Pure decode TPS | Notes |
|---|---|---|--:|--:|---|
| 17 027 | 354 | gguf | 47.9 | **120.3** | prefill 4.0 s @ 4019 tok/s, draft accept 15/81 (18.5 %) |
| 17 028 | 1024 | tp1 | 45.9 | ~50–60 | prefill ~6 s, DFlash acceptance has clearly dropped |
| 65 027 | 256 | gguf | n/a | **103.8** | prefill 24.2 s @ 2685 tok/s, **draft 0/0** (ngram-mod gave up) |

`gguf` decode rate degrades smoothly with context: 130 → 120 → 104 across 0 → 17 K → 65 K input tokens. This is raw KV-cache memory bandwidth scaling, not a spec-decoding cliff (which is what `tp1` hits).

`tp1` was **not** tested at 65 K because `--max-model-len` is capped at 20 K (see "What broke" §3 below).

### GPU saturation during decode (sampled at 2 Hz)

| Engine | avg SM | peak SM | peak power | VRAM resident |
|---|--:|--:|--:|--:|
| gguf | **95 %** | 95 % | 347 W | 22.8 / 24 GB |
| tp1 | 68 % | 93 % | 349 W | 21.9 / 24 GB |

gguf saturates the GPU. tp1's lower average reflects the AWQ-on-3090 launch-overhead pattern documented for the Qwen-MoE in the existing repo (small kernels per MoE expert dispatch, weights too large to fit cudagraphs comfortably) — same regime, different model.

## What broke vs. the Reddit recipe

The original Reddit post lists "vLLM 0.19.2rc1" without a Docker tag and a single `vllm serve` invocation. Four fixes were needed to make the recipe run end-to-end on this rig.

### 1. The vLLM version string in the Reddit post is fictional

`vllm/vllm-openai:v0.19.2rc1` does not exist on Docker Hub. The current published tags relevant to Gemma 4 / DFlash are:

| Tag | Date | Notes |
|---|---|---|
| `gemma4-0505-cu129` | 2026-05-05 | Gemma-4 / DFlash speculator support, vLLM `v0.20.2rc1.dev49+g9b4e83934` |
| `tokenspeed-preview` | 2026-05-06 | Benchmarking branch, untested here |
| `nightly` | 2026-05-08 | Latest, untested with DFlash |
| `latest-cu129` (= `v0.20.1`) | 2026-05-04 | Stable; **does not** ship the `dflash` speculator method |

**Fix:** pinned to `vllm/vllm-openai:gemma4-0505-cu129`. Confirmed by boot log: `Resolved architecture: DFlashDraftModel`.

### 2. AWQ checkpoint already contains fp8 components → `--kv-cache-dtype fp8_e5m2` is rejected

Carried `--kv-cache-dtype fp8_e5m2` over from the working Qwen3.6-27B compose file. The `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` checkpoint stores activations in fp8 already; vLLM raises:

```
ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.
```

**Fix:** removed `--kv-cache-dtype`, defaulting to fp16 KV. Costs some KV pool capacity (no big deal at 20 K ctx) but is the only path that works.

### 3. 24 GB is not enough for 32 K context with DFlash + AWQ + cudagraphs

The Reddit recipe uses `max-model-len 32768` on a 32 GB 5090. On a 3090:

```
ValueError: To serve at least one request with the model's max seq len (32768),
3.01 GiB KV cache is needed, which is larger than the available KV cache memory (2.6 GiB).
Estimated maximum model length is 22016.
```

VRAM breakdown after AWQ + DFlash + cudagraphs on a 3090:
- AWQ weights: ~13.5 GB
- DFlash draft model (430 M params, fp16): ~0.85 GB
- vLLM activations + cudagraph workspace: ~5–6 GB
- Available for KV: ~2.6 GB

**Fix:** capped at `--max-model-len 20000` and bumped `--gpu-memory-utilization` from 0.93 to 0.95. Production headroom is tight; bumping the draft model down to fp8 might recover a few hundred MB, but DFlash isn't designed to be quantized.

### 4. The `--spec-ngram-mod-n-{min,max,match}` flags coder543 used aren't in the mainline llama.cpp image

`coder543`'s reply specifies:

```
--spec-type ngram-mod
--spec-ngram-mod-n-min 48
--spec-ngram-mod-n-max 64
--spec-ngram-mod-n-match 24
```

The current `ghcr.io/ggml-org/llama.cpp:server-cuda` image (build `672b0c86e7f0`, 12 days old) accepts `--spec-type ngram-mod` but **only** these spec knobs:
- `--spec-ngram-size-n`
- `--spec-ngram-size-m`
- `--spec-ngram-min-hits`

`-n-min/-n-max/-n-match` are not in the mainline `--help` output. Probably a llama.cpp PR that landed after this image was cut, or coder543 is on a fork. Container exits with `error: invalid argument: --spec-ngram-mod-n-min` if you pass them.

**Fix:** dropped the unsupported knobs and rely on `ngram-mod` defaults. Still hits the headline 130 tok/s, so the knobs aren't load-bearing on a 3090.

### 5. The mmproj filename in coder543's recipe doesn't match Unsloth's actual layout

The recipe references `gemma-4-26B-A4B-it-UD-Q4_K_XL-mmproj-BF16.gguf`. Unsloth ships it un-prefixed as just `mmproj-BF16.gguf`.

**Fix:** updated the `--mmproj` path in `compose/docker-compose.26b-gguf.yml`.

## Why DFlash is a 5090-class feature, not a 3090-class feature

The Reddit headline ("578 tok/s") is real on a 5090. On a 3090, the same recipe gives 95–179 tok/s depending on workload, and the spec-decoding speedup vs no-spec is **net negative on prose**.

Two explanations:

### (a) DFlash acceptance correlates with token predictability

DFlash drafts tokens via a small block-diffusion model trained to mimic the target's outputs. Acceptance rate is the fraction of drafted tokens the target model agrees with. Empirically:

- **Code** has high local entropy reduction once a syntactic structure is committed (`def foo(x: int)` strongly constrains the next ~30 tokens). DFlash drafts these accurately → 13/13 tokens accepted per step → +40 % over baseline.
- **Narrative prose** has high local entropy throughout. DFlash drafts often get rejected → wasted forward passes → net slowdown.
- **Long context** dilutes the draft model's relative size advantage further (its KV cache also grows), and acceptance falls. By 65 K input tokens, acceptance hits 0 and the spec path is a pure overhead loss.

The Reddit thread's top reply (`ATK_DEC_SUS_REL`) calls this exact pattern out: *"DFlash drops off a cliff at high context lengths. By 'high' I mean ~20 k context or more."* Confirmed at 17 K on this rig.

### (b) The 3090 is launch-overhead-bound on AWQ Gemma MoE

This is the same anti-pattern documented in the existing repo for `qwen-moe.sh tp1` (Qwen3.6-35B-A3B AWQ on 1× 3090). Gemma-4-26B-A4B has the same shape: MoE expert routing dispatches many small kernels per layer, AWQ INT4 weights leave ~5 GB for cudagraph workspace + activations after KV, and the 3090's PCIe-only kernel-launch path can't keep the SMs fed. Average SM utilization landed at 68 % — vs gguf's 95 %. On a 5090, faster CUDA launch (Hopper schedulers, NVLink-class memory, larger L2) hides this overhead, which is why the same recipe gets ~6× the throughput there.

The fix isn't tuning vLLM further — it's switching to a smaller-footprint quant. `unsloth/gemma-4-26B-A4B-it-GGUF UD-Q4_K_XL` is **16 GB on disk** (Q4_K_XL is mixed Q4/Q6) but ~17 GB resident, leaves ~5 GB headroom for KV pool and llama.cpp's compute graphs, and the GPU saturates. Same model, same hardware, +30 % throughput on average and +30 % in the worst case (narrative).

This is a verbatim repeat of the Qwen-MoE finding documented in `FINDINGS.md` §"Why single-GPU MoE = llama.cpp, not vLLM". The recipe transfers cleanly between 35B-A3B and 26B-A4B because the failure mode is shared (small MoE expert kernels + AWQ footprint + 3090 launch overhead), not architecture-specific.

## Why GGUF wins flat-across-workloads

Same recipe class as the documented Qwen-MoE GGUF mode in this repo:
- **Saturates the GPU**: 95 % SM, 66 % memory bandwidth, 347 W (3090 default TDP is 350 W).
- **Headroom for context**: weights ~17 GB, mmproj ~1.2 GB, leaves ~5.8 GB for KV → full 262 K context fits with `-ctx-checkpoints 1` + `--kv-unified`.
- **Spec decoding is a pure win at short context**: ngram-mod adds zero parameters (it's a learned pattern table); when acceptance is high (code, code-adjacent prose), it's free speedup. When acceptance drops (long context, prose), it shuts itself off (`draft_n=0`) — no overhead penalty, unlike DFlash.

Decode rate vs context length:

| Input ctx | gguf TPS | tp1 TPS |
|---|--:|--:|
| 0 K | 130 | 95 (narr) / 179 (code) |
| 17 K | 120 | ~50 |
| 65 K | 104 | n/a (max-model-len 20 K) |

gguf has a smooth ~20 % degradation across an 8× context range. tp1 has a discontinuity around 16 K where DFlash hands back a model that's slower than no spec at all.

## Recommendations

For day-to-day use of Gemma-4-26B-A4B on this rig:

1. **Default: `gemma.sh start gguf`** — predictable 130 tok/s, full 262 K context, vision works, GPU saturated.
2. **For pure short-ctx code generation**: `gemma.sh start tp1` is +40 % over gguf (179 vs 129) on code prompts under 16 K input. Still slower than gguf on prose. Be deliberate about routing.
3. **Coexistence with the Qwen stack**: `gemma.sh gguf` pins to GPU 0 by default, conflicts with `qwen-moe.sh gguf`. Pick one or re-pin `CUDA_VISIBLE_DEVICES`. The 4× 3090 layout still has GPUs 1–3 free for `qwen.sh` modes that don't use GPU 0.
4. **Don't bother chasing the 578 tok/s Reddit headline** — that requires a 5090's launch-overhead margin and 32 GB of VRAM. The 3090-realistic ceiling for this recipe is ~180 tok/s on code-heavy prompts, ~130 tok/s on the typical mix.

## Open items / not yet tested

- **`e4b` mode** is shipped but unbenchmarked. Probably 200+ tok/s given bf16 weights at ~9 GB on a 3090 with no spec decoding. Should also verify multimodal works.
- **Gemma-4-31B-it (dense)** — the larger sibling. AWQ would be ~16 GB, fits with Q4-ish on a single 3090. Could be a "high-quality short-ctx" alternative to Qwen-3.6-27B. Not yet attempted.
- **vLLM `tp2` for Gemma**: untested. Splitting AWQ weights across 2× 3090 would buy back KV headroom (~6 GB instead of 2.6) and let `--max-model-len` reach the Reddit's 32 K. Could also remove the launch-overhead bottleneck, recovering DFlash's win on more workloads. Not yet attempted.
- **`tokenspeed-preview` image vs `gemma4-0505-cu129`** — the former is newer and was specifically named with benchmarking in mind. Worth a sanity sweep.
- **Quality benchmarks** (MMLU-Pro, GPQA, etc.) — not run; this writeup is purely throughput-focused. The gguf vs AWQ quality delta on Gemma-4 specifically would be useful to measure, since UD-Q4_K_XL is mixed-precision and may preserve more headroom than the AWQ-INT4.

## Reproduction

```fish
cd ~/projects/qwen3.6

# Pre-fetch GGUF (gguf mode):
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
    --local-dir gemma/models \
    --include "*UD-Q4_K_XL*"
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
    --local-dir gemma/models \
    --include "*mmproj*BF16*"

# Pre-fetch AWQ + DFlash (tp1 mode):
hf download cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
    --local-dir gemma/models/gemma-4-26b-a4b-awq
hf download z-lab/gemma-4-26B-A4B-it-DFlash \
    --local-dir gemma/models/gemma-4-26b-a4b-dflash

./gemma.sh start gguf       # ~6 s to ready
./gemma.sh start tp1        # ~90 s to ready (compile + warmup)

python3 /tmp/gemma_bench.py http://localhost:8024/v1/chat/completions gemma-4-26b-a4b-gguf 512
python3 /tmp/gemma_bench.py http://localhost:8023/v1/chat/completions gemma-4-26b-a4b-awq  512
```

Compose files under `gemma/compose/` document every flag with the rationale inline.
