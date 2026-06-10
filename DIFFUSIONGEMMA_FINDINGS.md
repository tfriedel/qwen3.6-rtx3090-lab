# DiffusionGemma-26B-A4B on a single RTX 3090 — day-zero findings

Date: 2026-06-10 (model + vLLM PR + llama.cpp PR all published this same day)
Hardware: 1 of 4× RTX 3090 (24 GB, Ampere SM_86), driver 595.x, CUDA 13.2, Docker + nvidia runtime
Sources:
- Announcement: [Google blog — "DiffusionGemma: faster text generation"](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)
- Model: [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (25.2B total / 3.8B active MoE, **block-diffusion** text generation: 256-token canvas, ≤48 denoising steps, entropy-bound sampler)
- vLLM support: [PR #45163](https://github.com/vllm-project/vllm/pull/45163) "[Model] Add DiffusionGemma Support" — **unmerged**; we run its CI image straight from vLLM's public ECR
- Quant: [`RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4) (16.8 GB)

## TL;DR

DiffusionGemma generates whole blocks of text per forward pass instead of one token at a time, and it shows: **195–221 TPS narrative / 159–189 TPS code decode on a single 3090** — ~1.6× the best autoregressive Gemma-4-26B-A4B result on this rig (131 TPS, llama.cpp + ngram-mod) — at 100 % SM util / 348 W. Output quality on smoke tests is fine (correct code, coherent prose), consistent with Google's own caveat that quality sits below autoregressive Gemma 4: use it for speed-critical interactive work, not as the default brain.

| Mode | Engine / quant | GPUs | Max ctx | TPS | When to use |
|---|---|---|---|---|---|
| `diffusion-gemma.sh start` | vLLM PR #45163 CI image / NVFP4 (Marlin FP4 weight-only) | 1 | 16 K | **195–221 narr / 159–189 code** | fast interactive generation, drafts, agent helpers |

Endpoint: `http://localhost:8031/v1`, model name `diffusiongemma-26b-a4b-nvfp4`, OpenAI-compatible, tool calling via the gemma4 parser.

Inverted spec-decode intuition: on this model **code decodes *slower* than narrative** — the entropy-bound sampler commits fewer canvas positions per denoising step on structured output (it is "careful" exactly where DFlash/ngram drafts were getting free wins), while free-flowing prose commits 14–19 tokens per step.

## How it's running (none of this is released yet)

Everything shipped today; nothing is in a released build:

- **vLLM**: PR #45163 (branch `dgemma`, commit `74b5964`) — open, unmerged. Instead of building from source, we pull the per-commit CI image vLLM's Buildkite publishes:
  `public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:74b5964f02c7e023fadd3004cfac8a61c52eef1f`
  (entrypoint is bash, so the launcher passes `--entrypoint vllm`). Once the PR merges into a nightly, swap `IMAGE` in `diffusion-gemma.sh` for the regular `vllm/vllm-openai` nightly.
- **llama.cpp**: draft [PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) + [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) exists (Q4_K_M = 16 GB) but is CLI-only (`llama-diffusion-cli`) — no server, so not wired into this repo. Revisit when `llama-server` support lands.

## The quant/parallelism maze on Ampere (read before changing the config)

Four quants exist; on 24 GB Ampere cards exactly **one** configuration works. We hit every wall on the way there:

| Quant | Size | Result on this rig |
|---|---|---|
| bf16 (google) | 48 GB | needs TP — and TP crashes (below) |
| FP8-dynamic (RedHatAI) | 25.4 GB | > 24 GB → needs 2 cards — and both TP=2 and PP=2 crash (below) |
| **NVFP4 (RedHatAI)** | **16.8 GB** | **works on 1 card** via Marlin FP4 weight-only fallback |
| GGUF (unsloth) | 16–25 GB | llama.cpp PR is CLI-only, no server yet |

Failure modes, in the order we found them:

1. **TP=2 → Marlin "Invalid thread config" crash at startup.** On Ampere, FP8/FP4 weight-only runs through the Marlin kernel, which requires 64-aligned GEMM dims. Unsharded, this model is fine (`moe_intermediate_size` 704 = 64×11, dense `intermediate_size` 2112 = 64×33) — but TP=2 halves them to 352 and 1056, neither 64-aligned. Instant `RuntimeError` in the profile run.
2. **TP=2 + `--enable-expert-parallel` → same crash.** EP keeps the 128 routed experts whole, but the dense MLP layers still TP-shard (the error shape moves from K=352 to K=1056). Dead end.
3. **PP=2 → `AssertionError` in `v1/worker/gpu/pp_utils.py:181`.** The diffusion sampler's pipeline-parallel token broadcast is simply not implemented in PR #45163. Dies during warmup.
4. **TP=1 NVFP4 at default memory settings → OOM after KV allocation.** The diffusion sampler materializes `[num_seqs, canvas 256, vocab 262144]` fp32 logits (+ scratch) — **~0.5 GB per seq slot, none of it counted by vLLM's memory profiler** (the PR was clearly tuned on 180 GB B200s). vLLM happily fills its `--gpu-memory-utilization` budget with KV cache, then the sampler asks for the next GiB and dies. At util 0.90 *and* 0.85 it OOMs **after** "Graph capturing finished".

Working configuration (what `diffusion-gemma.sh` encodes):

```
NVFP4, single GPU, --attention-backend TRITON_ATTN,
--max-model-len 16384, --max-num-seqs 2, --gpu-memory-utilization 0.79
```

util 0.79 deliberately leaves ~5 GB of "free" VRAM as off-the-books headroom for the sampler. Final steady state: 22.7/24.6 GB used. `TRITON_ATTN` is mandatory: the diffusion decoder's per-sequence causal masking needs FA4 on the flash-attn path, and FA4 is Blackwell-only — Triton implements it for everyone else.

NVFP4's native kernels also need SM_89+; on Ampere, Marlin dequants FP4→bf16 on the fly. Memory savings, no FP4 kernel speedup — all the throughput comes from the diffusion mechanism itself.

## Headline numbers — what we actually measured

Bench: `qwen36-27b-single-3090/scripts/bench.sh` against the OpenAI endpoint (1 concurrent request, temp 0.6, top-p 0.95), same prompts as every other model in this repo.

### Short-context decode (~30 input tokens)

| Workload | runs | wall TPS |
|---|---|--:|
| narrative (1000 tok) | 3 | **219 / 195 / 221** |
| code (800 tok) | 2 | **189 / 159** |

GPU during bench: 100 % SM util, 348 W, 22.7 GB. Compare Gemma-4-26B-A4B autoregressive on the same card: 131 TPS (gguf, flat) / 179 TPS (DFlash, code-only best case). DiffusionGemma's *narrative* number beats DFlash's *code* number.

### Diffusion-native metrics (from vLLM's `DiffusionDecoding metrics` log lines)

| Metric | Typical range observed |
|---|---|
| Tokens committed per denoising step | 10.8 – 19.7 (≈14 typical) |
| Denoising steps per 256-token canvas | 16 – 31 (≈18 typical) |
| Committed-token throughput (10 s windows, busy) | 180 – 230 tok/s |

Steps-per-canvas is content-dependent — structured/code-like output runs more denoising steps and commits fewer positions per step, which is why code is the *slow* workload here (the exact opposite of DFlash/MTP/ngram acceptance patterns elsewhere in this repo).

### Long context (e2e, includes prefill)

| Input tokens | Output | Wall | e2e TPS |
|---|---|--:|--:|
| 9 639 | 431 | 5.8 s | 74 |
| 15 639 | 441 | 6.0 s | 74 |

No DFlash-style collapse at 16 K — but the 16 K `--max-model-len` ceiling (KV + sampler headroom on 24 GB) means long-context work stays with `gemma.sh start gguf` (262 K) or the Qwen stacks.

### Functional checks

- Code smoke test: correct two-pointer `merge_sorted_lists` with sensible docstring/explanation. ✅
- `usage.completion_tokens` reported normally (committed tokens, not canvas slots). ✅
- Tool calling (`--tool-call-parser gemma4`): see TOOL_CALLING_ISSUES.md if anything regresses; verified a basic `get_weather` call end-to-end on day zero.
- Vision inputs: untested (model card claims image+video input; not exercised here).

## Caveats

- **Unmerged PR, pinned by commit digest.** The CI image tag is content-addressed to commit `74b5964` and ECR CI repos garbage-collect old tags eventually. If the image disappears, rebuild from the PR branch or wait for the merge.
- **Quality < autoregressive Gemma 4** per Google's own announcement. Treat as the speed option, not the quality option.
- **`--max-num-seqs 2`** — this endpoint is sized for interactive single-user work, not batch serving (each extra seq slot costs ~0.5 GB of unaccounted sampler memory).
- The model card's thinking mode (`<|think|>`) and the `enable_thinking` chat-template kwarg were not exercised; bench.sh passes `enable_thinking: false` and the template accepted it.

## Things not done

- llama.cpp GGUF path (PR #24423): CLI-only today. Wire a `gguf` mode into `diffusion-gemma.sh` when `llama-server` gains diffusion support — Q4_K_M would free ~6 GB vs NVFP4-on-Marlin and might allow 32 K+ ctx.
- bf16 quality-delta eval vs NVFP4 (RedHatAI reports 97–102 % recovery on B200 for FP8; no NVFP4-on-Ampere numbers exist).
- Multi-GPU: blocked upstream (Marlin dim alignment for TP, unimplemented PP broadcast). Watch PR #45163.
