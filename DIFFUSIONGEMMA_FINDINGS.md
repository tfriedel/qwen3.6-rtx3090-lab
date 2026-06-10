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
- **llama.cpp**: [PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) (branch `dgemma`, HEAD `c5fe75b`) + [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) (Q4_K_M = 16 GB). **Measured 2026-06-11 — ~3× slower than vLLM, not worth wiring up** (see "Going faster" below). Note: this branch is further along than "CLI-only" — it ships `examples/diffusion/` (`llama-diffusion-cli`), `examples/diffusion-gemma-server/`, and `examples/diffusion-gemma-eval/`, all CMake-wired. The server exists; it's just on the slower engine.

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

## Going faster — what we measured (2026-06-11)

Question: is vLLM the fastest way to run this, and can we push it harder on the 3090? Both lower-effort levers were tested and **both came back negative**. Throughput is `tokens-committed-per-step ÷ seconds-per-step`; the wins only live on the seconds-per-step side, which is hardware-bound on Ampere.

### 1. llama.cpp (Q4_K_M) is ~3× slower than vLLM — don't switch

Built PR #24423 (`llama-diffusion-cli`) with CUDA SM_86, benched the same prompts on a free 3090 (entropy-bound sampler, single stream):

| Workload | llama.cpp Q4_K_M | vLLM NVFP4 | Ratio |
|---|--:|--:|--:|
| narrative | 60.7–61.4 TPS | 195–221 TPS | ~3.4× slower |
| code | 74.4–75.3 TPS | 159–189 TPS | ~2.3× slower |

Per-step time ~220 ms (llama.cpp) vs ~70 ms (vLLM); both run **without** Flash Attention on Ampere (llama.cpp auto-disables FA "missing support"; vLLM uses TRITON_ATTN), so it's a clean forward-pass comparison and vLLM's Marlin GEMM + CUDA-graph MoE simply wins. Note one inversion: in llama.cpp **code is faster than narrative** (17.1 vs 13.3 tok/step) — the opposite of vLLM here — so the two EB-sampler implementations don't agree on commit behavior, despite identical nominal params. Doesn't change the verdict.

### 2. Sampler tuning gives no reliable speedup — the default is already near-optimal

Swept `max_denoising_steps {48,32,24} × entropy_bound {0.1,0.2}` on a GPU-2 instance (each via `--diffusion-config '{"canvas_length":256,"max_denoising_steps":N}'` + a copied `generation_config.json` with edited `entropy_bound`, pointed at by `--generation-config <dir>`). Steady-state TPS all landed in a 190–260 band dominated by run-to-run noise (±25, n=2). The mechanism is the clincher and is noise-independent: **the entropy-bound sampler self-converges at ~16–22 steps/canvas, already far below the 48 cap** — so lowering the cap to 32/24 almost never binds, and `entropy_bound` 0.1→0.2 doesn't measurably raise commit rate. Quality held even at the most aggressive corner (`eb 0.2 / steps 24` produced a correct, fully-commented quicksort), but there was no speed traded for it. **~210/230 TPS is the sampler ceiling on this rig.** To force fewer steps you'd need `max_denoising_steps ≤ 16` / `entropy_bound ≥ 0.4`, which clips real convergence — quality goes before speed arrives.

Two gotchas found in the PR source while doing this:
- `--override-generation-config` does **not** reach the diffusion sampler. It only merges in `get_diff_sampling_param()`, which this model skips — `custom_sampler` reads raw `try_get_generation_config()`. Tune via a copied `generation_config.json` + `--generation-config <dir>` instead.
- An explicit `--generation-config <dir>` applies that file's `max_new_tokens` as a **hard cap** (the checkpoint ships `max_new_tokens: 256`), silently truncating every request to 256 tokens regardless of the request's `max_tokens`. Under the default `"auto"` the request wins. If you point `--generation-config` anywhere, raise `max_new_tokens` in that file or all outputs clip at 256.

### 3. The only real lever left is seconds-per-step — and it's hardware/engineering, not config

The forward pass runs in **bf16** (NVFP4→Marlin dequant, no FP4 speedup on SM_86). Two things could lower it, neither available as a flag today:
- **TP=2** across two cards (BAR1 P2P is already enabled, commit `8a5c69e`) → ~1.5× the forward pass. Blocked by Marlin's 64-alignment (`moe_intermediate` 704→352, dense 2112→1056). Would need a dim-padding patch to PR #45163's model config.
- **int8 W8A8** — the only quant that actually accelerates compute on Ampere (2× bf16 int8 tensor cores). No int8 build of DiffusionGemma exists, and diffusion logits are precision-sensitive.

Dead ends confirmed: FP8 quant (no Ampere fp8 tensor cores → dequants to bf16), larger `--max-num-seqs` (already 100 % SM at batch-1, compute-bound), swapping the attention backend (attention is a small slice of the step; the per-seq causal mask needs the FA4 path Triton stands in for).

## Caveats

- **Unmerged PR, pinned by commit digest.** The CI image tag is content-addressed to commit `74b5964` and ECR CI repos garbage-collect old tags eventually. If the image disappears, rebuild from the PR branch or wait for the merge.
- **Quality < autoregressive Gemma 4** per Google's own announcement. Treat as the speed option, not the quality option.
- **`--max-num-seqs 2`** — this endpoint is sized for interactive single-user work, not batch serving (each extra seq slot costs ~0.5 GB of unaccounted sampler memory).
- The model card's thinking mode (`<|think|>`) and the `enable_thinking` chat-template kwarg were not exercised; bench.sh passes `enable_thinking: false` and the template accepted it.

## Things not done

- ~~llama.cpp GGUF path~~ — **done (2026-06-11): ~3× slower, not worth it.** See "Going faster" §1. A `gguf` serving mode in `diffusion-gemma.sh` would only make sense if `llama-server` diffusion support plus a future Ampere kernel speedup closed the gap; today it would just be a slower endpoint.
- ~~Sampler tuning for speed~~ — **done (2026-06-11): no reliable gain.** See "Going faster" §2. Default is already near the convergence point.
- bf16 quality-delta eval vs NVFP4 (RedHatAI reports 97–102 % recovery on B200 for FP8; no NVFP4-on-Ampere numbers exist).
- **TP=2 via a Marlin 64-alignment patch** — the one remaining real speed lever (~1.5× forward pass, BAR1 P2P already enabled). Pad `moe_intermediate` 704 and dense 2112 to stay 64-aligned under TP-sharding. Untried; heavy (patches PR #45163's model config). PP broadcast also still unimplemented upstream. See "Going faster" §3.
