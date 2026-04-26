# Qwen3.6-27B on a single RTX 3090 — replication findings

Date: 2026-04-25
Hardware: 1 of 4× RTX 3090 (24 GB, Ampere SM_86), driver 595.58.03, CUDA 13.2, Docker 26.1 + nvidia runtime
Source recipe: [r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1stjx29/), [`noonghunna/qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) (pinned vLLM image `vllm/vllm-openai@sha256:9bba4628a3b943e0dd33caefb94b811569ba1e97bdf23bee19a265c31b947ccb`, model `Lorbus/Qwen3.6-27B-int4-AutoRound`)

## TL;DR

Both shipped variants of the recipe were brought up successfully on one 3090. The recipe itself is **stale by one day** because Sandermage's `genesis-vllm-patches` repo was force-restructured on 2026-04-24 — the file the recipe mounts (`patch_genesis_unified.py`) no longer exists. Workarounds below.

| Variant                            | Max ctx | Bench TPS (narrative / code peak) | VRAM     | Genesis required? | Notes                                |
|------------------------------------|---------|-----------------------------------|----------|-------------------|--------------------------------------|
| Default (`docker-compose.yml`)     | 20 K    | **70–72 / 91**                    | 22.3 GB  | No                | fp8_e5m2 KV; tool calls work upstream |
| Tools-text (`...tools-text.yml`)   | 75 K    | **68–72 / 92**                    | 22.1 GB  | No                | fp8_e5m2 KV; no vision; needle recall verified at 70K |
| Long-ctx (`...longctx-experimental`)| 125 K   | **30–33 / 42**                    | 24.1 GB  | Yes               | TurboQuant 3-bit KV; needs `--enforce-eager` |
| **TP=2** (`...tp2.yml`, local addition)| 100 K | **90–96 / 120**                  | 22.4 GB × 2 | No            | 2× RTX 3090 tensor-parallel; fp8_e5m2; KV pool 172K tokens; needle verified at 90K; **TPS sweet spot** |
| **TP=4** (`...tp4.yml`, local addition)| 500 K | **88–97 / 117**                  | 22.4 GB × 4 | No            | 4× RTX 3090 tensor-parallel; fp8_e5m2; KV pool 507K tokens; needle verified at **470K** (475 s) |

Both expose the same OpenAI-compatible endpoint at `http://localhost:8020/v1/*`, model name `qwen3.6-27b-autoround`. Tool calling, vision, MTP spec-decode (n=3), streaming, thinking — all functional.

The README's "85 TPS" headline matches the **code peak** of the default variant on a 350 W-default 3090. Narrative TPS lands inside the published 60–105 band but on the lower end (the README's reference rig was capped at 230 W, ours runs default).

## What broke vs. the recipe

### 1. `patch_genesis_unified.py` is gone (upstream restructure 2026-04-24)

Sandermage's repo migrated from a monolithic ~3000-LOC text-replacement script to a modular Python package (`vllm/_genesis/` + `genesis_vllm_plugin/`) on 2026-04-24. The qwen36 setup script and both compose files still mount the dead file path. The qwen36 repo's own [issue #2](https://github.com/noonghunna/qwen36-27b-single-3090/issues/2) confirms.

**Fix for default (20K) variant:** skip Genesis entirely. The default uses `fp8_e5m2` KV, not TurboQuant — so the "hybrid attention gate bypass" Genesis used to provide is irrelevant. The Qwen3 tool-call parser fix has also landed upstream in this nightly. Verified: tool calls populate `tool_calls[]` cleanly without Genesis.

**Fix for long-ctx (125K) variant:** wire the new modular plugin. Two volume mounts + `pip install -e` in the entrypoint:

```yaml
volumes:
  - ../patches/genesis/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro
  - ../patches/genesis/genesis_vllm_plugin:/plugin:ro
entrypoint:
  - /bin/bash
  - -c
  - |
    set -e
    cp -r /plugin /tmp/genesis_vllm_plugin
    pip install --quiet --no-deps -e /tmp/genesis_vllm_plugin
    python3 -m vllm._genesis.patches.apply_all
    python3 /patches/patch_tolist_cudagraph.py
    exec vllm serve "$@"
```

Boot logs report `Genesis Results: 25 applied, 16 skipped, 0 failed` (most skips are env-opt-in patches like P5b, P7, P37, P58–P62).

### 2. Genesis preallocation is incompatible with `torch.compile`

The new Genesis ships several patches that pre-allocate fixed-shape tensors to eliminate per-step allocator churn (P22 TQ dequant pool, P26 TQ prefill output, P28 GDN core_attn_out, P32/P33 TQ cu/synth_seq_lens, P39a FLA KKT, P44 TQ mixed-batch attn_out, P46 GDN gating). These have static shapes that escape into `torch.compile`'s Dynamo trace and don't unify with the surrounding op's symbolic `48*s59` shapes — the engine crashes during `profile_run` with:

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
  call_function mul(FakeTensor(196608, 128), FakeTensor(48*s59, 128))
```

The README's `cudagraph_mode=NONE` workaround (which leaves torch.compile inductor on) is not enough — the failure is upstream in Dynamo before cudagraph runs.

**Fix:** `--enforce-eager`. Confirms ampersandru's `--enforce-eager` recipe in [issue #1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1). Subsumes `cudagraph_mode=NONE`. Costs an additional ~10–20% TPS over the README numbers but is the path Genesis is actually tested against (the README itself notes P7 requires `--enforce-eager`).

### 3. Genesis P28 prealloc must match `--max-num-batched-tokens`

P28 reads `GENESIS_GDN_MAX_BATCHED_TOKENS` from env, defaulting to 4096. The long-ctx compose configures `--max-num-batched-tokens 4128` (Mamba block_size constraint at 125K ctx). Mismatch → rmsnorm assertion in profile_run:

```
gdn_linear_attn.py:603: core_attn_out = self.norm(core_attn_out, z)
layernorm_guard.py:280: assert z.shape == x_shape_og
AssertionError
```

**Fix:** add `GENESIS_GDN_MAX_BATCHED_TOKENS=4128` to the compose env.

### 4. Memory budget shifts when Genesis adds preallocs

Recipe's `--gpu-memory-utilization 0.97` causes Genesis P22 to log a non-fatal CUDA OOM during boot (the per-layer dequant prealloc fails to allocate 520 MiB after vLLM has already sized the KV pool to fill remaining VRAM). At 0.92 the engine refuses to start because KV is too small for 125K context (1.94 GiB available, 2.38 GiB needed).

**Fix:** `--gpu-memory-utilization 0.95`. KV pool comes out at 210K tokens — comfortable margin over the 125K request — and Genesis preallocs fit.

### 5. `patch_tolist_cudagraph.py` Site B anchor is rewritten by Genesis P26

The recipe's local cudagraph-capture fix has two sites in `turboquant_attn.py`. With Genesis applied, P26 rewrites the `_prefill_attention` continuation branch where Site B lives, so the anchor never matches. Logs:

```
[tolist_cudagraph_fix] Site B (_prefill_attention) anchor NOT FOUND -- patch NOT applied.
```

This is benign for the long-ctx variant because we're running `--enforce-eager` (no cudagraph capture, so the `.tolist()` sync isn't illegal). Site A still applies and is also irrelevant in eager mode. The fix matters only on the unmodified-Genesis fast path that the recipe targets — which we're not using.

## Final configurations

### Default 20K variant (`compose/docker-compose.yml`)

Edits from upstream:
- `CUDA_VISIBLE_DEVICES=0` env var uncommented (we have 4× 3090s)
- Genesis volume mount commented out, entrypoint gated on file existence
- All else stock: `fp8_e5m2` KV, MTP n=3, 20 K context, 0.95 GPU mem util, vision on, tools on

Boot in ~90 s. Bench:

```
Warmup:    71.6 / 69.5 / 68.9 TPS   (1000 tok, narrative prompt)
Narrative: 70.9 / 71.0 / 72.2 TPS
Code:      90.6 / 91.9 TPS
MTP avg:   58–68%, position-wise [82%, 58%, 33%] → [87%, 69%, 50%]
```

### Tools-text 75K variant (`compose/docker-compose.tools-text.yml`)

Same Genesis-skip treatment as the default (fp8_e5m2 KV, no Genesis needed). `--language-model-only` drops the MoonViT vision tower (~0.86 GB savings → context grows from 20K to 75K).

Edits from upstream:
- `CUDA_VISIBLE_DEVICES=0`
- Genesis volume mount commented out, entrypoint gated on file existence
- All else stock: 75 K context, 0.97 GPU mem util, fp8_e5m2 KV, MTP n=3, vision off, tools on

Boot in ~180 s (KV-cache profiling is slower at higher max-model-len). Bench (short 1000-tok generations):

```
Warmup:    69.8 / 69.7 / 70.8 TPS
Narrative: 70.9 / 71.8 / 67.6 TPS
Code:      92.3 / 92.0 TPS
MTP avg:   53–63%
```

Long-context smoke test (needle-in-haystack):

```
prompt 55,076 tokens → 12 tokens generated in 57s; recalled "BLUE-MOOSE-42" correctly
prompt 70,078 tokens → 14 tokens generated in 22s (prefix cache warm); recalled "RED-PENGUIN-99" correctly
```

vLLM logs `GPU KV cache size: 24,000 tokens` at boot, which is **misleading** — the actual KV memory available is 3.29 GiB and is used dynamically with chunked prefill + paged allocation; the 75K prompt fits comfortably. The 24K figure appears to be page/block-count math, not effective per-request capacity.

### TP=2 multi-GPU variant (`compose/docker-compose.tp2.yml`, local addition)

The TPS sweet spot on this hardware. Same fp8_e5m2 + vision + tools setup as `default`, sharded across 2 cards. Per-card model footprint ~8.9 GB, leaving ~13–14 GB for KV. Total KV pool: 172,800 tokens — comfortably fits the configured 100 K context.

Config deltas vs `default`:
- `--tensor-parallel-size 2`, `CUDA_VISIBLE_DEVICES=0,1`
- Same `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` as tp4
- `--max-model-len 100000`, `--max-num-batched-tokens 4128`, `--max-num-seqs 2`
- `--gpu-memory-utilization 0.93`
- `shm_size: "24gb"`

Cold start ~235 s. Bench:

```
Warmup:    28.6 (cold) / 93.3 / 93.5 TPS
Narrative: 93.5 / 90.2 / 95.9 TPS
Code:      119.9 / 118.8 TPS
GPU state: 22.4 GB / 85% util / 330 W per card on GPU 0–1; GPU 2–3 idle
MTP avg:   54–62%
```

Long-context smoke: 90,053-token prompt → 14 tokens generated in 70.7 s, needle recalled correctly.

### TP=4 multi-GPU variant (`compose/docker-compose.tp4.yml`, local addition)

Not part of the upstream recipe — added here to use all four 3090s on this box. Sharded weights cut per-card model footprint to ~4.5 GB, leaving ~19.5 GB per card for KV. Total KV pool: 484,800 tokens at fp8_e5m2 (vs 24K on single-GPU `text` variant).

Config deltas vs `default`:
- `--tensor-parallel-size 4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` — required for consumer 3090s without NVLink (P2P is unreliable; the all-reduces go through host memory)
- `--max-model-len 500000`, `--max-num-batched-tokens 4128` (Mamba block-size constraint at high ctx)
- `--max-num-seqs 1` (the entire pool goes to one request — required to reach this context size)
- `--gpu-memory-utilization 0.95`
- `shm_size: "32gb"` (multi-process IPC for TP workers)
- No Genesis (fp8_e5m2 KV doesn't need it)

Note on `--max-model-len` vs KV pool size: vLLM logs `GPU KV cache size: 507,200 tokens` — that's the raw page count for the configured `gpu-memory-utilization` and `max-num-seqs`. With `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` set (recipe default), vLLM's pre-check is permissive — it accepts essentially any `--max-model-len` we set — but a single request's KV physically must fit in the pool, so the *honest* ceiling is the pool size. We tested 470K end-to-end (recalled "BRONZE-DRAGON-555" needle correctly in 475 s) and settled on a 500K request cap with a small safety margin against the 507K pool.

Sweep over the iterations on this hardware:

| Setting                                           | Pool tokens | Tested cap | Notes |
|---------------------------------------------------|-------------|------------|-------|
| `util=0.92` `max_seqs=2` `max_len=150000`         | 484,800     | 135 K      | first version |
| `util=0.92` `max_seqs=2` `max_len=256000`         | 484,800     | 230 K      | bumped cap, same pool |
| **`util=0.95` `max_seqs=1` `max_len=500000`**     | **507,200** | **470 K**  | dropped to one seq, won full pool — current |

Cold start ~235 s. Bench:

```
Narrative: 96.6 / 88.4 / 91.7 TPS
Code:      117.4 / 116.1 TPS
GPU state: 22.4 GB on each card, 37–88% util, 232–264 W per card (default 350 W cap)
MTP avg:   52–65%, climbs over warmup
```

Long-context smoke (current 500K config): 470,057-token prompt → 16 tokens generated in 475 s, "BRONZE-DRAGON-555" needle recalled correctly.

**TP scaling on this rig (decode TPS, narrative bench):**

| TP | Narr TPS | KV pool | Speedup vs TP=1 | Notes |
|----|----------|---------|-----------------|-------|
| 1  | 70–72    | ~24K    | 1.0×            | baseline (default variant) |
| 2  | 90–96    | 172K    | **~1.30×**      | sweet spot |
| 4  | 88–97    | 485K    | ~1.30×          | same TPS as TP=2, 3× the KV |

Decode is bandwidth-bound: every transformer layer all-reduces across all TP ranks, and on consumer 3090s without NVLink that all-reduce traverses host memory. Doubling cards 1→2 gives a ~30 % win because the parallel compute beats the *single* added all-reduce. Doubling again 2→4 gives nothing, because the all-reduce now does a 2-hop tree with 4-way PCIe contention, and the compute-per-card already wasn't the limit. So the TP=4 variant is **not** the right default for TPS — its only advantage is the bigger KV pool.

The setup that makes 2× / 4× viable at all:
- All cards on the same NUMA node (verified via `nvidia-smi topo -m`, all "NODE")
- `NCCL_P2P_DISABLE=1` forces NCCL through host memory — the path that works reliably on consumer cards (direct P2P over PCIe on 3090s is buggy and causes hangs)
- Per-shard model footprint stays small enough (~4.5 GB on TP=4, ~8.9 GB on TP=2) for SM occupancy

Topology matters: a rig with cards split across two NUMA sockets would pay a QPI/UPI hop on every all-reduce and the curve would shift further toward "lower TP is better."

### Long-ctx 125K variant (`compose/docker-compose.longctx-experimental.yml`)

Edits from upstream:
- `CUDA_VISIBLE_DEVICES=0`
- Genesis volume mounts replaced (new `vllm/_genesis` package + `genesis_vllm_plugin`)
- Entrypoint replaced (`pip install -e` plugin, then `python3 -m vllm._genesis.patches.apply_all`)
- `GENESIS_GDN_MAX_BATCHED_TOKENS=4128` env var added
- `--gpu-memory-utilization 0.95` (was 0.97)
- `--compilation-config '{"cudagraph_mode":"NONE"}'` replaced with `--enforce-eager`
- Stock otherwise: `turboquant_3bit_nc` KV, MTP n=3, 125 K context, vision on, tools on

Boot in ~150 s (more apply_all + KV pool sizing). Bench:

```
Warmup:    30.4 / 32.7 / 31.4 TPS
Narrative: 29.8 / 32.6 / 30.7 TPS
Code:      42.0 / 41.7 TPS
MTP avg:   52% → 77% → 80% across the run, position-wise climbs from [83,49,23] to [91,81,68]
```

KV pool: 210,528 tokens. GPU mem 24.12/24.57 GB. Power 284 W (default cap 350 W).

## Sibling: Qwen3.6-35B-A3B (MoE) — single 3090 = llama.cpp GGUF, multi-GPU = vLLM AWQ

Added 2026-04-26. Sibling MoE model — 35.95 B total params, 3 B active per token (qwen3_5_moe). Manage via `qwen-moe.sh` at the repo root.

**Engine choice: it depends on the GPU count.** A single-3090 deployment must use llama.cpp+GGUF, not vLLM. A 2-GPU deployment can stay in vLLM with AWQ. Direct measurement:

| Variant | Engine / quant | Max ctx | Gen TPS | SM util | Vision | VRAM | Status |
|---|---|---|---|---|---|---|---|
| **gguf** (`docker-compose.35b-a3b-gguf.yml`) | llama.cpp / IQ4_XS GGUF | 128 K | **115–133** | 96 % | ❌ (text-only) | 18.8 GB × 1 | **recommended for 1× GPU** |
| tp1 (`docker-compose.35b-a3b-awq-tp1.yml`) | vLLM / AWQ-INT4 | 20 K | ~18.6 | 22 % | ❌ | 23.4 GB × 1 | kept for parity, not recommended |
| **tp2** (`docker-compose.35b-a3b-awq.yml`) | vLLM / AWQ-INT4 | 200 K | **~149** | — | ✅ | 22.6 GB × 2 | **recommended for 2× GPU** |

The TP=1 vLLM number isn't a vLLM bug — it's a hard physics consequence of fitting a 21.6 GB AWQ MoE on 24 GB and being forced into `--enforce-eager` (no room for CUDA graphs or torch.compile cache). Detail below.

### Why single-3090 vLLM/AWQ collapses to 18 TPS

vLLM's `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (~22.78 GB on disk) requires **21.56 GiB** GPU memory after load. On a 24 GB 3090 that leaves ~0.7 GB for everything else. Both decode-time optimizations want exactly that ~0.7 GB:

- **CUDA graphs** (the default) capture into ~0.7 GB. With them on: `Available KV cache memory: -0.11 GiB` → engine refuses to start.
- **torch.compile inductor fusion at compile_sizes=[1]** writes the compiled graph cache into ~0.7 GB. With it on (cudagraphs off): `Available KV cache memory: 0.01 GiB` → KV pool can't hold even one request.
- **PIECEWISE cudagraphs at capture_sizes=[1]** (smallest possible): same ~0.7 GB swallowed.

Forced fallback: `--enforce-eager`. With every kernel launched from Python, the MoE expert routing path becomes a parade of small dispatches per layer per token. Direct GPU profiling under decode:

```
GPU 0 stats during vLLM/AWQ TP=1 decode (1500-tok generation):
SM utilization:    22 %
Memory BW util:    12 %
Power draw:        142 W (of 350 W cap)
Generation rate:   18.6 TPS
```

The GPU is idle ~78 % of decode time waiting for the next launch. Memory bandwidth is at 12 % (not memory-bound). Power 142 W (not power-bound). Pure kernel-launch-overhead bound — and the MoE makes this dramatically worse than the dense 27B because each layer has its expert dispatch as a separate small kernel.

The fix isn't "tune vLLM more"; it's "use a smaller weight quant so the optimization machinery has room to work." llama.cpp's IQ4_XS GGUF is **17.7 GB on disk → 18.8 GB GPU resident**, which leaves ~5 GB headroom for KV cache + compute graphs and lets the GPU saturate.

### Single-GPU sweet spot: llama.cpp + IQ4_XS GGUF (`docker-compose.35b-a3b-gguf.yml`)

Endpoint `:8022/v1`, model name `qwen3.6-35b-a3b-gguf`, container `llamacpp-qwen36-35b-a3b`. Image `ghcr.io/ggml-org/llama.cpp:server-cuda` (3.6 GB). Model `unsloth/Qwen3.6-35B-A3B-GGUF` file `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (~17 GB).

Bench at default config (128K ctx, q8_0 KV, flash-attn on):

```
GPU 0 stats during llama.cpp/IQ4_XS decode (1500-tok generation):
SM utilization:    96 %
Memory BW util:    52 %
Power draw:        317 W (of 350 W cap)
VRAM used:         18.8 GB / 24 GB
Generation rate:   115–133 TPS (3 runs)
```

Long-context recall (needle-in-haystack):

```
prompt   4K → prefill 2960 tok/s → HIT 'MAGENTA-OWL-7'
prompt  16K → prefill 3922 tok/s → HIT
prompt  32K → prefill 5367 tok/s → HIT
prompt  60K → prefill 4941 tok/s → HIT (filler ran out beyond this in test rig)
```

Config (in compose):

- `--n-gpu-layers 999 --ctx-size 131072 --parallel 1`
- `--flash-attn on -ctk q8_0 -ctv q8_0` (q8_0 KV is what makes 128K fit alongside 18 GB weights on a 24 GB card)
- `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` (Qwen3.6 recipe sampling defaults from the LocalLLaMA recipe)
- `--jinja --no-mmap`
- Pinned to GPU 0 (`CUDA_VISIBLE_DEVICES=0`)

This matches the published 3090 benchmark from `thc1006/qwen3.6-speculative-decoding-rtx3090` (135.7 tok/s with UD-Q4_K_XL; we're 1 quant tier smaller for more KV room and land at ~125 TPS). One commenter on the original benchmark thread put it bluntly: *"vllm is not going to run on a single 3090 with this model."* Confirmed.

### What NOT to do on single 3090 + A3B

- **Don't enable speculative decoding.** Per the same external bench (19-config matrix on a 3090): every spec-decode variant — `ngram-cache`, `ngram-mod`, classic draft with vocab-matched Qwen3.5-0.8B — is **net-negative** even at 100% draft acceptance. Mechanism: 8/256 expert routing means the verify pass loads the *union* of K positions' expert sets; at K ≪ 94 (the expert-saturation threshold derived in MoESD arXiv 2505.19645), the union load wipes out the savings from skipping forward passes. Worst case −60 % on structured prompts. A10B-class MoEs reach the threshold and gain from spec-decode; A3B does not on consumer Ampere.
- **Don't use vLLM AWQ on the single 3090.** See §"Why single-3090 vLLM/AWQ collapses to 18 TPS" above.
- **Don't run UD-Q4_K_M or UD-Q4_K_L** without checking: per a HF discussion thread linked from the benchmark post, some Unsloth quants left BF16 layers in (Unsloth eventually deleted UD-Q4_K_M for this reason; UD-Q4_K_L still has them as of writing). UD-IQ4_XS / UD-IQ4_NL / current Q4_K_S are clean.

### TP=2 two-GPU configuration (`docker-compose.35b-a3b-awq.yml`)

The recommended deployment when you have 2 GPUs free. Sharding halves the per-card model footprint (~10 GB each) and unlocks generous KV space:

- `--tensor-parallel-size 2`, `CUDA_VISIBLE_DEVICES=0,1`
- `--max-model-len 200000` (KV pool ceiling is 247K tokens; 200K leaves a ~47K-token headroom margin)
- `--max-num-batched-tokens 2048` (well above the 1056 Mamba floor)
- `--max-num-seqs 1` (single user)
- `--gpu-memory-utilization 0.95`
- Vision **enabled** (no `--language-model-only`)
- CUDA graphs **enabled** (no `--enforce-eager`)
- Same `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` as the 27B `tp2.yml` (consumer 3090s without NVLink)

Cold start ~200 s. Bench at this config:

```
Generation TPS: 148.8 / 148.7 / 149.0  (3× 600-token completions)
Prefill: 10.2K tok/s sustained at 60K-token prompt
Needle recall: HIT at 4K, 32K, 60K input prompts
Per-card: ~22.6 GiB used, both GPUs
```

That's **~2× the 27B's TP=2 throughput** (149 vs ~93 TPS) — exactly the MoE A3B speed advantage of activating only 3B params per token. Quality is lower than the 27B per Qwen's own benchmarks (1–10 pts depending on task; the 27B wins on every reasoning/coding axis), but for fast/cheap calls or large-context summarization the MoE is excellent.

**Untapped TP=2 optimization (not yet applied):** per the sibling repo `thc1006/qwen3.6-vllm-2x3090` clean A/B retest on 2× 3090, enabling `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` AND `--no-enable-prefix-caching` flips MTP from a regression to **+27.5% decode rate**. Tradeoff: prefix caching off, so repeated-prompt sessions lose the cache speedup. The previously-published vLLM MTP −12 % was confounded by prefix caching being on (vllm #38182: MTP drops cache hit rate ≈92 % → ≈71 %, masking compute speedup with cache-loss penalty). Worth probing on our rig if we want to push past 149 TPS.

### TP=2 max-context probe results

```
max-model-len    boot      KV pool        verdict
131,072 (128K)   200s      247K tokens    READY
196,608 (192K)   196s      247K tokens    READY
262,144 (256K)   200s      247K tokens    READY
327,680 (320K)   timeout (>240s)          rejected — pool ceiling exceeded
```

Pool size is constant at 247K because it's set by available memory after model load, independent of `max-model-len`. Settled on **200K** for production: comfortable margin against the 247K ceiling, fits any current workload.

### Quant choice rationale

| Quant | Size on disk | Single-3090? | TP=2 fit | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` (bf16) | ~72 GB | ❌ | ❌ | needs 4× 3090 with TP=4 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | ~36 GB | ❌ | ✅ | higher quality, ~30% slower than AWQ |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | ~18 GB | — | — | **Blackwell-only**, won't run on Ampere |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | ~23 GB | borderline | ✅ | vLLM-only path; recommended for TP=2 |
| **`unsloth/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`** | **~17 GB** | ✅ | n/a | **chosen for single-3090 (llama.cpp)** |

### Findings: dense 27B vs MoE 35B-A3B on this rig

The 27B-int4 (AutoRound, AutoRound's best) **wins on every quality benchmark** Qwen publishes (MMLU-Pro 86.2 vs 85.2, GPQA 87.8 vs 86.0, AIME 94.1 vs 92.7, HMMT-Feb 93.8 vs 83.6, SWE-bench 77.2 vs 73.4, LiveCodeBench 83.9 vs 80.4, Terminal-Bench 59.3 vs 51.5). The MoE wins on **speed** (~150 vs ~93 TPS at TP=2; ~125 vs ~70 TPS at single-GPU) and on memory-bound configurations (partial offload, low-VRAM rigs).

Side-by-side coexistence is possible but tight: the MoE TP=2 takes GPUs 0,1 — the 27B can run on its own `tp2.yml` if you re-pin its `CUDA_VISIBLE_DEVICES` to `2,3`. Both endpoints (`:8020` 27B / `:8021` MoE) then live independently. With the GGUF mode on GPU 0, plenty of room remains for the 27B `tp2.yml` on GPUs 1,2 (or 2,3) — three independent endpoints `:8020 / :8021 / :8022` can coexist within total GPU budget.

Both validated end-to-end: needle-in-haystack recall HIT at 60K real tokens, smoke chat completions correct.

### TP=1 single-GPU constraints (why it's so slow)

AWQ-INT4 weights take **21.56 GiB** on the 24 GiB 3090 (vs 18 GiB for the 27B-AutoRound — the MoE has more total parameters even after Q4 packing). After load that leaves only ~0.7 GiB for activations + KV pool, and several knobs become forced:

1. **`--language-model-only` (no vision).** The MoonViT vision tower's `fast_pos_embed_interpolate` allocates ~0.5 GiB during dummy profiling and OOMs even at 16 K context. Same workaround as the 27B `tools-text` variant.
2. **`--enforce-eager` (no CUDA graphs).** With CUDA graphs on, vLLM reports `Available KV cache memory: -0.11 GiB` — the graph capture pool eats the last GB. Eager mode frees that ~0.8 GiB for KV but costs **~8× decode speed** on this MoE (18 vs 149 TPS) — graph capture is dramatically more important here than on the 27B (which loses ~10–20 % from eager mode). Likely cause: many small expert kernels per token, each paying a launch-overhead tax that graphs amortize.
3. **`--max-num-batched-tokens 1088`.** Hard floor: the qwen3_5 hybrid uses Mamba-style SSM layers whose cache `align` mode requires `block_size ≤ max_num_batched_tokens`, with block_size = 1056 at this context size. Set lower → `AssertionError: In Mamba cache align mode, block_size (1056) must be <= max_num_batched_tokens`.
4. **`--max-num-seqs 1`.** Single-user only; KV pool can't serve concurrent sessions.
5. **`--gpu-memory-utilization 0.96`.** A hair higher than the 0.95 default, to scrape another 240 MiB into the KV pool.
6. **`--quantization compressed-tensors`.** Selects vLLM's `CompressedTensorsWNA16MarlinMoEMethod` for the AWQ MoE weights. Auto-detect would also pick this from the model's `recipe.yaml`, but pin explicitly.
7. **No `--kv-cache-dtype fp8_e5m2`.** vLLM's compressed-tensors MoE path rejects fp8 KV (`ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints` — misleading message; the checkpoint is INT4 not FP8). Fine in practice: the model has only 2 KV heads × 40 layers × 256 head_dim, so KV per token is light (~80 KB at fp16).

KV pool ends up at 0.69 GiB ≈ 8,448 token-equivalents — but because 24 of 40 layers are Mamba (constant-size state, not growing-with-position), real prompts much longer than 8 K still fit. Smoke tested up to 15 K input correctly; vLLM's startup check accepts max-model-len up to 20 K with this pool.

### TP=2 two-GPU configuration (`docker-compose.35b-a3b-awq.yml`)

The recommended deployment. Sharding halves the per-card model footprint (~10 GB each) and unlocks generous KV space:

- `--tensor-parallel-size 2`, `CUDA_VISIBLE_DEVICES=0,1`
- `--max-model-len 200000` (KV pool ceiling is 247K tokens; 200K leaves a ~47K-token headroom margin)
- `--max-num-batched-tokens 2048` (well above the 1056 Mamba floor)
- `--max-num-seqs 1` (single user)
- `--gpu-memory-utilization 0.95`
- Vision **enabled** (no `--language-model-only`)
- CUDA graphs **enabled** (no `--enforce-eager`)
- Same `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` as the 27B `tp2.yml` (consumer 3090s without NVLink)

Cold start ~200 s. Bench at this config:

```
Generation TPS: 148.8 / 148.7 / 149.0  (3× 600-token completions)
Prefill: 10.2K tok/s sustained at 60K-token prompt
Needle recall: HIT at 4K, 32K, 60K input prompts
Per-card: ~22.6 GiB used, both GPUs
```

That's **~2× the 27B's TP=2 throughput** (149 vs ~93 TPS) — exactly the MoE A3B speed advantage of activating only 3B params per token. Quality is lower than the 27B per Qwen's own benchmarks (1–10 pts depending on task; the 27B wins on every reasoning/coding axis), but for fast/cheap calls or large-context summarization the MoE is excellent.

### TP=2 max-context probe results

```
max-model-len    boot      KV pool        verdict
131,072 (128K)   200s      247K tokens    READY
196,608 (192K)   196s      247K tokens    READY
262,144 (256K)   200s      247K tokens    READY
327,680 (320K)   timeout (>240s)          rejected — pool ceiling exceeded
```

Pool size is constant at 247K because it's set by available memory after model load, independent of `max-model-len`. Settled on **200K** for production: comfortable margin against the 247K ceiling, fits any current workload.

### Quant choice rationale

| Quant | Size on disk | Single-3090? | TP=2 fit | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` (bf16) | ~72 GB | ❌ | ❌ | needs 4× 3090 with TP=4 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | ~36 GB | ❌ | ✅ | higher quality, ~30% slower than AWQ |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | ~18 GB | — | — | **Blackwell-only**, won't run on Ampere |
| **`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`** | **~23 GB** | **borderline** | **✅** | **chosen** — best speed/quality trade for this rig |
| `unsloth/Qwen3.6-35B-A3B-GGUF` | varies | yes | n/a | llama.cpp only (different stack) |

AWQ-INT4 via `compressed-tensors` was the only quant fitting all of: vLLM-native, runs on Ampere, fits on 1× 3090 (with concessions) AND scales to TP=2 cleanly.

### Findings: dense 27B vs MoE 35B-A3B on this rig

The 27B-int4 (AutoRound, AutoRound's best) **wins on every quality benchmark** Qwen publishes (MMLU-Pro 86.2 vs 85.2, GPQA 87.8 vs 86.0, AIME 94.1 vs 92.7, HMMT-Feb 93.8 vs 83.6, SWE-bench 77.2 vs 73.4, LiveCodeBench 83.9 vs 80.4, Terminal-Bench 59.3 vs 51.5). The MoE wins on **speed** (~150 vs ~93 TPS at TP=2) and on memory-bound configurations (partial offload, low-VRAM rigs).

Side-by-side coexistence is possible but tight: the MoE TP=2 takes GPUs 0,1 — the 27B can run on its own `tp2.yml` if you re-pin its `CUDA_VISIBLE_DEVICES` to `2,3`. Both endpoints (`:8020` 27B / `:8021` MoE) then live independently.

## Files modified / added

```
qwen36-27b-single-3090/compose/docker-compose.yml                       (modified)
qwen36-27b-single-3090/compose/docker-compose.tools-text.yml            (modified)
qwen36-27b-single-3090/compose/docker-compose.longctx-experimental.yml  (modified)
qwen36-27b-single-3090/compose/docker-compose.tp2.yml                   (new — 2-GPU variant, TPS sweet spot)
qwen36-27b-single-3090/compose/docker-compose.tp4.yml                   (new — 4-GPU variant, max context)
qwen36-27b-single-3090/compose/docker-compose.35b-a3b-awq.yml           (new — MoE vLLM TP=2, 200K ctx, 2× GPU recommended)
qwen36-27b-single-3090/compose/docker-compose.35b-a3b-awq-tp1.yml       (new — MoE vLLM TP=1, kept for posterity, 18 TPS)
qwen36-27b-single-3090/compose/docker-compose.35b-a3b-gguf.yml          (new — MoE llama.cpp IQ4_XS, 1× GPU recommended, 125 TPS)
qwen36-27b-single-3090/scripts/probe_moe_ctx.sh                         (new — context-fit probe for MoE variants)
qwen36-27b-single-3090/models/qwen3.6-35b-a3b-awq-int4/                 (downloaded, ~23 GB)
qwen36-27b-single-3090/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf            (downloaded, ~17 GB)
qwen-moe.sh                                                             (new — MoE launcher, sibling to qwen.sh, 3 modes)
```

## Things not done

- **`scripts/verify-full.sh`** (full functional test — streaming, thinking, needle recall). The post-Genesis log markers it greps for differ from the upstream-Genesis ones (e.g. `[OK] Qwen3 tool_call fix` doesn't appear in modular Genesis logs), so the script needs minor adaptation.
- **MTP-acceptance vs. README parity.** README reports 78–92% per-position accept; we see 52–80% on the long-ctx variant. The Genesis P60/P60b "SSM state pre-copy" patches that were credited with +23–40 pp acceptance are present in the modular package but **opt-in only** (`GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1`, `GENESIS_ENABLE_P60B_TRITON_KERNEL=1`). They're scoped to the ngram path per Genesis README, not MTP, so enabling them probably won't help us — but worth a probe.
- **Power-cap sweep.** README cites a +10% TPS uplift at 330 W vs. 230 W. We ran at default 350 W and saw 284 W under load (SM clocks saturate); no cap configured.
- **Tool-calling regression test under enforce-eager.** Verified at the basic level (`get_weather` populates `tool_calls[]`) on the default variant; not re-run on long-ctx variant. Would expect it to work given Genesis P12/P15/P27 are all applied.

## Pointers

- **Endpoint:** `http://localhost:8020/v1/*` (OpenAI-compatible, no auth)
- **Model name:** `qwen3.6-27b-autoround`
- **Container name:** `vllm-qwen36-27b`
- **Working tree:** `/home/thomas/projects/qwen3.6/qwen36-27b-single-3090/`
- **Model weights:** `…/models/qwen3.6-27b-autoround-int4/` (~18 GB, 10 shards)
- **vLLM image:** `vllm/vllm-openai@sha256:9bba4628a3b943e0dd33caefb94b811569ba1e97bdf23bee19a265c31b947ccb` (~24 GB)
- **Genesis tree:** `…/patches/genesis/` (full clone of `Sandermage/genesis-vllm-patches` main)
- **Tracking issues upstream:**
  - [vllm#40807](https://github.com/vllm-project/vllm/issues/40807) — CUDA graph `.tolist()` crash (worked around locally)
  - [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) — TurboQuant × spec-decode output corruption (closed for ngram, open for MTP via #40880)
  - [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) — MTP × TurboQuant × cudagraph (open; our `--enforce-eager` workaround supersedes this)
  - [vllm#40069](https://github.com/vllm-project/vllm/issues/40069) — TurboQuant + spec-decode/hybrid tracker
  - [noonghunna#2](https://github.com/noonghunna/qwen36-27b-single-3090/issues/2) — Genesis upstream restructure
  - [noonghunna#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) — `--enforce-eager` config corroboration

## Repro commands

```bash
# Switch between variants (both can't run simultaneously — same container name + port).
cd /home/thomas/projects/qwen3.6/qwen36-27b-single-3090/compose

# Default 20K variant
docker compose down
docker compose up -d
docker logs -f vllm-qwen36-27b   # wait for "Application startup complete"

# Tools-text 75K variant
docker compose down
docker compose -f docker-compose.tools-text.yml up -d
docker logs -f vllm-qwen36-27b   # ~180 s cold

# Long-ctx 125K variant
docker compose down
docker compose -f docker-compose.longctx-experimental.yml up -d
docker logs -f vllm-qwen36-27b   # ~150 s cold

# 2-GPU variant (TPS sweet spot)
docker compose down
docker compose -f docker-compose.tp2.yml up -d
docker logs -f vllm-qwen36-27b   # ~235 s cold

# 4-GPU variant (max context)
docker compose down
docker compose -f docker-compose.tp4.yml up -d
docker logs -f vllm-qwen36-27b   # ~235 s cold

# Smoke + bench (against whichever is up)
cd ..
curl -sf http://localhost:8020/v1/models
bash scripts/bench.sh
```
