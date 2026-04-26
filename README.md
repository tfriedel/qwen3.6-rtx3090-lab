# Qwen3.6-27B local inference

Local OpenAI-compatible endpoint for **Qwen3.6-27B** running on one RTX 3090 via vLLM. Replicated from [`noonghunna/qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) with the deltas this box needs documented in [`FINDINGS.md`](./FINDINGS.md).

## Layout

```
qwen3.6/
├── README.md                   this file
├── FINDINGS.md                 technical writeup — recipe deltas + bench results
├── qwen.sh                     start/stop/status helper for the four variants
└── qwen36-27b-single-3090/     cloned recipe (with local edits)
    ├── compose/
    │   ├── docker-compose.yml                       default 20K + vision (1 GPU)
    │   ├── docker-compose.tools-text.yml            75K text-only (1 GPU)
    │   ├── docker-compose.longctx-experimental.yml  125K + vision (1 GPU, eager)
    │   ├── docker-compose.tp2.yml                   100K + vision (2 GPUs, TP)  ← local addition
    │   └── docker-compose.tp4.yml                   150K + vision (4 GPUs, TP)  ← local addition
    ├── patches/
    │   ├── patch_tolist_cudagraph.py
    │   └── genesis/                                 Sandermage/genesis-vllm-patches clone
    ├── models/
    │   └── qwen3.6-27b-autoround-int4/              ~18 GB, Lorbus INT4 AutoRound
    └── scripts/                                     bench.sh, verify.sh, etc.
```

## Quick start

```fish
cd ~/projects/qwen3.6

./qwen.sh start default      # 20K context + vision (recommended for chat)
./qwen.sh status
./qwen.sh logs               # follow startup
./qwen.sh stop
```

The endpoint is OpenAI-compatible at `http://localhost:8020/v1`, model name `qwen3.6-27b-autoround`, no auth (any non-empty key is accepted).

## Variants

Pick the one matching your workload. Only one runs at a time (they share container name + port).

| Variant   | GPUs | Max ctx | TPS (narr / code) | Vision | Tools | KV dtype           | Genesis | When to use |
|-----------|------|---------|-------------------|--------|-------|--------------------|---------|-------------|
| `default` | 1    | 20 K    | 70–72 / 91        | yes    | yes   | `fp8_e5m2`         | no      | quick chat / vision, frees other GPUs |
| `text`    | 1    | 75 K    | 68–72 / 92        | no     | yes   | `fp8_e5m2`         | no      | long documents, code review, no images |
| `longctx` | 1    | 125 K   | 30–33 / 42        | yes    | yes   | `turboquant_3bit_nc` | yes (modular plugin) | full-repo + vision on a single card |
| **`tp2`** | 2    | 100 K   | **90–96 / 120**   | yes    | yes   | `fp8_e5m2`         | no      | **recommended default — TPS sweet spot, 100K ctx, frees 2 GPUs for other work** |
| `tp4`     | 4    | 500 K   | 88–97 / 117       | yes    | yes   | `fp8_e5m2`         | no      | only if you need >100K context — same TPS as tp2, costs 2 extra GPUs; needle verified at **470K (475 s)** |

Switch:

```fish
./qwen.sh restart longctx    # tears down current, brings up new
```

Cold start ~90 s (default), ~180 s (text), ~150 s (longctx), ~235 s (tp2 / tp4). The helper waits for `/v1/models` to respond before returning.

## Using the endpoint

Anywhere that talks OpenAI:

| Client | Configuration |
|---|---|
| **Open WebUI** | Settings → Connections → OpenAI API → URL `http://host.docker.internal:8020/v1`, key any string |
| **OpenCode** | `~/.config/opencode/config.json` provider entry, `base_url` `http://localhost:8020/v1` |
| **Aider** | `aider --openai-api-base http://localhost:8020/v1 --openai-api-key sk-x --model openai/qwen3.6-27b-autoround` |
| **Continue / Cline (VS Code)** | "OpenAI-compatible" provider, same URL |
| **curl** | `curl http://localhost:8020/v1/chat/completions -d '{"model":"qwen3.6-27b-autoround", "messages":[...]}'` |
| **Claude Code** | not directly — needs an Anthropic↔OpenAI proxy (e.g. `claude-code-router`) |

Tool calling is enabled (`qwen3_coder` parser), so agentic CLIs that make function calls work end-to-end.

## Bench numbers on this box

Default 20K, GPU 0 at default 350 W cap:

```
Warmup:    71.6 / 69.5 / 68.9 TPS   (1000-tok narrative)
Narrative: 70.9 / 71.0 / 72.2 TPS
Code:      90.6 / 91.9 TPS
MTP avg accept: 58–68%
```

Run yourself: `bash qwen36-27b-single-3090/scripts/bench.sh`.

Published headline ("85 TPS at 20K") matches our **code peak** on this rig. Narrative lands at the lower end of the published 60–105 band because the reference rig was capped at 230 W vs. our default 350 W.

Long-ctx 125K bench: 30–33 TPS narrative / 42 TPS code, KV pool 210K tokens, GPU 24.1/24.5 GB.

`tp2` 100K bench: **90–96 TPS narrative / 120 TPS code**, KV pool 172,800 tokens, 22.4 GB on each of GPU 0–1, 330 W per card. Verified end-to-end at 90K-token needle prompt (71 s).

`tp4` 500K bench: 88–97 TPS narrative / 117 TPS code, KV pool **507,200 tokens** (at `--gpu-memory-utilization 0.95`, `--max-num-seqs 1`), 22.4 GB on each of GPU 0–3. **Verified at 470K-token needle** (recalled correctly in 475 s). 500K is the practical ceiling on this rig — the request KV must fit inside the pool, and pushing higher requires more VRAM than four 3090s have.

**Tensor-parallel scaling on this rig:** doubling cards from 1→2 gives ~30 % decode speedup; doubling again from 2→4 gives essentially nothing. Decode is bandwidth-bound — every transformer layer does an all-reduce, and on consumer 3090s without NVLink that all-reduce traverses host memory. TP=2 is one round-trip; TP=4 is a tree-reduce with two hops and four-card contention on the same PCIe fabric. The compute savings from sharding wider don't beat the comm overhead. **`tp2` is the right default for TPS-sensitive workloads; `tp4` is only worth running when you genuinely need >100K context.**

Caveat: this depends on PCIe topology. All 4 cards here are on the same NUMA node (`nvidia-smi topo -m` shows `NODE` everywhere). A rig split across NUMA nodes would see a QPI/UPI hop on every all-reduce and the curve would shift further toward "lower TP is better."

## Hardware

- 4× RTX 3090 (24 GB each, Ampere SM_86), all on NUMA node 0, no NVLink
- Driver 595.x / CUDA 13.2
- ~25 GB free disk used (model + image)

Single-GPU variants (`default`, `text`, `longctx`) pin to GPU 0 via `CUDA_VISIBLE_DEVICES=0`; the `tp4` variant uses 0,1,2,3. Edit the env var in the active compose file to swap which card(s) get used.

## Why the local edits

The upstream recipe was published when Sandermage's `genesis-vllm-patches` shipped a monolithic `patch_genesis_unified.py`. That file was removed on **2026-04-24** (one day before we replicated). The compose entrypoints reference the missing file. We worked around it differently per variant:

- `default` and `text`: no Genesis needed (fp8_e5m2 KV bypasses the TurboQuant hybrid gate that Genesis was patching). Volume mount commented out, entrypoint gated on file existence.
- `longctx`: wired the new modular Genesis (`vllm/_genesis/` package + `genesis_vllm_plugin/` entry-point); also `--enforce-eager` (Genesis's static-shape preallocs are incompatible with `torch.compile` Dynamo), `GENESIS_GDN_MAX_BATCHED_TOKENS=4128` (P28 prealloc must match `--max-num-batched-tokens`), and `--gpu-memory-utilization 0.95` (Genesis adds extra preallocs on top of the KV pool).

Full breakdown including reproduction error messages and resolutions: [`FINDINGS.md`](./FINDINGS.md).

## Troubleshooting

```fish
./qwen.sh status            # check what's up
./qwen.sh logs              # follow container logs
docker stats vllm-qwen36-27b   # live GPU/CPU usage

# port already in use? something else owns 8020
lsof -i :8020

# "out of memory" on startup:
#   - default mode uses 22 GB; check no other GPU process: nvidia-smi
#   - longctx mode is at 24/24.5 GB; lower --gpu-memory-utilization in the compose

# model recall returning gibberish on long context: drop max_tokens in the request,
# or switch to longctx variant if you're hitting the 75K cap on text mode.
```

Bigger issues: see [`FINDINGS.md`](./FINDINGS.md) § Things not done + § Pointers (upstream issue links).
