# Qwen3.8-27B on 4× RTX 3090 — day-2 findings (2026-08-15)

Qwen released **Qwen3.8-27B** on 2026-08-13 ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0)
alongside the closed Qwen3.8-Max. This file records how it was added to this lab and what it does on this rig.

## TL;DR

- **`./qwen38.sh start tp2` → 108–114 narr / 145–147 code TPS** on 2× 3090, 100K ctx, vision + tools + thinking,
  prefix caching ON. Endpoint `http://localhost:8026/v1`, model name `qwen3.8-27b-awq`.
- **Zero engine work was needed.** Qwen3.8-27B is architecturally *identical* to Qwen3.6-27B: same
  `model_type: qwen3_5` / `Qwen3_5ForConditionalGeneration`, same 64-layer `3×(GatedDeltaNet→FFN) →
  1×(GatedAttention→FFN)` hybrid, same dims (5120 hidden / 17408 FFN), same 248320 vocab, same 262K native ctx,
  same 1-layer MTP head config keys. It is a continued-training generation on the 3.6 skeleton, so the pinned
  **2026-07-12 nightly image (v0.23.1rc1)** that serves 3.6 serves 3.8 unmodified — the whole port is two compose
  files + a launcher pointing at new weights.
- **Speed is 3.6-tp2-parity, not faster**: ~same TPS as `qwen.sh start tp2` (117–118/149–151). The 3.6 *throughput*
  trick did NOT carry over — see negative result below.
- **Why bother then:** quality. Qwen-published benchs vs 3.6-27B: Terminal-Bench 2.1 **73.0 vs 63.4**, GPQA-D
  **89.2 vs 87.8**, HLE **30.8 vs 24.0**, NL2Repo **42.3 vs 36.2**, OmniDocBench **91.1 vs 89.4** — but 3.6 keeps
  SWE-bench Pro (57.6 vs 53.5), LiveCodeBench v6 (89.6 vs 83.9) and IFBench. New in 3.8: `reasoning_effort`
  (xhigh/medium/low) and `preserve_thinking` chat-template controls; thinking is ON by default and the model
  skips thinking on trivial prompts by design.

## The quant

[`philbert440/Qwen3.8-27B-W4A16-AWQ`](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ) — 19.5 GB
compressed-tensors pack-quantized INT4 (AWQ g128, MSE observer), chosen over the alternatives because it is the
only 4-bit build that documents **all three** of: vision tower intact (333 BF16 tensors), **BF16 MTP head intact**
(`model-mtp.safetensors`, listed in `quantization_config.ignore`), and thinking-mode calibration (real
`<think>` traces — avoids the llm-compressor [#2680](https://github.com/vllm-project/llm-compressor/issues/2680)
empty-think-block corruption). Serves on stock vLLM, quantization auto-detected (no `--quantization` flag — the
3.6 recipes' `--quantization auto_round` does not apply here).

Alternatives noted, not benched:
- `cyankiwi/Qwen3.8-27B-AWQ-INT4` — 21 GB, g32 MSE (finer groups), vision + MTP in-shard. Same quantizer as our
  MoE AWQ; likely fine, arrived day-1 with no validation notes.
- `lued/Qwen3.8-27B-INT8-W8A16-MTP` — 31.6 GB W8A16, higher fidelity; would fit TP=2 with a smaller ctx budget.
- `goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound` — 28.3 GB AutoRound (nearest to our 3.6 default's pedigree, but
  oddly large and undocumented).
- `Lorbus` (our 3.6 quantizer, 1.2M downloads) had published **no 3.8 quant** as of 2026-08-15 — worth a re-check.
- GGUFs exist (unsloth 868K downloads, ggml-org official) — not pursued: llama.cpp was 3× slower than vLLM on the
  dense 3.6 and nothing suggests 3.8 differs.

## Measured (bench.sh, 2× 3090, idle host, P2P driver 610.43.02)

| Variant | Recipe | Narrative TPS | Code TPS | MTP accept |
|---|---|---|---|---|
| `qwen38 tp2` | k=3 + prefix cache ON (3.6 tp2 recipe, weights swapped) | **108–114** | **145–147** | mean len 2.66–3.48; per-pos 0.76–0.95 / 0.55–0.82 / 0.35–0.71 |
| `qwen38 tp2-mtp` k=4 | no prefix cache | 107–113 | 143–150 | pos-4 accept only **0.16–0.31** |
| `qwen38 tp2-mtp` k=3 | no prefix cache | 110–114 | 143–147 | mean len 2.60–2.85 |
| *(ref)* `qwen.sh tp2` (3.6) | k=3 + prefix cache | 117–118 | 149–151 | — |
| *(ref)* `qwen.sh tp2-mtp` (3.6) | k=4, no prefix cache | 117–127 | 164–169 | — |

GPU state during bench: both cards 100% util, ~21.7 GiB used, ~340 W.

### Negative result: the 3.6 "onegraph + k=4 + no-prefix-cache" throughput mode does not help 3.8

On 3.6, dropping prefix caching and raising MTP to k=4 bought +10% code TPS. On 3.8 the same recipe is **flat vs
tp2** at both k=4 and k=3. Root cause is visible in the SpecDecoding metrics: the 3.8 MTP head's acceptance decays
faster (position-4 acceptance 0.16–0.31; narrative mean acceptance length only 2.6–2.9 at k=3), so the extra draft
positions are mostly rejected and the saved verify passes never materialize. The quantizer's V100 card reports the
same shape (92.5% accept at K=2, "decaying acceptance" beyond). **Use `tp2`; `tp2-mtp` is kept (at k=3) purely for
future A/B** — e.g. if a quant with a better-preserved MTP head appears.

Consequence: 3.8 keeps prefix caching in its fastest mode, so the "re-submit the same large prompt" agentic case
is covered for free — the 3.6 tradeoff (tp2 vs tp2-mtp) does not exist for 3.8.

## Verified working

- Chat + thinking mode (reasoning content populated on non-trivial prompts; trivial prompts skip thinking by design)
- Vision (VLM path through the quantized body — solid-color probe answered correctly)
- Tool calling (`qwen3_coder` parser populates `tool_calls[]` correctly)
- MTP speculative decode (SpecDecoding metrics live, accept rates above)
- 100K `max_model_len` at `gpu-memory-utilization 0.93` on 2 cards, boot ~4 min

Not yet done: multi-turn tool-calling battery, long-ctx needle test, `reasoning_effort` A/B, tp4/500K variant, quality
side-by-side vs 3.6 on real coding tasks, cyankiwi-quant A/B.

## Files

```
qwen38.sh                                                        (new — launcher, port 8026, container vllm-qwen38-27b)
qwen36-27b-single-3090/compose/docker-compose.qwen38-tp2.yml     (new — recommended)
qwen36-27b-single-3090/compose/docker-compose.qwen38-tp2-mtp.yml (new — A/B experiment, no gain)
qwen36-27b-single-3090/models/qwen3.8-27b-awq-w4a16/             (downloaded, 19.5 GB)
```

Recipe deltas vs the 3.6 tp2 compose files (everything else byte-identical, incl. the FA2 FULL-cudagraph +
custom-allreduce env block): model path, served name, **no `--quantization` flag**, **no `--chat-template`
mount** (3.8's bundled `chat_template.jinja` carries the `reasoning_effort`/`preserve_thinking`/`enable_thinking`
logic; the 3.6 enhanced template predates those), port 8026.

## Community pointers (release megathread, r/LocalLLaMA `1voojjz`) — with local verification

**1. The weak MTP head is real, cross-engine, and temperature-dependent.** Multiple reports (llama.cpp on
2× A5000, vLLM on RTX 6000 Pro) of 3.8 MTP acceptance well below 3.6 (60–70% vs 80–90%). Community root-cause:
3.8's official *thinking* sampling is temp 1.0 (3.6's was 0.6), and higher temperature flattens the token
distribution the draft head has to hit. **Verified on this rig** (tp2, 800-tok narrative gen ×3):

| temp | mean acceptance length | avg draft acceptance |
|---|---|---|
| 1.0 (official thinking) | 2.34–2.43 | 44.6–47.6% |
| 0.6 (bench.sh default) | 2.60–2.85 | 53–62% |
| 0.3 | 2.58–2.59 | 52.6–52.9% |

So the headline TPS above (temp 0.6) is *optimistic* for thinking-mode workloads at official sampling — expect a
few TPS less at temp 1.0. This further cements the tp2-mtp negative result.

**2. `reasoning_effort` mechanics** (from the chat template — read locally to confirm): valid values are ONLY
`low` / `medium` / `xhigh`; anything else (incl. `high`) raises a template exception. It works by *system-prompt
injection*: `xhigh` (the default!) injects "think carefully… validate assumptions…", `medium` injects **nothing**,
`low` injects "keep thinking brief". Measured on the 12-coin puzzle: xhigh blew through an 8000-tok cap without
finishing; medium finished in 3,153 tok; low 3,258. Community reports the same 3–5× token burn at xhigh
(57K-tok one-shots). **Recommendation: pass `chat_template_kwargs: {"reasoning_effort": "medium"}` for daily
coding**, keep xhigh for peak-difficulty tasks. Caveat: switching effort mid-session changes the injected system
prompt → invalidates the prefix cache.

**3. Client config: vLLM puts thinking in `reasoning`, not `reasoning_content`.** Verified on this endpoint —
our v0.23.1rc1 build returns `message.reasoning`; llama.cpp uses `reasoning_content` (which is what the
megathread's OpenCode/Pi snippets assume). Point `reasoningField`/`compatibility.reasoningField` at `reasoning`
for this stack, and map thinking levels to low/medium/xhigh only.

**4. Agentic behavior shift:** 3.8 is strongly biased toward raw shell commands over configured MCP tools (likely
the Terminal-Bench training) — one user saw it ignore an explicitly-instructed filesystem MCP for reads. Watch
agent configs when swapping 3.6 → 3.8.

**5. Leads not yet tried on this rig:**
- [ninfer](https://github.com/Neroued/ninfer) — community reports 200 tok/s single-stream / 1,100 aggregate on a
  5090 (also `neroued/Qwen3.8-27B-NInfer` on HF). Ampere support unknown; biggest potential speed lever if it works.
- DSpark draft models ([llama.cpp PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173) merged;
  `RadixArk/Qwen3.8-27B-DSpark`) — mixed reports (one 5090 user: 100 tok/s; another: 0.29 acceptance, slower).
  DFlash drafter for 3.8 not out yet.
- exllamav3 6 bpw (`turboderp/Qwen3.8-27B-exl3`) — 70–150 tok/s at 220K ctx on a 5090.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v22) — fixes
  mid-thinking stops around tool calls seen in some UIs; try if tool-call glitches appear. The `peculiar-ragdoll`
  "Dirk" variant additionally suppresses xhigh + injects a conciseness prompt, but costs ~19% wall-clock in
  multi-turn (extra prefill every turn) and de-aligns the model from its xhigh-native post-training.

**6. Community quality consensus** (matches the benchmark table): clear step up for coding/agentic work (parallel
subagents, long tool-call chains work; "night and day" vs 3.6 on game one-shots), STEM/math roughly flat vs 3.6,
and the quality is gated on generous reasoning budgets — it buys capability with tokens.

## Sources

Primary: the HF model cards, [vLLM recipe page](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) (notes: vLLM ≥ 0.17.0,
MXFP4 broken on NVIDIA — NVFP4 is Blackwell-oriented and irrelevant to Ampere), and the quantizer's validation
writeup. Community: the r/LocalLLaMA release megathread (`1voojjz`) and its linked threads (`1vo9mj4`, `1vo9nn7`,
`1vo9qjv`, `1voa3ch`).
