# Qwen3.6-27B Tool Calling Issues

## Cause

This is a known, well-documented bug in the **Qwen3 parser stack**, not in your config or the model weights. Two compounding problems:

1. **Model emits `<tool_call>` while still inside an unclosed `<think>` block.** Qwen3.6 (and 3.5) sometimes skip the closing `</think>` and jump straight into a tool call. The failed output observed in the Pi agent log is exactly this — the entire `<tool_call>...</tool_call>` is parked inside the `thinking` field (`"thinkingSignature":"reasoning"`) and never escapes into `content` as a real toolCall, so `stopReason` becomes `"stop"` instead of `"toolUse"` and the agent thinks the task is done.

2. **`Qwen3ReasoningParser` treats the whole output as reasoning** when `</think>` is missing → tool-call parser receives empty content → tool call silently dropped. Same class of bug previously fixed for Kimi K2 (#33646).

3. Secondary: stale assistant turns with dangling `<think>` from earlier turns get re-wrapped by the chat template, nesting things further across multi-turn agent sessions.

This matches:
- **QwenLM/Qwen3.6 #150** "Qwen3.6-27B frequently stopped with empty tool call" (open, exactly the symptom).
- **vllm #35687** (MERGED 2026-04-24) "Treat `<tool_call>` as implicit reasoning end in Qwen3 parser" — fixes the streaming + non-streaming reasoning split so a stray `<tool_call>` ends reasoning even without `</think>`.
- Open upstream parser bugs **vllm #40785** (qwen3_coder) and **vllm #40787** (qwen3_xml) for related streaming/double-encoding issues.

## What we are currently evaluating

Two-layer fix applied in this repo, both shipped via `qwen36-27b-single-3090/compose/*.yml`:

### 1. vLLM nightly bump (parser side)

- Bumped pinned image from `9bba4628…` (2026-04-20) → `9b534fe6…` (2026-05-07, commit `51f22dcfd068fe8f1e3192da2a1e825b930223cf`).
- This crosses the merge of PR #35687, so the Qwen3 reasoning parser now treats `<tool_call>` as an implicit `</think>` and the silent-drop case in our logs should be fixed at the parser level.
- Applied to all 11 compose variants (default, text, longctx, tp2, tp2-mtp, tp4, tp4-2, bf16-tp4, 35b-a3b-awq{,-tp1,-mtp}). The 35b-a3b-gguf variant uses a different base image and was left alone.

### 2. Custom chat template (prompt side)

- Currently using **froggeric's** `qwen3.6/chat_template.jinja` from [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) on HF.
- Stored at `qwen36-27b-single-3090/patches/templates/qwen3.6-enhanced.jinja`.
- Mounted read-only into the container at `/patches/templates/qwen3.6-enhanced.jinja` and passed via `--chat-template` in every variant's command args.
- Features in this template:
  - `developer` role accepted (avoids crashes from OpenAI-style agent harnesses that send `role: "developer"`).
  - `<|think_on|>` / `<|think_off|>` inline toggles in system or user messages to control reasoning per-request.
  - `</thinking>` recognized in addition to `</think>`.
  - Non-ASCII characters in tool-call JSON args escaped as `\uXXXX`.
  - **Auto-close unclosed `<think>` before `<tool_call>`** — ported in by froggeric (`ex-arman68`) from allanchan339's template after the r/LocalLLaMA merge thread. This is the single piece that overlaps with the parser-side fix and gives defense-in-depth.
  - Hides historical reasoning by default to keep prompts compact and stable across long agent sessions.

### Alternative we considered

[fakezeta's merged gist](https://gist.github.com/fakezeta/9e8e039c60332fcb143c6e805558afe0), which combines froggeric + allanchan339. It adds a longer, stricter tool-call system prompt and parses string tool-call args as JSON. The JSON-parse piece is vLLM-specific (per froggeric's review in the thread it can break other runtimes) but fine for us. We chose froggeric's plain template instead because it is the cleaner, more portable baseline; can swap to fakezeta's if froggeric's prompt isn't strict enough about the XML format in practice.

### What we are NOT using

- **Sandermage Genesis plugin** (`patches/genesis/genesis_vllm_plugin/`) — only loaded by the experimental `longctx` mode that uses TurboQuant KV. The other variants use fp8_e5m2 KV which sidesteps the hybrid-gate bug Genesis works around. Many of Genesis's tool-call patches (P59, P60, P61) overlap with what is now in vLLM main, so re-enabling it on the new nightly would be redundant or actively conflicting.

## How to validate

After restarting (`./qwen.sh restart <mode>`):

1. Tail logs for the chat template line — vLLM logs `Using supplied chat template:` followed by the path; confirm it points at `/patches/templates/qwen3.6-enhanced.jinja`.
2. Run `scripts/verify-full.sh` if available to exercise tool-calls, streaming, thinking, needle-recall.
3. Run a real Claude/Pi agent session. The original failure signature was a turn ending with `stopReason: "stop"` while a `<tool_call>` block sat inside `thinking`. With the bumped parser the same model output should now produce `stopReason: "toolUse"` and a structured tool call in `content`.

## Fallback / rollback

- Old image still cached locally as `vllm/vllm-openai@sha256:9bba4628…`; revert by editing the `image:` line back.
- Template change is purely additive (a new file + two extra command args + one volume mount). Remove the `--chat-template` arg and the mount to fall back to the model-bundled template.

## Sources

- [QwenLM/Qwen3.6 issue #150 — empty tool call](https://github.com/QwenLM/Qwen3.6/issues/150)
- [vllm PR #35687 — Treat `<tool_call>` as implicit reasoning end](https://github.com/vllm-project/vllm/pull/35687)
- [allanchan339 — Qwen3.6 27B updated jinja](https://allanchan339.github.io/bug-fixes/2026/05/02/Qwen36-27B-updated-jinja.html)
- [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
- [r/LocalLLaMA — Qwen3.6 merged chat template from allanchan339 and froggeric](https://www.reddit.com/r/LocalLLaMA/comments/1t4cev0/qwen36_merged_chat_template_from_allanchan339_and/)
- [fakezeta merged gist](https://gist.github.com/fakezeta/9e8e039c60332fcb143c6e805558afe0)
- [vLLM Qwen3.5/3.6 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
