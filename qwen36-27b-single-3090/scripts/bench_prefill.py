#!/usr/bin/env python3
"""
Prefill / TTFT benchmark — sibling to bench.sh, built for issue #3 (LMCache).

bench.sh measures *decode* TPS. This script measures *prefill* cost: how long
it takes to process a large pasted-in code context before the first token comes
out. That's the metric LMCache is supposed to move, by offloading KV to CPU RAM
so an evicted prefix can be re-loaded instead of recomputed.

Three conditions, all on the SAME large code prefix P (~PREFIX_TOKENS tokens):

  1. cold          first-ever submission of P            -> full prefill
  2. warm-hot      immediate re-submission of P          -> vLLM GPU prefix-cache hit
  3. warm-evicted  re-submit P after flushing the GPU    -> the LMCache differentiator:
                   KV pool with distinct filler             OFF = full recompute (~cold)
                                                            ON  = load KV from CPU RAM

For each we report TTFT (time to first streamed token), total wall time, and the
server-reported cached_tokens (prompt_tokens_details.cached_tokens) so cache hits
are verifiable, not inferred from timing alone.

Env:
  URL            default http://localhost:8020
  MODEL          default qwen3.6-27b-autoround
  PREFIX_TOKENS  approx size of the code prefix P (default 16000)
  EVICT_TOKENS   distinct filler tokens to push through to evict P
                 (must exceed the GPU KV pool; default 200000)
  GEN_TOKENS     tokens to generate per timed request (default 8; TTFT-focused)
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("URL", "http://localhost:8020").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen3.6-27b-autoround")
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "16000"))
EVICT_TOKENS = int(os.environ.get("EVICT_TOKENS", "200000"))
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "8"))

# ~1 line of plausible code. Real-ish tokens (identifiers, punctuation, numbers)
# so the tokenizer count tracks line count the way a genuine source file would.
CODE_LINE = (
    "    result_{i} = compute_partial(buffer[{i}], scale={i}.5, "
    "offset=0x{i:04x})  # step {i}: accumulate into ring slot {i}\n"
)


def make_code_context(n_tokens, seed=0):
    """Deterministic pseudo-code block ~n_tokens tokens. seed varies the prefix
    so filler blocks don't share a cache prefix with P or each other."""
    # ~11 tokens/line for CODE_LINE above; build a bit over target then it's fine.
    n_lines = max(1, n_tokens // 11)
    header = (
        f"# module_{seed}.py  (synthetic benchmark context, block seed={seed})\n"
        f"import numpy as np\n\n"
        f"def pipeline_{seed}(buffer):\n"
    )
    body = "".join(CODE_LINE.format(i=(seed * 100000 + j)) for j in range(n_lines))
    return header + body


def request(messages, max_tokens, label, quiet=False):
    """Streamed chat completion. Returns (ttft_s, total_s, prompt_tokens,
    cached_tokens, completion_tokens)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    ttft = None
    prompt_tokens = cached = completion = None
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            obj = json.loads(chunk)
            choices = obj.get("choices") or []
            if choices and ttft is None:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    ttft = time.monotonic() - start
            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                details = usage.get("prompt_tokens_details") or {}
                cached = details.get("cached_tokens")
    total = time.monotonic() - start
    if not quiet:
        ttft_s = f"{ttft:.3f}" if ttft is not None else "  ?  "
        cached_s = "?" if cached is None else str(cached)
        print(
            f"  {label:<16} prompt_tok={prompt_tokens:<7} cached={cached_s:<7} "
            f"TTFT={ttft_s}s  total={total:.3f}s"
        )
    return ttft, total, prompt_tokens, cached, completion


def timed(label, prefix_text, question):
    msg = [{"role": "user", "content": prefix_text + "\n\n" + question}]
    return request(msg, GEN_TOKENS, label)


def main():
    print(f"URL={URL}  MODEL={MODEL}")
    print(f"PREFIX_TOKENS≈{PREFIX_TOKENS}  EVICT_TOKENS≈{EVICT_TOKENS}  GEN_TOKENS={GEN_TOKENS}\n")

    prefix = make_code_context(PREFIX_TOKENS, seed=0)
    question = "Summarize what this module does in one sentence."

    # Small warmup so the very first timed call isn't paying one-off init costs.
    request([{"role": "user", "content": "hi"}], 4, "warmup", quiet=True)

    print("=== 1. COLD (first-ever submission of P) ===")
    timed("cold", prefix, question)

    print("\n=== 2. WARM-HOT (immediate re-submit, GPU prefix cache still hot) ===")
    timed("warm-hot", prefix, question)

    if EVICT_TOKENS <= 0:
        print("\n(EVICT_TOKENS=0 → skipping eviction phase)")
        return

    print(f"\n=== EVICT (push ~{EVICT_TOKENS} distinct tokens through the KV pool) ===")
    # ~50 tok per CODE_LINE in practice; keep each filler block under max-model-len.
    block = 8000
    n = max(1, EVICT_TOKENS // block)
    for k in range(n):
        filler = make_code_context(block, seed=1000 + k)
        request(
            [{"role": "user", "content": filler + "\n\nReply with OK."}],
            1,
            f"evict-{k}",
            quiet=True,
        )
        print(f"    evict {k+1}/{n} ({(k+1)*block} distinct tokens pushed)")

    print("\n=== 3. WARM-EVICTED (re-submit P after eviction) ===")
    print("    OFF: expect full recompute (~cold, cached≈0)")
    print("    ON : expect KV re-load from CPU (fast, cached≈prefix)")
    timed("warm-evicted", prefix, question)


if __name__ == "__main__":
    main()
