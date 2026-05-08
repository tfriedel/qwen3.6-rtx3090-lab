#!/usr/bin/env python3
"""Long-prefill bench. Measures TTFT (= prefill time + 1 decode step)
for prompts of increasing length. Streams to capture TTFT precisely."""

import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("URL", "http://localhost:8020")
MODEL = os.environ.get("MODEL", "qwen3.6-27b-autoround")
LABEL = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"

# Build a prompt of approximately N tokens using a chunky filler
# (~4 chars/token average for English with Qwen tokenizer).
SEED = (
    "The history of computing is a long and winding road, full of both "
    "celebrated breakthroughs and forgotten dead ends. From the abacus to "
    "the analytical engine to the modern GPU, every advance has been built "
    "on the shoulders of those who came before. "
)


def make_prompt(target_tokens: int) -> str:
    chars = target_tokens * 4
    body = (SEED * (chars // len(SEED) + 1))[:chars]
    return (
        "Read the following passage carefully, then answer the question.\n\n"
        + body
        + "\n\nQuestion: in one sentence, what is the main subject of the passage?"
    )


def tokenize_count(text: str) -> int:
    req = urllib.request.Request(
        f"{URL}/tokenize",
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["count"]


def bench(prompt: str, max_tokens: int = 20):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    decoded = 0
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            delta = data["choices"][0]["delta"].get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                decoded += 1
    total = time.perf_counter() - t0
    return ttft, decoded, total


def main():
    # Warm up so weights are paged in and CUDA graphs are captured.
    print(f"=== label: {LABEL} ===")
    print("Warmup (small prompt)...")
    bench(make_prompt(200), max_tokens=10)

    sizes = [1024, 4096, 16384, 65536]
    print(f"{'tokens_in':>10}  {'ttft_s':>8}  {'prefill_tps':>12}  {'decoded':>8}  {'decode_tps':>10}")
    for target in sizes:
        prompt = make_prompt(target)
        actual = tokenize_count(prompt)
        ttft, decoded, total = bench(prompt, max_tokens=20)
        prefill_tps = actual / ttft if ttft else 0
        decode_time = total - ttft if ttft else total
        decode_tps = (decoded - 1) / decode_time if decoded > 1 and decode_time > 0 else 0
        print(f"{actual:>10}  {ttft:>8.3f}  {prefill_tps:>12.0f}  {decoded:>8}  {decode_tps:>10.2f}")


if __name__ == "__main__":
    main()
