# P2P benchmark + driver-install record

Captures the before/after of installing the
[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)
fork, which enables PCIe BAR1 P2P on consumer 3090/4090/5090 cards. Originally
ran on this 4× RTX 3090 EPYC box where stock NVIDIA driver disables P2P.

## Layout

- `INSTALL_LOG.md` — what was changed on the host (grub, kernel modules, depmod
  override, initramfs), full recovery procedure, and final verified bench numbers.
- `run.sh` — one-shot NCCL + p2pBandwidthLatencyTest run; takes a label arg.
- `prefill_bench.py` — long-prefill TTFT bench against a running vLLM endpoint
  (streams completions, measures prefill_tps for 1K/4K/16K/64K input prompts).
- `results/before/`, `results/after/` — NCCL micro outputs.
- `results/vllm/old-tp4-bench.txt`, `new-tp4-bench.txt` — short-decode workload
  (no measurable change, latency-bound).
- `results/vllm/old-prefill.txt`, `new-prefill.txt` — long-prefill workload
  (~1.6× faster across all sizes — the actual workload-level win).

## Headline numbers (TP=4 on 4× RTX 3090, no NVLink)

| Metric | Before fork | After fork + `NCCL_P2P_LEVEL=PHB` |
|---|---|---|
| NCCL all-reduce busbw (TP=4, 1 GB) | 6.4 GB/s | 24.4 GB/s (3.8×) |
| Pair P2P latency | 14 µs | 1.0 µs |
| vLLM TTFT @ 52K-token prompt | 23.1 s | 13.8 s (1.68×) |
| vLLM single-request decode TPS | ~95 | ~92 (no change — latency-bound) |

## Required env in vLLM compose

```
- NCCL_P2P_DISABLE=0
- NCCL_P2P_LEVEL=PHB
```

`NCCL_P2P_LEVEL=PHB` is essential: NCCL's default `PXB` refuses P2P that
crosses a PCIe Host Bridge, which on EPYC is *every* GPU pair (each GPU sits
on its own root port). Without this, NCCL silently falls back to SHM and you
get zero benefit on TP≥3. Already set in every TP compose file in
`qwen36-27b-single-3090/compose/`.

## Re-running

```sh
# NCCL micro-bench (no vLLM needed)
./run.sh check-after-kernel-bump

# vLLM long-prefill (start TP=4 stack first)
cd ../qwen36-27b-single-3090/compose && docker compose -f docker-compose.tp4.yml up -d
# wait for "Application startup complete"
cd ../../p2p-bench && python3 prefill_bench.py my-label
```
