#!/usr/bin/env bash
# Usage: ./run.sh <label>     e.g.  ./run.sh before    or   ./run.sh after-p2p
#
# Captures a triple of measurements that together cover the question
# "is GPU<->GPU bandwidth healthy?":
#
#   1. p2pBandwidthLatencyTest  -> raw peer bandwidth + latency matrix.
#                                  Without P2P enabled, the "P2P=Enabled"
#                                  matrix is the same as "P2P=Disabled"
#                                  (driver simply can't enable it).
#   2. nccl-tests all_reduce_perf -tp 4   -> what vLLM TP=4 actually uses.
#   3. nccl-tests all_reduce_perf -tp 2   -> what vLLM TP=2 uses.
#
# Output goes under results/<label>/.

set -euo pipefail

LABEL="${1:-unlabeled}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/results/$LABEL"
mkdir -p "$OUT"

P2P_BIN="$ROOT/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"
ALLREDUCE="$ROOT/nccl-tests/build/all_reduce_perf"

echo "=== environment ===" | tee "$OUT/env.txt"
{
  date
  echo "label: $LABEL"
  echo "kernel cmdline:"
  cat /proc/cmdline
  echo "driver:"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1
  echo "nvidia-smi nvlink --status:"
  nvidia-smi nvlink --status 2>&1 | head -8
  echo "topo:"
  nvidia-smi topo -m 2>&1
} | tee -a "$OUT/env.txt"

echo
echo "=== [1/3] p2pBandwidthLatencyTest ==="
"$P2P_BIN" 2>&1 | tee "$OUT/p2p.txt"

# nccl-tests sweep: 4MB to 1GB in factors of 2, 5 iters, 50 warmup
COMMON_ARGS="-b 4M -e 1G -f 2 -n 5 -w 50 -c 1"

echo
echo "=== [2/3] all_reduce TP=4 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ALLREDUCE" $COMMON_ARGS -g 4 2>&1 | tee "$OUT/allreduce_tp4.txt"

echo
echo "=== [3/3] all_reduce TP=2 (GPUs 0,1) ==="
CUDA_VISIBLE_DEVICES=0,1 "$ALLREDUCE" $COMMON_ARGS -g 2 2>&1 | tee "$OUT/allreduce_tp2.txt"

echo
echo "Results saved to $OUT"
