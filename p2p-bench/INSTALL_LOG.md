# aikitoria/open-gpu-kernel-modules install log
Date: 2026-05-08 (evening)
Host: a4000-224n3 (Ubuntu 24.04.4, kernel 6.8.0-111-generic, AMD EPYC, 4× RTX 3090)
Goal: enable PCIe BAR1 P2P on consumer 3090s to speed up vLLM TP=2 / TP=4 all-reduce.

## Pre-state captured
- NVIDIA driver: 595.71.05 (matches the fork's NVIDIA_VERSION exactly).
- Stock GPU modules installed via DKMS at `/lib/modules/6.8.0-111-generic/updates/dkms/nvidia*.ko.zst`.
- Original grub cmdline: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"`.
- `/proc/cmdline` at install time: `BOOT_IMAGE=/boot/vmlinuz-6.8.0-111-generic root=UUID=4fb0972c-5a99-4411-9443-8268f7e9d546 ro quiet splash vt.handoff=7`
- IOMMU: AMD-Vi enabled in BIOS (ivhd0..3 in /sys/class/iommu/), but in default *translating* mode (no `iommu=pt`).
- IOMMU groups: each GPU in its own group → ACS likely on; not changed (no kernel patch available).
- All 4 GPUs PCIe-only (`NODE` in nvidia-smi topo); no NVLink bridges.
- gdm3 + nvidia-persistenced hold modules → cannot rmmod live.

## Baseline benchmark captured
Location: `/ssdpool/thomas/projects/qwen3.6/p2p-bench/results/before/`
Built tools: `nccl-tests` and CUDA samples `p2pBandwidthLatencyTest`.
Script: `/ssdpool/thomas/projects/qwen3.6/p2p-bench/run.sh <label>`.
Highlights:
- p2pBandwidthLatencyTest: P2P connectivity matrix all zeros off-diagonal (driver refuses peer access). Unidirectional bandwidth ~11.4 GB/s (host-RAM bounce). Bidirectional ~16.3 GB/s. Latency 13–15 µs.
- NCCL all_reduce TP=4: avg busbw ~6.4 GB/s.
- NCCL all_reduce TP=2 (GPUs 0,1): avg busbw ~6.9 GB/s.

## Changes applied (NO reboot yet at end of log)

### 1. Grub kernel cmdline
- Backup: `/etc/default/grub.bak.20260508-233659`
- Edit: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"` → `"quiet splash amd_iommu=on iommu=pt"`
- Regenerated `/boot/grub/grub.cfg` via `sudo update-grub`. Recovery menu entry was NOT modified.
- Available grub menu entries:
  - 6.8.0-111-generic (default, now with new cmdline + new modules)
  - 6.8.0-110-generic (stock DKMS modules untouched)
  - 5.15.0-105-generic
  - recovery mode of 6.8.0-111

### 2. open-gpu-kernel-modules (aikitoria fork)
- Cloned to `/ssdpool/thomas/builds/open-gpu-kernel-modules` (from `https://github.com/aikitoria/open-gpu-kernel-modules.git`, branch master, depth=1).
- `version.mk`: NVIDIA_VERSION = 595.71.05 (matches userspace driver).
- Built with: `cd /ssdpool/thomas/builds/open-gpu-kernel-modules && make modules -j$(nproc)`
- Produced .ko files: nvidia.ko, nvidia-uvm.ko, nvidia-modeset.ko, nvidia-drm.ko, nvidia-peermem.ko (in `kernel-open/`).
- Installed: `sudo make modules_install -j$(nproc)` →
  - new files: `/lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia*.ko`
  - modules signed automatically (SIGN step in install output).
- modinfo verifies `version: 595.71.05`, `license: Dual MIT/GPL`, `supported: external`.

### 3. Stock DKMS modules: backed up but kept
- Moved aside (sudo mv) inside the same dkms tree:
  - `/lib/modules/6.8.0-111-generic/updates/dkms/nvidia*.ko.zst` →
    `/lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/nvidia*.ko.zst`
  - Files still under `updates/`, so depmod alone would still find them; see next step.

### 4. depmod override
- Created `/etc/depmod.d/nvidia-aikitoria.conf`:
  ```
  override nvidia * kernel/drivers/video
  override nvidia-uvm * kernel/drivers/video
  override nvidia-modeset * kernel/drivers/video
  override nvidia-drm * kernel/drivers/video
  override nvidia-peermem * kernel/drivers/video
  ```
- Ran `sudo depmod -a`. After this, `modinfo nvidia` resolves to `/kernel/drivers/video/nvidia.ko` (the new one).

### 5. Initramfs
- Regenerated via `sudo update-initramfs -u -k 6.8.0-111-generic`.
- Verified before and after: initramfs does NOT contain the GPU driver itself (only `forcedeth`, `hid-nvidia-shield`, `typec_nvidia`). The change is cosmetic — included for safety.

## Reboot — DONE, system came back

### Post-reboot verification (all green)
- `cat /proc/cmdline` shows `... quiet splash amd_iommu=on iommu=pt vt.handoff=7` ✓
- All 4 GPUs visible on driver 595.71.05 ✓
- `modinfo nvidia` resolves to `/lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia.ko`,
  `license: Dual MIT/GPL` (open kernel module path) ✓
- Modules loaded cleanly, gdm3 + nvidia-persistenced restarted normally ✓

### Post-install benchmark (results saved at `p2p-bench/results/after/`)

| Test | Before fork | After fork (default NCCL) | After + `NCCL_P2P_LEVEL=PHB` |
|---|---|---|---|
| p2p connectivity matrix | all 0 | all 1 | — |
| Pair unidir bandwidth | 11.4 GB/s | 26.3 GB/s | — |
| Pair bidir bandwidth | 16.3 GB/s | 52 GB/s | — |
| Pair P2P latency | 14 µs | 1.0 µs | — |
| **NCCL all-reduce TP=2 busbw** | **6.9 GB/s** | **23.4 GB/s (3.4×)** | — |
| **NCCL all-reduce TP=4 busbw** | **6.4 GB/s** | **6.4 GB/s (no change!)** | **24.4 GB/s (3.8×)** |

### Key gotcha discovered: NCCL_P2P_LEVEL

For TP=4, NCCL by default falls back to **SHM transport** (host-RAM bounce) even
though P2P is now hardware-enabled, because the default `NCCL_P2P_LEVEL=PXB`
refuses P2P that crosses a PCIe Host Bridge. On EPYC each GPU sits on its own
root port → all 4-GPU traffic crosses PHB → NCCL picks SHM.

NCCL transport log line that revealed it:
```
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

After setting `NCCL_P2P_LEVEL=PHB` (or `SYS`):
```
Channel 00 : 0[0] -> 1[1] via P2P/direct/direct   (implicit; SHM messages gone)
```
Bus bandwidth jumps from 6.4 → 24.4 GB/s.

For TP=2 it doesn't matter — NCCL chose P2P even at default level.

### Compose files updated
All vLLM TP compose files patched in
`/ssdpool/thomas/projects/qwen3.6/qwen36-27b-single-3090/compose/`:
- `docker-compose.tp4.yml`
- `docker-compose.tp2.yml`
- `docker-compose.tp2-mtp.yml`
- `docker-compose.bf16-tp4.yml`
- `docker-compose.tp4-2.yml`

Change applied in each:
```diff
- - NCCL_P2P_DISABLE=1
+ - NCCL_P2P_DISABLE=0
+ - NCCL_P2P_LEVEL=PHB
```

Originals saved as `*.yml.bak.preP2P` next to each.

## End-to-end vLLM impact (TP=4, qwen3.6-27b-autoround int4 + MTP)

### Short-decode workload (max_num_seqs=1, ~hundreds of tokens prompt)
Identical between OLD and NEW:
- Narrative 1000-tok: 95.1 → 91.8 TPS (within noise)
- Code 800-tok: 119.7 → 120.6 TPS

Reason: per-token all-reduce moves only ~4 KB → latency-bound, not bandwidth-bound.

### Long-prefill workload (TTFT for various prompt sizes)
| Tokens in | OLD TTFT | NEW TTFT | Speedup |
|---|---|---|---|
| 837 | 0.46 s | 0.28 s | 1.67× |
| 3,266 | 2.63 s | 1.88 s | 1.40× |
| 12,978 | 5.97 s | 3.34 s | 1.79× |
| 51,823 | 23.14 s | 13.81 s | **1.68×** |

Per-chunk all-reduce at chunked-prefill granularity (4128 tok × 2K hidden × 2B fp16 ≈ 17 MB) is squarely in the bandwidth-bound regime → ~1.6× faster prefill, matches the NCCL busbw improvement.

### Take-aways
- Keep the fork: real win on long-prompt workloads.
- The 470K-token needle-recall test (README mentions ~475 s under OLD) should drop to ~280 s under NEW.
- For short chat / agent decode the win is invisible; not a regression either.
- Set `NCCL_P2P_LEVEL=PHB` in any TP env that crosses root ports (already in all compose files).

### Bench scripts and raw results
- NCCL micro: `/ssdpool/thomas/projects/qwen3.6/p2p-bench/run.sh` → `results/{before,after}/`
- Long-prefill: `/ssdpool/thomas/projects/qwen3.6/p2p-bench/prefill_bench.py` → `results/vllm/{old,new}-prefill.txt`
- Short-decode: `qwen36-27b-single-3090/scripts/bench.sh` → `results/vllm/{old,new}-tp4-bench.txt`

## Post-reboot verification plan
1. `cat /proc/cmdline` → must show `iommu=pt`.
2. `nvidia-smi` → must list all 4 GPUs with driver 595.71.05.
3. `lsmod | grep nvidia` and `modinfo nvidia | grep filename` → must point at `/kernel/drivers/video/`.
4. Run `/ssdpool/thomas/projects/qwen3.6/p2p-bench/run.sh after` and compare:
   - p2pBandwidthLatencyTest connectivity matrix should become all 1s.
   - P2P=Enabled bandwidth should jump above ~20 GB/s.
   - NCCL all_reduce busbw should roughly double (target: ≥ 12 GB/s).

## Recovery procedure if 6.8.0-111-generic fails to come up

### Option A: select a different grub entry at boot
- Pick "Advanced options for Ubuntu" → 6.8.0-110-generic OR 6.8.0-111-generic recovery mode OR 5.15.0-105-generic.
- Requires console access (IPMI / KVM-over-IP / serial / hands at box).

### Option B: undo from a recovery shell on 6.8.0-111
Once you have a shell (recovery menu or alternate kernel):
```bash
# 1. Remove the depmod override
sudo rm /etc/depmod.d/nvidia-aikitoria.conf

# 2. Restore the stock DKMS modules to their original location
sudo mv /lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/nvidia-drm.ko.zst       /lib/modules/6.8.0-111-generic/updates/dkms/
sudo mv /lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/nvidia.ko.zst           /lib/modules/6.8.0-111-generic/updates/dkms/
sudo mv /lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/nvidia-modeset.ko.zst   /lib/modules/6.8.0-111-generic/updates/dkms/
sudo mv /lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/nvidia-uvm.ko.zst       /lib/modules/6.8.0-111-generic/updates/dkms/
sudo mv /lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/nvidia-peermem.ko.zst   /lib/modules/6.8.0-111-generic/updates/dkms/

# 3. (Optional) remove the new modules
sudo rm /lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia.ko \
        /lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia-uvm.ko \
        /lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia-modeset.ko \
        /lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia-drm.ko \
        /lib/modules/6.8.0-111-generic/kernel/drivers/video/nvidia-peermem.ko

# 4. Restore stock grub cmdline
sudo cp /etc/default/grub.bak.20260508-233659 /etc/default/grub
sudo update-grub

# 5. Refresh module deps and reboot
sudo depmod -a
sudo update-initramfs -u -k 6.8.0-111-generic
sudo reboot
```

### Option C: nuke from orbit (last resort)
- `sudo apt install --reinstall nvidia-dkms-595` (or whatever package version is installed) reinstalls and DKMS-rebuilds the stock modules, overwriting our changes.
- Followed by Option B step 4 (grub revert) and reboot.

## File locations
- Build tree: `/ssdpool/thomas/builds/open-gpu-kernel-modules/`
- Benchmark tree: `/ssdpool/thomas/projects/qwen3.6/p2p-bench/`
- This log: `/home/thomas/Documents/p2p-driver-install-log.md`
- Grub backup: `/etc/default/grub.bak.20260508-233659`
- Stock module backup: `/lib/modules/6.8.0-111-generic/updates/dkms-stock-backup/`
