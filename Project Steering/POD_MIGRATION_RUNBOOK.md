# Pod migration runbook — moving off the faulty pod2

**Why now:** RunPod flags *"a critical error on this machine"* on pod2 (`pretty_white_lark-migration`,
`1qkb7dfkjvxg0h`, 69.30.85.123:22091) and schedules **maintenance 2026-08-06 21:00 → 2026-08-08 21:00
MESZ** with the server down. pod2 also has **`oom_kill 6`** on its 50 GB container cap.

⛔ **pod4 is NOT part of this migration.** `interesting_gray_ant` / `v9ni8rpan3qyn3`
(69.30.85.48:22192) is healthy, runs `flagship-v1arch-v2bal-30k` (step 9750, `g_op_fwd_ade_m`
**0.0898** — the best curve in the programme) and the PI is keeping it. **Do not touch it.**

---

## 0. The one number that decides the method

| route | throughput | 3.25 GB ckpt | 77.8 GB ckpts | 526 GB data |
|---|---|---|---|---|
| **pod → pod DIRECT ssh** | **42 MB/s** (MEASURED, C56, cross-datacenter) | **77 s** | **31 min** | **3.5 h** |
| HF push/pull | ~118 MB/s when quota allows | 28 s | 11 min | — |
| ⛔ via this dev box | **0.92 MB/s** (MEASURED 2026-08-03: 200 MB in 218 s) | **59 min** | **~24 h** | ~7 days |

⇒ **Migrate pod → pod directly. Never relay through the dev box.**

---

## 1. Create the new pod

- **A40 (or better), same region if possible** — `CA-MTL-1` is where pod2 lives.
- ⚠️ **Ask for MORE THAN 50 GB RAM.** pod2's container cap is exactly 50 GB and it has been
  OOM-killed **6 times**. `free` inside the pod shows the 503 GB **host** and is misleading — the
  real limit is `/sys/fs/cgroup/memory/memory.limit_in_bytes`.
- Volume ≥ **700 GB** at `/workspace` (pod2 currently holds 526 GB data + 85 GB experiments).
- Template `runpod-torch-v240` (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) matches
  what the current stack runs on.
- Enable **SSH over exposed TCP** (the "Direct TCP ports" entry). You need the direct
  `IP:PORT`, not the `ssh.runpod.io` proxy.

## 2. Wire the two pods together (5 minutes)

⛔ **Generate the key on the DESTINATION and copy only the PUBLIC half.** Never copy a private key —
that is correctly classifier-blocked and never necessary.

**On the NEW pod:**
```bash
ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519
cat /root/.ssh/id_ed25519.pub
```

**On pod2 (the OLD pod)** — paste that public line:
```bash
mkdir -p ~/.ssh && echo "<paste the ssh-ed25519 line>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

**Verify from the NEW pod** (use pod2's DIRECT mapping, not the proxy):
```bash
ssh -n -o StrictHostKeyChecking=no -p 22091 root@69.30.85.123 "hostname; df -h /workspace | tail -1"
```

⚠️ **Use the direct port.** The `ssh.runpod.io` proxy genuinely cannot move files — `sftp` fails with
`subsystem request failed on channel 0` and `scp -O` exits 2. It serves an interactive shell only.

## 3. Stop v5f cleanly first

v5f writes `ckpt.pt` periodically; the on-disk copy is **step 1000** while the process is past 1150,
so a hard kill loses the delta. Either wait for the next save, or accept it.

```bash
# On pod2 — kill by EXPLICIT PID. NEVER pkill -f train_flagship_v4:
# that pattern matches your own ssh command and kills your session,
# returning empty output so it looks like nothing happened.
pgrep -f 'scripts/train_flagship_v4.py'
kill <PID>
```

## 4. Copy, in priority order

Run these **from the NEW pod**. Ordered so that a interruption still leaves you the irreplaceable
things.

```bash
NEW=/workspace ; OLD="-p 22091 root@69.30.85.123"

# (a) IRREPLACEABLE — trained checkpoints, 77.8 GB / 38 files, ~31 min
mkdir -p $NEW/experiments
rsync -a --info=progress2 -e "ssh -p 22091" root@69.30.85.123:/workspace/experiments/ $NEW/experiments/

# (b) The repo, plus pod2's uncommitted work (319 files — already saved to the project folder
#     at _pod_backup/pod2-2026-08-03/pod2_uncommitted.diff, so this is belt-and-braces)
git clone <repo> $NEW/TanitAD    # or rsync /workspace/TanitAD
rsync -a -e "ssh -p 22091" root@69.30.85.123:/workspace/TanitAD/ $NEW/TanitAD/

# (c) REBUILDABLE but slow to regenerate — 526 GB, ~3.5 h. Copy rather than rebuild.
rsync -a --info=progress2 -e "ssh -p 22091" root@69.30.85.123:/workspace/data/ $NEW/data/
```

⚠️ **`rsync` is preferred over `scp` here** because it resumes. If you must use `scp`, chunk it and
**verify by SIZE and by LOADING**, never by exit code — silent `tar`-over-ssh truncation with exit 0
has bitten this programme three times in one day.

## 5. Verify before trusting (this is the step people skip)

```bash
# byte-exactness
ssh -n -p 22091 root@69.30.85.123 "md5sum /workspace/experiments/flagship-v5f-w120-30k/ckpt.pt"
md5sum /workspace/experiments/flagship-v5f-w120-30k/ckpt.pt

# and a REAL load — a file can be the right size and still be unreadable
PYTHONPATH=/workspace/TanitAD/stack python3 -c "
import torch,glob
for f in glob.glob('/workspace/experiments/*/ckpt.pt'):
    try:
        ck=torch.load(f,map_location='cpu',weights_only=False)
        print('OK  ',f,'step',ck.get('step'))
    except Exception as e:
        print('FAIL',f,type(e).__name__)"
```

## 6. Before relaunching training on the new pod

1. ⛔ **Sync the stack and prove it with a real `import`.** A pod checkout drifts silently and a
   launch from a stale one resurrects fixed bugs. `git log` on the pod is **not** proof:
   ```bash
   PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
     python3 -c "import train_flagship_v4; print('IMPORT OK')"
   ```
2. **Check the real memory cap**, not `free`:
   ```bash
   cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # or memory.max on cgroup v2
   ```
3. **Test disk with a real write** — `df` reports the 965 TB MooseFS cluster and hides the per-pod
   quota:
   ```bash
   dd if=/dev/zero of=/workspace/_dtest bs=1M count=500 && rm -f /workspace/_dtest
   ```
4. **Relaunch v5f DETACHED** (the last death was partly a session-tied process):
   ```bash
   cd /workspace/TanitAD/stack
   setsid nohup env PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=8 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   python3 -u scripts/train_flagship_v4.py \
     --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
     --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
     --v2-lru 6 --require-parity \
     --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe 176x624 \
     --from-scratch --batch 8 --accum 8 \
     --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
     --cond-imagination --no-heldout-gate \
     --heldout-off-reason 'PI directive: resume without the heldout gate' \
     --out /workspace/experiments/flagship-v5f-w120-30k \
     >> /workspace/v5f_run.log 2>&1 < /dev/null &
   ```
   It **auto-resumes from `ckpt.pt`** — `--from-scratch` only controls trunk *init* and is overridden
   by the checkpoint load. Confirm you see `[resume] step <N>`.

## 7. Only then release pod2

Release it **after** step 5 passes on every artifact you care about. ⛔ Deleting or releasing is the
PI's call, never an agent's.

---

## Already secured to the project folder (2026-08-03)

| artifact | location |
|---|---|
| pod2's 319-file uncommitted diff | `_pod_backup/pod2-2026-08-03/pod2_uncommitted.diff` (3.09 MB) |
| pod2's `git status` + untracked list | `_pod_backup/pod2-2026-08-03/pod2_status.txt` |
| v5f `ckpt.pt` step 1000 | `thor:~/models/v5f/ckpt.pt` — pulled and **load-verified** |
| flagship v1 `speedjerk` | `thor:~/models/flagship-v1-speedjerk/ckpt.pt` — load-verified, also on HF |
| REF-C base + xl | `thor:~/models/refc-*` — also on HF |

⇒ **Nothing irreplaceable is single-disk right now.** The 77.8 GB of older experiment checkpoints on
pod2 are the main un-backed-up asset, and several of them (`speedjerk`, `phase0`) are already on HF.
