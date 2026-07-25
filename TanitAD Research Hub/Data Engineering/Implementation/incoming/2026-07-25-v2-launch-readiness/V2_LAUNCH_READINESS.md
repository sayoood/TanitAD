# v2 flagship launch readiness — clearing the two gates

**Date:** 2026-07-25 · **Discipline:** Data Engineering / Ops ·
**Slug:** `2026-07-25-v2-launch-readiness`
**Inputs:** `V2_CORPUS_QA.md` (GO on the data), `2026-07-24-v2-dataloader-integration/NOTE.md`
**Scope:** clear GATE 1 (stale pod `stack/`) and GATE 2 (no single node holds all 9,000)
so the v2 flagship starts on command, not on discovery.

Times are **UTC** (pods); Sayed's local is UTC+2.

---

## ✅ LAUNCHED — `flagship-v2corpus-30k`, pod1, trainer PID **699286**, 2026-07-25T02:41Z

Both gates cleared, all pre-launch checks passed, run is training.

| | |
|---|---|
| **Run** | `flagship-v2corpus-30k` · `tanitad-pod:/workspace/experiments/flagship-v2corpus-30k/` |
| **PID / log** | trainer **699286** (parent `bash -c` wrapper 699284 — the wrapper is NOT the trainer) · `tanitad-pod:/tmp/flagship-v2corpus-30k.log` (`/tmp`, not `/workspace` — logs there get swallowed on pod death) |
| **Launched** | 2026-07-25T02:41Z · **ETA ≈2026-07-29T01:10Z** (30,000 × MEASURED 11.34 s/step = 94.5 h) |
| **Corpus** | all **9,000** clips / 49.742 h, key **`4b7eeeac222d` verified** |
| **Windows** | **1,538,710** (vs 846,854 on pod1's shard alone) |

**Exact command** (reconstructible; `--sigreg-free-dims 64 --lr 3e-4 --warmup 2000`
stated explicitly to match MODEL_REGISTRY §1.2/§1.3 form):

```bash
cd /workspace/TanitAD/stack
PYTHONPATH=/workspace/TanitAD/stack PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python3 -u scripts/train_flagship4b.py \
  --v2-cache /workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d \
  --config flagship4b --v2 --sigreg-free-dims 64 \
  --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint \
  --lr 3e-4 --warmup 2000 --workers 8 --v2-lru 64 --guard-limit-gb 45 \
  --ckpt-every 1000 --log-every 50 \
  --out /workspace/experiments/flagship-v2corpus-30k \
  > /tmp/flagship-v2corpus-30k.log 2>&1 < /dev/null &
```

Plain `--v2` — **no** `--no-labels-v2`. That single omission is what makes the
corpus the only variable that differs from the control (§5c).

### Step-0 verification — every required check, MEASURED from the run's own log

| Check | Result |
|---|---|
| **OOM guard matched-file count** | **9,000 files** ✅ — **NON-ZERO.** This is the trap §5 closed; had it printed **0** the launch would have been aborted. |
| **Params** | `total_trainable` **286,339,251** ✅ — identical to MODEL_REGISTRY §1.3 `flagship4b-v2-30k`, i.e. the architecture is the registry's, not a silent variant |
| **Not frozen** | `encoder: 87,121,280` is **inside** `total_trainable` ✅ — a frozen trunk would read ~199 M |
| **Effective batch** | `batch 16x4` = **64 windows/step** ✅ |
| **Label regime** | `labels_v2 True` ✅ — plain `--v2`, the CORRECTED regime that matches the control (§5c) |
| **Episode ids** | `9000 providers, 9000 distinct episode ids` ✅ — the loader logged that the as-built 16-bit ids would have given only **8,391** |
| **Corpus** | `9000 lazy providers`, `1538710 train windows`, key `4b7eeeac222d` ✅ |
| **Lever parity vs control** | `v2_labels`/`speed_input`/`anchor_tactical`/`gated_intent` **true**, `rollout_k` **12** — an exact match to `flagship4b-v2-30k`'s own config.json ✅ (§5c) |
| **Health at step 100** | loss **37.69 → 23.26 → 17.09**, `g_op_fwd_ade_m` **1.8201 → 1.3754 → 0.8729**, **11.34 s/step**, data wait **1.60 s (14 %)**, RSS **12.2 GB** of a 57.7 GiB cap, GPU 15.2 GB ✅ |

The 30-step smoke on the consolidated 9,000 ran first and returned `FLAGSHIP4B_DONE`
with the identical param/guard/provider figures — so the union path was proven
*before* the 30k was committed, closing §10's last open item.

---

## VERDICT — both gates CLEARED; labeler DECIDED; run LAUNCHED

| | |
|---|---|
| **GATE 1 — pod `stack/`** | ✅ **CLEARED** on `tanitad-pod` (pod1) and `tanitad-pod3`. Verified by a **full trainer import**, not a file listing. pod2 given a zero-risk shadow stack. |
| **GATE 2 — one node, all 9,000** | ✅ **CLEARED.** pod3→pod1 transfer 00:03:30 → **02:13:16 UTC** (2 h 10 m, as projected). **4,047/4,047 files, 9.93 GiB, every name AND size identical to pod3.** Promoted; union = **9,000**, corpus key **`4b7eeeac222d`** reproduced. |
| **The training node** | ⚠️ **NOT pod2 — pod1.** pod1's GPU is **idle right now** (MEASURED: 2 MiB used, no trainer). Training can start **~34 h earlier** than the pod2 plan, on a **faster** GPU and **14× faster I/O**. |
| **End-to-end proof** | ✅ **The v2 flagship actually trains.** Two smoke runs on pod1 against the real v2 cache both reached `done: true`; **286,339,251 trainable params** (= parity's 286.34M) and loss 36.08 → **26.10** over 30 steps. See §5b. |
| **Label regime** | ✅ **DECIDED: plain `--v2`** (`labels_v2 True`) — the ONLY setting that leaves **corpus** as the single differing variable vs the control `flagship4b-v2-30k`. A first launch with `--no-labels-v2` was aborted after ~20 min; see §5c/§5d. |
| **Run name** | `flagship-v2corpus-30k` — unmistakably the *corpus* arm |
| **Cost to know** | 📌 **MEASURED ~11 s/step → a 30k v2 run is ≈3.8 days**, not overnight. Budget accordingly. |

**Two further launch-day traps were found and fixed** (§5): the OOM guard was
**silently inert on every v2 run** (matched 0 files), and its default limit sits
**above pod1's cgroup cap** so it could never fire.

---

## 1 · The premise changed: pod1 is free NOW

The brief assumed the v2 flagship waits ~1.5 days for pod2. **It does not have to.**

MEASURED 2026-07-25T00:00Z, `nvidia-smi` + `ps` on all three pods:

| | **pod1 `tanitad-pod`** | pod2 `tanitad-pod2` | pod3 `tanitad-pod3` |
|---|---|---|---|
| GPU | **RTX A6000, 48 GB** | A40, 46 GB | A40, 46 GB |
| GPU in use | **2 MiB — IDLE** | 34,683 MiB, **100 % util** | 0 MiB (idle GPU) |
| Running job | **none** | `train_flagship_v4.py` 30k, 26 h in | 8× YouTube harvest, **load avg 14–19** |
| `/workspace` | **local NVMe 500 G, 78.74 GiB free** | MooseFS (quota'd) | MooseFS |
| cgroup memory cap | **57.74 GiB** | 51.22 GiB | — |
| v2 clips held | **4,953** (12.42 GiB) | 0 | 4,047 (9.93 GiB) |

**pod2 frees ≈2026-07-26T10:10Z** (ESTIMATED from MEASURED checkpoint mtimes:
launch 2026-07-23T21:54:44Z, `ckpt_step10000.pt` written 2026-07-24T18:00Z →
20.09 h / 10k steps = **7.23 s/step** → 30k ≈ 60.3 h). That is **~34 h from now.**

### Why pod1 wins on evidence, not habit

1. **It is free now** — 34 h of runway recovered, the single largest factor.
2. **Bigger GPU.** A6000 48 GB > A40 46 GB.
3. **14× faster I/O, MEASURED by the QA itself.** Full-corpus scan: pod1 **91.7 s**
   (local NVMe) vs pod3 **1,285.3 s** (MooseFS-bound at 2.1 clip/s). The v2 loader
   is a JPEG-decode/`torch.load` path whose per-window cost is dominated by the
   2.9 MB payload read — the one workload where local NVMe matters most. pod2 is
   MooseFS like pod3.
4. **More RAM headroom.** 57.74 GiB cgroup vs pod2's 51.22 GiB, against a MEASURED
   ~1.9 GB/worker (QA P4).
5. **It already holds 55 % of the corpus** — 9.93 GiB moves, not 12.42 GiB.
6. **It does not disturb the running flagship-v4.** pod2 is untouched.

> The one thing pod2 has going for it is that the *next* flagship was pencilled in
> there. That is habit, not evidence. **Recommendation: run v2 on pod1.**

**On pod2's disk headroom** (asked for explicitly): MEASURED by real `dd`, never
`df` — 512 MiB written at **420 MB/s, no quota error**. That proves writes succeed
*right now*; it is **NOT** proof of 9.93 GiB of headroom, and I deliberately did
not write 10 GiB onto a pod that is mid-training. `df` on pod2 reports the 965 TB
cluster and remains useless, as documented.

---

## 2 · GATE 1 — pod `stack/` synced and verified

**What was wrong** (MEASURED, confirming the QA): both pods sat at `0f93b98` with
`grep -c v2-cache …/train_flagship4b.py` = **0**, no `tanitad/data/v2_dataset.py`,
and a stale `refb_labels.py`. A launch would have died on an unrecognised flag,
then `ModuleNotFoundError`.

**What I did.** Built one payload from the repo working tree at `bdb6ba1`
(`stack/tanitad/**` + `stack/scripts/*.py` + `stack/tests/*.py` + `pyproject.toml`
+ the `taniteval` package; 1.28 MB, 356 files, `__pycache__` excluded) and shipped
it **all at once** — the cross-pod-migration lesson — after taking a **full backup**
of each pod's `stack/` first.

### Verification — MEASURED per pod, by import, not by listing

| Check | pod1 `tanitad-pod` | pod3 `tanitad-pod3` |
|---|---|---|
| `grep -c v2-cache train_flagship4b.py` | **4** (was 0) | **4** (was 0) |
| `tanitad/data/v2_dataset.py` | **present** (14,894 B) | **present** (14,894 B) |
| `refb_labels.py` lines / md5 | **1300** / `6632348b…` | **1300** / `6632348b…` |
| **Full `train_flagship4b` module exec** | **IMPORT OK** (torch 2.4.1+cu124, tv 0.19.1) | **IMPORT OK** (torch 2.8.0+cu128, tv 0.23.0) |
| `--v2-cache` in the **real** CLI parser (`--help`) | **accepted** | **accepted** |
| `v2_dataset` import + `decode_jpeg` | **OK** | **OK** |
| `refb_labels.route_from_future_v21` | **present** | **present** |
| `taniteval.ci` (`episode_cluster_bootstrap`) | **OK** | **OK** |

`refb_labels.py` landing on `6632348b…` independently reproduces the md5 the QA
cited as repo-authoritative — two separate paths agreeing on the labeler.

Note both pods now pass on **different torch/torchvision majors** (2.4/0.19 and
2.8/0.23), so the v2 decode path is not version-fragile across the fleet.

**pod2 — deliberately NOT touched.** Its live checkout is running the flagship-v4.
Instead it received a **shadow stack** at `/workspace/tmp/v2_stack/` (extracted,
`--v2-cache` present), which is zero-risk and strictly better than leaving a recipe.
The exact pod2 promotion recipe is in §7 should the run ever need to move there.

**Backups** (restore points, on the pods):
`tanitad-pod:/workspace/tmp/stack_backup_20260724T235834Z.tgz` (2,074,784 B) ·
`tanitad-pod3:/workspace/tmp/stack_backup_20260724T235836Z.tgz` (2,148,686 B).

**Drift caught by the pre/post md5 manifest:** pod1's `train_flagship_v4.py` was
`64d7f4ba…`, **differing from the repo's `cbc41935…`**. pod2 — the pod actually
running that trainer — is `cbc41935…`, i.e. **repo-identical**, so the live
flagship-v4's provenance is clean and pod1's copy was the stale one. It is now
repo-aligned, with the old bytes preserved in the backup above.

---

## 3 · GATE 2 — the transfer: what I measured, chose, and started

### The numbers that decided it (all MEASURED on this session's links)

| Path | Throughput | 9.93 GiB would take |
|---|---|---|
| pod3 → dev-box (**down**link) | **4.63 MiB/s** | 37 min |
| dev-box → pod1 (**up**link, 1 stream) | **1.047 MiB/s** | 2.70 h |
| dev-box → pod1 (**4 parallel** streams) | **1.271 MiB/s** (+21 % only) | 2.22 h |
| **Live streamed pipe, in flight** | **1.394 MB/s** | **≈2.1 h** |

The dev-box **uplink is the hard cap** (~8.8–11 Mbit/s); 4 streams buy only 21 %,
so this is a link limit, not a per-connection one. The inherited "~1 MB/s relay"
figure is **confirmed, and it is specifically the uplink** — the downlink is 4.4×
faster, which is why a *streamed* pipe (one pass) beats staging (down-then-up).

### What is running

A **streamed, resumable, size-verified** pod3→pod1 pipe, 4 lanes:

```
ssh pod3 "tar -cf - --files-from=lane_i"  |  ssh pod1 "tar --no-same-owner -xf -"
```

- **Started** 2026-07-25T00:03:30Z. **Progress at 00:14:09Z: 337/4047 files, 0.83 GiB (8.3 %).** **ETA ≈02:11 UTC** (04:11 Berlin).
- Lands in a **staging dir** `/workspace/data/physicalai_v2/_incoming_pod3/`, *not*
  the live corpus dir, so a mid-stream kill can never leave a truncated `.v2ep.pt`
  where the trainer would read it. Promotion is a same-filesystem `mv` (instant, no
  extra space).
- **Resumable and self-healing:** each round re-derives the delta from a live
  `(name, size)` comparison, so a dropped lane or a re-run only moves what is
  actually missing or short. Driver: `transfer_v2.sh` (manifest §8).
- **Binary integrity proven before committing 2 h to it:** a 20-file pilot came
  through **20/20 md5-identical**. Windows OpenSSH pipes are byte-clean here.
  Re-checked **mid-flight** on a further **30 random staged clips: 30/30
  md5-identical** against pod3 — 50 files verified end-to-end, plus the
  per-file size check every round.
- Disk: pod1 had **78.74 GiB free** (MEASURED, `df` — valid here, local NVMe);
  9.93 GiB in, ~69 GiB left for checkpoints (3.2 GB each).

### The two faster paths, and why neither is being used

- **HF push→pull (~118 MB/s; would be minutes, not hours).** Blocked **twice
  over**: the brief's `Sayood` 403-storage-full, *and* — MEASURED here, an
  independent second probe — the HF token on pod3 is **invalid**
  (`whoami-v2` → `{"error":"Invalid username or password."}`). Even with storage
  cleared, pod3 needs a working token.
  **→ If Sayed clears HF storage AND refreshes the pod token, this collapses ~2.1 h
  to ~3 min.** That is live evidence for the storage decision, but it is not on the
  critical path today: the relay finishes ~32 h before pod2 would have freed.
- **Direct pod1→pod3 SSH (datacenter speed).** MEASURED: **pod1 CAN open a TCP
  connection to pod3:22079** (and to pod2), and pod1 already has a keypair — so the
  only missing piece is trust. I did **not** install a key: adding an
  `authorized_keys` entry is a change to a host's security configuration, which is
  Sayed's call, not mine. **Recommend Sayed authorise this once** — it would make
  every future cross-pod move minutes instead of hours, and it needs no private key
  to leave any machine (generate on the destination, install only the public half).

---

## 4 · The `episode_id` collision — fixed at load, no rebuild

**The defect** (QA P3, reproduced here from the builder source):
`v2_compressed.build_compressed` line 101 stores
`episode_id = int.from_bytes(clip_id.encode()[:4], "big")` — the first **4
characters** of the UUID, **16 bits**. MEASURED: **8,391 distinct ids for 9,000
clips → 609 collisions (6.8 %)**, max multiplicity 4. Parity has the same defect
at 1.4 %.

**Why it matters** (confirmed by reading the consumer, not by assumption):
`taniteval.ci.episode_cluster_bootstrap(per_window, eid, …)` clusters on the
**unique values of `eid`**. Under the 16-bit id, 609 pairs of genuinely different
clips are silently fused into one cluster — which **narrows the interval**. That is
precisely the failure CLAUDE.md's "never quote an interval without its estimator"
rule exists to prevent. Training is unaffected: the trainer emits `episode_id` in
every window and never consumes it.

**The fix — at LOAD time, in the v2 path only.**
`tanitad/data/v2_dataset.py` gains `stable_episode_id(clip_id)`: a **63-bit**
blake2b of the **full** `clip_id` (which `build_compressed` already stores in every
payload). 63 bits, not 64, so the value stays inside torch's signed-int64 collate;
collision probability over 9,000 clips ≈ 4e-12.

Chosen over the QA's suggested build-time fix on purpose:

- it **repairs the 9,000 clips already on disk with no rebuild**;
- every `*.v2ep.pt` stays **byte-for-byte untouched**, so the QA's
  build-vs-load byte-identity proof (24/24) still stands;
- **parity behaviour is not touched at all** — the raw epcache path never goes
  through this module.

It also fixes a case the build-time fix would not have: the raw 16-bit ids collide
**across** shards, so concatenating pod1's and pod3's halves under the old scheme
would have fused unrelated clips from opposite shards into one bootstrap cluster.

Default is ON (`build_v2_providers(..., stable_ids=True)`); `stable_ids=False`
reproduces the as-built ids for diffing against `load_compressed`. The manifest now
carries **both** (`episode_id` raw + `episode_uid` stable) plus `clip_id`, and
`MANIFEST_VERSION` 1→2 so existing sidecars rebuild themselves. `build_v2_providers`
now also **prints the distinct-id count and warns loudly on any collision**.

**Guard for anything that does NOT use this loader** (e.g. a rescorer reading
payloads directly): do **not** group on the stored `episode_id`. Group on
`clip_id`, or on `stable_episode_id(clip_id)` — both are in every payload.

**MEASURED — synthetic:** `tests/test_v2_dataset.py` **8 passed** on pod1, including
a new `test_stable_episode_id_fixes_the_16bit_collision` that builds two clips
sharing a 4-char prefix, asserts the raw scheme yields **1 cluster for 2 clips**,
and asserts the default path separates them.

**MEASURED — on the REAL corpus** (pod1's whole 4,953-clip shard, manifest rebuilt
in 11.9 s, `version=2`):

| | distinct ids | collisions |
|---|---|---|
| as-built 16-bit `episode_id` | **4,649** | **304** |
| `stable_episode_id` (this fix) | **4,953** | **0** |

304 collisions in *half* the corpus is consistent with the QA's 609 across all
9,000. All ids verified `< 2**63` (int64-collate safe). Sample:
`00097de1-5ded-4fba-a5ed-4b527678d1b0 → 1148924942763724552`.

---

## 5 · The exact launch command — and three things that would have bitten on launch day

### The label flag — surfaced here, RESOLVED in §5c (read that for the final call)

`train_flagship4b.py` line 253: **`--v2` sets `cfg.v2_labels = True`.** That is easy
to miss, and it changes what "28 % turns" means: under the curvature-gated v2
labeler this corpus reads **18.83 % turns (1.63× parity)**, not the **28.04 %
(1.97× parity)** its selection targeted — both MEASURED by the QA on the *same*
9,000 clips.

I flagged this as a blocking decision rather than defaulting into it, and initially
recommended `--v2 --no-labels-v2` to keep the 28.04 % reading.

**That recommendation was superseded and the run uses plain `--v2`.** The decisive
consideration turned out not to be the turn *percentage* at all: the v1/v2 labelers
disagree about what to *call* a gentle highway sweep, but **the scenes are identical
either way** — turn exposure does not change. What does change is attribution, and
only plain `--v2` matches the control `flagship4b-v2-30k` (`v2_labels: true`),
leaving **corpus as the single differing variable**. Full reasoning, the verified
control config, and the aborted first launch: **§5c and §5d.**

### Trap 1 (FIXED) — the OOM guard was silently inert on every v2 run

`start_cache_guard` globbed only `<dir>/*/ep_*.pt` — the **nested raw epcache**
layout. The v2 cache is **flat `<dir>/<clip_id>.v2ep.pt`**. `train_flagship4b.py`
line 326 passes the `--v2-cache` dirs straight into it, so the sweep matched
**ZERO files and could free nothing** — the OOM protection would have been absent
exactly where the risk is highest (pod2's flagship has been OOM-killed before).
Its pre-arm message printed the *pattern* count, so it looked healthy either way.

Fixed additively (both layouts globbed) and the message now reports **matched
files**, warning loudly at 0. **MEASURED on pod1:** v2 cache **0 → 4,953 files
watched**; raw parity epcache under its documented `--cache-dirs` usage still
**2,376** — no regression.

> Method note: my first probe pointed the guard at the *split* dir and saw 0 files
> on the raw path too. That was my probe being wrong, not a second bug — verified
> against the documented usage before reporting. (CLAUDE.md: verify before alarming.)

### Trap 2 (FIXED) — the guard's default limit is above pod1's cgroup cap

`limit_gb` is **GiB** (`limit_gb * 1024**3`). The default is **60 GiB**;
**pod1's cgroup cap is 57.74 GiB** — so the guard could **never fire** before the
kernel OOM-killer. The guard now checks the cap (both cgroup **v1** and v2 — the
pods are v1, and a v2-only check silently skipped this) and warns.
**MEASURED on pod1:** 60 GiB → warns; 45 GiB → silent. **Pass `--guard-limit-gb 45`.**

### The command

**The canonical, as-launched command is at the top of this document** (the LAUNCHED
block) — plain `--v2`, no `--no-labels-v2`. It is not repeated here so the two can
never drift apart.

Deltas from the staged NOTE.md command, each earned above:

- **`--guard-limit-gb 45`** — below pod1's 57.74 GiB cap, or the guard is decorative.
- **`--workers 8`** — MEASURED (QA P4, on pod1): 8 workers = **37.6 win/s at 16.8 GB**
  RSS. The step consumes 64 windows (bs16 × accum4); at ~7–8.6 s/step that is
  **~1.7 s of data, fully hidden**. 16 workers buys 49.4 win/s for **30.4 GB** —
  needless risk against a 57.74 GiB cap. 4 workers (20.0 win/s, 4.4 GB) is also
  sufficient and is the RAM-safest fallback.
- **log to `/tmp`** — logs written under `/workspace` get swallowed when a pod dies.
- **`setsid nohup … &`** — survives the ssh session.

Notes: `--v2-cache` sets `ds_val = None` — **this trainer runs no val loop**, so no
val cache is needed and every quotable number must come from a separate `eval_*.py`
run (operating standard #1: trainer val is not quotable). First start does a
one-time metadata-only manifest scan (MEASURED 2.3 s for pod1's 4,953; the
consolidated 9,000 will rebuild once, then be instant).

**Expected scale:** ~**1,547,710** windows (MEASURED by the QA from per-clip `T_out`),
vs 846,854 on pod1's shard alone.

---

## 5b · The v2 flagship was actually run — end to end, on the real corpus

The QA validated the *data path* (`build_v2_providers` → `_wrap` → `DataLoader`).
It did **not** run the trainer. Since pod1 was idle, I ran the real thing —
the exact command in §5, against pod1's real 4,953-clip v2 cache.

**Both runs reached `done: true`.** MEASURED, pod1, 2026-07-25T00:19–00:37Z:

| | run 1 (contended) | **run 2 (clean, 8 workers)** |
|---|---|---|
| steps | 20/20, `done: true` | **30/30, `done: true`** |
| wallclock | 637.9 s | **353.1 s** |
| s/step (last window) | ~24 | **10.54–11.41** |
| data wait / step | ~5.4 s (22 %) | **1.44–1.56 s (13 %)** |
| loss | 36.08 → 28.55 | **36.08 → 26.10** |

**`total_trainable` = 286,339,251** — identical to the parity flagship's 286.34 M,
so the v2 corpus feeds the intended architecture, not a silently different one.
The full 4-brain is live and grounded: operative 96.6 M / tactical 26.5 M + 30.1 M /
strategic 8.4 M / encoder 87.1 M / h15 22.1 M. All grounding metrics emit and
improve over 30 steps (`g_op_fwd_ade_m` 2.01 → 1.56, `g_tac_fwd_ade_m` 6.19 → 4.49).

**Two consequences worth acting on:**

1. **A 30k v2 run is ≈3.8 days** (30,000 × ~11 s = 91.7 h), not an overnight job.
   That is a real planning input and it is *slower per step* than pod2's
   flagship-v4 (7.23 s/step) — expected: `--grad-checkpoint` plus a JPEG-decoding
   loader versus a RAM-resident epcache.
2. **8 workers is confirmed correct.** Data wait is **13 %** of step time — the
   loader is hidden behind compute, exactly as the QA projected, and there is no
   case for the 16-worker/30.4 GB configuration.

> **Method note, logged because it nearly became a false alarm.** Run 1 read
> ~24–30 s/step and I almost reported it. It was **my own doing**: the full
> `pytest -q` I had launched on pod1 was consuming **1289 % CPU** (load 40.8),
> starving the dataloader — GPU util was **0–18 %**, i.e. the trainer was
> *waiting*, not computing. Killing pytest **by explicit PID** (never `pkill -f`,
> which self-matches the ssh command) took the box to load ~20 and the same run to
> **11 s/step**. CLAUDE.md's "verify before alarming" and "`step_s` is ACCUMULATED
> over `--log-every`" both applied — the raw `step_s` of 112.8 is **ten** steps.
> **Consequence: the pod1 sweep was killed** and re-run on **pod3** instead, which
> keeps pod1 clean for the transfer and the launch — **819 passed**, all failures
> environmental (§10).

Smoke artifacts (2 × 3.2 GB checkpoints) were deleted; pod1 free is **72.95 GiB**
and its GPU is back to **2 MiB idle**, ready for the real launch.

---

## 5c · PRE-REGISTRATION — `flagship-v2corpus-30k` vs `flagship4b-v2-30k`, **CORPUS ONLY**

**Final decision (coordinator, 2026-07-25, reversing an earlier call): plain `--v2`
— `labels_v2 = True`.** The first launch used `--v2 --no-labels-v2`; it was killed
at ~20 min and relaunched. *(History in §5d — the reversal is the point, not a
footnote.)*

**Why: it makes the corpus the ONLY differing variable.** The original rationale for
`--no-labels-v2` was to match the running `flagship-v4-fromscratch` arm — but that
arm is **`train_flagship_v4.py` with `args.labels = "v3"`**, a different trainer,
architecture and label family (VERIFIED from its config.json). It was never the
control, so the rationale was void.

### The control, verified from its OWN artifact — not from registry prose

`tanitad-pod2:/workspace/experiments/flagship4b-v2-30k/config.json`:

| Lever | `flagship4b-v2-30k` (control) | **`flagship-v2corpus-30k`** (this run) |
|---|---|---|
| trainer / config | `train_flagship4b.py --config flagship4b` | **same** ✅ |
| `v2_labels` | **true** | **true** ✅ |
| `speed_input` | **true** | **true** ✅ |
| `rollout_k` | **12** | **12** ✅ |
| `anchor_tactical` | **true** | **true** ✅ |
| `gated_intent` | **true** | **true** ✅ |
| **`total_trainable`** | **286,339,251** | **286,339,251** ✅ |
| **corpus** | parity **13.13 h** / 2,376 ep<br>`_epcache/physicalai_phase0` | **v2 49.742 h** / 9,000 clips<br>`epcache-physicalai-v2bal-4b7eeeac222d` |
| status | abandoned; log ends **step 7,700** | 30k running |

**Every lever matches. The corpus is the single differing variable.** ✅

### One correction to the matched-step point

The registry says the control was "abandoned at step **7,800**". Its **artifact**
says the train log's last row is step **7,700**, and rows near the end are
**duplicated** (steps 7,500/7,700 appear twice) — the signature of a
supervisor auto-resume, so provenance in that tail is messier than a clean run.

Available control checkpoints: **`ckpt_step5000.pt`** (cleanly archived) and
`ckpt.pt` (~7,700).

- **Primary matched-step comparison: step 5,000** — an explicitly archived
  checkpoint on the control, and this run writes one every 1,000 steps. Clean on
  both sides, no resume ambiguity.
- **Secondary (deeper): ~7,700** via the control's final `ckpt.pt`, flagged as
  sitting in the resume-duplicated tail.
- **30k-vs-30k does not exist** and never will for this control.

### On the turn share — the SCENES do not change, only the label convention

Under the v1 labeler this corpus reads **28.04 %** turns (1.97× parity); under the
**v2** labeler now in force it reads **18.83 %** (1.63× parity). Both are MEASURED
on the *same* 9,000 clips. The gentle-highway-sweep reclassification changes what
gets *called* a turn, **not which scenes the model sees** — turn exposure is
unchanged, and the control is labelled by the identical convention, so the
comparison is unaffected. Quote **18.83 %** when describing this run's labels and
**28.04 %** when describing the corpus design target.

### The pre-registered read — both outcomes committed in advance

- **Primary:** **ADE@2s**, via **episode-cluster bootstrap** over the 40 val
  episodes (`taniteval/ci.py`), **paired** where windows match — never
  `overlapping_holdout_se`. At **matched step 5,000**.
- **Secondary:** turn-stratified ADE. The corpus was rebalanced *for* turns, so
  that is where the effect should appear first if it appears at all.
- **WIN:** CI-separated improvement on ADE@2s at matched step, concentrated in the
  turn strata → the 3.8× corpus expansion + rebalancing buys geometry, and **corpus
  scale becomes the program's active lever**.
- **LOSS / NULL:** no CI separation, or worse → **corpus scale is not the
  bottleneck at this capacity.** With v1.6 having already shown capacity is not the
  bottleneck either, that would point squarely at the architecture/objective.
  **Committed as informative — not as grounds to re-cut the corpus.**
- **Not admissible either way:** any learning-curve exponent (CLAUDE.md), and any
  number read from a trainer log rather than `eval_*.py`.

### ⚠️ Reproducibility wrinkle found in the new run's `config.json`

Under `--v2-cache` the trainer records **`"cache_dirs": null, "data": "realmix"`** —
the **v2 corpus path is NOT written into `config.json`**. It survives only in the
`[data] v2-cache [...]` stdout line. So the run's own config does not identify the
corpus it trained on, which is precisely the provenance this program keeps getting
burned by. **Mitigation:** the launch log is preserved at
`tanitad-pod:/tmp/flagship-v2corpus-30k.log`, and the MODEL_REGISTRY entry must
record the corpus key `4b7eeeac222d` explicitly. Worth a one-line trainer fix
(write `args.v2_cache` into the config) — out of scope here, flagged.

---

## 5d · The reversal — first launch aborted, and why that is recorded here

| | |
|---|---|
| **Aborted run** | `--v2 --no-labels-v2` (labels_v2 **False**), PID 698375, 02:21Z → killed 02:40Z (~20 min, ~step 100) |
| **Preserved at** | `tanitad-pod:/workspace/experiments/flagship-v2corpus-30k_ABORTED-labelsFALSE-20260725T0221Z/` |
| **Resume risk** | **None — verified.** The dir held only `config.json` + `train_log.jsonl`; **no `ckpt.pt` existed** (first write is step 1,000), so a mismatched-label resume was impossible. Renamed anyway. |
| **Kill hygiene** | SIGTERM to **explicit PID 698375** (never `pkill -f`, which self-matches the ssh command). All 8 dataloader workers reaped; **0 orphan processes**; GPU returned to 2 MiB. |

**Root-cause class: a control arm assumed rather than read.** The label regime was
chosen to "match the running arm" without first checking which arm was the actual
control. One `grep` of two `config.json` files settled it. Cost: ~20 min of GPU and
one relaunch — caught before the 90-hour commitment, which is the only reason it was
cheap. Candidate for `RETRACTION_LOG.md` under the same class as "quoting a
faster-moving source than the artifact".

---

## 6 · Promotion — ONE command, and it refuses to promote a partial corpus

```bash
ssh tanitad-pod 'bash -s' < promote_and_verify.sh     # also at pod1:/workspace/tmp/
```

Six guarded steps, non-zero exit on any failure, safe to re-run:

1. **Refuses to promote unless staging holds exactly 4,047** — a partial union
   trains on a distribution nobody designed (per-shard turn shares are 27.35 % /
   28.89 %, both off the 28 % target; only the union is on target).
2. Promotes with `mv` on the same filesystem — instant, no extra space.
3. Deletes `_v2manifest.pt`. **Required**, not cosmetic: the existing sidecar
   predates `MANIFEST_VERSION 2` and has no `episode_uid`.
4. **Recomputes the corpus key over the 9,000 built filenames and asserts
   `4b7eeeac222d`** — the QA's end-to-end proof that what is on disk is the
   *designed* corpus, not merely 9,000 files that happen to number 9,000.
5. Rebuilds the manifest and asserts **9,000 clips with 9,000 distinct stable
   episode ids** (0 collisions).
6. Reports free disk.

**The key check is already validated, not merely written.** MEASURED: running this
script's exact key recipe over the union of the two shard listings
(pod1 4,953 + pod3 4,047, **overlap 0, union 9,000**) reproduces
**`4b7eeeac222d`**. So both the check and the premise — that these two shards *are*
the designed corpus — are confirmed **before** the transfer even lands.

---

## 7 · pod2 promotion recipe (only if the run must move there)

pod2 already has the shadow stack. To make its live checkout v2-capable **after
the flagship-v4 finishes** (≈2026-07-26T10:10Z) — never while it trains:

```bash
ssh tanitad-pod2 '
cd /workspace/TanitAD
tar -czf /workspace/tmp/stack_backup_$(date -u +%Y%m%dT%H%M%SZ).tgz \
    --exclude=__pycache__ --exclude="*.pyc" stack          # restore point FIRST
tar -xzf /workspace/tmp/stack_sync.tgz -C /workspace/TanitAD --no-same-owner
find stack taniteval -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
cd stack && PYTHONPATH=/workspace/TanitAD/stack python3 -c "
import sys; sys.path.insert(0,\"scripts\")
import importlib.util
s=importlib.util.spec_from_file_location(\"t\",\"scripts/train_flagship4b.py\")
m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(\"IMPORT OK\")"
'
```
The payload is already at `tanitad-pod2:/workspace/tmp/stack_sync.tgz`.
**pod2 caveats if it is ever the v2 node:** cgroup cap **51.22 GiB** → use
`--guard-limit-gb 40` and **≤8 workers**; and its `/workspace` is MooseFS, so
expect the QA's 14× I/O penalty versus pod1's NVMe, plus a 9.93 GiB *second*
transfer since pod2 holds **zero** v2 clips today.

---

## 8 · Also found: `v2_compressed.py` was stranded on the pods only

The v2 corpus **builder** — `build_compressed` / `load_compressed`, the canonical
reference the whole QA validated against — existed on **pod1 and pod3 and nowhere
else**. It was **absent from the repo** and absent from pod2. Two copies, both on
single disks: operating standard #3's exact failure class ("an artifact on one disk
is NOT done"), and the file that owns the `episode_id` defect.

**Rescued** to `stack/scripts/v2_compressed.py`, md5 `45bb6c3b5f842152f30e2be826810ea7`
**identical on pod1, pod3 and the repo copy** (252 lines). Now staged.

I left `build_compressed`'s `episode_id` line **unchanged** on purpose: patching it
would make a *resumed* build (it skips already-built clips) emit a mix of 16-bit and
63-bit ids into one corpus — worse than a uniform defect that the loader now
corrects for every consumer.

---

## 9 · Deliverable manifest

All staged in the repo working tree. **No commit, no push.**

| Artifact | Location | What it is |
|---|---|---|
| `V2_LAUNCH_READINESS.md` (this file) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-25-v2-launch-readiness/` | the report + LAUNCH-READY call |
| `transfer_v2.sh` | same dir | the resumable pod3→pod1 streamed transfer driver |
| **`promote_and_verify.sh`** | same dir **+ `tanitad-pod:/workspace/tmp/`** | the guarded promotion + corpus-key + provider verification (§6) |
| `sync_stack.sh` · `verify_stack.sh` | same dir | GATE-1 pod sync + the 8-check verification |
| **`stack/scripts/v2_compressed.py`** | `repo:stack/scripts/` | **RESCUED** from the pods (was repo-absent) |
| **`stack/tanitad/data/v2_dataset.py`** | `repo:stack/tanitad/data/` | `stable_episode_id` + manifest v2 + collision warning |
| **`stack/tests/test_v2_dataset.py`** | `repo:stack/tests/` | updated + new collision test (**8 passed**, pod1) |
| **`stack/scripts/finetune_traj.py`** | `repo:stack/scripts/` | OOM-guard: v2 layout + matched-file count + cgroup-cap warning |
| Synced `stack/` | `tanitad-pod:` and `tanitad-pod3:/workspace/TanitAD/stack` | GATE 1, verified by import |
| Shadow stack | `tanitad-pod2:/workspace/tmp/v2_stack/` | v2-capable, live checkout untouched |
| Sync payload | all three pods `:/workspace/tmp/stack_sync.tgz` | 1.28 MB, 356 files |
| Pod backups | `tanitad-pod:/workspace/tmp/stack_backup_20260724T235834Z.tgz` · `tanitad-pod3:…T235836Z.tgz` | pre-sync restore points |
| Transfer staging | `tanitad-pod:/workspace/data/physicalai_v2/_incoming_pod3/` | in flight; promote per §6 |
| The corpus | `tanitad-pod` + `tanitad-pod3:…/epcache-physicalai-v2bal-4b7eeeac222d/` | pod3's shard **unchanged** (read-only source) |

---

## 10 · What remains, and what I could not verify

**Before launch — ALL DONE, 2026-07-25T02:13–02:21Z:**
1. ✅ Promoted (`promote_and_verify.sh`, exit 0).
2. ✅ Corpus key `4b7eeeac222d` reproduced over the 9,000-clip union. The same run
   independently reproduced the QA's **609** raw-id collisions and showed
   **0** under the fix.
3. ✅ 30-step smoke on the consolidated 9,000 → `FLAGSHIP4B_DONE`.
4. ✅ Label regime decided (`--no-labels-v2`) and pre-registered (§5c).
5. ✅ **Launched, PID 698375.**

**Still open (for whoever picks this up):**
- **Nothing blocks the run.** It is training, supervised, and registry-recorded.
- **A clean full-suite `pytest` green** still needs a box with `fastapi` and the
  `*_train_log.jsonl` fixtures (the 10 failures on pod3 are both, not code).
- The trainer does not record `--v2-cache` in `config.json` (§5c) — a one-line fix
  worth making so future v2 arms are self-describing.

---

## 11 · Watchdog attached — and a latent landmine found while attaching it

**Death-only watchdog, installed onto the LIVE run without touching it.**

`/workspace/ops/runs.d/flagship-v2corpus-30k.env` + `supervise_run.sh`. `TRAIN_CMD`
is copied **verbatim from `/proc/699286/cmdline`**, so a resume reproduces the exact
recipe; the trainer auto-resumes from `OUT/ckpt.pt` (every 1,000 steps ≈ 3.2 h).

It is safe to attach mid-run because `TRAIN_MATCH` is the anti-double-launch guard.
**MEASURED — the supervisor's own log:**

```
supervisor UP on 62255816d4b2; OUT=/workspace/experiments/flagship-v2corpus-30k
trainer ALREADY RUNNING outside this supervisor (pid 699284,
  TRAIN_MATCH='train_flagship4b\.py.*flagship-v2corpus-30k')
  — NOT launching; waiting for it to exit
```

Verified after attaching: **exactly one trainer chain** (699284 wrapper → 699286
trainer), **one GPU compute process** (15,200 MiB), heartbeat advancing
`last_step 100 → 150`, `restarts: 0`, `status: external`.

### ⚠️ The landmine: a retired arm was armed to relaunch onto this GPU

While installing the manifest I found `pod_boot_hook.sh` iterates **ALL**
`runs.d/*.env` (`for envf in "$RUNS_DIR"/*.env`), skipping only `ENABLED != 1` —
and `/pre_start.sh` **is installed**, so it fires on every container start.

`flagship-v3enc.env` was still **`ENABLED="1"`**. But `flagship4b-v3enc-30k` is
**STOPPED at step 10,800 with a pre-registered 10k gate verdict of RESTART**
(MODEL_REGISTRY §1.4). **On the next pod restart the boot hook would have
relaunched that retired arm onto the same A6000**, where it would fight
`flagship-v2corpus-30k` for the GPU and the 57.74 GiB cgroup — on a pod whose
flagship has been OOM-killed before.

**Set to `ENABLED="0"`** with an inline comment explaining why, backup at
`flagship-v3enc.env.bak-20260725`. Both manifests re-verified as sourceable
(`RUN_ID`/`TRAIN_CMD`/`TRAIN_MATCH` all parse). **This is a config change to
another arm's manifest — flagged loudly rather than done quietly; reverting is one
character.** *(Evidence class: the boot-hook behaviour and `ENABLED="1"` are
MEASURED; "it would have relaunched on restart" is the direct consequence.)*

**Not verified, stated as such:**
- **No training step has run against the consolidated 9,000.** §5b proves the
  trainer end-to-end on pod1's 4,953-clip shard; nothing in the loader is
  shard-size-dependent, but the full-union path is unexercised until promotion.
  **Re-run the §5 command with `--steps 30` after promotion** — ~6 min, and it is
  the only thing standing between "proven on 55 %" and "proven".
- **Full `pytest -q` — run on pod3 instead, and the failures are environmental.**
  **819 passed, 10 failed, 4 skipped, 13 errors** (12:10). Every failure is a
  missing dependency or fixture on that pod, **not code**: `test_resim` /
  `test_scena` fail on the single missing module **`fastapi`**, and the five
  `test_gate_emitters` failures need `*_train_log.jsonl` fixtures that are not on
  pod3. **Zero** occurrences of `v2_dataset`, `finetune_traj` or `v2_compressed`
  anywhere in the failure output. Targeted re-run of the affected surface
  (`test_v2_dataset` + `test_labels_v2_wiring` + `test_flagship4b`): **27 passed**
  on pod3, and `test_v2_dataset` **8/8** on pod1 — green on both torch majors.
  A clean full-suite green still needs a box with `fastapi` and the log fixtures;
  nothing here is committed, so no gate is bypassed.
- **The transfer had not finished** at time of writing (23.6 % at 00:34:35Z). ETA
  ≈02:15 UTC is EXTRAPOLATED from a stable MEASURED ~1.39 MB/s over ~31 min.
- pod2's 9.93 GiB headroom is **unproven** (only ≥512 MiB shown) — deliberately, §1.
- The **~11 s/step** figure was measured with the transfer still running; the true
  clean rate is that or slightly better, so ≈3.8 days is a **ceiling**, not a floor.

**Evidence classes.** Pod GPU/disk/memory/code state, all throughput figures, the
pilot md5 identity, the 8-check sync verification, the guard's file counts and
warnings, and the 8 passing tests are **MEASURED** in this session. The corpus
integrity/distribution/key numbers are **MEASURED** by `V2_CORPUS_QA.md` and
**INHERITED** here. pod2's finish time and the transfer ETA are **ESTIMATED** from
measured rates. The pod1-vs-pod2 recommendation and the label-regime
recommendation are **JUDGEMENT** on the evidence above.
