# Pod migration — full inventory, the rescue, and the standing rule it overturned

**MEASURED 2026-07-28 17:5x–18:1x UTC.** PI request: *"screen the original CPU pod and try to save
data in order to use it with a new pod"*, following *"do an inventory over all the pods to screen if
we lost anything."*

## 0. Verdict first

✅ **NOTHING WAS LOST that we have not either recovered or priced.** Two genuine gaps, both small
and both named in §4. The headline worry — `flagship-v2corpus-30k` — is **intact and further along
than I reported.**

⭐ **The migration split the fleet across TWO datacenters.** The pods I could not reach were not
dead; they were in **US-TX-1** on reassigned ports while I probed **ca-mtl-1** endpoints. That single
fact explains every "outage" in the 12:57 and 17:57 reports.

---

## 1. The fleet, as it actually is

| host | endpoint | DC | GPU | volume | state |
|---|---|---|---|---|---|
| **old / `tanitad-pod`** | `38.147.83.15:39198` | **US-TX-1** | ⛔ **none (GPU removed)** | **444 GB** | source of the rescue |
| **pod2** | `69.30.85.123:22091` | ca-mtl-1 | A40 46 GB | **376 GB** | ✅ restore COMPLETE |
| **pod3** | `69.30.85.16:22079` | ca-mtl-1 | A40 | 495 GB | ✅ intact, idle |
| **eval** | `69.30.85.106:22073` | ca-mtl-1 | A40 | 512 B | ✅ fresh by design |
| **new** | `69.30.85.48:22192` | ca-mtl-1 | A40 46 GB | receiving | ✅ 493 MB/s real `dd` |

⚠️ **`tanitad-pod` in `~/.ssh/config` still says port 30107; the live port is 39198.** The port
moved, exactly the documented volume-resize signature — **not** a lost pod. `38.147.83.18:34126` is
stale and refuses.

## 2. ⭐ `flagship-v2corpus-30k` — intact, and FURTHER than reported

I reported it *"frozen at 23,850/30,000 (79.5 %)"*. **MEASURED:**

| item | value |
|---|---|
| last logged step | **24,550 / 30,000 = 81.8 %** |
| `ckpt.pt` | **3,415,808,330 B**, mtime **Jul 28 03:02** |
| milestones | `ckpt_step{5000,15000,20000}.pt`, 3.4 GB each |
| `train_log.jsonl` | 544,699 B, last write Jul 28 04:42 |

⚠️ `step_s: 544.1` is **accumulated over `--log-every`** (÷50 ⇒ ~10.9 s/step) — not a 544 s/step
pathology. The run slowed because **the pod lost its GPU**, not because of a code fault.

## 3. The rescue — and the rule it overturned

⭐⭐ **POD→POD DIRECT SSH WORKS. "Pods cannot SSH each other" is RETRACTED (C56).**

**MEASURED: 42 MB/s cross-datacenter** (US-TX-1 → ca-mtl-1) — `ckpt.pt` 3,415,808,330 B in **77 s**.
That is **42× the ~1 MB/s dev-box relay**, and it needs **no HF quota**, which is what made this
tractable at all while HF sits 403-storage-full.

**Why the old rule was wrong:** the genuinely blocked thing is **copying a private key** between pods
— correct, and it stays blocked. From that one blocked *method* I had concluded the *capability* was
absent. The standard alternative was never tried: **generate a keypair ON the destination and
authorise its PUBLIC key on the source.** No secret moves, so nothing is blocked.
A second contributor: the RunPod **proxy** really cannot transfer files (sftp →
`subsystem request failed on channel 0`; `scp -O` → exit 2) — a true limit of the *proxy* that I had
generalised into a false one about *pods*. The **direct** mapping
(`$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22`, read from the source's own env) was never probed.

**What that rule cost, measured:** a 22 GB move at **1.38 MB/s (~2 h)**; a 66 GB move written off
in-doc as *"18 h — unusable"*; and the R6 review lists it as **simultaneously blocking the formal
8-metric gate, a REF-C arm, and checkpoint backup.**

⚠️ **Trap found en route:** a nested `ssh` inside a piped script **consumes the rest of the script's
stdin** — the tail silently never runs and reads as a hang or a truncated log. Use `ssh -n`. This
cost two debugging rounds and produced two "empty" outputs I first mistook for failures.

### What is being rescued

Priority order, resumable at file granularity (a file whose local size already equals the remote size
is skipped, so a re-run is free). **Nothing is deleted on the source — deletion is not authorised.**

- **Tier 1 ✅ DONE** — `flagship-v2corpus-30k` resume set (`ckpt.pt` + config + log). The run is
  resumable on a GPU pod as of 17:59 UTC.
- **Tier 2** — the remaining ~54 GB of unique checkpoints: `flagship4b-v3enc-30k` (9.6 G),
  `refb-speed-30k` (5.9 G), `refb-{phase0,refbpatch,refbpatch-v2}-30k`,
  `flagship4b-v3enc-expA-nodrop-2k`, `p0-sB01-realmix`, `ft_trial`, `finetune_traj`,
  `ckpt_frozen.pt`, `axis6-{clean,relaxed}`.
- **Tier 3** — root `ckpt27k_flagship.pt`, `ckpt14k_frozen.pt`.
- **Tier 4** — data **unique to the old pod**: `cosmos` (41 G) and `physicalai_v2` (23 G vs pod3's
  17 G).

⛔ **DELIBERATELY NOT COPIED: `/workspace/data/physicalai_phase0` (302 G of the 365 G).** It carries
a `PARITY_OK` marker over the **same build** pod3 holds intact as
`pai_epcache/physicalai-train-e438721ae894` — **2376 episodes verified**. Copying it would spend ~2 h
and 302 GB duplicating a verified-good corpus. **This is written down rather than silently skipped:
if pod3 is ever lost, this decision must be revisited.**

## 4. The two genuine gaps

| gap | consequence |
|---|---|
| **`/workspace/smallval` empty on pod2** — arm A's `pw_A_old*.npz` did not return | arm A **completed `rc=0`** but its validation output is gone ⇒ **~4 h to re-run** if those numbers are needed |
| **`/workspace/experiments` empty on pod2** | no checkpoints there were unique; `flagship_v4_anchors_dense.pt` was recovered from pod3 (§5) |

## 5. ⭐ v5 / 176×624 is now LAUNCH-READY — every blocker cleared

1. **The PREP card's named residual blocker is GONE.** It read: *"the 120° VAL split (600 clips,
   ~24 GB) does not exist yet, and the trainer refuses to start without it."* pod2's restore brought
   it back — `physicalai-val-0c5f7dac3b11-w120-256x640cyl`, 603 entries / 20 G.
2. ⭐⭐ **PARITY IS CRYPTOGRAPHICALLY VERIFIED on the restored caches**, by the trainer's own
   preflight — not by my file count:
   - train: **2400 clips**, clip sha256 `e61a04553df5…` **matches the committed manifest**
     (sibling of `physicalai-train-e438721ae894`, skip-hash `f09e44db`)
   - val: **600 clips**, clip sha256 `0b176d2e5cb4…` **matches the committed manifest**
   ⇒ **the migration did not corrupt the caches.**
3. **No rebuild is needed for 176×624.** The cache's own `_geometry.json` states a centred sub-frame
   is *"a pure pixel slice… Nothing needs rebuilding to reach these numbers"* — it is a `--v2-subframe`
   argument on the existing 256×640 cache.
4. **pod2's checkout was 91 commits behind** at `0f93b98` — the documented drift trap. **Synced and
   verified by a real import** (not by `git log`): `seam_fail=1.5`, `seam_fail_frac=0.75`,
   `seam_fail_patience=50` ⇒ the redesigned population-over-time seam guard is live. Without this,
   an arm-B/C launch would have resurrected the max-based guard that a single outlier kills.
   ⚠️ **pod3 is ALSO at `0f93b98`** — sync it before its next launch.
5. **`--anchors-dense` recovered.** The path in **both published launch commands** is wrong
   (`/workspace/experiments/anchors/anchors_dense_1to20.pt` does not exist — `test_preflight_paths.py`
   documents exactly this); the real file is `flagship_v4_anchors_dense.pt`, found on pod3 at
   `/workspace/v4run/`, copied to pod2 **md5-verified `a51241e4ca547609a035e2f086c72b17`**, and
   **banked in this folder** so it is no longer single-disk.
6. **`PREFLIGHT: OK`** — run twice, the second time with anchors, on the real config.

⚠️ **ONE PRE-REGISTERED ITEM REMAINS OPEN, and it is the PI's call, not mine.** The v5 PREP card
records a counter-case at equal prominence: *"on EGO YAW RATE the wide frame is separated-WORSE
(−0.03546 R²)"* — while §2 of that card calls teaching the encoder to perceive ego-motion *"the real
item"* — and marks it **"Under investigation before v5 trains."** I have **not** launched. Launching
past a pre-registered hold would be choosing the outcome after seeing the convenience.

⚠️ Also standing from the PREP card: **v5 trains at 117.000°, not 120°** (the rig-clean slice costs
3° of field), and **v1's 0.4271 is not a valid comparator** for a wide-FOV arm — v1's encoder was
trained at 51.4°, so 100–120° frames are out-of-distribution for it in either direction.

## 6. Deliverable manifest

| artifact | where it lives |
|---|---|
| `MIGRATION_RESCUE.md` (this file) | repo, staged |
| `code/rescue_oldpod.sh` · `code/rescue_tier4_data.sh` | repo, staged; deployed at `newpod:/workspace/rescue_*.sh` |
| `flagship_v4_anchors_dense.pt` (43 KB, md5 `a51241e4…`) | repo, staged; live at `pod2:/workspace/experiments/` |
| `flagship-v2corpus-30k_train_log.jsonl` + `_config.json` | repo, `taniteval/results/trainlogs/` |
| rescued checkpoints (~60 GB) | `newpod:/workspace/rescue/` — **pod-resident, not in git** |
| rescue log | `newpod:/workspace/rescue/rescue.log` |
| C56 retraction | `Project Steering/RETRACTION_LOG.md` |
| corrected pod→pod rule | `CLAUDE.md` |

## 7. Recommended next actions

1. **Update `~/.ssh/config`:** `tanitad-pod` → port **39198**; pod2 → `69.30.85.123:22091`;
   pod3 → `69.30.85.16:22079`; eval → `69.30.85.106:22073`.
2. **Sync pod3's checkout** (also `0f93b98`) before it launches anything.
3. **PI decision — launch v5 176×624?** Everything is green except the pre-registered ego-yaw-rate
   hold in §5. Either clear that hold or accept it explicitly.
4. **Decide arm A's validation** — re-run (~4 h) or drop.
5. **The old pod can be released only after the rescue log shows Tier 4 complete** — and that is the
   PI's action, not mine; **deletion/termination is not authorised.**
