# POD SHIP RECEIPT — the refcv5-v2 trainer surface is ON `tanitad-a40`, proven by CONTENT and by BEHAVIOUR

**2026-09-06 · Architecture & Inference · zero GPU touched · NO TRAINING LAUNCHED.**
Machine-readable evidence: [`raw/pod_ship_20260906.json`](raw/pod_ship_20260906.json).

⛔ **This is a DEPLOYMENT audit. It carries no eval tier and no four-family table, deliberately.**
Stamping longitudinal/lateral/tactical/strategic metrics on a file-transfer receipt would be a
category error: nothing here predicts a trajectory. The four families bind the refcv5-v2 *run*,
not its shipment.

---

## 1. The headline

**Shipped commit: `78d2fb8`.** Every number below is stamped with the commit it was measured on —
see §8 for why that is not pedantry.

| | before | after |
|---|---|---|
| `…/stack/scripts/refc_v3_train.py` on the pod | 3,634 lines · `acad9ae6e336e52d1b61a314f93a39fa` · mtime 2026-09-05 23:38Z | **4,576 lines · `7f3043782a32456c95dc8f1d3b709761`** |
| repo `HEAD` blob (`78d2fb8`) | — | **4,576 lines · `7f3043782a32456c95dc8f1d3b709761`** |
| whole shipped subtree | 222/446 byte-identical | ⭐ **449/449 byte-identical** |
| `--help` **on the pod** | *(not attempted)* | ⭐ **exit 0, empty stderr, 35,446 B** |

⚠️ Both digests were asserted to be **32 hex characters** and explicitly checked against the
empty-input digest `d41d8cd98f00b204e9800998ecf8427e` before being called a match. A short or
empty digest is INCONCLUSIVE, never a pass — this bit us once already: an early `tar` invocation
failed (`Cannot connect to C: resolve failed`, a Windows drive letter read as an scp host) and
handed back exactly that empty digest.

⭐ **The pre-ship pod was `c8601d0^`** — the trainer from immediately *before* the P14 plumbing
commit. Its `sel_` census reads **8**, matching the pod digit for digit. That is what pins the
surface behind the plan's wrong-surface control (see `REFCV5_MISSING_PIECES_PLAN.md` §8.3).

## 2. The nine flags, proven ON THE POD, with negative controls

⭐ **A file that copied is not a trainer that parses.** md5 proves bytes; only execution proves
the program. `python3 refc_v3_train.py --help` was run **on `tanitad-a40`** with
`CUDA_VISIBLE_DEVICES=''`.

| must be PRESENT (P14 / P8 / P2 / P11) | hits | must be ABSENT | hits | probe-is-working (must be > 0) | hits |
|---|---|---|---|---|---|
| `--sel-refined` | 3 | `--max-speed-input` | **0** | `usage:` | 1 |
| `--sel-score-emitted` | 4 | `--str-goal-tok-head` | **0** | `--agents` | 3 |
| `--sel-score-emitted-t` | 2 | `--strategic-tokens` | **0** | `--anchors` | 3 |
| `--no-strategic` | 2 | `--p4-strategic` | **0** | `--steps` | 2 |
| `--goal-point-inject` | 3 | | | | |
| `--goal-point-geo-prior` | 2 | | | | |
| `--goal-point-t` | 2 | | | | |
| `--goal-point-w` | 2 | | | | |
| `--nav-args` | 2 | | | | |

⚠️⚠️ **A NEGATIVE CONTROL WAS RETIRED MID-SESSION, AND THAT IS THE POINT.** The first ship was
made at `be8c665`, where `--tac-goal-tok-head` read **0** and was a valid negative control. A
sibling then committed it: at `78d2fb8` the trainer is 4,576 lines and `--tac-goal-tok-head`
reads **2 — PRESENT**. ⛔ Quoting it as "absent" against `78d2fb8` or later would be false. **A
control's validity is scoped to the surface it was measured on**, and on a branch several agents
commit to, that surface moves in minutes. This is the same defect the plan's §8.3 documents,
reproduced inside the session that documented it.

⛔⛔ **THE NEGATIVE CONTROLS ARE WHY THIS RECEIPT EXISTS AT ALL.** On the FIRST ship every one of
the nine read **0** — and so did all four probe-is-working controls. Four zeros where non-zero was
required marked the census **INCONCLUSIVE**, not "nine flags absent". It was inconclusive because
**HEAD itself did not parse**:

```
ImportError: cannot import name 'goal_point' from 'tanitad.refs'
  refc_v3_train.py:75 -> refc_train.py:67 -> tanitad/refs/refc.py:156
```

`stack/tanitad/refs/goal_point.py` was **staged and never committed** while three HEAD files
imported it unconditionally at module level. Not a pod defect and not a currency defect — **the
ref was broken for everyone**. Fixed in `d1c929c`; full account in the plan's §8.4.

## 3. What was shipped, and why exactly that

⚠️ **Scope: `stack/tanitad` + `stack/scripts` only — 449 files, 3.34 MB of Python/JSON.** No data,
no checkpoints, no `.git`, nothing under `TanitAD Research Lab/`. This is the *python surface the
trainer runs on*, not a blind tree sync.

The substantive gap was **50 files**, enumerated before anything was sent:

* **32 real content drift** — including `refc_v3_train.py`, `refs/refc_v3.py`, `refs/refc.py`,
  `refs/refc_sampler.py`, `models/predictor.py`, `scripts/refcv5_preflight.py`: precisely the
  modules the composed arm needs.
* **18 absent entirely** — `refs/refc_strategic.py`, `refs/tac_goal_head.py`,
  `refs/anchor_twoseg.py`, `rl/control_space.py`, `rl/channel_guard.py`, `eval/constraints.py`,
  `models/_gradreach.py`, `effective_weights.py`, `scripts/build_b1_agent_join.py`,
  `scripts/sup_refcv5.sh` and 8 more.

⚠️ A further **174 files differed by LINE ENDINGS ONLY** and are *not* drift — see §4. Reporting
the raw diff (206 files) would have overstated the gap by 4×. **80 pod-only files** (67
`__pycache__`, 8 `.NEW`/`.PRE_*` backups, 4 pod-local) were left untouched: `tar` extract
overwrites only archived paths and deletes nothing. The pre-ship trainer is preserved at
`…/refc_v3_train.py.PRE_SHIP_20260906`.

## 4. ⛔ A DEFECT IN `ship_and_gate.sh`, MEASURED — its step 1 breaks its own step 2

Step 1 runs plain `git archive HEAD`. On this dev box `core.autocrlf=true`, **and `git archive`
honours it**:

```
packed by plain git archive : 270,331 B · md5 30889783c803ff87f76ac95f833a311a
HEAD blob                   : 265,852 B · md5 cee9196b5406608136a0ccbc0987c404
delta                       : +4,479 B == exactly one CR per line (4,479 lines)
```

Step 2 then compares the pod's file against `git show HEAD:… | md5sum` (an **LF** blob) and
**fails** — correctly, but with a signature that reads like a transfer fault rather than an EOL
conversion. **Fix: `git -c core.autocrlf=false -c core.eol=lf archive …`**, applied here, after
which packed md5 == blob md5 exactly.

⭐ The 174 EOL-only files are the collateral proof that an earlier ship ran without this pin. They
are now normalised to LF, so **a whole-subtree md5 gate is usable on this box for the first time**.

**Two further deviations from the script, both deliberate and both stated:**

1. **Step 0's dirty-worktree refusal was bypassed.** `stack/scripts/refc_v3_train.py` is `MM` (a
   sibling mid-edit at 4,576 lines). The guard is right in spirit — never ship a worktree — but
   step 1 ships **blobs from `HEAD`**, which worktree dirt cannot reach, and HEAD's blob md5 is
   the declared target. Verified after the fact: the packed trainer md5 equals the HEAD blob md5.
2. **Steps 3 (`pod_currency_audit`) and 4 (`refcv5_preflight`) were NOT run.** This is a
   ship-and-stop; the P1 gate belongs to a sibling. §5's audit is an equivalent whole-subtree
   content check done inline.

## 5. Currency audit — whole subtree, by content

449 files compared HEAD→pod by md5: **449 identical · 0 differ · 0 absent.**
Same-breath non-zero control: the identical count is 449, not 0, so the manifests were really read.
**Verdict: the pod's `stack/` equals `HEAD` `78d2fb8`.**

⚠️ **This receipt has a shelf life measured in minutes.** Three sibling commits landed on
`agent/arch-inf-20260803` while this ship was being verified (`be8c665` → `d1c929c` → `78d2fb8`
and beyond), one of which rewrote the trainer. The ship was re-run against the newest HEAD each
time. **The ship is idempotent and costs ~30 s: re-run `ship_and_gate.sh` immediately before
launch rather than trusting any receipt, this one included.**

## 6. ⚠️ THE 317 MB AGENT JOIN IS NOT IN GIT — reproducibility record

`b1train_agents.jsonl.xz` is **not versioned**. Only its `.meta.json` sidecar is committed. Two
copies exist, both verified by md5 in this session, neither under version control:

| location | path | bytes | md5 |
|---|---|---|---|
| dev box | `C:\Users\Admin\tanitad-caches\b1-train-join-20260906\b1train_agents.jsonl.xz` | 317,028,572 | `1c985e6d6ad34e605c4ebd30cb353558` |
| `tanitad-a40` | `/workspace/TanitAD/data/joins/b1train_agents.jsonl.xz` | 317,028,572 | `1c985e6d6ad34e605c4ebd30cb353558` |

⛔ **It must NOT be committed** — a 317 MB binary does not belong in this repo. It is made
reproducible instead. Contents: **849,263 rows · 4,427 of 4,572 clips joined · 1,806 s build ·
`digest_scope: compressed`.**

⚠️ The committed sidecar records `md5` and `n_lines` but **not the artifact's byte size**. It is
recorded here: **317,028,572 bytes.**

**Build command** (`stack/scripts/build_b1_agent_join.py`, argv exactly as recorded in the sidecar):

```
python stack/scripts/build_b1_agent_join.py \
  --pose-source   reconstruct \
  --ts-dir        C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov \
  --clips         b1_train_clips.json \
  --n-stack       3 \
  --ego-dir       C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo \
  --obstacle-dir  C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1train \
  --out           C:/Users/Admin/tanitad-caches/b1-train-join-20260906/b1train_agents.jsonl.xz
```

`b1_train_clips.json` **is** tracked, at
`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-06-b1-train-join/raw/b1_train_clips.json`.

### Does `--pose-source reconstruct` reproduce it byte-for-byte?

⭐ **On the EVAL join: YES, MEASURED.** Same builder, same 141 EVAL clips, same obstacle parquets,
**only `--pose-source` changed** (`v2ep` → `reconstruct`). Both runs produced md5
`3ddb42ecbd3926066795a94587af2aed` at **10,012,564 bytes** — `identical_md5: true`,
`identical_bytes: true` (`reconstruct_byte_identity.json`). Upstream, reconstructed poses are
**float32-identical** to banked `v2ep` poses on 20/20 clips (`max_dxy_m 0.0`, `max_dyaw_deg 0.0`,
`max_dv_mps 0.0`).

⛔ **On THIS TRAIN join: INFERRED, NOT MEASURED — and the distinction is not pedantry.** The TRAIN
join was only ever built with `--pose-source reconstruct`; **no `v2ep`-built TRAIN counterpart
exists**, because avoiding the ~161 GB TRAIN epcache on Thor was the entire point of reconstructing.
So the EVAL result demonstrates the *pose source* is substitutable; it does not demonstrate that
re-running the argv above regenerates *this* 317 MB file bit-for-bit. **What would settle it:**
re-run the recorded argv on the dev box (~30 min wall, zero GPU) and compare against
`1c985e6d6ad34e605c4ebd30cb353558`. Until then the honest statement is: **reproducible by
construction, unverified by re-execution.**

### ⭐ SETTLED 2026-09-07: BYTE-IDENTICAL, MEASURED

The re-execution named above was done. Full record:
[`../2026-09-07-join-repro/`](../2026-09-07-join-repro/README.md) ·
machine-readable [`raw/repro_result_20260907.json`](../2026-09-07-join-repro/raw/repro_result_20260907.json).

The recorded argv was replayed verbatim (only `--out` redirected to a new path; the live artifact
was never opened for writing), from a private `git archive HEAD` tree, 31 min, zero GPU.
**Ten axes, ten matches:**

| | original | re-build |
|---|---|---|
| ⭐ decompressed sha256 | `0283df30…37fc` | `0283df30…37fc` |
| ⭐ decompressed md5 | `8a9277a77b132e2631b133f9bea0758e` | same |
| `.xz` md5 | `1c985e6d6ad34e605c4ebd30cb353558` | same |
| bytes | 317,028,572 | 317,028,572 |
| rows / clips / boxes | 849,263 / 4,427 / 28,053,187 | identical |

⭐ **The decompressed digest ties to git**: this receipt's own committed sidecar records
`8a9277a77b132e2631b133f9bea0758e` as "the other candidate" it hashed when choosing digest scope —
and that is exactly what both sides hash to.

⇒ **§6's "must NOT be committed" now rests on a MEASURED recipe, not an inference.** Unversioned is
an acceptable risk for this artifact: the argv, the builder + 4 modules, `b1_train_clips.json` and
the obstacle re-pull script are all in git, and the only unversioned inputs are ~4.1 GB of upstream
PhysicalAI parquets. (The camera directory's 58.8 GB is mp4 the builder never opens — the real
timestamp input is **56 MB**.)

⚠️ **Three corrections to §6 and §7, measured:**
1. The artifact is in **four** places, not two — its exact bytes are also embedded at offset 0 of
   `b1_train_plus_eval_agents.jsonl.xz` on both the pod and the dev box. One of the four is a temp
   scratchpad and must not be counted as durable.
2. ⛔ **`b1_train_plus_eval_agents.jsonl.xz` — the file `--agents head` actually reads — has no
   sidecar and its digest was recorded nowhere.** It is
   **`0c31a3a63d7205e9fa50e8ac4204ef46`, 327,041,136 B**, and it is exactly `cat train eval`
   (proven by streaming the two parts through one md5).
3. ⚠️ **Nothing in `.gitignore` guards the 317 MB artifact** (`git check-ignore` matches no rule;
   control: `__pycache__/` matches correctly). One `cp` into a `raw/` dir plus a routine `git add`
   would put it in history irreversibly.

## 7. Is the pod ready to launch refcv5-v2?

⭐ **The DEPLOYMENT blocker is gone.** The trainer on the box is HEAD, byte for byte; it parses;
all nine levers are on its command line; the 317 MB join is staged beside it; the A40 is idle (only
`jupyter-lab` running) and a real `dd` write test — **not `df`, which reports the cluster and hides
the per-pod quota** — moved 200 MB at 291 MB/s.

⛔ **The launch itself is still gated, and NOT by anything above.** `REFCV5_MISSING_PIECES_PLAN.md`
§4 binds: the composed arm does not launch until **P1 is IN or explicitly OUT**, and that gate
belongs to a sibling. A second P1-IN blocker has since been found *and fixed* (`5a34fce`): the
trainer applies the same `--agent-join` to the EVAL dataset, and the B1 TRAIN join has **zero clip
overlap** with the 141-episode EVAL grid, so `--agents head` would have refused at startup. The
prepared fix concatenates the B1 EVAL join (`CONTROL_b1eval_recon.jsonl.xz`, 26,394 rows) onto the
TRAIN join — verified through the trainer's own reader at 875,657 rows.

⚠️ **Whoever launches must also honour a constraint the trainer enforces at startup:**
`--goal-point-inject` and `--nav-args` **cannot both be passed** — the first sets
`nav_inject = False` and the second then raises `SystemExit` (`refc_v3_train.py:380-384`). See
plan §8.5 for that and the rest of the cannot-enter list.
