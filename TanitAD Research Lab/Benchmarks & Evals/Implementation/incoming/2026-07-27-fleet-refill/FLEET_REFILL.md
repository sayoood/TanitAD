# Fleet refill — 2026-07-27

Owner: fleet-ops agent, drumbeat iteration 2026-07-27.
Repo HEAD at start: `fdc5b4f` → **advanced to `bdb4fec` mid-session** (the
orchestrator swept other agents' staged work while this ran; re-checked at
filing). Branch `agent/benchmarks-eval-20260721`.
All pod times are **UTC**; the dev box reads Europe/Berlin (UTC+2).

⚠️ **The `_decode_mp4` bug in §3.3 is still present at `bdb4fec`** — verified at
filing. `bdb4fec` did not touch `physicalai.py`; my fix is **staged on top of it**
and is the only thing standing between HEAD and a dead corpus builder.

---

## 0. Headline — what changed on the fleet

| host | before | after | left running? |
|---|---|---|---|
| `tanitad-pod` (pod1) | 🟢 training `flagship-v2corpus-30k` | **untouched** | n/a — not mine |
| `tanitad-pod2` | 🟡 `fov_extract.py` ×4 | 🔴 **still idle — build REFUSED on a parity blocker** (§3) | no |
| `tanitad-pod3` | 🔴 idle, repo drifted | ✅ **ran the owed IDM `steer` retrain to completion** (65 min, PID 1273634, exited clean) — **now idle again** | no — finished, all artifacts pulled |
| `tanitad-eval` | 🔴 idle + 5 zombies | 🟢 **free — 6 PIDs reaped** (§1) | no |

⚠️ **pod2, pod3 and eval are all idle as of filing.** The next iteration has three
free hosts; §4 lists what each is blocked on.

**Three results worth reading even if nothing else is:**

1. ⭐ **The `steer` regression IS a data-budget effect — and it was specifically a
   PhysicalAI regression.** At the v3 budget (68 episodes) the retrain reproduces
   v3 almost exactly (**+0.4175** vs published 0.408) and is **CI-separated WORSE
   than the deployed head on PhysicalAI**. By 400 episodes it **beats the deployed
   head, CI-separated** (paired ΔMAE **−0.0024 [−0.0039, −0.0010]**) — but the
   win is carried by **comma2k19** (R² +0.5648 → **+0.7951**); on PhysicalAI it
   recovers to **parity, not superiority**. §2.5.
2. ⛔⛔ **`_decode_mp4` is BROKEN at HEAD on every path, deployed included — a
   total corpus-build outage, introduced by `fdc5b4f`, and no test caught it
   because no test decodes a real mp4.** A CanonicalFrame is shadowed by PyAV's
   loop variable. **Fixed and staged**; full suite 1253-green. §3.3.
3. ⛔ **The pod2 wide-FOV cache build is NOT buildable on pod2 and I refused it.**
   The canonical corpus needs **≥2,400 raw clips**; pod2 holds **760** (31.7 %
   coverage). A rebuild there is an *episode re-selection*, which parity rules
   require be refused, not worked around. The geometry itself is sound — 120° is
   delivered exactly, now proven on real decoded frames. §3.

---

## 1. `tanitad-eval` — reaped, now free

**Six** processes were dead, not five. All killed **by explicit PID** (`pkill -f`
/ `pgrep -f` self-match the ssh command and would have killed my own session).
**No files were deleted.**

### 1.1 The five briefed PIDs — verified dead before killing

They are **not** alpasim eval workers: `fd 1`/`fd 2` on all five point at
`/workspace/scaled_refc_runtime.log`, i.e. they are **REF-C runtime-scaling**
workers. That log last changed **Jul 23 04:05**, ~4 days stale.

| PID | wchan | utime/stime (ticks) | elapsed | verdict |
|---|---|---|---|---|
| 1487279 | `pipe_read` | 7 / 2 | 4-01:20 | resource_tracker |
| 1487782 | `futex_wait_queue` | 1685 / 334 | 4-01:18 | worker |
| 1487785 | `pipe_read` | 1629 / 330 | 4-01:18 | worker |
| 1487926 | `futex_wait_queue` | 2001 / 330 | 4-01:18 | worker |
| 1487931 | `futex_wait_queue` | 1856 / 323 | 4-01:18 | worker |

**Zero-progress proof (MEASURED):** utime/stime sampled twice ~4 minutes apart
were **byte-identical** for all five. ~17 s of CPU across 4 days, blocked in
`futex_wait_queue` / `pipe_read` — the documented futex deadlock. No output file
was open for writing other than the already-complete log.

### 1.2 ⚠️ The sixth process — an undocumented watcher that could NEVER have fired

```
PID 1778612  bash -c while pgrep -f wave2.sh > /dev/null; do sleep 20; done; bash /workspace/idm3/wave3.sh
```

This is a **fresh instance of the `pgrep -f` self-match trap, in a new form**.
`pgrep -f wave2.sh` matches *its own parent's* command line — PID 1778612's
cmdline literally contains `wave2.sh` — so the guard is permanently true and
**`wave3.sh` would never have launched**. It had been spinning 1 h 28 m.

It was harmless only by luck: `wave4.sh` had independently already produced
`ship_tra.json` (04:25) and `ship_rot.json` (04:28), which are the artifacts IDM
v3 shipped. Had the loop ever fired it would have **overwritten those two
already-published files**. Reaped; verified `wave3.sh` did not fire and both
JSONs are intact.

> **Root-cause class (for `RETRACTION_LOG.md`): `pgrep -f` self-match.** The
> documented form is "it kills your own ssh session". This is the *inverse* form:
> a watcher that never fires and silently strands a chained job. The rule should
> read **"`pgrep -f <pat>` matches any process whose cmdline contains `<pat>`,
> including the shell that is running the pgrep"** — kill *and* wait by explicit
> PID.

**After:** GPU 0 % / 0 MiB, no user python processes, host free.

---

## 2. `tanitad-pod3` — repo synced, IDM `steer` retrain RUNNING

### 2.1 Repo sync — done NON-destructively (and this mattered)

pod3 was at `0f93b98` in **detached HEAD** with **69 modified tracked files** and
a large set of untracked `stack/scripts/*.py`. I did **not** reset it:
`git diff --ignore-all-space` still reports **33 files / 3,680 insertions**, so
those are **real local edits, not line-ending churn**. Resetting would have
destroyed them.

Instead: `git fetch` + a **new worktree** at `origin/main`:

- `/workspace/TanitAD` — untouched, still `0f93b98` + its 69 local edits
- `/workspace/TanitAD-main` — **clean worktree at `origin/main` = `2d903ba`**

**Real `import tanitad` verification (not a git log) — MEASURED:**

```
tanitad OK -> /workspace/TanitAD-main/stack/tanitad/__init__.py
comma2k19 OK -> /workspace/TanitAD-main/stack/tanitad/data/comma2k19.py
```

⚠️ **`origin/main` is BEHIND the local repo and does not contain the work this
task depends on.** Measured on pod3 and in the repo:

| thing | local `fdc5b4f` | `origin/main` `2d903ba` |
|---|---|---|
| `comma2k19.HEADING_MODE_HOLD` / `hold_heading_through_standstill` | present | **ABSENT** |
| `data/parity.register_geometry_sibling` | present | **ABSENT** |
| `data/calib.CanonicalFrame` | present | **ABSENT** |

So **no pod can obtain the heading repair or the geometry seam via git.** Both
had to be shipped by scp. This is an integration escalation — see §4.

### 2.2 Which label protocol I trained on — stated explicitly

**REPAIRED**, `idm3_labels.heading_repair`, `v_min = 0.5 m/s` — identical to
every v3 arm, so the numbers are comparable to v3's.

Two clarifications that matter for how the result is read:

- The repair used by the IDM line is **`idm3_labels.heading_repair`, applied
  post-hoc to the `poses` in the latent cache** — it is *not*
  `comma2k19.HEADING_MODE_HOLD` (the loader-level, opt-in, default-LEGACY fix).
  No corpus rebuild was needed, and none was done.
- I extended the repair predicate: stock `idm3_arms.repair_labels` tests
  `tag.startswith("cm_")`, which is **False** for the linked-in comma extras.
  Unpatched, 79 extra comma episodes would have trained on the **broken**
  arctan2-at-standstill heading while the original 64 were repaired — a silently
  mixed label protocol inside one training set. `idm4_steer.repair_labels_ext`
  fixes this.
- The repair rewrites the **yaw/heading** channel; it does **not** rewrite
  `steer`. It is applied for protocol identity with v3, not because it moves the
  headline channel.

### 2.3 ⚠️ The corpus, and the leak I had to measure my way out of

The retrain needs more episodes than the v3 cache holds. pod3 carries a second,
larger latent cache from the 2026-07-22 idm-proof run
(`/workspace/tmp/idm/latents`, **770** files, encoded with the **same** frozen
flagship-v1 encoder — `md5 b5f07d9e3dd2ca643949bc86832e6585`, the exact value
`idm2_encode.py` asserts).

**But those latents store only `{z, poses, actions}` — no `episode_id`, no
`src`.** Episode identity therefore **cannot be read off metadata**, and
assuming disjointness here is precisely the REF-A I-JEPA failure mode (~80 % val
leak). So I measured it: every episode in both caches was content-fingerprinted
(md5 of float32 `poses`).

**MEASURED** (`raw/fp_eval.json`, `raw/fp_pod3.json`, `raw/lat_overlap.json`):

| quantity | value |
|---|---|
| unique episodes, v3 cache | 104 |
| unique episodes, pod3 cache | **708** (770 files − **62 internal duplicates**; `pai_val` ⊂ `pai_a` ∪ `pai_b`) |
| shared between the two caches | **19** |
| union | **793** |
| **v3 VAL episodes found in the pod3 cache** | **4** → `cm_00018`, `cm_00039`, `pai_00000`, `pai_00018` |
| training pool after excluding all 36 val episodes | **757** (cm 121 / pai 636) |

⚠️ **Without the fingerprint check this run would have leaked 4 of its 36 val
episodes (11 %) into training.** They are excluded by content, not by name, and
the exclusion is asserted at runtime and recorded in the output JSON.

**The val split is provably unmoved.** `idm2_lib.split_tags()` derives the split
from whatever is in the latent dir, so adding episodes naively would have moved
the val set and silently destroyed every comparison. The extras are linked in
under prefixes **`cmx_` / `paix_`**, which `split_tags` cannot select — it
matches `t.startswith(dom + "_")` for `dom in ("pai", "cm")`, and neither
`"cmx_…"` nor `"paix_…"` starts with `"cm_"` or `"pai_"`. The script **asserts**
68 train / 36 val before training, and the A0 prediction array is asserted to
have exactly 4,195 rows.

**Data-budget increase: 68 → 757 episodes = 11.1×.** Per corpus (which is how
`steer` must be read, because it is not the same physical quantity across
corpora): **comma 42 → 121**, **PhysicalAI 26 → 636**.

### 2.4 ✅ The pod3 job — RAN TO COMPLETION, nothing left running

| | |
|---|---|
| **host** | `tanitad-pod3` |
| **PID** | 1273634 — **exited cleanly** (`IDM4_DONE 3918.9s`) |
| **ran** | 2026-07-27 **05:39 → 06:44 UTC** (65 min), 4 rungs × 3 seeds × 50 epochs |
| **log** | `pod3:/workspace/idmretrain/out/idm4.log` → pulled to `raw/idm4.log` |
| **results** | `pod3:/workspace/idmretrain/out/idm4_steer.json` → pulled to `raw/idm4_steer.json` |
| **checkpoint** | `pod3:/workspace/idmretrain/out/idm_head_v4_steer.pt` → pulled to `idm_head_v4_steer.pt` (11.6 MB, rung 757 seed 0) |

**pod3 is idle again.** All artifacts are in the repo; nothing is stranded.

Exact command (launched detached with `ssh -f` + `setsid nohup`):

```bash
cd /workspace/idmretrain && \
setsid nohup env PYTHONPATH=/workspace/TanitAD-main/stack \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  /workspace/venv/bin/python -u idm4_steer.py \
    --rungs 68 200 400 757 --seeds 0 1 2 --epochs 50 \
    --save-ckpt /workspace/idmretrain/out/idm_head_v4_steer.pt \
    --out /workspace/idmretrain/out/idm4_steer.json \
  > /workspace/idmretrain/out/idm4.log 2>&1 < /dev/null &
```

`OMP_NUM_THREADS=6` / `MKL_NUM_THREADS=6` set per the standing hazard note.
The JSON is rewritten after **every rung**, so a killed job still yields the
rungs it finished.

### 2.5 ⭐ Result so far — the data-budget hypothesis is being CONFIRMED

Pre-registered before running (both outcomes publishable): **H1** steer rises
with budget and the top rung beats A0 ⇒ data budget explains the v3 regression;
**H0** flat or short of A0 ⇒ it does not, and the standing explanation is retired.

Val: **36 episodes / 4,195 windows**, identical to v3's and to A0's stored
predictions. Estimator: **paired episode-cluster bootstrap**, unit = episode,
B = 2000 (`taniteval/ci.py`). `overlapping_holdout_se` never called.
All numbers **MEASURED**, `raw/idm4_steer_interim.json`.

**Baseline, re-measured on these exact windows:**
`A0 (idm_head_v1) steer R² = +0.7419` (pai +0.7340 / cm +0.5648).
This lands on the published **0.742**, so the baseline is confirmed rather than
inherited.

`steer` R² is the **seed-mean prediction** (3 seeds ensembled, which is what the
JSON reports — not the mean of per-seed R²).

**COMPLETE** — 4 rungs × 3 seeds × 50 epochs in **3,918.9 s** (65 min).

| rung (episodes) | cm / pai | train windows | **steer R² pooled** | **PhysicalAI** | **comma2k19** |
|---:|---|---:|---:|---:|---:|
| **A0 — the deployed head** | — | — | **+0.7419** | **+0.7340** | **+0.5648** |
| **68** (= v3's exact train set) | 42 / 26 | 15,875 | +0.4175 | +0.3711 | +0.5841 |
| **200** | 32 / 168 | 37,444 | +0.7262 | +0.7073 | +0.7453 |
| **400** | 64 / 336 | 74,854 | +0.7502 | +0.7317 | +0.7951 |
| ⭐ **757** | 121 / 636 | 141,628 | ⭐ **+0.7993** | ⭐ **+0.7858** | ⭐ **+0.8071** |

**Monotone in the budget on both corpora.** Per-seed at rung 757: +0.7866 /
+0.7857 / +0.7891 — a 0.0034 spread, so this is not a seed artifact.

**Paired ΔMAE vs A0** (negative = retrain better), episode-cluster bootstrap,
**per corpus, never pooled alone** — `steer` is `atan(L·κ)` on PhysicalAI and
`STEER_RATIO = 15.3` on comma2k19:

| rung | pooled | **PhysicalAI** (n = 14 ep) | **comma2k19** (n = 22 ep) |
|---:|---|---|---|
| 68 | +0.0036 [−0.0005, +0.0093] | ⛔ **+0.0161 [+0.0045, +0.0319] SEPARATED WORSE** | −0.0014 [−0.0033, +0.0006] |
| 200 | −0.0010 [−0.0031, +0.0016] | +0.0010 [−0.0052, +0.0089] | ⭐ **−0.0018 [−0.0034, −0.0005] SEP.** |
| 400 | ⭐ **−0.0024 [−0.0039, −0.0010] SEP.** | −0.0014 [−0.0044, +0.0019] | ⭐ **−0.0028 [−0.0045, −0.0014] SEP.** |
| ⭐ **757** | ⭐ **−0.0033 [−0.0047, −0.0021] SEP.** | ⭐ **−0.0042 [−0.0066, −0.0013] SEP.** | ⭐ **−0.0030 [−0.0048, −0.0017] SEP.** |

⭐⭐ **H1 is CONFIRMED, and at the top rung the retrain beats the deployed head on
BOTH corpora with each interval separated on its own** — not just pooled. The
model card's bar ("beat 0.742 on a paired, episode-disjoint read") is met with
room to spare, and episode-disjointness is *measured by content fingerprint*, not
assumed.

**The sharper story the per-corpus split tells:**

- **The v3 `steer` regression was specifically a PhysicalAI regression.** At the
  v3 budget the arm is **CI-separated WORSE than A0 on PhysicalAI** (+0.0161)
  while comma was already fine (−0.0014, not separated). 68 clips starved the
  corpus that had only **26** of them.
- **The recovery is ordered:** comma separates first (rung 200), pooled at rung
  400, **PhysicalAI only at rung 757** — which is exactly what a data-budget
  story predicts, since PhysicalAI is the corpus that was starved.
- ⚠️ The rung-400 PhysicalAI cell (−0.0014, not separated) was **UNPOWERED, not a
  refutation** — rung 757 resolves it at −0.0042. This is the ladder's own
  demonstration of why "not separated" must never be reported as "no effect".

**Bonus — the other channels at rung 757 (seed-mean, same windows):**

| channel | v4 @757 | A0 | note |
|---|---:|---:|---|
| `yaw_rate` | **+0.9188** | +0.8108 | also **above IDM v3's shipped +0.841** |
| `speed` | +0.8650 | +0.8651 | identical — speed was never budget-limited |
| `long_accel` | −0.0591 | −0.2398 | **still negative — stays unshipped**, per the v3 contract |

That `speed` is unmoved while `steer` and `yaw_rate` climb is a useful control:
the budget effect is specific to the rotation channels, not a global "more data
helps everything".

**Reading it:**

- ⭐ **The rung-68 control reproduces v3 to within seed noise** — steer R²
  **+0.4175** vs published **0.408**, and paired ΔMAE vs A0
  **+0.0036 [−0.0005, +0.0093]** vs the model card's
  **+0.0035 [−0.0005, +0.0093]**. The two intervals are essentially identical.
  This is the control that makes the rest of the ladder attributable to the
  **budget** rather than to anything I changed.
- **Tripling the budget (68 → 200) moves steer +0.309 R²**, from far below A0 to
  level with it; **6× (68 → 400) separates.** The curve is steep exactly where v3
  was sitting, which is why 68 episodes read as a recipe failure.
- ⚠️ **Rung 200 "not separated" was UNPOWERED, not refuted — and the higher rungs
  proved it.** At rung 200 the interval half-width was ±0.0023 MAE against a
  point estimate of −0.0010; separation needed roughly **|Δ| ≥ 0.0023** at
  n = 36 val episodes. Rungs 400 and 757 delivered the larger effect rather than
  a larger val set. **Do not read the rung-200 row as evidence of no effect.**
- ⚠️ **The per-corpus intervals rest on n = 14 (PhysicalAI) and n = 22
  (comma2k19) episodes**, and the val set is **fixed at 36** by the pairing
  requirement against A0's stored predictions. Those n's are small; the rung-757
  PhysicalAI interval [−0.0066, −0.0013] excludes 0 but is wide, so the *size* of
  the PhysicalAI win is poorly pinned even though its *sign* is established.
- ⚠️ **The rung-68 control also reproduces v3 per corpus**: the model card reports
  v3 `steer` **pai +0.360 / cm +0.583**; this control gives **+0.3711 / +0.5841**.
  Within seed noise — the card stands and the harness is faithful.
- ⚠️ **The pooled column remains mix-confounded** (rung 68 is 42 cm / 26 pai;
  rung 757 is 121 cm / 636 pai). It is reported for continuity with the model
  card, but **the per-corpus columns are the quotable ones** — and at rung 757
  they agree with the pooled read, so nothing hinges on the confound.
- ⚠️ **The pooled column is confounded and the per-corpus columns are the real
  read.** The pool is PhysicalAI-heavy (636/757), so climbing the ladder also
  shifts the corpus mix (rung 68 is 42 cm / 26 pai; rung 757 is 121 cm /
  636 pai). Since `steer` is `atan(L·κ)` on PhysicalAI and `STEER_RATIO = 15.3`
  on comma2k19 — **not the same physical quantity** — the pooled R² partly
  tracks the mix. The per-corpus columns are on a **fixed** val set and are not
  affected by this.

⛔ **The deployed `steer` head has NOT been replaced by me — that is a shipping
decision, not a fleet-ops one.** But the gate is now passed, so here is exactly
what is owed to ship it:

1. ✅ **Bar met.** "Do not replace `steer` unless the retrain beats 0.742 on a
   paired, episode-disjoint read" — beaten (**+0.7993** vs +0.7419), paired,
   episode-disjointness **measured by content fingerprint**, and CI-separated
   **on each corpus independently**.
2. **Ship the rung-757 head** (`idm_head_v4_steer.pt`, staged). ⚠️ It is **seed 0
   only**; the reported R² is the **3-seed ensembled prediction**, so a single
   -seed checkpoint will score slightly below the headline. Either ship the
   ensemble or re-derive the headline for the single seed — **do not quote the
   ensemble number for the seed-0 file.**
3. **Re-issue `MODEL_CARD_IDM_V3.md`**: it says *"`steer` is WORSE than the
   previous head (0.408 vs 0.742) … Do not use v3 for `steer`"* and calls it *"a
   data-budget regression, not a recipe improvement"*. **The diagnosis is now
   CONFIRMED and the prohibition is superseded** at ≥400 episodes.
4. **Update `MODEL_REGISTRY.md`** with the new head and this ladder as its
   provenance. Also note `yaw_rate` **+0.9188**, above v3's shipped **+0.841**.
5. **`long_accel` stays unshipped** (−0.0591). Better than A0's −0.2398, still
   negative; the v3 contract that excludes it is unchanged.

---

## 3. ⛔ `tanitad-pod2` — wide-FOV cache build REFUSED (parity), geometry PROVEN

pod2's `fov_extract.py` had exited and the host was fully idle (GPU 0 % / 0 MiB)
before I touched anything. I did **not** build the cache. Reason below; this is a
refusal on the parity rule, not a workaround and not a scope reduction.

### 3.1 The blocker — pod2 cannot reproduce the canonical corpus

**MEASURED** (`raw/wfov_preflight_2026-07-27.json`), from the canonical cache's
own marker on pod2:

```
_epcache/physicalai-train-e438721ae894/DONE = {"episodes": 2376, "skipped": 24}
ep_*.pt present                            = 2376
raw clips present on pod2                  = 760
clips required (2376 + 24 skipped)         = 2400
coverage                                   = 31.67 %
parity_rebuild_possible                    = false
```

**Absence probed at three locations before asserting** (per the standing rule):
`find /workspace -name '*.mp4'` → **760**, all in one directory; the second
candidate root `physicalai_phase0/camera/camera_front_wide_120fov` is **empty**;
no other video/HF-cache directory exists. The brief's inventory (760 R0 clips,
197 calibration chunks, 197 egomotion zips) is **correct** — what it did not
carry is that 760 ≠ the 2,400 the corpus needs.

**Why this is a refusal and not an obstacle to route around:** the binding runbook
is *rebuild → `register_geometry_sibling()` → commit manifest*, and the
registration mints a sibling key **only if uid digest + count + skip indices
match `e438721ae894` exactly**. At 760 clips all three fail, and — decisively —
`build_pai_cache.py` calls `split_clips(clips, val_frac=0.2, seed=…)` on the
**discovered** clip list, so a 760-clip run produces a **different train/val
episode selection**. That is exactly the "anything that re-selects episodes must
be refused" case. Building anyway would have cost ~1.2 h of decode and ~113 GB
for an artifact `corpus_key_of()` reads as NON-PARITY and the trainer refuses.

### 3.2 What I did instead — the geometry preflight PASSES

`wfov_preflight.py` mirrors the non-canonical branch of `build_pai_cache.py`'s
pre-decode geometry assert, decoding nothing.

| quantity | requested | **delivered (MEASURED)** |
|---|---|---|
| frame | 256 × 640 | 256 × 640 |
| projection | cylindrical | cylindrical |
| **HFOV** | 120.00° | **120.00°** — shortfall **0.00°**, `PASSES_BUILD_ASSERT: true` |
| `f_eff` | — | **305.577** |
| observed fraction | — | **0.9147** (8.5 % masked, not fabricated) |
| tokens @ patch 16 | — | **640** (grid 16 × 40) |
| `state_dim` | — | **2048** — invariant, as documented |
| cache key fragment | — | `{"geom": "256x640f305.5775cyl"}` |
| sensor | — | 1080 × 1920, **per-clip intrinsics: true** |

So the 120° cylindrical frame is **genuinely deliverable on real PhysicalAI
intrinsics** — it does not clamp and zoom the way the 100°-at-256² trap does.
The blocker is corpus coverage on this host, nothing about the geometry.

⚠️ **Correction to the brief: `f_eff` for this configuration is 305.577, not
266.** 266 is `F_REF`, the *deployed canonical 256×256* focal; it is not the
focal of a 120°/256×640 cylindrical frame. (The geometry doc's table lists
`256x640f366.693cyl` for **100°**; 120° at the same width is wider, hence the
smaller focal. Consistent.)

⚠️ **`exceeds_comma2k19_field: true`** — 120° exceeds comma2k19's entire 65.2°
field. The code raises this itself: comma2k19 **cannot supply this frame at any
resolution** and would have to be letterboxed with an explicit unobserved mask,
given its own frame, or dropped from the mix. That is a **PI decision**, and it
lands *before* any wide-FOV training, not after.

### 3.3 ⛔⛔ A LIVE, PROGRAM-BLOCKING BUG — `_decode_mp4` is broken at HEAD **on every path**

The preflight above validates the geometry against a `torch.zeros` probe. It
therefore **structurally cannot** catch a failure in the real mp4 → resample
path, because it never calls `_decode_mp4`. So I decoded real clips. All six
failed:

```
AttributeError: 'av.video.frame.VideoFrame' object has no attribute 'f_ref'
```

**Root cause — a variable-shadowing collision in `stack/tanitad/data/physicalai.py`:**

```python
fr = as_frame(frame, size, F_REF)        # line 516 — fr is the CanonicalFrame
...
for fr in c.decode(stream):              # line 539 — REBINDS fr to a PyAV VideoFrame
    ...
    outs.append(_remap_batch(torch.stack(buf), intr, fr, remap))   # gets the WRONG object
```

Every later use of `fr` — `_remap_batch`, `fr.to_dict()`, `fr.tag()`,
`ftheta_horizon_row(..., frame=fr)` — needs the **geometry**, not a video frame.
Only the loop variable was wrong.

⛔ **Blast radius — MEASURED on pod2, all three paths, before the fix:**

| path | result |
|---|---|
| **DEPLOYED 256×256 `ftheta_crop`** | ⛔ `AttributeError: … has no attribute 'half_angle_x_rad'` |
| wide 256×640 pinhole | ⛔ `AttributeError: … has no attribute 'half_angle_x_rad'` |
| wide 256×640 cylindrical | ⛔ `AttributeError: … has no attribute 'f_ref'` |

**This is not a wide-FOV problem — it is a total corpus-build outage.** At HEAD,
`stack/tanitad/data/physicalai.py` cannot decode a single PhysicalAI clip at any
geometry. The existing `physicalai-train-e438721ae894` cache is fine (it predates
the regression) but **it can no longer be reproduced from source**.

**Attribution — unambiguously `fdc5b4f`, and it is COMMITTED.** In `origin/main`
the function is `_decode_mp4(mp4, size)` and the resampler is called with the
**int** `size`:

```python
outs.append(ftheta_crop_resize(torch.stack(buf), intr, size))   # origin/main
```

so `for fr in c.decode(stream)` was harmless there — `fr` was *only* ever the
VideoFrame. The `fdc5b4f` diff adds the frame plumbing **into** that function
without renaming the loop variable:

```diff
-def _decode_mp4(mp4: Path, size: int) -> Tensor:
+def _decode_mp4(mp4: Path, size: int, frame: CanonicalFrame | None = None,
+    fr = as_frame(frame, size, F_REF)
```

`git status` shows `physicalai.py` clean against HEAD and
`git log -1 -- …/physicalai.py` is `fdc5b4f`. So the break lives entirely inside
the unpushed geometry work — **no pod running `origin/main` is affected**, but
the local repo HEAD and the whole v5 wide-FOV plan are.

**Why no test caught it:** `pytest -k "physicalai or calib or geometry or parity"`
is **195 passed, 1 skipped** *with the bug present*. **No test decodes a real
mp4.** That is the actual gap — the geometry work is heavily tested at the
ray-maths level and untested at the container level.

**FIXED and staged** (renamed the loop variable to `vframe`, plus a comment
saying why the name is reserved). Re-verified on pod2, same clip, all three paths:

| path | after the fix |
|---|---|
| DEPLOYED 256×256 | ✅ `(605, 3, 256, 256)`, tag **`256x256f266pin`**, horizon_row 129.8 |
| wide 256×640 pinhole | ✅ `(605, 3, 256, 640)`, tag `256x640f184.7521pin`, horizon_row 129.5 |
| wide 256×640 cylindrical | ✅ `(605, 3, 256, 640)`, tag **`256x640f305.5775cyl`**, horizon_row 130.2 |

The deployed path reproducing the documented canonical tag **`256x256f266pin`**
(the one that mints `14231cd29c74`) is the evidence that the fix restores the
*correct* behaviour and not merely a non-crashing one. Full suite re-run — see §5.

### 3.4 Real-frame validation — and the masked periphery is a RIG-B effect

With the fix in, 6 clips sampled across the corpus decode cleanly at
120° / 256×640 cylindrical (`raw/wfov_realframe_2026-07-27.json`, **MEASURED**).

| clip idx | `cy` | rig | observed_frac | zero-pixel frac |
|---:|---:|---|---:|---:|
| 0 | 750.4 | B | 0.9146 | 0.0855 |
| 152 | 748.8 | B | 0.9190 | 0.0811 |
| **304** | **549.2** | **A** | **1.0000** | **0.0000** |
| 455 | 757.7 | B | 0.9045 | 0.0955 |
| 607 | 752.0 | B | 0.9089 | 0.0913 |
| 759 | 757.7 | B | 0.9045 | 0.1253 |

⭐ **New finding: the ~9 % unobserved periphery at 120° is entirely a rig-B
effect. Rig A is FULLY observed (1.0000, zero black pixels).** The geometry doc's
single pooled "8.91 % unobserved" figure hides a clean rig split: **rig A 0 %,
rig B 9.0–9.6 %**. This is the *same* rig asymmetry the doc measures for the
deployed crop (11.2 % padded rows on rig B, 0 % on rig A) — the cylindrical path
converts it from **fabricated** pixels to **honestly masked** ones, but it does
**not** remove the asymmetry, and any wide-FOV model will still see a
rig-correlated mask.
⚠️ **n = 6 (5 rig B, 1 rig A)** — the rig-A cell is a single clip. Directionally
strong, but this needs a proper per-rig sample before it is quoted as a rate.

**Build-cost, now grounded in real decodes:** mean **24.8 s/clip** (605 frames,
0.0410 s/frame) ⇒ 2,400 clips at 16 workers ≈ **1.03 h**, corroborating the
geometry doc's independently-derived **~1.2 h**. `f_eff` was **305.577 on every
clip** (stdev 0.0), as it must be — it is fixed by the frame, not the sensor.

### 3.5 Where the build should actually run

The host that holds the full raw corpus. pod2 is not it. Disk is a **secondary**
concern only: `/workspace` on pod2 is at **523 GB** used, a real `dd` write test
passed **20 GB at 407 MB/s** (so headroom exists and `df`'s "243 T avail" is the
usual cluster-wide lie), but 20 GB does not prove the 112.9 GB the cache needs.

---

## 4. 🔴 Escalations — these need an owner, not a note in a file

0. ⛔⛔ **HIGHEST PRIORITY — I fixed a committed regression in another stream's
   file and it needs that stream's review before anyone commits.**
   `stack/tanitad/data/physicalai.py` `_decode_mp4` shadowed the CanonicalFrame
   with the PyAV decode variable, breaking **every** corpus-build path at HEAD
   including the deployed one (§3.3). Fix is **staged, not committed**: 8
   insertions / 2 deletions, loop variable renamed `fr` → `vframe`. Verified on
   real clips on pod2 (all three geometries) and `pytest -q` is
   **1253 passed, 7 skipped**. ⚠️ **The index also contains other agents' work —
   whoever commits must read `git status --short` first and follow the CLAUDE.md
   pathspec/`-F` rule.**
   ⚠️ **Companion gap: no test decodes a real mp4.** The targeted geometry suite
   is 195-green *with the bug present*. A container-level regression test is owed.
1. ⛔ **`origin/main` is missing three pieces of shipped work, so no pod can get
   them via git**: `comma2k19.HEADING_MODE_HOLD` + `hold_heading_through_standstill`,
   `parity.register_geometry_sibling`, `calib.CanonicalFrame`. All are committed
   locally at `fdc5b4f` on `agent/benchmarks-eval-20260721` and **unpushed**.
   Every pod that needs them today needs an scp. **Someone with push rights must
   land this branch** — and it must land *with* escalation 0's fix, because
   `fdc5b4f` as it stands would push a broken decoder to every pod.
2. ⛔ **The wide-FOV cache needs a host with ≥2,400 raw R0 clips.** pod2 has 760.
   Until such a host is identified the v5 wide-FOV schedule has no build step.
3. ⚠️ **120° exceeds comma2k19's entire field (65.2°).** A wide-FOV corpus forces
   a decision about comma2k19 — letterbox with mask, own frame, or drop. PI call.
4. ⚠️ **`RETRACTION_LOG.md` root-cause class to add: `pgrep -f` self-match, watcher
   form** — the guard that never releases, stranding a chained job (§1.2). The
   documented form only covers killing your own session.
5. ⚠️ **The pod3 idm-proof latent cache stores no `episode_id`.** Any future reuse
   is one careless line away from a val leak. It should be re-emitted with
   `episode_id`/`src`, or the fingerprint index in `raw/` should be adopted as
   the identity source of record.
6. ⚠️ **pod3 `/workspace/TanitAD` carries 33 files / 3,680 lines of real local
   edits** not in `origin/main` and not whitespace. Someone should triage whether
   that is work worth rescuing (`pod_git_drift.py` territory).
7. ⭐ **The IDM `steer` head can now be re-shipped** (§2.5) — model card + registry
   re-issue owed. This closes IDM_V3.md §9 escalation 4.
8. 🟢 **Three hosts are free** (pod2, pod3, eval) and pod1 has ~30 h left on
   `flagship-v2corpus-30k`. Obvious next jobs: the 757-episode recipe is now the
   best IDM we have and its **`yaw_rate` +0.9188** also beats v3's shipped
   +0.841, so a full re-ship + re-pseudo-label pass is available immediately, and
   it needs no new data.

---

## 5. Deliverable manifest

Everything below is **staged** (`git add`), never committed, never pushed.

| artifact | where it lives | only one copy? |
|---|---|---|
| `FLEET_REFILL.md` (this file) | `repo:…/incoming/2026-07-27-fleet-refill/` | no |
| **`stack/tanitad/data/physicalai.py`** — ⛔ the `_decode_mp4` fix (§3.3) | `repo:` **staged**, + `pod2:/workspace/wfov/stack/…` | no |
| `idm4_steer.py` — the retrain | `repo:…/2026-07-27-fleet-refill/` **and** `pod3:/workspace/idmretrain/idm4_steer.py` | no |
| `wfov_preflight.py` — geometry/parity gate | `repo:…/2026-07-27-fleet-refill/` **and** `pod2:/workspace/wfov/` | no |
| `wfov_realframe_check.py` — real-frame decode probe | `repo:…/2026-07-27-fleet-refill/` **and** `pod2:/workspace/wfov/` | no |
| `code/dump_lat_index.py`, `code/fingerprint_lat.py` | `repo:…/2026-07-27-fleet-refill/code/` | no |
| `raw/wfov_preflight_2026-07-27.json` | `repo:…` + `pod2:/workspace/wfov/` | no |
| `raw/wfov_realframe_2026-07-27.json` | `repo:…` + `pod2:/workspace/wfov/` | no |
| `raw/idm4_steer.json` — **final, all 4 rungs** | `repo:…` + `pod3:/workspace/idmretrain/out/` | no |
| `raw/idm4.log` | `repo:…` + `pod3:/workspace/idmretrain/out/idm4.log` | no |
| `raw/fp_eval.json`, `raw/fp_pod3.json` — episode fingerprints | `repo:…` | **effectively yes** — pod copies are scratch |
| `raw/lat_overlap.json` — the leak analysis | `repo:…` | **yes** |
| **`idm_head_v4_steer.pt`** — rung 757, seed 0, 11.6 MB | `repo:…` (pulled) + `pod3:/workspace/idmretrain/out/` | no |
| pod3 clean worktree at `origin/main` | `pod3:/workspace/TanitAD-main` | n/a (reproducible) |
| IDM harness + latents relayed eval→pod3 | `pod3:/workspace/idmretrain/` | no (source on eval) |
| geometry stack shipped to pod2 | `pod2:/workspace/wfov/stack/` | no (from repo) |

✅ **Nothing is stranded.** The pod3 job finished before filing and every artifact
— results JSON, log, and the checkpoint — was pulled into the repo and staged.
No job is left running on any host by this iteration.

### Verification run before filing

- `cd stack && pytest -q` → **1253 passed, 7 skipped** (with the `_decode_mp4`
  fix staged).
- Targeted `-k "physicalai or calib or geometry or parity or signals"` →
  **195 passed, 1 skipped** — note this was *also* green **before** the fix,
  which is the test gap in escalation 0.

### Measured operational corrections

- **The dev-box scp relay is ~5 MB/s, not ~1 MB/s.** 358 MB moved
  eval → local in **71 s** and local → pod3 at a comparable rate, md5-verified
  both ways (`d1e5ef0d73b6d09e0008a0eb3eacfb3a`). The "~1 MB/s, use the HF relay"
  note is pessimistic by ~5× for sub-GB payloads; HF relay is still right for
  multi-GB.
- **`tar` created on the Windows box throws `Cannot change ownership to uid …`
  for every member when extracted on a pod and exits non-zero — but the files
  extract correctly.** Use `--no-same-owner`, and do not read the exit code as a
  failed transfer.
