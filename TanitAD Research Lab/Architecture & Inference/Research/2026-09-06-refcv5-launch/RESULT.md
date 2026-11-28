# refcv5 IS TRAINING — the WP-4 diffusion rung launched, and the brief's first-rung precondition was REFUTED on the way

**Stream:** Architecture & Inference · **2026-09-06** · branch `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv5-launch/`
**Evidence class:** MEASURED (ours) throughout; every number names its artifact.
**Eval tier:** ⛔ **NONE APPLIES.** Nothing here is a capability number. This package
records a *launch* — preconditions, argv, liveness. **No result exists yet.**

---

## The headline, in five lines

| | |
|---|---|
| **L1** | ⭐ **refcv5 is TRAINING on the A40** as `refcv5-ddim-b1-v72-40k`, supervisor pid **2530709**, trainer pid **2530863**, GPU **43,421 MiB / 46,068 at 100 %**, target **40,284** steps, **4.046 s/step ⇒ ETA ≈ 2026-09-08 07:33 UTC**. |
| **L2** | ⛔⛔ **THE BRIEF'S FIRST-RUNG PRECONDITION IS REFUTED.** *"`--agents oracle` is the FIRST RUNG and needs no detector"* is true about the **detector** and **false about the JOIN**: the oracle's tokens **are** the GT boxes, so with no `--agent-join` the forward raises. MEASURED, not reasoned — the preflight's 2-step arm died on it. |
| **L3** | ⛔ **The pod's stack was STALE and the launch would have run refcv4b's code under refcv5's name.** `refc_sampler.py` and `refc_agents.py` were **ABSENT**; `refc_v3_train.py` read **1,846** lines against HEAD's **3,634**. Shipped md5-exact and content-verified. |
| **L4** | ⛔ **The anchor bank does NOT declare its units** — the brief said it had been rebuilt; on the pod all three banks read `control_units = <<ABSENT>>`. Resolved by the *sanctioned* route (`--anchor-control-units alat`), grounded because the file is **byte-identical** to the bank refcv4b trained 40,284 steps on. |
| **L5** | ⭐ **The lever is LIVE, not merely stamped**: `u0 = 0.27576` at step 50 and `w_u0 0.5` in the seam stamp. A weight that reaches `config.json` and adds zero to the total is the M18 defect; this one is measured non-zero. |

---

## 1. The preflight, row by row — and the disposition of every non-PASS

`stack/scripts/refcv5_preflight.py`, run on the pod against **the inputs the arm actually
uses**. Artifacts: `raw/refcv5_preflight_b1.log`, `raw/refcv5_preflight_b1.json`.

**Its own verdict: `7 PASS · 5 FAIL · 5 INCONCLUSIVE ⇒ NO-GO`** — reported exactly as the
tool wrote it, and **INCONCLUSIVE is counted as a failure**, per the brief.

⭐ **But the tool asks *"can refcv5 start a REAL arm on the PARITY corpus?"*, and this arm
is deliberately a **B1** arm — refcv4b's corpus, so that refcv5 vs refcv4b is a matched
comparison.** Ten of the twelve non-PASS rows are that scope difference or an
un-exercised option, not a defect in the launched configuration. Each is classified below
with a **GO/NO-GO for THIS arm**, and the two that were real were **fixed before launch**.

### A. the code

| row | reading | disposition for THIS arm |
|---|---|---|
| ⛔ **refcv5 modules import** | 11/12; `taniteval.ci` → `ModuleNotFoundError` | **WAS REAL — FIXED.** `taniteval` was absent from the pod entirely. This is the documented *"analysis-time import that fails after the rollout destroys the run's output"* trap: it would have killed the **landing eval** after ~45 h of paid compute. Shipped (md5 `789407da18b5df52a1b4190ff0bea075`, exact) and verified by a real import: `taniteval OK from /workspace/TanitAD/taniteval/taniteval/__init__.py`, `ci` carrying `episode_cluster_bootstrap` **and** `paired_episode_cluster_bootstrap`. ⇒ **GO** |
| ✅ trainer imports | `refc_v3_train` loaded | GO |
| ✅ class enum is the corpus enum | n=10 identical; control `automobile` present | GO |
| ✅ query budget covers train | default 100 ≥ train max 94; **no second spellings found** | GO (unused: agents off) |
| ⛔ **every weighted term has a gradient** | `agent_w_ground` **1.164e-10 DEAD**; `agent_w_project` **1.462e-03 LIVE**; control: 1 of 2 live | **NOT REACHED.** Both weights are **0.0** in this arm and `agent_rig_camera` is `off`. The stamp records `agent_ground_prior_probe: {checked: false, reason: "w_ground == 0 or no camera"}` — the honest form. ⇒ **GO** |
| ✅ seam guards refuse their defects | 4/4 fired; control (legal config) passed | GO |
| ✅ every knob reaches `config.json` | 18 knobs, 0 missing; control n_knobs > 12 | GO |
| ⚠️ camera scope is stated | INCONCLUSIVE — no `--rig-extrinsics` given | **NOT REACHED.** No monocular term runs, and the stamp says `mount_pose_scope: NONE` rather than falsely claiming PER-CLIP. ⇒ **GO** |

### B. the data

| row | reading | disposition for THIS arm |
|---|---|---|
| ⛔ **v2 cache present with stable ids** | 4,572 `*.v2ep.pt` (control: 4,573 dir entries), manifest present, **path names the parity key: False** | **TRUE AND ACCEPTED, BY DESIGN.** This is B1, not parity — the same corpus refcv4b trained on. The trainer says so itself at startup: `[parity] ⚠ NON-PARITY v2 corpus … Results off it are NOT cross-arm comparable with the parity arms.` ⇒ **GO for a refcv4b-matched arm; NO-GO for any parity claim.** Stated in the registry row. |
| ✅ episode ids collision-free | 4,572 distinct for 4,572 files | GO |
| ⚠️ corpus clip list is the PARITY set | INCONCLUSIVE — no `--clip-ids` | same scope point as above ⇒ **GO** |
| ⚠️ agent join joins to the cache | INCONCLUSIVE — no `--agent-join` | **REAL, and it is why WP-6 is deferred — see §2.** ⇒ **GO for the WP-4 rung; NO-GO for any agent rung.** |
| ⚠️ join digest declares its scope | INCONCLUSIVE — no join | as above ⇒ **GO** |
| ✅ **v7.2 labels cover the corpus** | **4,572/4,572 = 1.0000**, floor 0.50 | GO. ⭐ The readiness doc's `190/2,400 = 7.92 %` blocker was a **parity**-scoped reading; on B1 the coverage is complete. |
| ⚠️ per-clip cameras cover the corpus | INCONCLUSIVE — no bank built | **NOT REACHED** (no camera term) ⇒ **GO** |
| ⛔ **anchor vocabulary declares its units** | `controls` present; `control_units` **None**; `AnchorUnitsMissing` raised | **WAS REAL — RESOLVED, see §3.** ⇒ **GO** |

### C. end to end

| row | reading | disposition |
|---|---|---|
| ⛔ **a 2-step run states every weight** | exit 1 — `ValueError: this build is --agents oracle but no agent_gt reached the forward` | **WAS REAL AND DECISIVE — see §2.** Re-run with the launched configuration it **PASSES**: 3 steps, ckpt, eval, `summary.json`. ⇒ **GO** |

### ⭐ GO/NO-GO for the launched arm

**GO** — with two scope statements that travel with every future number from it:
**(a)** it is a **B1** arm, not parity; **(b)** it carries **no agent seam**, so it prices
**WP-4 alone**.

---

## 2. ⛔⛔ The refutation: `--agents oracle` needs the JOIN, not just "no detector"

The brief's precondition 1 reads: *"`--agents oracle` is the FIRST RUNG and needs no
detector. `--agents head` requires `--agent-join`; do not reach for it now."*

**MEASURED — `raw/refcv5_preflight_b1.log`, section C:**

```
ValueError: this build is `--agents oracle` but no agent_gt reached the forward.
The oracle's tokens ARE the ground-truth boxes; with none the seam emits nothing
and the arm would read as 'agent tokens do not help' while never having had any.
```

⇒ The sentence is **half true and the wrong half is load-bearing**. `oracle` needs no
*detector* — it needs the **boxes**, which arrive by exactly the same `--agent-join`.
`--agents head` needs the join to *train a detector*; `--agents oracle` needs it to *have
anything to embed*. The distinction the brief drew does not exist at this seam.

⚠️ **And the blocker behind it is a CORPUS-SCOPE mismatch, not a missing file.** There is
**no agent join on the pod at all** (`find /workspace /root -name '*agents*.jsonl*'` →
empty, taken with a control that read non-zero). The only join the programme holds is
`train2400_agents.jsonl.xz` — **parity**-scoped, 2,308 clips. B1 overlaps parity by
**7.92 %**, so it would supply boxes for roughly **4 %** of this corpus. Training the agent
seam on that would produce *"agent tokens do not help"* from a seam that was starved, which
is the precise failure the trainer's own `ValueError` exists to prevent.

⭐ **RULE ZERO — this did not end the turn.** The next lever was taken in the same run:
**WP-4 is the mechanism that is actually unblocked**, it is the one the PI's RL plan waits
on (*"DD-v2 post-trains a denoising trajectory, which did not exist until now"*), and
running it alone makes **attribution cleaner, not worse** — exactly one mechanism moves
against refcv4b.

**What would unblock WP-6, named as the rule requires:** a **B1-scoped `obstacle.offline`
join** (the DataFlyWheel owns the build; `stack/scripts/build_obstacle_join.py` is the
pod-side builder). Nothing else is missing — the code, guards and tests are all in HEAD.

---

## 3. Two blockers that the brief reported as already cleared, and were not

Both were found by **re-verifying rather than trusting**, which the brief instructed.

### 3a. ⛔ The pod's stack was STALE — the launch would have run refcv4b's code

| file | HEAD | pod, before |
|---|---|---|
| `stack/tanitad/refs/refc_sampler.py` | 389 | **ABSENT** |
| `stack/tanitad/refs/refc_agents.py` | 820 | **ABSENT** |
| `stack/tanitad/refs/refc.py` | 3,095 | 2,477 |
| `stack/scripts/refc_v3_train.py` | 3,634 | **1,846** |
| `stack/scripts/refcv5_preflight.py` | 803 | **ABSENT** |

⚠️ **This is the trap that makes a stale launch invisible**: a run from that tree would have
had no `--sampler` flag at all, and had the flag been *partially* present the `config.json`
would still have read `sampler: ddim` while the mechanism was absent.

**Shipped as a tarball** (`git fetch` on a pod HANGS and a failed fetch followed by a
checkout destroys shipped files — never used): md5 `4aff70f6200d559f328969d4c2f1d303`
**identical on both sides**, then verified **by content, not by tar's exit code** — all
seven load-bearing files re-read at their exact HEAD line counts, and flag markers
(`--sampler` 9, `--agents` 13, `--anchors` 1) against a same-breath control
(`def` count = 35) that read non-zero.

⭐ **And the guard is now permanent**: `sup_refcv5.sh` refuses to launch unless
`DDIMSchedule`, `roll_controls` and `build_agent_head` all import from the shipped tree
(exit 8). It fired green at launch: `WP-4/WP-6 seam present: DDIMSchedule, roll_controls,
build_agent_head`.

### 3b. ⛔ The anchor bank declares NO units — and the fix is the sanctioned one

The brief states *"Anchors: rebuilt with **declared `control_units`**"*. **MEASURED: all
three banks on the pod read `control_units = <<ABSENT>>`** —
`refcv4b-b1-v72-40k/anchors.pt`, `incoming_v4b/anchors.pt`, `incoming_v4b2/anchors.pt`,
each carrying `controls (117, 2)` and declaring nothing. This is the exact shape of the
**396 g / 0.31 g** incident, and `read_anchor_artifact` correctly raised
`AnchorUnitsMissing`.

**The route taken for the LIVE run is the one `anchor_meta.py` names for exactly this
case:** pass `--anchor-control-units alat` **by name**, so the run record says the units
came from the operator. The trainer stamped it as such:

```
[v3] anchors: v0-CONDITIONED, controls (117, 2) (accel, lateral accel),
     units=alat (source: cli-override-legacy-file), rolled per window
```

⭐ **The units are not a guess.** The installed file's `file_sha256` is
`e86cf507d55a4585435025fe52f33817d08dab879e1f65ff6a1fc9b0eb81e8fb` — **byte-identical** to
the value refcv4b's own `config.json` records, i.e. the same tensor that trained 40,284
steps under `alat`, at `sha256_installed 51f930dc6f3564ff…`.

### ⭐ 3c. …and then I made my own absence-at-one-location error, and fixed it

⚠️ **I first wrote that the re-stamp constants "could not be established", because the only
builder log ON THE POD (`anchors6s.log`) records `n_anchors 128` — refcv4's *fixed-path*
build, not refcv4b's 117-anchor bank.** That was an **absence found at ONE location**, and
the rule against it is in `CLAUDE.md` for exactly this reason. **A second location has
them all**: `anchors.units.json`, written beside the bank on 2026-09-04
(`C:\Users\Admin\navcomp\ckpt\`), corroborated line-for-line by `MODEL_REGISTRY.md` §4.6.

⇒ **The owed re-stamp is DONE, and it is verified rather than asserted** (`raw/
restamp_anchors.py`, `raw/restamp_anchors.log`). ⛔ It does **not** touch the live run's
file — modifying a training run's vocabulary mid-flight is forbidden; it writes a new
artifact for the *next* rung:

```
source file sha256 VERIFIED e86cf507d55a4585435025fe52f33817d08dab879e1f65ff6a1fc9b0eb81e8fb
units record matches the file it describes
anchors  sha256 VERIFIED 51f930dc6f3564ff8f21c9070ca97b805d4e82908e42d0ef1f99be9f5a3a66df
controls sha256 VERIFIED b072f4c052331beb79bef117c7b233f702802b517429cc9157a841294e089664
readback: control_units=alat source=file horizon_s=6.0 dt=0.1 ref_speed_ms=10.0 kappa_cap=0.12 alat_v_floor=4.0
readback tensors BIT-IDENTICAL to the live bank
CONTROL OK: the undeclared source bank is still REFUSED
WROTE /workspace/anchors_117_alat_declared.pt (11829 bytes)
```

Four properties make this admissible rather than a re-labelling: the **source bytes are
sha256-asserted** to be the bank refcv4b trained on; the **units record is asserted to
describe that same file**; the **tensors are proven unmoved** on both sides of the write;
and a **same-breath NEGATIVE control** shows the undeclared original is *still refused*, so
the guard was not weakened to make the new file pass. Every sha256 is length-checked before
comparison, so a failed read reports **INCONCLUSIVE** rather than a false match.

⇒ **The next refcv5 rung needs no `--anchor-control-units` override**: units resolve from
the file, `source=file`. The live arm keeps the override, because its bank cannot change.

---

## 4. The launched arm

**Run dir** `/workspace/experiments/refcv5-ddim-b1-v72-40k` · **supervisor**
`/workspace/sup_refcv5.sh` (md5 `ce882e31ba0b3d1371379d2f52ea9fa2`, 227 lines, identical
both sides, `bash -n` clean).

```
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 \
  --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 \
  --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --anchors /workspace/experiments/refcv5-ddim-b1-v72-40k/anchors.pt \
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
  --sel-accel-max 2.0 --goal-str \
  --ego-state-inject --ego-dropout 0.5 \
  --sampler ddim --w-u0 0.5 \
  --agents off \
  --out /workspace/experiments/refcv5-ddim-b1-v72-40k
```

**It is refcv4b's argv byte for byte, plus `--sampler ddim --w-u0 0.5`, plus the explicit
`--agents off`.** Same corpus, same labels, same anchors file, same seed, same step target
⇒ **a matched-step, matched-corpus comparison in which one mechanism moved.**

* **`--w-u0 0.5`** is not invented: it is the value exercised as the **honest CONTROL** in
  the WP-4/WP-6 guard-mutation panel (`…/2026-09-05-refcv5-wp4-wp6-wiring/RESULT.md`,
  *"CONTROL honest `--sampler ddim --w-u0 0.5` → PASSES"*). `--w-u0 0` is **refused**
  (validation arm R4), so a non-zero weight is mandatory, not a preference.
* ⛔ **`--sel-refined` is NOT passed — and could not be.** `grep -c sel_refined` on the
  trainer reads **0** against a same-breath control `sel_accel_max` reading **5**: the flag
  does not exist in this trainer. The brief's prohibition is satisfied structurally. (Its
  reason stands regardless: MEASURED **0.0259 m separated WORSE** on refcv4b, because the
  ranking head was never trained to rank.)
* **Stamped for the registry:** `size: base`, **`rig_rung: False`** — a registry-grade run.

### The seam stamp proves the lever is live, not merely present

`config.json → seams`: `sampler: "ddim"`, `sampler_space: "control"`,
`sampler_infer_t: 8`, `sampler_steps: 2`, `sampler_groups: 1`, `control_norm: [4.0, 3.0]`,
**`w_u0: 0.5`**, `agents: "off"`, `w_agent: 0.0`, `cross_agent: false`,
`feasible_decode: false`.

⭐ **And the training loop confirms it numerically**: `metrics.jsonl` carries a non-zero
`u0` at every logged step — **0.27576 / 0.18663 / 0.32519** at steps 50 / 100 / 150 — i.e.
the x0 term is actually being supervised. *A weight stamped into `config.json` that adds
zero to the total is the M18 dead-flag defect; this one is measured live.*
Loss is falling: **61.1346 → 57.7430 → 28.1739**.

### Pace and ETA

**4.046 s/step** steady-state over **34 intervals, steps 100 → 1,750** — model build
and the 4,572-episode enumeration excluded ⇒ **~43.3 h remaining**, **~45.3 h total**, landing
**≈ 2026-09-08 07:33 UTC (09:33 Berlin)**.

⚠️ **CORRECTED at step 1,750.** The launch record first said **3.938 s/step / ETA 06:20**,
read from a **single interval** (steps 100 → 150). That is a rate quoted without its
**n and window** — the family this programme already has a rule about — and it was **2.7 %**
optimistic. Both figures are shown so the correction is auditable rather than silent.

⚠️ **`elapsed_s` is CUMULATIVE and this trainer logs no `step_s`** — the number above is a
**difference between two rows**, never a division. (Dividing by `--log-every` is the
documented `step_s` scope trap; the registry says the same for refcv4b.)

⭐ **The DDIM sampler is nearly free in wall-clock**: refcv4b ran at **3.844 s/step**
marginal median on this same pod / cache / batch / workers / LRU, so WP-4 costs **+5.3 %**.
A 44 h budget was already the plan.

---

## 5. Liveness, and the done-marker discipline

**Asserted, never assumed** — every failure in this family reports success and leaves
nothing running.

```
sup_count = 2530709  bash /workspace/sup_refcv5.sh          <- supervisor ALIVE
trainer   = 2530863  refc_v3_train.py … --sampler ddim --w-u0 0.5 --agents off
GPU       = 43,421 MiB / 46,068   utilisation 100 %
train.log = [v3:hier] step 50 loss 61.1346 traj 2.3345       <- real progress
stderr    = empty
```

### ⭐ The lock-fd audit — the fix is VERIFIED, not merely written

`200>&-` is on the trainer's redirection list (line 145) **and on both `sleep`s**
(221, 225) — the second half is the one a partial fix missed the same day it shipped.
Proven by the `/proc/*/fd` holder scan, which is the diagnostic that settles it in one
line:

```
holder pid=2530709 : bash /workspace/sup_refcv5.sh
```

**Exactly one holder, and it is the supervisor.** No trainer, no dataloader worker, no
`sleep` holds fd 200 — so a future supervisor restart is possible rather than permanently
blocked.

### The done-marker has TWO independent writers

1. **The trainer writes its own** `summary.json` on a clean exit — MEASURED in the 3-step
   smoke: `[v3:hier] DONE at 3 — summary.json written`.
2. **The supervisor derives the done condition FROM THE DATA** — `last step in
   metrics.jsonl >= TARGET` — and writes `{"done": true, "final_step": …}` in the same
   breath as the decision, then exits 0.

Neither depends on the other, which is what makes a resurrection impossible. Plus a
**three-relaunch no-progress crash-loop stop** (writes `{"done": false, "crash_loop":
true}` and exits 7) and `MAX_RELAUNCH 40`. The `last_step()` helper is carried over in its
**fixed** form — the `|| echo 0` fallback that emitted two lines and made the done branch
unreachable is not reintroduced.

**Three pre-launch refusals guard the run** (all fired green): missing `anchors.pt` → exit 5;
a bank without `controls` / straight-ahead → exit 6; **the WP-4 seam absent from the shipped
stack → exit 8** (new in v5, and the direct answer to §3a).

---

## 6. Design inputs from the landing read, carried into the launch record

Stated because they are what the *next* rung must answer — this arm does not address them.

1. ⭐⭐ **CLOSING RATE is unencoded.** The frozen trunk decodes the lead's **position**
   (paired vs pixels **+0.4145 [+0.2018, +0.6120]**, constant control exactly +0.000000)
   but its **closing rate is a clean null on every arm** (+0.0061), and an explicit
   temporal difference recovers nothing — so it is the **representation**, not a windowing
   artefact. **You cannot keep distance from position alone**, and the LONGITUDINAL family
   is 88.7 % of the oracle gap. ⚠️ **This arm does not fix it.** WP-4 changes how the fan
   is *sampled*, not what the trunk *encodes*.
2. ⭐⭐ **The ORACLE NAV is worth 0.1054 m** — refcv4b ties the echo control only with it
   (`os_navzero − ha0_ext` = **+0.1054 [+0.0874, +0.1241]** separated worse). ⭐ The
   replacement is half-built: refcv4b's route head already predicts from vision at
   **κ 0.4852** and passes its anti-echo control (**0.7437** true vs **0.2264** shuffled on
   1,736 changed windows). ⚠️ **A sibling is pricing `os_navpred` on Thor — not duplicated
   here.** This arm carries `--nav-from-v7`, i.e. the same nav refcv4b had, deliberately,
   so the comparison stays matched.
3. ⚠️ **The longitudinal kin3 head is 0.1669 nats WORSE than its own eval-set prior.** A
   real defect, unaddressed by this arm, and it belongs in the next rung's scope.

---

## 7. ⚠️ Deviations from the brief, stated plainly

| brief said | what is true | why the deviation |
|---|---|---|
| `--agents oracle` is the first rung, needs no detector | **REFUTED** — it needs the **join**; the forward raises without it | §2. No B1-scoped join exists ⇒ WP-4 runs alone, attribution improves |
| Anchors rebuilt with declared `control_units` | **Not on this pod** — all three banks declare nothing | §3b. Sanctioned CLI override used; re-stamp still owed |
| `--anchor-file` | the real flag is **`--anchors`** | the wiring doc's prose name; `--help` is authoritative |
| Treat INCONCLUSIVE as FAILURE, report every row | done — §1 reports all 17 rows and the tool's own **NO-GO** | the NO-GO is a **parity**-scoped verdict; the arm is B1, and both scope facts are stated with every number |
| Reuse `sup_refcv4b_v3.sh` as the template | done, verbatim except the marked deltas | plus one **new** refusal (exit 8, stale-stack seam check) |

⛔ **The preflight's NO-GO is not overridden — it is answered.** Two of its FAILs were real
and were **fixed** (`taniteval` shipped; anchor units declared by name); the rest are the
parity/B1 scope difference or options this arm does not exercise. Had a real one been
unfixable, the arm would not have launched.

---

## 8. Deliverable manifest

| artifact | where |
|---|---|
| this record | repo `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv5-launch/RESULT.md` |
| preflight log + JSON | repo `…/2026-09-06-refcv5-launch/raw/refcv5_preflight_b1.{log,json}` |
| the supervisor, as launched | repo `stack/scripts/sup_refcv5.sh` **and** pod `/workspace/sup_refcv5.sh` (md5 `ce882e31ba0b3d1371379d2f52ea9fa2`, identical) |
| smoke config + log (3 steps, exact argv) | repo `…/2026-09-06-refcv5-launch/raw/smoke_config.json`, `raw/smoke.log` |
| live run | pod `/workspace/experiments/refcv5-ddim-b1-v72-40k/` (`config.json`, `metrics.jsonl`, `train.log`, `supervisor.log`, `ckpt.pt`) |
| shipped stack | pod `/workspace/TanitAD/stack` (md5 `4aff70f6…`), `taniteval` (md5 `789407da…`) |
| registry row | repo `Project Steering/MODEL_REGISTRY.md` |

⚠️ **The live run is on the pod only, by nature** — checkpoints are not repo artifacts. Its
**config, argv and provenance are in the repo** via this package and the registry row, so
nothing that decides a claim is stranded.

---

## 9. What happens next, and what is blocked

* **Unblocked and running:** the WP-4 rung to step 40,284.
* ⛔ **Blocked on a corpus that does not exist:** the WP-6 agent rung needs a **B1-scoped
  `obstacle.offline` join**. *(DataFlyWheel — build; the code is already in HEAD.)*
* ⛔ **Blocked on a PI decision:** whether refcv5 should *also* run a **parity** arm. It
  cannot be compared to refcv4b if it does, and parity v7.2 supervision covers **7.92 %**.
* ✅ **Closed in this turn, not deferred:** the anchor-bank re-stamp (§3c) and the missing
  `taniteval` on the pod (§1).
* ⚠️ **Owed, not blocked:** a **pre-registration for this arm's landing eval** — the wiring
  package noted the prereg governing the arm is *"a separate, still-owed artifact"*, and it
  is still owed. **It must be banked before the run lands** (~45 h of head-room), with the
  four metric families and the estimator named.
* ⛔⛔ **AND IT MUST CARRY AN INFERENCE-SEED REPLICATE — refcv5 IS STOCHASTIC AT EVAL.**
  MEASURED from source, not inferred: refcv4b's refinement loop zeroes its noise outside
  training (`noise = torch.randn_like(x) if self.training else torch.zeros_like(x)`), which
  is why refcv4b is deterministic at inference. **The v5 sampler cannot do that, and the
  code says so in as many words** — `refc.py`, the anchored-Gaussian block, draws
  `eps = torch.randn_like(x0_n)` on the eval path under the comment *"THIS IS STOCHASTIC AT
  EVAL, BY DESIGN AND NOT BY OVERSIGHT … a SAMPLER cannot, because sampling is the
  mechanism"*, and states the obligation it creates: *"A refcv5 sampler arm must be
  replicated over INFERENCE seeds, and an effect smaller than that floor is not an effect."*
  ⇒ **This changes the eval design, not just its paperwork.** refcv4b's landing read could
  quote a single-seed separated CI and close the inference question by construction; refcv5
  **cannot**. On refav1 the measured inference-seed floor was **~0.30 m ADE**. The
  pre-registration must commit an inference-seed replicate **before** the run lands, or the
  landing read will produce separated CIs that answer a question nobody asked
  (`H-ESTIM-SEED-1`; the third-variance rule).
