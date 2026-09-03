# RESULT — refcv3 now has an eval instrument, and its T1 definition is written down

**Owner:** Benchmarks & Evals FlyWheel · **date:** 2026-09-03 · **compute:** 0 GPU (dev-box CPU;
`nvidia-smi --query-compute-apps` showed no training process, and none was added) ·
**pre-registration:** `SPEC.md` in this directory, written before any code ran ·
**tier of every number in this file: INSTRUMENT CONTROL on a random-init model — NOT a refcv3 result.**

> ⛔⛔ **ESCALATE INTEGRATION — four items, none fixable by a note in a doc:**
> 1. ⭐ **`taniteval/tools/REFCV3_ARM.md` §2 IS THE DELIVERABLE THE MASTER MIND MUST REVIEW BEFORE
>    ANY refcv3 NUMBER IS QUOTED.** It is the T1 definition the killed agent never wrote, derived
>    with file:line. GATE 2 of `T1_CHECKLIST.md` is blocked on a human reading it, not on GPU.
> 2. ⛔ **THE TIER RULING IS STILL OPEN (BACKLOG R30).** The instrument stamps `os` as `T1` with
>    `status: UNRULED` on every block and frames every headline as a **margin over `ha0`**, so the
>    numbers are produced and correctly labelled either way — but **register decision 9 is waiting on
>    this ruling, not on compute**.
> 3. ⭐ **A NEW INSTRUMENT WAS REQUIRED AND IS NOT IN THE CHECKLIST: the SELECTION PROFILE.**
>    MEASURED on this package's own fixture — a random-init RefCV3 selected **one anchor on 42/42
>    windows** while the trivial profile read `trivial_frac = 0.0000` on that same arm. **The
>    trivial-profile gate, which the whole T1 runbook is built around, is BLIND to refcv3's
>    characteristic degeneracy.** `T1_CHECKLIST.md` GATE 6 needs a second bullet.
> 4. ⚠️ **BACKLOG R13 HAS LANDED and the checklist is now stale on it.** `t1_eval.DEFAULT_TIERS`
>    **does** carry `"ha0": "T1"` (`t1_eval.py:154`), verified by two differently-bound probes.
>    `T1_CHECKLIST.md` GATE-level defect #2 and `RESULT.md` §W4 in the epoch-conclusions package
>    still say it does not.

---

## 1. Headline

**refcv3 has an eval instrument.** `taniteval/tools/refcv3_arm.py` (1,890 lines) writes a
`t1_eval`-compatible dump for the arms `os` / `os_navshuf` / `ha` / `ha0` (+ opt-in `oracle_sel`),
reports the four binding metric families with `n` and episode-cluster-bootstrap CIs, prints **two**
degeneracy gates before any family row, and carries the units bridge, the tier stamps and the open
ruling. `stack/tests/test_refcv3_arm.py`: **18 passed**. The dump is also read by the **untouched**
`t1_eval.py --analyze-only` CLI, exit 0.

**The decision, made first and justified (SPEC §1): RESTART from `refav1_arm.py`, harvesting the
draft's engineering.** MEASURED by two differently-bound probes over the 1,746-line rescued draft
(`grep -c` through Bash, then `Select-String -LiteralPath` through PowerShell — both agree): **0**
occurrences of `ha0`, `trivial_profile`, `action_units`, `sel_score_v3`, `a_star`, `oracle_sel` and
`STEER_WHEELBASE`; **15** occurrences of `cl` and **3** of `roll_closed`. Its spine is the claim
`D-HF-COMPARABILITY` was written to forbid — it names the deployed arm `cl` and its docstring calls
it a *"Mirror of `t1_eval.roll_closed` … the loop collapses to the single tick"*. Finishing it meant
deleting its spine and adding five instruments; restarting inherits all five **by import** from
`refav1_arm.py` and keeps one convention in the programme. Six pieces of the draft's real work were
**harvested and credited at their definitions** (§4.3).

**The definition, in one sentence (REFCV3_ARM.md §2.7):**

> refcv3's `os` arm is **one forward pass** of `RefCV3Model` at the window origin, consuming the
> observed frames, the clip's v7.2 nav token and the measured `v0` at t0 **and nothing else**,
> emitting the whole 6 s path as the model's **own** `sel_score_v3`-ranked choice among its 128
> anchors; scored on the index-selected grid against the trainer's own waypoint targets. It is a
> one-shot planning-free trajectory prediction — **there is no action loop to close**, its tier is an
> open ruling, and it may only be compared to refav1 through each arm's **margin over the shared
> `ha0` floor**.

---

## 2. Why `roll_closed` cannot be ported — from source, not from the register

| fact | file:line | consequence |
|---|---|---|
| `forward(frames, nav_cmd, v0, steps, lan, nav_known)` — **no action argument** | `refc_v3.py:480` | there is no action to feed back |
| 128 anchors × 8 slots at `(5,10,15,20,30,40,50,60)` × 0.1 s | `refc_v3.py:196`, `:197`, `:106` | the whole 6 s path comes out of **one** forward |
| `t1_eval.roll_closed` feeds `(steer = atan(L·κ), a_j)` back per step | `t1_eval.py:760` | it needs an autoregressive predictor with a per-step readout; the flagship is supervised **and** autoregressive, which is the difference |
| `v0 = pose_last[:, 3]` | `refc_v3_train.py:445` | the only kinematic input, admissible under the PI ruling of 2026-09-02 |
| `traj_tgt = refb_labels.waypoint_targets(pose_last, fut_ext, horizons)` | `refc_v3_train.py:452` | the GT this tool scores against is the GT the model was trained against |

⚠️ **Two line-number corrections, stated rather than propagated.** `T1_CHECKLIST.md` cites
`physicalai.py:620` and `t1_eval.py:753`; MEASURED here they are **`physicalai.py:621`** (the
`steer = np.arctan(L·curv)` statement; the column-stack that writes it into `actions` is `:632`) and
**`t1_eval.py:760`**. Same statements, one/seven lines of drift.

### 2.1 ⛔ The selection gate — `sel_score_v3`, never `a_star`

The deployed path is `out["traj"]` and nothing else: `rank = sel_score_v3` (`= apply_seam_clamp(
sel_score, goal_gate·score)`, banked at `refc_v3.py:527`) masked by `reach_keep`, then `argmax`
(`:518–521`), assigned at `:525`; on the flat arm the core does the same with its own `sel_score`
(`refc.py:1531–1534`). The training loss scores something **different**: `a_star = dist.argmin(dim=1)`
is the anchor nearest the **ground truth** (`refc_v3_train.py:460`) and `recon = out["anchor_traj"]
[ar, a_star]` (`:463`) is what `loss_traj` measures — a **mean L1 per coordinate**, not an L2 ADE
(`:464–465`, `denom = sv.sum()*2`). The tool therefore (a) takes `out["traj"]`, (b) recomputes ADE as
an L2 norm from the dumped path, and (c) offers the oracle path only as the opt-in **T0 `oracle_sel`
ceiling**, never as the deployed arm.

---

## 3. MEASURED — the instrument controls (random-init model; ⛔ NOT refcv3 evidence)

**Fixture:** random-init `RefCV3Model` at `refc_v3_smoke_config(hier=True)`, encoder widened to the
corpus's 9 channels at 64 px; synthetic 3-clip v2 cache (40 raw frames/clip, JPEG, `n_stack = 3`,
so provider→RAW offset 2) with a 3-record v7.2 label blob covering left / follow / right.
**42 windows, 3 episodes, 5 arms, grid `2s` (dt 0.5 s, K 4, model slots 5/10/15/20).**

Every row below is a **control that had to read a known value**. That, and only that, is what these
numbers are for.

| control | read | the value it had to read | ✓ |
|---|---|---|---|
| `ha0` straight_frac / const_speed_frac | **1.0000 / 1.0000** (42/42) | a zero-control unicycle rollout is a straight line at constant speed | ✓ |
| `ha0` chord per step vs `v0·dt` | max \|Δ\| **< 1e-3 m** | the integrator advances on `v0` and never changes it | ✓ |
| trivial profile, `ha0` | `trivial_frac` **1.0000** | the instrument must see the floor as the floor | ✓ |
| trivial profile, `os` | `trivial_frac` **0.0000** | a random-init anchor model bends — it is not the CV plan | ✓ |
| `os` ≡ `os_navshuf` | **17 / 42** windows | exactly the 42 − 25 windows whose nav token the permutation left alone | ✓ |
| ADE `os` | 9.0374 [7.1672, 10.9010] | — | |
| ADE `ha` | 1.2134 [0.9640, 1.3550] | — | |
| ADE `ha0` | 1.0804 [1.0142, 1.1384] | — | |
| **paired `os − ha0`** | **+7.9571 [6.1530, 9.7625], separated** | a random-init model must **LOSE** to the trivial floor | ✓ |
| paired `os − os_navshuf` | **−0.0002 [−0.0024, 0.0016], not separated** | untrained weights carry no nav dependence | ✓ |
| **paired `ha − ha0`** | **+0.1330 [−0.0502, 0.2326], not separated** | ⭐ **the whole reason `ha0` exists**: holding a noisy observed steer is not better than doing nothing, so a win over `ha` alone is not skill | ✓ |
| `anchor_acc` | **0.0000** (chance 1/20 = 0.05) | untrained logits do not find the GT-nearest anchor | ✓ |
| future-only perturbation | `os` / `ha` / `ha0` move by **exactly 0.0**; `g` moves **> 1 m** | nothing after t0 reaches the arm — and the perturbation provably landed | ✓ |
| deliberate regression (`os := ha0`) | `identical_to.ha0` **42/42, frac 1.0**, paired ADE delta **0.0** | two arms that are equal are reported as equal, and the read is **VOID** | ✓ |
| unstamped arm in the dump | `SystemExit`, *"no T0/T1 tier stamp"* | the tier guard is not worked around | ✓ |
| unit conversion applied **once** | steer-reading == a hold of `tan(steer)/L_enc` under the kappa convention, to **< 1e-6**; a double application differs by **> 1e-3** | the `C-REFCV3-ARM-SAME-DEFECT` bridge is crossed exactly once | ✓ |
| untouched `t1_eval.py --analyze-only` | **exit 0** on the same dump | the dump really is the `t1_eval` contract | ✓ |

`stack/tests/test_refcv3_arm.py`: **18 passed** (`raw/pytest_refcv3_arm.log`).
`stack/tests/test_refav1_arm.py` + `test_refav1_kin_contract.py` +
`test_steer_curvature_interface.py`: **63 passed** — nothing was changed there, and the regression
run says so.

### 3.1 ⭐ The finding: the trivial profile is BLIND to refcv3's characteristic degeneracy

MEASURED, on the run above:

```
[trivial-profile] 42 windows
  os   n=42 straight=0.0000 const_speed=0.0000 CONSTANT-VELOCITY=0.0000  identical_to: os_navshuf=17/42
  ha0  n=42 straight=1.0000 const_speed=1.0000 CONSTANT-VELOCITY=1.0000
[selection-profile] 42 windows over 20 anchors
  n_distinct=1  modal=#2 (1.0000)  entropy=0.0000/2.9957 nats  agrees_with_oracle=0.0
  ⛔ VOID-RISK: the selection is a CONSTANT on 100.00% of windows — the arm is 'always anchor #2,
     refined'. Read every family row below as a property of that one anchor, NEVER as scene
     understanding. The trivial profile CANNOT see this.
```

**The `os` arm was maximally degenerate — one anchor for every scene — and the trivial profile read
`0.0000`,** because a constant anchor is neither straight nor constant-speed. A family table over
that arm would have been read as scene understanding. ⇒ The tool adds a **selection profile** beside
the trivial profile, printed before any family row, and `T1_CHECKLIST.md` GATE 6 should gain it.

⚠️ **A second, smaller instrument fix, also from watching it run:** `ha0` is *by definition* the
constant-velocity plan and therefore always appears in `degenerate_arms`. Escalating VOID-RISK on it
would make the loudest warning in the tool fire on every correct run and be learned as noise, so the
escalation now names `degenerate_arms_excluding_floor` (and separately flags `os` being bit-identical
to a non-floor arm on > 50 % of windows). `ha0` still appears in `degenerate_arms`; only the alarm
changed.

---

## 4. What the instrument does, and the four things it refuses

### 4.1 The arms, and the two names that must never appear

`os` (T1, **UNRULED**) · `os_navshuf` (T1, UNRULED) · `ha` (T1) · `ha0` (T1) ·
`oracle_sel` (T0, opt-in).
⛔ **No `cl` column** — a shared name is how two different procedures end up read as levels of one
quantity. ⛔ **No `ol` column**: refcv3 consumes no recorded actions, so "the recorded future
integrated from `v0`" is a property of the corpus, not a rollout of this model. It is written into
the dump manifest **and** the record as `ABSENT` with its structural reason and a pointer to `ha0`
as the shared floor.

### 4.2 The four families, and the states a missing one may take

| family | status on the fixture | note |
|---|---|---|
| ADE / LONGITUDINAL / LATERAL | **PRESENT** with `n` and CIs (`_ci_coverage.complete = True`) | speed / along-track / accel; cross-track / heading / curvature / yaw-rate |
| TACTICAL | **PRESENT** — trajectory-derived (`status: OK`) **plus** the declared heads and `anchor_acc` against chance `1/n_anchors` | the goal/anchor-selection half lives in `rec['refcv3']['tactical_declared']['anchor_selection']` |
| STRATEGIC | the trajectory-only row is **UNAVAILABLE by design** (a route class cannot be read off a 2 s path); the real one is the route head under **three** nav conditionings with the echo index and the changed subset | ⛔ inadmissible without `os_navshuf` beside it |
| LONGITUDINAL distance-keeping | **REFUSED** with `"WORK ITEM"` in the reason (no lead block on the fixture) | with `--lead-block` it is computed on the **common-grid view** — see §5 |
| `law` | **REFUSED by name**, tier T0 | consumes a **future** frame's latent; no refav1 analogue; can never enter a T1 row |

Two distinct shapes are used and both carry a reason and an `n`: **REFUSED/UNAVAILABLE** (the metric
exists for this model, its inputs are missing here) and **ABSENT** (the arm does not exist for this
model at all).

### 4.3 Harvested from the rescued draft, credited at each definition

The config rebuild through the trainer's own `build_parser` + `_pin_trainer_cfg` (⚠️
`refc_v3_train.py:1097` writes **no** model config — only `argv`, arm, horizons, image_hw, vocab,
nav — and the checkpoint carries only `model`/`opt`/`step`, so this is the *only* recoverable path);
`cross_check_config`'s refusal-naming-both-values; the **seam-clamp fail-loud neutralisation**
(`apply_seam_clamp` raises after N *consecutive* saturated calls and the counter lives on the
**model**, so a trained gate above the clamp would kill a 20k-window eval at window 50 — a
training-dynamics report firing inside an eval); `grid_slots`' index-select-only rule; the
`V3Dataset` subclass that skips the unused future-frame decode; and the **common-grid** lead-block
view. Each is marked *(harvested from the rescued draft)* in `refcv3_arm.py`.

---

## 5. The lead-block join, and its one honest limitation

The banked B1 EVAL block is one row per `(clip_id, RAW 10 Hz frame)` on a **0.2 s / K = 10** grid.
refcv3's grid is **not** a subset of it, so the join index-selects **both sides** onto the shared
instants — `{1.0, 2.0} s` for `--grid 2s` — with no resampling of the lead track and no interpolation
of the path, through `refav1_arm.join_lead_block`'s own guards (exact grid, the **label-free speed
proof** `dump v0 == block speed at the joined row`, NO_LABEL never scored as free flow).

⚠️ **The limitation, printed in the record rather than buried:** `join_lead_block` maps `t → 2t`, so
it reaches only **even** raw frames. refcv3's window origin is a provider frame index and
`RAW = provider + (n_stack − 1)` (`v2_dataset.py:36–38`, read from the cache manifest's own
`n_stack`), so an origin landing on an **odd** raw frame joins to `frame − 1`. The count is emitted
as `_odd_raw_frames` and the fix — a frame-identity mode on `join_lead_block` — is a **WORK ITEM**.
⛔ If the offset were wrong the **speed proof fails loudly per episode** (`SPEED_MISMATCH`) rather
than placing another clip's traffic on these windows; that is why it is a guard and not a comment.
⚠️ And a min-over-2-instants is coarser than the block's own 10-step min: rebuilding on refcv3's grid
(`build_lead_block_b1.py --dt 0.5 --k 4`) is the durable fix.

---

## 6. Work items this package created

| # | item | why |
|---|---|---|
| **W1** | ⭐ **`T1_CHECKLIST.md` GATE 6 needs the SELECTION PROFILE beside the trivial profile.** | MEASURED §3.1: the trivial profile is blind to a 100 %-constant anchor selection. |
| **W2** | **`T1_CHECKLIST.md` GATE-level defect #2 and the epoch package's W4 are STALE:** `t1_eval.DEFAULT_TIERS` already has `"ha0": "T1"` (`t1_eval.py:154`). | BACKLOG R13 landed; a stale blocker line is the `2 of 36` failure class. |
| **W3** | Two line-number citations drift (`physicalai.py:620→621`, `t1_eval.py:753→760`). | Cheap to fix; a wrong line number in a runbook costs a reader minutes at the worst moment. |
| **W4** | `join_lead_block` needs a **frame-identity** mode (§5). | Otherwise odd-raw-frame origins join to `frame − 1`. |
| **W5** | Rebuild the B1 lead block on refcv3's own grid (`--dt 0.5 --k 4`). | Makes the full 2 s horizon scoreable instead of `{1.0, 2.0} s`. |
| **W6** | ⛔ **The tier ruling (BACKLOG R30).** | A definition, not compute. Register decision 9 waits on it. |
| **W7** | `rebuild_config`'s `argv` path is **UNVERIFIED on a real `config.json`**. | Run step 1 of `REFCV3_ARM.md` §3.1 first; a `SystemExit` there is version skew, not a model problem. |

---

## 7. PROPOSED register row

Status stays **PROPOSED** until the Master Mind / PI reads it. Apply verbatim or amend.

### `D-REFCV3-ARM`

> ✅ **D-REFCV3-ARM — refcv3 HAS AN EVAL INSTRUMENT, ITS T1 DEFINITION IS WRITTEN DOWN, AND THE
> TRIVIAL-PROFILE GATE ALONE IS NOT SUFFICIENT FOR AN ANCHORED ONE-SHOT MODEL (MEASURED 2026-09-03,
> Benchmarks & Evals FlyWheel, 0 GPU; `taniteval/tools/refcv3_arm.py` +
> `taniteval/tools/REFCV3_ARM.md`; `Benchmarks & Evals/Research/2026-09-03-refcv3-arm/`).**
> `refcv3_arm.py` writes a `t1_eval`-compatible dump for `os` / `os_navshuf` / `ha` / `ha0`
> (+ opt-in T0 `oracle_sel`), reports the four binding families with `n` and paired
> episode-cluster-bootstrap intervals, and reuses `t1_eval.analyze`, `refav1_arm.trivial_profile`,
> the lead-block join and the unicycle action-unit bridge **by import** rather than by copy.
> **THE DEFINITION (the deliverable the killed agent never wrote):** the arm is ONE forward pass at
> t0 consuming observed frames + the v7.2 nav token + the measured `v0` and nothing else, with the
> path taken from the model's OWN `sel_score_v3` selection `out["traj"]` (`refc_v3.py:518-527`; the
> core's `sel_score` at `refc.py:1531-1534` on the flat arm) — ⛔ **NEVER `a_star`
> (`refc_v3_train.py:460`), which is the GT-nearest anchor and makes `eval_traj` an oracle-selected
> lower bound in a mean-L1-per-coordinate metric, not an ADE.** `t1_eval.roll_closed`
> (`t1_eval.py:760`) CANNOT be ported: `refc_v3.py:480`'s forward signature has **no action
> argument**, so there is no action to feed back. ⇒ the arm is named **`os`, never `cl`**; **`ol` is
> ABSENT with its structural reason**; **`ha0` is the only bit-comparable arm** and the admissible
> cross-model claim is each arm's **margin over that shared floor, per family, paired**.
> ⚠️ **THE TIER IS STAMPED `T1` WITH `status: UNRULED`** on every emitted block — whether the
> doctrine admits an action-free model at T1 is an open PI/Master-Mind ruling (BACKLOG R30), and the
> margin framing keeps every number correct either way. **⭐ THE NEW FINDING: an anchored one-shot
> model can be MAXIMALLY DEGENERATE while its trivial profile reads 0.0000.** MEASURED on a
> random-init RefCV3: the selection was a CONSTANT on **42/42** windows (`n_distinct = 1`, entropy
> 0.0000 of 2.9957 nats) while `straight_frac = const_speed_frac = 0.0000`, because a constant anchor
> is neither straight nor constant-speed. ⇒ a **SELECTION PROFILE** now prints beside the trivial
> profile and before any family row, and `T1_CHECKLIST.md` GATE 6 needs it. **Instrument controls all
> read their known values** (18 tests): `ha0` straight+const-speed on 42/42 with per-step chord
> `v0·dt` to < 1e-3; `os − ha0` **+7.96 m [6.15, 9.76] separated** (a random-init model must lose to
> the floor); `os − os_navshuf` **−0.0002 [−0.0024, 0.0016] not separated**; **`ha − ha0` +0.133
> [−0.050, 0.233] NOT separated — the measured reason `ha` is not the floor**; a future-only
> perturbation moves `os`/`ha`/`ha0` by **exactly 0.0** while GT moves > 1 m; the untouched
> `t1_eval.py --analyze-only` CLI reads the dump (exit 0). ⚠️ **BACKLOG R13 HAS LANDED**:
> `t1_eval.DEFAULT_TIERS` carries `"ha0": "T1"` at `t1_eval.py:154` (two differently-bound probes),
> so `T1_CHECKLIST.md`'s GATE-level defect #2 is stale. ⛔ **Nothing here is evidence about refcv3**:
> no real checkpoint was loaded, the pod `tanitad-refcv3` and Thor were not contacted, and
> `t1_eval.py` / `refav1_arm.py` / the model / the trainer were not edited.

---

## 8. DELIVERABLE MANIFEST

| # | artifact | where it lives | only one place? |
|---|---|---|---|
| 1 | **`taniteval/tools/REFCV3_ARM.md`** — the T1 definition (§2, the Master Mind's review item), the dump contract (§1), the real-read invocation (§3), what stays UNVERIFIED (§4), the decisions (§5). 486 lines, sha256 `1EC4F0E7ABC94DC5…` | `taniteval/tools/REFCV3_ARM.md` | **No** — repo, staged |
| 2 | **`taniteval/tools/refcv3_arm.py`** — the instrument. 1,890 lines, sha256 `6F4A9044527EF5D1…` | `taniteval/tools/refcv3_arm.py` | **No** — repo, staged |
| 3 | **`stack/tests/test_refcv3_arm.py`** — 18 tests. 607 lines, sha256 `0BF0ED8404B58CFB…` | `stack/tests/test_refcv3_arm.py` | **No** — repo, staged |
| 4 | `SPEC.md` (pre-registration, written before any code ran) | this directory | **No** — repo, staged |
| 5 | `RESULT.md` (this file) | this directory | **No** — repo, staged |
| 6 | `raw/console.log` — the full CLI transcript incl. both profiles and the untouched `t1_eval` run · sha256 `AD9B39CFAA494761…` | `…/2026-09-03-refcv3-arm/raw/` | **No** — repo, staged |
| 7 | `raw/pytest_refcv3_arm.log` — 18 passed · sha256 `09868E3C456CF2F8…` | same | **No** — repo, staged |
| 8 | `raw/refcv3_arm_fixture.json` — the full record (families, paired margins, both profiles, tier ruling, absent arms) · sha256 `A9C6172FDC7CD12A…` | same | **No** — repo, staged |
| 9 | `raw/t1_eval_on_refcv3_dump.json` — the **untouched** `t1_eval.py --analyze-only` reading the same dump · sha256 `CA6793F3EE0887EA…` | same | **No** — repo, staged |
| 10 | `raw/fixture_dump/` — the dump itself (`manifest.json` + 3 `ep*.npz` + 3 `decisions/ep*.npz`), the contract proof | same | **No** — repo, staged |
| 11 | proposed register row `D-REFCV3-ARM` | **§7 of this file only** | **YES — §7 is the single copy.** Not yet in `GOALS_AND_CLAIMS.md`; the Master Mind applies it. |

⛔ **Nothing in this package lives only on a pod, only in a worktree, or only in my context.** The
mirror at `C:\Users\Admin\tanitad-wt` was used to RUN (G: cannot import the stack) and is re-synced
FROM the repo, so it holds no unique content; every file above was authored in the repo and copied
outward.

**Not done, deliberately:** no commit, no push, no branch switch; no edit to `t1_eval.py`,
`refav1_arm.py`, `stack/tanitad/refs/refc_v3.py`, `stack/scripts/refc_v3_train.py`,
`stack/tanitad/data/v2_dataset.py` or the rescued draft under
`Implementation/incoming/2026-09-03-refcv3-arm-UNVERIFIED/` (read-only); the pod `tanitad-refcv3`
(TRAINING) and Thor were not contacted; no GPU job was started (`nvidia-smi
--query-compute-apps` showed only desktop processes on the 4060, and the whole package ran on CPU).
