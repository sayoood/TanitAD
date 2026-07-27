# The `vt_band` decision — what v5's early-stop stops on, priced

*2026-07-27 (Europe/Berlin; pods UTC). Owner: the vtband-decision stream.
Repo HEAD at start `c28a979`; **advanced to `d7d76bd` mid-session** (§7.1).
Hosts: **pod2** (real v5 config + real v5 caches, A40) + **tanitad-eval** (trained v4 head +
the raw val corpus, A40) + dev box (code/tests).
⛔ pod1 never contacted. ⛔ pod3 never contacted. ⛔ No v5 training launched. ⛔ No default changed.*

---

## 0. THE DECISION TABLE

⛔ **I did not choose. The PI chooses.** `crash_today` is the shipped behaviour and the RED baseline.

Probe values are the gate's own primary,
`pseudosim_composite_PSS_recovery_progress@twosided_v2` (higher is better), on
**8 held-out episodes / 528 windows, all finite**, arm
**`flagship-v4-fromscratch` @ step 15000** — a genuinely trained v4 head
(`goal_dropout 0.5`, `vt_gate 0.3836`, `rt_gate 0.1299`, **`sel_gate 0.1101`**).
Δ is the **paired episode-cluster bootstrap** vs `dropped` (`taniteval/ci.py`, B=2000,
unit = held-out episode). ⛔ Never `overlapping_holdout_se`.

| option | gate runs? | what the probe then MEANS | probe value | Δ vs `dropped` (paired, B=2000) | early-stop still fires? |
|---|---|---|---|---|---|
| **`crash_today`** (shipped) | ⛔ **NO** — `ValueError: cond_vtarget is on but no vt_band supplied`. Reproduced on the **real v5 config + real v5 caches + real 176×624 frame** (pod2) *and* on the raw corpus (eval) | nothing — the run dies at step 2 000 | — | — | ⛔ **n/a — a gate that cannot run cannot fire** |
| **`band0`** | ✅ yes | a planner told, on **every** window, *"target speed = **`v_stop`**, route = **LEFT**"*, with `route_graded = 0.0` saying *straight*. The longitudinal selection term stays **ON** | **0.5761** | **−0.0621 [−0.0878, −0.0371]** ⭐ **separated** | ✅ yes (streak 2) |
| **`dropped`** (`VT_DROPPED`/`ROUTE_DROPPED`) | ✅ yes | the deployed **no-goal** surface, on embedding rows that `goal_dropout = 0.5` trains on ~50 % of every batch. Longitudinal selection term **masked OFF** | **0.6383** (reference) | — | ✅ yes (streak 2) |
| **`produced`** (the model's own goal) | ✅ yes — ⚠️ needs a **signature change** (§3) | the **deployable** surface: `goal_head` on the encoded observation window. No future, no label | **0.6228** | **−0.0154 [−0.0222, −0.0083]** separated | ✅ yes (streak 2) |
| **`oracle`** (true band from the window's labels) | ⛔ **REFUSED**, not unimplemented | ill-defined: the gate probes a ego **rotated by ±8°**, and the label is minted from the **unperturbed** future poses (§4) | — | — | — |
| ⚠️ `zeros_naive` — *the patch the provenance invites* | ✅ yes | **A TRAP** (§2.3): also zeroes `vt_speed`, so the selector chases `(v0 − 5 m/s)` — it **brakes the probe** | 0.5685 | −0.0697 [−0.0991, −0.0425] separated | ✅ yes (streak 2) |

**Two diagnostics that split `band0`'s penalty** (same run, not candidates):

| diagnostic | probe value | Δ vs `dropped` | reading |
|---|---|---|---|
| `band0_vt_only` — `vt_band = 0`, route at its DROPPED row | **0.5805** | **−0.0578 [−0.0836, −0.0333] separated** | ⭐ **93 % of `band0`'s penalty is the VTARGET channel alone** |
| `band0_route_only` — route `= 0` (LEFT), `vt_band` at DROPPED | **0.6389** | **+0.0006 [−0.0010, +0.0024] NOT separated** | the `route = LEFT` error is **inert** on this checkpoint |

*(and it shows in the mechanism: `band0_vt_only` travels **24.99 m** — −8.5 % vs `dropped`, i.e.
essentially all of `band0`'s −9.1 % — while `band0_route_only` travels **27.08 m**, within 0.8 % of
`dropped`'s 27.30 m. Consistent with the trained gates: `vt_gate 0.3836` is **3.0×** `rt_gate 0.1299`.)*

---

## 1. THE ONE-LINE ANSWER TO EACH BRIEF QUESTION

1. **Does the gate run at all?** Only `crash_today` does not. All three candidates run,
   **on the real v5 config and the real v5 caches** (pod2, 176×624 cylindrical, `frame=` plumbed) —
   §6.1. ⭐ This alone unblocks the launch decision.
2. **What does each change about the probe's meaning?** Column 3 above.
3. **The numeric consequence.** Column 4/5, decomposed lateral/longitudinal in §2.2.
4. **Does the early-stop still fire?** ✅ under **every** option that runs, at `patience = 2`,
   against a **direction-checked** degradation — §5. **No candidate is disqualified under C13.**

---

## 2. ⛔ WHY THIS IS A DECISION AND NOT A PATCH — the three facts

### 2.1 Index 0 is a REAL class on BOTH categorical channels

`train_flagship_v4._goal_inputs` falls back to `torch.zeros` for both. Read against the
vocabularies (MEASURED, pinned by `test_vt_band_zero_is_v_stop_not_a_null` and
`test_route_zero_is_ROUTE_LEFT_and_contradicts_route_graded_zero`):

| channel | index 0 is | source |
|---|---|---|
| `vt_band` | ⛔ **`v_stop`** — the STOPPED target-speed band | `tanitad/lake/vocab.py`, `VTARGET_TOKENS[0]` |
| `route` | ⛔ **`ROUTE_LEFT`** | `refb_labels.py:76` — `ROUTE_LEFT, ROUTE_STRAIGHT, ROUTE_RIGHT = range(3)` |
| `route_graded` | `0.0` = **straight** — which **contradicts** the `LEFT` it is passed beside | `route_graded = tanh(mean_curv / CURV_TURN_PER_M)`; a genuine LEFT window carries `|graded| ≥ tanh(1) ≈ 0.7616` |

⇒ the "zeros" option does not probe a neutral planner. It probes one told *"you are coming to a
stop, and you are turning left"* — with a third channel saying *straight*. **The (LEFT, 0.0) pair is
a combination the training distribution essentially cannot contain.**

### 2.2 ⭐ THE PENALTY IS LONGITUDINAL, AND IT IS EXACTLY WHAT `v_stop` PREDICTS

The composite's admitted components are **`ego_progress` (w 5.0) and `recovery` (w 5.0)** —
`comfort` measured no range and was excluded, `no_collision`/`ttc` are not computable.
MEASURED, same 528 windows; the human travelled **27.47 m** on these windows:

| option | **LON** `ego_progress` | along-track (m) | raw ratio vs human | **LAT** `recovery` | cross-track end (m) |
|---|---|---|---|---|---|
| `band0` | **0.8408** | **24.83** | **0.8463** | 0.0460 | 2.751 |
| `dropped` | **0.9351** | **27.30** | **0.9913** | 0.0427 | 2.971 |
| `produced` | 0.9146 | 27.37 | **1.0063** | 0.0352 | 3.014 |
| `zeros_naive` | 0.8296 | 24.67 | 0.8351 | 0.0451 | 2.731 |

⭐ **`band0` makes the planner travel 24.83 m against `dropped`'s 27.30 m over the same 2 s — 2.47 m
less, i.e. −9.1 % — while `dropped` lands within 0.17 m of the human's 27.47 m** (per-window mean
ratio 0.9913). That is `v_stop` doing exactly what it says, measured on the deployable surface.
The lateral axis barely moves (cross-track end 2.751 m vs 2.971 m).

*(Against the human rather than against `dropped`, `band0` is 9.6 % short. `band0_vt_only` — the
VTARGET channel alone — travels 24.99 m, −8.5 % vs `dropped`: essentially all of the gap.)*

⚠️ **AND THE COMPOSITE UNDER-STATES IT.** `band0`'s `recovery` is *higher* than `dropped`'s
(0.0460 vs 0.0427) — because `recovery = 1 − xt_end/xt_hold` and `xt_hold = s_along·|tan dψ|`
shrinks with along-track travel, so **a slower plan gets paid on the recovery axis**. The true
penalty of conditioning on `v_stop` is larger than −0.0621.

### 2.3 ⛔ THE PATCH THAT LOOKS CONSERVATIVE IS THE WORST ONE

The tripwire's own docstring points at *"build the zeros the provenance promises"*. Doing that
literally also zeroes `vt_speed`, and `FlagshipV15Head.select` clamps `vt_speed` into the reachable
band around `v0` (MEASURED, real `v4_config()`: `reach = sel_accel_max·horizons[−1]·0.1 = 2.5·20·0.1 = 5.0 m/s`):

| `vt_speed` passed | `v_goal` at `v0 = [12, 8, 3]` | what the selector does |
|---|---|---|
| `v0` (`_goal_inputs`) | `[12.0, 8.0, 3.0]` | hold-v0 — an inert prior |
| **`0` (the naive patch)** | **`[7.0, 3.0, 0.0]`** | ⛔ **rank up the maximally decelerating candidate** |
| omitted (`None`) | — | term skipped entirely |

And a braking plan has `s_along → 0`, so `xt_hold → 0` on the gate's `dlat = 0` grid and `recovery`
goes **NaN by construction** — the composite would go **UP**. ⇒ a gate patched that way **reports a
healthier run while probing a planner that brakes.** Pinned by
`test_vt_speed_zero_makes_the_selector_chase_a_5mps_DECELERATION`.

### 2.4 The difference is in the SELECTOR too, not only the embedding

`condition()` returns `vt_keep = (band != VT_DROPPED)` and `select()` multiplies the longitudinal
penalty by it:

* `vt_band = 0` → `vt_keep = True` → longitudinal term **ON**
* `vt_band = VT_DROPPED` → `vt_keep = False` → longitudinal term **OFF**

`sel_gate` is **zero-init**, so at step 0 this is a strict no-op — and it is **learned**.
⭐ **MEASURED `sel_gate = 0.1101` at step 15000**, i.e. by the first real probe (step 2 000) the two
options differ in the *ranking* as well as the conditioning.

---

## 3. ⭐ IS `VT_DROPPED` IN-DISTRIBUTION? — established IN CODE, not by analogy

The brief's concern is the `ego=` shape: trained with a capability, scored without it. The
`v2_ego_dropout` precedent asks whether a withheld value is *a real ablation the model has seen*.
For `vt_band` the equivalent exists and is **stronger**, because it is a learned embedding row
rather than a zero-fill:

| fact | source | value |
|---|---|---|
| `V15Config.goal_dropout` | `tanitad/models/flagship_v15.py` | **0.5** |
| inherited unchanged by `V4Config` | `v4_config().goal_dropout` | **0.5** |
| the trainer never overrides it | `train_flagship_v4.train()` — no occurrence | ✅ |
| `condition()` masks per-example Bernoulli(0.5) → `VT_DROPPED` / `ROUTE_DROPPED` | `flagship_v15.py:458-460, 473-475` | ~**50 % of every batch** |
| `VT_DROPPED = N_VTARGET_BANDS = 23`, `ROUTE_DROPPED = 4` are their own `nn.Embedding` rows, deliberately distinct from the labeler's `ROUTE_UNKNOWN = 3` | `flagship_v15.py:73-83, 368, 375` | ✅ |

⇒ **`VT_DROPPED` is the single most frequently trained value of that channel.** All four facts are
pinned by `test_VT_DROPPED_is_IN_DISTRIBUTION_because_goal_dropout_ships_at_half` and
`test_the_dropped_rows_are_real_embedding_rows_distinct_from_UNKNOWN`, and
`make_goal_kwargs_fn("dropped", …)` **refuses** on a config with `goal_dropout == 0` rather than
feeding untrained `N(0, 0.02)` rows.

⚠️ **Stated honestly, and this is the part the PI should weigh:** this is still a
*withheld-capability* measurement. What makes it **not** the `ego=` bug is that `ego=None` skipped
the embedding entirely — an input the model had never been asked to handle — whereas `VT_DROPPED`
is an input it was trained to handle on half of every batch. ⭐ The head's refusal that started this
whole investigation is itself the `ego_guard` analogue **already built in**: `condition()` raising
on a missing `vt_band` is exactly what `taniteval/ego_guard.py` had to be bolted on to do.
**The crash is not a bug in the head — the bug is that `DeployableSurfacePlanner` walks into the
head's guard with an empty dict while advertising "withheld/unknown defaults (zeros)".**

⚠️ **Asymmetry worth naming:** `goal_dropout` only ever masks *toward* `VT_DROPPED`. Band 0
(`v_stop`) is seen only on the undropped half **and** only on windows whose true target speed
really is a stop.

---

## 4. THE THIRD OPTION — searched, found, and split in two

The brief asks whether a third option exists, e.g. the *true* band from the held-out window's own
labels. **Two distinct things were found.**

### 4.1 ⭐ `produced` EXISTS AND IS ALREADY SHIPPED CODE

`stack/scripts/goal_modes.py` already implements exactly this: `goal_head(states[:, -1])` → the four
strategic scalars → `route` / `route_graded` / `vt_band`. It is the `--goal-mode produced` eval path
and its own docstring calls it **"THE DEPLOYABLE PATH"**. Nothing had to be written; it had to be
*reached*.

⛔ **But it does NOT fit today's gate signature.** `goal_kwargs_fn(batch_size, device)` receives
neither `states` nor `v0`, and `traj` computes `states` **after** it has already built `kw`
(pinned by `test_todays_signature_cannot_carry_v0_or_states`). ⇒ wiring `produced` in is a
**signature change**, not a default change.

⚠️ **And so is every other option** — `_goal_inputs` sets `vt_speed = v0`, and `v0` is in scope at
that call site but not passed. **Today's signature cannot express even `band0` faithfully.**
`tanitad/train/heldout_goal.py` therefore standardises on `goal_kwargs_fn(states, v0)`, which all of
them can express; `StatesAwareSurfacePlanner` shows the exact 3-line reordering, and
`test_the_harness_is_equivalent_to_the_shipped_planner_when_the_goal_matches` proves the harness is
otherwise bit-identical to `DeployableSurfacePlanner`.

### 4.2 ⛔ `oracle` — priced and REFUSED, with the fatal reason third

| finding | detail |
|---|---|
| mechanically reachable | ✅ `pseudo_evaluate` already takes `goals=` and `traj` merges it into `kw` — **no signature change needed**, `HeldoutGate.probe` simply never passes it |
| blocker 1 — labels not in scope | the gate is handed val **episode objects** (`hg_eps = val_eps[:n]`), not `FlagshipV4Dataset` rows. `vt_band` is re-derivable from full-episode poses, but that is a new per-probe minting pass |
| blocker 2 — it leaks | `route` ≤ 25 s forward; `vt_band` = the 85th percentile of future speed over 10–20 s. An early-stop keyed on the future is not the deployable surface, and `GATE_PROTOCOL` §0.8 already demotes oracle-fed numbers |
| ⛔ **blocker 3 — FATAL, and unfixable by plumbing** | the gate probes the ego **rotated by dψ ∈ (−8, 0, +8)°**. The label is minted from the **unperturbed** future poses, so at the two perturbed points — **2 of 3 grid points, and the only ones on which `recovery` is even defined** — the "true" goal belongs to a pose the planner is not in. **There is no true label for a synthesised state.** |

---

## 5. ⭐ DOES THE EARLY-STOP STILL FIRE? — and the degradation that lied to me first

⛔ **An option under which the gate cannot fire is disqualified (C13).** Tested directly.

**The degradation used: the head's ranked pick replaced by a UNIFORM RANDOM candidate from the same
fan.** Chosen because it is the failure the gate was built for — `heldout_gate`'s own docstring:
*"Selection is the thing that regressed on v4."* It is applied identically under every option, so
"does the stop fire" is attributable to the option's baseline, not to a differently-sized insult.

Driven through the **shipped** `HeldoutGate.observe` — incumbent at probe 0, the degraded arm at
probes 1 and 2, `patience = 2`:

| option | baseline | degraded | paired Δ (B=2000) | direction OK? | **STOPS?** |
|---|---|---|---|---|---|
| `band0` | 0.5761 | 0.2205 | −0.3556 [−0.3781, −0.3315] sep | ✅ down | ✅ **True** (streak 2) |
| `dropped` | 0.6383 | 0.2226 | **−0.4157 [−0.4461, −0.3823]** sep | ✅ down | ✅ **True** (streak 2) |
| `produced` | 0.6228 | 0.2218 | −0.4010 [−0.4300, −0.3713] sep | ✅ down | ✅ **True** (streak 2) |
| `zeros_naive` | 0.5685 | 0.2205 | −0.3480 [−0.3684, −0.3263] sep | ✅ down | ✅ **True** (streak 2) |

⭐ **No candidate is disqualified.** ⚠️ But note the **dynamic range**: `dropped` has the largest
detectable drop (**−0.4157**) because it has the highest baseline. A gate whose baseline is already
depressed ~0.06 by a mis-specified goal has correspondingly less headroom to fall.

### 5.1 ⚠️ MY FIRST DEGRADATION WENT THE WRONG WAY — the self-check caught it

The brief warns that a sibling degraded a planner by slowing it and the composite went **UP +0.1698**
(a barely-moving plan scores `recovery = NaN` by construction). I designed around that trap — a pure
lateral drift that leaves along-track **bit-identical** — and **fell into a different one.**

**MEASURED (2 episodes, same arm): `recovery` 0.0291 → 0.2364, composite 0.6228 → 0.6903. The
"degradation" made the arm BETTER.** A constant-sign drift is a **correction** for a
systematically-biased planner: the model's signed cross-track error is one-sided, and `dyaw` is
symmetric, so +3 m re-centres one wing. `--degrade-selfcheck` refused it. Recorded, kept
reproducible behind `--degrade-mode lateral_drift`, and the class is the same as the sibling's:
**a degradation whose direction was assumed rather than measured.**

⭐ The random-selection degradation's direction is verified structurally too: along-track went **UP**
(24.8 → 36.0 m) and `ego_progress` collapsed (0.841 → 0.296) — so it is **over**-travel, not the
slow-plan/NaN artefact.

---

## 6. THE TWO LEGS, AND WHY BOTH WERE NEEDED

### 6.1 Leg A — real v5 config, real v5 caches (pod2), STRUCTURAL only

`--v2-val-cache …-w120-256x640cyl --frame-h 256 --frame-w 640 --frame-hfov 120 --projection
cylindrical --v2-subframe 176x624`, 600 providers, `val_eps[:8]`, 528 windows.

* ⛔ `crash_today` → **`ValueError: cond_vtarget is on but no vt_band supplied`** — **the defect
  reproduced on the exact v5 launch geometry.**
* ✅ every other option **RUNS**.

⛔ **Its probe VALUES are not quotable and are stamped so in the JSON**
(`FROM_SCRATCH_values_not_quotable: true`): no v5 checkpoint exists, so this leg is random-init.

⭐ **And that is itself the proof that Leg B was necessary.** At random init the options measured
**0.154655 (`band0`) / 0.154655 (`zeros_naive`) / 0.154653 (`dropped`) / 0.154862 (`produced`)** —
`band0` and `zeros_naive` **bit-identical**, `dropped` within **2 × 10⁻⁶** — because
`sel_gate = 0.0` **exactly** and all 24 VTARGET rows are i.i.d. `N(0, 0.02)`.
**An option priced on a from-scratch model would have concluded "they are the same."** That is the
C13 shape, and it is why every number in §0 comes from a trained head.

⚠️ **The same leg also shows why its VALUES must not be read as quality at all**: the random-init
planner emits plans travelling **117.36 m** where the human travelled 27.47 m — a progress ratio of
**6.64**. And the random-selection degradation made that arm **BETTER** (0.1547 → 0.2578, along-track
117.4 → 43.9 m), so `--degrade-selfcheck` **refused the direction** and the early-stop result there is
correctly reported as `STOPS = False` with `DIRECTION_OK = False`. ⛔ **That "False" is a statement
about a random model, not about any option.**

### 6.2 Leg B — trained head, raw val corpus (eval pod)

`flagship-v4-fromscratch` @ **step 15000**, `/root/valdata/physicalai-val-0c5f7dac3b11`,
`val_eps[:8]` → 528 windows / 8 episodes, every window finite under every option.
This is where §0's numbers come from.

⚠️ **Two caveats, stated rather than buried:**
1. **The corpus is the raw 256×256 deployed frame, not v5's 176×624 cylindrical.** No trained v5
   checkpoint exists, and a v4 checkpoint scored on v5 pixels would be off-distribution. What Leg B
   measures is *what the conditioning channels MEAN to a trained v4/v5-class head* — which is the
   semantic question the PI must decide. Leg A carries the config fidelity.
2. **This checkpoint predates `vision_rank = 16`** — its factorised heads are raw-2048. The driver
   builds the head to **match the checkpoint** (and refuses on any missing trained *parameter*)
   rather than loading `strict=False` into randomly-initialised readers. `vision_rank` feeds only
   the factorised **ranking** grafts; the `vt_band` decision lives in `condition()`/`vt_keep`.
   Recorded, not assumed away.

---

## 7. ⚠️ THINGS FOUND ALONG THE WAY THAT ARE NOT THE DECISION

### 7.1 The repo advanced mid-session and I nearly filed a false blocker

Between HEAD `c28a979` and `d7d76bd`, the **warp-geometry escalation was FIXED**:
`pseudo_evaluate` gained `frame=`, `HeldoutGateConfig` gained `frame`, `probe` forwards it, and
`train_flagship_v4` passes `frame=frame`. My first Leg A therefore hit
`WarpFrameRefused: no CanonicalFrame was supplied` under **every** option, and the obvious reading —
*"there is a second blocker in front of `vt_band`"* — **would have been wrong.** The refusal was
firing on **my driver**, which did not pass `frame=`. Fixed; Leg A re-run.

⇒ **On the real v5 launch command the warp blocker is already closed. `vt_band` is what remains.**
*(Class: asserting a blocker from one harness's failure without checking whether the harness or the
target owned it. Root-cause class = the same "absence found at ONE location" family.)*

### 7.2 ⚠️ `paired_episode_cluster_bootstrap` can print `separated: True` beside `0.0 [0.0, 0.0]`

On Leg A the JSON carries `delta 0.0, lo 0.0, hi 0.0, separated: true`. That is not a contradiction
and not a bug in the estimator — `separated` is computed on the **unrounded** bounds
(`ci.py:229 — bool(lo > 0 or hi < 0)`) while `delta`/`lo`/`hi` are **rounded to 4 dp** for
publication. With bit-identical arms the true bounds are ~1e-9 of one sign.
⚠️ **A reader, or an automated gate keying on `separated`, sees "separated" beside a zero effect.**
Harmless at Leg B's effect sizes; worth a `_degenerate` flag before anything automated reads it.
**Not fixed here — it is another module's instrument and this stream does not own it.**

### 7.3 Ops constraints — checked, not assumed

| constraint | how it was checked | result |
|---|---|---|
| ⛔ pod1 is TRAINING — do not touch | never contacted, at all | ✅ untouched |
| ⛔ pod3 on the D-B YouTube retry | never contacted, at all | ✅ untouched |
| ⚠️ pod2 `/workspace` hit its MooseFS quota this session | **real `dd`, 500 MiB, `oflag=direct`** — ⛔ never `df` | ✅ **571 MB/s**, headroom fine |
| pod2 / eval must be idle | `nvidia-smi` before launching | ✅ both **0 MiB used** of 46 068 MiB |
| explicit PIDs only, never `pkill -f` | nothing was killed; both jobs ran to completion | ✅ n/a |
| ⛔ no v5 training launched | the only pod2 job was a read-only probe driver | ✅ |
| ⛔ parity | nothing re-selects episodes; the gate's `val_eps[:8]` prefix is the trainer's own | ✅ |

⚠️ **On timing:** no wall-clock timing is quoted anywhere in this document, deliberately — the two
legs ran on different hosts against differently-warmed page caches, so any comparison would be
confounded. Nothing here depends on a duration.

### 7.4 A registry conflict, reported not resolved

`MODEL_REGISTRY.md` §1.5.5 records `flagship-v4-fromscratch` as
*"✅ READY, not launched … Zero GPU-day committed."* **pod2 carries
`/workspace/experiments/flagship-v4-fromscratch/ckpt.pt` at step 29999** plus
`ckpt_step{5,10,15,20}000.pt`, and eval carries a 15k copy. The arm was launched and ran to 30k.
⚠️ Per the standing rule the registry wins over prose — but here the **artifact** contradicts the
registry, so the registry row is what needs fixing. **Escalated, not edited** (agents do not own
that file's lineage rows).

---

## 8. ⭐ RECOMMENDATION — *a recommendation the PI may override*

⛔ **This is my reading, not a decision, and not a default I changed.**

**Recommend `dropped` (`VT_DROPPED` / `ROUTE_DROPPED`) for the v5 mid-run early-stop, with
`produced` as the strong second and the one to revisit at the gate step.** Reasoning, in order of
weight:

1. **`band0` is a measured mis-specification, not a neutral default.** It conditions on `v_stop` and
   `ROUTE_LEFT` on every window, costs **−0.0621 [−0.0878, −0.0371]** (separated), and the cost is
   **93 % attributable to the VTARGET channel alone** (−0.0578, separated; the route error is inert
   at +0.0006, not separated). Its mechanism is visible and directional: **9.0 % less along-track
   travel**. And the composite **under-states** it, because slow plans are paid on `recovery`.
2. **`dropped` is in-distribution by construction, in code**: `goal_dropout = 0.5` ships and the
   trainer never overrides it, so those rows are trained on ~50 % of every batch. It is the only
   option whose meaning does not depend on a label the gate does not have.
3. **It gives the early-stop the most headroom.** All options stop the degraded arm, but `dropped`
   has both the highest baseline (0.6383) and the largest detectable drop (**−0.4157**).
4. **It matches what the gate already claims to measure.** `DeployableSurfacePlanner`'s provenance
   advertises *"the deployed no-route state"*. `dropped` makes that string TRUE; `band0` leaves it
   false in a new way.
5. **`produced` is the more honest deployable surface and measures only −0.0154 below `dropped`** —
   but it needs the `goal_head` to be *trained*, and at the first probe (step 2 000 of a
   from-scratch run) it will not be. An early-stop whose conditioning quality drifts as a *second*
   head trains is comparing two moving things. ⚠️ **`produced` is the right primary for the 10 k
   gate; `dropped` is the right one for the mid-run stop.**

⚠️ **Whatever is chosen, the fix is a SIGNATURE change** (`goal_kwargs_fn(states, v0)`), because
today's `(batch_size, device)` cannot carry `v0` and therefore cannot express even `band0`
faithfully. ⛔ **Do not "build the zeros the provenance promises"** — §2.3.

**The one-line wiring, once the PI decides** (`heldout_gate.py` `traj`, plus passing the fn through
`probe`):

```python
states = self.world.encode_window(frames.to(self.device))
kw = (self.goal_kwargs_fn(states, v0d) if self.goal_kwargs_fn is not None else {})
out = self.head(states, v0d, **kw)
#   with  goal_kwargs_fn = heldout_goal.make_goal_kwargs_fn(<CHOICE>, head.cfg,
#                                                           goal_head=goal_head)
```

⚠️ `stack/tests/test_heldout_gate_real_head.py` will then FAIL by design — its docstring says to
**invert** it and record which goal state was chosen. `test_vtband_options.py` stays green either
way.

---

## 9. Suites — zero new skips

| suite | before (brief's baseline) | after | new skips |
|---|---|---|---|
| `stack/` (dev box) | 1489 passed, 12 skipped | ✅ **1509 passed, 12 skipped** | **0** |
| `taniteval/` (dev box) | 606 passed | ✅ **638 passed** | **0** |
| `stack/tests/test_vtband_options.py` on **pod2** (real torchvision) | — | ✅ **20 passed** | **0** |

⚠️ `taniteval` is **638, not the brief's 606** — it grew by 32 from *other* streams' work landed
between `c28a979` and `d7d76bd` (§7.1), not from this one. This stream added **no** taniteval tests.

New tests: `test_vtband_options.py` — **20**, 0 skips. ⚠️ The one grid assertion that would have
been an `importorskip` is deliberately a hard import, because it is load-bearing for the decision.

---

## 10. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `VTBAND_DECISION.md` (this) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-vtband-decision/` | no |
| ⭐ **`stack/tanitad/train/heldout_goal.py`** — the priced option registry (**inert; nothing imports it in the shipped path**) | `repo:stack/` (staged) + `pod2:/workspace/v5gate/stack/` + `eval:/root/vtband/stack/` | no |
| ⭐ **`stack/tests/test_vtband_options.py`** — 20 tests, 0 skips | `repo:stack/` (staged) + `pod2:` + `eval:` | no |
| `code/vtband_probe.py` — the measurement driver | `repo:…/code/` (staged) + `pod2:/workspace/vtband/code/` + `eval:/root/vtband/` | no |
| `raw/probe_v4fs15k_ep8_full.json` — **Leg B**, every number in §0/§2.2/§5 | `repo:…/raw/` (staged) + `eval:/root/vtband/raw/` | no |
| `raw/probe_v4fs15k_ep8.json` — Leg B first pass (4 options), reproduces §0 | `repo:…/raw/` (staged) + `eval:` | no |
| `raw/legA_v5config_structural.json` — **Leg A**, real v5 config/caches | `repo:…/raw/` (staged) + `pod2:/workspace/vtband/raw/` | no |
| `raw/legA.log`, `raw/probe_full.log` — the run transcripts | `repo:…/raw/` (staged) + pods | no |
| `raw/smoke2_lateral_drift_FAILED_direction_check.json` — §5.1, the degradation that lied | `repo:…/raw/` (staged) + `eval:` | no |

**I ran no `git commit`, no `git push`, and switched no branch.** I `git add`ed only my own paths.

⚠️ **THE INDEX CONTAINS A SIBLING STREAM'S WORK — do not commit it under this stream's message.**
`git diff --cached` also carries `…/incoming/2026-07-27-renderer-geometry/` plus
`taniteval/taniteval/{pseudosim,clhorizon}.py`, `stack/tanitad/train/heldout_gate.py`,
`stack/scripts/train_flagship_v4.py` and four test files — **that is the warp-geometry stream**, the
one whose `frame=` plumbing §7.1 describes. I staged none of it and modified none of it.
⚠️ Also present: `.claude/settings.local.json` (harness) and an untracked `4}` in the repo root —
neither is mine.

🔴 **INTEGRATION NEEDED, stated here and not in a README** (`AGENT_OPERATING_STANDARD` rule 3):
`stack/tanitad/train/heldout_goal.py` is **inert** — nothing in the shipped path imports it. It stays
inert until the PI picks an option and the 3-line `traj` reordering in §8 lands. **Until then
`train_flagship_v4 --heldout-gate` still dies at step 2 000**, and that is the only thing standing
between the current tree and a v5 launch.

🔒 **Confidentiality swept, not assumed:** every staged `code/` and `raw/` file was scanned for
UUID-shaped clip ids — see §11 for the count. Counts, gates and digests only.

---

## 11. Provenance of every number

| claim | class | source |
|---|---|---|
| `crash_today` refuses on the **real v5 config + real v5 caches + 176×624** | **MEASURED** | `raw/legA_v5config_structural.json`, `raw/legA.log` (pod2) |
| every other option runs on that config | **MEASURED** | same |
| Leg A values 0.154655/0.154655/0.154653 — bit-identical at random init | **MEASURED** | same; ⛔ **not quotable as quality**, stamped `FROM_SCRATCH_values_not_quotable` |
| probe values 0.5761 / 0.6383 / 0.6228 / 0.5685; n = 528/8 | **MEASURED** | `raw/probe_v4fs15k_ep8_full.json` (eval), arm `flagship-v4-fromscratch` @ step 15000 |
| paired Δ vs `dropped`, all separated | **MEASURED** | same — `paired_episode_cluster_bootstrap`, B=2000, unit = held-out episode |
| channel isolation: vt-only −0.0578 sep, route-only +0.0006 **not** sep | **MEASURED** | same |
| lat/lon decomposition; human along-track 27.47 m | **MEASURED** | same, via `pseudosim._cross_and_along` / `score_windows` |
| admitted components = `ego_progress` + `recovery` (w 5.0 each); comfort excluded | **MEASURED** | same (`components_admitted`) |
| trained `sel_gate 0.1101`, `vt_gate 0.3836`, `rt_gate 0.1299`, `goal_dropout 0.5` | **MEASURED** | same (`head_cfg`), read off the loaded checkpoint |
| early-stop fires under every option, streak 2, all Δ separated | **MEASURED** | same (`early_stop`, `degradation`) |
| the lateral-drift degradation went the WRONG way (0.6228 → 0.6903, recovery 0.0291 → 0.2364) | **MEASURED** | `raw/smoke2_lateral_drift_FAILED_direction_check.json` (2 episodes) |
| `VTARGET_TOKENS[0] == "v_stop"`; `ROUTE_LEFT == 0`; `ROUTE_DROPPED == 4`; `VT_DROPPED == 23` | **MEASURED** | `tanitad/lake/vocab.py`, `refb_labels.py:76,536`, `flagship_v15.py:73-83`; pinned by tests |
| `goal_dropout = 0.5` ships and the trainer never overrides it | **MEASURED** | `flagship_v15.py`, `v4_config()`, source scan of `train()`; pinned by test |
| `reach = 5.0 m/s`; `vt_speed = 0 → v_goal = (v0−5)⁺` | **MEASURED** | `flagship_v15.select`; computed and pinned by test |
| the gate's grid is `dlat = 0`, `dyaw (−8, 0, +8)`, `dlon (0,)` | **MEASURED** | `heldout_gate.probe_grid` / `pseudosim.GridSpec`; pinned by test |
| today's `goal_kwargs_fn(b, device)` cannot carry `v0`/`states` | **MEASURED** | `heldout_gate.py` source; pinned by test |
| the warp blocker is already fixed at `d7d76bd` | **MEASURED** | `pseudo_evaluate(..., frame=)`, `HeldoutGateConfig.frame`, `train_flagship_v4:1363` |
| `separated: True` beside a rounded `0.0 [0.0, 0.0]` | **MEASURED** | `raw/legA_v5config_structural.json` + `taniteval/ci.py:229` |
| registry §1.5.5 says `flagship-v4-fromscratch` was never launched; step-29999 ckpt exists | **MEASURED** | `MODEL_REGISTRY.md` §1.5.5 vs `pod2:/workspace/experiments/flagship-v4-fromscratch/` |
| the ~29.5 GPU-h the gate exists to save | INHERITED | `flagship-v5-retrain.PREP.md` cause #1 — **not re-derived here** |
| the sibling's +0.1698 slow-plan artefact | INHERITED | the brief — **not re-derived here** |
| pod1 is training; pod3 is on the D-B retry | **NOT PROBED** | neither was contacted this session |

🔒 No clip UUID appears in this document, in any artifact, or in any test fixture.
