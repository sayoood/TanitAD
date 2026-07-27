# `--heldout-gate` now reaches its probe — the wiring, the stop, and a core-estimator fix

*2026-07-27 (Europe/Berlin; pods UTC). Owner: the vtband-WIRING stream.
Repo HEAD at start `8e3491f`. Hosts: **pod2** (real v5 config + real v5 caches + the trained
`flagship-v4-fromscratch` checkpoints, A40) + dev box (code/tests).
⛔ pod1 never contacted (TRAINING). ⛔ pod3 never contacted. ⛔ No v5 run launched.
⛔ No YouTube. ⛔ `df` never used.*

---

## 0. HEADLINE

| job | status | the one thing that matters |
|---|---|---|
| **1. wire `vt_band`** | ✅ **DONE, proved end-to-end on the real v5 config** | `--heldout-gate` **reached and passed a probe at step 20** on `176x624` cylindrical + the real v5 caches: `primary_value 0.265049`, 264 windows / 4 episodes. Before this it raised `ValueError: cond_vtarget is on but no vt_band supplied`. |
| **1b. the stop still fires** | ✅ **DONE, on a direction-verified degradation** | Trained head (`flagship-v4-fromscratch` @15000), 528 windows / 8 episodes, through the **shipped** `HeldoutGate.probe`: **STOPS at streak 2**, Δ **−0.3969 [−0.4399, −0.3509]** separated, and the degradation is verified worse on **both** axes (`ego_progress` 0.935 → 0.294; along-track **UP** 27.30 → 36.13 m, so it is not the slow-plan NaN artefact). `ADMISSIBLE: true`. |
| **2. CI display** | ✅ **FIXED**, with a test that fails on the old rendering | `paired_episode_cluster_bootstrap` can no longer print an interval that contradicts its own `separated`. ⚠️ **4 committed numbers across 3 streams were rendered through the bad path** — named in §4.3, **not restated**. |
| **3. registry conflict** | 🔴 **ESCALATED, not edited** | `MODEL_REGISTRY.md` §1.5.5 is **FALSE**: `flagship-v4-fromscratch` ran to step **29999** and exited `rc=0`. Exact replacement text in §5. |

⚠️ **THE DEFAULT IS MINE, NOT THE PI's.** `--heldout-goal` defaults to **`dropped`**. `VTBAND_DECISION.md`
priced the options and **deliberately declined to choose**; I picked one so the gate could run at all.
**`--heldout-goal band0` / `--heldout-goal produced` overrides it and nothing else changes** — pinned by
`test_the_option_is_ONE_FLAG_and_the_alternatives_actually_change_the_probe`, which asserts the flag
moves the *probe value*, not just a provenance string.

---

## 1. WHAT WAS ACTUALLY BROKEN — and it was broken twice

`stack/tanitad/train/heldout_goal.py` shipped at `8e3491f` **complete, tested (20 tests) and INERT**:
nothing in the shipped path imported it. So the priced decision existed and the defect it priced did
not move. **A module can be correct, tested and inert at the same time** — and only a wiring test can
tell you which. `test_heldout_gate_goal_wiring.py::test_heldout_goal_is_actually_IMPORTED_by_the_shipped_path`
now asserts exactly that property, because prose already said "wired" once when it was not.

The fix is a **SIGNATURE change**, not a default change (as `VTBAND_DECISION.md` §3 established and I
re-verified in code rather than inheriting):

```python
# BEFORE — DeployableSurfacePlanner.traj
kw = (self.goal_kwargs_fn(b, self.device) if self.goal_kwargs_fn is not None else {})
...
states = self.world.encode_window(frames.to(self.device))     # ← AFTER kw
out = self.head(states, v0.to(self.device), **kw)

# AFTER
states = self.world.encode_window(frames.to(self.device))     # ← BEFORE kw
v0d = v0.to(self.device)
kw = (self.goal_kwargs_fn(states, v0d) if self.goal_kwargs_fn is not None else {})
out = self.head(states, v0d, **kw)
```

**Both halves are load-bearing and either alone is insufficient:**

* `(b, device)` carries no `v0`, and `_goal_inputs` sets `vt_speed = v0`. Without it even **`band0`
  cannot be expressed faithfully** — the only reachable substitute is `vt_speed = 0`, which is the
  trap (§1.2);
* `states` computed *after* `kw` makes **`produced` unreachable in principle**, since that option is a
  function of the encoded window.

### 1.1 The wiring, file by file

| file | change |
|---|---|
| `stack/tanitad/train/heldout_gate.py` | `DeployableSurfacePlanner` takes `goal_kwargs_fn(states, v0)` + `goal_option=`; `traj` reordered; new `GOAL_OPTION_DEFAULT` / `GOAL_OPTION_PROVENANCE`; `HeldoutGateConfig.goal_option`; `probe(..., goal_head=None)` **builds the fn itself** from `cfg.goal_option` when none is passed |
| `stack/tanitad/train/heldout_goal.py` | header corrected from *"NOTHING HERE IS WIRED IN"* to what is now true; `dropped`'s refusal **scoped** to heads that actually carry the rows (§1.3); `StatesAwareSurfacePlanner` demoted to a delegating shim |
| `stack/scripts/train_flagship_v4.py` | `--heldout-goal` (choices = the full option registry, default `dropped`); passed into `HeldoutGateConfig`; `goal_head=goal_head` forwarded at the probe call site; the option printed at launch **with its provenance**; carried in `--print-launch`; **`preflight_asserts` BLOCKS `crash_today`, `zeros_naive` and both diagnostics** |

### 1.2 ⛔ The zero-fill was NOT implemented, and the code refuses to reach it by accident

The brief's prohibition is enforced in three places, not asserted in prose:

1. `zeros_naive` exists **only** as a named, priced option — it is never a fallback and never a default;
2. `preflight_asserts` **BLOCKS** `--heldout-goal zeros_naive` with its measured mechanism
   (`v_goal = (v0−5)⁺` → braking plan → `s_along → 0` → `xt_hold → 0` on the `dlat = 0` grid →
   `recovery` NaN → **composite UP**);
3. `test_zeros_naive_and_crash_today_are_BLOCKED_by_preflight` fails if either block is removed.

⭐ **The failure mode this prevents is the nastiest one available here: a gate patched that way reads
HEALTHIER while probing a planner that brakes.** It would not crash. It would produce numbers.

### 1.3 One thing I changed in `heldout_goal.py` that the decision stream had not hit

`make_goal_kwargs_fn("dropped", cfg)` refused whenever `goal_dropout == 0`. Correct in intent — feeding
untrained `N(0, 0.02)` rows measures initialisation — but **unscoped**: a head conditioned on *neither*
channel never receives a categorical at all (`_filter` returns `{}`), so the refusal fired where there
was no untrained row to protect. Wired in as-is it made the gate raise against every stub head in the
suite, i.e. **it would have converted "the gate crashes on a real head" into "the gate crashes on a
toy head"** — a silently-unusable early-stop, which is the failure this module exists to remove. The
guard now checks `cond_vtarget or cond_route` as well. Pinned by
`test_the_dropped_refusal_is_SCOPED_to_heads_that_have_the_rows`, and the original refusal is still
pinned by `test_probe_REFUSES_dropped_on_a_config_that_turned_the_dropout_OFF`.

### 1.4 The default, stated as plainly as I can

**`dropped` is my choice, pending the PI's override.** Reasoning re-verified in code, not inherited:

| claim | how I checked it | value |
|---|---|---|
| `V15Config.goal_dropout` ships at 0.5 | read the dataclass | **0.5** |
| `V4Config` inherits it unchanged | `v4_config().goal_dropout` | **0.5** |
| the trainer never overrides it | **AST scan** for any assignment or kwarg named `goal_dropout` anywhere in `train_flagship_v4.py` | **none** |
| ⇒ `VT_DROPPED` / `ROUTE_DROPPED` are learned rows seen on ~50 % of every batch | follows from the three above | ✅ |

⚠️ The previous form of that check was a **substring scan** (`"goal_dropout" not in body`) and my own
`--heldout-goal` help text — which *documents* `goal_dropout = 0.5` — tripped it. A false positive that
pressures the next reader to delete the check is worse than no check, so it is now an AST scan for
**assignment**. *(Root-cause class: a tripwire whose predicate is broader than the property it guards.)*

The other two arguments for `dropped` I did **not** re-measure and mark **INHERITED** from
`VTBAND_DECISION.md`: the −0.0621 [−0.0878, −0.0371] cost of `band0` and the −0.4157 dynamic range.
They did not decide the wiring — the wiring works identically under all three — and the flag exists so
they do not have to.

---

## 2. ⭐ PROOF 1 — `--heldout-gate` REACHES A PROBE ON A REAL RUN

**MEASURED, pod2, `/tmp/v5_wiring_dropped.log` + `.../run/flagship-v5-WIRING-dropped/train_log.jsonl`.**
The command is `launch_smoke2.sh`'s — the **real v5 config and the real v5 caches** — with
`--heldout-every 20` instead of 2000 so the probe arrives in minutes instead of GPU-hours, and
`--steps 65` so three probes fit (⚠️ at `--steps 45` the shipped **preflight correctly BLOCKED the
launch**: two probes cannot satisfy `patience = 2`, and it said so before spending anything).

```
--v2-train-cache …/physicalai-train-e438721ae894-w120-256x640cyl   (parity VERIFIED, 2400 clips)
--v2-val-cache   …/physicalai-val-0c5f7dac3b11-w120-256x640cyl     (parity VERIFIED,  600 clips)
--frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe 176x624
--from-scratch --require-parity --heldout-gate --heldout-every 20 --heldout-episodes 4
--heldout-patience 2 --heldout-nboot 400 --heldout-goal dropped
```

What the run printed at launch:

```
[heldout-gate] ON — every 20 steps on 4 held-out episodes;
               primary=pseudosim_composite_PSS_recovery_progress@twosided_v2; patience=2;
               re-render frame=176x624f305.5775cyl (cylindrical). NOT gated on ade_0_2s.
[heldout-gate] goal='dropped' — vt_band=VT_DROPPED(23), route=ROUTE_DROPPED(4), … vt_keep=False.
[heldout-gate] ⚠️ default 'dropped' chosen by the vtband-WIRING stream 2026-07-27,
               PENDING THE PI's OVERRIDE — … Override with --heldout-goal {band0,produced}.
```

And at step 20 — **the probe the shipped code used to die at**:

| field | value |
|---|---|
| `primary_value` | **0.265049** |
| `n_windows` / `n_episodes` | **264 / 4** |
| `planner_calls` | 264 |
| `components_admitted` | `{ego_progress: 5.0, recovery: 5.0}` (comfort excluded — no discriminative range) |
| `surface.goal_conditioning` | `goal option 'dropped' via goal_kwargs_fn(states, v0)` |
| `warp.frame_tag` | `176x624f305.5775cyl` — **the train frame, not the deployed 256×256 pinhole** |
| `grid` | `dlat (0.0,)` · `dyaw (−8, 0, +8)°` · `dlon (0,)` |
| `role` | `incumbent (first probe)` |

⭐ **That single line is the launch unblocker.** ⛔ The probe VALUE is **not quotable as quality** — this
is a from-scratch model at step 20. What is quotable is that the gate *ran, scored, formed a composite,
pinned its admitted components and became the incumbent*, on the exact geometry v5 will launch on.

### 2.1 ⭐ AND THE WHOLE STOP PATH EXECUTED, INSIDE THE REAL TRAINER

The run did not merely probe — it **stopped itself**. All three probes, from
`raw/heldout_gate_probes_v5config.json`:

| step | `primary_value` | role | paired Δ vs incumbent (B=400) | separated-worse | streak | stop |
|---|---|---|---|---|---|---|
| 20 | 0.265049 | incumbent | — | — | 0 | false |
| 40 | 0.131927 | challenger | **−0.1331 [−0.1882, −0.0614]** sep | ✅ | 1 | false |
| 60 | 0.127348 | challenger | **−0.1377 [−0.2121, −0.0572]** sep | ✅ | **2** | ⭐ **true** |

```
[ckpt] milestone archived: ckpt_step20.pt
{"EARLY_STOP": true, "step": 60, "reason": "held-out primary (…@twosided_v2) separated-WORSE
 than the incumbent (step 20) for 2 consecutive probes (patience=2); last delta -0.1377
 [-0.2121, -0.0572], paired_episode_cluster_bootstrap, n_episodes=4. Training past this point
 is the ~29.5 GPU-h this gate exists to stop spending."}
```

`ckpt_best.pt` was written at the **incumbent** (step 20), not at the last step — the peak survives the
tail, which is the thing whose absence forced a hand recovery on the last run.

⚠️ **STATED CAREFULLY, because it would be easy to over-claim:** this stop was caused by **real
early-training dynamics under the λ_plan Phase-B ramp**, not by an injected degradation, and its
**direction was not pre-verified**. It therefore proves that the **stop PATH executes end-to-end in the
shipped trainer** — probe → paired bootstrap → streak → `stop` → best-checkpoint save → loop break →
`EARLY_STOP` record. It does **not** show that the gate discriminates a decayed model. That is §3's job,
and §3 uses a trained head and a direction-checked degradation precisely because this run cannot answer it.

*(It also incidentally re-confirms the fixed `ci.py` is inert on ordinary numbers: both live paired
intervals rendered at plain 4 dp with no `display_dp` and no `degenerate` flag.)*

---

## 3. ⭐ PROOF 2 — THE STOP STILL FIRES, ON A DEGRADATION WHOSE DIRECTION I MEASURED

§2.1 shows the stop *path* executing, but on a random-init model whose decline I did not induce and
whose direction I did not verify — so it cannot show the gate **discriminating**. This does.

**Setup** (`code/gate_stop_proof.py`, pod2, `raw/gate_stop_proof_v4fs15k.json`):

* arm **`flagship-v4-fromscratch` @ step 15000** — a genuinely trained head
  (`goal_dropout 0.5`, `cond_vtarget/cond_route` both on, `sel_gate` learned);
* `/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11`, `val_eps[:8]` →
  **528 windows / 8 held-out episodes**, the gate's own prefix;
* ⭐ **every probe goes through the SHIPPED `HeldoutGate.probe` with NO `goal_kwargs_fn` argument** —
  the trainer's exact call shape. Probe 0 clean (incumbent), probes 1–2 degraded, `patience = 2`;
* the degradation is injected into **`head.select`**, not into a planner wrapper, so
  `DeployableSurfacePlanner` and `HeldoutGate.probe` stay byte-identical to what a run executes.

⭐ **CROSS-CHECK, unplanned and worth stating:** the clean incumbent through the shipped gate measures
**0.638251** — the value `VTBAND_DECISION.md` reports for `dropped` through its own
`StatesAwareSurfacePlanner` harness, to six figures. The wiring did not change the measurement; it made
it reachable.

### 3.1 The degradation, and the direction check that has to come first

The head's **ranked pick is replaced by a uniform random candidate from its own fan**. Chosen because
it is the failure the gate was built for (`heldout_gate`: *"Selection is the thing that regressed on
v4"*), and because the candidate set is unchanged — so it cannot systematically slow the plan (the NaN
trap) or re-centre a lateral bias (the drift trap).

⚠️ **Two degradations in this program have already moved the WRONG way for structural reasons** — a
constant-sign lateral drift *re-centred* a one-sidedly-biased planner (recovery 0.029 → 0.236), and a
slowdown *raised* the composite +0.1698 because a barely-moving plan scores `recovery = NaN`. So
`gate_stop_proof.py` refuses to report a stop unless **all four** hold. **MEASURED, same 528 windows:**

| check | clean | degraded | verdict |
|---|---|---|---|
| composite went **DOWN** | 0.638251 | **0.241371** | ✅ |
| paired CI separated **and negative** | — | **−0.3969 [−0.4399, −0.3509]** | ✅ |
| `recovery` finite count **not collapsed** (the NaN artefact) | 352 / 528 | 339 / 528 (**96.3 %** kept) | ✅ |
| along-track **not shrunk** (the slow-plan artefact) | 27.30 m | **36.13 m — it went UP** | ✅ |

⭐ **The mechanism is visible and is the right one.** `ego_progress` collapses **0.9351 → 0.2943** while
along-track travel *increases* 27.30 → 36.13 m — i.e. the randomly-selected plan **over-travels**, which
is the opposite of the slow-plan/NaN trap by construction. Cross-track end goes 2.971 → 4.434 m, so both
axes degrade. **This degradation is not the one that lied.**

### 3.2 The result — driven through the shipped gate

| probe | arm | `primary_value` | paired Δ vs incumbent (B=2000, 8 episodes) | separated-worse | streak | **STOP** |
|---|---|---|---|---|---|---|
| 0 | clean (incumbent) | **0.638251** | — | — | 0 | false |
| 1 | DEGRADED | 0.251822 | **−0.3864 [−0.4236, −0.3562]** | ✅ | 1 | false |
| 2 | DEGRADED | 0.241371 | **−0.3969 [−0.4399, −0.3509]** | ✅ | **2** | ⭐ **true** |

```json
{"STOPS": true, "worse_streak": 2, "DIRECTION_OK": true, "ADMISSIBLE": true}
```

⭐ **`ADMISSIBLE` requires BOTH** — the gate stopped **and** the degradation is verified to have made
the arm worse. A stop on an unchecked degradation proves nothing about the gate, so the driver returns
a non-zero exit code when either half fails.

Head as loaded (**MEASURED off the checkpoint, not inherited**): `goal_dropout 0.5`,
`sel_gate 0.110144`, `vt_gate 0.383575`, `rt_gate 0.129860` — so the longitudinal selection term the
`dropped` option masks off is genuinely **learned and non-zero** at this checkpoint, which is what
makes the option choice a real one rather than a no-op.

---

## 4. JOB 2 — THE CORE ESTIMATOR COULD PRINT A SEPARATED VERDICT THAT READS AS A NULL

### 4.1 The defect

`taniteval/ci.py` decided `separated` on the **unrounded** percentile bounds
(`bool(lo > 0 or hi < 0)`) while it rounded `delta`/`lo`/`hi` to **4 dp** for publication. When the true
bounds are ~1e-9 of one sign the emitted record read:

```json
{"delta": 0.0, "lo": 0.0, "hi": 0.0, "ci95": 0.0, "p_delta_gt0": 1.0, "separated": true}
```

⚠️ **Every stream in this program reads this estimator's output**, and `HeldoutGate.observe` keys the
**v5 early-stop** on `separated`. A verdict printed beside a null effect is resolved by the reader in
whichever direction they already believed.

### 4.2 The fix — adaptive precision, marker as the fallback

⛔ **The statistics are untouched.** Testing rounded bounds would silently redefine `separated` as
*"separated by at least 5e-5"* — an unregistered threshold that would flip real verdicts. **The display
is what lies, so only the display is fixed** (`ci._render_bounds`).

Keep 4 dp — so **no existing number changes** — unless 4 dp would put the printed interval on the wrong
side of zero; in that case show exactly as many digits as it takes to agree with the verdict, up to
1e-12, and emit `display_dp` + `display_note` saying the extra digits are a rendering fix and not extra
precision. Past 1e-12 no number can carry the verdict, so `degenerate: true` + `degenerate_note` says
the arms are effectively identical and forbids quoting it.

**Why this and not the alternatives** (the brief offered three; all three were considered):

| option | why not |
|---|---|
| always more significant figures | changes **every** published interval to fix a degenerate case, and 12-dp bounds invite reading bootstrap noise as signal |
| scientific notation below a threshold | needs a threshold, which would itself become quotable. The value is only ever compared **against zero** — that is a SIGN question, not a magnitude one |
| an explicit marker **alone** | a reader who skips the marker still sees `0.0 [0.0, 0.0] separated`. The numbers must be right first; the marker is what remains when they cannot be |

⇒ the marker is the **fallback**, not the mechanism.

### 4.3 ⚠️ COMMITTED NUMBERS RENDERED THROUGH THE BAD PATH

A full scan of every tracked `*.json` in the repo containing `"separated"` (**264 files**) found
**4 nodes in 3 artifacts, from 3 different streams**, printing `separated: true` beside `0.0 [0.0, 0.0]`.
⛔ **Named, not restated** — the committed files carry only the summary node, so the per-window arrays
needed to re-render are not present and I did **not** re-derive them.

| artifact (committed) | node | n | `p_delta_gt0` | reading |
|---|---|---|---|---|
| `…/2026-07-27-vtband-decision/raw/legA_v5config_structural.json` (`8e3491f`) | `options/band0/paired_vs_reference` | 528 w / 8 ep | **1.0** | ⚠️ **already stamped `FROM_SCRATCH_values_not_quotable: true`** and flagged in that doc's §7.2. Probe values differ by **2e-6** (0.154655 vs 0.154653) — arithmetic, not evidence |
| same | `options/zeros_naive/paired_vs_reference` | 528 / 8 | **1.0** | same |
| `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44.json` (`2d6589b`) | `paired_common_start/deltas_vs_K20/longitudinal/160/d_closed_ade2s_m` | 80 / 21 | **0.975** | ⚠️ **the marginal case, and the one that matters.** `p = 0.975` puts `lo` exactly at the 2.5 % boundary; `\|lo\| < 5e-5`. **A separation this marginal should not be read as an effect** |
| `…/2026-07-28-tactical-action-input/artifacts/blockA/blockA_full_panel_20arm.json` (`d5d5afb`) | `paired/refc_base_produced__minus__refc_base_v0on/recovery` | 13184 / 40 | **0.9885** | ⚠️ `\|lo\| < 5e-5` on a 40-episode panel — a "separated" `recovery` delta of numerically **zero** |

🔴 **INTEGRATION NEEDED (`AGENT_OPERATING_STANDARD` rule 3), stated here and not in a README:** the last
two are **other streams' published artifacts**. I did not edit them. **Whoever owns
`2026-07-25-closedloop-horizon-and-shift` and `2026-07-28-tactical-action-input` must re-render those
two nodes from their per-window data and re-read the claims that lean on them** — under the fixed
`ci.py` both will now print either their true (tiny) bounds or a `degenerate` marker.

### 4.4 The test that fails on the old rendering

`taniteval/tests/test_ci_rendering.py` — 6 tests. Run against `git show HEAD:taniteval/taniteval/ci.py`
in a scratch tree (**MEASURED**, transcript in `raw/ci_rendering_test_PREFIX_vs_POSTFIX.txt`):

```
FAILED test_bit_identical_arms_do_not_print_separated_beside_a_zero_interval
   AssertionError: the printed interval contradicts the printed verdict:
                   delta=0.0 [0.0, 0.0] separated=True
FAILED test_a_tiny_but_REAL_separation_prints_enough_digits_to_show_it
FAILED test_ordinary_effect_sizes_are_still_rendered_at_exactly_4dp
FAILED test_render_bounds_never_needs_more_than_MAX_DISPLAY_DP
4 failed, 2 passed
```

against the fix: **6 passed**. The headline test asserts **agreement**, not a direction — whether the
estimator calls a case separated is a statistics question and deliberately not the test's business.
`test_the_SEPARATION_TEST_ITSELF_is_untouched_by_the_rendering_fix` recomputes the predicate from the
raw draws to prove the statistics did not move, and
`test_ordinary_effect_sizes_are_still_rendered_at_exactly_4dp` guarantees every ordinary interval in
the program is byte-identical to before.

---

## 5. JOB 3 — THE REGISTRY ROW IS FALSE. ESCALATED, NOT EDITED.

⛔ **I did not touch `MODEL_REGISTRY.md`** — agents do not own its lineage rows.

**The conflict.** §1.5.5 records `flagship-v4-fromscratch` as *"✅ READY, not launched … Zero GPU-day
committed."* **From the artifacts on pod2 (not from prose):**

| evidence | value | source |
|---|---|---|
| final step | **29999** | `pod2:/workspace/experiments/flagship-v4-fromscratch/metrics.json` |
| exit | **`trainer exited rc=0` … `clean finish (summary.json done)`** | `…/supervisor.log` |
| launched | **2026-07-23T21:54:44Z**, trainer pid 108011, restarts 0 | `…/supervisor.log` |
| finished | **2026-07-26T09:01:37Z** | `…/supervisor.log` |
| wallclock | **212 544.6 s = 59.04 h** (registry ESTIMATED ~53 h) | `metrics.json` |
| `--from-scratch` actually in force | `"from_scratch": true`, `trunk.ckpt: null`, `trunk.step: -1` | `…/config.json` |
| args as launched | `steps 30000, batch 16, accum 4 (eff 64), lr_head/trunk 1e-4, warmup 2000, floor 0.25, phase_a 2000, phase_b 8000` | `…/config.json` → `args` |
| checkpoints on disk | `ckpt.pt` (3.24 GB) + `ckpt_step{5,10,15,20}000.pt` | `ls -la` |
| held-out result | val `ade@2s` **0.5063**, `oracle_ade@2s` 0.1892, `sel_gap@2s` 0.3172, `miss@2m` 0.2145, n = 881 | `metrics.json` |
| train log | 661 rows, last step 29999, `plan_ade 0.3659`, `oracle_ade 0.1647` | `…/train_log.jsonl` |

⚠️ **This is high blast radius, and not hypothetically:** it is *the arm every selection experiment this
week used* — `VTBAND_DECISION.md`'s entire Leg B is `flagship-v4-fromscratch @ 15000`, and
`RETRACTION_LOG` 07-26 registers the closed-loop co-primary on `@29999`. A registry row saying the arm
was never launched, beside published numbers measured on it, is the exact shape the registry rule
exists to prevent.

### 5.1 ⭐ PROPOSED REPLACEMENT TEXT (for the PI / registry owner to apply)

Replace the §1.5.5 heading and **Status** row; keep **Distinguishing lever** and **Validation**
verbatim; replace **Cost** and **Code state**; add **Result** and **Location**.

```markdown
#### 1.5.5 flagship-v4 from-scratch — `flagship-v4-fromscratch` — ✅ **COMPLETE (30 k, rc=0)**

| Field | Value |
|---|---|
| **Status** | ✅ **LAUNCHED 2026-07-23T21:54:44Z and RAN TO COMPLETION.** `final_step 29999`, supervisor `trainer exited rc=0` / `clean finish`, 2026-07-26T09:01:37Z, restarts 0. ⚠️ This row read *"READY, not launched … Zero GPU-day committed"* until 2026-07-27 — it was **never updated after launch**, while the arm became the one every selection experiment of the week measured on. Spec: `…/incoming/2026-07-23-v4-fromscratch/V4_FROMSCRATCH_LAUNCH.md`; launch record `…/incoming/2026-07-23-v4-fromscratch-launch/LAUNCH_CONFIRMED.md`. |
| **Cost** | **MEASURED 59.04 h** (`wallclock_s 212544.6`) on pod2/A40 — vs the ~53 h ESTIMATE this row carried. |
| **Result** | **MEASURED (trainer `metrics.json`, n = 881 val windows):** `ade@2s` **0.5063**, `oracle_ade@2s` 0.1892, `sel_gap@2s` 0.3172, `miss@2m` 0.2145; `canary_ade@2s` **1.1409** from a random-init baseline of **15.674** (the co-evolution signature). ⚠️ **These are the TRAINER's numbers, not `eval_flagship_v4.py`'s — not quotable against v1.** The decision-grade read at step 15 000 is `ADE@2s 0.5839 [0.4962, 0.6821]`, **separated BEHIND v1** by `+0.1568 [+0.0630, +0.2504]` (`RETRACTION_LOG` 07-25). |
| **Code state** | `--from-scratch` / `--trunk none` in `stack/scripts/train_flagship_v4.py`; `from_scratch: true`, `trunk.ckpt: null`, `trunk.step: -1` in the run's own `config.json`. As launched: `steps 30000, batch 16, accum 4 (eff 64), lr_head 1e-4, lr_trunk 1e-4, warmup 2000, lam_mult_floor 0.25, phase_a 2000, phase_b 8000`. |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship-v4-fromscratch/` — `ckpt.pt` (3.24 GB, step 29999) + `ckpt_step{5,10,15,20}000.pt`; gated HF `Sayood/flagship-v4-fromscratch`. |
```

*(Root-cause class for `RETRACTION_LOG`: **a registry row whose state was pre-registered before the run
and never advanced by the launch.** The launch itself was recorded correctly in `LOOP_STATE.md` and in
four program reports — the registry, which is the ONLY quotable source, was the one place that missed
it. A pre-registration row and a status row should not be the same row.)*

---

## 6. THE FINAL v5 LAUNCH COMMAND

**Emitted by the trainer's own `--print-launch` on pod2 against the REAL caches, not written from
memory. `PREFLIGHT: OK`; both parity lines VERIFIED** (train 2400 clips `e61a04553df5…`, val 600 clips
`0b176d2e5cb4…`, skip-hash `f09e44db`).

```bash
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack
export OMP_NUM_THREADS=6
python3 scripts/train_flagship_v4.py \
  --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-lru 64 --v2-subframe 176x624 \
  --frame-h 256 --frame-w 640 --frame-hfov 120.0 --projection cylindrical \
  --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
  --out /workspace/experiments/flagship-v5-w120-rigclean-30k \
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 \
  --strategic full --long-horizon-k 50 \
  --steps 30000 --gate-step 10000 --batch 16 --accum 4 \
  --lr-head 0.0001 --lr-trunk 0.0001 --lam-mult-floor 0.25 --warmup 2000 \
  --workers 8 --eval-every 500 --save-every 1000 --eval-episodes 40 --rollout-k 4 \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 --heldout-patience 2 \
  --heldout-stride 8 --heldout-nboot 2000 --heldout-goal dropped \
  --seed 0 --device cuda --from-scratch --require-parity
```

⚠️ **`--heldout-goal dropped` is the only new token, and it is MY default, not the PI's.** Swap it for
`band0` or `produced` and nothing else in this command changes.
⛔ **I did not run it.** ⛔ It must not go on **pod1** (training).

**What has to be true on the target pod before this is fired** — checked here, not assumed:

| precondition | state |
|---|---|
| the `--heldout-gate` crash | ✅ **closed** (§2) |
| the warp-geometry blocker | ✅ closed at `d7d76bd` by the renderer stream; re-confirmed live — the probe re-rendered through `176x624f305.5775cyl`, not the deployed pinhole |
| `--v2-subframe 176x624` required (rig fingerprint) | ✅ present; preflight BLOCKS the launch without it |
| the wired code on the pod | ⚠️ **`/workspace/TanitAD/stack` on pod2 is at an OLD commit** (`0f93b98`). My changes were staged in the repo and copied to `/workspace/v5gate/stack`. **Sync `/workspace/TanitAD/stack` (or launch from `/workspace/v5gate/stack`) or the run gets the crashing gate back.** |

---

## 7. SUITES — zero new skips

| suite | brief's baseline | after | new skips |
|---|---|---|---|
| `stack/` (dev box) | 1509 passed, 12 skipped | ✅ **1523 passed, 12 skipped** | **0** |
| `taniteval/` (dev box) | 638 passed | ✅ **644 passed** | **0** |

New: `stack/tests/test_heldout_gate_goal_wiring.py` (**13**) + `taniteval/tests/test_ci_rendering.py`
(**6**); `test_heldout_gate_real_head.py` went 5 → 6 as the tripwire was **inverted rather than
deleted**, exactly as its own docstring instructed. `test_vtband_options.py` stayed 20 with two
assertions inverted and one made AST-precise. **No test was deleted, and nothing was converted to a
skip.**

⚠️ `taniteval` reads **644, not 638**: it grew by 32 between `8e3491f` and `98347e5` from the
renderer-geometry stream's `test_warp_geometry.py`, then by 6 from mine. `stack` likewise picked up
`+13` from me on top of the brief's baseline.

---

## 8. Deliverable manifest

⭐ **STATED PLAINLY, as the brief requires: `dropped` is a default chosen BY ME (the wiring stream),
not by the PI, and it is overridable in exactly one flag — `--heldout-goal band0|produced` — or by
changing the single constant `heldout_gate.GOAL_OPTION_DEFAULT`. Nothing else in the code or the launch
command changes.** The string `GOAL_OPTION_PROVENANCE` says so in the trainer's stdout at launch and in
**every probe record's JSON**, so a reader of a v5 log can never mistake it for an adjudicated choice.

| artifact | where it lives | only one place? |
|---|---|---|
| `VTBAND_WIRING.md` (this) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-vtband-wiring/` (staged) | no |
| ⭐ `stack/tanitad/train/heldout_gate.py` — the wiring: `(states, v0)` protocol, `goal_option`, `probe` builds the fn | `repo:stack/` (staged) + `pod2:/workspace/v5gate/stack/` | no |
| ⭐ `stack/scripts/train_flagship_v4.py` — `--heldout-goal`, `goal_head` forwarded, preflight blocks, `--print-launch` | `repo:stack/` (staged) + `pod2:/workspace/v5gate/stack/` | no |
| `stack/tanitad/train/heldout_goal.py` — no longer inert; scoped `dropped` refusal; shim | `repo:stack/` (staged) + `pod2:` | no |
| ⭐ `stack/tests/test_heldout_gate_goal_wiring.py` — **13 tests, NEW** | `repo:stack/` (staged) | no |
| `stack/tests/test_heldout_gate_real_head.py` — the tripwire **INVERTED**, 6 tests | `repo:stack/` (staged) | no |
| `stack/tests/test_vtband_options.py` — 20 tests, 2 assertions inverted, 1 made AST-precise | `repo:stack/` (staged) | no |
| ⭐ `taniteval/taniteval/ci.py` — `_render_bounds` + adaptive rendering | `repo:taniteval/` (staged) | no |
| ⭐ `taniteval/tests/test_ci_rendering.py` — **6 tests, NEW**, fail on the old rendering | `repo:taniteval/` (staged) | no |
| `code/gate_stop_proof.py` — the stop driver (shipped-gate-only) | `repo:…/code/` (staged) + `pod2:/workspace/vtwiring/code/` | no |
| `raw/heldout_gate_probes_v5config.json` — ⭐ **the probe records from the real v5 run** | `repo:…/raw/` (staged) + `pod2:/workspace/v5gate/run/flagship-v5-WIRING-dropped/train_log.jsonl` | no |
| `raw/v5_wiring_dropped.log` — the run transcript incl. the launch banner | `repo:…/raw/` (staged) + `pod2:/tmp/` | no |
| `raw/gate_stop_proof_v4fs15k.json` — the direction-checked stop proof | `repo:…/raw/` (staged) + `pod2:/workspace/vtwiring/raw/` | no |
| `raw/ci_rendering_test_PREFIX_vs_POSTFIX.txt` — the tests failing on `8e3491f`'s `ci.py` | `repo:…/raw/` (staged) | no |
| `raw/v5_launch_print_launch.txt` — `--print-launch` on pod2, `PREFLIGHT: OK` | `repo:…/raw/` (staged) | no |
| `raw/registry_v4fromscratch_artifact_evidence.txt` — the §5 artifact reads | `repo:…/raw/` (staged) | no |
| `pod2:/workspace/v5gate/launch_wiring_proof.sh` | pod2 only | ⚠️ **yes** — a 20-line reproduction wrapper around `launch_smoke2.sh`; the parameters it sets are all in §2, so nothing unreproducible is stranded |

**I ran no `git commit`, no `git push`, and switched no branch.** I `git add`ed only my own paths.

### 8.1 ⚠️ THE INDEX-SWEEP HAZARD FIRED AGAIN, MID-SESSION — third occurrence

`HEAD` advanced from `8e3491f` to **`98347e5`** *("renderer geometry FIXED…")* **while this stream was
working**, and that commit **swept every one of this stream's staged files into it**:

```
stack/scripts/train_flagship_v4.py            |  71 +-      taniteval/taniteval/ci.py                  | 105 +-
stack/tanitad/train/heldout_gate.py           | 149 +-      taniteval/tests/test_ci_rendering.py        | 173 ++
stack/tanitad/train/heldout_goal.py           |  94 +-      …/2026-07-27-vtband-wiring/code/gate_stop_proof.py | 280 ++
stack/tests/test_heldout_gate_goal_wiring.py  | 287 ++      …/2026-07-27-vtband-wiring/raw/ci_rendering_… |  30 ++
stack/tests/test_heldout_gate_real_head.py    | 204 +-      stack/tests/test_vtband_options.py          |  61 +-
```

✅ **Nothing is stranded** — every file is in the repo, which is what `AGENT_OPERATING_STANDARD` rule 1
and operating-standard rule 3 actually require. ⚠️ **But the attribution is wrong**: the entire
`vt_band` wiring and the `ci.py` estimator fix are now committed under a renderer-geometry message, and
`git log --grep` will not find them. *(That commit's own body records that `8e3491f` did the same thing
to the renderer stream in the other direction — so this is the **third** occurrence, and the two
streams have now swept each other.)* **Findable by path, not by message:** the files above.

⛔ I did not attempt to re-commit or split anything: `git commit -- <pathspec>` **segfaults on this
repo** (CLAUDE.md), and a corrective `--amend` would re-open the whole index and repeat the failure.
🔴 **This is for the orchestrator, not for me to fix.**

Also present in the tree and not mine: `.claude/settings.local.json` (harness) and an untracked `4}` in
the repo root.

⚠️ **A second git hazard, recorded because it cost me time and is not in CLAUDE.md:** two files written
into a **newly created directory** were **invisible to git** — `git status -uall`, `git ls-files
--others` and `git add <exact path>` all silently ignored them while `ls` showed them and `git add`
even printed its CRLF warning for them. Renaming the parent directory did **not** clear it. What worked
was **rewriting the file through a fresh inode from Bash** (`cp out; rm; cp back`). *(Root-cause class:
a staging failure that reports success — the most dangerous shape, because "I staged it" was true and
"it is staged" was not. Verify with `git ls-files --cached`, not with the exit code.)*

🔴 **INTEGRATION NEEDED (`AGENT_OPERATING_STANDARD` rule 3), all three stated here and not in a README:**

1. ⭐ **`MODEL_REGISTRY.md` §1.5.5 is FALSE and I did not edit it.** Replacement text ready to paste in
   §5.1, plus the `RETRACTION_LOG` root-cause class. **It is the arm every selection experiment this
   week measured on**, so this is the highest-blast-radius item in this report.
2. **Two other streams' committed artifacts carry a contradictory interval rendering** (§4.3) — they
   must be re-rendered from their per-window data by whoever owns them; I named them and did not
   restate their numbers.
3. **pod2's `/workspace/TanitAD/stack` is at `0f93b98`** and does not contain any of this. The v5
   launch must come from a synced tree or it gets the crashing gate back (§6).

🔒 **Confidentiality swept, not assumed:** every staged `code/` and `raw/` file was scanned for
UUID-shaped clip ids — see §9. Counts, gates, digests and step numbers only.

---

## 9. Provenance of every number, and the ops constraints

| claim | class | source |
|---|---|---|
| `--heldout-gate` REACHES and PASSES a probe on the real v5 config/caches; `primary_value 0.265049`, 264 w / 4 ep, frame `176x624f305.5775cyl` | **MEASURED (ours)** | `raw/heldout_gate_probes_v5config.json`, `raw/v5_wiring_dropped.log` (pod2) |
| the trainer's stop path executes end-to-end: probes 20/40/60, streak 2, `EARLY_STOP true`, `ckpt_best.pt` at the incumbent | **MEASURED (ours)** | same |
| ⚠️ that stop's cause is training dynamics, **direction NOT pre-verified** | **MEASURED, and explicitly NOT a discrimination claim** | same; §2.1 |
| the stop fires on a **direction-verified injected** degradation, trained head | **MEASURED (ours)** | `raw/gate_stop_proof_v4fs15k.json` (pod2), §3 |
| the pre-fix `ci.py` prints `delta=0.0 [0.0, 0.0] separated=True` | **MEASURED (ours)** | `raw/ci_rendering_test_PREFIX_vs_POSTFIX.txt` — `git show HEAD:taniteval/taniteval/ci.py` in a scratch tree |
| 4 committed nodes in 3 artifacts carry that rendering; 264 tracked JSONs scanned | **MEASURED (ours)** | §4.3; scan over every tracked `*.json` containing `"separated"` |
| `flagship-v4-fromscratch` reached step **29999**, `rc=0`, 59.04 h, launched 2026-07-23T21:54:44Z | **MEASURED (ours, from the artifact)** | `raw/registry_v4fromscratch_artifact_evidence.txt`; `pod2:/workspace/experiments/flagship-v4-fromscratch/{metrics,config}.json`, `supervisor.log`, `train_log.jsonl` |
| `goal_dropout = 0.5` ships and the trainer never assigns it | **MEASURED (ours)** | AST scan pinned by `test_VT_DROPPED_is_IN_DISTRIBUTION_because_goal_dropout_ships_at_half` |
| the final v5 launch command; `PREFLIGHT: OK`; parity VERIFIED both caches | **MEASURED (ours)** | `raw/v5_launch_print_launch.txt` — the trainer's own `--print-launch` on pod2 |
| `band0` costs −0.0621 [−0.0878, −0.0371]; `dropped`'s degradation range −0.4157; `produced` −0.0154 | **INHERITED** (`VTBAND_DECISION.md`, arm `flagship-v4-fromscratch` @15000) — **NOT re-derived here**, and none of it decided the wiring | that doc §0/§5 |
| the ~29.5 GPU-h the gate exists to save | **INHERITED** (`flagship-v5-retrain.PREP.md` cause #1) | not re-derived |
| the two mis-directed degradations (lateral drift; slow-plan +0.1698) | **INHERITED** (the brief + `VTBAND_DECISION.md` §5.1) — used only to *design* my direction check, never quoted as a result | — |
| v1's 0.4271 | ⛔ **NOT USED.** It is `wm_fidelity_ade_2s` — what the world model scores when handed the TRUE actions. Nothing here is held to it | — |
| pod1 is training; pod3 is idle with YouTube blocked | **NOT PROBED** | neither was contacted this session |

**Ops constraints — checked, not assumed:**

| constraint | how | result |
|---|---|---|
| ⛔ pod1 TRAINING — do not touch | never contacted at all | ✅ untouched |
| ⛔ pod3 / YouTube (D-B authorization SPENT) | never contacted at all | ✅ untouched |
| ⛔ never `df` | pod2 disk judged by a **real `dd`** write test only | ✅ |
| ⛔ never `pkill -f` | nothing was killed; both pod2 jobs ran to completion | ✅ n/a |
| ⛔ do NOT launch the real v5 run | the only pod2 jobs were a **65-step** proof and a read-only probe driver | ✅ |
| ⛔ parity sacred | nothing re-selects episodes; both parity lines VERIFIED by the trainer itself; the gate's `val_eps[:n]` prefix is the trainer's own | ✅ |
| `PYTHONPATH` / `OMP_NUM_THREADS=6` | set on every pod2 invocation | ✅ |
| no eval on a training pod | pod2 was idle (0 MiB) before each job and jobs were run **sequentially**, never concurrently | ✅ |

⚠️ **On timing:** no wall-clock comparison is quoted anywhere, deliberately — nothing here depends on a
duration, and the two pod2 jobs ran against differently-warmed page caches.
