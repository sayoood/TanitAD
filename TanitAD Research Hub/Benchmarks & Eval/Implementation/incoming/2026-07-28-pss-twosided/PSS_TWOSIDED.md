# THE EGO GUARD IS WIRED AND FIRES — AND THE TWO-SIDED PSS MOVES **11 OF 16 ARMS**, INCLUDING A VERDICT FLIP ON THE FLAGSHIP ITSELF

**Wall-clock date:** 2026-07-27 (Europe/Berlin). ⚠️ The directory is dated `2026-07-28` because the
brief named it so; the repo's narrative clock runs ~1 day ahead of wall-clock (known artefact).
**Stream:** Benchmarks & Eval · **Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `2b0f166`
**Source of both jobs:** `…/2026-07-28-tactical-action-input/TACTICAL_ACTION_INPUT.md` escalations **E1** and **E2**.
**Hosts:** dev box only (every number here is CPU arithmetic over committed dumps).
⛔ **pod1 was READ ONLY** (one `ps`/`nvidia-smi`); pod2, pod3 and `tanitad-eval` were not touched.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ✅ **THE E1 GUARD IS CALLED, AND IT FAILS ON THE LIVE BUG.** `tanitad.ego_plan.assert_ego_is_fed` existed, had 6 tests, and had **never been called**. It is now called at **every** planner call site in the shipped eval surface (`closedloop`, `planning`, `planner_p2`, `corpus_overlay`, `probe_overlay`) **and at the pseudo-simulation surface v5 is gated on** (`pseudo_evaluate`, beside the envelope assertion, **before any model is touched**). Demonstrated failing: `artifacts/guard_demo.json`. | `MEASURED` **tier 1** |
| **2** | ⭐ **AND THE BUG HAS TEETH — with a number.** On a trained-like 4-brain, dropping `ego=` shifts the tactical waypoints **0.1125 m mean / 0.2840 m max** and the strategic `ctx` by **L2 0.8994**. ⇒ scoring an ego-trained checkpoint ego-blind measures **a different model**, silently. | `MEASURED` **tier 1** |
| **2b** | ⚠️ **AND THE FIRST RUN OF THAT DEMO MEASURED 0.0 — I NEARLY PUBLISHED "THE BUG IS COSMETIC".** On a *freshly built* brain the shift is **exactly 0.0** because `FiLM.to_scale_shift` is zero-initialised (`predictor.py:25-26`), so the whole `cond` path is numerically dead at init. The source report's **E6** hit in practice, inside the instrument written to test E1. Both numbers are published. | `MEASURED` **tier 1** |
| **3** | **DECISION: REFUSE, not warn-and-record** — with an explicit, *semantic* escape hatch (`ego = 0`, which is a real ablation and in-distribution under `v2_ego_dropout`) and a non-free `warn` mode that **stamps `ego_input_DROPPED = True` into the emitted node**. Justified in §2.2 against four failure modes this program has already paid for. | decision |
| **4** | ⭐⭐ **THE TWO-SIDED TERM SEES WHAT THE PUBLISHED ONE COULD NOT: the source stream's own primary goes `+0.0078 [−0.0110, +0.0260] n.s.` → `+0.1121 [+0.0890, +0.1342] SEPARATED`** — a **14.4× larger** point estimate on the **same 15,981 rows** with the **same estimator**. ⇒ pre-registered **CONFIRM**. | `MEASURED` **tier 2** |
| **5** | ⛔⛔ **THE MOST CONSEQUENTIAL FLIP IS ON THE FLAGSHIP, NOT ON THE PROBE.** `v1_tactical_follow − cv_holdv0` was **−0.0178 [−0.0368, +0.0022] n.s.** — v1's deployed tactical plan was statistically **TIED** with holding `v₀`. Under `twosided_v2` it is **−0.1212 [−0.1418, −0.0991] SEPARATED-WORSE**. **The claim "the flagship's planner is at least as good as doing nothing" was an artefact of a metric that cannot see over-travel.** | `MEASURED` **tier 2** |
| **6** | ⭐ **THE RANKING CHANGES — 11 of 16 arms move, the largest by 6 places** — but ⭐ **`cv_holdv0` STILL RANKS FIRST AMONG REALISABLE ARMS, UNDER EVERY TERM TESTED** (`w` = 0, 0.5, 1, 1.5, 2, 3). Rank 1 overall is `v1_ego_oracle_lon` under every term including the published one — an **ORACLE**, so the panel's headline is unchanged by the fix. **The headline survives; the sub-claim under it does not (§5).** | `MEASURED` **tier 2** |
| **7** | ⭐⭐ **THE METRIC NO LONGER REFUSES A SOLVED PROBLEM, AND NO LONGER REWARDS DRIVING TOO FAST.** `oracle_lon_straight` (a straight plan walked to the **true** distance) was **INADMISSIBLE** under `clamp_v1` (`observed_range` 0.0219 < `RANGE_MIN` 0.05); it is now admissible and scores **0.6032 — the highest value in the entire panel**. And `v1_ego_double` (a **1.97×** over-travelling plan) goes from `ego_progress` **0.9866 — the panel's HIGHEST** — to **0.1164**. | `MEASURED` **tier 1** |
| **8** | ✅ **THE REPRODUCTION GATE IS EXACT: `max\|diff\| = 0.000000` across 16 published `clamp_v1` composites.** Every published PSS number recomputes bit-for-bit from these dumps under `progress_term="clamp_v1"`. The old metric is not merely "documented" — it is *still running*. | `MEASURED` **tier 1** |
| **9** | ⛔ **THE PER-ARM GATE IS REFUSED OUTRIGHT, not "available".** MEASURED here, not inherited: `comfort` mean is **0.0004** for `v1_tactical_follow` and **0.2882** for `v1_ego_v0` — **720×**, between two arms that differ only in *schedule*. My recompute script does not compute it. | `MEASURED` **tier 1** |
| **10** | ✅ **pod1 IS UNAFFECTED, VERIFIED NOT ASSUMED.** `train_flagship4b.py` imports **neither** `taniteval` nor `heldout_gate` nor `pseudosim` — proved by importing it against the modified tree and enumerating `sys.modules` (**leak set = `[]`**). PID **699286** still training `flagship-v2corpus-30k` at **2 d 10 h 29 m**, GPU 62 %. | `MEASURED` **tier 1** |

### 0.1 Pre-registered outcome

**CONFIRM.** Both conditions fired, measured on the same rows:
* *"the two-sided term detects the over-travel the blind term missed (the source stream's intervention now separates)"* — **+0.1121 [+0.0890, +0.1342] SEPARATED**, from **n.s.**
* *"and the guard fires on a real train/eval mismatch"* — **`EgoInputDropped` raised**, with the failing value published in `artifacts/guard_demo.json` and 8 tests that exercise the failing direction.

⚠️ **The REFUTE branch was live and did not fire, but half of it was right:** the brief's REFUTE read
*"the two-sided term does not change the ordering **or** the intervention's verdict"*. The intervention's
verdict changed decisively. **The top of the ordering did not** — `cv_holdv0` still ranks first among
realisable arms at every `w` I tested. So: **the blindness is real, decision-relevant for the
*flagship's* verdict, and NOT decision-relevant for the "nothing beats holding v₀" headline.**

### 0.2 Tier

**Tier 1** for §1–§3 (code, guard, term arithmetic — deterministic, CPU, model-free).
**Tier 2** for every composite number in §4–§6, inheriting all four of the panel's qualifiers verbatim
(**oracle goal where used**, **non-reactive log replay**, **no collision or TTC gate**, **`comfort`
dropped by the gate, not retuned**) plus the source stream's two (Block B arms are **plan transforms,
not trained planners**; two arms are **oracles**). ⛔ **`PSS` is NOT a Driving Score** and is not
compared to one.

---

## 1. ⭐ PRIORITY 1 — THE EGO GUARD, WIRED

### 1.1 What was actually broken

`TacticalPolicy.forward` and `StrategicPolicy.forward` guard the ego term with a **two-condition**
gate (`fourbrain.py:330`, `:78`):

```python
if self.ego_emb is not None and ego is not None:      # build-time AND call-time
    ctx = ctx + self.ego_emb(ego.to(ctx.dtype))
```

The build-time half was audited repeatedly and correctly reported off. **The call-time half was never
audited**: `ego=` is passed at exactly **three** call sites in the repo — `flagship_losses.py:245,
246, 351` — **all three in the trainer**. So a checkpoint trained *with* the lever is evaluated
*without* it, with no error and no log line.

⛔ **This is live.** pod1, read-only, at the time of writing:

```
699286  2-10:29:45  python3 -u scripts/train_flagship4b.py --v2-cache …
        --config flagship4b --v2 --sigreg-free-dims 64 --steps 30000 …
        --out /workspace/experiments/flagship-v2corpus-30k
```

`--v2` sets `v2_ego_to_planners = true`. `MEASURED`, tier 1.

### 1.2 ⭐ The bug has teeth — and the first measurement of that said it did not

`artifacts/guard_demo.json`, regenerable in seconds on CPU by `scripts/demo_guard_fails.py`:

| brain state | mean waypoint shift (ego fed vs dropped) | max | terminal | `ctx` L2 shift |
|---|--:|--:|--:|--:|
| ⚠️ **freshly built** (shipped zero-init FiLM) | **0.000000 m** | 0.000000 | 0.000000 | **0.000000** |
| ⭐ **trained-like** (FiLM non-zero) | **0.112516 m** | **0.283979** | 0.093695 | **0.899396** |

⚠️ **I ran the fresh-build version first and it returned 0.0 on every axis.** The cause is
`FiLM.to_scale_shift`'s zero-init (`predictor.py:25-26`): on a fresh brain the entire `cond` path —
`ctx` *and* any ego graft — is numerically dead. **The source report's E6, hit in practice, inside
the instrument written to demonstrate E1.** Had I published only that run I would have reported
*"the ego port has no effect, the escalation is cosmetic"*. Both rows are in the artifact.

⇒ On the state pod1's arm will actually be in, **dropping the ego measures a different model.**

### 1.3 Where the guard now sits

| file | call sites | ego SOURCE | can it feed? |
|---|---|---|---|
| `taniteval/taniteval/closedloop.py` | `closed_loop_rollout`, `open_loop_plan_rollout` | observed pose at tick 0, then **the simulated bicycle's own state** | ✅ |
| `taniteval/taniteval/planning.py` | `run` (2 strategic + 1 tactical) | observed poses `t`, `t−1` | ✅ |
| `taniteval/taniteval/planner_p2.py` | `head_action_seed`, `collect` | observed poses `t`, `t−1` | ✅ |
| `taniteval/taniteval/corpus_overlay.py` | `episode_rollouts` | observed poses `t`, `t−1` | ✅ |
| `taniteval/probe_overlay.py` | `probe` | observed poses `t`, `t−1` | ✅ |
| ⭐ `taniteval/taniteval/pseudosim.py` | `pseudo_evaluate` — **the v5 surface** | adapter must **declare** | ✅ (adapter contract) |

⭐ **The closed loop's ego is PROPRIOCEPTION OF THE SIMULATED VEHICLE, not a replay of the log.**
From tick 1 the loop plans on imagined latents; the logged pose history stops being the ego's history
the moment the loop deviates, and feeding it anyway would leak the future into a closed-loop arm. So
the speed channel tracks the loop's own `v` and the yaw rate is the previous tick's **executed**
bicycle rate — the identical formula `bicycle_integrate` uses, so the two cannot drift.

⭐ **The pseudo-simulation surface needed a different shape of guard.** `pseudo_evaluate` never
touches the policy brains; it calls an adapter's `.traj(frames, v0, goal)`. So the guard there asks
the **adapter** whether it handled the port: an adapter wrapping a model with trained `ego_emb`
weights must set `planner.ego_provenance`, or it is refused — **beside `assert_grid_in_envelope`,
before any model is loaded**, so a bad run costs zero GPU seconds. That is exactly the `panel_run.py`
shape that would have scored `flagship-v2corpus-30k` ego-blind.

### 1.4 ⚠️ MEASURED correction to the E1 census: 7 call-site files, not 8

`blindimag.py:101` is listed in the source report, in `ego_plan.py`'s docstring and in the guard's own
docstring as a call site. **It is inside the module docstring.** `blindimag` takes an injected
`plan_fn` and never touches the policy brains. Pinned in `_EXEMPTIONS`.

### 1.5 ⚠️ Two call sites are DELIBERATELY unguarded, and the list is tested

| site | why not guarded |
|---|---|
| `stack/tanitad/refs/refa.py:260` | `run_hierarchy(self, states, actions, nav_cmd)` forwards `ego=None`. **`stack/` MODEL code**, not an eval path; no REF-A checkpoint has ever set `v2_ego_to_planners`. ⇒ **E-A below.** |
| `stack/tanitad/models/fourbrain.py:614` | `propose_and_score`'s tactical call (the P2 candidate scorer). Same reason — and **pod1 is mid-run on a trainer that imports this exact file**. ⇒ **E-A below.** |

`test_the_exemption_list_is_a_decision_not_an_oversight` pins the list **and its length**, so adding a
hole is a decision someone has to write down. A guard with undocumented holes is the C13 defect one
level down.

### 1.6 ⛔ THE GUARD DEMONSTRATED FAILING — every rule, in both directions

`taniteval/tests/test_ego_guard.py`, **23 tests**. The failing direction is the point of the file:

| # | rule | ⛔ the value that makes it FAIL | ✅ the value that makes it pass |
|:--:|---|---|---|
| 1 | the guard | ego-trained policy + `ego=None` ⇒ **`EgoInputDropped`**, message names the call site | any non-`None` ego |
| 2 | the adapter hook | adapter over an ego-ckpt with no `ego_provenance` ⇒ **`EgoInputDropped`** | a declaring adapter |
| 3 | end-to-end | `pseudo_evaluate(undeclared_adapter, …)` ⇒ **raises before `.traj` is reached** (`.traj` raises if reached, so the refusal *proves* no model was touched) | a plain checkpoint |
| 4 | mode parser | `TANITEVAL_EGO_GUARD=wrn` ⇒ **`ValueError`**, never a silent fall-back to permissive | `refuse` / `warn` |
| 5 | warn mode | must still set `ego_input_DROPPED = True` in the node | — |
| 6 | ⭐ the coverage scan | a synthetic `model.tactical_policy(states, ctx)` line ⇒ **the scan flags it** (`test_the_guard_scan_itself_can_fail`) | `…, ego=ego)` |
| 7 | scan not vacuous | if a scanned path stops resolving the scan would go green over nothing ⇒ **`test_repo_layout_assumption_holds`** | — |
| 8 | pod1 isolation | `train_flagship4b.py` containing `taniteval` / `heldout_gate` / `pseudosim` ⇒ **fails** | it contains none |

⭐ **Rule 6 is the one that matters most.** The 2026-07-28 finding was not that a guard was missing —
`assert_ego_is_fed` existed and was tested. It was that a two-condition gate had been audited at one
condition, so **every** call site silently satisfied the false half. A source scan that fails when a
new planner call site forgets `ego=` is the only control that survives the next author.

---

## 2. The decision: **REFUSE**, with a semantic escape hatch

### 2.1 What ships

`taniteval.ego_guard.guard_mode()` returns `refuse` unless `TANITEVAL_EGO_GUARD=warn` is set
explicitly. An unrecognised value **raises** — a typo'd env var must never be the thing that disables
a guard.

### 2.2 Why refuse and not warn-and-record

| # | the argument | the failure it prevents, from this program's own log |
|:--:|---|---|
| 1 | **A warning still produces a number, and a number that exists gets quoted.** Refusal produces nothing to quote. | The harm is not a crash; it is a *silently wrong attribution* of a GPU-week arm as *"the ego lever does nothing"*. |
| 2 | **"Recorded somewhere" is a known-failed control here.** | An integration request in a README sat unread **10 days**; a stale absence-claim propagated into **≥7 documents**. A stderr line in a multi-hour eval log is weaker than both. |
| 3 | **Refusing costs ZERO published numbers.** | Every arm in the 2026-07-27 panel and every `MODEL_REGISTRY` checkpoint has `ego_emb is None`, so the guard is a **provable** no-op for all of them (tested). It can only fire on the new capability class. |
| 4 | **The escape hatch is better than a warning because it is SEMANTIC.** | An ego-ablated arm is a legitimate experiment — you get it by passing `ego = 0`, which is **in-distribution** when the run used `v2_ego_dropout` and is a *different object* from `ego=None` (that skips the `ego_emb` bias too). Making the ablation say so in code is the point. |

**`warn` mode exists for exactly one case** — a batch re-score of legacy arms where a stop would
strand the whole sweep — and is deliberately not free: explicit opt-in, an `EgoInputDroppedWarning`,
**and `ego_input_DROPPED = True` stamped into the emitted node**, so a warn-mode number is
identifiable as ego-blind **from the JSON alone**.

---

## 3. ⛔ PRIORITY 2 — THE TWO-SIDED TERM

### 3.1 The versioning, which is the load-bearing half

⚠️ **This changes the value of every published PSS number, so the fix is a new metric ID, not a new
definition under the old name.**

```python
PROGRESS_TERM_PUBLISHED = "clamp_v1"        # clamp(r, 0, 1) — EVERY pre-2026-07-28 number
PROGRESS_TERM_DEFAULT   = "twosided_v2"     # clamp(clamp_v1(r) - w*max(r-1,0), 0, 1), w = 1
metric_id(term) -> f"PSS_recovery_progress@{term}"
```

* The composite's `name` **is** the versioned ID. A reader of `name` cannot get the version wrong.
* `emit()` stamps `metric_id` and `progress_term`; `score_windows()` returns `_progress_term`;
  `composite()` **infers the term from the scores it was handed** rather than guessing.
* `heldout_gate.PRIMARY_NAME` — *"named, so it can never be quietly swapped"* — is now
  `pseudosim_composite_PSS_recovery_progress@twosided_v2`, with
  `PRIMARY_NAME_PUBLISHED_THROUGH_20260727` kept beside it, and `HeldoutGateConfig.progress_term`
  lets a gate reproduce the old one exactly.
* An unknown term **raises `UnknownProgressTerm`** and never falls back — a typo must not silently
  produce a number under the wrong ID.

⭐ **The default flips to the fixed term, deliberately.** A capability that must be passed explicitly
is a capability nobody passes — *that is the E1 bug found by the same report*. The versioned name,
not a frozen default, is what prevents a silent redefinition.

### 3.2 ⭐ The shape, and why it is this one

```
under = max(1 - r, 0)        over = max(r - 1, 0)        r = plan_along / human_along
score = clamp( clamp_v1(r) − w·over , 0, 1 )                     w = OVER_TRAVEL_WEIGHT = 1.0
```

1. ⭐ **It is a STRICT REFINEMENT.** For every `r ≤ 1` it is **BIT-identical** to `clamp_v1` — not
   equal to tolerance, *bit*-identical. Written as *"the published term minus an over-travel charge"*
   rather than as `1 − |1 − r|` **on purpose**: `1 − (1 − r)` is not `r` in float32, and my first
   implementation drifted by **2.5 × 10⁻⁴** on the 36 % of rows that under-travel, which would have
   crept into every paired delta between the two terms. Caught by the test, then designed out.
   ⇒ the under-travel half of the published term is preserved exactly and the change is **purely
   additive information** on the side the old term could not see.
2. **Zero-parameter at `w = 1`**, piecewise-linear, so the fix cannot be accused of being tuned to
   produce a ranking.
3. **The NaN mask is untouched** (`human > 0.5 m`), so both terms score the **same rows** — otherwise
   no paired delta between them is valid. Pinned.
4. **`recovery` is bit-identical between terms** (checked: `cv_holdv0` 0.077644 both,
   `v1_tactical_follow` 0.078493 both). The change touches `ego_progress` and nothing else.

⚠️ **Two-sided does NOT automatically mean symmetric, and I am not claiming it does.**
Over-travel — planning through space the car will not have — is *plausibly* more dangerous in driving
than under-travel. But **this surface has no collision gate and no cuboids**
(`COLLISION_UNAVAILABLE_REASON`), so danger is **not measurable here**, and any `w ≠ 1` would be an
assumption dressed as a measurement. ⇒ the default is the **minimum-assumption `w = 1`**, and the
ranking's sensitivity to `w` is **published (§6) rather than chosen**.

### 3.3 ⛔ Every rule can return a FAILING value — demonstrated

`taniteval/tests/test_pseudosim_progress_term.py`, **18 tests**:

| rule | ⛔ the value that makes it FAIL |
|---|---|
| the published term is BLIND | a 2× over-travelling plan scoring **1.0000** — identical to a perfect one (this is the *pass* condition for reproducing `clamp_v1`, and the *failing* value of the old metric) |
| the two-sided term SEES it | the same rows scoring **0.0000** |
| strict refinement | any `r ≤ 1` where the two terms differ by a single bit |
| the terms diverge above 1 | `max(clamp_v1 − twosided) ≠ 1.0` |
| the slope grid varies `w` | an over-travel slope ≠ `w` at any grid point |
| range | any term leaving `[0, 1]` on `r ∈ [−5, 20]` |
| versioning | `metric_id("clamp_v1") == metric_id("twosided_v2")` |
| the parser | `progress_term="twosided"` (a typo) silently returning a number instead of `UnknownProgressTerm` |
| row-set identity | the NaN mask differing between terms |

---

## 4. ⭐⭐ THE PANEL, RECOMPUTED — BOTH TERMS, SIDE BY SIDE

**20 arms · 40 val episodes · stride 8 · 21 grid points · 15,981 rows per arm · 0 rollout steps.**
Row identity `(ep_i, anchor, dlat, dyaw, dlon)` **asserted** across all 20 — a non-matching arm is
**refused**, not dropped; **0 refused**. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`,
**B = 2000, unit = val episode**. ⛔ `overlapping_holdout_se` appears nowhere.
⛔ **PANEL-WIDE gate** (`ego_progress` + `recovery` admitted, `comfort` dropped for **every** arm,
under **both** terms). Probes (`stand_still`, `v1_ego_half`, `v1_ego_double`, `oracle_lon_straight`)
are scored and reported but excluded from the gate and the ranking.
**One corpus** (`physicalai` val, 40 episodes) — nothing is pooled.
⚠️ **`selected_frac` does not exist on this surface**: there is no selection step, every arm is scored
on all 15,981 rows, so `selected_frac = 1.000` for every arm by construction. Every REF-C arm here is
`refc_nav_mode = produced` — **the model's own route head, image-only, no oracle nav, not gated**.

### 4.1 ✅ The reproduction gate — EXACT

| | |
|---|---|
| published `clamp_v1` composites checked | **16** |
| **max \|diff\|** | ⭐ **0.000000** |
| verdict | ✅ **PASS** |

Every published PSS value — `cv_holdv0` 0.5705, `v4_oracle` 0.5622, `refc_xl_produced` 0.5499,
`v1_tactical_follow` 0.5471, `v1_ego_v0` 0.5608, `v1_ego_oracle_lon` 0.5946, `refc_base_v0off`
0.4980, `v4_blind` 0.3749, … — recomputes **exactly** from these dumps under
`progress_term="clamp_v1"`. **The old metric is not documented; it is still running.**
⇒ nothing below can be blamed on the recomputation.

### 4.2 The panel

*(generated by `scripts/tables.py`; full version in `artifacts/tables.md`)*

| rank@v2 | arm | `PSS@clamp_v1` (PUBLISHED) | `PSS@twosided_v2` (NEW) | Δ | rank@v1 | `ego_progress` v1 → v2 |
|--:|---|---|---|--:|--:|---|
| 1 | ⚠️ `v1_ego_oracle_lon` | 0.5946 [0.5868, 0.6033] | 0.5943 [0.5865, 0.6029] | −0.0003 | 1 | 0.9805 → 0.9799 |
| **2** | ⭐ **`cv_holdv0`** | **0.5705 [0.5558, 0.5844]** | **0.5492 [0.5276, 0.5688]** | −0.0213 | 2 | 0.9407 → 0.9037 |
| 3 ⬆1 | `v1_ego_v0` | 0.5608 [0.5470, 0.5727] | 0.5403 [0.5212, 0.5581] | −0.0205 | 4 | 0.9324 → 0.8970 |
| 4 ⬇1 | `v4_oracle` | 0.5622 [0.5496, 0.5725] | 0.5362 [0.5187, 0.5515] | −0.0260 | 3 | 0.9462 → 0.9000 |
| 5 | `refc_xl_produced` | 0.5499 [0.5421, 0.5566] | 0.5259 [0.5112, 0.5388] | −0.0240 | 5 | 0.9438 → 0.9019 |
| 6 | `refc_xl_v0on` | 0.5499 [0.5421, 0.5566] | 0.5259 [0.5112, 0.5388] | −0.0240 | 6 | 0.9438 → 0.9019 |
| 7 ⬆4 | `refc_base_produced` | 0.5439 [0.5345, 0.5519] | 0.5228 [0.5066, 0.5369] | −0.0211 | 11 | 0.9317 → 0.8951 |
| 8 ⬆4 | `refc_base_v0on` | 0.5439 [0.5345, 0.5519] | 0.5228 [0.5066, 0.5369] | −0.0211 | 12 | 0.9317 → 0.8951 |
| 9 ⬆1 | `refc_small_produced` | 0.5444 [0.5360, 0.5514] | 0.5202 [0.5051, 0.5340] | −0.0242 | 10 | 0.9315 → 0.8894 |
| 10 ⬆4 | ⛔ `refc_xl_v0off` | 0.5166 [0.5047, 0.5271] | 0.4724 [0.4562, 0.4863] | −0.0442 | 14 | 0.8732 → 0.7950 |
| 11 ⬆4 | ⛔ `refc_base_v0off` | 0.4980 [0.4838, 0.5106] | 0.4591 [0.4428, 0.4737] | −0.0389 | 15 | 0.8377 → 0.7692 |
| 12 ⬇4 | `v1_tactical_oracle` | 0.5467 [0.5338, 0.5591] | 0.4305 [0.4026, 0.4571] | −0.1162 | 8 | 0.9047 → 0.6990 |
| **13 ⬇6** | ⛔ **`v1_tactical_follow`** | **0.5471 [0.5340, 0.5595]** | **0.4242 [0.3948, 0.4521]** | **−0.1229** | 7 | 0.9081 → 0.6902 |
| 14 ⬇5 | `v1_lat_straight` | 0.5460 [0.5288, 0.5608] | 0.4209 [0.3878, 0.4524] | −0.1251 | 9 | 0.9138 → 0.6911 |
| 15 ⬇2 | `nospeed_tactical_oracle` | 0.5394 [0.5242, 0.5540] | 0.4179 [0.3872, 0.4472] | −0.1215 | 13 | 0.8961 → 0.6802 |
| 16 | `v4_blind` | 0.3749 [0.3076, 0.4368] | 0.3322 [0.2711, 0.3917] | −0.0427 | 16 | 0.5999 → 0.5252 |

⛔⛔ **The v1 family falls off the panel.** `v1_tactical_follow` **7 → 13**, `v1_lat_straight`
**9 → 14**, `v1_tactical_oracle` **8 → 12**. Under `clamp_v1` the flagship's tactical plan outranked
every REF-C arm at base and small scale. Under `twosided_v2` it sits **below all of them and below
BOTH ego-ABLATED REF-C arms** (`refc_xl_v0off` 0.4724, `refc_base_v0off` 0.4591 vs
`v1_tactical_follow` 0.4242). **A REF-C decoder with its speed input deliberately zeroed now scores
above the flagship's deployed tactical head**, because v1 over-travels on **48.80 %** of windows with
a p95 ratio of **2.430×** and `clamp_v1` charged nothing for any of it.

### 4.3 The pre-registered contrasts

| contrast | Δ`PSS@clamp_v1` (PUBLISHED) | Δ`PSS@twosided_v2` (NEW) |
|---|---|---|
| ⭐ **`v1_ego_v0` − `v1_tactical_follow`** *(the source stream's primary)* | **+0.0078 [−0.0110, +0.0260] n.s.** | ⭐ **+0.1121 [+0.0890, +0.1342] SEP** |
| `v1_ego_v0` − `v1_tactical_oracle` *(replication)* | +0.0082 [−0.0104, +0.0261] n.s. | **+0.1057 [+0.0836, +0.1270] SEP** |
| `v1_ego_v0` − `cv_holdv0` | −0.0100 [−0.0170, −0.0033] SEP | −0.0090 [−0.0160, −0.0023] SEP |
| ⚠️ `v1_ego_oracle_lon` − `cv_holdv0` *(ORACLE)* | +0.0228 [+0.0064, +0.0412] SEP | +0.0438 [+0.0210, +0.0701] SEP |
| ⚠️ `v1_ego_oracle_lon` − `v1_tactical_follow` | +0.0407 [+0.0308, +0.0512] SEP | **+0.1652 [+0.1347, +0.1976] SEP** |
| `refc_xl_v0on` − `refc_xl_v0off` *(Block A, XL)* | +0.0332 [+0.0243, +0.0433] SEP | +0.0534 [+0.0474, +0.0596] SEP |
| `refc_base_v0on` − `refc_base_v0off` *(Block A, base)* | +0.0461 [+0.0354, +0.0579] SEP | +0.0639 [+0.0550, +0.0736] SEP |
| ⛔⛔ **`v1_tactical_follow` − `cv_holdv0`** | ⛔ **−0.0178 [−0.0368, +0.0022] n.s.** | ⛔ **−0.1212 [−0.1418, −0.0991] SEP** |
| `nospeed_tactical_oracle` − `cv_holdv0` | −0.0235 [−0.0430, −0.0037] SEP | −0.1260 [−0.1517, −0.1005] SEP |
| `refc_xl_produced` − `cv_holdv0` | −0.0203 [−0.0303, −0.0097] SEP | −0.0230 [−0.0341, −0.0118] SEP |
| `refc_base_produced` − `cv_holdv0` | −0.0252 [−0.0349, −0.0150] SEP | −0.0251 [−0.0351, −0.0153] SEP |
| `v4_oracle` − `cv_holdv0` | −0.0034 [−0.0138, +0.0078] n.s. | −0.0084 [−0.0188, +0.0029] n.s. |
| `v1_lat_straight` − `v1_tactical_follow` | +0.0006 [−0.0065, +0.0072] n.s. | −0.0020 [−0.0095, +0.0050] n.s. |
| ✅ `v4_oracle` − `v4_blind` *(G1 instrument sensitivity)* | +0.1882 [+0.1240, +0.2557] SEP | +0.2049 [+0.1506, +0.2650] SEP |
| ✅ `v1_ego_half` − `v1_tactical_follow` *(degradation guard)* | −0.2421 [−0.2565, −0.2285] SEP | −0.1189 [−0.1490, −0.0878] SEP |

**Four readings.**

1. ⭐ **CONFIRM.** The source stream's primary is now separated, at **14.4×** the point estimate, on
   the identical rows with the identical estimator.
2. ⛔⛔ **The asymmetry the source report proved is GONE, and it inverts.** Under `clamp_v1` the
   composite separated a **3.36×** degradation (+0.0332) and could not separate a **5.41×**
   improvement (+0.0078). Under `twosided_v2` the improvement (**+0.1121**) is now **2.1× larger**
   than the degradation (**+0.0534**) — the ordering the raw axis says it should be.
3. ⛔⛔ **The flagship's verdict flips.** *"v1's tactical plan is statistically tied with holding v₀"*
   was true only under a metric blind to over-travel. It is now separated-worse by −0.1212.
4. ✅ **Both instrument guards survive the change.** G1 (sighted vs blind) stays separated and gets
   *stronger*; the deliberate half-speed degradation stays separated-worse. The metric did not become
   permissive.

### 4.4 ⭐⭐ The probes — the two defects §5.1/§5.3 of the source report named are FIXED

| probe | what it is | `ego_progress`@v1 | admissible@v1 | `ego_progress`@v2 | admissible@v2 | `PSS@v2` |
|---|---|--:|---|--:|---|--:|
| ⛔ `v1_ego_double` | v1's curve at **2× speed** (1.97× the logged distance) | **0.9866 — the panel's HIGHEST** | ⛔ **INADMISSIBLE** (ceiling 97.49 %) | **0.1164** | ✅ | 0.0954 |
| ⚠️ `oracle_lon_straight` | straight plan walked to the **TRUE** distance | 0.9916 | ⛔ **INADMISSIBLE** (range 0.0219 < 0.05) | 0.9900 | ✅ | ⭐ **0.6032 — the panel's HIGHEST** |
| ⛔ `v1_ego_half` | v1's curve at **half** speed | 0.4864 | ✅ | 0.4837 | ✅ | 0.3100 |
| ⛔ `stand_still` | the published adversary | 0.0000 | ⛔ INADMISSIBLE | 0.0000 | ⛔ INADMISSIBLE | **REFUSED** |

* ⭐ **The metric can now tell "exactly right" from "twice too far".** A 1.97× over-travelling plan
  went from the panel's *highest* `ego_progress` to **0.1164**.
* ⭐ **The metric no longer refuses a solved problem.** §5.3 of the source report: an arm that gets
  the axis right was *refused by the gate that exists to keep the axis meaningful*. Under
  `twosided_v2` `oracle_lon_straight` is admissible and scores **0.6032 — above every other arm in
  the study**, which is what an arm with 0.331 m along-track error ought to do.
* ⭐ **And both fixes came free**, from the same one-line term — neither was designed for.

---

## 5. ⭐ THE HEADLINE QUESTION, ANSWERED EXPLICITLY

> **Does `cv_holdv0` still rank first?**

**Rank 1 overall is `v1_ego_oracle_lon` under EVERY term — including the published one.** It is an
**ORACLE** (v1's curve walked to the ego's own *logged* future distance), so it is an upper bound and
not a candidate. That was already true before this work; it is not a change.

> ⭐ **Among realisable arms: YES. `cv_holdv0` still ranks first, under every term tested
> (`w` = 0, 0.5, 1, 1.5, 2, 3), and its margin over the best learned arm is essentially unchanged
> (`v1_ego_v0 − cv_holdv0`: −0.0100 SEP → −0.0090 SEP).**

⛔ **But the sub-claim under the headline is REFUTED, and it is the more important one.**
The 2026-07-27 panel's force came from *two* statements: (a) *"a zero-parameter baseline beats every
learned arm"*, and (b) implicitly, *"and the flagship's own tactical planner is at worst tied with
it"* (`v1_tactical_follow − cv_holdv0` = **n.s.**). **(a) survives. (b) does not** — under a metric
that can see over-travel, v1's plan is **separated-worse than doing nothing by −0.1212**, and the gap
that was previously indistinguishable from zero is now the **fourth-largest |Δ|** in the contrast table.

⇒ **The metric fix does not rescue any learned arm. It makes the flagship's deficit visible.**

---

## 6. ⚠️ SENSITIVITY OF THE RANKING TO THE SHAPE

`score = clamp(clamp_v1(r) − w·max(r−1, 0), 0, 1)`; `w = 0` is `clamp_v1`, `w = 1` is `twosided_v2`.

| arm | `clamp_v1` (w=0) | w=0.5 | **`twosided_v2` (w=1)** | w=1.5 | w=2 | w=3 |
|---|---|---|---|---|---|---|
| ⚠️ `v1_ego_oracle_lon` | #1 0.5946 | #1 0.5944 | **#1 0.5943** | #1 0.5941 | #1 0.5939 | #1 0.5936 |
| ⭐ `cv_holdv0` | #2 0.5705 | #2 0.5585 | **#2 0.5492** | #2 0.5417 | #2 0.5353 | #2 0.5249 |
| `v1_ego_v0` | #4 0.5608 | #3 0.5494 | **#3 0.5403** | #3 0.5333 | #3 0.5272 | #3 0.5172 |
| `v4_oracle` | #3 0.5622 | #4 0.5471 | **#4 0.5362** | #4 0.5267 | #4 0.5180 | #4 0.5032 |
| `refc_xl_produced` | #5 0.5499 | #5 0.5363 | **#5 0.5259** | #5 0.5169 | #5 0.5090 | #7 0.4951 |
| `refc_xl_v0on` | #6 0.5499 | #6 0.5363 | **#6 0.5259** | #6 0.5169 | #6 0.5090 | #8 0.4951 |
| `refc_base_produced` | #11 0.5439 | #7 0.5316 | **#7 0.5228** | #7 0.5154 | #7 0.5089 | #5 0.4975 |
| `refc_base_v0on` | #12 0.5439 | #8 0.5316 | **#8 0.5228** | #8 0.5154 | #8 0.5089 | #6 0.4975 |
| `refc_small_produced` | #10 0.5444 | #9 0.5305 | **#9 0.5202** | #9 0.5116 | #9 0.5041 | #9 0.4918 |
| `refc_xl_v0off` | #14 0.5166 | #10 0.4920 | **#10 0.4724** | #10 0.4557 | #10 0.4415 | #10 0.4187 |
| `refc_base_v0off` | #15 0.4980 | #11 0.4762 | **#11 0.4591** | #11 0.4444 | #11 0.4315 | #11 0.4108 |
| `v1_tactical_oracle` | #8 0.5467 | #12 0.4759 | **#12 0.4305** | #12 0.3983 | #12 0.3744 | #12 0.3421 |
| `v1_tactical_follow` | #7 0.5471 | #13 0.4720 | **#13 0.4242** | #13 0.3906 | #13 0.3660 | #13 0.3332 |
| `v1_lat_straight` | #9 0.5460 | #14 0.4692 | **#14 0.4209** | #14 0.3867 | #14 0.3617 | #15 0.3284 |
| `nospeed_tactical_oracle` | #13 0.5394 | #15 0.4660 | **#15 0.4179** | #15 0.3845 | #15 0.3608 | #14 0.3299 |
| `v4_blind` | #16 0.3749 | #16 0.3519 | **#16 0.3322** | #16 0.3140 | #16 0.2969 | #16 0.2659 |

**What is robust to `w`, and what is not:**

* ⭐ **ROBUST: ranks 1–4 are identical for every `w > 0`.** Oracle, `cv_holdv0`, `v1_ego_v0`,
  `v4_oracle`. **The headline does not depend on the shape at all.**
* ⭐ **ROBUST: the entire v1 family sits below the entire REF-C family for every `w > 0`**, including
  the two `v0off` ablations. This is the study's main ordering change and it appears at `w = 0.5`
  already.
* ⚠️ **NOT robust: `v1_ego_v0` vs `v4_oracle` swaps at `w > 0`** (#4/#3 → #3/#4). Their intervals
  overlap heavily and the contrast is n.s. either way — a rank swap between two statistically
  indistinguishable arms is a *presentation* change, not a finding.
* ⚠️ **NOT robust: at `w = 3` the REF-C scale order inverts** (`refc_base` #5 overtakes `refc_xl` #7),
  and `nospeed`/`v1_lat_straight` swap #14/#15. **⇒ do not read the deep ranking off a chosen `w`.**
  At `w = 3` an arm is scored 0 at ratio 1.33, which charges over-travel far harder than the surface
  can justify without a collision gate.

⇒ **The choice of `w` is a presentation choice below rank 9 and irrelevant above it. `w = 1` is
reported as the primary because it is the minimum assumption, not because it wins anything.**

---

## 7. ⛔ THE PER-ARM GATE — REFUSED, NOT "AVAILABLE"

MEASURED here from the dumps, not inherited (`artifacts/tables.md` T6):

| arm | `comfort` mean |
|---|--:|
| `v1_tactical_follow` | **0.0004** |
| ⭐ `v1_ego_v0` | **0.2882 — 720×** |
| `v1_ego_oracle_lon` | 0.2453 |
| `v1_lat_straight` | 0.0099 |
| `cv_holdv0` | 1.0000 (saturated ⇒ inadmissible) |
| `v4_oracle` | 0.0000 (floored ⇒ inadmissible) |

Re-timing a plan at constant speed **smooths** it: the jerk that trips the comfort bound is an
artefact of independent per-horizon waypoint regression, and constant-speed resampling removes it. So
between two arms that differ **only in schedule**, `comfort` moves by **720×** — and under the
per-arm gate that term enters the composite and turns the source stream's primary from **n.s.** into
**+0.0836 SEPARATED**.

**⇒ Recommendation, and it is implemented: the per-arm gate should be REFUSED outright, not left
available as a "sensitivity".** `scripts/recompute_panel.py` does not compute it; the artifact records
the refusal and its reason. A sensitivity that can only ever be quoted when it agrees with the author
is not a sensitivity — and the one time it disagreed, it would have manufactured a `CONFIRM` out of a
smoothing artefact on the author's own primary. Keeping it "available and reported" is exactly how a
number that should never decide anything ends up in a headline.
⚠️ `panel_combine.py` (the 2026-07-27 script) still emits `_PSS_under_per_arm_gate_SENSITIVITY`. I did
**not** edit that script — it produced published artifacts and editing it would break their
reproduction. ⇒ **E-C below.**

---

## 8. Everything that is wrong with this work, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | ⛔ **No ego-trained checkpoint was scored.** The guard is proven to fire and the ego is proven to change the plan, but `flagship-v2corpus-30k` is still training and was not touched. **The guard is UNVERIFIED against a real ego-trained checkpoint** — only against a synthetic one with trained-like weights. | ⚠️ **the main caveat** |
| 2 | ⚠️ **`pose_scale` is taken from the trainer default (10.0)** at every wired call site, not read from the checkpoint. Correct for every 4-brain run in the registry (`--pose-scale` default), but **a mis-scaled ego is worse than no ego**. `ego_from_poses` takes it as an argument and the callers thread it; the loaders do not yet supply it from the ckpt config. ⇒ **E-B.** | ⛔ disclosed |
| 3 | ⚠️ **Two `stack/` call sites are deliberately unguarded** (§1.5), tested as an explicit exemption list. ⇒ **E-A.** | disclosed, pinned |
| 4 | ⚠️ **The closed loop's per-tick ego is a MODELLING CHOICE**, not a measurement: proprioception from the simulated bicycle. It is the only leak-free option, but it is not what a trainer saw (which is observed poses at every step). Untested against a real arm. | ⛔ disclosed |
| 5 | **The `w` grid is 5 points on one shape family.** A convex or exponential over-travel charge would not be captured. Chosen because piecewise-linear is the only shape that is a strict refinement of the published term. | deliberate |
| 6 | **`comfort` is dropped, not fixed** — inherited from the panel, deliberately not retuned. §7 shows the cost of the alternative. | inherited |
| 7 | **No collision, no TTC** — the val cache has no cuboids ⇒ **`PSS` is not a Driving Score**, and **which side of the ratio is more dangerous cannot be measured here.** That is precisely why `w = 1`. | inherited blocker |
| 8 | **2 s horizon, non-reactive log replay, lateral grid axis refused** — all inherited and all bounding. The 2 s horizon is why CV is a strong baseline by construction. | inherited |
| 9 | ⚠️ **Flipping `PROGRESS_TERM_DEFAULT` changes what a *future* gate run measures.** Mitigated by the versioned `name`/`metric_id`, the `progress_term` config field, and `PRIMARY_NAME` carrying the term — but it IS a default change and I am naming it. It does **not** touch pod1 (§9). | ⛔ deliberate, disclosed |
| 10 | **I did not re-run any model.** Every composite is arithmetic over dumps produced by other agents; the fidelity of those dumps is INHERITED from the source reports (which published a 2.2 mm cross-host port check). | inherited |
| 11 | ⚠️ **`v1_ego_v0`'s `recovery` defined-fraction is 0.8196 vs the control's 0.8423** (−2.27 pp), inherited from the source study and unchanged by the term. Its `recovery` half is therefore not a strict like-for-like; the `ego_progress` half, which carries the result, is on an identical NaN mask. | inherited, disclosed |

---

## 9. ✅ pod1 — VERIFIED, NOT ASSUMED

| check | result |
|---|---|
| `train_flagship4b.py` bytes | **unchanged** (`git diff HEAD` empty), and pinned by a test |
| `fourbrain.py` / `config.py` / `flagship_losses.py` bytes | **unchanged** (`git diff HEAD` empty) |
| ⭐ **real import test** — import `train_flagship4b` against the modified tree, enumerate `sys.modules` | **leak set = `[]`** — neither `taniteval.*`, nor `tanitad.train.heldout_gate`, nor `tanitad.ego_plan` enters its import graph |
| `tanitad/train/__init__.py` | **empty** ⇒ `heldout_gate` is not auto-imported by `tanitad.train.flagship_losses` |
| `TacticalPolicy.forward` signature | unchanged: `(self, states, ctx, ego=None)` |
| pod1 process, read-only | PID **699286**, `flagship-v2corpus-30k`, **2 d 10 h 29 m 45 s**, GPU **62 %**, 15,348 MiB |

⇒ **Nothing in this change can reach pod1's running arm or its resume path.** Everything I touched
lives in `taniteval/` plus `stack/tanitad/train/heldout_gate.py`, and the latter is imported only by
`train_flagship_v4.py` (the v4/v5 trainer). ⚠️ **Probed, not assumed:** `ps` on pod2, pod3 and
`tanitad-eval` shows **no** `train_flagship_v4` / `train_flagship4b` process on any of them, and
pod1's is `train_flagship4b`. And in any case a repo edit cannot reach a *running* process — pods
run their own copies — so the only exposure is a future run, which is exactly what `PRIMARY_NAME`
carrying the term makes visible.

🔒 **Parity untouched** — no episode re-selected; all 20 arms share the identical 15,981 rows and row
identity is *asserted*, not assumed. No clip UUID or raw PhysicalAI content appears in any artifact.

---

## 10. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **E-0** | ⛔⛔ **THE v2corpus GATE MUST NOW PASS `ego=`, AND THE HARNESS WILL REFUSE IF IT DOES NOT.** That is the intended behaviour and it will look like a broken harness the first time it fires. Whoever scores `flagship-v2corpus-30k` needs to know: build the ego with `taniteval.ego_guard.ego_from_poses(poses, last, pose_scale)` and pass it. **The wired eval paths do this automatically; a NEW script will not.** | **Benchmarks & Eval — BEFORE the v2corpus gate** |
| **E-A** | ⚠️ **Two `stack/` call sites remain unguarded by design** — `refs/refa.py:260` and `fourbrain.py:614` (`propose_and_score`). Both are model code, and one is in pod1's live import graph. **They should be guarded once pod1 is done**, not before. The exemption list is tested so it cannot grow silently. | **Architecture & Inference, after pod1 finishes** |
| **E-B** | ⚠️ **`pose_scale` must come from the checkpoint, not the default.** Every wired call site currently uses 10.0. Correct for every registry arm, but the loaders should surface `ckpt["args"]["pose_scale"]` and the eval entry points should thread it. **A mis-scaled ego decodes garbage — worse than no ego.** | **`taniteval` maintainer** |
| **E-C** | ⛔ **`panel_combine.py` still emits `_PSS_under_per_arm_gate_SENSITIVITY`.** I did not edit it (it produced published artifacts). Either it stops emitting that key, or the gate protocol states in writing that the per-arm gate may never be quoted. §7 is the evidence: it flipped a primary verdict on a 720× smoothing artefact. | **PI / `taniteval` maintainer** |
| **E-D** | ⛔⛔ **EVERY PSS NUMBER PUBLISHED BEFORE TODAY IS `@clamp_v1`, AND SIX CARRY A CHANGED VERDICT — four separation flips and two admissibility flips.** `MODEL_REGISTRY`, `PSEUDOSIM_ARM_PANEL.md`, `TACTICAL_ACTION_INPUT.md` and `V5_PLAN` all quote bare `PSS`. They need the `@clamp_v1` suffix, and the four verdict changes in §4.3 need propagating — in particular *"v1's tactical plan is tied with `cv_holdv0`"* is now **separated-worse**. Manifest in §11.2. | **Model-registry agent / PI** |
| **E-E** | ⭐ **v5 SHOULD BE GATED ON `@twosided_v2`, AND THE DEFAULT NOW DOES THAT.** `heldout_gate.PRIMARY_NAME` carries the term. ⚠️ **A decision is still owed on `w`:** I ship `w = 1` as the minimum assumption because this surface cannot measure danger. If the PI believes over-travel is more dangerous, `w = 1.5–2` changes nothing above rank 9 (§6) and is defensible — but it must be *chosen and recorded*, not inherited. | **PI — before the v5 gate** |
| **E-F** | ⚠️ **The E6 FiLM zero-init trap is not hypothetical — it bit this task.** Any v5 ego graft measured on a from-scratch brain will read **exactly 0.0** effect and "prove" the lever useless. Every ego-lever A/B must be run on a *trained* brain or state the FiLM's norm alongside the result. | **Architecture & Inference — v5** |
| **E-G** | ⭐ **A NEW RETRACTION CLASS, offered for `RETRACTION_LOG.md` — I did not append it myself.** ⛔ **`METRIC AUDITED ONLY IN THE DIRECTION IT WAS BUILT TO DETECT`.** `ego_progress` was validated against arms that travel too *little* (`stand_still`, `v4_blind`, `v1_ego_half` — all correctly punished) and was never probed with an arm that travels too *much*. It passed every adversary it had, and was blind on the other side of 1.0 for its whole life. *Detection heuristic: a metric defined by a one-sided `clamp` or `max`/`min` has an entire half of its input domain with zero gradient — enumerate what lives there before adopting it.* Sibling of the source report's E8 (`TWO-CONDITION GATE AUDITED AT ONE CONDITION`): **both are "the audit covered one side of a two-sided object".** | **PI / RETRACTION_LOG owner** |

---

## 11. Deliverable manifest

Repo dir: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-pss-twosided/`
Everything `git add`-ed into the working tree. ⛔ **I did not commit and did not push.**
⚠️ marks anything living in only ONE place — **there is nothing in that state.**

### 11.1 Artifacts and code

| artifact | where it lives | what it is |
|---|---|---|
| `PSS_TWOSIDED.md` | repo (this dir) | this report |
| `scripts/recompute_panel.py` | repo | the 20-arm recomputation under 6 terms; row-identity assertion, panel-wide gate, reproduction gate, paired contrasts. **No GPU, no model, no corpus.** |
| `scripts/demo_guard_fails.py` | repo | ⭐ the guard's **demonstrated failure** + the magnitude of the bug + the FiLM zero-init trap |
| `scripts/tables.py` | repo | regenerates **every table in this report** from the artifacts — the tables are generated, not hand-typed |
| `artifacts/panel_both_terms.json` | repo | ⭐ the full artifact: 20 arms × 6 terms, per-arm CIs, rankings, 6×23 paired blocks, the reproduction gate, the decomposition, the ratio distribution |
| `artifacts/guard_demo.json` | repo | the guard's failing value, its message, warn-mode stamp, and both controls |
| `artifacts/tables.md` | repo | the generated tables T1–T6, verbatim as transcribed above |
| `taniteval/taniteval/ego_guard.py` | repo | **NEW module** — the guard call, the adapter hook, `ego_from_poses`, the capability probe, the mode parser |
| `taniteval/tests/test_ego_guard.py` | repo | **NEW — 23 tests**, 8 of which pin a FAILING value, incl. the source-scan and the pod1-isolation pins |
| `taniteval/tests/test_pseudosim_progress_term.py` | repo | **NEW — 18 tests**, every rule in both directions |
| `taniteval/taniteval/pseudosim.py` | repo | versioned `PROGRESS_TERMS`, `metric_id`, `progress_from_ratio`, `UnknownProgressTerm`; guard hook in `pseudo_evaluate` |
| `taniteval/taniteval/{closedloop,planning,planner_p2,corpus_overlay}.py`, `taniteval/probe_overlay.py` | repo | the guard wired + `ego=` passed at every call site |
| `taniteval/tests/test_pseudosim.py` | repo | one assertion updated: the composite `name` is now the versioned ID |
| `stack/tanitad/train/heldout_gate.py` | repo | `PROGRESS_TERM`, versioned `PRIMARY_NAME`, `HeldoutGateConfig.progress_term`, term threaded into `_composite_of` |

**Reproduce everything, no GPU:**
```
python3 scripts/recompute_panel.py --sensitivity \
  --in-dir <…/2026-07-27-pseudosim-arm-panel/artifacts> \
  --in-dir <…/2026-07-28-tactical-action-input/artifacts/pw> \
  --in-dir <…/2026-07-28-tactical-action-input/artifacts/blockA> \
  --out artifacts/panel_both_terms.json          # 76.5 s on the dev box, 24 CPU
python3 scripts/demo_guard_fails.py              # seconds
python3 scripts/tables.py > artifacts/tables.md
```

### 11.2 ⛔ EVERY PUBLISHED NUMBER WHOSE VALUE CHANGES

**All of them.** Every `PSS_recovery_progress` value ever published is a `@clamp_v1` value. It stays
exactly computable and exactly reproducible (§4.1, `max|diff| = 0.000000`), but under the new default
term the number is different. The table below is the blast radius, per arm, with the rank move:

| arm | published `@clamp_v1` | new `@twosided_v2` | Δ | rank move | quoted in |
|---|--:|--:|--:|--:|---|
| `v1_ego_oracle_lon` | 0.5946 | 0.5943 | −0.0003 | — | `TACTICAL_ACTION_INPUT` §3.1 |
| `cv_holdv0` | 0.5705 | 0.5492 | −0.0213 | — | panel §3, `V5_PLAN`, registry |
| `v4_oracle` | 0.5622 | 0.5362 | −0.0260 | 3→4 | panel §3 |
| `v1_ego_v0` | 0.5608 | 0.5403 | −0.0205 | 4→3 | `TACTICAL_ACTION_INPUT` §3.1 |
| `refc_xl_produced` | 0.5499 | 0.5259 | −0.0240 | — | panel §3, registry §4.1 |
| `refc_xl_v0on` | 0.5499 | 0.5259 | −0.0240 | — | `TACTICAL_ACTION_INPUT` §6.3 |
| `refc_small_produced` | 0.5444 | 0.5202 | −0.0242 | 10→9 | panel §3 |
| `refc_base_produced` | 0.5439 | 0.5228 | −0.0211 | 11→7 | panel §3 |
| `refc_base_v0on` | 0.5439 | 0.5228 | −0.0211 | 12→8 | `TACTICAL_ACTION_INPUT` §6.5 |
| `refc_xl_v0off` † | 0.5166 | 0.4724 | −0.0442 | 14→10 | `TACTICAL_ACTION_INPUT` §6.3 (as a Δ, not a level) |
| ⛔ `v1_tactical_follow` | 0.5471 | 0.4242 | **−0.1229** | **7→13** | panel §3, `V5_PLAN` |
| ⛔ `v1_tactical_oracle` | 0.5467 | 0.4305 | **−0.1162** | **8→12** | panel §3 |
| ⛔ `v1_lat_straight` | 0.5460 | 0.4209 | **−0.1251** | **9→14** | `TACTICAL_ACTION_INPUT` §3.1 |
| ⛔ `nospeed_tactical_oracle` | 0.5394 | 0.4179 | **−0.1215** | 13→15 | panel §3 |
| `refc_base_v0off` | 0.4980 | 0.4591 | −0.0389 | 15→11 | `TACTICAL_ACTION_INPUT` §6.5 |
| `v4_blind` | 0.3749 | 0.3322 | −0.0427 | — | panel §3 |
| `v1_ego_half` (probe) | 0.3117 | 0.3100 | −0.0017 | probe | `TACTICAL_ACTION_INPUT` §3.1 |
| `v1_ego_double` (probe) | *refused* | 0.0954 | **now scorable** | probe | `TACTICAL_ACTION_INPUT` §5.2 |
| `oracle_lon_straight` (probe) | *refused* | 0.6032 | **now scorable** | probe | `TACTICAL_ACTION_INPUT` §5.3 |

**And four published VERDICTS change** (not just values):

| verdict, as published | under `@twosided_v2` |
|---|---|
| `v1_ego_v0 − v1_tactical_follow` **n.s.** ⇒ pre-registered `REFUTE` | **SEPARATED +0.1121** ⇒ the REFUTE was a metric artefact on this contrast |
| `v1_ego_v0 − v1_tactical_oracle` **n.s.** | **SEPARATED +0.1057** |
| ⛔ `v1_tactical_follow − cv_holdv0` **n.s.** (*"v1 is tied with doing nothing"*) | ⛔ **SEPARATED −0.1212** (*v1 is worse than doing nothing*) |
| `v1_ego_double` / `oracle_lon_straight` **INADMISSIBLE** | **admissible; scored 0.0954 / 0.6032** |

† `refc_xl_v0off`'s LEVEL was never published — only the paired Δ (+0.0332), which this recompute reproduces exactly.

⚠️ **The source report's headline sentence — *"the composite separates a 3.36× degradation and cannot
separate a 5.41× improvement"* — remains TRUE OF `@clamp_v1` and is FALSE OF `@twosided_v2`.** It
should be re-quoted with the suffix, not deleted: it is the measurement that motivated the fix.

### 11.3 Suites

| suite | result | vs the brief's target |
|---|---|---|
| `stack/` | **1385 passed, 12 skipped** at the start of my changes; **1415 passed, 12 skipped** at the end (115.0 s) | ✅ **zero new skips**, zero failures. ⚠️ **I added NO `stack/` tests** — my tests live in `taniteval/`. The **+30** are a concurrent sibling's (`stack/tests/test_rig_clean_fix.py`, the rig-clean-fix stream, written at 15:12 during my run, together with its edits to `stack/tanitad/data/calib.py` and `stack/scripts/v2_compressed.py`). Stated because a count that grew on its own would otherwise read as mine. |
| `taniteval/` | **606 passed, 0 skipped** (74.1 s) | ✅ 565 + **41 new** (23 `test_ego_guard` + 18 `test_pseudosim_progress_term`), **zero new skips**, zero failures. The one warning is pre-existing (`test_clhorizon`, `Mean of empty slice`). |

### 11.4 Self-refutations

| # | what | status |
|:--:|---|---|
| 1 | ⛔ **My guard-teeth demo first measured 0.000000 m and I nearly reported the bug as cosmetic.** Cause: `FiLM.to_scale_shift` zero-init on a fresh build. Both rows published (§1.2). | corrected |
| 2 | ⛔ **My first two-sided implementation (`1 − \|1 − r\|`) was NOT bit-identical to `clamp_v1` below ratio 1** — float32 drift of 2.5 × 10⁻⁴ on 36 % of rows, which would have contaminated every cross-term delta. Rewritten as *published-term-minus-a-charge* so the refinement claim is exact. Caught by my own test, which failed first. | corrected |
| 3 | ⛔ **My first guard placement broke a real test:** `test_closed_loop_rollout_uses_the_injected_plan_fn` asserts the injected-`plan_fn` path must not touch the policy brains, and my capability probe did. The test was right; the guard now runs only when the hierarchy path is used. | corrected |
| 4 | ⚠️ **I could have quoted rank 1 as `cv_holdv0` and been wrong.** Rank 1 overall is an ORACLE under *both* terms. Stated explicitly in §5 rather than shading it. | disclosed |
| 5 | ⚠️ **The source report's E1 census over-counted by one file** — `blindimag.py:101` is a docstring. That claim had already propagated into `ego_plan.py`'s docstring. Corrected in `_EXEMPTIONS` (§1.4). | corrected |
| 6 | **I did not guard two `stack/` call sites** and say so rather than claiming full coverage (§1.5, E-A). | deliberate, pinned |
| 7 | **I did not score a real ego-trained checkpoint** — pod1 is mid-run and forbidden. The guard's behaviour against `flagship-v2corpus-30k` is **UNVERIFIED** (§8.1). | by rule |
| 8 | **I did not hold anything to v1's 0.4271** — that is `wm_fidelity_ade_2s` (`rollout.py:170`, `actions_source="expert_future"`), not a planning bar. | correct by construction |
