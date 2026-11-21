# C2 — shipping the one measured deployable win, and the label it has been travelling under

**Date 2026-07-27** (Europe/Berlin; the deliverable directory is named for the brief's date)
· **Stream:** Architecture & Inference · **Repo HEAD at start:** `d07da62`
· **Hosts:** dev box (all analysis, CPU + one RTX 4060 timing run) + **`tanitad-eval`, IDLE, read
of one existing dump and one extract — nothing launched, nothing trained, no pod code modified.**
⛔ **pod1 and pod2 were never contacted.**

---

> # HEADLINE
>
> 1. ⭐ **CONFIRM — and at a layer the producing stream did not reach.** The shipped implementation
>    reproduces the UNGATED C2 cell to 4 dp: **as-trained 0.8563 → C2 0.5645**, paired
>    **−0.2918 [−0.4233, −0.1598]**, separated, `selected_frac` **1.000**, 881 windows / 40 episode
>    clusters. Verified at **three layers**, the strongest of which **re-derives the cost matrix from
>    the raw fan and the reference roll-out** (not from a stored column) and returns **881/881
>    identical picks**.
> 2. ⛔ **BUT THE BRIEF'S HEADLINE NUMBER IS NOT THE UNGATED RULE.** `−0.3366 / 0.5196–0.5221` is
>    labelled *"UNGATED, on 100 % of windows"*. It is `learned_gate_ALL_ridge_tau0` — a **fitted
>    ridge gate firing on 66.97 %** of windows over a 73-feature bank **including the 2-world-model
>    ensemble family the brief itself calls a dominated prerequisite.** Quoted as the ungated rule it
>    **overstates the shipped win by 0.0448 m (1.154×)**. **C19/M12 class: a conditional number
>    travelling without its firing rate.** *(`raw/c2_reverify.json § ADJUDICATION`, from the primary
>    JSON.)*
> 3. ⭐ **It is now a real, tested, selectable option:** `stack/tanitad/models/wm_reference_select.py`
>    (231 lines, the rule, no fitter) + `eval_flagship_v4.py --select-rule c2-wm-ref --c2-scorer <ckpt>`.
>    **30 new tests, 0 new skips.** `stack` **1348 → 1379 passed / 12 skipped** (+24 mine, +31 a
>    sibling's `test_ego_plan.py`); `taniteval` **559 → 565 passed / 0 skipped**.
> 4. ⛔ **DEFAULT OFF, and the reason is MEASURED, not precautionary.** The identical rule is
>    **separated-WORSE, +0.2090 [+0.0550, +0.3642]**, when an arm scores its own fan. **The rule's
>    SIGN is a property of the scorer.** So there is no default scorer either: `--select-rule
>    c2-wm-ref` without `--c2-scorer` **refuses to start**.
> 5. **Cost: +6.7 ms/window** (one 20-step roll at eval batch 16, RTX 4060) **+ 55.6 ms/window** for
>    the second frame encode a *foreign* scorer needs — **≈62 ms/window, ≈55 s for the 881-window
>    val set.** The per-candidate alternative (A1) is **256× the predictor steps** and **37.9×** the
>    wall-clock at matched batch.
> 6. ⚠️ **And it still does not clear the bar.** C2 at **0.5645** vs the legitimate same-surface bar
>    **0.4907** is a **1.15× miss**, and it leaves **0.3140 [0.2632, 0.3703] m** on the table against
>    the fan's own oracle (**0.2505**, reproduced exactly). This is a large, free, deployable
>    improvement to the *selector*. It is not a solved planner.

**Evidence class:** every number below is **MEASURED (ours)** with its artifact path, except two rows
explicitly tagged **INHERITED**. **Tier: CONFIRMED, not decision-grade** — 40 episodes, one fan, one
head; the tier is inherited from the producing stream and re-verification does not raise it.

---

## 1. RE-VERIFICATION — priority 1, done first, three layers

`code/c2_reverify.py` → `raw/c2_reverify.json`. Estimator: **paired episode-cluster bootstrap**
(`taniteval/ci.py`, B = 2000, unit = episode). ⛔ `overlapping_holdout_se` appears nowhere.
**Tolerance stated in advance: 5e-5 on the point estimate and on both bounds.**

### 1.1 The three layers, each strictly stronger than the last

| layer | what is trusted | what is recomputed | result |
|---|---|---|---|
| **L1** | the stored `costs.C2_wm_ref_proximity` column | pick → policy → paired interval, without calling `canary_proxy.py` | ✅ **PASS both arms** |
| **L2** ⭐ | **only the raw geometry** — `fan [881,256,20,2]` and `imag_ref [881,20,2]` | **the cost matrix itself**, by the shipped `wm_reference_cost` | ✅ **881/881 identical picks** |
| **L3** | only `fan` + `tgt` | `ade_0_2s` at WP_STEPS (5,10,15,20) — no stored per-arm column read at all | ✅ **PASS** |

### 1.2 The numbers

| arm (scorer of v4's fan) | as-trained | C2 | paired Δ | sep | `selected_frac` | vs published |
|---|---:|---:|---|:--:|---:|---|
| ⭐ **v1's world model** (`flagship-speedjerk-30k`) | 0.8563 | **0.5645** | **−0.2918 [−0.4233, −0.1598]** | ✅ | **1.000** | **exact, 0.0000** |
| ⛔ **v4 scoring its own fan** | 0.8563 | 1.0653 | **+0.2090 [+0.0550, +0.3642]** | ✅ | 1.000 | **exact, 0.0000** |

L3, from geometry alone: **0.5645**, **−0.2918 [−0.4233, −0.1598]**, `separated`, `p(Δ>0) = 0.000`,
881 windows / 40 clusters. `oracle_in_fan` recomputed = **0.2505**, matching the committed value.

### 1.3 ⚠️ L2 is decision-identical, NOT bit-identical — stated, not hidden

The published cost matrix was reduced on an **NVIDIA A40** in float32; mine on the **dev-box CPU**.
`.norm().mean()` is accumulation-order sensitive, so:

| | |
|---|---:|
| max abs cost disagreement | **1.144e-05 m** (3.7e-06 of the mean cost) |
| picks that still agree | **881 / 881** |
| windows whose winner-vs-runner-up gap is **below** that disagreement | **0** |
| ⚠️ windows whose gap is below **10×** it | **3** |
| ⚠️ worst-case headroom (tightest window) | **2.2×** |

⇒ the decision is robust on this corpus, **but only 2.2× clear at the tail**. A different
accumulation order on a future host could flip a small number of near-tied windows. Reported because
a "reproduces exactly" claim that quietly meant "on the same GPU" is exactly the class of thing this
program retracts.

### 1.4 ⛔ THE ADJUDICATION — what `−0.3366` actually is

`raw/c2_reverify.json § ADJUDICATION_of_the_brief_headline`, read from the primary
`…/2026-07-27-canary-proxy/raw/canary_proxy.json`:

| cell | JSON row | Δ | `ade_0_2s` | **`selected_frac`** | needs |
|---|---|---:|---:|---:|---|
| ⭐ **what ships** | `stage1…v1.UNGATED_C2_everywhere` | **−0.2918** | **0.5645** | **1.000** | one extra roll-out |
| the brief's headline | `stage3…v1.learned_gate_ALL_ridge_tau0` | −0.3366 | 0.5196 | **0.6697** | **a fitted ridge gate over 73 features incl. 2-WM ensemble** |
| the brief's "0.5221" | `stage3…v1.learned_gate_1WM_ridge_tau0` | −0.3342 | 0.5221 | **0.6788** | a fitted ridge gate (1WM features) |

**The producing document is right and internally consistent** — §1.2 publishes the ungated row as
−0.2918 and §5.2 recommends exactly that. **The conflation happened in relay.** Both "gate" rows are
also, per that document's own §4.4, **fold-seed fragile**; and the `ALL` row needs the second world
model whose availability the brief's own dominance argument says makes the gate pointless.

⇒ **Against the pre-registration: CONFIRM for the cell the shipped rule targets (−0.2918, exact),
and a RETRACTION of the brief's headline label.** Had I "reproduced −0.3366", I would have shipped a
gate.

### 1.5 What would have made this return a FAILING value — and that it can

| check | fails when | shown able to fail |
|---|---|---|
| L1/L3 `PASS` | recomputed mean or either bound moves > 5e-5 | red/green case 2: changing the cost reducer breaks L2/L3 fidelity (`raw/redgreen.json`) |
| L2 `picks_match_published` | the shipped formula differs from `cost_C2_ref` | ✅ went RED on `.mean()`→`.sum()` |
| L2 decision margin | any window's top-1/top-2 gap falls below the numerical disagreement | measured **0 / 881** — it is a real count, not a constant |
| the v4 arm | — | **it already fails: +0.2090, separated-WORSE.** The instrument reports a loss on a real arm, so it is not a one-sided instrument |

⚠️ **The tightest bar was NOT met on the first attempt**, and the fix is in the code: my initial L2
bar was *bit-exactness*, which came out **false**, and my first artifact-fidelity fixture used a
**single lead time** — where `mean` and `sum` coincide, so it **could not detect a changed reducer**.
The red/green sweep caught it (`raw/redgreen.json` case 3 came out GREEN), and the fixture now uses
two lead times. *A test that cannot fail is the defect one layer down.*

---

## 2. WHERE IT LIVES, AND THE DEFAULT

### 2.1 The rule — `stack/tanitad/models/wm_reference_select.py` (new, 231 lines)

```
ref        = rollout_decode(predictor, states, win_actions, None, step_readout, K)   # ONE roll
cost[b, n] = mean_k || fan[b, n, k] - ref[b, k] ||_2
pick[b]    = argmin_n cost[b, n]
```

**Why its own module, not inside the head.** It is a *selection rule over a frozen fan*, and it needs
a world model that is **not necessarily the head's own** — wiring it into `FlagshipV15Head` would
couple the head to a second checkpoint. It follows the `readout_selection.py` precedent exactly:
**the module ships a RULE and contains NO FITTER**, and
`test_the_module_ships_a_rule_and_cannot_fit_one` inspects every public signature for a ground-truth
argument. `MEASURED_ARMS` carries both measured rows **in code**, so the default is pinned to the
evidence rather than to an opinion, and a cross-file test asserts those constants still equal the
primary JSON.

Public surface: `wm_reference_rollout` · `wm_reference_cost` · `select_by_wm_reference` ·
`selection_telemetry` · `resolve_scorer_tag` · `WM_REFERENCE_SELECT_DEFAULT` · `SELF_SCORING` ·
`MEASURED_ARMS`.

### 2.2 The eval path — `stack/scripts/eval_flagship_v4.py` (+148 / −5)

`apply_c2_selection(out, horizons, ref, tag)` swaps the pick **and every key derived from it**
(`traj`, `wp_seq`, `waypoints`), placed **before `v15_losses`** so `ade` / `sel_gap` / `rank_acc` /
the dense path / the 4-wp path all describe the **deployed** pick. `out["anchor_traj"]` is **not
touched**, so `oracle_ade` and every coverage diagnostic are invariant *by construction* — asserted
in a test, not asserted in prose.

Provenance is stamped on the artifact: `res["select_rule"]`, `res["c2_scorer"]`, the
`diag["c2_selection"]` telemetry block, and the `method` string.

### 2.3 ⛔ The default is OFF — and there is no default scorer either

| guard | behaviour |
|---|---|
| `WM_REFERENCE_SELECT_DEFAULT` | `False` |
| `--select-rule` | `as-trained` (unchanged); `collect_planner(select_rule=...)` defaults the same |
| `--select-rule c2-wm-ref` with no `--c2-scorer` | **`SystemExit` on the ARGUMENTS**, before a 3 GB checkpoint is read |
| `--c2-scorer` with no `--select-rule c2-wm-ref` | **`SystemExit`** — it would be silently ignored |
| self-scoring | reachable only by passing the literal `self`, and it prints the +0.2090 measurement |

**Why OFF is not a hedge.** Of the **two** arms this rule has ever been measured on, **one is
separated-WORSE**. A default that flipped the deployed pick would do to every published `ade_0_2s`
exactly what the `vision_rank` default did to v4. The coupling is enforced:
`test_default_is_off_because_a_measured_arm_is_separated_worse` goes **RED** if the default is
flipped while any `MEASURED_ARMS` row is separated-and-not-better.

**No default that alters a running arm was changed.** `git diff` on `eval_flagship_v4.py` removes
exactly 5 lines, all of them my own edit anchors; **this stream modified no other existing file** in `stack/` or `taniteval/`. *(`stack/tanitad/ego_plan.py` + `test_ego_plan.py` were staged CONCURRENTLY by a sibling agent and are not mine — see §6's suite counts.)*

---

## 3. THE COST PER WINDOW — measured

`code/c2_cost.py` → `raw/c2_cost.json`. Real architecture from `_eval_cfg()` (state_dim 2048,
window 8, d_model 768, depth 10, action_dim 3, K = 20, 256 candidates), random weights — wall-clock
depends on shapes and ops, not on values. Host: dev box, **RTX 4060**.

| item | cost | note |
|---|---:|---|
| predictor steps, **C2** | **20 / window** | one roll-out per window |
| predictor steps, A1 (per-candidate) | 5 120 / window | **256×** |
| C2 roll, batch 1 / 4 / 8 / **16** / 32 | 109.8 / 28.1 / 13.9 / **6.74** / 4.67 ms per window | latency-bound below ~16 |
| A1 roll, batch 4 | 1 067 ms per window | **C2 is 37.9× cheaper at matched batch** |
| ⚠️ **second frame encode** (foreign scorer) | **55.6 ms / window**, batch-independent | **this dominates** |
| ⇒ **C2 with a foreign scorer, batch 16** | **≈62 ms / window** | **≈55 s for the 881-window val set** |
| C2 self-scoring | ≈6.7 ms / window (reuses `states`) | ⛔ but separated-**worse** |

**INHERITED cross-check** (`…/2026-07-26-v5-imagination-selection/raw/v5_v1.json`, not re-measured):
the producing run rolled 225 536 candidate roll-outs in **440.4 s** on an **A40**; C2 alone is 1/256
of that work, ⇒ **≈1.72 s** for all 881 windows on that host.

⇒ **The honest statement of the cost is not "one roll-out". It is "one roll-out plus a second
encode of the same frames", and the encode is 8× the roll.** The win is still essentially free at
eval scale.

---

## 4. THE EXACT COMMAND A v5 RUN USES

v5 trains with `train_flagship_v4` and is evaluated with `eval_flagship_v4.py`
(`…/2026-07-28-v5-trainer/V5_TRAINER.md` §0.1). **Run it twice with different `--key`s** — the two
`windows_<key>.pt` files are what the paired interval is computed from.

```bash
# on the eval host, PYTHONPATH is REQUIRED (cd alone is not enough)
export PYTHONPATH=/workspace/TanitAD/stack
export OMP_NUM_THREADS=6

# (1) the baseline — the as-trained selector, DEFAULT, byte-identical to today
python3 /workspace/TanitAD/stack/scripts/eval_flagship_v4.py \
    --ckpt   /workspace/_v5gate/flagship-v5-30k/ckpt.pt \
    --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
    --key    flagship-v5-30k \
    --out    /workspace/_v5gate/eval_v5_30k.json \
    --goal-mode produced --episodes 40 --stride 8

# (2) the SAME checkpoint, same windows, C2 selection
python3 /workspace/TanitAD/stack/scripts/eval_flagship_v4.py \
    --ckpt   /workspace/_v5gate/flagship-v5-30k/ckpt.pt \
    --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
    --key    flagship-v5-30k-c2 \
    --out    /workspace/_v5gate/eval_v5_30k_c2.json \
    --goal-mode produced --episodes 40 --stride 8 \
    --select-rule c2-wm-ref \
    --c2-scorer  /root/models/flagship-30k/ckpt.pt     # v1's world model
```

Then the **paired** read (never a difference of two independent intervals):

```python
from taniteval.ci import paired_episode_cluster_bootstrap   # B=2000, unit = episode
# per-window ade_0_2s from the two windows_<key>.pt files, same episode ids, same order
paired_episode_cluster_bootstrap(ade_c2, ade_as_trained, eid, n_boot=2000)
```

**Four things the command's reader must know.**

1. ⛔ `--c2-scorer` **has no default and never will.** `/root/models/flagship-30k/ckpt.pt` is v1 —
   the arm that measured −0.2918. `--c2-scorer self` is a **diagnostic**, measured +0.2090 worse.
2. `--goal-mode produced` matches the surface the −0.2918 was measured on. Under `oracle` (the
   script's default) the whole comparison is an upper bound, not a deployable number.
3. ⚠️ **The −0.2918 is a fact about v4's fan, not a promise about v5's.** The producing stream
   measured C2's value tracking scoring-world-model quality ~1:1, and the fan changes under v5.
   **The v5 run must MEASURE it. Do not carry the number forward.**
4. The two runs must use the **same** `--episodes/--stride` or the windows do not pair.

---

## 5. WHAT IT DOES **NOT** DO

- ⛔ **It does not clear the bar.** `0.5645` vs the legitimate same-surface bar **0.4907** is a
  **1.15× miss**. E-V5-1's REFUTE stands; C2 improves the selector inside it.
  ⚠️ **`0.4907` itself needs a deployment tag** (881-window / 40-episode surface; `a0` 0.4714) and is
  currently being quoted against 600-episode numbers elsewhere. **Do not hold C2 to v1's `0.4271`** —
  that is `wm_fidelity_ade_2s`, what the world model scores when *handed the true actions*
  (`taniteval/rollout.py:170`, `actions_source="expert_future"`). Not a planning bar. The shipped
  module's docstring and a test both pin that C2's roll receives **no** future actions.
- **It does not improve the fan.** It re-selects inside a frozen proposal set: `oracle_in_fan` is
  unchanged **by construction** (asserted), and C2 still leaves **0.3140 [0.2632, 0.3703] m**
  against it.
- **It is not uniformly better.** Per window: **54.9 % better, 32.2 % worse, 12.8 % identical pick**
  (median per-window Δ **−0.1069 m**). The whole-set −0.2918 is a *mean*, not a guarantee.
- **It carries no gate**, deliberately. `selected_frac` is emitted on **every** row the module
  produces so a future conditional variant cannot report a stratum win as a deployed one.
- **It is not validated with the reachability clamp.** `keep=` exists and is tested, but every
  measured row has `keep=None`; the combination has **no interval behind it** and the telemetry says
  so.
- **It was not run end-to-end through the eval script on a GPU.** The rule is verified against the
  real geometry (§1) and the wiring is unit-tested (§6), but no `eval_flagship_v4 --select-rule
  c2-wm-ref` process has executed against a live checkpoint. **Stated plainly rather than implied.**

---

## 6. THE GUARDS, AND THE PROOF THEY CAN FAIL

**30 new tests. 0 new skipped tests.**
`stack/tests/test_wm_reference_select.py` (16) · `stack/tests/test_eval_flagship_v4_select_rule.py`
(8) · `taniteval/tests/test_c2_published_policy.py` (6).

**Red/green — every guard removed one at a time, `raw/redgreen.json`, `ALL_PASS: true`, all files
restored bit-exact (sha256 checked):**

| guard removed | test | went RED |
|---|---|:--:|
| `WM_REFERENCE_SELECT_DEFAULT = False` → `True` | `test_wm_reference_select` | ✅ |
| cost reducer `.mean()` → `.sum()` | `test_wm_reference_select` | ✅ |
| cost reducer `.mean()` → `.sum()` | `test_c2_published_policy` (881-window fidelity) | ✅ *(GREEN on the first attempt — §1.5)* |
| reference roll handed future actions | `test_wm_reference_select` | ✅ |
| CLI `default="as-trained"` → `"c2-wm-ref"` | `test_eval_flagship_v4_select_rule` | ✅ |
| the "c2-wm-ref needs a scorer" refusal | `test_eval_flagship_v4_select_rule` | ✅ |
| the "cannot self-score by omission" refusal | `test_wm_reference_select` | ✅ |

**Degeneracy checks kept, and measured on the real corpus** (`raw/c2_reverify.json § L2.telemetry`):
`n_tied_argmin` **0 / 881** (so no tie-break by row order — the stable-argsort class cannot bite),
`n_constant_cost_rows` **0**, `n_distinct_picks` **184 / 256**,
`frac_pick_equals_baseline` **0.1283** (the rule genuinely moves the pick), `selected_frac` **1.000**.

**Suite status.** `stack` **1379 passed, 12 skipped** (baseline `1324 / 12`; **+24 mine**, **+31 a
sibling's `test_ego_plan.py`, staged concurrently — not mine, not touched**). `taniteval` **565
passed, 0 skipped** (baseline **559 / 0**; +6 mine). **Parity untouched** — nothing here re-selects
an episode; the 881 windows are the producing run's, read in place.

---

## 7. DELIVERABLE MANIFEST — **STAGED, NEVER COMMITTED, NEVER PUSHED**

| artifact | where it lives | only one place? | note |
|---|---|---|---|
| `C2_SHIP.md` (this file) | `repo:` staged | no | |
| **`stack/tanitad/models/wm_reference_select.py`** | `repo:` staged | no | the rule, no fitter |
| **`stack/scripts/eval_flagship_v4.py`** (+148/−5) | `repo:` staged | no | `--select-rule` / `--c2-scorer`, default unchanged |
| `stack/tests/test_wm_reference_select.py` (16) | `repo:` staged | no | |
| `stack/tests/test_eval_flagship_v4_select_rule.py` (8) | `repo:` staged | no | |
| `taniteval/tests/test_c2_published_policy.py` (6) | `repo:` staged | no | reads the tracked primary artifacts |
| `code/c2_reverify.py` → `raw/c2_reverify.json` | `repo:` staged | no | the 3-layer re-verification + the adjudication |
| `code/c2_cost.py` → `raw/c2_cost.json` | `repo:` staged | no | cost per window, RTX 4060 |
| `code/redgreen.py` → `raw/redgreen.json` | `repo:` staged | no | 7 guards removed, all RED |
| `code/extract_geom.py` | `repo:` staged | no | ran on `tanitad-eval` |
| ⭐ `raw/c2_geom_v1.pt` (38.2 MB) | `repo:` staged **+** `tanitad-eval:/workspace/_v5/c2_geom_v1.pt` | no | **RESCUED** — the fan + reference roll that make L2 reproducible offline; md5 `0ae126c3…` verified both ends |
| inputs: `v5_{v1,v4}_windows_reduced.pt`, `canary_proxy.json` | already tracked | no | read in place, unmodified |

🔴 **STILL IN ONE PLACE: `tanitad-eval:/workspace/_v5/v5_v1_windows.pt` (114 MB)** — the full v5 dump
carrying `imag` and `ctrv` `[881,256,20,2]` (rules **A1** and **C1**), which exist **nowhere else**.
I rescued only the C2 slice. **Escalated in §8.**

**Reproduce everything, no GPU, ~40 s:**
```
/c/Users/Admin/venvs/tanitad/Scripts/python.exe code/c2_reverify.py raw/c2_reverify.json raw/c2_geom_v1.pt
/c/Users/Admin/venvs/tanitad/Scripts/python.exe code/redgreen.py    raw/redgreen.json
```
(`code/c2_cost.py` needs a CUDA device; `code/extract_geom.py` needs the eval pod.)

**Dev-box parity note:** parity-independent. Every input is a persisted per-window output of the
pod-side harness, produced on the canonical `physicalai-train-e438721ae894` (2376 episodes, skip-hash
`f09e44db`). The dev box's own cache (`14231cd29c74`) was never touched. **No episode was re-selected.**

---

## 8. ⛔ ESCALATIONS — these must not sit in a file

1. ⭐ **`−0.3366` must stop being quoted as C2's ungated value.** It is a gate firing on **66.97 %**
   and needs a fitted ridge model over 2-WM features. **The shipped, training-free, no-gate number is
   `0.5645`, `−0.2918 [−0.4233, −0.1598]`.** The producing document is correct; **the relay is what
   drifted**, and it drifted inside a brief that was itself warning about exactly this failure class.
   ⚠️ **Needs the coordinator**: `Project Steering/V5_PLAN.md` §8's last bullet and any downstream
   brief carrying the −0.3366 label. **Agents do not edit steering files — I did not.**
2. 🔴 **Rescue or deliberately drop `tanitad-eval:/workspace/_v5/v5_v1_windows.pt` (114 MB).** It is
   the only copy of the A1/C1 imagination geometry (`imag`, `ctrv`) in the program. A pod is not
   storage. The pod also holds `v5_v4_windows.pt` by the same argument — **not verified, one probe
   only.**
3. ⚠️ **The `0.4907` bar needs its deployment tag before the v5 gate reads it.** It is an 881-window
   / 40-episode number (`a0` 0.4714) and is currently being compared against 600-episode figures.
   C2's 1.15× miss is stated against the 881-window form.
4. ⭐ **The cheapest next experiment is unchanged and still unowned: sweep the SCORER.** v1 was chosen
   because it was already dumped, never because it was the best available. `--c2-scorer` now makes
   that a one-flag sweep over REF-B v2 / REF-C XL / v3enc / v4.1 — **the instrument for it now
   exists in the repo.**

---

## 9. FOR `RETRACTION_LOG.md` — root-cause classes

- ⭐ **C19 recurrence, one hop downstream of the document that named it.** `−0.3366` (frac 0.6697)
  travelled as an unconditional number **into a brief whose own trap list says "report a whole-set
  policy value and a firing rate for anything conditional"**. ⇒ **the firing rate must be carried in
  the same cell as the delta, not in a neighbouring column** — a `selected_frac` in an adjacent table
  cell is not attached tightly enough to survive one relay. The shipped module emits it in the same
  dict as the pick, always.
- ⭐ **C-new: "reproduces exactly" needs its HOST.** My L2 bit-exactness bar failed on a pure
  CPU-vs-A40 float32 reduction-order difference (1.14e-05 m). The admissible bar for a
  cross-device reproduction is **does the DECISION move**, with the decision margin measured
  (here: 0/881 below the noise, but only **2.2×** clear at the tail).
- **C13 avoided, and only because it was checked.** My first artifact-fidelity fixture used a single
  lead time, where `mean` and `sum` are identical — **the fidelity test could not have detected a
  changed cost reducer.** The red/green sweep is what found it. **A guard nobody has seen fail is
  not a guard, and this is the second time in two streams that the sweep caught the tester rather
  than the code.**
- **C-II (unverified premise in a brief) — FOUND, and it is the headline.** The brief's own headline
  table mislabels a gated cell as ungated. Every *other* inherited number in the brief (0.8563,
  0.5645, +0.2090, 0.2505, 0.4271-is-not-a-bar) **reproduced exactly.**
