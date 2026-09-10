# P-RC21 — the 2,000-step RL arm RAN. What it says, and the next lever.

**Master Mind, 2026-09-10 (Europe/Berlin) / 2026-09-11 early.** Dev-box RTX 4060, no spend.
Run tree `C:/Users/Admin/tanitad-rlrun` (git-archived from the tip; the import probe confirmed
`tanitad` resolved to that tree, not the venv's editable G: install).

⛔ **TIER: T0, training-side.** The tool stamps this itself — `_tier: "T0 training-side;
capability claims are T1 only"`. ⛔ **Nothing here is a capability claim, and no row of it belongs
in the leaderboard.** One seed, no paired episode-cluster bootstrap, no CI.

---

## 1. It really ran

| counter | P1 | P2 |
|---|---|---|
| steps | **2,000** | 300 |
| windows scored | 4,000 | — |
| samples scored | **2,048,000** | — |
| optimizer steps | 2,000 | 300 |
| `done` | True | True |
| wall clock | **3 min 18 s** | ~2 min |

Trainable surface: **8,634,120 of 104,191,577 params (8.29 %)**, 68 tensors, decoder heads only.
Components that fired during training: `progress` 2000/2000, `comfort` 2000, `feasibility` 2000,
`collision` 1229, `headway` 666.

⭐ **The longitudinal terms this whole item exists for did fire on real windows.** `H-DDA-5` says
DD-V2's RL gain is a longitudinal-scale effect (EP +5.3, DAC +1.7, NC/TTC/comfort flat), and
`D-REFCV3-AXIS1` says **92.2 % of our own `os − ha` gap is along-track**. The machinery is now
reaching that axis.

## 2. ⛔ The result is NOT usable, and the run says so itself

**P1's own reward audit returns `INCONCLUSIVE`, not clean:**

> *"audit is UNDERPOWERED: weighted component(s) `['collision', 'headway']` were CONSTANT across
> the whole panel, so they cannot influence the ranking. Supply a context that exercises them
> (obstacles / lead_path / gt_traj) before trusting any verdict."*

⇒ ⛔ **Do not quote P1's deltas as evidence of anything.** For the record, and marked unusable:
`R1 +0.6427 → +0.6691` (Δ **+0.0264**), `R2 21.335 % → 20.924 %` (Δ **−0.410 pp**),
`R3 0.654 m → 2.102 m` (Δ **+221 %**).

⚠️ **That R3 move is a 3.2× worsening and it is the number to chase**, not to report. It is exactly
the shape of a reward that buys its headline term by giving something else away — the same trade
the refcv6 bar's non-regression clauses exist to forbid.

## 3. ⭐ THE FINDING — V2's central mechanism was NOT exercised, and cannot be from this pilot

**`use_gt_bar = False`** and **`frac_above_bar_mean = None`** in P1's summary.

⛔ **The pilot exposes no `--gt-bar` flag at all.** Its complete surface is:
`--batch --ckpt --lru --out --proximity-safe-m --reward --seed --steps --train-agents
--train-epdir --val-agents --val-epdir --w-anchor` — thirteen flags, and none of them reaches the
≥GT truncation.

⇒ This is a **V1-style baseline**, not a V2 arm. The stricter truncation
(`mask_positive = reward_group > reward_gt`) was implemented, and proved on the gradient path the
same day — moving only the bar drove a leaf parameter's `.grad` from `+3.20000076` to **exactly
0.0** — and the script that would exercise it **has no way to turn it on.**

⭐ **This is the `tac_goal_tok_head` failure in a new costume**: a mechanism that is built,
tested, gradient-verified and **unreachable from the caller**. *"Rollable and trained are different
claims."* ⇒ **The next lever is a one-flag change to the pilot, not a new experiment.**

## 4. ⚠️ P2 — the reward-hacking control fired, and the audit did not flag it

P2 runs `--reward hackable` deliberately, as a regression control. Its audit reports
**`verdict: clean`, `flagged: False`, `dead_components: []`** — while its own scores read:

| probe policy | score |
|---|---|
| `bullet_straight` | **1.5** |
| `teleport` | **1.5** |
| `shaky` | **1.5** |
| `spinner` | −0.0104 |
| `frozen` | 0.0 |

⛔ **Three distinct degenerate policies tie at the ceiling.** A reward that cannot separate
*drive straight through everything*, *teleport*, and *shake violently* is gameable by construction.

⚠️ **Scope this honestly rather than calling the audit broken:** the `verdict` field reports
**component liveness** — whether every weighted component varied enough to influence the ranking —
and by that definition `clean` is correct here, because nothing was constant. **The gap is that
liveness is not gameability.** ⇒ The audit needs a **separation** check: if two or more probe
policies tie at the maximum, the reward cannot rank them and the verdict must say so. Compare
P1's default reward, which produced a spread (1.45 / 1.26 / 0.95 / 0.50 / 0.30) and no ties.

⭐ **The control did its job — a deliberately hackable reward produced the ceiling-tie signature.**
It was the *verdict field* that was not looking for it.

## 5. The next levers, in cost order

1. ⭐ **Add `--gt-bar` to `rl_pilot_refc21.py`** and re-run P1. One flag; the mechanism underneath
   is already gradient-proven. **Until this runs, the V2 RL stage has not been tested at all.**
2. **Add a ceiling-tie separation check to the reward audit**, so `clean` cannot be returned when
   probe policies tie at the maximum. Deliberate-regression arm: the `hackable` reward must go RED.
3. **Give the audit a context that exercises `collision` and `headway`** (obstacles / lead_path /
   gt_traj), which is what P1's own INCONCLUSIVE verdict asks for. Without it no P1 verdict is
   admissible, with or without the bar.
4. **Then, and only then**, a seeded pair — because on this rig an arm with **zero levers moved**
   read "separably worse" on **5 of 9** family metrics.

## 6. Artifacts

`C:/Users/Admin/tanitad-data/rl-pilot/p1-grpo-2k/` and `…/p2-reg-2k/` — `pilot_summary.json`,
`config.json`, logs. Chain: `C:/Users/Admin/tanitad-rlrun/run_rc21.sh`, waiter
`wait_then_run.sh`.

⚠️ **Two gate notes, both caught by the gate rather than by the run:** the tree check first refused
the **correct** tree because Python printed Windows separators against a forward-slash pattern; and
the GPU-busy check correctly refused while a sibling stream held the card for `pytest`, then a
waiter launched the chain the moment it cleared. ⭐ Both refusals were **artifact-visible markers**,
not exit codes.
