# PRE-REGISTRATION — distance-keeping discrimination control (D-LEAD-1)

**Written and committed BEFORE the runner was executed.** Both outcomes, plus an
INSTRUMENT-FAIL branch, are fixed here. Date: 2026-08-03. Owner: Architecture & Inference.

## Why a control at all, before any arm is scored

`GATE_PROTOCOL` §0.7 and retraction **C63** say the same thing from two directions: C63 imported a
published metric (compounding ratio on decoded displacement) without measuring its **precondition**
on our stack, and its numbers turned out to be an artifact — a control arm that cost one batch would
have killed the design in minutes. Its prereg also had **no INSTRUMENT-FAIL branch**, so an
out-of-range result had nowhere to go.

Distance-keeping is exactly the same shape of import: TTC / time-headway are the field's standard
longitudinal criticality family ([arXiv 2603.28029](https://arxiv.org/html/2603.28029)) and are
first-class sub-scores inside every closed-loop score we benchmark against (NAVSIM PDMS/EPDMS,
Bench2Drive DS). ⇒ **before it scores a single arm, the instrument must be shown to move on our
windows.** A metric that cannot separate a good trajectory from an obviously bad one is not a
weak result, it is a broken gauge.

## H0 / H1

* **H0 (null):** on the PhysicalAI val windows, distance-keeping cannot distinguish the human's
  true future path from a hold-`v0` constant-velocity path.
* **H1:** it can — CV, which never brakes and never steers, closes on the lead that the human
  actually kept distance from.

## Design

| | |
|---|---|
| **arms** | **GT** = the human's true future ego positions · **CV** = hold-`v0`, straight ahead at the t0 speed. Both in the same window-origin frame. |
| **windows** | every t0 on a 1.0 s stride inside the measured `egomotion` ∩ `obstacle.offline` span, horizon 2.0 s, dt 0.5 s (4 waypoints — the grid the programme's ADE@2s uses). |
| **corpus** | the `obstacle.offline` chunks present on the dev box. ⛔ **read-only**; no clip is re-selected, parity key `physicalai-train-e438721ae894` / skip-hash `f09e44db` untouched. |
| **lead selection** | strictly causal at t0 (samples ≤ t0 only), `lead_state_gate.lead_frame`'s rule: nearest vehicle ahead, ≤ 80 m, inside \|lat\| < 2 m. Enforced at runtime by `assert_selection_causal`. |
| **estimator** | **paired episode-cluster bootstrap** (`taniteval.ci.paired_episode_cluster_bootstrap`), clusters = clips, B = 2000, seed 0. ⛔ never `overlapping_holdout_se` — it biases the point estimate as well as the interval (CLAUDE.md). |
| **primary** | Δ `min_ttc_s` (GT − CV). **Secondary:** Δ `headway_min_m`, Δ `time_gap_min_s`. |

## Outcomes, fixed in advance

1. ⭐ **PASS — instrument ADMISSIBLE.** Δ `min_ttc_s` CI excludes 0 **and** the sign is
   GT > CV (the human keeps more time-to-collision than a policy that never brakes).
   ⇒ wire `distance_keeping` into `four_families.longitudinal`, replacing the `UNAVAILABLE`
   stub, and report it for every arm from that point on.
2. **FAIL — NOT-APPLICABLE.** CI spans 0. ⇒ our val windows are too free-flow for distance-keeping
   to bite. Report the family as **NOT-APPLICABLE with its n and this reason**, per the binding
   rule's clause 5 — do **not** publish a metric that cannot move, and do **not** quietly retry
   with different thresholds.
3. ⛔ **INSTRUMENT-FAIL (the branch C63 lacked).** Any of: the CI excludes 0 with the **wrong sign**
   (CV safer than the human); `n_lead` < 100 windows or < 10 clusters; > 50 % of admissible windows
   censored at `TTC_CAP_S` in **both** arms. ⇒ **no verdict is issued.** The result is reported as
   an instrument defect with the diagnostic that fired, and the metric stays out of
   `four_families` until it is fixed. A wrong-signed separation is *not* a finding about driving.

## What this control does NOT establish — do not infer it later

1. It says **nothing about any TanitAD arm.** GT-vs-CV measures the *gauge*, not a model.
2. It does **not** close the 88.7 % longitudinal gap. It builds the instrument that can finally
   *see* the distance-keeping half of it.
3. **`min_ttc_s` is censored** at `TTC_CAP_S` = 30 s whenever the arm is not closing. Its mean is a
   mean over censored data — `n_closing` must be quoted beside it, always.
4. Coverage is **the chunks on this disk**, not the canonical 2,376-episode corpus. Any number here
   is a *gauge property*, not a corpus statistic, until it is re-run over the full corpus.
