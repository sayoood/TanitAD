# `DIR_YAW_RAD` 0.15 → 0.10 — the owed re-read, closed at 0 GPU

**MEASURED 2026-08-15 (ours)** · 0 GPU, banked artifacts only ·
`tools/gate_reread.py` → `results/gate_reread.json` · dev box,
`/c/Users/Admin/venvs/tanitad`. Closes escalation 4 of
`Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md`.

---

## The verdict-movement answer, first

**YES — the threshold change moves exactly ONE published verdict, and it is one that was
already retracted. It moves NO verdict on any panel that is currently load-bearing. But five
unswept κ sit close enough to a boundary that their verdict WORDS are not established at
0.10, and two of them are the paper's headline tactical claim.**

| | |
|---|---|
| banked panels carrying a gate-dependent κ | **16** (13 distinct results; 3 are pod-rescue duplicates) |
| of those, gate-swept (re-readable at 0.10 without a GPU) | **1** |
| verdicts that MOVE on swept `hier` panels | **0** |
| verdicts that MOVE on the swept Alpamayo comparison | **1** |
| unswept κ whose verdict COULD move inside the measured band | **5** (3 distinct) |

---

## 1. What the constant gates (from source, not from prose)

`DIR_YAW_RAD = rl.YAW_TURN_RAD = 0.15` at `taniteval/taniteval/hierarchy.py:164`. Its only
consumer is `_dir_of` (`:204-210`), which thresholds a **signed 2 s net heading change**
(`gt_net = wrap_to_pi(fut[:, GOAL_H-1, 2] - pl[:, 2])`, `:557`, `GOAL_H = K_MAX = 20` steps)
into `{L, S, R}`.

⚠️ **The blast radius in `RETRACTION_LOG.md` R-2026-08-06-yawgate is OVER-BROAD by one entry.**
It states the constant *"feeds `consistency.maneuver_vs_trajectory`,
`commanded_route_vs_maneuver`, `commanded_route_vs_trajectory` and every `*_turn_subset`."*
From source (`hierarchy.py:868-888`):

| field | gate-dependent? | why |
|---|---|---|
| `maneuver_vs_trajectory.*` | **YES** | `traj` = `_dir_of(traj_net)` |
| `commanded_route_vs_trajectory.*` | **YES** | same `traj` |
| `distributions.trajectory_dir`, `distributions.gt_dir` | **YES** | both `_dir_of` |
| **`commanded_route_vs_maneuver.*`** | ⭐ **NO** | **both sides are gate-free** — `route_n` is `route_logits.argmax` (`:869`) and `man_dir` is `MAN2DIR[man_pred]` (`:870`), a fixed 5→3 table. Its `turn` mask is `(route_n != S) \| (man_dir != S)`, also gate-free. |
| `distributions.{route_follow, route_commanded, maneuver_dir}` | **NO** | no `_dir_of` |

⇒ `commanded_route_vs_maneuver` — agreement, κ, and turn subset — **needs no re-read at any
gate**. That is one fewer number in the blast radius, established from source rather than
inherited.

## 2. Two instrument defects found while doing the re-read

**(a) ⛔ `verdict_stable` does not test the verdict that is published.**
`hierarchy._gate_sensitivity` (`:262`, `:278-282`) computes `verdict_stable` against
**κ ≥ 0.2**. The word actually published — `maneuver_consistency_verdict` in
`taniteval/taniteval/four_families.py:888-890` — uses a **different ladder: < 0.1 DECORATIVE,
< 0.4 WEAK, ≥ 0.4 SUBSTANTIAL**. On the one swept panel the two disagree:

| | across the full swept range 0.15 → 0.01 |
|---|---|
| `verdict_stable` **as reported** (κ ≥ 0.2) | **true** |
| verdict stability on the **published ladder** (0.1 / 0.4) | ⛔ **false** — SUBSTANTIAL at 0.15/0.10/0.06/0.04, **WEAK** at 0.02/0.01 |

The `GATE_RERUN_RESULT.md` headline *"Verdict: STABLE"* is therefore true of a threshold the
programme does not publish. ⚠️ It remains true for the **0.15 → 0.10** move specifically
(SUBSTANTIAL at both) — which is the move this item asks about — so no published statement is
retracted by this. But `verdict_stable` should test the ladder it ships.

**(b) ⚠️ The sweep does not cover the number sitting ON a boundary.** `_gate_sensitivity`
sweeps `maneuver_vs_trajectory_kappa` and `trajectory_vs_gt_kappa` only.
**`kappa_turn_subset` is gate-dependent and is NOT swept** — so the deployed arm's
`kappa_turn_subset = 0.2005`, which sits *on* the panel's own 0.2 line, cannot be re-read at
0.10 even on the swept panel. Work item, not an excuse.

## 3. Per affected number — old, new, verdict

### 3a. Recomputable (banked sweep exists)

**`flagship-v1arch-v2bal-30k`, 880 windows / 40 OOD-val q90 episodes**
(`…/2026-08-06-v1-defect-triage/results/hier_v1arch_gateswept.json.xz`):

| number | @ 0.15 (published) | @ 0.10 | Δ | verdict move |
|---|---|---|---|---|
| `maneuver_vs_trajectory_kappa` | **0.5787** | **0.5715** | −0.0072 | **none** — SUBSTANTIAL → SUBSTANTIAL |
| `trajectory_vs_gt_kappa` | 0.8260 | 0.7781 | −0.0479 | n/a (no ladder) |
| `frac_gt_turning` | 0.1341 | 0.1875 | +0.0534 | n/a |
| `kappa_turn_subset` | 0.2005 | ⛔ **not swept** | — | ⚠️ **undetermined, and it is the one on the line** |

**The Alpamayo comparison, 39 paired OOD clips**
(`…/2026-08-05-alpamayo2-super/comparison/a2_gate_audit.json`):

| series | @ 0.15 | @ 0.10 | Δ | verdict move |
|---|---|---|---|---|
| **`flagship_declared_maneuver` vs driven** | **0.4402** | **0.3743** | −0.0659 | ⭐ **YES — SUBSTANTIAL → WEAK** |
| `alpamayo_declared_lateral` vs driven | 0.1961 | 0.3004 | +0.1043 | none (WEAK both) — but it **crosses the 0.2 line** `_gate_sensitivity` calls a verdict |
| `flagship_executed_vs_gt` | 0.6176 | 0.7263 | +0.1087 | none |
| `alpamayo_executed_vs_gt` | 0.4882 | 0.7292 | +0.2410 | none — **but the two-arm RANKING flips** (ours ahead by 0.129 at 0.15, behind by 0.003 at 0.10) |

⇒ The one verdict that moves is on the 39-clip comparison, which
`RETRACTION_LOG.md` R-2026-08-06-yawgate **already retracted** and which the 880-window panel
supersedes by power. **No live decision moves.**

### 3b. NOT recomputable — 15 panels, and the reason is structural

`hierarchy.py` banked only the **thresholded** `traj_dir`/`gt_dir` until the 2026-08-06 fix
(`:610-620` is the fix). The continuous net yaw was discarded at write time. A 0.10 read of
these is a **GPU re-run, not a recompute** — including the registry's own headline:

| κ @ 0.15 | verdict | n | panel |
|---|---|---|---|
| 0.6938 | SUBSTANTIAL | 265 | `…/2026-07-23-v4-gate-emitters/artifacts/hierarchy_flagship-30k_v1.json` |
| 0.6053 | SUBSTANTIAL | 881 | `…/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-30k.json` |
| 0.5792 | SUBSTANTIAL | 881 | `…/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-v4.2b-dryrun.json` |
| **0.253** | **WEAK** | 418 | `…/2026-08-02-four-family-panel/hier_v1-lf19.json` ⚠️ **at risk** |
| **0.0072** | **DECORATIVE** | 418 | `…/2026-08-02-four-family-panel/hier_v2corpus-lf19.json` ⚠️ **at risk** |
| **0.6033** | SUBSTANTIAL | **6382** | `…/2026-08-05-v1arch-oodval-four-families/raw/v1arch_oodval_q90_4fam_LEAD.json` — **the `MODEL_REGISTRY.md` §-v1arch row and `Paper/TANITAD_PAPER.md:2057`** |
| 0.6058 | SUBSTANTIAL | 6382 | same run, RAWPIXELS control |
| 0.5827 / 0.6123 / 0.682 / 0.6879 | SUBSTANTIAL | 881 | pod-rescue `hier_flagship-nospeed / -speed / refa-dinov2 / refa-dynin-30k` |
| **0.0217** | **DECORATIVE** | 881 | pod-rescue `hier_refa-dynin.json` ⚠️ **at risk** |

*(plus 3 pod-rescue duplicates of rows already listed).*

### 3c. Which unswept verdicts could actually move

The 0.15 → 0.10 move has been MEASURED on **6 series** (§3a). Its range is
**[−0.0659, +0.2410]**. Asking only *"is the distance to the nearest verdict boundary inside
that band?"* — not predicting a value:

| κ | verdict | dist ↑ to 0.4 | dist ↓ to 0.1 | could move? |
|---|---|---|---|---|
| 0.253 (`hier_v1-lf19`) | WEAK | 0.147 | 0.153 | ⚠️ **YES, upward** (+0.147 < +0.241 observed) |
| 0.0072 (`hier_v2corpus-lf19`) | DECORATIVE | 0.0928 | — | ⚠️ **YES** |
| 0.0217 (`hier_refa-dynin`) | DECORATIVE | 0.0783 | — | ⚠️ **YES** |
| every κ ≥ 0.5792 | SUBSTANTIAL | ≥ 0.179 **downward** | — | **NO** — would need −0.179, far outside the observed −0.066 floor |

⭐ **The consequence that matters.** `Paper/TANITAD_PAPER.md:1867` (repeated at `:3127`) says
the manoeuvre-vs-trajectory agreement *"collapses from κ = 0.253 (v1, **weak**) to κ = 0.0072
(v2corpus) … the `--v2` pack's tactical machinery is **decorative**."* **Both verdict words sit
inside the measured move band.** The *collapse* itself is safe — 0.253 vs 0.0072 is a 35× gap,
far larger than any observed gate move — but **"weak" and "decorative" are not established at
0.10**, and this claim is currently unswept.

⭐ **The registry's `κ 0.6033 (SUBSTANTIAL)` is safe.** It needs −0.203 to leave SUBSTANTIAL;
the largest downward move ever measured is −0.0659, and the same arm/ckpt/corpus family on a
40-episode subset of the same q90 corpus moved only **−0.0072**. Evidence class for that
subset comparison: **MEASURED**; for the extrapolation to 6,382 windows: **ESTIMATED** — so
the registry row does not change, and it does not get a "verified at 0.10" stamp either.

## 4. ⭐ New MEASURED result: the gate is mis-scaled on the CANONICAL val corpus too

R-2026-08-06-yawgate established the mis-scaling on **OOD-val only** (39 clips, then 880
windows). 11 of the 15 unswept panels live on **canonical val**, where it had never been
measured. It can be measured **at 0 GPU**, and this is the part of the re-read nobody had
noticed was available.

**The reconstruction.** `taniteval/results/windows_<arm>.pt` banks `head_deg`, written by
`bench.py:399` as `driving_diagnostic.net_heading_change_deg(ep.poses, last)` =
`|wrap(poses[last+K_MAX,2] − poses[last,2])| · 180/π`, `K_MAX = 20`. `hierarchy.py:557` forms
the gate's input as `wrap_to_pi(fut[:,GOAL_H−1,2] − pl[:,2])`, `GOAL_H = K_MAX = 20`. **Same
poses, same horizon, same wrap ⇒ `head_deg · π/180` IS `|gt_net|`.** The sign is lost to
`.abs()`, so κ cannot be rebuilt — but every magnitude-only quantity can.

**Verified, not assumed.** `frac_gt_turning(0.15)` reconstructed from `head_deg` was checked
against `1 − P(straight)` of each panel's own `consistency.distributions.gt_dir`:

| arm | from the panel's `gt_dir` | from `head_deg` | |Δ| |
|---|---|---|---|
| flagship-30k · flagship-speed · flagship-nospeed · refa-dinov2 · refa-dynin-30k | 0.2168 | 0.2168 | **0.000000 — bit-exact, 5/5** |

**The measurement** (881 windows / 40 canonical-val episodes; a corpus property, so identical
across all 26 arms with a dump):

| corpus | n | median \|net yaw\| | p90 | gate ÷ median | frac turning @ 0.15 | @ 0.10 |
|---|---|---|---|---|---|---|
| **canonical val** *(NEW)* | 881 | **0.0181 rad** (1.04°) | **0.4611** | **8.3×** | **21.68 %** | **25.65 %** |
| OOD-val q90, 40 ep | 880 | 0.0171 | 0.2095 | 8.8× | 13.41 % | 18.75 % |
| OOD-val, 39 A2 clips | 39 | 0.0230 | 0.1850 | 6.5× | 17.95 % | — |

⇒ **The mis-scaling is a property of 2 s windows on this data, not of one corpus.** On
canonical val the published gate is **8.3× the human's own median turn** and still admits only
**21.7 %** of windows as "turning".
⚠️ **And canonical val is NOT the same distribution as OOD-val**: same tiny median, but a p90
of **0.4611 vs 0.2095** — a **2.2× heavier turn tail**. So the OOD-val gate sweep cannot be
transplanted onto the canonical-val panels; the 15 unswept κ genuinely need their own read.

## 5. What this closes and what it leaves open

| item | status |
|---|---|
| *"does 0.15 → 0.10 flip any interpretation?"* | ⭐ **ANSWERED.** One — the already-retracted 39-clip flagship declared-manoeuvre κ (SUBSTANTIAL → WEAK). None on any live panel. |
| the registry's `κ 0.6033 (SUBSTANTIAL)` | **stands**; no edit proposed. Not re-verifiable at 0.10 without a GPU. |
| the blast radius in R-2026-08-06-yawgate | ⚠️ **one entry too many** — `commanded_route_vs_maneuver` is gate-free (§1). |
| `verdict_stable` vs the published ladder | ⛔ **defect** (§2a) — 6-line fix in `hierarchy._gate_sensitivity`. |
| `kappa_turn_subset` not swept | ⛔ **gap** (§2b) — and it is the number on the boundary. |
| the paper's *"weak"* / *"decorative"* tactical words | ⚠️ **unswept and inside the move band** (§3c). Cheapest fix: re-run the two `-lf19` panels gate-swept (~2×140 s on an idle A40). |
| the other 12 unswept panels | **not urgent** — every one is ≥ 0.179 from its boundary, i.e. outside the measured band. |

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `DIR_YAW_GATE_REREAD.md` (this file) | repo `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-15-dir-yaw-gate-reread/` | staged |
| `tools/gate_reread.py` (0-GPU re-read + bit-exact cross-check) | same dir | staged |
| `results/gate_reread.json` (every number above, machine-readable) | same dir | staged |

**Nothing committed, nothing pushed.** No registry number was edited — §3c/§5 are the
escalation, per the source-of-truth rule.
