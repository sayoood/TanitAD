# `DIR_YAW_RAD` 0.15 → 0.10 — the two panels the PAPER quotes in words

**MEASURED 2026-08-16 (ours)** · **0 GPU, CPU only, 2.6 s** ·
`code/gate_envelope.py` → `raw/gate_envelope.json` · dev box `/c/Users/Admin/venvs/tanitad`.
Closes the item left open by
`…/Benchmarks & Eval/Implementation/incoming/2026-08-15-dir-yaw-gate-reread/DIR_YAW_GATE_REREAD.md` §5
(*"the paper's 'weak' / 'decorative' tactical words — unswept and inside the move band"*).

---

## The answer, first

| the paper's word | at gate 0.15 | at gate 0.10 | survives? |
|---|---|---|---|
| **v2corpus is "decorative"** | κ **0.0072** | κ ∈ **[−0.0398, +0.0572]** for *any* crossing rate 0–33.5 % | ⭐ **YES — unconditionally.** Never reaches the 0.1 line anywhere in the admissible set. |
| **the κ "collapse" v1 → v2corpus** | 0.253 vs 0.0072 | v1's floor stays above v2corpus's ceiling until a **14.6 %** crossing rate | ⭐ **YES.** 14.6 % is ~3× the largest band mass ever measured on this programme. |
| **v1 is "weak"** | κ **0.253** (WEAK) | a **0.96 %** crossing rate already admits **SUBSTANTIAL** | ⛔ **NO — not established.** Bounded away from DECORATIVE (needs 7.4 %), but the word itself is not admissible at 0.10. |

⇒ **Two of the three claims survive; one word was removed from the paper.** Both edits are in §4.

---

## 1. ⛔ This is NOT the re-run that was owed — and the re-run is not executable anywhere

The owed action was *"re-run the two `-lf19` panels gate-swept (~2×140 s on an idle A40)"*.
⚠️ That estimate was an **A40** estimate; the brief that reached me had it as *"CPU-only, ~140 s
each"*. It is neither, because the inputs are gone. Probed, and absent at every location
(rule: absence at ONE location is not absence):

| input needed | probe | state |
|---|---|---|
| corpus `/root/valdata/val19_leakfree` (19 leak-free episodes) | `find . -name 'ep_*.pt'` over the whole repo | **0 hits.** It lived on the eval pod, shut down 2026-08-02 (`Project Steering/POD_SHUTDOWN_2026-08-02.md`). |
| `windows_v1-lf19.pt` / `windows_v2corpus-lf19.pt` | `stack/experiments/pod-rescue-20260802/eval/` | **not rescued** — the bundle holds 7 JSONs and **no tensors**. |
| the swept panel's raw net yaw | `hier_v1arch_gateswept.json.xz` | banks summary fields only; **no per-window arrays**. |
| a GPU | fleet | the only one is the **Thor, training a 336 M model**. Out of bounds. |

The checkpoints *are* local (`_pod_backup/pod2-2026-08-03/ckpts/flagship4b-speedjerk-30k_ckpt.pt` =
v1, `flagship4b-v2-30k_ckpt.pt` = v2corpus — `make_v2_videos.py:15-17` names the mapping). **The
corpus is the blocker, not the weights.**

## 2. ⭐ What replaced it, and why it is exact rather than a guess

**The banked panel carries the COMPLETE sufficient statistic for the coherence κ.**
`hierarchy._kappa` (`taniteval/taniteval/hierarchy.py:248-252`) is a function of exactly three
quantities, and all three are in the JSON: the manoeuvre-direction marginal, the
trajectory-direction marginal, and the agreement rate. Verified as a **hard gate before anything
else runs** (`verify_reconstruction`):

| panel | md marginal | traj@0.15 marginal | agreement | κ reconstructed | κ published | Δ |
|---|---|---|---|---|---|---|
| `v1-lf19` | [21, 388, 9] | [10, 405, 3] | 0.9258 | **0.2530** | 0.253 | **0.000000** |
| `v2corpus-lf19` | [8, 6, 404] | [15, 384, 19] | 0.0646 | **0.0072** | 0.0072 | **0.000000** |

**And the gate move is MONOTONE, which turns "unknowable" into "bounded".** `_dir_of`
(`hierarchy.py:204-210`) thresholds a *signed* net yaw at ±g, so lowering g 0.15 → 0.10 can only
move a window **S→L** or **S→R** — never L→R, L→S, R→L or R→S. So κ@0.10 depends on exactly two
unknowns: **m**, how many trajectory windows sit in the band (0.10, 0.15], and which manoeuvre
classes those m windows carry. Both are constrained by the banked margins.

`code/gate_envelope.py` computes the **exact envelope [κ_min(m), κ_max(m)]** over

* **every** 3×3 confusion matrix consistent with the two marginals and the agreement rate
  (46 admissible for v1, 30 for v2corpus — enumerated, not chosen), and
* **every** crossing pattern consistent with that matrix.

Two correctness gates, both of which earned their place:
* `_selftest_delta_range` (300 random cases vs full integer brute force) — **caught a sign
  inversion in the first implementation** that would have published a mirrored envelope.
* `_selftest_reduce_cols` (200 cases) — gates the reduction that makes it run in 2.6 s instead
  of ~10⁶ solver calls.

## 3. The numbers

**`v1-lf19` — 418 windows / 19 episodes, ckpt 30000, κ@0.15 = 0.253 (WEAK)**

| crossing count m | as % of windows | κ_min | κ_max | verdict span |
|---|---|---|---|---|
| 0 | 0 % | 0.2530 | 0.2530 | WEAK |
| **4** | **0.96 %** | **0.2224** | **0.4017** | ⛔ **WEAK … SUBSTANTIAL** |
| 17 | 4.07 % | 0.1493 | 0.7530 | WEAK … SUBSTANTIAL |
| 30 | 7.18 % | 0.1008 | 0.8096 | WEAK … SUBSTANTIAL |
| **31** | **7.42 %** | **< 0.1** | — | first point DECORATIVE becomes admissible |
| 60 | 14.35 % | 0.0338 | 0.5496 | DECORATIVE … SUBSTANTIAL |

**`v2corpus-lf19` — 418 windows / 19 episodes, ckpt 29999, κ@0.15 = 0.0072 (DECORATIVE)**

| crossing count m | as % of windows | κ_min | κ_max | verdict span |
|---|---|---|---|---|
| 0 | 0 % | 0.0072 | 0.0072 | DECORATIVE |
| 17 | 4.07 % | −0.0382 | 0.0244 | DECORATIVE |
| 60 | 14.35 % | −0.0388 | 0.0333 | DECORATIVE |
| **140** | **33.49 %** | **−0.0398** | **0.0572** | ⭐ **still DECORATIVE** |

⭐ **Why v2corpus is immovable, and it is a real finding rather than luck.** Its manoeuvre head is
**degenerate — 404 of 418 windows (96.7 %) carry one class** (`distributions.maneuver_dir`
= [8, 6, 404]). `man_dir` is `MAN2DIR[man_pred]`, a fixed 5→3 table with **no gate in it**
(`hierarchy.py:163, 869-870`). A rater that emits a near-constant cannot agree with anything
beyond chance no matter where the *other* rater's threshold sits. **The "decorative" verdict was
never the gate's opinion in the first place.**

**Central line (ESTIMATED, not measured)** — crossers drawn *independently* of the manoeuvre class,
i.e. the null that the band carries the same manoeuvre mix as the rest of the straight pool:

| m | 5 | 17 | 30 | 45 | 60 |
|---|---|---|---|---|---|
| v1 | 0.2257 | 0.1776 | 0.1424 | 0.1142 | 0.0939 |
| v2corpus | 0.0074 | 0.0078 | 0.0082 | 0.0087 | 0.0093 |

Under that null v1 stays WEAK to m ≈ 57. **The envelope, not this line, is the admissible
statement** — the line is quoted only to show that the SUBSTANTIAL corner at m = 4 is an extreme
corner (it requires all four band windows to be declared-turn windows crossing the matching way,
out of ≤ 30 such windows in a pool of 405), not a central expectation.

### What m is plausible — MEASURED, on other corpora

| corpus | n | median \|net yaw\| | frac > 0.15 | frac > 0.10 | **band (0.10, 0.15]** |
|---|---|---|---|---|---|
| canonical val, GT | 881 | 0.0181 rad | 21.68 % | 25.65 % | **3.97 %** |
| OOD-val q90, GT | 880 | 0.0171 rad | 13.41 % | 18.75 % | **5.34 %** |

Reconstructed from `head_deg` banked per window (`bench.py:399` writes
`net_heading_change_deg(ep.poses, last)` over `K_MAX = 20` — the **same poses, horizon and wrap**
`hierarchy.py:557` uses for the gate's input) and from the one gate-swept panel.
⚠️ **These are GT band masses on OTHER corpora and cannot be transplanted onto `lf19`**: the lf19
corpus is far straighter (its own `gt_dir` gives only **2.87 %** of windows above 0.15, vs 21.68 %
on canonical val), so its band mass is smaller — scaling by the two measured
band/above-gate ratios (0.183, 0.398) puts **m ≈ 2–6 windows**. That is ESTIMATED, and it is
precisely why v1's word cannot be rescued: **m ≈ 2–6 straddles m\* = 4.**

⚠️ Also MEASURED and worth keeping: on the one panel that *was* swept, the same move cost only
**−0.0072** (κ 0.5787 → 0.5715). The envelope is a bound on what the banked numbers *admit*, not a
prediction; the one time this was actually measured, the move was tiny.

## 4. What was changed in the paper — exactly

`Paper/TANITAD_PAPER.md`, both sites the words live at (verified by content, not line number):

**(a) §"the four families say why it is worse"** — was:
> `manoeuvre-vs-trajectory agreement collapses from κ = **0.253** (v1, weak) to κ = **0.0072** (v2corpus)`

⇒ **"(v1, weak)" → "(v1)"**, the gate `DIR_YAW_RAD = 0.15` stamped on both κ, and a re-read
sentence added recording that *decorative* and *the collapse* hold at 0.10 while v1's word is not
established (0.96 % admits SUBSTANTIAL; DECORATIVE needs 7.4 %).

**(b) the §7 summary bullet** — was:
> `tactical κ 0.253 → 0.0072 (decorative)`

⇒ gate stamped, and *"the word and the collapse both survive a 0.10 re-read; v1's own word does
not, and is not quoted"* added.

**Nothing else changed.** `MODEL_REGISTRY.md` was not touched — no registry number is affected
(the registry's own κ 0.6033 was already shown safe in the 2026-08-15 pass).

## 5. Still open — escalated, not written into a doc as "please merge"

1. ⛔ **`hierarchy._gate_sensitivity` computes `verdict_stable` against κ ≥ 0.2** (`:262`,
   `:278-282`) **while the programme publishes a different ladder** — `< 0.1 DECORATIVE,
   < 0.4 WEAK, ≥ 0.4 SUBSTANTIAL` (`four_families.py:886-889`). A ~6-line fix, already logged
   2026-08-15 and still unfixed. **This is the defect that let the two words go unchecked**: the
   sweep reported "stable" about a threshold nobody quotes. Not touched here to keep this change
   to the paper + new files, but it should be the next thing done.
2. ⛔ **`kappa_turn_subset` is gate-dependent and is NOT swept.** For `v1-lf19` it is **−0.1898**
   on n = 37 turn-active windows; for `v2corpus-lf19` **0.0056** on n = 413. Neither is quoted in
   the paper, so nothing is at risk today, but the instrument gap stands.
3. ⚠️ **The corpus `val19_leakfree` is unrecoverable**, so these two panels can never be re-read
   exactly without rebuilding it (`Project Steering/eval_corpus/README.md` documents the build,
   and supersedes it with a 290-clip official split). Any future arm scored on lf19 should bank
   the raw net yaw — the 2026-08-06 fix (`hierarchy.py:610-620`) already does this for new panels.

## Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| `DIR_YAW_REREAD.md` (this file) | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-dir-yaw-reread/` | staged |
| `code/gate_envelope.py` (exact envelope + 2 brute-force self-tests) | same dir | staged |
| `raw/gate_envelope.json` (every number above, machine-readable) | same dir | staged |
| paper edits (2 sites) | `Paper/TANITAD_PAPER.md` | staged |

**Nothing committed, nothing pushed. No GPU used. No pod contacted.**
