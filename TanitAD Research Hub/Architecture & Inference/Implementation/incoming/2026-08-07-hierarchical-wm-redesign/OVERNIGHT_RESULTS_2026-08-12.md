# Overnight results — 2026-08-11 → 2026-08-12

**For:** Sayed · **Evidence class:** every number below is **MEASURED (ours)** with its artifact
path, unless stamped otherwise. **Source of truth:** `Project Steering/MODEL_REGISTRY.md` §1.14.

---

## The one thing to read first

**We now have T1 pseudo-closed-loop numbers, and they change the story.**

The stage-A repair wins decisively — closed-loop ADE **23.98 → 9.37 m**, paired
**−14.61 [−16.93, −12.20]**, separated. But a **hold-action control beats the repaired model
by 22×** (0.42 vs 9.37 m), and the divergence is **~99 % longitudinal**: the car holds its lane
while its speed integrates away.

**We cannot claim closed-loop driving competence for either arm.** That is not a setback — it is
the measurement the programme was missing, and it makes the v6 staged ladder empirically
motivated rather than merely architecturally argued.

---

## 1. T1 pseudo-closed-loop — the headline

TIER **T1 = PRIMARY**. 6 844 windows / 40 val episodes, stride 1, episode-cluster bootstrap.
`ol` = teacher-forced (**T0**, a WM diagnostic, never driving performance); `cl` = action-closed;
`ha` = hold-action control.

| arm | surface | tier | ADE (m) | FDE (m) | LON speed MAE (m/s) | LON along (m) | LAT cross (m) |
|---|---|---|---|---|---|---|---|
| `v5f-30k` | `cl` | **T1** | **23.9837** [21.44, 26.35] | 53.4756 | 26.9356 | 23.8965 | 0.9993 |
| `v5f-30k` | `ol` | T0 | 0.9397 [0.82, 1.07] | 2.8003 | 1.4431 | 0.8762 | 0.1947 |
| `v5f-30k` | `ha` | **T1** | 0.9597 [0.84, 1.09] | 2.8631 | 1.4531 | 0.8901 | 0.2072 |
| `stage-a-repaired` | `cl` | **T1** | **9.3697** [6.68, 12.26] | 19.5256 | 9.7291 | 9.2655 | 0.7446 |
| `stage-a-repaired` | `ol` | T0 | 0.3659 [0.29, 0.45] | 1.0231 | 0.5113 | 0.2990 | 0.1534 |
| `stage-a-repaired` | `ha` | **T1** | 0.4246 [0.35, 0.51] | 1.2242 | 0.5671 | 0.3487 | 0.1689 |

**Finding 1 — the repair works, on exactly the axis it targeted.** Paired
`stage-a-repaired − v5f-30k`: `cl` ADE **−14.6139 [−16.9319, −12.2010]**, `cl` LON speed MAE
**−17.2064 [−19.7815, −14.4927]**, `ol` −0.5739, `ha` −0.5351 — all separated, `p_delta_gt0` 0.0.
Stage-A restored action-response gain 0.27 → 0.971/0.966 with longitudinal sign 1.0, and the
closed loop improves 2.6×. That is a clean confirmation of the repair's mechanism.

**Finding 2 — the closed loop diverges.** Within-arm `cl − ol`: **+9.0039 [6.37, 11.85]**
(repaired), **+23.0439 [20.56, 25.39]** (v5f), both separated. T0 **0.3659** vs T1 **9.3697** on
the *same checkpoint and the same windows* is a **25× gap**. This is the strongest evidence yet
for the tier doctrine, and precisely the failure it exists to expose.

⚠️ Honest scope: `ha` is a strong baseline partly *because* the corpus is short-horizon and
near-constant-speed. That is what makes it the right "do nothing clever" floor — not a reason to
discount it.

---

## 2. The four families — the mechanism, and why the rule earned its cost

Rescored on the same grid (`FF_EXIT=0`, 6 arms, 15 paired contrasts). The binding rule is now
satisfied on these rows: `rule_satisfied: true`, only STRATEGIC unavailable.

### LONGITUDINAL — a runaway acceleration with a known sign

| | `stage-a-repaired · cl` |
|---|---|
| speed MAE / **bias** | 9.7291 / **+9.3892** (bias ≈ 96 % of the error) |
| along-track MAE / bias / final bias | 9.2655 / +9.0407 / **+18.5801** |
| accel MAE | **19.0948 m/s²** (> 1.9 g — physically impossible) |
| **ego progress ratio** | **1.7279** (median 1.0994) — drives 1.73× the human's distance |
| target-speed accuracy @ 0.5 / 1.0 / 2.0 m/s | 0.3398 / 0.5069 / 0.6564 |

⇒ The v6 longitudinal work item is **a systematic over-acceleration**, not scatter. That is far
more actionable than "ADE is 9.37".

### LATERAL — healthy, and not the problem

heading MAE **3.8776°** · yaw-rate MAE **4.9188 °/s** · curvature MAE **0.0186 1/m** ·
cross-track MAE **0.7446 m** (final 2.1565). All four members the rule names are present.

### TACTICAL — longitudinal decision-making is at chance

| decision axis | accuracy | Cohen's κ |
|---|---|---|
| lateral | 0.7515 | 0.3795 |
| **longitudinal** | 0.3327 | ⛔ **0.0405** |
| collapsed 5-way | 0.3036 | 0.1404 |
| lateral — hold-action | 0.8675 | 0.6427 |
| longitudinal — hold-action | 0.5586 | 0.2072 |

**κ 0.0405 is chance agreement.** And the collapsed 5-way sits *between* the two axes, reporting
neither — this is **the direct measurement of the lat/lon-mixing softmax defect**, visible only
because the family is reported factored. Per-class lateral: `lane_keep` 0.8092 / 0.8747;
`turn_left` 0.4994 / 0.6627; `turn_right` recall 0.6003 but **precision 0.2879** (1 195 predicted
against 573 true).

### TACTICAL goal-setting — the direction is right, the distance is wrong

Goal bearing MAE **4.8098°** (bias −1.83°) against goal range ratio **1.7584**, long-bias
**+18.58 m** vs lat-bias **−1.21 m**. ⇒ **the model knows WHERE to go and not HOW FAR** — a
sentence that ADE, and even ADE-plus-FDE, cannot express.

### STRATEGIC — `n/a` with reason and n = 6 844

PhysicalAI-AV has no map, no lane graph, no junction label, no traffic-light feature and no
route signal (the dataset card says verbatim *"we do not include open maps data"*). No rescore
can close this; the instrument is the VLM pipeline PH0→PH1→PH2. Distance-keeping is likewise
UNAVAILABLE pending a lead block on this dense grid — **a work item, not a pass**, and the half
where 88.7 % of the T0 oracle gap was measured to live.

---

## 3. The other verdicts banked overnight

| result | verdict |
|---|---|
| **W7-FULL** (selector-free planner, 256/256 candidates, oracle 0.1273) | ⛔ **FAIL 3.3348** vs gate 0.4505 — **winner's curse**: the argmin's error-rank is **132 of 256, the median**; top-m mean error flat at ~5.32 (the fan's own mean). A cheap untried fix exists: the anti-degeneracy progress term `--w-prog` has been **weight 0.0 in every W7 run**. |
| **H-COTRAIN** | **REJECTED within range** — neither pre-registered CONFIRM condition fired. Every probed physical variable became *more* decodable (curvature 0.213→0.513 enc, 0.225→0.704 pred; yaw 0.583→0.869), participation ratio **expanded 53 %** (4.53→6.94), P1 went FAIL→PASS at 20 k. The standing hypothesis that planner co-training crushes physical representation is **refuted within the measured range**. Scope: lowest available λ is 0.5, not 0. |
| **SIGReg** | ✅ **VALIDATED** — retention **1.53×** against a ≥0.8× gate (you asked for this explicitly). |
| **P8 BEV occupancy** (attempt 2) | ✅ **GATE PASS** — retention **0.932** at k=10 (IoU 0.01869 pred / 0.02005 enc, τ*=0.7 chosen on the *encoded* arm so the gate can only get harder), a **74× lift**. P4 permanence rides it: occluded recall ≥ visible. ⚠️ absolute IoU ~0.02 — the **ratio** is the quotable claim. |
| **I4a imagination ablation** | Imagination is **load-bearing**: intact 0.4011 / shuffled 1.2492 / zeroed 7.6493. |
| **Consumer invalidation** | Repairing a trunk invalidates every frozen consumer (frozen selector 0.7933 → **4.4159**). ⇒ v5.8f ships the **frozen-trunk assembly (0.4815)**, and this *is* the staged-training argument for v6. |

---

## 4. v6 — ready to start

**S-W is GO on code, blocked only on provisioning.** Trainer built and committed:
`stack/tanitad/models/v6.py` + `stack/scripts/train_v6_staged.py` + 80 CPU tests green.
Shared-encoder arm **87.89 M**; per-layer **120.74 M**; the E-ENC matched pair is 118.11 M
(2.2 % gap, so a real matched comparison). S-T is gated on S-W's gate JSON; **S-S is
structurally blocked until PH2** wires `g_str`.

**Good news on the 6 s horizon:** v6 derives `max_horizon` **per stage**, so S-W and S-S keep
94 windows/episode — the −42.6 % window loss hits **only S-T/S-J**. It is not a launch blocker.

**Decisions waiting for you** — full sheet in `PI_DECISIONS_2026-08-12.md`:

1. **S-W A40-hours before the first gate.** ESTIMATED 175–290 h / 30 k steps; recommendation is
   to authorise **≤12 A40-h to a step-500 re-cost**, with all four branches pre-registered.
2. **Which pod** — a dedicated A40 is recommended; both current pods have tracks.
3. E-ENC (recommend shared for the first S-W), W5/E-H1 in parallel, and an S-S gate amendment.

---

## 5. Artifacts and images

- **BEV belief reel** (camera │ what the WM believes │ belief ∩ truth):
  `p8_belief_reel.mp4`, `p8_belief_still.png`, `p8_belief_sheet.png` —
  on pod4 at `experiments/p8-occupancy-c/reel/` and on HF at `release/v58f/media/`.
- **Figures**: `Paper/figures/v6_architecture.png` (the v6 diagram you asked for),
  `v58f_results.png`, `winners_curse.png` (new).
- **Paper**: `Paper/TANITAD_PAPER.md` at **v1.0** (+882 lines) — the four verdicts with their
  mathematics, the winner's-curse formalism, the tier and four-family doctrines as method
  sections, and the v6 roadmap.

---

## 6. What is still open

| item | state |
|---|---|
| **VLM pilot (PH0)** | Bridge **fixed** (8 clips, 199 frames each, 0 failures — it had been silently skipping every clip on a key mismatch). The pilot then failed because the fallback arm `Qwen3.5-9B` is **text-only** and rejects video kwargs. Re-running the arm smoke on the now-free GPU with the 4-bit `gemma-4-31B` and `Qwen3.5-27B-FP8`. |
| **Distance-keeping / headway** | Needs a lead block on the T1 dense grid (`tools/build_lead_block.py`). |
| **STRATEGIC family** | Blocked on the corpus; PH2 is the instrument. |
| **HF release bundle** | Media pushed; the ckpt/gate bundle runs from pod5. |

### Two things that cost time overnight, now written into `CLAUDE.md`

- **`uv pip install <anything>` can silently replace torch with a wheel the driver cannot run.**
  Measured twice on pod4: `-U accelerate`, then `compressed-tensors`, each pulled
  torch 2.13+cu130 onto a CUDA-12.8 driver and took the GPU offline. Neither command names torch.
- **An analysis-time import that fails after the rollout destroys the output while the compute is
  already paid for.** Both T1 arms rolled all 40 episodes (~11 min each) and died on
  `from taniteval import selgap`. `--analyze-only` over the banked dumps recovered every number
  with **zero GPU**.
