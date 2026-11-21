# SPEC — P2 probe-leak audit (pre-registration of the audit itself)

`Research Lab ask (a), 2026-09-02 · register row D-V7-READINESS-2026-09-02 · Architecture & Inference FlyWheel · 2026-09-03 · 0 GPU (dev-box CPU only; Thor and every pod untouched)`

**Tier of everything in this package: `T0-DIAGNOSTIC`.** Nothing here is a driving claim. The one T1 instrument audited (the hold-action control) is audited for leak structure, not re-scored.

## 0. Why this SPEC exists, and the honest ordering

The Lab's Delta-JEPA brief (`TanitAD Research Lab/Frontier Scan/LEDGER_A2_jepa.md` entry 2026-09-02-01, backlog **GS-1**) warns that an action-decodability probe reading concatenated endpoints `[z_t, z_{t+1}]` can pass on *"action-correlated cues"* absorbed into `z_{t+1}` *"without requiring the model to represent the actual transition"*, and that **any P2 branch eliminated on such a probe must be re-opened**. The v7 launch gate P2 ("the model does not use its actions") rests on a chain of probes (MM-E10…E19, E-DEC-28b/30/40/48b/49/57/59, the L1–L3 ladder, the hold-action T1 read). This SPEC fixes, before the per-probe verdicts are written, **what counts as a leak, what a leak does to a conclusion, and what each 0-GPU re-score may claim under either outcome.**

⚠️ **Ordering, stated rather than implied.** The probe *code* was enumerated and read in the same session as this SPEC (the audit cannot be designed without knowing which tensors the probes consume). The commitments below — the taxonomy in §2, the survival rule in §3, the controls checklist in §4, and the outcome tables in §5 — were written **before any re-score was run and before any per-probe verdict was written into `RESULT.md`**, and were not edited afterwards. Every number in `RESULT.md` was produced after this file was saved. Two pre-run amendments are marked **[PRE-RUN AMENDMENT]** in §5: they record checkpoint availability discovered while staging assets, and an estimator defect found while reading the L3 code — both found *before* the re-scores were launched.

## 1. Scope — the probe set audited

Every instrument that the P2 narrative (`Project Steering/V7_LAUNCH_GATE.md` §P2, §P4, §P5; `V7_RECIPE_AND_SCALEUP.md` §2, §4, §8; `PREREG_MM_E19_K60_HORIZON.md`; the MM-E rows of `GOALS_AND_CLAIMS.md`) cites as evidence that the predictor does not use its actions, or that supported eliminating a P2 branch:

| id | instrument (code) | what P2 took from it |
|---|---|---|
| MM-E10 / E13 / E19 / E19-K8 / SMAS-1 | `actdiv_thor.py` / `actdiv_local.py` (action-divergence, roll variants) | action/scene ratio 0.004–0.006; 2k→30k acquired from ~0; k=60 → 0.50×; k8 control 1.04× |
| MM-E11 | same instrument on `o1ctrl30k` | O1 on → 0.40× (objective family eliminated) |
| MM-E12 | `code_actinfo_thor.py` (ridge latent→action) | the action is NOT redundant given the scene |
| MM-E17 / E18 | `condpath_thor.py` (per-stage roll spread); weight norms | attenuation 56×/128×; FiLM gain converged |
| E-DEC-28b/28c | `actionshuf.py`, `nrmse_shuf.py` | ẑ's advantage was smoothing; nrmse unchanged under action shuffle |
| E-DEC-30 (+ o11 degenerate) | `actchan.py` | 251 % action change moves ẑ 2–8 % of a 10 % latent nudge; o11p30k degenerate |
| E-DEC-40 | `deltaz.py` | Δz is 64 % drift; action adds nothing |
| E-DEC-48b | `confound2.py` | action marginal to the future scene ≈ 0 / negative |
| E-DEC-49 (+ echo test) | `egofuture.py`, `echocheck.py` | action → Δspeed +0.34; latent/ẑ ≈ 0 |
| E-DEC-57 | `kinident.py` | the action IS realised motion (r 0.9988) |
| E-DEC-59 / MM-E19 §3b | `latentmotion.py` | ego marginal over drift −0.0006 (t −0.48); same at k=60 |
| L3 (E-DEC-29, ladder) | `envpred.py` | `splitp30k` predictor adds nothing (t −3.69/−5.62/−6.26) |
| O11 breakout | `train_v6_staged.py::o11_counterfactual_action_loss` | the one arm that became action-sensitive (pick_acc 1.0) |
| T1 hold-action / echo | `taniteval/tools/t1_eval.py` (`roll_closed*`, `ha`), `taniteval/v0_antiecho.py` | P1: every arm loses to hold-action; echo clean |

Out of scope (audited for provenance only, not re-scored): the L2 environment-content probes (`spatialenv.py`, `egodom.py`, `gradprobe.py`) — they are P5/L2 instruments, not P2.

## 2. What counts as a leak — the taxonomy, with the operational test for each

A probe is **leak-free for a conclusion** only if the conclusion could not have been produced by the leak alone. The six classes below are the brief's, made operational:

| class | definition | operational test (applied to the probe's code, file:line quoted in RESULT) |
|---|---|---|
| **(i) target-in-input** | any input at probe time is a deterministic (or r ≥ 0.95) function of the target, or contains the realised future the target is built from — endpoint concatenation `[z_t, z_{t+1}]`, GT future actions fed to a rollout whose output is then scored against the same future, the `v`/`steer`/`ω` channels (realised motion, E-DEC-57) when the target is the realised motion | list the input tensors; for each, ask "is it computable from the target, or from frames the target is built from?" — quantified where data allows (r between the input and the target, model-free) |
| **(ii) input-derived target** | the target is a function of an input (Δz = z_{t+k} − z_t scored from z_t; Δv scored from v_t; a label derived from a probe input) | write the target as a function; if an input appears in it, the class applies and the **arithmetic component must be measured with an endpoint-shuffled control** |
| **(iii) energy normalisation by a target the model controls** | a ratio whose denominator is the target's own energy or a second measurement of the model that moves with the treatment (the SMAS finding on `scene_spread`; EM normalised by target energy) | name the denominator; if it is a model output or the target's energy, the class applies |
| **(iv) selection on the scored split** | any hyper-parameter (λ, PCA basis, bandwidth, standardisation, clip selection, band) chosen using rows that are then scored, or an inner split not shaped like the outer one | quote the selection code; the inner split must be clip-disjoint like the outer |
| **(v) n ≪ d** | effective n (clips for a clip-level control) below the feature dimension after any PCA | n, d printed per probe; d ≥ n_eff/4 is flagged |
| **(vi) none found** | none of (i)–(v) applies to the *conclusion drawn* | default only after (i)–(v) are each written out |

**A seventh column, from `CLAUDE.md`'s estimator rule, is recorded beside the six:** **(E) estimator** — an interval or t computed from overlapping or non-independent held-out scores (the `overlapping_holdout_se` family) is recorded as an estimator defect, because it manufactures *significance* rather than a sign.

**Direction matters and is recorded for every leak:** a leak can favour the *alternative* (make a probe read a positive that is not there) or favour the *null* (make it read an absence that is not there). The same leak can be harmless for one conclusion and fatal for another.

## 3. The survival rule (committed)

For a conclusion **C** drawn from a probe with leak **L**:

1. If **L favours the alternative** and the probe still read **null** → **C (the null) SURVIVES**, with the note that the null is if anything *understated*. Magnitude/attribution may still need re-quoting.
2. If **L favours the alternative** and the probe read **positive** → **C is RE-OPENED**: the positive may be manufactured; a control that removes L must be run (0-GPU if the banked raw allows, else costed).
3. If **L favours the null** and the probe read **null** → **C is RE-OPENED**: the absence may be manufactured.
4. If **L favours the null** and the probe read **positive** → **C SURVIVES** (the positive survived a leak working against it).
5. Class **(iii)** does not manufacture a sign but can hide a magnitude: the conclusion's *sign* survives; its *magnitude* and any *ratio bar* are re-quoted on a fixed denominator.
6. Class **(iv)/(v)** with a **null** read → RE-OPENED as *underpowered/manufactured*, never as *absence*; with a positive read → the positive is inadmissible until refit on the fit split.
7. Column **(E)** with any read → the *sign* may survive, the quoted **t / interval is withdrawn** until recomputed with independent held-out units.

A P2 *branch* is re-opened only if **every** probe supporting its elimination is re-opened; a branch supported by two independent leak-free reads is not re-opened by one leaky read.

## 4. The controls checklist applied to every probe

Present / absent is recorded for: constant-only (must read the no-information value exactly); raw-input (pixel) floor; shuffled-**action** control; shuffled-**target** (time-shuffled within clip); matched **null** through the identical code path; C0 identity (determinism); positive control that must read a known value; `n` and `d` printed; λ selected on the fit split **split the same way as the outer test** (`rangeprobe_rff.py:81-93`). A missing control is a work item, not a verdict.

## 5. Re-scores committed in advance (0-GPU, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1` asserted)

All use local banked checkpoints (md5-stamped in `raw/`) and the local copies of the banked corpora (Thor's first-24 `physicalai-val-w120-256x640cyl` for actdiv, md5-manifested; `physicalai-val130-heldout` + `val130_agents.jsonl` for latentmotion/envpred), the dev-box mirror stack `C:\Users\Admin\tanitad-wt\stack` stamped via `tanitad.__file__` (MM-C12). CPU-vs-CUDA numerics are expected to reproduce the banked incumbent to ≤ 1 % relative; a reproduction outside 5 % voids the re-score (C160) rather than being explained.

**[PRE-RUN AMENDMENT — checkpoint availability, found while staging assets.]** The dev box holds `postrain30k` (md5 `a58585883c27…`), `k8clip05p30k` (`2d744d6d2faa…`), `k60clip05p30k` (`7e3c776d19a6…`), `rdw8p30k` (`6e382ebe721b…`), `postrain30k_freeze` (`5f5e5c92cd8f…`) and `k4_30k` (`c914e6a0a4b0…`). **It does NOT hold `o11p30k/ckpt.pt` or `splitp30k/ckpt.pt`** (the 2026-08-25 scratchpad carries only their `config.json`/`train_log.jsonl`; `tanitad-caches`, `Temp/claude`, `k8-pull`, `mm-e19-pull` searched — absence at five locations, still one box). Both are Thor-only. ⇒ **R2 is INFEASIBLE in this package** and is costed instead; **R4 excludes the regression arm** and runs on the arms that are local.

**[PRE-RUN AMENDMENT — estimator, found while reading the L3 code.]** `envpred.py:211-217` (`loeo`) calls `v7tiny_probe.probe(Xf + [X[i]], …)`, and `probe()` (`v7tiny_probe.py:180-186`) scores a pooled cross-clip R² over the **second half** of the clip list (`fit = range(half)`, `te = range(half, n)`), so each "per-clip" score is a 12-clip pool sharing ~11 clips with its neighbours. The 24 scores are not independent; the banked L3/E-DEC-28b `t` values are computed as if they were (`envpred.py:238-241`). ⇒ **R4 reports both the banked-form estimator (so the banked numbers reproduce) and a corrected clip-level one (each clip scored exactly once on a fit that excludes it, within-clip r, paired clip-level t, clip bootstrap), and the banked t's are re-quoted only from the corrected form.**

**[PRE-RUN AMENDMENT — speed-channel scale, found while reading the actdiv family before R1 was launched.]** `actdiv_thor.py:47`, `code_actdiv_thor.py:47`, `actdiv_local.py:56` and `condpath_thor.py:55` hard-code `SPEED_SCALE = 30.0` and feed the predictor `v/30`; the trainer's `_lift3` (`train_v6_staged.py:3634-3635`) and the model's own planner path (`v6.py:5340`) use `flagship_v15.SPEED_SCALE = 10.0`, i.e. every v7-tiny arm trained on `v/10`. `actchan.py:73,106`, `actionshuf.py:82,140`, `envpred.py:99,183` and `egofuture.py:66,98` use the trained scale. ⇒ every banked actdiv/condpath number (MM-E10, E11, E13, E17, E19, E19-K8, SMAS-1) was taken with the speed channel at one third of its trained scale. **R1 therefore also runs the banked form at the trained scale (`v/10`)**, committed outcomes: **A** — the incumbent's h1 ratio at `v/10` is within 25 % of the `v/30` value ⇒ the scale error is inert for the ratio and the banked numbers stand with a footnote; **B** — it differs by more than 25 % ⇒ every actdiv-family number is re-quoted at the trained scale, and the "action-deaf" verdict is re-checked against the prereg's own 0.06 bar at that scale.

| re-score | what is removed / added | outcome A | outcome B |
|---|---|---|---|
| **R1 — actdiv with the speed channel varied** (`postrain30k`, `k8clip05p30k`, `k60clip05p30k`, `rdw8p30k`, `postrain30k_freeze`) | the banked instrument rolls `[steer, accel]` across windows and **holds each window's own `v`** (`actdiv_local.py:176-180`). Add variants: (B) roll the full `[steer, accel, v]` tuple; (C) roll `v` only. Plus GS-8's anchored read: ‖ẑ(s·a) − ẑ(0)‖ for s ∈ {0, 0.5, 1, 1.5, 2} (monotonicity) | **action_spread(B) ≤ 2× action_spread(banked)** ⇒ the "0.4–0.6 %" verdict is about the whole action tuple; P2's magnitude stands | **action_spread(B) > 2× banked** ⇒ the banked ratio measured *steer/accel at fixed speed*; P2's *magnitude* is re-quoted as the full-tuple number, its *sign* (still ≪ scene) re-checked; if the full-tuple ratio ≥ 0.06 (the prereg's own HORIZON-WORKS bar) the "action-deaf" verdict itself is RE-OPENED |
| **R2 — actdiv on `o11p30k`** (gate queue #1, GS-10) | **INFEASIBLE HERE (amendment above).** Costed: pull `ckpt.pt` (~133 MB) from Thor `/home/nvidia/v7tiny/o11p30k/`, then `tools/p2_leak_rescore.py actdiv --ckpt <path> --arm o11p30k` ≈ 2 min CPU / 4 min GPU; criterion committed: | h1 ratio **inside** 0.00408–0.00595, or above only with `nrmse_1 > 1.10` ⇒ the breakout never reached the predictor (or bought sensitivity degenerately); queue #4 should not run | h1 ratio **≥ 3× incumbent (≥ 0.018)** with `nrmse_1 ≤ 1.10` ⇒ P2 has a positive lever; #4 becomes the highest-value arm — **but** the class-(i) construction of O11 (RESULT §3) still requires the same-clip-negative control before "dynamical" can be claimed |
| **R3 — drift's arithmetic component** (E-DEC-59 rig, `latentmotion.py` band 0:8, K=4, `postrain30k`, `rdw8p30k`) | add an **endpoint-shuffled target** control: Δz′ = z_{π(t)} − z_t with π a within-clip permutation of the *future endpoint only*, scored by the identical RFF+ridge path (`rangeprobe_rff.rff_fold`, `panel_kfold.kfold_clip_scores`) | control r **≤ 20 %** of the true drift r ⇒ "drift" is mostly genuine self-predictability; P3's metric stands | control r **≥ 50 %** of the true drift r ⇒ drift is majority class-(ii) arithmetic (mean reversion of a change score on its own start point); P3's *metric* is re-opened (not P2's ego-marginal null, which is a difference on top of it); 20–50 % ⇒ "partial", reported as such |
| **R4 — L3 with the GT-future-action leak removed** (`envpred.py` rig; held-out lead-matched, K ∈ {1,3,6}; arms `rdw8p30k` [the banked E-DEC-28b arm], `postrain30k` and `postrain30k_freeze` [the gate-queue L3 read never taken]; `splitp30k` excluded per the amendment) | the banked rollout feeds the **true future actions** at every step (`envpred.py:184-190`); add a **hold-action** rollout (a_next = the last observed action, the T1 convention `t1_eval.py:736-737`) and a **shuffled-action** rollout | paired `zhat(GT) − zhat(hold)` on `n_agents` **\|t\| < 2.9** (corrected estimator) at every K ⇒ the L3 numbers did not carry the leak; the L3 verdicts survive as banked | **\|t\| ≥ 2.9** ⇒ the banked L3 numbers carried realised-future information through the action channel; every L3 number is re-quoted from the hold-action form, and the regression-arm premise is re-read |
| **R5 — data-side overlaps (model-free)** | quantify class-(i) overlaps on the corpus: r(closed-form ω, measured ω) [E-DEC-57 check]; r(accel, Δv over 1 and 4 ticks) [E-DEC-49 echo]; the share of the 4-tick yaw change carried by the first tick used inside `latentmotion.py:158` | reported as magnitudes; no outcome table — these are the quantities the verdict text cites |

**Anti-gates.** R1–R4 are one checkpoint each, one seed; R1 and R4 carry a clip-level bootstrap interval computed in the same script (the actdiv instrument has no banked interval — backlog **L-13**); CPU numbers are compared to CUDA-banked numbers only after the incumbent reproduction check. The h ≥ 2 heads are reported but **withdrawn** as evidence per MM-E14 (untrained heads).

## 6. What this audit cannot do, stated now

* It cannot contact Thor: `o11p30k` (final, killed at 7,600) and `splitp30k` are Thor-only; R2 is costed, not run; the regression arm's L3 status is re-read from the banked JSONs only.
* It cannot re-run the T1 read (needs the Thor dumps) — the hold-action instrument is audited for structure only.
* It cannot add the paired bootstrap to the *banked* MM-E19 factors — only to its own re-scores.
* It does not decide anything: register rows are PROPOSED for the Master Mind; nothing in the gate is closed by this package (PI closure rule, `V7_LAUNCH_GATE.md:12-15`).
