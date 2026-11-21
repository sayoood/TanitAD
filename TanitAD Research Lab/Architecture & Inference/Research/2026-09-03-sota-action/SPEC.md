<title>SPEC - SOTA pass: prediction / action sensitivity (theme 3 of 4)</title>

# SPEC - E-ARCH-SOTA-ACT-1: what the field has MEASURED about action-conditioning collapse in latent world models, and which fix is cheapest for us

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-action/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU
**Written BEFORE any primary of this pass was opened.** Staged in the same turn it was written.
**Trigger.** `V7_LAUNCH_GATE` P1/P2 (the programme's actual problem); `LAB_BACKLOG` rows P-1 (LatentAlign arm), P-2 (action-source mixture), P-4 (EB-JEPA control), GS-1/GS-2/GS-3/GS-8/GS-10, MM-1; `LEDGER_A2_jepa.md` entries 2026-09-02-01/-02.

## 1. Our MEASURED state

| fact | value | source (class · tier) |
|---|---|---|
| h=1 action/scene spread ratio (scene held fixed, action varied by ROLL) | **0.00408 – 0.00595** across 30k arms | MM-E10 (`GOALS_AND_CLAIMS.md:1637`) · MEASURED · T0 |
| no intervention raised it; two lowered it | O1 control objective → **0.40×** (MM-E11); `o5_k` 8→60 → **0.50×** (0.002983 vs 0.005950), decomposed one-variable as action **0.83×**, scene **1.71×** (MM-E19-K8) | `:1785`; `2026-09-01-mm-e19-k8-attribution` · MEASURED · T0, no interval (L-13) |
| the action is NOT redundant given the latent | cross-fitted ridge, 220 clips, n=880 | MM-E12 (`:1647`) · MEASURED · T0 |
| sensitivity is ACQUIRED from ~0, 20–50× too slowly | 2k vs 30k arms | MM-E13 (`:1648`) · MEASURED · T0 |
| h≥2 heads were never trained (‖W2‖ 0.0262 vs ‖W1‖ 8.8386) — a one-horizon predictor | MM-E14 (`:1649`) · MEASURED · T0 |
| the action signal is healthy at the embedding and destroyed in the FiLM conditioning (attenuation 56× / 128×); the FiLM gain CONVERGED (deafness is an optimum of the objective) | MM-E17 / MM-E18 (`:1697`, `:1689`) · MEASURED · T0 |
| action census: 0/32 arms move nrmse > 0.1 % under action shuffle; action pathway 2–9 % of the latent pathway | §13.0c · MEASURED · T0 |
| `copy_detector` CLEAN, echo 0.0000 on all three T1 arms | §13.10 T1 block · MEASURED · **T1** |
| our action channel IS realised motion (r 0.9988) | E-DEC-57 · INHERITED from the readiness report (row not re-opened here) |
| our O5 target is teacher-forced: the target encoder sees the realised future | `train_v6_staged.py:48/:1765/:4214` · INHERITED from `2026-08-31-command-conditioning-and-l3` (source lines cited there) |
| `o11p30k` — the only arm that ever became action-sensitive (pick_acc 1.000 for 12 rows after 5,200 steps at the floor), four config diffs incl. scratch init, abandoned at 25 % | INHERITED — `LAB_BACKLOG` P-3 / GS-10 |
| Delta-JEPA (lib `2606.31232`): LDAD decodes the action from `Δz = z_{t+1} − z_t` (two ENCODER outputs); `Δz` beats `concat[z_t, z_{t+1}]` on all four envs (Table 2); λ=0 nearly collapses; best λ = 50 | INHERITED from `LEDGER_A2_jepa.md` — to be RE-READ from the banked PDF in this pass (the brief asks for the anchored objective, the training change and the ablation numbers) |
| LatentAlign (lib `2605.09701`, DriveFuture): sigmoid-annealed target grounded → self-predicted; ablation 32.1 → 34.6 EPDMS | INHERITED from KNOWLEDGE_BASE 2026-08-31 — re-read here |
| EB-JEPA (lib `2602.03604`): 97 % Two-Rooms planning success | INHERITED, abstract-only so far — read in full here |

## 2. Questions

- **Q-A1.** For each of: Delta-JEPA/LDAD, LatentAlign, EB-JEPA, and every 2026 primary found on action-conditioning collapse (cycle/inverse-dynamics consistency, action dropout / classifier-free action guidance, contrastive action, action-prediction heads, counterfactual consistency, advantage-style action channels) — what is the **mechanism**, the **published effect size on an ACTION-SENSITIVITY metric** (not only planning success), and the **cost** (extra forward passes, extra modules, extra labels)?
- **Q-A2.** Which of these fixes survive a FROZEN trunk (they act on the predictor or the target), and which are encoder-shaping (they need a trainable trunk)?
- **Q-A3.** Which survive an action channel that is REALISED MOTION (r 0.9988) — i.e. do not become tautological when the action is decodable from the scene change?
- **Q-A4.** Does any primary measure the action-sensitivity metric OVER TRAINING (our MM-E13: acquired from ~0, too slowly) and report when it appears?
- **Q-A5.** What do the field's numbers PREDICT for our 0.004–0.006 ratio — is it in the "collapsed" regime of their diagnostics (Delta-JEPA Fig. 6 on LeWM), and which single arm discriminates fastest?

## 3. Hypotheses — both outcomes committed

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-SOTA-A1** | ≥ 1 banked primary reports an action-sensitivity METRIC (displacement vs. zero-action, counterfactual divergence, action-consistency transfer, …) before/after its fix | adopt the metric FORMULATION into `actdiv` (GS-8) so our number is comparable, and take its effect size as the prior for the arm | only planning-success proxies exist ⇒ our `actdiv` is ahead of the field and no published prior exists for the effect size; the arm is pre-registered on our own bar |
| **H-SOTA-A2** | The published fixes partition into (i) encoder-shaping via inverse dynamics on the displacement (LDAD / IDM-regularisation), (ii) cycle-consistency THROUGH the predictor (inverse dynamics on the PREDICTED transition), (iii) target construction (grounded → self-predicted anneal; counterfactual/negative-action targets), (iv) an anchored / advantage-style action channel — and (ii)–(iv) survive a frozen trunk while (i) does not | the first v7-tiny arm is drawn from (ii)–(iv); (i) is recorded as inapplicable for refav1 and applicable only to a trainable v7 trunk | the partition fails (e.g. (ii) is shown to need a trainable encoder too) ⇒ every published fix requires a trainable trunk and the freeze decision becomes the P2 decision |
| **H-SOTA-A3** | At least one fix is shown to work when the action is (near-)decodable from the observation pair — i.e. it is not tautological on a realised-motion channel | that fix is the first arm on the realised-motion corpus; GS-2's tautology test is still run first | none is shown ⇒ P2(b) (a command channel) regains rank, and the comma2k19 geometry decision (readiness §G.3) is the blocker |
| **H-SOTA-A4** | A primary reports the sensitivity metric across training and shows late emergence or a curriculum dependence | MM-E13's "20–50× too slow" is field-corroborated and a schedule lever is pre-registered | no such trace exists ⇒ MM-E13 is a TanitAD-only observation |

## 4. What would change our plan

1. A frozen-trunk-compatible fix with a measured action-sensitivity gain ⇒ it precedes the O11 re-run in the GPU order (readiness §G.2), because its prior is PUBLISHED where O11's is a confounded anomaly.
2. Evidence that every fix is encoder-shaping ⇒ the v7f trunk-freeze decision is the P2 decision and must be taken as one.
3. A published action-sensitivity metric formulation ⇒ `actdiv` adopts it (GS-8) before any arm is judged, so the arm's read is field-comparable.

## 5. Method and admissibility

As theme 1: banked PDFs only (tag `sota-2026-09-03-action`), ≥ 3 primaries read in full (Delta-JEPA in full, LatentAlign §method+ablation, EB-JEPA in full, plus the 2026 action-collapse primaries), every number stamped, agreement/disagreement stated, cheapest discriminating arm with both outcomes committed. Every proposed arm carries the constant-only control and the shuffled-action control (constitution §6.2).

## 6. Falsifiers

- H-SOTA-A2's partition is refuted by a single primary whose predictor-side consistency loss is shown to fail with a frozen encoder.
- H-SOTA-A3 B is refuted by a primary whose action is a realised-motion quantity (e.g. ego displacement) and whose fix still raises a sensitivity metric.
