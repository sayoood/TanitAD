# PREREG P5C-NEXT: what follows P5-C's GUARD-FAIL, ranked by the measured size of the term each arm attacks

Written 2026-11-20 (~07:30Z), P5C-NEXT agent, BEFORE any full per-proposal table or any next-arm number exists. Status: **DRAFT for
the MM**; it is frozen by sha256 when the MM lands it. ⛔ Nothing is launched by this document. Every GPU option is a PI decision (§5).

Package: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/` (below `SG/p5c_next/`). P5-C read: `E:/Projects/TanitAD-artifacts/p5c_read/out/`.

## 1 · Measured basis (each number: evidence class, artifact, tier, estimator)

### 1.1 Gap decomposition (MEASURED, `gap_decomp_p5c.json` / `.out`, tool `gap_decomp_p5c.py`)
Same exact method as D-P6-GAP-1 (`decomp_p6.py` = D-REFE-GAP-1b `ep_lowtail.py`): PDMS = NC·DAC·(5EP+5TTC+2C)/12, asserted per token
against the harness CSV. **Order A** (the banked order): gate on the reference's weight, then EP/TTC/C on the arm's gate. **Order B**:
gate on the arm's weight, sub-terms on the reference's gate. Another order gives another split, so both are printed.
**Controls, both PASS:** u400 reproduces 6.30 = 4.15 / 0.86 / 1.31 / −0.01 with 529 zeros, and P6 reproduces 3.35 = 3.04 / −1.20 / 1.53 / −0.02
with 384 zeros. P6's order B also reproduces the banked 1.35 / 2.05 / −0.03.
Tier T1 (NAVSIM v1.1 ego pseudo-sim, LQR tracker), full navtest, 12,146 tokens, one training seed, point decomposition, no interval.

| gap (PDMS ×100) | total | gate A (DAC=0 / NC=0 / NC=0.5) | EP A | TTC A | C A | gate B | EP B | TTC B | zeros |
|---|---|---|---|---|---|---|---|---|---|
| human − C1 | **2.00** | **2.84** (1.69 on 216 tok / 1.11 on 141 / 0.04 on 9) | **−2.24** | **1.41** | −0.01 | 1.21 | −1.15 | 1.95 | 357 |
| human − C0 | 3.17 | 2.95 | −1.24 | 1.47 | −0.02 | 1.31 | −0.11 | 1.99 | 372 |
| human − P7 (base) | 3.48 | 3.09 | −1.07 | 1.47 | −0.02 | 1.41 | +0.12 | 1.96 | 393 |
| C1 − C0 (ref C1) | 1.17 | 0.62 | 0.58 | −0.03 | −0.00 | −0.40 | 1.45 | 0.13 | — |

* **The whole remaining 2.00 is safety, not progress.** The zero tail plus TTC is 4.25 (order A) or 3.16 (order B). EP is negative in both
  orders: on its 11,789 gated tokens C1's EP is 92.44 against the human's 86.91 on the same tokens.
* TTC = 0 on 4.67 % of tokens (human 0 %). DAC = 0 on 1.78 %, NC = 0 on 1.16 %.
* **C1's gain over C0 is EP in both orders** (+0.58 / +1.45). The zero set churns: C1 357 and C0 372 zeros share only 225, so 279 of the
  504-token union flip between two 2k-step fine-tunes of one base. That is the signature of near-boundary picks, i.e. a selection problem.

### 1.2 Speed bias and roughness: labels or execution? (MEASURED)
**Labels** (`label_decomp_p5c.json`, rank-0 rows of the C1 table `p5r-rj5`, sha256 `3b198314…` verified, navtrain, 92,896 rows, no tracker).
Control PASS: label/bank reproduces the banked TB-6 rank-0 values 1.323 (acc) and 1.705 (heading rate).
The label path is bank `traj` (C0's label) → RJ5 rollout (CONTENT) → P5-R inversion (INVERSION):

| | speed, m/s (mean diff) | max\|acc\| ratio of medians | max\|heading rate\| ratio | max\|jerk\| ratio |
|---|---|---|---|---|
| content, RJ5 − bank | **+0.296** (54.0 % of rows faster, 0.2 % slower; +1.03 m at 4 s) | 1.134 | 1.041 | 1.135 |
| inversion, P5-R − RJ5 | +0.019 | 1.166 | **1.637** | 1.233 |
| total, label − bank | +0.315 | 1.323 | 1.705 | 1.400 |

**Student** (`roughness_p5c.json`, navtest picks, 12,146 tokens / 136 logs; log-cluster bootstrap with 2,000 draws, LOGS only):
* Speed, C1 − C0: plan **+0.24 [+0.19, +0.30] m/s**, tracked execution **+0.22 [+0.17, +0.27]**. The plan and its own execution differ by
  only +0.02, so the tracker adds nothing. The rank-0 label excess is +0.315 on ~69 % of draws (ESTIMATED share,
  PREREG_P5C §4.3), which predicts ≈ +0.22. ⇒ **The speed bias is IN THE LABEL CONTENT** (RJ5 is the faster driver), learned faithfully.
  It is the same thing that buys C1's EP. The excess over the human is C0's +0.27 (plan, bank teacher) plus RJ5's +0.24.
* Roughness, pick20 at 0.2 s (TB-6's own statistic): C1/C0 acc **1.139 [1.122, 1.156]**, heading rate **1.157 [1.139, 1.173]**. This is
  BELOW the label's 1.323 / 1.705. ⇒ the student ATTENUATES the label roughness and does not amplify it.
* **The dominant roughness belongs to the student's generation, in every arm, not to the label lever.** At 0.5 s against the human:
  C0 acc 1.28×, heading rate 2.63×, jerk 1.64×; C1 1.49× / 2.96× / 2.13×. Pass1 transparent comfort fails on 62.2 % (C0), 67.6 % (P7),
  77.4 % (C1), and 10.3 % (human).
* ⭐ **It lives on the 0.2 s grid that NAVSIM's tracker never reads** (smoke, n = 50, `propscore/smoke50_C1_s*.jsonl`, MEASURED,
  descriptive). The same C1 pick passes transparent comfort on **5/50** as pick20 (dt 0.2, the card's transparent arm) and on **48/50**
  as its 8 poses (dt 0.5). `to_navsim` reads 12 of the 20 poses (`p5c_read_prep.py:67`).
* Selection does not look at smoothness. The pick sits at the median roughness of its 64 proposals (rank 42–52 %), and the smoothest
  proposal has a median max\|acc\| of 0.70 against the pick's 1.72 (0.5 s).

### 1.3 Selection headroom (MEASURED smoke, n = 50: 25 C1-zeroed + 25 non-zeroed tokens, 38 logs, seed 20261120; descriptive, NOT decision-grade)
Instrument: `p5c_prop_score.py` scores all 64 proposals tracked and transparent in one NAVSIM call per token, and `p5c_prop_read.py` reads
them. **All controls PASS** (`propscore/smoke50_C1_read.json`):
* CTRL-T: the tracked pick equals the harness CSV, max \|d\| 1e-12 (0/50 over 1e-9).
* CTRL-P: pick20 transparent equals pass1, max \|d\| 0.0.
* CTRL-S: navsim_v1 argmax over the logits equals the pick on 50/50.
* RED arm (proposal axis rolled): CTRL-T **FAILS** on 5/6, as required (`propscore/red6_C1_read.json`).

Results:
* Zeroed tokens: **25/25 have ≥ 1 proposal with PDMS > 0** (selection-bound; u400's C3 was 86.0 % [80.2, 91.7]). Their oracle is 98.1.
  13 were zeroed by DAC and 12 by NC.
* Non-zeroed TTC = 0 tokens: 3/3 have a TTC = 1 safe proposal.
* Mix (full-navtest weights): pick 88.03, oracle of 64 **99.14**, random 70.79.
* Fixed safety-power rules, untuned: they recover zeroed tokens (a = 4: +15.9 on that stratum) but lose 0.7 to 1.0 on non-zeroed tokens.
  The mix is −0.2, n = 50. ⇒ the full table is needed, and the rule must be chosen on held-out logs.
* Cost: median **1.05 s/token**. A full arm is ≈ 3.6 CPU-h, **≈ 1.8 h wall at 2 processes** (ESTIMATED from the smoke), and ~75 MB of rows.

## 2 · Candidates, ranked by the measured term they attack (C1 vs human; A / B order)

| rank | arm | attacks | term size | GPU | why this rank |
|---|---|---|---|---|---|
| **1** | **S-RULE**: a selection rule over the existing 6 heads, tuned on TUNE logs, read on EVAL logs | gate + TTC | 2.84 + 1.41 (A); 1.21 + 1.95 (B) | **0** (CPU ~2 h/arm) | the largest term. Smoke: 25/25 zeroed tokens are selection-bound and the oracle is 99.14. Cheapest. |
| 2 | **SCORER-FT**: scorer-only fine-tune of C1's heads on on-policy C1 proposals with harness labels | gate + TTC | same | yes, PI | only if S-RULE cannot reach the headroom, i.e. the heads lack the information (D-REFE-MARGIN-HEADS-1, D-REFE-T2-FAN-1) |
| 3 | **TEACHER-N7**: distil the privileged RJ5 teacher (96.50 PDMS, DAC 99.98; D-REFE-TF-N7-1) | the generation-bound zero tail | unknown until the full table (smoke: 0/25 generation-bound) | yes, PI | ranks by the generation-bound share, which the full table measures |
| 4 | **GUARD-REPAIR**: (a) 0 GPU: plan20 := re-sampled from the 8-pose spline; (b) GPU: C1 + jerk/comfort term | the TM+C qualifier (transparent C −15.2 ⇒ −2.5 transparent PDMS) | **0 of the tracked 2.00** (tracked C 99.98) | (a) 0 / (b) yes | it repairs the guard, not the goal. (a) changes what the frozen transparent arm reads, so an MM/PI ruling comes first |
| 5 | speed-matched labels | G2 human-distance (speed_bias +0.22) | **negative for the goal** (EP is the term C1 leads) | yes | ⛔ not recommended for 94.55. Only if the PI rules G2 human-closeness binding over PDMS |

## 3 · Arm S-RULE: pre-registration (0 GPU; the first arm to run)

**Inputs.** Full per-proposal tables (`p5c_prop_score.py`, every token) for **C1** (primary), **C0** and **P7 base** (replicates).
Each table must pass CTRL-T, CTRL-P and CTRL-S on all 12,146 tokens, and the RED arm must fail. Otherwise INVALID with no verdict.

**Split (fixed now).** TUNE = the navtest logs whose `sha1(log_name)` hex digest ends in an even hex digit; EVAL = the rest.
Nothing is fitted on EVAL.

**Rule family (fixed now, 8 rules).** `agg(a, w) = (p_NC·p_DAC)^a · (5 p_EP + w p_TTC + 2 p_C)/(7+w)` over the sigmoid logits, with
a ∈ {1, 2, 4, 8} and w ∈ {5, 10}. (a = 1, w = 5) is the current navsim_v1 pick (control: it reproduces the CSV mean exactly).

**Selection.** On TUNE, take the rule with the highest mean tracked PDMS. Ties go to the smaller a, then the smaller w.

**Read.** On EVAL: Δ = PDMS(rule pick) − PDMS(navsim_v1 pick), tracked, ×100.
* Estimator: paired log-cluster bootstrap over the EVAL logs, 10,000 draws, seed 20260927, 95 % percentile.
* It answers *another draw of logs*.
* Inference is deterministic (seam re-run bit-identical, `p5c_read/out/seam_determinism_c1.json`).
* Training variance: the rule is training-free but checkpoint-specific. The replicate is therefore **across checkpoints** (C0, P7) and
  not across seeds. A seed replicate needs GPU (§5 B).

**Bar (committed): ADOPT iff all three hold**
1. CI-lo(Δ) > 0 on C1 EVAL;
2. no tracked sub-score (NC, DAC, TTC, EP, C) of rule − pick is separated adverse (CI-hi < 0) on C1 EVAL;
3. the same frozen rule gives Δ point > 0 on the EVAL logs of **both** C0 and P7.

**Both outcomes, committed now:**
* **ADOPT.** The rule becomes the selection rule for the next read. The C1 + rule level is quoted from EVAL only, with its CI, against
  94.55. The next lever is the generation-bound remainder (rank 3) and any TTC residue. A full harness seam of the rule's picks is
  confirmation only, because CTRL-T already makes the table harness-identical.
* **NOT ADOPTED.** The six heads do not carry the margin that the selection-bound headroom needs, consistent with
  D-REFE-MARGIN-HEADS-1. The next lever is SCORER-FT (rank 2, PI card §5 C), and its ceiling is the table's oracle.

**Four families, on the rule's picks, each with the same estimator and the same EVAL logs:**
* LONGITUDINAL (target speed, headway / TTC, progress) and LATERAL (heading, curvature, yaw rate, cross-track):
  `PK/eval/instruments/families_run.py` on a transp_score pass of the new picks (CPU, ~35 min at 2 processes, ESTIMATED from pass1's 1,900 s per shard).
* TACTICAL: command adherence (`tactical.json` form) plus **selection quality, now COMPUTED from the table**. It read NOT COMPUTED in P5-C.
* STRATEGIC: in full (the PI unpaused it on 11-16). `PK/eval/instruments/strategic_family.py` S_DEC + S_GOAL, reported with no bar.
* Tier: T1 executed (tracked) for the bar. Transparent is reported beside it, labelled *no tracker in this number*.

## 4 · Runner (built and smoked; NOT launched for the full set)
```
cd E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next
NAV="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
REFE="F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
for s in 0 1; do PYTHONPATH="$NAV;$REFE" TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nohup C:/Users/Admin/navsim-crun/venv/Scripts/python.exe p5c_prop_score.py --arm C1 \
  --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --out propscore/full_C1_s${s}of2.jsonl \
  --shard $s/2 > propscore/full_C1_s${s}of2.log 2>&1 < /dev/null & done
# then C0 (stage2_p5c_c0.proposals.npz, --arm C0) and base (stage2_p7_full.proposals.npz, --arm base), one arm at a time (<= 2 processes)
C:/Users/Admin/venvs/tanitad/Scripts/python.exe p5c_prop_read.py --rows "propscore/full_C1_s*of2.jsonl" --arm C1 \
  --csv E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p5c_c1/refe_stage2_p5c_c1.csv \
  --pass1 "E:/Projects/TanitAD-artifacts/p5c_read/out/pass1/rows_s*of2.jsonl" \
  --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --out propscore/full_C1_read.json --boot 10000
```
* The runner is resumable. It gates on commit free ≥ 5.5 GB at start and every 200 tokens.
* The TUNE/EVAL split, the 8-rule choice, the EVAL bootstrap and the ADOPT bar are implemented in `s_rule_read.py`. Its plumbing
  smoke on the 50 smoke rows passes the rule-(1,5)-reproduces-pick control (`propscore/smoke50_srule_plumbing.json`, labelled SMOKE,
  no verdict: that sample is 50 % zeroed by design). After the three tables:
  `python s_rule_read.py --primary "C1=propscore/full_C1_s*of2.jsonl=<c1 proposals.npz>" --replicate "C0=propscore/full_C0_s*of2.jsonl=<c0 npz>" "base=propscore/full_base_s*of2.jsonl=<p7 npz>" --out propscore/S_RULE_READ.json`
* ⚠️ The runner imports `transp_score.py` and `refe/` from the exported tip `F:/TanitAD_p5cread_tip` (EXPORT_SHA `751f3ae7`).
  `transp_score.py` there is byte-identical to the mirror tip `bcf267ce`.

## 5 · DECISION CARD (PI): GPU after P5-C. The L40 is busy with the field chain C → F → Cs → Fs for ~4 days.

| option | what | cost | the question it answers |
|---|---|---|---|
| **A (recommended now)** | **No GPU.** Run S-RULE: 3 tables, CPU ≈ 5.4 h wall on the dev box at ≤ 2 processes | 0 L40-h | does selection on the existing heads close the safety term (2.84 + 1.41)? |
| B | C1 seed-1 replicate (C1s1) | ~27 L40-h (ESTIMATED, PREREG_P5C §1 line 48) | is C1 − C0 = +1.17 above training noise? `H-ESTIM-SEED-1`; the bootstrap cannot answer it |
| C | SCORER-FT on C1 (heads only, on-policy C1 proposals, harness labels; needs a navtrain per-proposal table via the DB route) | ≤ 27 L40-h (ESTIMATED upper bound: a head-only fine-tune of the same model) + navtrain labelling CPU (UNVERIFIED) | can the heads learn the margin S-RULE shows is missing? |
| D | C1 + jerk/comfort term (and its replicate) | ~27 + 27 L40-h (ESTIMATED) | repairs the TM+C guard, not the tracked 2.00 |

**Recommendation:** A now. Queue B behind the field chain, because any lever claim on C1 needs it. C only if S-RULE reads NOT ADOPTED.
D only after an MM/PI ruling on whether the transparent arm should read pick20 or the 8 executed poses (§1.2 ⭐). Option 4(a) of §2 is
0 GPU and may make D unnecessary.
