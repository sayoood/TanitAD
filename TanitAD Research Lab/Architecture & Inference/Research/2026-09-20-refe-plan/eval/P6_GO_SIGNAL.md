# P6 GO signal: the evidence the Master Mind needs before the PI starts a pod

> 2026-10-11. **Status: NO SIGNAL.**
>
> PI, verbatim: *"I did not start the pod. I wait until you give me signal and we have evidence and confidence to
> surpass the paper."*
>
> The PI's rulings on P6-T, P6-N and P6-D (`SPEC_P6_PAPER_FAITHFUL.md` §7) settle **how** P6 runs. They do not
> authorise a launch.

## 1 · Where the 6.30 points are (MEASURED, 0 GPU)

**Source.** The NAVSIM harness hooks of the full 12,146-token navtest for SFT-4 B u400 (88.25) and for the human
(94.55). Script and output: `raw/2026-10-11-ep-gap-decomposition/ep_lowtail.py` → `ep_lowtail.out`.

**Method.** An exact sequential decomposition: the NC×DAC gate first, then the weighted terms on u400's gate.

| term | PDMS | what it is |
|---|---|---|
| **zeroed tokens** (NC×DAC = 0) | **4.15** | DAC = 0 on 3.42 % of tokens, NC = 0 on 0.96 % |
| TTC | 1.31 | TTC = 0 on 4.37 % of tokens; the human has 0 % |
| EP | 0.86 | |
| comfort | −0.01 | |
| **human − u400** | **6.30** | |

### Against the paper's student profile (PUBLISHED, Table A13)

| | paper | u400 |
|---|---|---|
| NC | 98.80 | 98.94 |
| TTC | 95.74 | 95.63 |
| DAC | 99.25 | 96.58 |
| EP | 92.72 | 81.20 |

- NC and TTC are already equal to the paper's.
- **The paper reaches 94.55 by being faster than the human.** Its EP is 92.72; the human scores 86.96 in our harness.
- The human reaches 94.55 by never being zeroed.
- **Both routes require the zero tail to shrink.** Only the paper's route also requires EP above the human.

### EP detail

- **Shorter than the human.** On the 7,275 non-zeroed tokens with 0.5 ≤ EP < 0.95, u400 drives about 5 % shorter
  than the human (median progress ratio 0.946).
- **Slower than its own teacher at speed.** At v0 6–15 m/s, u400's EP is 78–84 against our teacher's 88–93.

### Correction to `P6_CONFIDENCE.md`

That file says "EP is the largest gap". This is true against the paper's sub-score profile (11.5 EP points). It is
**not** true of the PDMS decomposition against the bar: the largest term there is the zero tail, 4.15 of the 6.30.

## 2 · The signal has two steps, on one pod with one data download

- **S1: start the pod for P2.** This covers the data fetch, ~11 h of PDM relabelling and ~20 h of training
  (ESTIMATED), run by `refe/p2r_pod.sh` per `eval/P2R_POD_RUNBOOK.md`.
- **S2: P6 on the same pod.** It runs only if P2 clears the GO rule in `eval/PREREG_P2_AMENDMENT_1.md`. The Master
  Mind reports the P2 read and the PI confirms.

⚠️ **No dev-box experiment can show 94.55 before training.** P2 is the first MEASURED evidence for the training
levers. So S1 asks for about 1.5 pod-days, not 7.

## 3 · Conditions for S1 (all are required)

| # | condition | status |
|---|---|---|
| C1 | The P2 chain is built and mutation-tested | ✅ `refe/test_p2r_chain.py`: 89 checks pass and 17 mutations go RED |
| C2 | P2's GO rule is pre-registered before any data | ✅ `eval/PREREG_P2_AMENDMENT_1.md` |
| **C3** | **Selection headroom on u400's own proposals** (dev box, pre-registered below) | ⏳ not run |
| C4 | M-LAB labeller cost ⇒ pod vCPU (P6-C) | ⏳ dev box, ~15 min |
| C5 | The launch gate is landed with `refe_p2` and `refe_p6` profiles (P6-G) | ⏳ Master Mind |
| C6 | P6-Y (yaw loss) is ruled | ⏳ PI |

### C3: pre-registration, written before any u400 proposal is scored

**Question.** Is the zero tail a **selection** failure, which P2 and P6's scorer levers can fix, or a **generation**
failure, where no safe proposal exists to pick?

**Population.**
- All **529** u400 tokens with NC×DAC = 0 on the full navtest (MEASURED from the CSV; 19 more have NC = 0.5).
- Plus an equal number of non-zeroed tokens drawn uniformly at random, seed 20261011.
- ⚠️ These are navtest tokens. This is a **diagnostic**: no parameter, weight or rule is fitted on it, and no
  selection rule derived from it may be quoted as a result.

**Measurement.**
- u400 is re-run on these tokens.
- **Fidelity gate:** its pick must reproduce the banked plan in `sft4_B_u400_navtest_full_seam.npz` (max |Δ| ≤ 1e-3 m).
- All 64 proposals per token are scored by the same harness: NC, DAC, EP, TTC, C and PDMS.

**Reads.**
- **(a) rescuable share:** among zeroed tokens, the share with ≥ 1 proposal scoring NC = DAC = 1.
- **(b) PDMS of three picks,** re-weighted to the full-navtest mix: the actual pick; a random pick (100 uniform
  draws); and the oracle pick (max true PDMS).
- **(c) EP headroom:** on non-zeroed tokens, the EP of the best safe proposal against the pick.

**Outcomes, committed in advance.**

| outcome | when | consequence |
|---|---|---|
| **SCORER-BOUND** | rescuable ≥ 50 % of zeroed tokens **and** random < actual | P2/P6's levers act where the gap is; C3 is green |
| **GENERATION-BOUND** | rescuable < 50 % | P6 as designed is aimed at the wrong component. Generation-side levers go first: the map-supervised drivable-area head (training only), P5 Tc20s labels, data. S1 is not given for P6 as designed. |
| **INSTRUMENT FAILURE** | random ≥ actual, or the fidelity gate fails | Stop and diagnose. No verdict. |

**Why 50 %.**
- The zero tail costs 4.15 PDMS.
- At ≥ 50 % rescuable, a perfect scorer recovers ≳ 2 PDMS from zeros alone. That is the scale P6's scorer levers
  claim.
- Below 50 %, even a perfect scorer cannot close the zero term.

**Cost (ESTIMATED).**
- About 70 k harness scorings (≈ 1,058 tokens × 64) on the dev-box CPU, about 2–3 h. The full run did 12,146 in
  1,336 s.
- u400 inference on about 1,058 tokens on the RTX 4060.
