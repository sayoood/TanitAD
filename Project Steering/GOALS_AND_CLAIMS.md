# GOALS & CLAIMS — the live register

`Created 2026-08-22 as part of the programme redesign (TANITAD_PROGRAMME.md §4).
⛔ RULE: any session that asserts, supports, or refutes a claim UPDATES this
file IN THE SAME TURN. Status ∈ OPEN / SUPPORTED / REFUTED / RETIRED. Every row
links evidence (artifact path or registry anchor). This file is the first thing
a fresh context reads after the constitution.`

## Programme goals (stable)

| id | goal | status |
|---|---|---|
| G1 | Sub-300M hierarchical 4-brain latent world model that drives (T1, four families) | OPEN |
| G2 | The 8 products of TANITAD_PROGRAMME.md §1 shipped production-ready | OPEN |
| G3 | Beat published SOTA on community benchmarks (NavSim …) via TanitEval | OPEN |

## Live claims & hypotheses (the working frontier)

| id | claim / hypothesis | status | evidence |
|---|---|---|---|
| H-RANK-1 | v6's trunk representation is dimensionally collapsed (encoder tokens ~1 effective direction) | **SUPPORTED** | rank probe 2026-08-22, `…/2026-08-19-simwam-analysis/raw/v6F_v7tiny_rank_probe.txt` (99.7 % top-1 energy) |
| H-RANK-2 | Collapse is a TRAINING dynamic, not initialisation (rank rises to ~16k then falls) | **SUPPORTED** | rank-vs-step: 6.43→8.96@16k→5.55@20k (σ² basis, ckpt sweep) |
| H-RANK-3 | SIGReg WEIGHT is the lever | **REFUTED** | 1000× λ sweep flat: partic 2.94→3.34 |
| H-RANK-4 | Term COUNT is a lever (LeWM two-term > six-term) | **SUPPORTED** | partic 2.94→4.43 (→5.00 @6k steps); still below DINOv3 8.56 |
| H-RANK-5 | Estimator CONDITIONING is a lever (rows n via bank, dims d via Sub-JEPA) | **REFUTED (both halves)** | n-raising: O(n²·M) OOM on 4060 AND Thor (bank64c died at first Epps-Pulley tensor). d-lowering: sub32c/sub64c VAL participation 3.33/3.56 ≈ lewm 3.50 (`~/v7tiny/val_rank_3way.json`) — the gate's train-pooled 5.5 was O4-sampling composition, not representation quality |
| H-RANK-9 | The gate's pooled participation reads the O4-BIASED train stream and overstates val-side participation (~5.5 vs ~3.4 same model) | **SUPPORTED** | val_rank_3way.json vs sub32c/sub64c stage_gate.json — gate numbers are cross-arm comparable but NOT a val-side measure |
| H-RANK-6 | Predictor/encoder capacity ratio (7.79×) drives collapse | **REFUTED** | `cap` (enc 256×6 / pred 128×2): VAL partic 3.55 ≈ lewm 3.50 |
| H-RANK-7 | k-step rollout (o5-k 6) rewards a static latent; k=1 helps | **SUPPORTED (modest)** | `k1`: VAL partic **4.05** vs lewm 3.50 (+16 %) — first val-side mover on the parity corpus |
| H-RANK-10 | Two-term+k1 keeps gaining rank past 6k steps (all-six peaked at 16k then collapsed — never measured for two-term) | **OPEN — champ30k running** | Thor `~/v7tiny/champ30k`, 30k steps ≈ 4 h; trajectory free from spectrum_pooled log rows |
| H-RANK-8 | 3-frame channel stacking makes Δz noise-like; n_stack=1 helps | OPEN (unrun) | lag-1 Δz autocorr −0.075 |
| H-INIT-1 | Residual-init defect (O(1) delta vs 1e-3 latent motion) broke every residual predictor | **SUPPORTED & FIXED** | paired G2: regress 45,712× → fixed 1.040× worse-than-hold, p<1e-4; fix in 3 modules/6 sites/5 models |
| H-GATE-1 | The O6 rank gate could never rule (spectrum n=24 vs ceiling 1024) | **SUPPORTED & FIXED** | preflight + pooled-spectrum-to-gate fix; Thor verifies "CAN rule, ceiling 1031" |
| H-GATE-2 | effective_rank(σ) is a valid collapse statistic | **REFUTED** | C132: 55 % top-1 energy PASSES floor 64 (130.91) while cleaner arm FAILS; gate now decides on participation (σ²), floor 8.56 = frozen DINOv3 |
| H-REP-1 | Rank is SUFFICIENT for a useful representation | **REFUTED** | C131: v1-era rank 24.93(σ)/7.59(σ²) with zero environment interpretation; passes rank, fails decodability. Pass = rank AND decodability |
| H-ENC-1 | Frozen encoders are REF-A's ceiling (D-A5) | **CHALLENGED** | frozen DINOv3 reads speed R² +0.147 vs our trained +0.0025; REF-A predictor carried the init defect; clean re-test = PREREG_E_ENC_3WAY arm C |
| H-ENC-2 | The 3-way (own ViT vs DINOv3-finetune vs DINOv3-frozen+4B) decides v7's encoder | OPEN (pre-registered) | `…/2026-08-19-simwam-analysis/PREREG_E_ENC_3WAY.md` — gated on collapse fix first |
| H-LBL-1 | REF-C v3 must train on the ALIGNED tactical/strategic vocabulary, not the ego-geometric subset | **SUPPORTED (PI ruling)** | HIERARCHY_VOCABULARY.md; label-agent session running the gap-closure |
| D-GATE-FLOOR | Revise pre-registered O6_RANK_FLOOR=64 (synthetic provenance) → participation ≥ 8.56 (real reference) + decodability co-criterion | **DECIDED by PI 2026-08-22** | C131/C132; `O6_PARTICIPATION_FLOOR` shipped |

## Standing rules with their origin (never re-litigate silently)

| rule | origin |
|---|---|
| Open-source research-licensed models/data are USABLE; augmented sets go PRIVATE on paid HF; HF Pro GPU/SSH is first-class compute | PI 2026-08-22 |
| Labels may use ego; inference is vision-only | PI 2026-08-03 |
| Goal input admissible; must not carry situation-classifier output | PI 2026-08-03 |
| Four metric families, never pooled; tier stamps on every number | PI 2026-08-02 / EVAL_DOCTRINE |
| ≥5 parallel streams; never idle; a report is not work | PI 2026-08-03 / 07-29 |
| Bank the primary or the deliverable is incomplete | PI 2026-08-18 |
| Thresholds calibrated on REAL references; statistic named, not just quantity | C131/C132, 2026-08-22 |
