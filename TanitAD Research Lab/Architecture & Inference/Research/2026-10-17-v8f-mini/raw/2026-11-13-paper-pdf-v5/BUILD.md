# Build note: the TanitAD-4B paper, version 5 (2026-11-13)

Source: raw/2026-11-13-paper-pdf-v4/tanitad_4b_paper.tex (sha256 e51ef9c2..., asserted by the patch). Edits: patch_tex_v5.py, seven edits, each asserted to match exactly once. No number of v4 is changed.
Engine: Tectonic 0.17.0, `tectonic --keep-logs --outdir build tanitad_4b_paper.tex`. Result: 13 pages (v4: 9), 0 overfull boxes, 0 undefined references. The underfull boxes in the log sit in narrow table cells and in lines v5 did not touch.
Pages 1 and 6 to 13 were rendered to PNG with the Windows PDF API and read. The new table of the nine statements is Table 1 (page 7, full width); the headline table of v4 is now Table 2 (page 9, unchanged content); the roadmap is Table 3 and sets alone on page 13.

## What changed
Every measured value added is a T0 or T1-kin (kinematic rollout on the model's own actions) diagnostic on logged frames, never driving performance. STRATEGIC family: PAUSED (PI item 25). No new measurement is made by this build: every number is copied from the proof programme, folder `raw/2026-11-12-v8f-proof/`, landed up to commit dd74d83f.
* Abstract: one passage (nine theorems, 56 solver conditions, 54 Lean 4 theorems; the pooling theorem; the trained-state read; hierarchy against flat is ahead on selection-critical windows and not established on all windows).
* Introduction: one contribution item (the machine-checked theory).
* New Section 6, "A machine-checked theory of the hierarchy": Table 1 (statements T1 to T9, their check, their measured premise) and four subsections with Propositions 3 to 6 (composed bound, jump bound and criterion; risk split and non-dominance; dominance under a hit-or-miss criterion, conditional; pooling), the commit-rule break-even, and a short list for rules, class weights, the yardstick, imagination, data and weights.
* Results: new Subsection 8.1, "Testing the premises of the theory" (eight paragraphs: hierarchy against a flat control; a yardstick for the world model; rules by construction; a class weight is one number; what the camera interface carries; the world-model state, untrained and trained; weight space; data).
* Outlook: one paragraph (the three experiments training on the pod; the next read).
* Conclusion: one sentence.
* References: three (Nachum et al. 2018, arXiv 1810.01257; Nachum et al. 2019, arXiv 1909.10618; Janner et al. 2019, arXiv 1906.08253), cited as PUBLISHED.

## Sources of the new numbers (all under raw/2026-11-12-v8f-proof/ unless a path says otherwise)
| claim in the paper | file | values as banked | rounding in the paper |
|---|---|---|---|
| solver totals | V8_THEORY.md (summary bullet), fv/FV_RESULT_V1.json, FV_RESULT_V2.json, FV_SPEC_V4_GATE.md, FV_RESULT_V5.json | 56 conditions, 30 mutated twins refuted, Z3; groups J and P written after reads they explain | as written |
| Lean totals | lean/LEAN_RESULT_POOL.md, lean/LEAN_RESULT_HIER.md, lean/src/lake_build.log | 35 + 19 theorems, no `sorry`, 50 of 50 audited on the standard axioms, build rc 0 | 54 |
| T1 premise | wm/VERDICT_WMCEIL.md, wm/VERDICT_WMSTACK.md, V8_THEORY.md (T1, JUMPCURVE) | lag correlation 0.975 / 0.883; learned skill +0.001 against ridge +0.022 at 0.2 s; -0.15 / -0.17 against +0.12 / +0.13 at 2 s; stack +0.040, +0.06, +0.15, +0.17; direct 2 s error 0.416 [0.389, 0.449] / 0.433 of ten one-step errors; ratios 0.74, 0.57, 0.52, 0.42 | 0.42 / 0.43 in Table 1 |
| hierarchy against flat, all windows | V8_THEORY.md (T2; H-V8F-FLATDAG-1) | -0.3451 [-0.4421, -0.2506] / -0.0544 [-0.1605, +0.0547] m, NOT ESTABLISHED | three decimals |
| by window class | hier/VERDICT_HIERWHERE.md, hier/pair2/VERDICT_HIERWHERE2.md | 1,305 and 1,094 windows; critical -0.4795 / -0.1589 (both separated) and -0.1817 (separated) / -0.1196 (not); bulk -0.1847 / +0.0702 and +0.3180 / +0.4928; critical minus bulk -0.2948, -0.2291, -0.4997, -0.6124; decomposition MIXED; logged mode missed on 53 to 71 % of windows | two decimals |
| commit rule | gate/VERDICT_GATEMIX.md, fv/FV_SPEC_V4_GATE.md, fv/GATE_BREAKEVEN.json | GATE-COSTS; AUROC 0.635 to 0.642; a gate banks 0.3 to 7.4 % of the oracle gain (ESTIMATED, binormal flag) | 0.64 |
| six-seed replicate | PREREG_V8F_PROOF_ADD11.md | 24 arms, six seeds, both training protocols; rule: all six seed differences agree in sign and the two-level interval excludes 0 | as written |
| rule shield | rule/VERDICT_RULESHIELD.md, cue/VERDICT_CUEPROBE.md | overrun share 91.6 / 84.7 % of 131 stop windows; with the logged rest distance overrun exactly 0 and -3.51 / -3.37 m; model cue: median error 11.8 / 9.6 m, flag on 10 / 18 windows; coverage 0.908 (119 of 131), margin about 7.4 m; stop flag AUROC ego 0.810 [0.744, 0.868], with 8x20 tokens -0.011 [-0.061, +0.043] | as written |
| bias edit | bias/VERDICT_BIASEDIT.md | differences 0.0085, 0.0256, 0.0171, 0.0085 against the gap 0.0684; AUROC 0.819 / 0.850 | 0.009 to 0.026; 0.068 |
| camera interface | res/VERDICT_RESPROBE.md, curve/VERDICT_RESCURVE.md, full/VERDICT_FULLRES.md | +0.2927 [+0.1082, +0.5512], +0.1491, +0.1637, +0.1801 [+0.0899, +0.2673]; 8x20 minus 4x10 at rank 256 +0.0111 [-0.0233, +0.0479]; full kernel +0.0755 [+0.0012, +0.1704] and +0.0429 [+0.0221, +0.0660]; R2 0.133, 0.401, 0.456; best fine set +0.0548 [+0.0003, +0.1114], bar 0.10; lead present 0.774 to 0.840, +0.0661 [+0.0326, +0.1033]; probes rising at 111 training episodes | two or three decimals |
| planted controls, untrained state | spatial/s0/VERDICT_S0.md, spatial/s0c/VERDICT_S0C.md, spatial/s0c/VERDICT_S0C_c.md | object 1.000 against 0.495; motion 1.000 against 0.493; gap R2 0.40, 0.26, 0.13; grid minus ego +0.27 [+0.08, +0.54]; grid minus mean +0.14 [-0.004, +0.29] | as written |
| trained state, two seeds | spatial/s1/VERDICT_S1EVAL_HIER_s0.md, VERDICT_S1EVAL_HIER_s1.md, VERDICT_S1_HIER_s0.md, VERDICT_S1_HIER_s1.md, SPREAD_S1_HIER_2seeds.md | presence AUROC 0.807 / 0.788 against 0.596, +0.211 [+0.105, +0.314] / +0.192 [+0.101, +0.286]; train gap R2 0.456 / 0.445 against 0.187; change over 2 s: ego +0.291, state minus ego -0.028 [-0.065, +0.008] / -0.025 [-0.059, +0.009]; two-seed range of gap R2 0.0103 | 0.81 / 0.79 against 0.60 in the abstract; 0.010 |
| decoded state | latvis/VERDICT_LATVIS.md | skill 0.361 [0.299, 0.423] / 0.353; token grid 0.777; rollout not closer than a copy | 0.36 / 0.35; 0.78 |
| weight average | wsp/VERDICT_WSP.md, wsp/TASKVEC.md | -0.0754 [-0.1029, -0.0469] m tactical ADE; kappa 0.738 [0.480, 0.988]; TASKVEC DIFFUSE, two head biases above seed noise (one replicate pair, no interval) | 0.075 [0.047, 0.103]; 0.74 [0.48, 0.99] |
| data selection | curve/VERDICT_RESCURVE.md (part C), PREREG_V8F_PROOF_ADD13.md, pod/chain2/arms_dataeff.txt | SELECT-FLAT at 25 and 50 %, 20 random draws; arms R-1/2, R-1/4, R-1/8, S-1/4, F-1/4 at two seeds, 19,164 steps | as written |
| state arms in training | PREREG_V8F_PROOF_ADD12.md, pod/chain2/arms_spatial.txt | arms G (grid, 40 cells x 16) and W (mean, width 640) at seeds 0 and 1 | as written |

## Not re-verified
* As v4: the count "2^7 corners" and the PUBLISHED statements of the cited papers. The three new references are cited for the statements `V8_THEORY.md` records from the banked primaries; their text was not re-read for this build.
* The status phrases "are training" reflect the pod queues at build time only (HIERREP, SPATIAL, DATAEFF); none of their arms is read in this version.
* Propositions 3 to 6 restate theorems of `lean/src/Hier.lean` and `lean/src/Pool.lean` in the paper's notation; the restatement itself is not machine-checked.

| file | sha256 |
|---|---|
| tanitad_4b_paper.tex | e4617dcc004f8e6a3123a85285885d7686b56fcf592017e7e2f5a2c849627dba |
| tanitad_4b_paper.pdf | 895e56bbc721665d35ad54f761349fcbdf108ffd9a45df12ecbd162b70b3cda6 |
| patch_tex_v5.py | 9f5003fee7b504aba51bd8f471a209ac4e56784fb9aacb4ab07139de873976d5 |
