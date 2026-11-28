#!/usr/bin/env bash
# FALLBACK ONLY. NOT RE-RUN. Derived from library_delta.json (the diff the verified run produced against tip d568e539).
# The staged library.json / LIBRARY.md / PDFs in this package are what was verified (--verify: 654 entries, 0 orphans).
# Use this only if the tip moved and the staged index would overwrite newer entries: run it in an EXPORT of the new tip
#   git -C <mirror> archive FETCH_HEAD | tar -x -C <empty dir>; cd <empty dir>
# then `python tools/kb_add.py --verify` and stage the changed Library files. Never hand-edit LIBRARY.md.
set -euo pipefail
REP="TanitAD Research Lab/Architecture & Inference/Research/2026-10-19-safety-head-lit/REPORT.md"

python tools/kb_add.py 2609.36438 --tag safety-head-lit --note "World4Scorer Tab.2 p7 outcome-label pairing control (ordering accuracy 64.2% -> 55.6% when bank labels shuffled); scene-matched bank following CLOVER" --cited-by "$REP"
python tools/kb_add.py 2609.20377 --tag safety-head-lit --note "Tab.3 p6 block C future-conditioned proposal scoring NC +0.1 DAC +0.2 TTC +0.5" --cited-by "$REP"
python tools/kb_add.py 2609.22317 --tag safety-head-lit --note "EditWM, candidate-specific cross-attention to corrected future features; screened, no sub-score ablation" --cited-by "$REP"
python tools/kb_add.py 2602.18887 --tag safety-head-lit --note "Tab.6 p7 per-agent per-step collision factor and per-step DAC with 9-point drivable-map lookup raise NC/DAC/TTC; Eq.3-4 p5" --cited-by "$REP"

python tools/kb_add.py 2601.22032 --tag safety-head-lit --note "Drive-JEPA, V-JEPA encoder + proposal-centric planner; screened, no sub-score table read for this report" --cited-by "$REP"
python tools/kb_add.py 2406.15349 --tag safety-head-lit --note "NAVSIM v1 metric definitions (NC/DAC/TTC/EP/C aggregation) behind every sub-score row" --cited-by "$REP"
python tools/kb_add.py 2503.12820 --tag safety-head-lit --note "Tab.3 p7 weighted confidence DAC 92.0->96.0; auxiliary perception tasks do not help (DAC 96.0->95.6)" --cited-by "$REP"
python tools/kb_add.py 2406.06978 --tag safety-head-lit --note "Tab.1 p4 distilling only the overall PDM score drops DAC 91.7->88.9; inference weighting W lifts DAC 91.7->96.1" --cited-by "$REP"
python tools/kb_add.py 2506.06659 --tag safety-head-lit --note "Tab.5 p7 coarse-to-fine trades DAC -1.1 for EP +3.9; Tab.2 p6 ViT-L recipe raises EP 85.0->91.3 with NC/DAC/TTC up" --cited-by "$REP"
python tools/kb_add.py 2504.01941 --tag safety-head-lit --note "Tab.3 p7 predicted future BEV states add NC +0.9 DAC +1.8 TTC +1.5 at the same trajectories; Tab.7 p8 unseen anchors" --cited-by "$REP"
python tools/kb_add.py 2606.07170 --tag safety-head-lit --note "Tab.7 p15 scorers trained on a fixed vocabulary break under search (iPad+search GTRS DAC 94.9->77.8); sec 4.3 p6-7 generalising scorer" --cited-by "$REP"
python tools/kb_add.py 2601.05083 --tag safety-head-lit --note "Tab.6 p8 separate+disentangled scoring branch and 6 sub-scores vs 1 (PDMS 90.0 vs 88.2); Fig.6 scoring attends different cameras by trajectory" --cited-by "$REP"
python tools/kb_add.py 2608.19085 --tag safety-head-lit --note "Tab.3 p10 action-conditioned future + safety hard negatives: NC +0.66 DAC +0.61 TTC +1.33 vs no-future at EP -1.39" --cited-by "$REP"
python tools/kb_add.py 2505.15111 --tag safety-head-lit --note "Tab.4 p8 proposal-aware spatial cross-attention (ProFormer) DAC 93.2->96.5 NC 96.9->98.1 TTC 90.8->94.3 vs BEVFormer" --cited-by "$REP"
python tools/kb_add.py 2506.06664 --tag safety-head-lit --note "Tab.1 p3 super-dense vocabulary + dropout for scorer generalisation off-vocabulary; Tab.2 p4 ensemble DAC 95.8/98.0 -> 99.3" --cited-by "$REP"
python tools/kb_add.py 2607.00399 --tag safety-head-lit --note "class-balanced high/low-quality candidate sampling to calibrate the verifier decision boundary (sec. III-B); screened, no sub-score table used" --cited-by "$REP"
python tools/kb_add.py 2606.24231 --tag safety-head-lit --note "continuous TTC label (App F.2.4 p20) + per-timestep aux heads; Tab.7 p15 weights as free EP<->TTC knob" --cited-by "$REP"
python tools/kb_add.py 2605.15120 --tag safety-head-lit --note "App. p8/p27 scorer calibration (s>=0.95 -> true mean 0.9753), Tab.20 p30 backbone swap moves test top-1 only 92.4->92.7; Tab.10 p16 261-candidate bank families" --cited-by "$REP"
python tools/kb_add.py 2510.24108 --tag safety-head-lit --note "Tab.8 p17 reward composition EP 63.3->83.8 with NC/DAC held; scorer token = 40 steps x 7 per-step features incl. motion differences (p16)" --cited-by "$REP"
python tools/kb_add.py 2603.29163 --tag safety-head-lit --note "Tab.1 p2 denser anchors keep helping (EPDMS 85.02->87.35, no saturation); Tab.6 p15 deformable aggregation vs MHA" --cited-by "$REP"
python tools/kb_add.py 2609.01799 --tag safety-head-lit --note "Tab.5 p8 longitudinal-only designed samples give the NC gain, lateral-only the DAC gain (scorer-only, frozen planners)" --cited-by "$REP"

python tools/kb_add.py --verify
