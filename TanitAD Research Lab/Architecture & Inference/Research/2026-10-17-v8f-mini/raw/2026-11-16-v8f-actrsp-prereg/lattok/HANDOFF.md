# LAT-TOK handoff (2026-11-16)
STATE: full run launched detached on the LoRA pod CPU (8 threads, nice 19, no GPU), ~15-20 min total (smoke passed; feature stage 1 done, analysis running).
Frozen prereg: AMENDMENT_1_BCAL_DRAFT.md sha256 4e122aa5...f53834 (verified before use). Rule, cells and controls are committed in the docstring of lattok_analysis.py.

## HEADLINE DEVIATION (read first)
The pod train cache is a RE-ENCODE (bf16, RTX 4000 Ada), not a copy of box 2's cache_full: `/root/data/cache_pod/train/compare_box2.jsonl` (4 clips) shows pool4x10 rel_l2 ~0.024, 0% of tokens bit-equal; 0/141 clip md5 match clips141.json. Measured here: pod-token z1 vs the banked Z1 rel L2 = 0.0151 (s0) / 0.0148 (s1), 0 bit-equal. So "(a) reproduces the banked z1 bit-equal" is FALSE; "(a) reproduces what the model computes from the same tokens" is TRUE (z1_from_mean(a) == wm.z1(adapt), bit-equal). Window ok-masks and raw actions match the banked windows exactly (act_ok_equal, okmask_equal true). I did not pull box 2 (forbidden). The MM decides whether a pod-cache verdict is admissible or box-2 clips must be pulled.

## Where
- pod: /root/lattok/{lattok.log (sentinel last line), out/lattok_result.json, feats/ (feat_s{0,1}.npy ~0.5 GB each + ctrl_feats.json), data/, stack/, run_lattok.sh}
- local: F:/TanitAD-scratch/actrsp/lattok/{lattok_feats.py, lattok_analysis.py, run_lattok.sh, summarize_lattok.py, smoke_log.txt}
## Next step
1. Poll: pssh 'tail -2 /root/lattok/lattok.log' (sentinel `ZZ LATTOK DONE ZZ` or `ZZ LATTOK FAIL ZZ <why>`).
2. pscp the json to F:/TanitAD-scratch/actrsp/lattok/lattok_result.json; `python summarize_lattok.py lattok_result.json`; write RESULT_LATTOK.md.
3. Controls to read in the result: lin.z1.acc.r2 must be ~0.211 (s0) / 0.205 (s1) and lin.z1.yaw.r2 ~ -0.048 / -0.050 (banked kstep / amend1 D); lin.*.shuf.r2 <= ~0; n_pairs 8306 == kstep n_pairs; a_mean_wm == kstep's.
4. Local scratch (stage, pkg, pkg2, tev, tars) already deleted, sizes in deletions_lattok.log. Still to delete when done (literal paths): pod /root/lattok/feats, /root/lattok/data, /root/lattok/smoke, /root/lattok/stack, /root/lattok/tanitad.tar, /root/lattok/tev.tar (log size each). Keep scripts, logs, result.
Memory note: analysis peaks ~7.3 GB RSS (set c); under the 8 GB cap.
