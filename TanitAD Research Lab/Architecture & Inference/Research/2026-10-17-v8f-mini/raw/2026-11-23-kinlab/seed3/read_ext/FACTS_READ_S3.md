# FACTS_READ_S3 (card PREREG_V8F_SEED3 75f3b191; code only: no s2 checkpoint, no s2 number, nothing pulled/rolled, no GPU)
Pairs (dry walk, `RH_S3_DRY=1 bash rh_read_s3_chain.sh`, rc 0, stops before the pull; log out/rh_read_s3_chain.log, tmp copy deleted):
- RH pass (RH_AKIND=RH, 13 pairs): P1 RH-B0 C*; P2 ACC/D1/D2 on RH and on B0; P4 RH(C*)-B0(K); BESIDE RH-B0 under the other 5 forms (incl. R0Hb20).
- KI pass (RH_AKIND=KI, 15 pairs): P1 KI-B0 C*; P2/P4 on KI; BESIDE KI-B0 x5; BESIDE_KI_minus_RH_R0Hb20 and _R0C10XHb20.
- Rollouts 18 = {B0,RH,KI} x 6 forms at seed 2 (T1-kin/hb0.2, rh_roll.py now accepts seed 2); T0 evals B0_s2, RH_s2, KI_s2 (rh_t0.py run).
Floors (frozen two-seed F_c, taken unchanged by rh_read.py --floors-from, one-seed reads only; refused unless seeds == [0,1]):
- RH pass: out/read/RESULT_ROLLHIST.json sha256 5613818e5748b8ee38fa6c88165bf5699dd549f04128c4af8f068643e0d99c41 (13 pairs, P1 six cells + strat arcs)
- KI pass: out/read_ki/RESULT_ROLLHIST_KI.json sha256 a12fef1c111721e8175bc4c4c70078489f3ef52a0cc6993493ae6210fb731bfa (15 pairs; KI's own floors, incl. BESIDE_KI_minus_RH_*)
- both hashes are pinned in the chain (gate0 fails if they change) and stamped into RESULT_S3_*.json (floors_file).
Selftests after the edit (CPU): `RH_AKIND=KI|RK|RA|RH python rh_read.py --selftest` all rc 0, "all hold: True", zero `"pass": false` (a new case S3-FLOORS-FROM-FILE added, incl. regression arms). The finished reads are unchanged: rh_t0.py default (no --bar) reproduced out/read_ki/GT0_ROLLHIST_KI.json BYTE-IDENTICAL; rh_common/rh_read defaults unchanged. No frozen reader (headway_read_r2.py, q3_read_r3.py, rh_strat.py, strat_roll_read.py) touched.
G-T0: rh_t0.py read --bar 0.1174 --beside-bar 0.0705 (KI s0,s1 test: HOLDS both under both bars). G2 = RH_s2 - B0_s2 upper bound <= 0.1174, 0.0705 beside.
Planted name check (plant_s3_check.py, 19/19 PASS): five names, precedence (BLOCKED > T0-FAILS > HARMS > REPLICATES > NO-EFFECT), -STRADDLE both ways (NO-EFFECT->REPLICATES, NO-EFFECT->HARMS), no suffix when the name does not change, bar inclusive at 0.1174, absent lever record -> BLOCKED, CLI writes RESULT_SEED3.json, mech_s3.py on synthetic T0 evals.
`bash -n` OK on rh_read_s3_chain.sh and rh_read_s3_retry.sh. Single quotes inside every python -c.
Decisions for the MM (not in the card):
1. MIXED family (some cells BETTER, not all) is neither BETTER nor WORSE (family rule as written): can give NO-EFFECT while a cell is BETTER. s3_name prints mixed_families.
2. Lever-acts record out/read_s3/LEVER_ACTS_S3.json (7 keys, s3_name.LEVER_KEYS) must be written from the pod records; absent -> SEED3-BLOCKED. The card's 8th item (checkpoints sha-equal) is computed from provenance.
3. Missing G2 / a NOT-COMPUTABLE P1 family -> NOT-DECIDABLE (outside the card's five names), never a made-up name.
4. Three-seed pooled reads: NOT-COMPUTED. The card freezes F_c for two seeds; rh_read.floor_of takes 1 or 2 seeds only (3 seeds falls to the one-seed stage-A branch).
5. Readers' spread arm in the s2 pass = stage-A UEA dumps (the existing one-seed branch); it only feeds the readers' own printed rule, not the name.
