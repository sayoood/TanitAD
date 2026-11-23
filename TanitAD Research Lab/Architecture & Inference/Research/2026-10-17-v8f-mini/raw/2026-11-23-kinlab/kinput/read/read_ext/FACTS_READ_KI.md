# FACTS_READ_KI (card PREREG_V8F_KINPUT 05e786a8; code only, no KI number exists)
RH_AKIND=KI: treated arm KI; KINDS (KI,B0); ROLL_KINDS (KI,); RUN_KINDS (KI,B0,RH). 15 pairs, listed in tmp/drywalk.txt:
- P1 KI-B0 C*; P2 ACC/D1/D2 on KI; P2 ACC/D1/D2 on B0; P4 KI(C*)-B0(K).
- BESIDE B0 pairs: KI-B0 under the other 5 forms. Their dict keys read "BESIDE_RH_minus_B0_<form>" (legacy name of the unchanged loop; the first run is KI).
- Beside KI-RH: BESIDE_KI_minus_RH_R0Hb20 and _R0C10XHb20 (the second is the card's ATTR; not duplicated).
- RK-RL (ATTR) pair: DROPPED, no KI analogue. KI-RK exists only at T0 (mech_ki.py beside), not at T1.
mech_ki.py: MECH4 = KI-RH at T0 ADE6, paired per seed, YES iff hi<0 at both seeds; beside KI-B0, KI-RK, RH-B0. Output out/read_ki/MECH4_KINPUT.json. Console names KI/RH/RK.
Chain: rh_read_ki_chain.sh (gate0 needs RH/B0 dumps+ckpts, RH/RK/B0 T0 evals; pulls KI_s0/s1 after ARM_DONE with pod sha256; PAIRS line single-quoted). Retry .sh/.cmd from RK's.
Tests (CPU, commit 10.55 GB at start, >=5.5): selftest KI, RK, RA all pass (ALL_PASS True). rh_common for RH/RHp3/RHK/RL/RA/RK identical to pre-edit. sha of out/read_rk + out/read_ra files unchanged (679aa9b0...).
Planted MECH4 (plant_ki_check.py, synthetic dumps): KI-RH -0.05 both seeds -> MECH4-YES; 0.0 -> MECH4-NO. bash -n both scripts OK. Dry walk printed pairs; chain not started; pull/roll/T0 not run.
Frozen readers, cells, floors, estimator, tier, executor untouched. rh_read.py edit: NEXT_KI, ki_answer, KI branch in main.
