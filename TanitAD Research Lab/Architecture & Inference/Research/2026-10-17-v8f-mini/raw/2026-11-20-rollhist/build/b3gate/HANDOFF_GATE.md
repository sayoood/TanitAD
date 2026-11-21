# HANDOFF_GATE (ROLLHIST B3 gate build, 2026-11-20) -- staged on F:, nothing landed, nothing on the pod
Dir: F:/TanitAD-scratch/rollhist/build/b3gate/ . Propose landing under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-15-v8f-datascale/rollhist_gate/` (study files, outside stack/; lander mods route).
Live checker: pod `/root/dscale10k/study/dscale10k_check.py` sha 7467e5d7.. = repo gate10k/pod_out/study copy (minted U_s0/U_s1 stage B). New file `dscale10k_check_rh.py` derives from the latest superset (reps1 copy ac37c1b6..); original untouched.
New: arms RH_s0/s1, B0_s0/s1 (stage B). Mint runs all stage-B checks against `U_sK.B.tmpl.json` + C-ARGV exact-vs-template, C-RH-PAIR (stock rollhist_pair_check.py), C-RH-BANKGEN (bank.json sha + generator record == declared token/kappa), G-DAGGER(-B0). Token binds commit, argv sha, data sha256s, template, pair argv, pair-check file; runner `run` refuses rc 2 without it.
Tests: 37 pass; mutation (delete generator comparison) -> header test RED (MUTATION_RECORD.md).
Wiring: `rh_arm.sh` (patched copy, `rh_arm.sh.diff`) + new `chk_arm_rh.sh`, `rh_evidence.py`. Pod install (MM): copy dscale10k_check_rh.py + dscale10k_merge.py to /root/rollhist/study/, chk_arm_rh.sh + rh_evidence.py + rh_arm.sh to /root/rollhist/.
Open for MM:
1. causal_g1: refused by design; needs a committed flag in rollhist_pair_check.py (MM-owned). causal_g1x works. render_argv_rh.py's in-process patch is not used by the new rh_arm.sh; rh_chain.sh step 3 still calls it (g1 only).
2. G-HYG/G-LIVE/G-CLOCK/G-EVAL/G-CKPT evidence is INHERITED from evidence_UE_sK.json (cdbff971); G-SUITE is fresh. Accept or order fresh probes.
3. No UE_sK.B argv exists on the pod; identity was proven with a filled template. Union data checks stubbed in unit tests only.
4. Pod paths untested (bash -n only); first run may need a path fix. rh_gate_record.py is kept as a supplementary record (pilot, bank-check, tree marker); drop if redundant.
Next step: MM review, land, install on pod, then launch chain.
