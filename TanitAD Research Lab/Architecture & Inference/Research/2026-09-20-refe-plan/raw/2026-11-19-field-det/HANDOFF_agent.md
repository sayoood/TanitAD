# HANDOFF: FIELDDET (2026-11-19) -- sweep still running detached

DONE and staged in `land/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/` (all LF except p7_ship_list.json, which keeps the tip's CRLF):
field_loss.py, test_field_loss.py, launch_gate.py, test_field_p7_profile.py, p7_ship_list.json. Mirror FETCH_HEAD = 40938dd88ba7c23ebcecc96cbda6ea295fd7fd7b.
New field_loss.py LF sha256 = da7b43030cbc89b43da74c3c9ce48b12450d3080c702374dc2e62dea929563b3.
Training-path identity: `training_path_identity.json`.

PENDING: the serial CPU sweep (58 test scripts mentioning launch_gate / LG. / gate_checks). It runs detached (WMI, bash PID 39772) and WAITS while free
commit memory < 6 GB (it was 1.5-4.6 GB when started: other chains hold the memory). Results append to `sweep/summary.tsv` (name, rc, seconds, last ZZ token),
logs `sweep/<test>.log`; the file ends with a `SWEEP_DONE` line. test_rls1_queue and test_gate_suite_prerun are re-run once on failure (`.rerun`).
test_field_loss.py (ZZTEST_PASS 54) and test_field_p7_profile.py (ZZFIELDP7PROF_PASS 502) already PASS and are NOT in the sweep list.
Judge: failures other than the known pre-existing ones (test_carve_feed, test_launch_gate [a D: path], test_p2r_chain, test_p7_ship_list, test_rls1_chain,
test_s1aux_chain, test_s1z_preflight, test_s1zsm_preflight, test_scorer1p6_preflight, test_sft4r_chain, test_vdac_stage0_forward; intermittent: test_rls1_queue,
test_gate_suite_prerun) are ours: compare against a pristine-tip run of the same test before blaming the edit.
CLEANUP after SWEEP_DONE: delete `F:/tanitad_scratch/fielddet/` (literal path; the sweep reads its tree from there; ~160 MB).
