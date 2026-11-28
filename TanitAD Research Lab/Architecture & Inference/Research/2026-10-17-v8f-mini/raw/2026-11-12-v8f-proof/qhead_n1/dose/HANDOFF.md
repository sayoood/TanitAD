# HANDOFF dose step (note ADD42_N2 section 3), build agent 2026-11-19

State: built and tested on the dev box; nothing ran on the pod. Tests 73 passed; mutations 13 of 13 red, suite green before and after (`MUTATIONS_DOSE.json`). `qhead_fit_n1.py` needs no change (DOSE_DEFS.md). Real-data control: the pack builder on the layout of the 1,200 probe clips reproduces the banked pack (d1 pass, difference 0). Read DOSE_DEFS.md (REVIEW 1: dev-box extract_states.py is CPU only).

Files (sha256 prefix): pod_qhead_dose_inputs.sh b2e99933b5c6, pod_qhead_dose_fits.sh 4bfae538c3e2, pod_dose_layout.py 3f76129eae8e, build_dose_pack.py 0803187a015a, dose_d1_pack_check.py d3e394f15f05, dose_d2_state_check.py e063f2485e50, MUTATIONS_DOSE.json b139edd757ca; unchanged tools qhead_fit_n1.py 57e1242713e0, qhead_model.py f6ff755e3dd4. Pod copies to compare: extract_states.py f7cfb17e72a8, pod_lead_apply.py c9b1ecef2302.

Order of commands (O=/workspace/v8proof/qhead_dose):
1. Push to `$O/tools/`: pod_qhead_dose_inputs.sh, pod_qhead_dose_fits.sh, pod_dose_layout.py, dose_d2_state_check.py, `../qhead_fit_n1.py`, `../qhead_model.py`; compare sha256 both sides, also `extract_states.py` and `pod_lead_apply.py` in /workspace/v8proof/probe.
2. `DRY=1 bash $O/tools/pod_qhead_dose_inputs.sh`, then `nohup setsid bash $O/tools/pod_qhead_dose_inputs.sh > $O/start_inputs.out 2>&1 &` until `ZZ QHEAD-DOSE INPUTS DONE 2` in `$O/run/dose_inputs.log`.
3. Pull `$O/layout/train_layout_0_4572.npz` (+ .json) to the dev box. Build: `python build_dose_pack.py --layout <layout> --join C:/Users/Admin/lorapod/ship2/join_train.npz --out train_pack_all.npz`.
4. d1 on the dev box: `python dose_d1_pack_check.py --new-pack train_pack_all.npz --banked-pack spatial/leadsup/read/train_pack.npz --out d1_pack.json` (exit 0).
5. Push the pack to `$O/pack/train_pack_all.npz` and d1_pack.json to `$O/controls/`.
6. d2 on the pod per ARM in LSD_s0 LSD_s1: `python $O/tools/dose_d2_state_check.py --new $O/states/${ARM}_dose_train.npz --banked /workspace/v8proof/probe/states/${ARM}_train.npz --layout $O/layout/train_layout_0_4572.npz --out $O/controls/d2_${ARM}.json` (exit 0).
7. `nohup setsid bash $O/tools/pod_qhead_dose_fits.sh > $O/start_fits.out 2>&1 &` until `ZZ QHEAD-DOSE FITS DONE 2 of 2`; read `<ARM>.qhead.json` (h5, selected step, shift, clip sets).

Not built: rollout queue and names tool for the `-QF` arms. Strays (deletes refused): work/ (9 MB).
