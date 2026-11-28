# HANDOFF qhead_n1 (build agent, 2026-11-19)

State: all seven build steps done and banked here, nothing landed, nothing run on real data.
Tests: 39 passed (14 fit, 25 h3 and names). Mutations: 7 of 7 red, unmutated suite green before and after (`MUTATIONS_QHEAD_N1.json`, all_ok true).
Read `QHEAD_N1_DEFS.md` first (decisions with file:line, REVIEW items, what is not verified, the fit command).

Next steps for the lead:
1. Adapt `../qhead/pod_qhead_queue.sh` and `qhead_arms.txt` to the QE arm names and `qhead_fit_n1.py` (REVIEW 1). Not done here.
2. Run the four fits on the pod; check each record: `h5_check_coverage_in_0.15_0.25`, `selected_step`, `shift_c_m`, `fold_check_max_abs_m`.
3. Run the four rollouts with the unchanged `v8f_t1_eval_rule_qhead.py`; the control rollout of the first run stands for h1 only if the tool sha256 prefix matches (it does for the tool file in `../qhead`).
4. Pull, then `bash qhead_read_n1.sh`.

Strays (cannot delete): ed_h3.py, ed_names.py, ed_test3.py, ed_fix2.py, fix1.py (derivation scripts), work/ (pytest basetemp, mutation scratch), read/ (empty).
One slip: a bare `python -` was started once by mistake (waited on stdin, stopped, did nothing); a stray `rm -f /dev/null` was refused by the permission layer and not retried.
