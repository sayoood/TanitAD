# HANDOFF imagplan_q (build agent, note PREREG_V8F_PROOF_ADD40_N7, 2026-11-19)
State: built, tested, mutation-listed, one CPU smoke done. Nothing run on the GPU, nothing committed or staged. Note N7 not edited. Folder `F:/TanitAD-scratch/v8proof/imagplan_q/`; pod tool folder `/workspace/v8proof/imagplan_q/` (all tool files there, sha256-equal to the dev box).

## What was built
* `v8f_t1_eval_imagplan_q.py` = copy of the planner wrapper patched by `make_wrapper_q.py` (16 edits, each asserted once); `imag_plan.py` unchanged copy (the rule is called with source cand-cv for cand-cvq); `qhead_model.py` unchanged copy of the ADD42 one.
  Mode `cand-cvq`; env `IP_QHEAD_FILE` (required, refused otherwise or for any other mode), `IP_QHEAD_CONST` (metres, replaces the head OUTPUT; q1 = 1e9, q2 = 0). Checks as the rule wrapper: no eval path, record next to file, head sha in record, checkpoint sha equal to the record's, plus h5 (check coverage in [0.15, 0.25]) true. Log keeps g_head, q_head, g_used per tick (NaN q for other sources).
* Pod: `pod_imagplan_run_q.sh`, `pod_imagplan_seq_q.sh` (tags IPo IPc IPl IPcq IPcqid IPcq0; same memory guard, 3-job limit; compare lines incl. `cmp IPcqid against IPc (q1, must be equal)`), made by `make_pod_scripts_q.py`. Dev box: `deploy_q1.sh` (make_deploy_chain_q.py), `chain_q1.sh`, `read_q1.py`, `read_q1.sh`.
* Run names (suffix q): IPoq_, IPcq_ (cand-cv), IPlq_, IPcqq_ (cand-cvq), IPcqidq_, IPcq0q_ + `<arm>`.

## Arms (all tree `/workspace/v8proof/tree_leadsup`, ckpt `runs/<queue>/<arm>/ckpt_last.pt`; head = banked fit, h5 true, fold check < 1e-5)
| arm | queue | ckpt sha256 (16) | head file (pod) | head sha256 (16) | sel. step | check coverage |
|---|---|---|---|---|---|---|
| LS_s0 | leadsup_a | df7f6b8094ad515f | qhead_dose_ls/heads/LS_s0.qhead.npz | b2220429eb64f82b | 2500 | 0.1909 |
| LS_s1 | leadsup_a | 055e57bda3def2ca | qhead_dose_ls/heads/LS_s1.qhead.npz | b04b174db4677b2c | 500 | 0.1646 |
| LSD_s0 | leadsup_b | 1b617e679ecf44bc | qhead_dose/heads/LSD_s0.qhead.npz | 35ac3f5f34d658cd | 2000 | 0.1881 |
| LSD_s1 | leadsup_b | c7b954dd88574eda | qhead_dose/heads/LSD_s1.qhead.npz | ed4c7de2ee7e7b02 | 1000 | 0.1955 |
Heads are under `/workspace/v8proof/qhead_dose*/heads`, not `runs/`. Records (banked copies): `qhead_n1/dose/pulled_ls/heads`, `qhead_n1/dose/pulled/heads`. The four checkpoint shas were re-read on the pod and equal the records.

## Tests and mutations (pod CPU, plain-python runner)
5 tests green (hand-computed min 12/3/70 with q 5 -> 5/3/5; identity with 1e9 bit for bit on executed paths and log columns; constant 0; head read at the tick frame, scaled to metres, checked against an independent numpy forward; refusals; other sources do not see the head). Mutations: 16 of 16 RED, all with the named catcher; control green before and after (`MUTATIONS_QPLAN.json`, `spec_qplan.json`, `pod_out_qplan.json`). Pins: `PINS_QPLAN.txt`.
Smoke (`pod_smoke_q.sh`, LS_s0, 8 windows, CPU, 12 s, rc 0): head loaded, imag log written (24 rows, q_head finite 24, g_used <= g_head, head sha = banked, const NaN). Folder `imagplan_q/smoke/IPcqSmoke_LS_s0` on the pod.

## REVIEW items (decisions the build forced; the note is not changed)
1. Gate: the activation gate (head gap <= 60 m) stays on the HEAD gap, g_used enters only the forecast and g0. Reason: control q5 (same active set) cannot hold if the gate sees min(g, q). Tested (a 70 m head gap with q 5 stays inactive).
2. Presence threshold: cand-cv default 0.5 for all four arms (the rule runs used 0.4 / 0.5 / 0.4 / 0.3 per arm). Kept as "presence gate unchanged".
3. q3 is checked from the imag log (head sha256 in the run) against the banked record and its check share in [0.15, 0.25].
4. -CONFIRMED is decided per name (cut, oracle, price) by `--combine`; a differing name is NOT-CONFIRMED and both are printed. The note's last sentence could also be read all-or-nothing.
5. q2 is read as pooled difference > 0 with interval excluding 0 (cell of the family reader, `ade6_m|all`); q4 as p2 of the card.
6. Reader q5 counts 2,399 / 642 / 56 come from the headway reader's counts; contact rate vs label is printed from cells whose name starts with C (not tested on real output).
7. Stale text: header comment lines 9-10 of `pod_imagplan_seq_q.sh` and line 3 of `deploy_q1.sh` still describe the old tags / folder; code is correct.

## Launch (the lead; not started by me)
`cd /f/TanitAD-scratch/v8proof/imagplan_q && nohup bash chain_q1.sh > chain_q1.out 2>&1 < /dev/null &`   (last line `ZZ CHAIN-Q1 DONE` or `STOP`; then `bash read_q1.sh`; IP_MAX_WINDOWS must be unset)

## Temp files left (no deletes made or attempted)
Dev box `imagplan_q/`: `_e.py` (edit helper), `read_q1_selftest/` (reader self-test, no data), `__pycache__` none. Pod `/workspace/v8proof/imagplan_q/`: `mut/` (two small tool trees, tmp, tmp_run), `smoke/`, `pod_out_control.json`, `pod_out_only.json` unused. I used `sed -i` once on my own `make_pod_scripts_q.py` (slip).
