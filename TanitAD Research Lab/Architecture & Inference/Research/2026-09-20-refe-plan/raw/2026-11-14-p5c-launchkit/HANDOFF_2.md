# HANDOFF_2: P5-C launch kit, items 6/7 and the chain (2026-11-14). CPU only; no git writes, no land, no ship, no pod, no GPU.
Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/` (now ~124 MB). Read order: `HANDOFF.md` (round 1), this file, `PROFILE.md` (new tables), `RUNBOOK_P5C.md` (sections 10, 11), `TESTS.md` (HANDOFF_2 round). Tip at build `e3fe96ac`, moved to `1da03de9`; every file the kit touches has the same blob there (MEASURED, `blobs.txt`).

## State: everything the brief asked is built, tested (REGRESSIONS 0) and banked in `files/`
1. **G-LIVE (f) warm-start identity**: `refe/p5c_gate.py`, wired in `gate_checks.check_live` (P5-C profiles only) + `instrumented_run` (build capture digests every tensor at the trainer's first `zero_grad`; arms `warm_not_loaded`, `warm_perturbed`) + `launch_gate.run_checks`. Bound to the profile pin `init_from_sha256` (= `P5C_BASE_SHA256`).
2. **G-LIVE (h) lever isolation on the real model**: same module; argv half in the test. C1 only; C0/C0r must carry no table.
3. **Post-training G-ADAM**: `refe/p5c_postgate.py` (exact trainable count, step 2,000, non-zero `exp_avg` and `exp_avg_sq`, frozen trunk bit-identical, traj head moved, C1 != C0, `config.json warm_start.sha256` == base; controls exact 0.0).
4. **Chain** `refe/p5c_pod.sh` (+ `--with-c0r`, `--dry-run`, `--clear-fail`): stages in `RUNBOOK_P5C.md` section 11. Never fills a pin, never writes a token or hand-writes a marker.
5. **Tests**: `test_p5c_live.py` (71), `test_p5c_postgate.py` (56), `test_p5c_chain.py` (83), all with mutants; ~32 other launch_gate-reading scripts + p6_chain, p6_levers (298 s with bound TMP), gate_suite_prerun all green or identical-on-tip (`TESTS.md`).
Files changed vs round 1: `gate_checks.py`, `launch_gate.py` (+3 suite entries, +1 `need_b` condition, +`build_cap`), `test_p5c_profile.py`. New: the six files above (`p5c_gate.py`, `p5c_postgate.py`, `p5c_pod.sh`, `test_p5c_live.py`, `test_p5c_postgate.py`, `test_p5c_chain.py`).

## Contradictions (card vs brief vs code), named
- **G-ADAM moment**: card 10.4 says non-zero `exp_avg_sq`, brief says first moment. The gate requires both (the card is not weakened).
- **Card (f)** asks >= 64 real rows through the model; built as a bit-level digest of ALL tensors (stronger: identical tensors + same code imply identical outputs on any rows). Not run on rows.
- **Card 10.3** says the checkpoint meta carries `init_from_sha256`; `train.py` writes it only to `config.json` (`warm_start.sha256`), not the meta. I did NOT edit `train.py` (closure-land rule); the post-gate checks `config.json`. MM: add it to the meta only if the card is to be met to the letter.
- **(h) gradient**: the WTA loss is an L1, so a small label shift changes the loss VALUE but can leave the proposal gradient identical (MEASURED, `test_p5c_live.py`); (h) judges the loss value, with a control that must read exactly 0.0.

## MM decisions
1. **Gate evidence from P7** (class: reading of `p7_pod.sh` + `gate_checks.judge_*`; ESTIMATED, not run): **G-LIVE-LAB (`lab.json`) is reusable** if P7's was measured at the same worker count (13) and chunks (2) on the same metric cache (it is model-independent): `P5C_REUSE_P7_LAB=1`; **G-LIVE-MEM must be RE-MEASURED** (the P5-C model has no head; the chain does it once with C1's argv, C0 has the same shapes); **G-EGR is re-measured** (cheap, uses the memory run's `ckpt_last.pt` as the probe); G-LIVE (f)/(h) and G-CKPT are produced by the gate itself.
2. **`P5C_LABEL` (tc20s | p5r) is required and never defaulted**: it is the open A2 decision (PI 11-12 "P5-R regardless of R4" vs frozen A1 Section 0). The chain refuses without it.
3. **`P5C_LABEL_WORKERS`** pinned (13 on the L40 pod), as P7; **`P5C_NSHARD`** defaults to `nproc` and refuses above it.
4. **C0r** runs only with `--with-c0r` after C1 and C0 (PI approval is the MM's to confirm; the chain cannot).
5. Round-1 items 1-5, 8, 9 still stand (A2 amendment, C0r naming, G-RESERVE dropped, `--scorer-mode`, `--resume`, land order).
6. Not done: the W-A7 launch blocker, W-TRANSP / W-FAM read blockers, the A2 text, and `train.py`'s meta key (above).

## Final land list (coordinated lander, a COPY of each file, LF) and what follows
Land `files/<repo path>` = 22 files: the 16 of round 1 (`REBASE.md`) with these three REPLACED by this round's versions (`gate_checks.py`, `launch_gate.py`, `test_p5c_profile.py`), plus the six new ones. Guard: tip blobs `train.py 0e115c0d`, `gate_checks.py 89a50121`, `launch_gate.py 0775df5b`, `test_ego_yaw.py 5a61e43f`, `test_launch_gate.py 74160da4`, `test_p7_head_wiring.py 48e8ef5a` (all re-read at `1da03de9`). After P7's last arm trains: land, fill the 3 pins, re-derive `SHIP_MANIFEST.json` + the ship list (`p7_ship_list.py derive --profile refe_p5c`), ship `refe/` with `p5c_pod.sh`; the label kit (`a1/POD_SHIP_LIST.txt` + `runner/p5c_pod_label.sh` + the kit's `traj_label_table.py`) goes to `$P5C_SHIP`. Re-run the pin-sensitive tests after the pin edit (fingerprints).

## Deletions (this session's scratch, literal paths)
`C:/Users/Admin/p5ck2/` 961 MB (tipA 480 MB, ovB 480 MB, res logs, tmp dirs; logs/scripts banked in `logs/h2/`; every file in `files/` byte-identical to the tested ovB copy, MEASURED with `cmp`). C: free after: 15 GB (it dipped to ~12 GB by other sessions' jobs; never below the 10 GB floor). No other deletions; E: hold grew by ~7 MB of log tails.
