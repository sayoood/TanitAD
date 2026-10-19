# HANDOFF L2-FRESH-STEPA (2026-10-19 16:45). SEALED; a chain is WAITING for the GPU lock; no agent should wait on it

**State.** (1) `PREREG_L2_FRESH.md` is SEALED: sha256 `5006ed844979578f4c42599cd79b09c15bd07862d9c9bda94b21481d5e8d4e54` (MM ruling verbatim inside; pins: carve `0629811f...`, heads `bded124e...` / `86f0b20f...`, P2 head, fold heads, 5 code files). Guard proof: accept (exit 0) + 7 RED copies refused (exit 5), `raw/guard_proof.json`. (2) Log-OOF companion built: 5 x 2 fold heads refit, banked OOF reproduced bit for bit (`raw/logoof/`, copy in `D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads/`). (3) READ 2 (universe D) prepared: `PREREG_L2_FRESH_D_DRAFT.md` (DRAFT), carve 99 tokens / 84 logs sha `80de7dd1...`, exposure measured (83/84 logs inside M-PDM and M-SCENE: D is log-SEEN by P2), fetch list 396 jpgs, G2b bridge built + tested on a fake planner (NOT run). (4) B chain (cmd PID 25036, since 16:03:40) still waits: `chain_active` held by another session + RAM < 8 GB. Untouched. (5) Step 7 (the read) has NOT run. Tests: `test_fresh.py` 27 passed, `test_readtwo.py` 12 passed.

**Paths.** Package `C:/Users/Admin/par18/l2fresh/` (READY.md has everything, incl. the exact bridge command in section G). B run dir `D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/`. Chain log `chain_B.log`. Staged repo copy `out/repo/...`, `out/paths.json` (re-run `python code/make_stage.py` after any new file).

**Check (one command, no polling turns).**
`grep -E "ZZGPU|ZZBUNDLE|STEP[0-9]|EXPORT|MERGE|SCORES|CHAIN_DONE|FAIL|G2B" C:/Users/Admin/par18/l2fresh/chain_B.log D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/step3.log`
Expected end `CHAIN_DONE_NO_READ`. Then verify `ZZBUNDLE_OK 400`, `labels.npz.g1.json` pass, `scores.npz.gates.json` G2a / G2c / G3 and `G2b_cross_device.pick_agreement` >= 0.99 (else the GPU re-score contingency, a new decision), `fresh_ep_scores.json` `pass: true` (C-FEAT).

**Next steps, in order.**
1. When the gates above hold, run the read (READY.md section D, two commands: `--seal-check-only`, then the read; CPU, ~1 min). Stated expectation in the seal: **O3**. Write the result, update `GOALS_AND_CLAIMS.md` (`D-REFE-L2-OFFPROP-1`) the same turn, act on the sealed section 5.4 (O1 Step B / O2 L2-BRANCH / O3 READ 2).
2. MM rulings pending on READ 2: D1 (declare the log-level G0 FAIL; stamp "token-unseen, LOG-SEEN by P2 (83/84); log-fresh for the L2 head"), D2 (is a 99-token READ 2, expected half-width ~0.054, worth the fetch and build), D3 (O3-on-READ-2 rule), D4 (PI approval of the 396-jpg pod fetch under the P6 overlap rule). Only then: the build in the D draft section 7.
3. The bridge (READY.md section G) only after the B chain released the lock and the MM decides READ 2 proceeds.

**Do not.** Edit `analyse_fresh.py`, `logoof_companion.py`, `score_fresh.py` (or `l2lib.py`, `stepA.py`): the seal pins their sha256 and the read then refuses (exit 5); a change needs a sealed Amendment 2 first. Run the read before `CHAIN_DONE_NO_READ`; read any A-quantity if G2b, G1 or C-FEAT fails; kill by name (`pkill -f`); touch another owner's `devbox_gpu.lock`; run the bridge while another job holds the GPU. To stop the chain: kill the bundle python (if it holds the lock), then the launcher python, then 25036, by explicit PID (`Get-CimInstance Win32_Process -Filter "ParentProcessId=25036"`).

**Limits.** At most 2 Python processes; no pod; D: nearly full (exFAT, 1 MiB clusters): write only small files into `l2_fresh_stepA/`. `_selftest_stepA/` is a code-path fixture (in-sample heads), NOT a result.
