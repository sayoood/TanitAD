# HANDOFF: Stage B chain for PREREG_V8F_PROOF_ADD11 (HIERREP). Built and tested on the dev box only; nothing launched.

## Ship and launch (lead)
1. Copy this folder to `/workspace/v8proof/hierrep/stageB/` and ALSO copy the card `F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD11.md` into it (the chain checks its sha256 = 94647519...29d1c). Optional: `r1_reference.json` = `{"ade6_all_windows_m": <dev-box ADE6 over all 2,399 windows of the stopbal_BAL-NULL_s0 T1-kin dump>, "n_windows": 2399}`. Absent = r1 PENDING (logged as WARNING, never a pass; the read prints PENDING).
2. Preview: `bash chain_hierrep.sh --dry-run` prints all 24 training and gate argv lines and the T1 / bank lines, touches nothing.
3. Launch detached (one chain only, it takes a flock): `nohup setsid bash /workspace/v8proof/hierrep/stageB/chain_hierrep.sh > /dev/null 2>&1 &`. It waits for `STAGEA_DONE.txt` (polls 60 s) and exits with `STAGEB_STOP.txt` if `STAGEA_STOP.txt` appears.
4. Watch: `/workspace/v8proof/hierrep/{stageB.log,STAGEB_STATUS.json}`; done = `STAGEB_DONE.txt` (names the reader rc); read output `hierrep/read/{RESULT_HIERREP.json,VERDICT_HIERREP.md}`. Re-launching after a STOP is safe: finished arms are skipped (token match + summary done at the final step + 141-file dump).

## What the chain does (order)
A: card sha, four pinned key files of commit 3e264bbf (sha256 of the git blobs), inputs (v9 md5; join / tracks / p10 sha256 prefixes 978f41bf / 1db431f1 / 14a062a9 from the earlier pod chain; lead and speed files hashed into `INPUTS_SHA256.txt`), eval grid and join copied to a no-ampersand `hierrep/pins/`. r1: T1-kin dump of `pulls/stopbal_BAL-NULL_s0/ckpt_last.pt` vs `r1_reference.json` (tol 0.005 m; FAIL stops). r2: `r2_regress_gate.py` (argv full, model built cut) must be REFUSED with `ZZDVB-REFUSEDZZ ... o_interface`. Then per seed 0..5, the whole seed before the next: gate+train+dump HIER_s, FLAT_s; bank of each frozen parent (4 shards); gate+train+dump DAG_s, FLATDAG_s. Finally the reader. Each gate = trainer `--selfcheck-only`, PASS line, token (`gate_token.py`: declared_vs_built PASS, G-LIVE pass, frozen-prior control exactly 0.0, commit, normalised argv sha256, data hashes); the training argv must equal the gated argv after dropping `--selfcheck-only` and the `--out` value, checked before and after training (`argv.json`).
Deviation from the brief's literal order: seeds are complete one at a time (Stage 1 and Stage 2 of seed s before s+1) so a killed chain leaves complete seeds for control r5. Pass `HR_SEEDS="0 1 2 3 4 5"` (default).

## Estimated A40 hours (ALL ESTIMATED, none measured on an A40)
Basis: the only pod precedent is B-prime, 0.185 s/step on an RTX 4000 Ada (3,565 s for 19,164 steps, LOOP_STATE); box 2 took 14,368 to 16,082 s on an RTX 4060.
* Stage 1 arm: 0.6 to 1.0 h. DAG / FLATDAG fine-tune (3,000 steps): 0.1 to 0.2 h. Bank (4 shards, 4,512 clips): 0.3 to 0.7 h each (no pod number exists). T1-kin dump (141 episodes, 2,399 windows, cuda): 0.05 to 0.15 h each. Selfcheck gate: about 0.03 to 0.1 h each (24 of them).
* Per seed about 3 to 5 h; six seeds about 18 to 30 h, plus r1/r2 under 0.3 h.

## Settings I chose (not in the card)
`OMP/MKL/OPENBLAS_NUM_THREADS=6` (7.65-core quota, rest for loader workers; env, not argv, so the argv stays the box-2 argv); bank children `--threads 1`, env 1 (4 shards x 1); `--dump-ticks --dump-heads` kept as in the verified recipe (set `HR_TICKS=0` to save disk: the read does not use them).

## Named gaps
1. Headway / time gap (LONGITUDINAL) and tactical goal-setting: the T1-kin dumps carry no lead block or goal label; printed as GAP rows (DEFS R14), not passes. STRATEGIC printed PAUSED.
2. `r2_regress_gate.py` ran only against a FAKE trainer module (stub run): the real trainer's refusal text and rc are first seen on the pod. If its refusal comes from another check than declared-vs-built, the chain still passes it when the log has `ZZDVB-REFUSEDZZ` and `o_interface`; read `r2_regression.json`.
3. The pod tree names no commit I could verify: the chain accepts `tree/COMMIT` or git HEAD if present (must start 3e264bbf, else STOP); otherwise it records "unverified" and relies on the four key-file sha256 pins.
4. Sha prefixes of join / tracks / p10 come from `chain_lorapod2.sh` (pins of the LoRA pod); if Stage A ships different bytes the chain stops at A-INPUTS (fail closed, read the message).
5. `df` is not used for disk (volume quota trap); 24 checkpoints + banks + dumps: ask the lead to confirm free quota before launch.

## Tests (dev box, no GPU)
`CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe test_hierrep_read.py`: 9 tests OK (planted effect HIER, null OPEN, mirrored FLAT, incomplete seed excluded, r3 failure marks READ NOT VALID, swap control, sign-flip exact, two-level bootstrap has seed variance, r1 PENDING/PASS/FAIL). `bash -n` clean; a stub run with fake trainer / T1 / bank tools (2 seeds) went through gates, tokens, argv checks, banks, dumps, the reader, and a second launch skipped 28 finished steps. Bad argv path: a mismatching `argv.json` stopped the chain (STAGEB_STOP.txt).
