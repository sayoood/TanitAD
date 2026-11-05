# Build brief: the (c) motion tokens and the (a) level head as two default-off v8F levers (MM, 2026-11-05)

Launch a Sonnet builder with this brief when an agent slot is free AND the motion probe (`motion_c/probe/RESULT.md`) has read.
If the probe reads MOTION PRESENT: build both levers below. If ABSENT / WEAK: build (a) only, and (c) becomes the pixel form
(`DESIGN_C.md` section 5), briefed separately.

## Files (one builder owns both: same two files)
`stack/tanitad/models/v8f_mini.py` (config + model), `stack/scripts/v8f_mini_train.py` (flags, FLAG_CHECKS G-DVB entries,
init-ckpt load of the new zero-init keys, gate selfcheck), new tests `stack/tests/test_v8f_motion_level.py`.
Start from the mirror tip (`git -C C:/Users/Admin/tanitad-push show <tip>:<path>`, `git ls-remote` first); work in a raw
`cat-file --batch` export on F:, never the E: tree. Output: a qland hold (mods-rw + stage) and SHA_CHANGES.txt, like
`raw/2026-11-05-recover-pertick-code/` (eccdabd). Do not edit `stack/scripts/launch_gate.py` or
`stack/ops/launch_gate_thor_env_failures_mapft.json` (another session's CAS).

## Lever (c): `--motion-tokens {off, diff}` (default off = today's model bit for bit)
* For each history frame k >= 1 and token cell c: m_kc = Linear_m(LN_m(f_kc − f_(k−1)c)) on the RAW cached features (before
  the shared adapter), plus the cell's `pos` and frame `time` embeddings; appended to the scene tokens of all three readers
  (S, T, O) with pad = pad(k) OR pad(k−1). Linear_m zero-initialised so a fresh arm starts at the init checkpoint's outputs.
* Mutation arms that must go RED: the appended tokens dropped; the difference replaced by f_kc; frame order reversed.

## Lever (a): `--level-head {off, on}` (default off = today's model bit for bit)
* Head: mean of the CURRENT tick frame's adapted tokens → MLP(d, d, 1) → v_hat (m/s); loss `--w-level` × Huber(v_hat, v_logged(t))
  on every window (label = logged speed: labels may use ego; inference reads vision only).
* O reads e_level(v_hat.detach() / 10) as one extra conditioning token at every tick (train and inference alike, so no
  exposure gap). The head never reads ego state, nav, limit or T's code (admissibility: vision only; disjoint from the
  situation classifier). Zero-init e_level's output.
* Mutation arms: the token dropped from O's cond; the head fed ego p10f instead of vision (must be refused by a guard);
  `--w-level 0` with the head on (refused: a head with no loss).

## Gate and tests (binding, PI 2026-09-26)
* Each flag has its FLAG_CHECKS G-DVB entry read back from the BUILT model; a flag that builds nothing is refused.
* G-LIVE on a real-data smoke: the new loss term finite; the new parameters get non-zero gradient; a control parameter that
  must read exactly 0.0.
* Init: loading the C0 init (`M2b_s0_dagoacc/ckpt_last.pt`) with a new lever on: every old key bit-identical, only the
  declared new keys missing, and the forward at step 0 equal to the off model's (zero-init).
* Full-suite comparison patched vs tip, NEW-ONLY must be 0 (`dagland/pertick/` runner pattern; create the --basetemp parent).

## The arms (C0's argv + one lever each; C0 argv = `v8f-recover/M2b_s0_recft_C0/argv.json`)
* CM (c): `--motion-tokens diff`. CL (a): `--level-head on --w-level <W>`, W chosen so the level loss is ~10 % of the total at
  step 0 (computed on the smoke, committed in the PREREG before the run).
* Read: `logged`, `hold_now`, `reindexed` (deciding), per `DESIGN_C.md` section 4.
