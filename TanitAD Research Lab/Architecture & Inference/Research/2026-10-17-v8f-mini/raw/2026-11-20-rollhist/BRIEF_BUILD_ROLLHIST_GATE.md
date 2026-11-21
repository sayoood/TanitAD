# Brief: a study-side launch-gate kind for the ROLLHIST arms (dev box only, CPU, no pod)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.
Stage only: no commit, no push, no pod, no GPU. No edits to protected files (launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, v8f_t1_eval.py). One pytest at a time. Check `(Get-PSDrive F).Free` before each run (stop below 4 GB).

## Why
CLAUDE.md binds: no training run starts without a PASS token from the programme's gate, bound to commit, argv sha256 and data-manifest sha256s, never hand-written.
- For v8f-mini, the stage-A and stage-B tokens come from the study-side checker `dscale10k_check.py`. Card `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_DRAFT.md` section 4 says so: `gate_block3/study/dscale10k_check.py`, `chk_arm3.sh mint`.
- The ROLLHIST B2 prep (`F:/TanitAD-scratch/rollhist/build/b2/`, read `HANDOFF_B2.md` and `REPORT_B2_PREP.json`) wrote a CHAIN-LOCAL record, `rh_gate_record.py`, because that checker knows only the DSCALE10K arm names.
- The MM's ruling: the arms must get real tokens from the checker. Add the kinds.

## Do
1. **Find the live checker.**
   - Find which copy of `dscale10k_check.py` the stage-B chain on the pod actually ran. Read `/root/dscale10k/` scripts with `pssh cat` only (`source C:/Users/Admin/lorapod/podenv.sh`); ⛔ no other pod action.
   - Match it by sha256 against the repo copies under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-15-v8f-datascale/` at the mirror tip. Report the path and sha.
2. **Add kinds `RH` and `B0`** (seeds 0 and 1), in a NEW file version, `dscale10k_check_rh.py`, beside it. Leave the original untouched.
   - Minting a stage-B RH or B0 token runs every check the existing stage-B mint runs, against the arm's `U_sK.B.tmpl.json`.
   - **RH:** the argv equals the template EXCEPT exactly:
     - the four DAgger flags (`--dagger-p 0.1`, `--dagger-bank`, `--dagger-bank-sha256`, `--dagger-seed`);
     - the two declarations (`--dagger-bank-readout <token>` `--dagger-bank-kappa 0.2`), where `<token>` is `causal_g1x` or `causal_g1` (a CLI argument of the mint, recorded in the token);
     - `--out`.
     The bank's `bank.json` sha256 equals `--dagger-bank-sha256`. Its `generator` record equals the declared token and kappa (`readout_token` from `stack/tanitad/data/v8f_dagger.py` at 2bda9e1a). The init checkpoint is the arm's own stage-A model (K-B2).
   - **B0:** the template with `--dagger-p 0`, no bank flags, no declarations, its own `--out`.
   - **Both:** `stack/scripts/rollhist_pair_check.py` PASS on the pair, and the token binds the commit 2bda9e1a, the argv sha256 and the data-manifest sha256s exactly as the stage-B token does.
   - `rollhist_pair_check.py` hard-codes `causal_g1x`. For `causal_g1`, do NOT patch it in process: refuse with a clear message that the checker needs a flag (a committed-code change, which is the MM's). The B2 `render_argv_rh.py` patches it in process; flag that in your report, and do not use it.
3. **Controls (unit tests, CPU, fake small banks and argv files)**, each a named test:
   - matching RH and B0 mint PASS;
   - a wrong declared token REFUSED;
   - a bank header without the generator record REFUSED;
   - a bank sha mismatch REFUSED;
   - B0 carrying any bank flag REFUSED;
   - RH with a fifth changed flag REFUSED;
   - an init checkpoint that is not the arm's stage A REFUSED;
   - **identity:** minting the banked UE_s0.B argv through the new file's UE path gives a token equal to the original file's, field for field.
   - **Mutation:** delete the generator-record comparison, and the "header without record" test must go RED. Record it.
4. **Wire it:** a patched copy of `F:/TanitAD-scratch/rollhist/build/b2/rh_arm.sh` (in your folder, not in place) that mints through `dscale10k_check_rh.py` and gates the trainer on the PASS token the way the stage-B arm script does. Show the diff.

## Deliverables in `F:/TanitAD-scratch/rollhist/build/b3gate/`
- `dscale10k_check_rh.py`;
- the tests and their pytest log;
- the mutation record;
- the patched `rh_arm.sh` with its diff;
- `REPORT_GATE.json`;
- `HANDOFF_GATE.md` (at most 800 tokens), naming the repo path you propose for landing;
- a sha256 manifest.

Hand back in at most 250 words. Never wait on anything for more than about 4 minutes.
