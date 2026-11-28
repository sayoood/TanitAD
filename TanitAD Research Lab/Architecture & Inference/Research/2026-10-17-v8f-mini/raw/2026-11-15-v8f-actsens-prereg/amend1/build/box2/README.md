# ACTSENS: box-2 launch package (world-model action sensitivity: `--wm-idm-gamma`, `--wm-sampler divrs`)

Card: `theory/actsens/PREREG_V8F_ACTSENS_DRAFT.md` (frozen, sha256 `b62227d9`). Code plan: `theory/actsens/REACHABILITY_ACTSENS.md`. Code: `actsens.diff` (a diff against ladder commit `1faa1a06`, to be landed by the Master Mind; the chain takes the landed commit as `ACTSENS_COMMIT`).
Pattern: `theory/egowm/box2/` (read, never edited). **Nothing here has touched box 2.** It ran only over stub transports and a fake box folder.

## Order of the chain (card section 2)
1. optional predecessor wait (`ACTSENS_PRED_NAME` + `ACTSENS_PRED_STATE`; unset = none, only `BOX2_BUSY.txt` excludes). UNVERIFIED which chain precedes this one on box 2.
2. **ship** to own tree `D:\tanitad\actsens\repo` (the four patched files plus the trainer's import closure, raw blobs from the commit, needles checked).
3. **pilot** `PILOT_DIV_s0`: 200 steps of DIV_s0's argv; it measures the DIVRS start-up. The same job gates the full-length IDM_s0 and DIV_s0 argvs. Projection over 6 runs; the 72 h rule needs an amendment file if exceeded.
4. **runs**: IDM_s0, NULL_s2, DIV_s0, IDM_s1, DIV_s1, NULL_s3. Each: disk, GPU idle (< 3,500 MiB, no python.exe), argv pin, `--selfcheck-only` gate, per-run token, training under the token, post check, bank with sha256 against box 2's copy, cleanup of the run's own gate dir.

## What the gate reads back (`actsens_check.gate_problems`)
`declared_vs_built == PASS`; argv sha; `steps_total` 19,164; data identity and R35's window count 40,358 / window-list sha; WM-only record (start state sha, 0 mismatches, WM re-initialised under the run's seed); the ladder read-back (k 10/10/6, ctx 3, strides 3 5, pooled windows, zcache of the chassis seed); WM parameters = 2,011,970 (R35's).
**IDM arms:** `levers.wm_idm_gamma_readback` declared and BUILT = [1000, 100, 100] (read from `HJEPAStack.built_levers()`, never `WMAux.gamma`); C3 (AMEND_ACTSENS_1 A1): the trainer's `live_first` (G-LIVE step 0, before any update) of IDM_s{k} against REF_s{k}'s `live_first`: unweighted terms and level/L2, level/L3 equal bit for bit, level/L1 difference = 900 x idm/L1 (1e-5 relative). `live.terms` (the LAST of G-LIVE's three AdamW steps) is never the C3 record. **DIV arms:** `levers.wm_sampler.kind == divrs`, the card's edges and cap, cell counts sum to 40,358, `wm_sampler_probe.pass` (card control 1). **NULL arms:** neither key present, sampler `uniform`. G-LIVE and planner identity as in the ladder.
**Post check:** non-WM tensors bit-equal to M2b's; C5 every optimiser parameter has Adam state; C2 `gamma_from_log` reads 1000/100/100 (IDM) or 100/100/100 at every logged step within 1e-4.

**Reference gates REF_s0, REF_s1 (AMEND_ACTSENS_1 A2, gate only: never train, no run token, not in the run order):** the argv is R35_s{k}'s exact argv with `--out` = the REF gate out and `--wm-idm-gamma 100 100 100` appended (the default dose, named so that `live_first` is recorded); both are frozen in `frozen/argvs_actsens.json` (sha256 pinned in `actsens_check.py`). Checks: the ladder, G-DVB and G-LIVE checks of the IDM arms; `live.terms` equal to `frozen/r35_live_terms.json` s{k} on EVERY key, bit for bit (C1 on real data); built gamma [100, 100, 100]; `live_first` present. Order: the pilot job gates REF_s0, IDM_s0, DIV_s0 in that order; IDM_s1's run stage gates REF_s1 before IDM_s1. An IDM arm whose REF gate is missing or refused is refused (ARM_REFUSED). The REF selfcheck is banked next to the pilot (`ref_selfcheck.json`) and next to each IDM bank. The trainer commit must carry `live_first` (a needle of `actsens_check.NEEDLES`, checked at ship time).

## Launch (Master Mind, from a session background task; do not run before the landed commit and the box-2 go)
```
export ACTSENS_COMMIT=<40-hex sha of the landed ACTSENS code commit>
export ACTSENS_SSH="$(command -v ssh)" ACTSENS_SFTP="$(command -v sftp)"     # absolute paths; PATH is never searched; ACTSENS_DRY must be unset
[ export ACTSENS_PRED_NAME=<NAME> ACTSENS_PRED_STATE=<dir> ]                # optional predecessor: DONE.txt first bytes "ZZ <NAME>-DONE"
[ export ACTSENS_AMENDMENT=<file> ]                                         # only if the pilot projects above 72 h and the PI amended
bash <this folder>/actsens_chain.sh
```
State: `C:/Users/Admin/actsens_state` (`ACTSENS_STATE`). Banks: `C:/Users/Admin/t2v8f_pulls/actsens_<tag>/`. Sentinels in their own files only: `STOP.txt` (`ZZ ACTSENS-STOP`), `DONE.txt` (`ZZ ACTSENS-DONE`), `ARM_REFUSED_<tag>.txt` (`ZZ ACTSENS-ARM-REFUSED`). Resume after a stop: inspect, then `ACTSENS_RESUME=1`.

## Failure behaviour and isolation
A gate, pin or token refusal of one full run writes `ARM_REFUSED_<tag>.txt` and the chain goes on with the next arm; the chain then ends with STOP, never DONE. Every other failure stops the chain. The chain acts only on `D:\tanitad\actsens\` and `D:\tanitad\runs\v8f_actsens\`, plus the read-only inputs the argvs name. Deletions: the run's own gate and pilot dirs, by literal path, after `BANK_VERIFIED`, logged (path, bytes). Checkpoints are never deleted.

## Evidence class
* MEASURED here: `tests/UNIT_RESULT.log`, `tests/DRYRUN_RESULT.log` (stub transport, fake box, stand-in trainer).
* UNVERIFIED until the first real run: that the real selfcheck carries the key names the checker reads (read from the ACTSENS trainer source as built here; the stand-in writes the same names); C3 bit-equality on box 2's GPU against R35's banked live terms (a mismatch REFUSES the IDM arm, fail closed); the DIVRS cell counts and start-up time on the 91-frame ladder windows (the pilot measures the start-up); which chain precedes this one.
