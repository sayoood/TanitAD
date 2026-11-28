# Fs launch gate 1c (MM 2026-11-24): G-LIVE's field_reach evidence form keeps 1b's non-probe keys

## What failed (MEASURED, pod `launch_gate_Fs.log`, 2026-11-24T07:29Z, package `code_86d174bb`)
G-HYG PASS, G-DVB PASS, then G-LIVE crashed in `gate_checks.check_live`:
```
det["field_reach"] = {k: (v if k in ("ran", "labelled") else {"loss": v["loss"], "groups": v["groups"]}) for k, v in fr.items()}
TypeError: list indices must be integers or slices, not str
```
Gate 1b (`raw/2026-11-23-fs-gate-1b/`, commit 95b3d174 here and side branch `agent/fs-replicate-gate1b` 86d174bb) made `field_reach` probe the first ACTIVE
sample and added `losses` (a list), `n_active` and `active_index` to its result. The evidence-form comprehension in `check_live` indexes every key except
`ran` / `labelled` as a probe dict. Fs did not launch; chain_e moved on to C1r as designed (the pod did not idle).

**Class: a fix verified at the FUNCTION and never through its CALLER.** 1b's tests and the pod probe called `field_reach` / `judge_field_reach` directly;
nothing ran `check_live` with a 1b-shaped result. Same family as CLAUDE.md "built, tested, and unreachable from its caller". This tip carried the same
line (`gate_checks.py:2828` at 9b71d987), so any field-profile gate on this branch would have crashed the same way.

## Fix
`field_reach_detail(fr)`: each dict-valued probe arm (`declared`, `zero`) is cut to `{loss, groups}`; every other key is kept as it is. `check_live` calls it.
Tests (in `test_scorerft_profile.py` here, `test_field_p7_profile.py` on the side branch, inside `chk_field_reach_active_sample`):
- every result shape (`fr`, `fr0`, `fr_n`, `frv`, not-run, unlabelled) goes through the helper: same keys, non-dict values unchanged, probe arms cut to `{loss, groups}`;
- `inspect.getsource(GC.check_live)` contains `det["field_reach"] = field_reach_detail(fr)` (the caller, not only the function);
- two mutations must go RED: the helper restored to the 1b-era comprehension (TypeError), and the call site reverted to it.

`patch_1c.py` applies the same edit to both trees (anchors must match exactly once).

## Tests (`tests.tsv`; CPU, serial, commit >= 6 GB, shared pytest lock)
| tree | file | result |
|---|---|---|
| side (86d174bb + 1c) | test_field_p7_profile.py (full, mutations) | ZZFIELDP7PROF_PASS 529 |
| side | test_launch_gate.py --static --no-banked | ZZLG_STATIC_PASS 1180 (first run FileNotFoundError: the export lacked PK/eval; rerun with it) |
| side | test_p5c_profile --fast / test_p5c_live / test_field_loss / test_gate_gpu_probes | 237 / 71 / 54 / 14 (GPU check skipped: CPU run) |
| main (9b71d987 + 1c, = this tip's two files) | test_scorerft_profile.py (full, mutations) | ZZSCORERFTPROF_PASS 349 |
| main | test_field_p7_profile.py (full) | ZZFIELDP7PROF_PASS 511 |
| main | test_launch_gate.py --static --no-banked | ZZLG_STATIC_PASS 1180 (same rerun) |
| main | test_p5c_profile --fast / test_p5c_live / test_field_loss / test_gate_gpu_probes | 302 / 71 / 54 / 14 |

## Where the fix lives
- This branch: `PK/refe/gate_checks.py`, `PK/refe/test_scorerft_profile.py` (this commit).
- Side branch `agent/fs-replicate-gate1c` = `cd3e3cde56dc0a9f24ab1408d7bceb97ca6ece19` (parent 86d174bb; only `gate_checks.py` and `test_field_p7_profile.py` differ):
  the commit Fs's package E2 is built from, so Fs stays byte-identical to F's package 10848403 apart from the gate.
- Pod: `chain_f.sh` v2 reads package E2 from `fs/PKG_FS2.txt` and refuses a package without the 1c line.
