# HANDOFF TF-N6 (built and tested, NOT launched; no navtest N6 arm built or scored)

2026-11-13, engineering agent for the REFe MM. **Integration needed from the MM: freeze the card, write the ledger entry, launch.**

## State
- Card draft: `PREREG_TF_N6_DRAFT.md` (bars 1 to 5, naming rule, both outcomes with next levers, disclosure). Card sha of the draft as copied for the guard check: `ad68a18b…` (will change when the MM edits it).
- Chain: `chain/n6_nt_{chain,build,analyze}.py`, `n6_make_pins.py`, `CHAIN_N6.md`, `launch.cmd.template` (pins sha `69bc1d73…`, 958 files, tip label 37645fcc6, cache listing digest pinned), `N6_PINS.template.json`, `replay_expect_navtest.json`.
- Tests: `chain/test_n6_chain.py`, **69 passed** (`chain/test_run_n6_chain.log`, `CUDA_VISIBLE_DEVICES=-1`); the 8 pinned L1 tests pass too. Required mutations all RED: tolerance off, NC guard dropped, human-first, DAC inverted (plus DDC guard, PDMS clause, tie rule); merge reads each npz once (with mutation arm).
- Guards on the real tree: `--check-only` PASS in 10 s (scratch ledger, scratch card copy); `--dry-run` prints 15 commands, no mutation flag, writes nothing (`chain/dryrun_check_N6.log`).
- Smoke (navtrain, 7 real instances, real build CLI): clean PASS 14/14 controls; `teacher_off` exercises the human path (7/7 DAC 1); `human_first` and `dac_inverted` go RED (`chain/smoke/SMOKE_N6.md`).

## Exact next steps (MM)
1. Review/edit the card, freeze: copy to `C:/Users/Admin/n6nt/run/n6/PREREG_TF_N6.md`, sha256 -> `<CARD_SHA256>` in `chain/launch.cmd.template`.
2. Ledger: copy `../n5_l1/ledger_K.json` to `C:/Users/Admin/n6nt/run/inputs/ledger_K.json`, append `TF-N6:<sha>:E3` n = 49.
3. If any pinned file changed: `n6/n6_make_pins.py --run-dir C:/Users/Admin/n6nt/run --tip <sha>`; new pins sha into the template.
4. `--check-only`, then launch via WMI. Est. 2.5 to 3.5 h CPU; disk: C: has ~6.5 GB (template passes `--min-free-gb 5`), or copy the tree to F: when >= 21 GB free.

## Blockers / UNVERIFIED
- The tolerance was NOT exercised on real data (the navtrain instances are easy: both repairs return a plan under the full certificate); it is exercised through the real repair loops on fake scorers. Whether it rescues any navtest token is unknown until the build runs.
- Whether the human future passes the certificate on navtest is UNVERIFIED (deliberately not looked up; navtest-seen design, disclosed in the card).
- N6-X is not scored (proved bit-identical to N5-A by C-X0 / C-UNCH, re-checked by C-REG): a design decision; say so if the MM wants it scored.
- A first-party `git` on E: refuses (dubious ownership); the TIP `families_run.py` was taken from the hold's `n3_dose/raw/instr/` (sha 44c62807 verified), not from git.

## Deletion log (rule 4)
Deletions were requested of my own scratch but the shell refused the `rm -rf` (permission denied), so NOTHING was deleted and I did not route around it. Remaining scratch I created, all re-creatable (HF `evidence/2026-11-13-tf-n5-a1/` and the pinned run_a1) or copied into the hold; **safe for the MM to delete by literal path:**
- `C:/Users/Admin/n6nt/dl` 53 MB (HF extract + pull script), `C:/Users/Admin/n6nt/tmp` 39 MB (pytest temp), `C:/Users/Admin/n6nt/smoke` 0.6 MB (copies in `chain/smoke/`), `C:/Users/Admin/n6nt/chk` 37 KB, `C:/Users/Admin/n6nt/reach_real.json`, `make_pins.log`, `test_run_n6_chain.log` (copies in the hold).
- KEEP: `C:/Users/Admin/n6nt/run` 129 MB = the run tree for the launch (the only copy of the assembled `inputs/` and the `w/` with the TIP families_run; re-creatable from `F:/tanitad_scratch/n4nt/run_a1` + HF + the hold).
Nothing under `F:/tanitad_scratch/n4nt/run_a1` or any file I did not create was modified (its `families_run.py` still sha b5ad959c).
