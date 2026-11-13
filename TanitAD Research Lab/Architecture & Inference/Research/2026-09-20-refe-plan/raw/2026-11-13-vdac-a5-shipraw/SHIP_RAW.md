# VDAC A5: HC-s0's pod G-SUITE failed on a file the ship list never listed (2026-11-13, MM)

## What happened (MEASURED, pod `/workspace/data/refe_vdac/`)
- The A5HC pause (2026-11-13 01:32 to 02:04Z) ran HC-s0 on the 06c9bd79 tree.
- HC-s0's preflight PASSED, with the cell-scaled G12 bound of 94acf07b (`ZZVDAC_PREFLIGHT_OK HC-s0`, `chain_hc-s0.log`).
- The launch gate then refused: `G-SUITE FAIL test_aux_drivable_head.py: rc 1, 'ZZTEST_PASS' NOT seen`, after **28.2 s** (`lg_HC-s0/evidence_G-SUITE.json`).
  - HA's gates ran the same script for 426 s and passed (`lg_HA-s0`, `lg_HA-SHUF`).
  - The other 6 suite scripts passed.
- No HC step trained. HC-SHUF was not run. P7 resumed (`P7_RESUMED pid 247636`).
- G-SUITE keeps no stdout of a failing script, so the pod run left only the rc.

## Diagnosis
- **Code and data are identical to the run HA passed on.**
  - The pod tree's `refe/*.py` and `*.sh` sha256 equal the dev-box export of 06c9bd79.
  - The label store is unchanged since 2026-10-12.
  - `navtest.json` is the registered LF form (`3ba1c964…`).
  - The test file, `aux_label_store.py`, `field_loss.py` and `model.py` are unchanged since HA's gate commit 16b3f34f.
- **The file set differs.** Neither A5 tree (`refe-plan-vdac`, `refe-plan-vdac.a5_5ea886fe`) has `raw/2026-10-11-aux-drivable/head_proto_selftest.out`. HA's tree (`refe-plan-vdac.914aeaaa`) and the S1 tree both have it.
- Check A1 of `test_aux_drivable_head.py` parses that file (`PROTO_OUT = os.path.join(PKG, "raw", "2026-10-11-aux-drivable", "head_proto_selftest.out")`, line 51). The unmutated suite fails, so `main()` returns 1 before the mutation loop: hence 28 s, not 426 s.
- **Reproduced on the dev box**, on the 06c9bd79 export:
  - A1 FAILS without the file (`FileNotFoundError`).
  - A1 PASSES with the git blob restored: `2,440,450 vs prototype 2,440,450`.
  - Every other non-store check passes in both runs.

## Root cause: `refe/p7_ship_list.py derive` has two gaps
1. **The static scan saw only `a / "b" / "c"` chains**, never `os.path.join(PKG, "raw", "d", "f")`.
   - It collected three bare constants instead.
   - A bare name is searched only under `refe/`, so none resolved, and the read was never listed. This is true even on a full tree.
2. **A candidate that resolves to nothing was dropped silently.** The A5 list was derived on a partial export (`refe code splits eval`), so two more files the suite names were also unlisted:
   - `raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md` (`test_s1aux_chain.py`);
   - `raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md` (`test_launch_gate.py`).
   - Both scripts tolerated their absence on the pod and passed.
- The traced mode (`--trace`) would have seen the `open()`, but it was not run: the trace needs the whole suite to pass, which needs the label store.

## The fix (this hold)
- `candidates()` also reads the constant tail of `os.path.join(...)`, `X.joinpath(...)` and `Path(X, ...)` calls.
- A candidate under `raw/` or `eval/` that names no file in the tree **REFUSES the derive**: `ZZP7SHIP_BAD unresolved=N`, and no list is written.
  - `--allow-unresolved PATH` lets one through, and the list discloses it (`unresolved_allowed`).
  - Unresolved names under the shipped roots (`refe/ code/ splits/`, which ship whole) are reported in `unresolved_in_roots` and not refused. On the A5 export, 18 of the 20 unresolved candidates were such temp-tree fixtures.
- MEASURED with the fix on the A5 export: exactly 2 candidates are refused, the two PREREG files above. Both are in git at the tip. The third file resolves once restored.

## Tests (dev box, full package export of 583d8903 + this hold, `CUDA_VISIBLE_DEVICES=-1`)
- `test_p7_ship_list.py`: pristine **25** → patched **33** PASS. The 8 new checks:
  - an `os.path.join` read and a `Path(...)` read are found;
  - a mutant without the Call branch does NOT find the join read;
  - a partial-tree derive REFUSES and writes no list;
  - `--allow-unresolved` passes and discloses;
  - a mutant without the refusal passes silently;
  - an in-root fixture is reported, not refused;
  - on the real package, the three raw files A5 lost are in the `refe_vdac_p6` list.
- `test_p7_chain.py` (reads the tool): pristine `ZZP7CHAIN_PASS 110`, patched `ZZP7CHAIN_PASS 110` (SAME; `evidence/`).
- `test_aux_drivable_head.py` on the 06c9bd79 export, without and with the restored file: `evidence/test_aux_drivable_head_06c9bd79_*.out`. The E checks fail in both runs because the label store is pod-only; that is expected off-pod.

## The re-ship that follows
- Derive the list on a FULL export of the landed tip, with the refusal live.
- Ship `refe code splits` plus the list's outside-root files (`eval/` and `raw/`).
- `check` it on the pod. The card stays in the data root (`/workspace/data/refe_vdac/PREREG_VDAC_P6_A5.md`), as POD_RUN_PLAN step B says.
- **Open instrument item:** G-SUITE (`gate_checks.check_suite`) discards a failing script's stdout. Keeping its tail would have shown A1 in the pod evidence and saved this diagnosis. Not done here: `gate_checks.py` is read by about 69 launch-gate tests.
