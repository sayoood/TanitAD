# HANDOFF P (dominance read re-pinned with rule R-SCN) -- 2026-11-06. DONE: items 1 to 6. IN FLIGHT: item 7 (the close-out chain).

MEASURED = ours, CPU. Nothing committed, staged, deleted, launched on a box; no ssh, no download; no real pull opened (stage2/pull_box2_run2*, stage3/pull_box2, stage4/pull_box2); `motion_c/t2_prereg/` untouched.
Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, `CUDA_VISIBLE_DEVICES=-1`; pytest scratch on F:. E: has about 1.2 GB free.

## NEW PIN: `code/read_t2_dominance.py` sha256 `d66647180a040318fc48b65bbdd068b637cf5d5a3e8b2ca0f91c4cee6bf1280a` (line 1 of `code/read_t2_dominance.py.sha256.txt`; the MM appends it to the card log)
Previous pins: `df4e8636...` (L), `6345d4c7...` (M); a copy of the 6345d4c7 file is on `F:/TanitAD-scratch/t2v8f_P/read_t2_dominance.pin6345d4c7.py`. `scene_exclusions.py` (83a5f647...) and `read_stage2.py` (f2136b39...) are UNCHANGED.
The one pin edited in a test: `FROZEN["code/read_t2_dominance.py"]` in `tests/test_t2v8f_stage4.py` line 58 (prose in its `check_contract` docstring still names 6345d4c7).

## What the read does with `--exclusions2/3/4` (all three or none; none = the previous pin, no `r_scn` key)
* per stage (`rscn_stage_check`): file ATTRIBUTED, no problems; its `stage` label; `manifest_sha256` equal AND `inputs.result_sha256` == sha256 of the result file on hand (the manifest sha is the stage-1 package's and is the SAME in all
  three stages, so it alone cannot tell stage 2's file from stage 3's); failed cells == exactly all cells of the excluded scenes, deciding AND beside (stages 2/3: 10 + 4; stage 4: 2 + 2); on disk each cell of an excluded scene is FAIL
  with 0 records and none DONE; every cell of an included scene DONE (result counts and directories); no pass or scene skipped; `n_cells_planned`; `overall` COMPLETE, or INCOMPLETE only with exclusions.
* union of the excluded scenes over stages 2, 3, 4 removed from BOTH C0C and P10 before `pair_check`; expected starts fall by the scenes' starts (read from the cells' `records=a/N`); post-condition `RSCN_PAIRS_ON_EXCLUDED`.
* G2(e): excluded scene -> `EXCLUDED (R-SCN): <reason>` (never a failure); included scenes still need `actor_attach.json`. G2(d) unchanged.
* `DOMINANCE_T2.json` key `r_scn` and a section "Excluded scenes (rule R-SCN)" in the .md: per stage the scene, reason, failed cells (deciding + beside of total), starts lost; the union; scenes and starts read on.

## Command for the real read (stage-4 builder input: concatenate the two sources lists, its CLI takes ONE `--sources`)
```
python -I -c "import json,sys; a=json.load(open(sys.argv[1])); a+=json.load(open(sys.argv[2])); json.dump(a,open(sys.argv[3],'w'),indent=1)" stage2/SCENE_SOURCES.json stage3/SCENE_SOURCES3.json stage4/SCENE_SOURCES4.json
python -I code/scene_exclusions.py --result <STAGE4_RESULT.json> --pull <pull4> --log deciding=<rollouts_deciding.log> --log beside=<rollouts_beside.log> --sources stage4/SCENE_SOURCES4.json --out stage4/EXCLUSIONS4.json
# stage 3 likewise with SCENE_SOURCES3.json -> stage3/EXCLUSIONS3.json; stage 2 is stage2/EXCLUSIONS2.json (built 19:31; its schema was checked, its result_sha256 is e8ccde22...)
python -I code/read_t2_dominance.py <the STAGE4_LAUNCH.md section-2 arguments> --exclusions2 stage2/EXCLUSIONS2.json --exclusions3 stage3/EXCLUSIONS3.json --exclusions4 stage4/EXCLUSIONS4.json
```
⚠ Each EXCLUSIONS file must be built from the SAME result file passed as `--resultN` (the read compares sha256s); rebuild it if the result changed. With launch sources only, the builder writes `idx: null` for a NO_DYNAMIC_LAYER scene and the read refuses (`RSCN_IDX_MISMATCH`).
⚠ `STAGE4_LAUNCH.md` section 2 still quotes pin 6345d4c7 and lacks the three flags: the MM updates it.

## Evidence
* `tests/test_t2v8f_dominance_rscn.py`: 23 tests green (synthetic 6-scene world with the drivers' stage layouts and the REAL builder's exclusions; scene 1 refused in stages 2+4, scene 5 in 3+4, scene 2 in stage 4 ONLY). Covers the six RED arms of the brief
  with named problems, the builder on stage-3 and stage-4 layouts (item 2), G2(e), the disclosure, and `main()` end to end (flags -> union -> scene loader, G2(e), JSON, md). 57 tests green together with the 18 dominance + 16 scene_exclusions tests.
* Mutation run `code/run_mutations_dom_p.py` -> `logs/mutations_dom_p.log`: **40 of 40 guards/wires, each deleted in memory, turn their designated test RED** (baseline 23 of 23 green). A bug the wiring test found: a variable `rs` shadowed in `write_md`.
* `tests/test_t2v8f_stage4.py` artifact/order/pairing/manifest tests green with the new pin (4 passed).

## IN FLIGHT: `audit/finish_p.sh` (detached, WMI): full suite -> `logs/pytest_full_P.log` (expect 151 passed, ~15 min), mutation rerun -> `logs/mutations_dom_p.log`, `audit/make_pins_p.py` -> `PINS_P.txt`; `logs/P_FINISHED.txt` ends `ZZ P-CHAIN-DONE ZZ`.
NEXT (fresh agent or the MM): read `logs/P_FINISHED.txt`; if the suite shows a failure, the failing test name is in `logs/pytest_full_P.log`. G2(b) still cites `logs/pytest_full_M.log` (128 passed, valid: the gate test files are unchanged); `pytest_full_P.log` is the newer green run.
