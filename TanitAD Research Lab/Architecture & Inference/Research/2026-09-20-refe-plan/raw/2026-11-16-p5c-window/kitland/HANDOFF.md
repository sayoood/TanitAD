# HANDOFF: P5-C kit land tree (REFe agent, 2026-11-16 ~17:25 local). STATE: tree + LAND_LIST done; serial test run IN FLIGHT (commit guard waiting); NOT landed.
Tip used: `1c48ee8fc55adcf12bdc0eb1e4c9ea0b491c7b98` (ls-remote == fetch == this, 2026-11-16). LAND GUARD OK: 13 guarded tip blobs read as REBASE.md (train 0e115c0d, gate_checks 89a50121, launch_gate 0775df5b, test_ego_yaw 5a61e43f, test_launch_gate 74160da4, test_p7_head_wiring 48e8ef5a, 7 pin tests c2e51ab4 fab9b75d 9445ddec fe829a52 9c8a73b8 07062a7b 7efa505f); 9 new files absent.

## Built (MEASURED)
- Tree F:/tanitad_scratch/kitland_tmp/t (refe code eval report splits stack tools taniteval @tip, `archive` with autocrlf=false) + kit files/ (22) + A3.1 p5c_pod.sh (11d657d9) and test_p5c_chain.py (67fda327) + 3 pins in launch_gate.py. Full PK copy for the ship-list derive: F:/.../kitland_tmp/full.
- launch_gate.py kit blob 86172cc3 -> pinned e87c5598. Diff = exactly 3 lines (444, 446, 447), see launch_gate.pins.diff: P5C_BASE_SHA256=2613fc17...fd3fe, P5C_LABEL_TABLE_SHA256=3b198314...637a2, P5C_RANK_POLICY="rank0". 0 CR.
- LAND_LIST.json: 22 items [repo_path, abs_src, "raw", base=<tip blob>|<absent>]; sources in land/refe/ (LF, 0 CR, paths <= 200). If a test assertion needs an edit, edit the copy in land/refe/ and re-run F:/tanitad_scratch/kitland_tmp/mk_landlist.py logic (blob shown in table).
- Env: PYTHONPATH set (`;` form via cygpath) to the tree's stack; `import tanitad` asserted -> F:\...\t\stack\tanitad (the venv editable install points at G:, avoided). CPU only, TMP per script, deleted after.

## In flight (detached bash, not model turns)
- drive.sh (54 scripts: the 10 kitchk + the 44 other refe tests that mention launch_gate; skipped: test_p7_ship_list, test_launch_gate full = environmental on kit round, listed in TESTS.md). Writes F:/tanitad_scratch/kitland_tmp/res/summary.txt (`name args rc secs commit tok`), logs res/<tag>.log, ends with ZZKL_END. Needs free commit >= 5 GB per script; box read 2.6-3.3 GB at 17:25 (P7 eval). It ABORTS after 60 min of waiting (line ABORT_COMMIT_LOW_60MIN) -> then report per brief.
- post.sh: after ZZKL_END runs `p7_ship_list.py derive --profile refe_p5c` (STATIC, no --trace) on full/ -> p5c_ship_list.json in the hold, res/derive.log, line `derive rc=`.

## Next steps for the successor
1. `grep -av ^wait .../res/summary.txt`. Expect rc 0 and tokens as LAND_NOTE_A31 section 3 (chain 126, profile --fast 233 / full 261, live 49, postgate 32, traj 166, lg static 1180, nesting 286, head_wiring 182, ego_yaw 160).
2. For each rc!=0: read res/<tag>.log tail; if it is a pin/fingerprint assertion, fix in a COPY in land/refe/ and log old->new; if environmental (dev data absent, raw/ absent from export, preflight.jsonl, carve_feed identity) compare with TESTS.md; else report unfixed. ENOSPC check: C: had 4.5 GB, E: 8.3 GB, F: 46 GB free at start.
3. SHIP_MANIFEST.json CANNOT be derived before landing: `launch_gate.py ship-manifest --pkg <pkg> --commit <landed 40-hex>` needs the landed commit (RUNBOOK step 2 is a post-land step). Only the ship list is derivable now (p5c_ship_list.json); it is a trace-less static list: RUNBOOK uses plain derive; --trace (~40 min suite) not run.
4. Cleanup (own scratch, literal paths, log in report): F:/tanitad_scratch/kitland_tmp/{t,full,tmp,res logs after copying summary} when done; keep nothing else.

## Deletions so far
launch_gate.kit_before_pins.py (my copy of the kit file, ~170 KB, re-creatable from the kit hold) removed from the hold. Nothing else.
