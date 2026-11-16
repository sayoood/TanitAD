# P5-C kit-land pod test run (CPU, pod venv-teacher, torch 2.7.1), 2026-11-16

Tree: F:/tanitad_scratch/kitland_tmp/t (t.tar sha256 272777ca...91eb58b8, verified both sides, 3483 files on both). Run: 54 scripts, 4 at a time, OMP_NUM_THREADS=3, no GPU.
Pod copy kept at /workspace/data/kitland/t (pins restored, byte-identical to the shipped file). Raw per-script lines: summary.txt.

**Totals: 46 rc=0, 8 rc=1. All 10 kit scripts PASS. No failure is pin-caused.**

Kit tokens (expected per LAND_NOTE_A31 s3): chain 126 OK, profile --fast 233 OK, profile full 261 OK, live 49 OK, postgate 32 OK, traj 166 OK, nesting 286 OK, head_wiring 182 OK, ego_yaw 160 OK, lg static **1181 (handoff expected 1180; pass, +1 test, not investigated)**.

Revert test: the 3 pin lines (444, 446, 447 of refe/launch_gate.py) set to None, the 8 failing scripts re-run (summary_nopins.txt). All 8 failed again with identical messages (logs_nopins/ vs logs_fail/), so none depends on a pin. Pins restored afterwards.

| script | rc | token | class | cause (log excerpt) |
|---|---|---|---|---|
| test_hnscreen_chain.py | 1 | none | (c) other, unexplained | unmutated suite 14 checks, 1 failing `G_control_landed_helper_alone_removes_a_stale_lock` (stale-lock helper; possibly Linux/pod lock semantics, not diagnosed) |
| test_p2r_chain.py | 1 | none | (b) env | `REQUIRED_FILE_ABSENT:SHIP_MANIFEST.json` (post-land artifact, not in tree) |
| test_rls1_chain.py | 1 | none | (b) env | `REQUIRED_FILE_ABSENT:SHIP_MANIFEST.json`, `raw/rl_pool/partition_runs/*.manifest.json` absent, `PARTITION_DRIFT:manifest_sha256` |
| test_s1aux_chain.py | 1 | none | (b) env | FileNotFoundError `raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md` (raw/ not in export) |
| test_s1z_preflight.py | 1 | none | (c) other, likely env | `AssertionError ('new','preflight.jsonl')` (preflight.jsonl differs; not diagnosed; handoff lists it as environmental) |
| test_s1zsm_preflight.py | 1 | none | (c) other, likely env | `AssertionError ('off','preflight.jsonl')`, same as above |
| test_sft4r_chain.py | 1 | none | (b) env | FileNotFoundError `raw/2026-10-04-sft4-smoke/r2_smoke2_g3mut.log` |
| test_vdac_stage0_forward.py | 1 | none | (c) other | REFUSED: `eval/s1_read.py` sha256 80a9f410... != frozen 58aa3a77... (frozen value is in test_vdac_stage0.py:488 `V1_SHA256`; tree file differs from the frozen hash; LF-normalised hash identical, so not a CRLF issue) |

Not run (per handoff): test_p7_ship_list, test_launch_gate full.

Needs MM decision: the (c) rows (hnscreen G_control, the two preflight.jsonl asserts, s1_read.py hash). None touches the pins, but they were not proven environmental here. Cheapest check: run the same 4 scripts on the dev box or the tip export with the kit absent (baseline), or read the first diverging line in each.

Pod cleanup: deleted /workspace/data/kitland/t.tar (196,556,800 B), launch_gate.pinned.bak (my backup, identical to restored file), tmp/ (0 MB). Local: deleted F:/tanitad_scratch/kitland_pod/t.tar (196,556,800 B), done1. Kept on pod: t/, run1/, run2/ logs, driver scripts (all under /workspace/data/kitland/).
