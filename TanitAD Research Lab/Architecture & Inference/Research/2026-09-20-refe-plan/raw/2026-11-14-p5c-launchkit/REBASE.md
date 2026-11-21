# REBASE.md: the P5-C wiring hold ported onto the current tip

Base of the hold: `fa67c63f`. Tip at build time: `99ffcd3a99253939b17c997e54b6bb765876bacd` (re-read at the end: tip had moved to `e3fe96ac5d53...`; every file the kit touches has the SAME blob there, MEASURED with `git rev-parse`) (`git ls-remote origin refs/heads/agent/arch-inf-20260803`; the mirror's own HEAD is a detached older commit, so every read used `git -C C:/Users/Admin/tanitad-push show 99ffcd3a:<path>`, a read-only command; the object was present). Method: `git merge-file` (3-way: hold / base / tip) per drifted file, then edits on top. All files LF. Full blob table: `blobs.txt`; remaining diff vs the tip: `diffs/<file>.diff`.

| file (`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`) | tip blob | base blob | tip drifted from base? | merge | conflicts | launchkit = tip + |
|---|---|---|---|---|---|---|
| `train.py` | `0e115c0d` | `0e115c0d` | no | hold applies as is | none | the hold's flag + (new) `traj_label_rank_policy` in the checkpoint meta and the startup line's per-rank split |
| `gate_checks.py` | `89a50121` | `b8b3884c` | **yes** (VDAC A5 rows, `suite_tail`) | `git merge-file`: clean | **0** (hold's hunks: `traj_label_not_passed` arm, DVB_KINDS entry, `traj_label_dvb_rows`; tip's hunks elsewhere) | hold + (new) rank-policy pin row, `p5c_init_rows`, `clock_bank_sha256(required=)` also for `p5c_rules` |
| `test_ego_yaw.py` | `5a61e43f` | `5a61e43f` | no | hold as is | none | one token (70 parser dests), as the hold |
| `test_launch_gate.py` | `74160da4` | `0691e902` | **yes** (the S1a test fix: `G19_/G20_/G21_` exemption and the `vdac_argv("HA-s0")` arms at `rls1_static`) | `git merge-file`: clean | **0** (the hold's one-token pin at :322 vs the S1a hunk at :1009) | hold's pin only (`diffs/test_launch_gate.py.diff` = 11 lines) |
| `test_p7_head_wiring.py` | `48e8ef5a` | `48e8ef5a` | no | hold as is | none | hold's S6 regex (`P5-C|traj_label|TLT\.|tgt_t`) and 70-dest pin |
| `traj_label_table.py` NEW | -- | -- | -- | -- | -- | hold + `--rank-policy` (required, no default), `--meta-json` (sha-bound), policy enforced at build AND in the reader `apply_to_rows` |
| `test_traj_label_table.py` NEW | -- | -- | -- | -- | -- | hold + checks `rank_policy`, `meta_json`, `dvb_policy_pin`; 9 more source mutants; fixtures declare a policy |
| `launch_gate.py` | `0775df5b` | `9d3646d1` | **yes** (taken from the tip, not from the hold: the hold never edited it) | -- | -- | the P5-C block (3 profiles, `p5c_argv`, `p5c_profile`, `p5c_refusals`) + one hook line in `profile_refusals` |
| `test_p5c_profile.py` NEW | -- | -- | -- | -- | -- | 249 checks, 14 source mutants |

| 7 pin tests: `test_div_gate`, `test_dose_gate`, `test_rep_gate`, `test_s1zsm_gate`, `test_scorer1p6_gate`, `test_vdac_gate`, `test_p7_integration` | tip blobs `c2e51ab4`, `fab9b75d`, `9445ddec`, `fe829a52`, `9c8a73b8`, `07062a7b`, `7efa505f` | = tip (undrifted) | no | one-line edits on the TIP version (profile count 26 -> 29, + the 3 P5-C names; p7_integration excludes `refe_p5c*` from "no other profile carries --ego-yaw-table") | none | `diffs/<name>.diff` |

Resolved conflict count: **0 of 2 drifted files** (16 files in `files/` in all: the 9 above + the 7 pin tests). The remaining "diff vs tip" is exactly the hold's change plus the items in the last column; nothing of the tip's newer code was overwritten (each `diffs/*.diff` is `tip -> launchkit` and contains only additions/edits of the lines named above).

## Other tip drift since `fa67c63f` that touches the P5-C inputs (checked, no action)
- `p7_ship_list.py`, `test_p7_ship_list.py`, `test_gsuite_tail.py`, `vdac_*`: not edited by the kit. `test_p7_ship_list.py` and `test_gsuite_tail.py` are in the test list (`TESTS.md`).
- The PK `raw/2026-11-11-p5c-readiness/` copy of the A1 kit is git-blob-equal to `E:/.../p5c/a1/` for `p5c_pod_label.sh` (`f9220d1f`), `p5c_dual_labels.py` (`50b9f0db`) and `p5c_table_gates.py` (`5c2f49cf`), the three blob ids A1 Point 2 freezes (MEASURED, `git rev-parse` vs `git hash-object`).

## Land guard (MM)
Land only if these tip blobs still read as above: `train.py 0e115c0d`, `gate_checks.py 89a50121`, `launch_gate.py 0775df5b`, `test_ego_yaw.py 5a61e43f`, `test_launch_gate.py 74160da4`, `test_p7_head_wiring.py 48e8ef5a`. Any other value = re-merge that file (`git merge-file <launchkit> <tip@99ffcd3a> <new tip>`).
