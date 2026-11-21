# P5-C launch kit: banked raw copy, NOT the code land (MM, 2026-11-14)

**Why this is here.** The kit was built by two agent rounds (CPU only, no git writes, no pod, no GPU). Until now it lived only on one disk, in the hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/`. This folder banks it, so it no longer exists on a single disk.

⛔ **This is NOT the code land.** Nothing under `refe/` changes in this commit. The 22 kit files land as code only **after P7's last arm trains**, because of the closure-land rule: a land that touches `refe/` between P7's bind and its end would void P7's bound records. The land order is in `HANDOFF_2.md`.

## Contents
- **`p5c_launchkit.tar.xz`**
  - Size: 362,700 bytes.
  - sha256: `71888abf7cd73d0c5b4172f03b3e13e2324b710277c438dd9190232fbfd021b7`.
  - Contents: 136 files, 2,059,424 bytes before compression. That is the whole hold:
    - `files/` holds the 22 files to land at their repo paths;
    - `runner/` holds `p5c_pod_label.sh`;
    - `diffs/` holds the diff from tip to kit for each file;
    - `logs/` holds test logs, digests and the agents' scripts.
  - Deterministic tar: names sorted, mtime 2026-11-14 00:00Z, owner 0.
  - Before banking, the MM scanned the whole hold for tokens, IPs, pod addresses and ssh details: 0 hits.
- **Plain copies of the docs**, so they can be read without extracting:
  - `HANDOFF.md`: round 1 (rebase, `--rank-policy`, profiles, runner patch);
  - `HANDOFF_2.md`: round 2 (G-LIVE (f) and (h), G-ADAM post-gate, the chain `p5c_pod.sh`, the final land list);
  - `REBASE.md`, `PROFILE.md`, `RUNBOOK_P5C.md`, `TESTS.md`, `blobs.txt`.

## State (MEASURED by the agents; the MM re-checked the guard blobs)
- **Tests:**
  - `test_p5c_live.py` 71, `test_p5c_postgate.py` 56, `test_p5c_chain.py` 83; 35 mutants RED;
  - the ~32 other scripts that read launch_gate, plus p6_chain, p6_levers and gate_suite_prerun: all green, or identical to the tip;
  - **REGRESSIONS 0** (`TESTS.md`).
- **Guard blobs** (MM, at tip `4e21490f`, 2026-11-14): `train.py 0e115c0d`, `gate_checks.py 89a50121`, `launch_gate.py 0775df5b`, `test_ego_yaw.py 5a61e43f`, `test_launch_gate.py 74160da4`, `test_p7_head_wiring.py 48e8ef5a`.
  - All are unchanged since the kit's build tip `1da03de9`.
  - The 9 other gate tests the kit edits are unchanged too.
- **Profiles:** `refe_p5c` (C1), `refe_p5c_c0` (C0) and `refe_p5c_c0r` (C0r) REFUSE until the 3 pins are filled: `P5C_BASE_SHA256`, `P5C_LABEL_TABLE_SHA256` and `P5C_RANK_POLICY`.

## Open before launch (each one goes into Amendment A2 or the land)
1. **Amendment A2 text.** It covers:
   - the PI's 2026-11-12 "P5-R regardless of R4" against A1 section 0;
   - the moved pins (the runner blob and `traj_label_table.py`);
   - G-RESERVE dropped;
   - no `--scorer-mode`;
   - `--resume`;
   - G-ADAM checking both moments;
   - the G-LIVE (f) tensor digest replacing the ≥ 64 rows;
   - `init_from_sha256` living in `config.json`;
   - `P5C_LABEL`.
2. **Gate evidence.** G-LIVE-LAB is reused from P7 only if it was measured at the same workers and chunks; G-LIVE-MEM and G-EGR are re-measured.
3. **C0r** runs only with PI approval (`--with-c0r`).
4. **W-A7** launch blocker, and the **W-TRANSP / W-FAM** read blockers.
