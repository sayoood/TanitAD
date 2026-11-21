# PREREG_P5C — Amendment A3.1 (pins only)

**Status: FROZEN by the Master Mind 2026-11-15** (sha256 in `PREREG_P5C_A3_1.sha256`).
- Drafted by a REFe agent. The MM reviewed it and re-ran every suite independently: 26 / 33 / 42 passed and `ZZP5CCHAIN_PASS 126`, on the hold copies with TMP on F: scratch.
- The MM added only this header, §4 and §5.
- **What existed at freeze:** frozen A3 (`ec14e588`, landed `854d5926`) and the bank-traj W-A7-PASS (`530fc1bf`). There was no RJ5-based label of any kind, no C1 table and no P5-C step.
- **Placement.**
  - The two `refe/` files belong to the unlanded P5-C launch kit. They land in `PK/refe/` with the kit only after P7 has trained its last step (closure rule, A2 §7).
  - Until then they are banked byte-identical under `raw/2026-11-11-p5c-readiness/a31/refe/`. A pin is a blob, so the copy and the later kit file are the same pin.

Fills the pins A3 §9 left blank, adds their deliberate-regression arms, and records their test runs. It changes no rule of A3 §1–§8. Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a31/`. Blobs are `git hash-object --no-filters`; every file has 0 CR bytes. Evidence class of every number: MEASURED (test logs in `HOLD/logs/final_*.log`).
Proposed repo homes (paths ≤ 200 chars): `PK/refe/` for the two `refe/` files; `PK/raw/2026-11-11-p5c-readiness/a31/` for the runner, W-A7 driver, the pod chain-C launcher and the new tests.

## 1 · Pins
| file (hold path) | old pin (blob) | new blob | sha256 |
|---|---|---|---|
| `refe/p5c_pod.sh` | A2 `cf823843` (sha256 `6b6e121f…`) | `11d657d93b5d380903af1b359243f2b8c31aac50` | `ca9161706bd90704a30a6d8d532f9406a6d9f111d439c0fc1336af3dac48f753` |
| `a1/p5c_pod_label.sh` (runner) | A2 `d25e1a92` (sha256 `f3b0046f…`) | `727e72d42382afad9e285dbc284d08a96822a656` | `ae8f9cb8be82a1ae89929670e3ae65ffe7ec0e0e0bee42678aed59f77a608324` |
| `w_a7/w_a7_read.py` (`--traj-source`) | `8be842d7` (sha256 `04d14afc…`) | `3b02634e39a8068b9b604f89a25bab80adaf184e` | `64daf021a8a64085799817a36b976fe7daddb2b390d02d498df61346e2c1c55e` |
| `pod_rj5c/rj5c_pod.sh` (new: chain C on the pod, A3 §7) | none | `b3619c0b86d4c851a50a0946f8a1a4804b9c95e6` | `020347978cf5e810fb7f55f11046047906a5e883e2dee9ca3fbc64a0d0d24410` |
| `pod_rj5c/control36_logs.txt` (new input of the above) | none | `be5d22939e34596592ab28c544efbe9ddd9fa4fe` | `b4e4f80fda1d45e5581428941a0713ca53e980215b0220274b96fa4bda5095ca` |
| `refe/test_p5c_chain.py` | A2 `e1b541f2` (sha256 `cb3b2c33…`) | `67fda3275f1de68b0d6d63b72e6f547ed390143b` | `73dcdc3ecc570e15254196c00225061553f804b313800a679663aaf9b2926f12` |
| `a1/test_p5c_pod_label_a31.py` (new) | none | `78dcf8ee289b9f3a590f3480b952f9110f0e3b24` | `653b47fec8bbe807081a5fa938273a36bf9cb7f0f52181066eb0d04a28a2b931` |
| `w_a7/test_w_a7_rj5.py` (new) | none | `619f74be6182322914590354c0efa1eb4845f17b` | `9bc8c701a4f04b941e523d48fd8a539fbed0ce40b33c107d3b95e04b76a200c0` |
| `pod_rj5c/test_rj5c_pod.py` (new) | none | `f102be929e61b7ed86f0add14bc69dfe1e8d8d7f` | `06867f5f71f4774e38a321dcd847a7446fcc34d8255a9f0d9e93db689e66b0b9` |

Unchanged and re-asserted: `p5c_rj5_path.py 409f4481`, `p5c_table_gates.py 5c2f49cf`, `p5c_dual_labels.py 50b9f0db`, `traj_label_table.py d65e01c4`, `test_w_a7.py f1deb689`, every A1/A2/A3 pin not above. New ship items the runner pins by sha256: `p5c_rj5_path.py` (`951d3e1d…`), `rj5_labels_local.jsonl.gz` (`23f59e5351b41d943e815e2d5847bb3ff936b7fb9fa0a50db1eb0a58f44ae686`), the RJ5 join (A3 §2).

## 2 · Test runs and RED arms
Run serially, `CUDA_VISIBLE_DEVICES=-1`, TMP on F: scratch; the dev box's navsim-crun venv for pytest, system Python for `test_p5c_chain.py`.
| suite | result | RED arms (the refusal is removed from the script text; the same scenario must go the other way) |
|---|---|---|
| `refe/test_p5c_chain.py` (`ZZP5CCHAIN_PASS 126`) | 126 ok, 0 failed | shard count from `nproc` (cap, default, and no-quota fallback: 3 arms; a stub `nproc` prints 128); `p5r-rj5` without the A3.1 sha, with a wrong one, without the join sha, with a wrong one, without declared platforms (6 arms); the 12 inherited A2 arms (its nproc and label-set targets were edited) |
| `a1/test_p5c_pod_label_a31.py` | 42 passed | 11 runner mutants: coverage bar lowered, join pin removed, control gates ignored, control pass fed the RJ5 bank, rekey check removed, class read from the ZZTB line, TB-6 rank 0 not a stop, RK0 not a stop, TB-0r not a fault, quota from nproc, A3.1 sha unchecked; 3 classifier-unit mutants (TB-5 unjudged, TB-9 not a fault, stop outranks fault) |
| `w_a7/test_w_a7_rj5.py` | 33 passed | 9 refusal removals (bank file as traj source, join sha, wrong key, status, rank, RJ5 sha, absent row, nothing moved, labels computed from the bank traj) + control (i′) always-pass mutant |
| `w_a7/test_w_a7.py` (unchanged copy, new driver) | 16 passed | bank-traj read byte-compatible |
| `pod_rj5c/test_rj5c_pod.py` | 26 passed | 13 refusals each with an arm: `--fetch`, `--fetch=`, other argument, trainer running, 6 sha mismatches (4 ship files, bank, control list), missing DB, quota under 14, non-empty scratch dir |

## 3 · Implementation facts the MM should confirm (none changes a §1–§8 rule)
1. **A3.1 sha mechanics.** The chain cannot embed A3.1's own sha. `P5C_LABEL=p5r-rj5` requires `P5C_A31_SHA256` (64 hex) equal to the sha256 of the shipped `PREREG_P5C_A3_1.md`, and `P5C_JOIN_SHA256` equal to the shipped join's.
2. **Rank policy.** The builder refuses a meta whose `rank_policy` differs from `--rank-policy rank0`. If the gates decide `all`, the runner writes the gates' decision to `rank_policy_gates_decision` and sets `rank_policy` to `rank0` in the meta copy before `provenance`.
3. **Order.** The control pass and its TB-0/TB-1 run before the RJ5 pass; a failure ends the run there.
4. **TB-7** not PASS is treated as a fault (A3 §4 does not name it).
5. **W-A7-RJ5** takes `--traj-source`, `--expect-join-sha256`, `--bank-run-result`, `--c1-table`, `--c1-records`; (i′) is the pinned gates' `_p5r_vs_control`.
6. **Chain C control list:** `control_by_log_bulk.json` has 36 entries but 35 distinct logs (one log has two entries). The launcher pins and checks 35.
7. The runner now builds only `p5r-rj5`; `P5C_LABEL=p5r|tc20s` still passes the chain's precheck but the runner refuses it at once. `a1/POD_SHIP_LIST.txt` (`763a093d`) needs the new ship items; not edited here.

## 4 · Ship addendum (MM; operational, no rule)
These go into `$P5CL_SHIP` in addition to `a1/POD_SHIP_LIST.txt` (`763a093d`). The runner pins each by sha256 (lines 98–114) or by the supplied sha.

| ship name | source | pin |
|---|---|---|
| `p5c_rj5_path.py` | `PK/raw/2026-11-11-p5c-readiness/a3/p5c_rj5_path.py` | sha256 `951d3e1d…` (blob `409f4481`) |
| `rj5_labels_local.jsonl.gz` | HF `Sayood/tanitad-refe-scorer1p6` `evidence/2026-11-15-rj5-labels-local/` (tar `c2f36453…`), member `label_local/rj5_labels_local.jsonl.gz` | sha256 `23f59e53…` |
| `rj5_bank_join.jsonl.gz` | the join **re-run with `--rows C=<pod chain-C rows>`** (A3 §2), from the banked local+A+B rows (HF `evidence/2026-11-15-rj5-rolls/`, tar `2faa9d0d…`) | its sha256, written to `LABEL_SELECTION.md` and passed as `P5C_JOIN_SHA256` BEFORE `feed` runs |
| `PREREG_P5C_A3_1.md` | this file | `P5C_A31_SHA256` = the sha256 in `PREREG_P5C_A3_1.sha256` |
| chain C on the pod | `a31/pod_rj5c/rj5c_pod.sh` + `control36_logs.txt` + A3 §7's roller files | §1 pins and A3 §7 pins |

**Window order (A3 §6 step 2, with chain C on the pod per the A3 §6 fallback).** The dev-box chain C stopped DISK-LOW on 2026-11-15: its fetch needs 12 GB projected free on C:, which held about 8 GB.
1. After P7's last step: `rj5c_pod.sh`. It refuses while any trainer runs. It rolls chain C and the pod R0 control on the 35 bulk control logs.
2. Re-run the join with C. Coverage must be ≥ 99.0 % (A3 §3.1), or there is no C1.
3. The label runner.
4. `LABEL_SELECTION.md`.
5. The launch gate.
6. C0, then (W-A7-RJ5 during C0) C1.
- Pass `P5C_NSHARD` ≤ floor(cgroup quota). The cgroup grants 15.3 CPUs; `nproc` reads 128.

## 5 · MM rulings on §3 (none changes an A3 rule)
1. **A3.1 sha mechanics:** accepted. The chain checks the supplied sha against the shipped file.
2. **Rank policy:** accepted. It is A3 §3.3 as written: forced `rank0`, with the gates' decision recorded and STOP still able to fire.
3. **Control pass first, abort on TB-0/TB-1:** accepted. It is A3 §4's fault branch and A3 §6's "found during C0 at no GPU cost" reasoning.
4. **TB-7 not PASS = fault:** accepted as INHERITED, not new. A2 §1 selects the label only on `ZZTB_ALL_PASS` or the R-D pattern, and TB-7 is in that set. A3 §4's fault list restates A2's gates "unchanged", and it does not exempt TB-7.
5. **W-A7-RJ5 inputs:** accepted.
6. **35 distinct control logs:** accepted.
   - `control_by_log_bulk.json` has 36 entries over 35 distinct bulk logs.
   - The MM re-tabulated the same rows (`raw/2026-11-15-rj5-label-path/mm_provenance/pod_probes_output_2026-11-15.txt`): **27 of the 35 bulk logs diverge and 8 are exact**. The one diverged local log is the CUDA-built `None` class, with 26 traj-different rows and 0 origin differences.
   - ⇒ **Correction to GAC `D-REFE-RJ5-PROV-1`:** its "28 of 36 bulk control logs" and "28 diverged logs, 721 of 2,452 rows" count that local log in. The bulk-only figures are 27 / 35 logs and 721 of 2,426 traj-different rows. The conclusion (a 1-ULP floor, not a bias) is unchanged.
7. **Runner builds only `p5r-rj5`:** accepted. C0 needs no table, and A3 forbids swapping C1 to the bank-traj label without a PI ruling.
