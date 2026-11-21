# LABEL_SELECTION (P5-C, C1): label `p5r-rj5`, rank policy `rank0`, table sha256 `3b198314…`. MM, 2026-11-16

Evidence class: **MEASURED**. Selection rule: `PREREG_P5C_A3.md` section 4, "All PASS ⇒ the MM writes `LABEL_SELECTION.md` … and fills `P5C_LABEL_TABLE_SHA256`". The gates run under A3.1, A3.2, and **A3.3, which is in force** (`PREREG_P5C_A3_3.md`, landed 2582b187).

## The selection
| pin (`refe/launch_gate.py`) | value |
|---|---|
| `P5C_LABEL_TABLE_SHA256` | `3b198314721fc7e75e6bd0ca9b569b5b0a56ba25178e4fb63a94b215995637a2` |
| `P5C_RANK_POLICY` | `"rank0"`, which is the table's `meta.rank_policy`; A3 section 3.3 forces it |
| label | `p5r-rj5` (RJ5 traj → Tc20s → P5-R) |

**Table:** `P5C_label_table_p5r-rj5_rank0.json.gz`, 94,502,145 B.
- **Pod copy:** `/workspace/data/p5c_label_a33/`.
- **HF (private):** `Sayood/tanitad-refe-scorer1p6`, folder `evidence/2026-11-16-p5c-label-table-a33/`. The LFS sha256 equals the value above (`ZZLABA33_BANK_VERIFIED`).
- **Same HF folder:** `C1_provenance.jsonl.gz` (sha256 `cc6e9bf0…`, 11,588,029 B) and `label_a33_small.tar` (`bd470434…`, the files in `raw/` here).

**Inputs:**
- join `rj5_bank_join.jsonl.gz` sha256 `4670f88ac409e71a1f7e1bf02328d93229fb3e153ad63f4127f08d53bbfed739` (coverage 1.0, 92,896 / 92,896);
- feed map `38a94869962da5c4de9d21b97399c3981bdcb21b486aa6b78f8f5263115d8fbc`;
- A3.1 sha `cda1567b…`;
- code: runner `3770cbe9`, gates `718290ce`, rj5 path `117434b3` (A3.3 pins).

## The run (the A3.3 reuse run)
- **Work dir:** `/workspace/data/p5c_label_a33`, a sha-verified copy of run 3's work dir. Log: `raw/p5c_label_a33.log`.
- **Reuse:** the labelling passes were not re-run.
  - All 30 label shards (15 control, 15 RJ5) were verified byte-identical to run 3's (`SHARDS_run3.sha256`, `614562c2…`), and the feed map was rebuilt equal to run 3's.
  - Every post-shard step ran again: rekey, fallback, the rank-0/1 repair runs, both gate passes, TB-0r, provenance, table build and verify. Pod clock 12:53–13:02Z.
- **Class:** `pass`. `class.json`: fault [], stop [], `rank_decision` "all"; the table is forced to rank0 under A3 section 3.3.

### Gates
| gate | set | result | value (n) |
|---|---|---|---|
| TB-0 | control | **PASS** | R-B max abs diff 0.0567 (2,400). R-A (Tc20s warm start vs the dev-box build) max **1.490e-9 ≤ 1e-8**, all ranks (2,400) |
| TB-1 | control | PASS | 2,400 / 2,400 at tol 1e-9 (unchanged) |
| TB-2 | main | PASS | 0.0 (150,603). The 1 fallback row is judged by its Tc20s knot deviation, 2.8e-17 ≤ 1e-9 |
| TB-3 / TB-4 | main | PASS | 150,603 / 150,603 |
| TB-5 | main | **PASS** | 150,603 / 150,603, label_fallback n 1, violations [] |
| TB-6 | main | PASS | rank 0 (decides): label/traj ratio of median max \|acc\| **1.323**, of median max \|heading rate\| **1.705**, bar ≤ 2 (92,896 rows). Rank 1 (reported only): 1.137 / 1.366 (57,707 rows) |
| TB-7 | main | PASS | rank 0: 92,896, rank 1: 57,707 |
| TB-0 / TB-1 | main | NOT_RUN | judged on the control pass: an RJ5 label differs from the bank-traj EVAL labels by construction (as in A3) |
| TB-0r | main | **PASS** | R-A max 2.548e-9 ≤ 1e-8. R-B ok: 16,133 common rows, every criterion |
| TB-9 | main + table | PASS | sidecar keys 150,603 = table keys 150,603 |
| RK0 | main | no-stop | decision `all`; the table is rank0 by A3 section 3.3 |
| LABEL-FALLBACK | main | OK | n 1, unlisted S2 candidates 0, reason `tb5_s2_tc20s` |

**Table verify (`table_verify.json`):**
- rows 150,603; covered 150,603 (coverage 1.0);
- differs from raw: 92,896, all rank 0;
- rank-1 rows stay raw under `rank0`.

## Disclosures that travel with every C1 number
1. **Both A3.3 rulings are POST-DATA** (PI, 2026-11-16).
   - R-A was raised from 1e-9 to 1e-8 for TB-0 and TB-0r. The measured cross-platform Tc20s floor is 2.55e-9.
   - The one TB-5 row takes its own Tc20s label, because P5-R flipped its tail heading by 3.58 rad on a near-standstill RJ5 traj.
   - R-B, the gate on the labels C1 trains on, is unchanged and passes in full.
2. **C1's rank-0 labels are 92,895 P5-R plus 1 Tc20s.**
   - The Tc20s row: token `d06ff3cbd9025da7`, RJ5 traj sha256 `926ffc66…`.
   - It is recorded per row in `C1_provenance.jsonl.gz` as `fallback_reason=tb5_s2_tc20s`.
3. **Pod R0 control = outcome B** (`D-REFE-P5C-JOINC-1`). The bank-builder platform differs from both the pod and the dev box. Disclosed per log (TB-8); it does not block C1.
4. **Two other runs are banked:**
   - Run 1 (`p5c_label`) and run 2 (`p5c_label_hsw`) failed TB-0 at 1e-9: `D-REFE-P5C-TB0-1`.
   - Run 3 (`p5c_label_a32`) failed on TB-5 and TB-0r: `D-REFE-P5C-RUN3-1`.

## What this unblocks, and what it does not
- **Unblocked:** C1's label pin. `P5C_RANK_POLICY` and `P5C_LABEL_TABLE_SHA256` are filled at the kit land.
- **Still blocked:** the base pin (`P5C_BASE_SHA256`). It waits on the P7 navtest read's verdict under SPEC_P7 section 5:
  - ADOPT or NOT PROVEN ⇒ B-P7 `2613fc17…`;
  - REFUTED or INVALID ⇒ B-P6.
- **Order (PI ruling):** C0 runs first, without a table. W-A7-RJ5 runs during C0. C1 then runs on this table. C0r runs only if C1 − C0 separates beyond B = 0.7031.
