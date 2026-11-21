# P2 held-out table, COMPLETION run: full coverage, every gate PASS (supersedes the PARTIAL coverage numbers of run 1)

Master Mind, 2026-10-18. Run 1 (landed `00da30d`) covered 603 of 3,137 gate sets (11 of 24 logs, 333 of 1,705 tokens).
The completion run (launched 09:37 Berlin, `HANDOFF_2.md`) supplied the missing inputs and resumed the SAME driver
(`code/p2ho_driver.py`, hash-checked in `CODE_SHA256.json`). Everything below is **MEASURED (ours)**, read off the JSON
named beside it. Descriptive only: no interval is claimed here, and nothing is fitted.

## 1. Coverage (`COVERAGE.json`, `P2HO_DONE`)
| | run 1 | completion |
|---|---|---|
| tokens banked / in plan | 333 / 333 | **1,705 / 1,705** |
| sets with a row | 603 | **3,137 / 3,137** |
| logs | 11 | **24 / 24** (all with a nuPlan DB) |
| blocked tokens | 1,372 | **0** |
| `PARTIAL` | true | **false** |

Inputs added, copied from the pod while it was idle (`copy/`, sha256 equal on both sides for all 5,499 files): camera
frames of the 13 pod-only logs and the nuPlan DBs of the 11 logs missing on the dev box.

## 2. Gates (all PASS)
| gate | result | artifact |
|---|---|---|
| disjointness (held-out set, navtest, P2 training bank, checkpoint) | PASS | `disjointness_proof.json` |
| fidelity A (scoring instrument vs banked labels, 12 sets / 768 proposals) | max abs diff rounded 0.0 on every head | `fidelity.json` |
| DG-DUMP (dump vs the landed full-navtest table, 4 tokens) | proposals and logits max abs **0.0**, picks equal; negative control 15.69 m | `dgdump.json` |
| G3 / G4 (table validity, pick from logits, PDMS formula) | G4a 1,705 / 1,705 in f32 and f64; G4b max abs **0.0**; DDC-in-product control 1.0 (has teeth) | `gate_table.json` |
| pre-head features reproduce the logits | max abs 0.00145 (fp16 input), mean 0.000175; shuffled-row control 15.25 | `featcheck.json` |
| **run-1 tokens recomputed (determinism)** | **333 / 333 compared: logits, proposals, PDMS, sub-scores, PDMS at pick all max abs 0.0; picks equal 333** | `recompute_check_333.json` |

## 3. Descriptive anatomy (`anatomy.json`; P2's OWN pick, not any SR-TTC result)
Mean PDMS x100 over 1,705 tokens: P2 pick 90.64, best-of-64 oracle 98.74, random selector 75.36; P2 pick zero on 56 tokens.

## 4. SR-TTC Amendment 2, A2.10 (the dev-gate admissibility conditions)
1. `P2HO_DONE` `PARTIAL` false, 1,705 tokens / 3,137 sets: **met**.
2. Every gate of this package PASSES: **met** (section 2).
3. Run-1 tokens bit-identical: **met** (`recompute_check_333.json`, PASS).
4. Landed with its sha256 before the SR-TTC gate forward runs: **met by this commit**. The table itself
   (`bulk/table_p2_heldout.npz`, sha256 `8eacc19b…`) and the set index (`bulk/heldout_sets_index.npz`) are committed;
   `MANIFEST.sha256.json` carries the sha256 of every bulk file.

⇒ **The SR-TTC dev gate is admissible.** It has NOT been read: no SR-TTC head exists yet.

## 5. Where the bulk lives
| file | sha256 | where |
|---|---|---|
| `table_p2_heldout.npz` (13.3 MB) | `8eacc19b35b1…` | repo `bulk/`, `C:/Users/Admin/p2heldout_work/final/`, `D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/` |
| `heldout_sets_index.npz` | `a40278fbe651…` | same three |
| `proposals.npz` (34.2 MB) | `a37422102add…` | C: and D: (verified by sha256 on both, 2026-10-18) |
| `prehead_features_p2_heldout.npz` (51.4 MB) | `e37b9c554bf2…` | C: and D: (verified by sha256 on both) |

Run 1's bulk stays at `D:/Projects/TanitAD-artifacts/p2_heldout_table/` (its manifest: `prev333/MANIFEST.sha256.json`).
`PLAN.md`, `HANDOFF.md` and the root `finalize_watcher.log` describe run 1; this file supersedes their coverage numbers.
