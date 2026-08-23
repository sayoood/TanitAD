# Deliverable manifest — `parity-leak-check`, 2026-07-28

Repo HEAD `a186204`. **Staged, never pushed, never committed.** No branch switched.
🔒 Gated-confidential: every artifact carries counts, tags (`ep_XXXXX`) and sha256 digests only.
**No clip UUIDs, no route/segment names, and no raw corpus bytes are in any staged file.**

## Documents

| artifact | where it lives | only one place? |
|---|---|---|
| `PARITY_LEAK_CHECK.md` — pre-registration, method, controls, verdict, `ci.py` verdict, escalations | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-parity-leak-check/` | no (repo) |
| `MANIFEST.md` (this file) | same | no (repo) |

## Code — all staged, all reproduce a staged number

| artifact | repo path | also at |
|---|---|---|
| `fingerprint_pai_cache.py` — 7 hash families + per-frame digests; `--mode full` / `--mode poses` (mmap) | `…/2026-07-28-parity-leak-check/code/` | `tanitad-pod3:/workspace/leakcheck_fingerprint.py`, `tanitad-eval:/root/leakcheck_fingerprint.py` |
| `intersect_pai.py` — the intersection + the SELF / SPIKE / MUTANT / DUPS / PARTIAL controls | same | `tanitad-pod3:/workspace/leakcheck_intersect.py` |
| `aux_checks.py` — subset check, view fidelity, `maneuvers` entropy, `episode_id` uniqueness, near-dups | same | `tanitad-pod3:/workspace/leakcheck_aux.py` |
| `reduce_fp.py` — strips `poses_b64` / `frame_digests` so the hash evidence can be staged | same | `tanitad-pod3:/workspace/leakcheck_reduce.py` |
| `ci_equivalence.py` — imports both `ci.py` files side-by-side, asserts the md5 **actually loaded**, drives 28 cases | same | `tanitad-eval:/root/leakcheck_ci_equivalence.py` |
| `scan_degenerate_records.py` — repo-wide scan for the display-defect signature | same | dev box only (pure repo scan) |
| `leak_summarize.py` — human-readable rendering of a result JSON | same | `tanitad-pod3:/workspace/leak_summarize.py` |
| `run_fingerprints_pod3.sh` — the pod3 driver (4 caches, read-only) | same | `tanitad-pod3:/workspace/leakcheck_run.sh` |

## Raw results — one JSON per number in the document

| artifact | repo path | what it settles |
|---|---|---|
| `ANSWER_0c5f40_x_train_FULL.json` | `…/raw/` | **THE VERDICT** — 0/40, six families, frame-level containment, all controls |
| `KNOWNPOSITIVE_f1b378_x_train.json` | `…/raw/` | the known-positive control — 62/80 by poses **and** by frames |
| `SPLITWIDE_0c5f600_x_train_POSES.json` | `…/raw/` | 0/600 split-wide; the 600/600-filename and 20-`episode_id` false positives |
| `aux_checks.json` | `…/raw/` | 40 ⊂ 600; view fidelity 2376/2376; `maneuvers` entropy; `episode_id` non-uniqueness; near-dups |
| `ci_equivalence.json` | `…/raw/` | **DISPLAY-ONLY** — md5s of the loaded modules, 11/12 functions byte-identical, 28 driven cases |
| `ci_diff.txt` | `…/raw/` | the two-hunk unified diff between `ef925f06…` and `c92618a0…` |
| `degenerate_records_scan.json` | `…/raw/` | the 18 published records carrying the display defect, of 12,621 scanned |
| `pod3_fingerprint_run.log` | `…/raw/` | the fingerprinting run log — 278.8 GB in 180 s |
| `hashes_fp_val_0c5f_deployed40_full.json` (40 eps) | `…/raw/` | hash-only fingerprints — re-derive the verdict without touching the corpus |
| `hashes_fp_train_e4387_full.json` (2,376 eps) | `…/raw/` | ditto, the parity train corpus |
| `hashes_fp_val_f1b378f295ae_full.json` (80 eps) | `…/raw/` | ditto, the known-positive control |
| `hashes_fp_val_0c5f_view600_poses.json` (600 eps) | `…/raw/` | ditto, the registered split |
| `hashes_fp_train_e4387_view_poses.json` (2,376 eps) | `…/raw/` | ditto, the train poses view |

## ⚠️ Deliberately NOT staged — and why

| artifact | where it lives | why it is not in the repo |
|---|---|---|
| `fp_*.jsonl` full fingerprints (36.4 MB across 5 files) | `tanitad-pod3:/workspace/leakcheck/`, `tanitad-eval:/root/leakcheck/` | they embed **`poses_b64`** — the raw float32 pose bytes of a **gated** corpus — and per-frame digests. The staged `hashes_*.json` carry the entire evidentiary content (every hash family, per episode) with **zero corpus bytes**, and regenerate the full form in **180 s** via the staged `run_fingerprints_pod3.sh`. This is a confidentiality decision, not an oversight. |

**Nothing produced by this agent exists in only one place**, except the pod-side full fingerprints
above, which are regenerable from staged code in three minutes and are deliberately excluded.

## Test suites

| suite | result | note |
|---|---|---|
| `taniteval/` | **697 passed, 0 failed, 0 skipped** | briefed 663 — suite has grown; **no new skips** |
| `stack/` | **1 failed, 1,575 passed, 12 skipped** | ⚠️ **pre-existing, not this agent's.** `tests/test_heldout_gate.py::test_the_admitted_component_set_is_PINNED_at_the_first_probe`, raised from `taniteval/taniteval/pseudosim.py:874`, caused by a **concurrent sibling's uncommitted edit** to `pseudosim.py`. Proven by isolation: the same test file with **HEAD's** `pseudosim.py` runs **18/18 green**. This agent modified **zero** files under `stack/` or `taniteval/`. See `PARITY_LEAK_CHECK.md` §8.4. |

## Compute discipline

* **pod1 (training, ~22,350/30,000) and pod2 (small validation) were never contacted.** Not one
  command was issued to either host.
* All work ran on **`tanitad-pod3`** (idle, 96 cores, GPU 0 %) and **`tanitad-eval`** (idle).
* **Read-only on every cache.** The only writes were to `pod3:/workspace/leakcheck/`,
  `eval:/root/leakcheck/` and the repo deliverable directory. Disk verified by `dd` before writing
  (466 MB/s, 500 MB) rather than by `df`.
* **Parity untouched.** No episode was re-selected, re-registered, moved or modified. No content-
  disjoint sub-split was constructed — it was not needed, and `PARITY_LEAK_CHECK.md` §1 records in
  advance that it would not have been registrable as parity anyway.
