#!/usr/bin/env python3
"""build_docs.py -- write SOURCES.md, CONFLICTS.md, ABSTRACT_PROPOSAL.md and paths.json for Paper v1.5.

Inputs (same directory): claims_table.md, claims_values.json, files_read.json, TANITAD_PAPER.md, inserted_text.txt
Run:  python -I build_docs.py     (after build_sources.py, assemble.py, verify_paper_numbers.py)
"""
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")


def rd(name):
    return open(os.path.join(HERE, name), encoding="utf-8", newline="").read()


def wr(name, s):
    with open(os.path.join(HERE, name), "wb") as f:
        f.write(s.encode("utf-8"))


paper = open(os.path.join(HERE, "TANITAD_PAPER.md"), "rb").read()
NL_PAPER = paper.count(bytes([10]))
base = open(os.path.join(HERE, "base_TANITAD_PAPER.md"), "rb").read()
NL_BASE = base.count(bytes([10]))
paper_blob = subprocess.run(["git", "hash-object", "--stdin"], input=paper, capture_output=True).stdout.decode().strip()
base_blob = subprocess.run(["git", "hash-object", "--stdin"], input=base, capture_output=True).stdout.decode().strip()
tip = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "rev-parse", "5fb509a"], capture_output=True, text=True).stdout.strip()
tip_blob = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "rev-parse", "5fb509a:Paper/TANITAD_PAPER.md"], capture_output=True, text=True).stdout.strip()
assert base_blob == tip_blob == "188d00e3c5480f71e6aeed0ef84d5f1489c38ecc", (base_blob, tip_blob)
assert b"\r" not in paper and b"\r" not in base

fr = json.load(open(os.path.join(HERE, "files_read.json"), encoding="utf-8"))
claims = json.load(open(os.path.join(HERE, "claims_values.json"), encoding="utf-8"))
table = rd("claims_table.md").rstrip("\n")
n_claims = len(claims)
n_rows = table.count("\n") - 1 + 1  # header + separator removed below
rows = [l for l in table.split("\n") if l.startswith("| ") and not l.startswith("| id ") and not l.startswith("|---")]

N_CRDIFF = sum(1 for r in fr["claim_files"] if not r.get("local_copy_identical_to_tip", False))
N_SAME = len(fr["claim_files"]) - N_CRDIFF
assert all(r["local_copy_equal_modulo_CR"] for r in fr["claim_files"])
RPT = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"

# --------------------------------------------------------------------------------------------------------------------
S = []
S.append(f"""# SOURCES: Paper v1.5 (`Paper/TANITAD_PAPER.md`), new §7.21 and its pointers: claim → JSON file → key → value

Drafted 2026-10-18 by a TanitAD paper agent from the Master Mind brief of the same date. **Never committed, never pushed**: the
files are staged under `C:/Users/Admin/paper15/` for the Master Mind to land (`paths.json` lists the targets).

## 0. Provenance

| item | value |
|---|---|
| base file | `Paper/TANITAD_PAPER.md` at the mirror tip **`{tip}`** (`5fb509a`) |
| base blob (`git rev-parse 5fb509a:Paper/TANITAD_PAPER.md`) | **`{tip_blob}`**, {len(base):,} bytes, {NL_BASE:,} lines, **0 CR (LF-only)** |
| v1.5 file (`TANITAD_PAPER.md`) | blob `{paper_blob}`, {len(paper):,} bytes, {NL_PAPER:,} lines, **0 CR**: line endings are the tip blob's |
| diff against the base | exactly six hunks, all insertions or one-line replacements: status line (`v1.4 (2026-10-17)` → `v1.5 (2026-10-18)`), version-note sentence, a dated pointer at the end of §7.20(g), the new **§7.21** before `## 8. Discussion`, one References line, one changelog entry. Nothing else moves. |
| section choice | **new §7.21**, not an extension of §7.20: §7.20 is the dated v1.4 record of one reading (88.25, the zero tail, P6); the four packages here are a second, later record with a different tier mix, and §7.20(g) is annotated rather than rewritten |
| abstract | **unchanged**: no number in it moves (88.25, 94.55, 65.9 %, 529, 86.0 %, 6.30 all stand); the proposed addition is `ABSTRACT_PROPOSAL.md` |
| files read | every package file below was read from the mirror tip (`git -C C:/Users/Admin/tanitad-push show 5fb509a:<path>`), never from E: (it lags) |
| number check | `verify_paper_numbers.py`: **207 required claims, 0 missing, 0 orphan numeric tokens** (every numeric token in the inserted text is a component of a claim below or is on the whitelist of non-measurements: section labels, dates, registered thresholds) |

**How to reproduce.** `python -I build_sources.py` (reads each raw JSON and writes `claims_table.md`, `claims_values.json`),
`python -I assemble.py` (anchored edits on the tip blob), `python -I verify_paper_numbers.py`, `python -I build_docs.py`.
Every number in §7.21 was read from the raw JSON/JSONL/log named below, **not from a RESULT.md**; the RESULT.md files were read
afterwards, only to find prose/JSON disagreements (`CONFLICTS.md`).

## 1. Tiers, estimators and evidence classes used in §7.21

| part | tier | estimator | what the interval answers |
|---|---|---|---|
| (a) headroom, (b) free points, (d) SELRULE-DAC | **T1** NAVSIM v1.1 PDMS (non-reactive pseudo-simulation; **not** closed-loop), full navtest 12,146 tokens / 136 logs | paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (`proxy_eval.boot`) | *another draw of logs* only; one training seed per state; selection deterministic |
| (b) family block | T1 `families6` | **single-arm**, unpaired, log-cluster bootstrap, **2,000** resamples | the instrument's own family intervals; not the paired rows |
| (c) R0d, SO-CARVE A1, A2 val side | **RL-val selection tier** (3,000 held-out sets / 75 logs): true sub-scores of the argmax pick; "never an endpoint" (`eval/rl_val_eval.py` line 24); **not T1** | same paired log-cluster bootstrap, 10,000, seed 20260927 | *another draw of logs* only; one seed, no replicate arm |
| (c) SO-CARVE main read | ESTIMATED harness (banked R0/C0 states, cached contexts, CPU; 302 train sets / 30 logs, 62 val sets / 13 logs) | same | same |
| H-ESTIM-SEED-1 floor | T1 | P2 seed 1 minus seed 0 = +0.33 [+0.04, +0.64], carried from §7.20 | *another training run*, n = 1 pair |

Evidence classes in the table: **MEASURED** (ours; the JSON path is the artifact), **ESTIMATED** (harness or arithmetic on a decomposition),
**POST-HOC** (computed after the pre-registered read, flagged where printed), **DERIVED** (arithmetic on two cited claims), **PUBLISHED**
(a banked primary, cited by library key), **INHERITED** (carried from existing paper text or the register, NOT re-read; section 3).

## 2. Files read (all under `{RPT}/raw/`, mirror tip `5fb509a`)

Two hashes per file. `git blob` is the object id on the tip; `sha256` is of the tip's bytes as committed. {N_CRDIFF} of these files were first
extracted to a local working copy with CRLF line endings (the other {N_SAME} were extracted byte-identical); the {N_CRDIFF} are byte-identical to the tip after stripping CR
(`files_read.json`, `local_copy_equal_modulo_CR`), and their sha256 prefixes agree with those quoted in the R0d package's own RESULT.md §5
(`kill_checks.jsonl` `31865f51`, `sft.jsonl` `5380d8d1`, `rlval_B_vs_ownA.json` `484aeb5d`, `safeobj_outcome.json` `c0ce2395`, `KILLED.json` `6183c3e7`).

| file (under `RP/raw/`) | git blob | sha256 (first 16) |
|---|---|---|""")
for r in fr["claim_files"]:
    S.append(f"| `{r['file']}` | `{r['git_blob_on_tip'][:12]}` | `{r['sha256_tip_bytes'][:16]}` |")
S.append("")
S.append("Full 64-hex digests: `files_read.json` (mirrored beside this file).")
S.append("")
S.append(f"""## 3. The claim table: {len(rows)} rows (each a number or a count printed in §7.21)

Columns: `id` (the id the verifier uses), the claim, the JSON file under `RP/raw/`, the key path inside it (`a/b/c`; `{{point, lo, hi}}` means
the three leaves are read as a point and a 95 % interval), the value as the JSON holds it **at full precision**, the string as printed in
§7.21, the evidence class, and the tier / estimator. A printed string is only ever the JSON value rounded to the stated decimals; the
rounding was chosen per row so that no separation call changes (a lower bound of 0.0002 is printed as +0.0002, not +0.00).

{table}
""")
S.append("""## 4. Numbers carried from existing paper text or the register, NOT re-read (INHERITED)

| number | where §7.21 uses it | its source |
|---|---|---|
| +0.33 [+0.04, +0.64] | the only T1 seed floor (P2 seed 1 − seed 0) | base paper line 3131 (§7.20), H-ESTIM-SEED-1 |
| 88.25 (u400, `88.2469` re-measured as the banked mean in claim b08) | the deployed scorer's level in the opening and in the gate paragraph of (b) | base paper line 3156 (§7.20); b08 is our own re-read of the banked mean from `freepoint_result.json` |
| 94.55 | the matched published row (PUBLISHED, arXiv:2609.06055 Table A13, §7.20); also the logged human under our harness (MEASURED, §7.20) | base paper lines 3160-3165; see CONFLICTS C8 on how the package labels it |
| "19" | median non-zero proposals over the SEL zero tokens (POST-HOC, §7.20(c)); cited only to say it is a different population from the all-token median 58 | base paper line 3206 |
| "1.4 to 4.5 times" | the register's ratio, quoted only to be declined; the ratios reproduce from JSON (1.450, 2.757, 4.486) but their denominators' intervals span 0 | `so-carve/RESULT.md` §0 and §8; register row D-REFE-SAFEOBJ-CARVE; CONFLICTS C5 |
| 3.54 [2.90, 4.23], SEL 445 / PROP 64 | §7.20(c)'s oracle-on-SEL ceiling, reproduced key for key from the full table (the new claims a23, a24, a25 are MEASURED from `P2_MEASURED_from_full.json`; the §7.20 values are carried only for the comparison sentence) | base paper §7.20(c); `p2-full-table/zero_subset_vs_landed.json` |
| "the rule that withdrew a ratio in §7.19b" | method precedent | base paper lines 178-180 |

## 5. Registered definitions quoted from prose (PREREG / SPEC / instrument text; not measurements)

Read from the tip; line numbers are of the tip blob.

| statement in §7.21 | source | sha256 (first 16) of the tip bytes |
|---|---|---|""")
dmap = {
    "eval/PREREG_RL_S1.md": "EPO: π = softmax(log(A+ε)/T), T = 0.1, β = 0.1 KL anchor, BCE on the six sub-scores unchanged (§1, lines 55-61); arms: A = BCE only, B = BCE + EPO, paired in one pass (line 95); C0/C1 = SFT-continued control, BCE only, identical sets / order / steps; R0/R1 = BCE + EPO (§3, lines 102-103)",
    "eval/rl_val_eval.py": "the RL-val tier text, line 24: hyper-parameter / early-stop use only, never an endpoint; the g5 bound (:52, :306 as cited by PREREG_SAFEOBJ line 113)",
    "raw/2026-10-17-safety-objective/SPEC_SAFEOBJ.md": "SAFEOBJ (§1): g = c_A − c_B with c = E_π[y] at T = a.epo_temp (0.1), arm A detached; registered ε = 0.0, η (dual step) = 4.0, λ_max = 10.0, λ0 = 0.0 (lines 43-49)",
    "raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md": "bar S / g5: TTC and NC may not fall by > 0.5 versus the control (line 113); K3 kills on two consecutive breaches, a single non-consecutive breach fails S without killing (line 168); outcome O4 = breached, dual held in-sample (in-sample violation ≤ 0.001 and λ > 0 on ≥ 50 % of post-ramp updates; line 194); ε 0.0, η 4.0, λ_max 10.0 rationale (lines 116-118)",
    "raw/2026-10-16-selrule-dac/PREREG_SELRULE_DAC.amended.md": "V(τ): argmax of the aggregate over proposals with p̂_DAC ≥ τ, fall back to the unmodified rule if none qualifies; τ grid {0.5 … 0.9} (line 31); the fit is on RL-val with u400 and the transfer to P2 is an assumption (Amendment 1, line 92)",
    "raw/2026-10-17-free-point-navtest/PREREG_FREEPOINT.md": "DEPLOYABLE-SAFE iff, against BASE, PDMS lower bound > 0 and the NC, DAC, TTC upper bounds are each ≥ 0; FP-NONE if neither R0B nor M25 qualifies (lines 25-29)",
    "raw/2026-10-18-so-carve/PREREG_SOCARVE_A2_TEMPSWEEP.md": "T grid {0.1, 0.05, 0.02, 0.01}; control 3: T = 1e-4 must reproduce the pick to 1e-6; PICK-ALIGNED iff |L_T − L_P1| ≤ 0.25 at all three states (lines 10-18)",
}
for r in fr["definition_files"]:
    S.append(f"| {dmap[r['file']]} | `{r['file']}` (blob `{r['git_blob_on_tip'][:12]}`) | `{r['sha256_tip_bytes'][:16]}` |")
S.append("""
Also quoted from prose, not from a JSON, and stated as such in §7.21: the free-point definitions of C0A and R0B
(`raw/2026-10-17-free-point-navtest/PLAN.md` §4: C0A = `rls1-c0` scorer state key `A` at update 1000, R0B = `rls1-r0` key `B` at update 1000;
M25 = `c + 0.25 (r − c)` on all seven outputs); and the carve provenance of SO-CARVE (`so-carve/RESULT.md` §2: read from the
trainer source, with the 16,000-set count independently confirmed by R0d's own `data` event, claim r_train_sets).

## 6. PUBLISHED rows (CLOVER, arXiv:2605.15120)

Banked in `TanitAD Research Lab/Library/papers/` (library key `2605.15120`); sha256 of the PDF in `library.json` =
`cdf0b4850472e7d5b035adc1775c93983dd1a7aafa786c4b50dbfabaac6e1053` (20,186,230 bytes). The values p01 to p07 above were re-read in that
banked PDF (Table 5; the Table 23 values agree except Stage 2's selected PDMS, which prints 0.9448 in Table 5 and 0.9443 in Table 23, see
CONFLICTS C11; the paper uses only the printed Gap, 0.0495). Nothing else from LAB-RUN-019 was re-read, so nothing else is used
(`TanitAD Research Lab/2026-10-18-LAB-RUN-019.md` is cited only as a survey pointer; sha256 `196f4edbed9061ca…`).
DriveZero (`2609.06055`, sha256 `82a00048648b155e…`) and NAVSIM (`2406.15349`, sha256 `d3bc66d321cfceccc…`) are cited by library key, as §7.20 already does.

## 7. What is not in any raw JSON

* "166 logs" (the RL-train log count) appears in `so-carve/RESULT.md` §2 and §11 but in no JSON; §7.21 does not print it (it prints the 16,000
  sets from R0d's `data` event).
* The temperature-criterion margin remarks, the "thin margin at u1000" and the near-tie diagnosis of control 3 are post-hoc reads of
  `tempsweep_a2.json` and `logs/a2_control3_diag.log` (the latter is claims s_c3_1e6 and s_c3_1e10 in the table); both are marked post hoc in the text.
* Counts such as "36 more zero-PDMS picks" (530 → 566) and "the 14 near-ties" are differences or counts of claims in the table (b-ids), not new measurements.
""")
wr("SOURCES.md", "\n".join(S))

# --------------------------------------------------------------------------------------------------------------------
C = f"""# CONFLICTS: where a package's prose disagrees with its own raw JSON, or the brief's wording needs a scope note

Paper v1.5 §7.21, drafted 2026-10-18. Rule applied throughout: **the raw JSON wins; the text of §7.21 follows the JSON and says so where it matters.**
No separation call (an interval's side of zero, or a bar crossed) changes between prose and JSON in any entry. The packages are read at mirror tip `5fb509a`.

## A. Prose versus raw JSON

**C1. The non-zero-pick term of the oracle gap: 5.88 (prose) against 5.875 (JSON).** `p2-full-table/RESULT.md` (anatomy table and Reading 1, 2),
`selrule-dac/RESULT_CONFIRM.md` (Reading 3) and the brief print **5.88 [5.54, 6.21]**. `p2-full-table/anatomy.json`,
`oracle_gap_by_cause_at_pick_pdms_pts/pick_nonzero` holds **5.874989315858168 [5.5348189393942935, 6.212801783721596]**: at two decimals that is 5.87 [5.53, 6.21];
the prose rounded the three-decimal print 5.875 upward. Effect: none on a separation call. §7.21 prints **5.875 [5.535, 6.213]** and derives the 62.4 % / 37.6 % split from
the JSON point (5.874989 / 9.412685 = 0.62416; from the prose 5.88 / 9.41 one would get 62.5 %). The four cause terms (5.875 + 2.502 + 1.012 + 0.024) add to 9.413, the JSON's oracle − pick.

**C2. "R0B removes the slow bias; the intervals do not overlap" (prose) against overlapping endpoints (JSON).** `free-point-navtest/RESULT.md` §3 ("intervals non-overlapping"),
the register row and the brief ("slow bias 'removed'"). `raw/result/families/fp_BASE_families.json` and `fp_R0B_families.json`,
`families/refcv6/longitudinal/ci/components/along_final_bias_m`: BASE **−0.8263 [−1.1897, −0.4778]**, R0B **−0.1950 [−0.5492, +0.1529]**. R0B's lower end (−0.5492) is
*below* BASE's upper end (−0.4778): the two intervals overlap on [−0.5492, −0.4778]. `speed_bias_mps` likewise overlaps (BASE [−0.3258, −0.1369], R0B [−0.1561, +0.0282]).
§7.21(b) therefore reports a point-estimate movement toward zero and says in so many words that it is **not** a separated change; no paired interval exists for the family metrics.
The "0 of 21 adverse" count is unaffected (it is the JSON's own `families_adverse_vs_BASE/R0B/adverse`, an empty list).

**C3. "TACTICAL rows carry no interval" (prose) against intervals present (JSON).** `free-point-navtest/RESULT.md` §3 ("rows without an interval, reported, not tested").
Both `fp_*_families.json` carry `families/refcv6/tactical/{{lateral_decision,longitudinal_decision,maneuver_5way_collapsed}}/ci/{{accuracy,kappa}}` as {{mean, lo, hi}}.
§7.21(b) prints them (12 claims t_*), labelled single-arm / 2,000 resamples, and reported, not tested.

**C4. SO-CARVE A2: "A2-ALIGNED(T* = 0.02)" (RESULT headline, register row) against `outcome: INVALID` (JSON).** `so-carve/tempsweep_a2.json`, key `outcome` = **INVALID**
(the pre-registered control 3 failed: T = 1e-4 must reproduce the pick to 1e-6; the differences read −0.015, +0.026, −0.017 with 0 picks differing from the argmax).
The RESULT body (§8) says the same ("By the prereg's own rule A2 is INVALID as written") while its headline and the draft register row say "tracks the pick". §7.21(c) follows the JSON:
the T = 0.02 reading is **PROVISIONAL, and by the registered rule invalid as written**; the rule is not relaxed. The post-hoc diagnosis (near-tie mass; exactly 0.0 at T = 1e-6 and 1e-10)
is from `logs/a2_control3_diag.log` and is marked post hoc.

**C5. "1.4 to 4.5 times smaller" (RESULT §0 and §8, the register row) against a ratio whose denominators span zero.** The ratios reproduce from `fullval_a1.json`
(`states/r0_u*/P1_TTC/L_va_full` over `states/r0_u*/P2_TTC/L_va_full`): 1.2333 / 0.8506 = 1.450, 0.6333 / 0.2297 = 2.757, 1.0000 / 0.2229 = 4.486. But the dual's quantity has
`ci_L_va_full` = [−0.1033, +0.5851] at u500 and [−0.1453, +0.6202] at u1000, so a ratio with that denominator is not interpretable (the rule that withdrew the lean-gap ratio in §7.19b).
§7.21(c) declines the ratio and reports the **paired difference** (quantity − pick), which is separated at one of three states (u1000 −0.777 [−1.320, −0.251]). The numbers are the same; the statistic is the admissible one.

**C6. Sign convention of the in-sample wedge (RZ-1).** `so-carve/RESULT.md` §5 and the register row give the R0d in-sample pick gap as **−0.297** (B minus A, ×100) and g as **−0.034**;
the post-hoc rows as pick −0.265 (se 0.089) and g −0.066 (se 0.032). §7.21(c) writes both as losses (positive = B below A), so g reads −0.034 and the pick **+0.297**; −0.265 becomes **+0.265**.
The magnitudes are the JSON's (`r0d-outcome/raw/sft.jsonl`, `con_g_ttc`, `con_p1_gap_ttc`; the post-hoc rows from `logs/r0d_insample_posthoc.log`); only the sign convention is changed, and it is stated in the text.

**C7. "166 logs" is in prose only.** `so-carve/RESULT.md` §2 and §11 ("16,000 sets in 166 logs, counted from the extracted `rlfan_sets_train` jsonl"); no raw JSON in the package holds it.
§7.21 does not print 166. The count that is printed, 16,000 RL-train sets with none in a held-out log, is R0d's own `data` event (`r0d-outcome/raw/sft.jsonl`, claim r_train_sets).

**C13 (minor). BASE against the banked u400, prose rounding.** `free-point-navtest/RESULT.md` §1 prints "−0.01 [−0.02, +0.00]"; `freepoint_result.json`
`identity/IC3_IC4/IC4/IC4b_paired_diff_BASEpod_minus_banked` holds −0.008178 [−0.024887, +0.000490]. §7.21 prints −0.0082 [−0.0249, +0.0005] (the upper end is positive, which "+0.00" hides).

## B. Class or scope notes (no number changes)

**C8. What 94.55 is.** `p2-full-table/RESULT.md` calls 94.55 "the human reference in D-REFE-GAP-CI … INHERITED". Paper §7.20 (base lines 3160-3165) distinguishes **two quantities that carry the value 94.55**:
the published matched student row (PUBLISHED, arXiv:2609.06055 Table A13, verified in the banked PDF) and our harness's logged human (94.55 [93.78, 95.23], MEASURED). §7.21(a) uses the **matched published row**,
class PUBLISHED, as §7.20 does; the 5.73-point / 60.9 % arithmetic (94.55 − 88.82; ÷ 9.41) is point arithmetic with no interval.

**C9. R0 against R0d in the proxy-wedge numbers.** The brief and the register row state "+0.22 vs +1.00 [+0.30, +1.73]" under the SAFEOBJ constraint. Those numbers are measured on the
**unconstrained R0's** states on the full RL-val (`fullval_a1.json`); R0d's own val-side E_π[TTC] was **not computed** (its score dumps are not on the dev box; `so-carve/RESULT.md` §9).
§7.21(c) says so (the wedge is measured at the R0 states and in-sample on R0d, not on R0d's val side) and keeps R0d's own pick loss (+0.767 [+0.065, +1.487]) beside R0's (+1.000 [+0.300, +1.735]) with the note that two runs without a replicate give neither an effect nor a null.
The upper end prints as +1.735 here; the prose's +1.73 is the same value (1.7348890) at two decimals.

**C10. A correction to my own draft (not a source conflict).** A first draft of §7.21(b) described C0A as "the control branch of the same pass". It is not: C0A is arm A of the
separate control run C0 (`rls1-c0`, state key A, update 1000), R0B is arm B of run R0 (`rls1-r0`, key B, update 1000) (`free-point-navtest/PLAN.md` §4; the "one pass" wording of `eval/PREREG_RL_S1.md` line 95 describes the two arms *within* one run).
Corrected before the final assembly; §7.21(b) now says C0A and R0B come from two runs.

**C11. A source-internal inconsistency in the literature row.** CLOVER (arXiv:2605.15120), Stage 2: the selected PDMS prints 0.9448 in Table 5 and 0.9443 in Table 23 (banked PDF).
§7.21(a) uses only the printed Gap, 0.0495 (×100 = 4.95), which is what the paper prints, not a difference recomputed from either selected value.

**C12. Line endings of the local extracts.** {N_CRDIFF} package JSON/JSONL files had CRLF added in the first local working copy. For those files the tip's blobs carry no CR. All values in SOURCES.md were read from the parse of the files (CR-insensitive),
and `files_read.json` records that every local copy is byte-equal to the tip after stripping CR; the sha256 digests in SOURCES.md are of the tip's bytes as committed and match those the R0d package quotes for itself.

## C. Things the brief states that the raw files confirm without change

* "9.41 splits as 5.88 / 2.50 / 1.01": 2.50 and 1.01 are 2.502 and 1.012 in the JSON; 5.88 is C1. The both-zero term (0.024, 4 tokens) completes the sum, which the brief does not list.
* "Only 64 of 12,146 tokens have no non-zero proposal": `anatomy.json` `n_best_of_64_zero` = 64.
* "R0B vs the deployed scorer: PDMS +0.55 [+0.24, +0.84], NC −0.25, TTC −0.40 (both separated adverse), EP +1.81": confirmed from `freepoint_result.json` (claims b22, b26, b28, b29).
* "SELRULE-DAC +0.0235 PDMS = 0.7 % of the DAC oracle": +0.0235 / 3.184 = 0.74 %; the lower bound is +0.0002, which §7.21 prints.
"""
wr("CONFLICTS.md", C)

# --------------------------------------------------------------------------------------------------------------------
A = """# ABSTRACT_PROPOSAL: Paper v1.5

**Decision for the Master Mind / PI: the abstract is left unchanged in `TANITAD_PAPER.md`.** No number it contains moves in v1.5 (88.25 [87.05, 89.36], 94.55, 65.9 % of the 6.30-point gap, 529 zero-scored tokens, 86.0 % [80.1, 91.7], AUC 0.46 against 0.92).
If the abstract should carry the §7.21 update, add this paragraph directly after the existing *Reference-arm update (v1.4, §7.20)* paragraph (below line 197 of the v1.4 file). Every number is a claim in `SOURCES.md`.

> *Reference-arm update (v1.5, §7.21):* inside the same model's own 64-proposal fan on the full navtest (T1; one training seed; paired log-cluster bootstrap, which answers another draw of logs only), the best-of-64 oracle reaches **98.23 PDMS [97.82, 98.62]** against the pick's **88.82 [87.86, 89.77]**. That **9.41-point [8.68, 10.18]** headroom is a ceiling, not an expected gain; **62.4 %** of it is ranking among non-zero proposals rather than the zero tail, and only 64 of 12,146 tokens have no non-zero proposal. The one RL lever that reaches that ranking buys progress (EP +1.81) and pays in safety (NC −0.25 and TTC −0.40, both separated adverse; PDMS +0.55 [+0.24, +0.84]), so no free point is deployable-safe. The safety constraint built to remove that payment failed because it held on an expectation under the training policy while its acceptance test reads the argmax, and a selection rule that re-weights the scorer's own drivable-area head is inert (+0.0235 PDMS, 0.7 % of its oracle reach) because those failures are confident. The 94.55 bar is still **not** cleared.

(About 190 words. The first sentence's tier statement and the last sentence's non-clearance are the same as §7.20's. The strategic family is paused (PI, 2026-09-19) and is not mentioned.)
"""
wr("ABSTRACT_PROPOSAL.md", A)

# --------------------------------------------------------------------------------------------------------------------
PKT = RPT + "/raw/2026-10-18-paper-v15/"
paths = [
    ["Paper/TANITAD_PAPER.md", "C:/Users/Admin/paper15/TANITAD_PAPER.md", "raw"],
]
for name in ["SOURCES.md", "CONFLICTS.md", "ABSTRACT_PROPOSAL.md", "paths.json", "claims_table.md", "claims_values.json", "files_read.json",
             "build_sources.py", "assemble.py", "verify_paper_numbers.py", "build_docs.py", "sec721_draft.md", "inserted_text.txt"]:
    paths.append([PKT + name, "C:/Users/Admin/paper15/" + name, "raw"])
paths.append([PKT + "base_blob.txt", "C:/Users/Admin/paper15/base_blob.txt", "raw"])
with open(os.path.join(HERE, "paths.json"), "wb") as f:
    f.write((json.dumps(paths, indent=1) + "\n").encode("utf-8"))
with open(os.path.join(HERE, "base_blob.txt"), "wb") as f:
    f.write((f"base file: Paper/TANITAD_PAPER.md @ mirror tip {tip}\nbase blob: {tip_blob}\nv1.5 blob: {paper_blob}\n").encode("utf-8"))
print("wrote SOURCES.md CONFLICTS.md ABSTRACT_PROPOSAL.md paths.json base_blob.txt; v1.5 blob", paper_blob)
