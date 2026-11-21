# R0e-pick: the three EVAL-side launch blockers (B3, B6, B7)  -  RESULT

Written 2026-10-18 by an R0e-blockers eval agent from the Master Mind brief of the same date. **Status: STAGED, not committed, not pushed.** Parent: `raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.draft.md` (a DRAFT; the PI decides whether R0e-pick runs). PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Read from mirror tip `3a28bc1`. Dev box only; no pod, no GPU, no network. Nothing under `PK/refe/` was touched (B2, B4, B5 wait for the P6 launch-gate work; B8 is outside this brief).

Every number below is **MEASURED (ours)** with the artifact named beside it, unless marked otherwise. Estimator for every interval: the **paired log-cluster bootstrap** (75 logs, 10,000 resamples, seed 20260927) for D3/D4 and the **standard error of the mean over the logged rows** for D1; neither answers "another training run" or "another inference run" (`H-ESTIM-SEED-1`).

## 1. Manifest (what lives where; all under `PK/`)
| file | status | base blob @ 3a28bc1 | sha256 of the staged file |
|---|---|---|---|
| `eval/safeobj_outcome.py` | EDITED (LF) | `acfe12f0eac2306dc72c66a0273b96b720b9f3d9` | `abfa88a1c361e4c7c3624c8300ad6e38a6ff214d5afe056c7572181f2a091ac3` |
| `eval/test_safeobj_outcome.py` | EDITED (CRLF kept) | `4499f1b726f6d47ce3a5d498ca9ac458aab5570a` | `c62dba175949e31a90dce119153fccc1b5de8c57cebd0284d5283fc1e4feccf2` |
| `eval/r0e_diagnostics.py` | NEW (LF) | - | `12ee20179ce0243f059e37c90e3deb694cad74404e3303f416df3a93a8a807f0` |
| `eval/test_r0e_diagnostics.py` | NEW (LF) | - | `dd78a5fb1a040d72413b365c59b4381af3b08b9439c3974fbf1c762cd70cbe33` |
| `raw/2026-10-18-r0e-blockers/PREREG_R0E_D4_NC.md` + `.sha256` | NEW | - | `a8dc326577f3029094c311ab42b4e293df491bd89036bdf1b83538472b03c4eb` (written BEFORE any NC number) |
| `raw/2026-10-18-r0e-blockers/r0e_d4_nc_alignment.py` | NEW | - | the D4 instrument (imports the banked `so_a2r2_tieaware.py` unchanged) |
| `raw/2026-10-18-r0e-blockers/d4_nc_alignment.json` | NEW | - | the D4 result |
| `raw/2026-10-18-r0e-blockers/r0d_diagnostics_banked*.json` | NEW | - | the B6 reader run on R0d's banked log (two scopes) |
| `raw/2026-10-18-r0e-blockers/logs/` | NEW | - | both full test logs + summaries, the D4 first-run copy, reader stdout |
| `paths.json` (beside this package's staging root) | NEW | - | `[repo path, abs src, "raw"]` for every staged file |

**Integration (escalated, not left as a note):** the Master Mind lands the four `eval/` files and this package. `SHIP_MANIFEST.json` (247 entries) lists no `eval/` file, so it needs no regeneration. The only other tests that name `safeobj_outcome.py` resolve to the `raw/2026-10-17-safety-objective/outcome/` copy and are unaffected (grep over the whole tree at 3a28bc1: no `refe/` file names it).

## 2. B3  -  the per-arm registered constraint temperature  -  DONE
**Defect (MEASURED on the base tool):** `_registered_view()` accepted the dual's `temp` key only at the module constant 0.1, so a `--epo-con-temp 0.01` arm's block would read `NOT_READ`, and there was no way to say what an arm was registered at.

**Change (`eval/safeobj_outcome.py`).**
* The registered T is **typed or read from the arm's registered argv, never inferred from the log**: `--con-temp NAME=T` (repeatable), or `--arms-file <arms.json> --arm-reg NAME=ARM` (the T that `arms.ARM.argv` resolves to: `--epo-con-temp`, else `--epo-temp`, else 0.1, the trainer's own resolution `scorer_finetune`: `temp=(a.epo_temp if a.epo_con_temp is None else a.epo_con_temp)`). If both are given they must agree (rc 3 otherwise). The arms-file sha256 and the argv tokens are recorded in the record (`registered.epo_con_temp_by_arm`, `epo_con_temp_default`, source).
* `read_log(..., registered_temp=)`: a data event whose `epo_con.temp` differs from **this arm's** registered T is REFUSED (the message names both numbers). A bool is never a temperature. A legacy block (no `temp` key, a trainer that predates the flag) implies T = 0.1 and is refused for any other registered T, so a pre-flag log can never evidence a 0.01 registration.
* **R0d's banked log is unchanged:** with no T named the default stays 0.1, and the tool reads `raw/2026-10-18-r0d-outcome/raw/sft.jsonl` to the same record fields as before (c = 750, same sft sha `5380d8d1…`, 528 of 550 post-ramp rows), and REFUSES it under a registered 0.01.
* Usage errors are rc 3: bad or non-positive T, a slot not given via `--arm`, a non-constrained slot (R1o, R0), duplicates, source disagreement, `--arms-file` without `--arm-reg`.

**Naming (a decision to know about).** The tool is slot-based and bar E's pairs are keyed R0d / R1d. **R0e-pick is therefore read in slot R0d** (`--arm R0d=<dir> --con-temp R0d=0.01`), and R1e-pick in slot R1d. No alias names were added: an alias would be a second key that bar E would have to learn.

**Tests (`eval/test_safeobj_outcome.py`, full form, run on the dev box).** **`ZZTEST_PASS 153  (1 unmutated suite of 189 checks + the CLI end to end, 152 RED mutations)`** (rc 0, 752.1 s, log `logs/test_safeobj_outcome.py.log`; tested file == staged file by sha256)
Base: `ZZTEST_PASS 136` (177 checks, 135 mutations, 616 s, same harness on `base_tree`). New: 12 `con_temp:` checks (a REAL 0.01 log accepted under 0.01 and refused under 0.1, and the reverse; the legacy block refused; other temperatures and a bool refused; each arm read against its own T; no-T writes the same record as before; the banked R0d log), the `con_temp_cli:` end-to-end checks inside `cli_checks` (unmutated suite 177 -> 189 checks), and 17 added RED mutations (135 -> 152), among them the one the brief asks for: **deleting the per-arm comparison turns `con_temp:*` RED** (`the per-arm comparison is deleted`), plus T inferred from the log, the bool guard, the legacy guard, one T for all arms, the arms-file resolution, the CLI drops, the disagreement check. One PRE-EXISTING mutation anchor (`lg = read_log(sft, c_upd, ctx)`) no longer matched after the signature change and was updated to the new call text; it was found by the first lane aborting on it, not by inspection, and `anchor_check.py` over all 152 anchors now reads 0 bad.
**Provenance bug found and fixed by the test itself:** the arms-file source dict was `{**src, "kind": ...}` so `**src` overrode the new kind; caught by `con_temp_cli:--con-temp_and_the_arms_file_may_agree`.

## 3. B6  -  the D1-D3 diagnostics reader (`eval/r0e_diagnostics.py`)  -  DONE
Reads the dual's logged fields only (`sft.jsonl`: λ, `con_g_*`, `con_p1_gap_*`, `epo_J`, `con_lam_pos_updates`, `epo_con.temp`/ramp), fail-closed: a non-contiguous 25-update series, a non-finite value, a missing built ramp, a log that ends before u1000 (D2), a missing or non-positive reference rise, or a T the log did not run at (D3) gives `NOT_READ` and **no number**, never a shorter window.
* **D1** wedge = 100·(g + p1gap) over the post-ramp rows (the draft's "31 post-ramp rows u226-u1000": first row u250), signed; |D1| ≤ 0.10 aligned; BORDERLINE iff within 1 se of the bar. Reported for TTC (decides) and NC (descriptive).
* **D2** epo_J mean over u751-1000 minus the u25 row, against R0d's rise; CROWDED iff rise < 0.5 × R0d's.
* **D3** validation L_T / L_P1 from the arm's own score dumps (tie-aware pick at τ0 = 2.3257e-3, the A2R2 machinery) against the in-sample 100-update window, per check, with a bootstrap CI; a registered T the log did not run at is refused.
* **ROW** refines O4 on D1 (P-O4-PICK / P-O4-WEDGE) and O3 on the λ-positive share and D2 (O3-a / O3-b); nothing else is refined.
* CLI: `python eval/r0e_diagnostics.py --arm-dir <dir> [--reference-log <R0d sft.jsonl>] [--through-update N | --outcome-json <safeobj_outcome.json> --slot R0d] [--scores-dir ... --registered-temp T --val-pdm ... --val-terms ...] --out <json>`; rc 0 all READ, 2 something NOT_READ, 3 usage.

**On R0d's banked log (MEASURED, `raw/2026-10-18-r0d-outcome/raw/sft.jsonl`; records `r0d_diagnostics_banked.json`, `..._through_first_breach.json`):**
| read | value |
|---|---|
| D1 TTC, 31 rows u250..u1000 | **-0.3165 ± 0.0524** points (se over rows): NOT ALIGNED (bar 0.10). The draft quotes -0.317 ± 0.052: reproduced. |
| D1 sensitivity, rows > ramp | -0.2940 (the 32-row window; same verdict) |
| D1 through the first breached check (u750, 21 rows) | -0.3310 ± 0.0700 -> ROW `P-O4-WEDGE` |
| D2 | J u25 0.8171 -> mean u751-1000 0.8766, rise **+0.0595** = 1.000 × the reference: NOT crowded |

**Tests (`eval/test_r0e_diagnostics.py`):** **`ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI end to end, 43 RED mutations)`** (rc 0, 26.5 s, log `logs/test_r0e_diagnostics.py.log`; tested file == staged file by sha256) Self-contained: planted logs hand-computed, the banked R0d log re-derived by an independent recomputation, the heavy `real_states:` check reproducing A2R2's banked `L_T`, `L_P1` and `diff` on the REAL R0 states at T 0.1 / 0.01 to 1e-9 (against the BANKED τ0, not the tool's own: a first version used the tool's τ and let a τ drifted to 1e-3 pass; a mutation caught it), a pin of every constant against the draft's §6/§8 text, and the CLI end to end. Three mutations were found weak by the sweep and strengthened before this was banked: `D2 fraction 0.5 -> 0.25` (needs the exact-half case), `D2 accepts a log that ends before u1000` (needs a killed fixture with u751+ rows but no u1000), `the tie set is strict` (needs a candidate exactly τ below the top).

**Decisions an owner may want to overrule (HYPOTHESIS-grade readings of the draft, not facts):**
1. D1's window follows the draft's own quote (u≥226, first row u250, 31 rows). The draft is silent on whether it means 31 or 32 rows; the 32-row variant is reported beside it in every record (R0d: -0.3165 vs -0.294, both far outside 0.10).
2. D3's in-sample window uses addendum R3's 4-row (100-update) window; the draft does not define it more tightly.
3. The test pins the draft's text. On freeze/rename (`PREREG_R0E_PICK.md`) the pin follows (`DRAFT` lists both names); if the sentences change, the pin fails loudly and should be re-read, not loosened.

## 4. B7  -  D4: NC alignment at the constraint temperature  -  DONE
Pre-registered first: `PREREG_R0E_D4_NC.md`, sha256 `a8dc3265…03c4eb` (timestamp 2026-10-18T07:29:41Z), both outcomes and a vacuity guard committed before any NC number was computed. The method is A2R2 unchanged (`so_a2r2_tieaware.py`, tie-aware pick at τ0, RL-val labels, the arm's own B and same-pass A at u250 / u500 / u1000, 3,000 sets, 75 logs), with the label swapped from TTC to NC (`comps["nc"]`). Controls: 4 of 4 pass (the TTC label reproduces the banked `a2r2_tieaware.json` to 0.0 abs diff; B := A reads exactly 0.0 for NC; the pick route equals `AT.picks` with 0 exact top ties; the planted control passes).

**Outcome: `D4-NC-ALIGNED(0.01)`, secondary `NC-A`** (NC does not separate T = 0.01 from A2R2's T* = 0.02; the T choice rests on TTC).
| T | M(T) = max_u \|L_T − L_P1^τ0\| (points; bar 0.25) | aligned | level form: max over arms and states of \|E_π[NC] − pick NC\| (points) |
|---|---|---|---|
| 0.1 | **0.2600** | no (by 0.010) | 0.642 |
| 0.02 | 0.1115 | yes | 0.130 |
| **0.01** | **0.0754** | **yes (margin 0.175)** | **0.0710** |
The level-form column is the brief's literal `max |E_pi[NC] - pick NC|`; it was ADDED to the output after the first run (descriptive, not an outcome input): `logs/d4_nc_alignment.first_run.json` differs from the final json by exactly that block, and the outcome, `M(T)` and every per-state number are byte-identical.

Per state at T = 0.01: diff -0.0399 [-0.129, +0.059] (u250), -0.0345 [-0.129, +0.049] (u500), -0.0754 [-0.187, +0.022] (u1000). The pick's own NC loss `L_P1` is 0.167 / 0.197 / 0.281 at u250 / u500 / u1000.

**How strong this is (MEASURED caveats, read before quoting).**
* **Thin evidence.** V = max_u \|L_P1\| = 0.2806 clears the 0.25 vacuity guard by only 0.031; the null predictor `L_T ≡ 0` is already within the bar at u250 and u500. Only u1000 can discriminate. The monotone shrinkage of M with T (0.260 -> 0.111 -> 0.075) and of the level gap (0.642 -> 0.130 -> 0.071) is the supporting pattern, not any single cell.
* **Every bootstrap CI spans 0 at T = 0.01** (and at T = 0.1 except u500). The estimator answers "another draw of logs" only.
* **R0 is unconstrained**: this says nothing about whether 0.01 stays aligned along a constrained trajectory (that is D1, in-run). One R0 run per state, offline selection tier (not T1); NC is a PDM collision term; LATERAL and TACTICAL instruments do not exist at this tier; STRATEGIC is PAUSED (PI ruling 2026-09-19), reported as paused, not as a pass.
* **No lever effect and no bar changed**: D4 is descriptive and decides no row. The draft's §11 sentence "NC is constrained at a temperature measured not to track the pick" is **not** triggered at 0.01.

## 5. What blocks nothing, and what is still open (named, per Rule Zero)
* **B3/B6/B7 are the whole eval-side scope; none is blocked.** The remaining launch blockers (B2, B4, B5 under `refe/`, B8) wait for the P6 launch-gate work and the PI; they are named in the brief, not touched here.
* The reader has been run only on R0d's banked log. Its first run on a REAL R0e-pick log will be the first time D3 sees a 0.01 dump; D3's `real_states:` check covers the R0 states only.
* If the PI wants NC watched at a temperature of its own, the next lever is a per-component constraint temperature (`--epo-con-temp-nc`): not needed on this evidence, and it would need its own G-DVB entry and reachability test.
