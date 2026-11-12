# PREREG TF-N1 AMENDMENT 1 (DRAFT): the N1 arms scored through the pinned W3 harness

Status: DRAFT for the Master Mind. Not frozen. Nothing in it has been scored on an N1 seam. When frozen, its sha256 is `<FILLED AT FREEZE>`.
Base card: `../PREREG_TF_N1.md`, sha256 `a5dd339eb4bf27ec545122d3288166ae49f52ada57ec029cf67629204bcc3fe2` (kept unchanged).
Void result: `../run_void/RESULT_N1_VOID.md`. Root cause of the void: `WTRANSP_DIAG.md` (this folder).

## 0. Why this amendment exists, and what the MM has already seen (DISCLOSURE)

TF-N1 was read once, on 2026-11-12, through W-TRANSP. Control C-ID-H failed (N1-T differs from the banked harness CSV by more than 1e-6 on 3,346
of 12,146 tokens), MM ruling 15.1 withdrew the untested harness fallback, and the read is **VOID**. A VOID read is still a read: K entry
n = 41 stays consumed.

**What has been seen before this amendment (all of it, and nothing else):**
1. N1-T's per-token differences to the banked harness CSV (counts per metric, the 7 flips, the EP distribution). This is the identity control's own scope.
2. The C-REG line: N1-W 52.38 against N1-T 89.80, bar -10, PASS.
3. The control results C-COV, C-BUILD, C-PIN, C-EST (all PASS).
4. For the root-cause diagnosis (`WTRANSP_DIAG.md`): N1-T rows only, plus the harness's metric cache and the nuPlan DB. One banked **non-N1** seam (the human seam of the
   E-0 check, 200 navtest tokens) was scored through the pinned harness and through W-TRANSP's scoring function, to test the repair and the harness route.

**What has NOT been seen:** any N1-C, N1-M or CERT aggregate or per-token value, any N1-W value other than the C-REG line, any family value.
The void run did compute W-TRANSP rows and families for those arms; they sit on the dev box (`fam/fam_*.json`, `score/rows_*.jsonl`), unopened.
This amendment pins their bytes (sha256, never values) and uses the families and transparent scores from them, flagged (section 3).

**The route change was not chosen after seeing an N1-C number.** The harness is the scorer the card named as the identity reference (base card
section 5: "W-TRANSP is the scorer of record on the condition that C-ID-H passes"); the condition failed on N1-T alone.

## 1. Unchanged from the base card

Section 2 population (12,146 / 12,110), section 3 inputs, section 4 arms (N1-C Tc20s PRIMARY, N1-T, N1-W, N1-M, CERT "teacher + map-certified
selection", human), section 6 bars and outcomes (CONFIRMED needs dPDMS >= +3.0, CI lower bound > 0, N1-C DAC >= 98.5; PAPER-LEVEL / NEAR / BELOW; expected values),
section 8 estimator (paired log-cluster bootstrap, 136 logs, 10,000 draws, seed 20260927, 95 % percentile), section 10 four metric families, section 11 E3 caveat,
and MM rulings 15.2 to 15.6. The plans are the void run's: `build/out/refe_tfn1_{tc20s,tw,m}.npz` (C-BUILD PASS in the void run; pinned by sha256 in
`A1_PINS.json`; token set, fingerprints and `sampling` verified equal to the registered T seam, 2026-11-12). Nothing is re-derived.

## 2. What changes

| item | base card | A1 |
|---|---|---|
| scorer of record | W-TRANSP tracked arm, conditional on C-ID-H | **the pinned W3 harness**: `eval/score_p4_a1.py` -> `eval/harness/score_e.py` (HARNESS_PINS, 21 files + pins file), each seam scored over all 12,146 tokens |
| the four primary scores | W-TRANSP | harness: the **registered T seam re-scored** (`inputs/refe_p4_teacher_navtest.npz`), `refe_tfn1_tc20s.npz`, `refe_tfn1_tw.npz`, `refe_tfn1_m.npz` |
| C-ID-H | W-TRANSP N1-T vs banked CSV | **the harness re-score of the T seam vs the banked CSV** `refe_p4_teacher_navtest.csv`, |d| <= 1e-6 on each of NC, DAC, DDC, EP, TTC, C, PDMS, on **all 12,146 tokens**, 0 missing |
| order | T, C, W, M together | **T first, alone; C-ID-H is checked at once (chain stage `cidh`); only if it passes are C, W, M scored.** If it fails the chain stops and no N1-C / N1-W / N1-M number exists |
| K | 41 | **42** (ledger entry n = 42, section 7). Bonferroni percentiles 2.5/42 and 100 - 2.5/42 = 0.0595 / 99.9405 (`tf_n1_analyze_a1.py`, one constant, diff in `tf_n1_analyze_a1.diff`) |
| human arm (descriptive) | W-TRANSP, the DB's logged future | the **banked NAVSIM human harness CSV** `refe_p4_human_navtest.csv` (a different human: NAVSIM's), so the base card's human diagnostic against the banked CSV is now an identity |
| CERT | per token, from W-TRANSP tracked DAC | per token, from the **harness** tracked DAC (`cert_choice`, unchanged rule; none-pass -> N1-C) |
| transparent scores and families | W-TRANSP rows | **unchanged W-TRANSP rows and families from the void run, flagged "not harness-identical"** (W-TRANSP rebuilds the scenery from the nuPlan DB; `WTRANSP_DIAG.md`). Descriptive only, never a bar |
| fallback | withdrawn (15.1) | none. **If C-ID-H fails again the read is VOID again**, the K entry stays consumed, and any further route needs its own amendment |

Controls under A1: C-PIN-A1 (every file of `A1_PINS.json` and of `N1_PINS.json`; the harness tree holds no unpinned file), C-COV (12,146 tokens, 0 error
rows, every arm on every token, 12,110 primary), C-BUILD (the void run's build report, pinned), C-EST (reproduces `proxy_eval.boot` on the re-scored T seam to <= 1e-9),
**C-ID-H (above, early and again in the read)**, C-REG (PDMS(N1-W) <= PDMS(N1-T) - 10 on the 12,110, both from the harness), the human diagnostic (never voids).
The analysis exits before any N1-C, N1-M or CERT aggregate if any control fails (base card section 7, unchanged).

**Two things the harness route cannot give, stated now:**
* The per-arm **transparent** score and the **families** are not in the harness CSVs. They stay W-TRANSP's. W-TRANSP differs from the harness by up to a few centimetres of object
  position (`WTRANSP_DIAG.md`), so family values are labelled "W-TRANSP scenery" and are not harness-identical. The CERT family file was built from W-TRANSP's own CERT choice; where the harness DAC
  would choose differently (not measured) the family and the harness CERT scores describe slightly different selections. CERT decides nothing.
* The harness is hard-wired to the navtest metric cache; its identity could be smoke-tested only on navtest tokens with a non-N1 seam (section 5).

## 3. Method, in the order it runs (`tf_n1_a1_chain.py`; CPU only, at most 2 processes, 0 GPU)

0. **Before the freeze** (the MM): `python a1_prep.py --run-dir C:/Users/Admin/n1_run --mirror C:/Users/Admin/tanitad-push/.git --tip <landed sha>`. It exports `eval/harness/`
   at that tip, checks it with the tree's own `harness_pins.py`, copies `score_p4_a1.py`, `tf_n1_a1_chain.py`, `tf_n1_analyze_a1.py` into the run tree, refuses unless the E: copy of
   `score_navtest_refe.py` (which `score_e.py` loads by an absolute path) equals the run tree's, and writes `n1/a1/A1_PINS.json`. The amendment quotes `A1_PINS.json`'s sha256: `<FILLED AT FREEZE>`.
1. **Freeze** this file (sha256), **append the ledger entry** (section 7) and land it, copy the landed ledger to `inputs/ledger_K.json`.
2. **Guards** (`launch_n1_a1.ps1 ... -CheckOnly`): the chain refuses unless the amendment sha, the ledger key `TF-N1-A1:<sha>:E3` with n = 42 (and the void n = 41 entry present, and no
   entry above 42), `N1_PINS.json`, `A1_PINS.json` and its sha, and the harness tree's file set all hold. 22 refusal and gate tests: `test_tf_n1_a1_chain.py`.
3. **Launch** by `Invoke-CimMethod Win32_Process Create`, `CreateFlags 0x08000200`, `< NUL` (`launch_n1_a1.ps1 -Print` shows the command; no `-ExecutionPolicy Bypass`).
4. Stages: `score_T` (one process) -> `cidh` (**STOP on failure**) -> `score_CWM` (C and W together, then M) -> `read` (`tf_n1_analyze_a1.py read --harness-csv N1T= N1C= N1W= N1M= ...`,
   controls first). Outputs under `<run>/a1/`; the void run's `out/` and `score/` are not touched. Each harness run retries on its own RAM guard (6 attempts) and its own status line
   must read PASS with 12,146 CSV rows; a CSV existing is not evidence.
5. After the run: bank `a1/` (CSVs, `cidh_early.json`, `n1_a1_result.json`, logs), verify by sha256, then delete `C:/Users/Admin/n1_run` by its literal path and log it.

## 4. Expected cost

Per seam, harness over 12,146 tokens, one process, about 2.3 GB RAM. The only timing is the smoke: 200 tokens in 99.2 s wall including start-up (MEASURED, `smoke/refe_a1smoke_human_sub200_status.json`),
i.e. 0.50 s per token, so **about 1.7 h per seam if start-up is negligible** (ESTIMATED; the card for the 2026-10-05 run quoted about 0.4 h, which would be 0.12 s per token and is not what this machine shows).
Wall: T 1.7 h, then C and W in parallel 1.7 h, then M 1.7 h = **about 5 h**; CPU-hours about 6.8; the read about 1 min. Lower bound if the older 0.4 h/seam is right: about 1.2 h wall.
The W-TRANSP stages (about 11 to 18 CPU-h in the base card) are NOT re-run. 0 GPU, no pod, no box 2.

## 5. Evidence that the route is operable (navtest tokens, a banked NON-N1 seam; no N1 seam was scored)

| check | result | artifact |
|---|---|---|
| `score_p4.py`'s `SP` stale scratchpad | repointed to the pinned `eval/harness/` of the tree it sits in; output root is an argument; relative paths made absolute (the harness runs with cwd = `eval/harness`); `RUN_V1_FIX1_ALLOW_DRIFT=1` as `stage2_sft.py:191`, because the drive-repointed `run_v1.py` differs from W3's original by drive roots only | `score_p4_a1.py`, `score_p4_a1.diff` |
| harness tree integrity | `harness_pins.verify` OK on a fresh export at tip `3b7adced`: 21 files, caller variant NT_FIX1 | `a1_prep.py` log |
| **identity smoke** | the human seam of `eval/raw/e0_human_sub200` (200 navtest tokens) re-scored through `score_p4_a1.py`: status PASS, PDMS 94.1195; against the banked `refe_e0b_human_sub200.csv` **|d| = 0.0 on NC, DAC, DDC, EP, TTC, C and score for all 200 tokens**, 0 missing | `smoke/smoke_identity.json` |
| `tf_n1_analyze.py read --harness-csv` | runs end to end on that output (as N1T, N1C, N1W, N1M, with the human CSV): C-COV true, C-EST true (dev 0), **C-ID-H true**, exits `ZZN1_CONTROLS FAIL` before any aggregate because C-BUILD, C-PIN, C-REG are meaningless on a smoke (as designed) | `smoke/analyze_smoke.log` |
| chain guards and the early gate | 22 tests green; mutation arm (the K guard disabled) goes red | `test_tf_n1_a1_chain.py` |
| launch mechanism | `Invoke-CimMethod Win32_Process Create`, flags 0x08000200, `< NUL`, tested on a harmless command: rc 0 | this session |

**Limits of this evidence (stated, not hidden):**
* The smoke seam is a **human** seam on **navtest** tokens. The card's rule was "non-navtest tokens only"; that was **impossible** for the harness, whose metric cache is the navtest cache
  (`run_v1.py:176`, `cache_name=navtest`) and which has no navtrain cache. No N1 arm (T, C, W, M, CERT) was scored. The human arm is a reference arm, not a candidate.
* The banked CSV `refe_e0b_human_sub200.csv` predates NT-FIX1; the A1 harness is the post-NT-FIX1 caller (its three fixes are non-numeric by its own RESULT). The 200-token bit identity is the evidence that they agree.
* **Untested at scale:** 12,146 tokens per seam, the RAM-guard retry path, and two harness runs in parallel. The banked teacher run (2026-10-05) is the precedent for the first two.
* The identity of the T-seam re-score to the banked CSV is exactly C-ID-H and is unknown until the chain runs. W-TRANSP's failure is explained (`WTRANSP_DIAG.md`), and the harness is the process that wrote the CSV.

## 6. What the verdicts mean

* `cidh` FAIL: the harness does not reproduce its own banked CSV on N1-T. The read is VOID a second time; no N1-C, N1-W or N1-M harness score exists. The K entry (42) stays consumed. The next step is
  diagnosing the harness-vs-banked difference (environment drift since 2026-10-05), not a third scorer.
* `cidh` PASS, then the read: bars and outcomes as the base card (section 1 above). C-ID-H is re-checked inside the read on the same CSV.
* Deliberate disclosure for the reader of the result: the harness scores the **same plans** as the void run (so N1-C's plan is not tuned between the two attempts), the families and transparent
  scores are W-TRANSP's, and the human arm is NAVSIM's.

## 7. K ledger (the MM appends; ledger at tip `0f15198c` holds 41 entries, max n = 41, the last being the void `TF-N1` entry)

```json
{"key": "TF-N1-A1:<sha256 of this file>:E3", "n": 42, "time": "<freeze UTC>", "nominee": "TF-N1-A1", "family": "TF", "set": "E3",
 "smoke": false, "arms": ["N1-C (candidate, PRIMARY)"],
 "non_selection_reads": ["N1-T (harness re-score, C-ID-H)", "N1-W (deliberate regression, C-REG)", "N1-M (CERT input)",
                         "CERT (pre-registered SECONDARY, decides nothing)", "human (descriptive, banked NAVSIM CSV)"]}
```
If another entry lands first, n and K shift: the MM corrects n here, the constant in `tf_n1_analyze_a1.py` and `--k`, re-runs `a1_prep.py`, and re-freezes. The chain refuses a ledger whose top n exceeds 42.

## 8. Before the freeze, the MM does (in order)

1. Land the hold; run `a1_prep.py` against the landed tip; note `A1_PINS.json`'s sha256 in section 3.
2. Decide whether the W-TRANSP repair (`WTRANSP_DIAG.md` section 4) is wanted for later work; it is not part of A1.
3. Freeze, append and land the ledger entry, copy the ledger to `inputs/ledger_K.json`, run the `-CheckOnly` launch, then launch.
