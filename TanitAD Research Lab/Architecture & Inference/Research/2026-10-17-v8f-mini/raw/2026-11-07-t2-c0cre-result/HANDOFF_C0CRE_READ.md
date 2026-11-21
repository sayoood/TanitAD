# HANDOFF_C0CRE_READ (2026-11-07, dev-box clock about 13:15). CPU only. Nothing staged, committed, pushed; box 2, busy markers, pods untouched.
MEASURED = ours, log named. UNVERIFIED = needs the real C0C-RE pull.

## Run this when `renc/c0cre/logs/c0cre_chain.out` shows `ZZ STAGE5 PULLED MK_DONE C0C-RE -> C:/Users/Admin/t2v8f_pulls/stage5_C0C-RE`
(the deciding run was launched 12:44; ESTIMATED 1.6 to 1.9 h. The BLIND run follows it on box 2 and does not block this.)
```
bash E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/c0cre_read.sh 2>&1 | tee E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/c0cre_read.run.log
```
It runs: pull check, `scene_exclusions.py` with both cell logs, the pinned read (pin 4, `--arm C0C-RE --arm-stage 5`), the pair C0C-RE - C0C, then `c0cre_outcome.py`. About 30 to 60 s (MEASURED 32 s on the BPC pull).
Outputs land beside it: `EXCLUSIONS5.json`, `score_dom/`, `pair_C0C-RE_C0C/`, `READ_ARGS.txt`, `RESULT_T2_RENC.json`, `RESULT_T2_RENC.md`. It refuses (2 no result file / log, 3 `EXCLUSIONS5.json` exists, 4/5/6 a reader wrote no JSON).
**When the BLIND pull has landed** (`stage5_C0C-RE-BLIND/pull5_C0C-RE-BLIND`): `C0CRE_BLIND_ONLY=1 bash .../c0cre_read.sh`. It reuses the deciding read, runs C0C-RE-BLIND - C0C-RE beside, and moves the old RESULT files to `RESULT_T2_RENC.before_blind.*` (delete them after you compare: same outcome expected, only the beside block is new). If both pulls are in when you start, the plain command does both.

## What to check
1. `exclusions5.out`: `ATTRIBUTED`, `problems: []`. BPC's identical stage lost 14 scenes (48 read); the md warns if C0C-RE reads other than 48.
2. `score_dom.out` and `pair_C0C-RE_C0C.out`: every gate true. Exit code of the script = the outcome code: 0 applied, 2 NOT-READ (a gate failed; the md names it), 3 INPUT-INVALID (a cross-check failed; the md lists it).
3. `RESULT_T2_RENC.md`: the outcome, the three RE-MOVES clauses per seed, the four families per seed (STRATEGIC shown, PAUSED), the committed next lever. The md says whether the inputs carry arm `C0C-RE` (a rehearsal banner appears otherwise).
4. `READ_ARGS.txt` hashes: `read_t2_dominance.py` 7fd65ffd..., `read_t2_pair.py` cbd0fdea....

## Section 6 as built (so you can overrule a reading by a dated card amendment)
* RE-DOMINANT = the pinned `outcome` is DOMINANT-T2 (so STRATEGIC counts inside the pinned D3; the three-family form is shown beside). Never counted as the scale signal; the md says a replicate is required.
* RE-MOVES = pair ADE6 separated-better at both seeds AND pair M1 crawl share not separated-worse at either AND no row separated-worse against P10 for C0C-RE that is not so for C0C (same seed; C0C from `stage4/score_q_identity`, cross-checked row by row against the pair's own C0C rows).
* Else RE-HARMS if any of those three fails on its worse side, else RE-NULL. The card does not split them; same next lever.
* Readings the card leaves open: "M1" = the pair's M1 row (M1 against P10 shown as a sensitivity); the HARMS/NULL split.

## Rehearsal (MEASURED; scratch deleted)
* `c0cre_outcome_selftest.out`: 26 of 26 checks. 9 of 9 mutations of the outcome script went RED (dropping a clause, ADE6 at one seed enough, ignoring either reader's gate outcome, no card check, no D2 cross-check, no reference cross-check, no HARMS label); two escaped the first test and made it gain isolating checks.
* Identity pull (2 scenes): steps 1 to 3 ran; exclusions ATTRIBUTED; the pinned read passed G4, G1, G2 (so the RE overlay and result wiring pass G2 a) and failed G3 as it must on 2 of 62 scenes; the outcome script then wrote NOT-READ (rc 2).
* Full pipeline on the complete BPC pull (`C0CRE_ARM=BPC C0CRE_RUN=BPC`): exclusions and dominance rows equal the banked BPC reads bit for bit; the pair BPC - C0C ran; the outcome script applied section 6 (RE-HARMS, banner REHEARSAL). Also: refusal on a missing pull (rc 2), no overwrite of `EXCLUSIONS5.json` (rc 3), dry-run wiring of the BLIND branch.
* NOT rehearsed: the pinned read's row stage on a real C0C-RE pull (cells named `C0C-RE_s0/_s1`, 62 scenes; the BPC pull stands in), the BLIND pair on real BLIND cells (same argument pattern as `pair_BPC_MC`), the box-2 side.

## Not in this deliverable
The T1-kin read of C0 on eval-141 through E' (card section 5, beside), the render panels beside P10's (card 6.1), the replicate (6.1), and the dated log line in the card and the GOALS_AND_CLAIMS update (Master Mind's).
