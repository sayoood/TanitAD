# BRIEF: C1's search-output EP-ADMISSION read, reported BESIDE C1 with NO authority (Master Mind, 2026-10-20 ~06:55Z)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- Work dir `C:/Users/Admin/par18/epadmc1/`. CPU only, at most 2 Python processes, `OMP_NUM_THREADS=2` (the SEARCH-DOSE chain and a fixture chain run on this box: never touch `C:/Users/Admin/par18/doser1`, `C:/Users/Admin/spc_dose`, `C:/Users/Admin/par18/c2prep*`, `C:/Users/Admin/c2prep_run`). If free RAM falls below 4 GB, stop and write the HANDOFF.
- Never commit or push. No navtest data.

## Why
C1 read **O1_CONFIRMED** (`.../raw/2026-10-19-search-p2-confirm/c1/`, `RESULT_SPC.json`). `MM_RULING_C2_LICENCE.md` §2.4 (sealed) commits to reporting, **beside C1 and with NO authority**: A2 and A1-r1 through the DQ6 adapter on C1's bank, with their C1-geometry error rates printed beside them (A2 false refusal 13.4 %, DAC power 7.4 %), and A3 / A4 / A5 of the search outputs. `ZZSPC_NEXT` names the same thing.

## Start
Pin the tip. Export `.../raw/2026-10-19-epadm-searchbank/` (the adapter `code/epadm_searchbank.py`, its `AMENDMENT_SEARCHBANK_DRAFT.md` §4 build command, `HANDOFF.md`, `RESULT.md`), `.../raw/2026-10-19-p6-read-prep/` (`ep_admission.py`) and whatever they import, and `.../raw/2026-10-19-search-p2-confirm/c1/`. Inputs (read-only, in place):
- C1's merged search and labels: `C:/Users/Admin/spc_c1/search_C1_P2HEAD.npz`, `C:/Users/Admin/spc_c1/labels_C1_P2HEAD.npz`;
- the B-REST bundle: `C:/Users/Admin/spc_c1/B_REST/bundle/bundle.npz` (sha256 `7b7dfff3…`).
B-REST labels may be read: C1 has read.

## Do
1. Build the search-output bank from C1's triple with the adapter (the same build as the carve-B identity proof, pointed at C1). Verify the adapter's identity/layout checks pass on it.
2. Run EP-ADMISSION (A1-r1, A2, A3, A4, A5) on that bank exactly as the frozen gate is defined (`ep_admission.py`); no parameter changes.
3. `RESULT.md` under `.../raw/2026-10-19-search-p2-confirm/c1/epadm/`. The headline line must read **"NO AUTHORITY (MM_RULING_C2_LICENCE.md §2.4): the C2 licence is C1's own read."** Then each criterion's verdict with its C1-geometry error rate beside it, taken from the epadm-searchbank simulation results (cite the file and key). Stamps: MEASURED, SELECTION tier, label-only, SEEN-SCENE, one training seed.
4. Manifest `epadmc1/out/paths.json` (`[repo path, abs src, "raw"]`, new files only), with the logs and bank summary json (not the bank arrays if > 5 MB; state where they live). `HANDOFF.md` (<= 400 tokens).

## Hand-back (<= 150 words)
Each criterion's verdict and error rate, the adapter identity result, and the manifest path.
