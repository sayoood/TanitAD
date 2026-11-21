# MANIFEST: stream SCORER-GYM (2026-11-07)

**State: DONE.** Nothing is running. Nothing was staged, committed or pushed. The folder sits outside the repo, and the Master Mind lands it.

| artifact | lives in | note |
|---|---|---|
| `LIT_REVIEW.md` | this folder only | 16 sources in the table, each read from its banked primary; one CLOVER number is INHERITED |
| `SCORER_GYM_DESIGN.md` | this folder only | design only, nothing built |
| `PREREG_DRAFT_SCORER1.md` | this folder only | DRAFT; the MM freezes it by sha256 |
| `BANK_LIST.json` | this folder only | 19 entries with sha256 and a one-line finding |
| `library/*.pdf` (19) | this folder | 17 are byte-identical copies from `TanitAD Research Lab/Library/`. **2 are NEW:** CaRL `2504.17838` and Gen-Drive `2410.05582`. The MM should bank both from the tip with `tools/kb_add.py`. |

## Proposed register rows (the MM writes them from the tip)

- **`D-REFE-U400-RECIPE-1`** (MEASURED from source): u400 was already trained with per-head BCE plus a listwise expected-true-PDMS term.
  - Source: `sft4_pod.sh:10` `--b-mode expreward --lam 1.0`; `scorer_finetune.py:113-119`.
  - In the full-information one-step setting, EPO (ZTRS eq. 7) is the same estimator.
  - ⇒ "Train the scorer with RL / listwise on true PDMS" is **not a new lever** on REFe. The genuinely new piece is multi-step credit (gym Stage B), together with a within-token objective on the safety heads.
- **`H-SCORER1-1`** (HYPOTHESIS, prereg DRAFT): a speed-balanced within-token pairwise loss on NC/TTC/DAC shrinks the P1 Q4 speed gap by at least 50 % and cuts CRUISE slow-downs by at least 10 points, with no PDMS or TTC loss beyond 0.5.
  - Stage 0, a zero-GPU check on navtrain held-out, decides first whether the loss is the right lever at all.
