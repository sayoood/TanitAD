# TF-N1: VOID. Control C-ID-H failed, so no arm number is read

2026-11-12, Master Mind. Source: `n1_result.json` and `read.log` (chain stop 11:26:12 local = 10:26:12Z).

- Pre-registration: `PREREG_TF_N1.md` sha256 `a5dd339e...`, frozen 06:48:28Z. K ledger entry n = 41; a VOID read is still a read, so it stays consumed.
- MM freeze ruling §15.1 withdrew the harness fallback, because it had never been tested. A C-ID-H failure therefore voids N1. No N1-C, N1-M or CERT aggregate was computed or read.
- The family files `fam_*.json` exist on the dev box (`LOCAL_ONLY_sha256.txt`). They are NOT opened and NOT banked in the repo: they hold arm numbers of a void read.

## Controls

| control | result |
|---|---|
| C-COV | PASS (12,146 tokens, 12,110 primary, 0 error lines, every arm on every token) |
| C-BUILD | PASS |
| C-PIN | PASS |
| C-EST | PASS (max dev 0.0) |
| C-REG | PASS (N1-W 52.38 against N1-T 89.80, bar −10) |
| **C-ID-H** | **FAIL**: W-TRANSP's N1-T against the banked harness CSV `refe_p4_teacher_navtest.csv` exceeds 1e-6 on 3,346 / 12,146 tokens. All-token PDMS: 89.7799 against 89.7861 |

## Diagnosis of C-ID-H (MEASURED; N1-T only, which is the control's own scope)
Per-token |Δ| between W-TRANSP N1-T and the harness CSV:

| metric | > 1e-6 | > 1e-3 | > 0.1 |
|---|---|---|---|
| EP | 3,343 | 239 | 10 |
| PDMS | 3,218 | 111 | 8 |
| NC | 3 | 3 | 3 (flips) |
| TTC | 4 | 4 | 4 (flips) |
| DAC, DDC, C | 0 | 0 | 0 |

- **Two different defects.**
  1. **EP** differs as a continuous quantity on 27.5 % of tokens. Most of it is below 1e-3, which looks like a numeric or normaliser difference rather than a plan difference.
  2. **Seven tokens flip** a discrete metric: 3 NC and 4 TTC. Examples: `2b482f615de15efb` NC 0 against 1; `76615d386ce6519e` TTC 1 against 0.
- ⇒ **W-TRANSP is not harness-identical on navtest.** This is also W-TRANSP's open control "tracked == pinned W3 harness" (INSTRUMENTS.md; PREREG_P5C §10.5 (i)), now MEASURED as failing.

## Next lever (Rule Zero)
- An amendment, written and hashed BEFORE any re-score, re-runs N1 through the pinned W3 harness route. It is a new K entry.
- The harness route must first be made operable and smoke-tested on non-navtest tokens: `score_p4.py`'s `SP` constant points at a stale scratchpad.
- The W-TRANSP EP and flip defect is diagnosed separately, as an instrument fix.
