# The adopted route fix on the FULL navtest: exact measurement (2026-10-04)

> PI rulings 2026-10-04: *"Can you adopt the route fix and measure performance?"*; *"Use pod for route fix measurement."*
> The pod path was blocked by bandwidth (another session's upload saturated the dev box's uplink: ~0.07 MB/s, > 12 h
> for the 12 GB of navtest frames + DBs). The dev-box GPU lock freed at 12:55 and the queued pipeline ran the inference
> there (597 s).

## Result (MEASURED; NAVSIM v1.1 PDMS, open loop, T1 tier; navsim_v1 rule, A7 repair ON, route fix ON)

All 12,146 navtest tokens, 136 logs. Log-cluster bootstrap (2,000 resamples, the parse tool's setting).

| | PDMS [95 % CI] | NC | DAC | EP | TTC | C | DDC | picks scoring 0 |
|---|---|---|---|---|---|---|---|---|
| final model, route fix OFF | 83.24 [81.77, 84.66] | 97.64 | 93.72 | 75.53 | 92.99 | 99.32 | 96.93 | 983 |
| **final model, route fix ON (adopted)** | **84.46 [83.18, 85.77]** | 98.13 | 94.71 | 76.70 | 93.69 | 99.93 | 97.61 | 837 |

- **Paired, ON − OFF:** **+1.22 [+0.59, +2.07]**, separated (log AND drive scopes). 1,178 tokens better, 1,039 worse, 9,929 tied.
- This supersedes the interim +1.05 [+0.52, +1.70] (84.28), which had used OFF rows on the 341 remaining changed tokens.
- **Same tokens, floors:** HUMAN 94.55, STOP 61.82, CV 20.65.
- **Reference:** DriveZero DINOv3 ViT-L 94.55 (Table A13; see `../../REVIEW_7_GAP_TO_PAPER.md`).

## Four families (MEASURED, families6 stage 1, the same 12,146 windows, OFF → ON)

- **Longitudinal (better):**
  - speed MAE 0.927 → 0.889 m/s; speed bias −0.298 → −0.328 m/s;
  - along-track MAE 1.337 → 1.258 m;
  - acceleration MAE 0.643 → 0.580 m/s²;
  - progress ratio vs the human 1.002 → 0.980;
  - distance keeping UNAVAILABLE: no lead track is supplied on navtest.
- **Lateral (better):**
  - heading MAE 2.97° → 2.72°;
  - yaw-rate MAE 3.77 → 3.23 °/s;
  - curvature MAE 0.0278 → 0.0237 1/m;
  - cross-track MAE 0.440 → 0.426 m.
- **Tactical, κ (better):**
  - lateral decision 0.834 → 0.848;
  - longitudinal decision 0.453 → 0.459;
  - 5-way manoeuvre 0.689 → 0.701.
- **Strategic: UNAVAILABLE on navtest.** The benchmark has no strategic term, and the harness has no NAVSIM strategic label builder yet (`4_families.json` gives the reason in full).

## Provenance and gates

**Seam composition:** `compose_full_seam.py` → `data/refe_navtest/seams/refe_navtest_final_routefix.npz` (md5 in `routefix_full/seam_md5.txt`), from three parts:
- Amendment 9's 3,045 confirmation rows;
- the adoption run's 341 rows;
- the OFF seam's 8,760 unchanged rows.

**Gates:** the changed set equals confirm ∪ rest; controls are 24/24 and 8/8 bit-identical with the OFF rows; fingerprints match on every row.

**A pipeline bug, fixed:**
- The chunk runner wrote its token list over the registered `a9/tokens_adopted_rest.json`, so compose first failed on KeyError `rest`.
- The file was restored from git (blob `461cd84c`, tokens verified identical).
- `run_a9_chunked.run_chunk` now writes `.chunk.json`.

**Seam exit code 1:** this is the frame-control gate on this batch's 25 frame-control tokens, the same declared DB-vs-OpenScene pose disagreement as the full navtest (max 6.18 m). Rows were produced for 349/349 with 0 misses.

**Harness:** three attempts were aborted by its RAM guard (other sessions' jobs held the dev box's memory); attempt 4 PASSED (12,146 / 12,146).

Files in `routefix_full/`: `3_parse.json`, `4_families.json`, the seam report, the per-token CSV (xz).
