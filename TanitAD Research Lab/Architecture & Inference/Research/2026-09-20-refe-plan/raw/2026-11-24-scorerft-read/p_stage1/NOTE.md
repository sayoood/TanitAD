# SCORER-FT arm P: card read, Stage 1 (MM, 2026-11-27)

**Verdict: OP6 FAIL.** The decision term made the pick worse. Card: `PK/raw/2026-11-24-scorerft-pick/PREREG_SCORERFT_PICK.md` (sha256 `7c39a57b…`). Read tool: tip `e7834b36`.

All numbers are MEASURED, tier T1 (NAVSIM v1.1 PDMS, LQR-tracked, open loop against logged agents), on 12,146 navtest tokens from 136 logs. The estimator is the paired log-cluster bootstrap, 10,000 draws, seed 20260927. It answers "another draw of LOGS"; training variance was not measured, since the card accepts one seed per arm at Stage 1.

## Headline

| quantity | value |
|---|---|
| P − C, tracked PDMS | **−2.5453 [−3.0485, −2.0851]** → P1 FAIL |
| G1 (no separated-adverse sub-score) | FAIL: NC −1.2679 [−1.6040, −0.9488], DAC −1.7125 [−2.2496, −1.2262], TTC −4.0672 [−4.7763, −3.3829] |
| EP | +0.6393 [+0.0749, +1.1806] |
| DDC (reported, not a guard) | −0.9015 [−1.3097, −0.5112] |
| P / C | 90.13 [89.11, 91.09] / 92.67 [91.90, 93.40]; both BELOW the 94.55 goal |
| M3, selection regret P − C | +2.5156 [+2.0754, +3.0027]: the regret grew |
| M4, oracle-of-64 P − C | −0.0297 [−0.0966, +0.0369]: the generator is not damaged |

- **Controls:** all 18 PASS, including CTRL-DET (full-seam bit-identity of the rep seam) and CTRL-CLIP (CLIP-SAME occ 8/8).
- **F_REFe qualifier** (descriptive; the card's bars do not move): |P − C| > 2·F_REFe = 0.2316 (`F_REFE_QUALIFIER.txt`).
- **Four families** (`report_p_stage1.txt`; JSON `read_p_stage1.json`):
  - LONGITUDINAL: separated adverse on the lead-agent terms (time_gap_min, min_ttc, lead_viol, lead_progress_ratio) and on speed MAE / bias / RMSE, plus 25 more components.
  - LATERAL: 8 components reported.
  - TACTICAL: selection quality computed; goal-setting NOT COMPUTABLE (n = 12,146).
  - STRATEGIC: reported in full.

## Mechanism (descriptive; navtest is snooped, so nothing may be fitted on it)

`decomp/`, on P's own 64-proposal tables:
- Pick 90.13 vs oracle 99.32.
- Regret 9.19 = safety 8.09 (DAC 3.14, TTC 2.45, NC 2.35, DDC 0.14; 2.31 tracker-induced) + EP 1.10.
- C, on the same instrument: safety 4.20, EP 2.47.

This is the third measured instance (after S-RULE and SCORER-FT T) of a progress-ward selection push buying EP with safety. See `D-REFE-SAFETYMISS-1` and `PK/raw/2026-11-26-refe-next/PLAN_REFE_9455.md`.

## Committed next arm (card §9, OP6) and its blocker

The card's OP6 arm is **P-top8**: the same routed, gap-weighted term, anchored on the scorer's whole top-8 rather than the argmax. Cost: ~14.2 L40-h + the weight rule.

It is **BLOCKED**:
- It needs a GPU, and the PI has ruled no additional pods (2026-11-22). The REFe pod has ended.
- It needs a PI go.

The MM recommends not spending on it. The binding constraint is safety-boundary discrimination. L2 SAFE-HEAD (`H-REFE-SAFEHEAD-1`) attacks it directly, and the parameter-free L1 TRK-SCORE and L1b ENS-SCORE are queued.

## How the read ran (four attempts; infra only, no number changed)

| # | what failed | fix |
|---|---|---|
| 1 | the seam's score step aborted on the harness RAM guard | memory-gated relaunch |
| 2 | the rep seam died at 2,413 s on numpy `_ArrayMemoryError` (5.62 MiB). The dev box's **commit** limit was reached: Chrome held ~21.7 GB of commit, and the pagefile grew until C: was full. | relaunch gated on commit ≥ 8 GB as well (`relaunch_p_memgate2.sh`) |
| 3 | the relaunch was refused at preflight with C: 0 GB, partly because of a stray 3.97 GB HF `.incomplete` file in the MM's scratchpad | file deleted |
| — | the rep seam then waited **7 h** on a stale GPU lock left by a crashed L2 cache dump | lock moved aside once its pid was confirmed dead; a reaper now guards our own `refe-*` locks |
| 4 | the families step failed: `p5c_families.py` needs `raw/2026-11-12-instruments/fam_ci/`, which `sft_read_p.sh`'s tip export omitted | path added to the export list (`scripts/sft_read_p.sh`), and the same tip's `fam_ci/` exported. `fam_ci.py` sha256 prefix `8f7ada4e` = the tip's. |

## Files

- `VERDICT_p_stage1.txt`, `report_p_stage1.txt`, `read_p_stage1.json`, `F_REFE_QUALIFIER.txt`, `seam_ledger.json`, `tactical.json`, `strategic_family_P_C.json`
- `families/{famci_P_vs_C.json,families_summary.json}`, `plans/convert_report.json`, `pod/*` (config, argv, w0, summary, metrics, clip probe, pull record)
- `decomp/*` (`oracle_decomp.py` / `safety_miss_split.py` outputs on P's tables; table sha256s)
- `CHAIN.log`, `scripts/*` (wrapper and launchers)
- ⚠ Not landed:
  - `hf_get.py`: a helper that reads the git-ignored key file in place.
  - The per-arm `families/fam_{P,C,base}.json`, about 50 MB each, re-creatable from `pass1/`.
  - The navtest 64-proposal tables are banked on HF instead: `Sayood/tanitad-refe-p5c` `propscore_tables_2026-11-26/`, verified by sha256.
- Backlog: SCORER-FT T has no MODEL_REGISTRY row. P's row is added with this land.
