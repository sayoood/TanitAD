# MANIFEST -- stream YSEL2 (H-SEL-1 fresh-token confirmation), 2026-11-07

Every artifact lives ONLY in the place named (nothing staged; the MM lands it). `$P` = `E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107/`.

| artifact | where | role |
|---|---|---|
| `PREREG_SEL_CONFIRM.md` + `PREREG_SEL_CONFIRM.sha256` | `$P` (one place) | the frozen pre-registration; MM lands before any read |
| `CODE_PIN.json` | `$P` | sha256 of the pinned code + token lists (its own sha is frozen in the prereg) |
| `POWER.md`, `power.json`, `power_sim.json` | `$P` | power + token plan + leakage bound + compute plan |
| `tokens_confirm.json` (sha `013b6771...`), `tokens_smoke.json` (sha `8d510492...`) | `$P` | the population (5,877 / 75 logs) and the 6 smoke C3 tokens |
| `tokens_confirm.SUPERSEDED_plumbing_seen.json` | `$P` | the pre-exclusion list (kept for the record, never read by the chain) |
| `make_confirm_tokens.py`, `make_pin.py` | `$P` | builders |
| `selc_rule.py`, `selc_gates.py`, `selc_analyze.py`, `selc_dump.py`, `selc_score.py`, `selc_chain.py` | `$P` | the frozen rule copy, preflight gates, the read, the GPU dump, the harness driver, the chain |
| `selc_power.py`, `selc_power_sim.py` | `$P` | power scripts (CPU, banked data only) |
| `gate_preflight.json`, `gate_ana_replay.json` | `$P` | G-RULE / G-PIN / G-ANA, green at freeze |
| `HANDOFF.md`, `MANIFEST.md` | `$P` | this hand-off |
| chain outputs (as they land): `CHAIN.log`, `stages/`, `markers/`, `logs/`, `dump/` (chunks, `dump_confirm.npz`, `smoke_identity.json`), `score/` (`*_gates.json`, `verdict_columns.npz`, `table_confirm.npz`), `CONFIRM_RESULT.md`, `confirm_result.json`, `CONFIRM_TABLE_APPENDIX.md`, `confirm_table_result.json` | `$P` | results; `dump/chunks/` is 75 files (do not copy to the repo; `dump_confirm.npz` carries the same data) |
| single-proposal seams, token chunk files, 469 harness run dirs | `F:/TanitAD_selc_20261107/{seams,tok,score}` (data dir, NTFS) | heavy intermediates, never for the repo |
| plumbing leftovers | `F:/TanitAD_selc_20261107/_plumb/`, session scratchpad `plumb/`, `plumb2/` | CPU plumbing only (labels `refe_selc_*`, never read) |

Integration to escalate: land `$P` minus `dump/chunks/` and the F: data into the repo next to YSEL (`.../2026-09-20-refe-plan/raw/2026-11-07-selection-confirm/`);
update `H-SEL-1` (and, at `selc_full.DONE`, `H-SEL-LANE-1`) in `GOALS_AND_CLAIMS.md` from the tip. nuPlan-derived per-token arrays are admissible in the
repo by the PI's ruling; `dump_confirm.npz` (~0.15 GB) may be better banked on HF than in git -- MM's call.
