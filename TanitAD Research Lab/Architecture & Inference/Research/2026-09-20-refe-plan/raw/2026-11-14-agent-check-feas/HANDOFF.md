# HANDOFF: agent check (D-VDAC-6) feasibility, 2026-11-14 (REFe Research Lab teammate; staged, nothing committed or pushed)

HOLD = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_feas/`. Files: `FEASIBILITY_AGENTS.md` (inventory, label design, ceiling plan), `PREREG_AGENT_CHECK_DRAFT.md` (the card, unfrozen), `ceiling_fitpack.json`, `auc_and_conversion.json`, `p6_agent_blindspot.json`, `scripts/` (S0-A scripts, probes, hand-copied timing output, `found_logs.txt`). Land into `PK/raw/2026-11-14-agent-check-feas/` (repo paths <= 200 chars).

## What was found (all MEASURED, two probes per absence)
* Agent annotations exist: nuPlan DB `lidar_box` (E:, 20 Hz, global frame, no confidence filter in the devkits), NAVSIM v1.1 `metric_cache` for navtest (12,146 pickles on F:, observation = 51 occupancy maps at 10 Hz, 5 s, logged futures), `navsim_logs` anns (navtest only). Navtrain metric caches/parts: none on this box (pod only); 48 of the 101 fit-pack logs have a DB locally.
* The ceiling needs no GPU: `F:/tanitad_scratch/p6fit/fit_pack.npz` (P6 logits + per-proposal NAVSIM sub-scores, 8,489 navtrain sets). TRUE agent veto +1.215 PDMS (TTC +1.181, NC +0.219); P6's own NC/TTC heads miss confidently (98.6 % of failing picks have own p_ok > 0.5; own top-5 AUC 0.787).
* One-token costs: `score_plans` 64 proposals 0.52 to 0.73 s warm; raster of 8 future steps 0.034 s; DB window query 0.070 s.
* The `selq2` regret split is banked: `F:/tanitad_scratch/selq2read/p6_e3_regret_split.json` (MEASURED, not inherited).
* Correction to report: my first ceiling script skipped the sigmoid and printed 91.18, which equals P6's navtest number by coincidence; the right navtrain pick PDMS is 94.49. Corrected script and JSON are the banked ones.

## Next step for the MM (in order)
1. Land the HOLD (docs + JSON + scripts) into the repo; read the draft card, decide Q-A..Q-D (section 1 of the card).
2. Build and run **S0-B** (CPU, ~0.3 to 0.5 h): the rasters and the footprint readout on the 3,976 local sets with R-true / R-t0 / R-cv / C-SHUF / C-REG. The scripts do NOT exist yet. This is the cheapest experiment that decides whether a head is worth training. Not keyed.
3. Only on S0B-GO ask the PI for ~8.3 to 9.6 L40-h (S1). Nothing was requested.

## Deletions (literal paths, own scratch, 0.07 MB total)
`C:/Users/Admin/agck/` files: a5.md (38,464 B, re-creatable via `git show`), auc_conv.py 2,629, blind.py 2,326, ceil_fitpack.py 3,017, found_logs.txt 1,918, inspect1.py 1,558, probe_db.py 1,502, raster_time.py 1,490, time1.py 1,752, tokcheck.py 892, tokcheck2.py 1,095, tokcheck3.py 670 (every script and found_logs.txt were copied into `scripts/` first); then the empty directory `C:/Users/Admin/agck`. No deletion was refused. No file was written under E:/Archive, F:/Archive, rj5db or rj5nt. `Keys.txt` untouched.

## Open questions
* Q-1: did P6 train on the 77 fit-extra logs? Held-out (24 logs, 1,705 sets) is the clean column until answered.
* Q-2: does P6's trainer feed only one time step? Absence found in `config.json` meta and `model.py` (two places); `train.py` not read.
* Q-3: the 53 missing DBs: which download chain lands them, and when? (S0-B on the full 8,489 waits for it, or runs on the pod.)
* Q-4: is a pose-only footprint readout admissible (card Q-C), and if S0-B reads STATIC-limited, may the next head see motion (Q-D, PI)?
* Q-5: navtrain-to-navtest scaling (1.45 to 1.93 on pick-failure shares) is ESTIMATED; the navtest ceiling 2.3 to 2.9 PDMS rests on it.
