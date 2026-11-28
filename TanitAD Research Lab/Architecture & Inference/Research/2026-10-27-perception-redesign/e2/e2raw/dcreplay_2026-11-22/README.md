# dcreplay_2026-11-22 -- offline replay of the K2DC map-dose controller (MM 4a3d05)

Banked from `E:/Projects/TanitAD-artifacts/refcv9_e2/dcreplay/` (written 2026-11-22 ~12:30-12:45Z). Register row: `H-R9-E2-DOSECTL` in
`Project Steering/GOALS_AND_CLAIMS.md`. Pre-registration that uses it: `../PREREG_K2DC02CL_2026-11-22.md`.

* `r9e2-K2DC02.tsv` (24 rows, K2DC02 seed 0) and `r9e2-K2DC02s1.tsv` (22 rows, K2DC02s1 seed 1, probes up to step 5500 -- the run was
  still training): the logged `step, dc_w_before, dc_w_eff, dc_share_read, ..., dcp_trunk_norm_map_hires, dcp_trunk_norm_total,
  dcp_trunk_proj_map_hires` rows of each run's metrics.jsonl.
* `dc_replay.py` -- PRE-COMMITTED (its header is the record): metric, candidates (DB control, FIX(w), EMA x DAMP), selection on K2DC02,
  K2DC02s1 held out, two controls that void the replay if they fail. Output `replay.json`. Imports the tip's
  `stack/tanitad/train/map_dose_ctl.py` (tip in `tip.txt`) for `recover` / `share_at` / `solve`.
* `dc_replay_bounds.py` -- POST-HOC (not pre-committed): the dead-beat rule with narrower bounds. Output `replay_bounds.txt`. The clamp
  [2.5, 3.5] row comes from here and is tested by the K2DC02cl training run, not by this replay.
* Run: `python dc_replay.py <map_dose_ctl.py> r9e2-K2DC02.tsv r9e2-K2DC02s1.tsv > replay.json` (same arguments for the bounds script).
* LIMIT (named in the header): each probe's statistics come from the model trained along the LOGGED weight sequence.
* Bytes: `replay.json`, `replay_bounds.txt` were written with CRLF on the dev box; the copies here are LF, `SHA256SUMS.txt` is over these LF bytes. The CRLF originals
  stay on E: with their own SHA256SUMS.txt.
