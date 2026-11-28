# TF-N4 A1 code fix: label construction uses the nuPlan-DB ego, scoring keeps the harness metric cache

Headline: **READY** (C-DERIV 200/200 at 0.0 on real navtest tokens, tests green, mutation RED). MM actions: re-pin the two changed files (see Decisions), land, rerun the build.

## Cause (MM diagnosis, `F:/tanitad_scratch/n4nt/diag/cderiv_route.json`)
Void run: labels re-derived from the metric-cache ego (`Ctx.from_parts`) deviate from banked N2-R by max 6.36 mm. N2-R labels were built from the nuPlan-DB ego (`sc.initial_ego_state`).

## Change (files in this folder; diffs `n4_geom.diff`, `n4_nt_build.diff` vs run-tree `n4/`)
`n4_geom.py`
- `LabelEgoMissing` exception; `Ctx.__init__(sc, parts=None, label_ego_fn=None)`; `Ctx.from_parts(parts, label_ego_fn=None)`.
- `Ctx.label_ego()` returns `(ego, src)`: DB route (sc given) = scenario ego, "db"; cache route = `label_ego_fn()` (lazy, cached once per Ctx) or **raises** `LabelEgoMissing` (no provider, or provider returns None); `force_cache_label_ego` (smoke mutation) returns the cache ego, "cache".
- `Ctx.labels()` passes that ego to BOTH `iterate20` and the `tracked8` closure given to `R.solve`; sets `label_ego_src`. `score`, `transparent`, `ext_dist`, `ego_frame`, drivable union, return vector keep the cache `self.ego`. `n4_repair.py` is untouched (it only calls `cx.labels`).

`n4_nt_build.py`
- `db_ego_fn(dbs, lg, tok)`: lazy `NS.build_scenarios_for_log(dbs[lg], [tok])` -> `scenario.initial_ego_state`; raises `LabelEgoMissing` if the log has no DB or the token is not returned.
- `ctx_stream` (cache route) builds `Ctx.from_parts(..., label_ego_fn=...)`; the DB is fetched only when `labels()` runs, i.e. C-DERIV tokens and tokens with builder DAC 0 (and `repair_all` smoke).
- `cmd_derive`: row field `label_ego` ("db" | "cache" | "none" = no label built); `except G.LabelEgoMissing: raise` before the generic handler, so a missing DB kills the shard loudly, never an error row or a cache fallback.
- `merge`: new control **C-LABELEGO** (read-back): every row that built labels (C-DERIV tokens + builder DAC-0 tokens) has `label_ego == "db"`; counts in the build report; part of the all-controls verdict.
- New smoke-only mutation `--mutate label_ego_cache` (needs `--smoke-allow-mutation`, chain never passes it) and `--cderiv-only` (derive only the 200 C-DERIV tokens, base score + label re-derivation, no repair arm; chain never passes it).
- Not changed: `n4_nt_chain.py` (its argv guard checks `--mutate` only; `--cderiv-only` is not in any chain argv).

## Tests (venv `C:/Users/Admin/navsim-crun/venv`, CPU, env of `n4_nt_chain.env(run)`, copy of `chain/` with the two patched files overlaid)
| run | result |
|---|---|
| `test_n4_a1.py` (8 new: labels use label_ego in iterate20 AND tracked8; cache route without provider refuses; provider None refuses; DB route uses scenario ego; mutation forces cache ego and reads "cache"; provider called once; missing-log refuses; mutation registered) | 8 OK |
| `python -m unittest test_n4_chain` (29) + test_n4_a1 | **37 OK**, 14.5 s |
| mutation of the unit test target (`labels()` back to `self.ego`) | **5 of 8 FAIL** (RED) |
| `test_n4_repair.py` (5, DB route) | **NOT RUN, UNVERIFIED**: needs `N4_ROWS` (RJ5 rolls/scores pack, `F:/tanitad_scratch/n4dac/rows`), absent on this box. `n4_repair.py` is byte-identical to the run tree; the DB-route semantics of `labels()` are unchanged (`self.ego` is the DB ego, `test_db_route_uses_scenario_ego`). |

## C-DERIV on real navtest (MEASURED; derive `--route cache --cderiv-only`, 200 smallest-sha1 tokens, bar 1e-9; no repair arm run, no N4 count)
| arm | n | rows with error | max dev (m) | median | n <= 1e-9 | n == 0.0 | `label_ego` |
|---|---|---|---|---|---|---|---|
| patched, DB ego | 200 | 0 | **0.0** | 0.0 | **200** | **200** | db 200 |
| MUTATION `label_ego_cache` | 200 | 0 | 6.358e-3 | 2.74e-5 | **0** | 0 | cache 200 |
The mutation reproduces the void run exactly (max 6.36 mm, median 2.7e-5, 0/200), so C-DERIV (and the C-LABELEGO read-back) go RED on real tokens when the label ego is the cache ego. Rows: `raw/cderiv_db_real.jsonl`, `raw/cderiv_mutation_cache_ego.jsonl`; drivers `raw/run_cderiv.py`, `raw/summ.py`, `raw/run_tests.py`.
The full `merge` (C-LABELEGO in a build report) was not run (needs the 12,146-token build); C-LABELEGO is exercised only by its code path and the per-row `label_ego` read-back above. UNVERIFIED end to end until the rebuild.

## ESTIMATED extra wall time (DB ego fetch)
Label tokens from the void report (`n4_build_report.json`, `dac0_tokens_builder`): **116** DAC-0 tokens + **200** C-DERIV tokens, overlap 0 (from the void derive rows) = **316 tokens** needing the DB ego.
MEASURED: mean 7.96 s/token (median 5.35 s, max 96 s; first token of a log is the slow one) for fetch + label + base score, vs 0.53 s/token on the cache route (mutation run, same tokens) => fetch ~ 7.4 s/token. ESTIMATED extra: 316 x 7.4 s = ~39 min CPU, ~20 min wall at 2 processes (~0.65 CPU-h). Against the void build's 0.57 CPU-h this roughly doubles the build. Fetch is per token, not batched per log (a per-log batch would cut it but was not needed).

## Decisions for the MM
1. `n4_geom.py` and `n4_nt_build.py` are PINNED files (`N4_PINS.json`, chain guard `a_pinned_file_edited_is_refused`): landing these copies into `chain/` requires regenerating the pins (`n4_make_pins.py`) and the chain test file; this is a change to the frozen build tooling, not the card (card numbers untouched).
2. Do not use `--cderiv-only` or `--mutate` in the chain.
