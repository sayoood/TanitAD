# HANDOFF A6-IMPL (2026-11-08) -- AMENDMENT_6 (sha256 073635ce...) applied to scorer + selector. CPU only, nothing staged/committed.

## 1. Scorer (`refe_behaviour_20261107/code/`)
`HumanPath` END extension = chord over the last 5.0 m of arc length of the MOVING part (raw 10 Hz samples with v >= 0.5; all of it if shorter; old rule only if the moving part has no length). New params `PATH_EXT_V_MIN`, `PATH_EXT_CHORD_M`; helper `_moving_chord_dir`; mutation marker `# MUT:pathext`. START extension unchanged (the amendment names only the end; say if you want it mirrored).
Tests 85/85 (3 new: noisy stopped human +-0.08 m zig-zag with noisy yaw and a control, chord unit test). `test_ch1_control_...bit_identically...` still matches the PRE-A3 golden hashes (only the two new echoed params are stripped before hashing) => scenes that never use the extension score byte-identically. Mutation check: ALL KILLED incl. new `pathext` site (revert -> the A6 test goes RED). `CODE_PIN.json` regenerated (24 files, `--check` OK, now carries `amendment_6_sha256`); previous kept as `CODE_PIN.before_a6.json`.

## 2. Selector (`a6/`, `a3/` untouched)
`a3_slots.py`: stop line = connector baseline START for every token (no polygon); station extension along the scorer's own `_moving_chord_dir` (imported, one definition, 2 Hz rear-axle window). Also fixed (A6-only, TL block only): TL stations used a path from `densify()` that can stop up to 0.5 m short of the last pose; they now use the path through the true last pose. Junction slots unaffected (J pools identical to a3). Legacy six slots: SUITE_CANDIDATES.json / _ALT.json sha256 identical to a3's (3d669b74 / 5633414d). The legacy `Path` class in select_suite_tokens.py (rear-axle, last-yaw extension) is NOT a stop-line station and was left alone.
Outputs: `SUITE_CANDIDATES_A6.json`, `MAP_FIELDS_A6.json`, `SUITE_PIN_A6.json` (+ `rows3_a6.pkl`, `audit_a6.py`). Rebuild: `python -I select_suite_tokens.py --select-only` (~7 min), then `python -I audit_a6.py` (~2 min).

## 3. Agreement selector vs scorer (13 tokens, `AGREEMENT_A6.json`)
Same line, same chord. Rear-axle: selector (2 Hz) minus scorer HumanPath (10 Hz): |diff| <= 0.002 m on 7 tokens, 0.04-0.16 m on 6 (max 0.163, 15715849), attributed to 2 Hz vs 10 Hz sampling of the chord/corners (not isolated per token). Shipped `stop_line_s` (box-centre) minus scorer rear-axle = -1.456..-1.462 m on all 13 (the COG offset 1.461 m).

## 4. Sim numbers (box-centre `stop_line_s`, ours / sim / ours-sim)
- 7a7ff4cc 76.15 / 76.08 / +0.07; b7c35b12 33.41 / 33.31 / +0.10: agree to ~0.1 m.
- 090d6a17 38.36 / 34.54 / +3.82; 15715849 63.00 / 61.10 / +1.90: **DO NOT AGREE**, and it is not a constant offset (not 1.46). Our lines lie 6.56 m and 5.82 m beyond the human's last position, 0.31 / -0.59 m lateral to the chord, consistent with the sim's "0.21 m from the continuation". Not explained by the t0-2 s start (travel there is 13.8 / 12.2 m). Open: the sim's path length or its line xy for these two. Needed from thor3: `stopline_check.json` line xy + path length for these two tokens.

## 5. Audit (`AUDIT_GAP.json`; gap = line station - front bumper at first stopped-on-red tick, human-safe)
- TL-G pool (n=54): p50 5.24, p75 5.61, p95 5.66, max 5.68 m. TL-S pool (n=58): p50 3.93, p75 4.02, p95 5.52, max 5.54 m. (Predicates apply the 6 m cap, so no pool member exceeds it.)
- Human-safe tokens that pass the predicate only with the cap lifted to 30 m (excluded by the 6 m cap, none changed by us): TL-G 77 tokens in 23 logs (min 6.47 m, max 28.3 m); TL-S 72 tokens in 12 logs (min 7.24, max 21.4 m). Full token names in the JSON. Cap not changed.
- The old "p50 5.58" is not quoted.

## 6. Candidate changes vs a3 (`DIFF_VS_A3.json`)
J-UL, J-R, J-S, TL-G: candidate lists AND ordered lists identical. TL-G pool 53->54 tokens (same 10 logs).
**TL-S changed:** pool 45->58 tokens (8 logs both). Same log (...51_veh-48_01248_01450) now yields `7c150adbe5225f20` (rank 3) instead of `f1992f61c13e5998` (a3 rank 4). New candidates: 090d6a17, c8d1d4dd, **7c150adb**, 7a7ff4cc, 15715849. Ranks 6-8 unchanged.
Bundle `bundle_a3` (rank <= 5 of the five A3 slots): **ADD token 7c150adbe5225f20** (its log is already in the bundle, new frames for t0 and t-2 s needed); **DROP f1992f61c13e5998** (no other role in the bundle's rank<=5 set). Nothing else changes. Not touched: bundle_a3. Stop line xy/stop_line_s of all bundle TL tokens change, so the bundle's MAP_FIELDS must come from `MAP_FIELDS_A6.json`.
