# Master Mind notes on the beside package (B1, B2, B3)

## 2026-11-11 ~18:12: B1 / B2 review
* Re-run by the Master Mind: `tests/test_decod_wmact.py` 10 passed (5.2 s, CPU); `CONTROLS_B1B2.json` reads `all_pass` true, tier T0 (MEASURED).
* **Decision, B2 cells the window cannot hold.** On arms with level strides (3, 5), the WMACT cell L3 / k3 needs more frames than the 16-frame training window carries, so it does not exist there (MEASURED in the agent's smoke rows). B2 stays as `BESIDE_RULES_SLOWWM.md` wrote it: the WMACT statistics unchanged on the same windows, and a cell that cannot exist is listed with n = 0 and its reason, never filled. Reading L3 / k3 on longer windows would be a different protocol and is not done. B2 has no bar, so this changes no outcome.
* **B1 real rows wait for the train stream.** The 141 WMACT train clips are streamed read-only from box 2 (`theory/wmact/HANDOFF_WMACT_TRAIN.md`), and their basis coordinates need the trained SLOW / PCA-B bases, which exist only after the SLOWWM chain's basis step. So `decod_build_train.py` runs with the SLOWWM read (2026-11-12), not before. Until then B1 runs only as `--smoke` (an episode-disjoint half split of eval-141), labelled as such.

## 2026-11-11 ~18:55: B3 review
* Re-run by the Master Mind: `tests/test_top64.py` 17 passed (30 s, CPU); `CONTROLS_B3.json` `all_pass` true (MEASURED). No leftover worker process (checked by command line).
* **Accepted reading where the rule is silent:** with an encoder window K = 3, z1 depends on three frames of u. J̄ shifts the top-64 u columns on all three frames together (the direction applied to the clip, the way a basis direction moves every frame). That joint J̄ defines S for the B3 read. The newest-frame-only J̄ is stored beside, with its angle to S. The 4,096 train frames are drawn with `default_rng(0).choice` from the ok frames of the `slow_basis.py cache --split train` output, with the basis sha and every clip sha checked.
* The runner bug the agent caught (the deciding runner's `run_stage` maps its own full-width work, so a naive wrapper would label full-width units as B3) is fixed and pinned by a test.
* On the planted arm every top-64 vs full plan difference reads the same 3.4393. A box-limit saturation of one action channel is a guess, not checked; read it on the first real SLOW arm before any B3 row is quoted.
* Cost ESTIMATED 4.5-5 h CPU per SLOW arm (13-15 h for three), from a 64-window smoke that shared the box with stray workers. B3 runs after the deciding read, never before it.
