# DEFINITIONS_H: how AGENT H reads the frozen card (written 2026-11-06 13:50, BEFORE any contrast was read; the card is untouched, sha256 c782928e...)

These fix every place where the card leaves a quantity to the implementer. They are not amendments; a reader who disagrees can recompute all of them from `raw/arrays/`.

1. **The hold executor.** `hermite_frames` (the executor) already compares the Hermite tangent norm `nrm` (metres per 0.2 s interval) with a threshold and holds the heading below it;
   the base threshold is `EPS_M = 1e-3`. `--exec-eps 0.1` sets that same comparison to 0.1 m per 0.2 s (0.5 m/s, the reader's own standstill mask). "Executed step" in the card = this tangent norm
   (the 10 Hz frames carry it as `v * 0.2`). Positions, speeds, replans, windows, causal bucket are not touched (tool patch: `code/apply_exec_patch.py`, a 14-line diff to the LF6 tool 949c4b7a).
2. **Arms and flags.** C0 = the RECOVER-FT arm C0 (`v8f-recover/M2b_s{0,1}_recft_C0`), the banked `C0_s*` flags: `--smooth sg72 --scene logged`, replan 10, nav deadreckon, periodic,
   causal bucket, ticks + heads. DAGOACC = `v8f-dagoacc/M2b_s{0,1}_dagoacc` with the SAME flags (the banked DAGOACC T1 run is unsmoothed; this one is not, so it is comparable to C0 here, not to that row).
   P10 = the floor the tool rolls in the same process (the prior alone), under the same executor as its model.
3. **Own floor.** Each run's P10 side is ITS OWN re-rolled floor (`extract_smooth.py --floor own`); under hold the floor's executed history changes, so the banked floor cannot be the comparator.
   The extractor's gross-mismatch guard against the banked floor is disabled for these runs (`--floor-guard-m 1e9`); the same-windows check stays.
4. **E and dE.** E_x(seed) = the reader's M1 row (`M1 crawl share`, pool "masked") of C0 minus P10 under executor x. dE = E_hold - E_base. Interval: paired episode-cluster bootstrap, B 2000, seed 0
   (`gapdecomp.decompose` on d_hold = a-b under hold against d_base = a-b under base, i.e. the same episode draws for both executors). "Separated-negative" = hi < 0.
5. **Outcome table.** Applied literally: 1 needs separated-negative at BOTH seeds and |dE| >= 0.5 E_base at both; 2 = separated-negative at both with |dE| < 0.5 E_base at either, or separated (negative) at one seed only;
   3 = separated at neither. **A separated-POSITIVE dE is outside the card's table** (the card names no outcome for it); it is reported as `UNCOMMITTED: hold INCREASES the excess`, not forced into a bin.
   E_base <= 0 would make "0.5 E_base" undefined and is reported the same way.
6. **The diagnostic.** "Launch windows" = the AUDIT's standstill-launch class: logged v0 < 0.1 m/s and logged 6 s extent |g[29]| > 10 m (n = 18 on eval-141); the gapdecomp class L (v0 < 6 and accelerating, n = 327)
   is reported beside it. "Executed-history yaw rate" = the cache-step yaw rate `ch1` of the executed rows t0+1 .. t0+20 (the rows a replan tick can read; the last tick is at t0+20); a window is flagged when
   max |ch1| > 60 deg/s. Rows t0+1 .. t0+30 are reported beside. Shares are reported for the P10 floor and for C0, per seed, under base and under hold.
7. **G-ID.** The strict reading is tested in two forms: (a) SAME SESSION, the unpatched LF6 tool against the patched tool at eps absent and at eps 1e-3, bit for bit on the executed paths, the tick records and the
   exec dump; (b) the patched tool at eps 1e-3 on the full eval-141 grid against the BANKED C0 rollout (a different session, tool 0721fc01). (b) is reported with its measured deviation; thread order / batch
   composition can move floats at the 1e-3 m level, and that is stated, not hidden.
