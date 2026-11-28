# SHIP-R8 AMENDMENT 4b -- the H2cn stage clamped to refcv8's FED speed ceiling (Master Mind, 2026-10-26 ~05:03 CET)

Sealed BEFORE any H2CN_CAP8 position, cell or clause exists. Amendments 1-4 and 5 stand; nothing registered is edited.

## 0. Why (disclosed)
AMENDMENT 4 read **PASS** on every clause for `H2CN_DEPLOY8` (`out/ship_r8/SHIP_R8_H2CN_DEPLOY8.json`, verdict generated 04:37:45).
Its BESIDE compliance row (A-R8-CLAMP-2 s2.3, no clause reads it) shows the stage plans above refcv8's FED speed ceiling on
S2 **0.0163** / S6 **0.0169** of compliant windows (episode-cluster rate, full plan 0-6 s) vs RETIME_DEPLOY8 0.0055-0.0060 / 0.0049-0.0055
and the model's own plan 0 -- about 3x RETIME. refcv8 runs with the cap ON (PI 2026-10-18); a capped car's stage must not plan above
the ceiling it was fed. This amendment reads the clamped stage; it never relabels Amendment 4's verdict.

## 1. The arm `H2CN_CAP8`
Per window: `s8` = Amendment 4's registered H2cn progress (the frozen stage `H2CN_STAGE_FROZEN.pt`, sha256 `f293b7bd...6f32`,
`h2cn_progress` unchanged, running max included). Then, on the 8-slot grid `T = (0.5, 1, 1.5, 2, 3, 4, 5, 6)` s with `s_cap[-1] = 0`, `t[-1] = 0`:
`s_cap[k] = min(s8[k], s_cap[k-1] + lim * (t[k] - t[k-1]) * (1 - 1e-6))`, where `lim` = the window's FED ceiling exactly as the
compliance row reads it (`cur["lim"][rows]`); a non-finite ceiling leaves the window unclamped. Positions
`eval_polyline(polyline_geometry(plan8), s_cap)`, cast as `H2CN_DEPLOY8`'s. Because a chord never exceeds its arc, every
waypoint-interval speed on any sub-grid is <= lim.

## 2. Controls (a failed control voids the clauses)
* **K-CAP-ID:** with `lim := +inf` everywhere, `H2CN_CAP8` is bit-equal to `H2CN_DEPLOY8` (positions, both seeds, both surfaces).
* **K-CAP-COMPLY:** the compliance row's `exceed_rate_strict` for `H2CN_CAP8` is **0** on every set at both seeds (by construction).
* **K-CAP-MONO:** `s_cap` is non-decreasing on every window.
* Reported, no pass/fail: windows changed by the clamp (count, share) per surface and seed.
* All Amendment 4 controls re-run unchanged (K-REG, K-DISJ, K-RETIME identity, RETIME TAC identity, H2CN_DEPLOY8 reproduced bit-for-bit).

## 3. Clauses
SHIP-R8 (1)-(7) + (1b) on `H2CN_CAP8`, verbatim from Amendment 4 with the arm swapped; both inference seeds; compliant windows and the
compliance row as registered; OVERALL as registered.

## 4. Committed next steps
* **PASS** -> the PI chooses the shipped stage between `H2CN_DEPLOY8` (Amendment 4 PASS, ~3x RETIME's ceiling exceedance) and
  `H2CN_CAP8` (0 exceedance by construction); the Master Mind's recommendation is `H2CN_CAP8`. NavSim only with a separate go.
* **FAIL** -> name the clause; `H2CN_DEPLOY8` stands as Amendment 4 registered it, with its compliance row put to the PI.
