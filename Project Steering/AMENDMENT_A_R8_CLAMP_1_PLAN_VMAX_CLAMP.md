# AMENDMENT A-R8-CLAMP-1 — refcv8 is evaluated WITH the hard speed cap from here

**Registered by the Master Mind 2026-10-18 ~19:20 Berlin. This commitment comes BEFORE any clamp-ON number on real windows
exists.** The only clamp numbers in existence are the audit's synthetic rig rows: 4 of 4 above-ceiling rows, 14.6–17.4 m/s
against 8.33 m/s (`Project Steering/audits/2026-10-18-refcv8-v7f-fixes/AUDIT.md`, de4e15b).

Amends: `SPEC_REFCV8_DRAFT.md` §5 (BAR-R8-E1 / E3 / NR) and the milestone battery
(`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/`).

## 1. Why

* PI decision 2, 2026-10-18, verbatim as landed in de4e15b: *"assure that refcv8 includes the fixes"*.
* The audit found refcv8's R1 equivalent MISSING. The max-speed ceiling only FILTERS the fan
  (`refcv6_selection.py:216`, `refc_v3.py:2634`). A row with no compliant candidate keeps the whole fan, and its emitted
  plan can exceed the fed ceiling.
* The fix is `cfg.plan_vmax_clamp` → `refcv6_selection.cap_plan_to_limit`. It is a per-interval projection of the SELECTED
  plan under a 2.5 m/s² braking envelope. Headings are kept, and non-binding rows come back bit-identical. It runs in the
  eval forward only.
* AUDIT.md:24 assigns the choice of carrying it to the PI / Master Mind. The PI asked for the fixes, so refcv8 carries it.

## 2. What changes — committed now

1. **Primary refcv8 rows.**
   * From the next refcv8 GPU battery onward (at the latest the final checkpoint, step 48,852), every refcv8 row is
     produced with `plan_vmax_clamp` ON, using the landed function, not a re-implementation.
   * The clamp-OFF rows on the SAME windows and the SAME DDIM draws are reported beside, for continuity.
2. **Bars.**
   * BAR-R8-E1, E3 and NR are read on clamp-ON refcv8.
   * The clamp-OFF verdict is printed beside them and cannot override.
   * The NR **reference** side is NOT decided here. It stays the open PI bar-instrument question `D-R8-NR-INSTRUMENT`
     (ceiling ON vs OFF on refcv7-50,400).
3. **Banked milestones (5k–30k).**
   * A clamp-ON read may be computed post-hoc, with the landed function applied to the banked emitted plans.
   * This is allowed ONLY on a surface where it is shown from source that no emitted plan feeds any later model input
     (no self-action feedback across windows or segments).
   * A surface where that is not shown is marked **NOT EXACT** and decides nothing.
4. **The R1 gap is reported at every milestone, in every read:**
   * the **exempt-row rate**: the fraction of windows whose emitted (clamp-OFF) plan exceeds the fed ceiling on the 0.5-s
     waypoint grid, with its size in m/s;
   * the clamp's **binding rate**;
   * the rate of rows with `v0` above the ceiling.

## 3. Known-value controls (must pass before any clamp-ON number is used)

* **(K1) Inert limit.** The post-hoc pipeline with the limit set to `+inf` on every row reproduces every banked row
  exactly: delta, CI, n.
* **(K2) Flag-off parity.** The GPU eval tree carrying the clamp code, run with the flag OFF, reproduces a banked dump
  bit-identically on a subset of at least 3 episodes. Only then is its clamp-ON roll admissible.
* **(K3) Mispairing.** Limits rolled by one episode block must change the binding set. Otherwise the read was not reading
  the per-window limit.
* **(K4) Post-hoc vs forward.** On that same subset, the post-hoc clamp equals the eval-forward clamp on the exact
  surfaces.

## 4. What this does NOT do

* It does not change training: the clamp is inference-only, and the live run's argv is untouched.
* It does not move a registered verdict that is already banked. The 5k–30k verdicts stay as banked, as early reads.
* It does not decide the NR reference instrument.
* Single training seed. Tier T1 open loop.

## Addendum 2026-10-18 ~20:40 Berlin: K4's tolerance, fixed BEFORE any K4 number exists

§3 (K4) says the post-hoc clamp "equals" the eval-forward clamp. The two run the same float32 arithmetic on two devices
(GPU forward vs CPU post-hoc), and 1e-6 m is below float32 spacing for any waypoint beyond ~8 m. "Equals" therefore means:

* every value is finite, AND
* max |d| ≤ 1e-6 m OR max float32 ULP over all rows ≤ 4.

Four ulp is ≤ 6e-5 m even at 256 m. A wrong clamp moves a waypoint by centimetres: the post-hoc S2 shift was 0.063 m mean.
Pinned by `code/test_r8_clamp_k2k4.py::test_k4_ulp_band_is_the_registered_criterion` in the milestone-battery package:
1 ulp PASS, 8 ulp FAIL, 1 cm FAIL, and the mutation ULP_TOL = 2 goes RED.
K4 has never run (its queue starts 30 min after the refcv8 30k queue ends).
