# PRE-REGISTRATION — MM-E4: the drift-attack tiny ladder (innovation-SIGReg × frozen-teacher)

**Written** 2026-08-29 ~19:30, before implementation and before any compute · **Author**
Master Mind · **Status** PREREGISTERED — ⛔ LAUNCH GATED ON THE PI's GO (interest expressed,
approval not yet given) · **Hypothesis** MM-E4 · **Tier** T0-DIAGNOSTIC throughout.

## Why

The drift attractor (~0.67–0.70 trainable-line; est. ~0.3 self-generated above the ~0.39
world-smoothness null — Paper §14) survives every measured knob (freeze/k/omega/O14/EMA).
The PI ranked the attack list (2026-08-29); this ladder runs the top two mechanisms and
their combination — they are mechanistically orthogonal (one constrains the DYNAMICS'
innovations, the other removes the MOVABLE TARGET), so the combined cell is informative,
not redundant.

```yaml
hypothesis: MM-E4
rig: v7-tiny 2k ladder (E-DEC-9 pattern), o14fut30k-line base (v6.0 vocab, steer cond),
     one variable per cell vs base; 2k scale-facts bind (vs-base RELATIVE reads;
     drift ~0.45 at 2k; cos the most sensitive axis)
cells:
  L0: base            # the o14base-class incumbent line at 2k (exists or re-run)
  L1: innovation-SIGReg   # O6's sketched test applied to Δz − g(z_t); g = per-batch
                          # ridge, FIT-SPLIT only (never tuned on scored windows);
                          # implementation reuses the existing SIGReg machinery
  L2: frozen-teacher      # --o5-target frozen (targets from the FIXED distill-init
                          # teacher via a learned projection; resync logic reused)
  L3: L1 + L2             # the combination
  rider: L2 + pred-depth+2  # the IWM capacity cell, free rider, exploratory only
one_variable_per_cell: verified by config diff BEFORE any read
controls: the ladder's standing set (constant reads zero; time-shuffled g must
  destroy L1's constraint — the deliberate-regression arm; seed band from the
  existing 2k replicates ~1.5 %)
reads (in order):
  drift        PRIMARY — vs base, beyond seed band
  cos/nrmse    ⛔ PROTECTED ASSET (E-DEC-69's lesson): any cell that pays >5 %
               cos is FAILED regardless of drift
  absorption   slim marginal must stay inside null
  participation rank gate (val-side)
outcomes (committed):
  DRIFT-DOWN:   any cell lowers drift beyond band with protections held
                => that cell earns the 30k confirmation arm (ONE arm, the winner)
  FLAT:         no cell moves drift at 2k => the 2k scale caveat applies (the
                attractor is a 30k phenomenon) — the BEST-protected cell may
                still earn a 30k arm on PI approval, argued from mechanism
  REGRESSION-FAILS: if the time-shuffled-g arm shows the same "gain" as L1,
                L1's read is an instrument artifact — read NOTHING from L1
  PRED-PAYS:    protections violated => cell dropped, numbers stated
cost: implementation ~2 days (L1 ~1 day: SIGReg input swap + ridge-in-loop;
      L2 ~1 day: --o5-target frozen + projection head + 6-test suite mirroring
      the EMA pattern) + 5 × ~35 min Thor at 2k + probe reads on the 4060;
      30k confirmation ~8.2 h Thor for at most one winner
schedule_constraint: fits inside v7f's data-preparation window; ⛔ yields Thor
      to the epcache build and the v7f launch — v7f is never delayed by this

---

## AMENDMENT A — PI APPROVAL + the #6 cell + time-minimization (2026-08-29 ~19:50, before implementation)

**PI, verbatim intent:** "let's do your recommendation regarding the drift measures and
add 6 to investigate; let's try to minimize the experimentation time." ⇒ **LAUNCH GATE
LIFTED** for the ladder as amended.

1. **Base cell corrected to the RECIPE line**: base = `o14fut10` (the EXISTING 2k arm
   with `--w-o14 1.0` — drift 0.4733 / nrmse 0.9746 banked). Testing drift levers off
   the O14 recipe would re-import the interaction blindness (the A7 lesson); the base
   also costs ZERO new compute.
2. **New cell L4 = #6 cross-view temporal targets, time-minimized variant**: the target
   latent is computed from a random AZIMUTHAL CROP of the target frame only (input path
   unchanged, no cache changes) — on a cylindrical projection a horizontal crop is a
   pure FOV restriction (column ∝ azimuth), so the view change is geometrically clean
   where a pinhole crop would not be. Implementation ~half a day instead of ~2.
3. **Ladder = 5 new arms** (L1 innovation-SIGReg · L2 frozen-teacher · L3 = L1+L2 ·
   L4 azimuthal-crop targets · rider pred-depth+2), each = the o14fut10 line + one
   variable, ~35 min each ⇒ **~3 h Thor total at 2k**.
4. **Scheduling rule (time-minimizing, v7f-protecting)**: whichever becomes ready
   first takes Thor — the drift arms (once implementation lands) or the B1 epcache
   build (once the validated manifest lands); the other follows immediately. The two
   never run concurrently (CPU-decode contention), and the v7f launch is never delayed:
   if the manifest arrives mid-ladder, the ladder yields after the current arm.
5. All committed reads/outcomes/protections of the base prereg unchanged, including
   the E-DEC-69 protected-asset rule and L1's time-shuffled-g regression arm.
