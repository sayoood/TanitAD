#!/usr/bin/env python3
"""Append this package's two self-corrections to RETRACTION_LOG.md (append-only, idempotent)."""
import io
import os
import shutil
import sys

LOG = ("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Project Steering/"
       "RETRACTION_LOG.md")
TMP = "C:/Users/Admin/feasdec/raw/_retraction_new.md"
MARK = "feasible-decode SPEC claimed a no-op that is true of the ratio and false of the component"

ENTRY = """

## 2026-09-05 — the feasible-decode SPEC claimed a no-op that is true of the ratio and false of the component; and "a scorer artifact, not a leak" was refuted by its own discriminator (Architecture & Inference FlyWheel)

Package: `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-feasible-decode/`.
Register: `D-FEASDEC-SPECERR-1`, `D-FEASDEC-STOPSTEP-1`. Both caught IN THE SAME TURN, by controls
the package ran on itself; neither reached a report.

### 1. RETRACTED — "any per-window monotone renormalisation of `along` moves rho by exactly zero"

`SPEC.md` §1 stated this as a *structural* fact and used it to argue in advance that three obvious
"reshape `progress`" proposals were provable no-ops. It ran the no-op **as a deliberate control**,
expecting a confirmation.

**MEASURED** (`raw/progress_rank_fix_240w.json`, `no_op_claim` block): the claim is **EXACT for the
UNCLAMPED ratio — 0.00e+00** — and **FALSE for the clamped component, which moves 0.0234**. The
`[-1, 1.5]` clamp is **not monotone**: it creates TIES, and a Spearman moves when ties move.

⭐ The correction is load-bearing rather than pedantic, because the tie-making IS half the lever the
package then built: `progress_v2` with the **cap only** reads rho(`peak_g`) **+0.1294** from
**+0.2900** — the cap alone more than halves the correlation, purely by creating ties.

**Root-cause CLASS — the `df` / `free` / `step_s` / cylindrical-FOV / anchor-units family, with the
scope being THE OBJECT A STRUCTURAL CLAIM IS ABOUT.** `progress` names two different tensors in this
codebase: the raw ratio `_progress` returns and the CLAMPED value `COMPONENTS["progress"]` returns.
A statement true of one was asserted of the other, and it read as an answer.
→ **Durable rule: a structural claim about a metric must NAME the exact object it holds for.** And
the control that confirms a *predicted* no-op is worth running precisely because it can refute the
prediction — this one did.

### 2. RETRACTED — "the residual envelope violation is a SCORER artifact on a stopped step"

The first build of `stack/tanitad/refs/feasible_decode.py` left **2 of 51,200** fan candidates
violating `envelope` under `clamp_entry`, both in `v0 = 0.000` windows with speeds
`[2.00, 4.00, 2.00, 0.00]`, reading **|kappa| = 0.3396** against a 0.2 cap — a stationary vehicle
performing a hard turn. The mechanism was identified correctly (`atan2(0, 0) == 0` is returned as a
HEADING, not as "undefined", so the step *before* a stop appears to have turned through the whole
previous heading), and I then drew the comfortable conclusion: that this is the SCORER's unguarded
heading, not a leak in the projection.

**MEASURED** (`raw/stopped_step_residual.json`): **REFUTED.** `models.kinematic
.unicycle_controls_from_path` — the programme's OTHER recovery, the one whose docstring explicitly
guards "WHERE THE EGO IS NOT MOVING, CURVATURE IS UNDETERMINED — NOT LARGE" — reads the **same
0.3396**. Both substitute a *direction* for a non-moving step instead of propagating the previous
one. It was a projection leak and it was fixed (hold the heading through a stop, freeze the yaw
there because `yaw_rate = v * kappa`; after: **0/51,200 on both arms, exact**).

**Root-cause CLASS — the same family, with the scope being WHOSE DEFECT IT IS.** ⭐ What made the
difference is that the discriminator was a **second IMPLEMENTATION**, not the same probe re-run —
the distinction `CLAUDE.md` draws under the `ls-tree` trap ("a second *probe* means a different
mechanism, not the same command run again"). It settled the question **against** the answer that
would have let the package ship as-is.

### 3. And the FIX was wrong twice, in the same family it was fixing

1. A **1e-6 m** held step is exact in float64 and **noise in float32**: at a 4 m path offset fp32
   resolves ~4.8e-7 m, so its `atan2` direction is meaningless. The regression fixture still read
   `envelope = 1.0` when scored in fp32 while reading a clean 0 in float64.
   → **The contract is what the fp32 CONSUMER recovers, never what our float64 arithmetic believes.**
2. Replacing it with a **1 mm ABSOLUTE** threshold then INFLATED *slow-but-moving* paths — a `ha0`
   hold at `v0 = 0.0005 m/s` takes perfectly well-conditioned 0.25 mm steps and was rewritten to
   1 mm — and drove the round-trip control (an already-feasible path must be a fixed point) from
   **0.00e+00 to 2.63e-03 m**.
   → **"Degenerate" is a statement about a step RELATIVE TO ITS OWN PATH.** A fixed constant that is
   right at one scale and wrong at another is the `df` / `step_s` family **introduced by the fix for
   another instance of it**.

⚠️ The sibling of the same class, kept because it is load-bearing: `fan_safety.score_paths` flags
with a strict `>`, so a projection clamping **exactly** to `a_max` saturates on the boundary and an
fp32 round-trip lands above it about half the time — MEASURED `envelope = 0.609375` on a
64-candidate fixture, which would have read as "the projection leaks" rather than as a boundary
artifact. A 1 % margin removes it.

→ **Pinned:** `stack/tests/test_feasible_decode.py` (15 tests) —
`test_a_stop_inside_the_window_does_not_manufacture_a_turn` scores in **float32**, the dtype the
consumer uses; `test_a_path_that_never_moves_keeps_its_exact_zeros` pins that the epsilon does not
fabricate motion; `test_roundtrip_on_an_already_feasible_path_is_exact` is the control that caught
the absolute-threshold regression.
"""


def main():
    if not os.path.isfile(LOG):
        print("LOG NOT READABLE -- mount down; not writing")
        return 2
    s = io.open(LOG, encoding="utf-8").read()
    if len(s) < 100000:
        print("LOG READ SHORT (%d bytes) -- refusing to write" % len(s))
        return 3
    if MARK in s:
        print("already present -- no-op")
        return 0
    out = s.rstrip("\n") + "\n" + ENTRY
    io.open(TMP, "w", encoding="utf-8", newline="\n").write(out)
    for i in range(8):
        try:
            shutil.copyfile(TMP, LOG)
            back = io.open(LOG, encoding="utf-8").read()
            if MARK in back and len(back) == len(out):
                print("WROTE ok on attempt %d: %d -> %d bytes"
                      % (i + 1, len(s), len(back)))
                return 0
        except Exception as e:      # noqa: BLE001
            print("attempt %d failed: %s" % (i + 1, e))
    print("COULD NOT VERIFY THE WRITE -- INCONCLUSIVE")
    return 5


if __name__ == "__main__":
    sys.exit(main())
