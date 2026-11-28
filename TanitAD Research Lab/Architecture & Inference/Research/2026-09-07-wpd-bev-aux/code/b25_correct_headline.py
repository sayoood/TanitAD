# -*- coding: utf-8 -*-
"""WP-D step 25 -- correct §0 and §6, which still assert the PRE-REPLICATE reading.

⛔ This is a RETRACTION inside the panel, and it is written as one: the old
sentence is not quietly softened, it is named and replaced.
"""
import io
import json
import os
import tempfile

RAW = r"C:\Users\Admin\wpd-probe\raw"
P = r"C:\Users\Admin\wpd-probe\PANEL_RESULT.md"
L = lambda n: json.load(io.open(os.path.join(RAW, n), encoding="utf-8"))
D1, FB, FC = L("paired_5b.json"), L("paired_floor_wpdD0b_vs_D0.json"), L("paired_floor_wpdD0c_vs_D0.json")
fl = lambda k: max(abs(FB["metrics"][k]["delta"]), abs(FC["metrics"][k]["delta"]))
rt = lambda k: abs(D1["metrics"][k]["delta"]) / fl(k)

s = io.open(P, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


# ---- §0 item 1 ---------------------------------------------------------------
sub("""1. ⛔⛔ **`E-BEV-AUX-1` IS REFUTED ON BOTH HALVES OF ITS OWN SUCCESS CONDITION — `F4` AND `F2`
   BOTH FIRED.** ⛔ **`F2` (§10):** the planner is separably **WORSE** with the aux term on, on
   **8 of 9** T1 four-family metrics — ADE **+0.02610 [+0.00970, +0.04240]**,
   with the same sign across LONGITUDINAL and LATERAL — and the *"it must still ACT"* control
   CLEARS, so it is not the planner stopping. ⛔ **`F4` (below): the gain is
   capacity/regularisation, not agent content.**""",
f"""1. ⛔⛔ **`E-BEV-AUX-1` DOES NOT MEET ITS SUCCESS CONDITION — BUT AFTER THE REPLICATE ARMS
   LANDED (2026-09-10) THE PANEL'S OWN *REASONS* FOR CALLING IT REFUTED NO LONGER HOLD, AND THIS
   ENTRY IS CORRECTED RATHER THAN SOFTENED.** ⛔ **RETRACTED from the 2026-09-08 version of this
   line:** *"REFUTED ON BOTH HALVES … `F4` AND `F2` BOTH FIRED … the planner is separably WORSE on
   **8 of 9** T1 metrics."* **`8 of 9` IS WRONG.** MEASURED (§10.4): the replicate `D0b` — D0's
   flags, D0's **seed**, **ZERO levers moved**, one differing argv token (`--out`) — is itself
   *"separably worse"* than D0 on **5 of the same 9 metrics**, reproducing the headline
   **ADE +0.02610** at **+0.02460** and *exceeding* the lever on **all three LONGITUDINAL**
   metrics. ⇒ **ADE ({rt('ADE_m'):.2f}× the replicate floor), FDE ({rt('FDE_m'):.2f}×) and the whole
   LONGITUDINAL family (0.40–0.60×) are RIG NOISE, not the lever.** ✅ **What survives is the
   LATERAL family** — heading **{rt('LAT_heading_mae_deg'):.1f}×**, curvature
   **{rt('LAT_curvature_mae_1pm'):.1f}×**, yaw-rate **{rt('LAT_yawrate_mae_radps'):.1f}×** the floor, with the
   replicates unseparated and mostly the OPPOSITE sign — so the defensible claim is **"the aux term
   costs LATERAL accuracy"**, on **3 of 9** metrics, and *not* *"the planner is worse"*.
   ⭐⭐ **This is the four-family rule earning itself in reverse: ADE was the spurious part.**
   ⛔ **`F4` (below) is ALSO no longer established as a MECHANISM**: `|D1 − D2|` = 0.00366 is
   **1.46× SMALLER** than the probe's own measured seed floor (§5b.1), so *"a shuffled target does
   the same"* is what this rig reports either way. ⇒ **The honest global verdict is `F5` —
   UNDERPOWERED at 4,000 steps** on `A3`, `A4` and most of `F2`; the claim **fails** only because
   **`A2` was never demonstrated** (a straddling CI, which no floor rescues).""")

# ---- §6 ----------------------------------------------------------------------
sub("""⭐ **`F2` (§10) is therefore the load-bearing half of the refutation**, and it is measured on a
different instrument at a different scale (ADE **+0.02610 [+0.00970, +0.04240]**, **4.9×**
this probe's seed floor) — see §7 for the arm that tests whether it too survives a replicate.""",
f"""⛔⛔ **AND `F2` WAS TESTED AGAINST ITS OWN REPLICATE FLOOR RATHER THAN ASSUMED — IT SPLITS
(§10.4).** The same two replicate checkpoints went through the identical T1 arm. `D0b` (**zero
levers moved**) is *"separably worse"* than D0 on **5 of 9** family metrics, so on this rig
`separated` has a **55.6 %** false-positive rate for a one-seed pair (the v7-tiny rig's recorded
figure is 14.3 %). ⇒ **ADE ({rt('ADE_m'):.2f}×), FDE ({rt('FDE_m'):.2f}×) and all three LONGITUDINAL
metrics (0.40–0.60×) are INSIDE the floor**; only **LATERAL heading / curvature / yaw-rate** clear
3× ({rt('LAT_heading_mae_deg'):.1f}× / {rt('LAT_curvature_mae_1pm'):.1f}× / {rt('LAT_yawrate_mae_radps'):.1f}×). ⇒ **`F2` survives
only as a LATERAL claim on 3 of 9 metrics**, and the ADE headline it was quoted by does not.""")

fd, tmp = tempfile.mkstemp(dir=os.path.dirname(P), suffix=".tmp")
os.close(fd)
io.open(tmp, "w", encoding="utf-8", newline="\n").write(s)
os.replace(tmp, P)
print("CORRECTED §0 item 1 and §6 ->", len(s), "chars")
assert "IS REFUTED ON BOTH HALVES" not in s
assert "8 of 9** T1 four-family metrics" not in s
print("POST_ASSERT_OK: the pre-replicate '8 of 9' headline is gone from §0")
for t in ("RETRACTED from the 2026-09-08 version", "55.6 %", "§10.4"):
    print(f"  contains {t!r}: {s.count(t)}")
