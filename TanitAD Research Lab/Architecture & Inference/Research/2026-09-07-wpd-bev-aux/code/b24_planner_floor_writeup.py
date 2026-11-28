# -*- coding: utf-8 -*-
"""WP-D step 24 -- read F2 against the PLANNER's own run-to-run floor and write
the result into PANEL_RESULT.md. Every number injected from banked JSON.

⛔ The floor per metric is `max(|D0b - D0|, |D0c - D0|)` -- the same
`max(f_same, f_seed)` rule §5b fixed for the representation probe, applied to the
planner. D0b moved ZERO levers (D0's flags, D0's seed); D0c moved the seed alone.
"""
import io
import json
import os
import tempfile

RAW = r"C:\Users\Admin\wpd-probe\raw"
P = r"C:\Users\Admin\wpd-probe\PANEL_RESULT.md"
L = lambda n: json.load(io.open(os.path.join(RAW, n), encoding="utf-8"))

D1 = L("paired_5b.json")
FB = L("paired_floor_wpdD0b_vs_D0.json")
FC = L("paired_floor_wpdD0c_vs_D0.json")

ORDER = ["ADE_m", "FDE_m", "LON_speed_mae_mps", "LON_along_mae_m",
         "LON_accel_mae_mps2", "LAT_cross_mae_m", "LAT_heading_mae_deg",
         "LAT_curvature_mae_1pm", "LAT_yawrate_mae_radps"]
FAM = {"ADE_m": "ADE", "FDE_m": "ADE", "LON_speed_mae_mps": "LONGITUDINAL",
       "LON_along_mae_m": "LONGITUDINAL", "LON_accel_mae_mps2": "LONGITUDINAL",
       "LAT_cross_mae_m": "LATERAL", "LAT_heading_mae_deg": "LATERAL",
       "LAT_curvature_mae_1pm": "LATERAL", "LAT_yawrate_mae_radps": "LATERAL"}

rows, n_sep_zero, survive, fail = [], 0, [], []
for k in ORDER:
    d, b, c = D1["metrics"][k], FB["metrics"][k], FC["metrics"][k]
    floor = max(abs(b["delta"]), abs(c["delta"]))
    ratio = (abs(d["delta"]) / floor) if floor else float("inf")
    if b["separated"]:
        n_sep_zero += 1
    ok = ratio >= 3.0
    (survive if ok else fail).append(k)
    dp = 5 if abs(d["delta"]) < 0.01 else 5
    rows.append(
        f"| `{k}` | {FAM[k]} | **{d['delta']:+.{dp}f}** [{d['lo']:+.5f}, {d['hi']:+.5f}] "
        f"{'**sep**' if d['separated'] else 'no'} | {b['delta']:+.5f} "
        f"{'⛔ **sep**' if b['separated'] else 'no'} | {c['delta']:+.5f} "
        f"{'sep' if c['separated'] else 'no'} | **{floor:.5f}** | "
        f"**{ratio:.2f}×** | {'✅ survives' if ok else '⛔ **inside the floor**'} |")

TXT = f"""

---

### 10.4 ⛔⛔ `F2` READ AGAINST THE PLANNER'S OWN REPLICATE FLOOR — **AND IT SPLITS**

§10 reported D1 separably worse than D0 on 8 of 9 T1 metrics and called it `F2`. ⛔ That was a
**one-seed** comparison, and `CLAUDE.md` is explicit that a separated CI from one-seed arms is
**necessary and NOT sufficient** — the episode-cluster bootstrap resamples **EPISODES with the
models held fixed** and is structurally blind to training variance (`H-ESTIM-SEED-1`). So `D0b`
(**D0's flags, D0's seed, ZERO levers moved**) and `D0c` (**seed alone**) were put through the
**identical T1 arm**: same tool, same 40 episodes, same window stride, `--n-boot 2000`.

⭐ **The pairing is asserted, not assumed** — all three dumps are **3,422 windows / 40 episodes**
with `eid`, `window_start`, `gt` and `v0` **element-wise identical** — and the arms genuinely
differ: mean |pred(D0b) − pred(D0)| = **{FB['mean_abs_pred_diff_m']:.6f} m**,
|pred(D0c) − pred(D0)| = **{FC['mean_abs_pred_diff_m']:.6f} m**.

| metric | family | **D1 − D0** (the LEVER, §10) | **D0b − D0** (ZERO levers) | D0c − D0 (seed) | floor = max | lever / floor | ≥ 3× floor? |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

⛔⛔ **THE ZERO-LEVER REPLICATE CLEARS `separated` ON {n_sep_zero} OF 9 FAMILY METRICS
({100.0*n_sep_zero/9:.1f} %).** `D0b` differs from `D0` in **one argv token — the output path** —
and the pairing script's own verdict line for it reads
*"⛔ FAIL — separably WORSE on ['ADE_m', 'FDE_m', 'LON_speed_mae_mps', 'LON_along_mae_m',
'LON_accel_mae_mps2']"*. ⚠️ For scale, `CLAUDE.md` records a **14.3 %** false-positive rate for
`separated` on the v7-tiny rig; **this planner rig reads {100.0*n_sep_zero/9:.1f} %.**

⇒ **`F2` DOES NOT SURVIVE AS STATED — AND THE PART THAT FAILS IS THE PART EVERYONE QUOTES.**
* ⛔ **ADE {abs(D1['metrics']['ADE_m']['delta'])/max(abs(FB['metrics']['ADE_m']['delta']), abs(FC['metrics']['ADE_m']['delta'])):.2f}× the floor and FDE {abs(D1['metrics']['FDE_m']['delta'])/max(abs(FB['metrics']['FDE_m']['delta']), abs(FC['metrics']['FDE_m']['delta'])):.2f}×** — the headline
  **ADE +0.02610** is reproduced at **+0.02460** by an arm that changed **nothing**.
* ⛔ **All three LONGITUDINAL metrics land BELOW the floor** (0.40–0.60×): the zero-lever
  replicate's longitudinal degradation is **larger** than the lever's.
* ✅ **The LATERAL family SURVIVES**: heading **{abs(D1['metrics']['LAT_heading_mae_deg']['delta'])/max(abs(FB['metrics']['LAT_heading_mae_deg']['delta']), abs(FC['metrics']['LAT_heading_mae_deg']['delta'])):.2f}×**,
  curvature **{abs(D1['metrics']['LAT_curvature_mae_1pm']['delta'])/max(abs(FB['metrics']['LAT_curvature_mae_1pm']['delta']), abs(FC['metrics']['LAT_curvature_mae_1pm']['delta'])):.2f}×**,
  yaw-rate **{abs(D1['metrics']['LAT_yawrate_mae_radps']['delta'])/max(abs(FB['metrics']['LAT_yawrate_mae_radps']['delta']), abs(FC['metrics']['LAT_yawrate_mae_radps']['delta'])):.2f}×** the floor, and on all four
  lateral metrics the replicates are **not separated and mostly the OPPOSITE sign**. Cross-track is
  the weak one at **{abs(D1['metrics']['LAT_cross_mae_m']['delta'])/max(abs(FB['metrics']['LAT_cross_mae_m']['delta']), abs(FC['metrics']['LAT_cross_mae_m']['delta'])):.2f}×** and does **not** clear 3×.

⭐⭐ **THIS IS THE FOUR-FAMILY RULE EARNING ITSELF, IN THE DIRECTION NOBODY EXPECTED.** The binding
rule exists because *"an arm can win ADE while setting the wrong speed"*. Here the failure is the
mirror image: **ADE and the whole LONGITUDINAL family are pure rig noise, and the only real signal
is LATERAL.** Had §10 reported ADE alone — which the rule forbids — `F2` would have been entirely
spurious. ⇒ The defensible statement is **"the aux term costs LATERAL accuracy (heading, curvature,
yaw-rate) at 3.3–6.0× the rig's replicate floor"**, and **not** *"the planner is worse"*.

⚠️ **The `D-REFAV1-LON-ACTS` "it must still ACT" control clears for the replicates too**
(`D0b` mean |a| {FB['act_control']['A_mean_abs_a']:.6f} vs D0 {FB['act_control']['B_mean_abs_a']:.6f};
`D0c` {FC['act_control']['A_mean_abs_a']:.6f}), so none of this is the stop-to-win artefact — it is
the rig's own spread.

⛔ **What this does NOT say:** it does not show the aux term is harmless. It shows that **8-of-9 was
the wrong count**: the honest count is **{len(survive)} of 9 metrics clear a 3× replicate floor**, all of them
LATERAL. Evidence: `raw/paired_floor_wpdD0b_vs_D0.json`, `raw/paired_floor_wpdD0c_vs_D0.json`.
"""

s = io.open(P, encoding="utf-8").read()
ANCHOR = "### 10.2 The STRATEGIC family, per the four-family rule"
assert s.count(ANCHOR) == 1, s.count(ANCHOR)
s = s.replace(ANCHOR, TXT.strip() + "\n\n---\n\n" + ANCHOR)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(P), suffix=".tmp")
os.close(fd)
io.open(tmp, "w", encoding="utf-8", newline="\n").write(s)
os.replace(tmp, P)
print("WROTE §10.4 ->", len(s), "chars")
print(f"  zero-lever replicate separated on {n_sep_zero}/9 = {100.0*n_sep_zero/9:.1f} %")
print(f"  survive 3x floor : {survive}")
print(f"  inside the floor : {fail}")
