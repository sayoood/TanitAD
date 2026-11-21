"""mapedge_q2.py -- PREREG_MAPEDGE_Q2.md: OWN-BND-T12 / DER-BND-T12 through MAPSEL's scorer, imported UNCHANGED.  CPU, 0 GPU.

    python mapedge_q2.py --out <dir> [--trace F:/refcv9_mapsel/trace_s0.npz --rast F:/refcv9_mapsel/rast_s0]

Runs only when Q1's MAPEDGE_RESULT.json says INSTRUMENT-PASS (headline) or FAIL with B1 passed (labelled "geometric prior").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MSDIR = Path("C:/Users/Admin/refcv8_pf/refcv9_mapsel")
sys.path.insert(0, str(MSDIR / "code"))
sys.path.insert(0, str(HERE))
import mapsel_score as MS  # noqa: E402
from mapedge import boundary  # noqa: E402

L = MS.L
NEW = (("OWN-BND-T12", "ownbnd", 2), ("DER-BND-T12", "derbnd", 2), ("NOOP-BND-T12", "noop", 2))
_orig_pass1 = MS.pass1


def pass1_bnd(tr, rdir, has_gt, hum=None, limit=None):
    F, acc_own, acc_der, don, n_samples, hum_pmin = _orig_pass1(tr, rdir, has_gt, hum, limit)
    n = F["own_off"].shape[0]
    for s, base in (("ownbnd", "own"), ("derbnd", "der"), ("noop", "own")):
        F[f"{s}_off"], F[f"{s}_lane"] = F[f"{base}_off"], F[f"{base}_lane"]
        F[f"{s}_edge"] = np.zeros_like(F["own_edge"])           # noop keeps all-False: Q2-K3
    zero = None
    for j in range(n):
        g = L.Geom(torch.from_numpy(tr["fan"][j]))
        own = np.load(MS.rast_file(rdir, tr["e_i"][j], tr["ws"][j]))["pdrv"]
        if zero is None:
            zero = torch.zeros(own.shape, dtype=torch.bool)
        F["ownbnd_edge"][j] = g.flags({"bad": zero, "edge": L.dilate(boundary(own >= L.PDRV_MIN_U8), L.EDGE_DIL), "lane": zero})[1].numpy()
        if don[j] >= 0:
            der = np.load(MS.rast_file(rdir, tr["e_i"][don[j]], tr["ws"][don[j]]))["pdrv"]
            F["derbnd_edge"][j] = g.flags({"bad": zero, "edge": L.dilate(boundary(der >= L.PDRV_MIN_U8), L.EDGE_DIL), "lane": zero})[1].numpy()
        if j % 500 == 0:
            print(f"[mapedge_q2] bnd pass {j + 1}/{n}", flush=True)
    return F, acc_own, acc_der, don, n_samples, hum_pmin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default="F:/refcv9_mapsel/trace_s0.npz")
    ap.add_argument("--rast", default="F:/refcv9_mapsel/rast_s0")
    ap.add_argument("--q1", default=str(HERE / "result" / "s0" / "MAPEDGE_RESULT.json"))
    ap.add_argument("--mapsel-result", default=str(MSDIR / "result" / "s0" / "MAPSEL_RESULT.json"))
    ap.add_argument("--dry", action="store_true", help="plumbing only on the smoke bank: no Q1 gate, --allow-partial, prints controls only (no outcome is read)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = hashlib.sha256((HERE / "PREREG_MAPEDGE_Q2.md").read_bytes()).hexdigest()
    if pre != (HERE / "PREREG_MAPEDGE_Q2.sha256").read_text().split()[0]:
        raise SystemExit("[mapedge_q2] PREREG_MAPEDGE_Q2.md does not match its seal: refusing")
    if a.dry:
        label = "DRY"
    else:
        q1 = json.loads(Path(a.q1).read_text())
        b1 = q1["bars"]["B1_rec_OWNBND_minus_OWNCODE5"]["pass"]
        if q1["verdict"].startswith("VOID") or not b1:
            raise SystemExit(f"[mapedge_q2] Q1 is {q1['verdict']}: Q2 does not run (PREREG_MAPEDGE sec. 4)")
        label = "headline" if q1["verdict"] == "INSTRUMENT-PASS" else "geometric prior, not perception (Q1 B2/B3 failed)"
    MS.pass1 = pass1_bnd
    MS.ARMS = tuple(MS.ARMS) + NEW
    msout = out / "mapsel_run"
    sys.argv = ["mapsel_score.py", "--seed", "0", "--trace", a.trace, "--rast", a.rast, "--out", str(msout), "--human-json", str(MSDIR / "result" / "K_HUMAN.json")] + (["--allow-partial"] if a.dry else [])
    MS.main()
    res = json.loads((msout / "MAPSEL_RESULT.json").read_text(encoding="utf-8"))
    if a.dry:
        mc, ff = res["map_compliance"]["S6c"]["arms"], res["four_families"]
        k3 = all(mc["NOOP-BND-T12"][k].get("rate") == mc["OWN-T1"][k].get("rate") for k in ("off_drivable", "edge_contact", "lane_cross")) and all(ff[x]["NOOP-BND-T12"] == ff[x]["OWN-T1"] for x in ("S6c_stage", "S2c_stage"))
        new_present = all(nm in mc for nm, _s, _t in NEW)
        print("ZZQ2DRY", "K3", k3, "new_arms_scored", new_present)
        return 0 if (k3 and new_present) else 1
    ref = json.loads(Path(a.mapsel_result).read_text(encoding="utf-8"))
    ctl = {}
    old = [nm for nm, _s, _t in ref.get("arms_declared", [])] or [k for k in ref["map_compliance"]["S6c"]["arms"] if k in res["map_compliance"]["S6c"]["arms"] and k not in [x[0] for x in NEW]]
    mism = []
    for nm in old:
        for k in ("off_drivable", "edge_contact", "lane_cross"):
            r0, r1 = ref["map_compliance"]["S6c"]["arms"][nm][k], res["map_compliance"]["S6c"]["arms"][nm][k]
            if r0.get("rate") != r1.get("rate") or r0.get("vs_BASE") != r1.get("vs_BASE"):
                mism.append(f"{nm}:{k}")
    ctl["Q2-K1_reproduces_MAPSEL_s0"] = {"pass": bool(old and not mism), "n_arms": len(old), "mismatches": mism[:20]}
    hard = {k: v for k, v in res.get("controls", {}).items() if isinstance(v, dict) and "pass" in v}
    hard_fail = sorted(k for k, v in hard.items() if v["pass"] is False)
    ref_fail = sorted(k for k, v in ref.get("controls", {}).items() if isinstance(v, dict) and v.get("pass") is False)
    ctl["Q2-K2_mapsel_controls_as_in_mapsel"] = {"pass": bool(hard_fail == ref_fail), "failing_here": hard_fail, "failing_in_mapsel": ref_fail}
    mc = res["map_compliance"]["S6c"]["arms"]
    ff = res["four_families"]
    same = all(mc["NOOP-BND-T12"][k].get("rate") == mc["OWN-T1"][k].get("rate") for k in ("off_drivable", "edge_contact", "lane_cross")) and \
        all(ff[s]["NOOP-BND-T12"] == ff[s]["OWN-T1"] for s in ("S6c_stage", "S2c_stage"))
    ctl["Q2-K3_noop_equals_OWN-T1"] = {"pass": bool(same)}
    cl = MS.arm_clauses(res, "OWN-BND-T12")
    e1c = mc["OWN-BND-T12"]["edge_contact"]["vs_BASE"]
    e3c = mc["DER-BND-T12"]["edge_contact"]["vs_BASE"]
    bars = {"E1": bool("hi" in e1c and e1c["hi"] < 0), "E2": bool(cl["C2"]), "E3": bool(not ("hi" in e3c and e3c["hi"] < 0)), "E4": bool(cl["C4"])}
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"])
    gt12 = MS.arm_clauses(res, "GT-T12")
    g12e = mc["GT-T12"]["edge_contact"]["vs_BASE"]
    gt_clears = bool("hi" in g12e and g12e["hi"] < 0 and gt12["C2"] and gt12["C4"])
    if ctl_fail:
        word, nxt = f"VOID (control failed: {ctl_fail})", "fix the wrapper; re-run (0 GPU)."
    elif all(bars.values()):
        word, nxt = "PASS", "register OWN-BND-T12 as an opt-in inference row (T1 stamp); carry the boundary edge into refcv9's map wiring (critic feature + calibrated rejection); NavSim only with a PI go."
    elif bars["E1"] and bars["E2"] and bars["E4"] and not bars["E3"]:
        word, nxt = "INCONCLUSIVE-DER", "the pick change alone lowers edge contact: read OWN minus DER; the boundary's window-specific value is not shown."
    elif not bars["E1"]:
        word = "FAIL (E1)"
        nxt = ("perception is the bottleneck at the curb: train the edge with a distance-transform / boundary loss (P3); GT-T12 is the bound." if gt_clears else
               "the fan holds no edge-clear candidate the selector reaches: run the fan-availability audit (share of windows with >= 1 GT-edge-clear eligible candidate).")
    else:
        word = "FAIL (" + ", ".join(k for k, v in bars.items() if not v) + ")"
        nxt = "the edge rule costs driving quality: a soft edge cost (rank penalty) instead of a hard filter."
    q2 = {"tool": "mapedge_q2.py", "prereg_sha256": pre, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "label": label,
          "controls": ctl, "bars": bars, "E1_cell": e1c, "E3_cell": e3c, "E2_worse_rows": cl["C2_worse_rows"], "C4": cl["C4"],
          "GT-T12_clears_E1_E2_E4": gt_clears, "verdict": word, "next": nxt,
          "rates_S6c": {nm: {k: mc[nm][k].get("rate") for k in ("off_drivable", "edge_contact", "lane_cross")} for nm in ("BASE", "HUMAN", "OWN-T1", "OWN-T12", "GT-T12", "OWN-BND-T12", "DER-BND-T12")},
          "edge_minus_HUMAN": {nm: mc[nm]["edge_contact"].get("minus_HUMAN") for nm in ("BASE", "OWN-T12", "GT-T12", "OWN-BND-T12", "DER-BND-T12")}}
    MS.jdump(q2, out / "Q2_RESULT.json")
    f = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    Lm = [word, "", f"# MAPEDGE Q2 (mechanical; PREREG_MAPEDGE_Q2 sha256 {pre[:12]}; arm label: {label})", "",
          "tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed", "",
          "| arm (S6c) | off-drivable | edge contact | lane cross | edge minus HUMAN |", "|---|---|---|---|---|"]
    for nm, r in q2["rates_S6c"].items():
        g = lambda v: f"{v:.4f}" if v is not None else "-"  # noqa: E731
        Lm.append(f"| {nm} | {g(r['off_drivable'])} | {g(r['edge_contact'])} | {g(r['lane_cross'])} | {g(q2['edge_minus_HUMAN'].get(nm))} |")
    Lm += ["", f"E1 OWN-BND-T12 edge contact vs BASE: {f(e1c)} -> {bars['E1']}", f"E2 no four-family row worse: {bars['E2']} {cl['C2_worse_rows'] or ''}",
           f"E3 DER-BND-T12 not separated below BASE: {f(e3c)} -> {bars['E3']}", f"E4 STOP guard: {bars['E4']}", f"GT-T12 clears E1/E2/E4: {gt_clears}", "",
           f"controls failed: {ctl_fail or 'none'}", "", f"NEXT: {nxt}", "", "ZZMEQ2ENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(Lm), encoding="utf-8")
    print(word, bars, ctl_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
