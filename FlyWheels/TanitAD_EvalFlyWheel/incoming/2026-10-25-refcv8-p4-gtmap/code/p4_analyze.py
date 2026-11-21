"""p4_analyze.py -- the P4 GT-MAP ORACLE read, MECHANICAL from PREREG_P4.md (sealed).  CPU only.  Writes P4.json + RESULT.md.

    R8_LAUNCH_TREE=<tree> PYTHONPATH='<tree>/stack;<tree>/taniteval' CUDA_VISIBLE_DEVICES=-1 \
      python -X utf8 p4_analyze.py --root <run dir> --seeds 0,1              # the scored read (p4_s{seed}.npz + dump_s{seed}/)
      python -X utf8 p4_analyze.py --root <smoke dir> --seeds 1 --smoke --plain-dump <dir> [--identity-npz <zero-delta p4_s1.npz>]

Interval: paired episode-cluster bootstrap (taniteval.ci via pair_read_r8.paired_cell), B 2000, seed 0, models FIXED -- answers
'another draw of EPISODES' only; NOT another training run (one training seed); INFERENCE: seeds 0 and 1 + floors F, F'.
Tier: T0-ORACLE (privileged GT map at inference) on the SHIP-R8 T1 self-action open-loop roll -- a bound, never driving performance.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG, SEAL = ROOT / "PREREG_P4.md", ROOT / "PREREG_P4.sha256"
BANK = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf")
PKG_CODE = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code")
REMAP = HERE / "refcv8_eval_remap_devbox_p4.json"
SLOTS = {"S2": [0, 1, 2, 3], "S6": [1, 3, 4, 5, 6, 7]}
DT = {"S2": 0.5, "S6": 1.0}
ARMS = ("TRUE", "OWNMAP", "SHUF")
CONTRASTS = {"TRUE-OWNMAP": ("TRUE", "OWNMAP"), "SHUF-OWNMAP": ("SHUF", "OWNMAP"), "OWNMAP-BASE": ("OWNMAP", "OWN"),
             "TRUE-BASE": ("TRUE", "OWN"), "SHUF-BASE": ("SHUF", "OWN")}
C1_TOL = 1e-5
N_BOOT, BOOT_SEED = 2000, 0
THIN = (2, 3, 5)
CLASS_NAMES = ("seen-no-class", "drivable", "lane", "crosswalk", "arrow/text", "non-drv-edge", "hatched", "sidewalk/verge")
REACH = {"miou_min": 0.80, "thin_min": 0.50, "thin_bands": (0, 1), "gain_min": 0.20}     # PREREG s5 (committed)
BEV_ZERO_S6 = {"0": 1.1505, "1": 1.1282}      # perc2plan RESULT.md (MEASURED, Thor): BEV_ZERO - BASE, S6 ADE -- context only
TIER = "T0-ORACLE (privileged SAM3 GT map injected at inference) on the SHIP-R8 T1 self-action open-loop roll: a BOUND, never driving"
QUESTION = ("paired episode-cluster bootstrap (taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' only; NOT another "
            "training run (one training seed, H-ESTIM-SEED-1); inference: seeds 0 and 1 + floors F (BASE s1-s0) and F' (contrast s1-s0)")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_tree(d: Path) -> dict:
    rows = {}
    for f in sorted(glob.glob(str(d / "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci_ = int(z["clip_index"][0])
        for j, w in enumerate(z["ws"]):
            rows[(ci_, int(w))] = {"os": z["os"][j], "g": z["g"][j]}
        fd = d / "decisions" / os.path.basename(f)
        if fd.exists():
            dd = np.load(fd, allow_pickle=True)
            for j, w in enumerate(dd["ws"]):
                rows.setdefault((ci_, int(w)), {})["plan8"] = dd["plan_full_nav_true"][j]
    return rows


def load_panel(pdir: Path) -> dict:
    rows = {}
    for f in sorted(glob.glob(str(pdir / "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci_ = int(z["clip_index"][0])
        for j, w in enumerate(z["ws"]):
            rows[(ci_, int(w))] = {"g": z["g"][j], "eid": int(os.path.basename(f)[2:5])}
    return rows


def load_p4(p: Path) -> dict:
    z = np.load(p, allow_pickle=True)
    o = {k: z[k] for k in z.files}
    o["meta"] = json.loads(str(z["meta_json"]))
    o["key"] = [(int(c), int(w)) for c, w in zip(o["clip_index"], o["ws"])]
    o["row"] = {k: i for i, k in enumerate(o["key"])}
    return o


_PR = {}


def pr_mod():
    if not _PR:
        sys.path.insert(0, str(PKG_CODE))
        import r8_repoint
        r8_repoint.REMAP_OVERRIDES = REMAP
        import pair_read_r8 as PR
        import r8_battery_patch as BP
        L, P = BP.apply()
        _PR.update(PR=PR, ra=P.RR.ra3().ra)
    return _PR["PR"], _PR["ra"]


def cell(new, ref, eid):
    PR, _ = pr_mod()
    return PR.paired_cell(new, ref, eid, N_BOOT, BOOT_SEED)


def ade(P, G):
    return np.linalg.norm(np.asarray(P, np.float64) - np.asarray(G, np.float64), axis=-1).mean(-1)


def reach(o) -> dict:
    """Pooled IoU (sum inter / sum union over windows) per pair x band x class, on the crop's target-seen cells."""
    if "reach_pairs" not in o:
        return {}
    P = o["reach_pairs"].sum(0)                   # [pairs, bands, 8, 2]
    pairs = o["meta"]["pairs"]
    out = {}
    for i, nm in enumerate(pairs):
        inter, union = P[i, :, :, 0], P[i, :, :, 1]
        tot_i, tot_u = inter.sum(0), union.sum(0)
        per_cls = {CLASS_NAMES[c]: (float(tot_i[c] / tot_u[c]) if tot_u[c] > 0 else None) for c in range(8)}
        present = [c for c in range(1, 8) if tot_u[c] > 0]
        miou = float(np.mean([tot_i[c] / tot_u[c] for c in present])) if present else None
        band = {f"band{b}": {CLASS_NAMES[c]: (float(inter[b, c] / union[b, c]) if union[b, c] > 0 else None) for c in THIN + (1,)}
                for b in range(P.shape[1])}
        out[nm] = {"miou_classes_1to7": miou, "per_class_all_bands": per_cls, "thin_and_drivable_by_band": band}
    return out


def reach_gate(R: dict) -> dict:
    t, o0 = R.get("TRUE_vs_gt"), R.get("own0_vs_gt")
    if not t or not o0 or t["miou_classes_1to7"] is None:
        return {"pass": False, "reason": "no TRUE reach statistics"}
    thin = {}
    ok_thin = True
    for b in REACH["thin_bands"]:
        for c in THIN:
            v = t["thin_and_drivable_by_band"][f"band{b}"][CLASS_NAMES[c]]
            thin[f"band{b}:{CLASS_NAMES[c]}"] = v
            if v is not None and v < REACH["thin_min"]:
                ok_thin = False
    gain = t["miou_classes_1to7"] - (o0["miou_classes_1to7"] or 0.0)
    ok = t["miou_classes_1to7"] >= REACH["miou_min"] and ok_thin and gain >= REACH["gain_min"]
    return {"pass": bool(ok), "true_miou": t["miou_classes_1to7"], "own0_miou": o0["miou_classes_1to7"], "gain": gain, "thin": thin,
            "bars": REACH}


def read_seed(root: Path, s: str, smoke: bool, plain_dump: Path | None, identity: Path | None) -> dict:
    o = load_p4(root / f"p4_s{s}.npz")
    run = load_tree(root / f"dump_s{s}")
    R = {"n_windows": len(o["key"]), "meta": {k: o["meta"][k] for k in o["meta"] if k not in ("child_argv",)}, "controls": {}}
    C = R["controls"]
    keys = o["key"]
    d2 = [float(np.abs(o["traj_OWN"][o["row"][k]] - run[k]["plan8"]).max()) for k in keys if k in run and "plan8" in run[k]]
    C["C2_base_capture_eq_dump"] = {"n": len(d2), "max_abs": max(d2) if d2 else None, "pass": bool(d2) and max(d2) == 0.0 and len(d2) == len(keys)}
    ref = load_tree(plain_dump) if plain_dump is not None else load_tree(BANK / f"dump_s{s}")
    k1 = [k for k in keys if k in run and k in ref]
    dos = max(float(np.abs(run[k]["os"] - ref[k]["os"]).max()) for k in k1) if k1 else None
    dpl = max(float(np.abs(run[k]["plan8"] - ref[k]["plan8"]).max()) for k in k1 if "plan8" in ref[k]) if k1 else None
    binding = plain_dump is not None or s == "1"
    C["C1_base_reproduces"] = {"against": str(plain_dump or BANK / f"dump_s{s}"), "n": len(k1), "max_abs_os": dos, "max_abs_plan8": dpl,
                               "tol": C1_TOL, "binding": binding,
                               "pass": (bool(k1) and len(k1) == len(keys) and max(dos, dpl) <= C1_TOL) if binding else None,
                               "note": None if binding else "seed 0: the banked s0 is the FULL arm-set roll (another sampler stream)"}
    cnt = o["meta"]["counters"]
    C["C3_recompute_identity"] = {"checked": cnt["recomp_zero_checked"], "equal": cnt["recomp_zero_equal"],
                                  "maxdiff": cnt["recomp_zero_maxdiff"], "own_calls": cnt["own_calls"],
                                  "pass": cnt["recomp_zero_checked"] == cnt["own_calls"] == cnt["recomp_zero_equal"] > 0}
    nogt = {a: int((~o[f"forced_{a}"]).sum()) for a in ARMS}
    eq_nogt = all(np.array_equal(o[f"traj_{a}"][~o[f"forced_{a}"]], o["traj_OWN"][~o[f"forced_{a}"]]) for a in ARMS)
    C["C3b_no_target_windows_are_base"] = {"n_by_arm": nogt, "pass": bool(eq_nogt)}
    if identity is not None:
        oi = load_p4(identity)
        m = {a: oi[f"forced_{a}"] for a in ARMS}
        dmax = {a: (float(np.abs(oi[f"traj_{a}"][m[a]] - oi["traj_OWN"][m[a]]).max()) if m[a].any() else None) for a in ARMS}
        seq = {a: bool((oi[f"sel_{a}"][m[a]] == oi["sel_OWN"][m[a]]).all()) for a in ARMS}
        C["C3c_zero_delta_identity_bit_exact"] = {"file": str(identity), "forced": {a: int(m[a].sum()) for a in ARMS},
                                                   "max_abs_traj": dmax, "sel_equal": seq,
                                                   "pass": all(v == 0.0 for v in dmax.values() if v is not None)
                                                   and all(seq.values()) and all(m[a].any() for a in ARMS)}
    C["C8_gt_item_eq_store"] = {"checked": cnt["gt_store_parity_checked"], "equal": cnt["gt_store_parity_equal"],
                                "pass": cnt["gt_store_parity_checked"] > 0 and cnt["gt_store_parity_checked"] == cnt["gt_store_parity_equal"]}
    C["applied_once"] = {"arm_forwards": cnt["arm_forwards"], "applied_not_once": cnt["applied_not_once"], "cap_not_once": cnt["cap_not_once"],
                         "mh_memo_hit": cnt["mh_memo_hit"], "mh_memo_miss": cnt["mh_memo_miss"],
                         "pass": cnt["applied_not_once"] == 0 and cnt["cap_not_once"] == 0 and cnt["arm_forwards"] > 0}
    C["refine_repro"] = {"maxdiff_fp32_decode_vs_model_logits": cnt["refine_repro_maxdiff"],
                         "argmax_agree_min": cnt["refine_repro_argmax_agree_min"],
                         "note": "CPU fp32 smoke: must be ~0; CUDA run: the model's logits ran under bf16 autocast -- recorded, not a gate"}
    if not smoke:
        bank = load_tree(BANK / f"dump_s{s}")
        kg = [k for k in keys if k in run and k in bank]
        g7 = max(float(np.abs(run[k]["g"] - bank[k]["g"]).max()) for k in kg) if kg else None
        C["C7_gt_parity"] = {"n": len(kg), "max_abs_g": g7, "pass": bool(kg) and g7 == 0.0}
    R["reach"] = reach(o)
    R["census"] = {"forced": {a: int(o[f"forced_{a}"].sum()) for a in ARMS}, "gt_label_windows": int(o["gt_label"].sum()),
                   "rel_norm_median": {a: (float(np.nanmedian(o[f"relnorm_{a}"])) if f"relnorm_{a}" in o else None) for a in ARMS},
                   "loss0_mean": {a: (float(np.nanmean(o[f"loss0_{a}"])) if f"loss0_{a}" in o else None) for a in ARMS},
                   "loss1_mean": {a: (float(np.nanmean(o[f"loss1_{a}"])) if f"loss1_{a}" in o else None) for a in ARMS},
                   "inv_wall_mean_s": (float(np.mean(o["inv_wall"])) if "inv_wall" in o else None),
                   "wall_by_arm_s": o["meta"].get("wall_by_arm_s"), "roll_wall_s": o["meta"].get("roll_wall_s")}
    R["dpsum"] = {a: o[f"dpsum_{a}"] for a in ARMS}
    R["_keys"] = keys
    if smoke:
        return R
    PR, ra = pr_mod()
    pan = {"S2": load_panel(BANK / f"panel_s{s}"), "S6": load_panel(BANK / f"s6_s{s}")}
    R["surfaces"] = {}
    R["_comps"] = {}
    for S in ("S2", "S6"):
        ks = [k for k in keys if k in pan[S]]
        G = np.stack([pan[S][k]["g"] for k in ks]).astype(np.float32)
        rows = [o["row"][k] for k in ks]
        eid = [pan[S][k]["eid"] for k in ks]
        comps = {arm: PR.components(ra, o[f"traj_{arm}"][rows][:, SLOTS[S]].astype(np.float32), G, DT[S]) for arm in ("OWN",) + ARMS}
        R["_comps"][S] = {"keys": ks, "eid": eid, "comps": comps}
        sur = {"n_windows": len(ks), "n_episodes": len(set(eid)), "means": {}, "rows": {}}
        for arm in ("OWN",) + ARMS:
            sur["means"][arm] = {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in comps[arm].items()}
        for fam, key, label, unit, lower, _nr in PR.METRICS:
            sur["rows"][key] = {"family": fam, "label": label, "unit": unit, "lower_is_better": lower,
                                **{c: cell(comps[a][key], comps[b][key], eid) for c, (a, b) in CONTRASTS.items()}}
        sur["STRATEGIC"] = {"status": "NOT APPLICABLE", "n": 0, "reason": "SHIP-R8 clause (6) --no-strategic; route checkpoint off (--r8-no-rc)"}
        sur["LONGITUDINAL_distance_keeping"] = {"status": "NOT COMPUTED", "n": 0,
                                                "reason": "the battery's per-window components carry no lead-agent headway/TTC instrument on "
                                                          "the os windows (pair_read_r8.METRICS); a named gap, not an omission"}
        R["surfaces"][S] = sur
    return R


def sep(c, lower):
    if not isinstance(c, dict) or "lo" not in c:
        return None
    if c["lo"] > 0:
        return "worse" if lower else "better"
    if c["hi"] < 0:
        return "better" if lower else "worse"
    return None


def build(root: Path, seeds: list, smoke: bool, plain_dump, identity) -> dict:
    res = {"tool": "p4_analyze.py", "prereg_sha256": sha256_file(PREREG) if PREREG.exists() else None,
           "seal": SEAL.read_text(encoding="utf-8").split()[0] if SEAL.exists() else None, "tier": TIER, "question": QUESTION,
           "rendered": time.strftime("%Y-%m-%d %H:%M:%S %z"), "per_seed": {}}
    for s in seeds:
        res["per_seed"][s] = read_seed(root, s, smoke, plain_dump, identity if s == seeds[-1] else None)
    if smoke or len(seeds) < 2:
        res["verdict"] = "NOT EVALUATED (smoke / one seed)"
        return res
    P0, P1 = res["per_seed"]["0"], res["per_seed"]["1"]
    inj_eq = all(np.array_equal(np.nan_to_num(P0["dpsum"][a], nan=-1e30), np.nan_to_num(P1["dpsum"][a], nan=-1e30)) for a in ARMS) \
        and P0["_keys"] == P1["_keys"]
    res["C6_same_injection_both_seeds"] = {"pass": bool(inj_eq)}
    floors = {}
    for S in ("S2", "S6"):
        c0, c1 = P0["_comps"][S], P1["_comps"][S]
        assert c0["keys"] == c1["keys"]
        fb = cell(c1["comps"]["OWN"]["ade_m"], c0["comps"]["OWN"]["ade_m"], c0["eid"])
        d0 = P0["surfaces"][S]["rows"]["ade_m"]["TRUE-OWNMAP"]["delta"]
        d1 = P1["surfaces"][S]["rows"]["ade_m"]["TRUE-OWNMAP"]["delta"]
        floors[S] = {"F_base_s1_minus_s0": fb, "F": abs(fb["delta"]), "F_contrast": abs(d1 - d0), "use": max(abs(fb["delta"]), abs(d1 - d0))}
    res["floors"] = floors
    gates = {"C1_s1": P1["controls"]["C1_base_reproduces"]["pass"],
             "C2": all(P["controls"]["C2_base_capture_eq_dump"]["pass"] for P in (P0, P1)),
             "C3": all(P["controls"]["C3_recompute_identity"]["pass"] and P["controls"]["C3b_no_target_windows_are_base"]["pass"]
                       for P in (P0, P1)),
             "C6": inj_eq, "C7": all(P["controls"].get("C7_gt_parity", {}).get("pass") for P in (P0, P1)),
             "C8": all(P["controls"]["C8_gt_item_eq_store"]["pass"] for P in (P0, P1)),
             "applied_once": all(P["controls"]["applied_once"]["pass"] for P in (P0, P1))}
    rg = reach_gate(P1["reach"])
    live = {}
    for s, P in (("0", P0), ("1", P1)):
        c = P["surfaces"]["S6"]["rows"]["ade_m"]["SHUF-OWNMAP"]
        live[s] = {"cell": c, "separated_worse": c["lo"] > 0}
    res["gates"] = {"validity": gates, "reach": rg, "liveness": live}
    prim = {}
    for s, P in (("0", P0), ("1", P1)):
        c = P["surfaces"]["S6"]["rows"]["ade_m"]["TRUE-OWNMAP"]
        prim[s] = {"cell": c, "effect": bool(c["hi"] < 0 and -c["delta"] > floors["S6"]["use"]),
                   "adverse": bool(c["lo"] > 0), "share_of_BEV_ZERO": (-c["delta"] / BEV_ZERO_S6[s])}
    res["primary_S6_ade_TRUE_minus_OWNMAP"] = prim
    worse = {}
    for S in ("S2", "S6"):
        for key, r in P1["surfaces"][S]["rows"].items():
            w0 = sep(P0["surfaces"][S]["rows"][key]["TRUE-OWNMAP"], r["lower_is_better"])
            w1 = sep(r["TRUE-OWNMAP"], r["lower_is_better"])
            if w0 == "worse" and w1 == "worse":
                worse.setdefault(S, []).append(key)
    res["family_separated_worse_both_seeds"] = worse
    if not all(gates.values()):
        v = "VOID (a validity control failed: " + ", ".join(k for k, x in gates.items() if not x) + ")"
    elif not rg["pass"]:
        v = "INCONCLUSIVE-REACH (the injected map did not reach the committed fidelity bar)"
    elif not (live["0"]["separated_worse"] and live["1"]["separated_worse"]):
        v = "INCONCLUSIVE-INERT (a WRONG map injected the same way does not hurt the planner: the channel cannot carry map content)"
    elif prim["0"]["adverse"] and prim["1"]["adverse"]:
        v = "ADVERSE (the TRUE map makes the planner WORSE than its own map, at both seeds)"
    elif prim["0"]["effect"] and prim["1"]["effect"]:
        v = "HEADROOM" + (" (with a family cost: " + json.dumps(worse) + ")" if worse else "")
    elif prim["0"]["effect"] or prim["1"]["effect"]:
        v = "PARTIAL (an effect beyond the floor at ONE inference seed only)"
    else:
        v = "NO-HEADROOM"
    res["verdict"] = v
    return res


def _c(c, dp=4):
    if not isinstance(c, dict) or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.{dp}f} [{c['lo']:+.{dp}f}, {c['hi']:+.{dp}f}]"


def render(res: dict) -> str:
    L = [f"# P4 GT-MAP ORACLE -- does refcv8's planner gain from a TRUE map in its BEV?", "",
         f"* Rendered {res['rendered']}; prereg `PREREG_P4.md` sha256 `{res['prereg_sha256']}` (seal `{res['seal']}`).",
         f"* **Tier: {res['tier']}.**", f"* **Interval:** {res['question']}.",
         "* Model: refcv8 EMA-48,852 (md5 b4364029b68f29a33eeb4aebbbbd002b), per-frame nav feed, eval139 (4,754 windows / 139 episodes), "
         "os-only, RTX 4060.", "", f"## Verdict (PREREG s6, mechanical): **{res['verdict']}**", ""]
    if "primary_S6_ade_TRUE_minus_OWNMAP" in res:
        L += ["| seed | S6 ADE TRUE-OWNMAP | effect beyond floor | share of BEV_ZERO (perc2plan) |", "|---|---|---|---|"]
        for s, p in res["primary_S6_ade_TRUE_minus_OWNMAP"].items():
            L.append(f"| s{s} | {_c(p['cell'])} | {p['effect']} | {p['share_of_BEV_ZERO']:+.3f} |")
        L += ["", "### Floors (inference seed)", ""]
        for S, f in res["floors"].items():
            L.append(f"* {S}: F (BASE s1-s0) = {f['F']:.4f} m {_c(f['F_base_s1_minus_s0'])}; F' (contrast s1-s0) = {f['F_contrast']:.4f} m; "
                     f"used max = **{f['use']:.4f} m**")
        L += ["", "### Gates", "", "```", json.dumps(res["gates"], indent=1, default=str)[:6000], "```", ""]
    for s, P in res["per_seed"].items():
        L += [f"## Seed {s}", "", "### Controls", "", "```", json.dumps(P["controls"], indent=1, default=str)[:5000], "```", "",
              "### Census", "", "```", json.dumps(P["census"], indent=1, default=str), "```", ""]
        if P.get("reach"):
            L += ["### Map reach (pooled IoU on the planner crop, target-seen cells)", "",
                  "| pair | mIoU 1-7 | drivable | lane | crosswalk | non-drv-edge |", "|---|---|---|---|---|---|"]
            for nm, r in P["reach"].items():
                pc = r["per_class_all_bands"]
                f = lambda v: "n/a" if v is None else f"{v:.3f}"     # noqa: E731
                L.append(f"| {nm} | {f(r['miou_classes_1to7'])} | {f(pc['drivable'])} | {f(pc['lane'])} | {f(pc['crosswalk'])} | "
                         f"{f(pc['non-drv-edge'])} |")
            L.append("")
        for S, sur in (P.get("surfaces") or {}).items():
            L += [f"### {S} -- n_windows {sur['n_windows']}, episodes {sur['n_episodes']} (all four families; arm - ref)", "",
                  "| family.metric | BASE mean | " + " | ".join(CONTRASTS) + " |", "|---" * (2 + len(CONTRASTS)) + "|"]
            for key, r in sur["rows"].items():
                cells = []
                for c in CONTRASTS:
                    x = r[c]
                    tag = sep(x, r["lower_is_better"])
                    cells.append(_c(x) + (f" **{tag}**" if tag else ""))
                bm = sur["means"]["OWN"].get(key)
                L.append(f"| {r['family']}.{key} | {('n/a' if bm is None else f'{bm:.5g}')} | " + " | ".join(cells) + " |")
            L += ["", f"* STRATEGIC: {sur['STRATEGIC']['status']} (n {sur['STRATEGIC']['n']}): {sur['STRATEGIC']['reason']}",
                  f"* LONGITUDINAL distance-keeping: {sur['LONGITUDINAL_distance_keeping']['status']} (n 0): "
                  f"{sur['LONGITUDINAL_distance_keeping']['reason']}", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plain-dump", default=None)
    ap.add_argument("--identity-npz", default=None)
    a = ap.parse_args()
    if not a.smoke:
        want = SEAL.read_text(encoding="utf-8").split()[0].strip()
        if sha256_file(PREREG) != want:
            raise SystemExit("[p4-analyze] PREREG_P4.md does not match its seal: refusing the scored read")
    root = Path(a.root)
    res = build(root, [s for s in a.seeds.split(",") if s], a.smoke, Path(a.plain_dump) if a.plain_dump else None,
                Path(a.identity_npz) if a.identity_npz else None)
    for P in res["per_seed"].values():
        for k in ("_keys", "_comps", "dpsum"):
            P.pop(k, None)
    name = "P4_SMOKE" if a.smoke else "P4"
    (root / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (root / ("RESULT_SMOKE.md" if a.smoke else "RESULT.md")).write_text(render(res), encoding="utf-8")
    print(f"[p4-analyze] verdict={res['verdict']} -> {root / (name + '.json')} ZZP4ANALYZEDZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
