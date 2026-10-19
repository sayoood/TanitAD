"""compare_r7_control.py -- the KNOWN-VALUE CONTROL of PREREG_R7_MATCHED_PHASE_TURN.md section 2 (CPU only, reads two dump trees).

    python compare_r7_control.py --new-root C:/.../battery/r7ctl_step50400_smoke --banked-root E:/refcv7_eval_kit/battery/step50400_a6text \
        --seeds 0,1 --out-json raw/r7_matched_phase/KNOWN_VALUE_CONTROL.json --out-md raw/r7_matched_phase/KNOWN_VALUE_CONTROL.md

Re-rolled refcv7-50,400 (the path intended for the milestones) vs the BANKED `step50400_a6text/dump_s{s}`, window by window
(identity = (clip_id, ws)), every array of `ep*.npz` and `decisions/ep*.npz`.

CRITERION (pre-registered, committed before any milestone number): `os` max abs diff <= 1e-3 m on EVERY compared window, at every
compared inference seed; GT and model-free arms BIT-IDENTICAL.  Anything else = VOID.

Classes of key (the class decides how a difference is read; a key not listed is INFORMATIONAL, never gating, and printed):
  EXACT  -- GT / measured inputs / model-free arms / labels: `g`, `v0`, `ha`, `ha0`, `ha0_ext`, decisions `ep_poses`, `pose_last`, `ego_t0`,
            `gt_future_ext`, `gt_future_valid_ext`, `ha_controls`, `lat_label`, `lon_label`, `route_label`, `nav_cmd`, `nav_valid`.
  GATE   -- `os` (S2 grid, 4 x 2).
  MODEL  -- other deterministic-given-the-window model outputs, reported with the same tolerance but NOT part of the pre-registered
            criterion: `plan_full_nav_true` (8-slot 0.5-6 s plan = what S6 and the 6-s chords read), `os_navzero`, `os_navflip`, `oracle_sel`,
            `plan_full_nav_zero`, `plan_full_nav_flipped`, `sel_bank_*`, and the integer `sel_idx*` (agreement rate).
  EXPECTED-TO-DIFFER on a clip-restricted run: `os_navshuf` and everything `*_nav_shuffled*` / `nav_cmd_shuf` -- the shuffle is a permutation
            among the nav-valid WINDOWS of the roll (refcv3_arm: `ra.shuffle_nav`), so restricting the window set changes it by construction.
Index keys (`eid`, `clip_index`) are not compared (they number the restricted episode list).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

TOL = 1e-3
EXACT_EP = ("g", "v0", "ha", "ha0", "ha0_ext")
GATE_EP = ("os",)
MODEL_EP = ("os_navzero", "os_navflip", "oracle_sel")
SHUF_EP = ("os_navshuf",)
SKIP_EP = ("eid", "clip_index", "ws")
EXACT_DEC = ("ep_poses", "pose_last", "ego_t0", "gt_future_ext", "gt_future_valid_ext", "ha_controls", "lat_label", "lon_label",
             "route_label", "nav_cmd", "nav_valid")
SHUF_DEC_SUBSTR = ("nav_shuffled", "nav_cmd_shuf")


def cls_of(key: str, where: str) -> str:
    if where == "ep":
        if key in SKIP_EP:
            return "skip"
        if key in EXACT_EP:
            return "exact"
        if key in GATE_EP:
            return "gate"
        if key in SHUF_EP:
            return "shuf"
        if key in MODEL_EP:
            return "model"
        return "info"
    if key == "ws":
        return "skip"
    if key in EXACT_DEC:
        return "exact"
    if any(s in key for s in SHUF_DEC_SUBSTR):
        return "shuf"
    if key.startswith(("plan_full_", "sel_bank_", "sel_idx", "sel_score", "gstr_", "fan_term", "reach_keep", "goal_", "lat_pred", "lon_pred",
                       "route_pred", "nav_injected", "anchor_acc", "a_star", "sel_agrees")):
        return "model"
    return "info"


def load_tree(root: str, seed: int):
    d = os.path.join(root, f"dump_s{seed}")
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    return d, {str(e["clip_id"]): int(e["file_index"]) for e in man["episodes"]}, man


def filter_off_diagnostic(new_root: str, banked_root: str, seed: int, tol: float) -> dict:
    """DIAGNOSTIC, NEVER PART OF THE VERDICT (the pre-registered criterion is on `os`).  The launch-tree path rolls every window TWICE under the
    same DDIM draw (r7_roll.Refcv6Proxy): once as built (max-speed ceiling filter ON) and once with the filter OFF (`filter_off.*` in
    refcv6_extras.npz, seed-0 full-arm rolls only).  Reported: (a) new filter-OFF plan vs the BANKED `os` / `plan_full_nav_true`; (b) new filter-ON
    plan vs the same (that is the verdict's arm); (c) in the BANKED roll, how often filter-ON differs from filter-OFF (0 = the ceiling was inert)."""
    out = {"seed": seed}
    try:
        dn = os.path.join(new_root, f"dump_s{seed}")
        db = os.path.join(banked_root, f"dump_s{seed}")
        zn = np.load(os.path.join(dn, "refcv6_extras.npz"))
        zb = np.load(os.path.join(db, "refcv6_extras.npz"))
        if "filter_off.traj" not in zn.files or "filter_off.traj" not in zb.files:
            return {"seed": seed, "status": "no filter_off arm in one of the two dumps (os-only roll, or the filter was off when rolled)"}
        mn = json.load(open(os.path.join(dn, "manifest.json"), encoding="utf-8"))
        mb = json.load(open(os.path.join(db, "manifest.json"), encoding="utf-8"))
        idn = {int(e["episode_index"]): str(e["clip_id"]) for e in mn["episodes"]}
        key_b = {}
        for e in mb["episodes"]:
            ez = np.load(os.path.join(db, f"ep{int(e['file_index']):03d}.npz"))
            dz = np.load(os.path.join(db, "decisions", f"ep{int(e['file_index']):03d}.npz"))
            for j, w in enumerate(ez["ws"]):
                key_b[(str(e["clip_id"]), int(w))] = (ez["os"][j], dz["plan_full_nav_true"][j])
        n = d_off_os = d_off_pf = d_on_os = d_on_pf = 0
        n_on_ne_off_new = 0
        for r in range(len(zn["ws"])):
            k = (idn[int(zn["clip_index"][r])], int(zn["ws"][r]))
            if k not in key_b:
                continue
            bos, bpf = key_b[k]
            n += 1
            d_off_os = max(d_off_os, float(np.abs(zn["filter_off.traj"][r, :4] - bos).max()))
            d_off_pf = max(d_off_pf, float(np.abs(zn["filter_off.traj"][r] - bpf).max()))
            d_on_os = max(d_on_os, float(np.abs(zn["nav_true.traj"][r, :4] - bos).max()))
            d_on_pf = max(d_on_pf, float(np.abs(zn["nav_true.traj"][r] - bpf).max()))
            n_on_ne_off_new += int(zn["nav_true.sel_idx"][r] != zn["filter_off.sel_idx"][r])
        out.update({
            "n_windows": n,
            "new_filter_OFF_vs_banked_os_max_abs": d_off_os, "new_filter_OFF_vs_banked_plan_full_max_abs": d_off_pf,
            "new_filter_ON_vs_banked_os_max_abs": d_on_os, "new_filter_ON_vs_banked_plan_full_max_abs": d_on_pf,
            "new_roll_windows_where_filter_changes_selection": n_on_ne_off_new,
            "banked_roll_n_windows": int(len(zb["ws"])),
            "banked_roll_windows_where_filter_changes_selection": int((zb["nav_true.sel_idx"] != zb["filter_off.sel_idx"]).sum()),
            "banked_roll_filter_on_vs_off_traj_max_abs": float(np.abs(zb["nav_true.traj"] - zb["filter_off.traj"]).max()),
            "reading": ("filter-OFF arm of the new path reproduces the banked plan to <= tol" if (d_off_os <= tol and d_off_pf <= tol and n > 0)
                        else "filter-OFF arm of the new path does NOT reproduce the banked plan")})
    except Exception as e:                                                      # noqa: BLE001 -- a diagnostic that cannot run says so
        out["status"] = f"diagnostic failed: {type(e).__name__}: {str(e)[:200]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--banked-root", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()
    t0 = time.time()
    res = {"tool": "compare_r7_control.py", "generated": time.strftime("%FT%T"), "new_root": a.new_root, "banked_root": a.banked_root,
           "tol_m": a.tol, "criterion": "os max abs diff <= tol on every compared window at every compared seed; GT/model-free bit-identical",
           "seeds": {}, "unreadable": []}
    verdict_ok = True
    reasons = []
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        try:
            dn, cn, mn = load_tree(a.new_root, s)
        except Exception as e:                                           # noqa: BLE001 -- a missing seed dump is a FAIL, named
            res["seeds"][str(s)] = {"status": f"NEW DUMP UNREADABLE: {type(e).__name__}: {str(e)[:200]}"}
            verdict_ok = False
            reasons.append(f"seed {s}: new dump unreadable")
            continue
        db, cb, mb = load_tree(a.banked_root, s)
        rows = {}                                                         # key -> accumulators
        n_win = 0
        n_clips = 0
        missing_clips = [c for c in cn if c not in cb]
        if missing_clips:
            verdict_ok = False
            reasons.append(f"seed {s}: {len(missing_clips)} clip(s) of the new roll absent from the banked dump")
        for clip, fi_new in sorted(cn.items(), key=lambda kv: kv[1]):
            if clip not in cb:
                continue
            fi_b = cb[clip]
            for where, sub in (("ep", ""), ("dec", "decisions")):
                pn = os.path.join(dn, sub, f"ep{fi_new:03d}.npz")
                pb = os.path.join(db, sub, f"ep{fi_b:03d}.npz")
                try:
                    zn, zb = np.load(pn), np.load(pb)
                    ws_n, ws_b = zn["ws"], zb["ws"]
                    pos = {int(w): i for i, w in enumerate(ws_b)}
                    if len(set(int(w) for w in ws_n)) != len(ws_n) or any(int(w) not in pos for w in ws_n):
                        verdict_ok = False
                        reasons.append(f"seed {s} {where} {clip[:8]}: window ids of the new roll are not a subset of the banked ones")
                        continue
                    idx = np.asarray([pos[int(w)] for w in ws_n])
                    if where == "ep":
                        n_win += len(ws_n)
                        n_clips += 1
                    for k in zn.files:
                        c = cls_of(k, where)
                        if c == "skip":
                            continue
                        if k not in zb.files:
                            r = rows.setdefault((where, k), {"cls": c, "absent_in_banked": True})
                            continue
                        xn, xb = zn[k], zb[k]
                        if k in ("ep_poses",):                          # per-EPISODE array, not per-window
                            xb_ = xb
                        else:
                            xb_ = xb[idx] if (xb.shape[:1] == (len(ws_b),)) else xb
                        r = rows.setdefault((where, k), {"cls": c, "max_abs": 0.0, "n_rows": 0, "n_diff_rows": 0, "n_nonfinite_mismatch": 0,
                                                         "dtype": str(xn.dtype), "shape_new": list(xn.shape)})
                        if xn.shape != xb_.shape:
                            r["shape_mismatch"] = [list(xn.shape), list(xb_.shape)]
                            continue
                        if np.issubdtype(xn.dtype, np.floating):
                            fn, fb = np.isfinite(xn), np.isfinite(xb_)
                            r["n_nonfinite_mismatch"] += int((fn != fb).sum())
                            both = fn & fb
                            d = np.where(both, np.abs(xn.astype(np.float64) - xb_.astype(np.float64)), 0.0)
                        else:
                            d = np.abs(xn.astype(np.float64) - xb_.astype(np.float64))
                        r["max_abs"] = max(r["max_abs"], float(d.max()) if d.size else 0.0)
                        per_row = d.reshape(d.shape[0], -1).max(axis=1) if d.ndim >= 1 and d.shape[0] > 0 and xn.shape[:1] == (len(ws_n),) else d.reshape(1, -1).max(axis=1)
                        r["n_rows"] += int(per_row.shape[0])
                        r["n_diff_rows"] += int((per_row > 0).sum())
                except Exception as e:                                    # noqa: BLE001 -- recorded; an unreadable file is a gap, not a pass
                    res["unreadable"].append(f"seed {s} {where} {clip[:8]}: {type(e).__name__}: {str(e)[:160]}")
                    verdict_ok = False
        sd = {"n_windows_compared": n_win, "n_clips_compared": n_clips, "keys": {}}
        gate_ok = True
        for (where, k), r in sorted(rows.items()):
            kk = f"{where}:{k}"
            sd["keys"][kk] = r
            c = r["cls"]
            if r.get("absent_in_banked") or r.get("shape_mismatch"):
                if c in ("exact", "gate"):
                    gate_ok = False
                    reasons.append(f"seed {s} {kk}: absent in banked / shape mismatch ({r.get('shape_mismatch')})")
                continue
            if c == "exact" and (r["max_abs"] != 0.0 or r["n_nonfinite_mismatch"]):
                gate_ok = False
                reasons.append(f"seed {s} {kk}: NOT bit-identical (max abs {r['max_abs']:.3e}, {r['n_diff_rows']}/{r['n_rows']} rows, "
                               f"{r['n_nonfinite_mismatch']} non-finite mismatches)")
            if c == "gate" and (r["max_abs"] > a.tol or r["n_nonfinite_mismatch"]):
                gate_ok = False
                reasons.append(f"seed {s} {kk}: max abs {r['max_abs']:.3e} m > {a.tol} m ({r['n_diff_rows']}/{r['n_rows']} windows differ; "
                               f"{r['n_nonfinite_mismatch']} non-finite mismatches)")
        if n_win == 0:
            gate_ok = False
            reasons.append(f"seed {s}: zero windows compared")
        if ("ep", "os") not in rows:
            gate_ok = False
            reasons.append(f"seed {s}: the `os` array was not compared at all")
        sd["gate_ok"] = gate_ok
        verdict_ok = verdict_ok and gate_ok
        res["seeds"][str(s)] = sd
    if res["unreadable"]:
        reasons.append(f"{len(res['unreadable'])} unreadable file(s)")
    res["filter_off_diagnostic"] = {str(sd_): filter_off_diagnostic(a.new_root, a.banked_root, sd_, a.tol) for sd_ in [int(x) for x in a.seeds.split(",") if x.strip()]}
    res["verdict"] = "PASS" if (verdict_ok and not reasons) else "FAIL (-> every refcv7 milestone number from this path is VOID)"
    res["reasons"] = reasons
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    L = [f"# Known-value control: re-rolled refcv7-50,400 vs the banked `step50400_a6text`\n",
         f"* generated {res['generated']}; new tree `{a.new_root}`; banked tree `{a.banked_root}`; tolerance on `os`: {a.tol} m",
         f"* **VERDICT: {res['verdict']}**"]
    for r_ in reasons:
        L.append(f"  * {r_}")
    for s, sd in res["seeds"].items():
        if "keys" not in sd:
            L.append(f"\n## seed {s}: {sd.get('status')}")
            continue
        L.append(f"\n## inference seed {s}: {sd['n_windows_compared']} windows / {sd['n_clips_compared']} clips; gate_ok = **{sd['gate_ok']}**\n")
        L.append("| file:key | class | max abs diff | rows differing / rows | non-finite mismatches |")
        L.append("|---|---|---|---|---|")
        for kk, r in sd["keys"].items():
            if r.get("absent_in_banked"):
                L.append(f"| {kk} | {r['cls']} | absent in banked | | |")
            elif r.get("shape_mismatch"):
                L.append(f"| {kk} | {r['cls']} | SHAPE {r['shape_mismatch']} | | |")
            else:
                L.append(f"| {kk} | {r['cls']} | {r['max_abs']:.3e} | {r['n_diff_rows']} / {r['n_rows']} | {r['n_nonfinite_mismatch']} |")
    L.append("\n## Diagnostic (NOT part of the verdict): the max-speed ceiling filter, ON vs OFF, new path vs banked roll\n")
    for s_, dg in res["filter_off_diagnostic"].items():
        L.append(f"* seed {s_}: " + json.dumps(dg, default=str))
    open(a.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps({"verdict": res["verdict"], "reasons": reasons[:10]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
