"""r7m_ceiling_effect.py -- does the max-speed CEILING FILTER move the sharp-turn heading slope?  CPU only, banked dumps only.

    python r7m_ceiling_effect.py --tree step20000ema_unc=C:/.../battery/step20000ema_unc [--tree ...] --out-json ... --out-md ...

WHY.  PREREG_R7_MATCHED_PHASE_TURN.md's known-value control found (KNOWN_VALUE_CONTROL.md): the banked refcv7-50,400 roll has the speed-ceiling
filter INERT (filter-on == filter-off in 4,754 / 4,754 windows) while the launch-tree path -- the path every refcv8 battery used -- has it
ACTING.  So the refcv8-vs-refcv7-50,400 `os` comparisons registered so far compare a ceiling-ON arm with a ceiling-OFF reference.  Every
`dump_s{s}/refcv6_extras.npz` also holds the SAME window rolled with the filter OFF (`filter_off.traj`, [N, 8, 2], same DDIM draw -- r7_roll.Refcv6Proxy),
and `os` == `nav_true.traj[:, :4]` exactly (checked below, per tree).  So the ceiling-OFF `os` of any rolled model is available WITHOUT a GPU, and the
slope can be read on BOTH instruments against the banked (inert) refcv7-50,400 reference that every panel embeds as `b_refcv7_50400_s{s}`.

WHAT IT COMPUTES (S2 grid; through-origin slope of the turn-aligned heading error on the turn-aligned GT heading, `heading_drift.shrinkage` math,
same resampling, differences on the SAME draws): per tree and inference seed and pooled -- s_ON (as banked in the panel), s_OFF (filter_off plan),
s_REF (refcv7-50,400), OFF-ON, ON-REF, OFF-REF with paired episode-cluster bootstrap CIs (B 2000, models FIXED: "another draw of EPISODES" only).
Controls printed beside the numbers: panel `os` vs `nav_true.traj[:, :4]` max abs (must be 0), panel window set vs extras window set, and the fraction
of windows where the filter changes the selected anchor (overall / in the sharp bin).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7m_slope_diff as SD  # noqa: E402
HD = SD.HD


def dump_rows(tree: str, seed: int):
    """{(clip_id, ws): (nav_true.traj[:4], filter_off.traj[:4], sel_on, sel_off)} from `dump_s{seed}`, plus the dump's own `os` check."""
    d = os.path.join(tree, f"dump_s{seed}")
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    z = np.load(os.path.join(d, "refcv6_extras.npz"))
    no_off = "filter_off.traj" not in z.files      # a tree rolled with the ceiling filter OFF (R7M_CEIL_OFF=1) carries no filter_off arm: OFF == ON by construction
    ci, ws = z["clip_index"], z["ws"]
    by_idx = {int(e["episode_index"]): e for e in man["episodes"]}
    out, chk = {}, 0.0
    for r in range(len(ws)):
        e = by_idx[int(ci[r])]
        out[(str(e["clip_id"]), int(ws[r]))] = (z["nav_true.traj"][r, :4], z["nav_true.traj"][r, :4] if no_off else z["filter_off.traj"][r, :4],
                                               int(z["nav_true.sel_idx"][r]), int(z["nav_true.sel_idx"][r] if no_off else z["filter_off.sel_idx"][r]))
    # integrity: the dump's `os` (S2 arm) is exactly the first four slots of nav_true.traj
    for e in man["episodes"]:
        ez = np.load(os.path.join(d, f"ep{int(e['file_index']):03d}.npz"))
        for j, w in enumerate(ez["ws"]):
            chk = max(chk, float(np.abs(ez["os"][j] - out[(str(e["clip_id"]), int(w))][0]).max()))
    dump_rows.last_no_off = no_off
    return out, chk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", action="append", required=True, help="LABEL=PATH (repeatable)")
    ap.add_argument("--contrast", action="append", default=[],
                    help="X:Y (labels of two --tree entries): X-Y slope differences on the ON instrument and on the OFF instrument, same draws")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-boot", type=int, default=HD.N_BOOT)
    ap.add_argument("--ref-prefix", default="b_refcv7_50400")
    ap.add_argument("--taniteval-root", default="E:/Projects/TanitAD/taniteval")
    ap.add_argument("--stack-root", default="E:/Projects/TanitAD/stack")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch, ci, ff = HD.setup(a.taniteval_root, a.stack_root)
    seeds = [int(x) for x in a.seeds.split(",")]
    store = {}
    res = {"tool": "r7m_ceiling_effect.py", "generated": time.strftime("%FT%T"), "n_boot": a.n_boot, "surface": "S2",
           "estimator": "episode-cluster bootstrap, models FIXED (not another training run; not another inference run)",
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)", "trees": {}}
    for spec in a.tree:
        lab, root = spec.split("=", 1)
        per = {}
        ctl = {}
        for s in seeds:
            refl = f"{a.ref_prefix}_s{s}"
            P0 = HD.load_panel(root, "S2", s, refl)
            rows, chk_dump = dump_rows(root, s)
            no_off = bool(dump_rows.last_no_off)
            miss = [k for k in P0["keys"] if k not in rows]
            if miss:
                raise SystemExit(f"{lab} s{s}: {len(miss)} panel windows absent from the dump extras")
            on = np.stack([rows[k][0] for k in P0["keys"]]).astype(np.float32)
            off = np.stack([rows[k][1] for k in P0["keys"]]).astype(np.float32)
            flip = np.asarray([rows[k][2] != rows[k][3] for k in P0["keys"]])
            ctl[f"s{s}"] = {"dump_os_vs_nav_true_traj_max_abs": chk_dump, "panel_os_vs_nav_true_traj_max_abs": float(np.abs(P0["os"] - on).max()),
                            "n_windows": len(P0["keys"]),
                            "filter_off_arm": ("ABSENT -> OFF = ON: this tree was rolled with the max-speed ceiling filter OFF (R7M_CEIL_OFF=1); its ON column is NOT "
                                               "a filter-ON instrument -- read contrasts on the OFF instrument" if no_off else "present")}
            dt = P0["dt"]
            B = HD.gt_bins(ff, torch, P0["g"], P0["v0"], P0["nav_cmd"], dt)
            dh = {"ON": HD.seg_signed_heading(ff, torch, P0["os"], P0["g"], dt), "OFF": HD.seg_signed_heading(ff, torch, off, P0["g"], dt),
                  "REF": HD.seg_signed_heading(ff, torch, P0["ref"], P0["g"], dt)}
            per[s] = {"dh": dh, "P0": P0, "B": B, "flip": flip, "dt": dt}
            ctl[f"s{s}"]["frac_windows_filter_changes_selection"] = float(flip.mean())
            ctl[f"s{s}"]["frac_sharp_windows_filter_changes_selection"] = float(flip[B["turn"] == 3].mean()) if (B["turn"] == 3).any() else None
            ctl[f"s{s}"]["n_sharp_windows"] = int((B["turn"] == 3).sum())
        P0, B, dt = per[seeds[0]]["P0"], per[seeds[0]]["B"], per[seeds[0]]["dt"]
        eid, G = P0["eid"], P0["g"]
        masks = SD.group_masks(B)
        views = {f"seed {s}": per[s]["dh"] for s in seeds}
        if len(seeds) == 2:
            views["pooled (2-seed mean heading error)"] = {k: 0.5 * (per[seeds[0]]["dh"][k] + per[seeds[1]]["dh"][k]) for k in per[seeds[0]]["dh"]}
        tr = {"controls": ctl, "views": {}}
        for vname, dhv in views.items():
            tr["views"][vname] = {}
            for gname in SD.GROUPS:
                for which in ("last segment", "segments 2..K"):
                    got = SD.boot_slopes(ff, torch, dhv, G, dt, eid, B, masks[gname], which, a.n_boot)
                    if got is None:
                        continue
                    sl, n_seg, n_win = got
                    d_on, d_off, d_ref = sl["ON"][1], sl["OFF"][1], sl["REF"][1]
                    tr["views"][vname][f"{gname} | {which}"] = {
                        "n_win": n_win, "n_seg": n_seg,
                        "slope": {k: SD.ci_of(sl[k][1], sl[k][0]) for k in sl},
                        "OFF - ON": SD.ci_of(d_off - d_on, sl["OFF"][0] - sl["ON"][0]),
                        "ON - REF": SD.ci_of(d_on - d_ref, sl["ON"][0] - sl["REF"][0]),
                        "OFF - REF": SD.ci_of(d_off - d_ref, sl["OFF"][0] - sl["REF"][0])}
        res["trees"][lab] = tr
        store[lab] = {s_: {"dh": per[s_]["dh"], "P0": per[s_]["P0"]} for s_ in seeds}
        print(f"[{time.time() - t0:6.1f}s] {lab} done", flush=True)
    res["contrasts"] = {}
    for spec in a.contrast:
        xl, yl = spec.split(":")
        res["contrasts"][spec] = {}
        for s_ in seeds:
            Px, Py = store[xl][s_]["P0"], store[yl][s_]["P0"]
            idx = HD.align(Px, Py)                                   # raises on a window-set / GT mismatch
            dx, dy = store[xl][s_]["dh"], {k: v[idx] for k, v in store[yl][s_]["dh"].items()}
            B = HD.gt_bins(ff, torch, Px["g"], Px["v0"], Px["nav_cmd"], Px["dt"])
            masks = SD.group_masks(B)
            arms = {"X_ON": dx["ON"], "X_OFF": dx["OFF"], "Y_ON": dy["ON"], "Y_OFF": dy["OFF"], "REF": dx["REF"]}
            row = {}
            for gname in SD.GROUPS:
                for which in ("last segment", "segments 2..K"):
                    got = SD.boot_slopes(ff, torch, arms, Px["g"], Px["dt"], Px["eid"], B, masks[gname], which, a.n_boot)
                    if got is None:
                        continue
                    sl, n_seg, n_win = got
                    dr = {k: v[1] for k, v in sl.items()}
                    row[f"{gname} | {which}"] = {
                        "n_win": n_win, "n_seg": n_seg, "slope": {k: SD.ci_of(sl[k][1], sl[k][0]) for k in sl},
                        "X-Y on the ON instrument": SD.ci_of(dr["X_ON"] - dr["Y_ON"], sl["X_ON"][0] - sl["Y_ON"][0]),
                        "X-Y on the OFF instrument": SD.ci_of(dr["X_OFF"] - dr["Y_OFF"], sl["X_OFF"][0] - sl["Y_OFF"][0]),
                        "Y-REF on the ON instrument": SD.ci_of(dr["Y_ON"] - dr["REF"], sl["Y_ON"][0] - sl["REF"][0]),
                        "Y-REF on the OFF instrument": SD.ci_of(dr["Y_OFF"] - dr["REF"], sl["Y_OFF"][0] - sl["REF"][0]),
                        "X-REF on the ON instrument": SD.ci_of(dr["X_ON"] - dr["REF"], sl["X_ON"][0] - sl["REF"][0]),
                        "X-REF on the OFF instrument": SD.ci_of(dr["X_OFF"] - dr["REF"], sl["X_OFF"][0] - sl["REF"][0])}
            res["contrasts"][spec][f"seed {s_}"] = row
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    f = lambda r: f"{r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' SEP' if r['separated'] else ''}"  # noqa: E731
    L = ["# Does the speed-ceiling filter move the sharp-turn heading slope? (S2, CPU, banked dumps)\n",
         f"* generated {res['generated']}; B {a.n_boot}; estimator: {res['estimator']}; tier: {res['tier']}",
         "* ON = the plan as banked in the panel (filter acting, as every refcv8 battery was rolled); OFF = the SAME window, same DDIM draw, filter OFF (`filter_off.traj`); "
         "REF = the banked refcv7-50,400 arm embedded in the panel (its filter was INERT: filter-on == filter-off in 4,754 / 4,754 windows).",
         "* slopes are through-origin (turn-aligned heading error on turn-aligned GT heading); negative = turns less than GT; paired CIs; CIs answer 'another draw of EPISODES' only."]
    for lab, tr in res["trees"].items():
        L.append(f"\n## {lab}\n")
        for sk, c in tr["controls"].items():
            L.append(f"* control {sk}: dump `os` vs `nav_true.traj[:, :4]` max abs {c['dump_os_vs_nav_true_traj_max_abs']:.1e}; panel `os` vs it {c['panel_os_vs_nav_true_traj_max_abs']:.1e} "
                     f"(both must be 0); filter changes the selected anchor on {c['frac_windows_filter_changes_selection']:.3f} of {c['n_windows']} windows, "
                     f"{c['frac_sharp_windows_filter_changes_selection']} of the {c['n_sharp_windows']} sharp windows")
        for key in ("sharp (>= 0.15) | last segment", "curved (GT |yaw| >= 0.02) | last segment", "sharp (>= 0.15) | segments 2..K"):
            L.append(f"\n### {key}\n")
            L.append("| view | n win / seg | s_REF | s_ON | s_OFF | ON - REF [CI] | OFF - REF [CI] | OFF - ON [CI] |")
            L.append("|---|---|---|---|---|---|---|---|")
            for vname, v in tr["views"].items():
                c = v.get(key)
                if not c:
                    continue
                L.append(f"| {vname} | {c['n_win']} / {c['n_seg']} | {c['slope']['REF']['point']:+.4f} | {c['slope']['ON']['point']:+.4f} | {c['slope']['OFF']['point']:+.4f} | "
                         f"{f(c['ON - REF'])} | {f(c['OFF - REF'])} | {f(c['OFF - ON'])} |")
    for spec, per_seed in res["contrasts"].items():
        L.append(f"\n## Contrast X:Y = {spec} (X - Y; ON = filter acting as in the refcv8 batteries, OFF = same windows with the filter off = the banked refcv7-50,400 instrument)\n")
        for sk, row in per_seed.items():
            for key in ("sharp (>= 0.15) | last segment", "curved (GT |yaw| >= 0.02) | last segment"):
                c = row.get(key)
                if not c:
                    continue
                L.append(f"* {sk} -- {key} (n {c['n_win']}): X-Y ON {f(c['X-Y on the ON instrument'])}; X-Y OFF {f(c['X-Y on the OFF instrument'])}; "
                         f"Y-REF ON {f(c['Y-REF on the ON instrument'])}; Y-REF OFF {f(c['Y-REF on the OFF instrument'])}; "
                         f"X-REF ON {f(c['X-REF on the ON instrument'])}; X-REF OFF {f(c['X-REF on the OFF instrument'])}")
    open(a.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("WROTE", a.out_json, a.out_md, f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
