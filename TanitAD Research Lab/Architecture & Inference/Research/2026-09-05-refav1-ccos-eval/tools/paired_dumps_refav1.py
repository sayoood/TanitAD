"""D-REFAV1-CCOS-EVAL — PAIRED comparison of two refav1 dumps on the SAME windows: arm B's `cl`
minus arm A's `cl` (e.g. ccos − cos), per metric of the four families, paired episode-cluster
bootstrap (`taniteval.ci.paired_episode_cluster_bootstrap`), plus each arm against the shared
floors (`ha0`, `ha0_ext`) — all on windows asserted identical by `ws`, `v0` AND the GT `g`
(bit-exact), never assumed. Per-window components come from `refav1_arm._components`
(four_families' own geometry + the canonical trajectory labeller), never re-derived.

Strata (optional, from the box panel's per-window `goalresp_rel`): HOLD (the decoded goal IS
the zero-action rollout, ‖g−z_ref‖/‖z_ref‖ < 1e-6) vs non-HOLD, joined on (episode index, t).

KNOWN-VALUE CONTROL: run with A = B (or the post-hoc `ha0_ext` copy of the same dump) and every
delta must read exactly 0.0000 with a zero-width interval; the tool prints it.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, sys
import numpy as np

_TOOLS = os.path.join(r"C:\Users\Admin\tanitad-wt", "taniteval", "tools")


def _load_arm():
    spec = importlib.util.spec_from_file_location("refav1_arm_paired", os.path.join(_TOOLS, "refav1_arm.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load(d):
    E = []
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        with np.load(f) as z:
            e = {k: z[k] for k in z.files}
        e["_eid"] = os.path.basename(f)[:-4]
        E.append(e)
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8")) if os.path.exists(os.path.join(d, "manifest.json")) else {}
    return E, man


def cat(E, k):
    return np.concatenate([e[k] for e in E])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dump", required=True); ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-dump", required=True); ap.add_argument("--b-name", default="B")
    ap.add_argument("--arm", default="cl")
    ap.add_argument("--panel", default=None, help="box_panel_282.json for the HOLD stratum")
    ap.add_argument("--stack", required=True); ap.add_argument("--taniteval", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.stack); sys.path.insert(0, a.taniteval)
    from taniteval import ci as CI
    ra = _load_arm()
    EA, mA = load(a.a_dump); EB, mB = load(a.b_dump)
    ws_a, ws_b = cat(EA, "ws"), cat(EB, "ws"); v0a, v0b = cat(EA, "v0"), cat(EB, "v0")
    ga, gb = cat(EA, "g"), cat(EB, "g")
    ci_a, ci_b = cat(EA, "clip_index"), cat(EB, "clip_index")
    if not (ws_a.shape == ws_b.shape and np.array_equal(ws_a, ws_b) and np.array_equal(v0a, v0b)
            and np.array_equal(ga, gb) and np.array_equal(ci_a, ci_b)):
        raise SystemExit(f"REFUSED: the two dumps are not on the same windows (ws/v0/g/clip_index differ); "
                         f"A {ws_a.shape} B {ws_b.shape}")
    eid = np.concatenate([[e["_eid"]] * len(e["ws"]) for e in EA])
    dt = float(mA.get("grid", {}).get("dt_s", 0.2))
    arms = {f"{a.a_name}.{a.arm}": cat(EA, a.arm), f"{a.b_name}.{a.arm}": cat(EB, a.arm)}
    for fl in ("ha0", "ha0_ext", "ha"):
        src = EB if all(fl in e for e in EB) else (EA if all(fl in e for e in EA) else None)
        if src is not None:
            arms[fl] = cat(src, fl)
    comps = {k: ra._components(v, ga, dt) for k, v in arms.items()}
    fam_of = ra._FAMILY_OF
    # strata
    strata = {"all": np.ones(len(eid), bool)}
    if a.panel and os.path.exists(a.panel):
        P = json.load(open(a.panel, encoding="utf-8"))
        hold = {(int(r["ep"]), int(r["t"])): (r["goalresp_rel"] < 1e-6) for r in P["rows"]}
        epi = np.concatenate([[int(e["clip_index"][0])] * len(e["ws"]) for e in EA])
        flags = np.array([hold.get((int(x), int(t)), None) for x, t in zip(epi, ws_a)], dtype=object)
        if all(f is not None for f in flags):
            h = flags.astype(bool)
            strata["HOLD"] = h; strata["nonHOLD"] = ~h
        else:
            print("[warn] panel join incomplete; no strata", sum(f is None for f in flags))
    pairs = [(f"{a.a_name}.{a.arm}", f"{a.b_name}.{a.arm}")]
    for fl in ("ha0", "ha0_ext"):
        if fl in arms:
            pairs += [(fl, f"{a.a_name}.{a.arm}"), (fl, f"{a.b_name}.{a.arm}")]
    out = {"tool": "paired_dumps_refav1.py", "a": {"name": a.a_name, "dump": a.a_dump, "cost": mA.get("cost")},
           "b": {"name": a.b_name, "dump": a.b_dump, "cost": mB.get("cost")}, "arm": a.arm,
           "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
           "same_windows_asserted": "ws, v0, g, clip_index bit-exact", "tier": "T1 (cl, ha0, ha0_ext, ha)",
           "estimator": "paired_episode_cluster_bootstrap", "n_boot": a.n_boot, "means": {}, "paired": {}}
    for k, c in comps.items():
        out["means"][k] = {mk: float(np.nanmean(v)) for mk, v in c.items()}
    for sname, msk in strata.items():
        blk = {}
        e_s = eid[msk]
        for x, y in pairs:
            d = {}
            for mk, fam in fam_of.items():
                xv, yv = comps[x][mk][msk], comps[y][mk][msk]
                keep = np.isfinite(xv) & np.isfinite(yv)
                if keep.sum() == 0:
                    continue
                r = CI.paired_episode_cluster_bootstrap(yv[keep], xv[keep], [e for e, kp in zip(e_s, keep) if kp],
                                                        n_boot=a.n_boot, seed=a.seed)
                d[mk] = {"family": fam, "delta": r.get("delta"), "lo": r.get("lo"), "hi": r.get("hi"),
                         "separated": r.get("separated"), "p_delta_gt0": r.get("p_delta_gt0"),
                         "n_windows": int(keep.sum()), "n_episodes": int(len(set(e for e, kp in zip(e_s, keep) if kp)))}
            blk[f"{y} - {x}"] = d
        out["paired"][sname] = {"n_windows": int(msk.sum()), "n_episodes": int(len(set(e_s.tolist()))), "pairs": blk}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    L = [f"paired {a.b_name}.{a.arm} − {a.a_name}.{a.arm} (and each vs the floors), n = {out['n_windows']} / {out['n_episodes']}, "
         f"paired episode-cluster bootstrap n_boot {a.n_boot}; A cost = {mA.get('cost', {}) and mA['cost'].get('metric') or 'cos (pre-flag)'} "
         f"B cost = {mB.get('cost', {}) and (mB['cost'].get('metric'), mB['cost'].get('weights'))}", ""]
    L.append("| stratum | pair | " + " | ".join(fam_of) + " |")
    L.append("|---|---|" + "---|" * len(fam_of))
    for sname, sb in out["paired"].items():
        for pr, d in sb["pairs"].items():
            cells = []
            for mk in fam_of:
                r = d.get(mk)
                if not r:
                    cells.append("—"); continue
                mark = "**" if r["separated"] else ""
                cells.append(f"{mark}{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{mark}")
            L.append(f"| {sname} (n={sb['n_windows']}/{sb['n_episodes']}) | {pr} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("means: " + "; ".join(f"{k}: ade {v['ade_m']:.4f}, cross {v['LAT_cross_mae_m']:.4f}, speed {v['LON_speed_mae_mps']:.4f}, TAClat {v['TAC_traj_lat_correct']:.4f}" for k, v in out["means"].items()))
    md = "\n".join(L); print(md)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
