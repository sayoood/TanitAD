"""NAVRBD step 0 inputs (runs on the pod, CPU).  The text-nav table, the nav_rb state table and the held-out grid for a join, by the tree's OWN builders (build_navtext_nav.build_split / clock_nav_qa,
build_navtext_nav_rb.build), eval split only.
  proof : python build_inputs_x.py proof TREE NAVTOK JOIN TRACKS PINNED_NAV PINNED_RB OUT_DIR   -> rebuild eval141 and assert the three arrays + the rb table equal the pinned files bit for bit
  build : python build_inputs_x.py build TREE NAVTOK JOIN TRACKS OUT_DIR [--expect-windows 7594 --expect-episodes 545]  -> navtext_nav_x.npz, navtext_nav_rb_x.npz, eval_grid_x.json + reports (+ sha256)
Grid rule = rbdist's (probe/count_rb10k.py): navtok flag 0 and a roundabout class at the NOW step, label present (has and lat_cls >= 0), stride 4, t = 3 mod 4; order = sorted clip id."""
import sys, os, json, hashlib, importlib.util
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_mod(tree, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(tree, "stack", "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_text_nav(tree, tab, J, tracks, split="eval"):
    BN = load_mod(tree, "build_navtext_nav")
    nav, nav_d, clamp, d_raw = BN.build_split(J, tab, split, tracks, -1)
    plus = BN.build_split(J, tab, split, tracks, +1)
    qa, qa_plus = BN.clock_nav_qa(J, nav, nav_d, d_raw), BN.clock_nav_qa(J, plus[0], plus[1], plus[3], "plus")
    return nav, nav_d, clamp, {"G-CLOCK-NAV": qa, "G-CLOCK-NAV_regression_arm_plus_travelled": qa_plus, "G-CLOCK-NAV_pass": bool(qa["pass"] and not qa_plus["pass"]),
                               "n_rowclamp": int(clamp.sum())}


def grid_of(tab, J, rbmod):
    vocab = tab["vocab"]
    isrb = rbmod.rb_class_mask(vocab)
    c, fl = np.asarray(tab["eval_cls_id"]), np.asarray(tab["eval_flag"])
    rb = (fl == 0) & np.where(c >= 0, isrb[np.clip(c, 0, None)], False)
    lab = np.asarray(J["has"]).astype(bool) & (np.asarray(J["lat_cls"]) >= 0)
    off, n = np.asarray(J["off"]), np.asarray(J["n"])
    idx = np.arange(len(c)) - np.repeat(off, n)
    m = rb & lab & (idx % 4 == 3)
    clips = [str(x) for x in J["clip"]]
    win = {}
    for ci, (o, k) in enumerate(zip(off, n)):
        t = np.flatnonzero(m[int(o):int(o + k)])
        if len(t):
            win[clips[ci]] = [int(x) for x in t]
    order = sorted(win)
    return {"order": order, "windows": {c: win[c] for c in order}, "n_windows": int(sum(len(v) for v in win.values())),
            "source": "NAVRBD step 0: rbdist rule (navtok flag 0 + roundabout class at NOW, label present has & lat_cls >= 0), stride 4, t = 3 mod 4; episode i = i-th sorted clip id"}


def main():
    mode, tree, navtok, join, tracks = sys.argv[1:6]
    sys.path.insert(0, os.path.join(tree, "stack"))
    from tanitad.data import v8f_mini_data as D
    _z = np.load(navtok, allow_pickle=False)

    class _Tab(dict):
        files = property(lambda self: list(self.keys()))
    tab = _Tab({k: _z[k] for k in _z.files if k.startswith("eval_") or k == "vocab"})
    J = D.load_join(join)
    TR = D.load_native_tracks(tracks)
    nav, nav_d, clamp, rep_nav = build_text_nav(tree, tab, J, TR)
    RB = load_mod(tree, "build_navtext_nav_rb")
    grid = grid_of(tab, J, RB)
    if mode == "proof":
        pn, prb, out = sys.argv[6:9]
        os.makedirs(out, exist_ok=True)
        P = np.load(pn, allow_pickle=False)
        eq = {"eval_nav": bool(np.array_equal(nav, P["eval_nav"])), "eval_nav_d": bool(np.array_equal(nav_d, P["eval_nav_d"], equal_nan=True)),
              "eval_rowclamp": bool(np.array_equal(clamp, P["eval_rowclamp"]))}
        align = {"clip": np.asarray(J["clip"]).astype(str), "off": np.asarray(J["off"]), "n": np.asarray(J["n"])}
        o, rrep, ok = RB.build(tab, align, {"eval": J}, grid, {"steps": None, "windows": grid["n_windows"], "episodes": len(grid["order"])})
        R = np.load(prb, allow_pickle=False)
        eq["eval_nav_rb"] = bool(np.array_equal(o["eval_nav_rb"], R["eval_nav_rb"]))
        res = {"pinned_nav_sha256": sha(pn), "pinned_rb_sha256": sha(prb), "bit_equal": eq, "all": all(eq.values()), "n_clips": int(len(J["clip"])), "G-CLOCK-NAV_pass": rep_nav["G-CLOCK-NAV_pass"],
               "rb_gate_ok": ok, "pinned_grid_sha256_not_compared": "the pinned eval grid is the 141-episode trunk-probe grid, not the rb-only grid"}
        # per-clip proof on a few clips: the clip_nav rule applied alone equals the pinned rows
        json.dump(res, open(os.path.join(out, "PROOF_eval141.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["all"] else 3)
    out = sys.argv[6]
    ew = int(sys.argv[sys.argv.index("--expect-windows") + 1]) if "--expect-windows" in sys.argv else None
    ee = int(sys.argv[sys.argv.index("--expect-episodes") + 1]) if "--expect-episodes" in sys.argv else None
    if ew is not None and (grid["n_windows"], len(grid["order"])) != (ew, ee):
        raise SystemExit(f"REFUSED: grid {grid['n_windows']} windows / {len(grid['order'])} episodes != expected {ew} / {ee}")
    os.makedirs(out, exist_ok=True)
    meta = json.dumps({"rule": "BUILD_SPEC section 5 (countdown, d_ann = max(30, 6 v), UTURN none, bare text d = 0); eval split only (NAVRBD step 0)", "inputs": {"navtok_sha256": sha(navtok), "tracks_sha256": sha(tracks), "join_sha256": sha(join)}})
    pn = os.path.join(out, "navtext_nav_x.npz")
    np.savez_compressed(pn, eval_clip=np.asarray(J["clip"]).astype(str), eval_off=np.asarray(J["off"]), eval_n=np.asarray(J["n"]), eval_nav=nav, eval_nav_d=nav_d, eval_rowclamp=clamp, meta_json=np.array(meta))
    rep_nav["navtext_nav_x_sha256"] = sha(pn)
    json.dump(rep_nav, open(os.path.join(out, "navtext_nav_x_report.json"), "w"), indent=1)
    align = np.load(pn, allow_pickle=False)
    align = {k: align["eval_" + k] for k in ("clip", "off", "n")}
    o, rrep, ok = RB.build(tab, align, {"eval": J}, grid, {"steps": None, "windows": grid["n_windows"], "episodes": len(grid["order"])})
    if not ok:
        print(json.dumps(rrep, indent=1, default=str)[:3000])
        raise SystemExit("REFUSED: G-CLOCK-RB failed (or its regression arm passed)")
    prb = os.path.join(out, "navtext_nav_rb_x.npz")
    np.savez_compressed(prb, **o, meta_json=np.array(json.dumps({"rule": rrep.get("rule", ""), "inputs": {"navtok_sha256": sha(navtok)}})))
    rrep["navtext_nav_rb_x_sha256"] = sha(prb)
    json.dump(rrep, open(os.path.join(out, "navtext_nav_rb_x_report.json"), "w"), indent=1, default=str)
    pg = os.path.join(out, "eval_grid_x.json")
    json.dump(grid, open(pg, "w"))
    summ = {"navtext_nav_x.npz": sha(pn), "navtext_nav_rb_x.npz": sha(prb), "eval_grid_x.json": sha(pg), "navtok_table_x.npz": sha(navtok), "join_X.npz": sha(join), "tracks": sha(tracks),
            "grid_windows": grid["n_windows"], "grid_episodes": len(grid["order"]), "G-CLOCK-NAV_pass": rep_nav["G-CLOCK-NAV_pass"], "G-CLOCK-RB_pass": rrep["eval"]["G-CLOCK-RB_pass"],
            "rb_on_steps": rrep["eval"]["G-CLOCK-RB"]["n_on"], "rb_grid_windows_on_at_now": rrep["eval"]["G-CLOCK-RB"].get("grid_windows_on_at_now"),
            "regression_arm_fails": not rrep["eval"]["G-CLOCK-RB_regression_arm_shift_5"]["pass"]}
    json.dump(summ, open(os.path.join(out, "INPUTS_SHA256.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
