#!/usr/bin/env python3
"""TRK-SCORE launcher (PREREG_TRKSCORE.md section 2.1). Patches `REFePlanner` IN-PROCESS, then runs the next script UNCHANGED (the
refe/ego_yaw.py pattern), so the navtest seam submits the PLANNED proposal at the tracker-consistent index k_trk.

    python trk_select.py --pkg <PK> --nav-root <navsim 3e8291b> --dump <trk.npz> [--red-all | --red-mod 10] [--keep-sim] \
        -- <PK>/refe/ego_yaw.py --table <yaw table> -- refe_navtest_seam.py <seam args>

* the patch: REFePlanner.__init__ also registers a forward pre-hook on model.score_dec[0] (it keeps the scoring context the forward used);
  REFePlanner.infer runs the ORIGINAL infer (one forward, the shipped pick k_ship), then trk_lib.trk_step, and returns (traj, score, k_trk).
  Nothing else changes: the seam's row is planner.executed(traj, k_trk), its proposal dump is the planned set with pick = k_trk.
* at exit (also when the seam exits non-zero: its known rc 1 at the frame control), the TRK dump is written: per token k_ship, k_id, k_trk,
  k_red, id_maxdiff, s_plan, s_trk, trk20, ship8, s0, red_d0, t_trk (+ sim, props8 with --keep-sim), and meta (code sha256, argv, counts).
* prints ZZTRK <n> id_match=<a>/<n> red_tokens=<r> red_switch=<s> red_d0_min=<m> switch=<k_trk != k_ship count>
Exit code: the wrapped script's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_seam_module(pkg: str):
    import importlib.util
    p = os.path.join(pkg, "eval", "refe_navtest_seam.py")
    spec = importlib.util.spec_from_file_location("refe_navtest_seam_trk", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                      # definitions only (the seam's main runs under __main__ guard)
    return m, p


class State:
    def __init__(self, a):
        self.a = a
        self.recs: dict = {}
        self.order: list = []
        self.n_calls = 0
        self.t0 = time.time()


def install(a, st: State, PL, TL, to_navsim):
    orig_init, orig_infer = PL.REFePlanner.__init__, PL.REFePlanner.infer

    def init(self, *ar, **kw):
        orig_init(self, *ar, **kw)
        self._trk_cap = {}
        self.model.score_dec[0].register_forward_pre_hook(lambda mod, args: self._trk_cap.__setitem__("sctx", args[1]))
        TL.check_set(self.model.queries.new_zeros(TL.N_PROP, TL.N_STEP, TL.TRAJ_DIM), self.model.cfg)

    def infer(self, ego, img):
        self._trk_cap.pop("sctx", None)
        traj, score, k_ship = orig_infer(self, ego, img)
        sctx = self._trk_cap.get("sctx")
        if sctx is None:
            raise RuntimeError("TRK: the scoring decoder's context was not captured during the forward")
        sc = getattr(self, "_scenario", None)
        tok = getattr(sc, "_initial_lidar_token", None)
        if tok is None:
            raise RuntimeError("TRK: no token on planner._scenario")
        red = a.red_all or TL.is_red_token(tok, a.red_mod)
        rec = TL.trk_step(self, ego, traj, score, int(k_ship), sctx, to_navsim, red=red, keep_sim=a.keep_sim)
        if tok not in st.recs:
            st.order.append(tok)
        st.recs[tok] = rec
        st.n_calls += 1
        if st.n_calls % 500 == 0:
            n = len(st.recs)
            idm = sum(r["k_id"] == r["k_ship"] for r in st.recs.values())
            print(f"    TRK {n} tokens: id_match {idm}/{n}, {time.time() - st.t0:.0f} s", flush=True)
        return traj, score, int(rec["k_trk"])

    PL.REFePlanner.__init__ = init
    PL.REFePlanner.infer = infer


def write_dump(a, st: State, code: dict, rc):
    import numpy as np
    toks = [t for t in st.order if t in st.recs]
    R = [st.recs[t] for t in toks]
    n = len(R)
    if n == 0:
        print("ZZTRK 0 (no token reached infer)", flush=True)
        return {"n": 0}
    out = dict(token=np.array(toks), k_ship=np.array([r["k_ship"] for r in R], np.int64), k_id=np.array([r["k_id"] for r in R], np.int64),
               k_trk=np.array([r["k_trk"] for r in R], np.int64), k_red=np.array([r["k_red"] for r in R], np.int64),
               id_maxdiff=np.array([r["id_maxdiff"] for r in R], np.float64), red_d0=np.array([r["red_d0"] for r in R], np.float64),
               t_trk=np.array([r["t_trk"] for r in R], np.float64), s_plan=np.stack([r["s_plan"] for r in R]),
               s_trk=np.stack([r["s_trk"] for r in R]), trk20=np.stack([r["trk20"] for r in R]), ship8=np.stack([r["ship8"] for r in R]),
               s0=np.stack([r["s0"] for r in R]))
    if a.keep_sim:
        out["sim"] = np.stack([r["sim"] for r in R])
        out["props8"] = np.stack([r["props8"] for r in R])
    red = out["k_red"] >= 0
    summ = {"n": n, "id_match": int((out["k_id"] == out["k_ship"]).sum()), "id_maxdiff_max": float(out["id_maxdiff"].max()),
            "switch": int((out["k_trk"] != out["k_ship"]).sum()), "red_tokens": int(red.sum()),
            "red_switch": int((out["k_red"][red] != out["k_trk"][red]).sum()) if red.any() else 0,
            "red_d0_min": float(np.nanmin(out["red_d0"][red])) if red.any() else None,
            "t_trk_mean_s": float(out["t_trk"].mean()), "wrapped_rc": rc, "red_mod": a.red_mod, "red_all": a.red_all,
            "keep_sim": a.keep_sim, "code": code, "argv": sys.argv, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["meta"] = np.array(json.dumps(summ))
    os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
    tmp = a.dump + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a.dump)
    json.dump(summ, open(os.path.splitext(a.dump)[0] + ".summary.json", "w", encoding="utf-8"), indent=1)
    print(f"ZZTRK {n} id_match={summ['id_match']}/{n} id_maxdiff_max={summ['id_maxdiff_max']:.3g} red_tokens={summ['red_tokens']} "
          f"red_switch={summ['red_switch']} red_d0_min={summ['red_d0_min']} switch={summ['switch']} t_trk_mean_s={summ['t_trk_mean_s']:.3f}", flush=True)
    return summ


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print(__doc__)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="the REFe package of the tip export (its refe/ and eval/)")
    ap.add_argument("--nav-root", required=True, help="navsim devkit @ 3e8291b (appended to sys.path)")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--red-mod", type=int, default=10)
    ap.add_argument("--red-all", action="store_true")
    ap.add_argument("--keep-sim", action="store_true")
    a = ap.parse_args(argv[:i])
    script, rest = argv[i + 1], argv[i + 2:]
    if os.path.exists(a.dump):
        print(f"REFUSING: {a.dump} exists (move it aside; a TRK dump is never overwritten)")
        return 2
    sys.path.insert(0, os.path.join(a.pkg, "refe"))
    sys.path.insert(0, HERE)
    import trk_lib as TL
    TL.tracker_setup(a.nav_root)                    # fail before the model loads if navsim cannot be imported here
    import planner as PL
    seam_mod, seam_path = load_seam_module(a.pkg)
    code = {os.path.basename(p): sha256_file(p) for p in (os.path.join(HERE, "trk_lib.py"), os.path.abspath(__file__), seam_path,
                                                          os.path.join(a.pkg, "refe", "planner.py"), os.path.join(a.pkg, "refe", "model.py"))}
    code["navsim"] = TL._T["navsim_file"]
    print(f"TRK launcher: dump {a.dump}; red_all={a.red_all} red_mod={a.red_mod}; code {json.dumps(code)}", flush=True)
    st = State(a)
    install(a, st, PL, TL, seam_mod.to_navsim)
    sys.argv = [script] + rest
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    rc = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        write_dump(a, st, code, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
