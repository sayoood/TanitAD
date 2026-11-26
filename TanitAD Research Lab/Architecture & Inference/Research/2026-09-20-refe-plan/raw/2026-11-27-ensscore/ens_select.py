#!/usr/bin/env python3
"""ENS-SCORE launcher (PREREG_ENSSCORE.md section 2.1). Patches `REFePlanner` IN-PROCESS, then runs the next script UNCHANGED (the
refe/ego_yaw.py / trk_select.py pattern), so the navtest seam submits the primary's PLANNED proposal at the ensemble index k_ens.

    python ens_select.py --pkg <PK> --member-ckpt <member model_final.pt> --yaw-table <navtest yaw table> --dump <ens.npz> \
        -- <PK>/refe/ego_yaw.py --table <yaw table> -- refe_navtest_seam.py <seam args>

* the patch: REFePlanner.__init__ (A, the seam's planner) also builds the MEMBER planner B with the planner's ORIGINAL __init__ and the
  SAME keyword arguments except `checkpoint` (so B's model is built and loaded by the planner's own code), refuses unless B's config and
  selection settings equal A's (CTRL-CFG), and registers two hooks: a forward pre-hook WITH KWARGS on A.model (the exact inputs of A's
  forward: image, ego vector, nav input, calib) and a forward pre-hook on each model's score_dec[0] (the scoring context).
  At A's first infer, B gets its OWN ego-yaw shim (EgoYawShim.from_checkpoint(member, the same table), A's allow_incoming, token from A's
  scenario): A's shim wraps A.model.forward BELOW the pre-hook, so the captured ego is the raw one and each model applies its own yaw.
  REFePlanner.infer runs the ORIGINAL infer (A's one forward, k_ship), then B.model on the captured inputs (B's own forward, k_b_own),
  then ens_lib.ens_step, and returns (traj_A, score_A, k_ens). The seam's row is A.executed(traj_A, k_ens); its proposal dump is A's set
  and A's own logits with pick = k_ens.
* at exit (also on a non-zero seam exit), the ENS dump is written: per token k_ship, k_ens, k_id, k_red, k_a_only, k_b_only, k_b_own,
  k_b_own_id, id_maxdiff_a/b, s_a, s_b, s_b_fwd, a_ens, ship8, ship8_b, props8_b, t_ens; meta (code sha256, member, shim scales, argv, GPU memory).
* prints ZZENS <n> id_match=<a>/<n> id_b_match=<b>/<n> red_switch=<r> switch=<k_ens != k_ship> b_only_switch=<k_b_only != k_ship>
Exit code: the wrapped script's (2 refusal).
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
    spec = importlib.util.spec_from_file_location("refe_navtest_seam_ens", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                      # definitions only (the seam's main runs under its __main__ guard)
    return m, p


class State:
    def __init__(self, a):
        self.a = a
        self.recs: dict = {}
        self.order: list = []
        self.n_calls = 0
        self.t0 = time.time()
        self.member: dict = {}


def build_member(PL, orig_init, kw: dict, member_ckpt: str):
    """The member planner B, built by the planner's ORIGINAL __init__ with A's keyword arguments and B's checkpoint."""
    b = PL.REFePlanner.__new__(PL.REFePlanner)
    kb = dict(kw)
    kb["checkpoint"] = member_ckpt
    orig_init(b, **kb)
    return b


def install(a, st: State, PL, EL, EY, to_navsim, build=build_member):
    import torch
    orig_init, orig_infer = PL.REFePlanner.__init__, PL.REFePlanner.infer

    def init(self, *ar, **kw):
        if ar:
            raise RuntimeError("ENS: REFePlanner must be constructed with keyword arguments (the member is built from the same kwargs)")
        orig_init(self, **kw)
        self._ens_cap = {}
        self.model.register_forward_pre_hook(lambda mod, args, kwargs: self._ens_cap.__setitem__("in_a", (args, dict(kwargs))), with_kwargs=True)
        self.model.score_dec[0].register_forward_pre_hook(lambda mod, args: self._ens_cap.__setitem__("sctx_a", args[1]))
        b = build(PL, orig_init, kw, a.member_ckpt)
        EL.check_members(self, b)
        b.model.score_dec[0].register_forward_pre_hook(lambda mod, args: self._ens_cap.__setitem__("sctx_b", args[1]))
        EL.check_set(self.model.queries.new_zeros(EL.N_PROP, EL.N_STEP, EL.TRAJ_DIM), self.model.cfg)
        self._ens_b = b
        self._ens_b_shim_done = False
        st.member.update(ckpt=a.member_ckpt, cfg_equal=True, per_sample_calib=bool(b.per_sample_calib), device=str(b.device))
        if torch.cuda.is_available():
            st.member["cuda_mem_after_load_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
        print(f"ENS: member loaded {a.member_ckpt} (cfg equal, per_sample_calib={b.per_sample_calib}, device={b.device})", flush=True)

    def ensure_member_shim(self):
        if self._ens_b_shim_done:
            return
        sa = getattr(self, "ego_yaw_shim", None)
        b = self._ens_b
        if sa is None:
            if EY.EgoYawShim.from_checkpoint(a.member_ckpt, a.yaw_table) is not None:
                raise RuntimeError("ENS: the member is a yaw checkpoint but the primary runs without the ego-yaw shim")
            st.member["yaw"] = None
        else:
            if sha256_file(a.yaw_table) != sa.table_sha256:
                raise RuntimeError("ENS: --yaw-table is not the table the primary's shim uses")
            sb = EY.EgoYawShim.from_checkpoint(a.member_ckpt, a.yaw_table, allow_incoming=sa.allow_incoming)
            if sb is None:
                raise RuntimeError("ENS: the primary is a yaw checkpoint but the member is not")
            sb.install(b.model, token_source=lambda: EY._scenario_token(self))
            b.ego_yaw_shim = sb
            st.member["yaw"] = {"scale_a": sa.scale, "scale_b": sb.scale, "table_sha256": sa.table_sha256, "allow_incoming": sa.allow_incoming}
        self._ens_b_shim_done = True

    def infer(self, ego, img):
        self._ens_cap.clear()
        traj, score, k_ship = orig_infer(self, ego, img)
        cap = self._ens_cap
        if "in_a" not in cap or "sctx_a" not in cap:
            raise RuntimeError("ENS: the primary forward's inputs / scoring context were not captured")
        ensure_member_shim(self)
        b = self._ens_b
        args, kwargs = cap["in_a"]
        with torch.no_grad():
            traj_b, score_b = b.model(*args, **kwargs)              # the member's OWN forward on the primary's exact inputs
        if "sctx_b" not in cap:
            raise RuntimeError("ENS: the member's scoring context was not captured")
        k_b_own = b._pick(score_b)
        sc = getattr(self, "_scenario", None)
        tok = getattr(sc, "_initial_lidar_token", None)
        if tok is None:
            raise RuntimeError("ENS: no token on planner._scenario")
        rec = EL.ens_step(self, b, traj, score, int(k_ship), cap["sctx_a"], traj_b[0], score_b[0], int(k_b_own), cap["sctx_b"], to_navsim)
        if tok not in st.recs:
            st.order.append(tok)
        st.recs[tok] = rec
        st.n_calls += 1
        if st.n_calls % 500 == 0:
            n = len(st.recs)
            idm = sum(r["k_id"] == r["k_ship"] for r in st.recs.values())
            print(f"    ENS {n} tokens: id_match {idm}/{n}, {time.time() - st.t0:.0f} s", flush=True)
        return traj, score, int(rec["k_ens"])

    PL.REFePlanner.__init__ = init
    PL.REFePlanner.infer = infer


KEYS_I = ("k_ship", "k_ens", "k_id", "k_red", "k_a_only", "k_b_only", "k_b_own", "k_b_own_id")
KEYS_F = ("id_maxdiff_a", "id_maxdiff_b", "t_ens")
KEYS_A = ("s_a", "s_b", "s_b_fwd", "a_ens", "ship8", "ship8_b", "props8_b")


def write_dump(a, st: State, code: dict, rc):
    import numpy as np
    toks = [t for t in st.order if t in st.recs]
    R = [st.recs[t] for t in toks]
    n = len(R)
    if n == 0:
        print("ZZENS 0 (no token reached infer)", flush=True)
        return {"n": 0}
    out = {"token": np.array(toks)}
    for k in KEYS_I:
        out[k] = np.array([r[k] for r in R], np.int64)
    for k in KEYS_F:
        out[k] = np.array([r[k] for r in R], np.float64)
    for k in KEYS_A:
        out[k] = np.stack([r[k] for r in R])
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    summ = {"n": n, "id_match": int((out["k_id"] == out["k_ship"]).sum()), "id_maxdiff_a_max": float(out["id_maxdiff_a"].max()),
            "id_b_match": int((out["k_b_own_id"] == out["k_b_own"]).sum()), "id_maxdiff_b_max": float(out["id_maxdiff_b"].max()),
            "switch": int((out["k_ens"] != out["k_ship"]).sum()), "red_switch": int((out["k_red"] != out["k_ens"]).sum()),
            "b_only_switch": int((out["k_b_only"] != out["k_ship"]).sum()), "t_ens_mean_s": float(out["t_ens"].mean()),
            "wrapped_rc": rc, "member": st.member, "cuda_max_mem_mb": gpu, "code": code, "argv": sys.argv,
            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["meta"] = np.array(json.dumps(summ))
    os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
    tmp = a.dump + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a.dump)
    json.dump(summ, open(os.path.splitext(a.dump)[0] + ".summary.json", "w", encoding="utf-8"), indent=1)
    print(f"ZZENS {n} id_match={summ['id_match']}/{n} id_maxdiff_a_max={summ['id_maxdiff_a_max']:.3g} id_b_match={summ['id_b_match']}/{n} "
          f"id_maxdiff_b_max={summ['id_maxdiff_b_max']:.3g} red_switch={summ['red_switch']} switch={summ['switch']} "
          f"b_only_switch={summ['b_only_switch']} t_ens_mean_s={summ['t_ens_mean_s']:.3f} cuda_max_mem_mb={gpu}", flush=True)
    return summ


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print(__doc__)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="the REFe package of the tip export (its refe/ and eval/)")
    ap.add_argument("--member-ckpt", required=True)
    ap.add_argument("--yaw-table", required=True, help="must be the table passed to ego_yaw.py --table (checked against the primary's shim)")
    ap.add_argument("--dump", required=True)
    a = ap.parse_args(argv[:i])
    script, rest = argv[i + 1], argv[i + 2:]
    if os.path.exists(a.dump):
        print(f"REFUSING: {a.dump} exists (move it aside; an ENS dump is never overwritten)")
        return 2
    if "--table" in rest and os.path.abspath(rest[rest.index("--table") + 1]) != os.path.abspath(a.yaw_table):
        print("REFUSING: --yaw-table differs from ego_yaw.py --table")
        return 2
    sys.path.insert(0, os.path.join(a.pkg, "refe"))
    sys.path.insert(0, HERE)
    import ens_lib as EL
    import ego_yaw as EY
    import planner as PL
    seam_mod, seam_path = load_seam_module(a.pkg)
    code = {os.path.basename(p): sha256_file(p) for p in (os.path.join(HERE, "ens_lib.py"), os.path.abspath(__file__), seam_path,
                                                          os.path.join(a.pkg, "refe", "planner.py"), os.path.join(a.pkg, "refe", "model.py"),
                                                          os.path.join(a.pkg, "refe", "ego_yaw.py"))}
    print(f"ENS launcher: dump {a.dump}; member {a.member_ckpt}; code {json.dumps(code)}", flush=True)
    st = State(a)
    install(a, st, PL, EL, EY, seam_mod.to_navsim)
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
