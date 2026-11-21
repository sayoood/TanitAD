"""n1dh_t0route_lib.py -- shared by n1dh_zpost.py (Z-POST) and n1dh_klim_run.py (K-LIM runner) of PREREG_V8F_N1DH (card sha256 f1df05bc, amendment 1 code of record f2e91e33).  CPU, 2 threads, zero training.

WHY A TOOL AT ALL (MEASURED from source, tree f2e91e33): the D1-HO dumps do NOT carry the route head's logits.  `v8f_t1_eval.HEAD_KEYS = ("ckpt","goal_prob","lat_logits","lon_logits","cons_pred")`
(stack/scripts/v8f_t1_eval.py:277, sha-pinned tool, not edited) is all `--dump-heads` keeps from forward_infer, and neither v8f_t1_eval_rbd.py nor v8f_t1_eval_hb.py adds `route_logits`; the model emits it only
as out["route_logits"] (stack/tanitad/models/v8f_mini.py:1198).  So the T0 route-head read has to run the checkpoint again.

HOW (no second implementation of the evaluator): the SAME argv the dump chain uses (hb_dump_n1dh.sh: hb wrapper kappa 0.2, inner v8f_t1_eval_rbd, deadreckon nav, rb exitdist channel, logged scene, periodic) is run IN PROCESS
with `--replan-steps 30`, so the rollout is ONE tick (H // m = 1): the tick-0 batch does not depend on m (ages_for(0), causal bucket at j = 0), so the tick-0 forward is the dump's tick-0 forward.  A class-level wrapper of
V8FMini.forward_infer records, per call, route_logits / traj30 / kin3[:,0] (and, for K-LIM, the batch) and returns the model's output unchanged.  With one tick there is exactly one forward_infer call per chunk, in window order.
Controls: K-ALIGN (recorded tick-0 plan traj30 equals the dump's `ol`, row for row, after resolving any order difference by value), K-OL-REAL (when a real D1-HO dump of the same arm is given: its `ol` equals ours).
Profiles: "ho" = the D1-HO inputs (defaults are the pod's /root layout; relocate with the flags; the held-out clip caches are ~10 GB, so on the dev box this profile needs them pulled first);
"e141" = the dev-box eval-141 inputs (used to smoke the machinery; a number from it is eval-141, never D1-HO).
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
SR = "F:/TanitAD-scratch/strat1/read"
RB141 = "F:/TanitAD-scratch/strat1/navrbd/b2/tables/navtext_nav_rbd.npz"
RAW141 = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def profile_inputs(profile, a=None):
    """-> dict of the evaluator's input paths for a profile.  `a` (argparse namespace, may be None) overrides the ho paths: ho_tracks, ho_cache, ho_grid, ho_join, ho_navtext, ho_rb"""
    g = (lambda k, d: (getattr(a, k, None) if a is not None else None) or d)
    if profile == "ho":
        return {"tracks": g("ho_tracks", "/root/dscale10k/in/labels/native_tracks_datascale10k.pt"), "cache": g("ho_cache", "/root/strat1/navrbd_step0/cache_view"),
                "grid": g("ho_grid", "/root/strat1/navrbd_step0/out/inputs/eval_grid_x_feasible.json"), "join": g("ho_join", "/root/dscale10k/merged/join_X.npz"),
                "navtext": g("ho_navtext", "/root/strat1/navrbd_step0/out/inputs/navtext_nav_x.npz"), "rb": g("ho_rb", "/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz")}
    if profile == "e141":
        return {"tracks": ART + "/sfresh_inputs/inputs/native_tracks.pt", "cache": ART + "/sfresh_inputs/cache", "grid": SR + "/inputs/eval_grid_windows.json",
                "join": SR + "/inputs/join_eval141.npz", "navtext": SR + "/inputs/navtext_nav.npz", "rb": RB141}
    raise ValueError(profile)


def tool_args(profile, ckpt, tag, out_dir, limit, max_windows, batch, a=None, kappa="0.2"):
    """the argv handed to v8f_t1_eval_hb.main: the dump chain's (hb_dump_n1dh.sh) with --replan-steps 30 (one tick), no tick dump, --no-analyze"""
    P = profile_inputs(profile, a)
    for k, v in P.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"profile {profile}: input {k} missing: {v}")
    return ["--kappa-max", kappa, "--inner", "v8f_t1_eval_rbd", "run", "--ckpt", ckpt, "--tag", tag, "--tracks", P["tracks"], "--eval-cache", P["cache"], "--eval-grid", P["grid"], "--eval-join", P["join"],
            "--out-dir", out_dir, "--replan-steps", "30", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", limit, "--device", "cpu", "--no-analyze",
            "--batch", str(batch)] + (["--max-windows", str(max_windows)] if max_windows else []) + [
            "--nav-text-npz", P["navtext"], "--nav-text-sha256", sha256_file(P["navtext"]), "--nav-rb", "on", "--nav-rb-feat", "exitdist", "--nav-rb-npz", P["rb"], "--nav-rb-sha256", sha256_file(P["rb"]),
            "--nav-rb-variant", "follow"]


def setup_paths(tree, hb_dir=None):
    """tree = the f2e91e33 export; hb_dir = the folder holding v8f_t1_eval_hb.py when it is not in tree/stack/scripts (the pod keeps it in /root/strat1/n1dh/hb; sha256 74fd8445)"""
    tree = os.path.abspath(tree).replace("\\", "/")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TANITEVAL_STACK_OVERRIDE"] = tree + "/stack"
    sys.dont_write_bytecode = True
    for p in ((hb_dir,) if hb_dir else ()) + (tree + "/stack/scripts", tree + "/taniteval", tree + "/stack", HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    return tree


class Capture:
    """class-level wrappers of V8FMini.forward_infer and v8f_mini_data.collate.  collate keeps the FULL collated batch of the tick (the trainer's format: kin3, gt, limit, ... -- the evaluator hands the model only
    INFERENCE_INPUT_KEYS, which has no kin3); forward_infer records what the model emits (output unchanged) and, with keep_batches, the full batch of that call."""

    def __init__(self, keep_batches=False):
        self.calls, self.batches, self.keep_batches, self.n_no_head, self.on = [], [], keep_batches, 0, True
        self.orig = self.orig_collate = None
        self.last_full = None

    def install(self, M, D):
        self.orig, self.orig_collate = M.V8FMini.forward_infer, D.collate
        cap = self

        def collate(items):
            full = cap.orig_collate(items)
            cap.last_full = full
            return full

        def forward_infer(model, batch):
            out = cap.orig(model, batch)
            if cap.on:
                if getattr(model, "route_head", None) is None or "route_logits" not in out:
                    cap.n_no_head += 1
                else:
                    full = cap.last_full
                    if full is None or "kin3" not in full or int(full["kin3"].shape[0]) != int(out["route_logits"].shape[0]):
                        raise RuntimeError("the collated batch of this tick was not captured (kin3 missing or a different batch size)")
                    cap.calls.append({"route_logits": out["route_logits"].detach().float().cpu().numpy(), "traj30": out["traj30"].detach().float().cpu().numpy(),
                                      "kin0": full["kin3"][:, 0].detach().float().cpu().numpy()})
                    if cap.keep_batches:
                        cap.batches.append({k: v.detach().clone() for k, v in full.items() if hasattr(v, "detach")})
            return out
        M.V8FMini.forward_infer = forward_infer
        D.collate = collate

    def uninstall(self, M, D):
        M.V8FMini.forward_infer = self.orig
        D.collate = self.orig_collate


def run_tick0(tree, profile, ckpt, out_dir, tag, limit="causal", max_windows=0, batch=32, a=None, keep_batches=False):
    """run the hb wrapper in process, one tick; returns (Capture, dump_dir).  Raises when the checkpoint has no route head (n_no_head > 0)."""
    setup_paths(tree, getattr(a, "hb_dir", None))
    import torch
    torch.set_num_threads(2)
    import tanitad.models.v8f_mini as M
    from tanitad.data import v8f_mini_data as D
    import v8f_t1_eval_hb as HB
    os.makedirs(out_dir, exist_ok=True)
    cap = Capture(keep_batches)
    cap.install(M, D)
    try:
        rc = HB.main(tool_args(profile, ckpt, tag, out_dir, limit, max_windows, batch, a))
    finally:
        cap.uninstall(M, D)
    if rc not in (0, None):
        raise RuntimeError(f"the T1 tool returned {rc}")
    return cap, os.path.join(out_dir, f"dump_{tag}")


def load_dump(d):
    import numpy as np
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert fs, d
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


def align(cap, dump):
    """K-ALIGN: the recorded tick-0 plans (float32 traj30) equal the dump's `ol` (float64 of the same float32).  Returns (perm, record): perm[i] = index into the recorded rows for dump row i."""
    import numpy as np
    rl = np.concatenate([c["route_logits"] for c in cap.calls])
    tr = np.concatenate([c["traj30"] for c in cap.calls])
    kin0 = np.concatenate([c["kin0"] for c in cap.calls])
    ol = dump["ol"]
    n = len(ol)
    rec = {"n_recorded": int(len(rl)), "n_dump": int(n)}
    if len(rl) != n:
        rec.update(pass_=False, why="row count differs")
        return None, rec
    if np.array_equal(tr.astype(np.float64), ol):
        rec.update(pass_=True, order="identical", max_abs=0.0)
        return np.arange(n), rec
    key = lambda x: np.ascontiguousarray(np.asarray(x, np.float32)).tobytes()          # noqa: E731
    idx = {}
    for i in range(n):
        idx.setdefault(key(tr[i]), []).append(i)
    perm = np.full(n, -1, np.int64)
    for j in range(n):
        lst = idx.get(key(ol[j]))
        if not lst:
            rec.update(pass_=False, why=f"dump row {j}: no recorded plan with the same bytes")
            return None, rec
        perm[j] = lst.pop(0)
    rec.update(pass_=bool((perm >= 0).all()), order="resolved by plan bytes", max_abs=0.0)
    return perm, rec


def kin_labels(g):
    """D1-HO's labels() verbatim (n1dh_ho_read.labels: maneuver_kinematics + factor_from_kinematics at DT 0.2); needs setup_paths + the reader's setup (torch, FF)"""
    import numpy as np
    import torch
    from taniteval import four_families as FF
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(g, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, 0.2)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)
