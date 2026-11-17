"""resolve_reach.py -- K-REACH-e of PREREG_V8F_CKPT120 resolved OFF the GPU, per arm, with the optimiser's parameter names taken from the BUILT model (CPU, torch).
The post-training gate (``stack/scripts/ckpt120_reach.py`` over ``zerograd_gate.analyze``) can only name an optimiser id by the shape alignment, which is ambiguous for the far layer (several 512x... tensors). The trainer registers parameters in a fixed order
(``v8f_mini_train.py`` lines 3831-3844 at 213250ce): decay [named_parameters, requires_grad, ndim >= 2, not sfar / ctx], nodecay [ndim < 2], ctx, ctx_nd, sfar, sfar_nd, then the LoRA matrices.  This script rebuilds that list from the
checkpoint's own cfg, checks every group size against the checkpoint's own ``opt.param_groups`` (a mismatch => NO names file, the gate then names by shape and may read INCONCLUSIVE; never a silent PASS), and runs the gate.
  python resolve_reach.py ARM [ARM ...]          -> reach/<arm>.json  (+ reach/<arm>.names.json)       exit 0 only if every arm reads PASS or HEAD-DEAD (HEAD-DEAD = the arm is INVALID: the read skips it and labels it so); INCONCLUSIVE => exit 2
  python resolve_reach.py --controls             -> reach/CONTROLS.json  on a BANKED trained checkpoint (S1_s0, whose ctx head is a trained head at lr x10): the gate machinery with the far-head prefix pointed at ``ctx_head.`` must read PASS,
                                                     a planted deleted-state head and a planted zero-state head must read HEAD-DEAD.  The control is run before any arm is trusted.
Never touches a GPU (CUDA_VISIBLE_DEVICES=-1) or the pod.
"""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c120_lib as C  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def _stack_path():
    for p in (C.TREE + "/stack", C.TREE + "/stack/scripts", C.TREE + "/taniteval"):
        if p not in sys.path:
            sys.path.insert(0, p)


def optimiser_names(ck):
    """-> (names list in optimiser-id order or None, info).  ``ck`` = the loaded checkpoint dict."""
    import torch  # noqa: F401
    _stack_path()
    from tanitad.models import v8f_mini as M
    m = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    m.load_state_dict(ck["model"])
    named = list(m.named_parameters())
    ctx_ids = {id(p) for p in m.ctx_head.parameters()} if getattr(m, "ctx_head", None) is not None else set()
    sfar_ids = {id(p) for p in m.s_far_out.parameters()} if getattr(m, "s_far_out", None) is not None else set()
    names = [n for n, p in named if p.requires_grad and p.ndim >= 2 and id(p) not in sfar_ids and id(p) not in ctx_ids]
    names += [n for n, p in named if p.requires_grad and p.ndim < 2 and id(p) not in sfar_ids and id(p) not in ctx_ids]
    sizes = [("decay", len(names) - sum(1 for n, p in named if p.requires_grad and p.ndim < 2 and id(p) not in sfar_ids and id(p) not in ctx_ids)),
             ("nodecay", sum(1 for n, p in named if p.requires_grad and p.ndim < 2 and id(p) not in sfar_ids and id(p) not in ctx_ids))]
    by_name = dict(named)
    if ctx_ids:
        cn = [n for n, p in named if id(p) in ctx_ids]
        names += [n for n in cn if by_name[n].ndim >= 2] + [n for n in cn if by_name[n].ndim < 2]
        sizes += [("ctx", sum(1 for n in cn if by_name[n].ndim >= 2)), ("ctx_nd", sum(1 for n in cn if by_name[n].ndim < 2))]
    if sfar_ids:
        sn = [n for n, p in named if id(p) in sfar_ids]
        names += [n for n in sn if by_name[n].ndim >= 2] + [n for n in sn if by_name[n].ndim < 2]
        sizes += [("sfar", sum(1 for n in sn if by_name[n].ndim >= 2)), ("sfar_nd", sum(1 for n in sn if by_name[n].ndim < 2))]
    opt = ck.get("opt") or ck.get("optimizer")
    groups = [(g.get("name") or ("decay" if i == 0 else "nodecay" if i == 1 else f"g{i}"), len(g["params"])) for i, g in enumerate(opt["param_groups"])]
    info = {"built_groups": sizes, "ckpt_groups": groups}
    n_lora = sum(n for g, n in groups if g == "lora")
    names += [f"lora.{i}" for i in range(n_lora)]                    # placeholders: the encoder's own matrices are not in the model's state dict
    built = dict(sizes)
    for g, n in groups:
        if g != "lora" and built.get(g) != n:
            info["mismatch"] = f"group {g}: built {built.get(g)} vs checkpoint {n}"
            return None, info
    if sum(n for _, n in groups) != len(names):
        info["mismatch"] = f"{len(names)} names for {sum(n for _, n in groups)} optimiser ids"
        return None, info
    return names, info


def resolve(arm, ckd=None, reach_dir=None):
    import torch
    reach_dir = reach_dir or C.REACH
    os.makedirs(reach_dir, exist_ok=True)
    ck_path = f"{ckd or C.CKD}/{arm}/ckpt_last.pt"
    out = f"{reach_dir}/{arm}.json"
    why = C.bank_refusal(arm, ckd)
    if why:
        json.dump({"schema": "ckpt120-reach-gate-v1", "verdict": "INCONCLUSIVE", "reasons": [why]}, open(out, "w"), indent=1)
        return "INCONCLUSIVE", out
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    names, info = optimiser_names(ck)
    del ck
    argv = [PY, C.TREE + "/stack/scripts/ckpt120_reach.py", "--ckpt", ck_path, "--out", out, "--expect-nav-s", "on" if arm in C.NAV_S_ARMS else "off"]
    if names is not None:
        npath = f"{reach_dir}/{arm}.names.json"
        json.dump(names, open(npath, "w"))
        argv += ["--param-names", npath]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", PYTHONPATH=";".join(os.path.normpath(p) for p in (C.TREE + "/stack", C.TREE + "/taniteval")))
    subprocess.run(argv, env=env, encoding="utf-8")
    rep = json.load(open(out, encoding="utf-8"))
    rep["names_source"] = "built model (trainer optimiser order)" if names is not None else "shape alignment (group sizes did not match the checkpoint)"
    rep["names_info"] = info
    json.dump(rep, open(out, "w"), indent=1)
    return rep["verdict"], out


def controls(ckpt=None, out=None):
    """the gate on a banked trained checkpoint, the far-head prefix pointed at its trained ctx head: PASS; planted deleted state and planted zero state: HEAD-DEAD."""
    import torch
    _stack_path()
    sys.path.insert(0, C.TREE + "/stack/scripts")
    import ckpt120_reach as R
    ckpt = ckpt or f"{C.SL.HERE}/ckpts/S1_s0/ckpt_last.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    names, info = optimiser_names(ck)
    old = dict(R.PREFIXES)
    R.PREFIXES["far"], R.PREFIXES["nav_s"] = "ctx_head.", "e_nav_s."
    res = {"ckpt": ckpt.replace("\\", "/"), "names_info": info, "stand_in": "ctx_head.* stands in for s_far_out.* (S1_s0 has no far head; its ctx head is a trained head at lr x10)"}
    try:
        base = R.analyze(ck, False, R.DEFAULT_CONTROL, names)
        res["trained_head_reads"] = base["verdict"]
        target = "ctx_head.0.weight"
        res["planted_deleted_state"] = R.analyze(ck, False, R.DEFAULT_CONTROL, names, _mutate=target)["verdict"]
        ck2 = copy.deepcopy(ck)
        opt = ck2.get("opt") or ck2.get("optimizer")
        victim = next(i for i, nm in R._names_by_id(ck2, names).items() if nm == [target])
        for k, v in list(opt["state"][victim].items()):
            if hasattr(v, "zero_"):
                v.zero_()
        res["planted_zero_state"] = R.analyze(ck2, False, R.DEFAULT_CONTROL, names)["verdict"]
    finally:
        R.PREFIXES.update(old)
    res["pass"] = bool(names is not None and res["trained_head_reads"] == "PASS" and res["planted_deleted_state"] == "HEAD-DEAD" and res["planted_zero_state"] == "HEAD-DEAD")
    json.dump(res, open(out or f"{C.REACH}/CONTROLS.json", "w"), indent=1)
    return res


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    os.makedirs(C.REACH, exist_ok=True)
    if a and a[0] == "--controls":
        r = controls()
        print("K-REACH-e controls:", {k: r[k] for k in ("trained_head_reads", "planted_deleted_state", "planted_zero_state", "pass")})
        return 0 if r["pass"] else 2
    rc = 0
    for arm in a:
        v, p = resolve(arm)
        print(f"K-REACH-e {arm}: {v} ({p})", flush=True)
        if v == "INCONCLUSIVE":
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
