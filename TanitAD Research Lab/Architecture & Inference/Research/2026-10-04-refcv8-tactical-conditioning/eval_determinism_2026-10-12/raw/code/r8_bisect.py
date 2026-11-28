"""Bisect the refcv8 eval non-determinism to its draw site(s). CPU, rig model, provisional A2 r8 recipe."""
import json, torch
import repro_lib as L
from tanitad.refs import refcv8_conditioning as C

res = {}
def run(tag, emit_live=None, patch=None, reset=False):
    cfg, m = L.build("r8")
    if emit_live is not None:
        m.core.decoder.r8_emit_live = emit_live
    bt = L.R.batch(cfg, b=3)
    undo = patch() if patch else None
    try:
        outs = []
        for _ in range(2):
            if reset:
                m.core.decoder.r8_gen = C.R8Generator(m.cfg.refcv8.seed)
            outs.append(L.fwd_forked(cfg, m, bt))
    finally:
        if undo: undo()
    d = L.differ(L.digests(outs[0]), L.digests(outs[1]))
    nb = int(m.core.decoder.anchors.shape[0])
    base_same = {}
    for k in ("anchor_traj", "u0_hat", "sel_score", "refined_logits", "offset", "goal_dist"):
        a, b = outs[0].get(k), outs[1].get(k)
        if torch.is_tensor(a) and a.dim() >= 2 and a.shape[1] > nb:
            base_same[k] = bool(torch.equal(a[:, :nb], b[:, :nb]))
    res[tag] = {"n_differ": len(d), "differ": d, "n_base": nb,
                "n_cand": int(outs[0]["r8_cand_src"].shape[1]) if "r8_cand_src" in outs[0] else None,
                "base_columns_identical": base_same}
    print(tag, len(d), d, base_same, flush=True)

def fresh(method):
    """Make ONE R8Generator method draw from a FRESH per-call generator (constant seed): that site becomes reproducible."""
    def patch():
        orig = getattr(C.R8Generator, method)
        def f(self, shape, device, *a, **k):
            g = torch.Generator(device=torch.device(device)); g.manual_seed(self.seed)
            if method == "rand":
                return torch.rand(shape, generator=g, device=device)
            return torch.randn(shape, generator=g, device=device, dtype=(a[0] if a else k.get("dtype", torch.float32)))
        setattr(C.R8Generator, method, f)
        return lambda: setattr(C.R8Generator, method, orig)
    return patch

run("A_baseline_emit_live_True")
run("B_baseline_emit_live_False(step0, emit_start 1000)", emit_live=False)
run("C_select_source_anchors_fixed(rand)", emit_live=False, patch=fresh("rand"))
run("D_eps_x_fixed(randn)", emit_live=False, patch=fresh("randn"))
def both():
    u1 = fresh("rand")(); u2 = fresh("randn")()
    return lambda: (u2(), u1())
run("E_both_fixed", emit_live=False, patch=both)
run("F_reset_r8_gen_before_each_call", emit_live=False, reset=True)
json.dump(res, open("r8_bisect.json", "w"), indent=1)
