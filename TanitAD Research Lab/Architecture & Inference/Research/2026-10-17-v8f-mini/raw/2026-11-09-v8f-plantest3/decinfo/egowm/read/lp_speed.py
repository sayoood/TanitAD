"""EGOWM planner harness: ``ladder_plantest/lp_lib.py`` with the speed channel. A THIN layer: lp_lib (and through it the frozen gates) is imported and never edited.

What changes for a stack that carries a speed channel (E-KIN / E-NOW), and nothing else:
  * every LEVEL-1 rollout the planner makes adds ``st.speed_embed(1, speed)`` to the action embedding, in ``rollout_h`` (context 3) and in the stack's own ``rollout`` (context 1; wrapped on the instance);
  * E-KIN: the speed at step h of a candidate = reflect_integrate(v_now, 2.0 x candidate scaled acc[0..h-1]) -- the CANDIDATE actions, differentiably (cumsum / cummin / clamp), so the planner's gradient
    sees the speed path; floored at 0. E-NOW: the measured speed at NOW, held;
  * context-3 history rows (the two real past states) carry the measured speed of their own frames (IMPL_CHOICES_EGOWM.md section 1, last bullet);
  * a speed stack asked to roll with no speed context raises (the stack's own rule), so a speed arm can never be planned without its speed. Levels 2 and 3 (td3) are not supported: TD3 as defined is not computable on strides (3, 5).
The measured speed is the trainer's: kin3[:, 0] of the native pose track (``eval141_speed.npz``), at the window's NOW frame g = t - 15 and at g-2, g-1. (lp_lib's ``ctx.v0`` is column 3 of the same track, used by its
window metrics only; the difference between the two is reported by ``speed_source_gap``.)
Use:  ctx = prepare_arm_speed(name, run_dir)  then  unit_speed(ctx, 'flat1', 'true', 'c1', idx, lam)  (same arguments and result as lp_lib.unit3)."""
import contextlib
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/egowm/tree")
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(THEORY, "ladder_plantest"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lp_lib as LP  # noqa: E402
import lr_lib as LR  # noqa: E402
import eval_speed as ES  # noqa: E402
import plantest_lib as P  # noqa: E402
from tanitad.models.v8f_mini import reflect_integrate, WM_ACC_SCALE  # noqa: E402

_NOW = {"v_now": None, "v_hist": None}          # tensors for the batch in flight: [b], [b, NH]


def plan_speeds(st, a, v_now):
    """Speed at the START of each of the H steps of candidate plans ``a`` [*lead, H, 2] (scaled units) from ``v_now`` [b] (leading dim b). E-NOW: v_now held. E-KIN: floored integral of the candidates' accelerations."""
    lead = a.shape[:-2]
    H = a.shape[-2]
    v0 = v_now.reshape(-1, *([1] * (len(lead) - 1))).expand(*lead)
    if st.ego_mode == "now":
        return v0.unsqueeze(-1).expand(*lead, H)
    acc = a[..., 0] * WM_ACC_SCALE                      # raw m/s^2 (WM_ACC_SCALE is the factor scale_actions divided by)
    return reflect_integrate(v0, acc[..., :H - 1])      # [*lead, H]: offsets 0 .. H-1


def rollout_h_speed(st, z0, a, hist=None, context=1):
    """lp_lib.rollout_h with the speed embedding added (level 1). Same structure, line for line; ``a`` (b, S, H, 2), ``hist`` as lp_lib's."""
    if not hasattr(st, "spd"):
        return _ORIG["rollout_h"](st, z0, a, hist, context)
    if _NOW["v_now"] is None:
        raise RuntimeError("a speed stack was planned with no measured speed (set_speed was not called for this batch)")
    pred = st.predictors[0]
    cap = getattr(pred, "context", None)
    if cap is not None and context > cap:
        raise ValueError(f"context {context} exceeds the level-1 predictor context {cap}")
    b, S, H, _ = a.shape
    n = b * S
    sp = plan_speeds(st, a, _NOW["v_now"])
    e = st.act_embed(a.reshape(n, H, -1)) + st.speed_embed(1, sp.reshape(n, H))
    zs, es = [], []
    if hist is not None:
        nh = hist["z"].shape[1]
        if nh != context - 1:
            raise ValueError(f"history has {nh} states, context {context} needs {context - 1}")
        if _NOW["v_hist"] is None or _NOW["v_hist"].shape[1] != nh:
            raise RuntimeError("context-3 history without the measured speed of its frames")
        zh = hist["z"].unsqueeze(1).expand(b, S, nh, -1).reshape(n, nh, -1)
        eh0 = st.act_embed(hist["a"]) + st.speed_embed(1, _NOW["v_hist"])
        eh = eh0.unsqueeze(1).expand(b, S, nh, -1).reshape(n, nh, -1)
        zs += [zh[:, i] for i in range(nh)]
        es += [eh[:, i] for i in range(nh)]
    else:
        nh = 0
    zs.append(z0.reshape(n, -1))
    for h in range(H):
        es.append(e[:, h])
        zc = torch.stack(zs[-context:], dim=1)
        ec = torch.stack(es[-context:], dim=1)
        zs.append(pred(zc, ec)[:, -1])
    out = torch.stack(zs[nh + 1:], dim=1)
    return out.reshape(b, S, H, -1)


_ORIG = {"rollout_h": LP.rollout_h}


@contextlib.contextmanager
def speed_planner(st):
    """Install for a speed stack: lp_lib.rollout_h -> the speed form, and ``st.rollout`` (the context-1 route and the final chosen-plan rollout) -> level 1 with the plan's speeds. Undone on exit."""
    if not hasattr(st, "spd"):
        yield
        return
    orig_rollout = st.rollout

    def rollout(level, z0, actions, *, context=1, speed=None):
        if level != 1:
            raise NotImplementedError("the EGOWM planner harness plans level 1 only (TD3 is not computable on strides (3, 5))")
        if speed is None:
            if _NOW["v_now"] is None:
                raise RuntimeError("a speed stack was planned with no measured speed (set_speed was not called for this batch)")
            speed = plan_speeds(st, actions, _NOW["v_now"])
        return orig_rollout(level, z0, actions, context=context, speed=speed)
    LP.rollout_h = rollout_h_speed
    st.rollout = rollout
    try:
        yield
    finally:
        LP.rollout_h = _ORIG["rollout_h"]
        del st.rollout                                      # back to the class method
        _NOW.update(v_now=None, v_hist=None)


def set_speed(ctx, ii):
    """Measured speed for the windows ``ii`` of this batch: NOW (frame g) and the two history frames (g-2, g-1)."""
    _NOW["v_now"] = torch.as_tensor(ctx.v_now[ii]).float()
    _NOW["v_hist"] = torch.as_tensor(ctx.v_hist[ii]).float()


def plan_all_speed(ctx, kind, gname, form="c1", idx=None, seed=P.PLAN_SEED, batch=P.BATCH, collect=True, lam=0.0, seed_mode="hold"):
    """lp_lib.plan_all3 line for line, plus ``set_speed`` before each batch (a speed stack only; harmless otherwise)."""
    if kind == "td3" and hasattr(ctx.st, "spd"):
        raise NotImplementedError("td3 with a speed stack")
    st = ctx.st
    k = "flat" if kind == "flat1" else kind
    n = ctx.z_now.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    if form == "c3":
        assert bool(ctx.hist_ok[idx].all()), "context 3 needs the two past frames"
    hold_all = LP.hold_c_of(ctx) if seed_mode == "hold" else None
    out = dict(actions=[], cost={}, init={}, iters={}, mono={})
    goal_lv = 3 if k == "td3" else 1
    with speed_planner(st):
        for j, s0 in enumerate(range(0, len(idx), batch)):
            ii = idx[s0:s0 + batch]
            zn = ctx.z_now[ii]
            g = P.goal_for(st, k, gname, ctx.z_fut[ii], zn, ctx.z_fut[torch.as_tensor(np.asarray(ctx.donor)[ii])])
            hist, cx = LP.hist_of(ctx, ii, form)
            hc = None if hold_all is None else hold_all[ii]
            if hasattr(st, "spd"):
                set_speed(ctx, ii)
            res = LP.run_planner3(st, kind, zn, g, P.make_gen(seed, j), hold_c=hc, lam=lam, seed_mode=seed_mode, hist=hist, context=cx)
            out["actions"].append(res.actions.detach())
            for lv in res.costs:
                out["cost"].setdefault(lv, []).append(res.costs[lv]); out["init"].setdefault(lv, []).append(res.init_costs[lv])
                out["iters"].setdefault(lv, []).append(torch.full((len(ii),), float(res.iters[lv])))
            if collect:
                with torch.no_grad():
                    z0s = st.initial_latents(zn, goal_lv)
                    for lv, roll in res.level_rollouts.items():
                        full = torch.cat([z0s[lv - 1].unsqueeze(1), roll], 1)
                        lifted = st.lift(full, lv, goal_lv, strict=True) if lv < goal_lv else full
                        d = (lifted - g.unsqueeze(1)).pow(2).sum(-1)
                        out["mono"].setdefault(lv, []).append(P.monotonicity(d))
    r = dict(actions=torch.cat(out["actions"]).numpy().astype(np.float64))
    for key in ("cost", "init", "iters", "mono"):
        for lv, v in out[key].items():
            if v:
                r[f"{key}{lv}"] = torch.cat(v).numpy().astype(np.float64)
    return r


def unit_speed(ctx, kind, gname, form, idx, lam, seed=0, seed_mode="hold", collect=True):
    import time
    t = time.time()
    r = plan_all_speed(ctx, kind, gname, form, idx=idx, seed=seed, collect=collect, lam=lam, seed_mode=seed_mode)
    r["plan_raw"] = LP.to_raw(ctx, r.pop("actions"))
    r["idx"] = np.asarray(idx)
    r["wall_s"] = np.array(time.time() - t)
    return r


def attach_speed(ctx, vd=None):
    """ctx.v_now [n] and ctx.v_hist [n, NH] from the trainer's measured speed (eval141_speed.npz) at g = t - 15 and g-2, g-1. A window whose NOW has no measured speed (clip row 0, filled by eval_speed.load) is counted."""
    vd = vd or ES.load()[0]
    n = len(ctx.eid)
    vn, vh = np.zeros(n, np.float32), np.zeros((n, LP.NH), np.float32)
    gap = []
    for i, (e, t) in enumerate(zip(ctx.eid, ctx.t)):
        g = int(t) - 15
        v = vd[str(e)]
        vn[i] = v[g]
        if g >= LP.NH:
            vh[i] = v[g - LP.NH:g]
        if np.isfinite(ctx.v0[i]):
            gap.append(abs(float(v[g]) - float(ctx.v0[i])))
    ctx.v_now, ctx.v_hist = vn, vh
    ctx.speed_source_gap = {"max_abs_m_s": float(np.max(gap)) if gap else None, "mean_abs_m_s": float(np.mean(gap)) if gap else None, "n": len(gap),
                            "note": "kin3[:,0] (trainer, pose differences) vs track column 3 (lp_lib ctx.v0, window metrics only)"}
    return ctx


def prepare_arm_speed(name, run_dir, frames_seed=None, expect_mode=None):
    """lp_lib.prepare_arm for a run folder that may hold a speed stack: BUILT-equals-argv, BUILT speed mode equals the argv's and the expected one, adapter hash, windows, then the measured speed."""
    model, cfg, argv, ck = LR.load_arm(run_dir)
    declared = (argv.get("wm-speed") or ["off"])[0]
    built_mode = model.wm.built_speed()["mode"] if model.wm.speed != "off" else "off"
    if built_mode != declared or (expect_mode or "off") != declared:
        raise LR.LadderMismatch(f"speed mode: BUILT {built_mode}, argv {declared}, expected {expect_mode or 'off'}")
    built = LR.built_values(model)
    checked = LR.check_built_vs_argv(built, argv)
    seed = int(name.rsplit("_s", 1)[1])
    fseed = {0: 0, 1: 1, 2: 0}[seed] if frames_seed is None else frames_seed
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s{fseed}.npz")
    sha = LR.adapter_sha(model)
    if sha != fr["adapter_sha"]:
        raise LR.LadderMismatch(f"{name}: adapter hash {sha[:12]} != cached frames {fr['adapter_sha'][:12]}")
    if int(argv["seed"][0]) != seed:
        raise LR.LadderMismatch(f"{name}: argv --seed {argv['seed']}")
    ev, tr = P.load_latents(LP.EVAL, "M2b_s0"), P.load_latents(LP.TRAIN, "M2b_s0")
    ctx = LP.build_ctx(model.wm, fr, name, ev, tr, LP.td3_defined(built))
    ctx.built, ctx.checked, ctx.adapter_sha, ctx.td3_defined, ctx.speed_mode = built, checked, sha, LP.td3_defined(built), built_mode
    return attach_speed(ctx)
