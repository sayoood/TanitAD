"""Controls of the EGOWM planner harness (lp_speed.py). CPU, 2 threads. Usage: python controls_plan.py  -> CONTROLS_PLAN.json
  pA  speed OFF: a non-speed checkpoint (R35H_s0) planned through lp_speed equals lp_lib.plan_all3, difference 0.0, 8 windows, contexts c1 and c3 (every returned array)
  pB  speed stack with the speed weights at ZERO (kin, now): the plan equals R35H's, difference 0.0, 8 windows, c1 and c3; regression arm: random weights -> the plan differs
  pC  the speeds the planner's rollouts are handed equal a hand integration of the CANDIDATE plan from the measured speed at NOW (kin) / the held speed (now), floor included; mutations: sign flipped, future speed -> red
  pD  gradient: d(goal cost)/d(candidate acc) of E-KIN passes through the speed path (autograd directional derivative equals a central finite difference; the speed-detached derivative does not)
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import lp_speed as S
import lp_lib as LP
import lr_lib as LR
import eval_speed as ES
import plantest_lib as P
from tanitad.models import v8f_mini as M

torch.set_num_threads(2)
R35H = f"{LR.PULLS}/ladder_R35H_s0"
IDX8 = np.arange(0, 1976, 247)[:8]
_C = {}


def base():
    if "base" not in _C:
        LR.memory_guard()
        _C["base"] = S.prepare_arm_speed("R35H_s0", R35H)
    return _C["base"]


def speed_ctx(mode, weights):
    b = base()
    ck = torch.load(os.path.join(R35H, "ckpt_last.pt"), map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**{**ck["cfg"], "wm_speed": mode})
    m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = m.load_state_dict(ck["model"], strict=False)
    assert set(r.missing_keys) == {f"wm.stack.spd.{i}.weight" for i in range(3)}, r.missing_keys
    if weights == "random":
        g = torch.Generator().manual_seed(7)
        for p in m.wm.stack.spd:
            p.weight.data = torch.randn(p.weight.shape, generator=g) * 0.5
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    ev, tr = P.load_latents(LP.EVAL, "M2b_s0"), P.load_latents(LP.TRAIN, "M2b_s0")
    ctx = LP.build_ctx(m.wm, fr, f"speed_{mode}_{weights}", ev, tr, False)
    ctx.speed_mode = mode
    return S.attach_speed(ctx)


def same(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def maxdiff(a, b):
    return max(float(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)).max()) for k in a)


def runs(ctx, fn, form):
    idx = IDX8 if form == "c1" else np.array([i for i in IDX8 if ctx.hist_ok[i]])
    return fn(ctx, "flat1", "true", form, idx=idx, collect=False)


def pA():
    b = base()
    out = {}
    for form in ("c1", "c3"):
        r0 = runs(b, LP.plan_all3, form)
        r1 = runs(b, S.plan_all_speed, form)
        out[form] = {"max_abs_diff": maxdiff(r0, r1), "arrays_bit_equal": same(r0, r1), "n_windows": int(len(r0["actions"]))}
    return {"passed": all(v["arrays_bit_equal"] and v["max_abs_diff"] == 0.0 and v["n_windows"] >= 6 for v in out.values()), "detail": out}


def pB():
    b = base()
    out = {}
    for mode in ("kin", "now"):
        z, rnd = speed_ctx(mode, "zero"), speed_ctx(mode, "random")
        for form in ("c1", "c3"):
            r0 = runs(b, LP.plan_all3, form)
            rz = runs(z, S.plan_all_speed, form)
            rr = runs(rnd, S.plan_all_speed, form)
            out[f"{mode}|{form}"] = {"zero_weights_max_abs_diff": maxdiff(r0, rz), "zero_weights_bit_equal": same(r0, rz), "random_weights_max_abs_diff_(regression: must be > 0)": maxdiff(r0, rr)}
    ok = all(v["zero_weights_bit_equal"] and v["zero_weights_max_abs_diff"] == 0.0 and v["random_weights_max_abs_diff_(regression: must be > 0)"] > 1e-4 for v in out.values())
    return {"passed": ok, "detail": out}


def hand_plan_speeds(mode, v0, acc_raw, H):
    """numpy loop: speeds at offsets 0..H-1 of one candidate (acc_raw [H] m/s^2)."""
    if mode == "now":
        return np.full(H, v0)
    s = [float(v0)]
    for h in range(H - 1):
        s.append(max(0.0, s[-1] + 0.2 * float(acc_raw[h])))
    return np.array(s)


def pC():
    rnd = {m: speed_ctx(m, "random") for m in ("kin", "now")}
    out, ok = {}, True
    g = torch.Generator().manual_seed(3)
    ii = IDX8
    for mode, ctx in rnd.items():
        st = ctx.st
        H, Sn = 12, 5
        a = torch.randn(len(ii), Sn, H, 2, generator=g) * 1.0
        a[0, :, 3:, 0] = -2.0                                   # window 0: hard deceleration so that the floor engages for kin
        ctx.v_now[ii[0]] = 4.0
        handed = {}
        orig = S._ORIG["rollout_h"]
        cls = M.ego_stack_cls()
        orig_cls = cls.rollout

        def spy(self_, level, z0, actions, *, context=1, speed=None):
            handed["speed"] = speed.detach().numpy().copy() if speed is not None else None
            return orig_cls(self_, level, z0, actions, context=context, speed=speed)
        cls.rollout = spy                                     # below the harness wrapper: records what the stack itself is handed
        try:
            with S.speed_planner(st):
                S.set_speed(ctx, ii)
                sp = S.plan_speeds(st, a, S._NOW["v_now"]).numpy()
                st.rollout(1, ctx.z_now[ii].unsqueeze(1).expand(len(ii), Sn, -1), a)
        finally:
            cls.rollout = orig_cls
        sp_h = handed["speed"]
        hand = np.stack([[hand_plan_speeds(mode, float(ctx.v_now[ii[w]]), a[w, s, :, 0].numpy() * 2.0, H) for s in range(Sn)] for w in range(len(ii))])
        err = float(np.abs(hand - sp_h).max())
        engaged = bool(mode == "kin" and (hand == 0.0).any())
        out[mode] = {"max_abs_err_m_s_(handed_to_rollout)": err, "max_abs_err_m_s_(plan_speeds)": float(np.abs(hand - sp).max()), "floor_engaged": engaged,
                     "e_now_constant_over_rollout": bool(mode != "now" or (sp_h == sp_h[..., :1]).all())}
        ok &= err < 1e-4 and (engaged or mode == "now")
    # regressions: sign flipped; E-NOW taking a future speed
    ctx = rnd["kin"]
    st = ctx.st
    a = torch.randn(4, 3, 12, 2, generator=g)
    with S.speed_planner(st):
        S.set_speed(ctx, IDX8[:4])
        good = S.plan_speeds(st, a, S._NOW["v_now"]).numpy()
        flipped = S.plan_speeds(st, a.clone().mul_(torch.tensor([-1.0, 1.0])), S._NOW["v_now"]).numpy()
    hand = np.stack([[hand_plan_speeds("kin", float(ctx.v_now[IDX8[w]]), a[w, s, :, 0].numpy() * 2.0, 12) for s in range(3)] for w in range(4)])
    out["REGRESSION_sign_flipped"] = {"max_abs_err_m_s": float(np.abs(hand - flipped).max()), "goes_red": bool(np.abs(hand - flipped).max() > 1e-3), "unmutated_err": float(np.abs(hand - good).max())}
    ctxn = rnd["now"]
    fut = np.roll(ctxn.v_now, -3)                                  # a speed from another (later) window
    out["REGRESSION_future_speed_now"] = {"differs_from_measured_at_NOW": bool(np.abs(fut[IDX8] - ctxn.v_now[IDX8]).max() > 1e-3)}
    ok &= out["REGRESSION_sign_flipped"]["goes_red"] and out["REGRESSION_future_speed_now"]["differs_from_measured_at_NOW"]
    return {"passed": bool(ok), "detail": out}


def pD():
    ctx = speed_ctx("kin", "random")
    st = ctx.st
    ii = IDX8[:3]
    H = 10
    g = torch.Generator().manual_seed(11)
    a = (torch.randn(len(ii), 2, H, 2, generator=g) * 0.5).requires_grad_(True)
    goal = ctx.z_fut[ii].unsqueeze(1)
    d = torch.randn(a.shape, generator=g)
    d = d / d.norm()

    def cost(aa):
        z0 = ctx.z_now[ii].unsqueeze(1).expand(len(ii), 2, -1)
        S.set_speed(ctx, ii)
        r = st.rollout(1, z0, aa)
        return (r[..., -1, :] - goal).pow(2).sum()
    with S.speed_planner(st):
        c = cost(a)
        gr, = torch.autograd.grad(c, a)
        dd = float((gr * d).sum())
        eps = 1e-2
        with torch.no_grad():
            fd = float((cost(a + eps * d) - cost(a - eps * d)) / (2 * eps))
        # speed-detached derivative: the same cost with the speed handed in as a constant
        orig_roll = st.rollout
        S.set_speed(ctx, ii)
        z0 = ctx.z_now[ii].unsqueeze(1).expand(len(ii), 2, -1)
        sp_const = S.plan_speeds(st, a, S._NOW["v_now"]).detach()
        base_roll = M.ego_stack_cls().rollout
        c2 = (base_roll(st, 1, z0, a, context=1, speed=sp_const)[..., -1, :] - goal).pow(2).sum()
        g2, = torch.autograd.grad(c2, a)
        dd2 = float((g2 * d).sum())
    rel = abs(dd - fd) / max(abs(fd), 1e-12)
    rel_det = abs(dd2 - fd) / max(abs(fd), 1e-12)
    return {"passed": bool(rel < 5e-2 and rel_det > 10 * rel), "detail": {"autograd_directional": dd, "finite_difference": fd, "rel_err_autograd": rel, "speed_detached_directional": dd2, "rel_err_speed_detached": rel_det}}


def main(names):
    todo = names or ["pA", "pB", "pC", "pD"]
    path = os.path.join(HERE, "CONTROLS_PLAN.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for n in todo:
        t = time.time()
        try:
            out[n] = globals()[n]()
        except BaseException as e:
            import traceback
            traceback.print_exc()
            out[n] = {"passed": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}
        out[n]["wall_s"] = round(time.time() - t, 1)
        print(n, out[n]["passed"], json.dumps(out[n])[:700], flush=True)
        json.dump(out, open(path, "w"), indent=1)
    return 0 if all(v.get("passed") for v in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
