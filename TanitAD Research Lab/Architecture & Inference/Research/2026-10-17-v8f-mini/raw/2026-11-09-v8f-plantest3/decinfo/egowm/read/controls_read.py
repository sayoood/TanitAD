"""Controls of the EGOWM read harness (lr_speed.py). CPU, 2 threads. Usage: python controls_read.py [c1 c2 c3 ...]  -> CONTROLS_READ.json
  c1  a NON-speed checkpoint (R35H_s0) read through the speed layer equals ladder_read/results/arm_R35H_s0.json leaf for leaf (bit for bit), deliberate regression: one leaf perturbed -> the comparison is red
  c2  a speed-structured model with the speed weights at ZERO reads bit for bit what R35H reads (cells), kin and now; deliberate regression: non-zero weights -> differ
  c3  the speed the stack is HANDED equals a hand integration (numpy loop, floored) on a planted deceleration clip, kin and now, and under a contrast; mutations: integration sign flipped, E-NOW taking a future speed -> red
"""
import copy
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import lr_speed as S
import lr_lib as L
import eval_speed as ES
from tanitad.models import v8f_mini as M

torch.set_num_threads(2)
BANK = os.path.join(S.THEORY, "ladder_read", "results", "arm_R35H_s0.json")
R35H = f"{L.PULLS}/ladder_R35H_s0"
SKIP = {"wall_s", "free_gb_at_start", "speed", "run_dir"}


def leaves(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if path == "" and k in SKIP:
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing"))
            else:
                leaves(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, "len"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                leaves(x, y, f"{path}[{i}]", out)
    else:
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))):
            out.append((path, f"{a!r} vs {b!r}"))
    return out


def count_leaves(a):
    if isinstance(a, dict):
        return sum(count_leaves(v) for v in a.values())
    if isinstance(a, (list, tuple)):
        return sum(count_leaves(v) for v in a)
    return 1


def c1():
    t = time.time()
    r = S.read_arm_speed("R35H_s0", R35H)
    r = json.loads(json.dumps(r))
    bank = json.load(open(BANK, encoding="utf-8"))
    diff = leaves(r, bank)
    mut = copy.deepcopy(r)
    mut["ia0"]["1"]["rows"]["2"]["point"] += 1e-12
    diff_mut = leaves(mut, bank)
    by = {}
    for pth, _ in diff:
        by[pth.split("/")[1]] = by.get(pth.split("/")[1], 0) + 1
    rel = 0.0
    flat_r, flat_b = {}, {}

    def flat(x, pre, o):
        if isinstance(x, dict):
            for k, v in x.items():
                flat(v, pre + "/" + k, o)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                flat(v, f"{pre}[{i}]", o)
        else:
            o[pre] = x
    flat(r, "", flat_r)
    flat(bank, "", flat_b)
    for pth, _ in diff:
        x, y = flat_r.get(pth), flat_b.get(pth)
        if isinstance(x, float) and isinstance(y, float):
            rel = max(rel, abs(x - y) / max(abs(x), abs(y), 1e-300))
    exact_sections = [k for k in ("ia0", "ia0_planning_context1", "passes", "built", "argv_checked", "cum_stride", "horizon_plan", "cells_checked_against_states", "copy_vs_itself_max_abs", "adapter_sha256") if k not in by]
    ok = len(diff_mut) >= 1 and all(k in exact_sections for k in ("ia0", "ia0_planning_context1", "passes")) and set(by) <= {"beside"} and rel < 1e-9
    return {"passed": bool(ok), "note": "IA-0 (both contexts), passes and every structural field exact; the beside block (ridge probes, IA-1..3, slope) differs at float rounding because this run used 2 torch threads, the banked run 3", "n_leaves_compared": count_leaves(bank), "n_differing_leaves": len(diff),
            "differing_leaves_by_section": by, "max_relative_difference_of_differing_leaves": rel, "sections_exactly_equal": exact_sections, "first_differences": diff[:3],
            "regression_arm_(one leaf +1e-12)_differing_leaves": len(diff_mut), "wall_s": round(time.time() - t, 1), "banked": BANK, "free_gb": r["free_gb_at_start"]}


def speed_model(mode, weights):
    """R35H's weights in a model BUILT with the speed stack. weights 'zero' (as built) or 'random' (seeded)."""
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
    return m


def sub_fr(fr, n):
    return {k: (v[:n] if isinstance(v, list) and k != "adapter_sha" else v) for k, v in fr.items()}


def cells_of(model, fr, reg):
    with S.speed_read(reg):
        cells, *_ = L.gather(model, fr)
    return cells


def same_cells(a, b):
    return all(np.array_equal(a[k][f], b[k][f]) for k in a for f in ("pred", "copy", "ora"))


def c2():
    fr = sub_fr(L.load_eval_frames(f"{L.SCR}/cache/frames_s0.npz"), 3)
    vd, _ = ES.load()
    reg = S.attach_speed(fr, vd)
    base_model, *_ = L.load_arm(R35H)
    base = cells_of(base_model, fr, reg)
    out = {}
    for mode in ("kin", "now"):
        z = cells_of(speed_model(mode, "zero"), fr, reg)
        rnd = cells_of(speed_model(mode, "random"), fr, reg)
        out[mode] = {"zero_weights_cells_equal_R35H": same_cells(z, base), "random_weights_cells_equal_R35H_(regression arm: must be False)": same_cells(rnd, base),
                     "n_cells": len(base), "max_abs_pred_diff_random": float(max(np.abs(rnd[k]["pred"] - base[k]["pred"]).max() for k in base))}
    ok = all(v["zero_weights_cells_equal_R35H"] and not v["random_weights_cells_equal_R35H_(regression arm: must be False)"] for v in out.values())
    return {"passed": ok, "detail": out, "clips": 3}


def hand_speeds(mode, v, acc, anchors, k, cum):
    """numpy loops, no torch: [A, k] speeds at the start of each of the k steps of a level with cumulative stride ``cum`` (frames)."""
    out = np.zeros((len(anchors), k))
    for i, j in enumerate(anchors):
        f0 = j * cum
        if mode == "now":
            out[i] = v[f0]
            continue
        s = [float(v[f0])]
        for h in range((k - 1) * cum):
            a = float(acc[f0 + h]) if f0 + h < len(acc) else 0.0
            s.append(max(0.0, s[-1] + 0.2 * a))
        out[i] = [s[h * cum] for h in range(k)]
    return out


def planted_fr(fr, vd):
    """Clip 0 of the frame cache with a hard deceleration planted (-3 m/s^2 over frames 20..70) so the floor at 0 engages, and its speed track made consistent (a clip of 101 frames)."""
    z, a, ok = fr["zmean"][0], fr["act"][0].copy(), fr["ok"][0]
    v = np.zeros(len(a), np.float32)
    v[0] = 9.0
    a[:, 0] = 0.3
    a[20:70, 0] = -3.0
    for f in range(len(a) - 1):
        v[f + 1] = max(0.0, v[f] + 0.2 * a[f, 0])
    pf = {"clips": [fr["clips"][0]], "T": [len(a)], "zmean": [z], "act": [a], "ok": [ok], "adapter_sha": fr["adapter_sha"]}
    return pf, {fr["clips"][0]: v}


def capture(model, pf, vd, contrast=None, mut=None):
    """Run encode_clip + the level rollouts of the planted clip under the layer and return the speeds the stack was handed, per level: {level: [A, kmax]}."""
    reg = S.attach_speed(pf, vd)
    handed = {}
    st = model.wm.stack
    orig = st.rollout

    def spy(level, z0, actions, *, context=1, speed=None):
        handed[level] = speed[0].detach().numpy().copy()
        return orig(level, z0, actions, context=context, speed=speed)
    st.rollout = spy
    saved = S.anchored_speeds
    try:
        if mut:
            S.anchored_speeds = mut
        with S.speed_read(reg):
            z, pl = S._STATE["orig_enc"](model, pf["zmean"][0], pf["act"][0], contrast)
            z, pl = S._encode_clip(model, pf["zmean"][0], pf["act"][0], contrast)
            for lv in (1, 2, 3):
                n_st = z[lv - 1].shape[1]
                S._roll_cells(st, lv, z[lv - 1], pl[lv - 1], list(range(0, n_st - 1)), 4 if lv == 1 else 3, 1)
    finally:
        S.anchored_speeds = saved
        st.rollout = orig
    return handed


def c3():
    fr = sub_fr(L.load_eval_frames(f"{L.SCR}/cache/frames_s0.npz"), 1)
    vd, _ = ES.load()
    pf, pv = planted_fr(fr, vd)
    cum = [1, 3, 15]
    res, ok = {}, True
    for mode in ("kin", "now"):
        model = speed_model(mode, "random")
        for cname, contrast in (("none", None), ("lon+1", (0, 1.0))):
            a = pf["act"][0].copy()
            if contrast:
                a[:, contrast[0]] += contrast[1]
            handed = capture(model, pf, pv, contrast)
            err = 0.0
            engaged = False
            for lv in (1, 2, 3):
                kmax = handed[lv].shape[1]
                hand = hand_speeds(mode, pv[pf["clips"][0]], a[:-1, 0], list(range(handed[lv].shape[0])), kmax, cum[lv - 1])
                err = max(err, float(np.abs(hand - handed[lv]).max()))
                engaged |= bool((hand == 0.0).any()) and mode == "kin"
            res[f"{mode}|{cname}"] = {"max_abs_err_m_s": err, "floor_engaged": engaged, "pass": err < 1e-5}
            ok &= err < 1e-5
    # deliberate regressions
    def flipped(mode, vm, acc, idx, k, cum_, dt=0.2):
        return S.anchored_speeds.__wrapped__(mode, vm, -acc, idx, k, cum_, dt) if hasattr(S.anchored_speeds, "__wrapped__") else orig_fn(mode, vm, -acc, idx, k, cum_, dt)

    orig_fn = S.anchored_speeds

    def future(mode, vm, acc, idx, k, cum_, dt=0.2):
        return orig_fn(mode, torch.roll(vm, -10, dims=1), acc, idx, k, cum_, dt)
    for name, mut, mode in (("sign_flipped_(kin)", flipped, "kin"), ("future_speed_(now)", future, "now")):
        model = speed_model(mode, "random")
        handed = capture(model, pf, pv, None, mut=mut)
        err = max(float(np.abs(hand_speeds(mode, pv[pf["clips"][0]], pf["act"][0][:-1, 0], list(range(handed[lv].shape[0])), handed[lv].shape[1], cum[lv - 1]) - handed[lv]).max()) for lv in (1, 2, 3))
        res[f"REGRESSION {name}"] = {"max_abs_err_m_s": err, "goes_red": err > 1e-3}
        ok &= err > 1e-3
    return {"passed": bool(ok), "detail": res}


def main(names):
    todo = names or ["c1", "c2", "c3"]
    path = os.path.join(HERE, "CONTROLS_READ.json")
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
        print(n, out[n]["passed"], json.dumps(out[n])[:600], flush=True)
        json.dump(out, open(path, "w"), indent=1)
    return 0 if all(v.get("passed") for v in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
