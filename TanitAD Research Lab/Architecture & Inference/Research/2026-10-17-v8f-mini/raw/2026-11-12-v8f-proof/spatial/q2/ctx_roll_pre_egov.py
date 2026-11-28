r"""ctx_roll.py -- CTXROLL (PREREG_V8F_PROOF_ADD17, note ADD17_N1): the SAME level-1 predictor rolled out to +2.0 s (a) from one state with context 1
(the banked rule), (b) seeded with the true history of its context (3 states), (c) from one state with the context grown by its own predictions.
Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU, no training, no fit: it writes the three rolled-out states in the
layout of the pixel read's latents file, so that rb_trained.py (the r-b lead read) runs on each unchanged.

  python ctx_roll.py --arm G_s0 --ckpt .../G_s0/ckpt_last.pt --tree C:/Users/Admin/v8proof_c/latvis_gw/tree
         --latents C:/.../LATENTS_GW_G_s0.npz --states-pt C:/.../latents/G_s0.pt --out-dir C:/Users/Admin/v8proof_c/ctxroll [--width 640]

Mapped mode (an arm whose pixel job has not written its latents file yet): --map-latents REF.npz --map-states-pt REF.pt take the window list
(eid, g0, t, fold, scaled actions) and the row mapping from a REFERENCE arm; the state file of every arm holds the same windows in the same order
(latvis_gw.cmd_collect asserts it against the banked window list).  With --latents also given the arm's own banked arrays are compared as usual, which
is the check that the mapping does not depend on the arm.  --emit-frame0 also writes LATENTS_F0_<ARM>.npz (start frame 0, context 1: the banked rule
of the r-b read, float32).

Start frame S0 = 4 of the 16-frame window, K = 10 steps of 0.2 s, target frame 14.  Outputs LATENTS_CTX_<ARM>_{R1,R3h,R3g}.npz (z0 = state of
frame 4, pred_2s = the rollout, true_2s = state of frame 14, eid, g0 = window g0 + 4, t, w_fold, act_scaled = transitions 4..13) and
RESULT_CTXROLL_<ARM>.json (controls x1..x4 of the card, the one-step and the latent diagnostics).  Exit 0 valid, 2 a control failed, 3 refused.
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

S0, K, FRAMES = 4, 10, 16
TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)"
TOL = {"x1": 4e-3, "x2": 4e-3, "x3": 1e-5, "x4": 4e-3, "x4i": 1e-4}   # note ADD17_N1: the banked latents file stores float16, so every comparison against it carries its rounding
MATCH_TOL, MATCH_GAP = 4e-3, 0.1


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_stack(tree, ckpt, width):
    """The arm's world-model stack on the CPU, STRICT key matching, tanitad imported from the given tree (asserted)."""
    tree = os.path.abspath(tree).replace(os.sep, "/")
    sys.path.insert(0, tree + "/taniteval")
    sys.path.insert(0, tree + "/stack")
    import torch
    import tanitad
    got = os.path.abspath(tanitad.__file__).replace(os.sep, "/")
    assert got.lower().startswith(tree.lower() + "/stack/"), ("tanitad imported from", got, "expected under", tree)
    from tanitad.models import v8f_mini as M
    sha_file = os.path.join(os.path.dirname(ckpt), "ckpt_last.sha256")
    want = open(sha_file, encoding="utf-8").read().split()[0].lower()
    have = sha256_file(ckpt)
    assert want == have, ("checkpoint sha256", want, have)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    st = model.wm.stack
    assert int(st.cfg[0].latent_dim) == width, (st.cfg[0].latent_dim, width)
    pred = st.predictors[0]
    info = {"ckpt_sha256": have, "step": int(ck.get("step", -1)), "tanitad_file": got, "wm_state": str(cfg.wm_state), "l1_width": int(st.cfg[0].latent_dim),
            "l1_pred_kind": str(getattr(st.cfg[0], "pred_kind", "flat")), "l1_pred_class": type(pred).__name__, "l1_pred_context": int(getattr(pred, "context", 0) or 0)}
    return torch, st, info


def seeded_roll(torch, st, Z, A, s, k, ctx):
    """Rollout of the level-1 predictor from frame s over k steps, seeded with the TRUE states of frames s-ctx+1..s and their actions; the context
    keeps the last ctx states (true ones are replaced by predictions as the rollout advances).  ctx = 1 is HJEPAStack.rollout(context=1)."""
    pred = st.predictors[0]
    e = st.act_embed(A)                                   # [n, T-1, E]: e[:, j] is the action of the transition j -> j+1
    zs = [Z[:, j] for j in range(s - ctx + 1, s + 1)]
    for h in range(k):
        p = s + h                                         # frame index of the newest state in zs
        zc = torch.stack(zs[len(zs) - ctx:], dim=1)
        ec = e[:, p - ctx + 1:p + 1]
        zs.append(pred(zc, ec)[:, -1])
    return zs[-1]


def chunked(fn, n, size=256):
    out = []
    for a in range(0, n, size):
        out.append(fn(slice(a, min(a + size, n))))
    return out


def match_rows(first, z0f):
    """For every window of a latents file (frame-0 state z0f, float16 on disk) the row of the state file: nearest within MATCH_TOL, unique by MATCH_GAP."""
    n = z0f.shape[0]
    rows, multi, unmatched, worst = np.full(n, -1, np.int64), [], 0, 0.0
    for j in range(n):
        dist = np.abs(first - z0f[j]).max(1)
        c = np.flatnonzero(dist <= MATCH_TOL)
        if len(c) == 0 or ((dist > MATCH_TOL).any() and float(dist[dist > MATCH_TOL].min()) < MATCH_GAP):
            unmatched += 1
            continue
        rows[j] = int(c[0])
        worst = max(worst, float(dist[c].max()))
        if len(c) > 1:
            multi.append([int(i) for i in c])
    return rows, multi, unmatched, worst


def load_states(torch, path, width):
    d = torch.load(path, map_location="cpu", weights_only=False)
    Z = d["Z1"]
    Z = torch.as_tensor(np.asarray(Z), dtype=torch.float32) if not torch.is_tensor(Z) else Z.float()
    assert Z.shape[1] == FRAMES and Z.shape[2] == width, tuple(Z.shape)
    return d, Z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--latents", default=None, help="the arm's own banked latents file of the pixel read (optional in mapped mode)")
    ap.add_argument("--states-pt", required=True)
    ap.add_argument("--map-latents", default=None, help="mapped mode: latents file of a REFERENCE arm (window list and row mapping)")
    ap.add_argument("--map-states-pt", default=None, help="mapped mode: state file of the same reference arm")
    ap.add_argument("--map-width", type=int, default=640)
    ap.add_argument("--emit-frame0", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args(argv)
    mapped = a.map_latents is not None
    if mapped != (a.map_states_pt is not None) or (not mapped and a.latents is None):
        print("REFUSED: give --latents, or --map-latents together with --map-states-pt")
        return 3
    os.makedirs(a.out_dir, exist_ok=True)
    torch, st, info = load_stack(a.tree, a.ckpt, a.width)
    torch.set_num_threads(4)
    d, Zall = load_states(torch, a.states_pt, a.width)
    if str(d.get("ckpt_sha256")) != info["ckpt_sha256"]:
        print("REFUSED: the state file was made from another checkpoint")
        return 3
    own = np.load(a.latents, allow_pickle=True) if a.latents else None
    if mapped:
        Lw = np.load(a.map_latents, allow_pickle=True)                     # window list of the reference arm
        dref, Zref = load_states(torch, a.map_states_pt, a.map_width)
        if Zref.shape[0] != Zall.shape[0] or not bool((np.asarray(dref["filled"]) == np.asarray(d["filled"])).all()):
            print("REFUSED: the reference state file does not hold the same windows as the arm's")
            return 3
        rows, multi, unmatched, worst = match_rows(Zref[:, 0].numpy(), np.asarray(Lw["z0"], np.float32))
        multi_diff = max([float((Zref[c[0]] - Zref[i]).abs().max()) for c in multi for i in c[1:]] + [0.0])
        if own is not None and not (np.array_equal(np.asarray(own["eid"]).astype(str), np.asarray(Lw["eid"]).astype(str)) and np.array_equal(own["g0"], Lw["g0"])):
            print("REFUSED: the arm's own latents file lists other windows than the reference")
            return 3
    else:
        Lw = own
        rows, multi, unmatched, worst = match_rows(Zall[:, 0].numpy(), np.asarray(own["z0"], np.float32))
        multi_diff = max([float((Zall[c[0]] - Zall[i]).abs().max()) for c in multi for i in c[1:]] + [0.0])
    if unmatched:
        print("REFUSED: %d windows of the latents file have no unique frame-0 match in the state file" % unmatched)
        return 3
    n = len(rows)
    Z = Zall[rows]                                                         # [n, 16, D] in the order of the window list
    A = torch.as_tensor(np.asarray(Lw["act_scaled"], np.float32))          # [n, 15, 2]; the scaling does not depend on the arm
    assert A.shape[1] == FRAMES - 1, tuple(A.shape)
    with torch.no_grad():
        r0 = torch.cat(chunked(lambda sl: st.rollout(1, Z[sl, 0], A[sl, :K], context=1)[:, K - 1], n))
        r1 = torch.cat(chunked(lambda sl: st.rollout(1, Z[sl, S0], A[sl, S0:S0 + K], context=1)[:, K - 1], n))
        ctx = info["l1_pred_context"] or 3
        r3h = torch.cat(chunked(lambda sl: seeded_roll(torch, st, Z[sl], A[sl], S0, K, ctx), n))
        r3g = torch.cat(chunked(lambda sl: st.rollout(1, Z[sl, S0], A[sl, S0:S0 + K], context=ctx)[:, K - 1], n))
        s1 = torch.cat(chunked(lambda sl: seeded_roll(torch, st, Z[sl], A[sl], S0, K, 1), n))
        x3 = float((s1 - r1).abs().max())
        pred = st.predictors[0]
        e = st.act_embed(A)
        o1 = torch.cat(chunked(lambda sl: pred(Z[sl, S0:S0 + 1], e[sl, S0:S0 + 1])[:, -1], n))
        o3 = torch.cat(chunked(lambda sl: pred(Z[sl, S0 - ctx + 1:S0 + 1], e[sl, S0 - ctx + 1:S0 + 1])[:, -1], n))
    nxt, cur, tgt = Z[:, S0 + 1], Z[:, S0], Z[:, S0 + K]
    mse = lambda u, v: float(((u - v) ** 2).mean())                       # noqa: E731
    one = {"copy": mse(cur, nxt), "context_1": mse(o1, nxt), "context_%d" % ctx: mse(o3, nxt)}
    one["ratio_context_1_over_copy"] = one["context_1"] / one["copy"]
    one["ratio_context_%d_over_copy" % ctx] = one["context_%d" % ctx] / one["copy"]
    lat = {"copy": mse(cur, tgt), "R1": mse(r1, tgt), "R3h": mse(r3h, tgt), "R3g": mse(r3g, tgt)}
    lat["skill_over_copy"] = {k: 1.0 - lat[k] / lat["copy"] for k in ("R1", "R3h", "R3g")}
    lat["frame0"] = {"copy": mse(Z[:, 0], Z[:, K]), "R1": mse(r0, Z[:, K])}
    # per-window errors (mean over the width) for the episode-cluster bootstrap of the one-step and the +2.0 s latent errors (ctx_verdict.py, step_boot)
    pw = lambda u, v: ((u - v) ** 2).mean(1).numpy().astype(np.float64)   # noqa: E731
    np.savez(os.path.join(a.out_dir, "DIAG_CTX_%s.npz" % a.arm), eid=np.asarray(Lw["eid"]).astype(str), step_copy=pw(cur, nxt), step_c1=pw(o1, nxt), step_c3=pw(o3, nxt),
             lat_copy=pw(cur, tgt), lat_R1=pw(r1, tgt), lat_R3h=pw(r3h, tgt), lat_R3g=pw(r3g, tgt), f0_copy=pw(Z[:, 0], Z[:, K]), f0_R1=pw(r0, Z[:, K]))
    # ---- controls
    if own is not None:
        x1 = float(np.abs(r0.numpy() - np.asarray(own["pred_2s"], np.float32)).max())
        x2 = float(np.abs(Z[:, K].numpy() - np.asarray(own["true_2s"], np.float32)).max())
        x2z = float(np.abs(Z[:, 0].numpy() - np.asarray(own["z0"], np.float32)).max())
        c1 = {"available": True, "max_abs_diff": x1, "tol": TOL["x1"], "pass": bool(x1 <= TOL["x1"])}
        c2p = bool(x2 <= TOL["x2"] and x2z <= TOL["x2"] and multi_diff <= 1e-6)
    else:
        x2 = x2z = None
        c1 = {"available": False, "pass": True, "note": "the pixel job of this arm has not written its latents file; compared when it has"}
        c2p = bool(multi_diff <= 1e-6)
    eid = np.array([str(x) for x in Lw["eid"]])
    g0 = np.asarray(Lw["g0"], np.int64)
    pos = {(e_, int(g)): j for j, (e_, g) in enumerate(zip(eid, g0))}
    pairs = [(j, pos[(eid[j], int(g0[j]) + S0)]) for j in range(n) if (eid[j], int(g0[j]) + S0) in pos]
    ja = np.array([p[0] for p in pairs], np.int64)
    jb = np.array([p[1] for p in pairs], np.int64)
    x4i_s = float((Z[ja, S0] - Z[jb, 0]).abs().max()) if len(pairs) else float("nan")       # float32 against float32: the same frame seen by two windows
    x4i_r = float((r1[ja] - r0[jb]).abs().max()) if len(pairs) else float("nan")
    c4 = {"pairs": len(pairs), "internal_state_max_abs_diff": x4i_s, "internal_rollout_max_abs_diff": x4i_r, "internal_tol": TOL["x4i"],
          "pass": bool(len(pairs) > 0 and x4i_s <= TOL["x4i"] and x4i_r <= TOL["x4i"])}
    if own is not None and len(pairs):
        x4s = float(np.abs(Z[ja, S0].numpy() - np.asarray(own["z0"], np.float32)[jb]).max())
        x4r = float(np.abs(r1[ja].numpy() - np.asarray(own["pred_2s"], np.float32)[jb]).max())
        c4.update({"state_max_abs_diff": x4s, "rollout_max_abs_diff": x4r, "tol": TOL["x4"], "pass": bool(c4["pass"] and x4s <= TOL["x4"] and x4r <= TOL["x4"])})
    controls = {"x1_reproduce_banked_rollout": c1,
                "x2_mapping": {"mode": "mapped from a reference arm" if mapped else "own latents file", "windows": int(n), "unmatched": int(unmatched), "match_tol": MATCH_TOL,
                               "match_gap": MATCH_GAP, "worst_match_diff": worst, "windows_with_several_rows": len(multi), "max_diff_between_those_rows": multi_diff,
                               "frame10_vs_banked_true_2s_max_abs_diff": x2, "frame0_vs_banked_z0_max_abs_diff": x2z, "tol": TOL["x2"], "pass": c2p},
                "x3_seeded_loop_context_1_equals_R1": {"max_abs_diff": x3, "tol": TOL["x3"], "pass": bool(x3 <= TOL["x3"])},
                "x4_overlap": c4}
    ok = all(v["pass"] for v in controls.values())
    res = {"arm": a.arm, "tier": TIER, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD17", "start_frame": S0, "steps": K, "context": ctx, "model": info,
           "controls": controls, "controls_pass": bool(ok), "one_step_mse": one, "latent_mse_at_2s": lat,
           "inputs": {"latents": (a.latents or "").replace(os.sep, "/"), "states_pt": a.states_pt.replace(os.sep, "/"), "ckpt": a.ckpt.replace(os.sep, "/"),
                      "map_latents": (a.map_latents or "").replace(os.sep, "/"), "map_states_pt": (a.map_states_pt or "").replace(os.sep, "/")}}
    with open(os.path.join(a.out_dir, "RESULT_CTXROLL_%s.json" % a.arm), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"controls": controls, "one_step_mse": one, "latent_mse_at_2s": lat, "model": info}, indent=1))
    if not ok:
        print("ZZ CTXROLL CONTROL-FAILED %s" % a.arm)
        return 2
    ident = {"eid": Lw["eid"], "t": np.asarray(Lw["t"]), "w_fold": np.asarray(Lw["w_fold"])}
    common = dict(ident, z0=cur.numpy(), true_2s=tgt.numpy(), g0=g0 + S0, act_scaled=A[:, S0:S0 + K].numpy())
    for name, r in (("R1", r1), ("R3h", r3h), ("R3g", r3g)):
        np.savez(os.path.join(a.out_dir, "LATENTS_CTX_%s_%s.npz" % (a.arm, name)), pred_2s=r.numpy(), **common)
    if a.emit_frame0:
        np.savez(os.path.join(a.out_dir, "LATENTS_F0_%s.npz" % a.arm), z0=Z[:, 0].numpy(), pred_2s=r0.numpy(), true_2s=Z[:, K].numpy(), g0=g0, act_scaled=A.numpy(), **ident)
    print("ZZ CTXROLL DONE %s" % a.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
