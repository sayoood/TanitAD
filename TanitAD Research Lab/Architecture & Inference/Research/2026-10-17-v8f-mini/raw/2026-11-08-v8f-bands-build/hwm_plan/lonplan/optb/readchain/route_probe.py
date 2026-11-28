"""Subprocess helper for test_route.py. Usage: route_probe.py ladder|route|ia3pk <ckpt> [n]. Prints one JSON line.
ladder = wmact_lib.load_model under the ladder tree (today's planner route); route = optb_route.load_model (strict); ia3pk = ia3pk_read.load_ckpt after install()."""
import hashlib, json, os, sys
sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
mode, ckpt = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
import torch
torch.set_num_threads(2)
out = {"mode": mode}
try:
    if mode == "ladder":
        os.environ["WMACT_CODE_ROOT"] = "F:/TanitAD-scratch/ladder_tree"
        sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wmact")
        import wmact_lib as W
        m, cfg, ck = W.load_model(ckpt)
    elif mode == "route":
        sys.path.insert(0, HERE)
        import optb_route as R
        m, cfg, ck = R.load_model(ckpt)
        out["remaps"] = R.STATE["remaps"]
    elif mode == "ia3pk":
        sys.path.insert(0, HERE)
        import optb_route as R
        R.install()
        sys.path.insert(0, "F:/TanitAD-scratch/actrsp/optb/read")
        import ia3pk_read as I
        wm, info, M = I.load_ckpt(ckpt)
        out["remaps"] = R.STATE["remaps"]; out["info"] = {k: info[k] for k in ("adapter_sha", "proj_sha", "step")}
        print(json.dumps(out), flush=True); os._exit(0)
    wm = m.wm
    st = wm.stack
    src = torch.load("F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt", weights_only=False)
    Z1 = src["Z1"]["R35_s0"][:n]
    A = wm.scale_actions(src["A_raw"][:n, :10]) if hasattr(wm, "scale_actions") else src["A_raw"][:n, :10]
    with torch.no_grad():
        z = st.rollout(1, Z1[:, 0], A, context=1)
    out["rollout_shape"] = list(z.shape)
    out["rollout_sha256"] = hashlib.sha256(z.detach().float().contiguous().numpy().tobytes()).hexdigest()
    out["n_params"] = sum(p.numel() for p in m.parameters())
    out["has_probe_buffer"] = bool(getattr(st, "rsp_probe_w", None) is not None) if hasattr(st, "rsp_probe_w") else None
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
print(json.dumps(out))
