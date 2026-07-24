"""A40 torch latency reference: independently reproduces 'CUDA graph is the rollout lever, fp16 is not'.
PREDICTOR-ONLY rollout proxy (20 predictor forwards; no step-readout/window-slide) — NOT the full tick.
Full-tick numbers stay cited from the 07-20 levers note (eager 100.29 -> composed 18.75 ms)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "/workspace/TanitAD/stack")
from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel


def build(action_dim=3):
    cfg = flagship4b_config()
    try:
        cfg.predictor.action_dim = action_dim
        w = WorldModel(cfg)
    except Exception:
        cfg = flagship4b_config(); w = WorldModel(cfg)
    return w.cuda().eval(), cfg


def cuda_time(fn, warmup=15, reps=80):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    a = np.array(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}


def imagine1(world, st, ac):
    out = world.imagine(st, ac)
    k = tuple(world.predictor.cfg.horizons)[0]
    return out[k]


def main():
    K = 20
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "K": K, "note": "predictor-only rollout proxy; NOT the full tick (cite 07-20 levers note)"}
    world, cfg = build(3)
    W = cfg.predictor.window; S = world.state_dim; A = cfg.predictor.action_dim
    st = torch.randn(1, W, S, device="cuda"); ac = torch.randn(1, W, A, device="cuda")
    with torch.no_grad():
        res["predictor_1call_fp32"] = cuda_time(lambda: imagine1(world, st, ac))
        res["rollout_k20_eager_fp32"] = cuda_time(lambda: [imagine1(world, st, ac) for _ in range(K)], warmup=8, reps=40)
        # CUDA-graph single-step capture, replayed K times
        try:
            s2 = st.clone(); a2 = ac.clone()
            strm = torch.cuda.Stream(); strm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(strm):
                for _ in range(3): _ = imagine1(world, s2, a2)
            torch.cuda.current_stream().wait_stream(strm)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _out = imagine1(world, s2, a2)
            res["rollout_k20_graph_step_fp32"] = cuda_time(lambda: [g.replay() for _ in range(K)], warmup=8, reps=40)
            res["graph_capture"] = "OK (single-step, replayed 20x)"
        except Exception as ex:
            res["graph_capture"] = "FAILED: %s: %s" % (type(ex).__name__, str(ex)[:200])
    # fp16 single-call on a FRESH half model (no shared mutation)
    with torch.no_grad():
        w16, _ = build(3); w16 = w16.half()
        st16 = st.half(); ac16 = ac.half()
        try:
            res["predictor_1call_fp16"] = cuda_time(lambda: imagine1(w16, st16, ac16))
        except Exception as ex:
            res["predictor_1call_fp16"] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:200])}
    Path("/root/bench_latency_report.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
