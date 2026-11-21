"""M17 cost measurement — `--agent-queries` 100 vs 32, through the SHIPPED path.

⛔ MEASURED, never assumed: M17 requires the decoder cost at N = 100 to be
reported against N = 32, with step time and peak memory, before the ruling is
acted on.

⚠️ SCOPE, stated because a number quoted outside it is the trap this programme
keeps paying for: this runs on the DEV BOX. The RTX 4060 was busy with another
stream (5.0 GB / 8.2 GB used, 100 % util) throughout, so no GPU load was added
and the timings are CPU. What transfers across devices is the RATIO and the
parameter/activation accounting; the absolute step time does NOT. The pod-side
re-measure is escalated.

Everything is built through `refc_v3_train`'s own helpers so the thing timed is
the thing that ships: `build_parser` -> `_pin_trainer_cfg` -> `RefCV3Model` ->
`compute_losses_v3`, with the agent target block the real dataset emits.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get("TANITAD_SCRIPTS", ""))

import refc_v3_train as t                                    # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402


class _PeakRSS(threading.Thread):
    """Peak process RSS while a step runs.

    ⚠️ On CPU there is no `torch.cuda.max_memory_allocated`, and the caching
    allocator does not exist, so peak RSS is the honest available proxy. It
    measures the WHOLE process, so only the DELTA between the two arms is
    quotable; the absolute is not an activation figure.
    """

    def __init__(self, hz=200.0):
        super().__init__(daemon=True)
        self.stop_flag = False
        self.peak = 0
        self.dt = 1.0 / hz

    def run(self):
        import psutil
        p = psutil.Process()
        while not self.stop_flag:
            try:
                self.peak = max(self.peak, p.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.dt)


def _batch(cfg, b, n_pad, seed=0):
    eps = t._synth_episodes(max(2, b), cfg.core, seed=seed)
    ds = t.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[i] for i in range(b)])
    # the agent target block, exactly the keys `compute_losses_v3` reads
    g = torch.Generator().manual_seed(seed)
    box = torch.stack([
        torch.rand(n_pad, 4, generator=g) *
        torch.tensor([50.0, 24.0, 3.0, 1.5]) +
        torch.tensor([3.0, -12.0, 3.5, 1.6]) for _ in range(b)])
    batch["agent_box"] = box
    batch["agent_yaw"] = torch.zeros(b, n_pad)
    batch["agent_cls"] = torch.zeros(b, n_pad, dtype=torch.long)
    batch["agent_valid"] = torch.ones(b, n_pad, dtype=torch.bool)
    batch["agent_occ"] = torch.zeros(b, n_pad)
    batch["agent_rates"] = torch.zeros(b, n_pad, 3)
    batch["agent_rates_mask"] = torch.zeros(b, n_pad, dtype=torch.bool)
    batch["agent_label"] = torch.ones(b, dtype=torch.bool)
    batch["agent_n_raw"] = torch.full((b,), n_pad, dtype=torch.long)
    batch["agent_n_truncated"] = torch.zeros(b, dtype=torch.long)
    return batch


def run(queries, size, hw, b, n_pad, steps, device, seed=0):
    torch.manual_seed(seed)
    p = t.build_parser()
    args = p.parse_args(["--out", "x", "--arm", "hier", "--size", size,
                         "--image-hw", str(hw[0]), str(hw[1]),
                         "--agents", "head", "--w-agent", "1.0",
                         "--agent-join", "synthetic-for-cost-only",
                         "--agent-queries", str(queries),
                         "--device", device])
    cfg = t._pin_trainer_cfg(v3.refc_v3_sized_config(size, hier=True), args)
    assert cfg.core.agents.queries == queries
    model = v3.RefCV3Model(cfg).to(device)
    model._w_agent = 1.0
    model._w_u0 = 0.0
    model._rig_camera, _ = t._build_rig_camera(cfg, args)
    n_par = sum(q.numel() for q in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = _batch(cfg, b, n_pad, seed=seed)

    def one():
        opt.zero_grad(set_to_none=True)
        out = t.compute_losses_v3(model, batch, device, mode="diffusion")
        out["loss"].backward()
        opt.step()
        return float(out["loss"])

    for _ in range(2):                                       # warm-up
        one()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    mon = _PeakRSS()
    base_rss = __import__("psutil").Process().memory_info().rss
    mon.start()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        one()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    mon.stop_flag = True
    mon.join(timeout=2.0)
    peak_cuda = (torch.cuda.max_memory_allocated() / 2**20
                 if device == "cuda" else None)
    return {"queries": queries, "size": size, "image_hw": list(hw),
            "batch": b, "n_pad": n_pad, "steps": steps, "device": device,
            "params": n_par,
            "step_s_median": statistics.median(ts),
            "step_s_mean": sum(ts) / len(ts),
            "step_s_min": min(ts), "step_s_max": max(ts),
            "peak_rss_mb": mon.peak / 2**20,
            "rss_delta_mb": (mon.peak - base_rss) / 2**20,
            "peak_cuda_mb": peak_cuda}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--hw", nargs=2, type=int, default=[256, 640])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-pad", type=int, default=94)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--queries", nargs="+", type=int, default=[32, 100])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for q in a.queries:
        r = run(q, a.size, tuple(a.hw), a.batch, a.n_pad, a.steps, a.device)
        rows.append(r)
        print(json.dumps(r), flush=True)
    base = next(r for r in rows if r["queries"] == min(a.queries))
    for r in rows:
        r["step_s_ratio_vs_min_q"] = r["step_s_median"] / base["step_s_median"]
        r["params_delta_vs_min_q"] = r["params"] - base["params"]
        r["rss_delta_ratio"] = (r["rss_delta_mb"] / base["rss_delta_mb"]
                                if base["rss_delta_mb"] else None)
    print(json.dumps(rows, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
