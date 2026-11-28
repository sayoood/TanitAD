#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Timing-instrumented AlpaSim driver for the A40 real-time measurement.

Wraps the flagship-v1 OR REF-C policy with per-drive timing and records, per step:
  - model_ms : the policy.plan() inference time (encode_window+strategic+tactical, OR REF-C diffusion)
  - gap_ms   : wall gap between consecutive drive() calls = the FULL per-step loop period
               (render + physics + controller + IPC + model), i.e. what caps the sim's Hz.
Writes a rolling JSON summary to --out. Same gRPC/canon plumbing as the real drivers.
Run: PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts <venv>/python rt_driver.py \
       --port 6789 --kind flagship|refc --ckpt <ckpt> [--preset base] --out /workspace/rt_x.json
"""
from __future__ import annotations
import argparse, json, logging, sys, time, statistics as st
from concurrent import futures
import grpc
from alpasim_grpc.v0.egodriver_pb2_grpc import add_EgodriverServiceServicer_to_server
sys.path.insert(0, "/workspace")
from refc_driver import RefCDriver, RefCPolicy  # noqa: E402
logger = logging.getLogger("rt_driver")


class TimedPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.model_s = []

    def __getattr__(self, name):        # delegate .step/.device/.window/etc. to the wrapped policy
        return getattr(self.inner, name)

    def plan(self, raw_frames, intr, v0, nav_cmd):
        t0 = time.perf_counter()
        r = self.inner.plan(raw_frames, intr, v0, nav_cmd)
        self.model_s.append(time.perf_counter() - t0)
        return r


class TimedDriver(RefCDriver):
    def __init__(self, policy, out):
        super().__init__(policy)
        self._out = out
        self._last = None
        self._gap_s = []

    def drive(self, request, context):
        now = time.perf_counter()
        if self._last is not None:
            self._gap_s.append(now - self._last)
        self._last = now
        resp = super().drive(request, context)
        if len(self._gap_s) and len(self._gap_s) % 5 == 0:
            self._dump()
        return resp

    def _dump(self):
        def stats(xs, drop=1):
            xs = xs[drop:] if len(xs) > drop else xs      # drop 1st (warmup) sample
            if not xs:
                return None
            xs2 = sorted(xs)
            return {"n": len(xs), "median_ms": st.median(xs) * 1e3,
                    "mean_ms": sum(xs) / len(xs) * 1e3,
                    "p10_ms": xs2[int(0.1 * len(xs2))] * 1e3,
                    "p90_ms": xs2[int(0.9 * len(xs2))] * 1e3}
        g = stats(self._gap_s)
        m = stats(self._p.model_s)
        rest = (g["median_ms"] - m["median_ms"]) if (g and m) else None
        d = {"per_step_loop_gap": g, "model_infer": m,
             "render_physics_ipc_ms_median": rest,
             "effective_hz_median": (1000.0 / g["median_ms"]) if g else None}
        json.dump(d, open(self._out, "w"), indent=2)


class ConstPolicy:
    """No-model control: returns a fixed forward plan (~0 compute) so the measured
    loop_gap isolates render + physics + IPC (no model, no GPU contention)."""
    step = 0; device = "cpu"; window = 8

    def plan(self, raw_frames, intr, v0, nav_cmd):
        xy = np.array([[max(v0, 1.0) * 0.5 * i, 0.0] for i in range(1, 5)], dtype=np.float32)
        return xy, np.zeros(4, dtype=np.float32)


def build_policy(kind, ckpt, preset, device):
    if kind == "constant":
        return ConstPolicy()
    if kind == "flagship":
        from flagship_v1_driver import FlagshipV1Policy
        return FlagshipV1Policy(ckpt=ckpt, device=device)
    return RefCPolicy(ckpt=ckpt, preset=preset, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=6789)
    ap.add_argument("--kind", choices=["flagship", "refc", "constant"], default="flagship")
    ap.add_argument("--ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--preset", default="base"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/workspace/rt_timing.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    policy = TimedPolicy(build_policy(a.kind, a.ckpt, a.preset, a.device))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_EgodriverServiceServicer_to_server(TimedDriver(policy, a.out), server)
    server.add_insecure_port(f"{a.host}:{a.port}")
    server.start()
    logger.info("rt_driver serving kind=%s on :%d out=%s", a.kind, a.port, a.out)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
