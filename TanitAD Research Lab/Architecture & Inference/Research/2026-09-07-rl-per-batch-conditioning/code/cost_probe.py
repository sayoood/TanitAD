"""The FINAL per-batch cost: ConditioningContract.check, the exact body that now runs
on every batch, against the RefCV3Model.forward it precedes. CPU-ONLY, zero GPU."""
from __future__ import annotations

import statistics
import time

import torch

from tanitad.rl import refc_adapter as A
from tanitad.refs import refc_v3 as v3


def _cfg_fleet_style():
    """The predicates that read TRUE on today's fleet (refcv4b/refcv5 argv:
    `--anchor-v0-conditioned`, `--ego-state-inject`)."""
    cfg = v3.RefCV3Config()
    cfg.ego_state_inject = True
    cfg.core.anchors.v0_conditioned = True
    cfg.core.sel_reach_clamp = True
    cfg.core.graft_lan = True
    cfg.core.nav_known_channel = True
    return cfg


class _M:
    def __init__(self, cfg):
        self.cfg = cfg


def _bench(fn, *, n, warmup=200, reps=7):
    for _ in range(warmup):
        fn()
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        per.append((time.perf_counter() - t0) / n)
    return min(per), statistics.median(per)


def main():
    m = _M(_cfg_fleet_style())
    batch = {"frames": torch.zeros(1), "v0": torch.zeros(1),
             "ego_state": torch.zeros(1, 5), "lan": torch.zeros(1, 4),
             "nav_known": torch.zeros(1), "gt_traj": torch.zeros(1, 8, 2),
             "obstacles": torch.zeros(1, 4, 2), "lead_path": torch.zeros(1, 1, 8, 2)}

    contract = A.ConditioningContract(m)
    contract.check(batch)                       # resolve once, as a rollout would
    chk_min, chk_med = _bench(lambda: contract.check(batch), n=20000)

    req_min, req_med = _bench(lambda: A.conditioning_requirements(m), n=2000)
    unc_min, unc_med = _bench(lambda: A.assert_conditioning(m, batch), n=2000)

    print(f"required channels on this build: {contract.required}")
    print(f"resolutions={contract.resolutions}  batches={contract.batches}")
    print(f"CHECK   ConditioningContract.check  min={chk_min*1e6:8.3f} us "
          f"med={chk_med*1e6:8.3f} us")
    print(f"RESOLVE conditioning_requirements   min={req_min*1e6:8.3f} us "
          f"med={req_med*1e6:8.3f} us")
    print(f"UNCACHED assert_conditioning        min={unc_min*1e6:8.3f} us "
          f"med={unc_med*1e6:8.3f} us")
    print(f"cache speedup on the per-batch path: {unc_min/chk_min:.2f}x")

    torch.manual_seed(0)
    for label, cfg in (("smoke  ", v3.refc_v3_smoke_config(hier=True)),
                       ("default", _cfg_fleet_style())):
        model = v3.RefCV3Model(cfg).eval()
        enc, W = cfg.core.encoder, cfg.core.window
        f = torch.zeros(1, W, enc.in_channels, enc.image_size, enc.image_size)
        with torch.no_grad():
            for _ in range(2):
                model(f)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                model(f)
                ts.append(time.perf_counter() - t0)
        c = min(ts)
        print(f"FORWARD {label} B=1 W={W} px={enc.image_size} ch={enc.in_channels}"
              f"  min={c*1e3:9.3f} ms med={statistics.median(ts)*1e3:9.3f} ms"
              f"  || check/forward = {chk_min/c*100:.5f} %"
              f"  resolve/forward = {req_min/c*100:.5f} %"
              f"  uncached/forward = {unc_min/c*100:.5f} %"
              f"  1 check = 1/{c/chk_min:,.0f} of a forward")


if __name__ == "__main__":
    main()
