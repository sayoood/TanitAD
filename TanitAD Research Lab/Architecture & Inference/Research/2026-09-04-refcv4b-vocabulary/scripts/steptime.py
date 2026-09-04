"""Where did 1.8 s/step go? Time a full forward+backward with the
v0-conditioned bank ON vs OFF, everything else identical.

MEASURED on the pod: refcv4 (fixed vocabulary) 2.016 / 2.099 s/step; refcv4b
(rolled vocabulary) 3.844 s/step. `roll_bank` alone times at 9.5 ms, so the
1.8 s is somewhere else and must be found, not guessed at.
"""
import sys
import time

import torch

sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
from tanitad.refs import refc_v3                                # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, HW = 2, (128, 320)          # small enough for an 8 GB card; the RATIO is
                               # what transfers, not the absolute number.
print("device", DEV, " batch", B, " image", HW)


def build(v0cond):
    torch.manual_seed(0)
    cfg = refc_v3.refc_v3_sized_config("base", hier=True)
    cfg.core.anchors.n_anchors = 117
    cfg.core.ego_valid_channel = True
    cfg.ego_state_inject = True
    cfg.ego_dropout = 0.5
    if v0cond:
        cfg.core.anchors.v0_conditioned = True
        cfg.core.anchors.control_units = "alat"
    m = refc_v3.RefCV3Model(cfg).to(DEV)
    if v0cond:
        a = torch.linspace(-4, 3, 13)
        a = torch.clamp(a - a[a.abs().argmin()], -4, 3)
        c = torch.linspace(-1, 1, 9) * 3.0
        ctrl = torch.stack(torch.meshgrid(a, c, indexing="ij"), -1).reshape(-1, 2)
        m.core.decoder.load_anchors(m.core.decoder.anchors.clone(),
                                    ctrl.to(DEV))
    return m


def timeit(m, n=12):
    ch = m.cfg.core.encoder.in_channels
    fr = torch.randn(B, 4, ch, *HW, device=DEV)
    nav = torch.zeros(B, dtype=torch.long, device=DEV)
    v0 = torch.rand(B, device=DEV) * 30
    es = torch.rand(B, 5, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-6)
    for i in range(n + 4):
        if i == 4:
            if DEV == "cuda":
                torch.cuda.synchronize()
            t = time.time()
        out = m(fr, nav_cmd=nav, v0=v0, steps=0, ego_state=es)
        loss = out["anchor_traj"].square().mean() + out["traj"].square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / n


for tag, flag in (("FIXED vocabulary  ", False), ("v0-CONDITIONED    ", True)):
    m = build(flag)
    dt = timeit(m)
    print("%s  %.1f ms / step" % (tag, dt * 1000))
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
