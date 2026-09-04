"""3-arm step-time panel separating the CODE change from the FLAG.

ARM A  = pre-11:40 forward path + FIXED vocabulary   (the ABORTED run's graph)
ARM C  = post-11:40 forward path + FIXED vocabulary  (CONTROL: the patch claims
         this is bit-identical to A, so C/A MUST read 1.00)
ARM B  = post-11:40 forward path + v0-CONDITIONED    (the LIVE run's graph)
"""
import os, sys, time, json
import torch

STACK = sys.argv[1]; V0COND = sys.argv[2] == "1"; NANC = int(sys.argv[3]); TAG = sys.argv[4]
sys.path.insert(0, STACK)
from tanitad.refs import refc_v3                                   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, HW, FR = 2, (128, 320), 4

torch.manual_seed(0)
cfg = refc_v3.refc_v3_sized_config("base", hier=True)
cfg.core.anchors.n_anchors = NANC
cfg.core.ego_valid_channel = True
cfg.ego_state_inject = True
cfg.ego_dropout = 0.5
cfg.core.sel_reach_clamp = True
cfg.core.sel_accel_max = 2.0
if V0COND:
    cfg.core.anchors.v0_conditioned = True
    try:    cfg.core.anchors.control_units = "alat"
    except Exception: pass
m = refc_v3.RefCV3Model(cfg).to(DEV)
if V0COND:
    a = torch.linspace(-4, 3, 13); a = torch.clamp(a - a[a.abs().argmin()], -4, 3)
    c = torch.linspace(-1, 1, 9) * 3.0
    ctrl = torch.stack(torch.meshgrid(a, c, indexing="ij"), -1).reshape(-1, 2)[:NANC]
    m.core.decoder.load_anchors(m.core.decoder.anchors.clone(), ctrl.to(DEV))

ch = m.cfg.core.encoder.in_channels
fr  = torch.randn(B, FR, ch, *HW, device=DEV)
nav = torch.zeros(B, dtype=torch.long, device=DEV)
v0  = torch.rand(B, device=DEV) * 30
es  = torch.rand(B, 5, device=DEV)
opt = torch.optim.Adam(m.parameters(), lr=1e-6)

N = 20
for i in range(N + 5):
    if i == 5:
        if DEV == "cuda": torch.cuda.synchronize()
        t = time.time()
    out = m(fr, nav_cmd=nav, v0=v0, steps=0, ego_state=es)
    loss = out["anchor_traj"].square().mean() + out["traj"].square().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
if DEV == "cuda": torch.cuda.synchronize()
dt = (time.time() - t) / N
print(json.dumps({"arm": TAG, "stack": STACK, "v0_conditioned": V0COND,
                  "n_anchors": NANC, "device": DEV, "batch": B, "hw": list(HW),
                  "frames": FR, "iters": N, "ms_per_step": round(dt*1000, 2),
                  "has_roll_bank_in_forward":
                      "bank = self.roll_bank" in open(
                          os.path.join(STACK, "tanitad", "refs", "refc.py"),
                          encoding="utf-8").read()}))
