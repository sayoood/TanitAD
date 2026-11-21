import sys, torch
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import REFe, REFeConfig, param_report
for bb in ("vits16","vitb16","vitl16"):
    for nc in (1,4):
        cfg = REFeConfig.for_backbone(bb, n_cameras=nc, cameras=('CAM_F0','CAM_L0','CAM_R0','CAM_B0')[:nc])
        m = REFe(cfg); r = param_report(m)
        print(f"{bb} ncam={nc}: total={r['total']:,} trainable={r['trainable']:,} pct={r['pct']:.2f}%  reg_compress={r['groups'].get('reg_compress',(0,0))[0]:,}")
        del m
