"""IDM-v2 DIAGNOSIS (d): is the ceiling HEAD CAPACITY or TRAINING BUDGET?

The "more training" hypothesis, tested three ways on the v1 recipe:
   * vs DATA   -- nested train subsets 8/17/34/68 clips
   * vs STEPS  -- 10/25/50/100/200 epochs at full data
   * vs CAPACITY -- d_model 128/256/512, depth 2/3/6
A flat curve means more of that axis cannot be the fix.

Writes /root/idm2/out/curve.json
"""
from __future__ import annotations
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm2_lib as L                       # noqa: E402
import idm_head as ih                      # noqa: E402

DEV = "cuda"
K = 4


def run(Ztr, Str, Ttr, Zva, Sva, Tva, epochs=50, seed=0, d_model=256, depth=3,
        batch=256, lr=3e-4, wd=0.01):
    torch.manual_seed(seed)
    std = ih.Standardizer.fit(Str)
    head = ih.IDMHead(state_dim=Ztr.shape[-1], d_model=d_model, depth=depth,
                      n_heads=4, window=2 * K + 1).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Ztr.shape[0]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // batch))
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            ld = ih.idm_loss(head(Ztr[ix]), Str[ix], Ttr[ix], std)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            opt.step(); sch.step()
    head.eval()
    with torch.no_grad():
        P = torch.cat([head(Zva[i:i + 1024])["scalars"].cpu()
                       for i in range(0, Zva.shape[0], 1024)]).numpy().astype(np.float64)
        TP = torch.cat([head(Zva[i:i + 1024])["traj"].cpu()
                        for i in range(0, Zva.shape[0], 1024)]).numpy().astype(np.float64)
    G = Sva.cpu().numpy().astype(np.float64)
    out = {nm: L.chan_metrics(P[:, j], G[:, j])
           for j, nm in enumerate(L.SCALARS)}
    out["_ade"] = float(np.linalg.norm(TP - Tva.cpu().numpy(), axis=-1).mean())
    out["_params"] = sum(p.numel() for p in head.parameters())
    out["_train_n"] = int(n)
    del head
    torch.cuda.empty_cache()
    return out


def main():
    tr_tags, va_tags = L.split_tags()
    tr = L.build_set(tr_tags, k=8, stride=1)
    va = L.build_set(va_tags, k=8, stride=2)
    sl = slice(8 - K, 8 + K + 1)
    Ztr = tr["Z"][:, sl].to(DEV).float(); Str = tr["S"].to(DEV); Ttr = tr["Traj"].to(DEV)
    Zva = va["Z"][:, sl].to(DEV).float(); Sva = va["S"].to(DEV); Tva = va["Traj"].to(DEV)
    res = {"n_train_eps": len(tr_tags), "n_val_eps": len(va_tags),
           "n_train_windows": int(Ztr.shape[0]), "n_val_windows": int(Zva.shape[0]),
           "vs_data": {}, "vs_steps": {}, "vs_capacity": {}}

    # ---- vs DATA (nested, domain-stratified) ----------------------------- #
    pai = [t for t in tr_tags if t.startswith("pai")]
    cmm = [t for t in tr_tags if t.startswith("cm")]
    for nclip in (8, 17, 34, 68):
        npai = max(1, round(nclip * len(pai) / len(tr_tags)))
        sub = set(pai[:npai] + cmm[:nclip - npai])
        m = torch.tensor(np.array([e in sub for e in tr["eid"]]), device=DEV)
        r = run(Ztr[m], Str[m], Ttr[m], Zva, Sva, Tva, epochs=50, seed=0)
        res["vs_data"][str(nclip)] = r
        print(f"data {nclip:>3} clips n={r['_train_n']:>6} "
              f"speed {r['speed']['r2']:+.4f} yaw {r['yaw_rate']['r2']:+.4f} "
              f"steer {r['steer']['r2']:+.4f} accel {r['long_accel']['r2']:+.4f} "
              f"ade {r['_ade']:.3f}", flush=True)

    # ---- vs STEPS -------------------------------------------------------- #
    for ep in (10, 25, 50, 100, 200):
        r = run(Ztr, Str, Ttr, Zva, Sva, Tva, epochs=ep, seed=0)
        res["vs_steps"][str(ep)] = r
        print(f"epochs {ep:>4} speed {r['speed']['r2']:+.4f} "
              f"yaw {r['yaw_rate']['r2']:+.4f} steer {r['steer']['r2']:+.4f} "
              f"accel {r['long_accel']['r2']:+.4f} ade {r['_ade']:.3f}", flush=True)

    # ---- vs CAPACITY ----------------------------------------------------- #
    for dm, dp in ((128, 3), (256, 3), (512, 3), (256, 2), (256, 6), (512, 6)):
        r = run(Ztr, Str, Ttr, Zva, Sva, Tva, epochs=50, seed=0,
                d_model=dm, depth=dp)
        res["vs_capacity"][f"d{dm}_L{dp}"] = r
        print(f"d_model {dm:>4} depth {dp} params {r['_params']/1e6:.2f}M "
              f"speed {r['speed']['r2']:+.4f} yaw {r['yaw_rate']['r2']:+.4f} "
              f"steer {r['steer']['r2']:+.4f} accel {r['long_accel']['r2']:+.4f} "
              f"ade {r['_ade']:.3f}", flush=True)

    L.jdump(res, "/root/idm2/out/curve.json")


if __name__ == "__main__":
    main()
