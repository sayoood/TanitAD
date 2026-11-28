"""IDM-v2 -- the ZERO-TRAINING long_accel candidate.

The error budget (`accel_budget.json`) says the *white* part of the deployed
head's per-frame speed error is only 0.360 m/s on PhysicalAI (the other 3.30 m/s
is a SMOOTH clip-level term that CANCELS in a derivative).  So differentiating
the SLIDING-WINDOW speed track should recover acceleration even though the
head's own `long_accel` output is worthless.

This measures it, on the PERSISTED head, with NO retraining at all:

    z -> A0 (sliding 9-frame window at every frame) -> v_hat[t]
      -> Savitzky-Golay derivative over w frames    -> a_hat[t]

evaluated against BOTH targets:
  * `accel_kin` = centred dv/dt of the pose speed -- the quantity the video
    actually shows, self-consistent with the trajectory
  * `long_accel` = the CAN label the head was trained on (ceiling 0.188 on
    PhysicalAI by §3.3 of the diagnosis, so this column cannot be good)

Writes /root/idm2/out/accel_derive.json
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm2_lib as L                      # noqa: E402
import idm_head as ih                     # noqa: E402

DEV = "cuda"


def taps(width, order=2):
    h = width // 2
    t = np.arange(-h, h + 1, dtype=np.float64)
    P = np.linalg.pinv(np.vander(t, order + 1, increasing=True))
    return P[0], P[1]


def main():
    tr_tags, va_tags = L.split_tags()
    d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
    h = ih.IDMHead(**d["config"]["head_kwargs"]).to(DEV)
    h.load_state_dict(d["state_dict"]); h.eval()

    WS = (9, 13, 17, 25)
    cols = {w: {"p": [], "eid": [], "dom": []} for w in WS}
    gt = {w: {"kin": [], "can": []} for w in WS}
    head_accel = {w: [] for w in WS}
    for tag in va_tags:
        ep = L.load_ep(tag)
        z = ep["z"].float()
        po = ep["poses"].float().numpy().astype(np.float64)
        ac = ep["actions"].float().numpy().astype(np.float64)
        T = z.shape[0]
        idx = np.arange(4, T - 4)
        offs = np.arange(-4, 5)
        Zw = z[torch.tensor(idx[:, None] + offs[None, :])].to(DEV)
        with torch.no_grad():
            P = torch.cat([h(Zw[i:i + 1024])["scalars"].cpu()
                           for i in range(0, Zw.shape[0], 1024)]).numpy().astype(np.float64)
        v = P[:, 0]
        a_head = P[:, 3]
        for w in WS:
            if v.size < w:
                continue
            _, kd = taps(w)
            a = np.convolve(v, kd[::-1], mode="valid") / L.DT
            c = idx[w // 2: v.size - w // 2]           # frame indices of the output
            kin = (po[c + 1, 3] - po[c - 1, 3]) / (2 * L.DT)
            cols[w]["p"].append(a)
            cols[w]["eid"].append(np.array([tag] * a.size))
            cols[w]["dom"].append(np.array([tag.split("_")[0]] * a.size))
            gt[w]["kin"].append(kin)
            gt[w]["can"].append(ac[c, 1])
            head_accel[w].append(a_head[w // 2: v.size - w // 2])

    out = {"n_val_eps": len(va_tags), "windows": {}}
    for w in WS:
        p = np.concatenate(cols[w]["p"])
        eid = np.concatenate(cols[w]["eid"])
        dom = np.concatenate(cols[w]["dom"])
        kin = np.concatenate(gt[w]["kin"])
        can = np.concatenate(gt[w]["can"])
        hd = np.concatenate(head_accel[w])
        rec = {"n": int(p.size), "deriv_window_frames": w,
               "derived_vs_kinematic": L.chan_metrics(p, kin),
               "derived_vs_can_label": L.chan_metrics(p, can),
               "HEAD_accel_vs_kinematic": L.chan_metrics(hd, kin),
               "HEAD_accel_vs_can_label": L.chan_metrics(hd, can),
               "per_domain": {}}
        rec["derived_vs_kinematic"]["boot_r2"] = L.boot_r2(p, kin, eid)
        for dd in ("pai", "cm"):
            m = dom == dd
            rec["per_domain"][dd] = {
                "derived_vs_kinematic": L.chan_metrics(p[m], kin[m]),
                "derived_vs_can_label": L.chan_metrics(p[m], can[m]),
                "HEAD_accel_vs_kinematic": L.chan_metrics(hd[m], kin[m]),
                "HEAD_accel_vs_can_label": L.chan_metrics(hd[m], can[m])}
        # paired: is the DERIVED accel better than the head's own accel output?
        rec["paired_derived_minus_head_vs_kin_mae"] = L.paired_mae(p, hd, kin, eid)
        rec["paired_derived_minus_head_vs_can_mae"] = L.paired_mae(p, hd, can, eid)
        out["windows"][str(w)] = rec
        print(f"w={w:>2} n={p.size}: derived vs KINEMATIC R2 "
              f"{rec['derived_vs_kinematic']['r2']:+.4f} "
              f"[{rec['derived_vs_kinematic']['boot_r2']['lo']:+.3f},"
              f"{rec['derived_vs_kinematic']['boot_r2']['hi']:+.3f}] "
              f"MAE {rec['derived_vs_kinematic']['mae']:.4f} | vs CAN "
              f"{rec['derived_vs_can_label']['r2']:+.4f} | HEAD-out vs kin "
              f"{rec['HEAD_accel_vs_kinematic']['r2']:+.4f} vs CAN "
              f"{rec['HEAD_accel_vs_can_label']['r2']:+.4f}", flush=True)
        for dd in ("pai", "cm"):
            q = rec["per_domain"][dd]
            print(f"      {dd}: derived/kin {q['derived_vs_kinematic']['r2']:+.4f} "
                  f"derived/CAN {q['derived_vs_can_label']['r2']:+.4f} | "
                  f"head/kin {q['HEAD_accel_vs_kinematic']['r2']:+.4f} "
                  f"head/CAN {q['HEAD_accel_vs_can_label']['r2']:+.4f}")
    L.jdump(out, "/root/idm2/out/accel_derive.json")


if __name__ == "__main__":
    main()
