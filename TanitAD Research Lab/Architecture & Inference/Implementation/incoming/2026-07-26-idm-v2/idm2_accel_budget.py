"""IDM-v2 -- the ERROR BUDGET that decides whether `long_accel` can EVER be
derived by differentiating a predicted speed, on ANY architecture over this
encoder.

Differentiating a per-frame speed sequence with the degree-2 Savitzky-Golay
derivative over w frames multiplies any NON-SMOOTH (white) component of the
speed error by  k_w = sqrt(sum tap^2)/dt.  For w=9, k = 1.291 s^-1; for w=17,
k = 0.433 s^-1.  So

    sigma_accel  >=  k_w * sigma_speed_white

and to reach R2 > 0 against a target of std sigma_a we need
    sigma_speed_white  <  sigma_a / k_w .

This script measures sigma_speed_white for the deployed head A0: the part of its
per-frame speed error that a degree-2 fit over the same window CANNOT absorb.
It is a property of the SUBSTRATE, not of a head design, so it bounds every
"predict a trajectory and differentiate" proposal.

Writes /root/idm2/out/accel_budget.json
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm2_lib as L                       # noqa: E402
import idm_head as ih                      # noqa: E402
from idm2_diag_labels import savgol_center  # noqa: E402

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
    out = {"kernels": {}, "per_domain": {}}
    for w in (9, 17):
        v, dv = taps(w)
        out["kernels"][f"w{w}"] = {"k_deriv_per_s": float(np.linalg.norm(dv) / L.DT),
                                   "smooth_gain": float(np.linalg.norm(v))}

    for dom in ("pai", "cm"):
        errs, accs = [], []
        for tag in [t for t in va_tags if t.startswith(dom)]:
            ep = L.load_ep(tag)
            z = ep["z"].float()
            po = ep["poses"].float().numpy().astype(np.float64)
            T = z.shape[0]
            # dense per-frame speed prediction from A0 (centre readout, k=4)
            idx = np.arange(4, T - 4)
            offs = np.arange(-4, 5)
            Zw = z[torch.tensor(idx[:, None] + offs[None, :])].to(DEV)
            with torch.no_grad():
                P = torch.cat([h(Zw[i:i + 1024])["scalars"][:, 0].cpu()
                               for i in range(0, Zw.shape[0], 1024)]).numpy()
            e = P.astype(np.float64) - po[idx, 3]
            errs.append(e)
            accs.append((po[idx + 1, 3] - po[idx - 1, 3]) / (2 * L.DT))
        E = np.concatenate(errs)
        A = np.concatenate(accs)
        A = A[np.isfinite(A)]
        rec = {"n_frames": int(E.size), "speed_err_rms": float(np.sqrt((E ** 2).mean())),
               "gt_kinematic_accel_std": float(A.std())}
        for w in (9, 17):
            kv, kd = taps(w)
            sm = np.concatenate([np.convolve(e, kv[::-1], mode="valid")
                                 for e in errs if e.size >= w])
            tg = np.concatenate([e[w // 2: e.size - w // 2] for e in errs if e.size >= w])
            white = tg - sm
            k = float(np.linalg.norm(kd) / L.DT)
            sig_a = float(np.sqrt((white ** 2).mean())) * k
            rec[f"w{w}"] = {
                "sigma_speed_white": float(np.sqrt((white ** 2).mean())),
                "sigma_speed_smooth": float(np.sqrt((sm ** 2).mean())),
                "k_deriv_per_s": k,
                "implied_sigma_derived_accel": sig_a,
                "gt_accel_std": float(A.std()),
                "implied_R2_ceiling_of_derived_accel":
                    float(1.0 - (sig_a / max(A.std(), 1e-9)) ** 2),
                "sigma_speed_white_REQUIRED_for_R2_0.3":
                    float(A.std() * (0.7 ** 0.5) / k)}
        out["per_domain"][dom] = rec
        print(f"\n--- {dom} (n={rec['n_frames']} frames) ---")
        print(f"  A0 per-frame speed error RMS {rec['speed_err_rms']:.3f} m/s; "
              f"GT kinematic accel std {rec['gt_kinematic_accel_std']:.3f} m/s^2")
        for w in (9, 17):
            r = rec[f"w{w}"]
            print(f"  w={w:>2}: white part of speed error "
                  f"{r['sigma_speed_white']:.3f} m/s (smooth part "
                  f"{r['sigma_speed_smooth']:.3f}) x k={r['k_deriv_per_s']:.3f}/s "
                  f"-> derived-accel sigma {r['implied_sigma_derived_accel']:.3f} "
                  f"m/s^2  => R2 ceiling {r['implied_R2_ceiling_of_derived_accel']:+.3f}")
            print(f"        would need white speed error < "
                  f"{r['sigma_speed_white_REQUIRED_for_R2_0.3']:.3f} m/s for R2 0.30 "
                  f"({r['sigma_speed_white']/max(r['sigma_speed_white_REQUIRED_for_R2_0.3'],1e-9):.1f}x "
                  f"better than today)")
    L.jdump(out, "/root/idm2/out/accel_budget.json")


if __name__ == "__main__":
    main()
