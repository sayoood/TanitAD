"""fit_and_write: fit E-GAIN on the TRAIN measurements (raw/TRAIN_MEASURE_shard*.npz) and write the edited checkpoints for s in {0, 1}, alpha in {0.5, 1.0}.
No eval-141 window is read here. Output: <EDIT_ROOT>/M2b_s{s}_wspedit_a{05|10}/ckpt_last.pt, raw/FIT_RESULT.json, raw/EDIT_PARAMS_s{s}.npz.
usage: fit_and_write.py [--dry]    (--dry: fit and report, write nothing)"""
import os
import sys
import glob
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch
import wsp_common as W
import wsp_edit as E

ALPHAS = {"05": 0.5, "10": 1.0}


def load_train_measure():
    files = sorted(glob.glob(W.HERE + "/raw/TRAIN_MEASURE_shard[0-9].npz"))
    assert len(files) >= 1, "no TRAIN_MEASURE shards"
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    idx = np.concatenate([p["flat_idx"] for p in parts])
    order = np.argsort(idx, kind="stable")
    assert np.array_equal(idx[order], np.arange(20000)), "shards do not cover the 20,000 windows exactly once"
    out = {}
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts], 0)[order]
    return out, files


def main():
    dry = "--dry" in sys.argv
    t0 = time.time()
    R, files = load_train_measure()
    ids_sha = W.sha256_file(W.HERE + "/raw/train_window_ids.txt")
    kin3, gb = R["kin3"], R["gb"]
    assert 3.0 <= kin3[:, 0].min() and kin3[:, 0].max() < 8.0
    res = {"train_window_ids_sha256": ids_sha, "n_fit_windows": int(len(kin3)), "shards": [os.path.basename(f) for f in files],
           "shard_sha256": {os.path.basename(f): W.sha256_file(f) for f in files}, "seeds": {}}
    for s in (0, 1):
        nm = f"s{s}"
        ck = torch.load(W.CKPT_C0[s], map_location="cpu", weights_only=False)
        sd = ck["model"]
        cfg = ck["cfg"]
        nb = cfg["nb"]
        r_res = E.readout(sd["basis"].double().numpy(), cfg["res_scale_m"])
        H, gv, ga = R[f"{nm}.h2"], R[f"{nm}.gv"], R[f"{nm}.ga"]
        ck_sha = W.sha256_file(W.CKPT_C0[s])
        seed_res = {"c0_ckpt": W.CKPT_C0[s], "c0_ckpt_sha256": ck_sha, "norm_r_res": float(np.linalg.norm(r_res)),
                    "train_median_gains": {"model_dsdv0": float(np.median(gv)), "model_dsda0": float(np.median(ga)), "base_dsdv0": float(np.median(gb[:, 0])), "base_dsda0": float(np.median(gb[:, 1]))},
                    "train_mean_plan_speed": float(R[f"{nm}.s_tot"].mean()), "train_mean_v0": float(kin3[:, 0].mean()),
                    "train_mean_s_res": float(R[f"{nm}.s_res"].mean()), "train_mean_s_net": float(R[f"{nm}.s_net"].mean()), "train_mean_s_base": float(R[f"{nm}.s_base"].mean()), "arms": {}}
        for tag, alpha in ALPHAS.items():
            f = E.fit_edit(H, kin3, gv, ga, gb, r_res, alpha)
            new_sd = E.apply_edit(sd, f["dW"], f["db"], nb)
            assert E.lateral_unchanged(sd, new_sd, nb)
            arm = {"alpha": alpha, "dg_v": f["dg_v"], "dg_a": f["dg_a"], "m_per_unit_alpha": f["m"], "kappa": f["kappa"], "lam_v": f["lam_v"], "lam_a": f["lam_a"],
                   "decode_r2_v0": f["decode_r2_v"], "decode_r2_a0": f["decode_r2_a"], "norm_dW": float(np.linalg.norm(f["dW"])), "norm_db": float(np.linalg.norm(f["db"])),
                   "norm_W_x_rows": float(sd[E.KEY_W][:nb].double().norm()), "lateral_bytes_identical": True}
            if not dry:
                out_dir = f"{W.EDIT_ROOT}/M2b_s{s}_wspedit_a{tag}"
                os.makedirs(out_dir, exist_ok=True)
                ck2 = dict(ck)
                ck2["model"] = new_sd
                ck2["wsp_edit"] = {"card": "PREREG_WSP_GAIN.md", "card_sha256": W.CARD_SHA, "alpha": alpha, "seed": s, "base_ckpt": W.CKPT_C0[s], "base_ckpt_sha256": ck_sha,
                                   "train_window_ids_sha256": ids_sha, "dg_v": f["dg_v"], "dg_a": f["dg_a"], "kappa": f["kappa"], "m": f["m"]}
                torch.save(ck2, out_dir + "/ckpt_last.pt")
                arm["ckpt"] = out_dir + "/ckpt_last.pt"
                arm["ckpt_sha256"] = W.sha256_file(out_dir + "/ckpt_last.pt")
                if tag == "10":
                    np.savez(W.HERE + f"/raw/EDIT_PARAMS_s{s}.npz", u=f["u"], w_v=f["w_v"], w_a=f["w_a"], dW_a10=f["dW"], db_a10=f["db"])
            seed_res["arms"][tag] = arm
        res["seeds"][str(s)] = seed_res
        print(f"s{s}: train median gains model dS/dv0 {seed_res['train_median_gains']['model_dsdv0']:.4f} dS/da0 {seed_res['train_median_gains']['model_dsda0']:.4f} "
              f"base {seed_res['train_median_gains']['base_dsdv0']:.4f} {seed_res['train_median_gains']['base_dsda0']:.4f}", flush=True)
        for tag in ALPHAS:
            a = seed_res["arms"][tag]
            print(f"   a{tag}: dg_v {a['dg_v']:+.4f} dg_a {a['dg_a']:+.4f} m {a['m_per_unit_alpha']:+.4f} kappa {a['kappa']:+.3f} decodeR2 v0 {a['decode_r2_v0']:.4f} a0 {a['decode_r2_a0']:.4f}", flush=True)
    res["wall_s"] = time.time() - t0
    if not dry:
        json.dump(res, open(W.HERE + "/raw/FIT_RESULT.json", "w"), indent=1)
    print("done", f"{res['wall_s']:.0f}s", "(dry)" if dry else "")


if __name__ == "__main__":
    main()
