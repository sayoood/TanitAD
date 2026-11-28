"""fit_and_write_d: build E-GAIN-D (PREREG_WSP_GAIN_D.md section 2) on the banked TRAIN measurements (wsp_edit/raw/TRAIN_MEASURE_shard*.npz) and write the edited checkpoints for s in {0, 1}.
  Delta-g = g_data - g_model (both on the 20,000 TRAIN fit windows; g_data from raw/GDATA.json, g_model = the median finite-difference gains of the shards), alpha = 1, one dose.
  Delta-m: the edited model's band-mean plan speed over the fit windows = the data's band-mean realised speed (shift = mean realised - mean model plan speed).
No eval-141 window is read here.
modes:
  (default)         write <EDIT_ROOT>/M2b_s{s}_wspedit_d/ckpt_last.pt and raw/FIT_RESULT_D.json
  --dry             fit and report, write nothing
  --repro-banked    the banked targets (base gains, shift 0.35, alpha 1.0) through THIS code, written to <EDIT_ROOT>/_repro_banked/M2b_s{s}_a10/ckpt_last.pt with the banked metadata; the sha256 must equal the
                    banked E-GAIN alpha 1.0 checkpoints' (wsp_edit/raw/FIT_RESULT.json). Writes raw/REPRO_BANKED.json; exit 3 on a mismatch."""
import os
import sys
import json
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch
import wsp_common as W
import wsp_edit as E
import fit_gdata as G


def main():
    mode_dry, mode_repro = "--dry" in sys.argv, "--repro-banked" in sys.argv
    t0 = time.time()
    got, head = W.assert_card_frozen()
    assert head == W.CARD_SHA, f"REFUSED: the D card prefix changed ({head})"
    R, files = G.load_shards()
    ids_sha = W.sha256_file(W.BANK + "/raw/train_window_ids.txt")
    assert ids_sha == "948a2a1d68e91764965baa408fe5aab309ff5dd85955e8be7f740abae108812b"
    kin3, gb = R["kin3"], R["gb"]
    banked = json.load(open(W.BANK + "/raw/FIT_RESULT.json"))
    gd = json.load(open(W.HERE + "/raw/GDATA.json"))
    s_gt = np.load(W.HERE + "/raw/s_gt_train20k.npy")
    g_target = (gd["g_data"]["dsdv0"], gd["g_data"]["dsda0"])
    res = {"card": "PREREG_WSP_GAIN_D.md", "card_prefix_sha256": head, "mode": "repro-banked" if mode_repro else "E-GAIN-D", "train_window_ids_sha256": ids_sha, "n_fit_windows": int(len(kin3)),
           "gdata_json_sha256": W.sha256_file(W.HERE + "/raw/GDATA.json"), "shard_sha256": {os.path.basename(f): W.sha256_file(f) for f in files}, "seeds": {}}
    mism = 0
    for s in (0, 1):
        nm = f"s{s}"
        ck = torch.load(W.CKPT_C0[s], map_location="cpu", weights_only=False)
        sd, cfg = ck["model"], ck["cfg"]
        nb = cfg["nb"]
        r_res = E.readout(sd["basis"].double().numpy(), cfg["res_scale_m"])
        H, gv, ga = R[f"{nm}.h2"], R[f"{nm}.gv"], R[f"{nm}.ga"]
        stot = R[f"{nm}.s_tot"].astype(np.float64)
        ck_sha = W.sha256_file(W.CKPT_C0[s])
        if mode_repro:
            f = E.fit_edit(H, kin3, gv, ga, gb, r_res, 1.0)                                  # the banked call, defaults = the base targets and 0.35
            out_dir = f"{W.EDIT_ROOT}/_repro_banked/M2b_s{s}_a10"
        else:
            shift = float(s_gt.mean() - stot.mean())
            f = E.fit_edit(H, kin3, gv, ga, gb, r_res, 1.0, shift=shift, g_target=g_target)
            out_dir = f"{W.EDIT_ROOT}/M2b_s{s}_wspedit_d"
        new_sd = E.apply_edit(sd, f["dW"], f["db"], nb)
        assert E.lateral_unchanged(sd, new_sd, nb)
        delta_out = (H.astype(np.float64) @ (new_sd[E.KEY_W][:nb].double() - sd[E.KEY_W][:nb].double()).numpy().T + (new_sd[E.KEY_B][:nb].double() - sd[E.KEY_B][:nb].double()).numpy())
        shipped_shift = float((delta_out @ r_res).mean())                                   # from the float32 weights that ship
        edited_mean = float(stot.mean() + shipped_shift)
        sr = {"c0_ckpt": W.CKPT_C0[s], "c0_ckpt_sha256": ck_sha, "norm_r_res": float(np.linalg.norm(r_res)),
              "g_model_train_median": {"dsdv0": f["g_v_model"], "dsda0": f["g_a_model"]}, "g_target": {"dsdv0": f["g_v_target"], "dsda0": f["g_a_target"]},
              "g_base_train_median": {"dsdv0": f["g_v_base"], "dsda0": f["g_a_base"]},
              "delta_g": {"dg_v": f["dg_v"], "dg_a": f["dg_a"]}, "m_per_unit_alpha": f["m"], "shift_target_m_s": f["shift"], "delta_m_m_s": f["shift"] - f["m"], "kappa": f["kappa"],
              "mean_model_plan_speed_train": float(stot.mean()), "mean_realised_speed_train": float(s_gt.mean()), "shipped_mean_shift_m_s": shipped_shift, "shipped_edited_mean_plan_speed": edited_mean,
              "shipped_edited_minus_realised_mean": edited_mean - float(s_gt.mean()), "lam_v": f["lam_v"], "lam_a": f["lam_a"], "decode_r2_v0": f["decode_r2_v"], "decode_r2_a0": f["decode_r2_a"],
              "norm_dW": float(np.linalg.norm(f["dW"])), "norm_db": float(np.linalg.norm(f["db"])), "norm_W_x_rows": float(sd[E.KEY_W][:nb].double().norm()), "lateral_bytes_identical": True}
        if not mode_dry:
            os.makedirs(out_dir, exist_ok=True)
            ck2 = dict(ck)
            ck2["model"] = new_sd
            if mode_repro:
                ck2["wsp_edit"] = {"card": "PREREG_WSP_GAIN.md", "card_sha256": W.BANKED_CARD_SHA, "alpha": 1.0, "seed": s, "base_ckpt": W.CKPT_C0[s], "base_ckpt_sha256": ck_sha,
                                   "train_window_ids_sha256": ids_sha, "dg_v": f["dg_v"], "dg_a": f["dg_a"], "kappa": f["kappa"], "m": f["m"]}
            else:
                ck2["wsp_edit"] = {"card": "PREREG_WSP_GAIN_D.md", "card_sha256": W.CARD_SHA, "alpha": 1.0, "seed": s, "base_ckpt": W.CKPT_C0[s], "base_ckpt_sha256": ck_sha,
                                   "train_window_ids_sha256": ids_sha, "g_data_dsdv0": g_target[0], "g_data_dsda0": g_target[1], "dg_v": f["dg_v"], "dg_a": f["dg_a"], "delta_m": f["shift"] - f["m"],
                                   "shift_target": f["shift"], "m": f["m"]}
            torch.save(ck2, out_dir + "/ckpt_last.pt")
            sr["ckpt"] = out_dir + "/ckpt_last.pt"
            sr["ckpt_sha256"] = W.sha256_file(out_dir + "/ckpt_last.pt")
            if mode_repro:
                want = banked["seeds"][str(s)]["arms"]["10"]["ckpt_sha256"]
                sr["banked_a10_ckpt_sha256"] = want
                sr["byte_identical_to_banked_a10"] = sr["ckpt_sha256"] == want
                # tensor-level comparison as well (a torch.save container difference would show here and not in the tensors)
                bk = torch.load(banked["seeds"][str(s)]["arms"]["10"]["ckpt"], map_location="cpu", weights_only=False)
                sr["state_dict_tensors_identical_to_banked_a10"] = E.state_identical(bk["model"], new_sd)
                mism += 0 if sr["byte_identical_to_banked_a10"] else 1
            else:
                np.savez(W.HERE + f"/raw/EDIT_PARAMS_D_s{s}.npz", u=f["u"], w_v=f["w_v"], w_a=f["w_a"], dW=f["dW"], db=f["db"])
        res["seeds"][str(s)] = sr
        print(f"s{s}: g_model {f['g_v_model']:.4f} {f['g_a_model']:.4f} -> target {f['g_v_target']:.4f} {f['g_a_target']:.4f}  dg_v {f['dg_v']:+.4f} dg_a {f['dg_a']:+.4f}  m {f['m']:+.4f} shift {f['shift']:+.4f} "
              f"delta_m {f['shift'] - f['m']:+.4f}  shipped mean edited plan {edited_mean:.4f} vs realised {s_gt.mean():.4f}  decodeR2 v0 {f['decode_r2_v']:.3f} a0 {f['decode_r2_a']:.3f}", flush=True)
        if mode_repro and not mode_dry:
            print(f"   repro sha {sr['ckpt_sha256'][:16]} banked {sr['banked_a10_ckpt_sha256'][:16]} byte-identical {sr['byte_identical_to_banked_a10']} tensors-identical {sr['state_dict_tensors_identical_to_banked_a10']}", flush=True)
    res["wall_s"] = time.time() - t0
    if not mode_dry:
        json.dump(res, open(W.HERE + ("/raw/REPRO_BANKED.json" if mode_repro else "/raw/FIT_RESULT_D.json"), "w"), indent=1)
    print("done", f"{res['wall_s']:.0f}s", "(dry)" if mode_dry else "")
    return 3 if (mode_repro and not mode_dry and mism) else 0


if __name__ == "__main__":
    sys.exit(main())
