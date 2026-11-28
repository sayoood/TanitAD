"""h2cn_freeze.py -- SHIP-R8 AMENDMENT 4 section 1: fit the H2cn progress stage on the 300 FRESH clips only, freeze it, and STOP.  CPU only.
NEW file (EvalFlyWheel 2026-10-26).  NO eval-window output of the stage is computed here (Amendment 4: registration precedes any eval-side output).

The fit is EXACTLY `l1c_h2cn_confirm.py`'s: `l1c_fit_score.load_fresh` (tree step48852ema_fresh, compliant fresh rows), bank F:/l1c/bank/ztac_fresh.npz
joined on (clip, ws), target arc(G) - arc(HP), `l1_fit_score.h2n_fit(X, Z, H, eid, 0)` once per surface (S2: 4 outputs, S6: 6 outputs).  The fitted networks
are taken out of h2n_fit's returned closure (no refit, no copy of the training code) and written with their standardisation stats and chosen (wd, epochs)
to ONE file, H2CN_STAGE_FROZEN.pt (deterministic content: no timestamp inside), + H2CN_STAGE_FROZEN.sha256 + H2CN_FREEZE_INFO.json.

Controls (any failure: no .pt / .sha256 is left behind, rc 4):
  K-SEAL     both amendments match their seals
  K-DISJ     fresh-clip disjointness vs eval139 + train4572 from independent sources (ship_r8_sub_common.fresh_disjointness) -- and its MUTATION arm
             (one eval139 clip injected into the fresh set) must read RED
  K-FIT      (wd, epochs, n_params, n_fresh) equal H2CN_RESULT.json's fits (the PASSED fit) -- cv_l1_sum is reported beside (exact or not)
  K-RELOAD   the stage re-loaded FROM THE FILE predicts the fresh FIT rows bit-for-bit like h2n_fit's own closure
  K-DETERM   the payload serialised twice (and rebuilt once) is byte-identical (the sha256 is a property of the content; serialised in memory, so the
             zip root is torch's fixed "archive", not a file name)

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 python -X utf8 h2cn_freeze.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ship_r8_sub_common as C  # noqa: E402

import numpy as np  # noqa: E402

H2CN_RESULT = Path("C:/Users/Admin/refcv8_pf/refcv9_l1c/result_h2cn/H2CN_RESULT.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-fresh", default="F:/l1c/bank")
    ap.add_argument("--tree-root", default=str(C.FRESH_TREE_ROOT))
    ap.add_argument("--tree", default=C.FRESH_TREE)
    a = ap.parse_args()
    t0 = time.time()
    seals = C.check_amendment_seals()
    print("[freeze] amendment seals OK", flush=True)
    for p in (C.STAGE_PT, C.STAGE_SHA):
        if p.exists():
            raise SystemExit(f"[freeze] REFUSING: {p} already exists (a frozen stage is written ONCE; move it aside deliberately to re-freeze)")
    C.setup_paths()
    import torch
    import l1c_fit_score as L
    F, AL, RT = L.F, L.AL, L.RT
    import retime_trainfit_r8 as TF
    tools = RT.tools_load()
    TF.TRAIN_ROOT, TF.TRAIN_TREE = a.tree_root, a.tree
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ftr = {}
    trs = {surf: L.load_fresh(tools, CR2, TF, surf, ftr) for surf in ("S2", "S6")}
    bank_path = Path(a.bank_fresh) / "ztac_fresh.npz"
    fresh = F.load_bank(bank_path)
    # ---- K-DISJ (before any fit) + its mutation arm
    disj = C.fresh_disjointness({s: np.unique(trs[s]["sha12"]) for s in trs})
    disj_mut = C.fresh_disjointness(None, inject_eval_clip=True)
    k_disj = bool(disj["PASS"] and not disj_mut["PASS"] and disj_mut["intersections"]["fresh12_x_eval139"] == 1)
    print(f"[freeze] K-DISJ pass={disj['PASS']} mutation_red={not disj_mut['PASS']}", flush=True)
    if not k_disj:
        C.WORK.joinpath("H2CN_FREEZE_WITHHELD.json").write_text(json.dumps({"K-DISJ": disj, "K-DISJ_mutation": disj_mut}, indent=1, default=str), encoding="utf-8")
        raise SystemExit("[freeze] WITHHELD: fresh-clip disjointness failed (or its mutation arm did not go RED)")
    ref = json.loads(H2CN_RESULT.read_text(encoding="utf-8"))
    fits, info_out, k_fit, k_reload = {}, {}, {}, {}
    for surf in ("S2", "S6"):
        tr = trs[surf]
        Ztr, jtr = F.join(fresh, tr["clip"], tr["ws"])
        Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
        t1 = time.time()
        pn, info = F.h2n_fit(tr["X"], Ztr, Htr, tr["eid"], 0)
        net, st = C.net_and_stats_of(pn)
        fits[surf] = {"net": net, "st": st, "info": info, "n_fresh": int(len(Htr)), "n_fresh_episodes": int(len(np.unique(tr["eid"]))),
                      "n_fresh_clips": int(len(np.unique(tr["sha12"])))}
        rf = ref["fits"][surf]
        k_fit[surf] = {"wd": [info["wd"], rf["H2cn"]["wd"]], "epochs": [info["epochs"], rf["H2cn"]["epochs"]], "n_params": [info["n_params"], rf["H2cn"]["n_params"]],
                       "n_fresh": [int(len(Htr)), rf["n_fresh"]], "cv_l1_sum": [info["cv_l1_sum"], rf["H2cn"]["cv_l1_sum"]],
                       "cv_l1_sum_exact": bool(info["cv_l1_sum"] == rf["H2cn"]["cv_l1_sum"]),
                       "cv_l1_sum_absdiff": float(abs(info["cv_l1_sum"] - rf["H2cn"]["cv_l1_sum"]))}
        k_fit[surf]["PASS"] = bool(info["wd"] == rf["H2cn"]["wd"] and info["epochs"] == rf["H2cn"]["epochs"] and info["n_params"] == rf["H2cn"]["n_params"]
                                   and int(len(Htr)) == rf["n_fresh"])
        fits[surf]["_pred_closure_fresh"] = pn(tr["X"], Ztr)
        fits[surf]["_XZ"] = (tr["X"], Ztr)
        info_out[surf] = {**info, "wall_s": round(time.time() - t1, 1), "n_fresh_rows": int(len(Htr)), "n_fresh_episodes": fits[surf]["n_fresh_episodes"],
                          "n_fresh_clips": fits[surf]["n_fresh_clips"], "train_R2_closure": F.r2(fits[surf]["_pred_closure_fresh"], Htr),
                          "bank_v0_vs_panel_maxabs": float(np.nanmax(np.abs(fresh["v0_in"][jtr] - tr["v0"])))}
        print(f"[freeze] {surf} fit wd={info['wd']} epochs={info['epochs']} cv={info['cv_l1_sum']!r} K-FIT={k_fit[surf]['PASS']} ({info_out[surf]['wall_s']} s)", flush=True)
    code = {n: C.sha256_file(p) for n, p in (("l1_fit_score.py", C.L1_CODE / "l1_fit_score.py"), ("l1c_fit_score.py", C.L1C_CODE / "l1c_fit_score.py"),
                                             ("l1c_h2cn_confirm.py", C.L1C_CODE / "l1c_h2cn_confirm.py"),
                                             ("progress_scalar.py", C.L1_CODE / "probe_repro/progress_scalar.py"),
                                             ("ship_r8_sub_common.py", HERE / "ship_r8_sub_common.py"), ("h2cn_freeze.py", Path(__file__).resolve()))}
    prov = {"amendment_4_sha256": seals["4"]["sha256"], "amendment_5_sha256": seals["5"]["sha256"], "code_sha256": code,
            "fresh_bank": {"file": str(bank_path), "sha256": C.sha256_file(bank_path)}, "fresh_tree": f"{a.tree_root}/{a.tree}",
            "fit_reference": {"file": str(H2CN_RESULT), "sha256": C.sha256_file(H2CN_RESULT)}, "fit_seed": 0, "torch": str(torch.__version__),
            "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "torch_get_num_threads": torch.get_num_threads()}}
    payload = C.stage_payload({s: {k: v for k, v in d.items() if not k.startswith("_")} for s, d in fits.items()}, prov)
    import io

    def ser(obj) -> bytes:                       # in-memory: torch.save names the zip root after a FILE's name, a buffer gets the fixed root "archive"
        f = io.BytesIO()
        torch.save(obj, f)
        return f.getvalue()
    b1 = ser(payload)
    k_det = bool(b1 == ser(payload) == ser(C.stage_payload({s: {k: v for k, v in d.items() if not k.startswith("_")} for s, d in fits.items()}, prov)))
    tmp1 = C.WORK / "_freeze_tmp1.pt"
    tmp1.write_bytes(b1)
    # ---- K-RELOAD from the temp file (bit-for-bit vs the closure, on the fresh FIT rows only)
    preds, blob, sha_tmp = C.load_stage(tmp1)
    for surf in ("S2", "S6"):
        X, Z = fits[surf]["_XZ"]
        p_file = preds[surf](X, Z)
        p_cl = fits[surf]["_pred_closure_fresh"]
        k_reload[surf] = {"bit_equal": bool(p_file.dtype == p_cl.dtype and p_file.shape == p_cl.shape and p_file.tobytes() == p_cl.tobytes()),
                          "max_abs": float(np.abs(p_file - p_cl).max()), "rows": int(len(X)), "d_out": int(p_cl.shape[1])}
    ok = bool(k_disj and all(v["PASS"] for v in k_fit.values()) and all(v["bit_equal"] for v in k_reload.values()) and k_det)
    info_json = {"tool": "h2cn_freeze.py", "role": "SHIP-R8 AMENDMENT 4 section 1: the frozen H2cn stage (fit on FRESH rows only); NO eval-window output computed",
                 "amendments": seals, "controls": {"K-SEAL": True, "K-DISJ": k_disj, "K-FIT": k_fit, "K-RELOAD": k_reload, "K-DETERM": k_det, "ALL_PASS": ok},
                 "disjointness": disj, "disjointness_mutation_arm": disj_mut, "fits": info_out, "provenance": prov,
                 "fit_reference_values": {s: ref["fits"][s] for s in ("S2", "S6")},
                 "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    if not ok:
        os.remove(tmp1)
        (C.WORK / "H2CN_FREEZE_WITHHELD.json").write_text(json.dumps(info_json, indent=1, default=str), encoding="utf-8")
        print("[freeze] WITHHELD: a control failed -> H2CN_FREEZE_WITHHELD.json  ZZH2CNFREEZEWITHHELDZZ", flush=True)
        return 4
    os.replace(tmp1, C.STAGE_PT)
    sha = C.sha256_file(C.STAGE_PT)
    if sha != sha_tmp or len(sha) != 64:
        raise SystemExit(f"[freeze] the moved stage file hashes to {sha}, the checked temp file to {sha_tmp}")
    C.STAGE_SHA.write_text(f"{sha}  {C.STAGE_PT.name}\n", encoding="utf-8")
    info_json["stage_file"] = {"path": str(C.STAGE_PT), "sha256": sha, "bytes": C.STAGE_PT.stat().st_size, "written_epoch": C.STAGE_PT.stat().st_mtime}
    (C.WORK / "H2CN_FREEZE_INFO.json").write_text(json.dumps(info_json, indent=1, default=str), encoding="utf-8")
    chk = json.loads((C.WORK / "H2CN_FREEZE_INFO.json").read_text(encoding="utf-8"))           # assert on the ARTIFACTS
    assert chk["controls"]["ALL_PASS"] is True and chk["stage_file"]["sha256"] == C.read_seal(C.STAGE_SHA) == C.sha256_file(C.STAGE_PT)
    print(f"[freeze] WROTE {C.STAGE_PT} sha256 {sha} wall {info_json['wall_s']} s  ZZH2CNFREEZEOKZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
