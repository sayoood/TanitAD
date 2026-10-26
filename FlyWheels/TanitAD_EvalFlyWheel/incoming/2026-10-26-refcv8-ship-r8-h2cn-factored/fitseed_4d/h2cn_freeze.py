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

AMENDMENT 4d (EvalFlyWheel 2026-10-26): `--fit-seed S` (default 0) re-trains the stage at fit seed S (`h2n_fit(..., S)`: CV folds AND MLP init/order) and
writes H2CN_FS{S}_STAGE_FROZEN.pt / .sha256 / H2CN_FS{S}_FREEZE_INFO.json (never overwrites).  Extra controls:
  K-REPRO0          (seed 0 only) the re-frozen stage == the REGISTERED stage: container sha256 equal, else every tensor bit-equal AND the predictions on all
                    FRESH fit rows bit-equal (the record says which held).  Failure = WITHHELD, rc 4.
  K-SEED-MOVES      (seed != 0) some parameter tensor differs from the seed-0 stage H2CN_FS0_STAGE_FROZEN.pt, on EVERY surface.
  K-FIT_vs_seed0    (seed != 0, DESCRIPTIVE only, never aborts) wd / epochs / cv_l1_sum against Amendment 4's seed-0 fit.
  K-DATA            (seed != 0, hard) n_fresh and n_params equal Amendment 4's (same data, same architecture); wd / epochs are free to move.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 python -X utf8 h2cn_freeze.py [--fit-seed S]
"""
from __future__ import annotations

import argparse
import hashlib
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
REG_STAGE_PT = C.REGISTERED_DIR / "H2CN_STAGE_FROZEN.pt"                  # READ ONLY (4d): the registered seed-0 stage
REG_STAGE_SHA = C.REGISTERED_DIR / "REGISTERED_H2CN_STAGE.sha256"         # READ ONLY (4d)
REG_STAGE_SHA_EXPECT = "f293b7bdc06aaeec19a10fd8cac9af993c4116401034a9d4ff9c28649e256f32"   # literal, from the 4d brief / Amendment 4 registration


def tensors_equal(blob_a: dict, blob_b: dict) -> dict:
    """bit-for-bit comparison of every stored tensor (state_dict + standardisation stats) and the scalar fit descriptors, per surface."""
    out = {}
    for surf in ("S2", "S6"):
        a, b = blob_a["surfaces"][surf], blob_b["surfaces"][surf]
        diff, n = [], 0
        for grp in ("state_dict", "st"):
            ka, kb = sorted(a[grp].keys()), sorted(b[grp].keys())
            if ka != kb:
                diff.append(f"{grp}:KEYSET")
                continue
            for k in ka:
                n += 1
                ta, tb = a[grp][k].detach().cpu().contiguous(), b[grp][k].detach().cpu().contiguous()
                if not (ta.dtype == tb.dtype and tuple(ta.shape) == tuple(tb.shape) and ta.numpy().tobytes() == tb.numpy().tobytes()):
                    diff.append(f"{grp}.{k}")
        scal = {k: bool(a[k] == b[k]) for k in ("wd", "epochs", "cv_l1_sum", "n_params", "d_in", "d_out", "n_fresh_rows")}
        out[surf] = {"n_tensors": n, "n_differ": len(diff), "differ": diff, "bit_equal": bool(len(diff) == 0), "scalars_equal": scal}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-fresh", default="F:/l1c/bank")
    ap.add_argument("--tree-root", default=str(C.FRESH_TREE_ROOT))
    ap.add_argument("--tree", default=C.FRESH_TREE)
    ap.add_argument("--fit-seed", type=int, default=0, help="AMENDMENT 4d: the `seed` argument of l1_fit_score.h2n_fit (CV folds AND MLP init/order)")
    a = ap.parse_args()
    t0 = time.time()
    seals = C.check_amendment_seals()
    seals["4d"] = C.check_amendment_4d_seal()
    print("[freeze] amendment seals OK (4, 5, 4d)", flush=True)
    SD = C.stage_paths(a.fit_seed)
    for p in (SD["pt"], SD["sha"], SD["info"], SD["withheld"], SD["tmp"]):
        if p.exists():
            raise SystemExit(f"[freeze] REFUSING: {p} already exists (a frozen stage is written ONCE; move it aside deliberately to re-freeze)")
    if a.fit_seed != 0 and not (C.stage_paths(0)["pt"].is_file() and C.stage_paths(0)["sha"].is_file()):
        raise SystemExit("[freeze] REFUSING: the seed-0 stage H2CN_FS0_STAGE_FROZEN.pt must exist first (K-REPRO0 passed) -- K-SEED-MOVES compares against it")
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
        SD["withheld"].write_text(json.dumps({"K-DISJ": disj, "K-DISJ_mutation": disj_mut}, indent=1, default=str), encoding="utf-8")
        raise SystemExit("[freeze] WITHHELD: fresh-clip disjointness failed (or its mutation arm did not go RED)")
    ref = json.loads(H2CN_RESULT.read_text(encoding="utf-8"))
    fits, info_out, k_fit, k_reload, k_data = {}, {}, {}, {}, {}
    for surf in ("S2", "S6"):
        tr = trs[surf]
        Ztr, jtr = F.join(fresh, tr["clip"], tr["ws"])
        Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
        t1 = time.time()
        pn, info = F.h2n_fit(tr["X"], Ztr, Htr, tr["eid"], a.fit_seed)
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
        k_data[surf] = {"n_fresh": [int(len(Htr)), rf["n_fresh"]], "n_params": [info["n_params"], rf["H2cn"]["n_params"]],
                        "PASS": bool(int(len(Htr)) == rf["n_fresh"] and info["n_params"] == rf["H2cn"]["n_params"])}
        fits[surf]["_pred_closure_fresh"] = pn(tr["X"], Ztr)
        fits[surf]["_XZ"] = (tr["X"], Ztr)
        info_out[surf] = {**info, "wall_s": round(time.time() - t1, 1), "n_fresh_rows": int(len(Htr)), "n_fresh_episodes": fits[surf]["n_fresh_episodes"],
                          "n_fresh_clips": fits[surf]["n_fresh_clips"], "train_R2_closure": F.r2(fits[surf]["_pred_closure_fresh"], Htr),
                          "bank_v0_vs_panel_maxabs": float(np.nanmax(np.abs(fresh["v0_in"][jtr] - tr["v0"])))}
        print(f"[freeze] {surf} fit_seed={a.fit_seed} wd={info['wd']} epochs={info['epochs']} cv={info['cv_l1_sum']!r} K-FIT(vs seed0 ref)={k_fit[surf]['PASS']} ({info_out[surf]['wall_s']} s)", flush=True)
    code = {n: C.sha256_file(p) for n, p in (("l1_fit_score.py", C.L1_CODE / "l1_fit_score.py"), ("l1c_fit_score.py", C.L1C_CODE / "l1c_fit_score.py"),
                                             ("l1c_h2cn_confirm.py", C.L1C_CODE / "l1c_h2cn_confirm.py"),
                                             ("progress_scalar.py", C.L1_CODE / "probe_repro/progress_scalar.py"),
                                             ("ship_r8_sub_common.py", HERE / "ship_r8_sub_common.py"), ("h2cn_freeze.py", Path(__file__).resolve()))}
    prov = {"amendment_4_sha256": seals["4"]["sha256"], "amendment_5_sha256": seals["5"]["sha256"], "amendment_4d_sha256": seals["4d"]["sha256"], "code_sha256": code,
            "fresh_bank": {"file": str(bank_path), "sha256": C.sha256_file(bank_path)}, "fresh_tree": f"{a.tree_root}/{a.tree}",
            "fit_reference": {"file": str(H2CN_RESULT), "sha256": C.sha256_file(H2CN_RESULT)}, "fit_seed": int(a.fit_seed), "torch": str(torch.__version__),
            "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "torch_get_num_threads": torch.get_num_threads()}}
    plain = {s: {k: v for k, v in d.items() if not k.startswith("_")} for s, d in fits.items()}
    payload = C.stage_payload(plain, prov, a.fit_seed)
    import io

    def ser(obj) -> bytes:                       # in-memory: torch.save names the zip root after a FILE's name, a buffer gets the fixed root "archive"
        f = io.BytesIO()
        torch.save(obj, f)
        return f.getvalue()
    b1 = ser(payload)
    k_det = bool(b1 == ser(payload) == ser(C.stage_payload(plain, prov, a.fit_seed)))
    tmp1 = SD["tmp"]
    tmp1.write_bytes(b1)
    # ---- K-RELOAD from the temp file (bit-for-bit vs the closure, on the fresh FIT rows only)
    preds, blob, sha_tmp = C.load_stage(tmp1)
    for surf in ("S2", "S6"):
        X, Z = fits[surf]["_XZ"]
        p_file = preds[surf](X, Z)
        p_cl = fits[surf]["_pred_closure_fresh"]
        k_reload[surf] = {"bit_equal": bool(p_file.dtype == p_cl.dtype and p_file.shape == p_cl.shape and p_file.tobytes() == p_cl.tobytes()),
                          "max_abs": float(np.abs(p_file - p_cl).max()), "rows": int(len(X)), "d_out": int(p_cl.shape[1])}
    # ---- AMENDMENT 4d controls
    k_repro0 = k_seed_moves = k_fit_desc = None
    if a.fit_seed == 0:
        reg_sha_file = C.hex64(REG_STAGE_SHA.read_text(encoding="utf-8-sig"))
        reg_sha_actual = C.sha256_file(REG_STAGE_PT)
        reg_ok = bool(len(reg_sha_file) == 64 and len(reg_sha_actual) == 64 and reg_sha_file == REG_STAGE_SHA_EXPECT == reg_sha_actual)   # the registered file IS what was registered
        new_sha = hashlib.sha256(b1).hexdigest()
        preds_reg, blob_reg, _ = C.load_stage(REG_STAGE_PT, registered_sha=REG_STAGE_SHA_EXPECT)
        te = tensors_equal(blob, blob_reg)
        pe = {}
        for surf in ("S2", "S6"):
            X, Z = fits[surf]["_XZ"]
            p_new, p_reg = preds[surf](X, Z), preds_reg[surf](X, Z)
            pe[surf] = {"bit_equal": bool(p_new.dtype == p_reg.dtype and p_new.shape == p_reg.shape and p_new.tobytes() == p_reg.tobytes()),
                        "max_abs": float(np.abs(p_new - p_reg).max()), "rows": int(len(X))}
        prov_new, prov_reg = blob["provenance"], blob_reg["provenance"]
        prov_diff = sorted(k for k in set(prov_new) | set(prov_reg) if k != "code_sha256" and prov_new.get(k) != prov_reg.get(k))
        code_diff = sorted(k for k in set(prov_new["code_sha256"]) | set(prov_reg["code_sha256"]) if prov_new["code_sha256"].get(k) != prov_reg["code_sha256"].get(k))
        sha_equal = bool(len(new_sha) == 64 and new_sha == REG_STAGE_SHA_EXPECT)
        tensors_ok = bool(all(v["bit_equal"] for v in te.values()))
        preds_ok = bool(all(v["bit_equal"] for v in pe.values()))
        k_repro0 = {"registered_sha256_expected": REG_STAGE_SHA_EXPECT, "registered_file_sha256": reg_sha_actual, "registered_seal_file_first_token": reg_sha_file,
                    "registered_file_is_what_was_registered": reg_ok, "refrozen_container_sha256": new_sha, "container_sha_equal": sha_equal,
                    "every_tensor_bit_equal": tensors_ok, "tensors": te, "fresh_row_predictions_bit_equal": preds_ok, "fresh_row_predictions": pe,
                    "provenance_keys_differing_outside_code_sha256": prov_diff, "provenance_code_sha256_entries_differing": code_diff,
                    "held": ("container sha256 equal" if sha_equal else ("tensors bit-equal AND fresh-row predictions bit-equal (container bytes differ)"
                                                                       if (tensors_ok and preds_ok) else "NONE")),
                    "PASS": bool(reg_ok and (sha_equal or (tensors_ok and preds_ok)))}
        print(f"[freeze] K-REPRO0 held={k_repro0['held']!r} container_sha_equal={sha_equal} tensors_bit_equal={tensors_ok} fresh_preds_bit_equal={preds_ok} PASS={k_repro0['PASS']}", flush=True)
    else:
        p0, blob0, sha0 = C.load_stage(C.stage_paths(0)["pt"], registered_sha=C.read_seal(C.stage_paths(0)["sha"]))
        te = tensors_equal(blob, blob0)
        pdesc = {}
        for surf in ("S2", "S6"):
            X, Z = fits[surf]["_XZ"]
            pdesc[surf] = {"max_abs_pred_diff_vs_seed0_on_fresh_fit_rows": float(np.abs(preds[surf](X, Z) - p0[surf](X, Z)).max())}
        k_seed_moves = {"vs": str(C.stage_paths(0)["pt"]), "vs_sha256": sha0, "per_surface": te, "predictions": pdesc,
                        "PASS": bool(all(any(d.startswith("state_dict.") for d in v["differ"]) for v in te.values()))}   # a PARAMETER tensor differs, on every surface
        k_fit_desc = {s: {"wd": [fits[s]["info"]["wd"], ref["fits"][s]["H2cn"]["wd"]], "epochs": [fits[s]["info"]["epochs"], ref["fits"][s]["H2cn"]["epochs"]],
                          "cv_l1_sum": [fits[s]["info"]["cv_l1_sum"], ref["fits"][s]["H2cn"]["cv_l1_sum"]], "equal_to_seed0_choice": k_fit[s]["PASS"]} for s in ("S2", "S6")}
        print(f"[freeze] K-SEED-MOVES PASS={k_seed_moves['PASS']} differing tensors S2={te['S2']['n_differ']}/{te['S2']['n_tensors']} S6={te['S6']['n_differ']}/{te['S6']['n_tensors']}; "
              f"K-FIT_vs_seed0 (descriptive) wd/epochs equal seed 0: S2={k_fit['S2']['PASS']} S6={k_fit['S6']['PASS']}", flush=True)
    if a.fit_seed == 0:
        ok = bool(k_disj and all(v["PASS"] for v in k_fit.values()) and all(v["bit_equal"] for v in k_reload.values()) and k_det and k_repro0["PASS"])
    else:
        ok = bool(k_disj and all(v["PASS"] for v in k_data.values()) and all(v["bit_equal"] for v in k_reload.values()) and k_det and k_seed_moves["PASS"])
    ctl = {"K-SEAL": True, "K-DISJ": k_disj, "K-RELOAD": k_reload, "K-DETERM": k_det}
    if a.fit_seed == 0:
        ctl.update({"K-FIT": k_fit, "K-REPRO0": k_repro0})
    else:
        ctl.update({"K-DATA": k_data, "K-SEED-MOVES": k_seed_moves, "K-FIT_vs_seed0 (descriptive)": k_fit_desc, "K-FIT_vs_seed0_raw (descriptive)": k_fit})
    ctl["ALL_PASS"] = ok
    info_json = {"tool": "h2cn_freeze.py", "fit_seed": int(a.fit_seed),
                 "role": f"SHIP-R8 AMENDMENT 4d: the H2cn stage re-fitted at fit seed {a.fit_seed} (fit on FRESH rows only); NO eval-window output computed",
                 "amendments": seals, "controls": ctl,
                 "disjointness": disj, "disjointness_mutation_arm": disj_mut, "fits": info_out, "provenance": prov,
                 "fit_reference_values": {s: ref["fits"][s] for s in ("S2", "S6")},
                 "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    if not ok:
        os.remove(tmp1)
        SD["withheld"].write_text(json.dumps(info_json, indent=1, default=str), encoding="utf-8")
        print(f"[freeze] WITHHELD: a control failed -> {SD['withheld'].name}  ZZH2CNFREEZEWITHHELDZZ", flush=True)
        return 4
    os.replace(tmp1, SD["pt"])
    sha = C.sha256_file(SD["pt"])
    if sha != sha_tmp or len(sha) != 64:
        raise SystemExit(f"[freeze] the moved stage file hashes to {sha}, the checked temp file to {sha_tmp}")
    SD["sha"].write_text(f"{sha}  {SD['pt'].name}\n", encoding="utf-8")
    info_json["stage_file"] = {"path": str(SD["pt"]), "sha256": sha, "bytes": SD["pt"].stat().st_size, "written_epoch": SD["pt"].stat().st_mtime}
    SD["info"].write_text(json.dumps(info_json, indent=1, default=str), encoding="utf-8")
    chk = json.loads(SD["info"].read_text(encoding="utf-8"))           # assert on the ARTIFACTS
    assert chk["controls"]["ALL_PASS"] is True and chk["stage_file"]["sha256"] == C.read_seal(SD["sha"]) == C.sha256_file(SD["pt"])
    print(f"[freeze] WROTE {SD['pt']} sha256 {sha} wall {info_json['wall_s']} s  ZZH2CNFREEZEOKZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
