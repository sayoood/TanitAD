"""a2_freeze.py -- PREREG_REFCV9_INMODEL_PA PHASE A2 freeze: the 9 fitted heads + R9PF-TL (R9PF's head, decision trunk_lon) -> 12 frozen head files,
their controls, R9PF_FIT_INFO.json and FROZEN_R9PF_PA.sha256 (the list the Master Mind registers).  CPU only.  NO eval139 window is read.

Controls (any failure: rc 4, R9PF_FIT_WITHHELD.json, nothing registered):
  K-FILES    all 9 fitted heads + their _FIT.json exist, every head's sha256 == its _FIT.json record (both 64 hex)
  K-TL       R9PF-TL_fs{k} carries R9PF_fs{k}'s state dict BIT FOR BIT, decision 'trunk_lon'
  K-RELOAD   each head re-loaded FROM ITS FILE (float32 R9ProgressHead, eval) reproduces the float64 fit on every calibration row: decision agreement
             (argmax) reported, max |logit| / |profile| difference reported; PASS iff decisions agree on >= 99.9 % of rows and max |profile| diff <= 1e-3 m
  K-BNSTATS  the head's BatchNorm statistics == the calibration rows' population mean / var (float32 cast) -- CONSTZ: z variance exactly 0
  K-LOSS     each fit's objective == refcv9_progress.r9_losses on a fixed batch (recorded by a2_fit.py)
  K-SHUF-PERM no episode keeps its own labels under the SHUF permutation
  K-DETERM   each payload re-serialised in memory twice is byte-identical
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

import numpy as np  # noqa: E402


def main() -> int:
    t0 = time.time()
    prereg = RC.check_prereg()
    RC.setup_paths()
    import torch
    R = RC.r9p()
    cache = RC.SCRATCH / "calib_rows.npz"
    d = np.load(cache, allow_pickle=False)
    n = len(d["v0"])
    ctl, fits, heads = {}, {}, {}
    # ---- K-FILES
    miss = []
    for arm in ("R9PF", "R9PF-CONSTZ", "R9PF-SHUF"):
        for k in RC.FIT_SEEDS:
            tag = f"{arm}_fs{k}"
            hp, fj = RC.HEADS / f"{tag}.pt", RC.SCRATCH / "fits" / f"{tag}_FIT.json"
            if not (hp.is_file() and fj.is_file()):
                miss.append(tag)
                continue
            fits[tag] = json.loads(fj.read_text(encoding="utf-8"))
            sh = RC.sha256_file(hp)
            if len(sh) != 64 or sh != fits[tag]["head_sha256"]:
                miss.append(f"{tag}:sha")
            heads[tag] = hp
    ctl["K-FILES"] = {"PASS": not miss, "missing_or_mismatch": miss}
    if miss:
        RC.jdump({"controls": ctl}, RC.WORK / "R9PF_FIT_WITHHELD.json")
        raise SystemExit(f"[freeze] WITHHELD: {miss}")
    # ---- R9PF-TL heads (R9PF's state dict, decision trunk_lon)
    ktl = {}
    for k in RC.FIT_SEEDS:
        src = torch.load(heads[f"R9PF_fs{k}"], map_location="cpu", weights_only=True)
        tag = f"R9PF-TL_fs{k}"
        hp = RC.HEADS / f"{tag}.pt"
        fit_tl = {**src["fit"], "arm": "R9PF-TL", "shared_with": f"R9PF_fs{k}",
                  "note": "R9PF-TL = R9PF's fitted head (same profiles, prereg s1) with the decision read from the trunk lon head (argmax over LON6 grouped to regimes)"}
        prov = {**src["provenance"], "a2_freeze_sha256": RC.sha256_file(__file__), "source_head": {"file": str(heads[f"R9PF_fs{k}"]), "sha256": RC.sha256_file(heads[f"R9PF_fs{k}"])}}
        payload = RC.head_payload("R9PF-TL", k, "trunk_lon", src["state_dict"], fit_tl, None, prov)
        b1 = RC.ser(payload)
        if b1 != RC.ser(payload):
            raise SystemExit("[freeze] non-deterministic TL serialisation")
        if not hp.exists():
            tmp = RC.HEADS / f".{tag}.tmp"
            tmp.write_bytes(b1)
            tmp.replace(hp)
        else:
            if RC.sha256_file(hp) != RC.sha256_bytes(b1):
                raise SystemExit(f"[freeze] {hp} exists with different content -- refusing")
        chk = torch.load(hp, map_location="cpu", weights_only=True)
        ktl[tag] = bool(chk["decision"] == "trunk_lon" and chk["r9_config"]["decision"] == "trunk_lon"
                        and all(chk["state_dict"][kk].dtype == src["state_dict"][kk].dtype and torch.equal(chk["state_dict"][kk], src["state_dict"][kk])
                                for kk in src["state_dict"]) and set(chk["state_dict"]) == set(src["state_dict"]))
        heads[tag] = hp
        fits[tag] = fit_tl
    ctl["K-TL"] = {"PASS": all(ktl.values()), "per_head": ktl}
    # ---- K-RELOAD / K-BNSTATS / K-DETERM per head
    rel, bns, det = {}, {}, {}
    zc = d["z"].astype(np.float32)
    x9 = d["x9"].astype(np.float32)
    for tag, hp in sorted(heads.items()):
        head, blob, _sh = RC.load_head(hp)
        src_tag = tag.replace("R9PF-TL", "R9PF")
        ref = np.load(RC.SCRATCH / "fits" / f"{src_tag}_fitrows_f64.npz")
        z = zc if blob.get("constz_mean") is None else np.broadcast_to(blob["constz_mean"].numpy(), zc.shape).copy()
        with torch.no_grad():
            o = head(torch.from_numpy(z), torch.from_numpy(x9))
        lg, pf = o["regime_logits"].numpy().astype(np.float64), o["profiles"].numpy().astype(np.float64)
        agree = float((lg.argmax(1) == ref["logits"].argmax(1)).mean())
        dp = float(np.abs(pf - ref["profiles"]).max())
        rel[tag] = {"decision_agreement": agree, "n_rows": n, "n_disagree": int((lg.argmax(1) != ref["logits"].argmax(1)).sum()),
                    "max_abs_logit_diff": float(np.abs(lg - ref["logits"]).max()), "max_abs_profile_diff_m": dp,
                    "PASS": bool(agree >= 0.999 and dp <= 1e-3)}
        rm, rv = blob["state_dict"]["norm.running_mean"].numpy(), blob["state_dict"]["norm.running_var"].numpy()
        b = {"mean_eq_fit_f32": bool(np.array_equal(rm, ref["mean"].astype(np.float32))), "var_eq_fit_f32": bool(np.array_equal(rv, ref["var"].astype(np.float32)))}
        IN = np.concatenate([x9.astype(np.float64), z.astype(np.float64)], 1)
        b["mean_vs_calib_rows_maxabs"] = float(np.abs(rm.astype(np.float64) - IN.mean(0)).max())
        if blob.get("constz_mean") is not None:
            b["constz_z_var_all_zero"] = bool((rv[9:] == 0).all())
            b["constz_mean_eq_calib_mean_f32"] = bool(np.array_equal(blob["constz_mean"].numpy(), d["z"].astype(np.float64).mean(0).astype(np.float32)))
        b["PASS"] = bool(b["mean_eq_fit_f32"] and b["var_eq_fit_f32"] and b.get("constz_z_var_all_zero", True) and b.get("constz_mean_eq_calib_mean_f32", True))
        bns[tag] = b
        det[tag] = bool(RC.ser(blob) == RC.ser(blob))
    ctl["K-RELOAD"] = {"PASS": all(v["PASS"] for v in rel.values()), "per_head": rel}
    ctl["K-BNSTATS"] = {"PASS": all(v["PASS"] for v in bns.values()), "per_head": bns}
    ctl["K-DETERM"] = {"PASS": all(det.values()), "per_head": det}
    ctl["K-LOSS"] = {"PASS": all(f["K-LOSS"]["PASS"] for t, f in fits.items() if not t.startswith("R9PF-TL")),
                     "per_fit": {t: f["K-LOSS"] for t, f in fits.items() if not t.startswith("R9PF-TL")}}
    ctl["K-SHUF-PERM"] = {"PASS": all(fits[f"R9PF-SHUF_fs{k}"]["label_permutation"]["episodes_keeping_own_labels"] == 0 for k in RC.FIT_SEEDS),
                          "per_fit": {k: fits[f"R9PF-SHUF_fs{k}"]["label_permutation"] for k in RC.FIT_SEEDS}}
    ctl["ALL_PASS"] = bool(all(v["PASS"] for v in ctl.values() if isinstance(v, dict)))
    a1 = json.loads((RC.RESULT / "R9PF_A1_PREREQ.json").read_text(encoding="utf-8"))
    frozen = {t: {"file": str(hp), "sha256": RC.sha256_file(hp)} for t, hp in sorted(heads.items())}
    banks = {"eval_ztac_s0": {"file": a1["eval_ztac_bank"]["s0"]["file"], "sha256": RC.sha256_file(a1["eval_ztac_bank"]["s0"]["file"])},
             "eval_ztac_s1": {"file": a1["eval_ztac_bank"]["s1"]["file"], "sha256": RC.sha256_file(a1["eval_ztac_bank"]["s1"]["file"])},
             "fresh_ztac": {"file": str(RC.FRESH_BANK), "sha256": RC.sha256_file(RC.FRESH_BANK)},
             "calib_rows": {"file": str(cache), "sha256": RC.sha256_file(cache)}}
    for k, v in banks.items():
        if k.startswith("eval") and v["sha256"] != a1["eval_ztac_bank"][k.split("_")[-1]]["sha256"]:
            ctl["ALL_PASS"] = False
            ctl["K-BANK-STABLE"] = {"PASS": False, "bank": k}
    summary = {t: {k: f.get(k) for k in ("wd", "epochs", "cv_obj", "cv_l1_mean_at_best", "cv_ce_mean_at_best", "train_regime_acc_vs_true_target",
                                          "train_regime_acc_vs_fit_target", "train_decision_counts", "n_params", "wall_s", "threads")}
               for t, f in sorted(fits.items())}
    info = {"tool": "a2_freeze.py", "role": "PREREG_REFCV9_INMODEL_PA PHASE A2: the frozen stage-2 heads (fit on FRESH calibration rows only); NO eval139 window read",
            "prereg": prereg, "controls": ctl, "frozen_heads": frozen, "banks": banks, "fits_summary": summary, "fits_full": fits,
            "calibration_rows": {"n": int(n), "n_episodes": int(len(np.unique(d["eid"]))), "class_counts": np.bincount(d["r_star"], minlength=3).tolist(),
                                 "definition": "fresh S6 compliant rows (H2cn's loader l1c_fit_score.load_fresh path; S6 subset of S2, all 8 slots valid, regime defined)"},
            "a1": {"file": str(RC.RESULT / "R9PF_A1_PREREQ.json"), "sha256": RC.sha256_file(RC.RESULT / "R9PF_A1_PREREQ.json"), "ALL_PASS": a1["ALL_PASS"]},
            "code_sha256": {p.name: RC.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
            "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    if not ctl["ALL_PASS"]:
        RC.jdump(info, RC.WORK / "R9PF_FIT_WITHHELD.json")
        print("[freeze] WITHHELD  ZZA2FREEZEWITHHELDZZ", flush=True)
        return 4
    RC.jdump(info, RC.WORK / "R9PF_FIT_INFO.json")
    lines = [f"{v['sha256']}  {Path(v['file']).as_posix()}" for v in frozen.values()]
    lines += [f"{v['sha256']}  {Path(v['file']).as_posix()}" for v in banks.values()]
    lines.append(f"{RC.sha256_file(RC.WORK / 'R9PF_FIT_INFO.json')}  {(RC.WORK / 'R9PF_FIT_INFO.json').as_posix()}")
    (RC.WORK / "FROZEN_R9PF_PA.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[freeze] OK {len(frozen)} heads  ZZA2FREEZEOKZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
