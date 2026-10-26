"""l1c_h2cn_confirm.py -- PREREG_L1C_H2CN.md: the nonlinear progress head (H2cn) on the fresh split, read under L1C's G.  CPU only.

    CUDA_VISIBLE_DEVICES=-1 python -X utf8 l1c_h2cn_confirm.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l1c_fit_score as L  # noqa: E402

F, AL, PH, PS, RT = L.F, L.AL, L.PH, L.PS, L.RT
ROOT = HERE.parent
N1 = {"0": -0.0130, "1": -0.0136}          # l1c_next.py N1 (EXPLORATORY), quoted in the prereg as KNOWN
H2C_L1C = {"0": -0.0094, "1": -0.0099}     # L1C_RESULT.json S2 H2c - H0


def worse(c, k):
    lower_better = not (k.startswith("TAC_") or k == "LON_speed_within_0p5")
    return bool(c is not None and ((c.get("lo") is not None and c["lo"] > 0) if lower_better else (c.get("hi") is not None and c["hi"] < 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-fresh", default="F:/l1c/bank")
    ap.add_argument("--tree-root", default="F:/l1c/tree")
    ap.add_argument("--tree", default="step48852ema_fresh")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = F.sha256_file(ROOT / "PREREG_L1C_H2CN.md")
    if pre != (ROOT / "PREREG_L1C_H2CN.sha256").read_text().split()[0]:
        raise SystemExit("[h2cn] PREREG_L1C_H2CN.md does not match its seal: refusing")
    import compliant_reread_r8 as CR
    import retime_trainfit_r8 as TF
    tools = RT.tools_load()
    keys = [m[1] for m in tools["PR"].METRICS]
    surfs = ("S2", "S6")
    first = {}
    ev = {surf: {s: F.load_eval(tools, CR, s, first, surf) for s in L.SEEDS} for surf in surfs}
    TF.TRAIN_ROOT, TF.TRAIN_TREE = a.tree_root, a.tree
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ftr = {}
    trs = {surf: L.load_fresh(tools, CR2, TF, surf, ftr) for surf in surfs}
    banks = {"eval0": F.load_bank(L.L1_BANK / "ztac_eval_s0.npz"), "eval1": F.load_bank(L.L1_BANK / "ztac_eval_s1.npz"),
             "fresh": F.load_bank(Path(a.bank_fresh) / "ztac_fresh.npz")}
    gall = json.loads(F.FROZEN_PF.read_text(encoding="utf-8"))["surfaces"]
    res = {"tool": "l1c_h2cn_confirm.py", "prereg_sha256": pre, "script_sha256": F.sha256_file(__file__), "fits": {}, "seeds": {},
           "tier": "T1 self-action open loop on logged eval139 frames", "estimator": "F.cell paired episode-cluster bootstrap B 2000 seed 0, episodes only, models fixed"}
    for surf in surfs:
        tr = trs[surf]
        Ztr, _ = F.join(banks["fresh"], tr["clip"], tr["ws"])
        Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
        t1 = time.time()
        pn, info = F.h2n_fit(tr["X"], Ztr, Htr, tr["eid"], 0)
        h = L.fit_heads(tr, Ztr)
        res["fits"][surf] = {"H2cn": {**info, "wall_s": round(time.time() - t1, 1)}, "n_fresh": int(len(Htr))}
        gains = np.asarray(gall[surf]["all"]["RETIME"], np.float64)
        for s in L.SEEDS:
            E = ev[surf][s]
            Z, _ = F.join(banks[f"eval{s}"], E["clip"], E["ws"])
            sp = AL.arc(E["HP"])
            arms = {"OS": E["O"], "H0": RT.retime_apply(E["O"], E["HP"], gains),
                    "H2cs": PH.at_progress(E["O"], sp + PS.ridge_pred(h["m_s"], E["X"])),
                    "H2c": PH.at_progress(E["O"], sp + F.head_pred(h["m2"], E["X"], Z)),
                    "H2cn": PH.at_progress(E["O"], sp + pn(E["X"], Z))}
            if surf == "S6":
                np.savez_compressed(out / f"arms_S6_s{s}.npz", clip=E["clip"], ws=E["ws"], eid=E["eid"], v0=E["v0"], G=E["G"],
                                    **{f"arm_{k}": np.asarray(v, np.float32) for k, v in arms.items()})
                continue
            C = {k: F.comp(tools, p, E["G"], E["dt"]) for k, p in arms.items()}
            pc = {"H2cn-H0": F.cell(tools, C["H2cn"]["ade_m"], C["H0"]["ade_m"], E["eid"]),
                  "H2cn-H2c": F.cell(tools, C["H2cn"]["ade_m"], C["H2c"]["ade_m"], E["eid"]),
                  "H2c-H0": F.cell(tools, C["H2c"]["ade_m"], C["H0"]["ade_m"], E["eid"])}
            fam = {k: F.cell(tools, C["H2cn"][k], C["H0"][k], E["eid"]) for k in keys if k in C["H2cn"]}
            res["seeds"][str(s)] = {"paired": pc, "four_families_H2cn-H0": fam, "n_windows": int(len(sp)), "n_episodes": int(len(np.unique(E["eid"])))}
            print(f"[h2cn] S2 seed {s} done", flush=True)
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "l1c_roc.py"), str(out)], capture_output=True, text=True)
    (out / "roc.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    roc = json.loads((out / "L1C_ROC.json").read_text(encoding="utf-8")) if (out / "L1C_ROC.json").exists() else None
    ctl = {"K-ROC": bool(roc and roc.get("controls_all_pass")),
           "K-H2c": all(abs(res["seeds"][s]["paired"]["H2c-H0"]["delta"] - H2C_L1C[s]) <= 1e-4 for s in ("0", "1"))}
    kr = {s: res["seeds"][s]["paired"]["H2cn-H0"]["delta"] - N1[s] for s in ("0", "1")}
    res["K-R_reproduction"] = {"diff_vs_N1": kr, "reproduces": all(abs(v) <= 1e-4 for v in kr.values())}
    P2n = all(res["seeds"][s]["paired"]["H2cn-H0"].get("hi") is not None and res["seeds"][s]["paired"]["H2cn-H0"]["hi"] < 0 for s in ("0", "1"))
    worse_both = [k for k in res["seeds"]["0"]["four_families_H2cn-H0"] if all(worse(res["seeds"][s]["four_families_H2cn-H0"].get(k), k) for s in ("0", "1"))]
    auc_read = roc["readings"].get("L1C_H2cn", {}).get("vs_H0") if roc else None
    Gn = (not worse_both) and auc_read != "AUC-DOWN"
    if not all(ctl.values()):
        word, nxt = f"VOID (control failed: {[k for k, v in ctl.items() if not v]})", "fix the instrument; re-score (0 GPU)."
    elif P2n and Gn:
        word, nxt = "PASS", "H2cn is a refcv8 STAGE-UPGRADE candidate: request the registered SHIP-R8 battery on the new stage from the PI, with these numbers."
    elif worse_both == ["LAT_cross_mae_m"] or (worse_both and set(worse_both) <= {"LAT_cross_mae_m", "LAT_cross_abs_final_m"}):
        word = "FAIL (Gn: " + ", ".join(worse_both) + ")"
        nxt = "the cross-track cost belongs to the progress mechanism, not the head class: read it against the inference-seed floor; carry the head to refcv9 as a trained clean-split stage."
    else:
        word = "FAIL (" + ("P2n " if not P2n else "") + ("Gn: " + ", ".join(worse_both) if worse_both else ("Gn: AUC-DOWN" if auc_read == "AUC-DOWN" else "")) + ")"
        nxt = "a fit that penalises the failing rows (constrained objective), pre-registered first."
    res.update(controls=ctl, bars={"P2n": P2n, "Gn": Gn}, worse_at_both_seeds=worse_both, auc_reading_vs_H0=auc_read,
               auc={s: {a: roc["seeds"][s]["auc"].get(a) for a in ("L1C_H0", "L1C_H2c", "L1C_H2cn", "stage")} for s in ("0", "1")} if roc else None,
               verdict=word, next=nxt, wall_s=round(time.time() - t0, 1))
    (out / "H2CN_RESULT.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    fmt = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and c.get("delta") is not None else "-"  # noqa: E731
    Lm = [word, "", f"# L1C-H2CN (mechanical; PREREG_L1C_H2CN sha256 {pre[:12]})", "", f"tier: {res['tier']}; estimator: {res['estimator']}; one training seed", "",
          "| pair (S2 ADE) | s0 | s1 |", "|---|---|---|"] + [f"| {p} | {fmt(res['seeds']['0']['paired'][p])} | {fmt(res['seeds']['1']['paired'][p])} |" for p in ("H2cn-H0", "H2cn-H2c", "H2c-H0")] + \
         ["", "| four families H2cn-H0 (S2) | s0 | s1 |", "|---|---|---|"] + \
         [f"| {k} | {fmt(res['seeds']['0']['four_families_H2cn-H0'][k])} | {fmt(res['seeds']['1']['four_families_H2cn-H0'].get(k))} |" for k in res["seeds"]["0"]["four_families_H2cn-H0"]] + \
         ["", f"braking AUC (S6 compliant) {json.dumps(res['auc'])}; H2cn vs H0: {auc_read}", f"rows worse at both seeds: {worse_both or 'none'}",
          f"K-R reproduction vs N1: {json.dumps(res['K-R_reproduction'])}", f"controls: {ctl}", "", f"NEXT: {nxt}", "", "ZZH2CNENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(Lm), encoding="utf-8")
    print(word, res["bars"], worse_both, auc_read, ctl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
