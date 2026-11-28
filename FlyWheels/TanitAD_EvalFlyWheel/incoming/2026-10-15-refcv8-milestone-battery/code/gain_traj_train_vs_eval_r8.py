"""gain_traj_train_vs_eval_r8.py -- POST-HOC ADDENDUM to PREREG_GAINTRAJ (NOT registered; exploratory; no role in the label).

Why it exists: the registered train read (PREREG section 6) found the 30k-EMA flat g* on the 139 TRAIN clips at 0.68 (compliant) but 0.50 (all windows) -- the compliance mask moves it by ~0.19 on train.
The registered eval number is compliant-only, so 'train vs eval agree?' needs the like-for-like pair on BOTH window sets.  This script computes eval-ALL g* (EMA-30k, both seeds, S2/S6) with the
same estimator/bootstrap as `gain_traj_r8` and puts train and eval side by side for both window sets, with an UNPAIRED (two different clip sets) bootstrap for the difference.
Control: the eval compliant subset recomputed here must equal the registered GAINTRAJ.json cell to 1e-12 (else it stops).

    CUDA_VISIBLE_DEVICES=-1 ... python -X utf8 gain_traj_train_vs_eval_r8.py     -> raw/gain_traj/GAINTRAJ_TRAIN_vs_EVAL.{json,md}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gain_traj_r8 as GT  # noqa: E402
import a0k0_headroom_retime_r8 as RT  # noqa: E402

OUT = GT.OUT


def arrays(S, sel):
    X = S["X"]
    return GT.arc(X["HP"][sel]), GT.arc(X["O"][sel]), GT.arc(X["G"][sel]), np.asarray(S["eid"])[sel]


def main() -> int:
    t0 = time.time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("set CUDA_VISIBLE_DEVICES=-1")
    GT.check_prereg()
    import compliant_reread_r8 as CR
    tools = RT.tools_load()
    main_json = json.loads((OUT / "GAINTRAJ.json").read_text(encoding="utf-8"))
    ev = {}                                    # (surf, seed, set) -> (A, B)
    first = {}
    for seed in GT.SEEDS:
        ctx = CR.prepare_seed(GT.CTRL_TREE, seed, tools, tools["rc"])
        for surf in GT.SURFS:
            S = RT.surface_inputs(tools, ctx, surf, first, GT.CTRL_TREE)
            m = np.asarray(S["mask"], bool)
            for name, sel in (("compliant", m), ("all", np.ones(len(m), bool))):
                sp, so, sh, eid = arrays(S, sel)
                _u, A, B = GT.episode_stats(sp, so, sh, eid)
                ev[(surf, seed, name)] = (A, B, int(sel.sum()))
            reg = main_json["cells"][GT.CTRL_TREE][surf][str(seed)]["g_star"]
            mine = float(ev[(surf, seed, "compliant")][0].sum() / ev[(surf, seed, "compliant")][1].sum())
            if abs(reg - mine) > 1e-12:
                raise SystemExit(f"[addendum] control failed: eval compliant g* {mine!r} != registered {reg!r} ({surf} s{seed})")
        del ctx
    import retime_trainfit_r8 as TF
    per, _ctxs = TF._load_train(tools, GT.SEEDS)       # re-points the archive root: AFTER the eval load
    tr = {}
    for seed in GT.SEEDS:
        for surf in GT.SURFS:
            S = per[surf][seed]
            m = np.asarray(S["mask"], bool)
            for name, sel in (("compliant", m), ("all", np.ones(len(m), bool))):
                sp, so, sh, eid = arrays(S, sel)
                _u, A, B = GT.episode_stats(sp, so, sh, eid)
                tr[(surf, seed, name)] = (A, B, int(sel.sum()))
    rows = {}
    for surf in GT.SURFS:
        for seed in GT.SEEDS:
            for name in ("compliant", "all"):
                At, Bt, nt = tr[(surf, seed, name)]
                Ae, Be, ne = ev[(surf, seed, name)]
                C_t, C_e = GT.draw_counts(len(At)), GT.draw_counts(len(Ae), seed=1)      # independent draws (different clip sets, different seeds)
                ct, ce = GT.gain_cell(At, Bt, C_t), GT.gain_cell(Ae, Be, C_e)
                d = GT.boot_gain(At, Bt, C_t) - GT.boot_gain(Ae, Be, C_e)
                lo, hi = GT.interval(d)
                rows[f"{surf}|{name}|s{seed}"] = {"train": {**ct, "n_windows": nt}, "eval_ema30k": {**ce, "n_windows": ne},
                                                  "diff_train_minus_eval": {"delta": ct["g_star"] - ce["g_star"], "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}}
    res = {"tool": "gain_traj_train_vs_eval_r8.py", "status": "POST-HOC addendum, NOT registered in PREREG_GAINTRAJ; no role in the label", "generated": time.strftime("%FT%T"),
           "prereg_sha256": GT.PREREG_SHA256, "script_sha256": GT.sha256_file(__file__), "tree_eval": GT.CTRL_TREE, "tree_train": f"{TF.TRAIN_ROOT}/{TF.TRAIN_TREE}",
           "estimator": "ratio sum(d*h)/sum(d*d); episode-cluster bootstrap B 2000 within each clip set; train-minus-eval difference from INDEPENDENT episode draws (two different clip sets); "
                        "answers 'another draw of episodes' only", "caveats": GT.CAVEATS + "; train clips are in this run's training set", "rows": rows}
    RT.jdump(res, OUT / "GAINTRAJ_TRAIN_vs_EVAL.json")
    L = ["# GAINTRAJ train vs eval -- EMA-30k flat g* (POST-HOC addendum, not registered; no role in the label)", "",
         "g* [95 % episode-cluster CI] on the 139 TRAIN-clip rolls (registered PREREG section 6) beside the eval139 EMA-30k value on the same window set; the `all windows` eval row is added post hoc "
         "(the registered eval number is compliant-only). Difference = unpaired (two different clip sets).", "",
         "| surface | window set | seed | train g* | eval g* | train - eval | n windows train / eval |", "|---|---|---|---|---|---|---|"]
    for k, v in rows.items():
        surf, name, s = k.split("|")
        t, e, d = v["train"], v["eval_ema30k"], v["diff_train_minus_eval"]
        L.append(f"| {surf} | {name} | {s[1:]} | {t['g_star']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] | {e['g_star']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}] | "
                 f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{' sep' if d['separated'] else ''} | {t['n_windows']:,} / {e['n_windows']:,} |")
    L += ["", "Control: the eval compliant rows reproduce the registered GAINTRAJ.json cells to 1e-12. Caveats: " + res["caveats"] + ".", ""]
    (OUT / "GAINTRAJ_TRAIN_vs_EVAL.md").write_text("\n".join(L), encoding="utf-8")
    print(f"WROTE {OUT / 'GAINTRAJ_TRAIN_vs_EVAL.md'} wall {round(time.time() - t0, 1)} s")
    print("ZZGTADDDONEZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
