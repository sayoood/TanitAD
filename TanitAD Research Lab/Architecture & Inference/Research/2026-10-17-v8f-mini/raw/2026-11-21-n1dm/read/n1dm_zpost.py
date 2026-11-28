"""n1dm_zpost.py -- Z-POST of PREREG_V8F_N1DM section 4: the route head's OWN argmax accuracy on D1-HO at T0 (tick 0, the executor of record's tick-0 batch), against the logistic 0.685 and CEIL 0.73, per seed;
beside it the decoded-plan T0 TAC-LAT of the same windows (`ol`).  Also the control arm (a label permutation across windows must read the chance level), K-ALIGN, K-OL-REAL, K-CONST.
Zero training, CPU, 2 threads, ONE process.  See n1dm_t0route_lib.py for why this tool exists (the dumps carry no route_logits) and how it runs the pinned evaluator in process, one tick.
Usage (D1-HO, the card's):   python -I n1dm_zpost.py --tree TREE --arm N1DM_s0 CKPT_S0 --arm N1DM_s1 CKPT_S1 --profile ho --out RESULT_N1DM_ZPOST.json [--work DIR] [--real-dump N1DM_s0 DIR ...]
     smoke (dev box):        python -I n1dm_zpost.py --tree TREE --arm T_s0 CKPT --profile e141 --max-windows 96 --out OUT.json
--zpost RESULT_N1DM_ZPOST.json joins the result to n1dm_ho_read.py (POST-WEAK).  A missing route head (an off model) is a REFUSAL, never a number.  Estimator: taniteval.ci.episode_cluster_bootstrap B 2000 seed 0
('another draw of EPISODES' only; training variance = the two seeds)."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import n1dm_t0route_lib as TL  # noqa: E402

ZPOST_LOGISTIC, ZPOST_CEIL = 0.685, 0.73
CL = ("lane_keep", "turn_left", "turn_right")
D1HO_MAJ_SHARE = 0.338215


def reader_ns(tree):
    """a namespace the D1-HO reader's setup() accepts: it gives the pinned ci / four_families modules and the paired / level estimators of the read"""
    import argparse as ap
    return ap.Namespace(tree=tree, inp=HERE + "/in", cmp_in=HERE + "/in")


def arm_block(R, name, cap, dump, real_dump, profile, perm_seed=0, mutate_shift=False):
    import numpy as np
    perm, kal = TL.align(cap, dump)
    blk = {"K-ALIGN": kal}
    if perm is None or not kal["pass_"]:
        blk["status"] = "INVALID (K-ALIGN failed)"
        return blk
    rl = np.concatenate([c["route_logits"] for c in cap.calls])[perm].astype(np.float64)
    glat, glon = TL.kin_labels(dump["g"])
    eid = dump["eid"]
    pred = rl.argmax(-1)
    if mutate_shift:                                                       # MUTANT: shift the labels by one window (the K-ALIGN-style regression the tests use)
        glat = np.roll(glat, 1)
    head_ok = (pred == glat).astype(np.float64)
    dec_lat, _ = TL.kin_labels(dump["ol"])
    dec_ok = (dec_lat == glat).astype(np.float64)
    sm = np.exp(rl - rl.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    blk["n_windows"], blk["n_episodes"] = int(len(glat)), int(len(np.unique(eid)))
    blk["route_head_acc_T0"] = R.level(head_ok, eid)
    blk["decoded_plan_acc_T0"] = R.level(dec_ok, eid)
    blk["head_minus_decoded"] = R.paired(head_ok, dec_ok, eid)
    blk["vs_logistic_0.685"] = blk["route_head_acc_T0"]["point"] - ZPOST_LOGISTIC
    blk["vs_CEIL_0.73"] = blk["route_head_acc_T0"]["point"] - ZPOST_CEIL
    blk["argmax_class_counts"] = {c: int((pred == i).sum()) for i, c in enumerate(CL)}
    blk["gt_class_share"] = {c: float((glat == i).mean()) for i, c in enumerate(CL)}
    blk["mean_max_posterior"] = float(sm.max(1).mean())
    # deliberate-regression arm: the labels permuted ACROSS windows (episodes break) must read the chance level sum_k p_pred(k) * p_gt(k)
    rs = np.random.RandomState(perm_seed)
    pl = glat[rs.permutation(len(glat))]
    chance = float(sum((pred == k).mean() * (glat == k).mean() for k in range(3)))
    pacc = float((pred == pl).mean())
    blk["K-PERM"] = {"permuted_label_acc": pacc, "chance_expected": chance, "abs_diff": abs(pacc - chance), "pass": bool(abs(pacc - chance) <= 0.03),
                     "note": "deliberate regression: a head scored against shuffled labels must read the chance level (tolerance 0.03 on n windows)"}
    maj = int(np.bincount(glat, minlength=3).argmax())
    cacc = float((np.full(len(glat), maj) == glat).mean())
    blk["K-CONST"] = {"majority_share": cacc, "must_read": D1HO_MAJ_SHARE if profile == "ho" else None, "pass": bool(cacc == float((glat == maj).mean()) and (profile != "ho" or round(cacc, 6) == D1HO_MAJ_SHARE))}
    if real_dump:
        try:
            rd = TL.load_dump(real_dump)
            same = bool(np.array_equal(rd["ol"], dump["ol"]) and np.array_equal(rd["eid"], dump["eid"]))
            blk["K-OL-REAL"] = {"real_dump": real_dump, "ol_equal": same, "pass": same}
        except Exception as e:                                          # noqa: BLE001
            blk["K-OL-REAL"] = {"real_dump": real_dump, "pass": False, "error": str(e)[:200]}
    blk["status"] = "OK" if (blk["K-PERM"]["pass"] and blk["K-CONST"]["pass"] and blk.get("K-OL-REAL", {"pass": True})["pass"]) else "INVALID (a control failed)"
    return blk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="git archive export of the-N1DM-commit (+ the hb wrapper, sha256 74fd8445)")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "CKPT"), required=True, help="NAME N1DM_s0 | N1DM_s1 (or any test name) and its checkpoint")
    ap.add_argument("--profile", choices=["ho", "e141"], default="ho")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=HERE + "/zpost_work", help="scratch dir for the one-tick dumps (delete after the read)")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", default="causal", choices=["causal", "logged"], help="causal = the executor of record's bucket (the dumps' --limit)")
    ap.add_argument("--real-dump", nargs=2, action="append", metavar=("NAME", "DIR"), default=[], help="the chain's real D1-HO dump of that arm (K-OL-REAL)")
    ap.add_argument("--mutant", action="store_true", help="shift the labels one window (the read must differ from the unshifted one)")
    ap.add_argument("--hb-dir", default=None, help="folder holding v8f_t1_eval_hb.py (sha256 74fd8445) when it is not in TREE/stack/scripts")
    for k in ("tracks", "cache", "grid", "join", "navtext", "rb"):
        ap.add_argument("--ho-" + k, default=None)
    a = ap.parse_args(argv)
    tree = TL.setup_paths(a.tree, a.hb_dir)
    import n1dm_ho_read as R
    R.setup(reader_ns(tree))
    rd = dict(a.real_dump)
    res = {"tool": "n1dm_zpost.py", "card": "PREREG_V8F_N1DM.md sha256 N1DM-DRAFT section 4 Z-POST; code of record the-N1DM-commit", "profile": a.profile, "tier": "T0 (tick 0 of the T1-kin/hb0.2 rollout, limit " + a.limit + ")",
           "evidence_class": "MEASURED (ours; checkpoint forward, zero training)" if a.profile == "ho" else "SMOKE on eval-141 inputs: machinery only, NOT a D1-HO number", "bars": {"logistic": ZPOST_LOGISTIC, "CEIL": ZPOST_CEIL},
           "estimator": "taniteval.ci.episode_cluster_bootstrap B 2000 seed 0; 'another draw of EPISODES' only", "max_windows": a.max_windows, "mutant": bool(a.mutant), "arms": {}}
    t0 = time.time()
    for name, ckpt in a.arm:
        out_dir = f"{a.work}/{name}"
        try:
            cap, dd = TL.run_tick0(tree, a.profile, ckpt, out_dir, f"ZPOST-{name}", limit=a.limit, max_windows=a.max_windows, batch=a.batch, a=a)
            if cap.n_no_head or not cap.calls:
                res["arms"][name] = {"status": f"REFUSED: the checkpoint's model has no route head ({cap.n_no_head} forward calls without route_logits); not a Z-POST number"}
                continue
            res["arms"][name] = arm_block(R, name, cap, TL.load_dump(dd), rd.get(name), a.profile, mutate_shift=a.mutant)
        except Exception as e:                                          # noqa: BLE001
            res["arms"][name] = {"status": f"FAILED: {type(e).__name__}: {str(e)[:300]}"}
        print(time.strftime("%H:%M:%S"), name, res["arms"][name].get("status"), round(time.time() - t0), "s", flush=True)
    ok = all(v.get("status") == "OK" for v in res["arms"].values())
    res["status"] = "VALID" if ok else "INVALID (" + "; ".join(f"{k}: {v.get('status')}" for k, v in res["arms"].items() if v.get("status") != "OK") + ")"
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for k, v in res["arms"].items():
        if "route_head_acc_T0" in v:
            print(f"{k}: route head T0 {v['route_head_acc_T0']['point']:.4f} [{v['route_head_acc_T0']['lo']:.4f},{v['route_head_acc_T0']['hi']:.4f}]  decoded plan T0 {v['decoded_plan_acc_T0']['point']:.4f}  "
                  f"vs 0.685 {v['vs_logistic_0.685']:+.4f}  vs 0.73 {v['vs_CEIL_0.73']:+.4f}  K-PERM {'ok' if v['K-PERM']['pass'] else 'FAIL'}")
    print("WROTE", a.out, res["status"][:80])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
