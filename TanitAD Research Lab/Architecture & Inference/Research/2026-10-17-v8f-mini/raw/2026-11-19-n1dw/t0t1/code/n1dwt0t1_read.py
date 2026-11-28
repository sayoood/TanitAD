#!/usr/bin/env python3
"""n1dwt0t1_read.py -- PREREG_N1D_T0T1 (sha256 5ef1ba4f...), T0-against-T1 read of N1D on D1-HO.  CPU, zero training.
T1 track = dump cl (rolled 6 s, m=10 causal); T0 track = dump ol (tick-0 open-loop plan; equals the probe extractor plan0, K-T0); GT = dump g.
Label = read_otband.labels (3-class kinematic) via n1dwprobe_probe.kin_labels (sha-pinned).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0."""
import glob, json, os, sys
PD = "/root/strat1/n1dwprobe/code"
sys.path.insert(0, PD)
import numpy as np
import n1dwprobe_probe as P

BAR, NB, SEED = 0.0383, 2000, 0
DUMPS = {"N1D_s0": "/root/strat1/navrbd/out/t1", "N1D_s1": "/root/strat1/navrbd/out/t1", "N1R_s0": "/root/strat1/navrbd_step0/out/t1", "N1R_s1": "/root/strat1/navrbd_step0/out/t1"}
BANKED = json.load(open("/root/strat1/n1dwt0t1/RESULT_NAVRBD_banked.json"))["D1_HO_each_acc"]


def load(arm):
    fs = sorted(glob.glob(f"{DUMPS[arm]}/dump_NAVRBD-HO-{arm}_m10_causal_heads/ep*.npz"))
    Z = [np.load(f) for f in fs]
    cat = {k: np.concatenate([z[k] for z in Z]) for k in ("g", "cl", "ol", "eid", "clip_index", "ws")}
    o = np.lexsort((cat["ws"], cat["clip_index"]))
    return {k: v[o] for k, v in cat.items()}


def pr(CI, a, b, e):
    r = CI.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(e), n_boot=NB, seed=SEED)
    return {"delta": float(np.mean(a) - np.mean(b)), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n_windows": int(len(a)), "n_episodes": int(len(np.unique(e)))}


def lvl(CI, a, e):
    r = CI.episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(e), reduce="mean", n_boot=NB, seed=SEED)
    return {"point": float(np.mean(a)), "lo": float(r["lo"]), "hi": float(r["hi"])}


def main():
    CI = P.load_ci()
    D = {a: load(a) for a in DUMPS}
    ref = D["N1D_s0"]
    eid = ref["eid"]
    prob = []
    if len(eid) != 5736 or len(np.unique(eid)) != 497:
        prob.append(f"grid {len(eid)}/{len(np.unique(eid))} != 5736/497")
    for a, d in D.items():
        for k in ("g", "eid", "clip_index", "ws"):
            if not np.array_equal(d[k], ref[k]):
                prob.append(f"{k} differs {a} vs N1D_s0")
    gl = P.kin_labels(ref["g"])
    lab = {a: {"T1": P.kin_labels(d["cl"]), "T0": P.kin_labels(d["ol"])} for a, d in D.items()}
    acc = {a: {t: (lab[a][t] == gl).astype(np.float64) for t in ("T1", "T0")} for a in D}
    res = {"card": "PREREG_N1D_T0T1.md sha256 5ef1ba4f142b380bd2d4bfbd426070e0bcc93587b53554d0217fb9110c85d9f7",
           "tier": "T0 (tick-0 open-loop plan) against T1-kin (self-action rollout, m=10, causal); zero training",
           "evidence_class": "MEASURED (ours; banked NAVRBD D1-HO dumps, CPU)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B {NB} seed {SEED} over 497 episodes; answers another-draw-of-EPISODES only (training variance = the two seeds)",
           "bar": BAR, "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "problems": prob}
    res["level"] = {a: {t: lvl(CI, acc[a][t], eid) for t in ("T0", "T1")} for a in D}
    res["T0_minus_T1"] = {a: pr(CI, acc[a]["T0"], acc[a]["T1"], eid) for a in D}
    res["N1D_minus_N1R_at_T0"] = {f"s{s}": pr(CI, acc[f"N1D_s{s}"]["T0"], acc[f"N1R_s{s}"]["T0"], eid) for s in (0, 1)}
    res["N1D_minus_N1R_at_T1"] = {f"s{s}": pr(CI, acc[f"N1D_s{s}"]["T1"], acc[f"N1R_s{s}"]["T1"], eid) for s in (0, 1)}
    res["fade_(N1D_T0-T1)_minus_(N1R_T0-T1)"] = {f"s{s}": pr(CI, acc[f"N1D_s{s}"]["T0"] - acc[f"N1D_s{s}"]["T1"], acc[f"N1R_s{s}"]["T0"] - acc[f"N1R_s{s}"]["T1"], eid) for s in (0, 1)}
    pj = json.load(open("/root/strat1/n1dwprobe/out/result/PROBE_RESULT.json"))
    labels = {}
    for s in (0, 1):
        a = f"N1D_s{s}"
        fp = pj["accuracy"][a]["tree"]["0"]["F-PLAN"]["point"]
        t0 = res["level"][a]["T0"]["point"]
        d = res["T0_minus_T1"][a]
        rl = bool(d["delta"] >= BAR and d["lo"] > 0)
        gap = fp - t0
        dl = bool((not rl) and gap >= BAR)
        labels[a] = {"label": "ROLLOUT-LOSES" if rl else "DECODE-LOSES" if dl else "NEITHER",
                     "T0_minus_T1": d["delta"], "T0_minus_T1_ci": [d["lo"], d["hi"]], "T0_acc": t0, "T1_acc": res["level"][a]["T1"]["point"],
                     "probe_Fplan_tree_acc": fp, "Fplan_minus_T0_point_diff_two_estimators": gap}
    res["labels_per_seed"] = labels
    l0, l1 = labels["N1D_s0"]["label"], labels["N1D_s1"]["label"]
    res["label_overall"] = l0 if l0 == l1 else "NEITHER"
    res["label_overall_note"] = "both seeds agree" if l0 == l1 else f"seeds disagree (s0 {l0}, s1 {l1}); the card requires an outcome to hold at both seeds, so the overall label is NEITHER and each seed is printed as one-seed"
    res["N1R_beside"] = {a: {"T0_minus_T1": res["T0_minus_T1"][a], "T0": res["level"][a]["T0"], "T1": res["level"][a]["T1"]} for a in ("N1R_s0", "N1R_s1")}
    c = {}
    k1 = pr(CI, acc["N1D_s0"]["T1"], acc["N1D_s0"]["T1"], eid)
    c["K1"] = {**k1, "pass": bool(k1["delta"] == 0.0 and k1["lo"] == 0.0 and k1["hi"] == 0.0)}
    kre = {}
    for a in D:
        v = float(acc[a]["T1"].mean())
        kre[a] = {"recomputed": v, "banked_RESULT_NAVRBD": BANKED[a], "abs_diff": abs(v - BANKED[a]), "pass": bool(abs(v - BANKED[a]) <= 1e-9)}
    c["K-RE"] = {"cells": kre, "pass": bool(all(v["pass"] for v in kre.values())), "banked_source": "raw/2026-11-16-v8f-strat1/navrbd/read/result/RESULT_NAVRBD.json D1_HO_each_acc"}
    maj = int(np.bincount(gl, minlength=3).argmax())
    cacc = float((gl == maj).mean())
    c["K-CONST"] = {"majority_accuracy": cacc, "expected": 0.338215, "pass": bool(round(cacc, 6) == 0.338215), "gt_class_counts": np.bincount(gl, minlength=3).tolist()}
    kt0 = {}
    for a in ("N1D_s0", "N1D_s1"):
        for tag in ("feat1", "feat2"):
            F = np.load(f"/root/strat1/n1dwprobe/out/{tag}/feat_{a}.npz")
            idx = {(int(c_), int(t)): i for i, (c_, t) in enumerate(zip(F["ci"], F["t0"]))}
            order = [idx[(int(c_), int(w))] for c_, w in zip(D[a]["clip_index"], D[a]["ws"])]
            diff = float(np.abs(F["plan0"][order] - D[a]["ol"]).max())
            kt0[f"{a}|{tag}"] = {"max_abs_diff_m": diff, "n_windows": len(order), "pass": diff == 0.0}
    c["K-T0"] = {"cells": kt0, "pass": bool(all(v["pass"] for v in kt0.values())), "note": "N1D arms only: the extractor was run for N1D only; N1R ol is read the same way from its dump, its parity vs an extractor is not measured"}
    res["controls"] = c
    res["controls_all_pass"] = bool(c["K1"]["pass"] and c["K-RE"]["pass"] and c["K-CONST"]["pass"] and c["K-T0"]["pass"] and not prob)
    res["informational_label_agreement_T0_T1"] = {a: float((lab[a]["T0"] == lab[a]["T1"]).mean()) for a in D}
    res["informational_confusion_GTrow_by_T0col"] = {a: np.bincount(3 * gl + lab[a]["T0"], minlength=9).reshape(3, 3).tolist() for a in D}
    res["informational_confusion_GTrow_by_T1col"] = {a: np.bincount(3 * gl + lab[a]["T1"], minlength=9).reshape(3, 3).tolist() for a in D}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
