#!/usr/bin/env python3
"""run_p5.py -- readout of trigger probe 5 (PREREG_FROZEN.md sections 4-7).  Dev box, CPU only, CUDA_VISIBLE_DEVICES=-1, <= 4 threads.
  python -I run_p5.py assemble | pca | lin | strata | eval | decide      (each stage banks to raw/ and is resumable)
assemble: box-2 feature shards (SCR/out) -> SCR/FEAT_all.npy fp16 [24608, 2 (TL, LEAD), 10240] in the row order of raw/frames.npz
pca     : PCA-64 per crop per population, fitted per OUTER fold on the outer-train rows (+ all rows), tp_lib.StreamPCA seed 0 -> raw/pca_<crop>_<pop>.npz
lin     : arms E, V+E (banked probe-1 OOF), TL+E, LEAD+E, C+E, C+V+E + controls C1/C2/C3 for EVERY arm name -> raw/res_<pop>_p5.json, raw/oof_<pop>_p5.npz
strata  : probe 3's strata on the banked + new OOF scores;  eval : no-refit eval-141 confirmation;  decide : the committed rule -> raw/VERDICT.json
Estimator: episode(clip)-cluster bootstrap B=2000 seed 0 (another draw of EPISODES; blind to training variance H-ESTIM-SEED-1).  Evidence class MEASURED; tier T0-class probe."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "4")
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = os.environ.get("P5_RAW", f"{HERE}/raw")                 # P5_RAW / P5_SCR / P5_POPS: smoke-test overrides only
SCR = os.environ.get("P5_SCR", "F:/TanitAD-scratch/trigger_probe5_scratch")
P1RAW = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe/raw"
P1SCR = "F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402

POPS = tuple(os.environ.get("P5_POPS", "pstop,pmove").split(","))
CROPS = ("TL", "LEAD")
K = 64
D_CROP = 10 * 1024
STRATA = ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart")
ARMS_NEW = ("TL+E", "LEAD+E", "C+E", "C+V+E")
ARMS_CTRL = ("E", "V+E") + ARMS_NEW           # controls run for EVERY arm name (the probe-1 `if src == "V"` trap is closed)
LEAK_ARMS = ("V+E",) + ARMS_NEW


def control_arms():
    """pytested: the control-arm list is non-empty for every source name."""
    return list(ARMS_CTRL)


def load_pop(pop):
    A = np.load(f"{P1RAW}/windows_train.npz", allow_pickle=False)
    idx = np.load(f"{P1RAW}/sample_{pop}.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    d = {"idx": idx, "y": A["go"][idx].astype(int), "clip": clip, "sha": sha, "outer": L.outer_fold(sha), "inner": L.inner_fold(sha), "E": A["E"][idx].astype(np.float64),
         "v0": A["v0"][idx], "name": np.array([str(A["clips"][c]) for c in clip]), "t": A["t"][idx].astype(int)}
    for k in STRATA:
        d[k] = A[k][idx]
    return d


def frame_rows(pop_or_eval):
    """global row of every window of a population in FEAT_all (raw/frames.npz order)."""
    F = np.load(f"{RAW}/frames.npz")
    key = {(str(c), int(t)): i for i, (c, t) in enumerate(zip(F["clip"], F["t"]))}
    if pop_or_eval == "eval":
        E = np.load(f"{P1RAW}/eval141_features.npz", allow_pickle=True)
        return np.array([key[(str(c), int(t))] for c, t in zip(E["clip"], E["t0"])])
    d = load_pop(pop_or_eval)
    return np.array([key[(str(c), int(t))] for c, t in zip(d["name"], d["t"])])


def stage_assemble():
    F = np.load(f"{RAW}/frames.npz")
    key = {(str(c), int(t)): i for i, (c, t) in enumerate(zip(F["clip"], F["t"]))}
    n = len(key)
    mm = np.lib.format.open_memmap(f"{SCR}/FEAT_all.npy", mode="w+", dtype=np.float16, shape=(n, 2, D_CROP))
    seen = np.zeros(n, bool)
    for jp in sorted(glob.glob(f"{SCR}/out/feat_shard_*.json")):
        j = json.load(open(jp))
        arr = np.load(jp.replace(".json", ".npy"), mmap_mode="r")
        assert arr.shape == (j["n"], 2, 10, 1024), (jp, arr.shape)
        for r, (c, t, _) in enumerate(j["frames"]):
            g = key[(c, int(t))]
            assert not seen[g], ("duplicate", c, t)
            mm[g] = np.asarray(arr[r]).reshape(2, D_CROP)
            seen[g] = True
    assert seen.all(), f"{int((~seen).sum())} frames missing"
    mm.flush()
    print("assembled", n, "frames", flush=True)


def feat_reader(rows, crop):
    X = np.load(f"{SCR}/FEAT_all.npy", mmap_mode="r")
    ci = CROPS.index(crop)
    return lambda r: np.asarray(X[rows[r], ci], np.float32)


def stage_pca():
    for pop in POPS:
        d = load_pop(pop)
        rows = frame_rows(pop)
        n = len(rows)
        for crop in CROPS:
            path = f"{RAW}/pca_{crop}_{pop}.npz"
            if os.path.exists(path):
                continue
            read = feat_reader(rows, crop)
            S = np.zeros((6, n, K), np.float32)
            t0 = time.time()
            for f in range(6):
                tr = np.flatnonzero(d["outer"] != f) if f < 5 else np.arange(n)
                p = L.StreamPCA(k=K, seed=0).fit(read, tr, D_CROP)
                S[f] = p.transform(read, np.arange(n))
                if f == 5:
                    np.savez(f"{SCR}/pcacomp_{crop}_{pop}_all.npz", mu=p.mu, comp=p.comp, S=p.S)
                print("pca", pop, crop, "fold", f, f"{time.time() - t0:.0f}s", flush=True)
            np.savez(path, S=S)


def sources(pop):
    S = {c: np.load(f"{RAW}/pca_{c}_{pop}.npz")["S"] for c in CROPS}
    S["V"] = np.load(f"{P1RAW}/pca_V_{pop}.npz")["S"]
    return S


def arm_fn(name, d, S):
    parts = []
    for p in name.split("+"):
        parts += list(CROPS) if p == "C" else [p]

    def f(fold):
        return np.concatenate([d["E"] if p == "E" else S[p][fold].astype(np.float64) for p in parts], 1)
    return f


def stage_lin():
    for pop in POPS:
        d = load_pop(pop)
        S = sources(pop)
        y, clip, outer, inner = d["y"], d["clip"], d["outer"], d["inner"]
        bk = np.load(f"{P1RAW}/oof_{pop}_V_linear.npz")
        assert (bk["y"] == y).all() and (bk["clip"] == clip).all(), "banked probe-1 rows differ"
        counts = L.boot_counts(clip)
        oof, res = {"E": bk["E"], "V+E": bk["V+E"]}, {"pop": pop, "n": int(len(y)), "n_go": int(y.sum()), "arms": {}, "controls": {}}
        t0 = time.time()

        def rec(o, labels):
            return {**L.auc_ci(o, labels, clip, counts=counts), "foldmean": L.auc_ci_fold(o, labels, clip, outer, counts=counts)}

        def run(name, labels, tag, extra_y=False):
            fn = arm_fn(name, d, S)
            if extra_y:
                base = fn
                fn = lambda f, b=base: np.concatenate([b(f), y[:, None].astype(np.float64)], 1)
            o, hps = L.cv_oof(fn, labels, outer, inner, "linear")
            oof[tag] = o
            r = {**rec(o, labels), "hp": {k: v["hp"] for k, v in hps.items()}}
            print(pop, tag, f"fold-mean {r['foldmean']['auc']:.4f} [{r['foldmean']['lo']:.4f},{r['foldmean']['hi']:.4f}] {time.time() - t0:.0f}s", flush=True)
            return r
        for nm in ("E", "V+E"):
            res["arms"][nm] = rec(oof[nm], y)
        for nm in ARMS_NEW:
            res["arms"][nm] = run(nm, y, nm)
        ys, yd = L.shuffle_labels(y, 1), L.donor_labels(y, clip, 2)
        for nm in control_arms():
            res["controls"][f"C1:{nm}"] = run(nm, ys, f"C1__{nm}")
            res["controls"][f"C2:{nm}"] = run(nm, yd, f"C2__{nm}")
        for nm in LEAK_ARMS:
            res["controls"][f"C3:{nm}"] = run(nm, y, f"C3__{nm}", extra_y=True)
        for nm in control_arms():                                        # C2 band rule: seeds 3-6 for an arm whose seed-2 CI misses 0.5
            c = res["controls"][f"C2:{nm}"]["foldmean"]
            if c["lo"] > 0.5 or c["hi"] < 0.5:
                aucs = [c["auc"]]
                for sd in (3, 4, 5, 6):
                    r = run(nm, L.donor_labels(y, clip, sd), f"C2s{sd}__{nm}")
                    res["controls"][f"C2s{sd}:{nm}"] = r
                    aucs.append(r["foldmean"]["auc"])
                res["controls"][f"C2band:{nm}"] = {"seeds_2_6_mean_foldmean_auc": float(np.mean(aucs)), "in_band_0.47_0.53": bool(0.47 <= np.mean(aucs) <= 0.53)}

        P = {}
        for a_, b_ in (("C+V+E", "V+E"), ("C+E", "E"), ("TL+E", "E"), ("LEAD+E", "E"), ("V+E", "E"), ("C+V+E", "E"), ("TL+E", "V+E"), ("LEAD+E", "V+E")):
            P[f"{a_}_minus_{b_}"] = {**L.dauc_ci(oof[a_], oof[b_], y, clip, counts=counts), "foldmean": L.dauc_ci_fold(oof[a_], oof[b_], y, clip, outer, counts=counts)}
        for tag in ("C1", "C2"):
            lab = ys if tag == "C1" else yd
            P[f"{tag}:C+V+E_minus_V+E"] = {**L.dauc_ci(oof[f"{tag}__C+V+E"], oof[f"{tag}__V+E"], lab, clip, counts=counts), "foldmean": L.dauc_ci_fold(oof[f"{tag}__C+V+E"], oof[f"{tag}__V+E"], lab, clip, outer, counts=counts)}
        res["paired"] = P
        res["seconds"] = time.time() - t0
        np.savez(f"{RAW}/oof_{pop}_p5.npz", **oof, y=y, clip=clip)
        json.dump(res, open(f"{RAW}/res_{pop}_p5.json", "w"), indent=1)


def controls_ok(res, tol=0.5):
    """PREREG 7 as a pure function of one population's result json -> (ok, reasons)."""
    why = []
    C = res["controls"]
    for nm in ("TL+E", "LEAD+E", "C+V+E", "V+E"):
        c = C[f"C1:{nm}"]["foldmean"]
        if not (c["lo"] <= tol <= c["hi"]):
            why.append(f"C1 {nm} CI [{c['lo']:.3f},{c['hi']:.3f}] misses 0.5")
    p = res["paired"]["C1:C+V+E_minus_V+E"]["foldmean"]
    if not (p["lo"] <= 0 <= p["hi"]):
        why.append("C1 paired null misses 0")
    for nm in control_arms():
        c = C[f"C2:{nm}"]["foldmean"]
        if not (c["lo"] <= tol <= c["hi"]):
            b = C.get(f"C2band:{nm}")
            if not (b and b["in_band_0.47_0.53"]):
                why.append(f"C2 {nm} misses 0.5 and the 5-draw band")
    for nm in ("TL+E", "LEAD+E", "C+V+E"):
        if C[f"C3:{nm}"]["foldmean"]["auc"] < 0.99:
            why.append(f"C3 leak {nm} AUC {C[f'C3:{nm}']['foldmean']['auc']:.3f} < 0.99")
    return (not why), why


def rule(d_by_pop, valid_by_pop):
    """PREREG 5, a pure function: d_by_pop[pop] = (delta, lo); returns HELP | NOT-HELP | INCONCLUSIVE | INCOMPLETE."""
    if any(p not in d_by_pop for p in POPS):
        return "INCOMPLETE"
    if any(valid_by_pop[p] and d_by_pop[p][0] >= 0.03 and d_by_pop[p][1] > 0 for p in POPS):
        return "CROPS HELP"
    if all(valid_by_pop[p] and d_by_pop[p][0] < 0.01 for p in POPS):
        return "CROPS DO NOT HELP"
    return "INCONCLUSIVE"


def stage_decide():
    dd, vv, why = {}, {}, {}
    for pop in POPS:
        p = f"{RAW}/res_{pop}_p5.json"
        if os.path.exists(p):
            r = json.load(open(p))
            f = r["paired"]["C+V+E_minus_V+E"]["foldmean"]
            dd[pop] = (f["delta"], f["lo"])
            vv[pop], why[pop] = controls_ok(r)
    out = {"verdict": rule(dd, vv), "dAUC_C+V+E_minus_V+E_foldmean": {k: {"delta": v[0], "lo": v[1]} for k, v in dd.items()}, "controls_ok": vv, "control_failures": why,
           "estimator": "episode-cluster bootstrap B=2000 seed 0", "tier": "T0-class probe of representations on logged inputs", "evidence": "MEASURED"}
    json.dump(out, open(f"{RAW}/VERDICT.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def stage_strata():
    out = {}
    for pop in POPS:
        d = load_pop(pop)
        o = np.load(f"{RAW}/oof_{pop}_p5.npz")
        sc = {k: o[k] for k in ("E", "V+E", "TL+E", "LEAD+E", "C+V+E")}
        y, clip, outer = d["y"], d["clip"], d["outer"]
        P = {}
        for s in STRATA:
            ms = d[s].astype(bool)
            nc = int(len(np.unique(clip[ms]))) if ms.any() else 0
            if not (ms.sum() >= 60 and y[ms].sum() >= 30 and (ms.sum() - y[ms].sum()) >= 30 and nc >= 20):
                P[s] = {"n": int(ms.sum()), "auc": "too small (<30 windows in a class or <20 episodes in the sample)"}
                continue
            cnt = L.boot_counts(clip[ms])
            P[s] = {"n": int(ms.sum()), "n_clips": nc, "auc": {k: L.auc_ci_fold(v[ms], y[ms], clip[ms], outer[ms], counts=cnt) for k, v in sc.items()},
                    "paired": {f"{a}_minus_{b}": L.dauc_ci_fold(sc[a][ms], sc[b][ms], y[ms], clip[ms], outer[ms], counts=cnt)
                               for a, b in (("TL+E", "E"), ("LEAD+E", "E"), ("C+V+E", "E"), ("C+V+E", "V+E"), ("V+E", "E"))}}
            print(pop, s, P[s]["n"], {k: round(v["auc"], 3) for k, v in P[s]["auc"].items()}, flush=True)
        out[pop] = P
    json.dump(out, open(f"{RAW}/results_strata_p5.json", "w"), indent=1, default=float)


def stage_eval():
    ev = np.load(f"{P1RAW}/eval141_features.npz", allow_pickle=True)
    rows_e = frame_rows("eval")
    X = np.load(f"{SCR}/FEAT_all.npy", mmap_mode="r")
    Ve = np.load(f"{P1SCR}/V_eval141.npy", mmap_mode="r")
    out = {}
    for pop in POPS:
        sel = ev["v0"] < 0.5 if pop == "pstop" else (ev["v0"] >= 0.5) & (ev["v0"] < 6.0)
        d = load_pop(pop)
        S = sources(pop)
        te = {}
        for ci, c in enumerate(CROPS):
            pc = np.load(f"{SCR}/pcacomp_{c}_{pop}_all.npz")
            te[c] = (np.asarray(X[rows_e[sel], ci], np.float32) - pc["mu"]) @ pc["comp"].T
        pv = np.load(f"{P1SCR}/pcacomp_V_{pop}_all.npz")
        te["V"] = (np.asarray(Ve[sel], np.float32) - pv["mu"]) @ pv["comp"].T
        Ee = ev["E"][sel].astype(np.float64)
        ye, eid = ev["go"][sel].astype(int), ev["eid"][sel]
        cnt = L.boot_counts(eid)
        sc, res = {}, {"n": int(sel.sum()), "n_go": int(ye.sum()), "n_episodes": int(len(np.unique(eid))), "arms": {}, "paired": {}}
        for nm in ("E", "V+E", "TL+E", "LEAD+E", "C+V+E"):
            parts = []
            for p in nm.split("+"):
                parts += list(CROPS) if p == "C" else [p]
            Xtr = np.concatenate([d["E"] if p == "E" else S[p][5].astype(np.float64) for p in parts], 1)
            Xte = np.concatenate([Ee if p == "E" else te[p].astype(np.float64) for p in parts], 1)
            pred, hp, _ = L.fit_probe(Xtr, d["y"], d["inner"], "linear")
            sc[nm] = pred(Xte)
            res["arms"][nm] = {**L.auc_ci(sc[nm], ye, eid, counts=cnt), "hp": hp}
            print("eval141", pop, nm, round(res["arms"][nm]["auc"], 4), flush=True)
        for a_, b_ in (("C+V+E", "V+E"), ("C+V+E", "E"), ("TL+E", "E"), ("LEAD+E", "E"), ("V+E", "E")):
            res["paired"][f"{a_}_minus_{b_}"] = L.dauc_ci(sc[a_], sc[b_], ye, eid, counts=cnt)
        out[pop] = res
    json.dump(out, open(f"{RAW}/results_eval141_p5.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"assemble": stage_assemble, "pca": stage_pca, "lin": stage_lin, "strata": stage_strata, "eval": stage_eval, "decide": stage_decide}[sys.argv[1]]()
