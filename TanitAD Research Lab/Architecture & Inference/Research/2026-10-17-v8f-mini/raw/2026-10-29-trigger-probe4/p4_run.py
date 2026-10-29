#!/usr/bin/env python3
"""p4_run.py -- the stages of trigger probe 4 (PREREG_FROZEN.md).  Every stage is a function of an input BUNDLE B (real: p4_lib.real_data(pop); the tests build a synthetic bundle of the same shape and run THE SAME functions),
is resumable (per arm-seed banking) and writes only under ``out_raw`` (small) / ``out_scr`` (big).
    python -I p4_run.py --stage arms|controls|post|eval141 --pop pstop|pmove      (post = strata + attention maps)
    python -I p4_run.py --stage tables                                            (reads everything; VERDICT.json, tables4.txt)
CUDA_VISIBLE_DEVICES=-1; torch / BLAS threads <= 6 (P4_THREADS)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np
import p4_lib as P

L = P.L
STR = ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart")
CTRL_ARMS = ("G8att", "G4att")
CTRL_SEED = 0                          # readout seed of the control fits (one fit per control arm; stated in PREREG 6)


def _grid_get(B, arm):
    grid = P.ARM_GRID[arm]
    return grid, (lambda f: P.fold_view(B["S"][grid], f))


def _save_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)


def _save_npz(path, d):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ stage: arms
def stage_arms(B, cfg, out_raw, out_scr, arms=P.ARMS, ram_min=4000, log=print):
    """4 arms x 3 readout seeds: per-fold fits, OOF logits of every (arm, seed) banked as soon as it finishes; the arm score = the mean of its seeds' logits."""
    pop, d = B["pop"], B["d"]
    oofp, infp = f"{out_raw}/oof4_{pop}.npz", f"{out_raw}/fits4_{pop}.json"
    oof = dict(np.load(oofp)) if os.path.exists(oofp) else {}
    info = json.load(open(infp)) if os.path.exists(infp) else {}
    for arm in arms:
        grid, sget = _grid_get(B, arm)
        kind, hw = P.ARM_KIND[arm], P.GRID_HW[grid]
        for seed in cfg["seeds"]:
            key = f"{arm}+E__s{seed}"
            if key in oof:
                continue
            P.require_ram(ram_min, f"arm {key} {pop}")
            t0 = time.time()
            o, att, inf = P.cv_oof_nn(sget, d["E"], d["y"], d["outer"], d["sha"], kind, hw, seed, cfg, collect_attn=(kind == "att"))
            assert np.isfinite(o).all()
            oof[key] = o
            if att is not None:
                np.save(f"{out_scr}/attn4_{pop}_{arm}_s{seed}.npy", att)
            info[key] = inf
            _save_npz(oofp, oof)
            _save_json(infp, info)
            fa = P.fold_mean_auc(o, d["y"], d["outer"])
            log(f"{pop} {key}: fold-mean AUC {fa:.4f}  epochs {[i['epochs_run'] for i in inf]}  {time.time() - t0:.0f}s")
        ks = [f"{arm}+E__s{s}" for s in cfg["seeds"]]
        if all(k in oof for k in ks):
            oof[f"{arm}+E"] = np.mean([oof[k] for k in ks], 0)
            _save_npz(oofp, oof)
    return oof


# ------------------------------------------------------------------------------------------------ stage: controls (C1 within-fold shuffle, C2 donor-label, with the conditional seeds 3-6)
def donor(d, seed):
    return L.donor_labels(d["y"], d["clip"], seed)


def stage_controls(B, cfg, out_raw, out_scr, ram_min=4000, log=print, arms=CTRL_ARMS):
    pop, d = B["pop"], B["d"]
    cp = f"{out_raw}/ctrl4_{pop}.npz"
    C = dict(np.load(cp)) if os.path.exists(cp) else {}
    counts = L.boot_counts(d["clip"], cfg["n_boot"], cfg["boot_seed"])

    def run(name, arm, labels):
        key = f"{name}:{arm}+E".replace(":", "__")
        if key in C:
            return C[key]
        P.require_ram(ram_min, f"control {key} {pop}")
        grid, sget = _grid_get(B, arm)
        t0 = time.time()
        o, _, _ = P.cv_oof_nn(sget, d["E"], labels, d["outer"], d["sha"], P.ARM_KIND[arm], P.GRID_HW[grid], CTRL_SEED, cfg)
        C[key] = o
        _save_npz(cp, C)
        r = P.rec(o, labels, d["clip"], d["outer"], counts)["fold"]
        log(f"{pop} control {name} {arm}+E: fold-mean {r['auc']:.4f} [{r['lo']:.4f},{r['hi']:.4f}]  {time.time() - t0:.0f}s")
        return o

    ys = P.shuffle_within_folds(d["y"], d["outer"], 1)
    for arm in arms:
        run("C1:shuffle", arm, ys)
    yd = donor(d, 2)
    for arm in arms:
        o = run("C2:donor", arm, yd)
        r = P.rec(o, yd, d["clip"], d["outer"], counts)["fold"]
        if not (r["lo"] <= 0.5 <= r["hi"]):
            for sd in (3, 4, 5, 6):
                run(f"C2:donor{sd}", arm, donor(d, sd))
    return C


# ------------------------------------------------------------------------------------------------ stage: post (strata + attention maps)
def attn_mean(out_scr, pop, arm, seeds):
    a = None
    for s in seeds:
        x = np.load(f"{out_scr}/attn4_{pop}_{arm}_s{s}.npy").astype(np.float32)
        a = x if a is None else a + x
    return a / len(seeds)


def stage_post(B, cfg, out_raw, out_scr, log=print):
    pop, d = B["pop"], B["d"]
    oof = dict(np.load(f"{out_raw}/oof4_{pop}.npz"))
    bk = B["bk"]
    sc = {"E": bk["E"], "V+E": bk["V+E"], "G4att+E": oof["G4att+E"], "G8att+E": oof["G8att+E"]}
    if "G4max+E" in oof:
        sc["G4max+E"], sc["G8max+E"] = oof["G4max+E"], oof["G8max+E"]
    if "P3:G8+E" in bk:
        sc["P3:G8+E"], sc["P3:G4+E"] = bk["P3:G8+E"], bk["P3:G4+E"]
    pairs = (("G4att+E", "E"), ("G8att+E", "E"), ("G8att+E", "G4att+E"), ("G8att+E", "V+E"))
    y, clip, outer = d["y"], d["clip"], d["outer"]
    out = {"note": "TRAIN only; TL_* are CLIP-level and VLM-derived (vlm-cot, disputed, untimed); lead_depart reads the future (a stratum, never a feature); BESIDE, never decisive", "pop": pop, "strata": {}}
    for s in STR:
        ms = d[s].astype(bool)
        r = {"n_sample": int(ms.sum()), "n_sample_clips": int(len(np.unique(clip[ms]))) if ms.any() else 0, "n_sample_go": int(y[ms].sum()) if ms.any() else 0}
        ok = ms.sum() >= 60 and y[ms].sum() >= 30 and (ms.sum() - y[ms].sum()) >= 30 and r["n_sample_clips"] >= 20
        if not ok:
            r["auc"] = "too small (<30 windows in a class or <20 episodes in the sample)"
        else:
            cnt = L.boot_counts(clip[ms], cfg["n_boot"], cfg["boot_seed"])
            r["auc"] = {k: P.rec(v[ms], y[ms], clip[ms], outer[ms], cnt) for k, v in sc.items()}
            r["paired"] = {f"{a}_minus_{b}": P.paired(sc[a][ms], sc[b][ms], y[ms], clip[ms], outer[ms], cnt) for a, b in pairs}
        out["strata"][s] = r
    _save_json(f"{out_raw}/results_strata4_{pop}.json", out)
    # attention maps (held-out softmax, mean over seeds, queries and windows; descriptive, NOT an attribution)
    maps, desc = {}, {}
    for arm in ("G8att", "G4att"):
        hw = P.GRID_HW[P.ARM_GRID[arm]]
        att = attn_mean(out_scr, pop, arm, cfg["seeds"])
        masks = {"ALL": np.ones(len(y), bool)}
        for s in STR:
            if isinstance(out["strata"][s]["auc"], dict):
                masks[s] = d[s].astype(bool)
        for s, ms in masks.items():
            for cls, mc in (("GO", ms & (y == 1)), ("STAY", ms & (y == 0))):
                m = P.attn_map(att, mc, hw)
                if m is not None:
                    maps[f"{arm}__{s}__{cls}"] = m
                    desc[f"{arm}__{s}__{cls}"] = {"n": int(mc.sum()), **P.attn_descriptors(m)}
        for q in range(cfg["n_query"]):
            maps[f"{arm}__ALL__query{q}"] = att[:, q].astype(np.float64).mean(0).reshape(hw)
    _save_npz(f"{out_raw}/maps4_{pop}.npz", maps)
    _save_json(f"{out_raw}/maps4_{pop}_descriptors.json", desc)
    plot_maps(maps, desc, pop, f"{out_raw}/attn_maps_{pop}.png")
    log(f"{pop} post: strata + {len(maps)} maps written")


def plot_maps(maps, desc, pop, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for arm in ("G8att", "G4att"):
        strata = [k.split("__")[1] for k in maps if k.startswith(f"{arm}__") and k.endswith("__GO")]
        if not strata:
            continue
        fig, ax = plt.subplots(len(strata), 3, figsize=(10.5, 1.9 * len(strata) + 0.6), squeeze=False)
        for i, s in enumerate(strata):
            go, st = maps[f"{arm}__{s}__GO"], maps.get(f"{arm}__{s}__STAY")
            for j, (title, m, cm) in enumerate((("GO", go, "viridis"), ("STAY", st, "viridis"), ("GO - STAY", None if st is None else go - st, "coolwarm"))):
                a = ax[i, j]
                if m is None:
                    a.axis("off")
                    continue
                v = np.abs(m).max() if j == 2 else None
                im = a.imshow(m, cmap=cm, vmin=(-v if j == 2 else None), vmax=v, aspect="equal")
                a.set_title(f"{s} {title}  n={desc.get(f'{arm}__{s}__{title}', {}).get('n', '')}" if j < 2 else f"{s} {title}", fontsize=8)
                a.set_xticks([])
                a.set_yticks([])
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        fig.suptitle(f"{pop} {arm}: held-out mean softmax attention (descriptive; not an attribution)", fontsize=9)
        fig.tight_layout()
        fig.savefig(path.replace(".png", f"_{arm}.png"), dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ stage: eval-141 no-refit confirmation (needs probe 3's G8_eval141.npy)
def pop_sel(ev, pop):
    return ev["v0"] < 0.5 if pop == "pstop" else (ev["v0"] >= 0.5) & (ev["v0"] < 6.0)


def stage_eval141(B, cfg, out_raw, out_scr, p3raw=P.P3RAW, p3scr=P.P3SCR, oldraw=P.OLDRAW, ram_min=4000, log=print):
    pop, d = B["pop"], B["d"]
    rp = f"{out_raw}/results_eval141_4_{pop}.json"
    exj, g8p = f"{p3raw}/eval_g_extract.json", f"{p3scr}/G8_eval141.npy"
    if not (os.path.exists(exj) and os.path.exists(g8p) and json.load(open(exj)).get("gate_pass")):
        _save_json(rp, {"pop": pop, "run": False, "reason": "probe 3's eval-141 G8 features (G8_eval141.npy) or their passed alignment gate (eval_g_extract.json) are absent: NOT RUN"})
        log(f"{pop} eval141: NOT RUN (probe 3 eval features absent)")
        return None
    ev = np.load(f"{oldraw}/eval141_features.npz", allow_pickle=False)
    sel = pop_sel(ev, pop)
    PC = np.load(f"{p3raw}/pcaG_{pop}.npz")
    G8e = np.load(g8p)[sel].astype(np.float32).reshape(-1, 160, 1024)
    Se = {"G8": P.project_tokens(G8e, [{"mu": PC["mu8"][5], "comp": PC["comp8"][5]}])[0], "G4": P.project_tokens(P.g4_from_g8(G8e), [{"mu": PC["mu4"][5], "comp": PC["comp4"][5]}])[0]}
    Ee, ye, eid = ev["E"][sel].astype(np.float64), ev["go"][sel].astype(int), ev["eid"][sel]
    N, y, E = len(d["y"]), d["y"], d["E"]
    sha = P.sha_ints(d["sha"])
    em, es = E.mean(0), E.std(0) + 1e-9
    Etr, Ete = ((E - em) / es).astype(np.float32), ((Ee - em) / es).astype(np.float32)
    res = {"pop": pop, "run": True, "n": int(len(ye)), "n_go": int(ye.sum()), "n_episodes": int(len(np.unique(eid))), "arms": {}, "paired": {}, "seeds": list(cfg["seeds"])}
    sc = {}
    pred, hp, _ = L.fit_probe(E, y, d["inner"], "linear")
    sc["E"] = pred(Ee)
    for arm in ("G8att", "G4att"):
        grid = P.ARM_GRID[arm]
        hw, T = P.GRID_HW[grid], P.N_TOK[grid]
        P.require_ram(ram_min, f"eval141 {arm} {pop}")
        X = np.asarray(P.fold_view(B["S"][grid], 5), np.float32).reshape(N, T, P.K_TOK)
        mu, sd = P.tok_standardiser(X)
        X = (X - mu) / sd
        Xe = ((Se[grid].reshape(-1, T, P.K_TOK) - mu) / sd).astype(np.float32)
        logits = []
        for seed in cfg["seeds"]:
            val = P.inner_val_mask(sha, seed, 5, cfg["val_group_mod"])
            m, inf = P.train_one("att", hw, X[~val], Etr[~val], y[~val], X[val], Etr[val], y[val], seed * 100 + 5, cfg)
            logits.append(P.predict(m, Xe, Ete))
        sc[f"{arm}+E"] = np.mean(logits, 0)
        res["arms"][f"{arm}+E__per_seed_auc"] = [float(L.WAuc(l, ye).auc()) for l in logits]
    counts = L.boot_counts(eid, cfg["n_boot"], cfg["boot_seed"])
    for k, v in sc.items():
        res["arms"][k] = L.auc_ci(v, ye, eid, counts=counts)
    for a, b in (("G8att+E", "E"), ("G4att+E", "E"), ("G8att+E", "G4att+E")):
        res["paired"][f"{a}_minus_{b}"] = L.dauc_ci(sc[a], sc[b], ye, eid, counts=counts)
    _save_json(rp, res)
    np.savez(f"{out_raw}/eval141_4_scores_{pop}.npz", **sc)
    log(f"{pop} eval141: " + json.dumps({k: round(v["auc"], 4) for k, v in res["arms"].items() if isinstance(v, dict)}))
    return res


# ------------------------------------------------------------------------------------------------ stage: tables (every number of RESULTS.md is printed here or sits in VERDICT.json)
def seed_aucs(oof, arm, seeds, y, outer):
    return [P.fold_mean_auc(oof[f"{arm}+E__s{s}"], y, outer) for s in seeds]


def controls_status(B, cfg, out_raw):
    """-> ({'G8att': bool|None, 'G4att': ..., 'pair': ...}, detail)."""
    pop, d = B["pop"], B["d"]
    cp = f"{out_raw}/ctrl4_{pop}.npz"
    C = dict(np.load(cp)) if os.path.exists(cp) else {}
    counts = L.boot_counts(d["clip"], cfg["n_boot"], cfg["boot_seed"])
    y, clip, outer = d["y"], d["clip"], d["outer"]
    ys, yd = P.shuffle_within_folds(y, outer, 1), donor(d, 2)
    flags, det = {}, {}
    k = lambda name, arm: f"{name}:{arm}+E".replace(":", "__")
    for arm in CTRL_ARMS:
        c1 = c2 = None
        dd = {}
        if k("C1:shuffle", arm) in C:
            r = P.rec(C[k("C1:shuffle", arm)], ys, clip, outer, counts)["fold"]
            c1 = P.c1_verdict(r)
            dd["C1"] = {**r, "pass": c1}
        if k("C2:donor", arm) in C:
            r = P.rec(C[k("C2:donor", arm)], yd, clip, outer, counts)["fold"]
            aucs = [r["auc"]]
            for sd in (3, 4, 5, 6):
                kk = k(f"C2:donor{sd}", arm)
                if kk in C:
                    aucs.append(P.rec(C[kk], donor(d, sd), clip, outer, counts)["fold"]["auc"])
            c2, how = P.c2_verdict(r, aucs)
            dd["C2"] = {**r, "pass": c2, "how": how, "draw_aucs": aucs}
        flags[arm] = None if (c1 is None or c2 is None) else bool(c1 and c2)
        det[arm] = dd
    pr = None
    if all(k("C1:shuffle", a) in C for a in CTRL_ARMS):
        pr = P.paired(C[k("C1:shuffle", "G8att")], C[k("C1:shuffle", "G4att")], ys, clip, outer, counts)["fold"]
        flags["pair"] = bool(pr["lo"] <= 0.0 <= pr["hi"])
        det["pair"] = {"C1_paired_null": {**pr, "pass": flags["pair"]}}
    else:
        flags["pair"] = None
    return flags, det


def p3_consistency(pop, bk, d, p3raw=P.P3RAW):
    """Input check (PREREG 1): the banked probe-3 G4+E / G8+E scores recomputed here must reproduce probe 3's own fold-mean AUC (resg_<pop>_linear.json) to 1e-9 -> same rows, same order, same estimator."""
    path = f"{p3raw}/resg_{pop}_linear.json"
    if not os.path.exists(path) or "P3:G8+E" not in bk:
        return {"checked": False, "reason": "probe-3 resg json or banked scores absent"}
    res = json.load(open(path))["arms"]
    out = {"checked": True, "ok": True, "arms": {}}
    for arm in ("G4+E", "G8+E"):
        mine = P.fold_mean_auc(bk[f"P3:{arm}"], d["y"], d["outer"])
        theirs = res[arm]["fold"]["auc"]
        out["arms"][arm] = {"mine": mine, "probe3": theirs, "abs_diff": abs(mine - theirs)}
        out["ok"] = bool(out["ok"] and abs(mine - theirs) < 1e-9)
    return out


def stage_tables(Bs, cfg, out_raw, log=print, p3raw=P.P3RAW):
    lines = []

    def pr(*a):
        s = " ".join(str(x) for x in a)
        log(s)
        lines.append(s)

    f3 = lambda r: f"{r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
    fd = lambda r: f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    Pdec, detail = {}, {}
    for B in Bs:
        pop, d = B["pop"], B["d"]
        y, clip, outer = d["y"], d["clip"], d["outer"]
        counts = L.boot_counts(clip, cfg["n_boot"], cfg["boot_seed"])
        oofp = f"{out_raw}/oof4_{pop}.npz"
        oof = dict(np.load(oofp)) if os.path.exists(oofp) else {}
        bk = B["bk"]
        pr(f"\n=== {pop}  n={len(y)}  GO={int(y.sum())}  clips={len(np.unique(clip))}  (TRAIN, T0-class probe of representations; fold-mean AUC, episode-cluster bootstrap B={cfg['n_boot']} seed {cfg['boot_seed']}) ===")
        sc, spread, seedtab = {}, {}, {}
        for nm in ("E", "V+E", "V+E(mlp)", "P3:G4+E", "P3:G8+E"):
            if nm in bk:
                sc[nm] = bk[nm]
        for arm in P.ARMS:
            if f"{arm}+E" in oof:
                sc[f"{arm}+E"] = oof[f"{arm}+E"]
                sa = seed_aucs(oof, arm, cfg["seeds"], y, outer)
                seedtab[arm] = sa
                spread[arm] = P.seed_spread(sa)
        for nm, v in sc.items():
            r = P.rec(v, y, clip, outer, counts)
            extra = f"  seeds {[round(x, 4) for x in seedtab[nm[:-2]]]} spread {spread[nm[:-2]]:.4f}" if nm[:-2] in seedtab else ""
            pr(f"  {nm:10s} fold-mean {f3(r['fold'])}   pooled {f3(r['pooled'])}{extra}")
        cells = {"spread": spread}
        for tag, a, b in (("A", "G8att+E", "V+E"), ("B", "G8att+E", "G4att+E")):
            if a in sc and b in sc:
                cells[tag] = P.paired(sc[a], sc[b], y, clip, outer, counts)["fold"]
                pr(f"  DECISION {tag}: dAUC({a} - {b}) = {fd(cells[tag])}   (pooled beside: {fd(P.paired(sc[a], sc[b], y, clip, outer, counts)['pooled'])})")
        chk = p3_consistency(pop, bk, d, p3raw)
        pr(f"  input check (probe-3 banked G4+E / G8+E reproduce probe 3's own fold-mean AUC to 1e-9): {json.dumps(chk)}")
        cells['input_check'] = chk
        flags, cdet = controls_status(B, cfg, out_raw)
        cells["ctrl"] = flags
        pr("  controls:", json.dumps({k: v for k, v in flags.items()}))
        for arm, dd in cdet.items():
            for cn, v in dd.items():
                pr(f"    {arm} {cn}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk in ("auc", "lo", "hi", "delta", "pass", "how")))
        beside = {}
        for a, b in (("G8att+E", "E"), ("G4att+E", "E"), ("G8att+E", "P3:G8+E"), ("G4att+E", "P3:G4+E"), ("G8att+E", "V+E(mlp)"), ("G8max+E", "V+E"), ("G4max+E", "V+E"), ("G8max+E", "G4max+E"), ("G4att+E", "V+E")):
            if a in sc and b in sc:
                beside[f"{a}_minus_{b}"] = P.paired(sc[a], sc[b], y, clip, outer, counts)["fold"]
                pr(f"  beside dAUC({a} - {b}) = {fd(beside[f'{a}_minus_{b}'])}")
        if "G8att" in seedtab and "V+E" in sc:
            per_seed = [P.fold_mean_auc(oof[f"G8att+E__s{s}"], y, outer) - P.fold_mean_auc(sc["V+E"], y, outer) for s in cfg["seeds"]]
            pr(f"  beside single-seed dAUC(G8att+E s - V+E): {[round(x, 4) for x in per_seed]}   (the decision arm is the 3-seed LOGIT MEAN; this is the single-fit view)")
            beside["A_per_seed_delta"] = per_seed
        if "G8att" in seedtab and "G4att" in seedtab:
            beside["B_per_seed_delta"] = [P.fold_mean_auc(oof[f"G8att+E__s{s}"], y, outer) - P.fold_mean_auc(oof[f"G4att+E__s{s}"], y, outer) for s in cfg["seeds"]]
            pr(f"  beside single-seed dAUC(G8att+E s - G4att+E s): {[round(x, 4) for x in beside['B_per_seed_delta']]}")
        Pdec[pop] = cells
        detail[pop] = {"input_check": chk, "controls": cdet, "beside": beside, "seed_fold_aucs": seedtab, "seed_spread": spread}
    dec = P.decide(Pdec)
    pr("\n=== VERDICT (committed rule, PREREG_FROZEN.md 4) ===")
    pr(dec["verdict"])
    for k, v in dec["cells"].items():
        pr(f"  {k}: " + json.dumps(v))
    ev = {}
    for B in Bs:
        p = f"{out_raw}/results_eval141_4_{B['pop']}.json"
        if os.path.exists(p):
            ev[B["pop"]] = json.load(open(p))
            r = ev[B["pop"]]
            if r.get("run"):
                pr(f"\neval-141 {B['pop']} n={r['n']} go={r['n_go']} episodes={r['n_episodes']}: " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in r["arms"].items() if isinstance(v, dict)) + "  paired " +
                   ", ".join(f"{k} {v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]" for k, v in r["paired"].items()))
            else:
                pr(f"\neval-141 {B['pop']}: {r['reason']}")
    _save_json(f"{out_raw}/VERDICT.json", {"verdict": dec["verdict"], "decision": dec, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "detail": detail, "eval141": ev})
    open(f"{out_raw}/tables4.txt", "w", newline="\n").write("\n".join(lines) + "\n")
    return dec


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("arms", "controls", "post", "eval141", "tables", "check", "ramguard"))
    ap.add_argument("--pop", default="")
    ap.add_argument("--arms", default="", help="comma list of arms for --stage arms (default: all four, decision arms first)")
    a = ap.parse_args()
    cfg = dict(P.CFG)
    assert cfg == P.CFG
    os.makedirs(P.RAW, exist_ok=True)
    os.makedirs(P.SCR, exist_ok=True)
    assert P.sha256_file(f"{P.HERE}/tp_lib.py") == P.TP_LIB_SHA, "tp_lib.py differs from the byte-identical probe 1/3 copy"
    if a.stage == "ramguard":
        print("free MB", P.require_ram(4000, "chain start"), flush=True)
        return
    if a.stage == "check":
        print(json.dumps({"markers": P.p3_markers_present(), "free_mb": P.free_mb()}))
        return
    pops = [a.pop] if a.pop else list(P.POPS)
    if a.stage == "tables":
        P.require_ram(4000, "tables")
        stage_tables([P.real_data(p) for p in P.POPS], cfg, P.RAW)
        return
    for pop in pops:
        P.require_ram(4000, f"{a.stage} {pop}")
        B = P.real_data(pop)
        {"arms": lambda: stage_arms(B, cfg, P.RAW, P.SCR, arms=tuple(x for x in a.arms.split(",") if x) or P.ARMS), "controls": lambda: stage_controls(B, cfg, P.RAW, P.SCR), "post": lambda: stage_post(B, cfg, P.RAW, P.SCR),
         "eval141": lambda: stage_eval141(B, cfg, P.RAW, P.SCR)}[a.stage]()
    print("stage done", a.stage, pops, flush=True)


if __name__ == "__main__":
    main()
