"""Beside PREREG_V8F_LADDER, deciding nothing (Master Mind, 2026-11-09, written after reading BAND-MET on every arm, the control included).
Question: does beating Copy-Last in latent MSE mean more than regressing towards the average? A predictor that merely shrinks towards the mean beats Copy-Last in MSE with no knowledge of the future: at zero autocorrelation the best shrink reads skill 0.5.
Baselines, no world model, no actions, cross-fitted over two clip folds (pos % 2): SHRINK = m + c (copy - m), scalar c by least squares; LINEAR = ridge from the anchor latent to the future latent, lambda = 0.1 x trace / D, fixed before running.
Reads, at every level's band end, both contexts: skill of each baseline against Copy-Last, and skill of the world model against each baseline, by the same two-part rule. Tier T0. Writes DIAG_SHRINK.json and DIAG_SHRINK.md."""
import json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import numpy as np
import torch
import lr_lib as L
import run_ladder_read as RL

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
BAND = {1: 2.0, 2: 6.0, 3: 12.0}


def fit_shrink(copy, ora):
    m = copy.mean(0)
    x, y = copy - m, ora - m
    return m, float((x * y).sum() / (x * x).sum())


def fit_linear(copy, ora):
    mx, my = copy.mean(0), ora.mean(0)
    x, y = (copy - mx).astype(np.float64), (ora - my).astype(np.float64)
    G = x.T @ x
    lam = 0.1 * np.trace(G) / G.shape[0]
    A = np.linalg.solve(G + lam * np.eye(G.shape[0]), x.T @ y)
    return mx, my, A


def cross_fit(copy, ora, pos):
    f = pos % 2
    s, l, cs = np.empty_like(copy), np.empty_like(copy), []
    for a in (0, 1):
        tr, te = f != a, f == a
        m, c = fit_shrink(copy[tr], ora[tr])
        s[te] = m + c * (copy[te] - m)
        cs.append(c)
        mx, my, A = fit_linear(copy[tr], ora[tr])
        l[te] = my + (copy[te] - mx).astype(np.float64) @ A
    return s, l, cs


def rk(x):
    return {k: x[k] for k in ("point", "lo", "hi", "above", "below", "thr_hi") if k in x}


def one(name):
    t0 = time.time()
    L.memory_guard()
    model, cfg, argv, ck = L.load_arm(f"{L.PULLS}/ladder_{name}")
    L.check_built_vs_argv(L.built_values(model), argv)
    fr = RL.frames_for(model, RL.seed_of(name))
    cells, states, cum, plan, _ = L.gather(model, fr)
    W = L.IL.boot_weights(np.arange(len(fr["clips"])))
    out = {}
    for lv in (1, 2, 3):
        k = [q for h, q, _, _ in plan[lv] if abs(h - BAND[lv]) < 1e-9][0]
        for tag in ("built", "ctx1"):
            c = cells[(lv, k, tag)]
            sel = c["ok"]
            pred, copy, ora, pos = c["pred"][sel], c["copy"][sel], c["ora"][sel], c["pos"][sel]
            s, l, cs = cross_fit(copy, ora, pos)
            ctl = L.skill_mom(copy, copy, ora, pos, W.shape[1]).row(W)
            assert ctl["point"] == 0.0, "control: Copy-Last against itself"
            rec = dict(k=k, n=int(len(pos)), shrink_c=cs,
                       pred_vs_copy=rk(L.skill_stats(pred, copy, ora, pos, W, (lv, k))),
                       shrink_vs_copy=rk(L.skill_stats(s, copy, ora, pos, W, (lv, k))),
                       linear_vs_copy=rk(L.skill_stats(l, copy, ora, pos, W, (lv, k))),
                       pred_vs_shrink=rk(L.skill_stats(pred, s, ora, pos, W, (lv, k))),
                       pred_vs_linear=rk(L.skill_stats(pred, l, ora, pos, W, (lv, k))))
            out[f"L{lv}|{tag}"] = rec
    print(name, f"{time.time() - t0:.0f}s", flush=True)
    return out


def fmt(x):
    return f"{x['point']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" + (" above" if x.get("above") else (" below" if x.get("below") else ""))


if __name__ == "__main__":
    arms = sys.argv[1:]
    p = f"{HERE}/DIAG_SHRINK.json"
    res = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    for a in arms:
        if a not in res:
            res[a] = one(a)
            json.dump(res, open(p, "w", encoding="utf-8"), indent=1)
    md = ["# DIAG_SHRINK: is beating Copy-Last more than regressing to the average?", "Beside `PREREG_V8F_LADDER.md`, written after reading BAND-MET on every arm; **decides nothing**. Tier **T0**, a world-model diagnostic, never a driving result. Baselines use no world model and no action, cross-fitted over two clip folds. `skill X vs Y` = 1 - sum|X - oracle|^2 / sum|Y - oracle|^2. Interval: episode-cluster bootstrap, B 2,000; `above` / `below` adds the same 199-draw derangement reference as the deciding row.", ""]
    for tag, title in (("built", "built context (3 states; the deciding form)"), ("ctx1", "planning form (context 1)")):
        md += [f"## {title}", "| run | level | n | SHRINK vs Copy-Last | LINEAR vs Copy-Last | world model vs Copy-Last | world model vs SHRINK | world model vs LINEAR |", "|---|---|---|---|---|---|---|---|"]
        for a in arms:
            for lv in (1, 2, 3):
                r = res[a][f"L{lv}|{tag}"]
                md.append(f"| {a} | L{lv} ({BAND[lv]:g} s) | {r['n']} | {fmt(r['shrink_vs_copy'])} | {fmt(r['linear_vs_copy'])} | {fmt(r['pred_vs_copy'])} | {fmt(r['pred_vs_shrink'])} | {fmt(r['pred_vs_linear'])} |")
        md.append("")
    open(f"{HERE}/DIAG_SHRINK.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))
    print("done")
