"""smoke_box2.py -- card ADD148 (LEADSPEED10K) with amendment 1: controls s1 (clock) and s5 (500-step real-data smoke) ON box 2. Tier T0; never driving performance.

  python smoke_box2.py --out D:/tanitad/leadspeed/smoke [--steps 500]

Reads D:/tanitad/cache_full/train (manifest, clips) and D:/tanitad/data/v9/lead_train4572.npz; checks the eval-141 clips (D:/tanitad/cache/eval/manifest.jsonl) are absent.
Writes SMOKE.json and smoke.log in --out. Prints QUEUE-COMPLETE at the end in every case. No model weight is saved beyond the smoke head (head_smoke.pt, small).
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leadspeed as L  # noqa: E402

CORPUS = "D:/tanitad/cache_full/train"
EVALM = "D:/tanitad/cache/eval/manifest.jsonl"
LEADF = "D:/tanitad/data/v9/lead_train4572.npz"
SEED, N_FIT, N_HO, N_CLOCK = 148, 400, 50, 200


def jl_lines(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--evalm", default=EVALM)
    ap.add_argument("--leadf", default=LEADF)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"card": "PREREG_V8F_PROOF_ADD148", "amendment": "AM1", "tier": "T0 (a head on cached tokens of logged frames); never driving performance", "controls": {}}
    try:
        import torch
        man = jl_lines(os.path.join(a.corpus, "manifest.jsonl"))
        ev = {r["clip"] for r in jl_lines(a.evalm)}
        res["eval_clips_in_corpus"] = int(sum(r["clip"] in ev for r in man))
        z = np.load(a.leadf, allow_pickle=True)
        sha, k, has, lead, gap, vlead = z["sha12"], z["k"], z["has"], z["lead"], z["gap"], z["v"]
        order = np.argsort(sha, kind="stable")
        sha_s = sha[order]
        uniq, start = np.unique(sha_s, return_index=True)
        rows = {u: order[s:e] for u, s, e in zip(uniq, start, list(start[1:]) + [len(sha_s)])}
        cand = [r for r in man if r["sha12"] in rows]
        rng = np.random.default_rng(SEED)
        pick = rng.permutation(len(cand))
        n_fit_all = int(round(0.9 * len(cand)))
        fit_all, ho_all = [cand[i] for i in pick[:n_fit_all]], [cand[i] for i in pick[n_fit_all:]]
        fit, ho = fit_all[:N_FIT], ho_all[:N_HO]
        res["split"] = {"corpus_clips": len(man), "with_labels": len(cand), "fit_all": len(fit_all), "held_out_all": len(ho_all), "smoke_fit": len(fit), "smoke_held_out": len(ho)}
        t0 = time.time()
        nbytes = 0
        data = []
        for r in fit + ho:
            p = os.path.join(a.corpus, r["file"])
            nbytes += os.path.getsize(p)
            d = torch.load(p, map_location="cpu", weights_only=False)
            tok = d["pool4x10"].view(torch.uint8).clone()            # keep fp8 bytes; cast per batch
            T = tok.shape[0]
            ix = rows[r["sha12"]]
            pres, g, clos = L.token_labels(k[ix], has[ix], lead[ix], gap[ix], T)
            data.append({"tok": tok, "v": d["v"].numpy().astype(np.float64), "pres": pres, "gap": g, "clos": clos, "k": k[ix], "vlead": vlead[ix]})
            del d
        dt_load = time.time() - t0
        res["load"] = {"clips": len(data), "seconds": round(dt_load, 1), "MB": round(nbytes / 1e6, 1), "MB_per_s": round(nbytes / 1e6 / max(dt_load, 1e-9), 1)}
        # ---- s1 clock
        per = {s: [] for s in (-2, -1, 0, 1, 2)}
        best0 = 0
        for dd in data[:N_CLOCK]:
            r = L.clock_check(dd["k"], dd["vlead"], dd["v"])
            for s, (m, n) in r.items():
                if n:
                    per[s].append(m)
            best0 += int(min(r, key=lambda s: r[s][0]) == 0)
        mae = {s: float(np.mean(v)) if v else float("inf") for s, v in per.items()}
        dd = data[0]
        rp = L.clock_check(dd["k"], dd["vlead"], np.concatenate([dd["v"][1:], [np.nan]]))
        planted_best = min(rp, key=lambda s: rp[s][0])
        res["controls"]["s1_clock"] = {"mae_by_shift": mae, "clips_best_at_0": best0, "clips": min(N_CLOCK, len(data)), "planted_shift_best": planted_best,
                                       "PASS": bool(min(mae, key=mae.get) == 0 and planted_best == 1)}
        # ---- s5 smoke
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(SEED)
        head = L.make_head(d_in=1024, width=64, temporal=True).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        fit_d, ho_d = data[:len(fit)], data[len(fit):]
        idx = [(ci, t) for ci, dd in enumerate(fit_d) for t in L.windows(dd["tok"].shape[0])]
        res["windows_fit"] = len(idx)
        res["share_windows_with_closing_label"] = round(float(np.mean([np.isfinite(fit_d[c]["clos"][t]) for c, t in idx])), 4)
        gen = np.random.default_rng(SEED)

        def batch(dset, pairs):
            x = torch.stack([dset[c]["tok"][t - L.H + 1:t + 1] for c, t in pairs]).view(torch.float8_e4m3fn).float()
            p = torch.tensor([float(dset[c]["pres"][t]) for c, t in pairs])
            g = torch.tensor([dset[c]["gap"][t] for c, t in pairs], dtype=torch.float32)
            c_ = torch.tensor([dset[c]["clos"][t] for c, t in pairs], dtype=torch.float32)
            return x.to(dev), p.to(dev), g.to(dev), c_.to(dev)

        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses, times = [], []
        for step in range(a.steps):
            ts = time.time()
            pairs = [idx[i] for i in gen.integers(0, len(idx), 64)]
            x, p, g, c_ = batch(fit_d, pairs)
            t = L.loss_terms(head(x), p, g, c_)
            loss = t["pres"] + t["gap"] + t["clos"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - ts)
            losses.append(float(loss.item()))
        fin = bool(np.isfinite(losses).all())
        first, last = float(np.mean(losses[:100])), float(np.mean(losses[-100:]))
        res["smoke"] = {"device": dev, "steps": a.steps, "batch": 64, "loss_first100": round(first, 4), "loss_last100": round(last, 4), "finite": fin,
                        "step_s_median": round(float(np.median(times)), 4), "step_s_mean": round(float(np.mean(times)), 4),
                        "peak_vram_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1) if dev == "cuda" else None}
        res["controls"]["s5_smoke"] = {"PASS": bool(fin and last < first)}
        head.eval()
        ys, ps = [], []
        with torch.no_grad():
            pairs = [(c, t) for c, dd in enumerate(ho_d) for t in L.windows(dd["tok"].shape[0]) if np.isfinite(dd["clos"][t])]
            for i in range(0, len(pairs), 256):
                x, _, _, c_ = batch(ho_d, pairs[i:i + 256])
                ps.append(head(x)[:, 2].float().cpu().numpy() * L.CLOS_SCALE)
                ys.append(c_.cpu().numpy())
        y, pr = (np.concatenate(ys), np.concatenate(ps)) if ys else (np.zeros(0), np.zeros(0))
        res["beside_held_out_closing_after_smoke"] = {"n": int(len(y)), "pearson_r": round(L.pearson(y, pr), 4) if len(y) > 2 else None, "names_nothing": True}
        torch.save(head.state_dict(), os.path.join(a.out, "head_smoke.pt"))
        res["controls"]["eval_absent"] = {"eval_clips_in_corpus": res["eval_clips_in_corpus"], "PASS": res["eval_clips_in_corpus"] == 0}
        res["all_pass_s1_s5_here"] = all(v["PASS"] for v in res["controls"].values())
    except Exception as ex:                                                                     # noqa: BLE001
        res["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:500])
        res["traceback"] = traceback.format_exc()[-2000:]
        res["all_pass_s1_s5_here"] = False
    json.dump(res, open(os.path.join(a.out, "SMOKE.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1), flush=True)
    print("QUEUE-COMPLETE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
