"""train_full_box2.py -- card PREREG_V8F_PROOF_ADD149 (LEADSPEEDFULL10K): the full run of the lead-speed head of card ADD148 on box 2, and its reads. Tier T0; never
driving performance. Prints QUEUE-COMPLETE at the end in every case (the chain's end marker).

  python train_full_box2.py --out D:/tanitad/leadspeed/full [--steps 30000 --n-fit 1500]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leadspeed as L  # noqa: E402
import tokstore  # noqa: E402

SEED = 148
MB_PER_CLIP = 0.1                           # amendment 3: tokens as read-only memmaps, 0.008 MB commit per clip measured on box 2 (box2_full/MEMPROBE4.out); v and labels small
SHAPE = (101, 40, 1024)
GUARD_GB, PROC_GB = 5.5, 2.5


def jl_lines(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def commit_avail_gb():
    try:
        import memstat
        return float(memstat.status()["commit_avail_GB"])
    except Exception:                                                                           # noqa: BLE001
        return float("nan")


def own_commit_gb():
    try:
        import memstat
        return float(memstat.own_commit_gb())
    except Exception:                                                                           # noqa: BLE001
        return 0.0                                                                              # unknown: the full 2.5 GB is budgeted


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(np.float64) + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clip_pairs(d, key="clos_head"):
    """One clip -> (closing label, closing head, acceleration label, acceleration head) arrays, built exactly as read_quantities builds S and A."""
    v, cl, ch = d["v"], d["clos_lab"], d[key]
    cs_l, cs_h, a_l, a_h = [], [], [], []
    for f in range(len(v)):
        if np.isfinite(cl[f]) and np.isfinite(ch[f]):
            cs_l.append(cl[f]); cs_h.append(ch[f])
        g = f - 5
        if g >= 0 and all(np.isfinite(x) for x in (cl[f], ch[f], cl[g], ch[g], v[f], v[g])):
            a_l.append(max(0.0, v[f] + cl[f]) - max(0.0, v[g] + cl[g]))
            a_h.append(max(0.0, v[f] + ch[f]) - max(0.0, v[g] + ch[g]))
    return [np.asarray(x, np.float64) for x in (cs_l, cs_h, a_l, a_h)]


def paired_bootstrap(per_clip, n_boot=2000, seed=150):
    """Card ADD150 section 2 (beside): S and A of the head ("clos_head") minus those of the reference head ("clos_ref") on the same clips and frames;
    clip-cluster bootstrap (clips resampled with replacement). Answers "another draw of clips", not "another training run"."""
    own = [clip_pairs(d, "clos_head") for d in per_clip]
    ref = [clip_pairs(d, "clos_ref") for d in per_clip]

    def sa(P, ix):
        c = [np.concatenate([P[i][j] for i in ix]) for j in range(4)]
        return L.pearson(c[0], c[1]), L.pearson(c[2], c[3])

    allix = np.arange(len(per_clip))
    so, ao = sa(own, allix)
    sr, ar = sa(ref, allix)
    rng = np.random.default_rng(seed)
    dS, dA = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, len(per_clip), len(per_clip))
        a1, b1 = sa(own, ix)
        a2, b2 = sa(ref, ix)
        dS.append(a1 - a2); dA.append(b1 - b2)
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    return {"estimator": "clip-cluster bootstrap, paired, %d resamples, seed %d; episodes (clips) only, one training seed per arm" % (n_boot, seed),
            "S_head": so, "S_ref": sr, "dS": so - sr, "dS_ci95": q(dS), "A_head": ao, "A_ref": ar, "dA": ao - ar, "dA_ci95": q(dA), "n_clips": len(per_clip)}


def read_quantities(per_clip):
    """per_clip: list of dict(v [T], clos_lab [T], clos_head [T], pres_lab [T], pres_head [T], gap_lab [T], gap_head [T]) with NaN where no window ends.
    -> S, A (Pearson), n_S, n_A, gap MAE on lead frames, presence AUC."""
    cs_l, cs_h, a_l, a_h, gl, gh, pl, ph = [], [], [], [], [], [], [], []
    for d in per_clip:
        v, cl, ch = d["v"], d["clos_lab"], d["clos_head"]
        T = len(v)
        for f in range(T):
            if np.isfinite(cl[f]) and np.isfinite(ch[f]):
                cs_l.append(cl[f]); cs_h.append(ch[f])
            g = f - 5
            if g >= 0 and all(np.isfinite(x) for x in (cl[f], ch[f], cl[g], ch[g], v[f], v[g])):
                a_l.append((max(0.0, v[f] + cl[f]) - max(0.0, v[g] + cl[g])) / 1.0)
                a_h.append((max(0.0, v[f] + ch[f]) - max(0.0, v[g] + ch[g])) / 1.0)
            if np.isfinite(d["pres_head"][f]):
                pl.append(bool(d["pres_lab"][f])); ph.append(d["pres_head"][f])
                if d["pres_lab"][f] and np.isfinite(d["gap_lab"][f]):
                    gl.append(d["gap_lab"][f]); gh.append(d["gap_head"][f])
    return {"S": L.pearson(cs_l, cs_h), "n_S": len(cs_l), "A": L.pearson(a_l, a_h), "n_A": len(a_l),
            "gap_mae_m": float(np.mean(np.abs(np.array(gl) - np.array(gh)))) if gl else float("nan"), "presence_auc": auc(pl, ph)}


def name_of(A, S):
    return "LEADSPEEDFULL10K-CARRIED" if A >= 0.30 else ("LEADSPEEDFULL10K-SPEED-ONLY" if S >= 0.30 else "LEADSPEEDFULL10K-NOT-CARRIED")


def consequence(name, Ae, Se):
    """Card ADD149 section 4, as committed."""
    if name == "LEADSPEEDFULL10K-CARRIED" and Ae >= 0.30:
        return "WITH-FORECAST: the head is the closing input of a reader-form rule arm with the forecast of card ADD144 (successor card)"
    if name == "LEADSPEEDFULL10K-SPEED-ONLY" and Se >= 0.30:
        return "CLOSING-ONLY: the head is the closing input without the acceleration (successor card)"
    return "BLOCKER-NAMED: no successor arm from this read; the blocker is named with the measured values (more fit clips needing a disk for the token store, or a second seed)"


def head_outputs(torch, head, tok, dev):
    """tok uint8 [T, 40, 1024] (fp8 bytes) -> pres prob, gap (m), closing (m/s) at every window end t >= H-1 (NaN before)."""
    T = tok.shape[0]
    ends = L.windows(T)
    P, G, C = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    with torch.no_grad():
        for i in range(0, len(ends), 128):
            e = ends[i:i + 128]
            x = torch.from_numpy(np.stack([np.asarray(tok[t - L.H + 1:t + 1]) for t in e])).view(torch.float8_e4m3fn).float().to(dev)
            o = head(x).float().cpu().numpy()
            P[e] = 1.0 / (1.0 + np.exp(-o[:, 0]))
            G[e] = o[:, 1] * L.GAP_SCALE
            C[e] = o[:, 2] * L.CLOS_SCALE
    return P, G, C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--n-fit", type=int, default=1500)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--corpus", default="D:/tanitad/cache_full/train")
    ap.add_argument("--leadf", default="D:/tanitad/data/v9/lead_train4572.npz")
    ap.add_argument("--evalc", default="D:/tanitad/cache/eval")
    ap.add_argument("--join", default="D:/tanitad/rhacc10k/pins/join_eval141.npz")
    ap.add_argument("--no-guard", action="store_true", help="tests only")
    ap.add_argument("--train-seed", type=int, default=SEED, help="cards ADD150 AM1 / ADD151: the training seed (init and batches); the clip split stays on seed 148")
    ap.add_argument("--ref-ckpt", default=None, help="card ADD150: a reference head read beside on the held-out clips")
    ap.add_argument("--ref-sha", default=None, help="sha256 prefix the reference checkpoint must have before it is read")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"card": "PREREG_V8F_PROOF_ADD149", "tier": "T0 (a head on cached tokens of logged frames); never driving performance", "controls": {},
           "leadspeed_py_sha256": sha_file(os.path.join(HERE, "leadspeed.py"))}
    try:
        import torch
        man = jl_lines(os.path.join(a.corpus, "manifest.jsonl"))
        evm = jl_lines(os.path.join(a.evalc, "manifest.jsonl"))
        z = np.load(a.leadf, allow_pickle=True)
        sha, k, has, lead, gap = z["sha12"], z["k"], z["has"], z["lead"], z["gap"]
        order = np.argsort(sha, kind="stable")
        sha_s = sha[order]
        uniq, start = np.unique(sha_s, return_index=True)
        rows = {u: order[s:e] for u, s, e in zip(uniq, start, list(start[1:]) + [len(sha_s)])}
        cand = [r for r in man if r["sha12"] in rows]
        pick = np.random.default_rng(SEED).permutation(len(cand))
        n_fit_all = int(round(0.9 * len(cand)))
        fit_all, ho = [cand[i] for i in pick[:n_fit_all]], [cand[i] for i in pick[n_fit_all:]]
        fit = fit_all[:a.n_fit]
        fit_ids, ho_ids, ev_ids = {r["clip"] for r in fit}, {r["clip"] for r in ho}, {r["clip"] for r in evm}
        res["seeds"] = {"split": SEED, "train": a.train_seed}
        res["split"] = {"with_labels": len(cand), "fit_all": len(fit_all), "fit_used": len(fit), "held_out": len(ho), "eval": len(evm)}
        res["controls"]["c2_disjoint"] = {"fit_and_held_out": len(fit_ids & ho_ids), "fit_and_eval": len(fit_ids & ev_ids), "PASS": not (fit_ids & ho_ids) and not (fit_ids & ev_ids)}
        need = len(fit) * MB_PER_CLIP / 1000.0                                               # amendment 2: only the fit clips stay in memory
        ca0, own0 = commit_avail_gb(), own_commit_gb()
        need0 = GUARD_GB + need + max(0.0, PROC_GB - own0)                                   # amendment 1: the process counted once
        guard = {"amendment": "AM1+AM2+AM3", "commit_avail_at_start_GB": round(ca0, 2), "own_commit_at_start_GB": round(own0, 3), "need_GB": round(need0, 2), "violations": 0}
        res["guard"] = guard
        if not a.no_guard and not (ca0 >= need0):
            res["name"] = "LEADSPEEDFULL10K-STOPPED-MEMORY"
            raise RuntimeError("memory guard at start: commit available %.2f GB < %.2f GB" % (ca0, need0))

        touch_sum = [0]

        def load(recs, labelled=True, after=0, touch=False):
            out = []
            for i, r in enumerate(recs):
                if not a.no_guard:
                    left = (len(recs) - i + after) * MB_PER_CLIP / 1000.0
                    ca = commit_avail_gb()
                    if not (ca >= GUARD_GB + left):
                        guard["violations"] += 1
                        guard["stop_at"] = {"clip_index": i, "of": len(recs), "commit_avail_GB": round(ca, 2), "need_GB": round(GUARD_GB + left, 2), "own_commit_GB": round(own_commit_gb(), 3)}
                        res["name"] = "LEADSPEEDFULL10K-STOPPED-MEMORY"
                        raise RuntimeError("memory guard during load: commit available %.2f GB < %.2f GB" % (ca, GUARD_GB + left))
                pth = os.path.join(a.corpus if labelled else a.evalc, r["file"])
                d = torch.load(pth, map_location="cpu", weights_only=False, mmap=True)
                v = d["v"].numpy().astype(np.float64)                                         # a copy; the mapping is released below
                del d
                tok = tokstore.open_tokens(pth, r.get("shapes", {}).get("pool4x10", SHAPE))     # amendment 3: read-only memmap, no commit
                if touch:
                    touch_sum[0] += int(tok.sum(dtype=np.uint64) % 997)                       # reads every page once (page cache)
                out.append({"clip": r["clip"], "tok": tok, "v": v})
            return out

        t0 = time.time()
        fit_d = load(fit, touch=True)
        for dd, r in zip(fit_d, fit):
            ix = rows[r["sha12"]]
            dd["pres"], dd["gap"], dd["clos"] = L.token_labels(k[ix], has[ix], lead[ix], gap[ix], dd["tok"].shape[0])
        def c5_check(recs, root, n):
            eq, tot = 0, 0
            for r in recs[:n]:
                pth = os.path.join(root, r["file"])
                d = torch.load(pth, map_location="cpu", weights_only=False, mmap=True)
                eq += int(np.array_equal(np.asarray(tokstore.open_tokens(pth, r.get("shapes", {}).get("pool4x10", SHAPE))), d["pool4x10"].view(torch.uint8).numpy()))
                tot += 1
                del d
            return eq, tot

        e1, n1_ = c5_check(fit, a.corpus, 50)
        e2, n2_ = c5_check(ho, a.corpus, 20)
        e3, n3_ = c5_check(evm, a.evalc, 20)
        p0 = os.path.join(a.corpus, fit[0]["file"])
        sh0 = tuple(fit[0].get("shapes", {}).get("pool4x10", SHAPE))
        off0, _ = tokstore.member(p0, sh0)
        d0 = torch.load(p0, map_location="cpu", weights_only=False, mmap=True)
        wrong = np.memmap(p0, dtype=np.uint8, mode="r", offset=max(0, off0 - 64), shape=sh0)
        planted_differs = not np.array_equal(np.asarray(wrong), d0["pool4x10"].view(torch.uint8).numpy())
        del d0, wrong
        res["controls"]["c5_token_bytes"] = {"fit_equal": [e1, n1_], "held_out_equal": [e2, n2_], "eval_equal": [e3, n3_], "planted_offset_differs": bool(planted_differs),
                                             "PASS": bool(e1 == n1_ and e2 == n2_ and e3 == n3_ and n1_ > 0 and planted_differs)}
        res["load"] = {"tokens": "read-only memmap (amendment 3)", "touch_check": touch_sum[0] % 997, "fit_clips": len(fit_d), "seconds": round(time.time() - t0, 1), "commit_avail_after_GB": round(commit_avail_gb(), 2), "own_commit_after_GB": round(own_commit_gb(), 3)}
        # ---- training
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(a.train_seed)                                                      # the split stays on SEED (148)
        head = L.make_head(d_in=1024, width=64, temporal=True).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        idx = [(ci, t) for ci, dd in enumerate(fit_d) for t in L.windows(dd["tok"].shape[0])]
        gen = np.random.default_rng(a.train_seed)

        def batch(pairs):
            x = torch.from_numpy(np.stack([np.asarray(fit_d[c]["tok"][t - L.H + 1:t + 1]) for c, t in pairs])).view(torch.float8_e4m3fn).float()
            p = torch.tensor([float(fit_d[c]["pres"][t]) for c, t in pairs])
            g = torch.tensor([fit_d[c]["gap"][t] for c, t in pairs], dtype=torch.float32)
            c_ = torch.tensor([fit_d[c]["clos"][t] for c, t in pairs], dtype=torch.float32)
            return x.to(dev), p.to(dev), g.to(dev), c_.to(dev)

        losses, ckpts, t1 = [], {}, time.time()
        for step in range(1, a.steps + 1):
            x, p, g, c_ = batch([idx[i] for i in gen.integers(0, len(idx), 64)])
            t = L.loss_terms(head(x), p, g, c_)
            loss = t["pres"] + t["gap"] + t["clos"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            if step % a.ckpt_every == 0 or step == a.steps:
                pth = os.path.join(a.out, "head_%d.pt" % step)
                torch.save(head.state_dict(), pth)
                ckpts[step] = sha_file(pth)
                print("step", step, "loss100", round(float(np.mean(losses[-100:])), 4), "s", round(time.time() - t1, 1), flush=True)
        n1 = min(1000, len(losses) // 2)
        fin = bool(np.isfinite(losses).all())
        res["train"] = {"steps": len(losses), "windows_fit": len(idx), "seconds": round(time.time() - t1, 1), "loss_first1000": round(float(np.mean(losses[:n1])), 4),
                        "loss_last1000": round(float(np.mean(losses[-n1:])), 4), "finite": fin, "ckpt_sha256": {str(s): h for s, h in ckpts.items()}}
        res["controls"]["c1_run"] = {"guard_violations": guard["violations"], "steps_done": len(losses), "PASS": bool(fin and len(losses) == a.steps and np.mean(losses[-n1:]) < np.mean(losses[:n1]))}
        # ---- the checkpoint of record, reloaded
        rec = os.path.join(a.out, "head_%d.pt" % a.steps)
        h2 = sha_file(rec)
        head.load_state_dict(torch.load(rec, map_location=dev))
        head.eval()
        res["controls"]["c4_checkpoint"] = {"sha256_at_save": ckpts.get(a.steps), "sha256_at_read": h2, "PASS": ckpts.get(a.steps) == h2}
        del fit_d

        ref_head = None
        if a.ref_ckpt:
            hr = sha_file(a.ref_ckpt)
            if not (a.ref_sha and hr.startswith(a.ref_sha)):
                raise RuntimeError("reference checkpoint sha256 %s does not start with %s" % (hr[:16], a.ref_sha))
            ref_head = L.make_head(d_in=1024, width=64, temporal=True).to(dev)
            ref_head.load_state_dict(torch.load(a.ref_ckpt, map_location=dev))
            ref_head.eval()
            res["ref"] = {"ckpt": a.ref_ckpt, "sha256": hr}

        def stream(recs, labelled, labeller, use_ref=False):
            """Amendment 2: read clips are loaded one at a time at read time (the guard applies before each), labelled, read, and released."""
            out = []
            for r in recs:
                dd = load([r], labelled=labelled)[0]
                dd["pres"], dd["gap"], dd["clos"] = labeller(r, dd)
                P, G, C = head_outputs(torch, head, dd["tok"], dev)
                C2 = head_outputs(torch, ref_head, dd["tok"], dev)[2] if (use_ref and ref_head is not None) else None
                out.append({"v": dd["v"], "clos_lab": dd["clos"], "clos_head": C, "pres_lab": dd["pres"], "pres_head": P, "gap_lab": dd["gap"], "gap_head": G})
                if C2 is not None:
                    out[-1]["clos_ref"] = C2
                del dd
            return out

        def lab_corpus(r, dd):
            ix = rows[r["sha12"]]
            return L.token_labels(k[ix], has[ix], lead[ix], gap[ix], dd["tok"].shape[0])

        pc_ho = stream(ho, True, lab_corpus, use_ref=True)
        res["held_out"] = read_quantities(pc_ho)
        if ref_head is not None:
            res["paired_vs_ref_held_out"] = paired_bootstrap(pc_ho)
        res["held_out_halves"] = [read_quantities(pc_ho[0::2]), read_quantities(pc_ho[1::2])]
        planted = [dict(d, clos_head=d["clos_lab"]) for d in pc_ho]
        pq = read_quantities(planted)
        res["controls"]["c3_planted_read"] = {"S": pq["S"], "A": pq["A"], "PASS": bool(abs(pq["S"] - 1.0) < 1e-9 and abs(pq["A"] - 1.0) < 1e-9)}
        # ---- eval-141 (for the consequence)
        J = np.load(a.join, allow_pickle=True)
        jclip = [str(c) for c in J["clip"]]
        def lab_eval(r, dd):
            ci = jclip.index(dd["clip"])
            o, n = int(J["off"][ci]), int(J["n"][ci])
            lv = (np.asarray(J["lead_valid"][o:o + n]) == 1).astype(int)
            hv = np.asarray(J["has"][o:o + n]).astype(int)
            return L.token_labels(2 * np.arange(n), hv, lv, np.asarray(J["lead_gap"][o:o + n], np.float64), dd["tok"].shape[0])

        res["eval141"] = read_quantities(stream(evm, False, lab_eval))
        res["token_reader_ADD146_beside"] = {"A": [0.0872, 0.0714], "S": [0.3337, 0.2731]}
        ok = all(v["PASS"] for v in res["controls"].values())
        res["controls_all_pass"] = ok
        if ok:
            A, S = res["held_out"]["A"], res["held_out"]["S"]
            res["name"] = name_of(A, S)
            res["consequence"] = consequence(res["name"], res["eval141"]["A"], res["eval141"]["S"])
    except Exception as ex:                                                                     # noqa: BLE001
        res["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:500])
        res["traceback"] = traceback.format_exc()[-2000:]
        res.setdefault("name", None)
    json.dump(res, open(os.path.join(a.out, "FULL.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1, default=float), flush=True)
    print("QUEUE-COMPLETE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
