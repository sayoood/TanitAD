"""train_acc_box2.py -- card PREREG_V8F_PROOF_ADD153 (ACCHIST10K): one arm of the acceleration head on box 2, and its reads. Tier T0; never driving performance.
The data, split, guards and controls c1 to c5 are those of train_full_box2.py (card ADD149 with amendments 1 to 3), imported, not changed.
Prints QUEUE-COMPLETE at the end in every case (the chain's end marker) unless --no-marker.

  python -I train_acc_box2.py --arm K25D --out D:/tanitad/leadacc/K25D [--train-seed 148 --label-run 5 --steps 30000 --n-fit 3984]
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "leadspeed"), HERE):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
import leadacc as A  # noqa: E402
import leadspeed as L  # noqa: E402
import tokstore  # noqa: E402
import train_full_box2 as F  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD153"


def to_x(torch, u8, dev):
    """uint8 [B, H, N, D] (fp8 bytes) -> float32 on dev; the cast runs on the device."""
    return torch.from_numpy(np.ascontiguousarray(u8)).to(dev).view(torch.float8_e4m3fn).float()


def head_outputs(torch, head, tok, dev, H, ends=None, bs=64):
    """-> presence prob, gap (m), closing (m/s), d (m/s) at every window end t >= 9 (NaN before)."""
    T = tok.shape[0]
    ends = A.window_ends(T) if ends is None else ends
    P, G, C, D = (np.full(T, np.nan) for _ in range(4))
    with torch.no_grad():
        for i in range(0, len(ends), bs):
            e = ends[i:i + bs]
            ws = [A.window(tok, int(t), H) for t in e]
            x = to_x(torch, np.stack([w for w, _ in ws]), dev)
            v = torch.from_numpy(np.stack([m for _, m in ws])).to(dev)
            o = head(x, v).float().cpu().numpy()
            P[e] = 1.0 / (1.0 + np.exp(-o[:, 0]))
            G[e] = o[:, 1] * L.GAP_SCALE
            C[e] = o[:, 2] * L.CLOS_SCALE
            D[e] = o[:, 3] * A.D_SCALE
    return P, G, C, D


def ref_outputs(torch, ref, tok, dev):
    """row 217's head (3 outputs, H 10, windows t >= 9): closing (m/s) and its derived d = c(f) - c(f - 5)."""
    C = F.head_outputs(torch, ref, tok, dev)[2]
    return C, A.d_labels(C)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(A.ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--n-fit", type=int, default=3984)
    ap.add_argument("--n-ho", type=int, default=None, help="smoke only: the first n held-out clips")
    ap.add_argument("--n-eval", type=int, default=None, help="smoke only: the first n eval clips")
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--train-seed", type=int, default=148)
    ap.add_argument("--label-run", type=int, default=5, choices=(5, 10), help="card section 2: 10 = the smoother d label of the stage-2 arm (training only)")
    ap.add_argument("--corpus", default="D:/tanitad/cache_full/train")
    ap.add_argument("--leadf", default="D:/tanitad/data/v9/lead_train4572.npz")
    ap.add_argument("--evalc", default="D:/tanitad/cache/eval")
    ap.add_argument("--join", default="D:/tanitad/rhacc10k/pins/join_eval141.npz")
    ap.add_argument("--ref-ckpt", default=None)
    ap.add_argument("--ref-sha", default=None)
    ap.add_argument("--no-guard", action="store_true", help="tests only")
    ap.add_argument("--no-marker", action="store_true", help="smoke driver: no QUEUE-COMPLETE line")
    a = ap.parse_args(argv)
    cfg = A.ARMS[a.arm]
    H = cfg["H"]
    os.makedirs(a.out, exist_ok=True)
    res = {"card": CARD, "tier": "T0 (a head on cached tokens of logged frames); never driving performance", "arm": a.arm, "cfg": dict(cfg, dil=list(cfg["dil"]) if cfg["dil"] else None),
           "receptive_field_frames": A.receptive_field(cfg["dil"]), "label_run": a.label_run, "controls": {}, "timing": {},
           "sha256": {n: F.sha_file(os.path.join(os.path.dirname(m.__file__), n)) for n, m in (("leadspeed.py", L), ("leadacc.py", A), ("train_full_box2.py", F))}}
    res["sha256"]["train_acc_box2.py"] = F.sha_file(os.path.abspath(__file__))
    try:
        import torch
        man = F.jl_lines(os.path.join(a.corpus, "manifest.jsonl"))
        evm = F.jl_lines(os.path.join(a.evalc, "manifest.jsonl"))
        z = np.load(a.leadf, allow_pickle=True)
        sha, k, has, lead, gap = z["sha12"], z["k"], z["has"], z["lead"], z["gap"]
        order = np.argsort(sha, kind="stable")
        sha_s = sha[order]
        uniq, start = np.unique(sha_s, return_index=True)
        rows = {u: order[s:e] for u, s, e in zip(uniq, start, list(start[1:]) + [len(sha_s)])}
        cand = [r for r in man if r["sha12"] in rows]
        pick = np.random.default_rng(F.SEED).permutation(len(cand))
        n_fit_all = int(round(0.9 * len(cand)))
        fit_all, ho = [cand[i] for i in pick[:n_fit_all]], [cand[i] for i in pick[n_fit_all:]]
        fit = fit_all[:a.n_fit]
        if a.n_ho is not None:
            ho = ho[:a.n_ho]
        if a.n_eval is not None:
            evm = evm[:a.n_eval]
        fit_ids, ho_ids, ev_ids = {r["clip"] for r in fit}, {r["clip"] for r in ho}, {r["clip"] for r in evm}
        res["seeds"] = {"split": F.SEED, "train": a.train_seed}
        res["split"] = {"with_labels": len(cand), "fit_all": len(fit_all), "fit_used": len(fit), "held_out": len(ho), "eval": len(evm)}
        res["controls"]["c2_disjoint"] = {"fit_and_held_out": len(fit_ids & ho_ids), "fit_and_eval": len(fit_ids & ev_ids), "PASS": not (fit_ids & ho_ids) and not (fit_ids & ev_ids)}
        need = len(fit) * F.MB_PER_CLIP / 1000.0
        ca0, own0 = F.commit_avail_gb(), F.own_commit_gb()
        need0 = F.GUARD_GB + need + max(0.0, F.PROC_GB - own0)
        guard = {"amendment": "card ADD149 AM1+AM2+AM3", "commit_avail_at_start_GB": round(ca0, 2), "own_commit_at_start_GB": round(own0, 3), "need_GB": round(need0, 2), "violations": 0}
        res["guard"] = guard
        if not a.no_guard and not (ca0 >= need0):
            res["name"] = "ACCHIST10K-STOPPED-MEMORY"
            raise RuntimeError("memory guard at start: commit available %.2f GB < %.2f GB" % (ca0, need0))
        touch_sum = [0]

        def load(recs, labelled=True, touch=False):
            out = []
            for i, r in enumerate(recs):
                if not a.no_guard:
                    left = (len(recs) - i) * F.MB_PER_CLIP / 1000.0
                    ca = F.commit_avail_gb()
                    if not (ca >= F.GUARD_GB + left):
                        guard["violations"] += 1
                        guard["stop_at"] = {"clip_index": i, "of": len(recs), "commit_avail_GB": round(ca, 2), "need_GB": round(F.GUARD_GB + left, 2)}
                        res["name"] = "ACCHIST10K-STOPPED-MEMORY"
                        raise RuntimeError("memory guard during load: commit available %.2f GB < %.2f GB" % (ca, F.GUARD_GB + left))
                pth = os.path.join(a.corpus if labelled else a.evalc, r["file"])
                d = torch.load(pth, map_location="cpu", weights_only=False, mmap=True)
                v = d["v"].numpy().astype(np.float64)
                del d
                tok = tokstore.open_tokens(pth, r.get("shapes", {}).get("pool4x10", F.SHAPE))
                if touch:
                    touch_sum[0] += int(tok.sum(dtype=np.uint64) % 997)
                out.append({"clip": r["clip"], "tok": tok, "v": v})
            return out

        def labels_corpus(r, T):
            ix = rows[r["sha12"]]
            pres, g, clos = L.token_labels(k[ix], has[ix], lead[ix], gap[ix], T)
            return pres, g, clos

        t0 = time.time()
        fit_d = load(fit, touch=True)
        for dd, r in zip(fit_d, fit):
            dd["pres"], dd["gap"], c5 = labels_corpus(r, dd["tok"].shape[0])
            dd["clos"] = c5
            dd["d"] = A.d_labels(c5 if a.label_run == 5 else A.closing_labels(dd["pres"], dd["gap"], run_max=a.label_run))
        res["timing"]["load_fit_s"] = round(time.time() - t0, 1)

        def c5_check(recs, root, n):
            eq, tot = 0, 0
            for r in recs[:n]:
                pth = os.path.join(root, r["file"])
                d = torch.load(pth, map_location="cpu", weights_only=False, mmap=True)
                eq += int(np.array_equal(np.asarray(tokstore.open_tokens(pth, r.get("shapes", {}).get("pool4x10", F.SHAPE))), d["pool4x10"].view(torch.uint8).numpy()))
                tot += 1
                del d
            return eq, tot

        e1, n1_ = c5_check(fit, a.corpus, 50)
        e2, n2_ = c5_check(ho, a.corpus, 20)
        e3, n3_ = c5_check(evm, a.evalc, 20)
        p0 = os.path.join(a.corpus, fit[0]["file"])
        sh0 = tuple(fit[0].get("shapes", {}).get("pool4x10", F.SHAPE))
        off0, _ = tokstore.member(p0, sh0)
        d0 = torch.load(p0, map_location="cpu", weights_only=False, mmap=True)
        wrong = np.memmap(p0, dtype=np.uint8, mode="r", offset=max(0, off0 - 64), shape=sh0)
        planted_differs = not np.array_equal(np.asarray(wrong), d0["pool4x10"].view(torch.uint8).numpy())
        del d0, wrong
        res["controls"]["c5_token_bytes"] = {"fit_equal": [e1, n1_], "held_out_equal": [e2, n2_], "eval_equal": [e3, n3_], "planted_offset_differs": bool(planted_differs),
                                             "PASS": bool(e1 == n1_ and e2 == n2_ and e3 == n3_ and n1_ > 0 and planted_differs)}
        # ---- c6 window alignment (bytes against the memmap)
        tk = fit_d[0]["tok"]
        T0 = tk.shape[0]
        ok6, checked = True, []
        for t in sorted({A.T_MIN, min(T0 - 1, H - 2), T0 - 1}):
            w, m = A.window(tk, t, H)
            s = t - H + 1
            exp_tok = np.asarray(tk[max(0, s):t + 1])
            npad = max(0, -s)
            ok6 &= bool(w.shape[0] == H and np.array_equal(w[npad:], exp_tok) and not w[:npad].any() and m.sum() == H - npad and not m[:npad].any())
            checked.append(int(t))
        res["controls"]["c6_window"] = {"frames_checked": checked, "H": H, "PASS": bool(ok6)}
        res["load"] = {"tokens": "read-only memmap (card ADD149 AM3)", "touch_check": touch_sum[0] % 997, "fit_clips": len(fit_d), "commit_avail_after_GB": round(F.commit_avail_gb(), 2),
                       "own_commit_after_GB": round(F.own_commit_gb(), 3)}
        # ---- training
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(a.train_seed)
        head = A.make_acc_head(H, cfg["dil"], cfg["motion"]).to(dev)
        res["n_params"] = int(sum(p.numel() for p in head.parameters()))
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        idx = [(ci, int(t)) for ci, dd in enumerate(fit_d) for t in A.window_ends(dd["tok"].shape[0])]
        gen = np.random.default_rng(a.train_seed)

        def batch(pairs):
            ws = [A.window(fit_d[c]["tok"], t, H) for c, t in pairs]
            x = to_x(torch, np.stack([w for w, _ in ws]), dev)
            v = torch.from_numpy(np.stack([m for _, m in ws])).to(dev)
            p = torch.tensor([float(fit_d[c]["pres"][t]) for c, t in pairs], device=dev)
            g = torch.tensor([fit_d[c]["gap"][t] for c, t in pairs], dtype=torch.float32, device=dev)
            c_ = torch.tensor([fit_d[c]["clos"][t] for c, t in pairs], dtype=torch.float32, device=dev)
            dd_ = torch.tensor([fit_d[c]["d"][t] for c, t in pairs], dtype=torch.float32, device=dev)
            return x, v, p, g, c_, dd_

        losses, dlosses, ckpts, t1 = [], [], {}, time.time()
        for step in range(1, a.steps + 1):
            x, v, p, g, c_, dd_ = batch([idx[i] for i in gen.integers(0, len(idx), 64)])
            t = A.loss_terms(head(x, v), p, g, c_, dd_)
            loss = t["pres"] + t["gap"] + t["clos"] + t["d"]
            if step == 1:                                                                       # c7: the d term reaches the d row and only it
                gw = torch.autograd.grad(t["d"], head.out.weight, retain_graph=True)[0]
                res["controls"]["c7_d_trained"] = {"d_term_step1": float(t["d"].item()), "grad_abs_d_row": float(gw[3].abs().sum()), "grad_abs_other_rows": float(gw[:3].abs().sum()),
                                                   "PASS": bool(math.isfinite(float(t["d"].item())) and float(gw[3].abs().sum()) > 0 and float(gw[:3].abs().sum()) == 0.0)}
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            dlosses.append(float(t["d"].item()))
            if step % a.ckpt_every == 0 or step == a.steps:
                pth = os.path.join(a.out, "head_%d.pt" % step)
                torch.save(head.state_dict(), pth)
                ckpts[step] = F.sha_file(pth)
                print("step", step, "loss100", round(float(np.mean(losses[-100:])), 4), "d100", round(float(np.mean(dlosses[-100:])), 4), "s", round(time.time() - t1, 1), flush=True)
        n1 = min(1000, len(losses) // 2)
        fin = bool(np.isfinite(losses).all())
        tr_s = time.time() - t1
        res["train"] = {"steps": len(losses), "windows_fit": len(idx), "seconds": round(tr_s, 1), "s_per_step": round(tr_s / max(1, len(losses)), 4),
                        "loss_first1000": round(float(np.mean(losses[:n1])), 4), "loss_last1000": round(float(np.mean(losses[-n1:])), 4),
                        "d_loss_first1000": round(float(np.mean(dlosses[:n1])), 4), "d_loss_last1000": round(float(np.mean(dlosses[-n1:])), 4),
                        "finite": fin, "ckpt_sha256": {str(s): h for s, h in ckpts.items()}}
        res["controls"]["c1_run"] = {"guard_violations": guard["violations"], "steps_done": len(losses),
                                     "PASS": bool(fin and len(losses) == a.steps and np.mean(losses[-n1:]) < np.mean(losses[:n1]))}
        rec = os.path.join(a.out, "head_%d.pt" % a.steps)
        h2 = F.sha_file(rec)
        head.load_state_dict(torch.load(rec, map_location=dev))
        head.eval()
        res["controls"]["c4_checkpoint"] = {"sha256_at_save": ckpts.get(a.steps), "sha256_at_read": h2, "PASS": ckpts.get(a.steps) == h2}
        del fit_d

        ref_head = None
        if a.ref_ckpt:
            hr = F.sha_file(a.ref_ckpt)
            if not (a.ref_sha and hr.startswith(a.ref_sha)):
                raise RuntimeError("reference checkpoint sha256 %s does not start with %s" % (hr[:16], a.ref_sha))
            ref_head = L.make_head(d_in=1024, width=64, temporal=True).to(dev)
            ref_head.load_state_dict(torch.load(a.ref_ckpt, map_location=dev))
            ref_head.eval()
            res["ref"] = {"ckpt": a.ref_ckpt, "sha256": hr}

        def stream(recs, labelled, labeller):
            out, tt = [], time.time()
            for r in recs:
                dd = load([r], labelled=labelled)[0]
                T = dd["tok"].shape[0]
                pres, g, clos = labeller(r, dd, T)
                P, G, C, D = head_outputs(torch, head, dd["tok"], dev, H)
                pc = {"v": dd["v"], "pres_lab": pres, "gap_lab": g, "clos_lab": clos, "clos_lab2": A.closing_labels(pres, g, run_max=10), "d_lab": A.d_labels(clos),
                      "pres_head": P, "gap_head": G, "clos_head": C, "d_head": D}
                if ref_head is not None:
                    pc["clos_ref"], pc["d_ref"] = ref_outputs(torch, ref_head, dd["tok"], dev)
                out.append(pc)
                del dd
            return out, time.time() - tt

        def summarise(pcs):
            cs = [A.clip_sums(A.clip_vectors(d)) for d in pcs]
            q = A.reads(cs)
            q.update(F.read_quantities(pcs))                                                   # S, A (= A_derived), gap MAE, presence AUC by row 217's own code
            q["A_row217_code"] = q.pop("A")
            q["A"] = A.reads(cs)["A"]
            return q, cs

        pc_ho, s_ho = stream(ho, True, lambda r, dd, T: labels_corpus(r, T))
        res["timing"]["read_held_out_s"] = round(s_ho, 1)
        q, cs = summarise(pc_ho)
        res["held_out"] = q
        res["held_out_halves"] = [summarise(pc_ho[0::2])[0], summarise(pc_ho[1::2])[0]]
        fl = A.paired(cs, "A", cs, "A_floor", seed=151)
        res["dA_floor_held_out"] = fl
        if ref_head is not None:
            cr = [A.clip_sums(A.clip_vectors(dict(d, clos_head=d["clos_ref"], d_head=d["d_ref"]))) for d in pc_ho]
            res["paired_vs_ref_held_out"] = {"S_ref": A.reads(cr)["S"], "A_ref": A.reads(cr)["A"], "dS": A.paired(cs, "S", cr, "S", seed=150), "dA": A.paired(cs, "A", cr, "A", seed=150)}
        planted = [dict(d, clos_head=d["clos_lab"], d_head=d["d_lab"]) for d in pc_ho]
        pq = A.reads([A.clip_sums(A.clip_vectors(d)) for d in planted])
        res["controls"]["c3_planted_read"] = {"S": pq["S"], "A": pq["A"], "A_derived": pq["A_derived"],
                                              "PASS": bool(abs(pq["S"] - 1.0) < 1e-9 and abs(pq["A"] - 1.0) < 1e-9 and abs(pq["A_derived"] - 1.0) < 1e-9)}
        J = np.load(a.join, allow_pickle=True)
        jclip = [str(c) for c in J["clip"]]

        def lab_eval(r, dd, T):
            ci = jclip.index(dd["clip"])
            o, n = int(J["off"][ci]), int(J["n"][ci])
            lv = (np.asarray(J["lead_valid"][o:o + n]) == 1).astype(int)
            hv = np.asarray(J["has"][o:o + n]).astype(int)
            return L.token_labels(2 * np.arange(n), hv, lv, np.asarray(J["lead_gap"][o:o + n], np.float64), T)

        pc_ev, s_ev = stream(evm, False, lab_eval)
        res["timing"]["read_eval_s"] = round(s_ev, 1)
        qe, cse = summarise(pc_ev)
        res["eval141"] = qe
        res["dA_floor_eval141"] = A.paired(cse, "A", cse, "A_floor", seed=151)
        ok = all(v["PASS"] for v in res["controls"].values())
        res["controls_all_pass"] = ok
        if ok:
            res["name"] = A.name_of(q["A"], fl["ci95"][0], q["S"])
            res["eval141_name"] = A.name_of(qe["A"], res["dA_floor_eval141"]["ci95"][0], qe["S"])
    except Exception as ex:                                                                     # noqa: BLE001
        res["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:500])
        res["traceback"] = traceback.format_exc()[-2000:]
        res.setdefault("name", None)
    json.dump(res, open(os.path.join(a.out, "ACC.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("traceback", "held_out_halves")}, indent=1, default=float), flush=True)
    if not a.no_marker:
        print("QUEUE-COMPLETE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    return res


if __name__ == "__main__":
    main()
    sys.exit(0)
