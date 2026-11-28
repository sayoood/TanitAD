"""rbvis.py -- RB-VIS: frozen N1R vision features -> attention-pool probe of TAC-LAT on roundabout windows.  Card: PREREG_RBVIS.md (written and hashed before any number).

Tier T0 vision-only decodability (NOT driving performance).  Windows are rbdist_zt's (same selection code).  Frames <= t only; no ego / kinematics / route text input.
Usage: python rbvis.py [--smoke] [--out DIR]     writes out/RESULT_RBVIS.json and the sentinel out/RBVIS.done (first line DONE or FAIL).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
sys.path.insert(0, S1 + "/rbdist")
sys.path.insert(0, S1 + "/rbdist/src")
import rbdist_zt as RD  # noqa: E402

TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache/train"
EVAL_CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval"
CKPT = S1 + "/navrb2/read/ckpts/{}/ckpt_last.pt"
T0 = S1 + "/navrb2/read/out/t0/eval_NAVRB2-{}_real.npz"
CARD = HERE + "/PREREG_RBVIS.md"
MARGIN = 0.0383
FOK = 4
EPOCH_CHOICES = (2, 4)
STRIDE = 2
IGNORE = -100
KEY = "pool4x10"


def sha(p):
    return RD.sha(p)


# ------------------------------------------------------------------ the frame reader (pure; the mutation target of the tests)
def read_window(x, t, _shift=0):
    """frames t-3 .. t of one clip tensor x [n, 40, 1024].  _shift is for the mutant test only (a reader of frame t+1 must turn the causality test RED)."""
    a = int(t) - (FOK - 1) + _shift
    assert a >= 0 and a + FOK <= x.shape[0], (t, a, x.shape)
    return x[a:a + FOK]


def clip_path(cache, clip):
    return f"{cache}/clips/{clip}.pt"


def load_clip(cache, clip):
    import torch
    return torch.load(clip_path(cache, clip), map_location="cpu", mmap=True, weights_only=False)[KEY]


def subsample(rows, clip_of, stride=STRIDE):
    """deterministic: inside each clip (rows in order), keep indices 0, stride, 2*stride ...  -> boolean mask"""
    keep = np.zeros(len(rows), bool)
    for c in np.unique(clip_of):
        ix = np.where(clip_of == c)[0]
        keep[ix[::stride]] = True
    return keep


# ------------------------------------------------------------------ windows (rbdist selection code)
def build_windows(log=print):
    S1D = S1
    tpath, jt, je = S1D + "/read/inputs/navtok_table.npz", S1D + "/ckpt120build/in/join_train.npz", S1D + "/read/inputs/join_eval141.npz"
    assert sha(tpath) == RD.PIN_TABLE
    tab = np.load(tpath, allow_pickle=False)
    JT, JE = dict(np.load(jt, allow_pickle=False)), dict(np.load(je, allow_pickle=False))
    for k in ("clip", "off", "n"):
        assert np.array_equal(tab["train_" + k], JT[k]) and np.array_equal(tab["eval_" + k], JE[k]), k
    tr_clips, ev_clips = [str(c) for c in JT["clip"]], [str(c) for c in JE["clip"]]
    disj = {"table_train_vs_eval": len(set(tr_clips) & set(ev_clips)),
            "train_cache_files_vs_eval": len({f[:-3] for f in os.listdir(TRAIN_CACHE + "/clips")} & set(ev_clips)),
            "eval_cache_files_vs_train": len({f[:-3] for f in os.listdir(EVAL_CACHE + "/clips")} & set(tr_clips))}
    assert all(v == 0 for v in disj.values()), disj
    ex_t = RD.text_arrays(S1 + "/rbdist/src/texts.json")
    rb_t, fl_t, _, _ = RD.split_arrays(tab, "train", ex_t)
    off_t = np.asarray(JT["off"], np.int64)
    n_t = np.asarray(JT["n"], np.int64)
    clip_row = np.repeat(np.arange(len(n_t)), n_t)
    ytr_all = RD.label_rule(JT["has"], JT["lat_cls"])
    rows_t = np.where((fl_t == 0) & rb_t & (ytr_all != IGNORE))[0]
    assert len(rows_t) == 24722 and len(set(clip_row[rows_t])) == 426, (len(rows_t), len(set(clip_row[rows_t])))
    c_t = clip_row[rows_t]
    t_t = rows_t - off_t[c_t]
    ok_t = t_t >= FOK - 1
    n_t_lt3 = int((~ok_t).sum())
    rows_t, c_t, t_t = rows_t[ok_t], c_t[ok_t], t_t[ok_t]
    # clips whose cache file frame count differs from train_n would shift every row: dropped and counted
    bad = set()
    for c in np.unique(c_t):
        if int(load_clip(TRAIN_CACHE, tr_clips[c]).shape[0]) != int(n_t[c]):
            bad.add(int(c))
    ok2 = ~np.isin(c_t, list(bad))
    rows_t, c_t, t_t = rows_t[ok2], c_t[ok2], t_t[ok2]
    keep = subsample(rows_t, c_t)
    train = {"row": rows_t[keep], "clip": c_t[keep], "t": t_t[keep], "y": ytr_all[rows_t[keep]]}
    # RD-side training set (for the S arm, identical to rbdist's: ALL 24,722 windows)
    rb_all = np.where((fl_t == 0) & rb_t & (ytr_all != IGNORE))[0]
    # ---- eval
    d = S1 + "/read/t1/dump_STRAT1-C0_s0_m10_causal_heads"
    zs = [np.load(os.path.join(d, f"ep{i:04d}.npz")) for i in range(141)]
    eid = np.concatenate([z["eid"] for z in zs]); cidx = np.concatenate([z["clip_index"] for z in zs]); ws = np.concatenate([z["ws"] for z in zs])
    rows_e = np.asarray(JE["off"], np.int64)[cidx] + ws.astype(np.int64)
    rb_e, fl_e, ex_e, di_e = RD.split_arrays(tab, "eval", ex_t)
    state_e = np.array([RD.rb_state(rb_e, fl_e, int(r)) for r in rows_e])
    yall = RD.label_rule(JE["has"][rows_e], JE["lat_cls"][rows_e])
    assert int(state_e.sum()) == 235 and len(set(eid[state_e])) == 16
    sel = state_e & (yall != IGNORE)
    assert int(sel.sum()) == 202 and len(set(eid[sel])) == 16, (int(sel.sum()), len(set(eid[sel])))
    ev = {"ci": cidx[sel].astype(np.int64), "t": ws[sel].astype(np.int64), "y": yall[sel], "eid": eid[sel], "all_index": np.where(sel)[0]}
    # S arm inputs (rbdist's own): features from table arrays
    off_e = np.asarray(JE["off"], np.int64)[cidx]
    Xtr = RD.feature_matrix(rb_t, fl_t, RD.split_arrays(tab, "train", ex_t)[2], RD.split_arrays(tab, "train", ex_t)[3], rb_all, np.repeat(off_t, n_t)[rb_all])
    Xe = RD.feature_matrix(rb_e, fl_e, ex_e, di_e, rows_e[sel], off_e[sel])
    info = {"disjointness": disj, "train_rb_windows_labelled": int(len(rb_all)), "train_episodes": 426, "dropped_t_lt3": n_t_lt3, "dropped_bad_frame_count_clips": len(bad),
            "train_windows_after_filters": int(ok2.sum()), "train_windows_kept_stride": int(len(train["row"])), "train_episodes_kept": int(len(set(train["clip"]))),
            "train_label_counts": np.bincount(train["y"], minlength=5).tolist(), "eval_rb_windows": 235, "eval_scored_windows": int(sel.sum()), "eval_scored_episodes": 16,
            "eval_label_counts": np.bincount(ev["y"], minlength=5).tolist(),
            "inputs_sha256": {"navtok_table.npz": RD.PIN_TABLE, "join_train.npz": sha(jt), "join_eval141.npz": sha(je)}}
    return {"train": train, "eval": ev, "tr_clips": tr_clips, "ev_clips": ev_clips, "S": (Xtr, ytr_all[rb_all], Xe), "info": info}


def gather(cache, clips, win, idx, _shift=0):
    import torch
    out = torch.empty((len(idx), FOK, 40, 1024), dtype=torch.float8_e4m3fn)
    by = {}
    for k, i in enumerate(idx):
        by.setdefault(int(win["clip" if "clip" in win else "ci"][i]), []).append(k)
    for c, ks in by.items():
        x = load_clip(cache, clips[c])
        for k in ks:
            out[k] = read_window(x, int(win["t"][idx[k]]), _shift)
    return out


# ------------------------------------------------------------------ frozen adapter + probe (design of weatherprobe/wprobe_fit.py)
def make_nets():
    import torch.nn as nn

    class Adapter(nn.Module):
        def __init__(self, sd=None, rand_seed=None):
            super().__init__()
            import torch
            self.ln, self.lin = nn.LayerNorm(1024), nn.Linear(1024, 256)
            self.pos, self.time = nn.Parameter(torch.zeros(40, 256)), nn.Parameter(torch.zeros(4, 256))
            if sd is not None:
                self.ln.weight.data, self.ln.bias.data = sd["adapter.0.weight"].float(), sd["adapter.0.bias"].float()
                self.lin.weight.data, self.lin.bias.data = sd["adapter.1.weight"].float(), sd["adapter.1.bias"].float()
                self.pos.data, self.time.data = sd["pos"].float(), sd["time"].float()
            else:
                g = torch.Generator().manual_seed(rand_seed)
                torch.manual_seed(rand_seed)
                self.ln, self.lin = nn.LayerNorm(1024), nn.Linear(1024, 256)
                self.pos.data, self.time.data = torch.randn(40, 256, generator=g) * 0.02, torch.randn(4, 256, generator=g) * 0.02
            for p in self.parameters():
                p.requires_grad_(False)

        def tokens(self, f8):
            f = f8.float()
            return self.lin(self.ln(f)) + self.pos[None, None] + self.time[None, :, None]

    class Probe(nn.Module):
        def __init__(self, seed, n_out=5):
            super().__init__()
            import torch
            torch.manual_seed(seed)
            self.n1 = nn.LayerNorm(256)
            self.q = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
            self.att = nn.MultiheadAttention(256, 4, batch_first=True)
            self.n2 = nn.LayerNorm(256)
            self.out = nn.Linear(256, n_out)

        def forward(self, tok):
            x = self.n1(tok)
            o, _ = self.att(self.q.expand(x.shape[0], -1, -1), x, x, need_weights=False)
            return self.out(self.n2(o[:, 0]))
    return Adapter, Probe


def fit(Adapter, Probe, adapter, feats, y, epochs, seed, log, bs=256, lr=1e-3, wd=0.01, warm=100):
    import torch
    import torch.nn as nn
    probe = Probe(seed)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    N = len(y)
    steps = epochs * ((N + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps - 1)))))
    rng = np.random.RandomState(seed)
    yt = torch.from_numpy(np.asarray(y, np.int64))
    for ep in range(epochs):
        perm = rng.permutation(N)
        tot, t0 = 0.0, time.time()
        for i in range(0, N, bs):
            ix = torch.from_numpy(perm[i:i + bs])
            with torch.no_grad():
                tok = adapter.tokens(feats[ix]).flatten(1, 2)
            loss = nn.functional.cross_entropy(probe(tok), yt[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
        log(f"    ep {ep + 1}/{epochs} loss {tot / N:.4f} {time.time() - t0:.0f}s")
    return probe


def predict(adapter, probe, feats, bs=256):
    import torch
    probe.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            out.append(probe(adapter.tokens(feats[i:i + bs]).flatten(1, 2)).float().numpy())
    return np.concatenate(out)


def run_arm(name, Adapter, Probe, adapter, tr_feats, ytr, ctr, ev_feats, log, y_fit=None):
    """E chosen on a train-only 20 % clip holdout (accuracy), then refit on all train windows.  -> dict(logits, E, holdout)"""
    uc = np.unique(ctr)
    perm = np.random.RandomState(0).permutation(len(uc))
    held = set(uc[perm[: len(uc) // 5]].tolist())
    is_held = np.array([c in held for c in ctr])
    yf = ytr if y_fit is None else y_fit
    fi, hi = np.where(~is_held)[0], np.where(is_held)[0]
    import torch
    scores = {}
    for E in EPOCH_CHOICES:
        log(f"{name}: holdout fit E={E} on {len(fi)} windows")
        pr = fit(Adapter, Probe, adapter, tr_feats[torch.from_numpy(fi)], yf[fi], E, 0, log)
        scores[E] = float((predict(adapter, pr, tr_feats[torch.from_numpy(hi)]).argmax(1) == yf[hi]).mean())
        log(f"{name}: E={E} held-out accuracy {scores[E]:.4f}")
    E = max(EPOCH_CHOICES, key=lambda e: (scores[e], -e))
    log(f"{name}: E chosen {E}; final fit on {len(yf)} windows")
    pr = fit(Adapter, Probe, adapter, tr_feats, yf, E, 0, log)
    return {"logits": predict(adapter, pr, ev_feats), "E": E, "holdout": {str(k): v for k, v in scores.items()}}


# ------------------------------------------------------------------ scoring (pure)
def timing_cls(y):
    y = np.asarray(y)
    return np.where(y == 0, 0, np.where(np.isin(y, (1, 2)), 1, 2))


def cells(pred, y):
    turn = np.isin(y, (1, 2))
    return {"timing3_accuracy": float((timing_cls(pred) == timing_cls(y)).mean()),
            "side_given_turn": {"n": int(turn.sum()), "accuracy": float((pred[turn] == y[turn]).mean()) if turn.any() else None},
            "macro_f1": RD.macro_f1(y, pred), "pred_counts": np.bincount(pred, minlength=5).tolist()}


def carries(d, margin=MARGIN):
    return bool(d["delta"] >= margin and d["separated"])


def verdict(vs, vh):
    """vs, vh: per-seed lists of paired dicts for V - S and V - H"""
    vc = all(carries(d) for d in vs)
    hg = vc and all(carries(d) for d in vh)
    sugg = (not vc) and any(d["delta"] >= MARGIN and not d["separated"] for d in vs)
    return {"VIS_CARRIES": vc, "HEAD_GAP": hg, "suggestive_unseparated": sugg,
            "reading": "VIS-CARRIES+HEAD-GAP" if hg else ("VIS-CARRIES (head already uses it)" if vc else "VIS-SILENT")}


def provenance():
    out = {}
    for n, c in (("train", TRAIN_CACHE), ("eval", EVAL_CACHE)):
        p = c + "/meta.json"
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            out[n] = {k: m.get(k) for k in ("pipeline", "script_sha256", "encoder_module_sha256", "model", "enc_dtype", "torch", "transformers")}
        else:
            man = json.loads(open(c + "/manifest.jsonl", encoding="utf-8").readline())
            out[n] = {"meta.json": "absent", "first_manifest_row": {k: man.get(k) for k in ("dtype", "enc_dtype", "source", "grids")}}
    out["same_encoder_pipeline"] = out["train"].get("pipeline") is not None and out["train"] == {**out["eval"]}
    return out


def main(out_dir, smoke=False):
    import torch
    torch.set_num_threads(int(os.environ.get("RBVIS_THREADS", "6")))
    os.makedirs(out_dir, exist_ok=True)
    LOG = open(out_dir + "/rbvis_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()
    t0 = time.time()
    sys.path.insert(0, S1 + "/navrb2/read/tree/taniteval")
    from taniteval import ci as CI
    assert sha(CARD) == open(HERE + "/PREREG_RBVIS.sha256").read().split()[0], "card changed after hashing"
    W = build_windows(log)
    info, tr, ev = W["info"], W["train"], W["eval"]
    log("WINDOWS", json.dumps(info))
    prov = provenance()
    log("PROVENANCE", json.dumps(prov))
    # ---- S (recomputed, asserted)
    Xtr, ytr_rd, Xe = W["S"]
    pS = RD.fit_predict(Xtr, ytr_rd, Xe, RD.FEATS["S"])
    ye, ee = ev["y"], ev["eid"]
    corrS = (pS == ye).astype(np.float64)
    want = json.load(open(S1 + "/rbdist/out/RESULT_RBDIST_ZT.json"))["arms"]["S"]["accuracy"]["point"]
    assert abs(float(corrS.mean()) - want) < 1e-12, (float(corrS.mean()), want)
    maj = int(np.bincount(ytr_rd, minlength=5).argmax())
    kconst = {"majority_class": maj, "S_pred_unique": sorted(set(pS.tolist())), "S_accuracy": float(corrS.mean()), "eval_share_of_majority": float((ye == maj).mean())}
    kconst["pass"] = bool(kconst["S_pred_unique"] == [maj] and kconst["S_accuracy"] == kconst["eval_share_of_majority"])
    # ---- H
    H = {}
    for s in (0, 1):
        z = np.load(T0.format(f"N1R_s{s}"), allow_pickle=True)
        assert np.array_equal(z["win_ci"][ev["all_index"]], ev["ci"]) and np.array_equal(z["win_t"][ev["all_index"]], ev["t"]) and np.array_equal(z["eid"][ev["all_index"]], ee)
        H[s] = z["lat_logits"][ev["all_index"]].argmax(1)
    # ---- features
    n_tr = len(tr["row"]) if not smoke else 600
    idx_tr = np.arange(n_tr) if not smoke else np.random.RandomState(0).permutation(len(tr["row"]))[:600]
    trw = {"clip": tr["clip"][idx_tr], "t": tr["t"][idx_tr]}
    ytr, ctr = tr["y"][idx_tr], tr["clip"][idx_tr]
    ev_idx = np.arange(len(ye))
    tr_f = gather(TRAIN_CACHE, W["tr_clips"], trw, np.arange(len(ytr)))
    ev_f = gather(EVAL_CACHE, W["ev_clips"], {"ci": ev["ci"], "t": ev["t"]}, ev_idx)
    log("features", tuple(tr_f.shape), tuple(ev_f.shape), f"{time.time() - t0:.0f}s")
    Adapter, Probe = make_nets()
    res = {"tool": "rbvis.py", "card": "PREREG_RBVIS.md", "card_sha256": sha(CARD), "smoke": smoke,
           "evidence_class": "MEASURED (ours; frozen DINOv3 pooled features, frozen N1R adapters, small probe fit on train RB windows, CPU)",
           "tier": "T0 vision-only decodability and T0 head read (NOT driving performance)",
           "estimator": RD.EST, "margin": MARGIN, "windows": info, "provenance": prov, "controls": {"K-CONST": kconst, "K-DISJOINT": {"pass": all(v == 0 for v in info["disjointness"].values()), **info["disjointness"]}},
           "S": {"accuracy": RD.level(CI, corrS, ee), **cells(pS, ye)}, "arms": {}}
    preds = {"S": pS, "H_s0": H[0], "H_s1": H[1]}
    for s in (0, 1):
        z = np.load(T0.format(f"N1R_s{s}"), allow_pickle=True)
        res["arms"][f"H_s{s}"] = {"accuracy": RD.level(CI, (H[s] == ye).astype(float), ee), **cells(H[s], ye), "source": T0.format(f"N1R_s{s}")}
    ads = {}
    for s in (0, 1):
        ck = torch.load(CKPT.format(f"N1R_s{s}"), map_location="cpu", weights_only=False)["model"]
        ads[f"V_s{s}"] = Adapter(ck)
        res["arms"].setdefault(f"V_s{s}", {})["ckpt_sha256"] = sha(CKPT.format(f"N1R_s{s}"))
        res["arms"][f"V_s{s}"]["adapter_tensor_sha256"] = {k: hashlib.sha256(ck[k].float().contiguous().numpy().tobytes()).hexdigest()[:16] for k in ("pos", "time", "adapter.0.weight", "adapter.1.weight")}
    ads["RAND"] = Adapter(None, rand_seed=0)
    out_np = {"y": ye, "eid": ee, "pS": pS, "H_s0": H[0], "H_s1": H[1]}
    for name in ("V_s0", "V_s1", "RAND"):
        r = run_arm(name, Adapter, Probe, ads[name], tr_f, ytr, ctr, ev_f, log)
        p = r["logits"].argmax(1)
        preds[name] = p
        res["arms"].setdefault(name, {}).update({"E": r["E"], "holdout": r["holdout"], "accuracy": RD.level(CI, (p == ye).astype(float), ee), **cells(p, ye)})
        out_np[name + "_logits"] = r["logits"]
        np.savez_compressed(out_dir + "/rbvis_logits.npz", **out_np)
        log(f"{name} done acc {float((p == ye).mean()):.4f} {time.time() - t0:.0f}s")
    # regression arm: V_s0 with train labels permuted between train clips
    uc = np.unique(ctr)
    pc = np.random.RandomState(0).permutation(len(uc))
    rows_of = {int(c): np.where(ctr == c)[0] for c in uc}
    ysh = ytr.copy()
    for i, c in enumerate(uc):
        dst, src = rows_of[int(c)], rows_of[int(uc[pc[i]])]
        ysh[dst] = ytr[src[np.arange(len(dst)) % len(src)]]
    r = run_arm("R-SHUF-TRAIN", Adapter, Probe, ads["V_s0"], tr_f, ytr, ctr, ev_f, log, y_fit=ysh)
    p = r["logits"].argmax(1)
    g = RD.paired(CI, (p == ye).astype(float), corrS, ee)
    res["controls"]["R-SHUF-TRAIN"] = {"accuracy": float((p == ye).mean()), "gain_over_S": g, "E": r["E"], "pass": bool(g["delta"] < MARGIN and not carries(g))}
    # paired contrasts
    vs, vh = [], []
    for s in (0, 1):
        v = (preds[f"V_s{s}"] == ye).astype(float)
        a = res["arms"][f"V_s{s}"]
        a["V_minus_S"] = RD.paired(CI, v, corrS, ee)
        a["V_minus_H"] = RD.paired(CI, v, (H[s] == ye).astype(float), ee)
        vs.append(a["V_minus_S"]); vh.append(a["V_minus_H"])
        res["arms"][f"H_s{s}"]["H_minus_S"] = RD.paired(CI, (H[s] == ye).astype(float), corrS, ee)
    ra = res["arms"]["RAND"]
    ra["V_minus_S"] = RD.paired(CI, (preds["RAND"] == ye).astype(float), corrS, ee)
    res["RAND_flag"] = {"RAND_carries": carries(ra["V_minus_S"]),
                        "reading": "information is in the cached DINOv3 features, not the trained adapter" if carries(ra["V_minus_S"]) else "RAND does not carry"}
    bad = [k for k, v in res["controls"].items() if not v.get("pass", True)]
    v = verdict(vs, vh)
    res["problems"] = [f"{k} did not read as committed" for k in bad]
    res["verdict"] = {"INVALID (control problem)": True, "rule_reading_if_controls_held": v["reading"], **v} if bad else v
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out_dir + "/RESULT_RBVIS.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("RESULT", json.dumps({"verdict": res["verdict"], "problems": res["problems"]}))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    done = a.out + "/RBVIS.done"
    try:
        r = main(a.out, a.smoke)
        open(done, "w").write("DONE\n" + json.dumps(r["verdict"]) + "\n")
    except BaseException:
        open(done, "w").write("FAIL\n" + traceback.format_exc())
        raise
