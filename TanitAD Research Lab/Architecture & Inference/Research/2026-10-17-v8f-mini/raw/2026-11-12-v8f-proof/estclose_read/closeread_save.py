#!/usr/bin/env python
"""closeread_save.py -- card PREREG_V8F_PROOF_ADD79 (ESTCLOSE10K), part A: fit and SAVE the two closing-speed estimators of the card, CPU only.

Per reader seed k: N10 = arm N (mean of three small networks, seeds 0, 1, 2) at H = 10; L25 = arm L (ridge) at H = 25. Fitted exactly as closeread_fit.run_cell fits them
(same build_rows, same split_by_clip(clips, 0.25, 0), ridge weight by select_ridge then ridge_fit on all fit rows, nets by the code of train_mlp, normalisation of the nets = mean / sd of the
TRAIN part of the split). The imported fit tool and closest.py are not edited; the net training is repeated here (train_state) so that the weights can be saved.

Control e0 (all must pass, else rc 1 and no estimator file is written; everything is computed first, written after):
  e0a  repeated training = the fit tool's estimate on the score rows, in the same process (<= 1e-7 m/s)
  e0b  numpy forward of the saved object (closest.ClosingEstimator, loaded from the exact bytes that are written) = the fit tool's estimate, row by row: L25 <= 1e-6, N10 <= 1e-4 m/s
  e0c  per-seed Pearson r of the saved estimator on the score rows against cells.L_H25.r_per_seed[k] / cells.N_H10.r_per_seed[k] of the --add78-result file: <= 0.005 apart
  e0d  the table rows (all rows of the score dump) contain the rows of build_rows (same X, same frames)
The tool prints PASS / FAIL and the differences, never an r value.

Writes (only after e0): closest_N10_s<k>.npz, closest_L25_s<k>.npz, planted_zero_H10.npz, planted_const2_H10.npz, planted_readerclos_H10.npz,
est_score_<N10|L25>_s<k>.npz (clip_index, frame, est for EVERY row of the score dump; NaN where the H frames ending at the row do not all carry a valid read), SAVE_RECORD.json.

  python closeread_save.py --dump-dir D --out-dir NEWDIR --add78-result JSON [--seeds 0,1]      |      --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("CLOSEREAD_DIR"), os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "closeread")):
    if _p and os.path.isfile(os.path.join(_p, "closeread_fit.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import closeread_fit as CF      # noqa: E402
import closest as CL            # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD79"
ESTS = {"N10": ("N", 10, "N_H10"), "L25": ("L", 25, "L_H25")}
TOL_TRAIN, TOL_L, TOL_N, TOL_R = 1e-7, 1e-6, 1e-4, 0.005
BANNED = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================================ rows
def window_rows(D, H):
    """every row r of the dump whose H frames r-H+1..r (same clip, consecutive) all carry a valid finite read: (idx, X [n, H, 3]) -- the window logic of closeread_fit.build_rows without the label condition."""
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    n = len(cid)
    f3 = np.stack([np.asarray(D["gap"], np.float64), np.asarray(D["p"], np.float64), np.asarray(D["clos"], np.float64)], 1)
    ok = np.asarray(D["valid"], bool) & np.isfinite(f3).all(1)
    cs = np.concatenate([[0], np.cumsum(~ok)])
    idx = np.arange(H - 1, n)
    win = (cs[idx + 1] - cs[idx - H + 1] == 0) & (cid[idx] == cid[idx - H + 1]) & (fr[idx] - fr[idx - H + 1] == H - 1)
    sel = idx[win]
    X = np.stack([f3[sel - H + 1 + k] for k in range(H)], 1) if len(sel) else np.zeros((0, H, 3))
    return sel, X


def label_ok(D):
    return np.asarray(D["lab_present"], bool) & (np.asarray(D["lab_run"], np.int64) >= CF.RUN_MIN) & np.isfinite(np.asarray(D["lab_closing"], np.float64))


def est_table(D, E):
    """estimate of the saved object at every row of the dump (NaN where there is no full history) -> (est [n_rows], idx of rows with a full history, X)"""
    sel, X = window_rows(D, E.H)
    est = np.full(len(D["frame"]), np.nan)
    if len(sel):
        est[sel] = E.apply(X)
    return est, sel, X


# ============================================================================================ training (a copy of closeread_fit.train_mlp that returns the weights)
def train_state(Xtr, ytr, Xva, yva, seed, max_epochs=120, patience=10, bs=256, lr=1e-3):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32))        # noqa: E731
    xt, yt, xv, yv = f(Xtr), torch.from_numpy(ytr.astype(np.float32)), f(Xva), torch.from_numpy(yva.astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hub = torch.nn.HuberLoss(delta=1.0)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            hub(net(xt[j])[:, 0], yt[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(hub(net(xv)[:, 0], yv))
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    sdict = {k: t.detach().cpu().numpy().copy() for k, t in net.state_dict().items()}
    weights = tuple(sdict[k] for k in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"))

    def predict(X):
        with torch.no_grad():
            return net(f(X))[:, 0].numpy().astype(np.float64)
    return predict, weights, mu, sd


def fit_cell(arm, fit_rows, score_rows, n_epochs):
    """-> (est by the repeated training on the score rows, the object's parts). The split and the selection are those of closeread_fit.run_cell."""
    Xf, yf = fit_rows["X"].reshape(len(fit_rows["y"]), -1), np.array(fit_rows["y"], np.float64)
    Xs = score_rows["X"].reshape(len(score_rows["y"]), -1)
    va = CF.split_by_clip(fit_rows["clip"], 0.25, 0)
    Xtr, ytr, Xva, yva = Xf[~va], yf[~va], Xf[va], yf[va]
    if arm == "L":
        lam = CF.select_ridge(Xtr, ytr, Xva, yva)
        m = CF.ridge_fit(Xf, yf, lam)
        return CF.ridge_predict(m, Xs), {"mu": m["mu"], "sd": m["sd"], "w": m["w"], "b": float(m["b"]), "lambda": lam}
    preds, nets, mu0, sd0 = [], [], None, None
    for s in CF.N_SEEDS:
        pr, w, mu, sd = train_state(Xtr, ytr, Xva, yva, s, max_epochs=n_epochs)
        preds.append(pr(Xs))
        nets.append(w)
        if mu0 is None:
            mu0, sd0 = mu, sd
        assert np.array_equal(mu, mu0) and np.array_equal(sd, sd0)
    return np.mean(preds, 0), {"mu": mu0, "sd": sd0, "nets": nets}


def object_bytes(arm, H, seed, parts, meta, fault=None):
    """the exact bytes of the estimator file. fault (selftest only): 'mu' shifts the stored normalisation"""
    mu = np.array(parts["mu"], np.float64)
    if fault == "mu":
        mu = mu + 0.5
    bio = io.BytesIO()
    if arm == "L":
        CL.save_linear(bio, H, mu, parts["sd"], parts["w"], parts["b"], seed=seed, meta=meta)
    else:
        CL.save_nets(bio, H, mu, parts["sd"], parts["nets"], seed=seed, meta=meta)
    return bio.getvalue()


# ============================================================================================ the save
def run_save(dump_dir, out_dir, add78, seeds, n_epochs=120, fault=None, log=print):
    """-> (rc, record). rc 0: all of e0 passed and the files are written; rc 1: e0 failed, nothing is written; rc 2: refused."""
    if os.path.exists(out_dir):
        log("REFUSED: %s exists" % out_dir)
        return 2, None
    fits, scores, lists = CF.load_dir(dump_dir, seeds) if all(os.path.isfile(os.path.join(dump_dir, "reader_s%d_lists.json" % k)) for k in seeds) else (None, None, None)
    if fits is None:
        fits = {k: CF.load_dump(os.path.join(dump_dir, "reader_s%d_fit.npz" % k)) for k in seeds}
        scores = {k: CF.load_dump(os.path.join(dump_dir, "reader_s%d_score.npz" % k)) for k in seeds}
        lists = {k: {} for k in seeds}
    checks, diffs, blobs, tables = {}, {}, {}, {}
    inputs = {}
    for k in seeds:
        for kind in ("fit", "score", "lists"):
            p = os.path.join(dump_dir, "reader_s%d_%s.%s" % (k, kind, "json" if kind == "lists" else "npz"))
            if os.path.isfile(p):
                inputs["reader_s%d_%s" % (k, kind)] = sha256_file(p)
    for k in seeds:
        tid = lists[k].get("train_ids") if lists[k] else None
        for name, (arm, H, cellkey) in ESTS.items():
            tag = "%s_s%d" % (name, k)
            fr, sr = CF.build_rows(fits[k], H), CF.build_rows(scores[k], H)
            est_ref, _info = CF.run_cell(arm, fr, sr, n_epochs, train_ids=tid)
            est_mine, parts = fit_cell(arm, fr, sr, n_epochs)
            d_train = float(np.max(np.abs(est_mine - est_ref)))
            checks["e0a_" + tag], diffs["e0a_" + tag] = bool(d_train <= TOL_TRAIN), d_train
            meta = {"card": CARD, "estimator": name, "arm": arm, "H": H, "reader_seed": k, "n_fit_rows": int(len(fr["y"])), "inputs": {i: h for i, h in inputs.items() if i.startswith("reader_s%d_" % k)}}
            meta["lambda"] = parts.get("lambda")
            blob = object_bytes(arm, H, k, parts, meta, fault)
            z = np.load(io.BytesIO(blob), allow_pickle=False)
            E = CL.ClosingEstimator({q: z[q] for q in z.files})
            est_saved = E.apply(sr["X"])
            d_fwd = float(np.max(np.abs(est_saved - est_ref)))
            checks["e0b_" + tag], diffs["e0b_" + tag] = bool(d_fwd <= (TOL_L if arm == "L" else TOL_N)), d_fwd
            try:
                want = float(add78["cells"][cellkey]["r_per_seed"][k])
                d_r = abs(CF.pearson(est_saved, sr["y"]) - want)
            except (KeyError, IndexError, TypeError, ValueError):
                d_r = float("nan")
            checks["e0c_" + tag], diffs["e0c_" + tag] = bool(np.isfinite(d_r) and d_r <= TOL_R), d_r
            est_all, sel, Xall = est_table(scores[k], E)
            lab = label_ok(scores[k])
            sub = sel[lab[sel]]
            pos = np.searchsorted(sel, sub)
            same = bool(len(sub) == len(sr["y"]) and np.array_equal(Xall[pos], sr["X"]) and np.array_equal(np.asarray(scores[k]["frame"])[sub], sr["frame"]))
            checks["e0d_" + tag], diffs["e0d_" + tag] = same, 0.0 if same else float("nan")
            d_tab = float(np.max(np.abs(est_all[sub] - est_saved))) if same and len(sub) else float("nan")
            checks["e0d2_" + tag], diffs["e0d2_" + tag] = bool(same and d_tab <= 1e-12), d_tab
            blobs["closest_%s.npz" % tag] = blob
            tables["est_score_%s.npz" % tag] = {"clip_index": np.asarray(scores[k]["clip_index"], np.int64), "frame": np.asarray(scores[k]["frame"], np.int64), "est": est_all}
    for key in checks:
        log("%s %s  largest difference %s" % ("PASS" if checks[key] else "FAIL", key, "nan" if not np.isfinite(diffs[key]) else "%.3g" % diffs[key]))
    ok = all(checks.values())
    log("ZZ CLOSEREAD-SAVE e0 %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        return 1, {"e0_all_pass": False, "checks": checks, "differences": diffs}
    os.makedirs(out_dir)
    outs = {}
    for fn, blob in blobs.items():
        with open(os.path.join(out_dir, fn), "wb") as f:
            f.write(blob)
    for fn, t in tables.items():
        np.savez(os.path.join(out_dir, fn), **t)
    CL.save_planted(os.path.join(out_dir, "planted_zero_H10.npz"), "zero", 10)
    CL.save_planted(os.path.join(out_dir, "planted_const2_H10.npz"), "const", 10, 2.0)
    CL.save_planted(os.path.join(out_dir, "planted_readerclos_H10.npz"), "reader_closing", 10)
    for fn in sorted(os.listdir(out_dir)):
        outs[fn] = sha256_file(os.path.join(out_dir, fn))
    rec = {"card": CARD, "seeds": list(seeds), "n_epochs": n_epochs, "inputs_sha256": inputs, "add78_result_sha256": None, "outputs_sha256": outs, "e0_all_pass": True, "e0_checks": checks, "e0_differences": diffs,
           "tolerances": {"e0a": TOL_TRAIN, "e0b_L25": TOL_L, "e0b_N10": TOL_N, "e0c": TOL_R}}
    return 0, rec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--add78-result", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-epochs", type=int, default=120)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    add78 = json.load(open(a.add78_result, encoding="utf-8"))
    rc, rec = run_save(a.dump_dir, a.out_dir, add78, seeds, a.n_epochs)
    if rc == 0:
        rec["add78_result_sha256"] = sha256_file(a.add78_result)
        with open(os.path.join(a.out_dir, "SAVE_RECORD.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, default=float)
    print("ZZ CLOSEREAD-SAVE rc %d" % rc)
    return rc


# ============================================================================================ selftest
def selftest(keep_dir=None):
    import tempfile
    fails = []

    def chk(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        if not cond:
            fails.append(name)
    base = keep_dir or os.path.join(HERE, "tmp_selftest_save")
    os.makedirs(base, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="st_", dir=base)
    fits, scores, _ep = CF.make_synth(seed=3, n_fit=40, n_score=40)
    lists = CF.synth_lists(fits)
    dd = os.path.join(tmp, "dump")
    os.makedirs(dd)
    for k in fits:
        np.savez(os.path.join(dd, "reader_s%d_fit.npz" % k), **fits[k])
        np.savez(os.path.join(dd, "reader_s%d_score.npz" % k), **scores[k])
        json.dump(lists[k], open(os.path.join(dd, "reader_s%d_lists.json" % k), "w", encoding="utf-8"))
    NE = 12
    cells = {}
    for name, (arm, H, ck) in ESTS.items():
        rs = []
        for k in (0, 1):
            e, _ = CF.run_cell(arm, CF.build_rows(fits[k], H), CF.build_rows(scores[k], H), NE, train_ids=lists[k]["train_ids"])
            rs.append(CF.pearson(e, CF.build_rows(scores[k], H)["y"]))
        cells[ck] = {"r_per_seed": rs}
    add78 = {"cells": cells}
    out = os.path.join(tmp, "out")
    lines = []
    rc, rec = run_save(dd, out, add78, [0, 1], NE, log=lines.append)
    chk("e0 passes on the synthetic dumps and the files are written (rc 0)", rc == 0 and rec["e0_all_pass"])
    want = ["closest_%s_s%d.npz" % (n, k) for n in ESTS for k in (0, 1)] + ["est_score_%s_s%d.npz" % (n, k) for n in ESTS for k in (0, 1)] + ["planted_zero_H10.npz", "planted_const2_H10.npz", "planted_readerclos_H10.npz"]
    chk("all expected files exist", all(os.path.isfile(os.path.join(out, f)) for f in want))
    chk("printed lines hold PASS / FAIL and no r value line", all(l.startswith(("PASS", "FAIL", "ZZ")) for l in lines))
    E = CL.ClosingEstimator.load(os.path.join(out, "closest_L25_s0.npz"))
    T = np.load(os.path.join(out, "est_score_L25_s0.npz"))
    sr = CF.build_rows(scores[0], 25)
    sel_est = E.apply(sr["X"])
    chk("L25 saved file: H 25, seed 0, meta card", E.H == 25 and E.arm == "L" and E.seed == 0 and E.meta["card"] == CARD)
    n = len(scores[0]["frame"])
    chk("est table has one row per score-dump row, NaN where the 25 frames are not all valid", len(T["est"]) == n and np.isnan(T["est"][:3]).all() and np.isnan(T["est"][np.asarray(scores[0]["frame"]) < 4 + 24]).all())
    key = {(int(c), int(f)): v for c, f, v in zip(T["clip_index"], T["frame"], T["est"])}
    cpos = {str(c): i for i, c in enumerate(scores[0]["clips"])}
    ci0 = np.asarray(scores[0]["clip_index"])[np.searchsorted(np.asarray(scores[0]["cid"]), np.arange(len(scores[0]["clips"])))]
    got = np.array([key[(int(ci0[cpos[str(c)]]), int(f))] for c, f in zip(sr["clip"], sr["frame"])])
    chk("est table at the label rows (looked up by clip_index, frame) equals the estimate on those rows", np.allclose(got, sel_est, atol=1e-12))
    Zp = CL.ClosingEstimator.load(os.path.join(out, "planted_const2_H10.npz"))
    chk("planted const2 returns 2.0", np.allclose(Zp.apply(np.zeros((3, 10, 3))), 2.0))
    rc2, _ = run_save(dd, out, add78, [0, 1], NE, log=lambda s: None)
    chk("an existing out dir is refused (rc 2)", rc2 == 2)
    bad = {"cells": {ck: {"r_per_seed": [r + 0.01 for r in v["r_per_seed"]]} for ck, v in cells.items()}}
    out2 = os.path.join(tmp, "out2")
    lines2 = []
    rc3, rec3 = run_save(dd, out2, bad, [0, 1], NE, log=lines2.append)
    chk("planted failure of e0 (r off by 0.01): rc 1, no out dir, FAIL printed", rc3 == 1 and not os.path.exists(out2) and any(l.startswith("FAIL e0c") for l in lines2))
    out3 = os.path.join(tmp, "out3")
    rc4, _ = run_save(dd, out3, add78, [0, 1], NE, fault="mu", log=lines2.append)
    chk("planted failure of the stored normalisation: rc 1 and no files", rc4 == 1 and not os.path.exists(out3))
    out4 = os.path.join(tmp, "out4")
    rc5, _ = run_save(dd, out4, {"cells": {}}, [0, 1], NE, log=lambda s: None)
    chk("an add78 file without the cells fails e0c (rc 1)", rc5 == 1 and not os.path.exists(out4))
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    chk("text of the tool holds none of the banned words", not [x for x in BANNED if x in src])
    print("ZZ SELFTEST-CLOSEREAD-SAVE ok: %s (rc %d)" % (not fails, 1 if fails else 0))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
