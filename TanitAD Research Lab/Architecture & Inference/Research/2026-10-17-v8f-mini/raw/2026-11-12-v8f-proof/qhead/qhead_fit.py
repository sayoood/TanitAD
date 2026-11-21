#!/usr/bin/env python
"""qhead_fit.py -- card PREREG_V8F_PROOF_ADD42 (QHEAD), section 3: per checkpoint a low-quantile gap head (pinball loss at 0.20) on the banked level-1 states of the TRAIN probe clips.
Runs ON THE POD (Linux, /workspace/venv/bin/python) or anywhere: every path is an argument. No evaluation file is read (control h4: a path of the evaluation split is refused, every file read is listed with its sha256).

  python qhead_fit.py --arm LS_s0 --seed 0 --states <ARM>_train.npz --pack train_pack.npz --out-dir DIR [--lead <ARM>_train_lead.npz] [--banked-head <ARM>_train_lead.npz.head.npz]
                      [--steps 19164] [--batch 64] [--device cpu]

Rows: state rows (clip, t) with 4 predecessors t-1..t-4 of the same clip (the banked head's history), whose train-pack label has presence 1 and a finite gap (any distance). The last 200 clips of the states file
(manifest order) are held out of the fit and used only for control h5; the fit uses the others. Recipe: that of the banked head (QHEAD_DEFS.md Q4): AdamW, lr 3e-4, weight decay 1e-2 on 2-D weights, warmup 200 then cosine to 0.1 x lr,
clip 1.0, batch 64, raw states (no normalisation), steps 19164; the loss line is the only difference. Writes <ARM>.qhead.npz (the head) and <ARM>.qhead.json (the record, written last). Tier T0.
Exit: 0 fit written (h5 may read False: the record says so), 3 refused or a gradient check failed.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import qhead_model as QM  # noqa: E402

import torch  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD42"
HELD_OUT_CLIPS = 200
H5_LO, H5_HI = 0.15, 0.25
DEFAULT = {"steps": 19164, "batch": 64, "lr": 3e-4, "wd": 1e-2, "warmup": 200, "clip_grad": 1.0}


def refuse(msg):
    raise SystemExit("REFUSED: " + msg)


def pinball(pred, y, q):
    """The pinball loss at level q of a prediction `pred` of the q-quantile of y (mean over the rows)."""
    e = y - pred
    return torch.maximum(q * e, (q - 1.0) * e).mean()


def lr_at(step, lr, warmup, total):
    """The trainer's schedule (v8f_mini_train.py class LRSchedule, lines 757-766): linear warmup to lr, then a cosine to 0.1 x lr at `total`."""
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1.0))))


def history_index(clip_idx, t, hist):
    """idx [n, hist]: row of frames t-hist+1 .. t of the same clip (oldest first), -1 where any is absent; ok [n]."""
    clip_idx = np.asarray(clip_idx, np.int64)
    t = np.asarray(t, np.int64)
    base = int(t.max()) + hist + 2 if t.size else 1
    key = (clip_idx * base + t + hist)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    idx = np.full((len(t), hist), -1, np.int64)
    for j in range(hist):
        want = key - (hist - 1 - j)
        pos = np.searchsorted(sk, want)
        pos = np.clip(pos, 0, len(sk) - 1)
        hit = sk[pos] == want
        idx[:, j] = np.where(hit, order[pos], -1)
    ok = (idx >= 0).all(1)
    idx[~ok] = -1
    return idx, ok


def build_rows(states, pack, hist):
    """-> dict(hist_idx [R, hist], y [R] label gap (m), clip [R] state clip index, row [R] state row of the end frame, counts)."""
    names = np.asarray(states["clips"]).astype(str)
    ci = np.asarray(states["clip_idx"]).astype(np.int64)
    t = np.asarray(states["t"]).astype(np.int64)
    p_names = np.asarray(pack["clips"]).astype(str)
    if len(p_names) != len(names) or not np.array_equal(p_names, names):
        refuse("the clip list of the pack differs from that of the state file (manifest order must be the same)")
    p_ci = np.asarray(pack["clip_idx"]).astype(np.int64)
    p_t = np.asarray(pack["t"]).astype(np.int64)
    base = int(max(t.max(), p_t.max())) + 2
    pk = p_ci * base + p_t
    po = np.argsort(pk, kind="stable")
    spk = pk[po]
    sk = ci * base + t
    pos = np.clip(np.searchsorted(spk, sk), 0, len(spk) - 1)
    found = spk[pos] == sk
    prow = np.where(found, po[pos], -1)
    hidx, ok_h = history_index(ci, t, hist)
    gap = np.where(found, np.asarray(pack["gap"], np.float64)[np.where(found, prow, 0)], np.nan)
    pres = np.where(found, np.asarray(pack["pres"], np.float64)[np.where(found, prow, 0)], 0.0)
    use = found & ok_h & (pres == 1) & np.isfinite(gap)
    rows = np.flatnonzero(use)
    return {"hist_idx": hidx[rows], "y": gap[rows], "clip": ci[rows], "row": rows,
            "counts": {"state_rows": int(len(t)), "matched_to_pack": int(found.sum()), "with_history": int((found & ok_h).sum()), "rows_used": int(use.sum()),
                       "n_clips": int(len(names))}}


def make_grad_check(head, Z, hist_idx, y, q, banked=None, device="cpu", read=None):
    """Control h6, gradient part, on one batch: grad abs-sum of every parameter of the new head (all > 0), of a control parameter outside the graph (exactly 0.0) and of every
    banked-head tensor, which the fit never puts in the graph (exactly 0.0)."""
    ctrl = torch.nn.Parameter(torch.zeros(3, device=device))
    held = {k: torch.nn.Parameter(torch.from_numpy(np.asarray(v, np.float32)).to(device)) for k, v in (banked or {}).items() if k in QM.TENSOR_KEYS}
    x = torch.from_numpy(np.asarray(Z[hist_idx], np.float32)).to(device)
    yy = torch.from_numpy(np.asarray(y, np.float32) / QM.GAP_SCALE_M).to(device)
    head.zero_grad(set_to_none=True)
    loss = pinball(head(x)[:, 0, 0], yy, q)
    if loss.requires_grad:                                    # a loss that never reaches the head leaves every gradient at None (read as 0.0)
        loss.backward()
    new = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in head.named_parameters()}
    other = {"control_outside_graph": float(ctrl.grad.abs().sum()) if ctrl.grad is not None else 0.0}
    for k, p in held.items():
        other["banked_" + k] = float(p.grad.abs().sum()) if p.grad is not None else 0.0
    head.zero_grad(set_to_none=True)
    r = {"new_head": new, "outside": other, "new_all_nonzero": bool(all(v > 0 for v in new.values())), "outside_all_exactly_zero": bool(all(v == 0.0 for v in other.values())),
         "n_banked_tensors_checked": len(held)}
    r["pass"] = bool(r["new_all_nonzero"] and r["outside_all_exactly_zero"])
    return r


def fit_head(Z, hist_idx, y, seed, steps, batch, lr, wd, warmup, clip_grad, q=QM.Q_LEVEL, device="cpu", hist=QM.HIST, log=None, read=None):
    """Fit a fresh QHead (torch seed = the base seed) on the rows. Z [n, D] states (float16 or float32), hist_idx [R, hist] rows of Z, y [R] label gap in m. -> head (eval mode)."""
    torch.manual_seed(int(seed))
    head = QM.QHead(Z.shape[1], hist).to(device)
    decay = [p for p in head.parameters() if p.ndim >= 2]
    nodecay = [p for p in head.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}], lr=lr)
    rng = np.random.RandomState(int(seed))
    R = len(y)
    perm = rng.permutation(R)
    pos = 0
    yt = torch.from_numpy(np.asarray(y, np.float32) / QM.GAP_SCALE_M)
    tl = []
    for step in range(int(steps)):
        if pos + batch > R:
            perm = rng.permutation(R)
            pos = 0
        sel = perm[pos:pos + batch]
        pos += batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr, warmup, steps)
        x = torch.from_numpy(np.asarray(Z[hist_idx[sel]], np.float32)).to(device)       # INPUT: the states of the history window
        pred = head(x)[:, 0, 0]
        loss = pinball(pred, yt[sel].to(device), q)                                      # LOSS: pinball at the quantile q
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), clip_grad)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            tl.append((step, float(loss.detach())))
            if log:
                log("step %d loss %.5f lr %.2e" % (step, float(loss.detach()), opt.param_groups[0]["lr"]))
    head.eval()
    head.train_loss_trace = tl
    return head


def predict(head, Z, hist_idx, device="cpu", chunk=4096):
    out = np.empty(len(hist_idx), np.float64)
    with torch.no_grad():
        for i in range(0, len(hist_idx), chunk):
            x = torch.from_numpy(np.asarray(Z[hist_idx[i:i + chunk]], np.float32)).to(device)
            out[i:i + chunk] = head(x)[:, 0, 0].double().cpu().numpy() * QM.GAP_SCALE_M
    return out


def coverage(q_pred, y):
    """Share of rows with the quantile output above the label gap (the card's coverage), and the mean amount (q - y)."""
    if len(y) == 0:
        return {"n": 0, "share_q_above_label": None, "mean_q_minus_label_m": None}
    return {"n": int(len(y)), "share_q_above_label": float((q_pred > y).mean()), "mean_q_minus_label_m": float((q_pred - y).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the seed of the base (0 or 1): the head is freshly initialised with it")
    ap.add_argument("--states", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lead", default="", help="optional: the banked head read of the train rows (<ARM>_train_lead.npz), for the record only")
    ap.add_argument("--banked-head", default="", help="optional: the six banked head tensors, for the gradient control and the record")
    for k, v in DEFAULT.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--hist", type=int, default=QM.HIST)
    ap.add_argument("--held-out", type=int, default=HELD_OUT_CLIPS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    inputs = [p for p in (a.states, a.pack, a.lead, a.banked_head) if p]
    for p in inputs:
        if QM.is_eval_path(p):
            refuse("a path of the evaluation split in the fit (control h4): " + p)
        if not os.path.isfile(p):
            refuse("no such file: " + p)
    rec_p = a.states + ".json"
    if not os.path.isfile(rec_p):
        refuse("no record %s next to the state file (its split must read train)" % rec_p)
    srec = json.load(open(rec_p, encoding="utf-8"))
    if srec.get("split") != "train":
        refuse("the state file record says split %r, not train" % srec.get("split"))
    out_h = os.path.join(a.out_dir, a.arm + ".qhead.npz")
    out_j = os.path.join(a.out_dir, a.arm + ".qhead.json")
    if os.path.exists(out_h) or os.path.exists(out_j):
        refuse("%s exists (a head is fitted once)" % out_h)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    files = {p.replace("\\", "/"): QM.sha256_file(p) for p in inputs}
    S = np.load(a.states, allow_pickle=False)
    P = np.load(a.pack, allow_pickle=False)
    Z = S["Z"]
    R = build_rows(S, P, a.hist)
    n_clips = R["counts"]["n_clips"]
    if n_clips <= a.held_out:
        refuse("%d clips cannot give %d held out" % (n_clips, a.held_out))
    held = R["clip"] >= (n_clips - a.held_out)
    fit_m = ~held
    banked = None
    if a.banked_head:
        bz = np.load(a.banked_head, allow_pickle=False)
        banked = {k: bz[k] for k in QM.TENSOR_KEYS}
    print("rows: %s | fit rows %d | held-out rows %d (clips %d..%d)" % (R["counts"], fit_m.sum(), held.sum(), n_clips - a.held_out, n_clips - 1), flush=True)
    torch.manual_seed(int(a.seed))
    probe = QM.QHead(Z.shape[1], a.hist)
    gc = make_grad_check(probe, Z, R["hist_idx"][fit_m][:a.batch], R["y"][fit_m][:a.batch], QM.Q_LEVEL, banked=banked, device=a.device)
    print("gradient check:", json.dumps(gc), flush=True)
    if not gc["pass"]:
        refuse("the gradient check failed: " + json.dumps(gc))
    head = fit_head(Z, R["hist_idx"][fit_m], R["y"][fit_m], a.seed, a.steps, a.batch, a.lr, a.wd, a.warmup, a.clip_grad, device=a.device, hist=a.hist,
                    log=lambda s: print(s, flush=True))
    qf = predict(head, Z, R["hist_idx"][fit_m], a.device)
    qh = predict(head, Z, R["hist_idx"][held], a.device)
    cov_fit, cov_held = coverage(qf, R["y"][fit_m]), coverage(qh, R["y"][held])
    extra = {}
    if a.lead:
        L = np.load(a.lead, allow_pickle=False)
        g_head = np.asarray(L["gap_m"], np.float64)[R["row"]]
        qall = np.empty(len(R["y"]))
        qall[fit_m], qall[held] = qf, qh
        used = np.minimum(g_head, qall)
        extra = {"fit_clips": {"share_used_above_label": float((used[fit_m] > R["y"][fit_m]).mean()), "mean_shift_m": float((g_head[fit_m] - used[fit_m]).mean())},
                 "held_out_clips": {"share_used_above_label": float((used[held] > R["y"][held]).mean()), "mean_shift_m": float((g_head[held] - used[held]).mean()),
                                    "share_head_above_label": float((g_head[held] > R["y"][held]).mean())}}
    h5 = bool(cov_held["share_q_above_label"] is not None and H5_LO <= cov_held["share_q_above_label"] <= H5_HI)
    QM.save_head(out_h, head)
    files_out = {out_h.replace("\\", "/"): QM.sha256_file(out_h)}
    rec = {"card": CARD, "arm": a.arm, "seed": a.seed, "tier": "T0 (a fit on logged frames; never driving performance)", "recipe": {k: getattr(a, k) for k in DEFAULT},
           "quantile": QM.Q_LEVEL, "hist": a.hist, "n_params": int(sum(p.numel() for p in head.parameters())), "counts": R["counts"], "n_fit_rows": int(fit_m.sum()),
           "n_held_out_rows": int(held.sum()), "n_held_out_clips": a.held_out, "fit_clip_max_index": int(R["clip"][fit_m].max()), "held_clip_min_index": int(R["clip"][held].min()), "n_fit_clips": int(n_clips - a.held_out), "coverage_fit": cov_fit, "coverage_held_out": cov_held,
           "h5_coverage_in_0.15_0.25": h5, "gradient_check": gc, "beside_lead_read": extra, "files_read_sha256": files, "head_file_sha256": files_out,
           "q4_only_train_files": bool(not [p for p in files if QM.is_eval_path(p)]), "states_record": {k: srec.get(k) for k in ("ckpt_sha256", "split", "lo", "hi", "step", "n_clips")},
           "loss_trace": head.train_loss_trace, "seconds": round(time.time() - t0, 1), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    QM.dump_json(out_j, rec)
    print("%s: fit rows %d held-out rows %d | coverage fit %.4f held-out %.4f | h5 %s | %.0f s" % (a.arm, rec["n_fit_rows"], rec["n_held_out_rows"], cov_fit["share_q_above_label"],
          cov_held["share_q_above_label"] if cov_held["n"] else float("nan"), h5, rec["seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
