#!/usr/bin/env python
"""qhead_fit_n1.py -- note PREREG_V8F_PROOF_ADD42_N1, sections 2 and 4: per checkpoint a low-quantile gap head (pinball loss at 0.20) on the banked level-1 states of the TRAIN probe clips,
fitted so that the quantile has to hold on clips the fit never saw. Second version of qhead_fit.py (same head, same inputs, same loss, same optimiser recipe).

  python qhead_fit_n1.py --arm LS_s0 --seed 0 --states <ARM>_train.npz --pack train_pack.npz --out-dir DIR [--lead <ARM>_train_lead.npz] [--banked-head <ARM>_train_lead.npz.head.npz]
                         [--fit-clips 800] [--select-clips 200] [--check-clips 200] [--select-every 250] [--steps 19164] [--batch 64] [--device cpu]

Three clip sets in manifest order (the order of the clip list of the state file, which the pack must share): fit = clips 0..fit-1, select = the next select clips, check = the next check clips.
  * the gradient steps use the fit rows only;
  * every --select-every steps and at the last step the pinball loss at 0.20 on the SELECT rows is read; the parameters of the step with the lowest value are kept (ties: the earlier step);
  * one constant c = the 0.80 quantile of (q_head - label) on the SELECT rows is taken off the output: the used output is q_head - c. It is folded into the output bias of the saved head (in the
    head's own units, c / 50), so the unchanged rollout tool loads the file as it loads a first-run head. The fit refuses to finish if the saved head does not reproduce q_head - c within 1e-5 m.
  * the CHECK rows enter no gradient step, no selection and not the shift: they give coverage and pinball only (control h5), and the record lists the clip ranges of the three sets (control h8).
No evaluation file is read (control h4). Writes <ARM>.qhead.npz (the head) and <ARM>.qhead.json (the record, written last). Tier T0.
Exit: 0 fit written (h5 may read False: the record says so), 3 refused (SystemExit text) or a gradient check failed.
"""
import argparse
import copy
import hashlib
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

CARD = "PREREG_V8F_PROOF_ADD42_N1"
H5_LO, H5_HI = 0.15, 0.25
SHIFT_Q = 0.80
FOLD_TOL_M = 1e-5
DEFAULT = {"steps": 19164, "batch": 64, "lr": 3e-4, "wd": 1e-2, "warmup": 200, "clip_grad": 1.0}
SETS = {"fit_clips": 800, "select_clips": 200, "check_clips": 200}
SELECT_EVERY = 250


def refuse(msg):
    raise SystemExit("REFUSED: " + msg)


def pinball(pred, y, q):
    """The pinball loss at level q of a prediction `pred` of the q-quantile of y (mean over the rows)."""
    e = y - pred
    return torch.maximum(q * e, (q - 1.0) * e).mean()


def pinball_np(pred, y, q=QM.Q_LEVEL):
    e = np.asarray(y, np.float64) - np.asarray(pred, np.float64)
    return float(np.maximum(q * e, (q - 1.0) * e).mean())


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


def clip_sets(n_clips, fit_clips, select_clips, check_clips):
    """The three sets as half-open ranges of clip indices in manifest order: fit [0, a), select [a, b), check [b, c). Refuses when the file has fewer clips than the sum."""
    for k, v in (("fit", fit_clips), ("select", select_clips), ("check", check_clips)):
        if int(v) <= 0:
            refuse("--%s-clips must be positive, got %s" % (k, v))
    a, b, c = int(fit_clips), int(fit_clips) + int(select_clips), int(fit_clips) + int(select_clips) + int(check_clips)
    if n_clips < c:
        refuse("the state file has %d clips, fewer than %d fit + %d select + %d check = %d" % (n_clips, fit_clips, select_clips, check_clips, c))
    return {"fit": [0, a], "select": [a, b], "check": [b, c]}


def set_mask(clip, rng_):
    return (clip >= rng_[0]) & (clip < rng_[1])


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


def predict(head, Z, hist_idx, device="cpu", chunk=4096):
    out = np.empty(len(hist_idx), np.float64)
    with torch.no_grad():
        for i in range(0, len(hist_idx), chunk):
            x = torch.from_numpy(np.asarray(Z[hist_idx[i:i + chunk]], np.float32)).to(device)
            out[i:i + chunk] = head(x)[:, 0, 0].double().cpu().numpy() * QM.GAP_SCALE_M
    return out


def eval_select(head, Z, sel_idx, sel_y, device="cpu"):
    """The selection value: the pinball loss at 0.20 on the SELECT rows, in metres."""
    return pinball_np(predict(head, Z, sel_idx, device), sel_y)


def fit_head(Z, hist_idx, y, seed, steps, batch, lr, wd, warmup, clip_grad, sel_idx, sel_y, select_every=SELECT_EVERY, q=QM.Q_LEVEL, device="cpu", hist=QM.HIST, log=None, read=None):
    """Fit a fresh QHead (torch seed = the base seed) on the FIT rows (hist_idx, y). Z [n, D] states, hist_idx [R, hist] rows of Z, y [R] label gap in m.
    Every `select_every` steps and at the last step the SELECT rows (sel_idx, sel_y) give the selection value; the parameters of the lowest value are kept (ties: the earlier step).
    -> head (eval mode, parameters of the selected step) with .train_loss_trace, .select_curve [[step, value_m]], .selected_step."""
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
    tl, curve = [], []
    best_val, best_step, best_sd = None, None, None
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
        done = step + 1
        if step % 500 == 0 or step == steps - 1:
            tl.append((step, float(loss.detach())))
            if log:
                log("step %d loss %.5f lr %.2e" % (step, float(loss.detach()), opt.param_groups[0]["lr"]))
        if done % int(select_every) == 0 or done == int(steps):
            head.eval()
            val = float(eval_select(head, Z, sel_idx, sel_y, device))                    # SELECT: the value read on the select rows
            head.train()
            curve.append([done, val])
            if best_val is None or val < best_val:
                best_val, best_step, best_sd = val, done, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_sd)
    head.eval()
    head.train_loss_trace = tl
    head.select_curve = curve
    head.selected_step = int(best_step)
    return head


def shift_value(q_select, y_select):
    """c: the 0.80 quantile of (q_head - label) on the select rows, so that the share of select rows with q_head - c above the label is 0.20."""
    return float(np.quantile(np.asarray(q_select, np.float64) - np.asarray(y_select, np.float64), SHIFT_Q))


def fold_shift(head, c):
    """Write the shift into the head's own output bias (units of gap / 50 m): the head then outputs q_head - c."""
    with torch.no_grad():
        head.mlp[2].bias -= float(c) / QM.GAP_SCALE_M


def coverage(q_pred, y):
    """Share of rows with the output above the label gap (the card's coverage), and the mean amount (q - y)."""
    if len(y) == 0:
        return {"n": 0, "share_q_above_label": None, "mean_q_minus_label_m": None}
    return {"n": int(len(y)), "share_q_above_label": float((q_pred > y).mean()), "mean_q_minus_label_m": float((q_pred - y).mean())}


def sha_text(path):
    return QM.sha256_file(path)


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
    for k, v in SETS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--select-every", type=int, default=SELECT_EVERY)
    ap.add_argument("--hist", type=int, default=QM.HIST)
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
    if a.select_every <= 0:
        refuse("--select-every must be positive")
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    files = {p.replace("\\", "/"): QM.sha256_file(p) for p in inputs}
    S = np.load(a.states, allow_pickle=False)
    P = np.load(a.pack, allow_pickle=False)
    Z = S["Z"]
    R = build_rows(S, P, a.hist)
    n_clips = R["counts"]["n_clips"]
    CS = clip_sets(n_clips, a.fit_clips, a.select_clips, a.check_clips)
    m_fit, m_sel, m_chk = (set_mask(R["clip"], CS[k]) for k in ("fit", "select", "check"))
    if min(m_fit.sum(), m_sel.sum(), m_chk.sum()) == 0:
        refuse("a clip set has no usable row: fit %d select %d check %d" % (m_fit.sum(), m_sel.sum(), m_chk.sum()))
    banked = None
    if a.banked_head:
        bz = np.load(a.banked_head, allow_pickle=False)
        banked = {k: bz[k] for k in QM.TENSOR_KEYS}
    print("rows: %s | fit %d select %d check %d | sets %s" % (R["counts"], m_fit.sum(), m_sel.sum(), m_chk.sum(), json.dumps(CS)), flush=True)
    torch.manual_seed(int(a.seed))
    probe = QM.QHead(Z.shape[1], a.hist)
    gc = make_grad_check(probe, Z, R["hist_idx"][m_fit][:a.batch], R["y"][m_fit][:a.batch], QM.Q_LEVEL, banked=banked, device=a.device)
    print("gradient check:", json.dumps(gc), flush=True)
    if not gc["pass"]:
        refuse("the gradient check failed: " + json.dumps(gc))
    # the rows of each use: the gradient steps see the FIT rows, the selection and the shift the SELECT rows; the CHECK rows nowhere
    fit_idx, fit_y = R["hist_idx"][m_fit], R["y"][m_fit]
    sel_idx, sel_y = R["hist_idx"][m_sel], R["y"][m_sel]
    chk_idx, chk_y = R["hist_idx"][m_chk], R["y"][m_chk]
    head = fit_head(Z, fit_idx, fit_y, a.seed, a.steps, a.batch, a.lr, a.wd, a.warmup, a.clip_grad, sel_idx, sel_y, a.select_every, device=a.device, hist=a.hist,
                    log=lambda s: print(s, flush=True))
    q0 = {"fit": predict(head, Z, fit_idx, a.device), "select": predict(head, Z, sel_idx, a.device), "check": predict(head, Z, chk_idx, a.device)}
    c = shift_value(q0["select"], sel_y)
    fold_shift(head, c)
    QM.save_head(out_h, head, extra={"shift_c_m": np.float64(c), "selected_step": np.int64(head.selected_step)})
    saved = QM.load_head(out_h)
    ys = {"fit": fit_y, "select": sel_y, "check": chk_y}
    idxs = {"fit": fit_idx, "select": sel_idx, "check": chk_idx}
    used = {k: predict(saved, Z, idxs[k], a.device) for k in ys}
    fold_err = max(float(np.max(np.abs(used[k] - (q0[k] - c)))) for k in ys)
    if not fold_err <= FOLD_TOL_M:
        refuse("the saved head does not reproduce q_head - c: max abs difference %.3g m (bar %g m)" % (fold_err, FOLD_TOL_M))
    cov = {k: coverage(used[k], ys[k]) for k in ys}
    pin = {k: pinball_np(used[k], ys[k]) for k in ys}
    extra = {}
    if a.lead:
        L = np.load(a.lead, allow_pickle=False)
        g_head = np.asarray(L["gap_m"], np.float64)[R["row"]]
        gm = {"fit": g_head[m_fit], "select": g_head[m_sel], "check": g_head[m_chk]}
        for k in ys:
            gu = np.minimum(gm[k], used[k])
            extra[k + "_clips"] = {"share_used_above_label": float((gu > ys[k]).mean()), "mean_shift_m": float((gm[k] - gu).mean()), "share_head_above_label": float((gm[k] > ys[k]).mean())}
    h5 = bool(cov["check"]["share_q_above_label"] is not None and H5_LO <= cov["check"]["share_q_above_label"] <= H5_HI)
    files_out = {out_h.replace("\\", "/"): QM.sha256_file(out_h)}
    rng_of = lambda cl: [int(cl.min()), int(cl.max())]                                  # noqa: E731
    rec = {"card": CARD, "arm": a.arm, "seed": a.seed, "tier": "T0 (a fit on logged frames; never driving performance)", "recipe": {k: getattr(a, k) for k in DEFAULT},
           "quantile": QM.Q_LEVEL, "hist": a.hist, "n_params": int(sum(p.numel() for p in head.parameters())), "counts": R["counts"],
           "clip_sets": CS, "clip_sets_convention": "half-open ranges [lo, hi) of clip indices of the state file, manifest order", "n_clips_in_file": n_clips,
           "n_fit_rows": int(m_fit.sum()), "n_select_rows": int(m_sel.sum()), "n_check_rows": int(m_chk.sum()),
           "clip_index_range_of_rows_used": {"gradient_steps": rng_of(R["clip"][m_fit]), "selection": rng_of(R["clip"][m_sel]), "shift": rng_of(R["clip"][m_sel]), "check_rows_only_coverage": rng_of(R["clip"][m_chk])},
           "select_every": a.select_every, "select_curve": head.select_curve, "selected_step": head.selected_step, "last_step": int(a.steps), "select_curve_metric": "pinball loss at 0.20 on the select rows, metres, output before the shift",
           "shift_c_m": c, "shift_quantile": SHIFT_Q, "fold_check_max_abs_m": fold_err, "fold_check_bar_m": FOLD_TOL_M,
           "coverage_fit": cov["fit"], "coverage_select": cov["select"], "coverage_check": cov["check"],
           "pinball_fit_m": pin["fit"], "pinball_select_m": pin["select"], "pinball_check_m": pin["check"],
           "h5_check_coverage_in_0.15_0.25": h5, "gradient_check": gc, "beside_lead_read": extra, "files_read_sha256": files, "head_file_sha256": files_out,
           "q4_only_train_files": bool(not [p for p in files if QM.is_eval_path(p)]), "states_record": {k: srec.get(k) for k in ("ckpt_sha256", "split", "lo", "hi", "step", "n_clips")},
           "tool_sha256": {"qhead_fit_n1.py": sha_text(os.path.abspath(__file__)), "qhead_model.py": sha_text(os.path.join(HERE, "qhead_model.py"))},
           "loss_trace": head.train_loss_trace, "seconds": round(time.time() - t0, 1), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    QM.dump_json(out_j, rec)
    print("%s: rows fit %d select %d check %d | selected step %d of %d | c %.4f m | coverage fit %.4f select %.4f check %.4f | h5 %s | %.0f s" % (
        a.arm, rec["n_fit_rows"], rec["n_select_rows"], rec["n_check_rows"], rec["selected_step"], a.steps, c, cov["fit"]["share_q_above_label"], cov["select"]["share_q_above_label"],
        cov["check"]["share_q_above_label"], h5, rec["seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
