#!/usr/bin/env python3
"""P5-C launch-gate probes that need the REAL model: G-LIVE (f) warm-start identity and G-LIVE (h) lever isolation (PREREG_P5C.md 10.2).  Library only; gate_checks.p5c_live_rows calls it.

(f) The arms fine-tune from P7's final checkpoint.  At step 0 (the build capture stops at the trainer's first `opt.zero_grad`, nothing trained) EVERY tensor of the built model must equal the
    base file's tensor bit for bit (dtype, shape, sha256 of the raw bytes), the optimiser must hold no state (fresh AdamW, card 3), no `ckpt_last.pt` may sit in --out (no resume), and the base
    FILE's sha256 must equal the profile's registered pin (P5C_BASE_SHA256).  All tensors identical at identical code implies identical trajectories and scores on any rows (the card's >= 64-row
    probe is a consequence, not a stronger test).  Deliberate-regression arms in gate_checks.instrumented_run (`warm_not_loaded`, `warm_perturbed`) must make the gate FAIL.
(h) Lever isolation on the real model, real loader rows: between C1 and C0 the only data difference is item[3]; the forward does not read it, so the score loss (fixed labels) and its gradient w.r.t. the
    scores are bit-identical, while the traj loss (the REAL wta_loss) differs; and a control (the table label replaced by the raw traj) reads EXACTLY 0.0.
Heavy imports (torch) are inside functions: importing this module reads nothing.
"""
from __future__ import annotations

import hashlib
import os


def tensor_digest(t) -> list:
    """[dtype, shape, sha256 of the raw bytes] of one tensor (CPU, contiguous)."""
    t = t.detach().cpu().contiguous()
    raw = t.reshape(-1).view(__import__("torch").uint8).numpy().tobytes() if t.numel() else b""
    return [str(t.dtype), list(t.shape), hashlib.sha256(raw).hexdigest()]


def file_sha256(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def step0_facts(model, opt, out_dir) -> dict:
    """What the trainer BUILT at step 0: a digest per state_dict tensor, the optimiser's state entries (0 = fresh) and whether `--out` already holds a resume file."""
    return {"tensors": {k: tensor_digest(v) for k, v in model.state_dict().items()},
            "opt_state_entries": len(opt.state), "ckpt_last_present": bool(out_dir) and os.path.exists(os.path.join(str(out_dir), "ckpt_last.pt"))}


def base_facts(path) -> dict:
    """The base file read the way `train.warm_start` reads it: FULL format = `model` (or the bare dict), PARTIAL = `model_partial`.  Digests per tensor + the FILE's sha256."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    partial = isinstance(sd, dict) and (sd.get("format") == "refe-partial-v1" or "model_partial" in sd)
    st = sd["model_partial"] if partial else (sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    return {"path": str(path), "format": "partial" if partial else "full", "file_sha256": file_sha256(path),
            "tensors": {k: tensor_digest(v) for k, v in st.items() if hasattr(v, "dtype")}}


def judge_warm_identity(step0, base, pin_sha256) -> list:
    """Failure strings ([] = pass).  `pin_sha256` is the profile's `init_from_sha256` (P5C_BASE_SHA256)."""
    if not isinstance(step0, dict) or "tensors" not in step0:
        return ["(f) no step-0 capture of the built model: the warm start was never observed"]
    if not isinstance(base, dict) or "tensors" not in base:
        return ["(f) the base checkpoint could not be read"]
    bad = []
    if not (isinstance(pin_sha256, str) and len(pin_sha256) == 64):
        bad.append("(f) the profile registers no base sha256 (P5C_BASE_SHA256 unfilled): the base is not bound")
    elif base["file_sha256"] != pin_sha256:
        bad.append(f"(f) the base file's sha256 {base['file_sha256'][:12]} is not the registered pin {pin_sha256[:12]}")
    built, ref = step0["tensors"], base["tensors"]
    if base["format"] == "full" and set(built) != set(ref):
        bad.append(f"(f) tensor names differ from the strict load: only in model {sorted(set(built) - set(ref))[:4]}, only in base {sorted(set(ref) - set(built))[:4]}")
    miss = sorted(set(ref) - set(built))
    if miss:
        bad.append(f"(f) base tensors absent from the built model: {miss[:4]}")
    diff = sorted(k for k in ref if k in built and built[k] != ref[k])
    if diff:
        bad.append(f"(f) {len(diff)} of {len(ref)} tensors differ from the base at step 0 (not bit-identical): {diff[:4]}")
    if step0.get("opt_state_entries") != 0:
        bad.append(f"(f) the optimiser holds state at step 0 ({step0.get('opt_state_entries')} entries): not a fresh AdamW (card 3)")
    if step0.get("ckpt_last_present"):
        bad.append("(f) --out already holds ckpt_last.pt: this start is a resume, not a fresh fine-tune")
    return bad


# ------------------------------------------------------------------------------------------------ (h) lever isolation
def _collate(items):
    import torch
    return [torch.stack([it[j] for it in items]) for j in range(len(items[0]))]


def _same(a, b) -> bool:
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def isolation_facts(model, ds_tab, ds_raw, idxs, yaw_wrap: bool = False, seed: int = 20261114) -> dict:
    """`ds_tab` and `ds_raw` are the SAME TargetBank with and without the table (ds_raw = ds_tab with `traj_label = None`: the flag-off path through the same `__getitem__`).  One model, fixed weights."""
    import torch
    import torch.nn.functional as F
    from model import wta_loss
    items_t, items_r = [ds_tab[i] for i in idxs], [ds_raw[i] for i in idxs]
    n = len(items_t[0])
    other_equal = all(_same(a[j], b[j]) for a, b in zip(items_t, items_r) for j in range(n) if j != 3)
    n_differs = sum(int(not _same(a[3], b[3])) for a, b in zip(items_t, items_r))
    bt, br = _collate(items_t), _collate(items_r)
    dev = next(model.parameters()).device
    was = model.training
    model.eval()

    def fwd(b):
        img, ego, goal, cal = b[0].to(dev), b[1].to(dev), b[2].to(dev), b[8]
        with torch.no_grad():
            traj, score = model(img, ego, goal, calib=cal if cal.numel() else None)[:2]
        return traj.float(), score.float()
    tr_t, sc_t = fwd(bt)
    tr_r, sc_r = fwd(br)
    model.train(was)
    g = torch.Generator().manual_seed(seed)
    lab = (torch.rand(sc_t.shape, generator=g) > 0.5).float().to(sc_t.device)

    def score_loss(sc):
        s = sc.detach().clone().requires_grad_(True)
        l_ = F.binary_cross_entropy_with_logits(s, lab)
        l_.backward()
        return float(l_.detach()), s.grad.clone()

    def traj_loss(tr, tgt):
        p = tr.detach().clone().requires_grad_(True)
        l_, _, _, _ = wta_loss(p, tgt.to(p.device), yaw_wrap=yaw_wrap, parts=True)
        l_.backward()
        return float(l_.detach()), p.grad.clone()
    ls_t, gs_t = score_loss(sc_t)
    ls_r, gs_r = score_loss(sc_r)
    lt_t, gt_t = traj_loss(tr_t, bt[3])
    lt_r, gt_r = traj_loss(tr_r, br[3])
    lt_c, gt_c = traj_loss(tr_t, br[3])                  # the control: the table label replaced by the raw traj on the SAME forward -> must equal the raw arm exactly
    return {"rows": len(idxs), "other_slots_equal": bool(other_equal), "item3_differs": int(n_differs),
            "forward_equal": bool(torch.equal(tr_t, tr_r) and torch.equal(sc_t, sc_r)),
            "score_loss_table": ls_t, "score_loss_raw": ls_r, "score_grad_diff_max": float((gs_t - gs_r).abs().max()),
            "traj_loss_table": lt_t, "traj_loss_raw": lt_r, "traj_loss_diff": lt_t - lt_r, "traj_grad_diff_max": float((gt_t - gt_r).abs().max()),
            "control_traj_loss_diff": lt_c - lt_r, "control_traj_grad_diff_max": float((gt_c - gt_r).abs().max())}


def judge_isolation(f) -> list:
    """Failure strings ([] = pass) for `isolation_facts`."""
    if not isinstance(f, dict) or "rows" not in f:
        return ["(h) the lever-isolation probe did not run"]
    bad = []
    if f["item3_differs"] < 1:
        bad.append("(h) no probed row has a table label that differs from the raw traj: the lever did not reach item[3] (or the rows probed carry no label)")
    if not f["other_slots_equal"]:
        bad.append("(h) the table changed a slot OTHER than item[3] (image, ego, goal, candidates or calib): the lever is not isolated")
    if not f["forward_equal"]:
        bad.append("(h) the model's forward differs between the table and raw batches: its inputs are not identical")
    if f["score_loss_table"] != f["score_loss_raw"] or f["score_grad_diff_max"] != 0.0:
        bad.append(f"(h) the score loss / its gradient moved with the table (loss {f['score_loss_table']!r} vs {f['score_loss_raw']!r}, grad diff {f['score_grad_diff_max']!r}): must be bit-identical")
    if f["traj_loss_diff"] == 0.0:             # the VALUE decides: the WTA loss is an L1, whose gradient is a sign and can be identical under a small shift (traj_grad_diff_max is recorded, not judged)
        bad.append(f"(h) the traj loss does not differ under the table (diff {f['traj_loss_diff']!r}): the label never reached the loss")
    if f["control_traj_loss_diff"] != 0.0 or f["control_traj_grad_diff_max"] != 0.0:
        bad.append(f"(h) the probe's own control (raw label on the same forward) reads {f['control_traj_loss_diff']!r} / {f['control_traj_grad_diff_max']!r}, must be EXACTLY 0.0: the probe cannot read zero")
    return bad


def pick_rows(ds_tab, k: int = 2) -> list:
    """The first `k` row indices whose table label differs from the row's raw traj (sorted), or [] if none."""
    out = []
    for i in sorted(ds_tab.traj_label or {}):
        if ds_tab.traj_label[i] != ds_tab.rows[i]["traj"]:
            out.append(i)
            if len(out) == k:
                break
    return out
