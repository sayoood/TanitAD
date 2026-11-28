"""P0 -- detection is MEASURED (SPEC_REFCV7 §14 A9, §15 A10): the refcv7 box-head metrics on VIS-1 positives with
IGNORE as DontCare, for BOTH slot heads (h in {box3d, agent}), on the LAST decoder layer (the one inference reads).

Registered BEFORE any refcv7 box number exists. refcv6 logged 224 metric keys and 0 detection metrics in 38,000
steps; the box-head audit's ``raw/LOGGING_SPEC_BOX.md`` names the keys that would have shown it (its §1 train row and
§2 eval row are emitted here under its own names), and A9 P0 adds the AP grid.

⭐ THE MATCHING RULE (declared; PREREG §4 / A9 "a detection greedy-matched to an IGNORE row leaves the PR count"):
per window, detections in DESCENDING presence order each take the NEAREST UNTAKEN row of POSITIVE ∪ IGNORE within
the distance -- ``box3d_head.box3d_match_rows(use_z=False)`` on that union, never re-implemented. Taking a POSITIVE is
a TP, taking an IGNORE row is DontCare (removed from every count), taking nothing is a FP. Per class (nuScenes):
detections whose argmax class is ``c`` against POSITIVES of class ``c`` ∪ ALL IGNORE rows.

⭐ THE DECISION RULE (A9, A10 §15.1): a detection is a slot with ``sigmoid(presence_logit) >= 0.5``
(``slot_presence.DETECTION_GATE``). P/R/F1 "@gate" are at 2 m. ⛔ NO calibration map (A10): nothing here converts
the focal head's score into a probability, so no ``sum_phat`` / ECE / G-LIVE-COUNT key is emitted.

⭐ THE BAND RULE (declared): bands are FORWARD distance ``cx`` (ego frame), [0, 20) / [20, 40) / [40, 60] m -- they
partition the decode box. A TP counts in its GT's band, a FP in its own ``cx``'s band (outside [0, 60] m only in
``all``). One greedy pass per window over all bands.

⛔ POOLED, NEVER A MEAN OF PER-BATCH RATIOS (LOGGING_SPEC §2): every metric is computed from the pooled rows of all
windows; ``tests/test_refcv7_box_head.py`` pins (tp, n_conf) = (1, 1) and (0, 9) -> precision 0.10, not 0.5.
⛔ AP is ``box3d_head.ap_from_rows``; the VIS-1 split is ``tanitad.data.vis1.vis1_split`` (the loss's own function).

KEYS (the Watch contract, :func:`metric_keys` / :func:`train_row_keys`):
  eval row, headline -- ``eval_{h}_prec@gate``, ``eval_{h}_rec@gate``, ``eval_{h}_f1@gate``, ``eval_{h}_conf_ratio``,
  ``eval_{h}_conf_ratio_alarm`` (1 outside [0.5, 1.5], A10 §15.3), ``eval_{h}_ap2m``, ``eval_{h}_auroc_matched``,
  ``eval_{h}_auroc_objectness``, ``eval_{h}_cls_acc_tp``, ``eval_{h}_cls_acc_tp_priorcorr`` (INFORMATIVE),
  ``eval_{h}_centre_err_p50``, ``eval_{h}_rec@gate_<cls>``, ``eval_{h}_npos_<cls>`` and the census
  ``eval_{h}_{n_conf, tp@gate, n_pos, n_ignore, n_dropped_hidden, n_ignore_masked_slots, n_windows}``;
  eval row, the A9 grid -- ``eval_{h}_det_ap{0p5,1,2,4}_{all|<cls>}_{all|0_20|20_40|40_60}``,
  ``eval_{h}_det_map{thr}_{band}``, ``eval_{h}_det_npos_{cls}_{band}``;
  calibration pass (INFORMATIVE, A10 §15.3) -- ``eval_{h}_calib_pr_gate``, ``eval_{h}_calib_prec``,
  ``eval_{h}_calib_rec``, ``eval_{h}_calib_n_pos``, ``eval_{h}_calib_n_windows``;
  train row -- ``{h}_n_pos``, ``{h}_n_ignore``, ``{h}_n_dropped_hidden``, ``{h}_n_ignore_masked_slots``,
  ``{h}_n_conf``, ``{h}_conf_ratio``, ``{h}_tp@gate``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["DIST_THRESHOLDS_M", "THR_KEY", "BANDS_X_M", "HEADS", "CONF_RATIO_BAND", "window_packs", "greedy_rows",
           "summarise", "metric_keys", "train_row_keys", "train_row_key_names", "calib_keys", "calib_key_names",
           "pr_equal_gate", "auroc", "OBJECTNESS_RADIUS_M", "PR_DIST_M", "CALIB_WINDOWS_FILE",
           "load_calib_windows", "calib_indices"]

DIST_THRESHOLDS_M: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
THR_KEY: dict = {0.5: "0p5", 1.0: "1", 2.0: "2", 4.0: "4"}
BANDS_X_M: tuple[tuple[str, float, float], ...] = (("0_20", 0.0, 20.0), ("20_40", 20.0, 40.0),
                                                   ("40_60", 40.0, 60.0))
HEADS: tuple[str, ...] = ("box3d", "agent")
OBJECTNESS_RADIUS_M: float = 2.0
PR_DIST_M: float = 2.0
#: A10 §15.3: the Watch alarm band on confident / VIS-1 positives (the audit's G-LIVE-GATE, tightened).
CONF_RATIO_BAND: tuple[float, float] = (0.5, 1.5)
_NAN = float("nan")


def _classes():
    from tanitad.models.agent_slots import AGENT_CLASSES
    return tuple(AGENT_CLASSES)


def _gate() -> float:
    from tanitad.models.slot_presence import DETECTION_GATE
    return float(DETECTION_GATE)


def _band_of(x: float) -> str | None:
    for name, lo, hi in BANDS_X_M:
        if lo <= x < hi or (name == BANDS_X_M[-1][0] and x == hi):
            return name
    return None


def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC with AVERAGE ranks for ties (a constant score reads EXACTLY 0.5) -- the audit's
    ``bha_analyze.auroc``, so refcv6's banked AUROC and refcv7's are the same statistic."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return _NAN
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    ranks = (start + (cnt + 1) / 2.0)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------------------------------------- #
# per-window packs                                                                                           #
# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def window_packs(pred: dict, tgt_real: dict, vis: dict, *, presence_cost: str = "focal",
                 match: dict | None = None, episode_ids=None, cls_weight=None,
                 with_match: bool = True) -> list:
    """One pack per batch element: the LAST layer's slots, the VIS-1 labels, the Hungarian matched set.

    ``tgt_real``: the target block, ``valid`` = every REAL row; ``vis`` = ``{"n_full", "n_vis", "known"}``.
    ``match``: the head's own last-layer training match on the POSITIVES (reused), else recomputed with
    ``presence_cost``. ``cls_weight`` [C]: the STAMPED class weights, for the INFORMATIVE prior-corrected argmax
    ``argmax(logit - ln w_c)`` (A10 §15.3); ``None`` -> the raw argmax twice."""
    from tanitad.data.vis1 import vis1_split
    from tanitad.models.agent_slots import match_slots
    from tanitad.models.slot_presence import ignore_presence_weight
    sp = vis1_split(tgt_real, n_full=vis["n_full"], n_vis=vis["n_vis"], vis_known=vis["known"])
    pos, ign = sp["pos"]["valid"], sp["ignore"]
    hidden = sp["in_filter"] & vis["known"].to(pos.device).bool() & (sp["vis_frac"] < 0.05)
    if match is None and with_match:
        match = match_slots(pred, sp["pos"], presence_cost=presence_cost)
    if match is None:              # train-row census only: no AUROC / pair error is read from this pack
        B0 = int(pred["presence_logit"].shape[0])
        match = {"rows": [torch.zeros(0, dtype=torch.long)] * B0,
                 "cols": [torch.zeros(0, dtype=torch.long)] * B0}
    logit = pred["presence_logit"].detach().float()
    box = pred["box"].detach().float()
    cl = pred["cls_logits"].detach().float()
    cls = cl.argmax(-1)
    if cls_weight is not None:
        lw = torch.log(torch.as_tensor(cls_weight, dtype=torch.float32, device=cl.device).clamp_min(1e-12))
        cls_corr = (cl - lw).argmax(-1)
    else:
        cls_corr = cls
    exempt_w = ignore_presence_weight(box, tgt_real["box"].float(), ign)
    B, N = int(logit.shape[0]), int(logit.shape[1])
    gt_box = tgt_real["box"].detach().float()
    gt_cls = tgt_real["cls"].detach()
    eps = (episode_ids.detach().reshape(-1).tolist() if torch.is_tensor(episode_ids)
           else (list(episode_ids) if episode_ids is not None else [None] * B))
    out = []
    for b in range(B):
        matched = np.zeros(N, dtype=bool)
        r, c = match["rows"][b], match["cols"][b]
        err = size_err = z_err = np.zeros(0)
        if r.numel():
            matched[r.cpu().numpy()] = True
            rr, cc = r.to(box.device), c.to(gt_box.device)
            d = box[b][rr][:, :2] - gt_box[b][cc][:, :2]
            err = d.norm(dim=-1).cpu().numpy().astype(np.float64)
            size_err = (box[b][rr][:, 2:4] - gt_box[b][cc][:, 2:4]).abs().sum(-1).cpu().numpy().astype(np.float64)
            if "cz" in pred and tgt_real.get("cz") is not None and tgt_real.get("zh_mask") is not None:
                zm = tgt_real["zh_mask"][b][cc].bool()
                dz = (pred["cz"].detach().float()[b][rr] - tgt_real["cz"].float()[b][cc]).abs()
                z_err = dz[zm].cpu().numpy().astype(np.float64)
        exempt = (exempt_w[b].cpu().numpy() == 0) & ~matched
        v = tgt_real["valid"][b].cpu().numpy().astype(bool)
        out.append({"ep": eps[b], "logit": logit[b].cpu().numpy().astype(np.float32),
                    "xy": box[b, :, :2].cpu().numpy().astype(np.float32),
                    "cls": cls[b].cpu().numpy().astype(np.int16),
                    "cls_corr": cls_corr[b].cpu().numpy().astype(np.int16),
                    "matched": matched, "exempt": exempt, "pair_err": err, "pair_size_err": size_err,
                    "pair_z_err": z_err,
                    "gt_xy": gt_box[b][v][:, :2].cpu().numpy().astype(np.float32),
                    "gt_cls": gt_cls[b][v].cpu().numpy().astype(np.int16),
                    "pos": pos[b][v].cpu().numpy().astype(bool), "ign": ign[b][v].cpu().numpy().astype(bool),
                    "hidden": hidden[b][v].cpu().numpy().astype(bool)})
    return out


def greedy_rows(pk: dict, dist_m: float, cls: int | None = None, min_conf: float | None = None) -> list:
    """``[(conf, kind, band, slot, gt)]`` for one window: kind 1 = TP, 0 = FP, -1 = DontCare.

    Calls ``box3d_head.box3d_match_rows`` on POSITIVE(class-filtered) ∪ IGNORE. ``min_conf`` restricts the ranked
    detections to ``conf >= min_conf`` -- exact for those slots, because greedy matching is score-ordered."""
    from tanitad.models.box3d_head import box3d_match_rows
    p = 1.0 / (1.0 + np.exp(-pk["logit"].astype(np.float64)))
    keep = np.ones(len(p), bool) if cls is None else (pk["cls"] == int(cls))
    if min_conf is not None:
        keep &= p >= float(min_conf)
    sel = np.nonzero(keep)[0]
    if sel.size == 0:
        return []
    keep_pos = pk["pos"] if cls is None else (pk["pos"] & (pk["gt_cls"] == int(cls)))
    gi = np.nonzero(keep_pos | pk["ign"])[0]
    A = int(gi.size)
    gxy = pk["gt_xy"][gi] if A else np.zeros((0, 2), np.float32)
    pred = {"box": torch.from_numpy(np.concatenate([pk["xy"][sel], np.ones((sel.size, 2), np.float32)], 1))[None],
            "presence_logit": torch.from_numpy(pk["logit"][sel])[None]}
    tgt = {"box": torch.from_numpy(np.concatenate([gxy, np.ones((A, 2), np.float32)], 1))[None],
           "valid": torch.ones(1, A, dtype=torch.bool)}
    per_elem, _ = box3d_match_rows(pred, tgt, dist_thresh_m=float(dist_m), use_z=False, score="presence")
    rows = []
    for (conf, hit, i, j) in per_elem[0]:
        s = int(sel[i])
        if hit:
            g = int(gi[j])
            if keep_pos[g]:
                rows.append((float(conf), 1, _band_of(float(pk["gt_xy"][g, 0])), s, g))
            else:
                rows.append((float(conf), -1, None, s, g))
        else:
            rows.append((float(conf), 0, _band_of(float(pk["xy"][s, 0])), s, -1))
    return rows


def _ap(rows, n_gt: int) -> float:
    from tanitad.models.box3d_head import ap_from_rows
    r = [(c, k) for (c, k, *_rest) in rows if k >= 0]
    return float(ap_from_rows(r, int(n_gt))["ap"]) if n_gt > 0 else _NAN


def _div(a, b) -> float:
    return (float(a) / float(b)) if b else _NAN


# --------------------------------------------------------------------------------------------------------- #
# the key contract                                                                                          #
# --------------------------------------------------------------------------------------------------------- #
def metric_keys(head: str) -> list:
    """Every eval-row key :func:`summarise` emits for ``head``."""
    C = _classes()
    bands = ["all"] + [b[0] for b in BANDS_X_M]
    ks = [f"eval_{head}_{k}" for k in ("prec@gate", "rec@gate", "f1@gate", "conf_ratio", "conf_ratio_alarm",
                                       "ap2m", "auroc_matched", "auroc_objectness", "cls_acc_tp",
                                       "cls_acc_tp_priorcorr", "centre_err_p50", "n_conf", "tp@gate", "n_pos",
                                       "n_ignore", "n_dropped_hidden", "n_ignore_masked_slots", "n_windows")]
    ks += [f"eval_{head}_rec@gate_{c}" for c in C] + [f"eval_{head}_npos_{c}" for c in C]
    for t in DIST_THRESHOLDS_M:
        for c in ["all", *C]:
            for b in bands:
                ks.append(f"eval_{head}_det_ap{THR_KEY[t]}_{c}_{b}")
        for b in bands:
            ks.append(f"eval_{head}_det_map{THR_KEY[t]}_{b}")
    for c in ["all", *C]:
        for b in bands:
            ks.append(f"eval_{head}_det_npos_{c}_{b}")
    return ks


def train_row_key_names(head: str) -> list:
    return [f"{head}_{k}" for k in ("n_pos", "n_ignore", "n_dropped_hidden", "n_ignore_masked_slots", "n_conf",
                                    "conf_ratio", "tp@gate")]


def calib_key_names(head: str) -> list:
    return [f"eval_{head}_calib_{k}" for k in ("pr_gate", "prec", "rec", "n_pos", "n_windows")]


def _census(packs, head: str, prefix: str, gate: float) -> dict:
    rows = [r for pk in packs for r in greedy_rows(pk, PR_DIST_M, None, min_conf=gate)]
    tp = sum(1 for r in rows if r[1] == 1)
    n_conf = sum(1 for r in rows if r[1] >= 0)                  # DontCare excluded (A10 §15.3)
    n_pos = int(sum(int(pk["pos"].sum()) for pk in packs))
    return {f"{prefix}{head}_n_pos": float(n_pos),
            f"{prefix}{head}_n_ignore": float(sum(int(pk["ign"].sum()) for pk in packs)),
            f"{prefix}{head}_n_dropped_hidden": float(sum(int(pk["hidden"].sum()) for pk in packs)),
            f"{prefix}{head}_n_ignore_masked_slots": float(sum(int(pk["exempt"].sum()) for pk in packs)),
            f"{prefix}{head}_n_conf": float(n_conf), f"{prefix}{head}_tp@gate": float(tp),
            f"{prefix}{head}_conf_ratio": _div(n_conf, n_pos)}, rows


def train_row_keys(packs: list, head: str, *, gate: float | None = None) -> dict:
    """LOGGING_SPEC §1, the cheap per-batch census (greedy matching over CONFIDENT slots only)."""
    out, _rows = _census(packs, head, "", _gate() if gate is None else float(gate))
    return out


def summarise(packs: list, head: str, *, gate: float | None = None) -> dict:
    """Every eval-row key (:func:`metric_keys`) for one head over the POOLED packs. NaN = undefined, never 0."""
    C = _classes()
    g = _gate() if gate is None else float(gate)
    out, rows_g = _census(packs, head, "eval_", g)
    tp, n_conf, n_pos = out[f"eval_{head}_tp@gate"], out[f"eval_{head}_n_conf"], out[f"eval_{head}_n_pos"]
    prec, rec = _div(tp, n_conf), _div(tp, n_pos)
    out[f"eval_{head}_prec@gate"] = prec
    out[f"eval_{head}_rec@gate"] = rec
    out[f"eval_{head}_f1@gate"] = (2 * prec * rec / (prec + rec)
                                   if (prec == prec and rec == rec and (prec + rec) > 0) else _NAN)
    cr = out[f"eval_{head}_conf_ratio"]
    lo, hi = CONF_RATIO_BAND
    out[f"eval_{head}_conf_ratio_alarm"] = (_NAN if cr != cr else (0.0 if lo <= cr <= hi else 1.0))
    out[f"eval_{head}_n_windows"] = float(len(packs))
    # per class, at the gate (recall) -- the TP's GT class decides, the count beside it
    tp_by = {c: 0 for c in range(len(C))}
    npos_by = {c: 0 for c in range(len(C))}
    for pk in packs:
        for gc, p in zip(pk["gt_cls"].tolist(), pk["pos"].tolist()):
            if p and 0 <= int(gc) < len(C):
                npos_by[int(gc)] += 1
    acc_raw, acc_corr, n_acc = 0, 0, 0
    for pk in packs:
        for r in greedy_rows(pk, PR_DIST_M, None, min_conf=g):
            if r[1] != 1:
                continue
            gc = int(pk["gt_cls"][r[4]])
            if 0 <= gc < len(C):
                tp_by[gc] += 1
                n_acc += 1
                acc_raw += int(int(pk["cls"][r[3]]) == gc)
                acc_corr += int(int(pk["cls_corr"][r[3]]) == gc)
    for ci, c in enumerate(C):
        out[f"eval_{head}_rec@gate_{c}"] = _div(tp_by[ci], npos_by[ci])
        out[f"eval_{head}_npos_{c}"] = float(npos_by[ci])
    out[f"eval_{head}_cls_acc_tp"] = _div(acc_raw, n_acc)
    out[f"eval_{head}_cls_acc_tp_priorcorr"] = _div(acc_corr, n_acc)
    errs = np.concatenate([pk["pair_err"] for pk in packs]) if packs else np.zeros(0)
    out[f"eval_{head}_centre_err_p50"] = float(np.median(errs)) if errs.size else _NAN
    # the A9 grid: AP per threshold x class x band, the class-mean mAP, and the counts
    bands = ["all"] + [b[0] for b in BANDS_X_M]
    ncls = {"all": None, **{c: i for i, c in enumerate(C)}}
    npos = {(c, b): 0 for c in ncls for b in bands}
    for pk in packs:
        for gxy, gc, p in zip(pk["gt_xy"], pk["gt_cls"], pk["pos"]):
            if not p:
                continue
            bd = _band_of(float(gxy[0]))
            for c, ci in ncls.items():
                if ci is None or int(gc) == ci:
                    npos[(c, "all")] += 1
                    if bd is not None:
                        npos[(c, bd)] += 1
    for (c, b), v in npos.items():
        out[f"eval_{head}_det_npos_{c}_{b}"] = float(v)
    for t in DIST_THRESHOLDS_M:
        cls_ap = {b: [] for b in bands}
        for c, ci in ncls.items():
            rows = [r for pk in packs for r in greedy_rows(pk, t, ci)]
            for b in bands:
                rb = rows if b == "all" else [r for r in rows if r[2] == b]
                ap = _ap(rb, npos[(c, b)])
                out[f"eval_{head}_det_ap{THR_KEY[t]}_{c}_{b}"] = ap
                if ci is not None and npos[(c, b)] > 0:
                    cls_ap[b].append(ap)
        for b in bands:
            out[f"eval_{head}_det_map{THR_KEY[t]}_{b}"] = float(np.mean(cls_ap[b])) if cls_ap[b] else _NAN
    out[f"eval_{head}_ap2m"] = out[f"eval_{head}_det_ap2_all_all"]
    # AUROC over the slots the presence loss scores (IGNORE-exempt slots excluded)
    S, M, O = [], [], []
    for pk in packs:
        keep = ~pk["exempt"]
        p = 1.0 / (1.0 + np.exp(-pk["logit"].astype(np.float64)))
        S.append(p[keep])
        M.append(pk["matched"][keep])
        pos_xy = pk["gt_xy"][pk["pos"]]
        if len(pos_xy):
            d = np.sqrt(((pk["xy"][:, None, :].astype(np.float64) - pos_xy[None].astype(np.float64)) ** 2).sum(-1))
            obj = d.min(1) <= OBJECTNESS_RADIUS_M
        else:
            obj = np.zeros(len(p), bool)
        O.append(obj[keep])
    if S:
        out[f"eval_{head}_auroc_matched"] = auroc(np.concatenate(S), np.concatenate(M))
        out[f"eval_{head}_auroc_objectness"] = auroc(np.concatenate(S), np.concatenate(O))
    else:
        out[f"eval_{head}_auroc_matched"] = out[f"eval_{head}_auroc_objectness"] = _NAN
    return out


# --------------------------------------------------------------------------------------------------------- #
# the INFORMATIVE calibration readout (A10 §15.3)                                                            #
# --------------------------------------------------------------------------------------------------------- #
def pr_equal_gate(packs: list) -> dict:
    """The P = R gate over pooled greedy 2 m rows (the audit's ``bha_analyze.pr_equal_gate`` rule: the first rank
    whose precision <= recall). INFORMATIVE only: it never replaces the declared 0.5 rule."""
    rows = sorted(((c, k) for pk in packs for (c, k, *_r) in greedy_rows(pk, PR_DIST_M, None) if k >= 0),
                  key=lambda r: -r[0])
    n_pos = int(sum(int(pk["pos"].sum()) for pk in packs))
    if not rows or n_pos == 0:
        return {"gate": _NAN, "precision": _NAN, "recall": _NAN, "n_pos": n_pos}
    h = np.asarray([k for (_c, k) in rows], dtype=np.float64)
    c = np.asarray([c for (c, _k) in rows], dtype=np.float64)
    tp = np.cumsum(h)
    prec = tp / np.arange(1, len(h) + 1)
    rec = tp / n_pos
    d = prec - rec
    k = int(np.argmax(d <= 0)) if (d <= 0).any() else len(d) - 1
    return {"gate": float(c[k]), "precision": float(prec[k]), "recall": float(rec[k]), "n_pos": n_pos}


def calib_keys(packs: list, head: str) -> dict:
    r = pr_equal_gate(packs)
    return {f"eval_{head}_calib_pr_gate": r["gate"], f"eval_{head}_calib_prec": r["precision"],
            f"eval_{head}_calib_rec": r["recall"], f"eval_{head}_calib_n_pos": float(r["n_pos"]),
            f"eval_{head}_calib_n_windows": float(len(packs))}


# --------------------------------------------------------------------------------------------------------- #
# the FIXED TRAIN calibration windows (A10 §15.3, INFORMATIVE)                                              #
# --------------------------------------------------------------------------------------------------------- #
#: the audit's 256 TRAIN windows (64 clips x 4), sha12 + window start only -- a banked artifact, never re-drawn.
CALIB_WINDOWS_FILE = "box_calib_train256.json"


def load_calib_windows(name: str = CALIB_WINDOWS_FILE) -> dict:
    """The banked calibration artifact, its window digest VERIFIED (a digest nothing recomputes is decoration)."""
    import hashlib
    import json
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "data" / name
    art = json.loads(p.read_text(encoding="utf-8"))
    blob = json.dumps(art["windows"], separators=(",", ":")).encode()
    if hashlib.sha256(blob).hexdigest() != art.get("windows_sha256"):
        raise SystemExit(f"[P0] {name}: the window list does not reproduce its own windows_sha256")
    if len(art["windows"]) != int(art["n_windows"]):
        raise SystemExit(f"[P0] {name}: {len(art['windows'])} windows listed, {art['n_windows']} declared")
    return art


def calib_indices(ds_index, sha12_of_window) -> tuple:
    """``(positions, n_missing)``: the dataset positions of the calibration windows, in the artifact's order.

    ``ds_index`` is the dataset's ``index`` (``[(episode_i, t)]``); ``sha12_of_window(episode_i)`` returns that
    episode's clip sha12. A window the dataset does not hold is COUNTED (the readout is informative, so the caller
    stamps the count rather than refusing the run)."""
    art = load_calib_windows()
    want = {(str(a), int(b)): k for k, (a, b) in enumerate(art["windows"])}
    found = {}
    for i, (e_i, t) in enumerate(ds_index):
        key = (str(sha12_of_window(e_i)), int(t))
        if key in want:
            found[want[key]] = i
    pos = [found[k] for k in sorted(found)]
    return pos, len(want) - len(pos)
