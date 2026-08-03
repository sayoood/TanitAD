"""D-TAC1 probe — the CHEAPEST DISCRIMINATING EXPERIMENTS for REF-C's tactical
head, run on an ALREADY-TRAINED checkpoint. No training. Forward passes only.

Pre-registration: ``Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md``
(both outcomes committed there BEFORE this is run).

THE QUESTION
============================================================================
MEASURED (REF-C-base 30k, canonical val, n = 859, LAN_E0_RESULTS.md section 5):
the 5-way tactical head predicts ``accelerate`` 0/93 and ``brake_stop`` 7/78,
while emitting the turns at almost exactly their true rate. Two mutually
exclusive explanations survive that observation, and they imply DIFFERENT fixes:

  READOUT   the head HAS longitudinal information; the mixed 5-way argmax and
            the class prior never let it surface. => factorising the readout
            (F2) + a prior-corrected decode (F3) is sufficient, and a large part
            is recoverable from the EXISTING checkpoint with no retrain at all.
  INPUT     the head has NO longitudinal information: ``maneuver_head`` reads
            ``pooled`` — the image embedding — while its own label is
            dv = v(t+2s) - v(t), and the ego speed reaches only the decoder.
            => F2 alone is a NULL RESULT and the head's INPUT must change (F1).

E-A1  COUNTERFACTUAL FACTORED DECODE (0 new training, 1 forward pass over val)
      The priority collapse is invertible: from the 5-way posterior recover
      P_lat and the CONDITIONAL P_lon (refc_tactical.invert_man5, exact
      round-trip pinned by tests/test_refc_tactical.py). Then ask, on the same
      windows and the same checkpoint, what the longitudinal decision would have
      been if it had never been mixed — raw argmax, and prior-corrected argmax.
      The threshold-FREE statistic is ``auc_lon_brake`` / ``auc_lon_accel``: an
      AUC near 0.5 means the information is absent (INPUT); an AUC well above
      0.5 with 0 emissions means it is present but unreadable (READOUT).

E-A2  LINEAR PROBE ON THE HEAD'S OWN INPUT (0 new training beyond a logistic
      regression on cached features). Fit a multinomial logistic regression to
      the longitudinal label from (a) ``pooled`` alone — literally what the head
      is given today, (b) ``v0`` alone, (c) ``pooled`` + ``v0``. Episode-disjoint
      folds. If (a) is at chance and (b)/(c) are well above it, F1 is the binding
      constraint and no architecture change can substitute for the input.

NEGATIVE CONTROLS, run FIRST and reported alongside (brief: prove the metric can
discriminate before quoting it):
  * ``shuffled``   the window <-> logits pairing is permuted. Every accuracy /
    AUC must fall to chance. A metric that still "separates" under this is
    measuring the class prior, not the model.
  * ``label_source`` the derived 5-way label is compared against the epcache's
    banked ``maneuvers`` field. A disagreement means the two label mints have
    drifted and NOTHING below is quotable.

Estimator on every interval: ``taniteval.ci.episode_cluster_bootstrap``, unit =
val episode. ``overlapping_holdout_se`` is never used (it biases the point
estimate — CLAUDE.md).

Usage (eval box / Thor — NEVER a training pod):
  OMP_NUM_THREADS=6 PYTHONPATH=<repo>/stack \\
  python scripts/refc_tactical_probe.py \\
      --ckpt /home/nvidia/models/refc-base/ckpt.pt \\
      --val-dir /home/nvidia/valdata/physicalai-val-0c5f7dac3b11 \\
      --preset base --out /tmp/dtac1_probe_refc-base-30k.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tanitad.refs import refc_tactical as tac

DEFAULT_STRIDE = 5
K_MAX = 20                     # farthest tactical waypoint (2 s @ 10 Hz)


# ---------------------------------------------------------------------------
# substrate
# ---------------------------------------------------------------------------

def load_val_episodes(val_dir: Path, limit: int | None = None):
    """``ep_*.pt`` -> episode objects. Unreadable files are RECORDED, never
    silently skipped (that would change every denominator downstream)."""
    class _Ep:
        __slots__ = ("feats", "poses", "episode_id", "maneuvers", "path")

    files = sorted(Path(val_dir).glob("ep_*.pt"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no ep_*.pt under {val_dir}")
    eps, unreadable = [], []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:
            unreadable.append({"file": f.name, "error": f"{type(exc).__name__}"})
            continue
        e = _Ep()
        e.feats, e.poses = d["frames_u8"], d["poses"].float()
        e.episode_id, e.maneuvers = d["episode_id"], d.get("maneuvers")
        e.path = str(f)
        eps.append(e)
    if not eps:
        raise SystemExit(f"every ep_*.pt under {val_dir} failed to load")
    return eps, {"val_dir": str(val_dir), "n_files_seen": len(files),
                 "n_episodes_loaded": len(eps), "unreadable": unreadable}


def build_model(ckpt: Path, preset: str, device: str):
    from tanitad.refs import refc as _refc
    presets = {"base": _refc.refc_config, "small": _refc.refc_small_config,
               "xl": _refc.refc_xl_config, "smoke": _refc.refc_smoke_config}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = presets[preset]()
    model = _refc.RefCModel(cfg)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # ``getattr`` on purpose: this probe must run against a POD checkout that
    # predates the D-TAC1 seam (a drifted `stack/` is the documented norm), so
    # it may not ship two new files just to read a field that means "off".
    if getattr(cfg, "factored_maneuver", False):
        raise SystemExit("this probe reads the 5-WAY head; point it at a "
                         "pre-D-TAC1 checkpoint")
    meta = {"ckpt": str(ckpt), "preset": preset,
            "ckpt_step": (ck.get("step") if isinstance(ck, dict) else None),
            "sd_missing": sorted(missing)[:8], "n_sd_missing": len(missing),
            "sd_unexpected": sorted(unexpected)[:8],
            "n_sd_unexpected": len(unexpected),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6,
                              3)}
    if len(missing):
        meta["⚠️"] = ("state_dict keys were MISSING from the checkpoint — those "
                      "modules are at random init and nothing below is quotable "
                      "for them.")
    return model.to(device).eval(), cfg, meta


@torch.no_grad()
def collect(model, cfg, eps, device: str, stride: int, batch: int,
            steps: int) -> dict:
    """One forward pass over the val windows -> the whole probe substrate."""
    window = int(cfg.window)
    LOG5, POOLED, V0, EID, LAT, LON, MAN5, MAN_BANKED = ([] for _ in range(8))
    for ep in eps:
        T = int(ep.feats.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(ep.feats[t:t + window])
                              for t in ch]).to(device).float().div_(255.0)
            v0 = ep.poses[last, 3].to(device)
            out = model(fw, nav_cmd=None, v0=v0, steps=steps)
            LOG5.append(out["maneuver_logits"].float().cpu())
            POOLED.append(out["pooled"].float().cpu())
            V0.append(ep.poses[last, 3].clone())
            EID.extend([ep.episode_id] * len(ch))
            # labels: the SAME endpoint kinematics the REF-C trainer reads.
            pose_last = ep.poses[last]
            fut = torch.stack([ep.poses[t + window: t + window + K_MAX]
                               for t in ch])
            lat, lon = tac.window_factored_labels(pose_last, fut,
                                                  horizon=K_MAX)
            LAT.append(lat)
            LON.append(lon)
            MAN5.append(tac.collapse(lat, lon))
            MAN_BANKED.append(ep.maneuvers[last] if ep.maneuvers is not None
                              else torch.full((len(ch),), -1,
                                              dtype=torch.long))
    return {"log5": torch.cat(LOG5), "pooled": torch.cat(POOLED),
            "v0": torch.cat(V0), "eid": EID, "lat": torch.cat(LAT),
            "lon": torch.cat(LON), "man5": torch.cat(MAN5),
            "man_banked": torch.cat(MAN_BANKED)}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _episode_cluster_bootstrap():
    """The ONE admissible estimator. ``taniteval`` is a sibling package of
    ``stack`` (repo_root/taniteval/taniteval), so it is not importable from a
    bare ``stack`` PYTHONPATH — add it rather than degrade to a weaker interval.
    A missing estimator FAILS LOUD: an interval without its estimator is not
    quotable (CLAUDE.md), so silently substituting one is worse than crashing."""
    try:
        from taniteval.ci import episode_cluster_bootstrap
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                               / "taniteval"))
        from taniteval.ci import episode_cluster_bootstrap
    return episode_cluster_bootstrap


def _ci(per_window, eid, label):
    r = _episode_cluster_bootstrap()(np.asarray(per_window, dtype=float), eid,
                                     reduce="mean")
    return {"metric": label, **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items()}}


def _auc(score, positive) -> float:
    """Rank-based ROC-AUC (ties averaged). Threshold-FREE, so it answers "is the
    information present" independently of any decode rule."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(positive).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within ties
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _class_report(pred, tgt, names) -> dict:
    n = len(names)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(np.asarray(tgt), np.asarray(pred)):
        conf[int(t), int(p)] += 1
    per = {}
    for k, nm in enumerate(names):
        n_true = int(conf[k].sum())
        per[nm] = {"n_true": n_true, "n_pred": int(conf[:, k].sum()),
                   "recall": round(conf[k, k] / n_true, 4) if n_true else None}
    macro = [v["recall"] for v in per.values() if v["recall"] is not None]
    return {"per_class": per, "confusion": conf.tolist(),
            "accuracy": round(float((np.asarray(pred) == np.asarray(tgt)).mean()),
                              4),
            "macro_recall": round(float(np.mean(macro)), 4) if macro else None,
            "never_predicted": [nm for nm, v in per.items()
                                if v["n_pred"] == 0]}


def _fit_logreg(X, y, n_cls, epochs=400, l2=1e-3, seed=0):
    """Multinomial logistic regression (torch, no sklearn dependency)."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    Xs = (X - mu) / sd
    W = torch.zeros(Xs.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xs @ W + b, y) \
            + l2 * (W * W).sum()
        loss.backward()
        opt.step()
    return (W.detach(), b.detach(), mu, sd)


def _apply_logreg(fit, X):
    W, b, mu, sd = fit
    Xs = (torch.as_tensor(X, dtype=torch.float32) - mu) / sd
    return (Xs @ W + b)


def linear_probe(feats: dict, y, eid, n_cls: int, names, folds: int = 2) -> dict:
    """Episode-disjoint K-fold multinomial LR per feature set.

    Reported per feature set: macro-recall (balanced accuracy — the metric a
    majority-class predictor CANNOT win) and the one-vs-rest AUC per class.
    """
    uniq = sorted(set(eid))
    assign = {e: i % folds for i, e in enumerate(uniq)}
    fold = np.array([assign[e] for e in eid])
    y = np.asarray(y)
    out = {}
    for name, X in feats.items():
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        pred = np.zeros(len(y), dtype=int)
        prob = np.zeros((len(y), n_cls), dtype=float)
        for f in range(folds):
            tr, te = fold != f, fold == f
            if not tr.any() or not te.any():
                continue
            fit = _fit_logreg(X[tr], y[tr], n_cls)
            lg = _apply_logreg(fit, X[te])
            pred[te] = lg.argmax(-1).numpy()
            prob[te] = torch.softmax(lg, -1).numpy()
        rep = _class_report(pred, y, names)
        rep["auc_ovr"] = {nm: round(_auc(prob[:, k], y == k), 4)
                          for k, nm in enumerate(names)}
        rep["n_features"] = int(X.shape[1])
        out[name] = rep
    return out


# ---------------------------------------------------------------------------
# the experiments
# ---------------------------------------------------------------------------

def experiment_a1(sub: dict, tau: float) -> dict:
    """E-A1 — counterfactual factored decode off the EXISTING 5-way head."""
    log5, lat, lon, eid = sub["log5"], sub["lat"], sub["lon"], sub["eid"]
    man5 = sub["man5"]
    log_lat, log_lon = tac.invert_man5(log5)

    lon_prior = tac.class_log_prior(lon, tac.N_LON)
    res = {
        "_what": "the SAME checkpoint, three decode rules, one forward pass",
        "n_windows": int(log5.shape[0]),
        "n_episodes": len(set(eid)),
        "label_marginal": {nm: round(float((lon == k).float().mean()), 4)
                           for k, nm in enumerate(tac.LON_CLASSES)},
        # (i) what is SHIPPED today
        "decode_5way_argmax": _class_report(log5.argmax(-1), man5,
                                            tac.MAN5_NAMES),
        # (ii) the mixing removed, prior untouched
        "decode_factored_raw": _class_report(log_lon.argmax(-1), lon,
                                             tac.LON_CLASSES),
        # (iii) the mixing removed AND the prior corrected
        "decode_factored_adjusted": _class_report(
            tac.logit_adjust(log_lon, lon_prior, tau).argmax(-1), lon,
            tac.LON_CLASSES),
        "lat_readout": _class_report(log_lat.argmax(-1), lat, tac.LAT_CLASSES),
        "tau": tau,
    }
    # THE threshold-free discriminator: is the information present at all?
    p_lon = log_lon.exp().numpy()
    res["auc_lon"] = {nm: round(_auc(p_lon[:, k], (lon == k).numpy()), 4)
                      for k, nm in enumerate(tac.LON_CLASSES)}
    res["auc_lon_active"] = round(
        _auc(1.0 - p_lon[:, tac.LON_STEADY], (lon != tac.LON_STEADY).numpy()), 4)
    # tau FRONTIER. A single tau is a point on a trade-off, and tau = 1 (the full
    # balanced posterior) is NOT automatically the right point: it maximises the
    # prior correction, not the metric. The whole curve is free — it is a
    # post-hoc transform of logits already computed — so report it rather than
    # defend one number.
    # ⚠️ Choosing tau by reading this curve is FITTING ON VAL. It is a FRONTIER
    # REPORT; a deployed tau must be picked on train/dev data (the model's own
    # EMA prior buffers) and only then confirmed here.
    res["tau_frontier"] = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        rep = _class_report(tac.logit_adjust(log_lon, lon_prior, t).argmax(-1),
                            lon, tac.LON_CLASSES)
        res["tau_frontier"].append({
            "tau": t, "accuracy": rep["accuracy"],
            "macro_recall": rep["macro_recall"],
            "recall": {nm: v["recall"] for nm, v in rep["per_class"].items()},
            "n_pred": {nm: v["n_pred"] for nm, v in rep["per_class"].items()}})
    # The label-side half of the defect, counted on THIS substrate: windows
    # whose longitudinal class the priority collapse destroys into a turn.
    turning = (man5 == tac.TURN_LEFT) | (man5 == tac.TURN_RIGHT)
    destroyed = int((turning & (lon != tac.LON_STEADY)).sum())
    res["label_side_collapse"] = {
        "n_destroyed_by_priority": destroyed,
        "frac": round(destroyed / int(log5.shape[0]), 4),
        "_what": ("windows where a longitudinal manoeuvre is live AND the 5-way "
                  "label calls a turn — the 5-way target cannot carry them, so "
                  "no decode rule can recover them from the 5-way head")}
    # CI on the decision that matters, with its estimator named.
    res["ci_lon_active_correct"] = _ci(
        ((log_lon.argmax(-1) != tac.LON_STEADY)
         == (lon != tac.LON_STEADY)).float().numpy(), eid,
        "factored-raw agreement on 'is a longitudinal manoeuvre happening'")
    res["ci_lon_adjusted_correct"] = _ci(
        ((tac.logit_adjust(log_lon, lon_prior, tau).argmax(-1) != tac.LON_STEADY)
         == (lon != tac.LON_STEADY)).float().numpy(), eid,
        "factored-adjusted agreement on 'is a longitudinal manoeuvre happening'")
    return res


def negative_control_a1(sub: dict, tau: float, seed: int = 0) -> dict:
    """Permute the window <-> logits pairing. Every statistic above must fall to
    chance; if it does not, the statistic is reading the class prior."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(sub["log5"].shape[0], generator=g)
    shuffled = dict(sub)
    shuffled["log5"] = sub["log5"][perm]
    r = experiment_a1(shuffled, tau)
    return {"_what": "logits permuted across windows — MUST be at chance",
            "auc_lon": r["auc_lon"], "auc_lon_active": r["auc_lon_active"],
            "decode_factored_raw_macro_recall":
                r["decode_factored_raw"]["macro_recall"],
            "decode_5way_accuracy": r["decode_5way_argmax"]["accuracy"]}


def experiment_a2(sub: dict) -> dict:
    """E-A2 — can the head's OWN input answer the longitudinal question?"""
    pooled = sub["pooled"].numpy()
    v0 = sub["v0"].numpy()
    feats = {"pooled_only  (what maneuver_head is given TODAY)": pooled,
             "v0_only      (the channel it is NOT given)": v0,
             "pooled_plus_v0 (the F1 proposal)":
                 np.concatenate([pooled, v0[:, None]], axis=1)}
    out = linear_probe(feats, sub["lon"].numpy(), sub["eid"], tac.N_LON,
                       tac.LON_CLASSES)
    out["_what"] = ("episode-disjoint 2-fold multinomial logistic regression on "
                    "the LONGITUDINAL label; macro_recall is balanced accuracy "
                    "(chance = 1/3), auc_ovr is threshold-free")
    out["_chance_macro_recall"] = round(1.0 / tac.N_LON, 4)
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--preset", default="base",
                    choices=("base", "small", "xl", "smoke"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None,
                    help="decoder steps (default: cfg.diffusion_steps)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="logit-adjustment strength for the adjusted decode")
    ap.add_argument("--dump", default=None,
                    help="bank the per-window substrate (5-way logits, pooled, "
                         "v0, labels, eid) to a .pt so every later re-analysis "
                         "— tau sweeps, new decode rules, new probes — is 0 GPU")
    args = ap.parse_args(argv)

    model, cfg, meta = build_model(Path(args.ckpt), args.preset, args.device)
    eps, coverage = load_val_episodes(Path(args.val_dir), args.max_episodes)
    steps = (args.steps if args.steps is not None
             else int(cfg.decoder.diffusion_steps))
    sub = collect(model, cfg, eps, args.device, args.stride, args.batch, steps)
    if args.dump:
        torch.save({**{k: v for k, v in sub.items()},
                    "ckpt": str(args.ckpt), "val_dir": str(args.val_dir),
                    "stride": args.stride, "decoder_steps": steps,
                    "lon_classes": list(tac.LON_CLASSES),
                    "lat_classes": list(tac.LAT_CLASSES)}, args.dump)
        print(f"[dtac1] banked substrate -> {args.dump}", flush=True)

    # --- control 0: the label source itself --------------------------------
    banked = sub["man_banked"]
    have = banked >= 0
    label_ctrl = {
        "n_windows_with_banked_maneuvers": int(have.sum()),
        "agreement_derived_vs_banked":
            round(float((sub["man5"][have] == banked[have]).float().mean()), 4)
            if bool(have.any()) else None,
        "_why": ("the derived 5-way label must reproduce the epcache's banked "
                 "`maneuvers`. A gap means the two mints have drifted (e.g. v1 "
                 "endpoint vs v2 curvature-gated) and every count below must be "
                 "read against the DERIVED label only."),
    }

    result = {
        "experiment": "D-TAC1 probe (E-A1 counterfactual decode + E-A2 input "
                      "probe)",
        "prereg": "Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md",
        "evidence_class": "MEASURED (ours)",
        "estimator": ("taniteval.ci.episode_cluster_bootstrap, unit = val "
                      "episode; overlapping_holdout_se is NEVER used"),
        "meta": meta, "coverage": coverage,
        "stride": args.stride, "decoder_steps": steps,
        "control_label_source": label_ctrl,
        "control_shuffled": negative_control_a1(sub, args.tau),
        "E_A1_counterfactual_decode": experiment_a1(sub, args.tau),
        "E_A2_input_probe": experiment_a2(sub),
    }

    a1 = result["E_A1_counterfactual_decode"]
    a2 = result["E_A2_input_probe"]
    auc_active = a1["auc_lon_active"]
    pooled_key = "pooled_only  (what maneuver_head is given TODAY)"
    v0_key = "v0_only      (the channel it is NOT given)"
    result["VERDICT"] = {
        "auc_lon_active_from_the_existing_head": auc_active,
        "pooled_only_macro_recall": a2[pooled_key]["macro_recall"],
        "v0_only_macro_recall": a2[v0_key]["macro_recall"],
        "reading": ("READOUT-limited (F2+F3 sufficient, part recoverable with "
                    "no retrain)" if (auc_active is not None
                                      and auc_active >= 0.65)
                    else "INPUT-limited (F1 required; F2 alone would be a null "
                         "result)" if (auc_active is not None
                                       and auc_active <= 0.55)
                    else "INDETERMINATE — see the pre-registration's tie-break"),
        "_decision_rule": ("thresholds 0.65 / 0.55 on auc_lon_active were fixed "
                           "in the pre-registration BEFORE this ran"),
    }

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["VERDICT"], indent=2), flush=True)
    print(f"[dtac1] wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":                       # pragma: no cover
    main()
