"""Behavioral-quality eval: tactical maneuver-selection + strategic planning.

Position/trajectory ADE (D1, driving_diagnostic) says WHERE the ego goes. This
script asks a different question: does the TanitAD-4B-M flagship pick sensible
MANEUVERS and represent route-scale INTENT, or does it just default to
lane-keeping? "Behavior beyond position."

WHAT THE FLAGSHIP ACTUALLY EMITS (STEP 0 — measured on ckpt27k_flagship.pt,
step 27000; state_dict introspection reproduced by `step0_findings()` below):

  module        emits                                     maneuver/route head?
  encoder       token grid -> compact state [B, S=2048]   no
  predictor     operative imagined latents k in {1,2,4}   no  (heads -> latent)
  tactical_pred tactical  imagined latents k in {8,16}     NO maneuver logits
  imagination   H15 belief field over the token grid      no
  inv_dyn       (steer, accel) from consecutive states    no

  There is NO trained maneuver-classification head, NO route/nav head, NO VQ
  codebook, NO parametric strategic component in the checkpoint (grep of the
  state_dict for maneuver|route|nav|strateg|vq|codebook returns NOTHING). The
  tactical layer's `heads.8` / `heads.16` are nn.Linear(d, state_dim) — they
  regress future LATENTS at 0.8 s / 1.6 s, not a maneuver distribution. The
  `TacticalSelector` (fourbrain.py) IS an imagine-and-select mechanism, but it
  is (a) not wired into the trained model, (b) goal-conditioned on an
  externally supplied sub-goal, (c) scored over hand-built action primitives.
  `StrategicGraph` is non-parametric (external k-means + Dijkstra) and decodes
  nothing from a forward pass. (The REF-B *opponent* model has real
  maneuver_head / route_head — but that is a different model, not the flagship.)

CONSEQUENCE FOR THE EVAL — we measure what EXISTS, two honest instruments:

  (1) DECODABILITY PROBE (primary). Fit a calibrated classifier (linear +
      MLP, A3 doctrine: probe reads the model's own imagined latents) from each
      latent source -> GT maneuver class, held-out by route (gates.split_by_
      episode). This is a PROBE, not an intrinsic selector: it says whether the
      representation CONTAINS the maneuver, a prerequisite for any future
      selection head. Reported vs the majority-class (lane_keep) baseline with
      balanced accuracy + macro-F1 (raw accuracy is meaningless under the
      highway lane-keep imbalance).

  (2) IMAGINE-AND-SELECT (secondary, loudly caveated). Run the actual
      imagine-and-select machinery (vectorized TacticalSelector logic, pinned
      equal to fourbrain.TacticalSelector by a test) over a 9-primitive
      vocabulary, goal-conditioned on the GT 2 s sub-goal, and classify the
      SELECTED primitive's maneuver. This tests whether goal-directed rollout
      picks the right maneuver — bounded by dynamics fidelity AND by the
      injected goal; it is NOT an autonomous learned selector.

STRATEGIC — probe the shared state -> coarse route intent (left/straight/right
from long-horizon heading change, refb_labels.nav_command semantics). The
verdict is recorded plainly: the strategic layer emits no intrinsic decodable
code in Phase 0, so this is a decodability proxy on the representation and the
strategic *selector* is a Phase-0 gap.

GT maneuver labels reuse scripts/refb_labels.classify_maneuver (documented
kinematic thresholds, 2 s horizon). Ego convention, episode loading, window
encoding, and the route-parity split are reused verbatim from
d1_probe_capacity / driving_diagnostic / gates — no reinvention.

Usage (pod1):
  python scripts/eval_behavior.py --ckpt /workspace/ckpt27k_flagship.pt \
      --cache-dirs /workspace/data/comma2k19/_epcache \
                   /workspace/data/physicalai/_epcache \
      --out /workspace/experiments/behavior_eval --episodes 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refb_labels as rl  # noqa: E402  (kinematic maneuver labels — reused)
from d1_probe_capacity import _ego  # noqa: E402  (ego-frame convention — reused)

from tanitad.eval.gates import split_by_episode  # noqa: E402  (route parity)
from tanitad.models.readout import RidgeProbe  # noqa: E402  (A3 calibrated probe)

MANEUVER_CLASSES = ("lane_keep", "turn_left", "turn_right", "accelerate",
                    "brake_stop")               # refb.MANEUVER_CLASSES order
ROUTE_CLASSES = ("route_left", "route_straight", "route_right")
N_MAN = len(MANEUVER_CLASSES)
N_ROUTE = len(ROUTE_CLASSES)
LANE_KEEP = rl.LANE_KEEP                          # 0 — majority baseline class

MANEUVER_H = rl.LABEL_HORIZON                     # 20 steps = 2 s @ 10 Hz
SELECT_H = MANEUVER_H                             # imagine-and-select rolls 2 s
ROUTE_H = 100                                     # 10 s route-intent lookahead
ROUTE_MIN = 50                                    # >=5 s of future -> route-valid
LATENT_SOURCES = ("encoder_state", "operative_k4", "tactical_k8", "tactical_k16")


# --------------------------------------------------------------------------- #
# Confusion-matrix math — pure, unit-tested                                    #
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: Tensor, y_pred: Tensor, n_classes: int) -> Tensor:
    """[N] true, [N] pred -> [C, C] counts, rows = true, cols = predicted."""
    y_true = y_true.long().flatten()
    y_pred = y_pred.long().flatten()
    idx = y_true * n_classes + y_pred
    return torch.bincount(idx, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes)


def per_class_prf(cm: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-class precision, recall, F1, support from a confusion matrix."""
    cm = cm.double()
    tp = cm.diag()
    support = cm.sum(1)                            # true count per class
    predicted = cm.sum(0)                          # predicted count per class
    precision = tp / predicted.clamp_min(1e-12)
    recall = tp / support.clamp_min(1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return precision, recall, f1, support


def macro_f1(cm: Tensor) -> float:
    """Unweighted mean F1 over classes PRESENT in ground truth (support > 0)."""
    _, _, f1, support = per_class_prf(cm)
    mask = support > 0
    return float(f1[mask].mean()) if bool(mask.any()) else float("nan")


def balanced_accuracy(cm: Tensor) -> float:
    """Mean per-class recall over classes present in ground truth."""
    _, recall, _, support = per_class_prf(cm)
    mask = support > 0
    return float(recall[mask].mean()) if bool(mask.any()) else float("nan")


def accuracy(cm: Tensor) -> float:
    total = cm.sum()
    return float(cm.diag().sum() / total) if int(total) > 0 else float("nan")


def prf_dict(cm: Tensor, class_names: tuple[str, ...]) -> dict:
    p, r, f1, sup = per_class_prf(cm)
    return {class_names[i]: {"precision": round(float(p[i]), 4),
                             "recall": round(float(r[i]), 4),
                             "f1": round(float(f1[i]), 4),
                             "support": int(sup[i])}
            for i in range(len(class_names))}


# --------------------------------------------------------------------------- #
# Calibrated classifier probe (A3): linear logistic + small MLP                #
# --------------------------------------------------------------------------- #
def _class_weights(y: Tensor, n_classes: int) -> Tensor:
    """Inverse-frequency weights (present classes), mean-normalized to ~1."""
    counts = torch.bincount(y.long(), minlength=n_classes).double()
    inv = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
    present = counts > 0
    inv = inv / inv[present].mean().clamp_min(1e-12)   # avg weight ~ 1
    return inv.float()


def fit_classifier(x_tr: Tensor, y_tr: Tensor, x_va: Tensor, n_classes: int,
                   kind: str = "linear", epochs: int = 300, lr: float = 1e-2,
                   seed: int = 0, device: str = "cpu"
                   ) -> tuple[Tensor, float]:
    """Fit a class-weighted probe on (x_tr, y_tr); return (val preds, train acc).

    Features are z-scored on TRAIN statistics (frozen, applied to val). Class
    weighting = inverse frequency so the lane-keep majority cannot swamp the
    minority maneuvers — the probe measures decodability, not the prior.
    """
    torch.manual_seed(seed)
    x_tr = x_tr.to(device).float()
    x_va = x_va.to(device).float()
    y_tr = y_tr.to(device).long()
    mu = x_tr.mean(0, keepdim=True)
    sd = x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr = (x_tr - mu) / sd
    x_va = (x_va - mu) / sd
    f = x_tr.shape[1]
    if kind == "linear":
        net: nn.Module = nn.Linear(f, n_classes)
    elif kind == "mlp":
        net = nn.Sequential(nn.Linear(f, 256), nn.GELU(),
                            nn.Linear(256, n_classes))
    else:
        raise ValueError(kind)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_classes).to(device))
    n = x_tr.shape[0]
    bs = min(4096, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            loss = lossfn(net(x_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        val_pred = net(x_va).argmax(-1).cpu()
        train_acc = float((net(x_tr).argmax(-1) == y_tr).float().mean())
    return val_pred, train_acc


def probe_metrics(y_true: Tensor, y_pred: Tensor, train_acc: float,
                  n_classes: int, class_names: tuple[str, ...],
                  majority_class: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    maj = float((y_true == majority_class).float().mean())
    return {
        "accuracy": round(accuracy(cm), 4),
        "balanced_accuracy": round(balanced_accuracy(cm), 4),
        "macro_f1": round(macro_f1(cm), 4),
        "majority_baseline_acc": round(maj, 4),
        "beats_majority_balacc": bool(balanced_accuracy(cm) > 1.0 / n_classes),
        "train_fit_acc": round(train_acc, 4),
        "n_val": int(cm.sum()),
        "per_class": prf_dict(cm, class_names),
        "confusion_matrix": cm.tolist(),
        "confusion_rows_true_cols_pred": list(class_names),
    }


# --------------------------------------------------------------------------- #
# Ground-truth labels from future kinematics                                   #
# --------------------------------------------------------------------------- #
def gt_maneuver(poses: Tensor, last: Tensor, horizon: int = MANEUVER_H) -> Tensor:
    """GT maneuver class per window (refb thresholds, 2 s). last -> [b]."""
    return rl.classify_maneuver(poses[last, 2], poses[last + horizon, 2],
                                poses[last, 3], poses[last + horizon, 3])


def route_intent(poses: Tensor, last: Tensor, T: int, turn_rad: float,
                 route_h: int = ROUTE_H, route_min: int = ROUTE_MIN
                 ) -> tuple[Tensor, Tensor]:
    """Coarse route intent from NET heading change over up to ``route_h`` steps.

    left/straight/right by |dyaw| vs ``turn_rad`` (nav_command semantics). A
    window is route-valid only if >= ``route_min`` future steps exist. Returns
    (class [b] in ROUTE_CLASSES order, valid mask [b]).
    """
    avail = (T - 1) - last
    h = torch.clamp(torch.minimum(avail, torch.full_like(avail, route_h)), min=1)
    dyaw = rl.wrap_to_pi(poses[last + h, 2] - poses[last, 2])
    cls = torch.full_like(last, 1)                # route_straight = 1
    cls[dyaw > turn_rad] = 0                       # route_left = 0
    cls[dyaw < -turn_rad] = 2                      # route_right = 2
    valid = avail >= route_min
    return cls, valid


def gt_dyaw_dv(poses: Tensor, last: Tensor, horizon: int = SELECT_H) -> Tensor:
    """[b, 2] = (wrapped heading change, speed change) over ``horizon``."""
    dyaw = rl.wrap_to_pi(poses[last + horizon, 2] - poses[last, 2])
    dv = poses[last + horizon, 3] - poses[last, 3]
    return torch.stack([dyaw, dv], dim=-1)


# --------------------------------------------------------------------------- #
# Collection — encode each window once, gather all latents + GT + meta         #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(world, episodes, corpora, device, window, turn_rad_strict,
            turn_rad_relaxed, stride=8, batch=8, keep_states=True) -> dict:
    cols = {k: [] for k in ("encoder_state", "operative_k4", "tactical_k8",
                            "tactical_k16", "man", "route_strict", "route_relax",
                            "route_valid", "sub_xy", "dyaw_dv", "eid_global",
                            "corpus", "ep_id", "step", "speed")}
    states_full, base_actions = [], []
    has_tac = world.tactical_pred is not None
    op_h = max(world.predictor.cfg.horizons)            # 4 (base250cam); 2 (smoke)
    tac_hs = sorted(world.tactical_pred.cfg.horizons) if has_tac else []
    for gi, (ep, corp) in enumerate(zip(episodes, corpora)):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = fr.shape[0]
        if T <= window + MANEUVER_H:
            continue
        starts = list(range(0, T - window - MANEUVER_H, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            st = world.encode_window(fw)                       # [b, W, S]
            op = world.imagine(st, aw)                          # {1,2,4}
            cols["encoder_state"].append(st[:, -1].cpu())
            cols["operative_k4"].append(op[op_h].cpu())         # max op horizon
            if has_tac:
                tac = world.tactical_pred(st, aw)               # {8,16}
                cols["tactical_k8"].append(tac[tac_hs[0]].cpu())
                cols["tactical_k16"].append(tac[tac_hs[-1]].cpu())
            if keep_states:
                states_full.append(st.cpu())
                base_actions.append(aw.cpu())
            last = torch.tensor([t + window - 1 for t in ch])
            cols["man"].append(gt_maneuver(ep.poses, last))
            rs, rv = route_intent(ep.poses, last, T, turn_rad_strict)
            rr, _ = route_intent(ep.poses, last, T, turn_rad_relaxed)
            cols["route_strict"].append(rs)
            cols["route_relax"].append(rr)
            cols["route_valid"].append(rv)
            cols["sub_xy"].append(_ego(ep.poses[last + SELECT_H, :2]
                                       - ep.poses[last, :2], ep.poses[last, 2]))
            cols["dyaw_dv"].append(gt_dyaw_dv(ep.poses, last))
            cols["speed"].append(ep.poses[last, 3])
            cols["eid_global"].extend([gi] * len(ch))
            cols["corpus"].extend([corp] * len(ch))
            cols["ep_id"].extend([int(ep.episode_id)] * len(ch))
            cols["step"].extend([int(x) for x in last.tolist()])
    out = {}
    for k, v in cols.items():
        if k in ("corpus",):
            out[k] = v
        elif k in ("eid_global", "ep_id", "step"):
            out[k] = torch.tensor(v)
        else:
            out[k] = torch.cat(v).float() if v else torch.empty(0)
    out["states_full"] = torch.cat(states_full).float() if states_full else None
    out["base_actions"] = torch.cat(base_actions).float() if base_actions else None
    out["has_tactical"] = has_tac
    out["op_horizon"] = op_h
    out["tac_horizons"] = tac_hs
    return out


# --------------------------------------------------------------------------- #
# Class balance                                                                #
# --------------------------------------------------------------------------- #
def class_balance(labels: Tensor, n_classes: int, class_names) -> dict:
    counts = torch.bincount(labels.long(), minlength=n_classes)
    n = int(counts.sum())
    return {"n": n, "counts": {class_names[i]: int(counts[i])
                               for i in range(n_classes)},
            "frac": {class_names[i]: round(float(counts[i]) / max(1, n), 4)
                     for i in range(n_classes)}}


def _corpus_scopes(corpus: list[str]) -> dict[str, Tensor]:
    scopes = {"_all": torch.ones(len(corpus), dtype=torch.bool)}
    for c in sorted(set(corpus)):
        scopes[c] = torch.tensor([x == c for x in corpus])
    return scopes


# --------------------------------------------------------------------------- #
# (1) Tactical maneuver decodability probe                                     #
# --------------------------------------------------------------------------- #
def maneuver_probe_eval(data, seeds, val_frac, device, epochs) -> dict:
    """Per latent source x corpus: linear + MLP probe -> GT maneuver, route
    parity split, mean/std over seeds; confusion matrix from seed[0]."""
    eid = data["eid_global"]
    y = data["man"].long()
    corpus = data["corpus"]
    scopes = _corpus_scopes(corpus)
    sources = [s for s in LATENT_SOURCES
               if s in data and data[s].numel() and
               not (s.startswith("tactical") and not data["has_tactical"])]
    report: dict = {}
    for src in sources:
        X = data[src]
        report[src] = {}
        for scope_name, scope_mask in scopes.items():
            scope_idx = scope_mask.nonzero(as_tuple=True)[0]
            n_ep = len(set(int(eid[i]) for i in scope_idx))
            if n_ep < 2 or len(scope_idx) < 40:
                report[src][scope_name] = {"skipped": f"too few (ep={n_ep}, "
                                           f"n={len(scope_idx)})"}
                continue
            per_kind = {}
            for kind in ("linear", "mlp"):
                accs, bals, f1s = [], [], []
                rep = None                        # first VALID split -> the matrix
                for seed in seeds:
                    tr, va = split_by_episode(eid.tolist(), val_frac, seed)
                    sset = set(scope_idx.tolist())
                    tr = [i for i in tr if i in sset]
                    va = [i for i in va if i in sset]
                    if len(tr) < 20 or len(va) < 10:
                        continue
                    pred, tr_acc = fit_classifier(
                        X[tr], y[tr], X[va], N_MAN, kind=kind,
                        epochs=epochs, seed=seed, device=device)
                    cm = confusion_matrix(y[va], pred, N_MAN)
                    accs.append(accuracy(cm))
                    bals.append(balanced_accuracy(cm))
                    f1s.append(macro_f1(cm))
                    if rep is None:
                        rep = (y[va], pred, tr_acc)
                if rep is None:
                    per_kind[kind] = {"skipped": "no valid splits"}
                    continue
                m = probe_metrics(rep[0], rep[1], rep[2], N_MAN,
                                  MANEUVER_CLASSES, LANE_KEEP)
                m["seed_mean_std"] = {
                    "accuracy": [round(_mean(accs), 4), round(_std(accs), 4)],
                    "balanced_accuracy": [round(_mean(bals), 4), round(_std(bals), 4)],
                    "macro_f1": [round(_mean(f1s), 4), round(_std(f1s), 4)],
                    "n_seeds": len(accs)}
                per_kind[kind] = m
            report[src][scope_name] = per_kind
    return report


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# --------------------------------------------------------------------------- #
# (2) Imagine-and-select (secondary, goal-conditioned)                         #
# --------------------------------------------------------------------------- #
def build_primitives(steer_mag: float, accel_mag: float, horizon: int,
                     device) -> tuple[Tensor, Tensor]:
    """9 constant-action primitives (3 steer x 3 accel), each [horizon, 2], and
    their intended maneuver class [9] (turn>accel priority, refb convention)."""
    steers = (-steer_mag, 0.0, steer_mag)
    accels = (-accel_mag, 0.0, accel_mag)
    acts, klass = [], []
    for s in steers:
        for a in accels:
            acts.append(torch.tensor([s, a], device=device).repeat(horizon, 1))
            if s > 0:
                klass.append(rl.TURN_LEFT)
            elif s < 0:
                klass.append(rl.TURN_RIGHT)
            elif a > 0:
                klass.append(rl.ACCELERATE)
            elif a < 0:
                klass.append(rl.BRAKE_STOP)
            else:
                klass.append(rl.LANE_KEEP)
    return torch.stack(acts), torch.tensor(klass, device=device)


@torch.no_grad()
def roll_operative(world, states: Tensor, actions: Tensor,
                   seq: Tensor) -> Tensor:
    """Recursively roll the operative 1-step head under action sequence ``seq``.

    states [N, W, S], actions [N, W, A], seq [K, A] (applied to every window) ->
    endpoint latent [N, S]. This is fourbrain.TacticalSelector's rollout,
    vectorized over windows (pinned equal by test_eval_behavior)."""
    s, a = states.clone(), actions.clone()
    for k in range(seq.shape[0]):
        a = torch.roll(a, -1, dims=1)
        a[:, -1] = seq[k]
        z = world.imagine(s, a)[1]
        s = torch.roll(s, -1, dims=1)
        s[:, -1] = z
    return s[:, -1]


@torch.no_grad()
def imagine_and_select_eval(world, data, device, seeds, val_frac, steer_mag,
                            accel_mag, comfort_weight, max_windows,
                            chunk=256) -> dict:
    """Goal-conditioned imagine-and-select over 9 primitives -> selected
    maneuver vs GT. Probe (A3) calibrates endpoint latent -> ego xy for scoring.

    CAVEAT (recorded in output): the selection is driven by the GT sub-goal and
    bounded by 2 s recursive-rollout fidelity of a predictor trained at k<=4;
    it is NOT an autonomous learned selector. It measures whether goal-directed
    rollout recovers the true maneuver."""
    if data["states_full"] is None:
        return {"skipped": "states not retained"}
    prims, prim_class = build_primitives(steer_mag, accel_mag, SELECT_H, device)
    eid = data["eid_global"]
    seed = seeds[0]
    tr, va = split_by_episode(eid.tolist(), val_frac, seed)
    if max_windows and len(va) > max_windows:
        va = va[:max_windows]
    states = data["states_full"]
    acts = data["base_actions"]
    sub_xy = data["sub_xy"]
    gt_man = data["man"].long()

    # A3 probe: endpoint latent under TRUE last-action hold -> GT ego xy @2 s.
    def endpoints(idx, seq):
        outs = []
        for j in range(0, len(idx), chunk):
            b = idx[j:j + chunk]
            outs.append(roll_operative(world, states[b].to(device),
                                       acts[b].to(device), seq).cpu())
        return torch.cat(outs) if outs else torch.empty(0)

    hold = acts[tr][:, -1:].mean(0).squeeze(0).to(device).repeat(SELECT_H, 1)
    z_fit = endpoints(tr, hold)
    probe = RidgeProbe(alpha=1.0).fit(z_fit, sub_xy[tr])

    # Score each primitive's rolled endpoint vs the GT sub-goal + comfort.
    scores = []
    for p in range(prims.shape[0]):
        z_end = endpoints(va, prims[p])
        xy = probe.predict(z_end)
        dist = (xy - sub_xy[va]).norm(dim=-1)
        comfort = prims[p].pow(2).mean()
        scores.append(dist + comfort_weight * comfort)
    sel = torch.stack(scores, dim=1).argmin(dim=1)          # [n_val] primitive idx
    sel_class = prim_class.cpu()[sel]
    truth = gt_man[va]
    cm = confusion_matrix(truth, sel_class, N_MAN)
    sel_dist = torch.bincount(sel_class, minlength=N_MAN)
    return {
        "note": ("GOAL-CONDITIONED on GT 2 s sub-goal; bounded by 2 s "
                 "recursive-rollout fidelity (predictor trained k<=4) AND by "
                 "the injected goal. NOT an autonomous learned selector."),
        "n_windows": int(len(va)),
        "primitive_vocabulary": {"steer_mag": steer_mag, "accel_mag": accel_mag,
                                 "n_primitives": int(prims.shape[0]),
                                 "horizon_steps": SELECT_H},
        "overall": {"accuracy": round(accuracy(cm), 4),
                    "balanced_accuracy": round(balanced_accuracy(cm), 4),
                    "macro_f1": round(macro_f1(cm), 4),
                    "majority_baseline_acc": round(
                        float((truth == LANE_KEEP).float().mean()), 4)},
        "per_class": prf_dict(cm, MANEUVER_CLASSES),
        "confusion_matrix": cm.tolist(),
        "confusion_rows_true_cols_pred": list(MANEUVER_CLASSES),
        "selected_class_distribution": {MANEUVER_CLASSES[i]: int(sel_dist[i])
                                        for i in range(N_MAN)},
        "by_corpus": _select_by_corpus(data, va, sel_class),
    }


def _select_by_corpus(data, va, sel_class) -> dict:
    truth = data["man"].long()
    corpus = data["corpus"]
    out = {}
    for c in sorted(set(corpus)):
        loc = [k for k, i in enumerate(va) if corpus[i] == c]
        if not loc:
            continue
        t = truth[[va[k] for k in loc]]
        p = sel_class[loc]
        cm = confusion_matrix(t, p, N_MAN)
        out[c] = {"n": len(loc), "accuracy": round(accuracy(cm), 4),
                  "balanced_accuracy": round(balanced_accuracy(cm), 4),
                  "macro_f1": round(macro_f1(cm), 4),
                  "confusion_matrix": cm.tolist()}
    return out


# --------------------------------------------------------------------------- #
# (3) Strategic route-intent probe                                             #
# --------------------------------------------------------------------------- #
def strategic_probe_eval(data, seeds, val_frac, device, epochs, turn_deg
                         ) -> dict:
    """Probe shared latents -> coarse route intent (left/straight/right). The
    strategic layer emits no intrinsic code, so this is a decodability proxy on
    the representation, reported per threshold + the honest gap verdict."""
    eid = data["eid_global"]
    corpus = data["corpus"]
    scopes = _corpus_scopes(corpus)
    valid = data["route_valid"].bool()
    out = {"verdict": (
        "STRATEGIC LAYER EMITS NO INTRINSIC DECODABLE CODE IN PHASE 0: the "
        "flagship checkpoint has no route/nav head and no VQ codebook; "
        "StrategicGraph is non-parametric (external k-means + Dijkstra). This "
        "section is a DECODABILITY PROXY — is route intent linearly present in "
        "the shared representation? — not an evaluation of a strategic "
        "selector, which does not yet exist (Phase-0 gap)."),
        "route_valid_windows": int(valid.sum()),
        "thresholds_deg": {"strict": turn_deg[0], "relaxed": turn_deg[1]}}
    for label, ykey in (("strict", "route_strict"), ("relaxed", "route_relax")):
        y = data[ykey].long()
        bal = {sc: class_balance(y[m & valid], N_ROUTE, ROUTE_CLASSES)
               for sc, m in scopes.items()}
        res = {"class_balance": bal, "probe": {}}
        for src in ("encoder_state", "tactical_k16"):
            if src not in data or not data[src].numel():
                continue
            if src == "tactical_k16" and not data["has_tactical"]:
                continue
            res["probe"][src] = {}
            for sc, m in scopes.items():
                idx = (m & valid).nonzero(as_tuple=True)[0]
                n_ep = len(set(int(eid[i]) for i in idx))
                if n_ep < 2 or len(idx) < 40:
                    res["probe"][src][sc] = {"skipped": f"ep={n_ep} n={len(idx)}"}
                    continue
                bals = []
                rep = None                        # first VALID split -> the matrix
                for seed in seeds:
                    tr, va = split_by_episode(eid.tolist(), val_frac, seed)
                    iset = set(idx.tolist())
                    tr = [i for i in tr if i in iset]
                    va = [i for i in va if i in iset]
                    if len(tr) < 20 or len(va) < 10:
                        continue
                    pred, tr_acc = fit_classifier(
                        data[src][tr], y[tr], data[src][va], N_ROUTE,
                        kind="linear", epochs=epochs, seed=seed, device=device)
                    bals.append(balanced_accuracy(
                        confusion_matrix(y[va], pred, N_ROUTE)))
                    if rep is None:
                        rep = (y[va], pred, tr_acc)
                if rep is None:
                    res["probe"][src][sc] = {"skipped": "no valid splits"}
                    continue
                res["probe"][src][sc] = probe_metrics(
                    rep[0], rep[1], rep[2], N_ROUTE, ROUTE_CLASSES,
                    rl.ROUTE_STRAIGHT)
                res["probe"][src][sc]["balacc_seed_mean_std"] = [
                    round(_mean(bals), 4), round(_std(bals), 4)]
        out[label] = res
    return out


# --------------------------------------------------------------------------- #
# (4) Worst-K maneuver errors (position-vs-behavior joint view)                #
# --------------------------------------------------------------------------- #
def worst_k_errors(data, device, val_frac, seed, epochs, src, k=25) -> dict:
    """Confidently-wrong maneuver predictions (probe on ``src``, seed split),
    with episode/step so they can be pulled up in the replay/viz tools."""
    eid = data["eid_global"]
    y = data["man"].long()
    tr, va = split_by_episode(eid.tolist(), val_frac, seed)
    if len(tr) < 20 or len(va) < 10 or src not in data or not data[src].numel():
        return {"skipped": True}
    # Refit with logits so we can rank by confidence of the wrong class.
    torch.manual_seed(seed)
    X, ytr = data[src], y
    x_tr = X[tr].to(device).float()
    x_va = X[va].to(device).float()
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    net = nn.Linear(x_tr.shape[1], N_MAN).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss(
        weight=_class_weights(ytr[tr], N_MAN).to(device))
    xs = (x_tr - mu) / sd
    yt = ytr[tr].to(device)
    for _ in range(epochs):
        perm = torch.randperm(xs.shape[0], device=device)
        for j in range(0, xs.shape[0], 4096):
            b = perm[j:j + 4096]
            loss = lossfn(net(xs[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.softmax(net((x_va - mu) / sd), -1).cpu()
    pred = prob.argmax(-1)
    truth = y[va]
    conf_wrong = prob.max(-1).values * (pred != truth).float()
    order = torch.argsort(conf_wrong, descending=True)[:k]
    items = []
    for o in order.tolist():
        if pred[o] == truth[o]:
            continue
        i = va[o]
        items.append({"corpus": data["corpus"][i],
                      "episode_id": int(data["ep_id"][i]),
                      "step": int(data["step"][i]),
                      "gt_maneuver": MANEUVER_CLASSES[int(truth[o])],
                      "pred_maneuver": MANEUVER_CLASSES[int(pred[o])],
                      "pred_confidence": round(float(prob[o].max()), 4),
                      "ego_speed_mps": round(float(data["speed"][i]), 3)})
    return {"latent_source": src, "split_seed": seed, "worst": items}


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def plot_confusions(report: dict, out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                          # pragma: no cover
        return [f"matplotlib unavailable: {e}"]
    made = []
    src = "tactical_k16" if "tactical_k16" in report["tactical_maneuver_probe"] \
        else next(iter(report["tactical_maneuver_probe"]), None)
    if src is None:
        return made
    for scope, per in report["tactical_maneuver_probe"][src].items():
        m = per.get("linear") if isinstance(per, dict) else None
        if not m or "confusion_matrix" not in m:
            continue
        cm = torch.tensor(m["confusion_matrix"]).double()
        cmn = cm / cm.sum(1, keepdim=True).clamp_min(1)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_MAN)); ax.set_yticks(range(N_MAN))
        ax.set_xticklabels(MANEUVER_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(MANEUVER_CLASSES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
        ax.set_title(f"maneuver probe ({src}, linear)\n{scope}  "
                     f"balacc={m['balanced_accuracy']} macroF1={m['macro_f1']} "
                     f"maj={m['majority_baseline_acc']}", fontsize=9)
        for a in range(N_MAN):
            for b in range(N_MAN):
                ax.text(b, a, f"{int(cm[a, b])}", ha="center", va="center",
                        fontsize=7, color="black" if cmn[a, b] < 0.5 else "white")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fn = out_dir / f"confusion_maneuver_{src}_{scope.strip('_')}.png"
        fig.savefig(fn, dpi=130); plt.close(fig)
        made.append(str(fn))
    return made


# --------------------------------------------------------------------------- #
# STEP-0 self-documentation from the loaded checkpoint                         #
# --------------------------------------------------------------------------- #
def step0_findings(world, state_dict: dict) -> dict:
    import re
    keys = list(state_dict.keys())
    hits = [k for k in keys
            if re.search(r"maneuver|route|nav|strateg|intent|vq|codebook", k, re.I)]
    tac_heads = sorted(k.split(".")[2] for k in keys
                       if k.startswith("tactical_pred.heads") and k.endswith("weight"))
    op_heads = sorted(k.split(".")[2] for k in keys
                      if k.startswith("predictor.heads") and k.endswith("weight"))
    return {
        "operative_horizons": op_heads,
        "tactical_present": world.tactical_pred is not None,
        "tactical_horizons": tac_heads,
        "tactical_heads_emit": "future LATENTS (nn.Linear -> state_dim), "
                               "NOT a maneuver distribution",
        "maneuver_route_strategic_vq_keys_in_ckpt": hits or "NONE",
        "intrinsic_maneuver_selector": bool(hits),
        "verdict": ("Flagship emits operative + tactical imagined LATENTS and an "
                    "H15 belief field. No trained maneuver head, no route head, "
                    "no VQ codebook. Maneuver-selection is measured by "
                    "decodability probe (primary) + goal-conditioned "
                    "imagine-and-select (secondary). Strategic selector is a "
                    "Phase-0 gap."),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def _corpus_of(cd: str) -> str:
    low = cd.lower()
    if "comma" in low:
        return "comma2k19"
    if "physical" in low:
        return "physicalai"
    return Path(cd).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output DIR")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--route-turn-deg", type=float, default=45.0)
    ap.add_argument("--route-turn-deg-relaxed", type=float, default=20.0)
    ap.add_argument("--select-steer", type=float, default=0.15)
    ap.add_argument("--select-accel", type=float, default=1.0)
    ap.add_argument("--select-comfort", type=float, default=0.01)
    ap.add_argument("--select-max-windows", type=int, default=2000)
    ap.add_argument("--no-select", action="store_true",
                    help="skip the imagine-and-select section")
    ap.add_argument("--git-hash", default="unknown")
    args = ap.parse_args()

    import math
    from tanitad.config import base250cam_config
    from tanitad.data.mixing import load_episode
    from tanitad.instruments.numerics import strict_numerics
    from tanitad.models.fourbrain import WorldModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes, corpora = [], []
    for cd in args.cache_dirs:
        vd = sorted(Path(cd).glob("*val*"))
        if not vd:
            print(f"[behavior] WARNING no *val* under {cd}", flush=True)
            continue
        for p in sorted(vd[-1].glob("ep_*.pt"))[:args.episodes]:
            episodes.append(load_episode(str(p), mmap=True))
            corpora.append(_corpus_of(cd))
    assert episodes, "no val episodes loaded"

    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    world.load_state_dict(sd)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()
    window = world.predictor.cfg.window
    s0 = step0_findings(world, sd)
    print(f"[behavior] {len(episodes)} val eps, step {step}, window {window}, "
          f"device {device}", flush=True)
    print(f"[behavior] STEP0: {s0['verdict']}", flush=True)

    tr_strict = math.radians(args.route_turn_deg)
    tr_relax = math.radians(args.route_turn_deg_relaxed)
    seeds = list(range(args.seed, args.seed + args.n_splits))

    with strict_numerics():
        data = collect(world, episodes, corpora, device, window, tr_strict,
                       tr_relax, stride=args.stride, batch=args.batch,
                       keep_states=not args.no_select)
    n = int(data["man"].numel())
    print(f"[behavior] collected {n} windows "
          f"({sum(data['route_valid'].tolist())} route-valid)", flush=True)

    corpus_counts: dict = {}
    for c in data["corpus"]:
        corpus_counts.setdefault(c, 0)
        corpus_counts[c] += 1

    scopes = _corpus_scopes(data["corpus"])
    gt_bal = {"maneuver": {sc: class_balance(data["man"][m], N_MAN,
                                             MANEUVER_CLASSES)
                           for sc, m in scopes.items()}}

    print("[behavior] maneuver decodability probe ...", flush=True)
    man_probe = maneuver_probe_eval(data, seeds, args.val_frac, device,
                                    args.epochs)
    print("[behavior] strategic route probe ...", flush=True)
    strat = strategic_probe_eval(data, seeds, args.val_frac, device, args.epochs,
                                 (args.route_turn_deg, args.route_turn_deg_relaxed))
    if args.no_select:
        select = {"skipped": "--no-select"}
    else:
        print("[behavior] imagine-and-select ...", flush=True)
        with strict_numerics():
            select = imagine_and_select_eval(
                world, data, device, seeds, args.val_frac, args.select_steer,
                args.select_accel, args.select_comfort, args.select_max_windows)
    worst = worst_k_errors(data, device, args.val_frac, args.seed, args.epochs,
                           "tactical_k16" if data["has_tactical"] else "encoder_state")

    report = {
        "exp": "behavior-eval-tactical-strategic",
        "ckpt": args.ckpt, "step": step, "git_hash": args.git_hash,
        "config": {"episodes_per_dir": args.episodes, "stride": args.stride,
                   "window": window, "n_splits": args.n_splits,
                   "val_frac": args.val_frac, "epochs": args.epochs,
                   "maneuver_horizon_steps": MANEUVER_H, "hz": 10,
                   "route_horizon_steps": ROUTE_H,
                   "select_horizon_steps": SELECT_H, "fp32": True,
                   "operative_latent_horizon": data["op_horizon"],
                   "tactical_latent_horizons": data["tac_horizons"]},
        "step0_findings": s0,
        "corpora": corpus_counts, "n_windows": n,
        "gt_class_balance": gt_bal,
        "tactical_maneuver_probe": man_probe,
        "tactical_imagine_and_select": select,
        "strategic_route_probe": strat,
        "worst_k_maneuver_errors": worst,
        "definitions": {
            "maneuver_classes": list(MANEUVER_CLASSES),
            "route_classes": list(ROUTE_CLASSES),
            "balanced_accuracy": "mean per-class recall over GT-present classes",
            "macro_f1": "unweighted mean F1 over GT-present classes",
            "probe_note": "A3 doctrine — classifier reads the model's own "
                          "imagined latents; DECODABILITY, not an intrinsic "
                          "selector. Class-weighted CE so lane-keep prior "
                          "cannot swamp minority maneuvers.",
        },
    }
    report["png_files"] = plot_confusions(report, out_dir)

    out_json = out_dir / "behavior_eval.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print("\n=== BEHAVIOR EVAL SUMMARY ===", flush=True)
    for sc in ("_all", "comma2k19", "physicalai"):
        b = gt_bal["maneuver"].get(sc)
        if b:
            print(f"  GT maneuver balance [{sc}]: {b['frac']}", flush=True)
    src = "tactical_k16" if data["has_tactical"] else "encoder_state"
    for sc in ("_all", "comma2k19", "physicalai"):
        cell = man_probe.get(src, {}).get(sc, {})
        lin = cell.get("linear") if isinstance(cell, dict) else None
        if lin and "balanced_accuracy" in lin:
            print(f"  probe {src} lin [{sc}]: balacc={lin['balanced_accuracy']} "
                  f"macroF1={lin['macro_f1']} maj={lin['majority_baseline_acc']}",
                  flush=True)
    if isinstance(select, dict) and "overall" in select:
        print(f"  imagine-and-select: {select['overall']}", flush=True)
    print(f"[behavior] report -> {out_json}", flush=True)
    print("BEHAVIOR_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
