"""TanitEval — the ``blind_conditioning_baseline`` FIREWALL.

THE ONE-SENTENCE RULE
---------------------
**Train a head on the symbolic context ALONE — no image — and if it matches the
real model, the target is a function of the conditioning and the decision
problem is inadmissible.**

WHY IT EXISTS (this is not hypothetical — it is a post-mortem)
--------------------------------------------------------------
``refb_labels.route_target(nav_cmd) = _NAV_TO_ROUTE[nav_cmd]``. The strategic
route head's TARGET was a deterministic lookup of the route COMMAND the same
head was conditioned on. A network with a 4-entry ``nn.Embedding`` on its FiLM
condition reaches cross-entropy **exactly 0.0** by copying that embedding to its
logits — which is what the deployed v1 did, by step ~14 500
(``taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl``). Downstream:

* ``route_acc_nav = 1.0000`` was read as the strategic level working;
* ``route_skill = 0.0000`` on two checkpoints, two window counts and two
  architectures — zero vision-derived route inference, *by construction*;
* the seam was reported ``load_bearing: true`` for months.

A blind baseline would have caught it on **day one, in CPU-minutes**: fed
``nav_cmd`` and nothing else, it scores **1.000** — which is the whole finding.

MEASURED 2026-07-26 on the same labels (``verify_pc1_labels.py``, 17 100 real
PhysicalAI windows): the route target equals ``_NAV_TO_ROUTE[nav_cmd]`` on
**100.00 %** of CE-eligible windows under the v1, v2 **and** v2.1 labelers
alike. No labeler swap fixes it; only a training signal that withholds the
command (LEVER A) can.

WHAT THE CHECK ACTUALLY MEASURES
--------------------------------
Three numbers, and the verdict needs all three:

``blind``      what the symbolic context alone buys.
``majority``   the base rate of the most common class — the floor. A blind
               score at the floor means the context carries nothing.
``real``       the image-using model's score on the SAME windows (optional; the
               check is still useful without it, as a pure circularity test).

===========================================  ====================================
condition                                    verdict
===========================================  ====================================
``blind >= 1 - DETERMINISTIC_EPS``           ``CIRCULAR`` — the target is a
                                             deterministic function of the
                                             conditioning. Inadmissible.
``blind >= real - MATCH_EPS`` (real given)   ``CIRCULAR`` — vision buys nothing
                                             the context did not already give.
``blind - majority >= SKILL_EPS``            ``LEAKY`` — the context carries
                                             real target information even if it
                                             does not determine it. Admissible
                                             only if every reported number is
                                             a *skill over the blind baseline*,
                                             never a raw score.
otherwise                                    ``CLEAN``
===========================================  ====================================

HONEST LIMITS
-------------
1. **A pass is not proof of a good problem** — only that this particular
   leak is absent. It says nothing about coverage, power, or whether the
   target means what its name says.
2. **The blind head must be at least as expressive as the leak.** A linear
   probe cannot express an XOR-shaped leak. The default is a 1-hidden-layer
   MLP; ``hidden=0`` gives the linear probe, and both are reported.
3. **Episode-clustered split, always.** A within-episode split makes almost any
   target look learnable from context, because context is near-constant inside
   an episode. The split here is over EPISODES and the interval is the
   episode-cluster bootstrap.
"""
from __future__ import annotations

import numpy as np
import torch

from taniteval import ci as _ci

BLOCK = "taniteval.blind_baseline/firewall"
VERSION = "1.0.0"
SPEC = ("TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
        "2026-07-26-4brain-preconditions/PRECONDITIONS_IMPLEMENTED.md (T3)")

#: blind >= 1 - eps  =>  the target IS the context (route_target's failure mode)
DETERMINISTIC_EPS = 0.02
#: blind within eps of the image-using model  =>  vision buys nothing
MATCH_EPS = 0.02
#: blind above the majority floor by eps  =>  the context leaks, even if it does
#: not determine. Same magnitude as `hierarchy.MIN_ACC`, same reason.
SKILL_EPS = 0.03

VERDICTS = ("CIRCULAR", "LEAKY", "CLEAN")


class CircularTarget(AssertionError):
    """A decision problem whose target is predictable from its conditioning."""


# ========================================================================== #
# the blind head                                                              #
# ========================================================================== #
def _fit_predict(x_tr, y_tr, x_te, hidden, n_class, seed, steps, lr):
    """Tiny classifier on symbolic features. CPU, deterministic, seconds."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d = x_tr.shape[1]
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, n_class))
    else:
        net = torch.nn.Linear(d, n_class)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xb, yb = torch.as_tensor(x_tr).float(), torch.as_tensor(y_tr).long()
    n = xb.shape[0]
    bs = min(256, n)
    for i in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(xb[idx]), yb[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(torch.as_tensor(x_te).float()).argmax(-1).numpy()


def _onehot_block(col):
    """A symbolic column -> one-hot. Categorical context must NOT be fed as an
    integer: a linear probe on the integer can only express a monotone ramp and
    would MISS a lookup-table leak — exactly the leak this file exists for."""
    col = np.asarray(col)
    if col.dtype.kind in "fc":                       # already continuous
        return col.reshape(-1, 1).astype(np.float64)
    vals = np.unique(col)
    return (col.reshape(-1, 1) == vals.reshape(1, -1)).astype(np.float64)


def build_context_matrix(context: dict) -> tuple:
    """``{name: [N]}`` -> ``([N, D] float, [names])``. Ints/strings one-hotted."""
    if not context:
        raise ValueError("blind baseline needs at least one context field")
    blocks, names = [], []
    n = None
    for k, v in context.items():
        b = _onehot_block(v)
        n = b.shape[0] if n is None else n
        if b.shape[0] != n:
            raise ValueError(f"context field {k!r} has {b.shape[0]} rows, "
                             f"expected {n}")
        blocks.append(b)
        names += [k] * b.shape[1]
    return np.concatenate(blocks, axis=1), names


# ========================================================================== #
# the firewall                                                                #
# ========================================================================== #
def blind_conditioning_baseline(context: dict, target, eid, *,
                                real_pred=None, problem="unnamed",
                                n_folds=4, hidden=16, seed=0, steps=400,
                                lr=0.05, n_boot=_ci.DEFAULT_N_BOOT):
    """Can the SYMBOLIC CONTEXT ALONE predict the target?

    ``context``   ``{field: [N]}`` — everything the real model gets BESIDES
                  pixels: nav command, maneuver class, map class, ego speed…
    ``target``    [N] int class labels (the decision problem's target).
    ``eid``       [N] episode ids — the resampling AND splitting unit.
    ``real_pred`` [N] optional predictions of the image-using model, so the
                  "vision buys nothing" arm of the test can run.

    Returns the firewall record; ``verdict`` is one of :data:`VERDICTS`."""
    y = np.asarray(target).astype(np.int64)
    eids = np.asarray([str(e) for e in eid])
    X, names = build_context_matrix(context)
    if not (len(y) == len(eids) == X.shape[0]):
        raise ValueError(f"length mismatch: X {X.shape[0]}, y {len(y)}, "
                         f"eid {len(eids)}")
    n_class = int(y.max()) + 1
    uniq = np.unique(eids)
    if len(uniq) < 2:
        raise ValueError("blind baseline needs >= 2 episodes: a within-episode "
                         "split makes almost any target look context-learnable")
    n_folds = int(min(n_folds, len(uniq)))

    # --- episode-clustered cross-validated blind prediction ----------------- #
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    folds = np.array_split(order, n_folds)
    blind = np.zeros_like(y)
    lin = np.zeros_like(y)
    for f in folds:
        te_eps = set(uniq[f])
        te = np.array([e in te_eps for e in eids])
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        blind[te] = _fit_predict(X[tr], y[tr], X[te], hidden, n_class,
                                 seed, steps, lr)
        lin[te] = _fit_predict(X[tr], y[tr], X[te], 0, n_class,
                               seed, steps, lr)

    correct = (blind == y).astype(np.float64)
    lin_correct = (lin == y).astype(np.float64)
    maj_cls = int(np.bincount(y, minlength=n_class).argmax())
    maj_correct = (y == maj_cls).astype(np.float64)

    def I(v):                                                   # noqa: E743
        return _ci.episode_cluster_bootstrap(v, list(eids), n_boot=n_boot,
                                             seed=seed)

    blind_acc = float(correct.mean())
    lin_acc = float(lin_correct.mean())
    maj_acc = float(maj_correct.mean())
    out = {
        "block": BLOCK, "version": VERSION, "spec": SPEC, "problem": problem,
        "n_windows": int(len(y)), "n_episodes": int(len(uniq)),
        "n_classes": n_class, "context_fields": sorted(set(names)),
        "blind_accuracy": I(correct),
        "blind_accuracy_linear_probe": I(lin_correct),
        "majority_base_rate": I(maj_correct),
        "blind_skill_over_majority": round(blind_acc - maj_acc, 6),
        "split": {"unit": "episode", "n_folds": n_folds,
                  "why": ("a within-episode split makes almost any target look "
                          "context-learnable, because the context is near-"
                          "constant inside an episode")},
        "head": {"hidden": hidden, "steps": steps, "lr": lr,
                 "note": ("categorical context is ONE-HOT encoded, never fed as "
                          "an integer -- an integer-coded lookup leak is "
                          "invisible to a linear probe")},
        "estimator": {"interval": "episode_cluster_bootstrap",
                      "n_boot": int(n_boot), "seed": int(seed),
                      "resampling_unit": "episode"},
        "thresholds": {"deterministic_eps": DETERMINISTIC_EPS,
                       "match_eps": MATCH_EPS, "skill_eps": SKILL_EPS},
    }

    real_acc = None
    if real_pred is not None:
        r = (np.asarray(real_pred).astype(np.int64) == y).astype(np.float64)
        real_acc = float(r.mean())
        out["real_model_accuracy"] = I(r)
        out["vision_gain_over_blind"] = _ci.paired_episode_cluster_bootstrap(
            r, correct, list(eids), n_boot=n_boot, seed=seed)

    # --- verdict ------------------------------------------------------------ #
    deterministic = blind_acc >= 1.0 - DETERMINISTIC_EPS
    matches_real = (real_acc is not None
                    and blind_acc >= real_acc - MATCH_EPS)
    leaky = (blind_acc - maj_acc) >= SKILL_EPS
    verdict = ("CIRCULAR" if (deterministic or matches_real)
               else "LEAKY" if leaky else "CLEAN")
    out.update({
        "verdict": verdict,
        "admissible": verdict != "CIRCULAR",
        "target_is_deterministic_in_context": bool(deterministic),
        "vision_buys_nothing": bool(matches_real),
        "context_leaks": bool(leaky),
        "_read": _READ[verdict],
        "summary": (f"{problem}: blind {blind_acc:.4f} (linear {lin_acc:.4f}) "
                    f"vs majority {maj_acc:.4f}"
                    + (f" vs real {real_acc:.4f}" if real_acc is not None else "")
                    + f" -> {verdict}  [n={len(y)}/{len(uniq)} eps]"),
    })
    return out


_READ = {
    "CIRCULAR": ("the target is recoverable from the conditioning alone. Any "
                 "score on it measures the lookup, not the model. This is the "
                 "`route_target = _NAV_TO_ROUTE[nav_cmd]` failure -- the head "
                 "reached CE 0.0 by copying its own conditioning embedding "
                 "while route_skill stayed 0.0000. INADMISSIBLE: fix the "
                 "target or withhold the conditioning during supervision."),
    "LEAKY": ("the context does not determine the target but carries real "
              "information about it. Admissible ONLY if every reported number "
              "is a SKILL OVER THIS BLIND BASELINE, never a raw accuracy."),
    "CLEAN": ("the symbolic context alone does no better than the majority "
              "class. A pass here rules out THIS leak and nothing else -- it "
              "says nothing about coverage, power, or whether the target means "
              "what its name says."),
}


# ========================================================================== #
# the registry — a decision problem cannot be registered without passing       #
# ========================================================================== #
DECISION_PROBLEMS: dict = {}


def register_decision_problem(name, *, target, conditioning, firewall,
                              owner=None, notes=None):
    """Register a decision problem — REFUSED unless its firewall record passes.

    ``firewall`` must be a record produced by
    :func:`blind_conditioning_baseline` **on this problem's own data**. There is
    deliberately no ``force`` argument: the whole point is that the check cannot
    be waived by the person whose result depends on waiving it.

        fw = blind_conditioning_baseline(ctx, y, eids, problem="route_v21")
        register_decision_problem("route_v21", target="route_target_v21",
                                  conditioning=["nav_cmd"], firewall=fw)
    """
    if not isinstance(firewall, dict) or firewall.get("block") != BLOCK:
        raise CircularTarget(
            f"cannot register decision problem {name!r}: `firewall` must be a "
            f"record from blind_conditioning_baseline (block={BLOCK!r}). A "
            "decision problem with no blind baseline is how "
            "`route_target = _NAV_TO_ROUTE[nav_cmd]` survived to a shipped "
            "checkpoint.")
    if firewall.get("verdict") not in VERDICTS:
        raise CircularTarget(f"{name!r}: malformed firewall record "
                             f"(verdict={firewall.get('verdict')!r})")
    if not firewall.get("admissible"):
        raise CircularTarget(
            f"REFUSING to register decision problem {name!r}: its target is "
            f"CIRCULAR. {firewall['summary']}\n{firewall['_read']}\n"
            f"Conditioning: {conditioning}. Fix the target or withhold the "
            "conditioning during supervision (see LEVER A, "
            "`cfg.v2_route_from_vision`), then re-run the firewall.")
    rec = {"name": name, "target": target,
           "conditioning": list(conditioning), "owner": owner, "notes": notes,
           "firewall": firewall, "verdict": firewall["verdict"],
           "must_report_skill_over_blind": firewall["verdict"] == "LEAKY"}
    DECISION_PROBLEMS[name] = rec
    return rec


def assert_registered(name):
    """Fail loud when a scored decision problem was never firewalled."""
    if name not in DECISION_PROBLEMS:
        raise CircularTarget(
            f"decision problem {name!r} is not registered: it has no "
            "blind_conditioning_baseline on record, so no number from it is "
            "admissible. Call register_decision_problem() first.")
    return DECISION_PROBLEMS[name]


# ========================================================================== #
# CLI — run the firewall on the REAL strategic route labels                    #
# ========================================================================== #
def main():
    """``python -m taniteval.blind_baseline --cache <epcache dir>``

    Runs the firewall on the actual ``refb_labels`` derivations, all three
    labeler versions, on real PhysicalAI poses. The expected (and MEASURED)
    outcome is **CIRCULAR on all three** — that is the PC1 finding."""
    import argparse
    import json
    import sys
    from pathlib import Path
    ap = argparse.ArgumentParser("taniteval.blind_baseline")
    ap.add_argument("--cache", required=True, help="dir with ep_*.pt")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sys.path.insert(0, "/root/TanitAD/stack")
    sys.path.insert(0, "/root/TanitAD/stack/scripts")
    import refb_labels as rl
    from tanitad.data.mixing import load_episode

    win, max_h = 8, 20
    variants = {
        "v1_route_target": lambda p, t: (rl.nav_command(p, t)[0],
                                         rl.nav_command(p, t)[1],
                                         rl.route_target(rl.nav_command(p, t)[0])),
        "v2_route_target": lambda p, t: (rl.nav_command_v2(p, t)[0],
                                         rl.nav_command_v2(p, t)[1],
                                         rl.route_target_v2(p, t)),
        "v21_route_target": lambda p, t: (rl.nav_command_v21(p, t)[0],
                                          rl.nav_command_v21(p, t)[1],
                                          rl.route_target_v21(p, t)[0]),
    }
    acc = {k: {"nav": [], "y": [], "eid": []} for k in variants}
    for f in sorted(Path(a.cache).glob("ep_*.pt"))[:a.episodes]:
        ep = load_episode(str(f), mmap=True)
        poses = torch.as_tensor(ep.poses)
        T = poses.shape[0]
        for t in range(0, max(0, T - win - max_h), a.stride):
            tl = t + win - 1
            for k, fn in variants.items():
                nav, valid, y = fn(poses, tl)
                if not valid:                 # the CE-eligible subset only
                    continue
                acc[k]["nav"].append(int(nav))
                acc[k]["y"].append(int(y))
                acc[k]["eid"].append(str(ep.episode_id))
    out = {}
    for k, d in acc.items():
        if len(set(d["eid"])) < 2:
            out[k] = {"error": "too few episodes"}
            continue
        out[k] = blind_conditioning_baseline(
            {"nav_cmd": np.asarray(d["nav"])}, np.asarray(d["y"]), d["eid"],
            problem=k, n_boot=a.n_boot)
        print(out[k]["summary"], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"[firewall] wrote {a.out}")


if __name__ == "__main__":
    main()
