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

⛔ THE RARE-EVENT REPAIR (2026-07-27) — READ BEFORE TRUSTING A ``CIRCULAR``
---------------------------------------------------------------------------
This firewall returned ``CIRCULAR`` on **all three** situation targets
(``…/2026-07-26-situation-classifier/artifacts/sc_results.json``) while its own
``context_leaks = 0`` refuted that verdict on every one of them. Both routes to
the wrong verdict were degeneracies of RAW ACCURACY on a rare-positive target:

============  ========  ========  =======  ==============================
target        blind     majority  real     why it wrongly said CIRCULAR
============  ========  ========  =======  ==============================
roundabout    0.9970    0.9970    0.9864   ``blind >= 1 - 0.02`` fires on the
                                           MAJORITY CLASS itself, at a positive
                                           rate of **0.0030**
intersection  0.9743    0.9743    0.8194   ``vision_buys_nothing`` compares
lane_change   0.9787    0.9788    0.9193   ACCURACIES, and a recall-seeking
                                           rare-event model MUST lose that
                                           comparison to "always predict
                                           negative"
============  ========  ========  =======  ==============================

``lane_change``'s blind head scored **below** the floor (skill −7.6e-05) and was
still called circular. **A firewall that returns CIRCULAR on a clean target is
worse than no firewall** — it retires an admissible decision problem.

The repair has three parts, all of the C13 class (*a test that cannot fail is
not a test*):

* the ``blind >= 1 - eps`` test is DEGENERATE when the majority class already
  clears the same bar, so it is disarmed there;
* the ``blind >= real - eps`` accuracy comparison is DEGENERATE when the real
  model does not itself clear the accuracy floor, so it is disarmed there;
* when raw accuracy is degenerate the verdict is decided on **balanced
  accuracy** (macro-averaged per-class recall), whose floor is 1/n_class for
  ANY imbalance — and if even that is undefined the verdict is ``REFUSED``,
  never a silent ``CLEAN`` and never a wrong ``CIRCULAR``.

``REFUSED`` is not admissible for registration: a problem the firewall cannot
adjudicate must not slip through, but it must not be libelled as circular either.

⚠️ Variable arity: :func:`blind_conditioning_baseline` is a FIXED-CLASS
classifier and cannot express a decision point whose option set changes size
(AlpaSim S1). Padding to a fixed arity gives a strictly WEAKER attack, i.e. a
lower bound on the leak — which is the wrong direction for a firewall. Use
:func:`blind_option_baseline` for those; the fixed-class entry point now REFUSES
to be misused for them rather than under-reporting.

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
4. **Raw accuracy is not a rare-event statistic.** See the repair above; when
   the base rate is extreme the verdict is decided on balanced accuracy and the
   record says so in ``statistic``.
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

VERDICTS = ("CIRCULAR", "LEAKY", "CLEAN", "REFUSED")


class CircularTarget(AssertionError):
    """A decision problem whose target is predictable from its conditioning."""


class FirewallRefused(AssertionError):
    """The firewall cannot adjudicate this target — and says so out loud."""


def balanced_accuracy(y_true, y_pred, n_class: int):
    """Macro-averaged per-class recall, and the classes it could be computed on.

    The floor is ``1/n_class`` for ANY class imbalance, which is exactly the
    property raw accuracy lacks and the whole reason a 0.003-positive-rate
    target broke the raw-accuracy tests. Returns ``(value, n_classes_present)``;
    the value is ``nan`` when no class has support.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean())
               for c in range(n_class) if (y_true == c).any()]
    return (float(np.mean(recalls)) if recalls else float("nan")), len(recalls)


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
                                lr=0.05, n_boot=_ci.DEFAULT_N_BOOT,
                                n_options=None):
    """Can the SYMBOLIC CONTEXT ALONE predict the target?

    ``context``   ``{field: [N]}`` — everything the real model gets BESIDES
                  pixels: nav command, maneuver class, map class, ego speed…
    ``target``    [N] int class labels (the decision problem's target).
    ``eid``       [N] episode ids — the resampling AND splitting unit.
    ``real_pred`` [N] optional predictions of the image-using model, so the
                  "vision buys nothing" arm of the test can run.

    ``n_options`` [N] optional per-window option-set size. Supplying it is how a
                  caller declares an OPTION-CHOICE problem; if the sizes are not
                  all equal this function REFUSES, because padding a
                  variable-arity option set into a fixed class vocabulary is a
                  strictly WEAKER attack — a lower bound on the leak, which is
                  the wrong direction for a firewall to err in.

    Returns the firewall record; ``verdict`` is one of :data:`VERDICTS`."""
    if n_options is not None:
        k = np.asarray(n_options).astype(np.int64)
        if k.size and int(k.min()) != int(k.max()):
            raise FirewallRefused(
                f"{problem!r}: option-set sizes vary ({int(k.min())}..."
                f"{int(k.max())}). This entry point is a FIXED-CLASS "
                "classifier; padding a variable-arity option set into it gives "
                "a strictly WEAKER attack, i.e. a LOWER BOUND on the leak — the "
                "wrong direction for a firewall. Use "
                "`blind_option_baseline(option_context, group, is_chosen, eid)`, "
                "which scores each real option and softmaxes over that decision "
                "point's own option set.")
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

    # --- ⛔ DEGENERACY AUDIT — can each test fail at all on THIS target? ----- #
    # C13, applied to the firewall itself. Every clause below was measured
    # firing on a clean target before it was disarmed; see the module docstring.
    det_degenerate = maj_acc >= 1.0 - DETERMINISTIC_EPS
    match_degenerate = (real_acc is not None
                        and real_acc < maj_acc + SKILL_EPS)
    max_possible_skill = 1.0 - maj_acc
    leak_degenerate = max_possible_skill < SKILL_EPS
    bal_blind, n_present = balanced_accuracy(y, blind, n_class)
    bal_real = (balanced_accuracy(y, np.asarray(real_pred).astype(np.int64),
                                  n_class)[0]
                if real_pred is not None else None)
    bal_floor = 1.0 / n_class
    rare = det_degenerate or match_degenerate or leak_degenerate
    out["degeneracy_audit"] = {
        "majority_base_rate_point": round(maj_acc, 6),
        "max_possible_blind_skill": round(max_possible_skill, 6),
        "deterministic_test_degenerate": bool(det_degenerate),
        "vision_buys_nothing_test_degenerate": bool(match_degenerate),
        "leak_test_degenerate": bool(leak_degenerate),
        "raw_accuracy_scale_is_degenerate": bool(rare),
        "balanced_accuracy_blind": (None if not np.isfinite(bal_blind)
                                    else round(bal_blind, 6)),
        "balanced_accuracy_real": (None if bal_real is None
                                   or not np.isfinite(bal_real)
                                   else round(bal_real, 6)),
        "balanced_accuracy_floor": round(bal_floor, 6),
        "n_classes_with_support": int(n_present),
        "_read": (
            "raw accuracy cannot decide this target: "
            + ("the MAJORITY CLASS alone clears `blind >= 1 - eps`; "
               if det_degenerate else "")
            + ("the real model does not clear the accuracy floor, so "
               "`blind >= real - eps` cannot distinguish 'vision buys nothing' "
               "from 'accuracy is the wrong statistic'; " if match_degenerate else "")
            + ("the largest attainable blind skill is below SKILL_EPS, so the "
               "LEAKY test cannot fire; " if leak_degenerate else "")
            + "the verdict is decided on BALANCED accuracy instead"
            if rare else "raw accuracy is a live statistic on this target"),
    }
    if bal_real is not None:
        out["balanced_accuracy_real"] = bal_real

    # --- verdict ------------------------------------------------------------ #
    if not rare:
        statistic = "accuracy"
        deterministic = blind_acc >= 1.0 - DETERMINISTIC_EPS
        matches_real = (real_acc is not None
                        and blind_acc >= real_acc - MATCH_EPS)
        leaky = (blind_acc - maj_acc) >= SKILL_EPS
        verdict = ("CIRCULAR" if (deterministic or matches_real)
                   else "LEAKY" if leaky else "CLEAN")
    elif n_present < 2 or not np.isfinite(bal_blind):
        # nothing left that can fail: refuse rather than invent a verdict.
        statistic = "none"
        deterministic = matches_real = leaky = False
        verdict = "REFUSED"
    else:
        statistic = "balanced_accuracy"
        deterministic = bal_blind >= 1.0 - DETERMINISTIC_EPS
        # only compare against a real model that HAS balanced skill to be bought
        matches_real = (bal_real is not None
                        and np.isfinite(bal_real)
                        and (bal_real - bal_floor) >= SKILL_EPS
                        and bal_blind >= bal_real - MATCH_EPS)
        leaky = (bal_blind - bal_floor) >= SKILL_EPS
        verdict = ("CIRCULAR" if (deterministic or matches_real)
                   else "LEAKY" if leaky else "CLEAN")

    out.update({
        "verdict": verdict,
        "statistic": statistic,
        "admissible": verdict not in ("CIRCULAR", "REFUSED"),
        "target_is_deterministic_in_context": bool(deterministic),
        "vision_buys_nothing": bool(matches_real),
        "context_leaks": bool(leaky),
        "_read": _READ[verdict],
        "summary": (f"{problem}: blind {blind_acc:.4f} (linear {lin_acc:.4f}) "
                    f"vs majority {maj_acc:.4f}"
                    + (f" vs real {real_acc:.4f}" if real_acc is not None else "")
                    + (f" | balanced blind {bal_blind:.4f} vs floor "
                       f"{bal_floor:.4f}" if statistic == "balanced_accuracy"
                       else "")
                    + f" -> {verdict} (on {statistic})"
                      f"  [n={len(y)}/{len(uniq)} eps]"),
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
    "REFUSED": ("⛔ THE FIREWALL COULD NOT ADJUDICATE THIS TARGET, and says so "
                "rather than guessing. Every test it owns is DEGENERATE here: "
                "on a target this imbalanced the majority class alone clears "
                "`blind >= 1 - eps`, the largest attainable blind skill is "
                "below SKILL_EPS, and comparing ACCURACIES to a recall-seeking "
                "model is a comparison that model must lose. This is NOT "
                "'circular' and NOT 'clean' -- it is UNADJUDICATED. Score the "
                "target with a rare-event statistic (AP / recall at a fixed "
                "budget) against a comparator that is actually chance "
                "(`taniteval.rank_metrics`), or re-pose the target so its "
                "classes have comparable support."),
}


# ========================================================================== #
# VARIABLE ARITY — the attack the fixed-class entry point cannot express       #
# ========================================================================== #
def _fit_predict_options(x_tr, m_tr, c_tr, x_te, m_te, hidden, seed, steps, lr):
    """Shared per-option scorer + masked softmax over each group's OWN options.

    This is what makes the attack arity-exact: the network sees one option at a
    time, so a decision point with 2 options and one with 7 are the same problem
    to it, and the softmax normalises over exactly the options that existed.
    Padding into a fixed class vocabulary (what the fixed-class entry point
    forces) throws that away and under-reports the leak.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d = x_tr.shape[-1]
    net = (torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, 1))
           if hidden else torch.nn.Linear(d, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.as_tensor(x_tr).float()
    M = torch.as_tensor(m_tr).bool()
    C = torch.as_tensor(c_tr).long()
    n = X.shape[0]
    bs = min(256, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad()
        s = net(X[idx]).squeeze(-1).masked_fill(~M[idx], float("-inf"))
        torch.nn.functional.cross_entropy(s, C[idx]).backward()
        opt.step()
    with torch.no_grad():
        s = net(torch.as_tensor(x_te).float()).squeeze(-1)
        return s.masked_fill(~torch.as_tensor(m_te).bool(),
                             float("-inf")).argmax(-1).numpy()


def blind_option_baseline(option_context: dict, group, is_chosen, eid, *,
                          real_choice=None, problem="unnamed", n_folds=4,
                          hidden=16, seed=0, steps=400, lr=0.05,
                          n_boot=_ci.DEFAULT_N_BOOT):
    """The firewall for a VARIABLE-ARITY option choice (AlpaSim S1's shape).

    Every array is per **(decision point, option)** row, length M:

    ``option_context``  ``{field: [M]}`` — what the model knows about each option
                        BESIDES pixels (its geometry, its class, the goal's
                        distance to it…).
    ``group``           [M] decision-point id — options of one decision point.
    ``is_chosen``       [M] bool, exactly one True per group: the taken option.
    ``eid``             [M] episode/scene id — the resampling AND splitting unit.
    ``real_choice``     [G] optional index (within each group) chosen by the
                        image-using model.

    Two floors are reported, and the LEAK-relevant one is ``chance``:

    ``chance``     ``mean_i 1/K_i`` — the EXACT expectation of a uniform pick
                   over each decision point's own option set. With variable
                   arity this, not a "majority class", is the floor a
                   circularity firewall must clear.
    ``best_fixed`` the strongest constant rule ("always take option j"), chosen
                   on the training folds and applied out of fold.
    """
    grp = np.asarray([str(g) for g in group])
    chosen = np.asarray(is_chosen).astype(bool)
    eids = np.asarray([str(e) for e in eid])
    X, names = build_context_matrix(option_context)
    if not (len(grp) == len(chosen) == len(eids) == X.shape[0]):
        raise ValueError("option_context / group / is_chosen / eid must all be "
                         "per-(decision point, option) rows of the same length")
    uniq_g, inv = np.unique(grp, return_inverse=True)
    G = len(uniq_g)
    counts = np.bincount(inv, minlength=G)
    if not np.all(np.bincount(inv, weights=chosen.astype(float),
                              minlength=G) == 1.0):
        raise ValueError("every decision point needs EXACTLY ONE chosen option")
    if counts.min() < 2:
        raise FirewallRefused(
            f"{problem!r}: a decision point with fewer than 2 options is not a "
            "choice — there is nothing for a blind head to get right or wrong.")
    K = int(counts.max())
    D = X.shape[1]
    Xp = np.zeros((G, K, D))
    Mp = np.zeros((G, K), bool)
    Cp = np.zeros(G, np.int64)
    slot = np.zeros(len(grp), np.int64)
    seen = np.zeros(G, np.int64)
    for r in range(len(grp)):
        gi = inv[r]
        slot[r] = seen[gi]
        seen[gi] += 1
        Xp[gi, slot[r]] = X[r]
        Mp[gi, slot[r]] = True
        if chosen[r]:
            Cp[gi] = slot[r]
    g_eid = np.empty(G, object)
    g_eid[inv] = eids                                  # constant within a group

    uniq_e = np.unique(g_eid)
    if len(uniq_e) < 2:
        raise ValueError("blind baseline needs >= 2 episodes")
    n_folds = int(min(n_folds, len(uniq_e)))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(uniq_e)), n_folds)
    blind = np.zeros(G, np.int64)
    fixed = np.zeros(G, np.int64)
    for f in folds:
        te_e = set(uniq_e[f])
        te = np.array([e in te_e for e in g_eid])
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        blind[te] = _fit_predict_options(Xp[tr], Mp[tr], Cp[tr], Xp[te], Mp[te],
                                         hidden, seed, steps, lr)
        # the strongest CONSTANT rule, chosen on TRAIN only
        best_j = int(np.bincount(Cp[tr], minlength=K).argmax())
        fixed[te] = np.minimum(best_j, counts[te] - 1)

    correct = (blind == Cp).astype(np.float64)
    fixed_correct = (fixed == Cp).astype(np.float64)
    chance_v = 1.0 / counts.astype(np.float64)

    def I(v):                                                   # noqa: E743
        return _ci.episode_cluster_bootstrap(v, list(g_eid), n_boot=n_boot,
                                             seed=seed)

    blind_acc, fixed_acc = float(correct.mean()), float(fixed_correct.mean())
    chance_acc = float(chance_v.mean())
    floor = max(fixed_acc, chance_acc)
    out = {
        "block": BLOCK, "version": VERSION, "spec": SPEC, "problem": problem,
        "arity": {"variable": bool(counts.min() != counts.max()),
                  "min": int(counts.min()), "max": int(counts.max()),
                  "mean": round(float(counts.mean()), 4)},
        "n_decision_points": int(G), "n_option_rows": int(len(grp)),
        "n_episodes": int(len(uniq_e)), "context_fields": sorted(set(names)),
        "blind_accuracy": I(correct),
        "best_fixed_index_accuracy": I(fixed_correct),
        "chance_accuracy": I(chance_v),
        "blind_skill_over_chance": round(blind_acc - chance_acc, 6),
        "blind_skill_over_best_fixed": round(blind_acc - fixed_acc, 6),
        "blind_vs_chance_paired": _ci.paired_episode_cluster_bootstrap(
            correct, chance_v, list(g_eid), n_boot=n_boot, seed=seed),
        "blind_vs_best_fixed_paired": _ci.paired_episode_cluster_bootstrap(
            correct, fixed_correct, list(g_eid), n_boot=n_boot, seed=seed),
        "split": {"unit": "episode", "n_folds": n_folds},
        "estimator": {"interval": "episode_cluster_bootstrap",
                      "n_boot": int(n_boot), "seed": int(seed),
                      "resampling_unit": "episode"},
        "thresholds": {"deterministic_eps": DETERMINISTIC_EPS,
                       "match_eps": MATCH_EPS, "skill_eps": SKILL_EPS},
        "note": ("arity-exact attack: one shared scorer per OPTION, softmaxed "
                 "over that decision point's own option set. A padded "
                 "fixed-class encoding is strictly weaker and would UNDER-report "
                 "the leak."),
    }

    real_acc = None
    if real_choice is not None:
        r = (np.asarray(real_choice).astype(np.int64) == Cp).astype(np.float64)
        real_acc = float(r.mean())
        out["real_model_accuracy"] = I(r)
        out["vision_gain_over_blind"] = _ci.paired_episode_cluster_bootstrap(
            r, correct, list(g_eid), n_boot=n_boot, seed=seed)

    det_degenerate = floor >= 1.0 - DETERMINISTIC_EPS
    match_degenerate = real_acc is not None and real_acc < floor + SKILL_EPS
    leak_degenerate = (1.0 - floor) < SKILL_EPS
    out["degeneracy_audit"] = {
        "floor_used": round(floor, 6), "chance": round(chance_acc, 6),
        "best_fixed": round(fixed_acc, 6),
        "max_possible_blind_skill": round(1.0 - floor, 6),
        "deterministic_test_degenerate": bool(det_degenerate),
        "vision_buys_nothing_test_degenerate": bool(match_degenerate),
        "leak_test_degenerate": bool(leak_degenerate),
    }
    if leak_degenerate and det_degenerate:
        verdict, deterministic, matches_real, leaky = "REFUSED", False, False, False
    else:
        deterministic = (not det_degenerate) and blind_acc >= 1.0 - DETERMINISTIC_EPS
        matches_real = ((not match_degenerate) and real_acc is not None
                        and blind_acc >= real_acc - MATCH_EPS)
        leaky = (not leak_degenerate) and (blind_acc - floor) >= SKILL_EPS
        verdict = ("CIRCULAR" if (deterministic or matches_real)
                   else "LEAKY" if leaky else "CLEAN")
    out.update({
        "verdict": verdict, "statistic": "option_choice_accuracy",
        "admissible": verdict not in ("CIRCULAR", "REFUSED"),
        "target_is_deterministic_in_context": bool(deterministic),
        "vision_buys_nothing": bool(matches_real),
        "context_leaks": bool(leaky), "_read": _READ[verdict],
        "summary": (f"{problem}: blind {blind_acc:.4f} vs chance "
                    f"{chance_acc:.4f} vs best-fixed {fixed_acc:.4f}"
                    + (f" vs real {real_acc:.4f}" if real_acc is not None else "")
                    + f" -> {verdict}  [G={G}/{len(uniq_e)} eps, "
                      f"K={counts.min()}..{counts.max()}]"),
    })
    return out


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
