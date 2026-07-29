"""TanitEval — E-CR: does our imagination COMPOUND, or does the task just get harder?

⭐ WHY THIS EXISTS — it resolves **C61**, a retraction of our own headline.

On 2026-07-29 we measured pure-imagination ADE at K=4/8/12/16/20 (0.4–2.0 s) and reported a local
exponent rising **1.03 → 1.63 → 1.87 → 1.91** as *"decay accelerates"*. The ADE table is MEASURED and
stands. **The interpretation does not**, because ADE-vs-horizon confounds two different causes:

    H-TASK      predicting 2 s ahead is intrinsically harder than 0.4 s ahead
    H-COMPOUND  the rollout feeds its own error forward and amplifies it

Every number we reported is equally consistent with both. The missing control is a **teacher-forced
arm at matched steps** — SkyJEPA (arXiv 2606.23444, verified 3-0):

    CR_k = e_k,rollout / e_k,teacher-forced      # 1.0 == no compounding
    ER_k = E[e_k − e_{k−1}]                      # per-step growth

⛔ **Do not use the exponent rise to justify an architecture change until this reports.**
Pre-registration with both outcomes committed: ``Project Steering/PREREG_deep_research_2026-07-29.md``.

## What makes the comparison sound

:func:`teacher_forced_transitions` is a **verbatim copy** of
``tanitad.models.metric_dynamics.rollout_transitions`` with **exactly one line changed** — the window
is advanced with the TRUE latent instead of the predicted one. Same predictor, same actions, same
readout, same windows, same order. That single-line discipline is the whole point: if the two arms
differed anywhere else, CR_k would measure the difference between two implementations rather than the
cost of recursion.

## What is reported, and what was deliberately NOT

``CR_k``  error on ACCUMULATED waypoints — matches our published ``ade_0_2s`` convention exactly.
``ER_k``  the per-step INCREMENT of that error, E[e_k − e_{k−1}].

⛔ **A per-step-displacement CR was designed and then DELIBERATELY DROPPED.** ``step_readout`` emits
each step's displacement in that step's own ROLLING frame, while a GT "step" obtained by differencing
origin-frame waypoints lives in the ORIGIN frame — the two differ by a rotation, so comparing them
would have produced a confident, frame-mismatched number. Recovering the matched frame needs the GT
yaw at every step; until that is built, ``ER_k`` carries the per-step information and is exact.
**A second metric that is subtly wrong is worse than one metric that is right.**

⚠️ Residual caveat that must travel with any result: even under teacher forcing the decoded pose keeps
integrating through SE(2), so ``CR_k`` on accumulated error is not a pure measure of LATENT
compounding — it bounds it from above. If CR_k comes back high, "the world model compounds" and "the
pose integrator compounds" are still not separated, and the matched-frame per-step metric becomes the
next thing to build.
"""
from __future__ import annotations

import torch

from taniteval.stack_guard import ensure_stack_on_path as _ensure_stack

_ensure_stack()

from tanitad.models.metric_dynamics import (accumulate_se2,  # noqa: E402
                                            rollout_transitions)
from driving_diagnostic import gt_ego_waypoints  # noqa: E402
from taniteval import rollout as _rollout  # noqa: E402

WIN = 8


def teacher_forced_transitions(predictor, states, actions, future_actions,
                               z_true, k):
    """``rollout_transitions`` with the state feedback replaced by ground truth.

    ⭐ THE ONLY DIFFERENCE from the canonical roll is the ``win_s`` update: it appends
    ``z_true[:, j]`` where the canonical version appends ``z_hat``. Everything else — the 1-step
    head, the action slide, the ``(z_prev, z_hat)`` pair shape, the loop bounds — is byte-identical
    to ``metric_dynamics.rollout_transitions`` so that any measured difference is attributable to
    recursion alone.

    ``z_true`` is ``[B, k, D]``: the ENCODED true future latents, aligned so that ``z_true[:, j]`` is
    the true latent of the frame the model has just predicted at step ``j``.
    """
    win_s, win_a = states, actions
    trans: list[tuple[torch.Tensor, torch.Tensor]] = []
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]
        trans.append((win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = (future_actions[:, j] if future_actions is not None
                      else win_a[:, -1])
            # ⛔ THE ONE CHANGED LINE — true latent, not z_hat.
            win_s = torch.cat([win_s[:, 1:], z_true[:, j].unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    return trans


def _accum_errors(step_readout, trans, k, gt_wp):
    """-> accumulated waypoint error [B, k] in metres, at every rollout step."""
    dpose = torch.stack([step_readout(trans[j][0], trans[j][1])
                         for j in range(k)], dim=1)          # [B, k, 3]
    wp = accumulate_se2(dpose)                               # [B, k, 2]
    return torch.linalg.norm(wp - gt_wp[:, :k], dim=-1)


@torch.no_grad()
def run(model, step_readout, episodes, device, k_max=20,
        report_k=(4, 8, 16, 20), speed_input=False, yaw_input=False,
        dyn_input=False, max_eps=None, stride=8):
    """Score the SAME windows twice — recursive rollout vs teacher-forced — and return CR_k/ER_k.

    ⚠️ ``episodes`` MUST be the canonical 40-episode parity val set. Anything that re-selects
    episodes breaks cross-arm comparability and must be refused (CLAUDE.md, parity is sacred).

    Returns per-episode-clustered arrays so the caller can run the **paired** episode-cluster
    bootstrap (``taniteval.ci``) — never a combination in quadrature, and never
    ``overlapping_holdout_se``.
    """
    model.eval()
    eps = episodes if max_eps is None else episodes[:max_eps]
    roll_a, tf_a, ep_ix = [], [], []

    for ei, ep in enumerate(eps):
        fr = ep.feats
        T = min(fr.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        for i in range(0, T - WIN - k_max, stride * 8):
            ch = list(range(i, min(i + stride * 8, T - WIN - k_max), stride))
            if not ch:
                continue
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WIN]) for t in ch]
                             ).to(device).float()
            fut = torch.stack([torch.as_tensor(fr[t + WIN:t + WIN + k_max])
                               for t in ch]).to(device).float()
            if fr.dtype == torch.uint8:
                fw = fw.div_(255.0)
                fut = fut.div_(255.0)
            aw = torch.stack([ep.actions[t:t + WIN] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + WIN:t + WIN + k_max] for t in ch]
                             ).to(device)
            aw, fa = _rollout.append_ego(aw, fa, ep.poses, last, speed_input,
                                         yaw_input, dyn_input, device)
            # ⚠️ signature is (poses, last, wp_steps) — NOT k=. Verified by reading
            # driving_diagnostic.py:101, not assumed.
            gt_wp = gt_ego_waypoints(ep.poses, last,
                                     wp_steps=list(range(1, k_max + 1))).to(device)

            states = model.encode_window(fw)
            b, kk = fut.shape[0], fut.shape[1]
            z_true = model.encode(fut.reshape(b * kk, *fut.shape[2:])
                                  ).reshape(b, kk, -1)

            tr_r = rollout_transitions(model.predictor, states, aw, fa, k_max)
            tr_t = teacher_forced_transitions(model.predictor, states, aw, fa,
                                              z_true, k_max)
            ra = _accum_errors(step_readout, tr_r, k_max, gt_wp)
            ta = _accum_errors(step_readout, tr_t, k_max, gt_wp)
            roll_a.append(ra.cpu()); tf_a.append(ta.cpu())
            ep_ix.append(torch.full((ra.shape[0],), ei, dtype=torch.long))

    RA = torch.cat(roll_a); TA = torch.cat(tf_a); E = torch.cat(ep_ix)

    def _cr(rollout, teacher):
        out = {}
        for k in report_k:
            if k > rollout.shape[1]:
                continue
            er = float(rollout[:, k - 1].mean())
            et = float(teacher[:, k - 1].mean())
            out[f"k{k}"] = {
                "e_rollout": round(er, 5), "e_teacher_forced": round(et, 5),
                "CR": round(er / max(et, 1e-9), 4),
                "ER": round(float((rollout[:, k - 1] - rollout[:, k - 2]).mean())
                            if k >= 2 else float("nan"), 5)}
        return out

    return {
        "estimator": "PAIRED episode-cluster bootstrap (taniteval.ci) — apply to "
                     "the per-window arrays returned here. NEVER "
                     "overlapping_holdout_se, NEVER quadrature.",
        "n_windows": int(RA.shape[0]),
        "n_episodes": int(E.max().item() + 1) if E.numel() else 0,
        "k_max": k_max,
        "CR": _cr(RA, TA),
        "_read": (
            "CR is on ACCUMULATED waypoint error, matching our published ade_0_2s convention. It "
            "BOUNDS latent compounding from above rather than isolating it: the decoded pose keeps "
            "integrating through SE(2) even when the latent is corrected. A matched-frame per-step "
            "metric was designed and dropped as frame-mismatched (see module docstring)."),
        "_prereg": (
            "CR flat near 1 (CI covering 1) => H-TASK: 'decay accelerates' is FALSIFIED, the "
            "acceleration is task difficulty, and NO architecture change is justified — E-ROLL, "
            "rollout-recovery training and the Koopman lever are all unmotivated. CR rising "
            "super-linearly with CI excluding 1 at k=16/20 => H-COMPOUND: rollout-recovery "
            "training is indicated, NOT a bigger horizon. Rising but CI covers 1 => UNDERPOWERED; "
            "report it as such and do not pick the convenient reading."),
        "_arrays": {"rollout_accum": RA, "tf_accum": TA, "episode": E},
    }
