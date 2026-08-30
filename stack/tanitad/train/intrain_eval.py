"""In-training VAL-SIDE eval — the thing the trainer has never had (PI Q3).

⛔ THE GAP THIS CLOSES, established at file:line before designing anything.
`train_v6_staged.py:6131` is the save-interval block and it contains **only** the
default-off X2 seam dump and `_save_ckpt`. The only `.eval()` calls in the trainer
are the VLM (`:957`) and the EMA target deepcopy (`:1063`) — neither is a
validation pass. `--v2-val-cache` is refused at startup (P4-5) because the
`ds_val` it built was never read. ⇒ **there is no held-out read anywhere inside a
run**: we cannot see a checkpoint degrade while it trains, and we select
checkpoints blind.

⚠️ AND IT MUST NOT REPEAT THE FAILURE THAT MAKES TRAINER NUMBERS UNQUOTABLE.
*"v1.6 is best-in-program"* came from a TRAINER LOG and was ~10 % optimistic
against `eval_*.py`. The cause was not dishonesty — a trainer watches a curve
under training conditions. So this module: runs under `model.eval()` **explicitly
and records the mode**; takes **held-out episodes**; emits the **same key names**
the post-hoc probes emit, so the numbers land on ONE scale rather than a second.
⛔ Every value it returns is stamped `T0 trainer-side`; it is a MONITOR, not a
capability claim, and a T1 claim still needs `taniteval/tools/t1_eval.py`.

⭐ THE DESIGN DECISION THAT MATTERS: IT IS NOT AN ADE WATCHER.
A distance metric is structurally blind to the programme's largest known failure.
MEASURED: v1.x reproduced an S-curve 97.9 % open-loop and **0.0 %** with the
action held — it was echoing the action channel, not driving — and the current v7
line shows the same signature (closed-loop vs hold-action gap ~1 %, H-ARCH-ACTINS).
An arm that ignores its action input can score EXCELLENT ADE. ⇒ the primary
quantity here is :func:`action_sensitivity`, and ADE is reported beside it as
context, never alone. This mirrors the four-families rule one level down: a scalar
distance cannot see a decision defect.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

__all__ = ["EvalReport", "action_sensitivity", "hold_action_baseline",
           "run_intrain_eval"]


class EvalReport(dict):
    """A plain dict, so it drops straight into ``metrics.json`` unchanged."""


def _ade(pred: Tensor, gt: Tensor) -> Tensor:
    """Mean L2 over waypoints. ``[..., S, 2]`` -> scalar."""
    return (pred - gt).norm(dim=-1).mean()


def action_sensitivity(pred_true: Tensor, pred_alt: Tensor,
                       eps: float = 1e-8) -> dict[str, float]:
    """⭐ How much does the prediction MOVE when the action input changes?

    ``pred_true`` = prediction under the recorded action; ``pred_alt`` =
    prediction under a *different* action (held, zeroed, or perturbed). Both
    ``[..., S, 2]`` in metres.

    Returns a RATIO and its two ABSOLUTE parts, because the ratio alone is
    unreadable at small magnitudes:

    * ``delta_m``     mean ||pred_true - pred_alt||, metres. **The primary number.**
    * ``scale_m``     mean ||pred_true||, metres — what the ratio divides by.
    * ``sensitivity`` ``delta_m / scale_m``.

    ⛔ HOW TO READ IT, and the reading is the point of the probe:
    ``sensitivity -> 0`` means the model's output does not depend on the action it
    was given. That is the ECHO/ACTION-INSENSITIVITY defect, and **a model in that
    state can still post an excellent ADE** — it is reproducing the future from
    the scene while ignoring the control. A falling ``sensitivity`` during a run
    is the single most important thing to see early, because every downstream
    closed-loop number collapses when it reaches zero.

    ⚠️ It is NOT a quality metric and a large value is not automatically good — an
    unstable model also moves a lot. Read it WITH ``val_ade``: high sensitivity +
    low ADE is the healthy quadrant; low sensitivity + low ADE is the echo trap.
    """
    delta = (pred_true - pred_alt).norm(dim=-1).mean()
    scale = pred_true.norm(dim=-1).mean()
    return {"delta_m": float(delta),
            "scale_m": float(scale),
            "sensitivity": float(delta / scale.clamp_min(eps))}


def hold_action_baseline(v0: Tensor, dt: float, n_steps: int) -> Tensor:
    """Constant-velocity straight-line rollout — the control that must be beaten.

    ⛔ THIS IS NOT A COURTESY BASELINE. MEASURED (H-CTRL-1, T1, 6,844 windows):
    the hold-action control beat a released arm on EVERY metric family, including
    a BETTER chance-corrected tactical kappa (0.6449 vs 0.3795). An in-training
    eval that does not carry it can watch a model improve against itself while
    losing to doing nothing.
    """
    t = torch.arange(1, n_steps + 1, device=v0.device, dtype=v0.dtype) * dt
    x = v0.reshape(-1, 1) * t.reshape(1, -1)
    return torch.stack([x, torch.zeros_like(x)], dim=-1)          # [B, S, 2]


@torch.no_grad()
def run_intrain_eval(forward_fn: Callable[[dict], Tensor],
                     batches: Iterable[dict],
                     *,
                     model: Any = None,
                     alt_action_fn: Callable[[dict], dict] | None = None,
                     dt: float = 0.1,
                     max_batches: int = 8,
                     ) -> EvalReport:
    """Run the held-out read. Cheap, deterministic, eval-mode, T0-stamped.

    ``forward_fn(batch) -> [B, S, 2]`` predicted waypoints in metres.
    ``alt_action_fn(batch) -> batch`` returns a COPY with the action channel
    replaced; if ``None``, the action-sensitivity block is reported ABSENT with a
    reason rather than silently skipped.

    ⛔ EVAL MODE IS SET HERE AND THE MODE IS RECORDED (TRAIN-C5: a readout that
    forgot `.eval()` overstated ADE by +175.7 % and its run-to-run wobble was
    dropout, not noise). The previous training mode is restored on exit, so a
    monitor can never leave the model in the wrong mode for the next step.

    ⚠️ ``max_batches`` bounds the cost. This runs every save-interval, so it must
    be a rounding error against the interval, not a second training loop: ~2
    forward passes x ``max_batches``, no backward.
    """
    was_training = bool(getattr(model, "training", False)) if model is not None else None
    if model is not None:
        model.eval()
    try:
        n = 0
        ade_sum = base_sum = 0.0
        # ⛔ A COUNT, NOT `if base_sum:`. The first version gated on the sum's
        # TRUTHINESS, so a baseline that scored a PERFECT 0.0 m was reported as
        # "no baseline computed" — absence and a real zero collapsed into the
        # same branch. That is the exact confusion this module's own
        # empty-val-set guard exists to prevent, reproduced one function down.
        n_base = 0
        sens: list[dict[str, float]] = []
        for batch in batches:
            if n >= max_batches:
                break
            pred = forward_fn(batch)
            gt = batch["gt_wp"]
            ade_sum += float(_ade(pred, gt))

            v0 = batch.get("v0")
            if v0 is not None:
                base = hold_action_baseline(torch.as_tensor(v0), dt, pred.shape[-2])
                base_sum += float(_ade(base.to(pred.device, pred.dtype), gt))
                n_base += 1

            if alt_action_fn is not None:
                sens.append(action_sensitivity(pred, forward_fn(alt_action_fn(batch))))
            n += 1

        if n == 0:
            # ⛔ absence is reported, never averaged into a number. An empty val
            # set that silently reports 0.0 is the failure family this whole
            # module exists inside.
            return EvalReport({"status": "ABSENT", "reason": "no val batches",
                               "n_batches": 0, "eval_mode": True,
                               "_tier": "T0 trainer-side"})

        rep = EvalReport({
            "status": "PRESENT",
            "n_batches": n,
            "val_ade_m": ade_sum / n,
            "eval_mode": True,
            "_tier": "T0 trainer-side monitor — NOT a capability claim; a "
                     "capability claim is T1 via taniteval/tools/t1_eval.py",
            "_read": "read action_sensitivity WITH val_ade_m: low sensitivity "
                     "and low ADE is the ECHO trap, not success",
        })
        if n_base:
            rep["hold_action_ade_m"] = base_sum / n_base
            # ⭐ the number that says whether the model beats doing nothing
            rep["beats_hold_action"] = rep["val_ade_m"] < rep["hold_action_ade_m"]
        if sens:
            rep["action_sensitivity"] = sum(s["sensitivity"] for s in sens) / len(sens)
            rep["action_delta_m"] = sum(s["delta_m"] for s in sens) / len(sens)
        else:
            rep["action_sensitivity"] = None
            rep["action_sensitivity_status"] = (
                "ABSENT — no alt_action_fn supplied, so action dependence was "
                "NOT measured. This is the primary quantity; a run without it "
                "cannot see the echo defect.")
        return rep
    finally:
        if model is not None and was_training:
            model.train()


# ---------------------------------------------------------------------------
# THE VAL ROUND — PI directive 2026-08-30: "one VAL ROUND at each 100 steps"
# ---------------------------------------------------------------------------
class EvalSplitError(RuntimeError):
    """Raised when the split artifact is missing, wrong, or leaking."""


class ValSplit:
    """The shipped eval split, loaded and VERIFIED. Never derived in the trainer.

    ⛔ A TRAINER-DERIVED SPLIT IS NOT REPRODUCIBLE ACROSS ARMS. Two arms with
    different seeds would score different episodes and their curves would not be
    comparable — so this loads a released artifact and refuses anything else.

    ⚠️ THREE SPLITS ARE NOW LIVE (v1 stress, v2, v3 representative). ``sha`` is
    recorded in every val record: with three in circulation an unstamped curve is
    unquotable the moment they diverge. v1/v2 remain resolvable (superseded, not
    deleted), which makes stamping the difference between a readable history and
    a guess.

    ⛔ AND THE INTERPRETATION RULE TRAVELS WITH THE FILE, not with whoever
    remembers it: ``interpretation.inadmissible_use`` is read from the artifact
    and re-stated in :meth:`stamp`, so a consumer cannot plot something the split
    forbids without it appearing beside the number.
    """

    def __init__(self, path, *, require_sha: str | None = None):
        import hashlib
        import json
        import pathlib
        p = pathlib.Path(path)
        raw = p.read_bytes()
        self.sha = hashlib.sha256(raw).hexdigest()
        if require_sha and not self.sha.startswith(require_sha):
            raise EvalSplitError(
                f"[val] ⛔ split sha mismatch: {self.sha[:16]} != {require_sha}. "
                f"Three splits are live and v1/v2 are superseded-but-resolvable, "
                f"so a wrong-but-valid file loads silently. Refusing.")
        d = json.loads(raw)
        self.path = str(p)
        self.version = d.get("version")
        self.eval_ids = frozenset(d["eval_clip_ids"])
        self.train_ids = frozenset(d.get("train_clip_ids") or ())
        self.interpretation = d.get("interpretation") or {}
        self.labels_md5 = d.get("based_on_labels_md5")
        if not self.eval_ids:
            raise EvalSplitError("[val] ⛔ split has no eval_clip_ids")
        # ⛔ THE LEAK GUARD, CHECKED AT LOAD not at review time.
        overlap = self.eval_ids & self.train_ids
        if overlap:
            raise EvalSplitError(
                f"[val] ⛔ {len(overlap)} clip(s) are in BOTH eval and train: "
                f"{sorted(overlap)[:3]}... A val curve computed 300 times per run "
                f"and leaking every time is worse than no curve, because a smooth "
                f"line reads as evidence.")

    def assert_held_out(self, scored_clip_ids) -> None:
        """⛔ THE GUARD THE VAL ROUND EXISTS OR DIES ON.

        MEASURED precedent: TRAIN-C5 was a readout left in train mode, +175.7 %
        ADE. A leak here is worse — on a split with rare competences it would
        look like unexpectedly GOOD numbers, the most believable possible failure,
        and we would celebrate it.
        """
        scored = frozenset(scored_clip_ids)
        stray = scored - self.eval_ids
        if stray:
            raise EvalSplitError(
                f"[val] ⛔ the val round scored {len(stray)} clip(s) NOT in the "
                f"eval split: {sorted(stray)[:3]}... These are TRAIN episodes. "
                f"Refusing — a leaking val curve is worse than none.")

    def stamp(self) -> dict:
        """What every val record carries so the number stays quotable."""
        return {"split_path": self.path, "split_sha": self.sha,
                "split_version": self.version, "n_eval_clips": len(self.eval_ids),
                "based_on_labels_md5": self.labels_md5,
                "inadmissible_use": self.interpretation.get("inadmissible_use"),
                "_read": "quote the AGGREGATE only; per-competence counts on this "
                         "split are too small to support per-competence claims"}


def o5_horizon_ratio(o5_step1: float, o5_stepK: float) -> dict:
    """⭐ THE FREE SIGNAL — already in the log stream, costs nothing.

    ``o5_step1 / o5_stepK``. MEASURED across five banked arms and three different
    recipes, it separates by SCALE rather than by recipe:
      2k arms   0.308, 0.361      ⇒ the k=1 shortcut is ACTIVE
      30k arms  1.030, 0.998, 1.111 ⇒ closed
    The mechanism: n_stack=3 + stride-1 + window=6 makes the k=1 target **67 %
    already-observed**, so k=1 is CHEAP while a longer horizon is not.

    ⚠️ WHAT IT DOES NOT ESTABLISH (carried verbatim): a ratio ~1.0 proves k=1 is
    not CHEAP; it does NOT prove no copying occurs. A model could copy the shared
    67 % and still find the remaining third as hard as a clean horizon.

    ⭐ Unlike ADE this moves when something STRUCTURAL is wrong rather than when
    driving is bad, which is why it belongs in a health monitor.
    """
    if not o5_stepK:
        return {"ratio": None, "status": "ABSENT",
                "reason": "o5_stepK is zero/absent — ratio undefined, not 1.0"}
    r = float(o5_step1) / float(o5_stepK)
    return {"ratio": r, "o5_step1": float(o5_step1), "o5_stepK": float(o5_stepK),
            "regime": ("shortcut-active" if r < 0.6 else
                       "closed" if r > 0.85 else "transitional"),
            "_read": "ratio ~1.0 proves k=1 is not CHEAP, NOT that no copying occurs"}


def should_run_val(step: int, *, monitor_every: int = 100,
                   save_every: int = 1000) -> dict:
    """Which cadence fires at this step. ONE split, TWO cadences.

    ⭐ The PI's directive is a val round every 100 steps; checkpoint SELECTION
    wants the same split at the save interval. Because both read the SAME
    artifact, the monitoring curve and the selection curve are the same curve
    sampled at different intervals — no independence problem, and no second
    budget. (The earlier two-split plan would have needed 85 + 56 clips; one
    split of 141 is strictly stronger than either half.)
    """
    return {"monitor": step > 0 and step % monitor_every == 0,
            "select": step > 0 and step % save_every == 0,
            "monitor_every": monitor_every, "save_every": save_every}


#: ⛔ v7.2 SUPERSEDES the v3 split-manifest. The split is now the DATA LAYOUT —
#: two label files — rather than a manifest a trainer must apply correctly.
#: ⭐ WHY THAT IS STRUCTURALLY BETTER: a trainer that opens the train file
#: CANNOT accidentally score eval clips, because they are not in the file. The
#: load-time leak guard below becomes belt-and-braces rather than the only thing
#: standing between us and a leak. It stays — a guard that has become redundant
#: is not a guard you delete, it is one that should never fire.
#: ⛔ CORRECTED 2026-08-30 — THESE PINS NAMED THE WRONG COPY, AND THE GUARD WOULD
#: THEREFORE HAVE REFUSED THE CANONICAL ARTIFACT AND ACCEPTED THE SUPERSEDED ONE.
#: The previous values (`0a586fe8…` / `bffc9df5…`) are the blobs under
#: `release/_v72_verify/labels/` — a round-trip VERIFY download taken before the
#: schema fix. The canonical build output is `release/v72/`, and its md5s are the
#: ones the producer published. MEASURED: both names resolve to exactly TWO copies
#: on this box with TWO DISTINCT md5s, differing by 2,359 B (train) and 60 B (eval).
#:
#: ⚠️ THE ROOT CAUSE IS NOT A STALE NUMBER, IT IS HOW THE ARTIFACT WAS IDENTIFIED.
#: The pins were captured through `glob.glob("C:/Users/Admin/**/<name>")[0]` — the
#: first hit of a recursive home-directory glob — so the *subject* of the assertion
#: was whichever copy the filesystem happened to yield. A test that pins a hash but
#: chooses its file nondeterministically pins nothing. ⇒ `resolve_v72` below fails
#: on ambiguity instead of picking, and `LabelManifest.to_dict()`'s own warning is
#: the rule this violated: *md5 is the identity — copies exist under several roots
#: and their md5s differ.*
V72 = {
    "train": {"name": "s2_labels_v7.2_train.jsonl.gz", "n": 4572,
              "md5": "0ff902130ce76886b8a925eceed9e3a5"},
    "eval": {"name": "s2_labels_v7.2_eval.jsonl.gz", "n": 147,
             "md5": "aa12c948f062181c3297265b51526ec5"},
}


def resolve_v72(side: str, roots):
    """The ONE path for ``side``, or a refusal naming every candidate.

    ⛔ Never returns "the first match". If several copies exist and their md5s
    DIFFER, that is an ambiguous subject and the caller must be told which copies
    it is choosing between — silently picking one is how the stale pin above was
    captured in the first place. Identical copies under different paths are fine:
    the bytes are the artifact, the path is not.
    """
    import glob as _glob
    name = V72[side]["name"]
    hits = sorted({os.path.realpath(h) for r in roots
                   for h in _glob.glob(os.path.join(r, "**", name),
                                       recursive=True)})
    if not hits:
        return None
    by_md5: dict[str, list[str]] = {}
    for h in hits:
        with open(h, "rb") as fh:
            by_md5.setdefault(hashlib.md5(fh.read()).hexdigest(), []).append(h)
    if len(by_md5) > 1:
        detail = "; ".join(f"{m} -> {', '.join(p)}" for m, p in sorted(by_md5.items()))
        raise EvalSplitError(
            f"[val] ⛔ {name} resolves to {len(hits)} copies with "
            f"{len(by_md5)} DISTINCT md5s: {detail}. Refusing to guess which is "
            f"the artifact — md5 is the identity, and taking the first glob hit "
            f"is exactly how this module's pins came to name a pre-fix verify "
            f"download. Name the canonical root explicitly.")
    return hits[0]

#: ⚠️ THE OVERLAP THAT MUST BE STAMPED ON EVERY COMPARISON.
#: v7.2 eval = the v3 eval set + 6 clips that were held out of training for
#: val40 leakage and then used for nothing. Recovering them cost no labelling
#: run and no training data — but it means OUR EVAL SET NOW OVERLAPS THE
#: DEPLOYED val40 BY 6 CLIPS (15 % of val40, 4 % of our eval).
#:
#: ⛔ NOT a leak: neither side is trained on, and train contains zero val40
#: clips. But our val curve and the published open-loop statistic are NO LONGER
#: FULLY INDEPENDENT, and any comparison between them must say so.
#:
#: ⚠️⚠️ AND IT IS FAR TIGHTER THAN 15 % SUGGESTS. Per D-VAL40-NOLABELS, val40's
#: ONLY labelled clips are exactly these 6. ⇒ if anyone scores val40 on the
#: LABEL-BASED families, they are scoring precisely the clips that are also in
#: our eval set: the two numbers would share their ENTIRE SUPPORT, not 15 % of
#: it. On the label families they are not two measurements, they are one.
VAL40_OVERLAP = {
    "n_clips": 6, "pct_of_val40": 0.15, "pct_of_our_eval": 0.04,
    "is_leak": False,
    "_read": "trajectory/ADE comparisons: partially dependent, state it. "
             "LABEL-FAMILY comparisons: the two numbers share their ENTIRE "
             "support (val40's only labelled clips ARE these 6) — they are not "
             "independent measurements and must never be presented as agreement.",
}


def v72_stamp(side: str, path: str, md5: str) -> dict:
    """The stamp a v7.2 number carries. ⛔ It must name WHICH SIDE, not just the
    blob: with three label releases and three splits live and all resolving, a
    number that says only "v7.2" does not say what it scored."""
    if side not in V72:
        raise EvalSplitError(f"[val] unknown v7.2 side {side!r}; have {sorted(V72)}")
    exp = V72[side]
    if md5 != exp["md5"]:
        raise EvalSplitError(
            f"[val] ⛔ v7.2 {side} md5 {md5} != {exp['md5']}. Three label "
            f"releases resolve; the wrong one opens cleanly. Refusing.")
    return {"labels_release": "v7.2", "labels_side": side,
            "labels_path": path, "labels_md5": md5,
            "n_clips": exp["n"],
            "val40_overlap": VAL40_OVERLAP if side == "eval" else None,
            "_read": "per-competence numbers are INADMISSIBLE at n=147 "
                     "(~13 stop-launch, ~24 highway); quote the aggregate"}
