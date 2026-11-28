#!/usr/bin/env python3
"""On-policy scorer labels, label_version 4 -- MEASURE 5 "teach the scorer about slow plans". BEHIND A FLAG, DEFAULT OFF.

WHY. EXPLORATORY (eval/raw/e6_sub200_ep015/): at the first on-policy epoch the whole proposal fan got ~1 m/s faster, so
slow plans are missing from the sets the scorer learns from, and a STOP probe (stop_candidate.json `scorer_on_stop`)
measured the scorer misjudging plans unlike its own proposals: predicted no-collision 0.76 for standing still where
the harness says 0.97, TTC 0.76 vs 0.965. The scorer has only ever been taught on its own (now fast) proposals.

WHAT. For a label-free fraction of samples (`--slow-frac`), the labelled SET additionally carries slowed copies of some
of its own proposals -- the same path traversed `factor` times as fast, built by the ONE shared construction
`refe/slow_copies.py` (also the test-time measure's, SPEC_NAVTEST Amendment 6) -- plus one standing-still plan
(zeros[T, 3], the form the STOP probe fed the scorer and W3's STOP agent submits). Each copy is labelled by EXACTLY the
v3 labeller's functions (the teacher's calculators via `onpolicy_label.Scorer.score`, EP relative to the same teacher
advance, NAVSIM's own drivable area + comfort via `navsim_dac.navsim_dac_and_comfort`, `apply_navsim`).

⭐ WHERE THE COPIES GO -- THE ONLY DESIGN THE RUNNING TRAINER CAN READ WITHOUT A RESTART: the copies REPLACE K of the
set's 64 slots. Measured against the live trainer's own code (train.py at the tip, blob 439a76ca):
  * APPEND (a 64+K set): `OnPolicyBank.__init__` refuses any set whose traj is not exactly (n_prop, horizon, 2)
    (train.py:447-450) -- it is counted `n_incomplete` and DROPPED, while the sample's older v3 line survives. The
    copies would be labelled, written, and never trained on.
  * AN EXTRA SET: the bank keeps ONE set per (log, token, step, rank), ranked by (ckpt_step, label_version)
    (train.py:461-470), so a second set either REPLACES the sample's own-proposal set or is discarded. Serving two
    needs OnPolicyBank + TargetBank.__getitem__ + the loop changed, i.e. a restart.
  * REPLACE K OF 64 (this file): shape (64, 20, 2) passes (train.py:447-450); every target carries
    `navsim_dac.violation`, so NAVSIM's drivable area is used (train.py:457-460); (ckpt_step, 4) supersedes
    (ckpt_step, 3) (train.py:464-470); the next epoch boundary re-reads every `onpolicy_*.jsonl` (train.py:1279-1287,
    bank_snapshot :213-222); `__getitem__` serves the set with an all-ones mask (train.py:665-683) and the loop scores it
    as its own set through `REFe.forward(score_extra=...)` -> `score_trajectories` (train.py:1325-1342, model.py:672-682,
    :760), BCE on all 64 slots.
⚠️ THE TRADE-OFF, MEASURED, NOT ASSUMED: the scoring decoder's self-attention spans the set with no mask
(model.py:414-416), so a copy changes the CONTEXT its set-mates are scored in. stop_candidate.json: ONE out-of-
distribution member (STOP) moved the 64's logits by up to 4.16 and their argmax on 128/200 tokens, while an
in-distribution one (a duplicate of the pick) moved it on 5/200. Labels are per trajectory, so a mixed set teaches
context-robust scoring -- but inference sets are pure 64, hence `--slow-frac` keeps pure sets in the diet (default
0.5), and the effectiveness test must read the scorer on PURE sets (no loss on the 64) as well as with copies.

WHICH PROPOSALS ARE COPIED, AND WHY. Every choice is LABEL-FREE: the decoder attends over the set, so a composition
chosen from the labels (e.g. "copy the plans that collide") would let the scorer read the label off the set structure
-- a leak it can never use at inference.
  * sources ('scorer_top', default): the dump-time checkpoint's own top `--slow-n-src` proposals by the PLANNER'S OWN
    aggregate (`REFePlanner.aggregate`, rule navsim_v1 -- called, not copied) of the logits the patched dump writes
    (`onpolicy_dump.py --emit-logits`). These are the plans the planner would pick; their slowed twins are exactly the
    alternatives the scorer must rank, and it is the design the harness is scoring (eval/slow_copies.py ranks r00..).
    With no logits in the chunk (an unpatched dump) the sources fall back to a seeded random draw, and the set says so.
  * the sources STAY in the set (the within-set contrast: one path at 1.0 / 0.75 / 0.5 of its speed);
  * the K replaced slots are a seeded random draw among the NON-sources -- never the scorer's bottom ranks, which
    would stop supervising exactly the plans the scorer believes are bad (a confirmation loop).
COST. Mixed sets label the teacher + all 64 originals (so the discrimination rule and every original's label are
EXACTLY v3's) + K copies. DEFAULT factors = 0.75 only (K = 8 x 1 + 1 = 9: +14 % rollouts per carrying set, ~+7 % at
--slow-frac 0.5): VALIDITY (eval/PREREG_MEASURE5.md) found the teacher's collision label one-directionally wrong on
0.5x copies (222 vs 9 against NAVSIM), so 0.5 is opt-in only (`--slow-factors 0.75,0.5`: K = 17, +26 %); at the default
--slow-frac 0.5 that is ~+13 % on average. The replaced originals' labels go to a SIDE file (`slowaux_*`, never read by
the trainer, whose glob is `onpolicy_*.jsonl`), so the set line stays as lean as v3's for the epoch-boundary parse.

FLAG OFF (the default) = the v3 labeller: the same lines byte-for-byte except the timing fields (`sec`, `at`,
`labeller`), label_version 3. Pinned by eval/validate_slow_labels_v4.py against the real `onpolicy_label.py` worker.

  worker   python onpolicy_label_v4.py --queue <props dir> --out <sets dir> --rank 0|1 --worker v4w0 --slow-copies
  (every v3 flag is accepted with its v3 meaning; the --slow-* flags only matter with --slow-copies)

--repair-last-heading (measure 5r, "v4r", eval/PREREG_MEASURE5R.md; DEFAULT OFF): every REFe candidate the labelling
functions see (the 64 originals, the copies, STOP) is first passed through `planner.repair_last_heading` -- SPEC
Amendment 7's executed-plan repair, imported, never copied: heading[19] := heading[18], positions untouched. The
REFERENCE candidate (teacher / human future) and the SERVED set (`traj`, `yaw`: what the scorer reads) are NOT repaired:
at inference the scorer scores the unrepaired proposals and the planner repairs AFTER selection, so a slot's label
becomes the outcome of EXECUTING that slot's plan. Lines carry label_version 5 and "repair": "last_heading_hold".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import onpolicy_label as OL  # noqa: E402  the v3 labeller, UNCHANGED: Scorer, KEEP, apply_navsim, _t, _done_keys
import navsim_dac as ND  # noqa: E402
import slow_copies as SC  # noqa: E402  the ONE time-rescaling construction

LABEL_VERSION = 4
LABEL_VERSION_REPAIRED = 5            # --repair-last-heading (measure 5r): supersedes 4 and 3 in the trainer's rank
REPAIR_TAG = "last_heading_hold"
MUTATIONS = ("", "label_source", "label_source_nocheck",       # deliberate-regression arms, validation only
             "repair_wrong_index", "repair_served", "repair_skip_copies")   # measure 5r G-R1's arms
# --nc-at-fault (measure 5b, OFF by default): the key the TEACHER'S OWN at-fault classifier writes (see
# install_teacher_at_fault). It is NOT one of v3's signals, so the discrimination count must not see it.
AT_FAULT_KEY = "ttc.NuPlanTTC.at_fault.info"
V4_ONLY_KEYS = frozenset({AT_FAULT_KEY})


def install_teacher_at_fault() -> str:
    """Make the teacher's NuPlanTTC ALSO export its own at-fault collision event, as `ttc.NuPlanTTC.at_fault.info`.

    WHY (MEASURED, eval/raw/e6_sub200_ep015/v4_label_validity.json): the teacher's `NuPlanCollision.info` fires on ANY
    overlap while the ego moves faster than 0.05 m/s -- its `BEHIND_COS_THRESHOLD` is defined and never used
    (driverl .../collision/nuplan_collision.py:19,42) -- so a slowed plan that a log-replayed follower drives into reads
    as a collision. On half-speed copies it said collision on 222 of 1,600 slots where NAVSIM's harness finds no
    at-fault collision (9 the other way). The SAME teacher code base already carries nuPlan's at-fault rule, inside
    NuPlanTTC (`_classify_current_at_fault_collisions_for_ttc`: active-front, or moving into a stopped agent; NOT an
    active-rear hit, NOT a stopped ego). This wrapper calls the original forward unchanged and then records that
    classification of the current collisions -- the teacher's own rule, exported, not a rule of ours.
    Process-wide and idempotent; only the v4 labeller installs it, and only under --nc-at-fault."""
    from driverl.env.engine.reward_calculator.nuplan_ttc import NuPlanTTC
    if getattr(NuPlanTTC.forward, "_v4_at_fault", False):
        return "teacher at-fault export already installed"
    orig = NuPlanTTC.forward

    def forward(self, scenario_data, log_scenario_data, rewards_and_infos, **kwargs):
        orig(self, scenario_data, log_scenario_data, rewards_and_infos, **kwargs)
        polygons_now, _, valid = scenario_data.get_recent_agent_polygons()
        controlled = scenario_data.agent_control_manager.controlled_mask
        cur = self._collision.detect_current_collisions_knn(
            polygons_now, valid, controlled, scenario_data.agent_nearest_indices[:, :, -1, :])
        at_fault = self._classify_current_at_fault_collisions_for_ttc(scenario_data, polygons_now, cur)
        rewards_and_infos["NuPlanTTC"]["at_fault"] = {"info": at_fault.any(dim=2) & controlled}

    forward._v4_at_fault = True
    forward.__wrapped__ = orig
    NuPlanTTC.forward = forward
    return "teacher at-fault export INSTALLED (NuPlanTTC.forward wrapped; the original runs unchanged first)"


def apply_at_fault(d: dict) -> dict:
    """--nc-at-fault: the NC key the TRAINER reads (`collision.NuPlanCollision.info`, ScorerBank.components) takes the
    teacher's at-fault event; the raw collision is kept once under `teacher_collision.NuPlanCollision.info`
    (idempotent -- the apply_navsim pattern)."""
    if "teacher_collision.NuPlanCollision.info" not in d:
        d["teacher_collision.NuPlanCollision.info"] = d.get("collision.NuPlanCollision.info")
    af = d.get(AT_FAULT_KEY)
    if af is None or isinstance(af, str):
        raise RuntimeError(f"--nc-at-fault: no teacher at-fault event on this candidate ({af!r})")
    d["collision.NuPlanCollision.info"] = float(af)
    return d


@dataclass(frozen=True)
class SlowSpec:
    factors: tuple = (0.75,)           # validated; 0.5 is opt-in (see the module docstring)
    n_src: int = 8
    stop: bool = True
    frac: float = 0.5
    sources: str = "scorer_top"          # 'scorer_top' (needs the dump's logits) | 'random'
    seed: int = 20260927

    def __post_init__(self):
        if not all(0.0 < float(f) < 1.0 for f in self.factors):
            raise ValueError(f"slow factors must be in (0, 1): {self.factors}")
        if self.sources not in ("scorer_top", "random"):
            raise ValueError(f"--slow-sources must be scorer_top or random, got {self.sources!r}")
        if not 0.0 <= float(self.frac) <= 1.0:
            raise ValueError(f"--slow-frac must be in [0, 1], got {self.frac}")

    @property
    def n_copies(self) -> int:
        return int(self.n_src) * len(self.factors) + int(bool(self.stop))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = [float(f) for f in self.factors]
        return d

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _u64(*parts) -> int:
    """A deterministic 64-bit draw from the sample's IDENTITY only (never from a label, never from wall-clock)."""
    h = hashlib.sha256(json.dumps([str(p) for p in parts]).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def carries(key: tuple, spec: SlowSpec) -> bool:
    """Does this sample's set carry copies? A hash of (seed, log, token, step, rank) against --slow-frac."""
    return (_u64(spec.seed, "carry", *key) / 2.0 ** 64) < float(spec.frac)


def v1_aggregate(logits) -> np.ndarray:
    """[M, 6] logits -> [M] float64: the PLANNER'S OWN selection aggregate (REFePlanner.aggregate, rule navsim_v1),
    called through a shim so there is no second copy of the rule to drift from it."""
    import types

    import torch
    from planner import REFePlanner
    shim = types.SimpleNamespace(rule="navsim_v1", V1_W=REFePlanner.V1_W, PDM_W=REFePlanner.PDM_W)
    return REFePlanner.aggregate(shim, torch.as_tensor(np.asarray(logits, dtype=np.float64))).numpy()


def plan_slow(key: tuple, M: int, logits, spec: SlowSpec):
    """The label-free set composition for one sample, or None (a pure set).

    Returns {"slots": [K replaced slot indices, ascending], "src": [source proposal per slot, -1 = STOP],
    "factor": [time factor per slot, 0.0 = STOP], "src_rank": [the source's rank by the dump-time scorer, -1 when not
    ranked], "sources": how the sources were chosen}. Copies are listed source-major (s0 @ f0, s0 @ f1, s1 @ f0, ...,
    STOP last) and assigned to the ascending slots in that order."""
    if spec.n_copies + spec.n_src > M:
        raise ValueError(f"{spec.n_copies} copies + {spec.n_src} kept sources do not fit a set of {M}")
    if not carries(key, spec):
        return None
    rng = np.random.default_rng(_u64(spec.seed, "slots", *key))
    lg = None if logits is None else np.asarray(logits, dtype=np.float64)
    if spec.sources == "scorer_top" and lg is not None and lg.shape == (M, 6) and np.isfinite(lg).all():
        order = np.argsort(-v1_aggregate(lg), kind="stable")
        src = [int(i) for i in order[:spec.n_src]]
        rank_of = {int(i): r for r, i in enumerate(order)}
        how = "scorer_top"
    else:
        src = sorted(int(i) for i in rng.choice(M, spec.n_src, replace=False))
        rank_of = {}
        how = "random" if spec.sources == "random" else "random(no-logits)"
    copies = [(s, float(f)) for s in src for f in spec.factors] + ([(-1, 0.0)] if spec.stop else [])
    src_set = set(src)
    nonsrc = np.asarray([i for i in range(M) if i not in src_set])
    slots = sorted(int(i) for i in rng.choice(nonsrc, len(copies), replace=False))
    return {"slots": slots, "src": [s for s, _ in copies], "factor": [f for _, f in copies],
            "src_rank": [rank_of.get(s, -1) for s, _ in copies], "sources": how}


def repair_fn(mutate: str = ""):
    """The executed-plan repair the labeller applies under --repair-last-heading: refe/planner.py's OWN function
    (SPEC_NAVTEST Amendment 7). `repair_wrong_index` is a G-R1 deliberate-regression arm (heading[18] := heading[17])."""
    if mutate == "repair_wrong_index":
        def wrong(t):
            out = np.array(t, copy=True)
            out[..., -2, 2] = out[..., -3, 2]
            return out
        return wrong
    from planner import repair_last_heading
    return repair_last_heading


def copy_traj(P: np.ndarray, src: int, factor: float) -> np.ndarray:
    """One copy [T, 3] float64: STOP (factor 0, src -1) is zeros -- the shared construction's domain is (0, 1]."""
    if factor == 0.0:
        return np.zeros(P.shape[1:], dtype=np.float64)
    return SC.slow_copy(np.asarray(P[src], dtype=np.float64), factor)


def ndiff_over(got: dict, names: list) -> int:
    """The builder's discrimination count (onpolicy_label.Scorer.score) restricted to `names`: the number of the
    teacher's signals whose value differs across those candidates. Over the teacher + the 64 originals it is EXACTLY
    v3's `nd` (the keys of `got` are the ones Scorer.score counted: no `_` prefix, no `@last`)."""
    n = 0
    for k in got["teacher"]:
        if k in V4_ONLY_KEYS:
            continue
        vals = set()
        for c in names:
            v = got[c].get(k)
            if not isinstance(v, str) and v is not None:
                vals.add(round(float(v), 6))
        if len(vals) > 1:
            n += 1
    return n


def selfcheck(props, plan: dict, labelled: np.ndarray, served: np.ndarray, repair=None) -> str | None:
    """Re-derive every copy from the RAW props line and assert the array that was LABELLED and the array that is
    SERVED are both that copy, and every other slot is untouched. Independent of the path that built them: it catches
    a copy slot labelled or served with the wrong trajectory (the 'label the original' wiring bug)."""
    P0 = np.asarray(props, dtype=np.float64)
    M = P0.shape[0]
    slots, srcs = plan["slots"], plan["src"]
    if len(set(slots)) != len(slots) or set(slots) & {s for s in srcs if s >= 0}:
        return "replaced slots are not distinct non-source slots"
    for i, (slot, s, f) in enumerate(zip(slots, srcs, plan["factor"])):
        exp = np.zeros(P0.shape[1:]) if f == 0.0 else SC.slow_copy(P0[s], f)
        exp_lab = exp if repair is None else repair(exp)          # measure 5r: the copy is LABELLED as executed
        if not np.array_equal(labelled[i], exp_lab):
            return f"the trajectory LABELLED for slot {slot} is not copy(src {s}, factor {f})"
        if not np.array_equal(served[slot], exp):
            return f"the trajectory SERVED in slot {slot} is not copy(src {s}, factor {f})"
    for j in sorted(set(range(M)) - set(slots)):
        if not np.array_equal(served[j], P0[j]):
            return f"kept slot {j} is not the original proposal"
    return None


def serve_set(P: np.ndarray, targets: list, plan: dict, C: np.ndarray, ctg: list) -> tuple:
    """The served set: the originals with copy i (and its targets) in slot plan['slots'][i]. Returns (served [M,T,3],
    targets [M]). The ONE assembly, used by label_one and by diag_slow_labels_v4_train.py."""
    served, stg = np.array(P, dtype=np.float64, copy=True), list(targets)
    for i, slot in enumerate(plan["slots"]):
        served[slot] = C[i]
        stg[slot] = ctg[i]
    return served, stg


def _line(r, S, nd, served, targets, teacher, label_version, slow, worker, t1, extra=None) -> str:
    """The set line. Key order and rounding are v3's (onpolicy_label.worker); `slow` / `extra` only when ON."""
    M = served.shape[0]
    d = {"kind": "onpolicy_set", "label_version": label_version, "comfort_source": "navsim",
         "log_name": r["log_name"], "token": r["token"],
         "step": int(r["step"]), "rank": int(r["rank"]), "ckpt_step": int(r["ckpt_step"]),
         "aug": r.get("aug"), "ndiff": int(nd), "stride": S.stride,
         "traj": [[[round(float(x), 5), round(float(y), 5)] for x, y in served[k, :, :2]] for k in range(M)],
         "yaw": [[round(float(v), 6) for v in served[k, :, 2]] for k in range(M)],
         "targets": targets,
         "teacher_targets": teacher}
    if slow is not None:
        d["slow"] = slow
    if extra:
        d.update(extra)
    d.update({"labeller": worker, "sec": round(time.time() - t1, 2),
              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return json.dumps(d) + "\n"


def label_one(S, sc, r: dict, spec: SlowSpec | None, worker: str, mutate: str = "",
              nc_at_fault: bool = False, repair: bool = False) -> dict:
    """ONE sample -> {"line": set line or None, "aux": side-file line or None, "status": ..., timings}.

    spec None and nc_at_fault False = the v3 labeller exactly. The originals are scored in the SAME Scorer.score call
    as the copies (one context, one teacher rollout); each candidate's rollout is independent of the others, which
    validate_slow_labels_v4.py asserts bit-for-bit against a copies-free call. nc_at_fault needs
    install_teacher_at_fault() in this process (the worker does it)."""
    if mutate not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutate!r}")
    t1 = time.time()
    out = {"line": None, "aux": None, "status": "written", "sec_scoring": 0.0, "sec_navsim": 0.0, "n_copies": 0}
    tr = np.asarray(r["teacher"], dtype=float)
    P = np.asarray(r["props"], dtype=float)                                         # [M, T, 3]
    M = P.shape[0]
    key = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
    plan = plan_slow(key, M, r.get("logits"), spec) if spec is not None else None
    # measure 5r: the LABELLED originals are the executed (repaired) plans; the SERVED ones stay as proposed
    rep = repair_fn(mutate) if repair else None
    PL = P
    if rep is not None and mutate != "repair_served":
        PL = np.asarray(rep(P), dtype=float)
    cands = [("teacher", OL._t(tr[:, :2]), OL._t(tr[:, 2]))] + \
            [(k, OL._t(PL[k, :, :2]), OL._t(PL[k, :, 2])) for k in range(M)]
    C = L = None
    if plan is not None:
        C = np.stack([copy_traj(P, s, f) for s, f in zip(plan["src"], plan["factor"])])   # [K, T, 3] float64
        L = C
        if mutate in ("label_source", "label_source_nocheck"):
            # ⛔ DELIBERATE REGRESSION (validation only): label the ORIGINAL in each copy's place
            L = np.stack([P[s] if s >= 0 else C[i] for i, s in enumerate(plan["src"])])
        if rep is not None and mutate not in ("repair_served", "repair_skip_copies"):
            L = np.asarray(rep(L), dtype=float)
        cands += [(f"s{i}", OL._t(L[i, :, :2]), OL._t(L[i, :, 2])) for i in range(len(L))]
    got, nd_all = S.score(sc, r, cands)
    out["sec_scoring"] = time.time() - t1
    nd = ndiff_over(got, ["teacher"] + list(range(M)))            # the builder's rule over v3's candidates only
    if plan is None and AT_FAULT_KEY not in got["teacher"] and nd != nd_all:
        raise RuntimeError(f"ndiff_over {nd} != Scorer.score's {nd_all} on a copies-free set: the rule drifted")
    if nd < 3:
        out["status"] = "skipped_ndiff"
        return out

    def keep(c) -> dict:
        d = {kk: got[c].get(kk) for kk in OL.KEEP}
        if nc_at_fault:
            d[AT_FAULT_KEY] = got[c].get(AT_FAULT_KEY)
            apply_at_fault(d)
        return d

    t2 = time.time()
    ego = sc.get_ego_state_at_iteration(int(r["step"]))
    ndv, ncf = ND.navsim_dac_and_comfort(PL, ego, sc.map_api, grid="refe20")         # EXACTLY v3's call (PL is P
    targets = [OL.apply_navsim(keep(k), ndv[k], ncf[k]) for k in range(M)]           # unless --repair-last-heading)
    served, stg, slow = P, targets, None
    if rep is not None and mutate == "repair_served":
        served = np.asarray(rep(P), dtype=float)          # ⛔ G-R1 arm m2: repair the SERVED set, label unrepaired
    if plan is not None:
        cdv, ccf = ND.navsim_dac_and_comfort(L, ego, sc.map_api, grid="refe20")      # the copies, the same function
        ctg = [OL.apply_navsim(keep(f"s{i}"), cdv[i], ccf[i]) for i in range(len(L))]
        served, stg = serve_set(served, targets, plan, C, ctg)
        if mutate != "label_source_nocheck":
            err = selfcheck(r["props"], plan, L, served, repair=repair_fn() if repair else None)
            if err is not None:
                out["status"] = "selfcheck_failed"
                out["error"] = err
                return out
        slow = {"carries": True, "slots": plan["slots"], "src": plan["src"], "factor": plan["factor"],
                "src_rank": plan["src_rank"], "sources": plan["sources"], "spec": spec.sha()}
        out["aux"] = json.dumps({
            "kind": "onpolicy_slowaux", "label_version": LABEL_VERSION, "log_name": r["log_name"],
            "token": r["token"], "step": int(r["step"]), "rank": int(r["rank"]), "ckpt_step": int(r["ckpt_step"]),
            "spec": spec.as_dict(), "slots": plan["slots"],
            "dropped_targets": [targets[s] for s in plan["slots"]],
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
        out["n_copies"] = len(plan["slots"])
    elif spec is not None:
        slow = {"carries": False, "spec": spec.sha()}
    out["sec_navsim"] = time.time() - t2
    lv = LABEL_VERSION if (spec is not None or nc_at_fault) else OL.LABEL_VERSION
    if repair:
        lv = LABEL_VERSION_REPAIRED
    teacher = keep("teacher") if nc_at_fault else {kk: got["teacher"].get(kk) for kk in OL.KEEP}
    extra = {"nc_source": "teacher_at_fault"} if nc_at_fault else {}
    if repair:
        extra["repair"] = REPAIR_TAG
    out["line"] = _line(r, S, nd, served, stg, teacher, lv, slow, worker, t1, extra=extra or None)
    return out


def startup_check(spec: SlowSpec | None) -> str:
    """Fail in seconds, not after the expensive part: every v4-only dependency, on synthetic inputs, with controls
    that must read known values. Returns a one-line report or raises."""
    if spec is None:
        return "measure 5 OFF: the v3 path, no v4-only dependency"
    P = np.cumsum(np.full((64, 20, 3), 0.1), axis=1)
    P[:, :, 2] = 0.0
    for f in spec.factors:
        c = SC.slow_copy(P[0], f)
        if c.shape != P[0].shape or np.array_equal(c, P[0]):
            raise RuntimeError(f"slow_copy({f}) is not a distinct same-shape copy")
    if not np.array_equal(SC.slow_copy(P[0], 1.0), P[0]):
        raise RuntimeError("slow_copy(1.0) is not the identity (the shared construction's pinned property)")
    lg = np.zeros((64, 6))
    lg[37] = 5.0                                                           # a known top-1: slot 37
    if spec.sources == "scorer_top":
        agg = v1_aggregate(lg)
        if agg.shape != (64,) or int(np.argmax(agg)) != 37:
            raise RuntimeError("the planner's v1 aggregate does not rank a known top-1 first")
    probe = SlowSpec(factors=spec.factors, n_src=spec.n_src, stop=spec.stop, frac=1.0, sources=spec.sources,
                     seed=spec.seed)
    plan = plan_slow(("probe", "probe", 0, 0), 64, lg, probe)
    if plan is None or len(plan["slots"]) != spec.n_copies or (spec.sources == "scorer_top" and plan["src"][0] != 37):
        raise RuntimeError(f"plan_slow did not produce the declared composition: {plan}")
    return (f"measure 5 ON, spec {spec.sha()} ({spec.sources} sources): slow_copy {SC.__file__}"
            f"{', the planner v1 aggregate ranks a known top-1 first' if spec.sources == 'scorer_top' else ''}, "
            f"{spec.n_copies} copies per carrying set")


def repair_check() -> str:
    """Fail in seconds: planner.repair_last_heading on a synthetic plan must leave x, y and heading[0..18] bit-identical
    and set heading[19] = heading[18] exactly (the control: the synthetic heading[19] differs from heading[18])."""
    P = np.stack([np.arange(20.0), 0.1 * np.arange(20.0), np.linspace(0.0, 0.95, 20)], -1)[None].repeat(3, 0)
    P[:, 19, 2] = 2.5
    Q = np.asarray(repair_fn()(P))
    ok = (np.array_equal(Q[..., :2], P[..., :2]) and np.array_equal(Q[..., :19, 2], P[..., :19, 2])
          and np.array_equal(Q[..., 19, 2], P[..., 18, 2]) and P[0, 19, 2] != P[0, 18, 2] and Q is not P)
    if not ok:
        raise RuntimeError("planner.repair_last_heading does not implement Amendment 7's repair on a synthetic plan")
    import planner
    return f"planner.repair_last_heading ({planner.__file__}) verified on a synthetic plan"


def worker(a, spec: SlowSpec | None) -> int:
    """onpolicy_label.worker's queue protocol, unchanged (claim by atomic rename, one flushed line per sample,
    resume by skipping samples already in its own file, `.done_<worker>` on completion, a status json)."""
    print(f"  startup check: {startup_check(spec)}", flush=True)
    rep_on = bool(getattr(a, "repair_last_heading", False))
    if rep_on:
        print(f"  --repair-last-heading: {repair_check()}", flush=True)
    ncaf = bool(getattr(a, "nc_at_fault", False))
    if ncaf:
        print(f"  --nc-at-fault: {install_teacher_at_fault()}", flush=True)
    S = OL.Scorer(a.rank, a.stride)
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"onpolicy_r{a.rank}_{a.worker}.jsonl")
    aux_path = os.path.join(a.out, f"slowaux_r{a.rank}_{a.worker}.jsonl")
    done = OL._done_keys(out_path)
    stat_path = os.path.join(a.queue, f"status_r{a.rank}_{a.worker}.json")
    lv = LABEL_VERSION if (spec is not None or ncaf) else OL.LABEL_VERSION
    if rep_on:
        lv = LABEL_VERSION_REPAIRED
    st = {"worker": a.worker, "repair": REPAIR_TAG if rep_on else None, "rank": a.rank, "samples": 0, "written": 0, "skipped_ndiff": 0, "failed": 0,
          "chunks": 0, "sec_scoring": 0.0, "label_version": lv, "nc_at_fault": ncaf,
          "slow_spec": spec.as_dict() if spec is not None else None, "slow_sets": 0, "slow_copies": 0,
          "selfcheck_failed": 0, "mutate": a.mutate or None,
          "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"  worker {a.worker} rank {a.rank}: {len(done):,} samples already in {out_path}; measure 5 "
          f"{'ON ' + json.dumps(spec.as_dict()) if spec is not None else 'OFF'}; NC "
          f"{'= the teacher AT-FAULT event' if ncaf else '= the teacher collision event (v3)'}; label_version {lv}"
          f"{'  *** MUTATION ' + a.mutate if a.mutate else ''}", flush=True)
    fh = open(out_path, "a", encoding="utf-8")
    fa = open(aux_path, "a", encoding="utf-8") if spec is not None else None
    while True:
        mine = sorted(glob.glob(os.path.join(a.queue, f"props_r{a.rank}_*.jsonl.{a.worker}")))
        chunk = mine[0] if mine else None
        if chunk is None:
            for c in sorted(glob.glob(os.path.join(a.queue, f"props_r{a.rank}_*.jsonl"))):
                try:
                    os.rename(c, c + "." + a.worker)
                    chunk = c + "." + a.worker
                    break
                except OSError:
                    continue
        if chunk is None:
            if a.once:
                break
            time.sleep(a.poll_s)
            continue
        rows = []
        try:
            with open(chunk, encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            # ⚠️ A CLAIM THAT LOST THE RENAME RACE. MEASURED 2026-09-27 on the dev box's exFAT D: -- two workers'
            # os.rename of ONE chunk both returned, and the loser died opening a name that never existed (v3's worker
            # has the same lines). On Linux the loser's rename raises and never gets here. Skip: the chunk is the
            # other worker's, and a crash here would only cost this worker's remaining throughput.
            print(f"    lost the claim race for {os.path.basename(chunk)}; moving on", flush=True)
            continue
        by_log: dict = {}
        for r in rows:
            by_log.setdefault(r["log_name"], []).append(r)
        for lg, rs in by_log.items():
            try:
                scs = S.scenarios(lg, sorted({r["token"] for r in rs}))
            except Exception as exc:
                st["failed"] += len(rs)
                print(f"    {lg[:34]}: scenario build FAILED {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                continue
            for r in rs:
                key = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]), int(r["ckpt_step"]))
                if key in done:
                    continue
                st["samples"] += 1
                sc = scs.get(r["token"])
                if sc is None:
                    st["failed"] += 1
                    continue
                try:
                    res = label_one(S, sc, r, spec, a.worker, a.mutate, nc_at_fault=ncaf, repair=rep_on)
                except Exception as exc:
                    st["failed"] += 1
                    print(f"    {r['token']}: labelling FAILED {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                    continue
                st["sec_scoring"] += res["sec_scoring"]
                st["sec_navsim_dac"] = st.get("sec_navsim_dac", 0.0) + res["sec_navsim"]
                if res["status"] == "skipped_ndiff":
                    st["skipped_ndiff"] += 1
                    done.add(key)
                    continue
                if res["status"] == "selfcheck_failed":
                    st["selfcheck_failed"] += 1
                    st["failed"] += 1
                    print(f"    {r['token']}: SELF-CHECK FAILED, set NOT written: {res.get('error')}", flush=True)
                    continue
                if res["aux"] is not None:
                    fa.write(res["aux"])
                    fa.flush()
                fh.write(res["line"])
                fh.flush()
                done.add(key)
                st["written"] += 1
                st["slow_sets"] += int(res["n_copies"] > 0)
                st["slow_copies"] += int(res["n_copies"])
        os.rename(chunk, chunk.rsplit(".", 1)[0] + ".done_" + a.worker)
        st["chunks"] += 1
        st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["sec_per_sample"] = round(st["sec_scoring"] / max(st["written"] + st["skipped_ndiff"], 1), 2)
        tmp = stat_path + ".tmp"
        json.dump(st, open(tmp, "w"), indent=1)
        os.replace(tmp, stat_path)
        print(f"  chunk {os.path.basename(chunk)}: {len(rows)} samples -> written {st['written']:,} total "
              f"({st['slow_sets']} with {st['slow_copies']} copies), skipped(ndiff) {st['skipped_ndiff']}, "
              f"failed {st['failed']} (self-check {st['selfcheck_failed']}), {st['sec_per_sample']} s/sample",
              flush=True)
        if a.max_chunks and st["chunks"] >= a.max_chunks:
            break
    fh.close()
    if fa is not None:
        fa.close()
    print(f"ZZOPLABEL4_WORKER_EXIT {a.worker} written {st['written']} slow_sets {st['slow_sets']} "
          f"selfcheck_failed {st['selfcheck_failed']}")
    return 0


def spec_from_args(a) -> SlowSpec | None:
    if not a.slow_copies:
        return None
    return SlowSpec(factors=tuple(float(x) for x in a.slow_factors.split(",") if x.strip()), n_src=a.slow_n_src,
                    stop=not a.slow_no_stop, frac=a.slow_frac, sources=a.slow_sources, seed=a.slow_seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True, choices=(0, 1))
    ap.add_argument("--stride", type=int, default=2, help="the bank's stride (build_scorer_targets default)")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", default="v4w0")
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument("--once", action="store_true", help="exit when the queue is empty")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--slow-copies", action="store_true",
                    help="MEASURE 5 ON (label_version 4). Default OFF = exactly the v3 labeller.")
    ap.add_argument("--slow-factors", default="0.75",
                    help="time factors in (0, 1); 0.75 is the validated default, 0.5 is opt-in (validity: its collision labels are one-directionally wrong)")
    ap.add_argument("--slow-n-src", type=int, default=8)
    ap.add_argument("--slow-no-stop", action="store_true", help="omit the standing-still plan")
    ap.add_argument("--slow-frac", type=float, default=0.5, help="fraction of samples whose set carries copies")
    ap.add_argument("--slow-sources", default="scorer_top", choices=("scorer_top", "random"))
    ap.add_argument("--slow-seed", type=int, default=20260927)
    ap.add_argument("--nc-at-fault", action="store_true",
                    help="MEASURE 5b (OFF by default): the NC label = the teacher's OWN at-fault collision rule "
                         "(NuPlanTTC's), not its raw overlap event; the raw value is kept beside it")
    ap.add_argument("--repair-last-heading", action="store_true",
                    help="MEASURE 5r (OFF by default): label every REFe candidate as EXECUTED -- planner."
                         "repair_last_heading (Amendment 7) before the teacher and NAVSIM labelling; served set unchanged")
    ap.add_argument("--mutate", default="", choices=MUTATIONS,
                    help="DELIBERATE REGRESSION arms for eval/validate_slow_labels_v4.py only")
    ap.add_argument("--preflight", action="store_true",
                    help="check every dependency (the v4-only ones AND the teacher's Scorer) and exit; writes nothing")
    a = ap.parse_args()
    if a.mutate and os.environ.get("REFE_V4_ALLOW_MUTATE") != "1":
        print("  REFUSING: --mutate is a validation arm; set REFE_V4_ALLOW_MUTATE=1 to run it deliberately")
        return 2
    spec = spec_from_args(a)
    if a.preflight:
        print(f"  {startup_check(spec)}", flush=True)
        if a.nc_at_fault:
            print(f"  {install_teacher_at_fault()}", flush=True)
        OL.Scorer(a.rank, a.stride)
        lv = LABEL_VERSION if (spec is not None or a.nc_at_fault) else OL.LABEL_VERSION
        if a.repair_last_heading:
            print(f"  {repair_check()}", flush=True)
            lv = LABEL_VERSION_REPAIRED
        print(f"ZZOPLABEL4_PREFLIGHT_OK rank {a.rank} label_version {lv} nc_at_fault {a.nc_at_fault} "
              f"repair {a.repair_last_heading}")
        return 0
    return worker(a, spec)


if __name__ == "__main__":
    sys.exit(main())
