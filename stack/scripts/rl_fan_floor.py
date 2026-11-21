#!/usr/bin/env python3
"""``rl_fan_floor`` — the RL rung's PRIMARY readout, driven over a BANKED FAN.

⛔ WHY: `B2` of the 2026-09-05 RL package records that ``fan_floor@k``,
fan-collision-vs-replay and diversity **do not exist**, and that they are a hard
blocker on the RL rung's primary endpoint. This is the driver; the arithmetic and
every guard live in ``tanitad.rl.fan_floor``.

WHAT IT CONSUMES — a banked fan ``.npz`` carrying at minimum::

    fan2   [W, K, S, 2]  float   the EMITTED candidate set (ego frame)
    eid    [W]           int     episode id  -> the bootstrap's cluster key

and, optionally, any of::

    c_progress c_headway c_collision c_comfort c_feasibility  [W, K]
    f_contact f_infeasible f_unsafe                            [W, K] uint8
    rank conf                                                  [W, K]
    has_lead                                                   [W]     bool

⭐ THE QUALITY SIGNAL IS RULE-BASED AND ECHO-FREE. ``--quality composed`` builds
``sum_j w_j * c_j`` from the banked per-candidate reward components — progress,
headway, collision, comfort, feasibility — i.e. DD-v2's PDMS *structure* with its
map half removed (PhysicalAI has no map). ⛔ It is NEVER ADE/FDE to the ego's
recorded future: ``fan_floor.assert_quality_admissible`` refuses those by name,
because a floor computed on them measures how tightly the fan hugs the human,
which is the imitation loss and not the RL endpoint.

TWO MODES
  ``--bank X.npz``                 single-fan readout + the mandatory controls.
  ``--bank ARM.npz --base B.npz``  paired episode-cluster bootstrap of every
                                   metric, ARM vs BASE, on the SAME windows.

⛔ TIER: everything printed here is **T0 / fan-level** — a property of what the
model EMITS. A driving claim is T1 with the four families. ⛔ ESTIMATOR: the
paired episode-cluster bootstrap answers *"would another draw of EPISODES say
this?"* — not another training run (`H-ESTIM-SEED-1`) and not another inference
run. Both are named in the output so the interval is never read as the other
question.

Output is ASCII-only on purpose: this repo's console is cp1252 and a non-ASCII
``print()`` is fatal there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.rl import fan_floor as FF                        # noqa: E402

#: DD-v2's structure with the map half removed. Matches
#: ``rewards.DEFAULT_WEIGHTS``' intent: the safety terms PIN, the rest RANK.
COMPOSED_WEIGHTS = {
    "c_progress": 5.0,
    "c_headway": 5.0,
    "c_comfort": 2.0,
    "c_collision": 12.0,
    "c_feasibility": 12.0,
}


def _load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_quality(d, kind):
    """Return (quality [W, K] float64 tensor, name, provenance string)."""
    if kind == "composed":
        parts, used = None, []
        for key, w in COMPOSED_WEIGHTS.items():
            if key not in d:
                continue
            v = torch.as_tensor(np.asarray(d[key], dtype=np.float64))
            parts = v * w if parts is None else parts + v * w
            used.append("%s*%.1f" % (key, w))
        if parts is None:
            raise SystemExit("no c_* reward components in the bank; "
                             "cannot build the composed quality")
        return parts / sum(COMPOSED_WEIGHTS[k] for k in COMPOSED_WEIGHTS
                           if k in d), "reward_composed", "+".join(used)
    if kind == "safety":
        parts, used = None, []
        for key in ("c_collision", "c_headway"):
            if key not in d:
                continue
            v = torch.as_tensor(np.asarray(d[key], dtype=np.float64))
            parts = v if parts is None else parts + v
            used.append(key)
        if parts is None:
            raise SystemExit("no safety components in the bank")
        return parts / float(len(used)), "safety_composed", "+".join(used)
    if kind == "geom_feasibility":
        # ⭐ A CONTROL QUALITY, NOT A SHIPPED METRIC. Recomputed from the
        # GEOMETRY rather than read from a banked column, which is the only way
        # the collapse self-test can exercise its FLOOR half (see `self_test`):
        # a precomputed column cannot move when the waypoints move.
        #
        # ⚠️ dt IS LOAD-BEARING. The banked fan is the 2 s prefix on the **0.5 s
        # grid** (origin + 4 slots; `fan_safety.py`'s own docstring), while
        # `rewards.kinematics` defaults to `DT_S = 0.1`. Taking the default
        # inflates every acceleration by 25x and every jerk by 125x -- a correct
        # formula under the wrong units, which is this programme's most-repeated
        # trap. The grid is derived from the waypoint count and stated in the
        # provenance string so the number carries its units.
        from tanitad.rl import rewards as R
        fan = torch.as_tensor(np.asarray(d["fan2"], dtype=np.float32))
        W, K, S = fan.shape[0], fan.shape[1], fan.shape[2]
        dt = float(d["dt_s"]) if "dt_s" in d else (2.0 / (S - 1))
        kin = R.kinematics(fan.reshape(W * K, S, 2), dt=dt)
        over = ((kin.accel.abs() / R.A_MAX_MPS2).amax(dim=-1)
                .clamp_min((kin.kappa.abs() / R.KAPPA_MAX_1PM).amax(dim=-1)))
        return ((-over).to(torch.float64).reshape(W, K),
                "geom_feasibility",
                "CONTROL-ONLY: -max(|a|/%.1f, |kappa|/%.2f) from fan2 at "
                "dt=%.2f s" % (R.A_MAX_MPS2, R.KAPPA_MAX_1PM, dt))
    if kind.startswith("c_"):
        if kind not in d:
            raise SystemExit("component %s not in the bank" % kind)
        return (torch.as_tensor(np.asarray(d[kind], dtype=np.float64)),
                kind.replace("c_", ""), kind)
    raise SystemExit("unknown --quality %r" % kind)


def readout(d, quality_kind, ks, seed, draws):
    fan = torch.as_tensor(np.asarray(d["fan2"], dtype=np.float64))
    q, qname, prov = build_quality(d, quality_kind)
    flag = None
    for key in ("f_contact", "f_unsafe", "f_infeasible"):
        if key in d:
            flag = torch.as_tensor(np.asarray(d[key], dtype=np.float64))
            flagname = key
            break
    rank = (torch.as_tensor(np.asarray(d["rank"], dtype=np.float64))
            if "rank" in d else None)
    if rank is not None and not torch.isfinite(rank).all():
        # `rank` carries -inf on masked candidates. A sentinel read as a number
        # is how a whole ridge panel silently became its own constant control.
        # Push masked candidates to the BOTTOM of the ranking instead.
        finite = rank[torch.isfinite(rank)]
        rank = torch.where(torch.isfinite(rank), rank,
                           torch.full_like(rank, float(finite.min()) - 1.0))
    r = FF.summarise_fan(fan, q, quality_name=qname, ks=ks,
                         collision_flag=flag, rank=rank,
                         rand_draws=draws, seed=seed)
    r.quality_provenance = prov
    r.flag_name = flagname if flag is not None else None
    return r


def _fmt(x):
    return "%.6f" % float(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", required=True, help="banked fan .npz (the ARM)")
    ap.add_argument("--base", default=None, help="banked fan .npz (the BASE)")
    ap.add_argument("--quality", default="composed",
                    help="composed | safety | c_progress | c_headway | ...")
    ap.add_argument("--ks", default="1,5,8,10,32,64")
    ap.add_argument("--taus", default="0.0,0.25,0.5,0.75")
    ap.add_argument("--rand-draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--lead-only", action="store_true",
                    help="restrict to windows with a lead agent")
    ap.add_argument("--out", default=None, help="write the panel as JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="run the collapse control on the SAME bank and refuse "
                         "to proceed if the instrument cannot fail it")
    a = ap.parse_args(argv)

    ks = tuple(int(x) for x in a.ks.split(",") if x.strip())
    taus = tuple(float(x) for x in a.taus.split(",") if x.strip())

    d = _load(a.bank)
    if a.lead_only and "has_lead" in d:
        m = np.asarray(d["has_lead"]).astype(bool)
        d = {k: (v[m] if getattr(v, "shape", (0,))[:1] == m.shape else v)
             for k, v in d.items()}
    arm = readout(d, a.quality, ks, a.seed, a.rand_draws)

    print("== rl_fan_floor : PRIMARY readout (T0, fan-level; never a driving claim)")
    print("bank            : %s" % a.bank)
    print("windows x cands : %d x %d" % (arm.n_windows, arm.n_candidates))
    print("quality         : %s = %s" % (arm.quality_name, arm.quality_provenance))
    print("collision flag  : %s" % (arm.flag_name or "(absent)"))
    print("")
    print("%-16s %12s %12s   %s" % ("metric", "mean", "rand_best@k", "note"))
    for k in sorted(arm.floor):
        rb = arm.rand_best.get(k)
        print("%-16s %12s %12s   %s" % (
            "fan_floor@%d" % k, _fmt(arm.floor[k].mean()),
            _fmt(rb.mean()) if rb is not None else "-",
            "order stat over K=%d; NOT headroom" % arm.n_candidates))
    for t in sorted(arm.quantile):
        print("%-16s %12s %12s   K-free form" % (
            "fan_q%.2f" % t, _fmt(arm.quantile[t].mean()), "-"))
    print("%-16s %12s %12s   metres, OUR definition (not DDv2's)" % (
        "fan_diversity", _fmt(arm.diversity.mean()), "-"))
    print("%-16s %12s %12s   metres" % (
        "fan_endpoint_std", _fmt(arm.endpoint_std.mean()), "-"))
    for name in sorted(arm.collision):
        print("%-16s %12s %12s   fraction of the candidate SET" % (
            "fan_collision_%s" % name, _fmt(arm.collision[name].mean()), "-"))

    panel = {
        "tool": "stack/scripts/rl_fan_floor.py",
        "tier": "T0-fan",
        "evidence_class": "MEASURED (ours)",
        "bank": a.bank,
        "quality": arm.quality_name,
        "quality_provenance": arm.quality_provenance,
        "collision_flag": arm.flag_name,
        "n_windows": arm.n_windows,
        "n_candidates": arm.n_candidates,
        "arm": {k: float(v.mean()) for k, v in arm.flat().items()},
        "notes": [
            "fan_floor@k is an ORDER STATISTIC over K: not comparable across "
            "different K, and MAXIMISED BY MODE COLLAPSE. Read it beside "
            "fan_diversity, always.",
            "rand_best@k is E[max over a random k-subset] - what a no-skill "
            "selector reaches. Any 'headroom' reading of @k must be quoted "
            "beside it (2026-09-05 retraction: a best-of-N statistic read as a "
            "skill gap).",
            "fan_diversity is OUR definition; DDv2's 42.3->30.3 levels are NOT "
            "comparable, only the direction.",
        ],
    }

    if a.self_test:
        # ⛔ The instrument must be able to FAIL a knowingly-collapsed fan. The
        # control has TWO halves and they are reported separately, because on a
        # bank whose quality columns are PRECOMPUTED only one of them can fire.
        fan = torch.as_tensor(np.asarray(d["fan2"], dtype=np.float64))
        col = FF.collapse_fan(fan, 0.9)
        d2 = dict(d)
        d2["fan2"] = col.numpy()
        arm2 = readout(d2, a.quality, ks, a.seed, a.rand_draws)
        kmax = max(kk for kk in arm.floor)
        v = FF.floor_verdict(arm2, arm, k=kmax)
        div_fired = v["rel_diversity"] <= -0.30
        floor_moved = abs(v["d_floor"]) > 1e-12
        v["div_half"] = "FIRED" if div_fired else "DID-NOT-FIRE"
        v["floor_half"] = "FIRED" if floor_moved else "INERT"
        panel["self_test"] = v
        print("")
        print("-- SELF TEST (deliberate 90 pct fan collapse on this same bank)")
        print("   diversity half : %s   d_diversity %s (rel %.4f)"
              % (v["div_half"], _fmt(v["d_diversity"]), v["rel_diversity"]))
        print("   floor half     : %s   d_floor@%d %s"
              % (v["floor_half"], kmax, _fmt(v["d_floor"])))
        if not div_fired:
            print("   ** INSTRUMENT FAILS ITS OWN CONTROL: a 90 pct collapse "
                  "did not register in diversity. Do not trust the numbers "
                  "above. **")
            return 2
        if not floor_moved:
            v["limit"] = (
                "PARTIAL: the floor half is INERT because --quality %r reads "
                "PRECOMPUTED per-candidate columns, which do not change when "
                "the GEOMETRY is collapsed. Only the diversity half was "
                "exercised here. Re-run with --quality geom_feasibility to "
                "exercise both." % a.quality)
            print("   PARTIAL: the floor half is INERT on a bank with "
                  "precomputed quality columns -- collapsing the geometry")
            print("            cannot move a column that was computed before "
                  "the collapse. Diversity fired; the floor did not.")
            print("            Re-run with --quality geom_feasibility for the "
                  "complete control.")
        else:
            print("   OK: BOTH halves fire, so a floor gain here is readable "
                  "against a knowingly-collapsed fan.")

    if a.base:
        sys.path.insert(0, os.path.join(os.path.dirname(_STACK), "taniteval"))
        from taniteval import ci as CI                       # noqa: E402
        db = _load(a.base)
        if a.lead_only and "has_lead" in db:
            m = np.asarray(db["has_lead"]).astype(bool)
            db = {k: (v[m] if getattr(v, "shape", (0,))[:1] == m.shape else v)
                  for k, v in db.items()}
        base = readout(db, a.quality, ks, a.seed, a.rand_draws)
        FF.assert_equal_k(torch.as_tensor(np.asarray(d["fan2"])),
                          torch.as_tensor(np.asarray(db["fan2"])))
        eid = np.asarray(d["eid"]).reshape(-1)
        af, bf = arm.flat(), base.flat()
        print("")
        print("-- PAIRED episode-cluster bootstrap (arm - base), n_boot=%d"
              % a.n_boot)
        print("   estimator answers: 'would another draw of EPISODES say this?'")
        print("   NOT another training run (H-ESTIM-SEED-1) and NOT another "
              "inference run.")
        print("%-20s %11s %11s %11s  %s" % ("metric", "delta", "lo", "hi", "sep"))
        rows = {}
        for name in sorted(set(af) & set(bf)):
            out = CI.paired_episode_cluster_bootstrap(
                af[name].numpy(), bf[name].numpy(), eid,
                n_boot=a.n_boot, seed=a.seed)
            rows[name] = out
            print("%-20s %11.6f %11.6f %11.6f  %s" % (
                name, out["delta"], out["lo"], out["hi"],
                "yes" if out["separated"] else "no"))
        panel["base"] = {k: float(v.mean()) for k, v in bf.items()}
        panel["paired"] = rows
        kmax = max(arm.floor)
        panel["verdict"] = FF.floor_verdict(arm, base, k=kmax)
        print("")
        print("-- SHAPE verdict @%d : %s" % (kmax, panel["verdict"]["verdict"]))
        print("   (point deltas only; one seed is necessary-not-sufficient)")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(panel, fh, indent=1)
        print("")
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
