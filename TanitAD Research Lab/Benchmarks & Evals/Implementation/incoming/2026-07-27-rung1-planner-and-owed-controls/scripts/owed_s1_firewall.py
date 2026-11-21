#!/usr/bin/env python3
"""OWED ROWS 23-25 — the `S1-BLINDvsMAJORITY` firewall, re-adjudicated. ZERO GPU.

These three rows could not be re-POWERED by the predecessor and cannot be
re-powered here either: n is bounded by **20 AlpaSim scenes**, not by episodes,
and buying more is a download decision. What CAN be done at the same n, and was
not done, is all of the following — each of which can return a FAILING verdict:

  1. REPRODUCE the committed `S1_RESULTS.json` from its own driver, so that
     anything below is known to be a property of the data and not of a re-run.
  2. ⭐ INTERVAL THE HEADLINE STATISTIC. `s1_slice.py:266-281` publishes
     `acc_blind` as the MAXIMUM of two blind attacks (a learned option scorer and
     a deterministic argmin-distance attack) but computes
     `blind_vs_majority_paired` from `run_firewall`'s `correct_blind`, which is
     the LEARNED attack ONLY. On variant H the headline blind is 0.5000 and the
     intervalled one is 0.2500. **The published interval does not interval its own
     published point estimate.**
  3. ⭐ THE LEAK-RELEVANT CONTRAST. The firewall tests `blind` vs **majority**.
     A circularity firewall's question is `blind` vs **chance**. That contrast has
     a published point estimate (+0.2820 E / +0.1500 NOGOAL) and **has never been
     given an interval**. It is computed here, paired, at the scene level.
  4. M8 — `mde`, `max_possible_effect` and `can_fire` for every contrast, using
     the bound the artifact itself states (`acc_ceiling = 1.0`).

⛔ NO FIREWALL IS RE-IMPLEMENTED. `run_firewall` is imported from the stream's own
`blind_conditioning_baseline.py` (the instrument that produced the committed
numbers — importing it is reproduction, not duplication) and the PACKAGED
`taniteval.blind_baseline.blind_conditioning_baseline` is run as an INDEPENDENT
second instrument on the same data. `taniteval.ci` supplies every interval.

Validation runs in BOTH directions (`--selftest`), and the failing direction must
fire or the script exits non-zero.

Usage:
    python owed_s1_firewall.py --gates <4brain-gates dir> --out <raw dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval", _REPO / "stack"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from taniteval import ci as _ci                              # noqa: E402
from taniteval import blind_baseline as _bb                  # noqa: E402

B_BOOT, SEED = 2000, 0
VARIANTS = ("E", "H", "NOGOAL")


# --------------------------------------------------------------------------- #
# M8 — the power ceiling. This is a PROOF for a bounded metric, not a forecast. #
# --------------------------------------------------------------------------- #
def power_ceiling(interval: dict, max_possible: float) -> dict:
    """`mde` = the row's own 95 % half-width; an effect must EXCEED it to separate.

    For a BOUNDED metric the largest effect that can physically exist is known,
    so `mde >= max_possible` is a proof that the test cannot fire at ANY observed
    value — C13's "a guard that cannot fail is not a guard", pointed at the
    negative-control side.
    """
    mde = float(interval["ci95"]) if "ci95" in interval else \
        float(interval["hi"] - interval["lo"]) / 2.0
    return {
        "mde": round(mde, 6),
        "max_possible_effect": round(float(max_possible), 6),
        "mde_as_pct_of_max_possible": (None if max_possible <= 0 else
                                       round(100.0 * mde / max_possible, 1)),
        "can_fire": bool(max_possible > mde),
        "_read": ("this control CANNOT separate at any observed value -- its "
                  "verdict is VOID, not a pass"
                  if max_possible <= mde else
                  "this control can separate, if the effect exceeds its MDE"),
    }


def paired(a, b, eid, label):
    r = _ci.paired_episode_cluster_bootstrap(np.asarray(a, float),
                                             np.asarray(b, float),
                                             [str(e) for e in eid],
                                             n_boot=B_BOOT, seed=SEED)
    r["contrast"] = label
    r["ci95"] = round(float((r["hi"] - r["lo"]) / 2.0), 6)
    r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
    return r


# --------------------------------------------------------------------------- #
# Rebuild the per-decision-point vectors the committed artifact did not keep    #
# --------------------------------------------------------------------------- #
def rebuild(gates_dir: Path):
    """Re-run the stream's OWN firewall driver pieces to recover, per variant:
    `correct_blind_learned`, `correct_blind_headline`, `correct_major`,
    `chance_i` (= 1/K_i, the EXACT expectation of a uniform random choice over
    that decision point's own option set) and the scene ids."""
    sys.path.insert(0, str(gates_dir))
    import s1_slice as S                                       # noqa: E402
    from blind_conditioning_baseline import run_firewall       # noqa: E402

    dps = json.load(open(gates_dir / "s1_decision_points.json", encoding="utf-8"))
    res = [d for d in dps if d.get("target_branch") is not None]
    out = {}
    for variant in VARIANTS:
        groups, clusters, kept = [], [], []
        for d in res:
            f = S.build_features(d, variant)
            if f is None:
                continue
            groups.append((f, int(d["target_branch"])))
            clusters.append(d["scene_id"])
            kept.append(d)
        if not groups:
            out[variant] = None
            continue
        r = run_firewall(groups, clusters, variant)
        # --- the deterministic attack, EXACTLY as s1_slice.py:247-265 ---------- #
        det_correct, det_rows = [], []
        if variant in ("E", "H"):
            for i, d in enumerate(kept):
                cls_ = d.get("option_centerlines_egoframe") or []
                if len(cls_) != len(d["options"]):
                    continue
                if variant == "E":
                    g = S.route_polyline(d, 30.0)
                else:
                    gp = S.goal_point(d, 150.0, 200.0)
                    g = None if gp is None else np.asarray([gp])
                if g is None:
                    continue
                dd = [1e9 if len(cl) < 2 else
                      float(np.mean(S._pt_to_polyline_dist(np.asarray(g, float), cl)))
                      for cl in cls_]
                det_rows.append(i)
                det_correct.append(int(int(np.argmin(dd)) == int(d["target_branch"])))
        out[variant] = {
            "n": len(groups),
            "scenes": clusters,
            "correct_blind_learned": list(map(int, r["correct_blind"])),
            "correct_major": list(map(int, r["correct_major"])),
            "chance_i": [1.0 / g[0].shape[0] for g in groups],
            "det_rows": det_rows,
            "det_correct": det_correct,
            "acc_blind_learned": round(float(np.mean(r["correct_blind"])), 4),
            "acc_major": round(float(np.mean(r["correct_major"])), 4),
            "acc_chance": round(float(np.mean([1.0 / g[0].shape[0] for g in groups])), 4),
            "n_clusters": r["n_clusters"],
        }
        if det_correct:
            out[variant]["acc_blind_deterministic"] = round(float(np.mean(det_correct)), 4)
    return out, res


# --------------------------------------------------------------------------- #
# ⛔ the failing direction — both instruments must FIRE on a planted leak        #
# --------------------------------------------------------------------------- #
def selftest(reb) -> dict:
    """Fidelity AND failure. An instrument validated in one direction only is not
    validated: the whole point of this task is that a control which cannot fail
    reads exactly like a control that passed."""
    t = {}
    # (1) the interval estimator: a==b must NOT separate; a==b+1 MUST separate.
    v = reb["NOGOAL"]
    a = np.asarray(v["correct_blind_learned"], float)
    same = paired(a, a, v["scenes"], "selftest_identical")
    shift = paired(a + 1.0, a, v["scenes"], "selftest_shifted_by_1")
    t["ci_identical_not_separated"] = (not same["separated"]) and abs(same["delta"]) < 1e-12
    t["ci_shifted_separated"] = bool(shift["separated"] and abs(shift["delta"] - 1.0) < 1e-9)
    # (2) the PACKAGED firewall: an echo context must be CIRCULAR, noise CLEAN.
    y = np.asarray([int(x) for x in _echo_targets(reb)])
    eids = reb["NOGOAL"]["scenes"]
    echo = _bb.blind_conditioning_baseline({"echo": y}, y, eids, problem="selftest_echo",
                                           n_boot=200)
    rng = np.random.default_rng(0)
    noise = _bb.blind_conditioning_baseline({"noise": rng.integers(0, 5, size=len(y))},
                                            y, eids, problem="selftest_noise", n_boot=200)
    t["packaged_firewall_echo_is_CIRCULAR"] = echo["verdict"] == "CIRCULAR"
    t["packaged_firewall_noise_is_not_CIRCULAR"] = noise["verdict"] != "CIRCULAR"
    # (3) the MDE emitter: must return can_fire=False on the known-void case and
    #     True on a wide-open one.
    t["mde_void_case_can_fire_false"] = (
        power_ceiling({"ci95": 0.5555}, 0.2500)["can_fire"] is False)
    t["mde_open_case_can_fire_true"] = (
        power_ceiling({"ci95": 0.0100}, 0.2500)["can_fire"] is True)
    t["ALL_PASS"] = all(bool(x) for k, x in t.items() if k != "ALL_PASS")
    t["_read"] = ("every instrument used below is validated in BOTH directions: "
                  "it reproduces a known value AND it fires on a planted failure")
    return t


def _echo_targets(reb):
    """The NOGOAL variant's own targets, recovered from its correct/major vectors
    is not possible; the echo self-test only needs SOME label vector of the right
    length with >=2 classes, so the majority-correct vector is used."""
    v = reb["NOGOAL"]
    return v["correct_major"]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gates = Path(a.gates).resolve()
    outd = Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    committed = json.load(open(gates / "S1_RESULTS.json", encoding="utf-8"))
    reb, res = rebuild(gates)

    st = selftest(reb)
    print("[selftest]", json.dumps({k: v for k, v in st.items() if k != "_read"}), flush=True)
    if not st["ALL_PASS"]:
        print("SELF-TEST FAILED — nothing below is admissible", flush=True)
        (outd / "owed_s1_firewall.json").write_text(
            json.dumps({"selftest": st, "ABORTED": True}, indent=2), encoding="utf-8")
        return 3

    out = {
        "block": "owed_controls/S1-BLINDvsMAJORITY",
        "frozen_list_rows": [23, 24, 25],
        "estimator": {"interval": "paired_episode_cluster_bootstrap",
                      "module": "taniteval.ci", "n_boot": B_BOOT, "seed": SEED,
                      "resampling_unit": "AlpaSim scene (episode cluster)"},
        "instruments": {
            "reproduction": "2026-07-26-4brain-gates/s1_slice.py + "
                            "blind_conditioning_baseline.run_firewall (IMPORTED, not copied)",
            "independent_second": "taniteval.blind_baseline.blind_conditioning_baseline (PACKAGED)",
            "intervals": "taniteval.ci (PACKAGED)"},
        "selftest": st,
        "n_scenes_available": committed["coverage"]["n_scenes_with_dp"],
        "power_bar_single_arm_clusters": committed["power"]["bar_single_arm"],
        "variants": {},
    }

    fid = {}
    for v in VARIANTS:
        c = committed["firewall"][v]
        r = reb[v]
        fid[v] = {
            "acc_blind_learned": {"committed": c.get("acc_blind_learned_mlp", c["acc_blind"]),
                                  "recomputed": r["acc_blind_learned"]},
            "acc_major": {"committed": c["acc_major"], "recomputed": r["acc_major"]},
            "acc_chance": {"committed": c["acc_chance"], "recomputed": r["acc_chance"]},
            "n": {"committed": c["n"], "recomputed": r["n"]},
            "n_clusters": {"committed": c["n_clusters"], "recomputed": r["n_clusters"]},
        }
        if "acc_blind_deterministic" in r:
            fid[v]["acc_blind_deterministic"] = {
                "committed": c.get("acc_blind_deterministic_argmin_dist"),
                "recomputed": r["acc_blind_deterministic"]}
    fid["REPRODUCES"] = all(
        abs(float(d["committed"]) - float(d["recomputed"])) < 5e-4
        for vv in VARIANTS for k, d in fid[vv].items()
        if d["committed"] is not None)
    out["fidelity_vs_committed"] = fid
    print(f"[fidelity] REPRODUCES = {fid['REPRODUCES']}", flush=True)

    for v in VARIANTS:
        c = committed["firewall"][v]
        r = reb[v]
        sc = r["scenes"]
        blind_l = np.asarray(r["correct_blind_learned"], float)
        maj = np.asarray(r["correct_major"], float)
        chance = np.asarray(r["chance_i"], float)
        ceiling = float(c["acc_ceiling"])

        # ---- headline blind: the STRONGEST attack, per the driver's own rule --- #
        headline_src = "learned option scorer"
        blind_h = blind_l.copy()
        if "det_correct" in r and r["det_correct"] and \
                r.get("acc_blind_deterministic", -1) > r["acc_blind_learned"]:
            headline_src = "deterministic argmin distance(goal, option centreline)"
            blind_h = blind_l.copy()
            for i, row in enumerate(r["det_rows"]):
                blind_h[row] = float(r["det_correct"][i])

        acc_bl = float(blind_l.mean())
        acc_bh = float(blind_h.mean())
        acc_mj = float(maj.mean())
        acc_ch = float(chance.mean())

        b_vs_m_learned = paired(blind_l, maj, sc, "blind_LEARNED - majority")
        b_vs_m_headline = paired(blind_h, maj, sc, "blind_HEADLINE - majority")
        b_vs_c_learned = paired(blind_l, chance, sc, "blind_LEARNED - chance")
        b_vs_c_headline = paired(blind_h, chance, sc, "blind_HEADLINE - chance")

        # ---- M8 power ceilings ------------------------------------------------ #
        pc = {
            "blind_minus_majority": power_ceiling(b_vs_m_headline, ceiling - acc_mj),
            "blind_minus_chance": power_ceiling(b_vs_c_headline, ceiling - acc_ch),
            "committed_row_as_published": power_ceiling(
                {"ci95": c["blind_vs_majority_paired"]["ci95"]}, ceiling - acc_mj),
        }

        # ---- the packaged instrument, independently ---------------------------- #
        pack = _packaged(gates, v)

        # ---- the verdict, per PRE_REGISTRATION §A.3 ---------------------------- #
        mismatch = abs(acc_bh - acc_bl) > 1e-9
        if b_vs_c_headline["separated"] and b_vs_c_headline["delta"] > 0:
            verdict = "CONTROL FAILS"
            why = ("the blind head is separated-ABOVE chance: the conditioning "
                   "carries real target information, which is exactly what a "
                   "circularity firewall exists to surface")
        elif not pc["committed_row_as_published"]["can_fire"]:
            verdict = "VOID"
            why = ("MDE exceeds the largest effect that can physically exist -- "
                   "the published verdict is zero evidence at any observed value")
        elif mismatch:
            verdict = "VOID-BY-MISMATCH"
            why = ("the published interval is computed on the LEARNED attack "
                   "while the published point estimate is the STRONGER "
                   "deterministic attack -- the interval does not interval its "
                   "own headline")
        else:
            verdict = "UNDER-POWERED (OWED)"
            why = ("not separated, and the MDE is larger than the leak sizes this "
                   "firewall exists to catch")

        out["variants"][v] = {
            "name": c["name"],
            "n_decision_points": r["n"], "n_clusters": r["n_clusters"],
            "acc_blind_headline": round(acc_bh, 4),
            "acc_blind_headline_source": headline_src,
            "acc_blind_learned": round(acc_bl, 4),
            "acc_majority": round(acc_mj, 4),
            "acc_chance": round(acc_ch, 4),
            "committed_blind_vs_majority_paired": c["blind_vs_majority_paired"],
            "recomputed": {
                "blind_LEARNED_minus_majority": b_vs_m_learned,
                "blind_HEADLINE_minus_majority": b_vs_m_headline,
                "blind_LEARNED_minus_chance": b_vs_c_learned,
                "blind_HEADLINE_minus_chance": b_vs_c_headline,
            },
            "power_ceiling": pc,
            "headline_interval_statistic_mismatch": bool(mismatch),
            "mismatch_size": round(acc_bh - acc_bl, 4),
            "packaged_instrument": pack,
            "VERDICT": verdict,
            "why": why,
        }
        print(f"[{v}] blind_h={acc_bh:.4f} learned={acc_bl:.4f} maj={acc_mj:.4f} "
              f"chance={acc_ch:.4f} | b-c {b_vs_c_headline['delta']:+.4f} "
              f"[{b_vs_c_headline['lo']:+.4f},{b_vs_c_headline['hi']:+.4f}] "
              f"sep={b_vs_c_headline['separated']} | can_fire="
              f"{pc['committed_row_as_published']['can_fire']} -> {verdict}", flush=True)

    (outd / "owed_s1_firewall.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[write] {outd / 'owed_s1_firewall.json'}", flush=True)
    return 0


def _packaged(gates: Path, variant: str) -> dict:
    """The PACKAGED firewall on the SAME data, as an independent second instrument.

    ⚠️ It is NOT a drop-in: `taniteval.blind_baseline` classifies over a FIXED
    class set from a per-item context matrix, while S1 is a VARIABLE-ARITY option
    set. The honest encoding is to pad the per-option geometry to the maximum
    arity and let the packaged head predict the branch index. This is a WEAKER
    attack than the option scorer by construction (it cannot share weights across
    options), so it is reported as a LOWER BOUND on the blind signal and never as
    the headline. Reporting it is the point: R-E asked whether the packaged module
    could have replaced the copy, and the answer is a measurement, not an opinion.
    """
    sys.path.insert(0, str(gates))
    import s1_slice as S                                       # noqa: E402
    dps = json.load(open(gates / "s1_decision_points.json", encoding="utf-8"))
    res = [d for d in dps if d.get("target_branch") is not None]
    rows, y, eid = [], [], []
    for d in res:
        f = S.build_features(d, variant)
        if f is None:
            continue
        flat = np.zeros(3 * f.shape[1])
        flat[:min(3, f.shape[0]) * f.shape[1]] = f[:3].reshape(-1)
        rows.append(flat)
        y.append(int(d["target_branch"]))
        eid.append(d["scene_id"])
    if len(rows) < 4 or len(set(eid)) < 2:
        return {"ran": False, "why": "too few decision points"}
    X = np.asarray(rows)
    ctx = {f"f{i}": X[:, i] for i in range(X.shape[1])}
    r = _bb.blind_conditioning_baseline(ctx, np.asarray(y), eid,
                                        problem=f"S1_{variant}", n_boot=B_BOOT)
    return {"ran": True, "verdict": r["verdict"],
            "blind_accuracy": r["blind_accuracy"]["mean"] if "mean" in r["blind_accuracy"]
            else r["blind_accuracy"],
            "blind_accuracy_linear_probe": r["blind_accuracy_linear_probe"],
            "majority_base_rate": r["majority_base_rate"],
            "blind_skill_over_majority": r["blind_skill_over_majority"],
            "context_leaks": r["context_leaks"],
            "note": ("PADDED fixed-arity encoding -- a WEAKER attack than the "
                     "variable-arity option scorer, so a LOWER BOUND")}


if __name__ == "__main__":
    raise SystemExit(main())
