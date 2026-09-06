#!/usr/bin/env python3
"""THE HONEST BASELINE: the output-scored speed constraint, with the flag OFF.

⛔ WHAT IS BEING SCORED, AND WHY IT IS LEGITIMATE. No model runs, so the
trajectory scored here is the EGO'S OWN REALISED path over the 2-6 s tactical
band -- the human driver. That is exactly the right baseline for an
output-scored constraint: it says what the metric reads on a trajectory that was
never given the ceiling, which is the number a flag-OFF arm must be compared
against. ``an input that does not exist cannot echo``.

⛔⛔ AND THE CONTROL THAT MAKES THE NUMBER READABLE. The SAME instrument is run
against the RAW ``v_hi_ms`` ceiling. That must read frac_over = 0.000000 on every
clip BY CONSTRUCTION -- the raw value IS the max of the speed being scored. If it
did not read exactly zero the join would be broken; if the quantized number is
quoted without it, a reader cannot tell a real safety result from an arithmetic
identity.

TIER: none. No model runs; a T0/T1 stamp would be a category error.

ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, "C:/Users/Admin/tanitad-wt/taniteval")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/taniteval")

from tanitad.data import egomotion_source as ES        # noqa: E402
from tanitad.eval import constraints as C              # noqa: E402
from tanitad.eval import speed_limit_scoring as SLS    # noqa: E402
from taniteval import ci as CI                         # noqa: E402

HZ = 10.0
BAND = (2.0, 6.0)


def run(labels_gz: Path, join_json: Path, out_json: Path) -> dict:
    join = json.load(open(join_json, encoding="utf-8"))
    assert join["units"] == "m_s"
    lab = {}
    for line in gzip.open(labels_gz, "rt", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            lab[r["clip_id"]] = r
    # ⛔⛔ THREE CEILING CONFIGURATIONS, AND THE FIRST RUN OF THIS SCRIPT IS WHY.
    # I predicted the RAW control would read frac_over = 0.000000 exactly. It read
    # 0.018207 on 2,262 clips -- because I had combined the posted ceiling with
    # the KINEMATIC 0.35 g comfort ceiling, and THAT one binds. The identity
    # ("the raw ceiling is the max of the speed being scored") holds only for the
    # POSTED-LIMIT CEILING ALONE. So both are reported: `posted_only` carries the
    # identity and is the instrument check; `combined` is the realistic envelope
    # and its raw arm is NOT vacuous. Quoting the combined raw number as "the
    # vacuity control" would have been true-but-wrong-for-the-reader.
    clips, over_q, over_r, under_q, man, shot = [], [], [], [], [], []
    over_q_only, over_r_only = [], []
    n_allows_q = n_scored = 0
    for jr in join["rows"]:
        cid = jr["clip_id"]
        r = lab.get(cid)
        if r is None:
            continue
        tr = ES.load(cid)
        p, key = tr.poses, tr.key_index
        lo = key + int(round(BAND[0] * HZ))
        hi = key + int(round(BAND[1] * HZ))
        if hi >= len(p):
            continue
        v = p[lo:hi + 1, 3]
        kap = tr.curvature[lo:hi + 1]
        # ⛔ `kinematic_speed_ceiling` returns (v_max, valid) -- a TUPLE. Binding
        # it to one name and calling np.isfinite on it silently builds a [2, K]
        # mask that indexes a 1-D speed array; caught here by an IndexError,
        # which is the good case. Unpack, always.
        kin, kin_ok = C.kinematic_speed_ceiling(kap)
        mf = ((r.get("a_tac") or {}).get("lat") or "LANE_KEEP") != "LANE_KEEP"
        s = SLS.score_against_posted_limit(v, jr["v_hi_ms"], manoeuvre_flag=[mf],
                                           kinematic=kin, kin_valid=kin_ok)
        s_only = SLS.score_against_posted_limit(v, jr["v_hi_ms"],
                                                manoeuvre_flag=[mf])
        q, raw = s["quantized"], s["vacuity_control"]["raw"]
        if q["status"] != "OK" or s_only["quantized"]["status"] != "OK":
            continue
        over_q_only.append(s_only["quantized"]["frac_over_ceiling"])
        over_r_only.append(s_only["vacuity_control"]["raw"]["frac_over_ceiling"])
        clips.append(cid)
        over_q.append(q["frac_over_ceiling"])
        over_r.append(raw["frac_over_ceiling"])
        fu = q["frac_under_when_allowed"]
        under_q.append(np.nan if fu is None else fu)
        n_allows_q += q["n_situation_allows"]
        n_scored += q["n_scored"]
        man.append(1.0 if mf else 0.0)
        shot.append(q["mean_overshoot_ms"])

    eid = np.array(clips)
    oq = np.array(over_q, float)
    orr = np.array(over_r, float)
    uq = np.array(under_q, float)
    fin = np.isfinite(uq)

    b_q = CI.episode_cluster_bootstrap(oq, eid, reduce="mean", dp=6)
    b_r = CI.episode_cluster_bootstrap(orr, eid, reduce="mean", dp=6)
    pb = CI.paired_episode_cluster_bootstrap(oq, orr, eid)
    b_u = (CI.episode_cluster_bootstrap(uq[fin], eid[fin], reduce="mean", dp=6)
           if fin.any() else None)

    doc = {
        "control_units": "m_s",
        "tier": ("NONE -- no model runs. The scored trajectory is the EGO'S OWN "
                 "realised path over 2-6 s; a T0/T1 stamp would be a category "
                 "error."),
        "what_was_scored": ("ego realised speed over the 2-6 s tactical band vs "
                            "min(posted-limit ceiling, kinematic 0.35 g ceiling)"),
        "n_clips": len(clips), "n_steps_scored": int(n_scored),
        "estimator": "episode_cluster_bootstrap / paired (taniteval.ci), "
                     "cluster = clip",
        "variance_question": ("sampling variance over CLIPS from this corpus -- "
                              "NOT seed variance, NOT split variance"),
        "QUANTIZED_ceiling": {
            "frac_over_ceiling_mean": b_q["mean"], "ci": [b_q["lo"], b_q["hi"]],
            "n_clips_with_any_overshoot": int((oq > 0).sum()),
            "frac_clips_with_any_overshoot": float((oq > 0).mean()),
            "mean_overshoot_ms_over_clips": float(np.mean(shot)),
            "frac_under_when_allowed_mean": (b_u["mean"] if b_u else None),
            "frac_under_ci": ([b_u["lo"], b_u["hi"]] if b_u else None),
            "n_clips_with_an_allowing_step": int(fin.sum()),
            "n_situation_allows_steps": int(n_allows_q),
        },
        "COMBINED_raw_ceiling": {
            "frac_over_ceiling_mean": b_r["mean"], "ci": [b_r["lo"], b_r["hi"]],
            "n_clips_with_any_overshoot": int((orr > 0).sum()),
            "reading": ("NOT a vacuity control: this ceiling is "
                        "min(raw posted, kinematic 0.35 g), and the kinematic "
                        "half binds. The vacuity identity lives in "
                        "POSTED_ONLY_vacuity_identity below."),
        },
        "POSTED_ONLY_vacuity_identity": {
            "quantized_frac_over_mean": float(np.mean(over_q_only)),
            "quantized_n_clips_with_overshoot": int((np.array(over_q_only) > 0).sum()),
            "raw_frac_over_mean": float(np.mean(over_r_only)),
            "raw_n_clips_with_overshoot": int((np.array(over_r_only) > 0).sum()),
            "raw_expected": 0.0,
            "raw_identity_holds": bool(np.all(np.array(over_r_only) == 0.0)),
            "reading": ("⛔ THE INSTRUMENT CHECK AND THE WHOLE ARGUMENT IN ONE "
                        "ROW. Against the RAW ego-derived ceiling alone, "
                        "frac_over is exactly 0.000000 on every clip BY "
                        "CONSTRUCTION -- the ceiling IS the max of the speed "
                        "being scored, so the metric can never fire and says "
                        "nothing about driving. Against the QUANTIZED posted-limit "
                        "ceiling the same instrument CAN fire. A non-zero raw "
                        "number would mean the join is broken."),
        },
        "paired_quantized_minus_raw": {
            "delta": pb["delta"], "ci": [pb["lo"], pb["hi"]],
            "separated": bool(pb["separated"]), "n_episodes": pb["n_episodes"]},
        "manoeuvre_rate": float(np.mean(man)),
        "_vacuity_gate": ("manoeuvre_rate is reported beside every satisfaction "
                          "number: a constraint-satisfaction zero bought by "
                          "declining the manoeuvre is not a safety result."),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


if __name__ == "__main__":
    d = run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print("n_clips %d   n_steps %d   manoeuvre_rate %.4f"
          % (d["n_clips"], d["n_steps_scored"], d["manoeuvre_rate"]))
    q, r = d["QUANTIZED_ceiling"], d["COMBINED_raw_ceiling"]
    o = d["POSTED_ONLY_vacuity_identity"]
    print("-- POSTED LIMIT ALONE (the instrument check) --")
    print("  quantized frac_over %.6f   clips with overshoot %d"
          % (o["quantized_frac_over_mean"], o["quantized_n_clips_with_overshoot"]))
    print("  RAW       frac_over %.6f   clips with overshoot %d   identity holds: %s"
          % (o["raw_frac_over_mean"], o["raw_n_clips_with_overshoot"],
             o["raw_identity_holds"]))
    print("-- COMBINED min(posted, kinematic 0.35 g) --")
    print("QUANTIZED  frac_over  %.6f [%.6f, %.6f]   clips with any overshoot "
          "%d (%.2f%%)" % (q["frac_over_ceiling_mean"], q["ci"][0], q["ci"][1],
                           q["n_clips_with_any_overshoot"],
                           100 * q["frac_clips_with_any_overshoot"]))
    print("           frac_under_when_allowed %s  (n clips with an allowing step "
          "%d, steps %d)" % (q["frac_under_when_allowed_mean"],
                             q["n_clips_with_an_allowing_step"],
                             q["n_situation_allows_steps"]))
    print("RAW        frac_over  %.6f [%.6f, %.6f]   clips with any overshoot %d"
          % (r["frac_over_ceiling_mean"], r["ci"][0], r["ci"][1],
             r["n_clips_with_any_overshoot"]))
    p = d["paired_quantized_minus_raw"]
    print("paired q-raw  %+.6f [%+.6f, %+.6f]  %s"
          % (p["delta"], p["ci"][0], p["ci"][1],
             "SEPARATED" if p["separated"] else "not separated"))
