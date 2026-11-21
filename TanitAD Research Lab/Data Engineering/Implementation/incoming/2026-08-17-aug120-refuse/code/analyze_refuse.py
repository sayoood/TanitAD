"""Verify the aug120 re-fuse BY CONTENT (C77), and measure the label deltas.

⛔ A RUN IS NOT COMPLETE BECAUSE THE FILE COUNT MATCHES. C77 was banked by an
agent that reported "the main exec has completed" over a corpus holding zero
detections. Completion here is evaluated on what the records CARRY: the
perception census state, the per-concept totals, the error census, the engine
stamp, and — the one the brief names — the count still carrying
`perception.absent`.

Deltas are decomposed A0->A1 (fuser code) -> A2 (v2 corpus) -> A3 (strategic
leg) so no single number has to carry three causes. Intervals are the
episode-cluster bootstrap from `taniteval.ci`; `overlapping_holdout_se` is
never used.

⚠️ ONE CLIP = ONE EPISODE HERE, so the cluster bootstrap coincides with a
clip-level bootstrap. That is stated rather than left for a reader to notice:
the clustering is a no-op BY CONSTRUCTION (n_windows == n_episodes == 201),
not by accident, and it is still the right estimator to name.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

FAMILIES = {
    # new field name           -> (A0 legacy field name, human label)
    "g_str": ("g_str", "STRATEGIC goal (g_str)"),
    "a_str": (None, "STRATEGIC action (a_str)"),
    # ⚠️ v1 emitted the ACTION vocabulary under the GOAL field's name; the
    # rename is the fix, so the legacy counterpart of a_tac_* IS g_tac_*.
    "a_tac_lat": ("g_tac_lat", "TACTICAL lateral (a_tac_lat)"),
    "a_tac_lon": ("g_tac_lon", "TACTICAL longitudinal (a_tac_lon)"),
}


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(p).startswith("_"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        out[r["clip_id"]] = r
    return out


def tok(rec, field, legacy=False):
    key = field
    if legacy:
        return ((rec.get("vocab") or {}).get(key) or {}).get("token")
    return ((rec.get("vocab") or {}).get(key) or {}).get("token")


def content_verify(recs, name):
    """The C77 check: what do the records CARRY?"""
    per_concept = collections.Counter()
    per_scene = collections.Counter()
    errs = collections.Counter()
    census_states = collections.Counter()
    engines = collections.Counter()
    track_frames = collections.Counter()
    n_absent = n_scene = n_tracks = 0
    schema_bad = adm_bad = prov_lie = 0
    goal_ev = collections.Counter()
    census_scene = collections.Counter()
    n_sign_present = 0
    for cid, r in recs.items():
        p = r.get("perception") or {}
        n_absent += int("absent" in p)
        n_tracks += len(p.get("tracks") or [])
        for t in (p.get("tracks") or []):
            per_concept[t["concept"]] += 1
            track_frames[t.get("n_frames", 0)] += 1
        cen = p.get("census")
        census_states[(cen or {}).get("state", "MISSING_KEY")] += 1
        eng = p.get("engine") or {}
        engines[(eng.get("schema_version"),
                 eng.get("confidence_threshold"))] += 1
        sc = p.get("scene")
        if sc:
            n_scene += 1
            per_scene.update({k: int(v) for k, v in
                              (sc.get("per_scene_hits") or {}).items()})
            errs["scene_err"] += int(sc.get("n_err_scene") or 0)
        schema_bad += int(r.get("schema_version") != "ph1-fused-v1")
        adm = r.get("inference_admissible") or []
        # labels may use ego; INFERENCE IS VISION-ONLY
        adm_bad += int("ego" in adm or "alpamayo" in adm)
        vp = (r.get("_provenance") or {}).get("vlm")
        mode = "past"          # MEASURED 201/201 on this cohort
        prov_lie += int(vp == "vision")
        ge = ((r.get("corroboration") or {}).get("goal_evidence") or {})
        if ge:
            goal_ev[ge.get("verdict")] += 1
            n_sign_present += int(bool(ge.get("sign_like_object_present")))
        cs = ((r.get("corroboration") or {}).get("census_vs_scene") or {})
        if cs:
            census_scene[cs.get("verdict")] += 1
    return {
        "arm": name, "n_records": len(recs),
        "PERCEPTION_ABSENT": n_absent,
        "census_states": dict(census_states),
        "n_tracks_total": n_tracks,
        "per_concept_track_totals": dict(per_concept.most_common()),
        "track_length_hist_n_frames": dict(sorted(track_frames.items())),
        "n_with_scene_channel": n_scene,
        "per_scene_hit_totals": dict(per_scene.most_common()),
        "error_census": dict(errs) if errs else {},
        "perception_engines": [{"schema_version": k[0],
                                "confidence_threshold": k[1], "n_clips": v}
                               for k, v in sorted(engines.items(),
                                                  key=lambda kv: str(kv[0]))],
        "perception_engine_mixed": len(engines) > 1,
        "schema_wrong": schema_bad,
        "inference_whitelist_violations": adm_bad,
        "provenance_says_vision_while_ego_prompted": prov_lie,
        "goal_evidence_verdicts": dict(goal_ev),
        "n_sign_like_object_present": n_sign_present,
        "census_vs_scene_verdicts": dict(census_scene),
    }


def family_delta(a, b, field_a, field_b, clips, ci):
    """Per-token prevalence deltas + the overall change rate, both bootstrapped
    on the clip (= episode) as the resampling unit."""
    ta = [tok(a[c], field_a) if field_a else None for c in clips]
    tb = [tok(b[c], field_b) for c in clips]
    changed = [float(x != y) for x, y in zip(ta, tb)]
    eid = list(clips)
    rate = ci.episode_cluster_bootstrap(changed, eid)
    toks = sorted({t for t in ta + tb if t is not None},
                  key=lambda s: str(s))
    rows = []
    for t in toks + [None]:
        ia = [float(x == t) for x in ta]
        ib = [float(x == t) for x in tb]
        na, nb = int(sum(ia)), int(sum(ib))
        if na == 0 and nb == 0:
            continue
        d = ci.paired_episode_cluster_bootstrap(ib, ia, eid)
        rows.append({"token": t if t is not None else "(null/abstain)",
                     "n_before": na, "n_after": nb,
                     "delta_n": nb - na,
                     "delta_frac": d["delta"], "lo": d["lo"], "hi": d["hi"],
                     "separated": d["separated"],
                     "estimator": d["estimator"]})
    conf = collections.Counter()
    for x, y in zip(ta, tb):
        if x != y:
            conf[(str(x), str(y))] += 1
    return {"n_clips": len(clips),
            "changed_n": int(sum(changed)),
            "changed_rate": rate["mean"], "lo": rate["lo"], "hi": rate["hi"],
            "estimator": rate["estimator"],
            "n_episodes": rate["n_episodes"], "n_windows": rate["n_windows"],
            "per_token": rows,
            "top_transitions": [{"from": k[0], "to": k[1], "n": v}
                                for k, v in conf.most_common(10)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--a3", required=True)
    ap.add_argument("--taniteval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.taniteval)
    from taniteval import ci

    arms = {k: load(getattr(a, k)) for k in ("a0", "a1", "a2", "a3")}
    clips = sorted(arms["a0"])
    for k, v in arms.items():
        assert sorted(v) == clips, f"{k} clip set differs from a0"

    out = {"class": "MEASURED", "n_clips": len(clips),
           "content_verification": {k.upper(): content_verify(v, k.upper())
                                    for k, v in arms.items()},
           "deltas": {}}

    steps = [("A0->A1_fuser_code", "a0", "a1", True),
             ("A1->A2_v2_corpus", "a1", "a2", False),
             ("A2->A3_strategic_leg", "a2", "a3", False),
             ("A0->A3_TOTAL", "a0", "a3", True)]
    for label, x, y, legacy in steps:
        d = {}
        for new_f, (old_f, human) in FAMILIES.items():
            fa = (old_f if legacy else new_f)
            if fa is None:
                d[new_f] = {"note": "field did not exist in A0 — no paired "
                                    "before; see the A1 census instead",
                            "human": human}
                continue
            d[new_f] = {"human": human,
                        **family_delta(arms[x], arms[y], fa, new_f, clips, ci)}
        out["deltas"][label] = d

    # ---- the prose / census defects, counted rather than described --------- #
    def phrase_census(recs, needle):
        return sum(1 for r in recs.values()
                   if needle in (r.get("scenario_description") or ""))
    out["prose"] = {
        arm.upper(): {
            "records_saying_'no agents'": phrase_census(v, "no agents"),
            "records_saying_'UNAVAILABLE'": phrase_census(v, "UNAVAILABLE"),
            "records_saying_'0 agents detected'":
                phrase_census(v, "0 agents detected"),
            "records_saying_'-lane-ego-carriageway'":
                phrase_census(v, "-lane-ego-carriageway"),
        } for arm, v in arms.items()}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out["content_verification"][k] for k in
                      ("A0", "A3")}, indent=1)[:4000])
    print("ANALYZE_DONE ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
