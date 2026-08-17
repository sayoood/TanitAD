"""P3 — DECLARE the episode selection BEFORE any result exists.

⛔ The 2026-08-16 run took eval clips by PROVIDER ORDER and landed 13 lead-carrying
eval episodes. The episode-cluster bootstrap clusters on EPISODES, so 13 was the
real n behind every interval it published. This selects by LEAD-BEARING CONTENT.

THE DESIGN, fixed here and not revisited after seeing a number:
  stratum   clips with n_lead_frames >= 120 over the FULL 2,308-clip train join
            (MEASURED: 302 clips). A STRATUM, not the extreme tail — sampling the
            top-130 outright would pick a 130-clip subpopulation defined by an
            order statistic, and its lead density would not be a sample of
            anything.
  sample    130 clips drawn UNIFORMLY AT RANDOM from that stratum, seed 0.
  split     70 EVAL / 60 TRAIN, assigned by the same seeded permutation, so both
            sides are draws from ONE distribution — an eval set made of the
            lead-richest clips and a train set of the rest would be a covariate
            shift dressed as a split.

⚠️ STATED, not hidden: this eval set is deliberately LEAD-ENRICHED and is NOT a
random sample of the corpus. The primary metric (`lead_gap_abs_err_m`) is only
DEFINED on windows carrying a GT in-corridor lead, so the claim it supports is
"on windows that carry a lead" either way; enrichment buys power for exactly that
claim and buys nothing else. Corpus-level prevalence is NOT estimable from it.
"""
import json, sys
from pathlib import Path
import numpy as np

cen = json.loads(Path(sys.argv[1]).read_text("utf-8"))
out = Path(sys.argv[2])
MIN_LEAD, K, N_EVAL, SEED = 120, 130, 70, 0

pool = [d for d in cen["per_clip"] if d["n_lead_frames"] >= MIN_LEAD]
pool.sort(key=lambda d: d["clip_id"])              # deterministic base order
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(pool))
sel = [pool[i] for i in perm[:K]]
ev = sorted(d["clip_id"] for d in sel[:N_EVAL])
tr = sorted(d["clip_id"] for d in sel[N_EVAL:])
assert not (set(ev) & set(tr))

def agg(ids):
    d = {c["clip_id"]: c for c in sel}
    return {"n_clips": len(ids),
            "n_labelled_frames": int(sum(d[i]["n_labelled_frames"] for i in ids)),
            "n_lead_frames": int(sum(d[i]["n_lead_frames"] for i in ids)),
            "mean_lead_frac": round(float(np.mean([d[i]["lead_frac"] for i in ids])), 4),
            "mean_lead_gap_m": round(float(np.mean([d[i]["mean_lead_gap_m"] for i in ids])), 3),
            "mean_ingrid_per_frame": round(float(np.mean([d[i]["mean_ingrid_per_frame"] for i in ids])), 3)}

rec = {"_evidence_class": "MEASURED (ours; selection is a pure function of the join census)",
       "_declared_before_any_result": True,
       "stratum": {"rule": "n_lead_frames >= %d over the full train join" % MIN_LEAD,
                   "n_clips_in_stratum": len(pool),
                   "n_clips_in_corpus_join": cen["n_clips"],
                   "n_clips_with_any_lead": cen["totals"]["n_clips_with_any_lead"]},
       "sample": {"K": K, "seed": SEED, "method": "uniform without replacement from the stratum"},
       "split": {"n_eval": N_EVAL, "n_train": K - N_EVAL,
                 "method": "the SAME seeded permutation — one distribution, two halves"},
       "eval_clips": ev, "train_clips": tr,
       "eval_agg": agg(ev), "train_agg": agg(tr),
       "prior_run_comparison": {
           "2026-08-16": {"eval_clips_total": 20, "eval_clips_carrying_a_lead": 13,
                          "selection": "provider order over a 61-clip HF prefix"},
           "this_run": {"eval_clips_total": N_EVAL,
                        "eval_clips_carrying_a_lead": int(sum(
                            1 for i in ev if {c["clip_id"]: c for c in sel}[i]["n_lead_frames"] > 0)),
                        "selection": "seeded uniform sample of the >=120-lead-frame stratum"}}}
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in rec.items() if k not in ("eval_clips", "train_clips")}, indent=1))
