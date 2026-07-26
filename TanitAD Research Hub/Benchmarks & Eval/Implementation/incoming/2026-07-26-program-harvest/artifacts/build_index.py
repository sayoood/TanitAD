"""Build harvest_index.json -- machine-readable, re-runnable, diffable."""
import json, os, re, datetime

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
OUT = os.path.join(ROOT, "TanitAD Research Hub", "Benchmarks & Eval", "Implementation",
                   "incoming", "2026-07-26-program-harvest")

proj = json.load(open(os.path.join(SP, "h1_projected.json"), encoding="utf-8"))
ranked = json.load(open(os.path.join(SP, "h1_ranked.json"), encoding="utf-8"))

def slim(r):
    return {k: r.get(k) for k in
            ("file", "json_path", "effect", "lo", "hi", "half_width", "proximity",
             "prox_at600_mean", "would_flip_mean", "would_flip_conservative",
             "n_windows", "n_episodes", "estimator", "est_class", "ntier",
             "is_panel_stratum", "is_headline", "dup_files")}

n40 = [slim(r) for r in proj["n40"]]
n12 = [slim(r) for r in proj["n12"]]
dep = [slim(r) for r in ranked["deprecated"]]

# firewall / negative-control inversion set (H1.3)
FIREWALL_PAT = re.compile(r"firewall|blind_|leak|_shuf|shuffle|majority|control|guard|canary_baseline",
                          re.I)
firewall = [r for r in (n40 + n12 + [slim(x) for x in proj["nonpanel40"] + proj["nonpanel12"]])
            if FIREWALL_PAT.search(r["json_path"])]
# dedupe
seen = set(); fw = []
for r in firewall:
    k = (r["file"], r["json_path"])
    if k in seen:
        continue
    seen.add(k); fw.append(r)
fw.sort(key=lambda r: -(r["proximity"] or 0))

index = {
  "schema_version": 1,
  "generated": "2026-07-26",
  "generator": "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/"
               "2026-07-26-program-harvest/artifacts/{h1_sweep,h1_rank,h1_project,build_index}.py",
  "task": "program-wide harvest of all agent deliverables, feeding Project Steering/BOOST_PROGRAM.md",
  "commissioned_by": "Sayed, 2026-07-26",
  "compute": "CPU only, no pod, no GPU",
  "evidence_discipline": {
    "sweep_statistics": {"class": "MEASURED (ours, this generator)", "tier": "PROVISIONAL",
                         "upgrade_path": "a second agent re-runs artifacts/h1_sweep.py and "
                                         "reproduces separated_false_nodes=2051"},
    "artifact_claims": {"class": "INHERITED", "tier": "PROVISIONAL",
                        "note": "reading a report does not upgrade its tier"},
    "excluded": "overlapping_holdout_se / _jack nodes are listed under h1.deprecated_estimator "
                "and are NOT results -- they bias point estimates -6.67%..+11.69%, up to x-4.15 "
                "with sign flips"
  },
  "confidentiality": "no PhysicalAI clip UUIDs or raw content in this file",
  "corpus": {
    "incoming_dirs": 135,
    "json_files_parsed": 642,
    "md_files": 222, "md_bytes": 3188631, "py_files": 373
  },
  "h1_unpowered_nulls": {
    "headline": "any verdict resting on a 40-episode 'not separated' is UNPOWERED, not refuted "
                "(MODEL_REGISTRY.md 1.2a, MEASURED)",
    "shrinkage_measured": {"lo": 2.8, "mean": 3.4, "hi": 3.9,
                           "source": "MODEL_REGISTRY.md 1.2a, 8 open-loop metrics, "
                                     "40->600 episodes; sqrt(15) predicts 3.87"},
    "known_flip": {"metric": "along_track_vs_cv (vs_floor_paired.cv.long_abs_2s_m)",
                   "at40": [-0.0278, 0.5304], "at600": [0.1926, 0.3104],
                   "point_estimate_movement_pct": 0.7,
                   "proximity_in_this_sweep": 0.911,
                   "note": "CALIBRATION: the one known flip ranks in the top ~5% of the n~40 "
                           "population, so the ranking statistic locates known positives"},
    "counts": {
      "separated_false_nodes": 2051,
      "with_effect_and_interval": 1785,
      "after_dedupe": 1502,
      "valid_estimator": 1480,
      "deprecated_jack": 21,
      "n40_bucket": len(n40),
      "n40_would_separate_at_mean_shrinkage": sum(1 for r in n40 if r["would_flip_mean"]),
      "n40_would_separate_at_weakest_shrinkage": sum(1 for r in n40 if r["would_flip_conservative"]),
      "n12_bucket": len(n12)
    },
    "ranking_statistic": {
      "name": "proximity", "definition": "abs(effect) / half_width",
      "caveat": "SCREENING statistic, not a power calculation. Ranks closeness to separation; "
                "does NOT predict a flip, because the point estimate can move."
    },
    "n40_nulls": n40,
    "n12_nulls": n12,
    "deprecated_estimator": dep,
    "firewall_inversion": {
      "warning": "INVERTED VALUE. For firewall / negative-control / leakage / shuffle checks a NULL "
                 "is the DESIRED verdict. 'not separated at n=40' is not a refuted leak -- it is a "
                 "leak we could not see. Re-adjudicate these FIRST: they can only REMOVE results.",
      "rows": fw
    }
  },
  "h2_unused_capabilities": {"file": "h2_unused_capabilities.json",
                             "md": "H2_UNUSED_CAPABILITIES.md",
                             "owner": "harvest subagent H2"},
  "h3_stranded_integrations": {"file": "h3_stranded.json", "md": "H3_STRANDED_INTEGRATIONS.md",
                               "owner": "harvest subagent H3"},
  "h4_contradictions": {"file": "h4_contradictions.json", "md": "H4_CONTRADICTIONS.md",
                        "owner": "harvest subagent H4"},
  "h5_unconnected_levers": [
    {"rank": 1, "finding": "Argoverse 2 is credential-free and its lane graph is byte-verified "
                           "(successors/predecessors/neighbours/is_intersection on 7692 segments "
                           "across 85 maps, 100% field presence; anonymous s3 GET HTTP 200)",
     "source": "TanitAD Research Hub/Data Engineering/Implementation/incoming/"
               "2026-07-26-credential-free-lanegraph/LANEGRAPH_ALTERNATIVES.md",
     "answers": "Q-TOPO: the strategic-brain topology must come from AlpaSim or an external corpus",
     "answering_stream": "Data Engineering / lane-graph alternatives",
     "blocked_stream": "Architecture & Inference / 4-brain dominance (S1,S2,S4,HP-4)",
     "cost_to_connect": "one ~40-line ingest driver mirroring stack/scripts/ingest_nuscenes.py, "
                        "plus the ~147 MiB pull already on the PI decision list (BOOST 5.5)",
     "verified_by_me": ["git ls-files -> stack/tanitad/data/argoverse2.py PRESENT",
                        "git ls-files -> stack/tests/test_argoverse2.py PRESENT",
                        "test -f stack/scripts/ingest_argoverse2.py -> ABSENT",
                        "grep -> no production importer; only lake/schema.py SOURCE_REGISTRY + tests"],
     "class": "MEASURED (mine)", "tier": "CONFIRMED"},
    {"rank": 2, "finding": "AlpaSim scenes embed trajdata.VectorMap (130-472 lane polygons, "
                           "130-393 road edges, wait-lines per scene) and trafficsim (SMART/CAT-K, "
                           "Apache-2.0) is in-tree on the pod and has NEVER been enabled",
     "source": "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/"
               "2026-07-26-alpasim-consolidation/ALPASIM_STATE.md ; "
               "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
               "2026-07-26-4brain-dominance-program/4BRAIN_DOMINANCE_PROGRAM.md",
     "answers": "Q-TOPO and Q-HP4 simultaneously, plus tactical gates T1-T4",
     "cost_to_connect": "~1h read-only VectorMap connectivity probe on the (free) eval pod; "
                        "then a 1-3d trafficsim one-scene rollout. ZERO GPU.",
     "caveat": "the prior probe gate0_prereq_probe.json measured COUNTS ONLY and its trajectory "
               "read ERRORED -- connectivity is NOT yet established",
     "class": "INHERITED", "tier": "PROVISIONAL"},
    {"rank": 3, "finding": "the deployed tick is 74.24 GFLOP (not 401.9) and its DRAM bytes are "
                           "95.5% rollout, so every FLOP-trading lever is worth ~0%",
     "source": "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
               "2026-07-26-orin-thor-optimization/ORIN_THOR_STATE_AND_PLAN.md",
     "answers": "the H2 sensor-attention cost model (84.8-85.6% encoder-compute saving) is priced "
                "in the wrong currency for the deployment target",
     "cost_to_connect": "zero compute -- re-derive H2's cost table in DRAM bytes",
     "caveat": "sign is NOT obviously against H2: cheap FLOPs may make an occasional second "
               "encode nearly free, strengthening selective activation",
     "class": "MEASURED (each stream)", "tier": "PROVISIONAL (the cross-application is inference)"},
    {"rank": 4, "finding": "TWO INDEPENDENT streams support training-time off-path / viewpoint "
                           "augmentation: (a) P1/C14 -- the yaw warp is geometrically EXACT so ~half "
                           "the envelope is our arm's OOD sensitivity; (b) the own-dynamics-encoder "
                           "line -- in-domain rig-A speed R2 ~0.80-0.85 collapses cross-rig, and "
                           "data-diversity was REFUTED as the cause (the deficit is REPRESENTATIONAL)",
     "source": "Project Steering/RETRACTION_LOG.md 2026-07-26 C14 ; "
               "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
               "2026-07-22-own-dynamics-encoder/RESULTS_camcond.md",
     "answers": "Q-BarB -- wm_canary_ade_2s must fall 2.07x and has NO identified lever, UNOWNED",
     "why_missed": "the dynenc stream is filed as REFUTED; its NEGATIVE result is a POSITIVE "
                   "constraint on the fix (neither conditioning nor more corpus)",
     "class": "INHERITED (both measurements); the coincidence is inference",
     "tier": "PROVISIONAL -- a HYPOTHESIS for a lever, must not enter a kill conjunction"},
    {"rank": 5, "finding": "the wm_canary descent rate BOOST_PROGRAM 3.3 uses (-21.6% per 20k) is a "
                           "TRAINER-LOG number; the eval-grade pair is 2.0739@15k -> 1.1409@30k = "
                           "-45.0% per 15k, and the trainer series reads 1.4900 at step 8-10k, "
                           "i.e. LOWER at an EARLIER step -- they are not the same statistic",
     "source": ["TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/"
                "2026-07-26-v4-restart-lever/raw/lambda_verdict.json",
                "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/"
                "2026-07-25-flagship-v4-midtrain-eval/flagship-v4-fromscratch-15k.json",
                "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/"
                "2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-oracle.json"],
     "answers": "Q-BarB's difficulty estimate, which underwrites the recommendation not to restart v4",
     "retraction_classes": ["C1 -- only eval_*.py output is quotable",
                            "2026-07-25 -- a metric NAME is not a metric DEFINITION"],
     "explicit_non_claim": "I do NOT claim Bar B is reachable. Two points fit no rate and the "
                           "program forbids extrapolation without window+R2+n. The claim is only "
                           "that the plan's number comes from the wrong instrument.",
     "cost_to_connect": "zero -- both eval points already exist; a third eval point settles the rate",
     "class": "MEASURED (mine, three raw JSONs)", "tier": "CONFIRMED"},
    {"rank": 6, "finding": "HP-4 is ~17 scenes away, not a corpus rebuild away (0 of 23 topology "
                           "classes reach the >=40-cluster bar; best S|S at 38 scenes)",
     "source": "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
               "2026-07-26-vectormap-corridor/VECTORMAP_CORRIDOR.md",
     "answers": "Q-HP4; converts an 'impossible' into a procurement item and justifies BOOST 5.6",
     "class": "INHERITED", "tier": "PROVISIONAL (the 17 is ESTIMATED in its own source)"},
    {"rank": 7, "finding": "obstacle.offline (3D agent tracks, ~97% of corpus, 87481 cuboids, 10 "
                           "dynamic classes) was refused on a gate that tested EGO 2s along-track "
                           "prediction (+1.16% [-0.92,+3.19]) -- a different quantity from "
                           "world-model fidelity",
     "source": "TanitAD Research Hub/Data Engineering/Research/2026-07-21-lead-state-gate.md",
     "answers": "Q-BarB -- wm_canary is scene-latent fidelity, not ego waypoint accuracy",
     "retraction_class": "C12 -- a composite null blamed on the wrong half, at capability level",
     "explicit_non_claim": "do NOT re-ingest on this reasoning alone; that is the "
                           "mechanism-without-measurement the original gate correctly refused (C3). "
                           "The step is a SECOND gate with a WM-fidelity target on the 1.1 GB slice "
                           "already pulled.",
     "class": "INHERITED (the gate); the re-scoping is inference", "tier": "PROVISIONAL"},
    {"rank": 8, "finding": "the n=600 re-scoring path is already built and proven (RESULT_v1_600ep "
                           "exists; the 600 build is a MEASURED order-preserving superset of the 40)",
     "source": "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/"
               "2026-07-26-pod2-eval-host/artifacts/RESULT_v1_600ep.json",
     "answers": "all of H1 -- the 789-null backlog is blocked on nothing but eval-pod time",
     "class": "MEASURED", "tier": "CONFIRMED"},
    {"rank": 9, "finding": "H18 grounding dominance corrected UP to paired delta +2.9568 m and "
                           "would need an 8.65x interval widening to un-separate, against a "
                           "worst-ever-measured 2.06x",
     "source": "Project Steering/RETRACTION_LOG.md 2026-07-25 (the _jack entry)",
     "answers": "the hierarchy question, currently carried as '0 of 3 seams load-bearing'",
     "why_missed": "a retraction's good news travels worse than its bad news",
     "class": "INHERITED", "tier": "PROVISIONAL"},
    {"rank": 10, "finding": "PC3 is unblocked in code (corridor.from_windows) but unmeasured on any "
                            "real arm, because no archived arm has pred_dense",
     "source": "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
               "2026-07-26-4brain-dominance-program/4BRAIN_DOMINANCE_PROGRAM.md",
     "answers": "Q-CL -- closed-loop measurability (stream S-1), the program's #1 blocker",
     "cost_to_connect": "one flag on the next eval run",
     "class": "INHERITED", "tier": "PROVISIONAL"}
  ],
  "checkpoint_availability": {
    "note": "INHERITED from MODEL_REGISTRY.md -- NOT probed (no pod access this session)",
    "multi_copy_safe_to_rescore": ["flagship-30k (v1): HF gated + eval pod + pod2",
                                   "refc-xl-30k: HF gated + eval pod + pod3",
                                   "refc-base-30k: HF gated + eval pod + pod3",
                                   "refa-dinov2: HF + eval pod"],
    "single_disk_AT_RISK": [
      {"arm": "flagship-v16-ab-ft", "where": "pod2 only",
       "note": "registry 1.4b: 'Sayood/flagship-v16-ab-ft holds NO weights and 1.4b has no Location row'"},
      {"arm": "flagship-v4.1-10k", "where": "pod2 + eval copy",
       "note": "registry 1.5: 'single pod disk -- HF-back it once a transfer path is chosen'"},
      {"arm": "flagship-v3enc-10k", "where": "tanitad-pod",
       "note": "registry 1.4: 'DO NOT RECYCLE tanitad-pod' -- only 10k state that will ever exist"},
      {"arm": "dynenc-branchB", "where": "pod3 + durable MooseFS",
       "note": "registry 10.1: HF push BLOCKED by the safety classifier; needs a PI/user action"}
    ],
    "local_free_repairing": {
      "path": "taniteval/results/windows_*.pt",
      "count": 27,
      "what_it_buys": "re-pair ANY two arms at n=40 on the correct paired episode-cluster bootstrap, "
                      "on the dev box, in minutes, with no pod and no GPU. This is how "
                      "v16_vs_v1_paired_bootstrap.json and jack_recompute.json were produced. "
                      "It CANNOT give n=600 -- but it re-adjudicates every _jack number for free."
    }
  }
}

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "harvest_index.json")
with open(p, "w", encoding="utf-8") as fh:
    json.dump(index, fh, indent=1, ensure_ascii=True)
print("wrote", p, os.path.getsize(p), "bytes")
print("n40:", len(n40), "n12:", len(n12), "deprecated:", len(dep), "firewall_rows:", len(fw))
for r in fw[:12]:
    print(f"  fw prox {r['proximity']:.3f} ne={r['n_episodes']}  {os.path.basename(r['file'])} @ {r['json_path']}")
