#!/usr/bin/env python3
"""READ 2 (universe D, the scene-unseen tokens with a local nuPlan DB) PREPARATION.  Read-only, CPU, NO label and NO score; no pod, no fetch, no GPU.

  (a) the carve: the 103 log-less navtrain tokens that have a DB on this box (`raw/logless251_probe.json`), minus those whose log is one of the 24 held-out logs (Step A's logs), carved with Step A's OWN sampler
      (`stepA.sample_stratified` through `fresh_carve.carve`, mode log_disjoint_c) and a NEW recorded seed.  The universe is SHORT (< n): all tokens are taken, and the sampler's draw at n = |universe| is
      asserted to be the identity, so the seed is recorded and has no effect (stated, not hidden).  -> `raw/fresh_sample_D_scene_unseen.json` (same format as the other carve records; token-list sha256).
  (b) log-level M-PDM / M-SCENE exposure of D's logs, from the SAME sources the G0 check used (named by file:line in the output):
        M-PDM   = tokens P2's PDM-label loss was served (served_keys.tsv.gz, heldout_log == "0") and their logs through navtrain_token_table.tsv.gz
        M-SCENE = the final-run target bank `train_grow` (g0_traingrow_scan2.json `token_to_logs`)
      -> `raw/readtwo_exposure.json`: G0 (token and log level, with the deliberate-overlap control) and, per D log, the number of M-PDM tokens / sets and M-SCENE tokens of that log.
  (c) the fetch list: the exact `<log>/<CAM>/<hash>.jpg` names of the carve's tokens (4 cameras each), from `raw/logless251_probe_needed_frames.json`; -> `raw/readtwo_fetch_list.{json,txt}` + `raw/readtwo_plan_D.json`
      (token_log, db, token_ts_us, token_root: the planned landing root) for the patched bundle.

    python readtwo_prep.py --out C:/Users/Admin/par18/l2fresh/raw
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fresh_carve as FC  # noqa: E402
import stepA  # noqa: E402

SEED_D = 20261020             # a NEW seed (the date + 1; Step A 20260927, B 20261019), recorded BEFORE any D label or score exists; it has no effect while the universe is SHORT
PROBE = "C:/Users/Admin/par18/l2fresh/raw/logless251_probe.json"
NEEDED = "C:/Users/Admin/par18/l2fresh/raw/logless251_probe_needed_frames.json"
FRAMES_ROOT_D = "C:/Users/Admin/par18/l2fresh/frames_D"          # planned landing root of the fetch (this package; no other agent's dir)
POD_SRC = "/workspace/data/navtrain_pixels/navtrain_current_N/<log>/<CAM>/<hash>.jpg"
MEAN_JPG_MIB_PRIOR_COPY = 1069.9 / 5488                           # 5,488 files = 1,069.9 MiB (p2heldout HANDOFF_2.md, MEASURED by that agent's copy; INHERITED here)


def rd(p):
    return FC.read_tsv(p)


def source_lines() -> dict:
    """the G0 sources, by file:line, VERIFIED against the file text (a stale line number would be the stale-absence-claim class)"""
    src = open(os.path.join(HERE, "fresh_carve.py"), encoding="utf-8").read().splitlines()
    want = {"served_keys.tsv.gz path (M-PDM tokens)": 'SERVED = f"{PK}/raw/rl_pool/served_keys.tsv.gz"', "navtrain_token_table.tsv.gz path (token -> log)": 'TOKTAB = f"{PK}/raw/rl_pool/navtrain_token_table.tsv.gz"',
            "g0_traingrow_scan2.json path (M-SCENE = train_grow)": 'SCAN = "C:/Users/Admin/par18/l1run/code/raw/g0_traingrow_scan2.json"',
            "M-PDM token set: served keys with heldout_log == 0": 'train_tok_A = {r["token"] for r in read_tsv(SERVED) if r["heldout_log"] == "0"}',
            "M-PDM log set: the token table's log of those tokens": 'train_log_A = {ttab[t]["log"] for t in train_tok_A if t in ttab}',
            "M-SCENE token -> logs": 'scan = json.load(open(SCAN, encoding="utf-8"))["token_to_logs"]'}
    out = {}
    for k, text in want.items():
        hits = [i + 1 for i, ln in enumerate(src) if text in ln]
        if len(hits) != 1:
            raise SystemExit(f"source line for {k!r} not found exactly once in fresh_carve.py ({hits})")
        out[k] = f"code/fresh_carve.py:{hits[0]}"
    out["original G0 check (L1 Step A)"] = "C:/Users/Admin/par18/l1run/code/g0_overlap.py:5 (SOURCE A) and :49 (--scan = the final-run target bank); its output l1run/code/raw/g0_overlap.json"
    return out


def build(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    probe = json.load(open(PROBE))
    found = probe["found"]                                                     # token -> [log, ts_us, db path]
    plan = json.load(open(FC.PLAN))
    sA = json.load(open(f"{FC.W}/sample.json"))
    tl = plan["token_log"]
    logsA = {tl[t] for t in sA["tokens"]}
    rows = [{"token": t, "log": v[0], "served": False} for t, v in sorted(found.items())]
    rec = FC.carve("log_disjoint_c", plan, sA["tokens"], 400, SEED_D, rows)
    rec["mode"] = "scene_unseen_D"
    rec["rule"] = ("the 103 navtrain tokens whose log the token table does not record (never served to P2's PDM-label loss, absent from train_grow) that have a nuPlan DB on this box, "
                   "minus those whose DB log is one of Step A's 24 logs; carved with Step A's sampler")
    toks, logs, _ = FC.universe("log_disjoint_c", plan, sA["tokens"], rows)
    # the sampler's draw at n = |universe| is the identity (so the SHORT 'take all' is what Step A's sampler would return); asserted, not assumed
    ident = sorted(stepA.sample_stratified(toks, logs, len(toks), SEED_D))
    rec["sampler_identity_at_n_equals_universe"] = bool(ident == sorted(toks))
    if not rec["sampler_identity_at_n_equals_universe"]:
        raise SystemExit("Step A's sampler at n = |universe| is not the identity: the SHORT 'take all' would not be a sampler draw")
    rec["seed_note"] = f"seed {SEED_D} is recorded; the universe ({len(toks)}) is smaller than n_requested (400), so every token is taken and the seed has no effect"
    rec["tokens_sha256"] = FC.sha_tokens(rec["tokens"])
    rec["parent_prereg_sha256"] = stepA.PREREG_SHA256
    rec["stepA_sample_tokens_sha256"] = FC.sha_tokens(sA["tokens"])
    rec["stepA_seed"] = sA["seed"]
    rec["excluded_in_the_24_heldout_logs"] = sorted(t for t, v in found.items() if v[0] in logsA)
    rec["g0"] = FC.g0_for(rec, sA["tokens"])
    json.dump(rec, open(os.path.join(out_dir, "fresh_sample_D_scene_unseen.json"), "w", encoding="utf-8"), indent=1)

    # ---- (b) per-log exposure, from the same sources as g0_for
    ttab = {r["token"]: r for r in rd(FC.TOKTAB)}
    served_rows = [r for r in rd(FC.SERVED) if r["heldout_log"] == "0"]
    sets_per_tok = collections.Counter(r["token"] for r in served_rows)
    pdm_tokens_by_log, pdm_sets_by_log = collections.defaultdict(set), collections.Counter()
    for t, n in sets_per_tok.items():
        if t in ttab and ttab[t]["log"]:
            pdm_tokens_by_log[ttab[t]["log"]].add(t)
            pdm_sets_by_log[ttab[t]["log"]] += n
    scan = json.load(open(FC.SCAN, encoding="utf-8"))["token_to_logs"]
    scene_tokens_by_log = collections.defaultdict(set)
    for t, ls in scan.items():
        for l in ls:
            scene_tokens_by_log[l].add(t)
    dlogs = sorted(set(rec["logs"]))
    tok_per_log = collections.Counter(rec["logs"])
    per_log = {l: {"d_tokens": tok_per_log[l], "mpdm_tokens": len(pdm_tokens_by_log.get(l, ())), "mpdm_sets": int(pdm_sets_by_log.get(l, 0)), "mscene_tokens": len(scene_tokens_by_log.get(l, ()))} for l in dlogs}
    n_pdm, n_scene = sum(1 for v in per_log.values() if v["mpdm_tokens"] > 0), sum(1 for v in per_log.values() if v["mscene_tokens"] > 0)

    def dist(key):
        v = sorted(x[key] for x in per_log.values())
        return {"min": v[0], "median": v[len(v) // 2], "max": v[-1], "sum": int(sum(v))}

    exp = {"what": "log-level M-PDM and M-SCENE exposure of universe D's logs (MEASURED, CPU, no label or score read)", "evidence": "MEASURED (ours); sources below",
           "sources": source_lines(), "carve": {"n_tokens": rec["n"], "n_logs": len(dlogs), "tokens_sha256": rec["tokens_sha256"], "status": rec["status"]},
           "g0": rec["g0"], "logs_with_any_mpdm_token": n_pdm, "logs_with_any_mscene_token": n_scene,
           "per_log_distribution": {"mpdm_tokens": dist("mpdm_tokens"), "mpdm_sets": dist("mpdm_sets"), "mscene_tokens": dist("mscene_tokens"), "d_tokens": dist("d_tokens")},
           "reading": ("a D log with M-PDM tokens is a log P2's PDM-label loss trained on (through OTHER tokens of that log): the D TOKENS are unseen, the D LOG is not. "
                       "G0 A1.2 is evaluated on the fit logs; D is not the L2 fit set (the L2 heads' fit logs are the 24 held-out logs, none of which is in D), so what the log-level overlap licenses is a statement about P2's "
                       "exposure, and about the scene bank, not about the L2 head."),
           "per_log": per_log}
    json.dump(exp, open(os.path.join(out_dir, "readtwo_exposure.json"), "w", encoding="utf-8"), indent=1)

    # ---- (c) fetch list and the plan for the patched bundle
    need = json.load(open(NEEDED))["tokens"]
    fetch, plan_d = [], {"what": "plan entries of universe D for the patched bundle (token_log, db, token_ts_us, token_root)", "tokens": rec["tokens"], "token_log": {}, "db": {}, "token_ts_us": {}, "token_root": {}}
    for t in rec["tokens"]:
        lg, ts, dbp = found[t]
        plan_d["token_log"][t], plan_d["db"][lg], plan_d["token_ts_us"][t], plan_d["token_root"][t] = lg, dbp, int(ts), FRAMES_ROOT_D
        for cam, fr in sorted(need[t]["frames"].items()):
            fetch.append({"token": t, "cam": cam, "path": fr["filename_jpg"], "dt_ms": fr["dt_ms"]})
    paths = sorted(f["path"] for f in fetch)
    fl = {"what": "the jpgs READ 2 needs: nearest frame per camera to each D token's timestamp (probe_logless251.py); pod source layout " + POD_SRC, "n_tokens": rec["n"], "n_jpgs": len(paths), "n_unique_paths": len(set(paths)),
          "max_dt_ms": max(f["dt_ms"] for f in fetch), "estimated_mib": round(len(paths) * MEAN_JPG_MIB_PRIOR_COPY, 1),
          "estimate_basis": "mean 0.195 MiB per jpg from the earlier pod copy of 5,488 files = 1,069.9 MiB (p2heldout HANDOFF_2.md; INHERITED); the MM brief says about 85 MB for 412 jpgs (103 tokens)",
          "list_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(), "landing_root": FRAMES_ROOT_D, "items": fetch,
          "superset_in_the_probe": {"n_tokens": len(need), "n_jpgs": sum(len(v["frames"]) for v in need.values()), "note": "the probe lists all 103 DB tokens (412 jpgs); the 4 tokens in the 24 held-out logs are not in D"}}
    json.dump(fl, open(os.path.join(out_dir, "readtwo_fetch_list.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(out_dir, "readtwo_fetch_list.txt"), "w", newline="\n").write("\n".join(paths) + "\n")
    json.dump(plan_d, open(os.path.join(out_dir, "readtwo_plan_D.json"), "w", encoding="utf-8"), indent=1)
    return {"carve": {k: rec[k] for k in ("status", "n", "universe_tokens", "universe_logs", "tokens_sha256", "seed")}, "n_logs_in_carve": len(dlogs), "excluded": rec["excluded_in_the_24_heldout_logs"],
            "g0": {k: rec["g0"].get(k) for k in ("evaluable", "A1_2_verdict")}, "g0_mpdm": rec["g0"].get("M-PDM"), "g0_mscene": rec["g0"].get("M-SCENE"), "logs_with_mpdm": n_pdm, "logs_with_mscene": n_scene,
            "fetch": {k: fl[k] for k in ("n_jpgs", "n_unique_paths", "max_dt_ms", "estimated_mib", "list_sha256")}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print("ZZREADTWO_PREP", json.dumps(build(a.out), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
