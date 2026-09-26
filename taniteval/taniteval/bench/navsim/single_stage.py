"""``python -m taniteval.bench navsim_v2 --split navtest_single_stage …`` — NAVSIM v2 EPDMS, ONE stage.

W8 (EvalFlyWheel, 2026-09-26; PI: "wire the navtest single-stage split"). The devkit's ONE-STAGE
runner (``run_pdm_score_one_stage.py`` @0a380a9) over the original navtest frames:

    preflight (runtime, devkit yaml counts, metric cache by CONTENT, every needed log on the named dir,
               venv import probe)
    -> export (devkit AgentInputs, single-stage mode) -> STOP seam
    -> official ONE-STAGE scoring per arm (CV, STOP, [HUMAN]) through the promoted E1 wrapper
    -> scores/<arm>.csv (UNMODIFIED) · artifacts/<arm>.json · criteria/<arm>.txt · plans/ · scenes.json
    -> summary.json (schema-validated)

Built ON the two-stage suite, never beside it: same arm specs, seam agent, scorer guards, wrapper,
artifact gates, criteria check, estimator module and contract. What differs is the protocol's:

* ONE summary row, ``average_all_frames`` = ``pdm_score_df[score_cols].mean(skipna=True)``
  (run_pdm_score_one_stage.py:293) — the headline, read as TEXT from the ``score`` column;
* no stage 2, no synthetic scene, no mapping; ``per_stage`` carries ONE block and says so;
* HUMAN is a legitimate REFERENCE here (every token is an original frame with a logged future);
  on a two-stage split it stays refused (E1 finding 3);
* background traffic NON-REACTIVE log replay (``traffic_agents=non_reactive``, passed explicitly);
* the interval: W2's registered ``navsim_log_cluster_bootstrap`` with the SINGLE-STAGE aggregation
  (``single_stage_token_mean``) over the run's ``log_name`` clusters (navtest: 136 >= the RG-14 floor 8).

⚠ ``--tokens-file`` scores a SUBSET (the smoke). A subset run is forced to ``_scratch/`` and is never
claim-bearing: two-frame extended comfort (EC) pairs ADJACENT scored tokens
(run_pdm_score_one_stage.py:129-156), so a subset's per-token ``score`` differs from the full split's.

⛔ MODEL ARMS are REFUSED on this split for now: the suite's bridge has no navtest frame bank wired
(W3's 32-shard bank exists in W3's format, ``D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank``).
Unblock = a ``model_arms.DEFAULT_BANKS['navtest_single_stage']`` entry + a stage-1-only bank join.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

from ..contract import REPO, NAVSIM_FLOORS
from . import artifacts as ART
from . import plans as PL
from . import profiles as P
from . import scoring as SC
from . import summarize as S

SUMMARY_ROW = "average_all_frames"
TIE = 1e-12
#: the one-stage CSV's metric columns (run_pdm_score_one_stage.py:283-290 keeps PDMResults fields + EC + score)
SUB_COLS = dict(S.SUB)

HUMAN_SPEC = {"kind": "reference", "official_agent": "human_agent",
              "declared_inputs": ["logged future trajectory (PRIVILEGED, label-side only)"],
              "spec": {"agent": "human_agent (the devkit's HumanAgent: the logged ego future, 8 poses at 0.5 s)",
                       "ii": dict(cameras=False, lidar=False, ego_velocity=False, ego_acceleration=False,
                                  ego_pose_history=False, driving_command=False, privileged=True),
                       "sensor_set": "none — PRIVILEGED: returns the logged human future (a reference, never a claim)",
                       "setting": "reference ceiling: the logged expert; post-#151 human filter ON",
                       "goal": "none", "route_input": False}}


def _ci():
    """W2's NavSim estimator, loaded by file path (``navsim`` is also the DEVKIT's package name)."""
    spec = importlib.util.spec_from_file_location("taniteval_bench_navsim_ci",
                                                  REPO / "taniteval" / "adapters" / "navsim_ci.py")
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    return ci


def resolve_arms_single(requested: list, ckpt) -> tuple:
    """-> (ordered [(name, info)], notes). STOP + CV always (floors first); HUMAN only when requested."""
    from .benchmark import FLOOR_SPECS
    req = [x.strip().upper() for x in requested if x.strip()]
    if ckpt is not None:
        raise P.Refusal("model arms / --ckpt are not wired on a single-stage split yet: the suite's bridge has no "
                        "navtest frame bank (W3's bank exists in W3's format). Unblock = model_arms.DEFAULT_BANKS"
                        "['navtest_single_stage'] + a stage-1-only bank join. Run the floors with --ckpt none.")
    allowed = {"CV", "STOP", "HUMAN"}
    for a in req:
        if a not in allowed:
            raise P.Refusal(f"arm {a!r} is not available on a single-stage split (available {sorted(allowed)}; "
                            "ECHO needs a checkpoint, model arms are not wired here)")
    order, notes = [], []
    for f in ("CV", "STOP"):
        added = f not in req
        order.append((f, {**FLOOR_SPECS[f], "added_by_rule": added}))
        if added:
            notes.append(f"{f} added by rule (MANDATORY NavSim floor)")
    if "HUMAN" in req:
        order.append(("HUMAN", {**HUMAN_SPEC, "added_by_rule": False}))
    return order, notes


def load_token_scores(path) -> dict:
    """{token: {col: float}} of a one-stage CSV's TOKEN rows (text-parsed; '' -> NaN)."""
    hdr, raw = S.read_raw_rows(path)
    out = {}
    for t, r in raw.items():
        if t == SUMMARY_ROW or t.startswith("extended_pdm_score"):
            continue
        out[t] = {c: (float(r[c]) if r.get(c) not in (None, "", "nan") else float("nan"))
                  for c in list(SUB_COLS.values()) + ["score"] if c in r}
        out[t]["valid"] = r.get("valid")
    return out


def average_row_check(tok: dict, avg_score: float) -> dict:
    """C-AVG: the devkit's ``average_all_frames`` score == the skipna mean of the token scores."""
    v = [r["score"] for r in tok.values() if not math.isnan(r["score"])]
    mine = float(np.mean(v)) if v else float("nan")
    d = abs(mine - avg_score) if v else float("inf")
    return {"recomputed_mean": mine, "devkit_average_all_frames": avg_score, "abs_diff": d, "tol": 1e-12,
            "n": len(v), "pass": bool(d <= 1e-12)}


def interval_single(ci, protocol: str, tok: dict, t2l: dict, head: float, n_logs: int) -> dict:
    scores = {t: r["score"] for t, r in tok.items()}
    try:
        blk = ci.interval_from_run(protocol=protocol, clusters_by_unit={t: t2l[t] for t in scores},
                                   scores=scores, official_value=head)
        verdict, why = ci.is_admissible_interval(blk, protocol=protocol, max_clusters=n_logs)
    except Exception as e:                                               # noqa: BLE001
        return {"status": "UNAVAILABLE", "n": n_logs, "reason": f"W2 estimator raised {type(e).__name__}: {e}"[:400]}
    if verdict == "PASS" and blk.get("lo") is not None:
        return {"status": "OK", "estimator": blk.get("estimator"), "cluster_unit": blk.get("cluster_unit", "log_name"),
                "lo": blk["lo"], "hi": blk["hi"], "n_clusters": int(blk.get("n_clusters")), "detail": blk,
                "question_answered": getattr(ci, "QUESTION_ANSWERED", None)}
    return {"status": "UNAVAILABLE", "n": int(blk.get("n_clusters") or n_logs),
            "reason": f"W2 estimator verdict {verdict}: {why}"[:600], "detail": blk}


def paired_single(ci, a: dict, b: dict, t2l: dict, head_a: float, head_b: float) -> dict:
    common = sorted(set(a) & set(b))
    if not common:
        return {"status": "UNAVAILABLE", "reason": "no common tokens", "n": 0}
    d = np.asarray([a[t]["score"] - b[t]["score"] for t in common], dtype=np.float64)
    out = {"status": "OK", "headline_delta": head_a - head_b, "n_common": len(common),
           "token_mean_delta": float(np.nanmean(d)),
           "wins": int((d > TIE).sum()), "ties": int((np.abs(d) <= TIE).sum()), "losses": int((d < -TIE).sum()),
           "_wtl_scope": "per-token wins/ties/losses on the common tokens (tie band 1e-12)",
           "submetric_mean_deltas": {k: float(np.nanmean([a[t][c] - b[t][c] for t in common]))
                                     for k, c in SUB_COLS.items() if c in a[common[0]] and c in b[common[0]]}}
    try:
        out["interval"] = ci.paired_log_cluster_bootstrap(
            ci.single_stage_contributions([a[t]["score"] for t in common]),
            ci.single_stage_contributions([b[t]["score"] for t in common]),
            [t2l[t] for t in common], aggregation=ci.AGG_SINGLE_STAGE, official_a=head_a, official_b=head_b)
    except Exception as e:                                               # noqa: BLE001
        out["interval"] = {"status": "UNAVAILABLE", "n": len(common),
                           "reason": f"paired log-cluster bootstrap failed: {type(e).__name__}: {e}"[:300]}
    return out


def per_log_single(tok: dict, t2l: dict) -> dict:
    out = {"_statistic": "PLAIN mean of the official per-token `score` by OpenScene log (single stage)"}
    by = {}
    for t, r in tok.items():
        by.setdefault(t2l.get(t, "?"), []).append(r["score"])
    for ln in sorted(by):
        v = [x for x in by[ln] if not math.isnan(x)]
        out[ln] = {"n": len(by[ln]), "score_mean": (float(np.mean(v)) if v else None)}
    return out


class ArmReuseRefused(P.Refusal):
    """A single-stage arm adoption the identity check would not allow — never downgraded to a warning."""


def adopt_scored_arm(*, arm: str, src_run: Path, run, prof: "P.SplitProfile", tokens_expected: set,
                     devkit_sha: str, patches: list, preflight: dict, export_sha256: str | None,
                     new_seam: Path | None) -> dict:
    """``--reuse-scored-arms`` (W8 2026-09-26): adopt ONE arm that a previous run of this split SCORED
    COMPLETELY, instead of re-scoring it — the single-stage twin of W7's ``floor_reuse`` (MEASURED on
    navhard: RAM-guard aborts killed two attempts at ~80 % of an arm; a retry that re-scores the arms that
    already PASSED re-buys that exposure). Raises :class:`ArmReuseRefused` on ANY mismatch; there is no
    fall-back to a partial adoption. Checked, each against THIS run: the source arm PASSED on exactly the
    expected token count; split, protocol, runner script, traffic policy; devkit sha and the patch set by
    name AND raw-bytes blob; the metric cache path AND its CACHE_DONE token/manifest sha256; the token set
    BY VALUE; the agent-input export sha256; a seam arm's seam BYTE-EQUAL."""
    from .floor_reuse import _patch_key
    src_run = Path(src_run)
    bad = []
    try:
        sbr = json.loads((src_run / "bench_run.json").read_text(encoding="utf-8"))
        sc = json.loads((src_run / "raw" / arm / f"{arm}.counts.json").read_text(encoding="utf-8"))
    except Exception as e:                                               # noqa: BLE001
        raise ArmReuseRefused(f"{arm}: source run {src_run} unreadable ({type(e).__name__}: {e})")
    n = len(tokens_expected)
    for k, want in (("status", "PASS"), ("rc", 0), ("log_successful", n), ("log_failed", 0),
                    ("csv_valid_rows", n), ("runner_script", P.runner_for(prof)),
                    ("traffic_agents", prof.traffic_agents)):
        if sc.get(k) != want:
            bad.append(f"source {arm}.{k} = {sc.get(k)!r}, need {want!r}")
    if (sbr.get("split") or {}).get("name") != prof.name or sbr.get("protocol") != prof.protocol:
        bad.append(f"split/protocol differ: {(sbr.get('split') or {}).get('name')!r}/{sbr.get('protocol')!r}")
    if (sbr.get("devkit") or {}).get("sha") != devkit_sha:
        bad.append(f"devkit sha differs: {(sbr.get('devkit') or {}).get('sha')} != {devkit_sha}")
    if _patch_key((sbr.get("devkit") or {}).get("patches")) != _patch_key(patches):
        bad.append("patch set differs (name or raw-bytes blob)")
    if str(sc.get("cache")) != str(prof.cache).replace(os.sep, "/"):
        bad.append(f"metric cache path differs: {sc.get('cache')} != {prof.cache}")
    try:
        spre = json.loads((src_run / "raw" / "preflight.json").read_text(encoding="utf-8"))
    except Exception:                                                    # noqa: BLE001
        spre = {}
    s_cd, n_cd = ((spre.get("cache") or {}).get("cache_done") or {}), ((preflight.get("cache") or {}).get("cache_done") or {})
    for k in ("tokens_sha256", "manifest_sha256"):
        if not s_cd.get(k) or s_cd.get(k) != n_cd.get(k):
            bad.append(f"metric cache identity {k}: source {s_cd.get(k)} != this run {n_cd.get(k)}")
    got = set(load_token_scores(src_run / "scores" / f"{arm}.csv")) if (src_run / "scores" / f"{arm}.csv").exists() else set()
    if got != set(tokens_expected):
        bad.append(f"token set differs: {len(got)} vs {n} ({len(got - set(tokens_expected))} extra, "
                   f"{len(set(tokens_expected) - got)} missing)")
    s_exp = src_run / "raw" / "export_record.json"
    s_sha = json.loads(s_exp.read_text(encoding="utf-8")).get("sha256") if s_exp.exists() else None
    if new_seam is not None and s_sha != export_sha256:
        bad.append(f"agent-input export differs: {s_sha} != {export_sha256}")
    if new_seam is not None:
        s_seam = src_run / "raw" / "seams" / Path(new_seam).name
        if not s_seam.exists():
            bad.append(f"source seam {s_seam} absent")
        else:
            a, b = np.load(s_seam, allow_pickle=False), np.load(new_seam, allow_pickle=False)
            if not (list(a["token"]) == list(b["token"]) and np.array_equal(a["poses"], b["poses"])):
                bad.append("seam differs (tokens or poses) — the arm itself changed")
    if bad:
        raise ArmReuseRefused(f"{arm}: adoption from {src_run.name} REFUSED — " + "; ".join(bad))
    import shutil
    dst = run.p(f"raw/{arm}")
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((src_run / "raw" / arm).iterdir()):
        if f.is_file():
            shutil.copyfile(f, dst / f.name)
    rec = {"arm": arm, "source_run": str(src_run).replace(os.sep, "/"), "identity_checks": "all passed",
           "source_counts_status": sc.get("status"), "n_tokens": n,
           "why": "the arm PASSED completely in the source run under the identical devkit, patches, cache, inputs, "
                  "runner and traffic policy — a re-score would reproduce its rows (deterministic scorer, E1 C7)"}
    (dst / f"{arm}.reused.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return {**sc, "csv": str(dst / f"{arm}.devkit.csv"), "reused_from": rec["source_run"]}


def run_single_stage(ctx, prof: "P.SplitProfile") -> None:                   # noqa: C901
    from . import benchmark as NB
    from .export import ensure_export
    from . import seams as SM
    a, run = ctx.args, ctx.run
    tokens = P.read_tokens_file(a.tokens_file) if getattr(a, "tokens_file", None) else None
    cache_override = getattr(a, "metric_cache", None)
    if cache_override and tokens is None:
        raise P.Refusal("--metric-cache is only accepted with --tokens-file: a FULL-split run reads the split's "
                        f"verified canonical cache {prof.cache}")
    prof_run = dataclasses.replace(prof, cache=Path(cache_override)) if cache_override else prof
    arms, notes = resolve_arms_single(a.arms, a.ckpt)
    for n in notes:
        ctx.log(f"[navsim] {n}")
    y = P.read_split_yaml(prof.name)
    t2l = P.token_to_log(prof)
    toks = sorted(tokens) if tokens is not None else sorted(y["stage_one"])
    logs_run = sorted({t2l[t] for t in toks})
    dk = P.devkit_sha_measured()
    ctx.set_protocol(prof.protocol)
    subset = ({"n_tokens": len(toks), "n_logs": len(logs_run), "tokens_file": str(a.tokens_file).replace(os.sep, "/"),
               "why_not_a_result": ("a SUBSET: EC pairs adjacent scored tokens (run_pdm_score_one_stage.py:129-156), so "
                                    "per-token scores differ from the full split; forced to _scratch/, not claim-bearing")}
              if tokens is not None else None)
    ctx.set_split(prof.name, len(toks), len(logs_run), n_stage_one=len(toks), n_stage_two=0, single_stage=True,
                  devkit_split=prof.tts, notes=list(prof.notes), **({"subset": subset} if subset else {}))
    ctx.set_devkit(P.DEVKIT_REPO, dk["sha"], P.devkit_patches(), pin=P.DEVKIT_PIN_NOTE, sha_probe=dk,
                   runtime=str(P.CR).replace(os.sep, "/"), wrapper=str(P.WRAPPER.relative_to(REPO)).replace(os.sep, "/"),
                   runner="navsim.planning.script.run_pdm_score_one_stage", fix151="post")
    loop = {"single_stage": ("OPEN (PI ruling 2026-09-02): one query on a real logged frame; the plan is fixed and "
                             "propagated by LQR + kinematic bicycle at 10 Hz over 4 s; the ego is never re-queried"),
            "background_traffic": (f"NON-REACTIVE log replay (traffic_agents={prof.traffic_agents}, "
                                   "log_replay_traffic_agents.py; agents intersecting the ego at t0 are removed)"),
            "reactive_background_traffic": "NO (log replay; the ego is non-reactive in every NAVSIM runner)"}
    # ⚠ every loop VALUE is a string: W5's renderer formats them (MEASURED 2026-09-26: a bool here raised
    # AttributeError in benchreport/render.py:160 and the smoke's report failed). The machine-readable flag
    # lives beside the loop, not inside it.
    ctx.set_stamps("T1-family", loop, "MEASURED", closed_loop=False, traffic_agents=prof.traffic_agents,
                   reactive_background_traffic=False)
    ctx.set_claim_bearing(tokens is None)
    for name, info in arms:
        ctx.add_arm(name, info["kind"], info["declared_inputs"], added_by_rule=info.get("added_by_rule", False),
                    status="FAILED")
    pre = P.preflight(prof_run, tokens=tokens)
    run.write_json("raw/preflight.json", pre)
    if getattr(a, "dry_run", False):
        run.write_json("raw/plan.json", {"arms": [n for n, _ in arms], "notes": notes, "preflight": pre,
                                         "runner": P.runner_for(prof), "subset": subset,
                                         "overrides_example": SC.overrides_for(prof_run, exp_name="<exp>",
                                                                               worker="sequential", tokens=tokens,
                                                                               token_log=t2l)})
        for n, _ in arms:
            ctx.arm_rec(n)["status"] = "SKIPPED"
        ctx.rec["status"] = "DRY_RUN_PASSED"
        ctx.rec["dry_run"] = True
        ctx.rec["dry_run_note"] = "DRY RUN: preflight passed; nothing scored"
        return
    exp_dir = P.EXP_ROOT / "runs" / ctx.run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    doc, exp_rec = ensure_export(prof_run, tokens=tokens, log=ctx.log)
    run.write_json("raw/export_record.json", exp_rec)
    seam_dir = run.p("raw/seams")
    seam_dir.mkdir(parents=True, exist_ok=True)
    seams = {name: SM.make_stop_seam(doc, seam_dir / f"{name}.npz", arm=name)
             for name, info in arms if info.get("seam") == "stop"}
    scored, counts, adopted = {}, {}, {}
    reuse_src = getattr(a, "reuse_scored_arms", None)
    for name, info in arms:
        rep = None
        if reuse_src:
            src_counts = Path(reuse_src) / "raw" / name / f"{name}.counts.json"
            src_ok = src_counts.exists() and json.loads(src_counts.read_text(encoding="utf-8")).get("status") == "PASS"
            if src_ok:                     # a PASSED source arm is adopted or the run REFUSES — never half-trusted
                rep = adopt_scored_arm(arm=name, src_run=Path(reuse_src), run=run, prof=prof_run, tokens_expected=set(toks),
                                       devkit_sha=dk["sha"], patches=P.devkit_patches(), preflight=pre,
                                       export_sha256=exp_rec.get("sha256"), new_seam=seams.get(name))
                adopted[name] = rep["reused_from"]
                ctx.log(f"[navsim] {name}: ADOPTED from {Path(reuse_src).name} (identity checks passed; not re-scored)")
            else:
                ctx.log(f"[navsim] {name}: the source run did not PASS this arm — scoring it")
        if rep is None:
            rep = SC.score_arm(arm=name, prof=prof_run, raw_dir=run.p(f"raw/{name}"), exp_dir=exp_dir,
                               seam=seams.get(name), official_agent=info.get("official_agent"),
                               ram_floor_mb=getattr(a, "ram_floor_mb", 3000.0), log=ctx.log,
                               baseline_blobs=(ctx.rec.get("git") or {}).get("suite_code_blobs") or {},
                               tokens=tokens, token_log=t2l)
        counts[name] = rep
        if rep["status"] == "PASS":
            ctx.write_scores(name, rep["csv"])
            scored[name] = run.p(f"scores/{name}.csv")
            ctx.arm_rec(name)["status"] = "OK"
    gallery_inputs = {name: PL.write_plans(run.p(f"plans/{name}.npz"), arm=name,
                                           hooks=NB._hooks(run.p(f"raw/{name}"), name), seam=seams.get(name))
                      for name in scored}
    gallery_inputs["scenes"] = PL.write_scenes(run.p("scenes.json"), doc, prof=prof_run, frame_bank=None)
    run.write_json("raw/gallery_inputs.json", gallery_inputs)

    # ------------------------------------------------------------------ summary
    ci = _ci()
    tok, raw, heads = {}, {}, {}
    for name in scored:
        tok[name] = load_token_scores(scored[name])
        _, raw[name] = S.read_raw_rows(scored[name])
        heads[name] = float(raw[name][SUMMARY_ROW][S.HEADLINE_COLUMN])          # TEXT-parsed, the file's value
    tokset = set(toks)
    summ_arms = {}
    for name, info in arms:
        spec = info["spec"]
        if name not in scored:
            summ_arms[name] = {
                "kind": info["kind"], "status": "FAILED", "declared_inputs": ctx.arm_rec(name)["declared_inputs"],
                "headline": {"status": "UNAVAILABLE", "reason": f"scoring FAILED: {counts[name]['failures']}", "n": 0},
                "per_stage": {}, "submetrics": {}, "per_log": {},
                "paired": {f: ({"status": "SELF"} if f == name else
                               {"status": "UNAVAILABLE", "reason": "this arm was not scored", "n": 0})
                           for f in ("STOP", "CV")},
                "interval": {"status": "UNAVAILABLE", "reason": "not scored", "n": 0},
                "families": S.families_refused("not scored"), "files": {"counts": f"raw/{name}/{name}.counts.json"}}
            continue
        import pandas as pd
        full_df = pd.read_csv(scored[name], index_col=0)
        ctl = {"C4_formula": S.c4(full_df, False), "C_AVG": average_row_check(tok[name], heads[name]),
               "C_COUNT": {k: counts[name].get(k) for k in ("log_successful", "log_failed", "csv_token_rows",
                                                            "csv_valid_rows", "expected_tokens")}}
        interval = interval_single(ci, prof.protocol, tok[name], t2l, heads[name], len(logs_run))
        hooks = NB._hooks(run.p(f"raw/{name}"), name)
        art = ART.build_arm_artifact_single_stage(
            arm=name, spec=spec, split=prof.name, devkit_split=prof.tts, protocol=prof.protocol, raw_rows=raw[name],
            hooks=hooks, tokens=tokset, n_logs=len(logs_run), interval=interval, loop=loop, controls=dict(ctl))
        ctx.write_artifact(name, art)
        ctl["criteria_check"] = ctx.run_criteria(name)
        ctl["navsim_gates"] = art["navsim_gate_selfcheck"]["gates"]
        ctl["navsim_gate_mutations_all_red"] = art["navsim_gate_selfcheck"]["mutations_all_red"]
        avg = raw[name][SUMMARY_ROW]
        sub_avg = {k: (float(avg[c]) if avg.get(c) not in (None, "", "nan") else None) for k, c in SUB_COLS.items()}
        n_ec_nan = int(sum(1 for r in tok[name].values() if math.isnan(r.get("two_frame_extended_comfort", float("nan")))))
        paired = {}
        for f in ("STOP", "CV"):
            if f == name:
                paired[f] = {"status": "SELF"}
            elif f in scored:
                paired[f] = paired_single(ci, tok[name], tok[f], t2l, heads[name], heads[f])
            else:
                paired[f] = {"status": "UNAVAILABLE", "reason": f"floor {f} was not scored in this run", "n": 0}
        summ_arms[name] = {
            "kind": info["kind"], "status": "OK", "declared_inputs": ctx.arm_rec(name)["declared_inputs"],
            "added_by_rule": info.get("added_by_rule", False),
            "headline": {"value": heads[name], "x100": 100.0 * heads[name], "column": S.HEADLINE_COLUMN,
                         "row": SUMMARY_ROW, "n": int(len(tok[name])),
                         "statistic": ("official ONE-STAGE EPDMS: the devkit's average_all_frames row = skipna mean of "
                                       "the per-token `score` (run_pdm_score_one_stage.py:293); per token "
                                       "NC·DAC·DDC·TLC·(5EP+5TTC+2LK+2HC+2EC)/16, EC NaN -> /14 (L177-191)")},
            "per_stage": {"single_stage": {"score": heads[name], "column": S.HEADLINE_COLUMN, "row": SUMMARY_ROW,
                                           "n": int(len(tok[name])), "submetrics": sub_avg},
                          "_note": "ONE-STAGE split: there is no stage 2, no synthetic scene and no mapping"},
            "submetrics": {"variant": "EPDMS_v2", "row": SUMMARY_ROW, "denominator": 16,
                           "formula": "NC·DAC·DDC·TLC·(5EP+5TTC+2LK+2HC+2EC)/16 (EC NaN -> /14)",
                           "values": sub_avg, "n_tokens_EC_nan_dropped_to_14": n_ec_nan},
            "per_log": per_log_single(tok[name], t2l),
            "paired": paired, "interval": interval,
            "families": S.families_from_artifact(art, art["four_families"].get("_w1_scope", "")),
            "statistics": {"token_mean_score": float(np.nanmean([r["score"] for r in tok[name].values()])),
                           "n_tokens": len(tok[name])},
            "caveats": NB._caveats(spec) + ([{"id": "SUBSET_RUN", "one_line": subset["why_not_a_result"]}] if subset else [])
            + ([{"id": "ARM_ADOPTED_FROM_SOURCE_RUN",
                 "one_line": f"{name} was NOT re-scored: its rows are adopted, byte-identical, from {adopted[name]}",
                 "record": f"raw/{name}/{name}.reused.json"}] if name in adopted else [])
            + [{"id": "FIX151_POST_ONLY_COMPARABLE",
                "one_line": ("post-#151 EPDMS (devkit 0a380a9): comparable ONLY to published rows marked fix151=post; "
                             "the pre/unverified rows of this column are a different implementation")}],
            "modality": {"sensor_set": spec["sensor_set"], "setting": spec["setting"],
                         "ego_status_used": bool(spec["ii"].get("ego_velocity") or spec["ii"].get("ego_acceleration")),
                         "vision_only_claimed": False, "privileged": bool(spec["ii"].get("privileged", False))},
            "controls": ctl,
            "files": {"scores": f"scores/{name}.csv", "artifact": f"artifacts/{name}.json",
                      **({"plans": f"plans/{name}.npz"} if (gallery_inputs.get(name) or {}).get("status") == "OK" else {}),
                      "criteria": f"criteria/{name}.txt", "counts": f"raw/{name}/{name}.counts.json",
                      "wrapper_manifest": f"raw/{name}/{name}_manifest.json", "hooks": f"raw/{name}/{name}_hooks.json",
                      "final_scores_frame": f"raw/{name}/{name}_final_scores_frame.csv"},
        }
    summary = {
        "schema": "taniteval.bench.summary/1", "run_id": ctx.run_id, "benchmark": "navsim_v2",
        "protocol": prof.protocol, "split": prof.name, "claim_bearing": tokens is None, "evidence_class": "MEASURED",
        "stamps": {"tier": "T1-family", "loop": loop, "closed_loop": False, "traffic_agents": prof.traffic_agents,
                   "reactive_background_traffic": False},
        "headline_metric": {"name": "EPDMS (NAVSIM v2, one-stage)", "column": S.HEADLINE_COLUMN, "higher_is_better": True,
                            "statistic": "official ONE-STAGE EPDMS (row average_all_frames)",
                            "forbidden_column": S.FORBIDDEN_COLUMN, "range": [0.0, 1.0]},
        "floors": list(NAVSIM_FLOORS), "arms": summ_arms,
        "controls": {"arm_notes": notes, "route_command_is_a_route_level_oracle": NB._route_oracle_caveat(),
                     "floors_provenance": ({"status": "REUSED", "arms": adopted,
                                            "mandatory_floors_rule": ("SATISFIED by arms scored ON THE SAME TOKENS: each "
                                                                      "adopted arm PASSED in its source run under the identical "
                                                                      "devkit, patches, cache identity, inputs, runner and "
                                                                      "traffic policy; see raw/<arm>/<arm>.reused.json")}
                                           if adopted else {"status": "SCORED_IN_THIS_RUN", "arms": sorted(scored)}),
                     "model_scores_provenance": {"status": "SCORED_IN_THIS_RUN"},
                     "gallery_inputs": gallery_inputs,
                     "scoring_counts": {k: {kk: v.get(kk) for kk in ("status", "failures", "wall_s", "log_successful",
                                                                    "log_failed", "csv_valid_rows", "agent_calls",
                                                                    "runner_script", "traffic_agents")}
                                        for k, v in counts.items()}},
        "provenance": {"devkit": ctx.rec.get("devkit"), "split_yaml_counts": pre.get("yaml_counts"),
                       "ckpt": ctx.rec.get("ckpt"), "subset": subset,
                       "metric_cache": pre.get("cache"), "logs_dir": str(prof.logs_dir).replace(os.sep, "/"),
                       "harness": {"fix151": "post", "basis": "MEASURED: devkit 0a380a9 = post-#151 (E1: README changelog "
                                                             "2025/09/29 + pdm_score.py:194-219)"},
                       "promoted_from": ["W1 navsim_v2 suite (two-stage)", "W3 navsim_v1 navtest cache recipe"]},
    }
    ctx.write_summary(summary)
    if any(v.get("retryable") for v in counts.values()):
        ctx.rec["retryable"] = True
        ctx.rec["retryable_reason"] = ("the promoted wrapper's RAM guard aborted an arm (available memory below the "
                                       "floor while other jobs held the box) — nothing about the harness failed")
    ok_floors = all(f in scored for f in NAVSIM_FLOORS)
    ctx.rec["status"] = ("COMPLETE" if len(scored) == len(arms) else ("PARTIAL" if ok_floors else "FAILED"))
    ctx.rec["device_used_override"] = "none"
