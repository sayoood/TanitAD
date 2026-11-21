"""FIXTURE ADAPTERS: E1 / E2 banked NavSim outputs -> a BUILD_PLAN §1 run directory (W1 schema v1).

W1 owns ``python -m taniteval.bench``; until its NavSim producer lands, the report is built and tested
against the programme's EXISTING real outputs:

* **E2** ``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw/`` —
  refcv4b zero-shot on ``warmup_two_stage`` (camera arms stage-2 only) + STOP / CV / ECHO floors.
* **E1** ``…/2026-09-19-navsim-warmup-reference-epdms/raw/`` — the CV reproduction of the HF warmup
  leaderboard (18.5356) and the human reference, which is UNDEFINED on a two-stage split.

Every number written here is COMPUTED from the devkit's own per-token CSVs (read verbatim, ``score``
column, never ``pdm_score``) and — for E2 — CROSS-CHECKED against E2's independently written
``raw/scores_summary.json`` (E2's ``parse_scores.py``): any field both compute must agree to
``XCHECK_TOL`` or the adapter REFUSES (raises). Nothing is retyped.

The outputs are FIXTURE conversions, not bench runs: ``bench_run.json`` and ``summary.json`` carry
``provenance.fixture = true`` and ``provenance.leaderboard_eligible = false`` so W4 never counts them
beside W1's canonical run of the same protocol.

    python -m taniteval.benchreport.adapt e2 <out_dir>      # E2 -> run dir
    python -m taniteval.benchreport.adapt e1 <out_dir>      # E1 -> run dir
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from .metrics import EPDMS_V2, SPEED_BANDS, TIE_TOL, band_of, mean, wtl

REPO = Path(__file__).resolve().parents[3]
EVAL_INCOMING = REPO / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming"
E2_RAW = EVAL_INCOMING / "2026-09-19-navsim-refcv4b-bridge" / "raw"
E1_RAW = EVAL_INCOMING / "2026-09-19-navsim-warmup-reference-epdms" / "raw"
E2_RESULT = E2_RAW.parent / "RESULT.md"
E1_RESULT = E1_RAW.parent / "RESULT.md"

XCHECK_TOL = 1e-12
SUMMARY_SCHEMA = "taniteval.bench.summary/1"
BENCH_RUN_SCHEMA = "taniteval.bench.bench_run/1"
PROTOCOL = "EPDMS_v2_warmup_two_stage"
DEVKIT_REPO = "autonomousvision/navsim"
DEVKIT_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
STAMPS = {
    "tier": "T1-family",
    "loop": {"stage_one": "OPEN (PI 2026-09-02)",
             "stage_two": "UNRULED (pre-rendered perturbed start)",
             "background_agents": "IDM-reactive in both stages (run_pdm_score.py:77-79, 132-134)"},
    "closed_loop": False,
    "evidence_class": "MEASURED",
}
DEVKIT_PATCHES = [
    {"name": "navsim/common/dataclasses.py PosixPath-safe unpickler (blob 596cb7d)", "kind": "pre-existing"},
    {"name": "venv fcntl.py flock shim (sha256 75184a48…)", "kind": "pre-existing"},
    {"name": "nuPlan setup.py (packaging only)", "kind": "pre-existing"},
    {"name": "E1 navsim_win.py --patch-loader: MetricCacheLoader path-separator fix", "kind": "in-process"},
]
HEADLINE_METRIC = {
    "name": "EPDMS",
    "column": "score",
    "higher_is_better": True,
    "statistic": ("official two-stage combined EPDMS = the devkit's `extended_pdm_score_combined` row "
                  "of the per-token `score` column (NAVSIM v2, warmup_two_stage)"),
}
S2U_DEF = ("S2-EPDMS-u (E2 SPEC §2): mean over the 16 stage-2 groups (8 reactive_all_mapping entries × "
           "{orig, prev} sides) of the uniform mean of the official per-scene `score` of the stage-2 "
           "tokens. ⛔ NOT a two-stage EPDMS — a stage-2-only statistic with uniform weights.")
WARMUP_INTERVAL = {"status": "UNAVAILABLE",
                   "reason": ("the settled estimator is the LOG-CLUSTER bootstrap (clusters = log_name, "
                              "B = 2000, n ≥ 8 clusters; W2 2026-09-19/20) and warmup_two_stage has only "
                              "7 log groups — below the floor. Point estimates and paired per-scene counts "
                              "only. ⚠️ A run that did not capture the devkit's pre-CSV frame (`weight`, "
                              "`log_name`, dropped at run_pdm_score.py:422) has no interval either."),
                   "n": 7,
                   "estimator_if_it_were_settled": "log_cluster_bootstrap(cluster=log_name, B=2000)"}
# ⛔ BINDING (W2): the NavSim driving command is a route-level ORACLE — carried in the run so the
# report's caveat is DATA, not only renderer prose (the renderer also derives it from declared_inputs).
COMMAND_ORACLE_CAVEAT = {
    "id": "navsim.driving_command_is_a_route_oracle",
    "applies_when": "the arm declares driving_command[t0]",
    "text": ("the NavSim driving_command is a ROUTE-LEVEL ORACLE (a function of ego pose + the nuPlan "
             "route + the map, reproduced 1,902/1,902 with OpenScene's own function); the route is the "
             "EXPERT'S driven path at roadblock granularity (79.8 % of forward blocks driven vs a 34.6 % "
             "null control), and on stage 2 command and route are COPIED from the expert's own frame "
             "(5,462/5,462 navhard) ⇒ an arm fed it is NOT evidence of route-following"),
    "source": "W2 (EvalFlyWheel, estimator-and-gates), 2026-09-19/20",
}


class AdapterError(RuntimeError):
    """The source files disagree with each other or with the contract — never guessed around."""


# ------------------------------------------------------------------------------------ small utils
def _rel(p: Path) -> str:
    """Repo-relative when the source lives in the repo, absolute otherwise (tests use a tmp copy)."""
    try:
        return Path(p).relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).as_posix()


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x: str) -> float:
    return float("nan") if x in ("", None) else float(x)


def read_devkit_csv(path: Path) -> tuple[dict, dict]:
    """-> (token rows {token: {col: value}}, summary rows {name: {col: value}}). Refuses ``pdm_score``."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        if "pdm_score" in cols:
            raise AdapterError(f"{path}: carries a pdm_score column — refusing (the EPDMS is `score`)")
        if "score" not in cols or "token" not in cols:
            raise AdapterError(f"{path}: no `token`/`score` column")
        toks, summ = {}, {}
        for r in rd:
            t = r["token"]
            vals = {c: (r[c] if c in ("token", "valid", "") else _f(r[c])) for c in cols if c}
            (summ if t.startswith("extended_pdm_score") else toks)[t] = vals
    return toks, summ


def _stage_view(row: dict, stage: int) -> dict:
    suf = "_stage_one" if stage == 1 else "_stage_two"
    out = {k: row.get(c + suf, float("nan")) for k, c in EPDMS_V2.columns.items()}
    out["score"] = row["score"]
    return out


def _nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _mean_or_none(xs):
    xs = [x for x in xs if not _nan(x)]
    return mean(xs)


# ------------------------------------------------------------------------------------ scene metadata
def load_scene_meta(e2_raw: Path = E2_RAW) -> tuple[dict, list]:
    """token -> {stage, log, v0, command, scene_token, map_name, pickle}; + the reactive_all_mapping.

    Source: E2's ``navsim_agent_inputs.json`` — the devkit's own ``AgentInput`` export
    (``SceneLoader.get_agent_input_from_token``), per scorer token."""
    doc = json.loads((e2_raw / "navsim_agent_inputs.json").read_text(encoding="utf-8"))
    meta = {}
    for t, r in doc["tokens"].items():
        ego = r["ego_statuses"][-1]
        vx, vy = ego["ego_velocity"]
        cmd = ego.get("driving_command") or []
        meta[t] = {"stage": int(r["stage"]), "log": r["log_name"], "v0": math.hypot(vx, vy),
                   "command_onehot": list(cmd),
                   "command": (["left", "straight", "right", "unknown"][cmd.index(max(cmd))]
                               if cmd and max(cmd) > 0 else "none"),
                   "scene_token": r.get("scene_token"), "map_name": r.get("map_name"),
                   "pickle": r.get("pickle"), "frame_type": r.get("frame_type"),
                   "num_future_frames": r.get("num_future_frames")}
    return meta, doc["reactive_all_mapping"]


def s2_group_uniform(s2_scores: dict, mapping: list) -> dict:
    """E2 SPEC §2's S2-EPDMS-u, replicated from ``parse_scores.py::s2_group_uniform``."""
    gm, sizes = [], []
    for _orig, _prev, pairs in mapping:
        for grp in ([p[0] for p in pairs], [p[1] for p in pairs]):
            vals = [s2_scores.get(t, float("nan")) for t in grp]
            if any(_nan(v) for v in vals):
                return {"status": "UNAVAILABLE", "reason": f"{sum(_nan(v) for v in vals)} group tokens "
                        f"missing/NaN", "n": sum(not _nan(v) for v in vals)}
            gm.append(math.fsum(vals) / len(vals))
            sizes.append(len(grp))
    return {"value": math.fsum(gm) / len(gm), "n_groups": len(gm), "group_sizes": sizes,
            "n_scenes": int(sum(sizes))}


# ------------------------------------------------------------------------------------ per-arm stats
def stage_block(rows: dict, tokens: list, stage: int) -> dict:
    """Uniform per-stage scene statistics over ``tokens`` (the arm's own rows)."""
    views = [_stage_view(rows[t], stage) for t in tokens]
    subs = {k: _mean_or_none(v[k] for v in views) for k in EPDMS_V2.submetrics}
    zero = {k: _mean_or_none((1.0 if v[k] == 0 else 0.0) for v in views if not _nan(v[k]))
            for k in EPDMS_V2.multipliers}
    partial = {k: _mean_or_none((1.0 if 0 < v[k] < 1 else 0.0) for v in views if not _nan(v[k]))
               for k in EPDMS_V2.multipliers}
    scores = [v["score"] for v in views]
    nz = sum(1 for s in scores if s == 0)
    return {"n": len(tokens), "aggregation": "uniform scene mean over this stage's tokens",
            "scene_mean": _mean_or_none(scores), "submetrics": subs,
            "zero_rates": zero, "partial_rates": partial,
            "n_zero_score": nz, "zero_score_rate": nz / len(tokens) if tokens else None}


def official_rows(summ: dict) -> dict:
    out = {}
    for name, r in summ.items():
        key = name.replace("extended_pdm_score_", "")
        stage = 1 if key == "stage_one" else 2 if key == "stage_two" else None
        subs = {}
        if stage is not None:
            subs = {k: r.get(c + ("_stage_one" if stage == 1 else "_stage_two"))
                    for k, c in EPDMS_V2.columns.items()}
            subs = {k: (None if _nan(v) else v) for k, v in subs.items()}
        out[key] = {"score": r["score"], "submetrics": subs}
    return out


def paired_block(a_rows: dict, f_rows: dict, tokens: list, meta: dict,
                 a_s2u: dict, f_s2u: dict) -> dict:
    """Arm − floor, per scene, on the given (stage-2) tokens."""
    d = [a_rows[t]["score"] - f_rows[t]["score"] for t in tokens]
    res = {"status": "OK", "scope": "stage_two", "n_common": len(tokens),
           "scene_mean_delta": mean(d), **{k: v for k, v in wtl(d).items() if k != "n"},
           "statistic_deltas": {"S2_EPDMS_u": (a_s2u["value"] - f_s2u["value"]
                                               if "value" in a_s2u and "value" in f_s2u else None)},
           "submetric_deltas": {}}
    for k in EPDMS_V2.submetrics:
        av = _mean_or_none(_stage_view(a_rows[t], 2)[k] for t in tokens)
        fv = _mean_or_none(_stage_view(f_rows[t], 2)[k] for t in tokens)
        res["submetric_deltas"][k] = None if av is None or fv is None else av - fv
    by_log, by_band = {}, {}
    for t, x in zip(tokens, d):
        by_log.setdefault(meta[t]["log"], []).append(x)
        by_band.setdefault(band_of(meta[t]["v0"]), []).append(x)
    res["by_log"] = {lg: {"n": len(xs), "scene_mean_delta": mean(xs),
                          **{k: v for k, v in wtl(xs).items() if k not in ("n", "tie_tol")}}
                     for lg, xs in sorted(by_log.items())}
    res["by_speed_band"] = {}
    for bid, lab, lo, hi in SPEED_BANDS:
        xs = by_band.get(bid, [])
        res["by_speed_band"][bid] = ({"n": 0, "status": "UNAVAILABLE", "reason": "no scene in band"}
                                     if not xs else
                                     {"n": len(xs), "scene_mean_delta": mean(xs),
                                      **{k: v for k, v in wtl(xs).items() if k not in ("n", "tie_tol")}})
    return res


def per_log_block(rows: dict, tokens: list, meta: dict) -> dict:
    by = {}
    for t in tokens:
        by.setdefault(meta[t]["log"], []).append(rows[t]["score"])
    return {"scope": "stage_two", "statistic": "uniform scene mean of `score`",
            "logs": {lg: {"n": len(v), "value": mean(v)} for lg, v in sorted(by.items())}}


def per_band_block(rows: dict, tokens: list, meta: dict) -> dict:
    by = {}
    for t in tokens:
        by.setdefault(band_of(meta[t]["v0"]), []).append(rows[t]["score"])
    bands = {}
    for bid, lab, lo, hi in SPEED_BANDS:
        v = by.get(bid, [])
        bands[bid] = {"label": lab, "lo": lo, "hi": (None if math.isinf(hi) else hi), "n": len(v),
                      "value": mean(v) if v else None}
    return {"scope": "stage_two", "statistic": "uniform scene mean of `score`",
            "v0_source": "|ego_velocity[t0]| from the devkit AgentInput export (navsim_agent_inputs.json)",
            "bands": bands}


# ------------------------------------------------------------------------------------ four families
_REFUSED = ("UNAVAILABLE", "REFUSED", "UNDEFINED")
_CONFIG_KEYS = {"n", "n_windows", "dt_s", "tier", "ci", "min_ds_m", "min_ds_mps", "excluded_below_min_ds",
                "n_bearing", "version", "classes", "class_order", "never_predicted"}
NO_FUTURE_S2 = ("stage-2 synthetic scenes carry no logged future (num_future_frames = 0, E1 MEASURED 204/204): "
                "no geometry family is computable there for ANY arm")


def _is_config(k) -> bool:
    k = str(k)
    return k.startswith("_") or k in _CONFIG_KEYS or k.startswith("n_") or k.startswith("min_")


def _has_value_leaf(block, top: bool = False) -> bool:
    """True if the block holds at least one computed metric value.

    ⚠️ At the FAMILY level the block's own ``status`` is NOT decisive: an artifact can mark a family
    UNAVAILABLE for its geometry part while carrying a computed readout inside it (E2's STRATEGIC
    nav-compliance, 0.926 on n 204). Below the top, a dict with a refused status holds no value."""
    if isinstance(block, bool):
        return False
    if isinstance(block, (int, float)):
        return not (isinstance(block, float) and math.isnan(block))
    if isinstance(block, dict):
        if not top and block.get("status") in _REFUSED:
            return False
        return any(_has_value_leaf(v) for k, v in block.items() if not _is_config(k))
    return False


def family_entry(scopes: dict, n_default: int) -> dict:
    """scopes: {scope_name: family block from the instrument}. -> W1-schema family {status, n, reason, scopes}."""
    ok = {s: b for s, b in scopes.items() if isinstance(b, dict) and _has_value_leaf(b, top=True)}
    refused = {s: b for s, b in scopes.items() if s not in ok}
    reasons = []
    for s, b in refused.items():
        r = (b or {}).get("reason") if isinstance(b, dict) else None
        reasons.append(f"[{s}] {r or 'no instrument output'}")
    for s, b in ok.items():
        inner = [k for k, v in b.items() if isinstance(v, dict) and v.get("status") in _REFUSED]
        if b.get("status") in _REFUSED or inner:
            reasons.append(f"[{s}] partial — refused leaves: {', '.join(inner) or '(family-level status)'}"
                           + (f"; family status {b.get('status')}: {b.get('reason')}"
                              if b.get("status") in _REFUSED else ""))
    if ok and not reasons:
        status = "OK"
    elif ok:
        status = "PARTIAL"
    else:
        status = "UNAVAILABLE"
    n = max([int(b.get("n") or b.get("n_windows") or 0) for b in ok.values() if isinstance(b, dict)]
            or [n_default])
    out = {"status": status, "n": n, "scopes": scopes}
    if status != "OK":
        out["reason"] = " | ".join(reasons) or "no instrument output"
    return out


def families_from(artifact: dict | None, stage1: dict | None, n_s2: int, why_absent_s1: str) -> dict:
    """Both stages, always: a stage with no instrument output is a named refusal, never an omission."""
    fams = {}
    for f in ("longitudinal", "lateral", "tactical", "strategic"):
        scopes = {}
        if artifact is not None:
            blk = dict(artifact.get("four_families", {}).get(f, {}))
            extra = artifact.get(f)          # e.g. artifact.longitudinal.distance_keeping
            if isinstance(extra, dict):
                blk["_artifact_extra"] = extra
            scopes["stage_two"] = blk
        else:
            scopes["stage_two"] = {"status": "UNAVAILABLE", "reason": NO_FUTURE_S2, "n": n_s2}
        if stage1 is not None:
            scopes["stage_one"] = stage1.get(f, {"status": "UNAVAILABLE", "reason": "absent from the instrument "
                                                 "output", "n": 0})
        else:
            scopes["stage_one"] = {"status": "UNAVAILABLE", "reason": why_absent_s1, "n": 0}
        fams[f] = family_entry(scopes, n_s2)
    return fams


# ------------------------------------------------------------------------------------ git / files
def _git_head() -> tuple[str, dict]:
    try:
        h = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=60).stdout.strip()
    except Exception as e:                                      # noqa: BLE001
        return "UNAVAILABLE", {"reason": f"{type(e).__name__}: {e}"}
    if len(h) == 40:
        return h, {"note": "HEAD at ADAPTER time (the source run's own HEAD was not recorded by E2)"}
    return "UNAVAILABLE", {"reason": f"git rev-parse returned {h!r}"}


def _files_manifest(run_dir: Path) -> dict:
    out = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and "report" not in p.relative_to(run_dir).parts[:1]:
            rel = p.relative_to(run_dir).as_posix()
            sz = p.stat().st_size
            out[rel] = {"bytes": sz, "sha256": sha256_of(p),
                        "location": "repo" if sz <= 20 * 1024 * 1024 else "offrepo"}
    return out


def _write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _clean_nan(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean_nan(v) for v in o]
    return o


# ------------------------------------------------------------------------------------ E2
E2_ARMS = (
    # (E2 source name, run arm name, kind, role, short label)
    ("A1_ego_cmd", "A1_ego_cmd", "model", "bar", "A1 · frames + ego t0 + cmd"),
    ("A2_vision_pure", "A2_vision_pure", "model", "ablation", "A2 · frames only"),
    ("A3_ego_nocmd", "A3_ego_nocmd", "model", "ablation", "A3 · frames + ego t0, no cmd"),
    ("A1NT_ego_cmd_nearest", "A1NT_ego_cmd_nearest", "model", "sensitivity", "A1NT · nearest-time history"),
    ("A2NT_vision_pure_nearest", "A2NT_vision_pure_nearest", "model", "sensitivity",
     "A2NT · nearest-time, frames only"),
    ("A4_blind_ego_cmd", "A4_blind_ego_cmd", "model", "diagnostic", "A4 · frames-blind (grey)"),
    ("CV_official", "CV", "floor", "floor", "CV · constant velocity (devkit)"),
    ("STOP_zero", "STOP", "floor", "floor", "STOP · all-zero plan"),
    ("ECHO_ha0_ext", "ECHO", "floor", "floor", "ECHO · constant a0, κ0 echo"),
)
E2_PAIR_NAMES = {"CV": "CV_official", "STOP": "STOP_zero", "ECHO": "ECHO_ha0_ext"}
FLOORS = ("STOP", "CV", "ECHO")


def _xcheck(label: str, ours, theirs, errs: list) -> None:
    if ours is None and theirs is None:
        return
    if ours is None or theirs is None or abs(float(ours) - float(theirs)) > XCHECK_TOL:
        errs.append(f"{label}: adapter {ours!r} vs E2 scores_summary {theirs!r}")


def build_e2_run(out_dir: Path, e2_raw: Path = E2_RAW, *, bank_dir: str | None = None,
                 synthetic_dir: str | None = None, copy_plans: bool = True) -> Path:
    """E2's raw outputs -> a §1 run directory at ``out_dir`` (created; must not exist or be empty)."""
    e2_raw = Path(e2_raw)
    meta, mapping = load_scene_meta(e2_raw)
    s1 = sorted(t for t, m in meta.items() if m["stage"] == 1)
    s2 = sorted(t for t, m in meta.items() if m["stage"] == 2)
    theirs = json.loads((e2_raw / "scores_summary.json").read_text(encoding="utf-8"))
    ff_stage1 = json.loads((e2_raw / "four_families_stage1.json").read_text(encoding="utf-8"))

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise AdapterError(f"{out_dir} exists and is not empty — refusing to overwrite a run dir")
    for sub in ("scores", "artifacts", "criteria", "plans"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows, summ = {}, {}
    for src, arm, *_ in E2_ARMS:
        p = e2_raw / f"score_{src}.csv"
        rows[arm], summ[arm] = read_devkit_csv(p)
        missing = set(meta) - set(rows[arm])
        if missing or set(rows[arm]) - set(meta):
            raise AdapterError(f"{p}: token set != the AgentInput export ({len(missing)} missing)")
        shutil.copyfile(p, out_dir / "scores" / f"{arm}.csv")

    # stand-in detection from the seam files (the source of truth for WHO answered each token)
    standin = {}
    for src, arm, *_ in E2_ARMS:
        seam = e2_raw / f"seam_{src}.npz"
        if arm == "CV":
            standin[arm] = []                                   # the devkit agent itself
            continue
        import numpy as np
        z = np.load(seam, allow_pickle=False)
        standin[arm] = sorted(str(t) for t, s in zip(z["token"], z["source"]) if str(s) == "cv_standin")
        if copy_plans:
            np.savez_compressed(out_dir / "plans" / f"{arm}.npz", token=z["token"], poses=z["poses"],
                                source=z["source"], sampling=z["sampling"], arm=np.array(arm),
                                seam_file=np.array(seam.name))
    if copy_plans:
        import numpy as np
        z = np.load(e2_raw / "seam_K4_seam_cv.npz", allow_pickle=False)
        np.savez_compressed(out_dir / "plans" / "CV.npz", token=z["token"], poses=z["poses"],
                            source=z["source"], sampling=z["sampling"], arm=np.array("CV"),
                            seam_file=np.array("seam_K4_seam_cv.npz"))

    s2u = {arm: s2_group_uniform({t: rows[arm][t]["score"] for t in s2}, mapping) for arm in rows}
    errs: list = []
    arms_out = {}
    for src, arm, kind, role, label in E2_ARMS:
        r = rows[arm]
        th = theirs["arms"][src]
        manifest = e2_raw / f"seam_{src}.manifest.json"
        man = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        declared = (man.get("declared_inputs") if man else None)
        if arm == "CV":
            declared = ["ego_velocity[t0]"]       # devkit constant_velocity_agent.py:34 reads only this
        elif arm == "STOP":
            declared = []
        hybrid = bool(standin[arm])
        if hybrid != ("_HYBRID_WARNING" in th):
            errs.append(f"{arm}: seam stand-in detection ({hybrid}) != E2 _HYBRID_WARNING ({'_HYBRID_WARNING' in th})")
        orows = official_rows(summ[arm])
        st2 = stage_block(r, s2, 2)
        if "stage_two" in orows:
            st2["official_row"] = {**orows["stage_two"],
                                   "aggregation": "devkit stage-2 row: Gaussian-kernel weights from the stage-1 "
                                                  "endpoints (NAVSIM v2 two-stage aggregation)",
                                   "admissible": not hybrid}
        if hybrid:
            st2["official_row"]["reason"] = ("HYBRID: its kernel weights come from the devkit CV stand-in's "
                                             "stage-1 endpoints, not this arm's")
            st1 = {"status": "UNAVAILABLE", "n": len(standin[arm]),
                   "reason": (f"STAND-IN: the {len(standin[arm])} stage-1 rows are the devkit "
                              "ConstantVelocityAgent (declared stand-in; 0/192 stage-1 camera jpgs on "
                              "the box) — not this arm (E2 COMMS D2)")}
        else:
            st1 = stage_block(r, s1, 1)
            if "stage_one" in orows:
                st1["official_row"] = {**orows["stage_one"], "aggregation": "devkit stage-1 row (uniform)",
                                       "admissible": True}
        comb = (orows.get("combined") or {}).get("score")
        if comb is None:
            # the devkit's aggregation produced no combined row (it can fail AFTER every token is scored)
            headline = {"status": "UNAVAILABLE", "n": len(meta),
                        "reason": ("the devkit wrote no `extended_pdm_score_combined` row for this arm — its "
                                   "two-stage aggregation did not complete. The per-token `score` rows exist "
                                   "and every per-stage / per-log / paired number below is computed from them.")}
        elif hybrid:
            headline = {"status": "UNAVAILABLE", "n": len(meta),
                        "reason": (f"HYBRID — the official two-stage EPDMS of this arm is not its own: "
                                   f"{len(standin[arm])}/{len(meta)} rows (all of stage 1) were answered by "
                                   f"the devkit CV stand-in, which also sets the stage-2 kernel weights. The "
                                   f"devkit's row is kept in `statistics.official_rows` and is NOT reported as "
                                   f"this arm's EPDMS (E2 RESULT §2).")}
        else:
            headline = {"value": comb, "x100": comb * 100, "unit": "fraction", "column": "score",
                        "statistic": "EPDMS_two_stage_combined", "n": len(meta)}
        stats = {"S2_EPDMS_u": {**s2u[arm], "column": "score", "scope": "stage_two", "definition": S2U_DEF},
                 "official_rows": orows}
        if comb is not None:
            stats["EPDMS_two_stage_combined"] = ({"value": comb, "admissible": True, "column": "score"}
                                                 if not hybrid else
                                                 {"value": comb, "admissible": False,
                                                  "reason": headline["reason"], "column": "score"})
        # ---- cross-check against E2's independently computed summary
        _xcheck(f"{arm}.S2_EPDMS_u", s2u[arm].get("value"), th["S2_EPDMS_u"].get("value"), errs)
        _xcheck(f"{arm}.S2_scene_mean", st2["scene_mean"], th["S2_scene_mean"], errs)
        for k in EPDMS_V2.submetrics:
            _xcheck(f"{arm}.S2.{k}", st2["submetrics"][k], th["S2_submetric_means"][k], errs)
        for k in EPDMS_V2.multipliers:
            _xcheck(f"{arm}.S2.zero.{k}", st2["zero_rates"][k], th["S2_multiplier_zero_rates"][k], errs)
        plog = per_log_block(r, s2, meta)
        for lg, v in plog["logs"].items():
            _xcheck(f"{arm}.S2_by_log.{lg}", v["value"], th["S2_by_log"].get(lg), errs)
        for k, v in th["official_summary_rows"].items():
            _xcheck(f"{arm}.{k}", (orows.get(k.replace("extended_pdm_score_", "")) or {}).get("score"), v, errs)
        if not hybrid:
            _xcheck(f"{arm}.S1_scene_mean", st1["scene_mean"], th["S1_scene_mean"], errs)
            for k in EPDMS_V2.submetrics:
                _xcheck(f"{arm}.S1.{k}", st1["submetrics"][k], th["S1_submetric_means"][k], errs)

        # ---- four families (instruments E2 ran; everything else refused with its reason)
        art_p = e2_raw / f"artifact_{src}.json"
        art = json.loads(art_p.read_text(encoding="utf-8")) if art_p.exists() else None
        why_s1 = (st1["reason"] if hybrid else
                  f"four_families not run for {arm} on stage 1 in E2 (E2 ran our instruments on stage 1 for "
                  f"CV and A4 only, raw/four_families_stage1.json) — a WORK ITEM, not a pass")
        fam = families_from(art, ff_stage1.get(src), len(s2), why_absent_s1=why_s1)
        files = {"scores": f"scores/{arm}.csv"}
        if art is not None:
            shutil.copyfile(art_p, out_dir / "artifacts" / f"{arm}.json")
            files["artifact"] = f"artifacts/{arm}.json"
        crit = e2_raw / f"criteria_check_{src}.txt"
        if crit.exists():
            shutil.copyfile(crit, out_dir / "criteria" / f"{arm}.txt")
            files["criteria"] = f"criteria/{arm}.txt"
        if copy_plans:
            files["plans"] = f"plans/{arm}.npz"
        arms_out[arm] = {
            "kind": kind, "role": role, "label": label, "status": "OK",
            "declared_inputs": list(declared or []),
            "caveats": ([dict(COMMAND_ORACLE_CAVEAT)]
                        if any("driving_command" in d for d in (declared or [])) else []),
            "meaning": (man.get("spec") or {}).get("meaning") if man else None,
            "hybrid": hybrid,
            "headline": headline,
            "statistics": stats,
            "per_stage": {"stage_one": st1, "stage_two": st2},
            "submetrics": {"scope": "stage_two", "aggregation": "uniform scene mean",
                           "values": st2["submetrics"], "zero_rates": st2["zero_rates"]},
            "per_log": plog,
            "per_speed_band": per_band_block(r, s2, meta),
            "paired": {},
            "interval": dict(WARMUP_INTERVAL),
            "families": fam,
            "files": files,
            "provenance": {"source_csv": f"{_rel(e2_raw)}/score_{src}.csv",
                           "source_arm_name": src},
        }
    # ---- paired vs every floor (W1 cross-field rule: every arm is paired against every floor)
    for src, arm, *_ in E2_ARMS:
        for fl in FLOORS:
            if fl == arm:
                arms_out[arm]["paired"][fl] = {"status": "SELF"}
                continue
            pb = paired_block(rows[arm], rows[fl], s2, meta, s2u[arm], s2u[fl])
            ha, hf = arms_out[arm]["headline"], arms_out[fl]["headline"]
            if "value" in ha and "value" in hf:
                pb["headline_delta"] = ha["value"] - hf["value"]
            else:
                pb["headline_delta"] = None
                pb["headline_delta_reason"] = "headline UNAVAILABLE for " + (
                    arm if "value" not in ha else fl) + " (HYBRID)"
            arms_out[arm]["paired"][fl] = pb
            key = f"{src}__minus__{E2_PAIR_NAMES[fl]}"
            tp = theirs["pairs"].get(key)
            if tp is not None:
                for k in ("wins", "ties", "losses"):
                    if pb[k] != tp[k]:
                        errs.append(f"pair {key}.{k}: adapter {pb[k]} vs E2 {tp[k]}")
                _xcheck(f"pair {key}.S2_EPDMS_u_delta", pb["statistic_deltas"]["S2_EPDMS_u"],
                        tp["S2_EPDMS_u_delta"], errs)
                _xcheck(f"pair {key}.scene_mean_delta", pb["scene_mean_delta"], tp["scene_mean_delta"], errs)
                for k in EPDMS_V2.submetrics:
                    _xcheck(f"pair {key}.{k}", pb["submetric_deltas"][k], tp["submetric_mean_deltas"][k], errs)
    if errs:
        raise AdapterError("E2 cross-check FAILED (adapter vs E2's parse_scores.py):\n  " + "\n  ".join(errs))

    git_head, git_info = _git_head()
    utc = "2026-09-19T11:36:00Z"        # E2's last scoring write (score_STOP_zero.csv 13:35 Berlin)
    run_id = "20260919T113600Z-navsim_v2-refcv4b_b1_v72_40k-" + hashlib.sha256(
        (e2_raw / "scores_summary.json").read_bytes()).hexdigest()[:6]
    model_arm = json.loads((e2_raw / "seam_A1_ego_cmd.manifest.json").read_text(encoding="utf-8"))["model"]
    scenes = {t: {k: m[k] for k in ("stage", "log", "v0", "command", "command_onehot", "scene_token",
                                    "map_name", "pickle", "frame_type", "num_future_frames")}
              for t, m in sorted(meta.items())}
    scenes_doc = {"source": f"{_rel(e2_raw)}/navsim_agent_inputs.json",
                  "what": "per scorer token: stage, log, |v0| (m/s), NavSim command at t0, scene/pickle ids",
                  "synthetic_scene_pickles": synthetic_dir or
                  "C:/Users/Admin/navsim-crun/data/openscene/warmup_two_stage/synthetic_scene_pickles",
                  "frame_bank": bank_dir or "C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus",
                  "maps_root": "C:/Users/Admin/navsim-crun/data/maps",
                  "tokens": scenes}
    _write_json(out_dir / "scenes.json", scenes_doc)
    provenance = {"adapter": "taniteval.benchreport.adapt.build_e2_run (W5)", "fixture": True,
                  "leaderboard_eligible": False,
                  "why_not_eligible": "a W5 FIXTURE conversion of E2's banked outputs; W1's bench run of the same "
                                      "protocol is canonical",
                  "source_dir": _rel(e2_raw),
                  "source_result": _rel(E2_RESULT),
                  "xcheck": {"against": "raw/scores_summary.json (E2 parse_scores.py)", "tolerance": XCHECK_TOL,
                             "status": "PASS"},
                  "source_sha256": {"scores_summary.json": sha256_of(e2_raw / "scores_summary.json"),
                                    "navsim_agent_inputs.json": sha256_of(e2_raw / "navsim_agent_inputs.json")}}
    summary = {
        "schema": SUMMARY_SCHEMA, "run_id": run_id, "benchmark": "navsim_v2", "protocol": PROTOCOL,
        "split": "warmup_two_stage", "claim_bearing": True, "evidence_class": "MEASURED",
        "stamps": dict(STAMPS), "headline_metric": dict(HEADLINE_METRIC),
        "floors": list(FLOORS), "primary_arm": "A1_ego_cmd",
        "statistics": {"S2_EPDMS_u": {"label": "S2-EPDMS-u", "definition": S2U_DEF, "official": False,
                                      "higher_is_better": True},
                       "EPDMS_two_stage_combined": {"label": "EPDMS (official two-stage combined)",
                                                    "definition": HEADLINE_METRIC["statistic"],
                                                    "official": True, "higher_is_better": True}},
        "estimator": {"interval": dict(WARMUP_INTERVAL), "paired_counts": {"tie_tol": TIE_TOL,
                      "rule": "per-scene arm − floor on the stage-2 tokens; |d| ≤ tie_tol is a tie"}},
        "arms": arms_out,
        "controls": {"K4_seam_transparency": theirs.get("K4_seam_transparency"),
                     "BAR_E2_1": theirs.get("BAR_E2_1"),
                     "E2_controls_table": "RESULT.md §4 (K0–K9, KB, KC, KF, KX)"},
        "report": {"status": "REPORT_PENDING"},
        "provenance": provenance,
    }
    summary = _clean_nan(summary)
    _write_json(out_dir / "summary.json", summary)
    bench = {
        "schema": BENCH_RUN_SCHEMA, "run_id": run_id, "utc": utc, "utc_end": utc,
        "git_head": git_head, "git": git_info,
        "command": ["python", "-m", "taniteval.benchreport.adapt", "e2", str(out_dir)],
        "ckpt": {"path": model_arm["ckpt"], "sha256": None, "registry_key": "refcv4b-b1-v72-40k",
                 "sha256_status": "UNAVAILABLE — E2 recorded md5 only", "md5": model_arm["ckpt_md5"],
                 "step": model_arm.get("step")},
        "benchmark": "navsim_v2", "protocol": PROTOCOL,
        "devkit": {"repo": DEVKIT_REPO, "sha": DEVKIT_SHA, "patches": DEVKIT_PATCHES},
        "split": {"name": "warmup_two_stage", "n_scenes": len(meta), "n_logs": len({m["log"] for m in meta.values()}),
                  "n_stage_one": len(s1), "n_stage_two": len(s2)},
        "arms": [{"name": arm, "kind": kind, "declared_inputs": arms_out[arm]["declared_inputs"], "status": "OK"}
                 for _s, arm, kind, *_r in E2_ARMS],
        "device": {"requested": "cpu", "used": "cpu"},
        "wall_s": 0.0, "wall_s_note": "adapter conversion; E2's own walls are in raw/score_*.counts.json",
        "claim_bearing": True, "status": "PARTIAL",
        "status_reason": "camera arms are stage-2 only (no stage-1 camera frames on the box): their official "
                         "two-stage EPDMS is HYBRID and UNAVAILABLE",
        "stamps": dict(STAMPS), "report": {"status": "REPORT_PENDING"}, "provenance": provenance,
    }
    _write_json(out_dir / "bench_run.json", bench)
    bench["files"] = _files_manifest(out_dir)
    _write_json(out_dir / "bench_run.json", bench)
    return out_dir


# ------------------------------------------------------------------------------------ E1
def build_e1_run(out_dir: Path, e1_raw: Path = E1_RAW, e2_raw: Path = E2_RAW) -> Path:
    """E1 -> run dir: CV (official two-stage) + human (reference, UNDEFINED on two-stage) + STOP REFUSED.

    E1 predates the STOP-floor rule, so its run has NO STOP arm. The schema requires one; it is emitted
    as ``status: REFUSED`` with the reason, so the report bannerizes the run INCOMPLETE instead of
    silently passing it. Scene metadata (token -> stage / log / v0) comes from E2's AgentInput export
    of the SAME split (E2 control KX: E1's and E2's CV CSVs identical over 223 rows, max |Δ| 0.0)."""
    e1_raw = Path(e1_raw)
    meta, mapping = load_scene_meta(e2_raw)
    s1 = sorted(t for t, m in meta.items() if m["stage"] == 1)
    s2 = sorted(t for t, m in meta.items() if m["stage"] == 2)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise AdapterError(f"{out_dir} exists and is not empty — refusing to overwrite a run dir")
    for sub in ("scores", "artifacts", "criteria"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    csvs = sorted((e1_raw / "A1").glob("devkit_*.csv"))
    if len(csvs) != 1:
        raise AdapterError(f"E1 A1: expected exactly one devkit CSV, found {len(csvs)}")
    cv_rows, cv_summ = read_devkit_csv(csvs[0])
    if set(cv_rows) != set(meta):
        raise AdapterError("E1 CV CSV token set != the split's token set")
    shutil.copyfile(csvs[0], out_dir / "scores" / "CV.csv")
    orows = official_rows(cv_summ)
    comb = orows["combined"]["score"]
    s2u = s2_group_uniform({t: cv_rows[t]["score"] for t in s2}, mapping)
    st1 = stage_block(cv_rows, s1, 1)
    st1["official_row"] = {**orows["stage_one"], "aggregation": "devkit stage-1 row (uniform)", "admissible": True}
    st2 = stage_block(cv_rows, s2, 2)
    st2["official_row"] = {**orows["stage_two"], "aggregation": "devkit stage-2 row: Gaussian-kernel weights "
                           "from the stage-1 endpoints", "admissible": True}
    art_cv = json.loads((e1_raw / "artifacts" / "A1_constant_velocity_agent_warmup_two_stage.json"
                         ).read_text(encoding="utf-8"))
    art_h = json.loads((e1_raw / "artifacts" / "A2_human_agent_warmup_two_stage.json").read_text(encoding="utf-8"))
    shutil.copyfile(e1_raw / "artifacts" / "A1_constant_velocity_agent_warmup_two_stage.json",
                    out_dir / "artifacts" / "CV.json")
    shutil.copyfile(e1_raw / "artifacts" / "A2_human_agent_warmup_two_stage.json",
                    out_dir / "artifacts" / "HUMAN.json")
    shutil.copyfile(e1_raw / "artifacts" / "A1_criteria_check.txt", out_dir / "criteria" / "CV.txt")
    shutil.copyfile(e1_raw / "artifacts" / "A2_criteria_check.txt", out_dir / "criteria" / "HUMAN.txt")

    def fam_from_artifact(art, scope):
        fams = {}
        for f in ("longitudinal", "lateral", "tactical", "strategic"):
            fams[f] = family_entry({scope: art.get("four_families", {}).get(f, {})}, len(s1))
        return fams

    human_reason = art_h["benchmark"]["navsim"]["score"]["reason"]
    stop_reason = ("NOT RUN by E1: the STOP-floor rule postdates E1 (E2 integration ask 5, 2026-09-19). E2 "
                   "measured it on the identical harness: 2026-09-19-navsim-refcv4b-bridge/raw/score_STOP_zero.csv")
    arms = {
        "CV": {"kind": "floor", "role": "floor", "label": "CV · constant velocity (devkit)", "status": "OK",
               "declared_inputs": ["ego_velocity[t0]"],
               "headline": {"value": comb, "x100": comb * 100, "unit": "fraction", "column": "score",
                            "statistic": "EPDMS_two_stage_combined", "n": len(meta)},
               "statistics": {"S2_EPDMS_u": {**s2u, "column": "score", "scope": "stage_two", "definition": S2U_DEF},
                              "EPDMS_two_stage_combined": {"value": comb, "admissible": True, "column": "score"},
                              "official_rows": orows},
               "per_stage": {"stage_one": st1, "stage_two": st2},
               "submetrics": {"scope": "stage_two", "aggregation": "uniform scene mean",
                              "values": st2["submetrics"], "zero_rates": st2["zero_rates"]},
               "per_log": per_log_block(cv_rows, s2, meta),
               "per_speed_band": per_band_block(cv_rows, s2, meta),
               "paired": {"CV": {"status": "SELF"},
                          "STOP": {"status": "UNAVAILABLE", "reason": "the STOP floor was not run in E1", "n": 0}},
               "interval": dict(WARMUP_INTERVAL),
               "families": fam_from_artifact(art_cv, "stage_one"),
               "files": {"scores": "scores/CV.csv", "artifact": "artifacts/CV.json", "criteria": "criteria/CV.txt"},
               "provenance": {"source_csv": f"{_rel(csvs[0])}"}},
        "HUMAN": {"kind": "reference", "role": "reference", "label": "Human · log replay (devkit HumanAgent)",
                  "status": "FAILED", "declared_inputs": ["privileged: logged future trajectory"],
                  "headline": {"status": "UNAVAILABLE", "reason": human_reason, "n": len(s2)},
                  "per_stage": {"stage_one": {"status": "UNAVAILABLE", "n": len(s1),
                                              "reason": ("scored in-process by the two-stage runner but never "
                                                         "written (runner exit 1); the one-stage runner's stage-1 "
                                                         "human EPDMS 0.951255 (E1 A2b) is a DIFFERENT statistic "
                                                         "and lives in controls")},
                                "stage_two": {"status": "UNAVAILABLE", "n": len(s2),
                                              "reason": "UNDEFINED: synthetic scenes carry num_future_frames = 0"}},
                  "submetrics": {"status": "UNAVAILABLE", "reason": "no CSV (runner exit 1)", "n": 0},
                  "per_log": {"status": "UNAVAILABLE", "reason": "no CSV (runner exit 1)", "n": 0},
                  "paired": {"CV": {"status": "UNAVAILABLE", "reason": "no CSV (runner exit 1)", "n": 0},
                             "STOP": {"status": "UNAVAILABLE", "reason": "no CSV and no STOP floor", "n": 0}},
                  "interval": dict(WARMUP_INTERVAL),
                  "families": {f: {"status": "UNAVAILABLE", "n": 0,
                                   "reason": "the human IS the ground truth these instruments score against "
                                             "(pred = GT); a reference, not a result (E1 RESULT §5)"}
                               for f in ("longitudinal", "lateral", "tactical", "strategic")},
                  "files": {"artifact": "artifacts/HUMAN.json", "criteria": "criteria/HUMAN.txt"}},
        "STOP": {"kind": "floor", "role": "floor", "label": "STOP · all-zero plan", "status": "REFUSED",
                 "declared_inputs": [],
                 "headline": {"status": "UNAVAILABLE", "reason": stop_reason, "n": 0},
                 "per_stage": {"status": "UNAVAILABLE", "reason": stop_reason, "n": 0},
                 "submetrics": {"status": "UNAVAILABLE", "reason": stop_reason, "n": 0},
                 "per_log": {"status": "UNAVAILABLE", "reason": stop_reason, "n": 0},
                 "paired": {"STOP": {"status": "SELF"},
                            "CV": {"status": "UNAVAILABLE", "reason": "STOP not run in E1", "n": 0}},
                 "interval": dict(WARMUP_INTERVAL),
                 "families": {f: {"status": "UNAVAILABLE", "n": 0, "reason": stop_reason}
                              for f in ("longitudinal", "lateral", "tactical", "strategic")},
                 "files": {}},
    }
    controls = json.loads((e1_raw / "controls.json").read_text(encoding="utf-8"))
    stage1_one_stage = (e1_raw / "stage_one_summary.txt").read_text(encoding="utf-8")
    git_head, git_info = _git_head()
    run_id = "20260919T102420Z-navsim_v2-none-" + hashlib.sha256(csvs[0].read_bytes()).hexdigest()[:6]
    provenance = {"adapter": "taniteval.benchreport.adapt.build_e1_run (W5)", "fixture": True,
                  "leaderboard_eligible": False,
                  "why_not_eligible": "a W5 FIXTURE conversion of E1's banked outputs",
                  "source_dir": _rel(e1_raw),
                  "source_result": _rel(E1_RESULT),
                  "scene_meta_source": "E2 raw/navsim_agent_inputs.json (same split; E2 KX identical CV CSVs)"}
    summary = _clean_nan({
        "schema": SUMMARY_SCHEMA, "run_id": run_id, "benchmark": "navsim_v2", "protocol": PROTOCOL,
        "split": "warmup_two_stage", "claim_bearing": True, "evidence_class": "MEASURED",
        "stamps": dict(STAMPS), "headline_metric": dict(HEADLINE_METRIC), "floors": ["STOP", "CV"],
        "primary_arm": "CV",
        "statistics": {"S2_EPDMS_u": {"label": "S2-EPDMS-u", "definition": S2U_DEF, "official": False,
                                      "higher_is_better": True},
                       "EPDMS_two_stage_combined": {"label": "EPDMS (official two-stage combined)",
                                                    "definition": HEADLINE_METRIC["statistic"],
                                                    "official": True, "higher_is_better": True}},
        "estimator": {"interval": dict(WARMUP_INTERVAL)},
        "arms": arms,
        "controls": {"E1_controls_json": controls, "stage_one_one_stage_runner": stage1_one_stage},
        "report": {"status": "REPORT_PENDING"}, "provenance": provenance})
    _write_json(out_dir / "summary.json", summary)
    bench = {
        "schema": BENCH_RUN_SCHEMA, "run_id": run_id, "utc": "2026-09-19T10:24:20Z", "utc_end": None,
        "git_head": git_head, "git": git_info,
        "command": ["python", "-m", "taniteval.benchreport.adapt", "e1", str(out_dir)],
        "ckpt": {"path": None, "sha256": None, "registry_key": None},
        "benchmark": "navsim_v2", "protocol": PROTOCOL,
        "devkit": {"repo": DEVKIT_REPO, "sha": DEVKIT_SHA, "patches": DEVKIT_PATCHES},
        "split": {"name": "warmup_two_stage", "n_scenes": len(meta),
                  "n_logs": len({m["log"] for m in meta.values()})},
        "arms": [{"name": "CV", "kind": "floor", "declared_inputs": ["ego_velocity[t0]"], "status": "OK"},
                 {"name": "HUMAN", "kind": "reference", "declared_inputs": ["privileged: logged future trajectory"],
                  "status": "FAILED"},
                 {"name": "STOP", "kind": "floor", "declared_inputs": [], "status": "SKIPPED"}],
        "device": {"requested": "cpu", "used": "cpu"}, "wall_s": 534.6,
        "claim_bearing": True, "status": "PARTIAL",
        "status_reason": "human UNDEFINED on a two-stage split; STOP floor not run (predates the rule)",
        "stamps": dict(STAMPS), "report": {"status": "REPORT_PENDING"}, "provenance": provenance}
    _write_json(out_dir / "bench_run.json", bench)
    bench["files"] = _files_manifest(out_dir)
    _write_json(out_dir / "bench_run.json", bench)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m taniteval.benchreport.adapt", description=__doc__.split("\n\n")[0])
    ap.add_argument("which", choices=("e1", "e2"))
    ap.add_argument("out_dir")
    ap.add_argument("--no-plans", action="store_true", help="E2: do not copy the per-arm plans (seam poses)")
    a = ap.parse_args(argv)
    if a.which == "e2":
        p = build_e2_run(Path(a.out_dir), copy_plans=not a.no_plans)
    else:
        p = build_e1_run(Path(a.out_dir))
    print(f"[adapt] wrote {p} at {_dt.datetime.now(_dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
