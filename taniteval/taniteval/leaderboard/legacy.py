"""Convert pre-contract banked outputs into W1-schema ``summary.json`` records (taniteval.bench.summary/1).

Until W1's CLI has produced real run directories, the leaderboard reads E2's banked
``scores_summary.json`` THROUGH this converter, so there is exactly one reader path (the W1 one) and
the converted record is validated by W1's own ``schema_check`` before anything is rendered.
The converted record says so in ``provenance.not_produced_by_bench_cli``; nothing is written into
``taniteval/results/bench/`` (that tree belongs to real CLI runs).

Conversions are verbatim except: (a) E2's floor keys are renamed to the contract's canonical floor
names (``STOP_zero``→``STOP``, ``CV_official``→``CV``, ``ECHO_ha0_ext``→``ECHO``); (b) the paired
``headline_delta`` between two arms whose official combined EPDMS exists is the difference of the
two MEASURED values (the combined score has no per-scene paired statistic) — stated in provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CANON = {"STOP_zero": "STOP", "CV_official": "CV", "ECHO_ha0_ext": "ECHO"}


def _ckpt_obj(spec: dict) -> dict:
    """W1's checkpoint triple for a converted run, with the display string from ITS function.

    ⚠️ The banked E2 record identifies its checkpoint by **md5** only, and W1's contract requires a
    sha256 for any run with model arms. The sha256 here was MEASURED over the file on disk whose md5
    was RE-COMPUTED and matches the banked one exactly (`ckpt_sha256_provenance` records that chain),
    which is what licenses attaching it to E2's run — ⛔ an md5 is never relabelled as a sha256, and a
    digest is never copied from a report. `registry_key` is present, so the display is the key."""
    ck = dict(spec.get("ckpt_obj") or {})
    ck.setdefault("path", spec.get("ckpt_path"))
    ck.setdefault("sha256", spec.get("ckpt_sha256"))
    if spec.get("ckpt_sha256_provenance"):
        ck["sha256_provenance"] = spec["ckpt_sha256_provenance"]
    ck.setdefault("registry_key", spec.get("ckpt_registry_key"))
    if spec.get("ckpt_md5"):
        ck["md5"] = spec["ckpt_md5"]
    if spec.get("ckpt"):
        ck["as_recorded"] = spec["ckpt"]            # the operator's verbatim line, never parsed
    try:
        from taniteval.bench.contract import ckpt_display
    except Exception:                                # the contract is the only formatter; no local fallback
        raise
    ck["registry_key_display"] = ckpt_display(ck)
    return ck


def _devkit_obj(spec: dict) -> dict:
    """W1's devkit object. The harness line is kept VERBATIM under `as_recorded` rather than parsed
    into a patch count that nobody measured."""
    return {"repo": spec.get("devkit_repo", "autonomousvision/navsim"), "sha": spec.get("devkit_sha"),
            "patches": spec.get("devkit_patches") or [], "as_recorded": spec["harness"]}


_FAMILY_REASON = {
    "longitudinal": "EPDMS carries EP (ego progress) and TTC only — compliance outcomes, not our speed / along-track / "
                    "distance-keeping metrics; the EP and TTC sub-metrics are in `submetrics`",
    "lateral": "EPDMS carries LK (lane keeping) only; heading / curvature / yaw-rate are not computed on NAVSIM here",
    "tactical": "NC and TLC are OUTCOMES, not decision quality (CRITERIA_REGISTRY benchmarks.navsim.EPDMS_submetrics._family_caveat)",
    "strategic": "DAC and DDC are route-level COMPLIANCE, not goal-setting quality (same caveat)",
}


def e2_to_summary(root: Path, spec: dict, e1_controls: dict | None = None) -> dict:
    src = Path(root) / spec["path"]
    raw = src.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    A, pairs, est = d["arms"], d.get("pairs", {}), d.get("_estimator", {})
    keys = [a["key"] for a in spec["arms"]]
    floors = [CANON[k] for k in keys if k in CANON]
    n_tok = {k: (A[k].get("n_stage1") or 0) + (A[k].get("n_stage2") or 0) for k in keys}

    def official(k):
        return None if "_HYBRID_WARNING" in A[k] else A[k]["official_summary_rows"]["extended_pdm_score_combined"]

    arms = {}
    for a in spec["arms"]:
        k = a["key"]
        v = A[k]
        name = CANON.get(k, k)
        mf = Path(root) / spec["manifest_pattern"].format(arm=k)
        declared = json.loads(mf.read_text(encoding="utf-8")).get("declared_inputs", []) if mf.exists() else [a.get("declared", "")]
        off = official(k)
        # ⭐ W-25's discipline: COUNT THE STAND-INS FROM THE SEAM ARTIFACT'S OWN `source` COLUMN, never
        # from the producer's reported total. The reported total is a claim about the run; the column is
        # the run. (Both are printed when they agree, so a future disagreement is visible, not silent.)
        rws = json.loads(mf.read_text(encoding="utf-8")).get("rows") if mf.exists() else None
        n_si = sum(1 for r in rws if r.get("source") == "cv_standin") if isinstance(rws, list) else None
        n_rows = len(rws) if isinstance(rws, list) else None
        rep = (json.loads(mf.read_text(encoding="utf-8")).get("n_cv_standin_rows") if mf.exists() else None)
        cnt = ("" if n_si is None else
               f" {n_si} of {n_rows} scored rows came from the devkit CV stand-in — COUNTED FROM THE SEAM "
               f"ARTIFACT'S OWN `source` column" +
               (f", and the producer's reported `n_cv_standin_rows` agrees ({rep})" if rep == n_si else
                f"; ⛔ the producer REPORTED {rep} — they disagree, and the column wins") + ".")
        head = ({"value": off, "column": "score", "statistic": "EPDMS, official two-stage runner, combined", "n": n_tok[k], "x100": off * 100}
                if off is not None else
                {"status": "UNAVAILABLE", "n": n_tok[k],
                 "reason": "HYBRID — the official runner's stage 1 for this arm is a devkit-CV stand-in, so the "
                           "combined score is not this arm's (E2 _HYBRID_WARNING)." + cnt})
        paired = {}
        for fk in keys:
            if fk not in CANON:
                continue
            fname = CANON[fk]
            if fk == k:
                paired[fname] = {"status": "SELF"}
            elif off is not None and official(fk) is not None:
                paired[fname] = {"status": "OK", "headline_delta": off - official(fk), "n_common": min(n_tok[k], n_tok[fk]),
                                 "note": "difference of two MEASURED official combined scores; no per-scene paired statistic exists for it"}
            else:
                paired[fname] = {"status": "UNAVAILABLE", "n": n_tok[k],
                                 "reason": "headline UNAVAILABLE for one side (HYBRID official stage 1)"}
        s2p = {}
        for fk in keys:
            if fk in CANON and fk != k:
                p = pairs.get(f"{k}__minus__{fk}")
                if p is not None:
                    s2p[CANON[fk]] = {"delta": p["S2_EPDMS_u_delta"], "delta_x100": p["S2_EPDMS_u_delta"] * 100,
                                      "wins": p["wins"], "ties": p["ties"], "losses": p["losses"], "n": p["n_scenes"],
                                      "what": p.get("what", "")}
                    continue
                q = pairs.get(f"{fk}__minus__{k}")   # the same paired difference, banked the other way round
                s2p[CANON[fk]] = None if q is None else {
                    "delta": -q["S2_EPDMS_u_delta"], "delta_x100": -q["S2_EPDMS_u_delta"] * 100,
                    "wins": q["losses"], "ties": q["ties"], "losses": q["wins"], "n": q["n_scenes"],
                    "what": f"REVERSED from the banked pair {fk} − {k} (exact negation; wins and losses swapped)"}
        arms[name] = {
            "kind": {"floor": "floor", "control": "reference", "model": "model", "ablation": "model"}[a["kind"]],
            "role": a["kind"], "label": a["label"], "status": "OK",
            "declared_inputs": [x for x in declared if x] or ["none"],
            "headline": head,
            "per_stage": {"stage1": {"n": v.get("n_stage1"), "scene_mean": v.get("S1_scene_mean"),
                                     "official_stage_score": v["official_summary_rows"]["extended_pdm_score_stage_one"],
                                     "is_cv_standin": "_HYBRID_WARNING" in v},
                          "stage2": {"n": v.get("n_stage2"), "scene_mean": v.get("S2_scene_mean"),
                                     "official_stage_score": v["official_summary_rows"]["extended_pdm_score_stage_two"]}},
            "per_log": {"stage2_S2_EPDMS_u_by_log": v.get("S2_by_log", {})},
            "submetrics": {"stage1": v.get("S1_submetric_means", {}), "stage2": v.get("S2_submetric_means", {}),
                           "stage2_multiplier_zero_rates": v.get("S2_multiplier_zero_rates", {})},
            "paired": paired,
            "interval": {"status": "UNAVAILABLE", "n": int(est.get("n", 0)),
                         "reason": est.get("reason", "warmup never carries an interval")},
            "families": {f: {"status": "UNAVAILABLE", "n": int(v.get("n_stage2") or 0), "reason": r} for f, r in _FAMILY_REASON.items()},
            "files": {"source": spec["path"], "manifest": spec["manifest_pattern"].format(arm=k) if mf.exists() else None},
            "statistics": {"S2_EPDMS_u": {"value": v["S2_EPDMS_u"]["value"], "x100": v["S2_EPDMS_u"]["value"] * 100,
                                          "n": v["S2_EPDMS_u"].get("n_scenes"), "n_groups": v["S2_EPDMS_u"].get("n_groups"),
                                          "statistic": d.get("_primary", ""), "paired": s2p},
                           # W-25: keep the devkit's printed combined row beside S2-EPDMS-u, under a name
                           # that says it is HYBRID. Deleting it hides what the runner actually printed.
                           **({} if off is not None else {"official_combined_row_HYBRID": {
                               "value": v["official_summary_rows"]["extended_pdm_score_combined"],
                               "x100": v["official_summary_rows"]["extended_pdm_score_combined"] * 100,
                               "what": "the official runner's printed combined row for this arm — HYBRID "
                                       "(stage 1 is the devkit CV stand-in), so it is NOT this arm's EPDMS"}})},
        }
    out = {
        "schema": "taniteval.bench.summary/1",
        "run_id": spec["id"],
        "benchmark": "navsim_v2",
        "protocol": spec["protocol"],
        "split": "warmup_two_stage",
        "claim_bearing": False,
        "evidence_class": "MEASURED",
        "stamps": {"tier": "T1-family", "loop": {"stage1": "OPEN", "stage2": "UNRULED"}, "closed_loop": False,
                   "background_traffic": "IDM-reactive in both stages", "evidence_class": "MEASURED"},
        "headline_metric": {"name": "EPDMS", "column": "score", "higher_is_better": True,
                            "statistic": "official two-stage runner, combined — the HF warmup leaderboard's statistic"},
        "floors": floors,
        "arms": arms,
        "controls": {"k4_seam_transparency": d.get("K4_seam_transparency", {}), "bar_e2_1": d.get("BAR_E2_1", {})},
        "provenance": {"converted_by": "taniteval.leaderboard.legacy.e2_to_summary", "not_produced_by_bench_cli": True,
                       "source": spec["path"], "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "result_doc": spec.get("result_doc"),
                       # ⛔ W1's shapes, not free text: `ckpt` is the triple (+ the display string computed ONCE
                       # by `contract.ckpt_display`, never re-derived downstream) and `devkit` is an object.
                       # This converter is the PRODUCER of this summary, so formatting it here is the single
                       # formatting — a renderer that composed its own wording would be the second one.
                       "ckpt": _ckpt_obj(spec), "devkit": _devkit_obj(spec),
                       "registry_anchor": spec.get("registry_anchor"), "tier_loop": spec["tier_loop"]},
    }
    if e1_controls is not None:
        out["controls"]["hf_reproduction"] = e1_controls
    return out
