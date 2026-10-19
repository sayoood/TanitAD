"""summarize_g0_r8.py -- a SLIM, bankable summary of a refcv8 G0 artifact (the full g0.json is ~10 MB and stays on E:).

    python summarize_g0_r8.py --g0 E:/refcv8_eval_kit/battery/step5000/g0.json --tag step5000 \
        --out-json <package>/raw/step5000/G0_SUMMARY_step5000.json --out-md <package>/raw/step5000/G0_SUMMARY_step5000.md

Reads the artifact (or `<stem>.partial.json` when the run did not finish), never recomputes a verdict: every number is
copied from the judge's own output. Tier: T1-adjacent -- this is a REPRODUCTION gate on the run's own in-run eval row, not
a driving metric. Evidence class of each number: MEASURED (the g0.json named in the output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADLINE = ("eval_loss", "eval_traj", "eval_tacv6_goal_conf_bce", "eval_r8", "eval_r8_total", "eval_r8_cons",
            "eval_r8_alloc_l1", "eval_r8_sat", "eval_r8_subscore", "eval_r8_v9_cons", "eval_r8_v9_mae_speed",
            "eval_r8_v9_mae_lat_c", "eval_r8_v9_mae_lon_c")


def _g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summarize(g0_path: Path) -> dict:
    p = Path(g0_path)
    used = p
    if not p.exists() and p.with_suffix(".partial.json").exists():
        used = p.with_suffix(".partial.json")
    g = json.load(open(used, encoding="utf-8"))
    rp = p.with_suffix(".r8_repoint.json")
    rep = json.load(open(rp, encoding="utf-8")) if rp.exists() else {}
    v = g.get("verdict") or {}
    terms = v.get("terms") or {}
    out = {
        "artifact": str(used), "finished": bool(g.get("finished")), "partial_only": used != p,
        "tool": g.get("tool"), "ckpt": g.get("ckpt"), "ckpt_md5": g.get("ckpt_md5"), "config_md5": g.get("config_md5"),
        "step": g.get("step"), "n_seeds": len(g.get("by_seed") or {}), "seeds": g.get("seeds"),
        "started": g.get("started"), "finished_at": g.get("finished"), "wall_s": g.get("wall_s"),
        "gpu": g.get("gpu"), "torch": g.get("torch"), "tanitad_file": g.get("tanitad_file"),
        "cuda_max_memory_allocated_gib": g.get("cuda_max_memory_allocated_gib"),
        "state_dict": {k: (len(x) if isinstance(x, list) else x) for k, x in (_g(g, "model", "state_dict") or {}).items()
                       if k in ("strict", "missing", "unexpected", "n_keys", "step")},
        "param_breakdown_equal": _g(g, "model", "param_breakdown", "equal"),
        "n_eval_windows": _g(g, "dataset", "n_windows"), "n_eval_episodes": _g(g, "dataset", "n_episodes"),
        "perm_sha256": g.get("perm_sha256"),
        "GATE_VERDICT": v.get("G0"), "gate_amendment": v.get("amendment"),
        "gate_registration": v.get("registration"),
        "verdicts_by_rule": {
            "as_registered": _g(g, "verdict_as_registered", "G0"), "A2": _g(g, "verdict_A2", "G0"),
            "A5": _g(g, "verdict_A5", "G0"), "A6": _g(g, "verdict_A6", "G0"), "A7": _g(g, "verdict_A7", "G0")},
        "reasons_gate_first40": (v.get("reasons") or [])[:40], "n_reasons_gate": len(v.get("reasons") or []),
        "reasons_as_registered_first10": (_g(g, "verdict_as_registered", "reasons") or [])[:10],
        "n_terms": v.get("n_terms"), "by_class_counts": v.get("by_class_counts"), "medians": v.get("medians"),
        "mutation_detection": {m: {k: d.get(k) for k in ("detected", "n_terms_out", "how")}
                               for m, d in (v.get("mutation_detection") or {}).items()},
        "M1_detected": _g(v, "mutation_detection", "m1", "detected"),
        "blind_spots_named": v.get("blind_spots_named"),
        "wrapper_clause": _g(g, "wrapper_control", "clause"),
        "requires_grad_control": {k: _g(g, "requires_grad_control", k) for k in ("n_terms", "n_differ", "max_rel_all")},
        "a7_keys": sorted((g.get("a7") or {}).keys()),
        "seed_group_report": _g(g, "a7", "seed_group"),
        "pose_sync_patch": {"enabled": rep.get("pose_sync_patch_enabled"), "stats": rep.get("pose_sync_stats"),
                            "remap_overrides": rep.get("remap_overrides"), "loader_file": rep.get("loader_file"),
                            "loader_KIT": rep.get("loader_KIT"), "launch_tree": rep.get("launch_tree"),
                            "wrapper_status": rep.get("g0_wrapper_status")},
        "headline_terms": {},
    }
    for k in HEADLINE:
        t = terms.get(k)
        if t:
            out["headline_terms"][k] = {x: t.get(x) for x in ("inrun", "mean", "sd", "cls", "class", "verdict", "tol",
                                                               "pi_lo", "pi_hi", "dev")}
    # the terms that are OUT, by class (names only; the count is the signal)
    outs = {}
    for k, t in terms.items():
        if t.get("verdict") not in ("OK", "EXCLUDED", "DERIVED"):
            outs.setdefault(t.get("cls") or t.get("class") or "?", []).append(k)
    out["terms_not_ok_by_class_first15"] = {c: ks[:15] for c, ks in outs.items()}
    out["n_terms_not_ok_by_class"] = {c: len(ks) for c, ks in outs.items()}
    return out


def to_md(s: dict, tag: str) -> str:
    L = [f"## refcv8 G0 -- {tag}  (EARLY READ at step {s.get('step')} of 48,852; never a bar verdict)", "",
         f"* **Gate verdict: {s.get('GATE_VERDICT')}** under amendment `{s.get('gate_amendment')}` "
         f"(registration: {json.dumps(s.get('gate_registration'), default=str)[:200]}); n_seeds {s.get('n_seeds')}; "
         f"finished {s.get('finished')}{' (PARTIAL artifact only)' if s.get('partial_only') else ''}.",
         f"* Verdicts by rule: {json.dumps(s.get('verdicts_by_rule'))}.",
         f"* Mutations (M1 must be detected, else G0 is VOID): "
         f"{json.dumps(s.get('mutation_detection'))}; M1 detected = **{s.get('M1_detected')}**.",
         f"* Checkpoint md5 `{s.get('ckpt_md5')}`; config md5 `{s.get('config_md5')}`; step {s.get('step')}; "
         f"strict load: {json.dumps(s.get('state_dict'))}; param_breakdown equal: {s.get('param_breakdown_equal')}.",
         f"* Eval set: {s.get('n_eval_episodes')} episodes, {s.get('n_eval_windows')} windows; in-run subset perm sha256 "
         f"`{s.get('perm_sha256')}`.",
         f"* Pose-sync patch (r8_repoint): {json.dumps(s.get('pose_sync_patch', {}).get('stats'), default=str)[:400]}.",
         f"* Terms judged: {s.get('n_terms')} {json.dumps(s.get('by_class_counts'))}; not-OK by class: "
         f"{json.dumps(s.get('n_terms_not_ok_by_class'))}.", "",
         "| term | in-run | seeds mean | sd | class | verdict |", "|---|---|---|---|---|---|"]
    for k, t in (s.get("headline_terms") or {}).items():
        L.append(f"| `{k}` | {t.get('inrun')} | {t.get('mean')} | {t.get('sd')} | {t.get('cls') or t.get('class')} | "
                 f"{t.get('verdict')} |")
    L += ["", f"First reasons: {json.dumps((s.get('reasons_gate_first40') or [])[:8])}", "",
          "Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own "
          "in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw "
          "prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the "
          "run', not 'is the model good'."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True)
    ap.add_argument("--tag", default="step5000")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    s = summarize(Path(a.g0))
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    Path(a.out_md).write_text(to_md(s, a.tag), encoding="utf-8")
    print(json.dumps({k: s[k] for k in ("GATE_VERDICT", "gate_amendment", "n_seeds", "M1_detected", "finished")},
                     default=str))


if __name__ == "__main__":
    main()
