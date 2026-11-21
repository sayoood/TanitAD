"""Aggregate every raw JSON of this package into ``raw/summary.json``.

⛔ Every number in ``RESULT.md`` comes from here, and every number here comes
from a file in ``raw/`` or ``logs/``. Nothing is typed twice.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
RAW, LOGS = PKG / "raw", PKG / "logs"


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def jsonl(p: Path):
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def grad_table(rows, keys=None):
    """module -> (n_rows, n_rows_with_zero_grad, min, median, max)."""
    names = set()
    for r in rows:
        for k in r.get("grad", {}):
            if k.endswith("_grad_abs_sum"):
                names.add(k[len("gp_"):-len("_grad_abs_sum")])
    out = {}
    for nm in sorted(names):
        vals = [r["grad"]["gp_%s_grad_abs_sum" % nm] for r in rows
                if "gp_%s_grad_abs_sum" % nm in r.get("grad", {})]
        if not vals:
            continue
        nparams = next((int(r["grad"]["gp_%s_n_params" % nm]) for r in rows
                        if "gp_%s_n_params" % nm in r.get("grad", {})), None)
        out[nm] = {"n_params": nparams, "n_rows": len(vals),
                   "n_rows_exact_zero": sum(1 for v in vals if v == 0.0),
                   "min": min(vals), "median": st.median(vals),
                   "max": max(vals)}
    return out


def main():
    s: dict = {"evidence_class": "MEASURED"}

    asm = load(RAW / "assembly.json")
    if asm:
        s["assembly"] = {
            "git_head": asm["git_head"],
            "argv": asm["argv"],
            "payload_frame": asm["payload_frame"],
            "encoder_image_hw": asm["encoder_image_hw"],
            "encoder_in_channels": asm["encoder_in_channels"],
            "window": asm["window"],
            "trunk_spec_from_timm": asm["trunk_spec_from_timm"],
            "trunk_provenance": asm["trunk_provenance"],
            "n_params_total": asm["n_params_total"],
            "param_breakdown": asm["param_breakdown"],
            "unreachable_spec_seams": sorted(
                k for k, v in asm["spec_seam_cli_reachability"].items()
                if not v.get("reachable")),
            "seams_absent_from_built_model": sorted(
                k for k, v in asm["spec_seam_cli_reachability"].items()
                if v.get("in_built_model") is False),
        }

    fw_rows = jsonl(LOGS / "forward_full139.jsonl") or \
        (load(RAW / "forward.json") or {}).get("rows", [])
    if fw_rows:
        losses = {}
        for r in fw_rows:
            for k, v in r["losses"].items():
                losses.setdefault(k, []).append(v)
        s["forward"] = {
            "n_clips": len(fw_rows),
            "batch_shapes": fw_rows[0]["batch_shapes"],
            "t_s": {"mean": st.mean(r["t_s"] for r in fw_rows),
                    "min": min(r["t_s"] for r in fw_rows),
                    "max": max(r["t_s"] for r in fw_rows),
                    "total_h": sum(r["t_s"] for r in fw_rows) / 3600.0},
            "rss_gb_peak": max(r["rss_gb"] for r in fw_rows),
            "n_nonfinite_values": sum(len(r["nonfinite"]) for r in fw_rows),
            "nonfinite_keys": sorted({k for r in fw_rows
                                      for k in r["nonfinite"]}),
            "loss_terms_absent": fw_rows[0]["loss_terms_absent"],
            "loss_terms_zero_on_some_clip": sorted(
                {k for r in fw_rows for k in r["loss_terms_zero"]}),
            "n_clips_with_zero_route": sum(
                1 for r in fw_rows if "route" in r["loss_terms_zero"]),
            "per_loss": {k: {"min": min(v), "median": st.median(v),
                             "max": max(v), "n_exact_zero":
                             sum(1 for x in v if x == 0.0),
                             "n_nonfinite": sum(1 for x in v
                                                if x != x or abs(x) == float("inf"))}
                         for k, v in sorted(losses.items())},
            "grad": grad_table(fw_rows),
        }

    pc_rows = jsonl(LOGS / "perception.jsonl") or \
        (load(RAW / "perception.json") or {}).get("rows", [])
    if pc_rows:
        ok = [r for r in pc_rows if "map" in r]
        withbox = [r for r in ok if r.get("box", {}).get("loss_total")
                   is not None]
        pj = load(RAW / "perception.json") or {}
        s["perception"] = {
            "n_clips_attempted": len(pc_rows),
            "n_skipped": sum(1 for r in pc_rows if "skipped" in r),
            "skip_reasons": sorted({r["skipped"] for r in pc_rows
                                    if "skipped" in r}),
            "head_params": pj.get("head_params"),
            "fmap_s16_shape": ok[0]["fmap_s16_shape"] if ok else None,
            "bev_shape": ok[0]["bev_shape"] if ok else None,
            "map_logits_shape": ok[0]["map_logits_shape"] if ok else None,
            "label_grid": ok[0]["map"]["label_grid"] if ok else None,
            "map_loss": {
                "min": min(r["map"]["loss"] for r in ok),
                "median": st.median(r["map"]["loss"] for r in ok),
                "max": max(r["map"]["loss"] for r in ok),
                "n_exact_zero": sum(1 for r in ok if r["map"]["loss"] == 0.0),
                "n_nonfinite": sum(1 for r in ok
                                   if r["map"]["loss"] != r["map"]["loss"]),
            } if ok else None,
            "map_seen_frac": {
                "min": min(r["map"]["seen_frac"] for r in ok),
                "median": st.median(r["map"]["seen_frac"] for r in ok),
            } if ok else None,
            "map_grad_to_trunk": grad_table(
                [{"grad": r["map"]["grad"]} for r in ok if "grad" in r["map"]]),
            "box_grad_to_trunk": grad_table(
                [{"grad": r["box"]["grad"]} for r in withbox
                 if "grad" in r.get("box", {})]),
            "n_with_boxes": len(withbox),
            "box_loss": {
                "min": min(r["box"]["loss_total"] for r in withbox),
                "median": st.median(r["box"]["loss_total"] for r in withbox),
                "max": max(r["box"]["loss_total"] for r in withbox),
            } if withbox else None,
            "n_z_targets": {
                "min": min(r["box"].get("n_z_targets", 0) for r in withbox),
                "median": st.median(r["box"].get("n_z_targets", 0)
                                    for r in withbox),
                "max": max(r["box"].get("n_z_targets", 0) for r in withbox),
                "total": sum(r["box"].get("n_z_targets", 0) for r in withbox),
            } if withbox else None,
            "control_no_zh_all_zero_loss_z": (
                all(r["box"]["control_no_zh"]["loss_z"] == 0.0
                    and r["box"]["control_no_zh"]["n_z"] == 0
                    for r in withbox if "control_no_zh" in r["box"])
                if withbox else None),
            "control_delta_total_min_abs": (
                min(abs(r["box"]["control_no_zh"]["delta_total"])
                    for r in withbox if "control_no_zh" in r["box"])
                if withbox else None),
            "join_vs_parquet_max_abs_dz": (
                max(r["box"]["join_vs_parquet_max_abs_dz"] for r in withbox
                    if "join_vs_parquet_max_abs_dz" in r["box"])
                if withbox else None),
            "t_s_mean": st.mean(r["t_s"] for r in pc_rows if "t_s" in r)
            if any("t_s" in r for r in pc_rows) else None,
            "rss_gb_peak": max((r["rss_gb"] for r in pc_rows
                                if "rss_gb" in r), default=None),
        }

    for name, key in (("identity.json", "identity"),
                      ("zero_grad_forensics.json", "zero_grad_forensics"),
                      ("pretrained_guard.json", "pretrained_guard"),
                      ("join_frame_key.json", "join_frame_key"),
                      ("cost_model.json", "cost_model"),
                      ("tip_recheck.json", "tip_recheck"),
                      ("tactical.json", "tactical")):
        d = load(RAW / name)
        if d is None:
            continue
        if key == "identity":
            s[key] = {k: v for k, v in d.items()
                      if k not in ("per_key_max_abs_delta", "traceback")}
            s[key]["n_keys_compared"] = len(d.get("per_key_max_abs_delta", {}))
        elif key == "zero_grad_forensics":
            s[key] = d.get("verdicts")
        elif key == "pretrained_guard":
            s[key] = {m: {"stem": a["stem"],
                          "control": a["CONTROL_shipped_build"],
                          "mut_he_reinit": a["MUT_A_he_reinit_stem"],
                          "mut_pretrained_false": a["MUT_B_pretrained_false"],
                          "feature_info": a["feature_info"]}
                      for m, a in d["arms"].items()}
        elif key in ("join_frame_key", "cost_model", "tip_recheck"):
            s[key] = d
        elif key == "tactical":
            rows = d.get("rows", [])
            s[key] = {"config_overrides": d.get("config_overrides"),
                      "n_clips": len(rows),
                      "losses_first": rows[0]["losses"] if rows else None,
                      "grad": grad_table(rows)}

    (RAW / "summary.json").write_text(json.dumps(s, indent=1),
                                      encoding="utf-8")
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v)
                      for k, v in s.items()}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
