"""Merge the per-arm `bench_encoder_share.py` runs into the cross-arm ratios.

⚠️ ONE ARM PER PROCESS is forced by memory, not chosen: three real trunks +
three AdamW states + three device batches OOM'd on the 44 GB A40 (MEASURED —
`torch.OutOfMemoryError` at 44.06/44.43 GiB in use). Splitting the processes
re-introduces the confound the rig-fix stream was bitten by — a fixed order
makes the first arm systematically different — so the DRIVER runs the whole set
TWICE in ROTATED order and this merger takes the median across both passes,
reporting the spread so a large one is visible instead of averaged away.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

BASE = "256x640"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict] = {}
    batches: set[int] = set()
    for p in a.inputs:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        batches.add(int(d["batch"]))
        for tag, arm in d["arms"].items():
            per.setdefault(tag, {"full": [], "enc": []})
            per[tag]["full"].append(arm["full_step_s"]["median"])
            per[tag]["enc"].append(arm["encoder_only_s"]["median"])
            meta.setdefault(tag, {k: arm[k] for k in (
                "frame", "token_grid", "n_tokens", "images_encoded_per_step",
                "obs_window", "n_future_encoded", "encoder_params_M",
                "total_params_M")})

    # ⛔ every pass must be at the SAME micro-batch or the ratios are not matched
    assert len(batches) == 1, f"inputs mix micro-batches {sorted(batches)}"
    batch_of = batches.pop()
    out: dict = {"evidence_class": "MEASURED (ours; artifact = this JSON)",
                 "passes": len(a.inputs), "inputs": a.inputs,
                 "micro_batch": batch_of, "arms": {}}
    for tag, v in per.items():
        out["arms"][tag] = {
            **meta[tag],
            "full_step_s": {"median": round(st.median(v["full"]), 5),
                            "samples": [round(x, 5) for x in v["full"]],
                            "spread_pct": round(
                                100 * (max(v["full"]) - min(v["full"]))
                                / st.median(v["full"]), 2)},
            "encoder_only_s": {"median": round(st.median(v["enc"]), 5),
                               "samples": [round(x, 5) for x in v["enc"]],
                               "spread_pct": round(
                                   100 * (max(v["enc"]) - min(v["enc"]))
                                   / st.median(v["enc"]), 2)},
        }
    b = out["arms"][BASE]
    for tag, A in out["arms"].items():
        sr = A["full_step_s"]["median"] / b["full_step_s"]["median"]
        er = A["encoder_only_s"]["median"] / b["encoder_only_s"]["median"]
        A["step_ratio_vs_256x640"] = round(sr, 4)
        A["encoder_ratio_vs_256x640"] = round(er, 4)
        A["encoder_share_of_step_standalone"] = round(
            A["encoder_only_s"]["median"] / A["full_step_s"]["median"], 4)
        A["encoder_share_implied_by_step_ratio"] = (
            None if abs(1 - er) < 1e-9 else round((1 - sr) / (1 - er), 4))
    out["headline"] = {
        "PRIMARY_run_level_step_ratio_vs_256x640": {
            t: out["arms"][t]["step_ratio_vs_256x640"] for t in out["arms"]},
        "encoder_only_ratio_vs_256x640": {
            t: out["arms"][t]["encoder_ratio_vs_256x640"] for t in out["arms"]},
        "encoder_share_of_a_full_step_BRACKET": {
            t: sorted(x for x in (
                out["arms"][t]["encoder_share_of_step_standalone"],
                out["arms"][t]["encoder_share_implied_by_step_ratio"])
                if x is not None)
            for t in out["arms"]},
        "how_to_read": (
            "the PRIMARY is the direct one and needs no decomposition: it is "
            "the ratio of two REAL training steps. The share is reported as a "
            "BRACKET of two estimators of the same quantity (encoder timed "
            "standalone / step, and the share implied by the step ratio) "
            "because a standalone encoder timing does not reproduce the kernel "
            "overlap or memory pressure it sees inside the step."),
        "micro_batch": batch_of,
        "scope": (
            f"ONE A40, micro-batch {batch_of} (the largest that fits at "
            f"256x640 — see capacity_*.json; --batch 16 OOMs there), real "
            "modules and a real batch off the REGISTERED v5 val cache, DATA "
            "LOADING EXCLUDED. A COMPUTE ratio, not a wall-clock run ratio — "
            "the +6 % decode cost of the slice is CPU work in the DataLoader "
            "workers and is not in these numbers."),
    }
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["headline"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
