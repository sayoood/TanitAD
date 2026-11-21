"""r7m_stamp.py -- bank a SMALL, clip-id-free stamp of one refcv7 matched-phase battery tree (PREREG_R7_MATCHED_PHASE_TURN.md).

    python r7m_stamp.py --tree C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc --out raw/r7_matched_phase/BATTERY_STAMP_r7step20000_unc.json

Copies only whitelisted fields of `battery_summary.json` (ckpt + md5 + step, S2 size, the baselines the panels embedded, the inference-seed replicate,
the bars, the r8 wrapper stamp, per-seed roll wall time).  Raw clip ids live in the work tree and are NOT copied.  The dumps themselves stay in the
archive (single copy, C:/Users/Admin/refcv8_eval_archive/battery/<tag>); this stamp only says what they are.
"""
import argparse
import json
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ceiling-filter-off", default="unknown", help="what the chain was launched with (R7M_CEIL_OFF): 1 / 0 / unknown; the roll log line is the cross-check")
    a = ap.parse_args()
    s = json.load(open(os.path.join(a.tree, "battery_summary.json"), encoding="utf-8"))
    st = s.get("stages") or {}
    r8 = s.get("r8") or {}
    ceil_off = None                                   # was the roll's max-speed ceiling filter switched OFF (R7M_CEIL_OFF=1)?  Read from the roll children's own log line.
    seen = []
    for sd in (0, 1):
        lg = os.path.join(a.tree, f"roll_s{sd}.log")
        try:
            txt = open(lg, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if "ceiling filter:" in txt:
            seen.append("ceiling filter: OFF (R7M_CEIL_OFF=1)" in txt)
    ceil_off = (all(seen) if all(seen) or not any(seen) else "MIXED") if seen else None
    out = {"tool": "r7m_stamp.py", "ceiling_filter_off_per_roll_log": ceil_off, "ceiling_filter_off_per_chain_env": a.ceiling_filter_off, "generated": time.strftime("%FT%T"), "tree": a.tree, "tag": s.get("tag"), "ckpt": s.get("ckpt"),
           "ckpt_md5": s.get("ckpt_md5"), "step": s.get("step"), "config": s.get("config"), "config_md5": s.get("config_md5"),
           "started": s.get("started"), "finished": s.get("finished"), "S2": s.get("S2"), "SMOKE_RESTRICTED_S2": s.get("SMOKE_RESTRICTED_S2"),
           "g0": st.get("g0"), "baselines_used": s.get("baselines_used"), "baselines_missing": s.get("baselines_missing"),
           "inference_seed_replicate": s.get("inference_seed_replicate"),
           "bars": [{k: b.get(k) for k in ("id", "verdict")} for b in (s.get("bars") or [])],
           "rolls": {k: {kk: v.get(kk) for kk in ("seed", "arm_set", "wall_s", "n_windows", "n_episodes", "skipped", "cuda_max_memory_allocated_gib")}
                     for k, v in st.items() if k.startswith("roll_s") and isinstance(v, dict)},
           "r8": {"wrapper": r8.get("wrapper"), "argv": r8.get("argv"), "pose_sync_policy": r8.get("pose_sync_policy"), "wall_s": r8.get("wall_s"),
                  "no_baselines": r8.get("no_baselines"), "os_only_all": r8.get("os_only_all"),
                  "remap_overrides": (r8.get("repoint") or {}).get("remap_overrides"),
                  "remap_overrides_file": (r8.get("repoint") or {}).get("remap_overrides_file"),
                  "child_rewrites": (r8.get("patch") or {}).get("child_rewrites"),
                  "refcv7_baselines": (r8.get("patch") or {}).get("refcv7_baselines"),
                  "baselines_dropped": (r8.get("patch") or {}).get("baselines_dropped"),
                  "seam_telemetry": (r8.get("repoint") or {}).get("seam_telemetry")}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
