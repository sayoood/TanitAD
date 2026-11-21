"""bank_battery_r8.py -- bank the refcv8 battery passes into the package SANITIZED and SLIM, and write BATTERY_SUMMARY_<tag>.json.

    python bank_battery_r8.py <work_root E:/refcv8_eval_kit/battery> <package raw/step5000> --tag step5000 \
        --passes unc,corr --chain-log <battery chain log> --timings <timings file> --g0-gate <battery_gate json>

Per pass `<tag>_<pass>` (unc = the standalone / refcv7-comparable clock, the paired read; corr = the corrected clock, E1/E3 only):
copies the SMALL json / log / txt / md files (< 5 MiB; dump_*, panel_*, s6_* and the raw-id window files stay on E:) through the
battery's own sanitiser (UUIDs -> sha12), and distils `battery_summary.json` + `PAIRED_READ_*.json` into one summary.
Refuses (exit 1) on any UUID hit in the destination.  Re-runnable: each call rewrites the summary from what exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code")
import sanitize_for_bank as S  # noqa: E402

MAX = 5 * 2**20
SKIP_DIRS = ("dump_", "panel_", "s6_s", "decisions", "batch_cache")
SKIP_FILES = ("windows_s0.json", "windows_s1.json")                 # raw clip ids (the sha12 twin is banked separately)


def _g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def distil_pass(work: Path) -> dict:
    sp = work / "battery_summary.json"
    if not sp.exists():
        return {"status": "NO battery_summary.json", "work_dir": str(work)}
    j = json.load(open(sp, encoding="utf-8"))
    st = j.get("stages") or {}
    out = {"status": "COMPLETE" if (j.get("bars") and j.get("finished")) else "INCOMPLETE",
           "started": j.get("started"), "finished": j.get("finished"), "tag": j.get("tag"), "ckpt_md5": j.get("ckpt_md5"),
           "step": j.get("step"), "S2": j.get("S2"), "SMOKE_RESTRICTED_S2": j.get("SMOKE_RESTRICTED_S2"),
           "G0_NONBLOCKING": j.get("G0_NONBLOCKING"), "tanitad_file": j.get("tanitad_file"),
           "r8": {k: (j.get("r8") or {}).get(k) for k in ("pose_sync_policy", "g0_nonblocking", "no_baselines", "smoke_cpu",
                                                          "wall_s")},
           "child_rewrites": _g(j, "r8", "patch", "child_rewrites"),
           "baselines_dropped": _g(j, "r8", "patch", "baselines_dropped"),
           "baselines_used": j.get("baselines_used"), "baselines_missing": j.get("baselines_missing"),
           "rolls": {}, "pairing": {}, "void_gates_3_4": {}}
    for s in (0, 1):
        r = st.get(f"roll_s{s}") or {}
        out["rolls"][f"s{s}"] = {k: r.get(k) for k in ("arm_set", "wall_s", "n_windows", "n_episodes", "skipped",
                                                         "cuda_max_memory_allocated_gib", "device", "frame_memo",
                                                         "window_restriction")}
        pn = st.get(f"panel_s{s}") or {}
        out["pairing"][f"s{s}"] = pn.get("pairing")
        out["void_gates_3_4"][f"s{s}"] = pn.get("void_gates_3_4")
    rep = j.get("inference_seed_replicate") or {}
    out["inference_seed_replicate"] = {"max_abs_path_diff_m": rep.get("max_abs_path_diff_m"), "floor_m": rep.get("floor_m"),
                                       "ade": rep.get("ade")}
    out["bars_r7_form_informative"] = [(b.get("id"), b.get("verdict")) for b in (j.get("bars") or []) if isinstance(b, dict)]
    out["strategic"] = j.get("strategic")
    return out


def bank_dir(src: Path, dst: Path) -> dict:
    rec = {"banked": [], "left_on_E": []}
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        parts = [x for x in rel.parts[:-1]]
        if any(part.startswith(SKIP_DIRS) or part in SKIP_DIRS for part in parts) or p.name in SKIP_FILES:
            continue
        if p.suffix.lower() not in (".json", ".log", ".txt", ".md"):
            continue
        if p.stat().st_size > MAX:
            rec["left_on_E"].append(f"{rel} ({p.stat().st_size} B)")
            continue
        S.sanitize_file(p, dst / rel)
        rec["banked"].append(str(rel).replace("\\", "/"))
    return rec


def bank_stamps(tag: str, total: int = 48852):
    """tier + stamps of a banked milestone summary, from the tag (`step<N>[ema]`); was hard-coded to step 5,000."""
    import re
    m = re.match(r"step(\d+)(ema)?$", tag)
    if not m:
        where = f"tag {tag} (step not parsed)"
        final = False
    else:
        n = int(m.group(1))
        final = n >= total
        where = (f"FINAL checkpoint, step {n:,} of {total:,}" if final else f"EARLY READ at step {n:,} of {total:,}")             + f", {'EMA' if m.group(2) else 'raw'} weights"
    tier = f"T1 self-action OPEN LOOP on logged eval139 frames; {where}"
    stamps = ["single training seed; training-seed floor at full scale unmeasured",
              ("the pre-registered bar verdict (final checkpoint)" if final else
               f"no bar verdict: the SPEC bars apply to the final checkpoint; these are early reads ({where})"),
              "numbers are PROVISIONAL unless G0 is PASS as coded or under the registered amendments (A7 + W-R8-1 [+ W-R8-2]): "
              "read passes.<pass>.paired_read.g0.G0_amended (SPEC_REFCV8 section 5)"]
    return tier, stamps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_root")
    ap.add_argument("package_raw")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--passes", default="unc,corr")
    ap.add_argument("--chain-log", default=None)
    ap.add_argument("--timings", default=None)
    ap.add_argument("--g0-gate", default=None)
    ap.add_argument("--status", default=None, help="free-text chain status stamped into the summary")
    ap.add_argument("--total-steps", type=int, default=48852, help="the run's --steps (refcv8-r101-s0: 48,852)")
    a = ap.parse_args()
    root, dst = Path(a.work_root), Path(a.package_raw)
    dst.mkdir(parents=True, exist_ok=True)
    tier, stamps = bank_stamps(a.tag, a.total_steps)
    summ = {"tool": "bank_battery_r8.py", "tag": a.tag, "written": time.strftime("%FT%T%z"), "chain_status": a.status,
            "tier": tier, "stamps": stamps, "passes": {}, "banked": {}}
    if a.g0_gate and Path(a.g0_gate).exists():
        summ["g0_gate"] = json.load(open(a.g0_gate, encoding="utf-8"))
    for ps in [x for x in a.passes.split(",") if x]:
        work = root / f"{a.tag}_{ps}"
        if not work.exists():
            summ["passes"][ps] = {"status": "NOT RUN", "work_dir": str(work)}
            continue
        try:
            d = distil_pass(work)
        except Exception as exc:                                  # noqa: BLE001 -- the bank is the LAST step: record, never crash it
            d = {"status": "DISTIL ERROR", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "work_dir": str(work)}
        pr = work / f"PAIRED_READ_{a.tag}_{ps}.json"
        if pr.exists():
            p = json.load(open(pr, encoding="utf-8"))
            d["paired_read"] = {"file": pr.name, "clock": p.get("clock"), "g0": p.get("g0"),
                                "bars": {k: (v.get("verdict") if k != "BAR-R8-NR" else
                                             {"verdict": v.get("verdict"), "per_family": v.get("per_family")})
                                         for k, v in (p.get("bars") or {}).items()},
                                "inference_floor_ade": {s: (f or {}).get("ade_m") for s, f in (p.get("inference_floor") or {}).items()},
                                "strategic": p.get("strategic"),
                                "controls_all_bit_identical": {k: v.get("all_bit_identical") for k, v in (p.get("controls") or {}).items()},
                                "windows_check": {f"{surf}_s{s}": (sd.get("windows_check") or {}).get("equal")
                                                  for surf, v in (p.get("surfaces") or {}).items()
                                                  for s, sd in (v.get("seeds") or {}).items()}}
        else:
            d["paired_read"] = {"status": "ABSENT (the paired read wrote no artifact)"}
        summ["passes"][ps] = d
        summ["banked"][ps] = bank_dir(work, dst / f"battery_{ps}")
    if a.timings and Path(a.timings).exists():
        summ["timings_raw"] = Path(a.timings).read_text(encoding="utf-8").splitlines()[-60:]
    if a.chain_log and Path(a.chain_log).exists():
        summ["chain_log_tail"] = [S._sub(x) for x in Path(a.chain_log).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]]
    summ = S._walk(summ)
    json.dump(summ, open(dst / f"BATTERY_SUMMARY_{a.tag}.json", "w", encoding="utf-8"), indent=1, default=str)
    hits = S.scan(dst)
    print(json.dumps({"summary": str(dst / f"BATTERY_SUMMARY_{a.tag}.json"), "passes": {k: v.get("status") for k, v in summ["passes"].items()},
                      "uuid_hits": hits}))
    if hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
