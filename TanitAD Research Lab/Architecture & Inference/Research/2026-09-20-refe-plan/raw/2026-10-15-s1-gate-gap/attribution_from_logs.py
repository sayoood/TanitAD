#!/usr/bin/env python3
"""Attribution of the GPU-idle gap before an S1 arm, computed from the pod logs copied to <logs>/ (chain.log, lg/<ARM>/*, runs/<ARM>/*).  Writes attribution.json.

    python attribution_from_logs.py --logs <s1gap dir> --arm N-s0 --out attribution.json

Every figure in the JSON carries its evidence class.  No pod access: it reads the copied files only.  Timestamps: chain.log and the evidence / token JSON are the pod's UTC; file mtimes
in the copy are the same instants (checked below against the JSON's own `finished_utc` / `created_utc`; a mismatch is reported, not hidden).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

UTC = dt.timezone.utc


def mt(p: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(p.stat().st_mtime, UTC).replace(microsecond=0)


def parse_utc(s: str) -> dt.datetime:
    return dt.datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def mins(a: dt.datetime, b: dt.datetime) -> float:
    return round((b - a).total_seconds() / 60.0, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--arm", default="N-s0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = Path(a.logs)
    chain = [ln for ln in (L / "chain.log").read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]

    def at(ln: str) -> dt.datetime:
        return parse_utc(ln.split()[0])
    # the attempt that trained the arm: the LAST START before ARM_TRAIN_START of this arm
    i_train = next(i for i, ln in enumerate(chain) if f"ZZS1AUX_ARM_TRAIN_START {a.arm}" in ln)
    i_start = max(i for i, ln in enumerate(chain[:i_train]) if "ZZS1AUX_START" in ln)
    att = chain[i_start:i_train + 1]

    def first(tag: str) -> str:
        return next(ln for ln in att if tag in ln)
    t_start, t_env, t_fetch, t_pre = at(first("ZZS1AUX_START")), at(first("ZZS1AUX_ENV_OK")), at(first("ZZS1AUX_FETCH_OK")), at(first("ZZS1AUX_PRECHECK_OK"))
    t_clear, t_gate_ok, t_train = at(first("ZZS1AUX_GPU_CLEAR")), at(first(f"ZZS1AUX_ARM_GATE_OK {a.arm}")), at(first(f"ZZS1AUX_ARM_TRAIN_START {a.arm}"))
    lg, runs = L / "lg" / a.arm, L / "runs" / a.arm
    ev = {c: json.loads((lg / f"evidence_{c}.json").read_text(encoding="utf-8")) for c in ("G-HYG", "G-DVB", "G-PREFLIGHT", "G-SUITE", "G-CLOCK")}
    tok_path = next(lg.glob("PASS_*.json"))
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    t_hyg, t_suite = parse_utc(ev["G-HYG"]["finished_utc"]), parse_utc(ev["G-SUITE"]["finished_utc"])
    t_mint = parse_utc(tok["created_utc"])
    t_row = mt(runs / "manifest_row.log")
    t_verify_last = mt(runs / "token_verify.log")
    checks = {"G-HYG.finished_utc == mtime": t_hyg == mt(lg / "evidence_G-HYG.json"), "G-SUITE.finished_utc == mtime": t_suite == mt(lg / "evidence_G-SUITE.json"),
              "token.created_utc == mtime": t_mint == mt(tok_path), "manifest row mtime == ARM_GATE_OK": t_row == t_gate_ok}
    seg = {"A_gate_start_to_HYG_DVB_PREFLIGHT": {"min": mins(t_clear, t_hyg), "what": "launch_gate.py run start: imports (profile_refusals imports scorer_finetune -> torch), tree_manifest + data_manifest #1, G-HYG, G-DVB, G-PREFLIGHT"},
           "B_G-SUITE": {"min": mins(t_hyg, t_suite), "what": "five test scripts, serially (check_suite)"},
           "C_finalize_and_mint": {"min": mins(t_suite, t_mint), "what": "finalize: tree_manifest + data_manifest #2, evidence hashes, HMAC, write token"},
           "D_verify_400_and_403": {"min": mins(t_mint, t_row), "what": "token_ok :400 (after the mint) and token_ok :403 (unconditional), then the manifest row (~1 s)", "per_verify_min": round(mins(t_mint, t_row) / 2, 2)},
           "E_wait_clear_and_verify_414": {"min": mins(t_gate_ok, t_train), "what": "wait_clear (3 idle polls x 30 s = >= 60 s by design) then token_ok :414 (the binding verify)"}}
    total = mins(t_clear, t_train)
    for s in seg.values():
        s["share_pct"] = round(100 * s["min"] / total, 1)
    verify_each = [seg["D_verify_400_and_403"]["per_verify_min"], round(seg["E_wait_clear_and_verify_414"]["min"] - 1.0, 2)]
    remeasure = {"verify_each_min": verify_each, "A_incl_imports_min": seg["A_gate_start_to_HYG_DVB_PREFLIGHT"]["min"], "C_min": seg["C_finalize_and_mint"]["min"],
                 "five_remeasures_min_total": round(seg["A_gate_start_to_HYG_DVB_PREFLIGHT"]["min"] + seg["C_finalize_and_mint"]["min"] + seg["D_verify_400_and_403"]["min"] + verify_each[1], 2)}
    remeasure["share_of_gap_pct"] = round(100 * remeasure["five_remeasures_min_total"] / total, 1)
    remeasure["evidence_class"] = "MEASURED (pod timestamps) for the segments; the split of a segment into 're-measure' vs 'imports / checks' is INFERRED (A also holds the imports and three cheap checks)"
    data = tok["binding"]["data"]
    inputs = {k: {"kind": v["kind"], "n_files": v.get("n_files"), "bytes": v.get("bytes")} for k, v in sorted(data.items())}
    big = max((v for v in data.values() if v["kind"] == "dir"), key=lambda v: v["n_files"])
    n_files = big["n_files"]
    hashing_upper_s = (t_fetch - t_env).total_seconds()
    tree_upper_s = max((t_pre - t_start).total_seconds(), 1.0)
    per_remeasure_s = remeasure["verify_each_min"][0] * 60
    listing_s_lower = per_remeasure_s - hashing_upper_s - tree_upper_s
    inference = {
        "checkpoint_sha256_upper_bound_s": hashing_upper_s,
        "checkpoint_sha256_evidence": f"chain.log ENV_OK {t_env:%H:%M:%S} -> FETCH_OK {t_fetch:%H:%M:%S}: file_pin_ok = stat + md5sum + sha256sum of the same {data['--ckpt']['bytes']:,}-byte file ran inside that window (second resolution)",
        "code_tree_upper_s": tree_upper_s,
        "code_tree_evidence": "chain.log START -> PRECHECK_OK: s1aux_checks.py tree (the same tree_manifest) ran inside that window (second resolution)",
        "image_listing_files": n_files,
        "image_listing_s_lower_bound": round(listing_s_lower, 1),
        "per_file_ms_lower_bound": round(1000 * listing_s_lower / n_files, 3),
        "other_eleven_inputs": "1-8 files each (token binding.data): negligible",
        "evidence_class": "INFERRED: per-re-measure cost (MEASURED, verify) minus two MEASURED upper bounds; the pod wrote no per-phase timestamps",
    }
    # per-arm preflight (inside the gap for every arm whose pre_ stage is not already done)
    pre = []
    for i, ln in enumerate(chain):
        m = re.search(r"ZZS1AUX_REFUSED stage=arm_(\S+): scorer_finetune.py --preflight did not pass", ln)
        if m:
            prev = chain[i - 1]
            pre.append({"arm": m.group(1), "from": prev.split()[0], "to": ln.split()[0], "min": mins(at(prev), at(ln)), "from_event": prev.split()[1] if len(prev.split()) > 1 else ""})
    pf = runs / "preflight.jsonl"
    out = {"arm": a.arm, "gap_min": total, "gap_from": f"{t_clear:%FT%TZ} ZZS1AUX_GPU_CLEAR", "gap_to": f"{t_train:%FT%TZ} ZZS1AUX_ARM_TRAIN_START", "timestamp_consistency": checks,
           "segments": seg, "remeasure": remeasure, "inside_one_remeasure": inference, "data_inputs": inputs, "g_suite_scripts_on_pod": ev["G-SUITE"]["details"].get("scripts") if isinstance(ev["G-SUITE"].get("details"), dict) else None,
           "preflight_inside_gap_for_arms_2_to_8_min": pre, "preflight_first_attempt_min": mins(parse_utc("2026-10-14T18:28:36Z"), mt(pf).replace(tzinfo=UTC)) if pf.exists() else None,
           "preflight_note": "N-s0's own preflight is NOT in the 24.8 min (pre_N-s0 was already done by the first attempt); every later arm pays about 6 min of it inside the gap (MEASURED: H2-s0, H1-s0 FAILED at the end of their preflight, so the duration is the full run)",
           "evidence_class": "MEASURED (pod timestamps in the copied logs) unless a block says INFERRED"}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"gap_min": total, "segments_min": {k: v["min"] for k, v in seg.items()}, "five_remeasures_min": remeasure["five_remeasures_min_total"],
                      "share_pct": remeasure["share_of_gap_pct"], "listing_s_lower_bound": inference["image_listing_s_lower_bound"], "per_file_ms": inference["per_file_ms_lower_bound"],
                      "consistency": checks, "preflight": pre}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
