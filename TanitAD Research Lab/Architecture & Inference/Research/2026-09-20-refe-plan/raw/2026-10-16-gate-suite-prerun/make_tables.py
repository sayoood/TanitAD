#!/usr/bin/env python3
"""Render the MEASURED artifacts of this package as plain data tables under tables/*.txt (pipe tables, no prose).  Idempotent; stdlib only; run by finish_all.sh as its last step, or by hand.

Inputs (this directory):  baseline/scripts_before/summary.txt, baseline/scripts_after/summary.txt, baseline/test_launch_gate_static_before.log, arm_report.json,
logs/test_gate_suite_prerun_full.log, measure_p2rep/measure.json, baseline/tree_manifest_at_run.json.   A missing input is rendered as `NOT AVAILABLE: <path>`, never dropped.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
PK = D.parents[1]
sys.path.insert(0, str(PK / "refe"))
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402


def read(p: Path) -> str | None:
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None


def na(p) -> str:
    return f"NOT AVAILABLE: {p}"


def summary(p: Path) -> dict:
    out = {}
    for ln in (read(p) or "").splitlines():
        m = re.match(r"^(\S+) rc=(\d+) secs=(\d+) :: (.*)$", ln)
        if m:
            cnt = re.findall(r"ZZ\w+PASS (\d+)", m.group(4))
            out[m.group(1)] = {"rc": int(m.group(2)), "secs": int(m.group(3)), "count": int(cnt[-1]) if cnt else None}
    return out


def tests_table() -> str:
    b, a = summary(D / "baseline/scripts_before/summary.txt"), summary(D / "baseline/scripts_after/summary.txt")
    if not b or not a:
        return na("baseline/scripts_{before,after}/summary.txt")
    lg = re.findall(r"ZZLG_STATIC_PASS (\d+)", read(D / "baseline/test_launch_gate_static_before.log") or "")
    rows = [f"test_launch_gate.py --static --no-banked standalone, BEFORE: {lg[-1] if lg else 'NOT AVAILABLE'}", "",
            "| script (as check_suite runs it: cwd refe/, REFE_GATE_NESTED=1) | before: rc / checks / s | after: rc / checks / s | verdict |", "|---|---|---|---|"]
    for k, x in b.items():
        y = a.get(k)
        fmt = lambda r: "-" if r is None else f"rc {r['rc']} / {r['count']} / {r['secs']} s"          # noqa: E731
        v = "same" if y and (x["rc"], x["count"]) == (y["rc"], y["count"]) else "DIFFERENT"
        rows.append(f"| {k}.py | {fmt(x)} | {fmt(y)} | {v}{' (RED before AND after: pre-existing)' if k == 'test_rl_s1b' and y and y['rc'] == x['rc'] == 1 else ''} |")
    return "\n".join(rows)


def arms_table() -> str:
    p = D / "arm_report.json"
    if not p.is_file():
        return na(p)
    rep = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| # | arm | expected | observed | the gate's first reason (verbatim, cut at 130 chars) |", "|---|---|---|---|---|"]
    for i, (n, r) in enumerate(rep.items(), 1):
        why = (r.get("first_reason") or ("(accepted)" if r["observed"] == "ACCEPTED" else "(end-to-end arm: see the token's suite_source)")).replace("|", "/")[:130]
        rows.append(f"| {i} | {n}{' [e2e]' if r['e2e'] else ''} | {r['expected']} | {r['observed']} | {why} |")
    bad = [n for n, r in rep.items() if r["expected"] != r["observed"]]
    rows.append("")
    rows.append(f"{len(rep)} arms: {sum(1 for r in rep.values() if r['expected'] == 'ACCEPTED')} positive twins/controls (must be ACCEPTED), {sum(1 for r in rep.values() if r['expected'] == 'REJECTED')} must be REJECTED (live fallback). "
                f"expected != observed: {bad or 'none'}")
    return "\n".join(rows)


def matrix_table() -> str:
    t = read(D / "logs/test_gate_suite_prerun_full.log")
    if t is None:
        return na("logs/test_gate_suite_prerun_full.log")
    rows = ["| # | mutation (one acceptance condition / plumbing hop neutralised in a COPY of refe/) | arms that must turn ACCEPTED (= test RED) | caught |", "|---|---|---|---|"]
    n = 0
    for m in re.finditer(r"^\s{4}(\S+)\s+RED on (\d+)/(\d+)\s+(\[.*\])$", t, re.M):
        n += 1
        rows.append(f"| {n} | {m.group(1)} | {m.group(3)} | {m.group(2)}/{m.group(3)} {m.group(4)[:100]} |")
    tail = re.findall(r"ZZTEST_PASS \d+", t)
    rows.append("")
    rows.append(f"{n} mutations; last token of the full run: {tail[-1] if tail else 'NO ZZTEST_PASS (failed or unfinished)'}")
    return "\n".join(rows)


def measure_block() -> str:
    p = D / "measure_p2rep/measure.json"
    if not p.is_file():
        return na(p)
    m = json.loads(p.read_text(encoding="utf-8"))
    pr, we, wo = m["prerun"], m["run_with_evidence"], m["run_without_flag"]
    src = we.get("suite_source") or {}
    rows = [f"| step (dev box, profile {m['profile']}, {m['host']['cpus']} cpus, {m['host']['ram_gib']} GiB RAM, SHARED with other jobs) | seconds | rc | other |", "|---|---|---|---|",
            f"| suite-prerun | {pr['seconds']} | {pr['rc']} | verdict {pr.get('verdict')}; peak RSS of the process tree {pr.get('peak_rss_gib')} GiB; peak threads {pr.get('peak_threads')}; peak processes {pr.get('peak_processes')}; CPU {pr.get('cpu_seconds')} s; evidence {pr.get('evidence_bytes')} bytes |",
            f"| run --only G-SUITE --suite-evidence (ACCEPTED path) | {we['seconds']} | {we['rc']} | G-SUITE {we.get('g_suite_status')}; path {src.get('path')}; binding re-measure {src.get('verify_seconds')} s |",
            f"| run --only G-SUITE, no flag (today's gate, LIVE) | {wo['seconds']} | {wo['rc']} | G-SUITE {wo.get('g_suite_status')}; suite_source {wo.get('suite_source')} |", "",
            f"whole-process seconds include python start, `import train` and the data manifest; live - accepted = {round(wo['seconds'] - we['seconds'], 1)} s on this box"]
    return "\n".join(rows)


def savings_table() -> str:
    b = summary(D / "baseline/scripts_before/summary.txt")
    if not b:
        return na("baseline/scripts_before/summary.txt")
    secs = lambda prof: sum(b.get(e["script"].removesuffix(".py"), {}).get("secs", 0) for e in L.PROFILES[prof]["suite"])          # noqa: E731
    anchor_dev = secs("refe_p2rep")
    k = 22 * 60 / anchor_dev
    rows = ["| profile (arms it gates) | suite scripts | dev-box s (BASELINE, 4-way contended) | pod G-SUITE | saved per arm (minus ~0.2 min re-measure) |", "|---|---|---|---|---|"]
    arms = {"refe_p2": "P2", "refe_p2rep": "P2-s1", "refe_p6": "P6", "refe_field": "L_field F", "refe_field_ctrl": "L_field C", "refe_rls1": "RL-S1 C0 C1 R0 R1 R2", "refe_rls1b": "RL-S1b C1 R0b R1b",
            "refe_rls1c": "RL-S1c R0c", "refe_aux_s1": "aux S1: 9 arms", "refe_aux_s1c": "aux S1c: 2 arms"}
    for prof, who in arms.items():
        s = secs(prof)
        pod = 22.0 if prof == "refe_p2rep" else s * k / 60
        names = ", ".join(e["script"].removesuffix(".py").removeprefix("test_") for e in L.PROFILES[prof]["suite"])
        red = " (test_rl_s1b is RED: its 35 s is a lower bound)" if "test_rl_s1b.py" in [e["script"] for e in L.PROFILES[prof]["suite"]] else ""
        rows.append(f"| {prof} ({who}) | {names}{red} | {s} | {'22 min MEASURED' if prof == 'refe_p2rep' else f'{pod:.0f} min ESTIMATED'} | {pod - 0.2:.0f} min {'' if prof == 'refe_p2rep' else 'ESTIMATED'} |")
    rows.append("")
    rows.append(f"ESTIMATED rows: pod seconds = dev-box baseline seconds x {k:.2f} (= 22 min / the {anchor_dev} s the baseline took for refe_p2rep's three scripts); assumes one pod/dev ratio for every script; order of magnitude (+-2x). "
                "Only the refe_p2rep row is anchored to a measurement (P2-s1, 2026-10-15: GPU free 15:33Z -> TRAIN_START 16:13Z = 40 min, G-SUITE 22 min).")
    return "\n".join(rows)


def files_table() -> str:
    rows = ["| file | bytes | sha256 |", "|---|---|---|"]
    for rel in ("refe/launch_gate.py", "refe/gate_token.py", "refe/test_gate_suite_prerun.py"):
        p = PK / rel
        rows.append(f"| {rel} | {p.stat().st_size} | {hashlib.sha256(p.read_bytes()).hexdigest()} |" if p.is_file() else f"| {rel} | missing | |")
    p = D / "baseline/tree_manifest_at_run.json"
    if p.is_file():
        then, now = json.loads(p.read_text(encoding="utf-8")), GT.tree_manifest(PK)
        rows += ["", f"shipped tree after all runs vs before them ({len(now)} files): {'UNCHANGED' if then == now else 'CHANGED ' + str(GT.tree_diff(then, now))}; digest {GT.tree_digest(now)[:16]}"]
    else:
        rows += ["", na(p)]
    return "\n".join(rows)


def main() -> int:
    out = D / "tables"
    out.mkdir(exist_ok=True)
    parts = {"tests_before_after": tests_table(), "regression_arms": arms_table(), "mutation_matrix": matrix_table(), "measure_p2rep": measure_block(), "savings_estimate": savings_table(),
             "files": files_table()}
    allt = []
    for name, txt in parts.items():
        (out / f"{name}.txt").write_text(txt + "\n", encoding="utf-8", newline="\n")
        allt.append(f"===== {name} =====\n{txt}\n")
    (out / "ALL.txt").write_text("\n".join(allt), encoding="utf-8", newline="\n")
    n_na = sum(t.count("NOT AVAILABLE") for t in parts.values())
    print(f"tables/ written: {', '.join(parts)}; NOT AVAILABLE blocks: {n_na}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
