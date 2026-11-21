#!/usr/bin/env python3
"""Unattended: when the refcv4b bridge pass ends, SCORE it, ANALYSE it, and decide BAR-W3-M1.

    PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/after_bridge_chain.py

⛔ WHY THIS EXISTS. The CPU pass is ~7.5 h. A pass that finishes into an empty room is a pass that
has to be re-noticed, and "the operator will run the next step" has cost this programme whole
nights. The chain waits on the ARTIFACT (never on an exit code), then runs the two steps that turn
a seam into a verdict, and writes a DONE marker either way.

⛔ NO PIPES AROUND THE STEPS. Every step is a ``subprocess.run`` whose ``returncode`` is read
directly and whose output goes to a file — `$?` after a pipe is the LAST element's status, which
has twice reported a clean exit for a step that failed or was killed.

Handles the partial case honestly: if the bridge stopped early (a RAM stall, a kill), the seam
holds fewer than 12,146 tokens, and the chain scores **exactly those** with the floors restricted
to the same tokens — a paired read that can never be mistaken for the published split, because the
analyser suppresses every published verdict under a restriction.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RAW = PKG / "raw"
SEAM = RAW / "bridge_navtest" / "seam_A1_ego_cmd.npz"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
PY_NAVSIM = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
PY_TANITAD = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FULL_N = 12146
FLOORS = ("CV", "STOP")                      # BAR-W3-M1: A1 > max(STOP, CV)


def log(msg: str):
    print(f"[chain {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bridge_alive(pattern: str = "*run_bridge*") -> bool:
    """⚠️ Match the EXECUTABLE plus a token that is not in this probe's own command line — a
    filter containing the string it searches for matches its own echoed command (measured three
    times in this programme). PowerShell is not python.exe, so this cannot self-match."""
    ps = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '" + pattern + "' }).Count")
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
        return int((r.stdout or "0").strip() or 0) > 0
    except Exception:
        return True                          # unreadable ⇒ assume alive; never a false "finished"


def run(step: str, argv: list, cwd=PKG, env=None) -> int:
    out = RAW / "chain_bridge" / f"{step}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, PYTHONIOENCODING="utf-8",
             PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval", **(env or {}))
    with open(out, "w", encoding="utf-8") as f:
        p = subprocess.run(argv, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=e)
    log(f"{step}: rc={p.returncode} -> {out.name}")
    return p.returncode


def bar_verdict(arms: dict, arm_key: str, partial: bool) -> dict:
    """BAR-W3-M1 exactly as the SPEC states it: **A1 > max(STOP, CV)**.

    ⛔ Beating ONE floor is not the bar. On navtest v1 a do-nothing plan scores three times
    constant velocity, so `max` is the whole point of the rule and an `any` here would read as a
    PASS on the easy side. The paired intervals travel WITH the verdict, because a point estimate
    above a floor and an interval straddling zero are different claims."""
    pdms = {a: arms[a]["x100"]["PDMS"] for a in list(FLOORS) + [arm_key] if a in arms}
    missing = [f for f in FLOORS if f not in pdms]
    paired = {f: (arms.get(arm_key, {}).get("paired_interval", {}) or {}).get(f) for f in FLOORS}
    floor = max((pdms[f] for f in FLOORS if f in pdms), default=None)
    got = pdms.get(arm_key)
    verdict = ("INCONCLUSIVE" if (missing or got is None or floor is None)
               else ("PASS" if got > floor else "FAIL"))
    return {"bar": "BAR-W3-M1: refcv4b A1_ego_cmd navtest PDMS_v1 > max(STOP, CV)",
            "n": arms.get(arm_key, {}).get("n"), "full_split": not partial,
            "PDMS_x100": pdms, "max_floor_x100": floor, "VERDICT": verdict,
            "missing_floors": missing,
            "paired_vs_floor": {f: ({"delta": (paired[f] or {}).get("delta"),
                                     "ci95": (paired[f] or {}).get("ci95"),
                                     "separated": (paired[f] or {}).get("separated")}
                                    if paired.get(f) else None) for f in FLOORS},
            "stamps": {"tier": "T1-family", "loop": "OPEN", "evidence": "MEASURED", "estimator":
                       "navsim_log_cluster_bootstrap (W2) — answers 'another draw of LOGS', blind "
                       "to training and inference variance"},
            "_note": ("⛔ A verdict on a PARTIAL seam is PROVISIONAL and is not the pre-registered "
                      "reading; the SPEC's bar is the full 12,146-token split." if partial else
                      "The pre-registered full-split reading.")}


def main() -> int:
    t0 = time.time()
    state = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # ── 1. wait on the ARTIFACT, with the process as the secondary signal ─────────────────── #
    while True:
        if SEAM.exists() and not bridge_alive():
            break
        if not bridge_alive() and not SEAM.exists():
            # ⭐ THE RECOVERY THE RESUME PARTS EXIST FOR. The bridge writes the seam only at the
            # end of the arm, so a kill leaves banked parts and no seam — work already paid for.
            # Assemble it (no model, no GPU, no device gate) and carry on with a PARTIAL read
            # rather than reporting the whole pass as lost.
            parts = sorted((SEAM.parent / "chunks_A1_ego_cmd").glob("part_*.npz"))
            if parts:
                log(f"bridge gone without a seam; assembling {len(parts)} banked parts")
                state["assembled_from_parts"] = len(parts)
                state["assemble_rc"] = run("assemble", [
                    PY_TANITAD, "code/run_bridge_navtest.py", "--arms", "A1_ego_cmd",
                    "--inputs", EXPORT,
                    "--bank", "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank",
                    "--out", str(SEAM.parent), "--device", "cpu", "--assemble-only"])
                if SEAM.exists():
                    break
            state["outcome"] = "BRIDGE_GONE_WITHOUT_SEAM"
            break
        if time.time() - t0 > 14 * 3600:
            state["outcome"] = "TIMEOUT_14H"
            break
        time.sleep(60)
    if state.get("outcome"):
        log(state["outcome"])
        json.dump(state, open(RAW / "chain_bridge_DONE.json", "w", encoding="utf-8"), indent=1)
        return 2
    time.sleep(10)                            # let the final manifest land
    with np.load(SEAM, allow_pickle=False) as z:
        toks = [str(t) for t in z["token"].tolist()]
    state.update(n_seam=len(toks), n_requested=FULL_N, partial=len(toks) != FULL_N)
    log(f"seam: {len(toks)} tokens (partial={state['partial']})")

    # ── 2. score the seam through the v1.1 devkit (NAVSIM venv) ───────────────────────────── #
    label = "A1_navtest" if not state["partial"] else f"A1part{len(toks)}_navtest"
    arm_key = label.rsplit("_", 1)[0]
    sel = None
    if state["partial"]:
        doc = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
        sel = RAW / f"{arm_key}_tokens.json"
        json.dump({"rule": f"the {len(toks)} tokens the bridge completed — a PAIRED subset, "
                           "NOT the published split", "tokens": toks,
                   "token_log": {t: doc[t]["log_name"] for t in toks}},
                  open(sel, "w", encoding="utf-8"), indent=1)
    argv = [PY_NAVSIM, "code/run_v1.py", "score", "--label", label,
            "--arm", f"SEAM:{SEAM.as_posix()}", "--cache-name", "navtest", "--patch-loader"]
    if sel is not None:
        argv += ["--tokens", str(sel)]
    state["score_rc"] = run("score", argv)
    csv_p = RAW / label / f"{label}.csv"
    state["score_csv"] = str(csv_p) if csv_p.exists() else None
    if not csv_p.exists():
        state["outcome"] = "SCORE_PRODUCED_NO_CSV"
        json.dump(state, open(RAW / "chain_bridge_DONE.json", "w", encoding="utf-8"), indent=1)
        return 3

    # ── 3. analyse, paired against the banked floors on the same tokens ───────────────────── #
    argv = [PY_TANITAD, "code/analyze_navtest.py", "--split-run", "navtest",
            "--arms", f"CV,HUMAN,STOP,{arm_key}"]
    tag = ""
    if state["partial"]:
        tag = f"part{len(toks)}"
        argv += ["--restrict-tokens", str(sel), "--tag", tag]
    state["analyze_rc"] = run("analyze", argv)
    ap = RAW / f"analysis_navtest{('_' + tag) if tag else ''}.json"
    if not ap.exists():
        state["outcome"] = "ANALYSIS_MISSING"
        json.dump(state, open(RAW / "chain_bridge_DONE.json", "w", encoding="utf-8"), indent=1)
        return 4
    d = json.load(open(ap, encoding="utf-8"))

    # ── 4. BAR-W3-M1, written as the SPEC states it ───────────────────────────────────────── #
    bar = bar_verdict(d["arms"], arm_key, state["partial"])
    bar["artifacts"] = {"seam": str(SEAM), "csv": str(csv_p), "analysis": str(ap)}
    json.dump(bar, open(RAW / "BAR_W3_M1.json", "w", encoding="utf-8"), indent=1)
    state["outcome"] = "DONE"
    state["bar"] = {k: bar[k] for k in ("VERDICT", "PDMS_x100", "max_floor_x100", "n")}
    state["wall_h"] = round((time.time() - t0) / 3600, 2)
    json.dump(state, open(RAW / "chain_bridge_DONE.json", "w", encoding="utf-8"), indent=1)
    log(f"BAR-W3-M1 {bar['VERDICT']}  {json.dumps(bar['PDMS_x100'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
