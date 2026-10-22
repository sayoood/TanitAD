#!/usr/bin/env python3
"""Stage 2 of a scorer fine-tune (eval/PREREG_SFT1.md / PREREG_SFT3.md / PREREG_SFT4.md, identical rule): the stage-1
winner's full checkpoint replaces model_final.pt in the UNCHANGED seam (route fix ON, A7 ON) on Amendment 5's 923
confirmation tokens (43 logs), against the DEPLOYED system's own scores on the same tokens (the adopted route-fix run,
`refe_navtest_final_routefix`, model_final.pt, navsim_v1).

  1. seam (refe_navtest_seam.py, unchanged; --rule as registered) under the shared dev-box GPU lock
  2. score (W3's harness via score_e.py, the drive-repointed wrapper; retried while the harness RAM guard fires)
  3. per-token PDMS(arm) - PDMS(deployed), paired log-cluster bootstrap over the logs (10,000, seed 20260927)
  4. the four families of both on the SAME tokens (families6), longitudinal / lateral adverse separation (proxy_eval)
  Registered rule: ADOPT iff the lower bound > 0 AND no longitudinal or lateral component separates adversely;
  REFUTED iff the upper bound < 0; otherwise NOT PROVEN.
  Consistency control: every token whose executed poses equal the deployed poses must score IDENTICALLY.

    python eval/stage2_sft.py --ckpt <local model_sft_X.pt> --name sft3_A [--rule navsim_v1|navsim_v1_lane]
Writes raw/2026-10-05-sft-stage2/<name>/result.json and prints ZZSTAGE2 <name> <verdict> <mean> [lo, hi].

P6-READ-GAPS (2026-10-22), two changes and nothing else (every other line is the 2026-10-12 instrument, whose estimator, seam arguments and
scoring calls are unchanged; `python eval/test_p6_read_gaps.py` proves the seam command is the old one when --dump-proposals is absent):
  * G3  the harness (score_e.py, w3e/, ev6e/families6.py) is the BANKED copy under eval/harness/, sha256-pinned and refused on any mismatch
        (eval/harness_pins.py), no longer a previous session's %TEMP% scratchpad.
  * G2  --dump-proposals: the seam also writes all 64 proposals + the scorer logits (REPORT ONLY, same forward pass, ~95 MB on the full navtest),
        the input of the TACTICAL selection-quality rows (eval/selection_quality.py).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, str(PKG / "eval"))
import eval_checkpoint as EC  # noqa: E402
import harness_pins as HP  # noqa: E402
import proxy_eval as PE  # noqa: E402

# G3: the banked, sha256-pinned harness (was: a previous session's %TEMP% scratchpad, a085b71c-.../scratchpad)
HARNESS = PKG / "eval" / "harness"
SP = str(HARNESS)
EV6E = f"{SP}/ev6e"
D = Path("E:/Projects/TanitAD/data/refe_navtest")
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TOKENS = D / "a5_confirm_tokens.json"
BASE_SEAM = D / "seams" / "refe_navtest_final_routefix.npz"
BASE_CSV = D / "score" / "refe_navtest_final_routefix" / "refe_navtest_final_routefix.csv"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"


def take_lock(me):
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode()); os.close(fd)
                return
            except FileExistsError:
                pass
        print(f"waiting for the GPU lock {time.strftime('%T')}", flush=True)
        time.sleep(60)


def release(me):
    try:
        if open(LOCK, encoding="utf-8").read() == me:
            os.remove(LOCK); print(f"GPU lock released {time.strftime('%T')}", flush=True)
    except OSError:
        pass


def free_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def families(seam_npz: Path, toks: set, label: str, out_dir: Path):
    z = np.load(seam_npz)
    tt = z["token"]
    idx = [i for i, t in enumerate(tt) if str(t) in toks]
    sub = out_dir / f"{label}_seam.npz"
    np.savez(sub, token=tt[idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx], sampling=z["sampling"], arm=z["arm"])
    out = out_dir / f"{label}_families.json"
    subprocess.call([PY_T, "families6.py", "--seam", str(sub), "--inputs", EXPORT, "--stage", "1", "--label", label,
                     "--out", str(out), "--n-boot", "2000"], cwd=EV6E, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    stdout=open(out_dir / f"{label}_families.log", "w"), stderr=subprocess.STDOUT)
    return json.load(open(out, encoding="utf-8")) if out.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rule", default="navsim_v1", choices=("navsim_v1", "navsim_v1_lane"))
    ap.add_argument("--tokens", default=str(TOKENS), help="W3-format token set; default = Amendment 5's 923 (the registered set)")
    ap.add_argument("--exploratory", action="store_true",
                    help="a look on a NON-registered token set: the verdict is labelled EXPLORATORY and decides nothing")
    ap.add_argument("--dump-proposals", action="store_true",
                    help="G2: the seam also writes every proposal + the scorer's raw logits to seams/stage2_<name>.proposals.npz (REPORT ONLY; "
                         "the seam's poses are unchanged); the input of eval/selection_quality.py")
    a = ap.parse_args()
    # G3: refuse to start (no lock, no 4.8 h seam) unless the banked harness is the pinned one
    hp = HP.require(HARNESS, caller=PKG / "eval" / "score_navtest_refe.py")        # the caller score_e.py loads (same PKG)
    out_dir = PKG / "raw" / "2026-10-05-sft-stage2" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    TOK = Path(a.tokens)
    toks = sorted(json.load(open(TOK, encoding="utf-8"))["tokens"])
    seam = D / "seams" / f"stage2_{a.name}.npz"
    dump = D / "seams" / f"stage2_{a.name}.proposals.npz"
    # ⛔ the harness refuses any label not starting with "refe" (its scratch must never collide with W3's / refcv6's);
    # MEASURED 2026-10-05: "stage2_sft3_A" failed all score attempts in seconds. The selftests never scored, so missed it.
    label = f"refe_stage2_{a.name}"
    res = {"name": a.name, "ckpt": a.ckpt, "rule": a.rule, "tokens": str(TOK), "n_tokens": len(toks), "exploratory": a.exploratory,
           "deployed_seam": str(BASE_SEAM), "deployed_csv": str(BASE_CSV),
           "estimator": "paired log-cluster bootstrap over the confirmation logs, 10,000 resamples, 95 %, seed 20260927",
           "tier": "NAVSIM v1.1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents",
           "harness_pins": {"n_files": hp["n_files"], "pins_file_sha256": hp["pins_file_sha256"], "caller": hp["caller"]}}
    if a.dump_proposals:
        res["proposal_dump"] = {"requested": True, "path": str(dump)}
    if not seam.exists():
        me = json.dumps({"job": f"refe-stage2-{a.name}", "pid": os.getpid(), "host": socket.gethostname(),
                         "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        take_lock(me)
        try:
            env = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in EC.env_driverl().items()}
            cmd = [PY_D, "refe_navtest_seam.py", "--ckpt", a.ckpt, "--frames", str(D / "frames"),
                   "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", EXPORT,
                   "--tokens", str(TOK), "--out", str(seam), "--arm", f"REFe_{a.name}", "--rule", a.rule]
            if a.dump_proposals:
                cmd += ["--dump-proposals", str(dump)]
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=str(PKG / "eval"), env=env, stdout=open(out_dir / "seam.log", "w"),
                                 stderr=subprocess.STDOUT)
            res["seam"] = {"rc": rc, "seconds": round(time.time() - t0)}
        finally:
            release(me)
        if res["seam"]["rc"] != 0:
            res["verdict"] = "INCOMPLETE (seam failed)"
            json.dump(res, open(out_dir / "result.json", "w"), indent=1); print("ZZSTAGE2", a.name, res["verdict"]); return 1
    csv_p = D / "score" / label / f"{label}.csv"
    # ⛔ the harness's OWN status must read PASS -- a CSV existing is not evidence (MEASURED 2026-10-05: without --tokens
    # it scored the whole 12,146-token filter, marked the 11,223 tokens absent from the seam failed, guard C2 = FAIL,
    # and still wrote a CSV whose absent rows carry empty scores). --tokens restricts the scene filter to the subset.
    status = {}
    stp = out_dir / "score_status.json"
    if stp.exists():
        status = json.load(open(stp, encoding="utf-8"))
    # ⛔ WinError 206 (MEASURED 2026-10-11, all six attempts of sft4_B_u400_navtest_full): --tokens becomes ONE Hydra
    # override, scene_filter.tokens=[...], and 12,146 tokens make it 230,812 chars against Windows' 32,767 limit, so every
    # attempt died in CreateProcess before scoring anything. When the set IS the whole navtest filter, the filter itself
    # is the restriction: omit --tokens, and the harness's C2 then demands that ALL the filter's tokens score.
    full = set(toks) == set(json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"])
    tok_arg = [] if full else ["--tokens", str(TOK)]
    res["scored_as"] = "the whole navtest filter (no --tokens)" if full else "a --tokens subset"
    for att in range(1, 7):
        if status.get("status") == "PASS":
            break
        while free_mb() < 4000:
            time.sleep(60)
        lp = out_dir / f"score_{att}.log"
        # RUN_V1_FIX1_ALLOW_DRIFT: when the caller (eval/score_navtest_refe.py) is the post-NT-FIX1 variant it byte-pins W3's run_v1.py to the
        # ORIGINAL's sha256; the banked copy is that file with exactly six `D:/` -> `E:/` (HARNESS_PINS.json, raw/harness_provenance.jsonl), so the
        # pin must be told. It is recorded in the harness's counts.json (run_v1_matches_pinned: false). With the pre-NT-FIX1 caller it is inert.
        subprocess.call([PY_D, "score_e.py", "--label", label, "--seam", str(seam), *tok_arg,
                         "--out", str(D / "score")], cwd=SP,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"),
                        stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
        status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
        json.dump(status, open(stp, "w"), indent=1)
        if status.get("status") != "PASS":
            time.sleep(120)
    res["harness_status"] = status
    if status.get("status") != "PASS" or not csv_p.exists():
        res["verdict"] = "INCOMPLETE (the harness did not PASS)"
        json.dump(res, open(out_dir / "result.json", "w"), indent=1); print("ZZSTAGE2", a.name, res["verdict"]); return 1
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: E[t]["log_name"] for t in toks}
    arm = {t: 100.0 * float(r["score"]) for t, r in PE.read_csv(csv_p).items()}
    base = {t: 100.0 * float(r["score"]) for t, r in PE.read_csv(BASE_CSV).items()}
    rep = json.load(open(seam.with_suffix(".report.json"), encoding="utf-8"))
    sel = [t for t in toks if t in arm and t in base]
    zb, za = np.load(BASE_SEAM), np.load(seam)
    bp = {str(t): i for i, t in enumerate(zb["token"])}
    ap_ = {str(t): i for i, t in enumerate(za["token"])}
    same = [t for t in sel if t in bp and t in ap_ and np.array_equal(zb["poses"][bp[t]], za["poses"][ap_[t]])]
    mism = [t for t in same if arm[t] != base[t]]
    res["gates"] = {"rows": {"ok": len(sel) == len(toks) and rep.get("misses") == 0, "scored": len(sel), "asked": len(toks),
                             "seam_misses": rep.get("misses")},
                    "consistency": {"ok": not mism, "same_pose_tokens": len(same), "same_pose_different_score": len(mism)},
                    "frame_control_reported": rep.get("frame_control")}
    Dd = {t: arm[t] - base[t] for t in sel}
    mu, lo, hi = PE.boot(Dd, tl)
    res["primary"] = {"mean": mu, "ci95": [lo, hi], "pdms_deployed": float(np.mean([base[t] for t in sel])),
                      "pdms_arm": float(np.mean([arm[t] for t in sel])), "better": sum(v > 0 for v in Dd.values()),
                      "worse": sum(v < 0 for v in Dd.values()), "tied": sum(v == 0 for v in Dd.values()),
                      "zero_deployed": sum(base[t] == 0 for t in sel), "zero_arm": sum(arm[t] == 0 for t in sel),
                      "n_logs": len({tl[t] for t in sel})}
    fb = families(BASE_SEAM, set(sel), "deployed", out_dir)
    fa = families(seam, set(sel), a.name, out_dir)
    adv = PE.fam_adverse(fb, fa) if fb and fa else {"missing": True}
    res["families_adverse_arm_vs_deployed"] = adv
    gates_ok = res["gates"]["rows"]["ok"] and res["gates"]["consistency"]["ok"]
    no_adv = not adv.get("missing") and not adv.get("adverse")
    if not gates_ok:
        res["verdict"] = "NOT PROVEN (a gate failed)"
    elif lo > 0 and no_adv:
        res["verdict"] = "ADOPT"
    elif hi < 0:
        res["verdict"] = "REFUTED"
    else:
        res["verdict"] = "NOT PROVEN"
    if a.exploratory:
        res["verdict"] = "EXPLORATORY (not registered; decides nothing): would read " + res["verdict"]
    tail = ""
    if a.dump_proposals:                       # G2: the dump is REQUESTED, so its absence is stated, never silent (a seam re-run costs 4.8 h)
        present = dump.exists()
        res["proposal_dump"].update({"present": present, "bytes": dump.stat().st_size if present else None})
        tail = " proposal_dump=" + ("ok" if present else "MISSING")
    json.dump(res, open(out_dir / "result.json", "w"), indent=1)
    print("ZZSTAGE2", a.name, res["verdict"], f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]" + tail, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
