#!/usr/bin/env python3
"""YAXD chain (PREREG_AXD.md section 6). Idempotent; every stage logs to <root>/logs; LAST ACTION writes YAXD/markers/axd_<tag>.DONE|FAILED|SKIPPED (json).
    python axd_chain.py --tag smoke --tokens <w3 token json> [--red-tokens <json>] --no-wait
    python axd_chain.py --tag full                      # waits for YAX/markers/yax_full.DONE|FAILED; exits SKIPPED if YAX X2 is CANDIDATE
Stages: wait -> pins -> seam (GPU lock, never forced) -> [smoke: RED seams] -> per PDMS arm: harness score, families6, perwindow -> axd_analyze -> [full, no P1 CANDIDATE: P2 hook].
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axd_common as A  # noqa: E402

C = A.C
import ax_eval_arm as AE  # noqa: E402  (YAX, read-only: seam_env, GL (the shared dev-box GPU lock), run_score, run_families, run_perwindow)

MK = HERE / "markers"
CODE = ["axd_common.py", "axd_seam.py", "axd_analyze.py", "axd_chain.py", "axd_run_all.py"]


def log(root, msg):
    line = f"[{C.now()}] {msg}"
    print(line, flush=True)
    with open(Path(root) / "PIPELINE.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def marker(tag, kind, **kw):
    MK.mkdir(exist_ok=True)
    C.jdump(dict(kw, kind=kind, t=C.now(), pid=os.getpid()), MK / f"axd_{tag}.{kind}")


def wait_yax(root, give_up_h=48.0):
    t0 = time.time()
    while True:
        done, failed = (A.YAX_MARKERS / "yax_full.DONE").exists(), (A.YAX_MARKERS / "yax_full.FAILED").exists()
        if done or failed:
            st = C.jload(A.YAX_STATUS) if A.YAX_STATUS.exists() else {}
            v = st.get("verdicts", {})
            log(root, f"YAX finished ({'DONE' if done else 'FAILED'}); verdicts {v}")
            return v
        if time.time() - t0 > give_up_h * 3600:
            raise RuntimeError("YAX marker never appeared within the give-up window")
        time.sleep(120)


def check_pins(root):
    pin = json.loads((HERE / "CODE_PIN_AXD.json").read_text())
    bad = [f for f in CODE if C.sha256_file(HERE / f) != pin.get(f)]
    pre = (HERE / "PREREG_AXD.sha256").read_text().split()[0]
    if C.sha256_file(A.PREREG) != pre:
        bad.append("PREREG_AXD.md")
    if bad:
        raise RuntimeError(f"REFUSING: pin mismatch {bad} (re-pin + AMENDMENT required)")
    log(root, "pins OK")


def seam_cmd(arms_out, tokens, a0_out, regress="none", device="cuda"):
    return [C.PY_D, str(HERE / "axd_seam.py"), "--arms-out", str(arms_out), "--regress", regress, "--",
            "--ckpt", str(C.U400), "--frames", str(C.D / "frames"), "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", C.export_path(),
            "--tokens", str(tokens), "--out", str(a0_out), "--arm", "REFe_A0", "--rule", "navsim_v1_lane", "--device", device]


def run_seams(root, jobs, give_up_h, no_lock=False):
    """jobs: list of (label, cmd, logfile, done_file). ONE lock hold for all of them (never forced; released in finally)."""
    todo = [j for j in jobs if not Path(j[3]).exists()]
    if not todo:
        return
    body = None if no_lock else AE.GL.acquire_blocking(give_up_h * 3600, owner="yaxd_ax_eval", log=lambda m: log(root, m))
    log(root, f"GPU lock taken: {body}" if body else "CPU seam: no GPU lock (smoke only)")
    try:
        for label, cmd, lf, done in todo:
            t0 = time.time()
            with open(lf, "wb") as f:
                rc = subprocess.run(cmd, cwd=str(C.EVAL), env=AE.seam_env(), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            ok = "ZZAXDSEAM OK" in C.tail(lf, 400)
            log(root, f"seam {label}: rc {rc} ok {ok} in {time.time() - t0:.0f} s")
            if label == "main" and not ok:
                raise RuntimeError(f"seam failed: {C.tail(lf, 1500)}")
            Path(done).write_text(json.dumps({"rc": rc, "ok": ok}))
    finally:
        if body:
            AE.GL.release_gpu_lock(AE.GL.LOCK, body)
            log(root, "GPU lock released")


def red_check(root, red_root, regress):
    """end-to-end RED arm on the real seam: the named gate must read RED (PREREG section 3)."""
    import numpy as np
    import axd_analyze as AN
    sys.path.insert(0, str(HERE))
    z = np.load(red_root / "seam_arms" / "records.npz")
    rec = {k: z[k] for k in z.files}
    for arm in ("IDT", "IDTm") + A.CANDIDATES:
        rec[f"poses__{arm}"] = np.load(red_root / "seam_arms" / arm / "seam.npz")["poses"]
    toks = [str(t) for t in rec["token"]]
    csel = np.asarray([toks.index(str(t)) for t in rec["cruise_token"]], dtype=np.int64)
    got = {"transform_not_applied": not AN.gate_reach(rec, "CLAMP0")["PASS"] and not AN.gate_reach_idt_gain(rec)["PASS"],
           "ax0_leak": len(csel) > 0 and not AN.gate_zero(rec, rec["poses__AX0"][csel])["PASS"],
           "idt_perturbed": not AN.gate_idt(rec)["PASS"], "memo_stale": not AN.gate_memo(rec)["PASS"]}[regress]
    log(root, f"RED arm {regress}: gate red = {got} (cruise n {len(csel)})")
    return got


def score_arm(root, name, tokens, toks, tag):
    outd = root / "eval" / name
    if (outd / "arm.json").exists():
        return
    res = {"name": name, "started": C.now(), "tokens": str(tokens)}
    res["score"] = AE.run_score(name, tokens, outd, toks, tag=f"axd{tag}")
    res["families"] = AE.run_families(name, outd, toks)
    AE.run_perwindow(outd, tokens)
    res["finished"] = C.now()
    C.jdump(res, outd / "arm.json")
    log(root, f"arm {name} scored: {res['score']['status'].get('status')}")


def run(a):
    root = A.RUN_ROOT / a.tag
    for d in ("logs", "eval", "tokens", "analysis"):
        (root / d).mkdir(parents=True, exist_ok=True)
    marker(a.tag, "STARTED", root=str(root))
    if not a.no_wait:
        v = wait_yax(root)
        if v.get("X2") == "CANDIDATE":
            log(root, "YAX X2 is CANDIDATE -> YAXD exits without running (brief)")
            marker(a.tag, "SKIPPED", why="YAX X2 CANDIDATE", verdicts=v)
            return 0
    if a.tag == "full" or not a.skip_pins:
        check_pins(root)
    tokens = root / "tokens" / "tokens.json"
    if not tokens.exists():
        shutil.copy(a.tokens, tokens)
    toks = sorted(C.jload(tokens)["tokens"])
    free = shutil.disk_usage("C:/").free / 2 ** 30
    log(root, f"tokens {len(toks)}; C: free {free:.1f} GiB")
    if free < 2.0:
        raise RuntimeError("C: below 2 GiB free")
    a0 = root / "eval" / "A0"
    a0.mkdir(exist_ok=True)
    jobs = [("main", seam_cmd(root / "seam_arms", tokens, a0 / "seam.npz", device=a.device), root / "logs" / "seam.log", root / "logs" / "seam.DONE")]
    reds = ("transform_not_applied", "ax0_leak", "idt_perturbed", "memo_stale") if a.red_tokens else ()
    for r in reds:
        rr = root / "red" / r
        (rr / "eval").mkdir(parents=True, exist_ok=True)
        jobs.append((r, seam_cmd(rr / "seam_arms", a.red_tokens, rr / "eval" / "A0.npz", r, device=a.device), root / "logs" / f"seam_red_{r}.log", root / "logs" / f"seam_red_{r}.DONE"))
    if a.tag == "full" and (a.device != "cuda" or a.no_lock):
        raise RuntimeError("REFUSING: the full run is GPU-only and always takes the shared lock")
    run_seams(root, jobs, a.give_up_h, no_lock=a.no_lock)
    red_ok = {r: red_check(root, root / "red" / r, r) for r in reds}
    if reds and not all(red_ok.values()):
        raise RuntimeError(f"a RED arm did not turn its gate red: {red_ok}")
    for n in A.CANDIDATES:
        d = root / "eval" / n
        d.mkdir(exist_ok=True)
        for f in ("seam.npz", "seam.report.json"):
            if not (d / f).exists():
                shutil.copy(root / "seam_arms" / n / f, d / f)
    for n in A.PDMS_ARMS:
        score_arm(root, n, tokens, toks, a.tag)
    lf = root / "logs" / "analyze.log"
    rc = subprocess.call([C.PY_D, str(HERE / "axd_analyze.py"), "--root", str(root), "--tokens", str(tokens)] + (["--smoke"] if a.tag != "full" else []),
                         stdout=open(lf, "w"), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    p1 = C.jload(root / "analysis" / "P1.json")
    verdicts = {t: p1["arms"][t]["decision"]["verdict"] for t in A.CANDIDATES}
    log(root, f"analyze rc {rc}; P1 verdicts {verdicts}; candidates {p1['P1_candidates']}")
    gates = {k: v.get("PASS") for k, v in p1["gates"].items() if "PASS" in v}
    if a.tag != "full":
        mech = {k: v for k, v in gates.items() if k != "G-POS"}
        if not all(mech.values()):
            raise RuntimeError(f"smoke: mechanical gate red {mech}")
    marker(a.tag, "DONE", verdicts=verdicts, candidates=p1["P1_candidates"], gates=gates, red_arms=red_ok, root=str(root))
    if a.tag == "full" and not p1["P1_candidates"] and (HERE / "axd_p2_chain.py").exists():
        log(root, "no P1 CANDIDATE -> P2 (AXNOISE fine-tune), Rule Zero")
        return subprocess.call([C.PY_D, str(HERE / "axd_p2_chain.py")], stdin=subprocess.DEVNULL)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", default=str(C.A5_TOKENS))
    ap.add_argument("--red-tokens", default=None)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-lock", action="store_true", help="smoke on CPU only")
    ap.add_argument("--skip-pins", action="store_true", help="smoke only; the full run always checks the pins")
    ap.add_argument("--give-up-h", type=float, default=12.0)
    a = ap.parse_args()
    try:
        return run(a)
    except Exception:
        why = traceback.format_exc()[-2500:]
        print(why, flush=True)
        marker(a.tag, "FAILED", why=why)
        return 1


if __name__ == "__main__":
    sys.exit(main())
