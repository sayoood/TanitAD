#!/usr/bin/env python3
"""Arm read chain (PREREG_SCORER1P6): wait for the E3 pass verdict, emit-changed, harness-score each arm's unscored picks, read. Markers: read/markers/read.DONE | read.FAILED. CPU only."""
import glob, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PKG + "/eval")
import stage2_sft as S2
E = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/eval"; E2 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/e2"
PASS = Path("F:/tanitad_scratch/p6arms/pass"); OUT = Path("F:/tanitad_scratch/p6arms/read"); MK = OUT / "markers"; LOG = OUT / "chain_read.log"
ARMS = ("C0", "C0r", "S1E")
COMMON = ["--p6-proposals", "F:/TanitAD_p6read_work/pass2/proposals_p6final.npz", "--e2-tokens", E2 + "/e2_tokens.json", "--e2-table", E2 + "/e2_table_p6final.npz"]
def say(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(time.strftime("[%F %T] ") + m + "\n")
def ntok(p):
    j = json.load(open(p, encoding="utf-8"))
    return len(j if isinstance(j, list) else j.get("tokens", j))
def main():
    MK.mkdir(parents=True, exist_ok=True)
    for m in ("read.DONE", "read.FAILED"):
        if (MK / m).exists(): raise SystemExit("marker exists " + m)
    pl = PASS / "arm_pass.log"
    while True:
        v = [l for l in pl.read_text(encoding="utf-8", errors="replace").splitlines() if l.startswith("ZZARMPASS_")] if pl.exists() else []
        if v: break
        time.sleep(120)
    say("pass verdict: " + v[-1][:300])
    if not v[-1].startswith("ZZARMPASS_OK"): raise RuntimeError("E3 pass did not pass: " + v[-1][:300])
    for a in ARMS:
        if not (PASS / a / "e3_logits.npz").exists(): raise RuntimeError("missing " + str(PASS / a / "e3_logits.npz"))
    armargs = sum([["--arm", f"{a}={PASS / a}"] for a in ARMS], [])
    em = OUT / "emit"
    rc = subprocess.call([S2.PY_D, "read_scorer1p6.py", "emit-changed", *COMMON, *armargs, "--out", str(em)], cwd=E, stdout=open(OUT / "emit.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    say(f"emit-changed rc {rc}")
    if rc != 0: raise RuntimeError(f"emit-changed rc {rc}")
    ens3_csv = None
    for seam in sorted(em.glob("seam_*.npz")):
        name = seam.stem[5:]; tok = em / f"tokens_{name}.json"; n = ntok(tok)
        say(f"{name}: {n} tokens to score")
        if n == 0: continue
        label = f"refe_p6s1_{name}"; sd = OUT / "score"; status = {}
        for att in range(1, 7):
            while S2.free_mb() < 4000: time.sleep(60)
            d = sd / label
            if d.is_dir(): shutil.rmtree(d)
            lp = OUT / f"score_{name}_{att}.log"
            subprocess.call([S2.PY_D, "score_e.py", "--label", label, "--seam", str(seam), "--tokens", str(tok), "--out", str(sd)], cwd=S2.SP,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"), stdout=open(lp, "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
            status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
            say(f"{name} score attempt {att}: {status.get('status')} rows {status.get('csv_valid_rows')}")
            if status.get("status") == "PASS": break
            time.sleep(120)
        if status.get("status") != "PASS": raise RuntimeError(f"harness did not PASS for {name}")
        cs = [c for c in glob.glob(str(sd / label / "*.csv")) if "_logs" not in c]
        if len(cs) != 1: raise RuntimeError(f"{name}: csv not unique {cs}")
        if name == "ENS3": ens3_csv = cs[0]
        else: shutil.copyfile(cs[0], PASS / name / "e3_changed_scores.csv")
    cmd = [S2.PY_D, "read_scorer1p6.py", "read", *COMMON, "--p6-csv", "E:/Projects/TanitAD/data/refe_navtest/score/refe_p6final/refe_p6final.csv", *armargs,
           "--c3-tokens", "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_all.json", "--out", str(OUT / "verdict")] + (["--ens3-csv", ens3_csv] if ens3_csv else [])
    rc = subprocess.call(cmd, cwd=E, stdout=open(OUT / "read.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    say(f"read rc {rc}")
    (MK / ("read.DONE" if rc == 0 else "read.FAILED")).write_text(json.dumps({"rc": rc, "out": str(OUT / "verdict")}) + "\n", encoding="utf-8")
if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception:
        tb = traceback.format_exc(); say("stopped: " + tb[-1500:]); (MK / "read.FAILED").write_text(tb[-3000:], encoding="utf-8"); sys.exit(1)
