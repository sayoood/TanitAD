#!/usr/bin/env python3
"""Arm read chain, second version: the harness passes tokens on one command line (WinError 206 above about 32 KB), so each arm's changed tokens are scored in chunks of 1,000 and merged.
Reuses read/emit/ from chain_read.py. Markers read/markers/read2.DONE | read2.FAILED. CPU only."""
import csv, glob, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "F:/tanitad_scratch/p6arms/read")
import chain_read as C
S2 = C.S2; OUT = C.OUT; PASS = C.PASS; MK = C.MK; CH = 1000
def score_chunk(name, seam, toks, tl, k):
    label = f"refe_p6s1_{name}_k{k:02d}"; sd = OUT / "score"
    tj = OUT / "emit" / f"tokens_{name}_k{k:02d}.json"
    json.dump({"tokens": toks, "token_log": {t: tl[t] for t in toks}}, open(tj, "w", encoding="utf-8"))
    for att in range(1, 7):
        while S2.free_mb() < 4000: time.sleep(60)
        d = sd / label
        if d.is_dir(): shutil.rmtree(d)
        lp = OUT / f"score_{name}_k{k:02d}_{att}.log"
        subprocess.call([S2.PY_D, "score_e.py", "--label", label, "--seam", str(seam), "--tokens", str(tj), "--out", str(sd)], cwd=S2.SP,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"), stdout=open(lp, "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        st = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
        C.say(f"{name} chunk {k} ({len(toks)} tokens) attempt {att}: {st.get('status')} rows {st.get('csv_valid_rows')}")
        if st.get("status") == "PASS":
            cs = [c for c in glob.glob(str(sd / label / "*.csv")) if "_logs" not in c]
            if len(cs) != 1: raise RuntimeError(f"{label}: csv not unique {cs}")
            return cs[0]
        time.sleep(120)
    raise RuntimeError(f"harness did not PASS for {label}")
def main():
    for m in ("read2.DONE", "read2.FAILED"):
        if (MK / m).exists(): raise SystemExit("marker exists " + m)
    em = OUT / "emit"; ens3_csv = None
    for seam in sorted(em.glob("seam_*.npz")):
        name = seam.stem[5:]; j = json.load(open(em / f"tokens_{name}.json", encoding="utf-8")); toks = j["tokens"]; tl = j["token_log"]
        C.say(f"{name}: {len(toks)} tokens in {(len(toks) + CH - 1) // CH} chunks")
        if not toks: continue
        rows, header = [], None
        for k in range(0, len(toks), CH):
            p = score_chunk(name, seam, toks[k:k + CH], tl, k // CH)
            with open(p, encoding="utf-8", newline="") as f:
                r = csv.reader(f); h = next(r)
                header = header or h
                if h != header: raise RuntimeError(f"header differs in {p}")
                rows += [x for x in r if x[1] not in ("average", "")]
        got = {x[1] for x in rows}
        if got != set(toks): raise RuntimeError(f"{name}: scored {len(got)} tokens, needed {len(toks)}; missing {len(set(toks) - got)}")
        dst = OUT / "score" / f"merged_{name}.csv"
        with open(dst, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for i, x in enumerate(rows): w.writerow([i] + x[1:])
        C.say(f"{name}: merged {len(rows)} rows -> {dst}")
        if name == "ENS3": ens3_csv = str(dst)
        else: shutil.copyfile(dst, PASS / name / "e3_changed_scores.csv")
    armargs = sum([["--arm", f"{a}={PASS / a}"] for a in C.ARMS], [])
    cmd = [S2.PY_D, "read_scorer1p6.py", "read", *C.COMMON, "--p6-csv", "E:/Projects/TanitAD/data/refe_navtest/score/refe_p6final/refe_p6final.csv", *armargs,
           "--c3-tokens", "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_all.json", "--out", str(OUT / "verdict")] + (["--ens3-csv", ens3_csv] if ens3_csv else [])
    rc = subprocess.call(cmd, cwd=C.E, stdout=open(OUT / "read.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    C.say(f"read rc {rc}")
    (MK / ("read2.DONE" if rc == 0 else "read2.FAILED")).write_text(json.dumps({"rc": rc, "out": str(OUT / "verdict")}) + "\n", encoding="utf-8")
if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception:
        tb = traceback.format_exc(); C.say("read2 stopped: " + tb[-1500:]); (MK / "read2.FAILED").write_text(tb[-3000:], encoding="utf-8"); sys.exit(1)
