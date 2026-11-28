#!/usr/bin/env python3
"""SR-TTC S0 post-step (zero model turns).  Waits for the driver to finish, then
  * S0_DONE   -> MANIFEST.sha256.json: sha256 + bytes of EVERY output file (the bank npz/report files, the run's small result files, the code), written to run/ and to work/bank/;
                 `paths_s0_results.json` (repo-relative path, absolute source, "raw") for the SMALL result files only (the bulk features stay on C:), each scanned for an IPv4 address /
                 ssh target / token pattern first: a file that matches is NOT listed (the repo is PUBLIC);  FINALIZED.txt
  * S0_FAILED -> nothing but FINALIZED.txt noting the failure
  * the driver process died without writing a marker -> S0_FAILED is written here (rc 9) so a dead run can never read as "still running".
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I srttc_s0_post.py [tag]
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SRC = "C:/Users/Admin/srttc_s0"
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
RUN = f"{SRC}/run" if tag == "main" else f"{SRC}/run/_{tag}"
WORK = f"{SRC}/work" if tag == "main" else f"{SRC}/work/_{tag}"
PK_RAW = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-sel-rank"
# an IPv4 NOT embedded in a longer dotted number (nuPlan log names are dotted timestamps), an ssh target, a token shape, the ssh-prefix variable
SECRET = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])|root@|hf_[A-Za-z0-9]{8,}|-p\s*\d{4,5}\s+root|PODSSH=")
VERSIONISH = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pid_alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def has_secret(p):
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return True
    if SECRET.search(txt):
        return True
    return False


def main():
    deadline = time.time() + 7 * 24 * 3600
    marker, dead_since = None, None
    while time.time() < deadline:
        for m in ("S0_DONE", "S0_FAILED"):
            if os.path.exists(f"{RUN}/{m}"):
                marker = m
                break
        if marker:
            break
        try:
            pid = int(open(f"{RUN}/driver.pid").read().strip())
        except (OSError, ValueError):
            pid = None
        if pid and not pid_alive(pid):
            dead_since = dead_since or time.time()
            if time.time() - dead_since > 120:
                with open(f"{RUN}/S0_FAILED", "w", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=9 the driver (pid {pid}) is not running and wrote no marker. Re-issue the SAME launch command to resume: done batches are skipped.\n")
                marker = "S0_FAILED"
                break
        else:
            dead_since = None
        time.sleep(30)
    if marker is None:
        return 1
    time.sleep(10)
    notes = [f"marker={marker}"]
    if marker == "S0_DONE":
        files = {}
        for p in sorted(glob.glob(f"{WORK}/bank/*")) + sorted(glob.glob(f"{RUN}/*.json")) + sorted(glob.glob(f"{RUN}/state/*.json")) + sorted(glob.glob(f"{RUN}/plan/*.json")) \
                + sorted(glob.glob(f"{RUN}/S0_*")) + sorted(glob.glob(f"{RUN}/logs/*.log")) + sorted(glob.glob(f"{SRC}/code/*.py")) + sorted(glob.glob(f"{SRC}/code/landed/*.py")) \
                + sorted(glob.glob(f"{RUN}/batches/*/*.json")) + sorted(glob.glob(f"{RUN}/batches/*/*.tsv")):
            if os.path.isfile(p) and not p.endswith(".part") and "MANIFEST.sha256" not in p:
                files[p.replace(SRC + "/", "").replace("\\", "/")] = {"path": p.replace("\\", "/"), "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
        doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "what": "SR-TTC S0 outputs: sha256 + bytes of every file as written (bulk features stay on C: only)", "n_files": len(files), "files": files}
        for out in (f"{RUN}/MANIFEST.sha256.json", f"{WORK}/bank/MANIFEST.sha256.json"):
            with open(out, "w", encoding="utf-8", newline="\n") as w:
                json.dump(doc, w, indent=1)
        listed, withheld = [], []
        small = [p for p in sorted(glob.glob(f"{RUN}/*.json")) + sorted(glob.glob(f"{RUN}/S0_*")) + sorted(glob.glob(f"{RUN}/state/*.json")) + sorted(glob.glob(f"{RUN}/plan/*.json"))
                 + sorted(glob.glob(f"{RUN}/batches/*/stage_info.json")) + sorted(glob.glob(f"{RUN}/batches/*/audit.json")) + sorted(glob.glob(f"{RUN}/batches/*/copy_manifest_*.tsv"))
                 + sorted(glob.glob(f"{RUN}/logs/dump_*.log")) + [f"{RUN}/logs/podcopy.log", f"{RUN}/logs/podsets.log", f"{RUN}/logs/dgdump.log"]
                 if os.path.isfile(p) and os.path.getsize(p) < 20 * 2 ** 20]
        for p in small:
            rel = os.path.relpath(p, RUN).replace("\\", "/")
            if has_secret(p):
                withheld.append(rel)
            else:
                listed.append([f"{PK_RAW}/raw/s0/{rel}", p.replace("\\", "/"), "raw"])
        with open(f"{RUN}/paths_s0_results.json", "w", encoding="utf-8", newline="\n") as w:
            json.dump(listed, w)
        notes += [f"manifest {len(files)} files", f"results listed {len(listed)}", f"withheld (secret-pattern match) {withheld}"]
    with open(f"{RUN}/FINALIZED.txt", "w", encoding="utf-8") as w:
        w.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + " " + "; ".join(notes) + "\n")
    print("; ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
