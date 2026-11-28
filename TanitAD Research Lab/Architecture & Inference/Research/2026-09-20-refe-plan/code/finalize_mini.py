"""Stage 0 finisher: adopt the finished nuPlan-mini download, land it on D:, verify, extract,
run the released DriveRL teacher on the first scenarios, and render videos.

Every wait is BOUNDED and names its failure mode (a wait loop must match the way the thing fails):
  * Chrome's *.crdownload stalls > 10 min            -> FALLBACK: resumable curl straight to D:
  * no download in flight and no zip for 2 min       -> FALLBACK curl
  * Chrome holds a full-size .crdownload (awaiting a  -> copy it out (Chrome allows shared read);
    "Keep" click)                                        if the copy's CRCs fail -> FALLBACK curl
  * zip byte count != 8,550,100,030 or testzip fails  -> STAGE0_FAIL, nothing downstream runs
Data lives on D: ONLY (PI 2026-09-20). Run outputs stay at C:/Users/Admin/dz/out (short paths).
Markers on stdout: STAGE0_DONE / STAGE0_FAIL  (a watcher greps for them).
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

DL = "C:/Users/Admin/Downloads"
DATA = "D:/Projects/TanitAD/data/nuplan"
ZIP_D = f"{DATA}/nuplan-v1.1_mini.zip"
URL = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_mini.zip"
EXPECT = 8_550_100_030  # Content-Length from the S3 HEAD, 2026-09-20
PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
# the rebuilt interpreter (eval/EVAL_VENV.md); the %TEMP% driverl-venv was deleted 2026-09-24
VPY = os.environ.get("REFE_DRIVERL_PY", "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe")
LINK_D = f"{DATA}/dblinks/driverl_val14"  # plain dir on D: (the C: junction approach was retired 2026-09-20)
OUT_ROOT = "C:/dzo/m-nr-n"  # short: Windows MAX_PATH, see run_mini_teacher.sh
SCORE_TSV = "C:/Users/Admin/dz/DriveZero/DriveRL/output/task_logs/m-nr-n/score_summary.tsv"
N_SCEN, N_RENDER = 8, 3
FROM_RUN = "--from-run" in sys.argv  # resume at the teacher run: zip verified + DBs extracted already


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def fail(m):
    log(m)
    print("STAGE0_FAIL", flush=True)
    sys.exit(1)


def curl_to_d():
    log(f"FALLBACK: resumable curl -> {ZIP_D}")
    os.makedirs(DATA, exist_ok=True)
    r = subprocess.run(["curl", "--ssl-no-revoke", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10",
                        "-o", ZIP_D, URL], capture_output=True, text=True)
    log(f"curl exit {r.returncode}; tail: {r.stderr[-300:]}")


def zip_ok(path):
    if not os.path.exists(path) or os.path.getsize(path) != EXPECT:
        log(f"size check FAILED for {path}: {os.path.getsize(path) if os.path.exists(path) else 'missing'} != {EXPECT}")
        return False
    log("size OK; testing every member CRC (minutes for 8.5 GB)...")
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except Exception as e:
        log(f"zipfile error: {e}")
        return False
    if bad:
        log(f"CRC FAILED at member {bad}")
        return False
    log("CRC OK for all members")
    return True


# ---------------------------------------------------------------- 1. get the zip onto D:
os.makedirs(DATA, exist_ok=True)
t0 = time.time()
last_size, last_grow = None, time.time()
while not (os.path.exists(ZIP_D) and os.path.getsize(ZIP_D) == EXPECT):
    finals = [p for p in glob.glob(DL + "/nuplan*mini*.zip") if os.path.getsize(p) == EXPECT]
    if finals:
        log(f"final zip in Downloads: {finals[0]} -> moving to D: (nothing stays on C:)")
        shutil.move(finals[0], ZIP_D)
        break
    crs = glob.glob(DL + "/*.crdownload")
    if crs:
        s = os.path.getsize(crs[0])
        if s != last_size:
            last_size, last_grow = s, time.time()
        if s >= EXPECT and time.time() - last_grow > 90:
            log("Chrome holds a FULL-SIZE .crdownload (awaiting a 'Keep' click?) -> copying it out")
            shutil.copyfile(crs[0], ZIP_D)
            if zip_ok(ZIP_D):
                break
            os.remove(ZIP_D)
            curl_to_d()
            break
        if time.time() - last_grow > 600:
            log(f"Chrome download STALLED at {s/1e9:.2f} GB for 10 min")
            curl_to_d()
            break
        log(f"waiting on Chrome: {s/1e9:.2f}/{EXPECT/1e9:.2f} GB")
    else:
        if time.time() - t0 > 120:
            log("no .crdownload and no final zip for 2 min -> the browser download is gone")
            curl_to_d()
            break
    if time.time() - t0 > 3 * 3600:
        fail("TIMEOUT 3 h waiting for the zip")
    time.sleep(15)

if FROM_RUN:
    log("--from-run: zip already verified (CRC OK 11:07) and extracted (64 DBs 11:22); skipping to the teacher run")
elif not zip_ok(ZIP_D):
    fail("zip on D: is not valid; not extracting")

# ---------------------------------------------------------------- 2. extract on D:
if not FROM_RUN:
    log(f"extracting to {DATA} ...")
    with zipfile.ZipFile(ZIP_D) as z:
        names = z.namelist()
        log(f"{len(names)} members; first: {names[:3]}")
        z.extractall(DATA)
# mini-specific search (never `**/*.db`: test/val DBs will share DATA later and sort first)
cands = [LINK_D, f"{DATA}/data/cache/mini"]
db_dir = next((c for c in cands if glob.glob(c + "/*.db")), None)
if db_dir is None:
    fail(f"no mini *.db files under {cands}")
dbs = sorted(glob.glob(db_dir + "/*.db"))
log(f"{len(dbs)} .db files in {db_dir}  (RELAYED expectation: 64 logs)")

# ---------------------------------------------------------------- 3. link-free layout on D:
# MEASURED 2026-09-20: an NTFS junction (C:) into the exFAT volume lists EMPTY to os.listdir -> the
# devkit found "No log files"; exFAT holds no symlinks. Their runner only needs
# DRIVERL_EVAL_DB_LINK_ROOT/driverl_val14 to EXIST, so the DBs live there as a plain directory.
if os.path.normpath(db_dir).lower() != os.path.normpath(LINK_D).lower():
    os.makedirs(os.path.dirname(LINK_D), exist_ok=True)
    if os.path.exists(LINK_D):
        fail(f"{LINK_D} already exists; refusing to overwrite")
    log(f"renaming {db_dir} -> {LINK_D} (same volume, instant)")
    os.rename(db_dir, LINK_D)
    db_dir = LINK_D
n_link = len(glob.glob(LINK_D + "/*.db"))
if n_link == 0:
    fail(f"no .db files at {LINK_D}")
log(f"{n_link} .db at {LINK_D}")

# ---------------------------------------------------------------- 4. run the teacher (first light: 8 scenarios, non-reactive, no TTS)
env = dict(os.environ)
env.pop("DRIVERL_EVAL_DRY_RUN", None)
# ⛔ Do NOT export MSYS_NO_PATHCONV here (first run, 2026-09-20 11:22, exit 2): it stops Git Bash from
# converting /c/Users/... into C:\Users\... for the NATIVE python, which then looked for
# C:\c\Users\...\driverl_release_preflight.py. run_mini_teacher.sh scopes that flag to its mklink line.
env.pop("MSYS_NO_PATHCONV", None)
cmd = ["bash", f"{PKG}/code/run_mini_teacher.sh", db_dir, "nr", str(N_SCEN), "0"]
log("RUN " + " ".join(cmd))
run_log = f"{PKG}/raw/stage0_teacher_run.log"
os.makedirs(f"{PKG}/raw", exist_ok=True)
with open(run_log, "w", encoding="utf-8") as f:
    p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, text=True)
txt = open(run_log, encoding="utf-8", errors="replace").read()
res = re.findall(r"^RESULT .*$", txt, re.M)
log(f"teacher run exit {p.returncode}; RESULT lines: {res}")
if p.returncode != 0:
    log("tail of run log:\n" + txt[-2500:])
    fail("teacher run failed")

# ---------------------------------------------------------------- 5. render the first logs
logs = sorted(glob.glob(OUT_ROOT + "/**/*.msgpack.xz", recursive=True))
log(f"{len(logs)} simulation logs")
media = f"{PKG}/media"
os.makedirs(media, exist_ok=True)
rendered = []
for lp in logs[:N_RENDER]:
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(lp).replace(".msgpack.xz", ""))[:60]
    out = f"{media}/teacher_mini_{tag}.mp4"
    r = subprocess.run([VPY, f"{PKG}/code/render_teacher_bev.py", lp, out], capture_output=True, text=True)
    log(f"render exit {r.returncode}: {(r.stdout or r.stderr)[-300:].strip()}")
    if r.returncode == 0:
        rendered.append(out)

json.dump({"zip": ZIP_D, "db_dir": db_dir, "n_db": len(dbs), "result_lines": res, "score_tsv": SCORE_TSV,
           "logs": len(logs), "rendered": rendered, "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(f"{PKG}/raw/stage0_run.json", "w"), indent=1)
print("STAGE0_DONE", flush=True)
