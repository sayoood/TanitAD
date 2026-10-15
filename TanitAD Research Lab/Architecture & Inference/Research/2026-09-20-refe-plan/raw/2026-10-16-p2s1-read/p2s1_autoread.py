#!/usr/bin/env python3
"""P2-s1 auto-reader: a DETACHED dev-box job that starts the navtest read of the P2 training-seed replicate BY ITSELF.

    launch (detached, survives the session):   powershell -NoProfile -File launch_p2s1_autoread.ps1
    foreground / dry run:                      python p2s1_autoread.py --dry-run

WHAT IT DOES (every wait is a script loop, no model turns):
  1. POLL  Sayood/tanitad-refe-p2-joint-pdm-s1 every --poll-min (10) for the chain's FINAL marker (refe/p2rep_pod.sh, p2r_sync.py final):
         run/summary.json  done=true with model_final_sha256 (train.py writes it LAST)   AND
         run/model_final.pt in the listing, its server-side LFS sha256 == summary's         AND
         logs/chain.log, when pushed: any `ZZP2REP_FINAL_OK model_final sha256 <hex>` equals it.
     A light tick pushes summary.json before model_final.pt, so `done=true` ALONE is not the marker (MEASURED in the push code: p2r_sync.collect).
  2. DOWNLOAD run/model_final.pt (HF_HUB_DISABLE_XET=1) to <ck-dir>/model_final.pt and REFUSE unless its sha256 equals the published one.
  3. PREFLIGHT the instrument (stage2_sft.py md5, the scratch dependencies stage2 hard-codes, the token file, disk).
  4. GPU GATE  wait_idle.py (the neighbour jobs' idle waiter), then the shared lock + chain_active marker must be free.
     stage2_sft.py takes devbox_gpu.lock ITSELF (O_EXCL, a different body than aux_navtest_forward's), so this script does NOT hold it: holding it
     would make stage 2 wait for ever.  The gate proves the GPU idle and the lock free, then hands over in the same second.
  5. RUN eval/P2_READOUT_RUNBOOK.md section 3, verbatim:  stage2 (attempt 1 ends INCOMPLETE (seam failed) BY DESIGN) ; continue only if
     the seam .npz exists ; stage2 (attempt 2 reuses the seam) ; stage3_full --name <name> --ckpt <ck> --expect-ckpt-sha256 <sha>.
  6. READ  p2rep_read.py (RR1 / RR2 / RR3 -> exactly one row of PREREG_P2_REPLICATE section 3).
  7. MARKER  out/P2S1READ_DONE  (a RECORDED outcome: REPLICATED | REPLICATED, FLOOR-BOUND | NOT REPLICATED)
             out/P2S1READ_FAILED (anything else, INVALID included: no outcome is recorded; the file names the phase and the reason)

SECURITY. The HF token is read IN PLACE from Keys.txt (first `hf_` match), kept in this process only, handed to huggingface_hub as an object, and
never printed, written to a file, put in argv or inherited by a child (stage 2/3 get an environment without HF_TOKEN).  Every log line is scrubbed.
A re-run of the same command RESUMES (the checkpoint is re-verified by hash, stage 2 reuses its seam).  Never `pgrep -f` / `pkill -f`: a single-instance lock.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")        # MUST precede any huggingface_hub import (dev-box uploads/downloads hang in CloseWait under hf_xet)

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK_SUFFIX = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
MIRRORS = ("E:/Projects/TanitAD/", "D:/Projects/TanitAD/")
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO = "Sayood/tanitad-refe-p2-joint-pdm-s1"
NAME = "p2s1_full"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
WAIT_IDLE_CANDS = ("D:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py", "E:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py")
KEYS_CANDS = ("D:/Projects/TanitAD/Keys.txt", "E:/Projects/TanitAD/Keys.txt")
DATA_DEFAULT = "E:/Projects/TanitAD/data/refe_navtest"
CK_DIR_DEFAULT = "E:/Projects/TanitAD/data/refe_runs_eval/p2s1"
STAGE2_MD5 = "831a8d76bb5b8be2016288b02cd29ee5"            # R1's instrument, unchanged (eval/stage3_full.py STAGE2_SFT_MD5)
STAGE3_MD5_RECORDED = "a501ef4eac66f58b9e6115f61ef48ba5"   # recorded, not gated: the version that produced P2's banked R2
PROXY_MD5_RECORDED = "d177f0c9ab493390cb2b5bb8811c7cd3"
FAMILIES6_MD5 = "e275637c93e1f18ec80a347719680e02"
SP_FALLBACK = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
TOKENS_MD5 = "3a8b3653fd70891e50a230a37eb76ae8"
MODEL_FINAL_BYTES_P2 = 1288484600                          # P2-s0's model_final.pt (same architecture): informational only
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FINAL_OK_RE = re.compile(r"ZZP2REP_FINAL_OK\s+model_final\s+sha256\s+([0-9a-f]{64})")
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")


class Refuse(Exception):
    """a verification failed: write P2S1READ_FAILED and stop (never retried silently)"""


# ---------------------------------------------------------------- utilities
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_token(cands) -> str | None:
    """the first `hf_...` token in the first readable Keys file, IN PLACE.  Returned to the caller; never printed or written here."""
    for c in cands:
        try:
            m = re.search(r"hf_[A-Za-z0-9]+", Path(c).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if m:
            return m.group(0)
    return None


def pid_alive(pid) -> bool:
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"], capture_output=True, timeout=30)
        return f'"{int(pid)}"' in r.stdout.decode("utf-8", "replace")
    except Exception:
        return True                                      # cannot tell = assume alive (never steal a lock on a failed probe)


def find_pkg_root(cli=None) -> Path:
    for c in ([cli] if cli else []) + [m + PK_SUFFIX for m in MIRRORS]:
        if c and (Path(c) / "eval" / "stage2_sft.py").is_file():
            return Path(c)
    raise Refuse("no package root with eval/stage2_sft.py (E:/ or D:/ mirror); pass --pkg-root")


def find_first(cands):
    for c in cands:
        if Path(c).is_file():
            return str(c)
    return None


def stage2_scratch(pkg_root: Path) -> tuple:
    """SP and EV6E as stage2_sft.py hard-codes them (parsed from its source, so a moved scratch dir is seen here, not 5 h in)"""
    try:
        m = re.search(r'^SP = "([^"]+)"', (pkg_root / "eval" / "stage2_sft.py").read_text(encoding="utf-8"), re.M)
        sp = m.group(1) if m else SP_FALLBACK
    except OSError:
        sp = SP_FALLBACK
    return sp, sp + "/ev6e"


# ---------------------------------------------------------------- the marker (pure: tested)
def final_marker(tree, read_text) -> dict:
    """tree: {repo path: {"size": int, "sha256": str|None}} or None (no repo).  read_text(path) -> str.
    status: waiting (the chain is not over) | inconsistent (over, but the published hashes disagree) | final"""
    def w(why):
        return {"status": "waiting", "why": why}

    def i(why):
        return {"status": "inconsistent", "why": why}

    S, M, C = "run/summary.json", "run/model_final.pt", "logs/chain.log"
    if tree is None:
        return w("repo does not exist yet (or is not visible to this token)")
    if S not in tree:
        return w("run/summary.json not pushed yet")
    try:
        summ = json.loads(read_text(S))
    except Exception as e:                                # a half-written / unreadable file is a mid-push state, not a verdict
        return w(f"run/summary.json unreadable ({type(e).__name__})")
    if summ.get("done") is not True:
        return w("summary.json: done is not true (training or final steps running)")
    sha = str(summ.get("model_final_sha256") or "").lower()
    if not HEX64.match(sha):
        return i("summary.json says done=true but carries no valid model_final_sha256")
    if M not in tree:
        return w("summary.json is done=true but run/model_final.pt is not in the repo yet (a light tick pushes summary first)")
    ent = tree[M]
    lfs = (ent.get("sha256") or "").lower() or None
    if lfs is not None and lfs != sha:
        return i(f"remote LFS sha256 {lfs[:12]}.. != summary model_final_sha256 {sha[:12]}..")
    chain_ok = chain_failed = False
    if C in tree:
        try:
            txt = read_text(C)
        except Exception:
            txt = ""
        found = FINAL_OK_RE.findall(txt)
        if found:
            if found[-1] != sha:
                return i(f"chain.log ZZP2REP_FINAL_OK sha256 {found[-1][:12]}.. != summary {sha[:12]}..")
            chain_ok = True
        pos_fail, pos_ok = txt.rfind("ZZP2REP_FINAL_FAILED"), txt.rfind("ZZP2REP_FINAL_OK")
        chain_failed = pos_fail > pos_ok
    if chain_failed and lfs is None:
        return w("chain.log ends with ZZP2REP_FINAL_FAILED and the server hash is unavailable: not verifiable yet")
    return {"status": "final", "why": "summary done=true + model_final.pt listed" + (" + LFS sha256 equal" if lfs else " (no server hash)") + (" + chain FINAL_OK equal" if chain_ok else ""),
            "sha256": sha, "size": int(ent.get("size") or 0), "steps": summ.get("steps"), "server_hash_verified": lfs is not None,
            "chain_final_ok": chain_ok, "chain_final_failed": chain_failed, "size_equals_p2_s0": int(ent.get("size") or 0) == MODEL_FINAL_BYTES_P2}


# ---------------------------------------------------------------- backends
class HFBackend:
    def __init__(self, repo, token):
        try:
            import truststore
            truststore.inject_into_ssl()
        except Exception:
            pass
        from huggingface_hub import HfApi
        self.repo, self.token, self.api = repo, token, HfApi(token=token)

    def tree(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            items = list(self.api.list_repo_tree(self.repo, recursive=True, repo_type="model"))
        except RepositoryNotFoundError:
            return None
        out = {}
        for it in items:
            size = getattr(it, "size", None)
            if size is None:
                continue
            lfs = getattr(it, "lfs", None)
            get = (lambda k: lfs.get(k) if isinstance(lfs, dict) else getattr(lfs, k, None)) if lfs is not None else (lambda k: None)
            out[it.path] = {"size": int(get("size") or size), "sha256": get("sha256")}
        return out

    def read_text(self, path):
        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory() as td:
            return Path(hf_hub_download(repo_id=self.repo, filename=path, repo_type="model", token=self.token, local_dir=td)).read_text(encoding="utf-8", errors="replace")

    def fetch(self, path, dest_dir) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=self.repo, filename=path, repo_type="model", token=self.token, local_dir=str(dest_dir)))


class ScriptedBackend:
    """an HF stand-in for the dry run and the tests: `steps` is a list of {"tree": ..., "text": {path: str}, "blobs": {path: bytes}}; each tree() call
    advances one step (the last step repeats)"""

    def __init__(self, steps):
        self.steps, self.i = steps, -1

    @property
    def cur(self):
        return self.steps[min(max(self.i, 0), len(self.steps) - 1)]

    def tree(self):
        self.i += 1
        return self.cur["tree"]

    def read_text(self, path):
        return self.cur["text"][path]

    def fetch(self, path, dest_dir) -> Path:
        p = Path(dest_dir) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.cur["blobs"][path])
        return p


# ---------------------------------------------------------------- download + verify (pure given a backend: tested)
def download_verified(backend, info: dict, ck_path: Path, stage_dir: Path, min_free_bytes: int = 0, log=print) -> str:
    sha, size = info["sha256"], info["size"]
    if ck_path.is_file() and ck_path.stat().st_size == size and sha256_file(ck_path) == sha:
        return "already present, sha256 verified"
    anc = stage_dir
    while not anc.exists() and anc != anc.parent:
        anc = anc.parent
    free = shutil.disk_usage(anc).free
    if min_free_bytes and free < max(min_free_bytes, 2 * size):
        raise Refuse(f"disk: {free / 2**30:.1f} GiB free < {max(min_free_bytes, 2 * size) / 2**30:.1f} GiB needed for the download")
    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp = backend.fetch("run/model_final.pt", stage_dir)
    got_size, got = tmp.stat().st_size, sha256_file(tmp)
    if got != sha or (size and got_size != size):
        try:
            tmp.unlink()
        except OSError:
            pass
        raise Refuse(f"REFUSED: downloaded model_final.pt sha256 {got} ({got_size} B) != published {sha} ({size} B)")
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, ck_path)
    return f"downloaded {got_size} B, sha256 verified == published"


# ---------------------------------------------------------------- the plan: runbook section 3, verbatim (pure: tested)
def build_plan(cfg: dict, sha: str) -> dict:
    py, ev, pk = cfg["py"], Path(cfg["pkg_root"]) / "eval", Path(cfg["pkg_root"])
    ck, tokens, name = cfg["ck_path"], f"{cfg['data']}/navtest_full_tokens.json", cfg["name"]
    s2 = [py, str(ev / "stage2_sft.py"), "--ckpt", ck, "--name", name, "--rule", "navsim_v1", "--tokens", tokens]
    return {
        "idle_gate": [py, cfg["wait_idle"]],
        "stage2_attempt1": s2,
        "stage2_attempt2": list(s2),
        "stage3": [py, str(ev / "stage3_full.py"), "--name", name, "--ckpt", ck, "--expect-ckpt-sha256", sha],
        "read": [py, str(HERE / "p2rep_read.py"), "--name", name, "--r2-result", str(pk / "raw" / "p2_readout" / name / "r2_result.json"),
                 "--out-dir", str(Path(cfg["out"]) / "read")],
        "seam_npz": f"{cfg['data']}/seams/stage2_{name}.npz",
        "seam_report": f"{cfg['data']}/seams/stage2_{name}.report.json",
        "r2_result": str(pk / "raw" / "p2_readout" / name / "r2_result.json"),
    }


# ---------------------------------------------------------------- the job
class Job:
    def __init__(self, cfg: dict, backend, dry: bool = False, sleep=time.sleep, secrets=()):
        self.c, self.b, self.dry, self.sleep = cfg, backend, dry, sleep
        self.out = Path(cfg["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.secrets = [s for s in secrets if s]
        self.state = {"phase": "init", "started": utc(), "dry_run": dry, "pid": os.getpid(), "repo": cfg["repo"], "name": cfg["name"]}

    # -- logging / state / markers
    def scrub(self, s) -> str:
        s = TOKEN_RE.sub("hf_***", str(s))
        for sec in self.secrets:
            s = s.replace(sec, "hf_***")
        return s

    def log(self, msg):
        line = f"{utc()} {self.scrub(msg)}"
        print(line, flush=True)
        try:
            with open(self.out / "autoread.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def set_state(self, **kw):
        self.state.update(kw, updated=utc())
        tmp = self.out / "autoread_state.json.tmp"
        try:
            tmp.write_text(self.scrub(json.dumps(self.state, indent=1, default=str)), encoding="utf-8")
            os.replace(tmp, self.out / "autoread_state.json")
        except OSError:
            pass

    def marker(self, kind: str, payload: dict):
        for other in ("P2S1READ_DONE", "P2S1READ_FAILED", "P2S1READ_DRYRUN_OK"):
            if other != kind and (self.out / other).exists():
                (self.out / other).unlink()
        body = {"marker": kind, "when_utc": utc(), "repo": self.c["repo"], "name": self.c["name"], **payload}
        (self.out / kind).write_text(self.scrub(json.dumps(body, indent=1, default=str)), encoding="utf-8")
        self.log(f"ZZ{kind} {json.dumps({k: v for k, v in payload.items() if k in ('outcome', 'phase', 'reason')}, default=str)}")

    # -- subprocess runner (the dry run only prints)
    def child_env(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN")}
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run_step(self, label, argv, cwd, outfile=None, timeout_h=None) -> int:
        assert not any(TOKEN_RE.search(a) for a in argv), "a token-shaped string in argv"
        self.log(("WOULD RUN " if self.dry else "RUN ") + f"[{label}] cwd={cwd} :: " + " ".join(f'"{a}"' if " " in a else a for a in argv))
        if self.dry:
            return 0
        t0 = time.time()
        with open(outfile, "ab") if outfile else open(os.devnull, "wb") as fh:
            p = subprocess.Popen(argv, cwd=str(cwd), env=self.child_env(), stdout=fh, stderr=subprocess.STDOUT)
            limit = (timeout_h or self.c["step_timeout_h"]) * 3600
            while True:
                try:
                    rc = p.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    self.set_state(step=label, step_pid=p.pid, step_elapsed_s=round(time.time() - t0))
                    if time.time() - t0 > limit:
                        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                        raise Refuse(f"step {label} exceeded {limit / 3600:.0f} h and was killed")
        self.log(f"[{label}] exit {rc} after {time.time() - t0:.0f} s")
        return rc

    # -- phases
    def poll(self) -> dict:
        c, errs, inconsistent = self.c, 0, 0
        deadline = time.time() + c["max_wait_h"] * 3600
        n = 0
        while True:
            n += 1
            try:
                m = final_marker(self.b.tree(), self.b.read_text)
                errs = 0
            except Exception as e:
                errs += 1
                m = {"status": "error", "why": f"{type(e).__name__}: {self.scrub(e)[:200]}"}
                if errs >= c["max_errors"]:
                    raise Refuse(f"{errs} consecutive poll errors; last: {m['why']}")
            self.set_state(phase="poll", poll_n=n, last_poll=utc(), last_status=m["status"], last_why=m["why"])
            if m["status"] == "final":
                self.log(f"poll {n}: FINAL marker: {m['why']} (sha256 {m['sha256'][:12]}.., {m['size']} B, steps {m.get('steps')})")
                return m
            inconsistent = inconsistent + 1 if m["status"] == "inconsistent" else 0
            if inconsistent >= c["max_inconsistent"]:
                raise Refuse(f"the chain's published hashes disagree for {inconsistent} consecutive polls: {m['why']}")
            self.log(f"poll {n}: {m['status']}: {m['why']}")
            if time.time() > deadline:
                raise Refuse(f"no final marker within {c['max_wait_h']} h")
            self.sleep(c["poll_s"])

    def preflight(self, plan):
        c = self.c
        pk = Path(c["pkg_root"])
        sp, ev6e = stage2_scratch(pk)
        checks = {
            "stage2_sft.py md5 == R1's pin": md5_file(pk / "eval" / "stage2_sft.py") == STAGE2_MD5,
            "stage3_full.py exists": (pk / "eval" / "stage3_full.py").is_file(),
            "score_e.py (stage 2 scratch)": (Path(sp) / "score_e.py").is_file(),
            "families6.py md5": (Path(ev6e) / "families6.py").is_file() and md5_file(Path(ev6e) / "families6.py") == FAMILIES6_MD5,
            "token file md5": (Path(c["data"]) / "navtest_full_tokens.json").is_file() and md5_file(Path(c["data"]) / "navtest_full_tokens.json") == TOKENS_MD5,
            "frames dir": (Path(c["data"]) / "frames").is_dir(),
            "p2rep_read inputs": subprocess.run([c["py"], str(HERE / "p2rep_read.py"), "--check-inputs"], capture_output=True, text=True).returncode == 0,
        }
        self.state["instrument_md5"] = {"stage3_full.py": md5_file(pk / "eval" / "stage3_full.py") if (pk / "eval" / "stage3_full.py").is_file() else None,
                                        "proxy_eval.py": md5_file(pk / "eval" / "proxy_eval.py") if (pk / "eval" / "proxy_eval.py").is_file() else None,
                                        "recorded_for_P2": {"stage3_full.py": STAGE3_MD5_RECORDED, "proxy_eval.py": PROXY_MD5_RECORDED}}
        for k, v in checks.items():
            self.log(f"preflight {'ok  ' if v else 'FAIL'} {k}")
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise Refuse("preflight failed: " + "; ".join(bad))

    def gpu_gate(self, plan):
        """proves the dev-box GPU idle and the shared lock free; stage2_sft.py then takes the lock itself"""
        c = self.c
        if self.dry:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            self.log(f"WOULD RUN [idle_gate] {' '.join(plan['idle_gate'])} ; then require no {LOCK} and no {MARK}. now: GPU used {used} MiB; lock {'HELD ' + Path(LOCK).read_text(encoding='utf-8')[:120] if Path(LOCK).exists() else 'free'}")
            return
        deadline = time.time() + c["gpu_wait_max_h"] * 3600
        while True:
            self.set_state(phase="gpu_gate")
            left = deadline - time.time()
            if left <= 0:
                raise Refuse(f"the dev-box GPU was not free within {c['gpu_wait_max_h']} h")
            try:
                r = subprocess.run(plan["idle_gate"], timeout=left, env=self.child_env())
                idle = r.returncode == 0
            except subprocess.TimeoutExpired:
                raise Refuse(f"wait_idle.py did not report an idle GPU within {c['gpu_wait_max_h']} h")
            held = self.lock_state()
            if idle and held is None:
                self.log("GPU idle and the shared lock is free: handing over to stage2_sft.py (it takes the lock itself)")
                return
            self.log(f"GPU gate: idle={idle}, lock={held}: waiting 60 s")
            self.sleep(60)

    def lock_state(self):
        """None if free; else a description.  A dead holder's lock is removed only when older than 10 min (aux_navtest_forward's rule)."""
        if Path(MARK).exists():
            return f"{MARK} present (a chain is active)"
        if not Path(LOCK).exists():
            return None
        try:
            holder, age = json.loads(Path(LOCK).read_text(encoding="utf-8")), time.time() - os.path.getmtime(LOCK)
        except (OSError, ValueError):
            return "lock unreadable"
        if "pid" in holder and not pid_alive(holder["pid"]) and age > 600:
            self.log(f"WARN: removing the stale GPU lock of dead pid {holder['pid']} (age {age:.0f} s)")
            try:
                Path(LOCK).unlink()
                return None
            except OSError:
                pass
        return f"held by {holder}"

    def execute(self, info: dict, plan: dict):
        c, pk = self.c, Path(self.c["pkg_root"])
        ck = Path(c["ck_path"])
        s2out = self.out
        self.set_state(phase="stage2_attempt1")
        rc1 = self.run_step("stage2_attempt1", plan["stage2_attempt1"], pk, s2out / "stage2_att1.out")
        self.log("attempt 1 ends `ZZSTAGE2 <name> INCOMPLETE (seam failed)` BY DESIGN (the C1 frame-control gate); the seam .npz decides whether to continue")
        if not self.dry and not Path(plan["seam_npz"]).is_file():
            raise Refuse(f"stage2 attempt 1 (rc {rc1}) left no seam .npz at {plan['seam_npz']}: the seam really died (see stage2_att1.out)")
        self.set_state(phase="stage2_attempt2")
        rc2 = self.run_step("stage2_attempt2", plan["stage2_attempt2"], pk, s2out / "stage2_att2.out")
        if rc2 != 0:
            raise Refuse(f"stage2 attempt 2 exited {rc2} (the harness did not PASS, see stage2_att2.out)")
        self.set_state(phase="stage3")
        rc3 = self.run_step("stage3", plan["stage3"], pk, s2out / "stage3.out")
        if not self.dry and not Path(plan["r2_result"]).is_file():
            raise Refuse(f"stage3 (rc {rc3}) wrote no r2_result.json at {plan['r2_result']}")
        self.set_state(phase="read", stage3_rc=rc3)
        rcr = self.run_step("read", plan["read"], pk, s2out / "read.out")
        return rc3, rcr

    def finish(self, info, plan, rc3, rcr) -> int:
        """writes the DONE / FAILED marker of a finished pipeline; returns the exit code (0 only for a RECORDED outcome)"""
        if self.dry:
            return 0
        res = json.load(open(Path(self.c["out"]) / "read" / "p2rep_result.json", encoding="utf-8")) if (Path(self.c["out"]) / "read" / "p2rep_result.json").is_file() else None
        if res is None:
            raise Refuse(f"p2rep_read.py (rc {rcr}) wrote no p2rep_result.json (see read.out)")
        common = {"sha256": info["sha256"], "ckpt": self.c["ck_path"], "stage3_rc": rc3, "read_rc": rcr, "p2rep_result": str(Path(self.c["out"]) / "read" / "p2rep_result.json"),
                  "report_md": str(Path(self.c["out"]) / "read" / "P2REP_READ.md"), "r2_result": plan["r2_result"], "instrument_md5": self.state.get("instrument_md5")}
        rr1, rr2 = res["RR1"]["primary"], res["RR2"]
        summary = {"outcome": res["outcome"], "RR1": [rr1["mean"], rr1["ci95"]], "adverse": res["RR1"]["adverse_subscores"],
                   "delta_seed": [rr2["delta_seed_mean"], rr2["delta_seed_ci95"]], "F": rr2["F"], "D_s0": res["D_s0"], "D_s1": res["D_s1"]}
        if res["outcome"] == "INVALID":
            self.marker("P2S1READ_FAILED", {"phase": "read", "reason": "INVALID read: failed gates " + ", ".join(res["failed_gates"]) + " (no outcome is recorded)", **common, **summary})
            return 1
        self.marker("P2S1READ_DONE", {**common, **summary})
        return 0

    def run(self) -> int:
        c = self.c
        try:
            self.log(f"start: repo {c['repo']} name {c['name']} dry_run={self.dry} poll {c['poll_s']} s, pkg {c['pkg_root']}")
            if not self.dry and (self.out / "P2S1READ_DONE").is_file() and not c["force"]:
                self.log("P2S1READ_DONE already exists: nothing to do (--force re-runs)")
                return 0
            if not self.dry:
                for stale in ("P2S1READ_FAILED",):
                    if (self.out / stale).exists():
                        self.log(f"clearing the previous {stale} (a relaunch is a retry)")
                        (self.out / stale).unlink()
            info = self.poll()
            self.set_state(phase="download", final=info)
            self.log("download: " + download_verified(self.b, info, Path(c["ck_path"]), Path(c["stage_dir"]), min_free_bytes=c["min_free_gib"] * 2**30, log=self.log))
            self.set_state(phase="verified", sha256=info["sha256"], ckpt=c["ck_path"])
            plan = build_plan(c, info["sha256"])
            if c.get("plan_hook"):                       # tests substitute stub commands here; production leaves it None
                plan = c["plan_hook"](plan)
            if c.get("extra_negative"):
                c["extra_negative"](self)
            self.set_state(phase="preflight", plan=plan)
            (c.get("preflight_hook") or self.preflight)(plan)
            self.gpu_gate(plan)
            rc3, rcr = self.execute(info, plan)
            code = self.finish(info, plan, rc3, rcr)
            if self.dry:
                self.marker("P2S1READ_DRYRUN_OK", {"plan": plan, "final": info})
            self.set_state(phase=("dry_run_done" if self.dry else ("done" if code == 0 else "failed")))
            return code
        except Refuse as e:
            self.log(f"REFUSED/FAILED in phase {self.state.get('phase')}: {e}")
            self.marker("P2S1READ_FAILED", {"phase": self.state.get("phase"), "reason": str(e)})
            self.set_state(phase="failed")
            return 1
        except Exception as e:                           # a bug is a failure with a name, never a silent death
            import traceback
            self.log("UNEXPECTED " + traceback.format_exc()[-1500:])
            self.marker("P2S1READ_FAILED", {"phase": self.state.get("phase"), "reason": f"unexpected {type(e).__name__}: {self.scrub(e)[:300]}"})
            self.set_state(phase="failed")
            return 1


class SingleInstance:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "started": utc()}).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    holder = {}
                if holder.get("pid") and pid_alive(holder["pid"]) and holder["pid"] != os.getpid():
                    raise SystemExit(f"p2s1_autoread: another instance is running (pid {holder['pid']}, since {holder.get('started')}): {self.path}")
                try:
                    self.path.unlink()
                except OSError:
                    raise SystemExit(f"p2s1_autoread: cannot clear the stale lock {self.path}")

    def __exit__(self, *a):
        try:
            self.path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- dry run
def dry_scenario(tmp: Path):
    """five polls the chain's push pattern can really produce, ending in a consistent FINAL; plus a corrupt-download twin"""
    blob = b"fake model_final.pt " * 211 + b"\x00\x01\x02"
    sha = hashlib.sha256(blob).hexdigest()
    summ_run = json.dumps({"done": False, "steps": 600})
    summ_done = json.dumps({"done": True, "steps": 1209, "model_final": "model_final.pt", "model_final_sha256": sha})
    chain = f"2026-10-16T11:49:00 ZZP2REP_TRAIN_DONE\n2026-10-16T11:52:00 ZZP2REP_FINAL_OK model_final sha256 {sha} remote=sha256-verified\n"
    ck = {"size": 4096, "sha256": None}
    mf = {"size": len(blob), "sha256": sha}
    steps = [
        {"tree": None, "text": {}, "blobs": {}},                                                                                    # 0 no repo yet
        {"tree": {"run/summary.json": {"size": 40, "sha256": None}, "run/ckpt_last.pt": ck}, "text": {"run/summary.json": summ_run}, "blobs": {}},     # 1 training
        {"tree": {"run/summary.json": {"size": 200, "sha256": None}, "run/ckpt_last.pt": ck}, "text": {"run/summary.json": summ_done}, "blobs": {}},   # 2 done=true, no model_final yet
        {"tree": {"run/summary.json": {"size": 200, "sha256": None}, "run/model_final.pt": {"size": len(blob), "sha256": "0" * 64}},                    # 3 hashes disagree
         "text": {"run/summary.json": summ_done}, "blobs": {}},
        {"tree": {"run/summary.json": {"size": 200, "sha256": None}, "run/model_final.pt": mf, "logs/chain.log": {"size": len(chain), "sha256": None}},   # 4 FINAL
         "text": {"run/summary.json": summ_done, "logs/chain.log": chain}, "blobs": {"run/model_final.pt": blob}},
    ]
    corrupt = [dict(steps[4], blobs={"run/model_final.pt": blob[:-1] + b"\xff"})]
    return steps, corrupt, sha


def dry_run(args) -> int:
    pk = find_pkg_root(args.pkg_root)
    out = HERE / "out" / "dryrun"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    steps, corrupt, sha = dry_scenario(out)
    token = read_token([args.keys_file] if args.keys_file else KEYS_CANDS)
    cfg = make_cfg(args, pk, out=out, ck_path=str(out / "ck" / "model_final.pt"), stage_dir=str(out / "dl"), poll_s=0.05, min_free_gib=0)
    cfg["ck_path"] = str(out / "ck" / "model_final.pt").replace("\\", "/")
    ok = True

    def negative(job):                                   # a corrupt download must be refused, and leave nothing at the destination
        nonlocal ok
        bad_ck = out / "ck_corrupt" / "model_final.pt"
        try:
            download_verified(ScriptedBackend(corrupt), job.state["final"], bad_ck, out / "dl_bad", log=job.log)
            ok = False
            job.log("DRY-RUN FAIL: a corrupt download was ACCEPTED")
        except Refuse as e:
            job.log(f"dry-run negative control OK: corrupt download refused ({str(e)[:90]}..); destination exists: {bad_ck.exists()}")
            ok = ok and not bad_ck.exists()

    cfg["extra_negative"] = negative
    job = Job(cfg, ScriptedBackend(steps), dry=True, secrets=[token])
    job.log(f"token: {'found in Keys.txt (' + str(len(token)) + ' chars, never printed)' if token else 'NOT FOUND in Keys.txt'}")
    rc = job.run()
    job.log("check (the fake download landed at the checkpoint path, its sha256 equals the published one): "
            + str(Path(cfg['ck_path']).is_file() and sha256_file(cfg['ck_path']) == sha))
    ok = ok and rc == 0 and Path(cfg["ck_path"]).is_file() and sha256_file(cfg["ck_path"]) == sha
    if not args.no_live and token:                        # one real, read-only listing of the replicate's repo
        try:
            t = HFBackend(args.repo, token).tree()
            job.log("LIVE read-only listing of " + args.repo + ": " + ("repo not found / not visible yet" if t is None else
                    f"{len(t)} files; model_final.pt present: {'run/model_final.pt' in t}; marker now: " + final_marker(t, HFBackend(args.repo, token).read_text)["status"]))
        except Exception as e:
            job.log(f"LIVE listing failed ({type(e).__name__}: {job.scrub(e)[:160]}): informational in a dry run")
    job.log("DRY RUN " + ("PASS" if ok else "FAIL"))
    print("ZZP2S1AUTOREAD_DRYRUN_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def make_cfg(args, pk: Path, out=None, ck_path=None, stage_dir=None, poll_s=None, min_free_gib=None) -> dict:
    data = os.environ.get("REFE_NAVTEST_DIR", DATA_DEFAULT).replace("\\", "/")
    ck_dir = (args.ck_dir or CK_DIR_DEFAULT).replace("\\", "/")
    return {"repo": args.repo, "name": args.name, "pkg_root": str(pk), "data": data, "py": PY_T, "wait_idle": find_first(WAIT_IDLE_CANDS) or WAIT_IDLE_CANDS[0],
            "out": str(out or (HERE / "out")), "ck_path": ck_path or f"{ck_dir}/model_final.pt", "stage_dir": stage_dir or f"{ck_dir}_dl",
            "poll_s": poll_s if poll_s is not None else args.poll_min * 60, "max_wait_h": args.max_wait_h, "max_errors": args.max_errors,
            "max_inconsistent": args.max_inconsistent, "gpu_wait_max_h": args.gpu_wait_max_h, "step_timeout_h": args.step_timeout_h,
            "min_free_gib": min_free_gib if min_free_gib is not None else args.min_free_gib, "force": args.force, "extra_negative": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=REPO), ap.add_argument("--name", default=NAME)
    ap.add_argument("--pkg-root", default=None, help="the package root whose eval/ is run (default: the E: mirror stage2_sft.py is hard-wired to, then D:)")
    ap.add_argument("--ck-dir", default=None, help=f"where model_final.pt lands (default {CK_DIR_DEFAULT})")
    ap.add_argument("--keys-file", default=None, help="default: D:/Projects/TanitAD/Keys.txt, then E:")
    ap.add_argument("--poll-min", type=float, default=10.0)
    ap.add_argument("--max-wait-h", type=float, default=96.0, help="give up (FAILED) if no final marker appears in this long")
    ap.add_argument("--max-errors", type=int, default=36, help="consecutive failed polls before FAILED (36 = 6 h of an HF/network outage)")
    ap.add_argument("--max-inconsistent", type=int, default=6, help="consecutive polls with disagreeing published hashes before FAILED")
    ap.add_argument("--gpu-wait-max-h", type=float, default=24.0)
    ap.add_argument("--step-timeout-h", type=float, default=12.0)
    ap.add_argument("--min-free-gib", type=float, default=6.0)
    ap.add_argument("--force", action="store_true", help="re-run even if P2S1READ_DONE exists")
    ap.add_argument("--dry-run", action="store_true", help="poll -> (fake) download -> sha check -> command assembly; no GPU, no stage 2/3; markers go to out/dryrun/")
    ap.add_argument("--no-live", action="store_true", help="dry run: skip the one read-only live listing of the replicate's repo")
    a = ap.parse_args(argv)
    if a.dry_run:
        return dry_run(a)
    pk = find_pkg_root(a.pkg_root)
    token = os.environ.get("HF_TOKEN") or read_token([a.keys_file] if a.keys_file else KEYS_CANDS)
    if not token:
        print("p2s1_autoread: no HF token (Keys.txt unreadable or has no hf_ token)")
        return 3
    out = HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(a, pk)
    with SingleInstance(out / "autoread.lock"):
        return Job(cfg, HFBackend(a.repo, token), dry=False, secrets=[token]).run()


if __name__ == "__main__":
    sys.exit(main())
