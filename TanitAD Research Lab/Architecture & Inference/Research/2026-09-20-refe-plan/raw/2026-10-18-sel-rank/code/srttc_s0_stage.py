#!/usr/bin/env python3
"""SR-TTC S0 -- STAGING of one batch of fit logs from the pod to the dev box (nuPlan DBs + the needed camera frames), and the deletion of the staged copies once the batch is banked.

Uses `podcopy.py` (derived from the held-out completion run's copier; the diff is in its docstring).  SCOPE CHANGE (coordinator, 2026-10-18): P6 trains on the pod for ~7 days, so pod READS are
allowed while it trains, under P6's overlap rule: (1) the pod-side reader runs as `nice -n 19 ionice -c3`; (2) ONE stream at a time; (3) no pod-side python / hashing / other CPU work: tar, tail,
stat and ls only; (4) before EVERY 120 MB batch the tail of P6's chain log is read and a `P6_GUARD` line makes this module skip the batch and retry in 10 min.  Never touches P6's processes
or locks, never stages onto the pod.  Staged copies live on D: (`STAGE_ROOT`).

Pod access: the ssh prefix is read at run time from the file named in pod_cfg.local; it lives only in this process' environment and the child's and is NEVER written anywhere.
Staged (and therefore deletable) files are exactly the ones listed in `copy_manifest_*.tsv` of the batch: files that were already on the dev box before are never touched.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
POD_CFG = "C:/Users/Admin/srttc_s0/pod_cfg.local"      # one line: the path of the file that holds the ssh prefix (kept out of the repo on purpose)
POD_DBS = "/workspace/data/navtrain_dbs"
POD_PIX = "/workspace/data/navtrain_pixels"
PODCOPY = os.path.join(HERE, "podcopy.py")
P6_CHAIN_LOG = "/workspace/data/refe_p6/chain.log"
NICE = "nice -n 19 ionice -c3"           # coordinator 2026-10-18: pod reads run at the lowest CPU and IO priority, one stream at a time
STAGE_ROOT = "D:/Projects/TanitAD-artifacts/srttc_s0_stage"        # coordinator 2026-10-18: D: has ~197 GB free; keep >= 20 GB free on D: and >= 5 GB on C:
MIN_FREE_GB_STAGE = 21.0                # podcopy refuses a batch that would leave less than this on the staging drive (D:)
RETRY_P6_S = 600                        # a P6_GUARD line in P6's chain log: skip the batch, retry in 10 min

# remote: stdin = '<log>/<CAM>/<hash>.jpg' lines; stdout = '<bytes>\t<shard>\t<rel>' or 'MISSING\t\t<rel>'   (= copy/pod_stat.sh of the held-out completion run, inline)
_STAT_FRAMES = ('declare -A SH; while IFS= read -r rel; do [ -z "$rel" ] && continue; log=${rel%%/*}; '
                'if [ -z "${SH[$log]+x}" ]; then d=$(ls -d ' + POD_PIX + '/*/"$log" 2>/dev/null | head -1); SH[$log]=${d%/*}; fi; sh=${SH[$log]}; f="$sh/$rel"; '
                'if [ -f "$f" ]; then printf "%s\\t%s\\t%s\\n" "$(stat -c %s "$f")" "${sh##*/}" "$rel"; else printf "MISSING\\t\\t%s\\n" "$rel"; fi; done')
_STAT_DBS = ('while IFS= read -r rel; do [ -z "$rel" ] && continue; f=' + POD_DBS + '/"$rel"; '
             'if [ -f "$f" ]; then printf "%s\\t%s\\n" "$(stat -c %s "$f")" "$rel"; else printf "MISSING\\t%s\\n" "$rel"; fi; done')


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class PodError(Exception):
    pass


class Stager:
    def __init__(self, run_dir, work, python, log=print, stop_event=None, status=None, stage_root=STAGE_ROOT):
        self.run, self.work, self.python, self.log, self.status = run_dir, work, python, log, status
        self.stage_root = stage_root
        self.stop = stop_event
        self.env = dict(os.environ, PODSSH=open(open(POD_CFG, encoding="utf-8").read().strip(), encoding="utf-8").read().strip())
        import shlex
        self.pre = shlex.split(self.env["PODSSH"])
        self.stage_db_dir = os.path.join(stage_root, "dbs")
        os.makedirs(self.stage_db_dir, exist_ok=True)

    # ---- low-level
    def _ssh(self, cmd, stdin=b"", timeout=900):
        r = subprocess.run(self.pre + [cmd], input=stdin, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")

    def p6_ok(self):
        """(ok, line): False when the last 20 lines of P6's chain log show `P6_GUARD` (a guard PAUSE or slowdown) or cannot be read.  Read-only `tail`, lowest priority."""
        try:
            rc, out, _ = self._ssh(f"{NICE} tail -n 20 {P6_CHAIN_LOG}", timeout=90)
        except subprocess.SubprocessError:
            return False, "ssh timeout"
        if rc != 0:
            return False, f"ssh/tail rc={rc}"
        for ln in out.splitlines():
            if "P6_GUARD" in ln:
                return False, ln.strip()[:200]
        return True, ""

    def wait_p6(self, what):
        while True:
            if self.stop is not None and self.stop.is_set():
                raise PodError("aborted")
            ok, why = self.p6_ok()
            if ok:
                return
            msg = f"staging SKIPS {what}: P6 chain log shows a guard line / unreadable ({why}); retry in {RETRY_P6_S // 60 if 'P6_GUARD' in why else 1} min"
            self.log(f"[{now()}] {msg}")
            if self.status:
                self.status(msg)
            time.sleep(RETRY_P6_S if "P6_GUARD" in why else 60)

    def _podcopy(self, lst, dest, manifest, what):
        """run podcopy.py; waits and retries on 'pod busy' (3) / ssh failure (6); a sha mismatch (4) is fatal."""
        for attempt in range(1, 100000):
            rc = subprocess.call([self.python, PODCOPY, "--list", lst, "--dest", dest, "--manifest", manifest, "--max-batch-mb", "120", "--min-free-gb", str(MIN_FREE_GB_STAGE), "--p6-log", P6_CHAIN_LOG],
                                 env=self.env, stdout=open(os.path.join(self.run, "logs", "podcopy.log"), "a", encoding="utf-8"), stderr=subprocess.STDOUT)
            if rc == 0:
                return
            if rc == 4:
                raise PodError(f"{what}: size mismatch (podcopy rc 4): REPORT, do not retry")
            if rc == 5:
                raise PodError(f"{what}: disk guard (podcopy rc 5): free space on the staging drive < {MIN_FREE_GB_STAGE} GiB after the batch")
            msg = f"podcopy {what} rc={rc} (3 = P6_GUARD in P6's chain log, 6 = ssh): waiting {RETRY_P6_S // 60 if rc == 3 else 1} min (attempt {attempt}); resumable, complete files are skipped"
            self.log(f"[{now()}] {msg}")
            if self.status:
                self.status(msg)
            if self.stop is not None and self.stop.is_set():
                raise PodError("aborted")
            time.sleep(RETRY_P6_S if rc == 3 else 60)
        raise PodError(f"{what}: podcopy kept failing")

    # ---- audit (subprocess in the driverl-eval interpreter)
    def audit(self, batch_dir, tokens_json, frame_roots):
        cmd = [self.python, "-I", os.path.join(HERE, "srttc_s0_audit.py"), "--tokens-json", tokens_json, "--out", batch_dir, "--stage-db-dir", self.stage_db_dir]
        for r in frame_roots:
            cmd += ["--images-root", r]
        with open(os.path.join(batch_dir, "audit.log"), "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if rc:
            raise PodError(f"audit rc={rc} ({batch_dir}/audit.log)")
        return {n: json.load(open(os.path.join(batch_dir, n), encoding="utf-8")) for n in ("audit.json", "needed_frames.json", "needed_dbs.json")}

    # ---- one batch
    def stage(self, bid, logs, tokens):
        """-> dict(plan=<batch_tokens.json path>, frames_root, copied_dbs=[paths], bytes_staged, missing_on_pod={...}).  tokens = {token: log} of the batch."""
        bd = os.path.join(self.run, "batches", bid)
        os.makedirs(bd, exist_ok=True)
        tj = os.path.join(bd, "tokens_in.json")
        json.dump({"tokens": tokens}, open(tj, "w", encoding="utf-8"))
        frames_root = os.path.join(self.stage_root, "frames", bid)
        os.makedirs(frames_root, exist_ok=True)
        info = {"batch": bid, "logs": logs, "frames_root": frames_root, "copied_dbs": [], "missing_dbs_on_pod": [], "missing_frames_on_pod": 0, "bytes_staged": 0, "staged_at": now()}
        a1 = self.audit(bd, tj, [frames_root])
        # ---- 1. DBs for the logs with none on this box
        need_db = a1["needed_dbs.json"]["logs"]
        if need_db:
            self.wait_p6(f"DB stat of {bid}")
            rc, out, err = self._ssh(f"{NICE} bash -c {shlex.quote(_STAT_DBS)}", "\n".join(f"{lg}.db" for lg in need_db).encode() + b"\n")
            if rc:
                raise PodError(f"DB stat rc={rc} {err[:200]!r}")
            rows, miss = [], []
            for ln in out.splitlines():
                p = ln.split("\t")
                if p[0] == "MISSING":
                    miss.append(p[1][:-3])
                elif len(p) == 2:
                    rows.append((int(p[0]), POD_DBS, p[1]))
            info["missing_dbs_on_pod"] = miss
            if rows:
                lst = os.path.join(bd, "copy_list_dbs.tsv")
                open(lst, "w", encoding="utf-8", newline="\n").write("".join(f"{b}\t{base}\t{rel}\n" for b, base, rel in rows))
                self._podcopy(lst, self.stage_db_dir, os.path.join(bd, "copy_manifest_dbs.tsv"), f"DBs of {bid}")
                info["copied_dbs"] = [os.path.join(self.stage_db_dir, r[2]).replace("\\", "/") for r in rows]
                info["bytes_staged"] += sum(r[0] for r in rows)
        a2 = self.audit(bd, tj, [frames_root]) if need_db else a1
        # ---- 2. frames for the tokens whose DB is here but whose four frames are not
        nf = a2["needed_frames.json"]["tokens"]
        rels = sorted({fn for v in nf.values() for fn in v["frames_filename_jpg"].values()})            # '<log>/<CAM>/<hash>.jpg' exactly as the DB names them
        if rels:
            self.wait_p6(f"frame stat of {bid}")
            rc, out, err = self._ssh(f"{NICE} bash -c {shlex.quote(_STAT_FRAMES)}", "\n".join(rels).encode() + b"\n")
            if rc:
                raise PodError(f"frame stat rc={rc} {err[:200]!r}")
            rows, nmiss = [], 0
            for ln in out.splitlines():
                p = ln.split("\t")
                if p[0] == "MISSING":
                    nmiss += 1
                elif len(p) == 3:
                    rows.append((int(p[0]), f"{POD_PIX}/{p[1]}", p[2]))
            info["missing_frames_on_pod"] = nmiss
            if rows:
                lst = os.path.join(bd, "copy_list_frames.tsv")
                open(lst, "w", encoding="utf-8", newline="\n").write("".join(f"{b}\t{base}\t{rel}\n" for b, base, rel in rows))
                self._podcopy(lst, frames_root, os.path.join(bd, "copy_manifest_frames.tsv"), f"frames of {bid}")
                info["bytes_staged"] += sum(r[0] for r in rows)
        # ---- 3. the final plan of this batch
        a3 = self.audit(bd, tj, [frames_root])
        info["plan"] = os.path.join(bd, "batch_tokens.json")
        info["n_tokens_in"], info["n_tokens_ok"] = len(tokens), a3["audit.json"]["n_tokens_ok"]
        info["blocked_reasons"] = a3["audit.json"]["blocked_reasons"]
        json.dump(info, open(os.path.join(bd, "stage_info.json"), "w", encoding="utf-8"), indent=1)
        return info

    def audit_only(self, b):
        """smoke without the pod: the plan of the locally available tokens of the batch (no copy)."""
        bd = os.path.join(self.run, "batches", b["id"])
        os.makedirs(bd, exist_ok=True)
        tj = os.path.join(bd, "tokens_in.json")
        json.dump({"tokens": b["tokens"]}, open(tj, "w", encoding="utf-8"))
        a3 = self.audit(bd, tj, [])
        return {"batch": b["id"], "logs": b["logs"], "frames_root": os.path.join(self.stage_root, "frames", b["id"]), "copied_dbs": [], "plan": os.path.join(bd, "batch_tokens.json"),
                "n_tokens_in": len(b["tokens"]), "n_tokens_ok": a3["audit.json"]["n_tokens_ok"], "blocked_reasons": a3["audit.json"]["blocked_reasons"], "bytes_staged": 0}

    def estimate_bytes(self, logs, tokens_by_log):
        """a conservative ALLOCATED-disk estimate of a batch BEFORE copying: mean 146 MB per DB (A2.3, MEASURED over 240 local fit-log DBs; the worst single DB seen is ~1 GB) + 4.2 MB of frames per token
        (4 jpgs of ~0.2 MB, each rounded up to D:'s ~1 MB allocation unit: exFAT, the project notes measured 5 GB of objects occupying ~38 GB)."""
        n_db = sum(1 for lg in logs if not os.path.exists(os.path.join(self.stage_db_dir, f"{lg}.db")))
        return int(n_db * 200e6 + sum(tokens_by_log[lg] for lg in logs) * 4.2e6)

    def cleanup(self, info):
        """delete the staged copies of a banked batch: exactly the files this stager copied (never a pre-existing file)."""
        n, freed = 0, 0
        for p in list(info.get("copied_dbs", [])) + [os.path.join(self.stage_db_dir, f"{lg}.db") for lg in info.get("logs", [])]:       # the stage dir is OURS: every DB in it was copied by a stager
            ap = os.path.abspath(p)
            if os.path.dirname(ap) == os.path.abspath(self.stage_db_dir) and os.path.isfile(ap):
                freed += os.path.getsize(ap)
                os.remove(ap)
                n += 1
        fr = os.path.abspath(info["frames_root"])
        if os.path.isdir(fr) and os.path.abspath(os.path.join(self.stage_root, "frames")) == os.path.dirname(fr):
            for dp, _dn, fns in os.walk(fr):
                for fn in fns:
                    freed += os.path.getsize(os.path.join(dp, fn))
                    n += 1
            shutil.rmtree(fr, ignore_errors=True)
        return n, freed
