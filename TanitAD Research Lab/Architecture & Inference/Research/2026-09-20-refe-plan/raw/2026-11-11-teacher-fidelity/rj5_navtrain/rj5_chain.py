#!/usr/bin/env python3
"""RJ5 navtrain chain: per-box, detached, resumable. For each BATCH of logs: fetch the DBs (unless they are already local) ->
roll (N single-thread processes of rj5_roll.py) -> optional dual label (p5c_dual_labels.py, unchanged) -> delete the DBs this chain
fetched -> write `batches/batch_NNNN.done` (row counts + sha256 of every output). Re-running the same command resumes.

  python rj5_chain.py --bank bank_p7_rank0_slim.jsonl.gz --logs-file logs_box2.txt --work D:/tanitad/rj5/run --db-dir D:/tanitad/rj5/db
         --procs 8 --batch-logs 12 --arms RJ5 [--control-every 20] [--label] [--fetch] [--held-db-dir E:/.../trainval] [--max-batches n]

Disk guard: before a fetch, the batch is refused (rc 6, DISK-LOW) if free space on the DB drive minus
--est-gb-per-log x (logs to fetch) is below --min-free-gb; after a fetch, the batch's DBs are deleted and the chain stops (rc 6)
if free space is below --min-free-gb. (Until 2026-11-13 --min-free-gb was declared and never read.)
Deletes ONLY `<db-dir>/<log>.db` files that this process fetched in the current batch (literal path, under --db-dir, logged).
Control arm: every --control-every-th log of a batch is also rolled at R0 (the registered jerk 1.0), whole log; `bank_dmax` of those rows is the
identity control against the bank (max |traj - bank traj|). Fallbacks (error rows) are counted per batch and kept in the rows file.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg, fp):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(fp, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def split_batches(logs, k):
    return [logs[i:i + k] for i in range(0, len(logs), k)]


def pick_control_logs(batch, every, offset=0):
    """Every `every`-th log by GLOBAL index (offset = index of the batch's first log), so small batches do not
    put a control on every batch (2026-11-13: with --batch-logs 2 the old per-batch index rolled R0 on half the logs)."""
    return [lg for i, lg in enumerate(batch) if every and (i + offset) % every == 0]


def count_rows(path):
    ok = err = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "traj" in r:
                    ok += 1
                elif "error" in r:
                    err += 1
    return ok, err


def disk_guard(free_gb, n_fetch, est_gb_per_log, min_free_gb):
    """(ok, projected_free_gb) for fetching n_fetch DBs of ~est_gb_per_log each with free_gb free now."""
    projected = free_gb - n_fetch * est_gb_per_log
    return projected >= min_free_gb, projected


def run_py(args, env, logfile):
    with open(logfile, "a", encoding="utf-8", newline="\n") as lf:
        return subprocess.Popen([sys.executable] + args, env=env, stdout=lf, stderr=subprocess.STDOUT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--logs-file", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--db-dir", required=True, help="where fetched DBs are written (and read by the rollers)")
    ap.add_argument("--held-db-dir", default=None, help="DBs already local: rolled in place, never deleted")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--batch-logs", type=int, default=12)
    ap.add_argument("--arms", default="RJ5")
    ap.add_argument("--control-every", type=int, default=20, help="also roll R0 on every k-th log of a batch (0 = no control)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--fetch-script", default=None)
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--label-script", default=None)
    ap.add_argument("--min-free-gb", type=float, default=0.8)
    ap.add_argument("--est-gb-per-log", type=float, default=0.3, help="inflated DB size assumed per fetched log (smoke 2026-11-13: 0.14 GiB)")
    ap.add_argument("--max-batches", type=int, default=0)
    a = ap.parse_args(argv)

    os.makedirs(a.work, exist_ok=True)
    bdir = os.path.join(a.work, "batches")
    os.makedirs(bdir, exist_ok=True)
    clog = os.path.join(a.work, "chain.log")
    logs = [x.strip() for x in open(a.logs_file, encoding="utf-8") if x.strip()]
    logs = sorted(set(logs))
    batches = split_batches(logs, a.batch_logs)
    log(f"RJ5_CHAIN start logs={len(logs)} batches={len(batches)} procs={a.procs} arms={a.arms} fetch={a.fetch} label={a.label}", clog)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "1"
    done_n = 0
    for bi, batch in enumerate(batches):
        marker = os.path.join(bdir, f"batch_{bi:04d}.done")
        if os.path.exists(marker):
            continue
        if a.max_batches and done_n >= a.max_batches:
            break
        bfile = os.path.join(bdir, f"batch_{bi:04d}_logs.txt")
        with open(bfile, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(batch) + "\n")
        t0 = time.time()
        held = set()
        if a.held_db_dir:
            held = {lg for lg in batch if os.path.exists(os.path.join(a.held_db_dir, lg + ".db"))}
        need = [lg for lg in batch if lg not in held]
        fetched = []
        if need:
            if not a.fetch:
                log(f"batch {bi}: {len(need)} logs have no local DB and --fetch is off: they will be listed nodb", clog)
            else:
                ok_disk, proj = disk_guard(shutil.disk_usage(a.db_dir).free / 2**30, len(need), a.est_gb_per_log, a.min_free_gb)
                if not ok_disk:
                    log(f"batch {bi}: DISK-LOW before fetch: projected free {proj:.2f} GB < {a.min_free_gb} GB; stop (re-run resumes)", clog)
                    return 6
                nfile = os.path.join(bdir, f"batch_{bi:04d}_fetch.txt")
                with open(nfile, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\n".join(need) + "\n")
                rc = subprocess.run([sys.executable, a.fetch_script, "--out", a.db_dir, "--logs-file", nfile,
                                     "--held-root", "/nonexistent", "--jobs", "8"], env=env, encoding="utf-8",
                                    stdout=open(os.path.join(a.work, f"fetch_{bi:04d}.log"), "w", encoding="utf-8"),
                                    stderr=subprocess.STDOUT).returncode
                fetched = [os.path.join(a.db_dir, lg + ".db") for lg in need if os.path.exists(os.path.join(a.db_dir, lg + ".db"))]
                log(f"batch {bi}: fetch rc={rc} got={len(fetched)}/{len(need)} "
                    f"free={shutil.disk_usage(a.db_dir).free / 2**30:.2f} GB", clog)
                if rc != 0:
                    log(f"batch {bi}: FETCH FAILED; stop (re-run resumes)", clog)
                    return 4
                free_now = shutil.disk_usage(a.db_dir).free / 2**30
                if free_now < a.min_free_gb:
                    for p in fetched:
                        if os.path.dirname(os.path.abspath(p)) == os.path.abspath(a.db_dir) and p.endswith(".db"):
                            os.remove(p)
                    log(f"batch {bi}: DISK-LOW after fetch: free {free_now:.2f} GB < {a.min_free_gb} GB; deleted {len(fetched)} fetched DBs; stop", clog)
                    return 6
        # DB dir the rollers read: held logs may live elsewhere, so give each roller a merged view via two --db-dir passes is avoided:
        # held DBs are only used when --held-db-dir == --db-dir (dev box); otherwise rolled in two sub-batches below.
        groups = []
        if held:
            groups.append((a.held_db_dir, sorted(held)))
        if fetched or (need and not a.fetch):
            groups.append((a.db_dir, sorted(need)))
        outdir = os.path.join(a.work, "rows")
        os.makedirs(outdir, exist_ok=True)
        procs = []
        for gi, (dbd, glogs) in enumerate(groups):
            gfile = os.path.join(bdir, f"batch_{bi:04d}_g{gi}_logs.txt")
            with open(gfile, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(glogs) + "\n")
            ctl = pick_control_logs(glogs, a.control_every, bi * a.batch_logs)
            for si in range(a.procs):
                tag = f"b{bi:04d}g{gi}s{si}"
                procs.append(run_py([os.path.join(HERE, "rj5_roll.py"), "--bank", a.bank, "--db-dir", dbd, "--out-dir", outdir,
                                     "--arms", a.arms, "--logs-file", gfile, "--shard", f"{si}/{a.procs}", "--tag", tag],
                                    env, os.path.join(a.work, f"roll_{tag}.log")))
            if ctl and "R0" not in a.arms.split(","):
                cfile = os.path.join(bdir, f"batch_{bi:04d}_g{gi}_ctl.txt")
                with open(cfile, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\n".join(ctl) + "\n")
                tag = f"b{bi:04d}g{gi}ctl"
                procs.append(run_py([os.path.join(HERE, "rj5_roll.py"), "--bank", a.bank, "--db-dir", dbd, "--out-dir", outdir,
                                     "--arms", "R0", "--logs-file", cfile, "--shard", "0/1", "--tag", tag],
                                    env, os.path.join(a.work, f"roll_{tag}.log")))
        rcs = [p.wait() for p in procs]
        rows_files = sorted(f for f in os.listdir(outdir) if f.startswith(f"rj5_b{bi:04d}"))
        ok = err = 0
        for f in rows_files:
            o, e = count_rows(os.path.join(outdir, f))
            ok += o
            err += e
        log(f"batch {bi}: roll rcs={rcs} rows_ok={ok} rows_err={err}", clog)
        if a.label:
            log(f"batch {bi}: label step is run by rj5_label.py (see PLAN); skipped here", clog)
        # delete only the DBs this batch fetched
        freed = 0
        for p in fetched:
            if os.path.dirname(os.path.abspath(p)) == os.path.abspath(a.db_dir) and p.endswith(".db"):
                freed += os.path.getsize(p)
                os.remove(p)
        log(f"batch {bi}: deleted {len(fetched)} fetched DBs, {freed / 2**20:.0f} MB", clog)
        if any(rc not in (0, 3) for rc in rcs):
            log(f"batch {bi}: a roller died (rcs {rcs}); NOT marked done, stop (re-run resumes from the rows files)", clog)
            return 5
        man = {"batch": bi, "logs": len(batch), "rows_ok": ok, "rows_err": err, "roll_rcs": rcs, "sec": round(time.time() - t0, 1),
               "files": {f: sha256_file(os.path.join(outdir, f)) for f in rows_files}}
        with open(marker, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1)
        done_n += 1
    log("ZZRJ5_CHAIN_DONE", clog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
