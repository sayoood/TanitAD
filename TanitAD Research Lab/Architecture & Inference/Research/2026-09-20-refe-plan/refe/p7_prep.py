#!/usr/bin/env python3
"""Small, testable helpers for the P7 pod chain (refe/p7_pod.sh).  Standard library only, plus `online_labeller.parts_path` for `twins` (the CONSUMER's own path function, never a re-implementation).
One opaque marker line per subcommand (the chain parses a marker, never prose); rc 0 only when the condition holds.

  twins plan|apply  POST_MERGE step (d), as a CHAIN stage (R67: a procedure replayed from a document drops the step that lives in another document).  For every lane_rank row of the P7 bank
                    (`--bank`, all targets_rank*.jsonl) the twin part `<log>/<token>__lane_rank_<k>.pkl.xz` is HARD-LINKED (copy fallback, atomic + verified) from `--src` into `--dst` (the metric cache).
                    100 % COVERAGE OR STOP: a twin part missing in the source, a logged part missing in the destination, or a different file already sitting at the destination REFUSES before anything is
                    written.  `--reserve`: a bank row whose log is in the session reserve REFUSES (placing it would put a reserve key back into the metric cache, a G-RESERVE leak).  `plan` is read-only.
                    ZZP7TWINS_OK|ZZP7TWINS_BAD state= rows= distinct= to_place= present= missing_src= missing_logged= conflict= reserve_rows= ...
  noparts           the scorer-target coverage guard on the first labeller events of an attempt (R67: `no_parts` 29 of 256 sets per step for 1,730 steps, printed on every step, read by nothing).
                    Reads `<run>/metrics.jsonl` (non-JSON / NUL-padded lines are skipped, as PREREG_P6 Amendment (d) item 4 requires).  rc 0 `ZZP7NOPARTS_OK`; rc 2 `ZZP7NOPARTS_WAIT` (fewer than --events
                    labeller events so far); rc 1 `ZZP7NOPARTS_BAD` (no_parts / sets > --max-frac over the first --events events).  The guard reads the FIRST events of the file: a resumed run's
                    earlier segment is judged too, which is what a resume must not hide.
  local-check       a path (symlinks followed) must NOT sit on a network / FUSE filesystem (MEASURED 2026-10-25: the labeller stalled 141-430 s after every checkpoint save while the run dir and
                    metric cache lived on the network volume).  `--mounts` is a mounts file (default /proc/self/mounts).   ZZP7LOCAL_OK path= fstype= | ZZP7LOCAL_BAD path= fstype= why
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "ceph", "lustre", "glusterfs", "gpfs", "beegfs", "mfs", "moosefs")


def sha256_file(p, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_network_fs(fstype: str) -> bool:
    t = (fstype or "").lower()
    return t.startswith("fuse") or t in NETWORK_FS


def fs_of(path, mounts_file="/proc/self/mounts"):
    """(mount point, fstype) of the longest mount-point prefix of the REAL path; (None, None) when the mounts file cannot be read."""
    real = os.path.realpath(str(path)).replace("\\", "/")
    if len(real) > 1 and real[1] == ":":
        real = real[2:]                                                       # a drive letter (the dev box); a pod path never has one
    try:
        lines = Path(mounts_file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None
    best = (None, None)
    for ln in lines:
        f = ln.split()
        if len(f) < 3:
            continue
        mp = f[1].replace("\\040", " ")
        if real == mp or real.startswith(mp.rstrip("/") + "/") or mp == "/":
            if best[0] is None or len(mp) >= len(best[0]):
                best = (mp, f[2])
    return best


def local_check(path, mounts_file="/proc/self/mounts") -> tuple:
    mp, ft = fs_of(path, mounts_file)
    if ft is None:
        return False, f"ZZP7LOCAL_BAD path={path} fstype=unknown the mounts file {mounts_file} cannot say which filesystem this is (never guessed local)"
    if is_network_fs(ft):
        return False, f"ZZP7LOCAL_BAD path={path} fstype={ft} mount={mp} a network / FUSE filesystem: checkpoint saves stall the labeller there; put it on local disk (symlink at the same path)"
    return True, f"ZZP7LOCAL_OK path={path} fstype={ft} mount={mp}"


def _lane_rows(bank: Path):
    files = sorted(bank.glob("targets_rank*.jsonl"))
    if not files:
        raise SystemExit(f"ZZP7TWINS_BAD no targets_rank*.jsonl in {bank}")
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                if '"lane_rank"' not in ln:
                    continue
                r = json.loads(ln)
                a = r.get("aug")
                if isinstance(a, dict) and a.get("kind") == "lane_rank":
                    yield r


def twins(a) -> int:
    import online_labeller as OL
    bank, src, dst = Path(a.bank), Path(a.src), Path(a.dst)
    reserve_logs = frozenset()
    if a.reserve:
        import session_reserve as SR
        reserve_logs = SR.reserve_logs(a.reserve)
    seen, todo, pairs = set(), [], {}
    rows_n = miss_src = miss_logged = conflict = same_inode = identical = reserve_rows = 0
    first_bad = []
    ranks = set()
    for r in _lane_rows(bank):
        rows_n += 1
        ranks.add(int(r["aug"]["value"]))
        if r["log_name"] in reserve_logs:
            reserve_rows += 1
            first_bad.append(f"reserve_row:{r['log_name']}")
            continue
        s = OL.parts_path(src, r["log_name"], r["token"], r["aug"])
        d = OL.parts_path(dst, r["log_name"], r["token"], r["aug"])
        lg = OL.parts_path(dst, r["log_name"], r["token"], None)
        if d in seen:
            continue
        seen.add(d)
        if not lg.is_file():
            miss_logged += 1
            first_bad.append(f"no_logged:{lg.name}")
        if not s.is_file():
            miss_src += 1
            first_bad.append(f"no_src:{s.name}")
            continue
        pairs[d] = s
        if d.exists():
            try:
                same = os.path.samefile(s, d)
            except OSError:
                same = False
            if same:
                same_inode += 1
            elif d.stat().st_size == s.stat().st_size and sha256_file(d) == sha256_file(s):
                identical += 1
            else:
                conflict += 1
                first_bad.append(f"conflict:{d.name}")
            continue
        todo.append((s, d))
    base = (f"rows={rows_n} distinct={len(seen)} ranks={'/'.join(map(str, sorted(ranks))) or '-'} to_place={len(todo)} present={same_inode + identical} missing_src={miss_src} "
            f"missing_logged={miss_logged} conflict={conflict} reserve_rows={reserve_rows} first_bad={';'.join(first_bad[:3]) or '-'}")
    bad = bool(miss_src or miss_logged or conflict or reserve_rows or rows_n == 0)
    if rows_n == 0:
        print(f"ZZP7TWINS_BAD state=no_lane_rank_rows {base} (a P7 bank without twins is not the registered bank)")
        return 1
    if bad:
        print(f"ZZP7TWINS_BAD state={'plan' if a.mode == 'plan' else 'refused_before_writing'} {base}")
        return 1
    if a.mode == "plan":
        print(f"ZZP7TWINS_OK state=plan {base}")
        return 0
    how, linked, copied, errors, err_first = a.how, 0, 0, 0, ""
    t0 = time.time()
    for s, d in todo:
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            if how in ("auto", "link"):
                try:
                    os.link(s, d)
                    linked += 1
                    continue
                except OSError as e:
                    if how == "link":
                        raise
                    how = "copy"
                    err_first = err_first or f"link_unsupported_errno{e.errno}"
            tmp = d.with_name("." + d.name + ".p7next.tmp")
            with open(s, "rb") as fi, open(tmp, "wb") as fo:
                h = hashlib.sha256()
                while True:
                    b = fi.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    fo.write(b)
                fo.flush()
                os.fsync(fo.fileno())
            if tmp.stat().st_size != s.stat().st_size or sha256_file(tmp) != h.hexdigest():
                tmp.unlink()
                raise OSError("copy_verification_failed")
            os.replace(tmp, d)
            copied += 1
        except OSError as e:
            errors += 1
            err_first = err_first or f"{d.name}:{type(e).__name__}:{getattr(e, 'errno', '')}"
            if errors >= 5:
                break
    after_missing = sum(1 for d, s in pairs.items() if not d.is_file() or d.stat().st_size != s.stat().st_size)      # the destination, read from the filesystem, not from the loop's counters
    ok = errors == 0 and after_missing == 0
    print(f"{'ZZP7TWINS_OK' if ok else 'ZZP7TWINS_BAD'} state=apply {base} linked={linked} copied={copied} errors={errors} dst_missing_after={after_missing} err_first={err_first or '-'} wall_s={round(time.time() - t0, 1)}")
    return 0 if ok else 1


def noparts(a) -> int:
    mp = Path(a.run) / "metrics.jsonl"
    sets = nop = n = seen = 0
    if mp.is_file():
        with open(mp, "rb") as fh:
            for raw in fh:
                raw = raw.replace(b"\x00", b"").strip()
                if not raw.startswith(b"{"):
                    continue
                try:
                    r = json.loads(raw)
                except ValueError:
                    continue
                if r.get("event") != "labeller":
                    continue
                seen += 1
                if seen <= a.skip:
                    continue
                n += 1
                sets += int(r.get("sets", 0))
                nop += int(r.get("no_parts", 0))
                if n >= a.events and not a.count_only:
                    break
    if a.count_only:
        print(f"ZZP7NOPARTS_COUNT {seen}")
        return 0
    if n < a.events:
        print(f"ZZP7NOPARTS_WAIT events={n}/{a.events} sets={sets} no_parts={nop}")
        return 2
    frac = nop / max(sets, 1)
    if sets == 0 or frac > a.max_frac:
        print(f"ZZP7NOPARTS_BAD events={n} sets={sets} no_parts={nop} frac={frac:.6f} max_frac={a.max_frac} "
              f"(R67: scorer targets are missing for these samples; check the metric cache's twin parts, never a fallback)")
        return 1
    print(f"ZZP7NOPARTS_OK events={n} sets={sets} no_parts={nop} frac={frac:.6f} max_frac={a.max_frac}")
    return 0


def local_cmd(a) -> int:
    ok, msg = local_check(a.path, a.mounts)
    print(msg)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("twins")
    p.add_argument("mode", choices=("plan", "apply"))
    p.add_argument("--bank", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--reserve", default=None)
    p.add_argument("--how", choices=("auto", "link", "copy"), default="auto")
    p = sub.add_parser("noparts")
    p.add_argument("--run", required=True)
    p.add_argument("--events", type=int, default=20)
    p.add_argument("--skip", type=int, default=0, help="judge the labeller events AFTER the first SKIP ones (a resume judges its own segment, not the old one)")
    p.add_argument("--count-only", action="store_true", help="print the number of labeller events in the file (the chain's --skip for the next attempt)")
    p.add_argument("--max-frac", type=float, default=0.001)
    p = sub.add_parser("local-check")
    p.add_argument("--path", required=True)
    p.add_argument("--mounts", default="/proc/self/mounts")
    a = ap.parse_args(argv)
    return {"twins": twins, "noparts": noparts, "local-check": local_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
