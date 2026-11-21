#!/usr/bin/env python3
"""Small, testable helpers for the P6 pod chain (refe/p6_pod.sh, SPEC_P6 D6 / D7 / G-LIVE-MEM / G-LIVE-LAB). Pure standard library + numpy; no NAVSIM stack.

Every subcommand prints ONE opaque marker line (so the chain parses a marker, never greps prose) and returns 0 only when its condition holds.

  command-table   D6 (P6-N-SRC): {"<log>/<token>": [l, s, r, u]} from the NAVSIM metadata pickles (navsim_logs/trainval/<log>.pkl, one list of frame dicts per log, each
                  carrying `token` and `driving_command`), for exactly the RANK-0, NON-AUGMENTED rows of the bank (the rows annotate_commands.py annotates).
                  ZZCMD_TABLE_OK|ZZCMD_TABLE_BAD rank0=N matched=M missing=K logs_missing=J   -- BAD (rc 1) when missing/N > --max-missing-frac (a table that does not cover the bank)
  lane-ranks      D7: the distinct lane-rank values the bank's twins carry.     ZZLANE_RANKS 1 2
  lane-distinct   D7 acceptance, from ARTIFACTS: a lane-rank twin's route (centerline / route lanes) must DIFFER from the logged route's for the same token, else
                  PDM-Closed did not honour the DriveRL route patch and the twin is labelled on the wrong route.   ZZD7_DISTINCT_OK|ZZD7_DISTINCT_BAD k= sampled= differ= frac=
  quarantine      D7 declared fallback: move every rank-K twin parts file out of the cache (the labeller then masks those samples: trajectory loss only).
  sample-tokens   an even stride over the cache's logged-route parts, as "log:token,..." (G-LIVE-LAB / D9 verify inputs).     ZZTOKENS <list>
  mem-argv        the G-LIVE-MEM trainer argv: the launch ARGS with --epochs/--resume dropped and --steps 4 at the LAUNCH --accum (4 x 64 = 256 micro-batches), a scratch --out.
  egr-ok          the chain-level fail-fast read of hf_egress.py gate's g_egr_result.json (the launch gate re-judges it).     ZZEGR_OK / ZZEGR_BAD <why>
  resume-bank     a RESUME must read the bank the run STARTED on: the `--targets` train.py recorded in <run>/config.json (args.targets, train.py:1500-1511) against the chain's CBANK.
                  ZZRESUME_BANK_SAME bank=<p> (rc 0) | ZZRESUME_BANK_MISMATCH run=<p> chain=<p> [run_n_tuples= run_targets_sha256=] (rc 3) | ZZRESUME_BANK_UNREADABLE <why> (rc 4)

P7 ONLY (P7-WIRE-RESERVE, MM_RULINGS Q7; refe/session_reserve.py).  Reached ONLY from p6_pod.sh section 5a, which runs ONLY when P7_RESERVE is set: with it unset nothing below is imported, called or read,
and every subcommand above behaves exactly as before.  One marker line each, `ZZRESERVE_<WHAT>_OK ...` (rc 0) or `ZZRESERVE_<WHAT>_BAD <why>` (rc 1).
  reserve-verify      the reserve file's sha256 against the module's pin, and its content against the pinned digests.            ZZRESERVE_OK sessions=16 logs=116 sha256=<12>
  reserve-filter      a bank directory -> its reserve-free copy (same targets_rank*.jsonl names, byte-for-byte rows, rank-0 AND twins) + reserve_filter_report.json; REFUSES a row with no resolvable log.
                      ZZRESERVE_FILTER_OK rows_in= rows_out= excluded= unresolved=0 targets_sha256=<16>
  reserve-calib       calib_table.json without the reserve's logs.                                                                ZZRESERVE_CALIB_OK logs_in= logs_out= excluded=
  reserve-move        MOVE the reserve's <log>/ dirs out of the D9 metric cache (--kind cache) or its <log>_<CAM>.zip / <log>/ out of an image root (--kind images); reversible (--restore).
                      ZZRESERVE_MOVE_OK kind= moved= already= | ZZRESERVE_RESTORE_OK restored=
  reserve-labelstore  the reserve's shards out of an aux label store into a reserve-only store (the caller re-runs `aux_label_store.py finalize` on both).   ZZRESERVE_LABELSTORE_OK moved=
  reserve-audit       zero reserve keys, and every key's log resolved, in the bank, cache, label store, image root and calibration table.   ZZRESERVE_AUDIT_OK inputs=... | ZZRESERVE_AUDIT_LEAK ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import posixpath
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

MEM_STEPS = 4                                   # 4 windows at the LAUNCH --accum (64: 256 micro-batches >= gate_checks.MEM_MIN_MICRO).  It was --accum 50 until R42:
MEM_MIN_MICRO = 200                             # the same run is G-LIVE-LAB's L1 stall reading, and the stall fraction scales like 1/accum


# ------------------------------------------------------------------------------------------------ the bank
def _bank_rows(bank):
    for fp in sorted(Path(bank).glob("targets_rank*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    continue


def rank0_keys(bank) -> set:
    """(log, token) of every rank-0, non-augmented row: the rows annotate_commands.py writes a command into."""
    return {(r["log_name"], r["token"]) for r in _bank_rows(bank) if int(r.get("rank", 0)) == 0 and not r.get("aug")}


def lane_ranks(bank) -> list:
    out = set()
    for r in _bank_rows(bank):
        aug = r.get("aug")
        if isinstance(aug, dict) and aug.get("kind") == "lane_rank":
            out.add(int(aug.get("value")))
    return sorted(out)


# ------------------------------------------------------------------------------------------------ D6
def build_command_table(logs_dir, bank, max_missing_frac=0.005) -> tuple:
    """(table, stats). The metadata pickles are the official NAVSIM/OpenScene files, loaded the way NAVSIM loads them."""
    want = rank0_keys(bank)
    by_log: dict = {}
    for lg, tok in want:
        by_log.setdefault(lg, set()).add(tok)
    table, logs_missing = {}, 0
    for lg, toks in sorted(by_log.items()):
        p = Path(logs_dir) / f"{lg}.pkl"
        if not p.is_file():
            logs_missing += 1
            continue
        with open(p, "rb") as fh:
            frames = pickle.load(fh)
        for fr in frames:
            t = fr.get("token")
            if t in toks:
                dc = fr.get("driving_command")
                if dc is not None:
                    table[f"{lg}/{t}"] = [int(x) for x in list(dc)]
    n = len(want)
    stats = {"rank0": n, "matched": len(table), "missing": n - len(table), "logs_missing": logs_missing,
             "missing_frac": (n - len(table)) / max(n, 1), "max_missing_frac": max_missing_frac}
    stats["ok"] = bool(n > 0 and stats["missing_frac"] <= max_missing_frac)
    return table, stats


# ------------------------------------------------------------------------------------------------ D7
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def canon(o, depth=0, seen=None) -> str:
    """A canonical, process-independent string of a parts element (arrays by bytes, containers recursively, objects by their __dict__, addresses stripped).
    Bounded: deeper than 6 levels, or a cycle, collapses to the type name. UNVERIFIED on the real NAVSIM parts objects until the first pod run."""
    seen = seen if seen is not None else set()
    if o is None or isinstance(o, (bool, int, float, str, bytes)):
        return repr(o)
    if depth > 6 or id(o) in seen:
        return "<" + type(o).__name__ + ">"
    seen = seen | {id(o)}
    try:
        import numpy as np
        if isinstance(o, np.ndarray) and o.dtype != object:
            return f"nd:{o.dtype.str}:{o.shape}:{hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest()}"
        if isinstance(o, np.generic):
            return repr(o.item())
    except ImportError:
        pass
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(canon(x, depth + 1, seen) for x in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(sorted(canon(k, depth + 1, seen) + ":" + canon(v, depth + 1, seen) for k, v in o.items())) + "}"
    if isinstance(o, (set, frozenset)):
        return "s{" + ",".join(sorted(canon(x, depth + 1, seen) for x in o)) + "}"
    if hasattr(o, "__dict__"):
        return type(o).__name__ + canon(vars(o), depth + 1, seen)
    return _ADDR.sub("", repr(o))


def route_digest(parts) -> str:
    """The route-defining elements of D9's parts tuple (traj, ego, obs, centerline, lanes, dam): the centreline and the route lane ids."""
    return hashlib.sha256((canon(parts[3]) + "|" + canon(parts[4])).encode("utf-8")).hexdigest()


def lane_distinct(bank, cache, k, n=40, min_frac=0.9) -> dict:
    import online_labeller as OL
    seen, sampled, differ = set(), 0, 0
    twin_rows = 0
    for r in _bank_rows(bank):
        aug = r.get("aug")
        if not (isinstance(aug, dict) and aug.get("kind") == "lane_rank" and int(aug.get("value")) == int(k)):
            continue
        twin_rows += 1
        key = (r["log_name"], r["token"])
        if key in seen:
            continue
        tw, lg = OL.parts_path(cache, key[0], key[1], aug), OL.parts_path(cache, key[0], key[1], None)
        if not (tw.is_file() and lg.is_file()):
            continue
        seen.add(key)
        sampled += 1
        differ += int(route_digest(OL.load_parts(tw)) != route_digest(OL.load_parts(lg)))
        if sampled >= n:
            break
    frac = differ / sampled if sampled else 0.0
    return {"k": int(k), "twin_rows_seen": twin_rows, "sampled": sampled, "differ": differ, "frac": round(frac, 4), "min_frac": min_frac,
            "ok": bool(sampled > 0 and frac >= min_frac)}


def quarantine(cache, k, dst) -> int:
    n = 0
    for p in sorted(Path(cache).glob(f"*/*__lane_rank_{int(k)}.pkl.xz")):
        out = Path(dst) / p.parent.name / p.name
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, out)
        n += 1
    return n


def sample_tokens(cache, n) -> list:
    """An even stride over the logged-route parts (twin files excluded): log:token."""
    allp = []
    for d in sorted(Path(cache).iterdir()) if Path(cache).is_dir() else []:
        if d.is_dir():
            allp += [(d.name, f.name[: -len(".pkl.xz")]) for f in sorted(d.iterdir()) if f.name.endswith(".pkl.xz") and "__" not in f.name]
    if not allp:
        return []
    step = max(len(allp) // max(n, 1), 1)
    return [f"{lg}:{tk}" for lg, tk in allp[::step][:n]]


# ------------------------------------------------------------------------------------------------ the trainer argv of the memory run
def _drop(argv, flag, nvalues=0):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1 + nvalues
        else:
            out.append(argv[i])
            i += 1
    return out


def _set(argv, flag, value):
    out, i, done = [], 0, False
    while i < len(argv):
        if argv[i] == flag:
            out += [flag, value]
            i += 2
            done = True
        else:
            out.append(argv[i])
            i += 1
    return out if done else out + [flag, value]


def mem_argv(args: str, out_dir: str) -> list:
    """The launch argv with the schedule cut to MEM_STEPS windows at the LAUNCH --accum (--epochs derives the step count and is dropped), no --resume,
    a scratch --out, no periodic checkpoint. Everything that sets the memory (batch, backbone, amp, compile, labeller, workers) AND the L1 stall
    (the accumulation: one drained micro-batch per window, R42) is the LAUNCH value."""
    argv = args.split()
    if "--accum" not in argv or argv.index("--accum") + 1 >= len(argv):
        raise SystemExit("the launch ARGS carry no --accum: the stall run must use the launch accumulation (R42)")
    if MEM_STEPS * int(argv[argv.index("--accum") + 1]) < MEM_MIN_MICRO:
        raise SystemExit(f"{MEM_STEPS} windows x --accum {argv[argv.index('--accum') + 1]} < {MEM_MIN_MICRO} micro-batches (G-LIVE-MEM)")
    argv = _drop(_drop(_drop(argv, "--resume"), "--epochs", 1), "--grow-scenes", 1)
    for f, v in (("--steps", str(MEM_STEPS)), ("--log-every", "1"), ("--ckpt-every-min", "999"), ("--out", out_dir)):
        argv = _set(argv, f, v)
    if any(re.search(r"\s", t) for t in argv):
        raise SystemExit("whitespace inside an argv token")
    return argv


def egr_check(path) -> tuple:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ck = d.get("ckpt") or {}
    why = []
    if d.get("pass") is not True:
        why.append("gate_not_pass")
    if not ck:
        why.append("no_ckpt_record")
    else:
        if ck.get("push_ok") is not True:
            why.append("ckpt_push_failed")
        if not ck.get("local_size") or ck.get("local_size") != ck.get("remote_size"):
            why.append(f"ckpt_size {ck.get('local_size')}!={ck.get('remote_size')}")
    return (not why), ",".join(why)


def _norm_bank_path(p) -> str:
    """A path as a string comparison needs it: forward slashes, no doubled or trailing slash, no `./`.  NOT realpath: the run's bank may be on another machine's volume."""
    s = re.sub(r"/{2,}", "/", str(p).strip().replace("\\", "/"))
    return posixpath.normpath(s) if s else s


def resume_bank(config_path, chain_bank) -> tuple:
    """(rc, marker line): does the bank this chain is about to train on name the bank the run STARTED on?

    The run's own record is `<run>/config.json`, written ONCE by train.py on a fresh start (train.py:1500-1511): `args` is `vars(argparse namespace)`, so `args.targets` is the
    `--targets` the run was launched with (parsed, so `--targets=X`, an abbreviated flag and a re-ordered argv all read the same; `argv` is the raw sys.argv and does not).
    ⛔ This compares the PATH.  That is a cheap early PROXY for the bank's identity (a stat-and-read-one-file check, run before any stage spends), NOT the authority: train.py's
    own resume identity (IDENTITY_ARGS + n_tuples / per_rank / targets_sha256, train.py:993-996 and 1036-1055) compares the bank's CONTENT and deliberately not its path, and
    that stays the content check.  The two fail differently: a bank MOVED to another path with the same bytes is refused here (point P6_CBANK at the run's recorded path and
    the trainer's content check decides), and a bank REBUILT IN PLACE under the same path passes here and is refused by the trainer's preflight.
    ⛔ A config.json that cannot be read or parsed, or that carries no `args.targets`, is never treated as 'fresh': it refuses (rc 4)."""
    cp = Path(config_path)
    try:
        cfg = json.loads(cp.read_text(encoding="utf-8"))
        run_bank = cfg["args"]["targets"]
        if not isinstance(run_bank, str) or not run_bank.strip():
            raise ValueError(f"args.targets is {run_bank!r}, not a path")
        idn = cfg.get("identity") if isinstance(cfg.get("identity"), dict) else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:      # json.JSONDecodeError and UnicodeDecodeError are ValueErrors
        return 4, f"ZZRESUME_BANK_UNREADABLE {config_path}: {type(e).__name__}: {str(e)[:160]}".replace(chr(10), " ")
    if _norm_bank_path(run_bank) == _norm_bank_path(chain_bank):
        return 0, f"ZZRESUME_BANK_SAME bank={_norm_bank_path(chain_bank)}"
    extra = ""
    if idn.get("n_tuples") is not None:
        extra += f" run_n_tuples={idn['n_tuples']}"
    if idn.get("targets_sha256"):
        extra += f" run_targets_sha256={str(idn['targets_sha256'])[:12]}"
    return 3, f"ZZRESUME_BANK_MISMATCH run={run_bank} chain={chain_bank}{extra}"


# ------------------------------------------------------------------------------------------------ P7: the session reserve (P7-WIRE-RESERVE)
def _one_line(s, n=300) -> str:
    return " ".join(str(s).split())[:n]


def reserve_cmd(a) -> int:
    """The P7-only subcommands.  `session_reserve` is imported HERE, never at module level: the P6 chain does not load it."""
    import session_reserve as SR
    tag = {"reserve-verify": "", "reserve-filter": "FILTER_", "reserve-calib": "CALIB_", "reserve-move": "MOVE_", "reserve-labelstore": "LABELSTORE_", "reserve-audit": "AUDIT_"}[a.cmd]
    try:
        if a.cmd == "reserve-move" and a.restore:
            print(f"ZZRESERVE_RESTORE_OK restored={SR.restore_moved(a.to)['restored']}", flush=True)
            return 0
        reserve = SR.load_reserve(a.reserve)                        # refuses unless the file IS the registered draw (sha256 pinned)
        if a.cmd == "reserve-verify":
            info = SR._reserve_sha_info(a.reserve)
            print(f"ZZRESERVE_OK sessions={info['sessions']} logs={info['logs']} sha256={info['file_sha256'][:12]}", flush=True)
            return 0
        if a.cmd == "reserve-filter":
            tl = SR.token_log_map(a.token_table) if a.token_table else None
            r = SR.filter_bank_dir(a.bank, a.out, reserve, token_log=tl, reserve_info=SR._reserve_sha_info(a.reserve))
            print(f"ZZRESERVE_FILTER_OK rows_in={r['n_in']} rows_out={r['n_out']} excluded={r['n_excluded']} unresolved={r['n_unresolved']} targets_sha256={r['targets_sha256'][:16]}", flush=True)
            return 0
        if a.cmd == "reserve-calib":
            r = SR.filter_calib(a.calib, a.out, reserve)
            print(f"ZZRESERVE_CALIB_OK logs_in={r['logs_in']} logs_out={r['logs_out']} excluded={r['excluded']}", flush=True)
            return 0
        if a.cmd == "reserve-move":
            r = SR.move_reserved(a.kind, a.root, a.to, reserve, dry_run=a.dry_run)
            print(f"ZZRESERVE_MOVE_OK kind={a.kind} moved={r['n_moved']} already={r['n_already']} planned={r['n_planned']}{' dry_run=1' if a.dry_run else ''}", flush=True)
            return 0
        if a.cmd == "reserve-labelstore":
            r = SR.split_label_store(a.store, a.reserve_store, reserve)
            print(f"ZZRESERVE_LABELSTORE_OK moved={r['n_moved']}", flush=True)
            return 0
        bank = sorted(Path(a.bank).glob("targets_rank*.jsonl")) if Path(a.bank).is_dir() else [Path(a.bank)]
        r = SR.audit_inputs(reserve, bank_files=bank, metric_cache_dir=a.cache, label_store=a.label_store, image_index=a.images, calib_rows=a.calib)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        c = r["counts"]
        if r["ok"]:
            print("ZZRESERVE_AUDIT_OK inputs=" + ",".join(c["inputs_audited"]) + " " + " ".join(f"{k}={v}" for k, v in sorted(c["examined"].items())), flush=True)
            return 0
        print("ZZRESERVE_AUDIT_LEAK leaks=" + ",".join(f"{k}:{v}" for k, v in sorted(c["leak_totals"].items()) if v) + " unresolved=" + ",".join(f"{k}:{v}" for k, v in sorted(r["unresolved"].items()) if v)
              + " empty=" + ",".join(c["empty_inputs"]), flush=True)
        return 1
    except (SR.ReserveError, OSError, ValueError, KeyError) as e:
        print(f"ZZRESERVE_{tag}BAD {type(e).__name__}: {_one_line(e)}", flush=True)
        return 1


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("command-table")
    p.add_argument("--logs", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-missing-frac", type=float, default=0.005)
    p = sub.add_parser("lane-ranks")
    p.add_argument("--bank", required=True)
    p = sub.add_parser("lane-distinct")
    p.add_argument("--bank", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--min-frac", type=float, default=0.9)
    p = sub.add_parser("quarantine")
    p.add_argument("--cache", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--dst", required=True)
    p = sub.add_parser("sample-tokens")
    p.add_argument("--cache", required=True)
    p.add_argument("--n", type=int, default=128)
    p = sub.add_parser("mem-argv")
    p.add_argument("--args", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("egr-ok")
    p.add_argument("--json", required=True)
    p = sub.add_parser("resume-bank")
    p.add_argument("--config", required=True)
    p.add_argument("--bank", required=True)
    # P7 only (session_reserve.py): see the module docstring
    p = sub.add_parser("reserve-verify")
    p.add_argument("--reserve", required=True)
    p = sub.add_parser("reserve-filter")
    p.add_argument("--reserve", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-table", default=None)
    p = sub.add_parser("reserve-calib")
    p.add_argument("--reserve", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("reserve-move")
    p.add_argument("--reserve", default=None)
    p.add_argument("--kind", choices=("cache", "images"), default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--to", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--restore", action="store_true")
    p = sub.add_parser("reserve-labelstore")
    p.add_argument("--reserve", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--reserve-store", required=True)
    p = sub.add_parser("reserve-audit")
    p.add_argument("--reserve", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--calib", default=None)
    p.add_argument("--label-store", default=None)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd.startswith("reserve-"):
        if a.cmd == "reserve-move" and not a.restore and not (a.reserve and a.kind and a.root):
            ap.error("reserve-move needs --reserve, --kind and --root (or --restore --to)")
        return reserve_cmd(a)
    if a.cmd == "command-table":
        table, st = build_command_table(a.logs, a.bank, a.max_missing_frac)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out).with_suffix(".tmp")
        tmp.write_text(json.dumps(table), encoding="utf-8")
        os.replace(tmp, a.out)
        print(f"{'ZZCMD_TABLE_OK' if st['ok'] else 'ZZCMD_TABLE_BAD'} rank0={st['rank0']} matched={st['matched']} missing={st['missing']} logs_missing={st['logs_missing']}", flush=True)
        return 0 if st["ok"] else 1
    if a.cmd == "lane-ranks":
        print("ZZLANE_RANKS " + " ".join(str(k) for k in lane_ranks(a.bank)), flush=True)
        return 0
    if a.cmd == "lane-distinct":
        r = lane_distinct(a.bank, a.cache, a.k, a.n, a.min_frac)
        print(f"{'ZZD7_DISTINCT_OK' if r['ok'] else 'ZZD7_DISTINCT_BAD'} k={r['k']} sampled={r['sampled']} differ={r['differ']} frac={r['frac']}", flush=True)
        return 0 if r["ok"] else 1
    if a.cmd == "quarantine":
        print(f"ZZD7_QUARANTINED {quarantine(a.cache, a.k, a.dst)}", flush=True)
        return 0
    if a.cmd == "sample-tokens":
        toks = sample_tokens(a.cache, a.n)
        print("ZZTOKENS " + ",".join(toks), flush=True)
        return 0 if toks else 1
    if a.cmd == "mem-argv":
        print("ZZMEM_ARGV " + " ".join(mem_argv(a.args, a.out)), flush=True)
        return 0
    if a.cmd == "resume-bank":
        rc, line = resume_bank(a.config, a.bank)
        print(line, flush=True)
        return rc
    ok, why = egr_check(a.json)
    print("ZZEGR_OK" if ok else f"ZZEGR_BAD {why}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
