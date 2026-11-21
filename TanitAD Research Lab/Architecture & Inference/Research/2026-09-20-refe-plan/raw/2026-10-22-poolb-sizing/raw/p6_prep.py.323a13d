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
    a = ap.parse_args(argv)
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
