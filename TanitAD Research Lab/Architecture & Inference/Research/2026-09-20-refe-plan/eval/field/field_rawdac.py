#!/usr/bin/env python3
"""FIELD-RAWDAC: per-proposal `navsim_dac_violation` for ONE arm's 64 proposals, in BOTH modes, on the C3 tokens.

Why: M1 of PREREG_AUX_FIELD_LOSS_P7.md = per token the share of the 64 proposals that are NAVSIM-DAC-compliant, AND the same share on the raw
plan (`refe.navsim_dac.navsim_dac_violation(mode="raw")`). No per-proposal `mode="raw"` table existed, so field_read.py v1 substituted the
transparent NAVSIM scorer (Q-F2). This builds the card's form. Both forms are kept and labelled in field_read.py.

Per token it rebuilds ego_state + map_api from the navtest DB exactly as eval/validate_navsim_dac.py does
(navtrain_scenarios.build_scenarios_for_log, history_rows=1, future_rows=80) and calls navsim_dac_violation(props[i], ego, map_api,
grid="navsim8", mode=...) twice. Output jsonl, one row per token: {token, log, pick, raw:[64], sim:[64], t_s}  (1.0 = OFF the drivable area).
`<out>.header.json` records the seam npz sha256, the rows' arm and the settings.

Subcommands
  score  (default)  --arm A --proposals <seam>.proposals.npz --out rows.jsonl [--shard i/n] [--tokens c3.json] [--limit n] [--red]
  check             --rows "rows*.jsonl" --proposals <npz> --table "full_ARM_s*of2.jsonl" [--red-rows "red*.jsonl"] [--out ctrl.json]
                    CTRL-SIM, CTRL-PICK and (with --red-rows) RED. Exit 0 only when every control PASSES.
CONTROLS (field_read.py runs CTRL-SIM and CTRL-PICK itself through ctrl_sim()/ctrl_pick(), and refuses on failure)
  CTRL-SIM   `sim` agrees with the 64-proposal table's tracked DAC (`tr.dac == 0` <=> `sim == 1`): per-proposal agreement >= SIM_BAR (the
             validator's 0.95); per-token agreement (all 64 identical) is printed with n. The number is reported whatever it is.
  CTRL-PICK  every row's `pick` equals the seam npz `pick` of the same token.
  RED        `--red` rolls the proposal axis by one; CTRL-SIM's per-token agreement must then DROP (strictly, and below its bar).
Resumable (rows without "error" are skipped); sharded by sha1(token) % n; CPU only; gated on free commit >= 5.5 GB; at most 2 processes.
Environment (the drive letters moved; these are the measured locations on this box 2026-11-20):
  REFE_NUPLAN_DB_ROOT / --db-dir   E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test      (navtest DBs, <log>.db)
  NUPLAN_MAPS_ROOT / --map-root    E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0
  --toklog                         F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz  (token -> log; else the table's `log`)
Run with the navsim-crun python, PYTHONPATH = <navsim devkit>;<PK/refe>  (';' on Windows), exactly like p5c_prop_score.py.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
MAP_ROOT = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
TOKLOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SIM_BAR = 0.95
COMMIT_FLOOR_GB = 5.5
EXPECT_PROPS = 64


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def commit_free_gb() -> float:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)"], capture_output=True, text=True)
    try:
        return float(r.stdout.strip().replace(",", "."))
    except ValueError:
        return 0.0


def wait_commit(tag: str):
    while True:
        g = commit_free_gb()
        if g >= COMMIT_FLOOR_GB:
            return g
        print(f"[{tag}] commit free {g} GB < {COMMIT_FLOOR_GB}; waiting 60 s", flush=True)
        time.sleep(60)


def load_rows(paths) -> dict:
    """token -> row, rows without "error" only. `paths` = list of files / globs."""
    out = {}
    for p in ([paths] if isinstance(paths, str) else paths):
        for fp in sorted(glob.glob(p)) or [p]:
            with open(fp, encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if "error" not in r and "token" in r:
                        out[r["token"]] = r
    return out


def load_tokens(path):
    d = json.load(open(path, encoding="utf-8"))
    return list(d["tokens"] if isinstance(d, dict) else d)


# ------------------------------------------------------------------------------------------------------------------ controls
def ctrl_sim(rows: dict, table: dict, tokens=None, bar: float = SIM_BAR) -> dict:
    """`sim` vs the table's tracked DAC. violation(sim == 1) <=> tracked dac == 0. Pure. Raises ValueError on a missing token / bad shape."""
    toks = list(tokens) if tokens is not None else sorted(rows)
    miss = [t for t in toks if t not in rows or t not in table]
    if miss:
        raise ValueError(f"CTRL-SIM: {len(miss)} tokens missing from the rawdac rows or the table (first {miss[0]})")
    S = np.array([rows[t]["sim"] for t in toks], float)
    T = np.array([table[t]["tr"]["dac"] for t in toks], float)
    if S.shape != T.shape or S.shape[1:] != (EXPECT_PROPS,):
        raise ValueError(f"CTRL-SIM: shapes sim {S.shape} vs table {T.shape}")
    ok = (S == 1.0) == (T == 0.0)
    per_prop, per_tok = float(ok.mean()), float(ok.all(1).mean())
    return {"n_tokens": len(toks), "n_proposals": int(ok.size), "per_proposal_agreement": per_prop, "per_token_agreement": per_tok,
            "violation_rate_sim": float((S == 1.0).mean()), "violation_rate_table": float((T == 0.0).mean()), "bar_per_proposal": bar,
            "pass": bool(per_prop >= bar)}


def ctrl_pick(rows: dict, pick_by_token: dict, tokens=None) -> dict:
    toks = list(tokens) if tokens is not None else sorted(rows)
    miss = [t for t in toks if t not in rows or t not in pick_by_token]
    if miss:
        raise ValueError(f"CTRL-PICK: {len(miss)} tokens missing (first {miss[0]})")
    bad = [t for t in toks if int(rows[t]["pick"]) != int(pick_by_token[t])]
    return {"n_tokens": len(toks), "n_mismatch": len(bad), "pass": not bad}


def npz_picks(npz_path: str) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    return {str(t): int(p) for t, p in zip(z["token"], z["pick"])}


# ------------------------------------------------------------------------------------------------------------------ score
def score(a) -> int:
    import navsim_dac as ND
    import navtrain_scenarios as NS
    g = wait_commit(f"{a.arm} start")
    z = np.load(a.proposals, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    P, pick = z["proposals"], z["pick"]
    assert P.shape[1:] == (EXPECT_PROPS, 8, 3), P.shape
    if a.red:
        P = np.roll(P, 1, axis=1)
    sha = sha256_file(a.proposals)
    want = set(load_tokens(a.tokens)) if a.tokens else set(toks)
    absent = sorted(want - set(toks))
    if absent:
        raise SystemExit(f"{len(absent)} requested tokens are absent from the proposals dump (first {absent[0]})")
    tlog = {}
    if os.path.exists(a.toklog):
        op = gzip.open if a.toklog.endswith(".gz") else open
        d = json.load(op(a.toklog, "rt", encoding="utf-8"))
        d = d.get("tokens", d)
        tlog = {t: (v["log_name"] if isinstance(v, dict) else v) for t, v in d.items()}
    done = set(load_rows(a.out)) if os.path.exists(a.out) else set()
    si, sn = (int(x) for x in a.shard.split("/"))
    todo = [i for i, t in enumerate(toks) if t in want and t not in done and int(hashlib.sha1(t.encode()).hexdigest(), 16) % sn == si]
    by_log: dict = {}
    for i in todo:
        by_log.setdefault(tlog.get(toks[i]), []).append(i)
    if None in by_log:
        raise SystemExit(f"{len(by_log[None])} tokens have no log in {a.toklog}")
    hdr = {"arm": a.arm, "proposals": os.path.abspath(a.proposals), "proposals_sha256": sha, "n_tokens_dump": len(toks), "n_requested": len(want),
           "tokens_file": a.tokens, "shard": a.shard, "red": bool(a.red), "grid": "navsim8", "modes": ["raw", "simulated"], "db_dir": a.db_dir,
           "map_root": a.map_root, "navsim_dac": ND.__file__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out + f".header.s{si}of{sn}.json", "w", encoding="utf-8") as fh:
        json.dump(hdr, fh, indent=1)
    print(f"[{a.arm}] commit free {g} GB; shard {a.shard}: {len(todo)} tokens in {len(by_log)} logs ({len(done)} done); seam sha256 {sha[:16]}", flush=True)
    n, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg, idx in sorted(by_log.items()):
            if a.limit and n >= a.limit:
                break
            if n and n // 200 != (n - len(idx)) // 200:
                wait_commit(f"{a.arm} {n}")
            tl0 = time.time()
            db = os.path.join(a.db_dir, f"{lg}.db")
            try:
                scs = {sc._initial_lidar_token: sc for sc in
                       NS.build_scenarios_for_log(db, [toks[i] for i in idx], map_root=a.map_root, history_rows=1, future_rows=80)}
            except Exception as e:  # noqa: BLE001
                for i in idx:
                    fo.write(json.dumps({"token": toks[i], "log": lg, "error": f"scenario build {type(e).__name__}: {str(e)[:200]}"}) + "\n")
                    n += 1
                fo.flush()
                continue
            t_build = (time.time() - tl0) / max(len(idx), 1)
            for i in idx:
                if a.limit and n >= a.limit:
                    break
                ts = time.time()
                try:
                    sc = scs.get(toks[i])
                    if sc is None:
                        raise KeyError("token not in the built scenarios (margin guard)")
                    ego = sc.get_ego_state_at_iteration(0)
                    props = np.asarray(P[i], np.float64)
                    raw = ND.navsim_dac_violation(props, ego, sc.map_api, grid="navsim8", mode="raw")
                    sim = ND.navsim_dac_violation(props, ego, sc.map_api, grid="navsim8", mode="simulated")
                    rec = {"token": toks[i], "log": lg, "pick": int(pick[i]), "raw": [float(x) for x in raw], "sim": [float(x) for x in sim],
                           "t_s": round(time.time() - ts + t_build, 3), "red": bool(a.red)}
                except Exception as e:  # noqa: BLE001
                    rec = {"token": toks[i], "log": lg, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n += 1
    print(f"ZZRAWDAC_SHARD_DONE {a.arm} {a.shard} {n} {round(time.time() - t0, 1)} s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------ check
def check(a) -> int:
    rows = load_rows(a.rows)
    table = load_rows_table(a.table)
    picks = npz_picks(a.proposals)
    out = {"n_rows": len(rows), "proposals": a.proposals, "proposals_sha256": sha256_file(a.proposals)}
    try:
        out["CTRL-SIM"] = cs = ctrl_sim(rows, table)
        out["CTRL-PICK"] = cp = ctrl_pick(rows, picks)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    ok = cs["pass"] and cp["pass"]
    print(f"CTRL-SIM  n={cs['n_tokens']} tokens, {cs['n_proposals']:,} proposals: per-proposal agreement {100 * cs['per_proposal_agreement']:.2f} % "
          f"(bar {100 * SIM_BAR:.0f} %), per-token (all 64 identical) {100 * cs['per_token_agreement']:.2f} %; violation rate sim "
          f"{100 * cs['violation_rate_sim']:.2f} % vs table {100 * cs['violation_rate_table']:.2f} % -> {'PASS' if cs['pass'] else 'FAIL'}")
    print(f"CTRL-PICK n={cp['n_tokens']} tokens, {cp['n_mismatch']} mismatches -> {'PASS' if cp['pass'] else 'FAIL'}")
    if a.red_rows:
        red = load_rows(a.red_rows)
        toks = sorted(set(red) & set(rows))
        rs = ctrl_sim(red, table, toks)
        base = ctrl_sim(rows, table, toks)
        drop = rs["per_token_agreement"] < base["per_token_agreement"] and rs["per_token_agreement"] < SIM_BAR and not rs["pass"]
        out["RED"] = {"n_tokens": len(toks), "per_token_agreement_unrolled": base["per_token_agreement"], "per_token_agreement_rolled": rs["per_token_agreement"],
                      "per_proposal_agreement_rolled": rs["per_proposal_agreement"], "pass": bool(drop)}
        print(f"RED       n={len(toks)} tokens: per-token agreement {100 * base['per_token_agreement']:.2f} % -> {100 * rs['per_token_agreement']:.2f} % rolled "
              f"(per-proposal {100 * rs['per_proposal_agreement']:.2f} %) -> {'PASS (the control drops)' if drop else 'FAIL (the control did not drop)'}")
        ok = ok and drop
    out["pass"] = bool(ok)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    print("ZZRAWDAC_CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def load_rows_table(paths) -> dict:
    return load_rows(paths)     # same shape: jsonl, "token", no "error"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--arm", required=True)
    s.add_argument("--proposals", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--shard", default="0/1")
    s.add_argument("--tokens", default=None, help="json list (or {'tokens': [...]}) of tokens; default all tokens of the dump")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--red", action="store_true", help="RED arm: roll the proposal axis by one; CTRL-SIM per-token agreement must then drop")
    s.add_argument("--db-dir", default=os.environ.get("REFE_NUPLAN_DB_ROOT", DB_DIR))
    s.add_argument("--map-root", default=os.environ.get("NUPLAN_MAPS_ROOT", MAP_ROOT))
    s.add_argument("--toklog", default=TOKLOG)
    c = sub.add_parser("check")
    c.add_argument("--rows", nargs="+", required=True)
    c.add_argument("--proposals", required=True)
    c.add_argument("--table", nargs="+", required=True)
    c.add_argument("--red-rows", nargs="+", default=None)
    c.add_argument("--out", default=None)
    argv = sys.argv[1:]
    if argv and argv[0] not in ("score", "check", "-h", "--help"):
        argv = ["score"] + argv
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return check(a)
    if a.cmd == "score":
        return score(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
