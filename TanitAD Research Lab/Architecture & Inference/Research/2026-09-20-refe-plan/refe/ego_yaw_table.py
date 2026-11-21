#!/usr/bin/env python3
"""A TOKEN -> REAL YAW RATE TABLE for REFe's ego slot 4 (WP YAW, 2026-11-06; PLAN: TanitAD-artifacts/yawrate_wp/PLAN.md).

WHY. REFe's 7-D ego vector is [vx, vy, ax, ay, yaw_rate, tire_steering_angle, speed]. Slots 4 and 5 are 0.0 in EVERY banked sample
(D-REFE-DEADSLOTS-1): nuPlan's DB loader builds `EgoState.build_from_rear_axle(...)` from ego_pose's x, y, qw..qz, vx, vy, acceleration_x,
acceleration_y ONLY (nuplan_scenario_queries.py `get_ego_state_for_lidarpc_token_from_db` / `get_sampled_ego_states_from_db`), so the
`angular_velocity` argument keeps its default 0.0 and `tire_steering_angle` is hard-coded 0.0. The DB HAS the yaw rate: `ego_pose.angular_rate_z`.
This module reads it with THE SAME lidar_pc -> ego_pose join the loader uses, and banks it as a gzipped JSON `{token: angular_rate_z}`.

UNITS (MEASURED by `units` below against d(heading)/dt of the same DB's quaternions): rad/s, CCW-positive, z up, ego/body frame.
THE KEY is the lowercase-hex `lidar_pc.token` -- the bank row's `token` field (build_teacher_rollouts.make_row: `sc.scenario_name` ==
`NuPlanScenario.token` == `_initial_lidar_token`; navtrain rows have step 0).  It is the NAVSIM token, so one table serves a bank and an eval.

THE CONTROL (mandatory, `control`): the SAME join's vx, vy, acceleration_x, acceleration_y must equal -- bit for bit, float64 -- (a) what the
bank's scenario builder hands the trainer (`navtrain_scenarios.build_scenarios_for_log` -> `scenario.get_ego_state_at_iteration(0)` ->
`dynamic_car_state.rear_axle_velocity_2d / _acceleration_2d`), (b) the devkit query itself, and (c) where a banked row exists, the row's
`ego[0..3]`.  A join that disagrees on vx is the WRONG ROW and the table is refused.

    python ego_yaw_table.py build   --split navtrain --yaml <navtrain.yaml> --db-root <splits> --out table_navtrain.json.gz [--allow-partial]
    python ego_yaw_table.py build   --split navtest  --yaml <navtest.yaml>  --db-root <splits> --out table_navtest.json.gz
    python ego_yaw_table.py control --table table_navtrain.json.gz --db-root <splits> --n 240 --min-logs 12 [--bank-dir <dir>]
    python ego_yaw_table.py units   --table table_navtrain.json.gz --db-root <splits> --logs 12
  <splits> is searched as splits/*.db (top-level; a pod's flat navtrain_dbs/ too), then trainval/, test/, mini/ -- coverage is reported per location.
  THE TABLE IS RAW rad/s.  The trainer feeds ego[4] = YAW_SCALE (10.0, defined once, below) * the table value; the checkpoint meta records `ego_yaw_scale`.
    python ego_yaw_table.py check-bank --table table_navtrain.json.gz --bank-dir <dir>      # every bank row's token must be in the table
    python ego_yaw_table.py info    --table table_navtrain.json.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCHEMA = "ego-yaw-table-v1"
FIELD = "ego_pose.angular_rate_z"
YAW_TAG = "nuplan_angular_rate_z"                  # the checkpoint meta["ego_yaw"] value
EGO_YAW_SLOT = 4                                   # REFeConfig.ego_dim comment: vx, vy, ax, ay, YAW_RATE, steering, speed
# MM DECISION 2026-11-06 (YF measured): REFe's ego inputs are RAW (speed ~10, ax/ay ~1) and the measured yaw rate is |w| <= ~0.35 rad/s, i.e. 0.18 % of the ego_enc pre-activation spread.
# The trainer therefore feeds ego[4] = YAW_SCALE * angular_rate_z.  ONE definition, here: the trainer, the shim (ego_yaw.py) and the tests IMPORT it, nothing re-types the number; the
# checkpoint's meta carries it as `ego_yaw_scale`, and the shim multiplies by the META's value (a checkpoint without one is refused).  The TABLE stays in raw rad/s.
YAW_SCALE = 10.0
# WHERE THE nuPlan DBs LIVE (MEASURED by YF 2026-11-06: 193 of the proxy's 407 logs are LOOSE top-level files): beside the `trainval/`, `test/` and `mini/` directories there are loose
# `splits/*.db`.  A log is resolved in THIS order, and coverage is reported per location.  A pod's flat `navtrain_dbs/` is the top-level location.
LOCATIONS = ("top", "trainval", "test", "mini")
UNITS = "rad/s, CCW-positive (z up), ego body frame"
# THE JOIN. Byte-for-byte the FROM / JOIN of the devkit's get_ego_state_for_lidarpc_token_from_db (nuplan_scenario_queries.py:559-600), with
# `angular_rate_z` ADDED to the select list and `lp.token` selected instead of filtered on, so ONE pass reads a whole log.
JOIN_SQL = """
    SELECT  lower(hex(lp.token)) AS token,
            ep.angular_rate_z,
            ep.vx,
            ep.vy,
            ep.acceleration_x,
            ep.acceleration_y
    FROM ego_pose AS ep
    INNER JOIN lidar_pc AS lp
        ON lp.ego_pose_token = ep.token
"""


class TableRefused(Exception):
    """a table that is malformed, mis-hashed, or does not cover what it was asked to"""


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def f64_bits(x: float) -> bytes:
    return struct.pack("<d", float(x))


# ------------------------------------------------------------------------------------------------ the join
def ro_connect(db_path: str) -> sqlite3.Connection:
    """READ-ONLY (`mode=ro`): the table builder must never be able to write to a nuPlan DB."""
    uri = "file:" + Path(db_path).resolve().as_posix() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def index_dbs_located(db_root: str) -> dict:
    """{log_name: (absolute .db path, location)} over splits/*.db (top), trainval/, test/, mini/ -- first hit in that order wins.  A log present in TWO locations is recorded under
    `index_dbs_located.duplicates` and refused if the two files differ in size (one of them is then not the same log)."""
    out: dict = {}
    dups: dict = {}
    for loc in LOCATIONS:
        pat = os.path.join(db_root, "*.db") if loc == "top" else os.path.join(db_root, loc, "*.db")
        for pth in sorted(glob.glob(pat)):
            lg = os.path.basename(pth)[:-3]
            if lg in out:
                dups.setdefault(lg, [out[lg][1]]).append(loc)
                if os.path.getsize(pth) != os.path.getsize(out[lg][0]):
                    raise TableRefused(f"log {lg} is in {out[lg][1]} AND {loc} with DIFFERENT sizes ({os.path.getsize(out[lg][0])} vs {os.path.getsize(pth)}): not the same log")
                continue
            out[lg] = (os.path.abspath(pth), loc)
    index_dbs_located.duplicates = dups
    return out


index_dbs_located.duplicates = {}


def join_rows(db_path: str, want: set | None = None) -> dict:
    """{lidar_pc token (lowercase hex): (angular_rate_z, vx, vy, acceleration_x, acceleration_y)} for one log, `want` restricting the tokens kept.
    A token that appears twice in one DB with DIFFERENT values is refused (the join is then not a function of the token)."""
    con = ro_connect(db_path)
    out: dict = {}
    try:
        for tok, wz, vx, vy, ax, ay in con.execute(JOIN_SQL):
            if want is not None and tok not in want:
                continue
            row = (wz, vx, vy, ax, ay)
            if tok in out and out[tok] != row:
                raise TableRefused(f"{db_path}: token {tok} maps to two different ego_pose rows")
            out[tok] = row
    finally:
        con.close()
    return out


# ------------------------------------------------------------------------------------------------ the table file
def save(path, table: dict, meta: dict) -> str:
    """Write the gzipped JSON deterministically (sorted keys, mtime 0, no wall clock inside) and return its sha256."""
    for k, v in table.items():
        if not isinstance(v, float) or not math.isfinite(v):
            raise TableRefused(f"token {k}: value {v!r} is not a finite float")
    meta = dict(meta, schema=SCHEMA, field=FIELD, units=UNITS, n_tokens=len(table))
    blob = json.dumps({"meta": meta, "table": table}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(blob)
    return sha256_file(path)


def load(path, expect_sha256: str | None = None):
    """(table {token: float}, meta {...}, sha256 of the FILE).  Refuses a file whose schema, token count or values are not what `save` wrote."""
    sha = sha256_file(path)
    if expect_sha256 is not None and sha != expect_sha256:
        raise TableRefused(f"{path}: sha256 {sha[:12]} != the pinned {expect_sha256[:12]}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    meta, table = d.get("meta") or {}, d.get("table")
    if meta.get("schema") != SCHEMA or not isinstance(table, dict):
        raise TableRefused(f"{path}: not an {SCHEMA} file (schema {meta.get('schema')!r})")
    if meta.get("n_tokens") != len(table):
        raise TableRefused(f"{path}: meta says {meta.get('n_tokens')} tokens, the table holds {len(table)}")
    bad = [k for k, v in table.items() if not isinstance(v, (int, float)) or not math.isfinite(v)]
    if bad:
        raise TableRefused(f"{path}: {len(bad)} non-finite values, e.g. {bad[:3]}")
    return table, meta, sha


# ------------------------------------------------------------------------------------------------ build
def ask_set(split: str, yaml_path: str):
    """(log_names, tokens) of the NAVSIM split file -- the SAME reader the bank builder uses (navtrain_scenarios.load_split)."""
    import navtrain_scenarios as NS
    logs, toks = NS.load_split(yaml_path)
    return list(logs), set(toks)


def build(split: str, yaml_path: str, db_root: str, limit_logs: int | None = None, tokens: set | None = None):
    """-> (table, meta, report).  `asked` = every token of the split file (or `tokens`, an explicit bank/eval token set); `found` = those whose lidar_pc row exists in a DB we hold.
    Logs are resolved over splits/*.db, trainval/, test/, mini/ (LOCATIONS) and the report carries coverage PER LOCATION."""
    logs, want = ask_set(split, yaml_path)
    if tokens is not None:
        want = set(tokens)
    local = index_dbs_located(db_root)
    logset = set(logs)
    covered = sorted(logset & set(local)) if logs else sorted(local)
    if limit_logs:
        covered = covered[:limit_logs]
    table: dict = {}
    per_log: dict = {}
    by_loc: dict = {loc: {"logs_indexed": 0, "logs_in_split_file": 0, "logs_read": 0, "tokens_found": 0} for loc in LOCATIONS}
    for lg, (_p, loc) in local.items():
        by_loc[loc]["logs_indexed"] += 1
        by_loc[loc]["logs_in_split_file"] += int(lg in logset) if logs else 0
    n_zero = 0
    for lg in covered:
        path, loc = local[lg]
        rows = join_rows(path, want)
        for tok, (wz, *_rest) in rows.items():
            if tok in table and f64_bits(table[tok]) != f64_bits(wz):
                raise TableRefused(f"token {tok} appears in two logs with different angular_rate_z (log {lg})")
            table[tok] = float(wz)
            n_zero += int(wz == 0.0)
        per_log[lg] = len(rows)
        by_loc[loc]["logs_read"] += 1
        by_loc[loc]["tokens_found"] += len(rows)
    missing = sorted(want - set(table))
    vals = sorted(table.values())
    nz = len(vals)
    absv = sorted(abs(v) for v in vals)
    q = lambda xs, p: xs[min(int(p * (len(xs) - 1) + 0.5), len(xs) - 1)] if xs else None   # noqa: E731
    not_held = sorted(logset - set(local)) if logs else []
    report = {
        "split": split, "yaml": str(yaml_path), "yaml_sha256": sha256_file(yaml_path), "db_root": str(db_root), "location_order": list(LOCATIONS),
        "tokens_asked": len(want), "tokens_found": len(table), "tokens_missing": len(missing),
        "coverage": (len(table) / len(want)) if want else None, "partial": bool(missing),
        "logs_in_split_file": len(logs), "logs_held_locally": len(logset & set(local)) if logs else None, "logs_not_held": len(not_held), "logs_not_held_examples": not_held[:3],
        "logs_with_tokens": sum(1 for v in per_log.values() if v), "n_value_exactly_zero": n_zero, "by_location": by_loc,
        "logs_in_more_than_one_location": len(index_dbs_located.duplicates), "missing_examples": missing[:5],
        "stats": {"min": vals[0] if nz else None, "max": vals[-1] if nz else None, "mean": (sum(vals) / nz) if nz else None,
                  "abs_p50": q(absv, 0.5), "abs_p99": q(absv, 0.99)},
    }
    meta = {"split": split, "source_yaml": os.path.basename(str(yaml_path)), "source_yaml_sha256": report["yaml_sha256"],
            "join": "ego_pose INNER JOIN lidar_pc ON lp.ego_pose_token = ep.token (devkit get_ego_state_for_lidarpc_token_from_db)",
            "key": "lowercase hex lidar_pc.token", "tokens_asked": len(want), "tokens_found": len(table), "partial": bool(missing),
            "tokens_asked_sha256": hashlib.sha256("\n".join(sorted(want)).encode()).hexdigest(),
            "n_logs": report["logs_with_tokens"], "table_values": "raw rad/s (the trainer applies YAW_SCALE, not the table)"}
    return table, meta, report


# ------------------------------------------------------------------------------------------------ the control
def control(table_path: str, db_root: str, n: int = 240, min_logs: int = 12, bank_dir: str | None = None,
            seed: int = 20261106, map_root: str | None = None) -> dict:
    """Prove the join returns the row the TRAINER's ego comes from.  Needs the nuPlan devkit (the driverl-eval venv).
    Sample >= n tokens spread over >= min_logs logs; for EACH token compare, bit for bit (float64), vx, vy, acceleration_x, acceleration_y of
      (join) vs (scenario builder -> EgoState) vs (devkit get_ego_state_for_lidarpc_token_from_db) vs (a banked row's ego[0..3], when one exists),
    and the table's value vs the join's angular_rate_z.  Any disagreement is recorded and the verdict is FAIL."""
    import random
    import navtrain_scenarios as NS
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db
    table, meta, sha = load(table_path)
    local = {lg: p for lg, (p, _loc) in index_dbs_located(db_root).items()}
    # tokens grouped by log: read each candidate DB's lidar_pc once
    rng = random.Random(seed)
    logs = sorted(local)
    rng.shuffle(logs)
    bank = {}
    if bank_dir:
        for p in sorted(glob.glob(os.path.join(bank_dir, "**", "targets_rank*.jsonl"), recursive=True)):
            if "_INVALID" in p or "_quarantine" in p:
                continue
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    r = json.loads(ln)
                    bank.setdefault(r["token"], r)
    picked: dict = {}
    per_log_cap = max(1, math.ceil(n / min_logs))
    for lg in logs:
        rows = join_rows(local[lg], set(table))
        toks = sorted(rows)
        if not toks:
            continue
        rng.shuffle(toks)
        # prefer tokens a bank row exists for, so (c) is exercised wherever the bank is held
        toks.sort(key=lambda t: t not in bank)
        # the scenario builder's own margin guard drops edge frames: take what it yields
        got = [t for t in toks[: per_log_cap * 2]]
        picked[lg] = got
        if sum(min(len(v), per_log_cap) for v in picked.values()) >= n and len(picked) >= min_logs:
            break
    tot = {"tokens": 0, "logs": 0, "bit_equal_builder": 0, "bit_equal_devkit": 0, "bank_rows_compared": 0, "bit_equal_bank": 0,
           "table_equals_join": 0, "slot4_builder_zero": 0, "slot5_builder_zero": 0, "bank_slot4_zero": 0, "bank_speed_equal": 0}
    fails: list = []
    used_logs = set()
    for lg, toks in picked.items():
        db = local[lg]
        join = join_rows(db, set(toks))
        scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(db, toks, **({"map_root": map_root} if map_root else {}))}
        taken = 0
        for tok in toks:
            if taken >= per_log_cap:
                break
            sc = scs.get(tok)
            if sc is None:
                continue                                              # the builder's margin guard dropped this frame: not a bank frame
            wz, vx, vy, ax, ay = join[tok]
            st = sc.get_ego_state_at_iteration(0)
            v, a = st.dynamic_car_state.rear_axle_velocity_2d, st.dynamic_car_state.rear_axle_acceleration_2d
            dk = get_ego_state_for_lidarpc_token_from_db(db, tok)
            dv, da = dk.dynamic_car_state.rear_axle_velocity_2d, dk.dynamic_car_state.rear_axle_acceleration_2d
            jb = (f64_bits(vx), f64_bits(vy), f64_bits(ax), f64_bits(ay))
            same_b = jb == (f64_bits(v.x), f64_bits(v.y), f64_bits(a.x), f64_bits(a.y))
            same_d = jb == (f64_bits(dv.x), f64_bits(dv.y), f64_bits(da.x), f64_bits(da.y))
            tot["tokens"] += 1
            used_logs.add(lg)
            tot["bit_equal_builder"] += int(same_b)
            tot["bit_equal_devkit"] += int(same_d)
            tot["table_equals_join"] += int(f64_bits(table[tok]) == f64_bits(wz))
            tot["slot4_builder_zero"] += int(float(st.dynamic_car_state.angular_velocity) == 0.0)
            tot["slot5_builder_zero"] += int(float(st.tire_steering_angle) == 0.0)
            if not (same_b and same_d):
                fails.append({"log": lg, "token": tok, "join": [vx, vy, ax, ay], "builder": [v.x, v.y, a.x, a.y], "devkit": [dv.x, dv.y, da.x, da.y]})
            r = bank.get(tok)
            if r is not None:
                tot["bank_rows_compared"] += 1
                e = r["ego"]
                eq = jb == (f64_bits(e[0]), f64_bits(e[1]), f64_bits(e[2]), f64_bits(e[3]))
                tot["bit_equal_bank"] += int(eq)
                tot["bank_slot4_zero"] += int(e[4] == 0.0 and e[5] == 0.0)
                tot["bank_speed_equal"] += int(f64_bits(e[6]) == f64_bits(math.hypot(vx, vy)))
                if not eq:
                    fails.append({"log": lg, "token": tok, "join": [vx, vy, ax, ay], "bank": e[:4]})
            taken += 1
    tot["logs"] = len(used_logs)
    ok = (tot["tokens"] >= n and tot["logs"] >= min_logs and tot["bit_equal_builder"] == tot["tokens"]
          and tot["bit_equal_devkit"] == tot["tokens"] and tot["table_equals_join"] == tot["tokens"]
          and tot["bit_equal_bank"] == tot["bank_rows_compared"] and not fails)
    return {"verdict": "PASS" if ok else "FAIL", "table_sha256": sha, "n_requested": n, "min_logs": min_logs, **tot,
            "bank_dir": bank_dir, "failures": fails[:10]}


# ------------------------------------------------------------------------------------------------ units
def units(table_path: str, db_root: str, n_logs: int = 12, seed: int = 20261106) -> dict:
    """angular_rate_z against d(heading)/dt of the SAME DB's quaternions: slope ~ 1 and corr ~ 1 => rad/s, CCW-positive (heading is CCW from +x)."""
    import random
    import numpy as np
    import navtrain_scenarios as NS
    table, _meta, sha = load(table_path)
    local = {lg: p for lg, (p, _loc) in index_dbs_located(db_root).items()}
    rng = random.Random(seed)
    names = sorted(local)
    rng.shuffle(names)
    xs, ys, per = [], [], []
    done = 0
    for lg in names:
        con = ro_connect(local[lg])
        try:
            rows = con.execute("SELECT timestamp, qw, qx, qy, qz, angular_rate_z FROM ego_pose ORDER BY timestamp").fetchall()
        finally:
            con.close()
        if len(rows) < 400:
            continue
        a = np.array(rows, dtype=np.float64)
        t = a[:, 0] * 1e-6
        yaw = np.arctan2(2 * (a[:, 1] * a[:, 4] + a[:, 2] * a[:, 3]), 1 - 2 * (a[:, 3] ** 2 + a[:, 4] ** 2))
        yaw = np.unwrap(yaw)
        # a 0.5 s central difference: ego_pose is 20 Hz and the IMU rate is noisy at one-sample spacing
        k = 5
        d = (yaw[2 * k:] - yaw[:-2 * k]) / (t[2 * k:] - t[:-2 * k])
        w = a[k:-k, 5]
        # a 0.5 s moving average of the measured rate so both sides are the same bandwidth
        wc = np.convolve(a[:, 5], np.ones(2 * k) / (2 * k), mode="same")[k:-k]
        m = np.isfinite(d) & np.isfinite(wc)
        if m.sum() < 100 or np.std(wc[m]) < 1e-6:
            continue
        c = float(np.corrcoef(wc[m], d[m])[0, 1])
        slope = float(np.dot(wc[m], d[m]) / np.dot(wc[m], wc[m]))
        per.append({"log": lg, "n": int(m.sum()), "corr": c, "slope_dyaw_vs_wz": slope, "max_abs_wz": float(np.max(np.abs(w)))})
        xs.append(wc[m]); ys.append(d[m])
        done += 1
        if done >= n_logs:
            break
    X, Y = np.concatenate(xs), np.concatenate(ys)
    slope = float(np.dot(X, Y) / np.dot(X, X))
    corr = float(np.corrcoef(X, Y)[0, 1])
    return {"table_sha256": sha, "n_logs": done, "n_points": int(X.size), "pooled_corr": corr, "pooled_slope": slope,
            "frac_same_sign": float(np.mean(np.sign(X[np.abs(X) > 0.02]) == np.sign(Y[np.abs(X) > 0.02]))),
            "per_log": per, "conclusion": ("rad/s, CCW-positive (heading increases with +angular_rate_z)"
                                           if corr > 0.95 and 0.9 < slope < 1.1 else "UNITS/SIGN NOT CONFIRMED")}


# ------------------------------------------------------------------------------------------------ bank coverage
def bank_tokens(bank_dir: str) -> list:
    """(log_name, token) of every row of every targets_rank*.jsonl under `bank_dir` (not the _INVALID / _quarantine trees)."""
    out = []
    for p in sorted(glob.glob(os.path.join(bank_dir, "**", "targets_rank*.jsonl"), recursive=True)):
        if "_INVALID" in p or "_quarantine" in p:
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                out.append((r.get("log_name", ""), r["token"]))
    return out


def check_bank(table_path: str, bank_dir: str) -> dict:
    table, _meta, sha = load(table_path)
    rows = bank_tokens(bank_dir)
    miss = [(lg, t) for lg, t in rows if t not in table]
    return {"table_sha256": sha, "bank_rows": len(rows), "distinct_tokens": len({t for _l, t in rows}), "missing_rows": len(miss),
            "missing_examples": miss[:5], "verdict": "PASS" if rows and not miss else "FAIL"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True, choices=("navtrain", "navtest"))
    b.add_argument("--yaml", required=True)
    b.add_argument("--db-root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit-logs", type=int, default=None)
    b.add_argument("--allow-partial", action="store_true",
                   help="write the table even when some asked tokens' logs are not held (flagged `partial` in the meta and the report). WITHOUT it a build that cannot cover every asked token "
                        "REFUSES -- a table that silently lacks tokens is the first step to a silent 0.0")
    b.add_argument("--tokens-file", default=None, help="replace the asked set by an explicit token list (JSON list, or {'tokens': [...]}), e.g. a bank's or an eval's tokens")
    c = sub.add_parser("control")
    c.add_argument("--table", required=True)
    c.add_argument("--db-root", required=True)
    c.add_argument("--n", type=int, default=240)
    c.add_argument("--min-logs", type=int, default=12)
    c.add_argument("--bank-dir", default=None)
    c.add_argument("--map-root", default=None)
    u = sub.add_parser("units")
    u.add_argument("--table", required=True)
    u.add_argument("--db-root", required=True)
    u.add_argument("--logs", type=int, default=12)
    k = sub.add_parser("check-bank")
    k.add_argument("--table", required=True)
    k.add_argument("--bank-dir", required=True)
    i = sub.add_parser("info")
    i.add_argument("--table", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        extra = None
        if a.tokens_file:
            d = json.loads(Path(a.tokens_file).read_text(encoding="utf-8"))
            extra = set(d["tokens"] if isinstance(d, dict) else d)
        table, meta, rep = build(a.split, a.yaml, a.db_root, a.limit_logs, tokens=extra)
        if rep["partial"] and not a.allow_partial:
            print(json.dumps(rep, indent=1))
            print(f"REFUSED: {rep['tokens_missing']:,} of {rep['tokens_asked']:,} asked tokens ({rep['logs_not_held']} logs) have no DB under {a.db_root} "
                  f"(searched {list(LOCATIONS)}). Build where the DBs are, or pass --allow-partial to write a table that says it is partial.")
            return 4
        sha = save(a.out, table, meta)
        rep["table_sha256"], rep["out"], rep["bytes"] = sha, str(a.out), os.path.getsize(a.out)
        Path(str(a.out) + ".report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))
        print(f"ZZYAWTABLE_BUILT {len(table)} {sha}")
        return 0
    if a.cmd == "control":
        res = control(a.table, a.db_root, a.n, a.min_logs, a.bank_dir, map_root=a.map_root)
        print(json.dumps(res, indent=1))
        print(f"ZZYAWTABLE_CONTROL_{res['verdict']} tokens={res['tokens']} logs={res['logs']}")
        return 0 if res["verdict"] == "PASS" else 1
    if a.cmd == "units":
        res = units(a.table, a.db_root, a.logs)
        print(json.dumps(res, indent=1))
        return 0
    if a.cmd == "check-bank":
        res = check_bank(a.table, a.bank_dir)
        print(json.dumps(res, indent=1))
        # a machine-readable line for the pod chain (p7_pod.sh stage 5b greps it): PASS only when EVERY bank row's token is in the table (100 %), FAIL otherwise
        print(f"ZZYAWTABLE_BANK_{res['verdict']} rows={res['bank_rows']} distinct_tokens={res['distinct_tokens']} missing_rows={res['missing_rows']}")
        return 0 if res["verdict"] == "PASS" else 1
    if a.cmd == "info":
        table, meta, sha = load(a.table)
        print(json.dumps({"sha256": sha, "meta": meta}, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
