#!/usr/bin/env python3
"""WP YAW tests (TanitAD-artifacts/yawrate_wp/PLAN.md): the token -> angular_rate_z table, the trainer flag, the launch-gate entries, the eval/deploy shim.

  python test_ego_yaw.py            everything, incl. the real-data table control and the source-mutation harness (minutes, CPU)
  python test_ego_yaw.py --fast     no real-data control, no mutation harness (about a minute)
  python test_ego_yaw.py --check N  one named check (what the mutation harness runs on a mutated copy of this tree)

Run under CUDA_VISIBLE_DEVICES=-1 with an interpreter that has torch + numpy + yaml (the driverl-eval venv, or the tanitad venv for the non-devkit parts).  Prints ZZEGOYAW_PASS <n>.
A check that cannot go RED is not a check: every positive assertion has its negative twin, and the MUTATIONS table DELETES the emitting line from a copy of the tree and requires the check to fail.
Any skip prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ego_yaw_table as EYT  # noqa: E402

N = 0
MIRROR = Path("C:/Users/Admin/tanitad-push/.git")
TIP_TRAIN_BLOB = "0b2170eb5c1357471a06152374f4b463a86e2c3c"           # refe/train.py at the tip this work started from (a521c0736a70)
TIP_PLANNER_BLOB = "6fb6598a2ecf230153a4d7adfb790afc30728070"
DB_ROOT = Path("E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
TABLES = Path("E:/Projects/TanitAD-artifacts/yawrate_wp/tables")


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def skip(why):
    print(f"ZZTEST_SKIP {why}", flush=True)


def git_hash_object(path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()


# ------------------------------------------------------------------------------------------------ fixtures
def make_nuplan_like_db(path: Path, n: int = 5, decoy: bool = True, salt: int = 0):
    """A minimal nuPlan-shaped DB: ego_pose + lidar_pc with the real column names.  Values are awkward floats (1/3 ...) so a bit-for-bit comparison means something.
    With `decoy`, a SECOND ego_pose row shares each lidar_pc row's TIMESTAMP but not its token -- a join by time (the wrong row) reads the decoy."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE ego_pose (token BLOB PRIMARY KEY, timestamp INTEGER, x REAL, y REAL, z REAL, qw REAL, qx REAL, qy REAL, qz REAL, vx REAL, vy REAL, vz REAL,"
                " acceleration_x REAL, acceleration_y REAL, acceleration_z REAL, angular_rate_x REAL, angular_rate_y REAL, angular_rate_z REAL, epsg INTEGER, log_token BLOB)")
    con.execute("CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, next_token BLOB, prev_token BLOB, ego_pose_token BLOB, lidar_token BLOB, scene_token BLOB, filename TEXT, timestamp INTEGER)")
    exp = {}
    for i in range(n):
        ept = bytes([0xE0 + salt, i] + [i] * 6)
        lpt = bytes([0xA0 + salt, i] + [i + 1] * 6)
        vx, vy, ax, ay, wz = (1 + i) / 3.0, -(i + 2) / 7.0, 0.1 * (i + 1) / 9.0, -(i + 3) / 11.0, 0.01 * (i + 1) / 3.0 * (-1) ** i
        ts = 1_000_000 + i * 50_000
        con.execute("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ept, ts, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, vx, vy, 0.0, ax, ay, 0.0, 0.0, 0.0, wz, 0, b"\0" * 8))
        con.execute("INSERT INTO lidar_pc VALUES (?,?,?,?,?,?,?,?)", (lpt, None, None, ept, b"\1" * 8, b"\2" * 8, "f", ts + 7))
        if decoy:
            con.execute("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (bytes([0xD0 + salt, i] + [i] * 6), ts + 7, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 99.0, 99.0, 0.0, 99.0, 99.0, 0.0, 0.0, 0.0, 99.0, 0, b"\0" * 8))
        exp[lpt.hex()] = (wz, vx, vy, ax, ay)
    con.commit()
    con.close()
    return exp


def make_bank(root: Path, yaw_in_bank: float = 0.0, n_scenes: int = 4) -> str:
    """6 tuples in 4 scenes (2 with a rank-1 twin), the REAL banks' property: ego[4] == ego[5] == 0.0 (D-REFE-DEADSLOTS-1).  Synthetic images (`--synthetic`)."""
    rng = np.random.default_rng(5)
    d = root / "bank"
    d.mkdir(parents=True, exist_ok=True)
    rows0, rows1 = [], []
    for s in range(n_scenes):
        gy = [0.0, 9.0, -9.0, 1.0][s % 4]
        e = [float(x) for x in rng.normal(0, 0.3, 7)]
        e[4], e[5] = float(yaw_in_bank), 0.0
        base = {"image": [f"log{s}/CAM_F0/a{s}.jpg"], "cameras": ["CAM_F0"], "ego": e, "goal": [30.0, gy, 60.0, 2 * gy],
                "traj": [[float(0.7 * (t + 1)), float(gy * (t + 1) / 80.0), float(0.02 * s)] for t in range(20)],
                "step": 0, "token": f"tok{s:04x}", "log_name": f"log{s}", "scenario_type": "t", "source": "fixture"}
        rows0.append({**base, "rank": 0})
        if s < 2:
            rows1.append({**base, "rank": 1, "goal": [30.0, -gy, 60.0, -2 * gy], "aug": {"kind": "lane_rank", "value": 1, "divergence_m": 2.0}})
    for nm, rows in (("targets_rank0.jsonl", rows0), ("targets_rank1.jsonl", rows1)):
        (d / nm).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return str(d)


FIX_YAW = {f"tok{s:04x}": v for s, v in enumerate((0.25, -0.5, 0.0078125, 0.3333333333333333))}      # exactly representable + one awkward float


def make_table(root: Path, tokens=None, name="yaw.json.gz") -> str:
    tab = dict(FIX_YAW if tokens is None else tokens)
    p = root / name
    EYT.save(p, {k: float(v) for k, v in tab.items()}, {"split": "fixture"})
    return str(p)


def argv_default(bank: str, out: str, extra=()) -> list:
    return ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", bank, "--scorer-targets", bank + "_none", "--batch", "1",
            "--accum", "2", "--steps", "3", "--lr", "5e-5", "--log-every", "1", "--ckpt-every-min", "999", "--out", out] + list(extra)


def argv_p6(bank: str, out: str, *, steps: int = 2, accum: int = 1, extra=()) -> list:
    """The P6-style rehearsal argv of test_p6_levers.py (online synthetic labeller, command nav mode, scorer sees the trunk): the argv under which EVERY declared trainable group, the scorer's
    included, receives a gradient -- so G-LIVE's group checks are not what fails in these tests."""
    return ["--backbone", "vits16", "--weights", "none", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", bank, "--scorer-targets", bank + "_none",
            "--yaw-wrap", "off", "--nav-mode", "command", "--labeller", "online", "--label-backend", "synthetic", "--label-workers", "2",
            "--batch", "1", "--accum", str(accum), "--steps", str(steps), "--lr", "2e-4", "--log-every", "1", "--ckpt-every-min", "999", "--out", out,
            "--scorer-sees-trunk"] + list(extra)


def cfg_for_bank():
    from model import REFeConfig
    c = REFeConfig.for_backbone("vits16")
    c.n_cameras = 1
    return c


def same_item(a, b) -> bool:
    """two dataset items (tuples of tensors) bit-for-bit equal"""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if not (isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


# ------------------------------------------------------------------------------------------------ S1 the table, the join, the control
def chk_table():
    td = Path(tempfile.mkdtemp(prefix="yt_tab_"))
    try:
        exp = make_nuplan_like_db(td / "2021.01.01.00.00.00_veh-00_00001_00099.db")
        db = str(td / "2021.01.01.00.00.00_veh-00_00001_00099.db")
        got = EYT.join_rows(db)
        ok(set(got) == set(exp), (sorted(got), sorted(exp)))
        ok(all(got[t] == exp[t] and all(EYT.f64_bits(a) == EYT.f64_bits(b) for a, b in zip(got[t], exp[t])) for t in exp), "the join returns the LINKED ego_pose row, bit for bit")
        ok(all(v[1] != 99.0 and v[0] != 99.0 for v in got.values()), "a join by TIMESTAMP would have read the decoy rows (99.0): this one reads the token-linked row")
        wanted = {sorted(exp)[0], "ff" * 8}
        ok(set(EYT.join_rows(db, wanted)) == {sorted(exp)[0]}, "want restricts the tokens kept")
        # the connection is READ-ONLY
        con = EYT.ro_connect(db)
        try:
            con.execute("UPDATE ego_pose SET angular_rate_z = 0.0")
            ok(False, "a write through the builder's connection succeeded")
        except sqlite3.OperationalError as e:
            ok("readonly" in str(e).lower(), str(e))
        finally:
            con.close()
        ok(EYT.join_rows(db) == got, "the DB is unchanged after the read-only attempt")
        # the table file: deterministic, round-trips, refuses what it did not write
        tab = {t: float(v[0]) for t, v in exp.items()}
        p1, p2 = td / "a.json.gz", td / "b.json.gz"
        s1 = EYT.save(p1, tab, {"split": "t"})
        s2 = EYT.save(p2, tab, {"split": "t"})
        ok(s1 == s2 and p1.read_bytes() == p2.read_bytes(), "the same table saves to the same bytes (no wall clock, mtime 0)")
        t2, m2, sh = EYT.load(p1)
        ok(sh == s1 and t2 == tab and m2["n_tokens"] == len(tab) and m2["schema"] == EYT.SCHEMA and m2["field"] == EYT.FIELD and "rad/s" in m2["units"])
        ok(all(EYT.f64_bits(t2[k]) == EYT.f64_bits(tab[k]) for k in tab), "float64 round-trips exactly through JSON")
        for bad, why in (({"a": float("nan")}, "NaN"), ({"a": float("inf")}, "inf"), ({"a": 1}, "an int")):
            try:
                EYT.save(td / "bad.json.gz", bad, {})
                ok(False, f"save accepted {why}")
            except EYT.TableRefused:
                ok(True)
        try:
            EYT.load(p1, expect_sha256="0" * 64)
            ok(False, "a pinned sha256 mismatch was accepted")
        except EYT.TableRefused as e:
            ok("sha256" in str(e))
        raw = json.loads(gzip.open(p1, "rt", encoding="utf-8").read())
        for mut, frag in ((lambda d: d["meta"].__setitem__("schema", "other"), "not an"), (lambda d: d["meta"].__setitem__("n_tokens", 99), "meta says"),
                          (lambda d: (d["table"].__setitem__("zz", "x"), d["meta"].__setitem__("n_tokens", len(d["table"]))), "non-finite")):
            d = json.loads(json.dumps(raw))
            mut(d)
            q = td / "mut.json.gz"
            with gzip.open(q, "wt", encoding="utf-8") as f:
                json.dump(d, f)
            try:
                EYT.load(q)
                ok(False, f"load accepted a table with {frag}")
            except EYT.TableRefused as e:
                ok(frag in str(e), str(e))
        # build(): coverage is reported as found / asked, with the missing tokens named
        import yaml
        ylog = "2021.01.01.00.00.00_veh-00_00001_00099"
        want = sorted(exp)[:3] + ["ee" * 8]
        (td / "split.yaml").write_text(yaml.safe_dump({"log_names": [ylog, "2021.01.01.00.00.00_veh-00_NOT_HELD"], "tokens": want}), encoding="utf-8")
        table, meta, rep = EYT.build("navtrain", str(td / "split.yaml"), str(td))
        ok(rep["tokens_asked"] == 4 and rep["tokens_found"] == 3 and rep["tokens_missing"] == 1 and abs(rep["coverage"] - 0.75) < 1e-12 and rep["partial"] and meta["partial"], rep)
        ok(rep["logs_in_split_file"] == 2 and rep["logs_held_locally"] == 1 and rep["missing_examples"] == ["ee" * 8], rep)
        ok(table == {t: float(exp[t][0]) for t in sorted(exp)[:3]})
        ok(rep["by_location"]["top"]["tokens_found"] == 3 and rep["by_location"]["top"]["logs_read"] == 1 and rep["location_order"] == ["top", "trainval", "test", "mini"], rep["by_location"])
        # a build that cannot cover every asked token REFUSES (rc 4) unless told --allow-partial, and the partial table says so
        outp = td / "part.json.gz"
        cli = ["build", "--split", "navtrain", "--yaml", str(td / "split.yaml"), "--db-root", str(td), "--out", str(outp)]
        ok(EYT.main(cli) == 4 and not outp.exists(), "partial coverage without --allow-partial must refuse and write nothing")
        ok(EYT.main(cli + ["--allow-partial"]) == 0 and EYT.load(outp)[1]["partial"] is True)
        # MM 2026-11-06: logs are resolved over splits/*.db, then trainval/, test/, mini/ -- and coverage is reported PER LOCATION
        loc_root = td / "locroot"
        for sub in ("trainval", "test", "mini"):
            (loc_root / sub).mkdir(parents=True)
        e_top = make_nuplan_like_db(loc_root / "LOG_TOP.db", n=2, decoy=False, salt=1)
        e_tv = make_nuplan_like_db(loc_root / "trainval" / "LOG_TV.db", n=3, decoy=False, salt=2)
        e_te = make_nuplan_like_db(loc_root / "test" / "LOG_TE.db", n=4, decoy=False, salt=3)
        shutil.copyfile(loc_root / "trainval" / "LOG_TV.db", loc_root / "test" / "LOG_TV.db")                 # the SAME log in two places (identical file): first location wins, recorded
        idx = EYT.index_dbs_located(str(loc_root))
        ok({k: v[1] for k, v in idx.items()} == {"LOG_TOP": "top", "LOG_TV": "trainval", "LOG_TE": "test"} and set(EYT.index_dbs_located.duplicates) == {"LOG_TV"}, idx)
        (loc_root / "test" / "LOG_TV.db").write_bytes(b"different size")
        try:
            EYT.index_dbs_located(str(loc_root))
            ok(False, "the same log in two locations with different sizes was accepted")
        except EYT.TableRefused as e:
            ok("DIFFERENT sizes" in str(e))
        (loc_root / "test" / "LOG_TV.db").unlink()
        all_tok = set(e_top) | set(e_tv) | set(e_te)
        (td / "loc.yaml").write_text(yaml.safe_dump({"log_names": ["LOG_TOP", "LOG_TV", "LOG_TE", "LOG_GONE"], "tokens": sorted(all_tok)}), encoding="utf-8")
        t3, m3, r3 = EYT.build("navtest", str(td / "loc.yaml"), str(loc_root))
        bl = r3["by_location"]
        ok(len(t3) == len(all_tok) == 9 and not r3["partial"] and r3["coverage"] == 1.0, r3)
        ok((bl["top"]["tokens_found"], bl["trainval"]["tokens_found"], bl["test"]["tokens_found"], bl["mini"]["tokens_found"]) == (2, 3, 4, 0), bl)
        ok(r3["logs_not_held"] == 1 and r3["logs_not_held_examples"] == ["LOG_GONE"] and bl["top"]["logs_in_split_file"] == 1, r3)
        # check_bank: a bank row whose token is not in the table is named
        bank = make_bank(td)
        tp = make_table(td)
        ok(EYT.check_bank(tp, bank)["verdict"] == "PASS")
        tp2 = make_table(td, {k: v for k, v in FIX_YAW.items() if k != "tok0002"}, "yaw_missing.json.gz")
        cb = EYT.check_bank(tp2, bank)
        ok(cb["verdict"] == "FAIL" and cb["missing_rows"] == 1 and cb["missing_examples"][0][1] == "tok0002", cb)
        # the CLI the pod chain calls (p7_pod.sh stage 5b): rc AND a ZZ line, both directions
        env_ = dict(os.environ, PYTHONIOENCODING="utf-8")
        cli_ok = subprocess.run([sys.executable, str(HERE / "ego_yaw_table.py"), "check-bank", "--table", tp, "--bank-dir", bank], capture_output=True, text=True, env=env_, encoding="utf-8")
        ok(cli_ok.returncode == 0 and "ZZYAWTABLE_BANK_PASS rows=" in cli_ok.stdout and "missing_rows=0" in cli_ok.stdout and "ZZYAWTABLE_BANK_FAIL" not in cli_ok.stdout, cli_ok.stdout[-300:])
        cli_bad = subprocess.run([sys.executable, str(HERE / "ego_yaw_table.py"), "check-bank", "--table", tp2, "--bank-dir", bank], capture_output=True, text=True, env=env_, encoding="utf-8")
        ok(cli_bad.returncode == 1 and "ZZYAWTABLE_BANK_FAIL" in cli_bad.stdout and "missing_rows=1" in cli_bad.stdout and "ZZYAWTABLE_BANK_PASS" not in cli_bad.stdout, cli_bad.stdout[-300:])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_real_table_control():
    """The mandatory control, on REAL data: the join's vx, vy, acceleration_x, acceleration_y equal -- bit for bit -- what the bank's scenario builder and the devkit hand the trainer,
    and a banked row's ego[0..3].  The mutated join (vx <- vy) must make it FAIL.  Needs the nuPlan devkit (the driverl-eval venv) + the dev-box DBs + the dev-box table."""
    tab = TABLES / "ego_yaw_navtrain_devbox.json.gz"
    try:
        import nuplan  # noqa: F401
    except Exception:
        return skip("real-data control: no nuPlan devkit in this interpreter")
    if not tab.exists() or not DB_ROOT.exists():
        return skip("real-data control: dev-box table / DBs absent")
    kw = dict(table_path=str(tab), db_root=str(DB_ROOT), n=20, min_logs=10, map_root="E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0",
              bank_dir="E:/Projects/TanitAD/data/refe_navtrain")
    r = EYT.control(**kw)
    ok(r["verdict"] == "PASS" and r["tokens"] >= 20 and r["logs"] >= 10 and r["bit_equal_builder"] == r["tokens"] and r["bit_equal_devkit"] == r["tokens"], r)
    ok(r["table_equals_join"] == r["tokens"] and r["bit_equal_bank"] == r["bank_rows_compared"] and r["bank_slot4_zero"] == r["bank_rows_compared"], r)
    orig = EYT.JOIN_SQL
    try:
        EYT.JOIN_SQL = orig.replace("ep.vx,", "ep.vy,", 1)           # the WRONG column: vx <- vy
        bad = EYT.control(**kw)
    finally:
        EYT.JOIN_SQL = orig
    ok(bad["verdict"] == "FAIL" and bad["failures"], "the control must go RED on a join that reads the wrong column")
    # and a join on the wrong ROW (by timestamp, the decoy) would disagree on vx: covered by chk_table's decoy rows


def chk_real_table_tokens():
    tn, tt = TABLES / "ego_yaw_navtest.json.gz", TABLES / "ego_yaw_navtrain_devbox.json.gz"
    if not tn.exists():
        return skip("real-table token checks: tables absent")
    nav, meta, sha = EYT.load(tn)
    ok(len(nav) == 12146 and meta["tokens_found"] == meta["tokens_asked"] == 12146 and meta["split"] == "navtest")
    allf = Path("E:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json")
    if allf.exists():
        d = json.loads(allf.read_text(encoding="utf-8"))
        toks = set(d["tokens"] if isinstance(d, dict) else d)
        ok(len(toks) == 12146 and toks == set(nav), "the navtest table's tokens are exactly the eval's 12,146")
    c3 = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-c3-selection-headroom/tokens.json")
    if c3.exists():
        ok(set(json.loads(c3.read_text(encoding="utf-8"))["tokens"]) <= set(nav), "the C3 selection tokens are inside the navtest table")
    if tt.exists():
        tr, m2, _ = EYT.load(tt)
        ok(not (set(tr) & set(nav)), "navtrain and navtest tokens are disjoint (no eval token in the train table)")
        ok(m2["split"] == "navtrain" and m2["tokens_found"] == len(tr) <= m2["tokens_asked"] == 103288 and m2["partial"] == (len(tr) < 103288))


# ------------------------------------------------------------------------------------------------ S2 the loader: default identity, the override, the refusals
def chk_loader():
    import train as T
    td = Path(tempfile.mkdtemp(prefix="yt_ld_"))
    try:
        bank = make_bank(td)
        tp = make_table(td)
        cfg = cfg_for_bank()
        d0 = T.TargetBank(bank, None, cfg, synthetic=True)
        d1 = T.TargetBank(bank, None, cfg, synthetic=True, ego_yaw_table=None)
        ok(d0.ego_yaw is None and d0.ego_yaw_sha256 is None and len(d0) == len(d1) == 6)
        for i in range(len(d0)):
            ok(same_item(d0[i], d1[i]), f"default (kwarg omitted) vs explicit None, item {i}")
            r = d0.rows[i]
            ok(torch.equal(d0[i][1], torch.tensor(r["ego"], dtype=torch.float32)) and float(d0[i][1][4]) == 0.0 and float(d0[i][1][5]) == 0.0,
               "no flag: ego is the bank's own vector, slots 4 and 5 are 0.0")
        pa = T.build_parser().parse_args(["--targets", bank])
        ok(pa.ego_yaw_table is None)
        # the override
        dy = T.TargetBank(bank, None, cfg, synthetic=True, ego_yaw_table=tp)
        ok(dy.ego_yaw_sha256 == EYT.sha256_file(tp) and dy.ego_yaw_cov["covered"] == 6)
        ok(EYT.YAW_SCALE == 10.0, "MM decision 2026-11-06: YAW_SCALE = 10.0")
        ok(dy.ego_yaw["tok0001"] == -0.5, "the table holds RAW rad/s; the loader applies the scale")
        for i in range(len(dy)):
            r, a, b = dy.rows[i], d0[i], dy[i]
            tok = r["token"]
            ok(float(b[1][4]) == float(np.float32(EYT.YAW_SCALE * FIX_YAW[tok])), (i, float(b[1][4]), FIX_YAW[tok]))
            ok(b[1].dtype == torch.float32 and b[1].shape == a[1].shape)
            keep = [0, 1, 2, 3, 5, 6]
            ok(torch.equal(b[1][keep], a[1][keep]) and float(b[1][5]) == 0.0, "only slot 4 moved; slot 5 (steering) stays exactly as the bank has it")
            ok(all(same_item((x,), (y,)) for k, (x, y) in enumerate(zip(a, b)) if k != 1), "image, goal, targets, candidates: untouched")
        ok(dy[0][1][4] == dy[4][1][4] if dy.rows[4]["token"] == dy.rows[0]["token"] else True)
        twins = {}
        for i, r in enumerate(dy.rows):
            twins.setdefault(r["token"], set()).add(float(dy[i][1][4]))
        ok(all(len(v) == 1 for v in twins.values()) and any(int(r["rank"]) == 1 for r in dy.rows), "a goal-augmented twin carries its frame's yaw")
        # the refusals
        miss = make_table(td, {k: v for k, v in FIX_YAW.items() if k != "tok0001"}, "miss.json.gz")
        try:
            T.TargetBank(bank, None, cfg, synthetic=True, ego_yaw_table=miss)
            ok(False, "a table missing a bank row's token was accepted")
        except SystemExit as e:
            ok("NOT in the table" in str(e) and "tok0001" in str(e) and "REFUSED" in str(e), str(e)[:200])
        bank_nz = make_bank(td / "nz", yaw_in_bank=0.125)
        try:
            T.TargetBank(bank_nz, None, cfg, synthetic=True, ego_yaw_table=tp)
            ok(False, "a bank that already carries a yaw rate was accepted")
        except SystemExit as e:
            ok("ALREADY carry" in str(e), str(e)[:200])
        ok(T.TargetBank(bank_nz, None, cfg, synthetic=True)[0][1][4] == np.float32(0.125), "the same bank without the flag is served as it is")
        for badtab, frag in ((str(td / "nope.json.gz"), "refused"),):
            try:
                T.TargetBank(bank, None, cfg, synthetic=True, ego_yaw_table=badtab)
                ok(False)
            except SystemExit as e:
                ok(frag in str(e))
        g = td / "garbage.json.gz"
        g.write_bytes(b"not a gzip")
        try:
            T.TargetBank(bank, None, cfg, synthetic=True, ego_yaw_table=str(g))
            ok(False, "a garbage table was accepted")
        except SystemExit as e:
            ok("refused" in str(e))
        c4 = cfg_for_bank()
        c4.ego_dim = 4
        try:
            T.TargetBank(bank, None, c4, synthetic=True, ego_yaw_table=tp)
            ok(False, "a model with no slot 4 was accepted")
        except SystemExit as e:
            ok("ego slot 4" in str(e))
        # the per-sample backstop
        del dy.ego_yaw["tok0003"]
        try:
            dy[3]
            ok(False, "the per-sample lookup served a row with no table entry")
        except RuntimeError as e:
            ok("tok0003" in str(e) and "REFUSED" in str(e))
        # the identity of a run: a resume under a different table is refused; a run without the flag is unchanged
        old = {"args": {k: None for k in T.IDENTITY_ARGS}, "ego_yaw_sha256": "a" * 64, "ego_yaw_scale": 10.0}
        new = {"args": {k: None for k in T.IDENTITY_ARGS}, "ego_yaw_sha256": "b" * 64, "ego_yaw_scale": 10.0}
        ok(any("ego_yaw_sha256" in m for m in T._identity_diff(old, new)) and not any("ego_yaw_sha256" in m for m in T._identity_diff(old, dict(old))))
        ok(any("ego_yaw_sha256" in m for m in T._identity_diff({"args": old["args"]}, new)), "a pre-WP-YAW checkpoint does not resume under a yaw table")
        ok(T._identity_diff({"args": old["args"]}, {"args": old["args"]}) == [], "two runs without the flag: no new identity key")
        ok(any("ego_yaw_scale" in m for m in T._identity_diff(old, dict(old, ego_yaw_scale=1.0))), "a resume under a different yaw scale is refused")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ S3 the trainer + the checkpoint meta
def _smoke(argv, work, arm=None):
    import gate_checks as C
    return C.instrumented_run(argv, mode="smoke", work=work, arm=arm)


def chk_ckpt_meta():
    td = Path(tempfile.mkdtemp(prefix="yt_meta_"))
    try:
        bank, tp = make_bank(td), None
        tp = make_table(td)
        s1 = _smoke(argv_default(bank, str(td / "o1"), ("--ego-yaw-table", tp)), td / "w1")
        ok(s1.rc == 0, (s1.rc, s1.error))
        meta = torch.load(td / "o1" / "model_final.pt", map_location="cpu", weights_only=False)["meta"]
        ok(meta.get("ego_yaw") == "nuplan_angular_rate_z" == EYT.YAW_TAG and meta.get("ego_yaw_table_sha256") == EYT.sha256_file(tp) and meta.get("ego_yaw_slot") == 4, meta)
        ok(meta.get("ego_yaw_scale") == EYT.YAW_SCALE == 10.0, meta)
        # ONE constant: the trainer's loader, the checkpoint's meta and the shim built FROM that checkpoint produce the very same float32, bit for bit
        import ego_yaw as EY
        import train as T
        sh = EY.EgoYawShim.from_checkpoint(str(td / "o1" / "model_final.pt"), tp)
        ok(sh is not None and sh.scale == meta["ego_yaw_scale"] == EYT.YAW_SCALE and sh.train_table_sha256 == EYT.sha256_file(tp))
        ds = T.TargetBank(bank, None, cfg_for_bank(), synthetic=True, ego_yaw_table=tp)
        for i, r in enumerate(ds.rows):
            with sh.token(r["token"]):
                shimmed = sh.apply(torch.zeros(1, 7))
            ok(shimmed[0, 4].item() == ds[i][1][4].item() and shimmed[0, 4].dtype == ds[i][1].dtype, (r["token"], shimmed[0, 4].item(), ds[i][1][4].item()))
        s0 = _smoke(argv_default(bank, str(td / "o0")), td / "w0")
        ok(s0.rc == 0, (s0.rc, s0.error))
        m0 = torch.load(td / "o0" / "model_final.pt", map_location="cpu", weights_only=False)["meta"]
        ok(not any(k.startswith("ego_yaw") for k in m0), m0)
        ok("ego_yaw_scale" not in m0, "no flag: no scale in the meta")
        # ckpt_last.pt (the resume file) carries the same meta
        ck = torch.load(td / "o1" / "ckpt_last.pt", map_location="cpu", weights_only=False)
        ok(ck["meta"].get("ego_yaw") == "nuplan_angular_rate_z" and ck["identity"].get("ego_yaw_sha256") == EYT.sha256_file(tp), "the resume file carries the table's content hash")
        ok(ck["identity"].get("ego_yaw_scale") == EYT.YAW_SCALE, "and the scale")
        ok("ego_yaw_sha256" not in torch.load(td / "o0" / "ckpt_last.pt", map_location="cpu", weights_only=False)["identity"])
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ S4 the launch gate
def _prof():
    import launch_gate as L
    return {**L.PROFILES["refe_p6"], "forbidden_flags": (), "required_flags": {}, "provisional_flags": ()}


def chk_gate_registry():
    import gate_checks as C
    import train as T
    from types import SimpleNamespace
    ok(C.DVB_KINDS.get("ego_yaw_table") == "built" and C.EGO_YAW_SLOT == EYT.EGO_YAW_SLOT == 4)
    dests = {x.dest for x in T.build_parser()._actions if x.dest != "help"}
    ok(set(C.DVB_KINDS) == dests and len(dests) == 69, (sorted(dests ^ set(C.DVB_KINDS))))
    # RED: the entry removed -> the gate refuses the trainer's flag as "NO declared-vs-built entry"
    dummy = SimpleNamespace(parser_unknown=[], rc="X", error=None, log="-")
    saved = C.DVB_KINDS.pop("ego_yaw_table")
    try:
        ev = C.check_dvb(["--targets", "x"], dummy, None, _prof())
        ok(ev["status"] == "FAIL" and "ego_yaw_table" in " ".join(ev["reasons"]) and "NO declared-vs-built entry" in " ".join(ev["reasons"]), ev["reasons"])
    finally:
        C.DVB_KINDS["ego_yaw_table"] = saved
    # G-HYG still passes with the new namespace read
    ok(C.check_hyg()["status"] == "PASS")


def _build_smoke(td, extra, arm_b=None, arm_s=None, tag=""):
    import gate_checks as C
    import launch_gate as L
    bank = make_bank(td)
    argv = argv_p6(bank, str(td / f"o{tag}"), extra=extra)
    b_argv = C._set(C._drop(argv, "--steps", 1), "--epochs", "1")
    b = C.instrumented_run(b_argv, mode="build", work=td / f"wb{tag}", arm=arm_b)
    s = C.instrumented_run(argv, mode="smoke", work=td / f"ws{tag}", arm=arm_s)
    L._free(b)
    return argv, b_argv, b, s


def chk_dvb_flag_set():
    """G-DVB with the flag named: the row exists, is OK, and the trainer's CALL SITE hands the table to the loader."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="yt_dvb_"))
    try:
        tp = make_table(td)
        argv, b_argv, b, s = _build_smoke(td, ("--ego-yaw-table", tp))
        ok(b.rc == "STOPPED_AT_BUILD" and s.rc == 0, (b.rc, b.error, s.rc, s.error))
        prof = {**_prof(), "expect_schedule": None}
        ev = C.check_dvb(b_argv, b, s, prof)
        rows = [r for r in ev["details"]["rows"] if r["dest"] == "ego_yaw_table"]
        ok(ev["status"] == "PASS" and rows and all(r["ok"] for r in rows) and rows[0]["declared"] == tp == rows[0]["built"], (ev["reasons"], rows))
        # the profile's content pin: the right sha passes, a wrong one is a launch blocker
        pin = EYT.sha256_file(tp)
        ev_ok = C.check_dvb(b_argv, b, s, {**prof, "ego_yaw_sha256": pin})
        ok(ev_ok["status"] == "PASS" and any(r["how"].startswith("the table file's sha256") and r["ok"] for r in ev_ok["details"]["rows"]), ev_ok["reasons"])
        ev_bad = C.check_dvb(b_argv, b, s, {**prof, "ego_yaw_sha256": "0" * 64})
        ok(ev_bad["status"] == "FAIL" and "sha256" in " ".join(ev_bad["reasons"]).lower() or "ego_yaw_table" in " ".join(ev_bad["reasons"]), ev_bad["reasons"])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_dvb_red_not_passed():
    """RED arm (G-DVB): the flag is declared but the loader is built WITHOUT the table -> declared != built."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="yt_dvbr_"))
    try:
        tp = make_table(td)
        argv, b_argv, b, s = _build_smoke(td, ("--ego-yaw-table", tp), arm_b="yaw_table_not_passed", arm_s="yaw_table_not_passed")
        ev = C.check_dvb(b_argv, b, s, {**_prof(), "expect_schedule": None}, arm="yaw_table_not_passed")
        ok(ev["status"] == "FAIL" and "--ego-yaw-table: declared" in " ".join(ev["reasons"]) and "built None" in " ".join(ev["reasons"]), ev["reasons"])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_live_flag_set():
    """G-LIVE with the flag named: the model RECEIVED a non-zero yaw, ego_enc.0.weight[:, 4] has a finite non-zero gradient, the control reads exactly 0.0 / non-zero."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="yt_live_"))
    try:
        tp = make_table(td)
        bank = make_bank(td)
        argv = argv_p6(bank, str(td / "o"), extra=("--ego-yaw-table", tp))
        s = C.instrumented_run(argv, mode="smoke", work=td / "w")
        ok(s.rc == 0, (s.rc, s.error))
        ev = C.check_live(argv, s, _prof())
        ok(ev["status"] == "PASS", ev["reasons"])
        f = ev["details"]["ego_yaw_live"]
        ok(f["flag"] and f["ego4_abs_max_received"] > 0.0 and f["col4_grad_abs_sum_max"] > 0.0 and f["finite"], f)
        ok(f["probe"]["zero"] == 0.0 and f["probe"]["one"] > 0.0, f["probe"])
        ok(abs(f["ego4_abs_max_received"] - EYT.YAW_SCALE * max(abs(v) for v in FIX_YAW.values())) < 1e-5, ("the model received the SCALED yaw", f))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_live_control_unset():
    """The CONTROL: no flag, the bank's dead slot -> the model receives exactly 0.0 and the column-4 gradient reads EXACTLY 0.0."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="yt_ctl_"))
    try:
        bank = make_bank(td)
        argv = argv_p6(bank, str(td / "o"))
        s = C.instrumented_run(argv, mode="smoke", work=td / "w")
        ok(s.rc == 0, (s.rc, s.error))
        ev = C.check_live(argv, s, _prof())
        ok(ev["status"] == "PASS", ev["reasons"])
        f = ev["details"]["ego_yaw_live"]
        ok((not f["flag"]) and f["ego4_abs_max_received"] == 0.0 and f["col4_grad_abs_sum_max"] == 0.0 and f["other_cols_grad_abs_sum_max"] > 0.0, f)
        ok(f["probe"]["zero"] == 0.0 and f["probe"]["one"] > 0.0, f["probe"])
        # and the judge goes RED on a column that moves while its input is exactly zero (a broken probe / a leak into the column)
        bad = dict(f, col4_grad_abs_sum_max=0.5)
        ok(C.judge_ego_yaw(bad, f["probe"]) and "must be exactly 0.0" in " ".join(C.judge_ego_yaw(bad, f["probe"])))
        ok(C.judge_ego_yaw(f, dict(f["probe"], zero=0.25)) and C.judge_ego_yaw(f, dict(f["probe"], one=0.0)), "a probe whose control fails is refused")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_live_red_override_removed():
    """RED arm (G-LIVE): flag set, table LOADED, override line removed -> the yaw never reaches the model -> the gate FAILs."""
    import gate_checks as C
    td = Path(tempfile.mkdtemp(prefix="yt_red_"))
    try:
        tp = make_table(td)
        bank = make_bank(td)
        argv = argv_p6(bank, str(td / "o"), extra=("--ego-yaw-table", tp))
        s = C.instrumented_run(argv, mode="smoke", work=td / "w", arm="yaw_override_removed")
        ok(s.rc == 0, (s.rc, s.error))
        ev = C.check_live(argv, s, _prof(), arm="yaw_override_removed")
        txt = " ".join(ev["reasons"])
        ok(ev["status"] == "FAIL" and "RECEIVED ego[4] == 0.0" in txt and "EXACTLY zero" in txt, ev["reasons"])
        f = ev["details"]["ego_yaw_live"]
        ok(f["ego4_abs_max_received"] == 0.0 and f["col4_grad_abs_sum_max"] == 0.0, f)
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ S5 the shim
class _Stub(torch.nn.Module):
    """forward(img, ego, goal, calib=None) -> ego (so the shim's effect is directly readable)"""
    def forward(self, img, ego, goal, calib=None):
        return ego


def chk_shim():
    import ego_yaw as EY
    td = Path(tempfile.mkdtemp(prefix="yt_shim_"))
    try:
        tab_path = make_table(td)
        tab, _m, sha = EYT.load(tab_path)
        m = _Stub()
        base = torch.zeros(1, 7)
        base[0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        base[0, 6] = 7.0
        before = m(None, base, None).clone()
        shim = EY.EgoYawShim(tab, sha, EYT.YAW_SCALE).install(m)
        # no token -> REFUSED
        try:
            m(None, base, None)
            ok(False, "a forward with no token was served")
        except RuntimeError as e:
            ok("NO token" in str(e))
        with shim.token("tok0001"):
            out = m(None, base, None)
        ok(float(out[0, 4]) == float(np.float32(EYT.YAW_SCALE * -0.5)) and torch.equal(out[0, [0, 1, 2, 3, 5, 6]], base[0, [0, 1, 2, 3, 5, 6]]) and float(base[0, 4]) == 0.0, "slot 4 set, the rest bit-identical, the input NOT mutated")
        ok(float(shim._token is None) == 1.0, "the token context was restored")
        shim.set_token(["tok0000", "tok0003"])
        out2 = m(None, torch.zeros(2, 7), None)
        shim.clear()
        ok(out2[:, 4].tolist() == [float(np.float32(EYT.YAW_SCALE * 0.25)), float(np.float32(EYT.YAW_SCALE * 0.3333333333333333))], out2[:, 4].tolist())
        try:
            with shim.token("nope"):
                m(None, base, None)
            ok(False, "an unknown token was served")
        except RuntimeError as e:
            ok("NOT in the yaw table" in str(e))
        try:
            with shim.token("tok0001"):
                m(None, torch.full((1, 7), 0.5), None)
            ok(False, "an incoming non-zero ego[4] was overwritten in strict mode")
        except RuntimeError as e:
            ok("two sources" in str(e))
        # token_source wins over set_token (the planner path)
        shim.remove()
        ok(torch.equal(m(None, base, None), before), "remove() restores the model call exactly")
        holder = type("P", (), {})()
        holder.model = m
        holder._scenario = type("S", (), {"_initial_lidar_token": "tok0002"})()
        # a checkpoint that wants yaw: a stub file with the meta the trainer writes
        ck = td / "yawckpt.pt"
        torch.save({"model": {}, "meta": {"ego_yaw": "nuplan_angular_rate_z", "ego_yaw_scale": EYT.YAW_SCALE, "ego_yaw_table_sha256": "c" * 64, "ego_yaw_slot": 4}}, ck)
        noscale = td / "noscale.pt"
        torch.save({"model": {}, "meta": {"ego_yaw": "nuplan_angular_rate_z", "ego_yaw_table_sha256": "c" * 64, "ego_yaw_slot": 4}}, noscale)
        scale3 = td / "scale3.pt"
        torch.save({"model": {}, "meta": {"ego_yaw": "nuplan_angular_rate_z", "ego_yaw_scale": 3.0, "ego_yaw_table_sha256": "c" * 64, "ego_yaw_slot": 4}}, scale3)
        plain = td / "plain.pt"
        torch.save({"model": {}, "meta": {"backbone": "vits16"}}, plain)
        ok(EY.attach_to_planner(holder, str(plain), tab_path) is None and m.forward.__func__ is _Stub.forward if hasattr(m.forward, "__func__") else True, "a non-yaw checkpoint: nothing is wrapped")
        ok(not hasattr(m, "__dict__") or "forward" not in m.__dict__, "a non-yaw checkpoint leaves the model's forward untouched")
        sh = EY.attach_to_planner(holder, str(ck), tab_path)
        ok(sh is not None and holder.ego_yaw_shim is sh and sh.train_table_sha256 == "c" * 64)
        out3 = m(None, base, None)
        ok(float(out3[0, 4]) == float(np.float32(EYT.YAW_SCALE * 0.0078125)), "the planner path reads the scenario's initial token")
        ok(sh.report()["n_calls"] == 1 and sh.report()["n_rows"] == 1)
        try:
            sh.install(m)
            ok(False, "double install accepted")
        except RuntimeError:
            ok(True)
        sh.remove()
        # from_checkpoint refusals
        ok(EY.EgoYawShim.from_checkpoint(str(plain), None) is None)
        # MM 2026-11-06: a yaw-trained checkpoint WITHOUT a scale is refused (the multiplier it was trained with is unknown)
        try:
            EY.EgoYawShim.from_checkpoint(str(noscale), tab_path)
            ok(False, "a yaw checkpoint without ego_yaw_scale was accepted")
        except RuntimeError as e:
            ok("ego_yaw_scale" in str(e) and "REFUSED" in str(e))
        for bad in (0.0, float("nan"), float("inf"), "10"):
            try:
                EY.EgoYawShim(tab, sha, bad)
                ok(False, f"scale {bad!r} accepted")
            except RuntimeError:
                ok(True)
        # the shim multiplies by the META's scale, not by a number of its own: a checkpoint trained at 3.0 is fed 3.0 x rad/s even while ego_yaw_table.YAW_SCALE is 10.0
        sh3 = EY.EgoYawShim.from_checkpoint(str(scale3), tab_path)
        sh3.install(m)
        with sh3.token("tok0001"):
            ok(float(m(None, torch.zeros(1, 7), None)[0, 4]) == float(np.float32(3.0 * -0.5)) and sh3.scale == 3.0)
        sh3.remove()
        try:
            EY.EgoYawShim.from_checkpoint(str(ck), None)
            ok(False, "a yaw checkpoint with no eval table was accepted")
        except RuntimeError as e:
            ok("train/eval mismatch" in str(e))
        other = td / "other.json.gz"
        with gzip.open(other, "wt", encoding="utf-8") as f:
            json.dump({"meta": {"schema": EYT.SCHEMA, "field": "something_else", "n_tokens": 1}, "table": {"a": 0.5}}, f)
        try:
            EY.EgoYawShim.from_checkpoint(str(ck), str(other))
            ok(False, "a table of another field was accepted")
        except RuntimeError as e:
            ok("is not" in str(e))
        # allow_incoming replaces (the closed-loop sensor case is NOT this module's, but the switch exists and is explicit)
        sh2 = EY.EgoYawShim.from_checkpoint(str(ck), tab_path, allow_incoming=True)
        sh2.install(m)
        with sh2.token("tok0001"):
            ok(float(m(None, torch.full((1, 7), 0.5), None)[0, 4]) == float(np.float32(EYT.YAW_SCALE * -0.5)))
        sh2.remove()
        # patch_planner_class on a stub module named `planner` (the real one needs the devkit and a ViT-L)
        import types
        stub = types.ModuleType("planner")

        class REFePlanner:
            def __init__(self, checkpoint=None, **k):
                self.model = _Stub()
                self._scenario = type("S", (), {"_initial_lidar_token": "tok0000"})()
        stub.REFePlanner = REFePlanner
        saved = sys.modules.get("planner")
        sys.modules["planner"] = stub
        try:
            undo = EY.patch_planner_class(tab_path)
            p1 = REFePlanner(checkpoint=str(ck))
            p2 = REFePlanner(checkpoint=str(plain))
            ok(hasattr(p1, "ego_yaw_shim") and not hasattr(p2, "ego_yaw_shim"))
            ok(float(p1.model(None, torch.zeros(1, 7), None)[0, 4]) == float(np.float32(EYT.YAW_SCALE * 0.25)) and float(p2.model(None, torch.zeros(1, 7), None)[0, 4]) == 0.0)
            undo()
            ok(not hasattr(REFePlanner(checkpoint=str(ck)), "ego_yaw_shim"))
            # the launcher: an UNMODIFIED script that builds the planner itself (the navtest seam) gets the shim, and its argv arrives intact
            script = td / "fake_seam.py"
            script.write_text("import json, sys\nimport planner as PL\np = PL.REFePlanner(checkpoint=sys.argv[1])\n"
                              "import torch\nout = p.model(None, torch.zeros(1, 7), None)\n"
                              "json.dump({'argv': sys.argv, 'shim': hasattr(p, 'ego_yaw_shim'), 'yaw': float(out[0, 4])}, open(sys.argv[2], 'w'))\n", encoding="utf-8")
            res = td / "res.json"
            argv0 = list(sys.argv)
            try:
                rc = EY.main(["--table", tab_path, "--", str(script), str(ck), str(res)])
            finally:
                sys.argv = argv0
            r = json.loads(res.read_text(encoding="utf-8"))
            ok(rc == 0 and r["shim"] is True and r["yaw"] == float(np.float32(EYT.YAW_SCALE * 0.25)) and r["argv"][1:] == [str(ck), str(res)], r)
            ok(EY.main(["--table", tab_path]) == 2, "no `--`: usage, rc 2")
        finally:
            if saved is None:
                sys.modules.pop("planner", None)
            else:
                sys.modules["planner"] = saved
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_shim_real_model():
    """The shim on the REAL REFe (ViT-S, CPU, no weights): the model's output with the shim at token t equals the output with ego[4] set to table[t] BY HAND, and differs from ego[4] == 0."""
    import ego_yaw as EY
    from model import REFe
    td = Path(tempfile.mkdtemp(prefix="yt_shimr_"))
    try:
        tab_path = make_table(td)
        tab, _m, sha = EYT.load(tab_path)
        cfg = cfg_for_bank()
        torch.manual_seed(0)
        model = REFe(cfg).eval()
        img = torch.randn(1, 3, cfg.img_h, cfg.img_w)
        ego = torch.zeros(1, cfg.ego_dim)
        ego[0, :4] = torch.tensor([3.0, 0.1, 0.2, -0.1])
        ego[0, 6] = 3.0
        goal = torch.zeros(1, 2 * cfg.n_goal_points)
        goal[0, 0] = 20.0
        with torch.no_grad():
            ref0 = model(img, ego, goal)
            e2 = ego.clone()
            e2[0, 4] = float(np.float32(EYT.YAW_SCALE * tab["tok0001"]))
            ref1 = model(img, e2, goal)
            shim = EY.EgoYawShim(tab, sha, EYT.YAW_SCALE).install(model)
            with shim.token("tok0001"):
                got = model(img, ego, goal)
            shim.remove()
            again = model(img, ego, goal)
        ok(torch.equal(got[0], ref1[0]) and torch.equal(got[1], ref1[1]), "shim == setting ego[4] by hand, bit for bit")
        ok(not torch.equal(got[0], ref0[0]), "and ego[4] reaches the output (the shim is not inert)")
        ok(torch.equal(again[0], ref0[0]), "removed: the original behaviour")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_scale_single_source():
    """MM 2026-11-06: YAW_SCALE = 10.0 has ONE definition (ego_yaw_table.py), imported by the trainer, the shim and the tests; the trainer READS it at run time (not a copy), the checkpoint meta
    records it, and the shim multiplies by the META's value."""
    import ast
    import re
    ok(EYT.YAW_SCALE == 10.0)
    defs = []
    for f in sorted(HERE.glob("*.py")):
        if f.name.startswith("test_"):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.upper() == "YAW_SCALE":
                        defs.append((f.name, node.lineno))
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value in ("ego_yaw_scale",) and isinstance(v, ast.Constant):
                        defs.append((f.name, node.lineno, "literal in a dict"))
    ok(defs == [("ego_yaw_table.py", defs[0][1])] if defs else False, defs)
    tr = (HERE / "train.py").read_text(encoding="utf-8")
    ey = (HERE / "ego_yaw.py").read_text(encoding="utf-8")
    ok(len(re.findall(r"EYT\.YAW_SCALE", tr)) >= 4 and 'meta["ego_yaw_scale"] = EYT.YAW_SCALE' in tr and "EYT.YAW_SCALE * self.ego_yaw[tok]" in tr, "the trainer imports the constant")
    ok(not re.search(r"\b10(\.0)?\b", ey.split('"""', 2)[2]), "the shim types no number of its own")
    # the trainer reads the constant at run time: moving it moves the loader's output
    import train as T
    td = Path(tempfile.mkdtemp(prefix="yt_sc_"))
    try:
        bank, tp = make_bank(td), None
        tp = make_table(td)
        ds = T.TargetBank(bank, None, cfg_for_bank(), synthetic=True, ego_yaw_table=tp)
        i = next(k for k, r in enumerate(ds.rows) if r["token"] == "tok0001")
        before = float(ds[i][1][4])
        saved = EYT.YAW_SCALE
        EYT.YAW_SCALE = 7.0
        try:
            ok(float(ds[i][1][4]) == float(np.float32(7.0 * -0.5)) and before == float(np.float32(10.0 * -0.5)), "the loader reads ego_yaw_table.YAW_SCALE at call time")
        finally:
            EYT.YAW_SCALE = saved
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_pins_untouched():
    """The pinned harness and the planner are byte-identical to the tip; `eval/harness_pins.py --verify` passes."""
    pl = HERE / "planner.py"
    ok(git_hash_object(pl) == TIP_PLANNER_BLOB, "refe/planner.py is byte-identical to the tip")
    pins = HERE.parent / "eval" / "harness_pins.py"
    if not pins.exists():
        return skip("harness_pins.py not exported in this tree")
    r = subprocess.run([sys.executable, str(pins), "--verify"], capture_output=True, text=True, cwd=str(HERE.parent))
    ok(r.returncode == 0, (r.stdout + r.stderr)[-400:])


def chk_loader_vs_tip():
    """LOCAL-ONLY (SLOW): the flag-less loader == the TIP's loader, item for item.  It needs the dev box's git mirror, so it must not ride a pod's G-SUITE: there it printed ZZTEST_SKIP, and a skip
    FAILS the gate (AMENDMENT YAW package, 2026-11-07; it sat inside chk_loader before).  The flag-less behaviour is still asserted on the pod by chk_loader (no flag: ego is the bank's own vector)."""
    import train as T
    if not MIRROR.exists():
        return skip("default identity against the tip's train.py: mirror absent")
    td = Path(tempfile.mkdtemp(prefix="yt_ldtip_"))
    try:
        bank = make_bank(td)
        cfg = cfg_for_bank()
        d0 = T.TargetBank(bank, None, cfg, synthetic=True)
        tip_src = subprocess.run(["git", "--git-dir", str(MIRROR), "cat-file", "blob", TIP_TRAIN_BLOB], capture_output=True, check=True).stdout
        tipd = td / "tip"
        tipd.mkdir()
        for f in HERE.glob("*.py"):
            if f.name != "train.py":
                shutil.copyfile(f, tipd / f.name)
        (tipd / "train.py").write_bytes(tip_src)
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_tip", str(tipd / "train.py"))
        TT = importlib.util.module_from_spec(spec)
        sys.modules["train_tip"] = TT
        spec.loader.exec_module(TT)
        dt = TT.TargetBank(bank, None, cfg, synthetic=True)
        ok(len(dt) == len(d0) and all(same_item(dt[i], d0[i]) for i in range(len(d0))), "default identity: this tree's loader == the tip's loader, every item bit for bit")
        ok("ego_yaw_table" not in TT.build_parser().parse_args(["--targets", bank]).__dict__, "the tip has no such flag (the identity above is against a loader that cannot know it)")
    finally:
        shutil.rmtree(td, ignore_errors=True)


CHECKS = {"table": chk_table, "loader": chk_loader, "ckpt_meta": chk_ckpt_meta, "gate_registry": chk_gate_registry, "dvb_flag_set": chk_dvb_flag_set,
          "dvb_red_not_passed": chk_dvb_red_not_passed, "live_flag_set": chk_live_flag_set, "live_control_unset": chk_live_control_unset,
          "live_red_override_removed": chk_live_red_override_removed, "shim": chk_shim, "shim_real_model": chk_shim_real_model, "scale_single_source": chk_scale_single_source}
# local-only: they need the dev box's DBs / tables / eval tree, or pin a SHA of a file that a later landing may legitimately change (so they must not ride a pod's G-SUITE)
SLOW = {"real_table_control": chk_real_table_control, "real_table_tokens": chk_real_table_tokens, "pins_untouched": chk_pins_untouched, "loader_vs_tip": chk_loader_vs_tip}

# (file, old, new, the check that must go RED).  Each DELETES / neutralises ONE emitting line of a COPY of this tree.
MUTATIONS = [
    ("train.py", "            ego_t[EYT.EGO_YAW_SLOT] = EYT.YAW_SCALE * self.ego_yaw[tok]\n", "", "live_flag_set"),                              # the override line itself
    ("train.py", "            ego_t[EYT.EGO_YAW_SLOT] = EYT.YAW_SCALE * self.ego_yaw[tok]\n", "            ego_t[EYT.EGO_YAW_SLOT] = self.ego_yaw[tok]\n", "loader"),   # the override without the scale
    ("train.py", '            meta["ego_yaw_scale"] = EYT.YAW_SCALE', "            pass", "ckpt_meta"),                                                          # the meta no longer records the scale
    ("ego_yaw.py", "[self.scale * self.table[t] for t in toks]", "[self.table[t] for t in toks]", "shim"),                                                # the shim ignores the scale
    ("train.py", "                    ego_yaw_table=a.ego_yaw_table or None,                    # WP YAW (None = the unchanged loader)\n", "                    ego_yaw_table=None,\n", "dvb_flag_set"),   # the call site forgets the table
    ("train.py", '            meta["ego_yaw"] = EYT.YAW_TAG\n', "", "ckpt_meta"),                                                              # the checkpoint no longer says so
    ("train.py", "            lacking = [r for r in self.rows if r.get(\"token\", \"\") not in self.ego_yaw]", "            lacking = []", "loader"),            # the missing-token refusal
    ("gate_checks.py", '    "ego_yaw_table": "built",\n', "", "gate_registry"),                                                              # the G-DVB registry entry
    ("gate_checks.py", "            reasons += judge_ego_yaw(f_, p_)\n", "", "live_red_override_removed"),                                            # the G-LIVE check is not wired in
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="ytmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:60]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_ego_yaw.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
            print(f"  mutation RED as required: {fn} [{old.strip()[:56]}] -> {check}", flush=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        {**CHECKS, **SLOW}[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        for name, fn in SLOW.items():
            fn()
            print(f"  {name}: ok ({N})", flush=True)
        run_mutations()
    print(f"ZZEGOYAW_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
