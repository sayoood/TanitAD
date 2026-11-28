#!/usr/bin/env python3
"""Tests for P7-WIRE-RESERVE: refe/session_reserve.py, the P7-only subcommands of refe/p6_prep.py, and the dormant section 5a of refe/p6_pod.sh.

No pod, no GPU, no network.  Every guard has a DELIBERATE-REGRESSION arm that must turn RED (a check that cannot go RED is not a check):
  * a planted leaked bank row (rank 0) and a planted leaked twin (rank 1) -> `audit_inputs` says so, `filter_bank` removes them;
  * a planted D9 cache key (a logged-route part and a lane-rank twin part), a planted label-store shard, a stale manifest, an image zip / frame dir / reference, a calibration row;
  * a LOGLESS key in every input (bank row with no `log_name`, a file at the cache root, a shard whose header disagrees, a zip that names no log, a calibration key '') -> REFUSED / unresolved, never passed;
  * a reserve file with one byte changed -> `load_reserve` refuses (and the REAL chain refuses before it touches anything);
  * the REAL p6_pod.sh in the sandbox: with P7_RESERVE set the reserve leaves every input and ARGS names the reserve-free copies; a leak planted after the fact, a P6 run directory, a held P6 lock and a
    logless bank row each REFUSE with nothing moved; with P7_RESERVE UNSET the chain's call log is identical to the tip's p6_pod.sh (P6's default path is untouched).
`raw/mutation_check.py` deletes the single exclusion line of `filter_bank` (and eight other lines) and requires this suite to go RED.

Run:  python refe/test_session_reserve.py [--only SUBSTRING]   -> ZZSESSRES_PASS <n>      (also collectable by pytest)
Needs bash + numpy for the sandbox / label-store tests (the driverl-eval venv); when they are absent a plain ZZINFO line says so and `ZZSESSRES_PASS` lists what did NOT run.  Never prints ZZTEST_SKIP.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_reserve as SR  # noqa: E402
import p6_prep as PP  # noqa: E402

FIXTURE = HERE / "fixtures" / "p7_reserve_sessions.json"                       # byte-identical to PK/raw/2026-10-22-p7-session-reserve/raw/reserve_sessions.json (its sha256 is pinned in the module)
PKG_RAW = HERE.parent / "raw" / "2026-10-22-p7-session-reserve" / "raw"
TOKEN_TABLE = PKG_RAW / "src" / "navtrain_token_table.tsv.gz"
OPTIONAL = {"ran": [], "absent": []}


def info(msg):
    print("ZZINFO " + msg, flush=True)


def optional(name, available, why=""):
    """An optional real-data / tool check: a PLAIN ZZINFO line when absent (never ZZTEST_SKIP), and the run's summary lists it."""
    (OPTIONAL["ran"] if available else OPTIONAL["absent"]).append(name)
    if not available:
        info(f"{name} ABSENT ({why}); not counted as run")
    return available


# ------------------------------------------------------------------------------------------------ synthetic data
S_RES1, S_RES2 = "2021.06.07.17.46.49_veh-35", "2021.09.14.16.46.51_veh-45"             # two sessions of the synthetic reserve
S_TR1, S_TR2, S_TR3 = "2021.05.12.19.36.12_veh-35", "2021.06.09.14.50.36_veh-26", "2021.10.01.19.16.42_veh-28"
RES = frozenset({S_RES1, S_RES2})


def mk(session, a=0, b=100):
    return f"{session}_{a:05d}_{b:05d}"


R1A, R1B, R2A = mk(S_RES1, 10, 200), mk(S_RES1, 300, 400), mk(S_RES2, 5, 50)          # reserved logs (R1A, R1B share a session)
T1A, T2A, T3A = mk(S_TR1, 1, 99), mk(S_TR2, 7, 70), mk(S_TR3, 3, 33)                    # training logs
ALL_LOGS = (R1A, R1B, R2A, T1A, T2A, T3A)


def row(log, tok, rank=0, aug=None, **kw):
    r = {"log_name": log, "token": tok, "step": 0, "rank": rank}
    if aug is not None:
        r["aug"] = aug
    r.update(kw)
    return r


GH = {"kind": "goal_horizon", "value": 16}
LR1 = {"kind": "lane_rank", "value": 1}


def bank_rows():
    """(rank0 rows, rank1 rows) over ALL_LOGS, two tokens per log; rank-1 twins of three tokens (one reserved, two training), both kinds."""
    r0 = [row(lg, t, command=1) for lg in ALL_LOGS for t in ("a1", "b2")]
    r1 = [row(R1A, "a1", 1, GH), row(R1A, "a1", 1, LR1), row(R2A, "b2", 1, LR1), row(T1A, "a1", 1, GH), row(T1A, "a1", 1, LR1), row(T3A, "b2", 1, LR1)]
    return r0, r1


def write_jsonl(path, rows, newline="\n"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"".join(json.dumps(r).encode() + newline.encode() for r in rows))


def write_bank(d, r0, r1):
    write_jsonl(Path(d) / "targets_rank0.jsonl", r0)
    write_jsonl(Path(d) / "targets_rank1.jsonl", r1)


def write_aux_shard(store, log, tokens, map_name="us-nv-las-vegas-strip", header_log=None):
    """A stdlib twin of aux_label_store.write_shard (same byte layout: magic | u32 LE header_len | header JSON | blobs)."""
    entries, blobs, off = [], [], 0
    for t in sorted(tokens):
        b = b"\x00" * 8
        entries.append({"t": t, "o": off, "l": len(b), "s": "bank", "p": [0.0, 0.0, 0.0], "q": None})
        blobs.append(b)
        off += len(b)
    head = json.dumps({"schema": "aux-drv-label-v1", "log": header_log or log, "map": map_name, "raster": {}, "entries": entries}, separators=(",", ":")).encode("utf-8")
    p = Path(store) / "shards" / f"{log}.aux1"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(SR._AUX_MAGIC + struct.pack("<I", len(head)) + head + b"".join(blobs))
    return p


def write_manifest(store, logs):
    files = {f"shards/{lg}.aux1": {"bytes": 1, "sha256": "0" * 64, "n": 2, "map": "m"} for lg in logs}
    (Path(store) / "manifest.json").write_text(json.dumps({"schema": "aux-drv-label-v1", "files": files}), encoding="utf-8")


def touch(p, data=b""):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def make_inputs(root, logs=ALL_LOGS):
    """A complete synthetic input set over `logs`: bank dir, D9 cache, label store (+ manifest), image root, calibration table."""
    root = Path(root)
    r0 = [row(lg, t, command=1) for lg in logs for t in ("a1", "b2")]
    r1 = [row(logs[0], "a1", 1, GH), row(logs[-1], "b2", 1, LR1)]
    write_bank(root / "bank", r0, r1)
    for lg in logs:
        for t in ("a1", "b2"):
            touch(root / "mc" / lg / f"{t}.pkl.xz", b"p")
        touch(root / "mc" / lg / "a1__lane_rank_1.pkl.xz", b"p")
        write_aux_shard(root / "store", lg, ("a1", "b2"))
        touch(root / "pix" / "navtrain_current_1" / lg / "CAM_F0" / "h1.jpg", b"j")
    touch(root / "pix" / f"{logs[0]}_CAM_F0.zip", b"z")
    touch(root / "pix" / ".done_1")
    write_manifest(root / "store", logs)
    (root / "calib.json").write_text(json.dumps({"fields": ["fx"], "logs": {lg: {"CAM_F0": [1.0]} for lg in logs}}), encoding="utf-8")
    return root


def audit(root, reserve=RES, **over):
    root = Path(root)
    kw = dict(bank_files=sorted((root / "bank").glob("targets_rank*.jsonl")), metric_cache_dir=root / "mc", label_store=root / "store", image_index=root / "pix", calib_rows=root / "calib.json")
    kw.update(over)
    return SR.audit_inputs(reserve, **kw)


def tree(root):
    """Sorted (relative path, size or None for a dir) of everything under root: a snapshot a restore must reproduce exactly."""
    out = []
    for dp, dn, fn in os.walk(root):
        for d in dn:
            out.append((os.path.relpath(os.path.join(dp, d), root).replace("\\", "/"), None))
        for f in fn:
            out.append((os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"), os.path.getsize(os.path.join(dp, f))))
    return sorted(out)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ 1. the unit and the reserve file
class UnitTests(unittest.TestCase):
    def test_session_of_and_its_refusals(self):
        self.assertEqual(SR.session_of("2021.06.07.17.46.49_veh-35_02607_03120"), "2021.06.07.17.46.49_veh-35")
        self.assertEqual(SR.session_of(R1A), S_RES1)
        for bad in ("L0", "", None, 5, b"2021.06.07.17.46.49_veh-35_02607_03120", S_RES1, S_RES1 + "_0001", "2021.06.07.17.46.49_veh-35_02607_03120\n", " " + R1A, "veh-35_02607_03120",
                    "2021.06.07.17.46.49_veh-35_02607_03120_1"):
            with self.assertRaises(SR.UnresolvedLog, msg=repr(bad)):
                SR.session_of(bad)
        # the bare rsplit would have PASSED the first of these as a "session" that is in no reserve: the exact failure the refusal exists for
        self.assertEqual("L0".rsplit("_", 2)[0], "L0")

    def test_is_reserved_needs_a_resolvable_log(self):
        self.assertTrue(SR.is_reserved(R1A, RES))
        self.assertFalse(SR.is_reserved(T1A, RES))
        with self.assertRaises(SR.UnresolvedLog):
            SR.is_reserved("L0", RES)

    def test_reserve_must_be_a_nonempty_set_of_sessions(self):
        for bad in (frozenset(), set(), "2021.06.07.17.46.49_veh-35", frozenset({"L0"}), frozenset({R1A}), None):
            with self.assertRaises(SR.ReserveError, msg=repr(bad)):
                SR._as_reserve(bad)


class ReserveFileTests(unittest.TestCase):
    def test_fixture_is_the_registered_draw(self):
        res = SR.load_reserve(FIXTURE)
        self.assertIsInstance(res, frozenset)
        self.assertEqual(len(res), 16)
        logs = SR.reserve_logs(FIXTURE)
        self.assertEqual(len(logs), 116)
        self.assertEqual({SR.session_of(lg) for lg in logs}, set(res))
        self.assertEqual(SR.sha256_file(FIXTURE), SR.RESERVE_FILE_SHA256)
        doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(sum(x["tokens"] for x in doc["sessions"]), SR.RESERVE_SCENES)             # 10,141
        self.assertEqual(SR.BANK_RANK0_SCENES - SR.RESERVE_SCENES, doc["reserve"]["bank_scenes_left_for_training"])
        steps = 25 * math.ceil((SR.BANK_RANK0_SCENES - SR.RESERVE_SCENES) / 256)
        self.assertEqual(steps, 9075)                                                               # SESSION_RESERVE.md section 0: 25 x ceil(92,896 / 256)

    def test_one_byte_changed_refuses(self):
        raw = FIXTURE.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            for pos in (0, 100, len(raw) // 2, len(raw) - 2):                     # a brace, a letter, mid-file, near the end
                b = bytearray(raw)
                b[pos] ^= 0x01
                p = touch(Path(tmp) / f"m{pos}.json", bytes(b))
                with self.assertRaises(SR.ReserveError, msg=f"byte {pos}"):
                    SR.load_reserve(p)
            for name, data in (("trunc", raw[:-1]), ("append", raw + b" "), ("empty", b""), ("sess", raw.replace(S_RES1.encode(), S_TR1.encode(), 1))):
                with self.assertRaises(SR.ReserveError, msg=name):
                    SR.load_reserve(touch(Path(tmp) / f"{name}.json", data))
            with self.assertRaises((SR.ReserveError, OSError)):
                SR.load_reserve(Path(tmp) / "absent.json")

    def test_crlf_checkout_is_the_same_text(self):
        """MEASURED 2026-10-22: `git archive` on the dev box (core.autocrlf=true) exports this file as CRLF, 19,971 B / sha 4f5b1f1d...; the blob is 19,121 B / 53f3bcd4....  A CRLF checkout must still load."""
        raw = FIXTURE.read_bytes()
        self.assertNotIn(b"\r", raw)
        with tempfile.TemporaryDirectory() as tmp:
            crlf = touch(Path(tmp) / "crlf.json", raw.replace(b"\n", b"\r\n"))
            self.assertNotEqual(SR.sha256_file(crlf), SR.RESERVE_FILE_SHA256)
            self.assertEqual(SR.load_reserve(crlf), SR.load_reserve(FIXTURE))

    def test_the_content_checks_are_live(self):
        """The file hash alone is not the only guard: with the pinned hash re-pointed at an altered file, the schema / count / digest checks must still refuse (a check that cannot go RED is dead code)."""
        doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cases = {}
        d = json.loads(json.dumps(doc)); d["schema"] = "other"; cases["schema"] = d
        d = json.loads(json.dumps(doc)); d["sessions"] = d["sessions"][:-1]; cases["15 sessions"] = d
        d = json.loads(json.dumps(doc)); d["sessions"][0]["logs"][0]["log"] = mk(S_TR1, 1, 2); cases["a log outside its session"] = d
        d = json.loads(json.dumps(doc)); d["reserve"]["logs_sha256"] = "0" * 64; cases["digest"] = d
        with tempfile.TemporaryDirectory() as tmp:
            for name, dd in cases.items():
                p = touch(Path(tmp) / "x.json", (json.dumps(dd, indent=1, sort_keys=True) + "\n").encode())
                with mock.patch.object(SR, "RESERVE_FILE_SHA256", SR.sha256_file(p)):
                    with self.assertRaises(SR.ReserveError, msg=name):
                        SR.load_reserve(p)
            p = touch(Path(tmp) / "ok.json", (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())             # control: the unaltered content under a re-pointed pin loads
            with mock.patch.object(SR, "RESERVE_FILE_SHA256", SR.sha256_file(p)):
                self.assertEqual(len(SR.load_reserve(p)), 16)


# ------------------------------------------------------------------------------------------------ 2. filter_bank
class FilterTests(unittest.TestCase):
    def test_rank0_and_twins_are_excluded_by_their_own_log(self):
        r0, r1 = bank_rows()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_bank(tmp / "in", r0, r1)
            a = SR.filter_bank(tmp / "in" / "targets_rank0.jsonl", tmp / "out" / "targets_rank0.jsonl", RES)
            b = SR.filter_bank(tmp / "in" / "targets_rank1.jsonl", tmp / "out" / "targets_rank1.jsonl", RES)
            self.assertEqual((a["n_in"], a["n_out"], a["n_excluded"], a["n_unresolved"]), (12, 6, 6, 0))
            self.assertEqual((b["n_in"], b["n_out"], b["n_excluded"], b["n_unresolved"]), (6, 3, 3, 0))     # the 3 twins of reserved logs go, the 3 of training logs stay
            out1 = [json.loads(x) for x in (tmp / "out" / "targets_rank1.jsonl").read_text().splitlines()]
            self.assertEqual([(r["log_name"], r["aug"]["kind"]) for r in out1], [(T1A, "goal_horizon"), (T1A, "lane_rank"), (T3A, "lane_rank")])
            for r in out1 + [json.loads(x) for x in (tmp / "out" / "targets_rank0.jsonl").read_text().splitlines()]:
                self.assertNotIn(SR.session_of(r["log_name"]), RES)
            self.assertEqual(a["sha256_out"], sha(tmp / "out" / "targets_rank0.jsonl"))
            self.assertEqual(a["sha256_in"], sha(tmp / "in" / "targets_rank0.jsonl"))
            self.assertEqual(b["by_rank_excluded"], {"1": 3})
            self.assertEqual(sorted(a["excluded_sessions"]), sorted(RES))

    def test_survivors_are_byte_for_byte(self):
        """Never re-serialised: odd spacing, key order, a CRLF line and a missing final newline survive unchanged (the last gains only its newline)."""
        lines = [b'{"rank": 0,   "token": "a1", "log_name": "%s", "x": 1.50}\n' % T1A.encode(),
                 b'{"log_name": "%s", "token": "z9", "rank": 0}\r\n' % R1A.encode(),
                 b'{"token": "b2", "log_name": "%s", "rank":0}\r\n' % T2A.encode(),
                 b'\n',
                 b'{"log_name": "%s","token":"c3","rank":0}' % T3A.encode()]
        with tempfile.TemporaryDirectory() as tmp:
            src = touch(Path(tmp) / "in.jsonl", b"".join(lines))
            r = SR.filter_bank(src, Path(tmp) / "out.jsonl", RES)
            self.assertEqual((r["n_in"], r["n_out"], r["n_excluded"]), (4, 3, 1))                  # the blank line is not a row
            self.assertEqual((Path(tmp) / "out.jsonl").read_bytes(), lines[0] + lines[2] + lines[4] + b"\n")
            self.assertEqual(r["sha256_out"], sha(Path(tmp) / "out.jsonl"))

    def test_a_row_with_no_resolvable_log_is_refused_and_nothing_is_written(self):
        bads = [{"token": "a1", "rank": 0}, {"log_name": "", "token": "a1"}, {"log_name": None, "token": "a1"}, {"log_name": "L0", "token": "a1"}, {"log_name": S_RES1, "token": "a1"}, {"log_name": 7}]
        with tempfile.TemporaryDirectory() as tmp:
            for i, bad in enumerate(bads):
                src = Path(tmp) / f"in{i}.jsonl"
                write_jsonl(src, [row(T1A, "ok1"), bad, row(R1A, "ok2")])
                out = Path(tmp) / f"out{i}.jsonl"
                with self.assertRaises(SR.UnresolvedLog, msg=repr(bad)):
                    SR.filter_bank(src, out, RES)
                self.assertFalse(out.exists() or out.with_name(out.name + ".tmp").exists(), "a refused filter must leave no output")
            src = touch(Path(tmp) / "junk.jsonl", json.dumps(row(T1A, "x")).encode() + b"\nnot json\n[1, 2]\n")                  # unparseable and non-object lines
            with self.assertRaises(SR.UnresolvedLog) as cm:
                SR.filter_bank(src, Path(tmp) / "junk.out", RES)
            self.assertIn("2 of 3", str(cm.exception))
            self.assertFalse((Path(tmp) / "junk.out").exists())

    def test_in_place_and_missing_and_empty_reserve_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = touch(Path(tmp) / "in.jsonl", json.dumps(row(T1A, "x")).encode() + b"\n")
            with self.assertRaises(SR.ReserveError):
                SR.filter_bank(src, src, RES)
            with self.assertRaises(SR.ReserveError):
                SR.filter_bank(Path(tmp) / "absent.jsonl", Path(tmp) / "o.jsonl", RES)
            with self.assertRaises(SR.ReserveError):
                SR.filter_bank(src, Path(tmp) / "o.jsonl", frozenset())

    def test_token_table_cross_check_catches_a_twin_with_the_wrong_parent_log(self):
        r0, r1 = bank_rows()
        tl = {"a1": R1A, "b2": R1A}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = [row(R1A, "a1", 1, GH), row(R1A, "zz", 1, GH)]                      # zz is not in the map: counted, not refused
            write_jsonl(tmp / "g.jsonl", good)
            r = SR.filter_bank(tmp / "g.jsonl", tmp / "g.out", RES, token_log=tl)
            self.assertEqual((r["n_token_checked"], r["n_token_unmapped"], r["n_excluded"]), (1, 1, 2))
            write_jsonl(tmp / "w.jsonl", [row(T1A, "a1", 1, LR1)])                      # a twin that names a TRAINING log for a token the table puts in a RESERVED log: it would have leaked
            with self.assertRaises(SR.AttributionConflict):
                SR.filter_bank(tmp / "w.jsonl", tmp / "w.out", RES, token_log=tl)
            self.assertFalse((tmp / "w.out").exists())

    def test_assert_rows_clear_is_the_consumer_side_refusal(self):
        r0, r1 = bank_rows()
        self.assertEqual(SR.assert_rows_clear([r for r in r0 + r1 if SR.session_of(r["log_name"]) not in RES], RES), 12 - 6 + 6 - 3)
        with self.assertRaises(SR.ReserveError) as cm:
            SR.assert_rows_clear(r0 + r1, RES, what="TargetBank rows")
        self.assertIn("9 of 18 TargetBank rows", str(cm.exception))
        with self.assertRaises(SR.UnresolvedLog):
            SR.assert_rows_clear([row(T1A, "a"), {"token": "no-log"}], RES)

    def test_filter_bank_dir_report_and_the_trainers_identity_digest(self):
        r0, r1 = bank_rows()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_bank(tmp / "in", r0, r1)
            rep = SR.filter_bank_dir(tmp / "in", tmp / "out", RES)
            self.assertEqual((rep["n_in"], rep["n_out"], rep["n_excluded"], rep["n_unresolved"]), (18, 9, 9, 0))
            self.assertEqual(rep["by_rank_in"], {"0": 12, "1": 6})
            self.assertEqual(rep["by_rank_out"], {"0": 6, "1": 3})
            h = hashlib.sha256()                                                        # train._digest_files: basename + bytes of each rank file, in sorted order
            for fp in sorted((tmp / "out").glob("targets_rank*.jsonl")):
                h.update(fp.name.encode())
                h.update(fp.read_bytes())
            self.assertEqual(rep["targets_sha256"], h.hexdigest())
            self.assertNotEqual(rep["targets_sha256"], rep["source_targets_sha256"])
            on_disk = json.loads((tmp / "out" / "reserve_filter_report.json").read_text())
            self.assertEqual(on_disk["targets_sha256"], rep["targets_sha256"])
            self.assertEqual(sorted(p.name for p in (tmp / "out").glob("targets_*.jsonl")), ["targets_rank0.jsonl", "targets_rank1.jsonl"])     # the report does not match the trainer's glob
            # a stray targets_*.jsonl would be a silently dropped arm (train.TargetBank refuses it): so does this
            write_jsonl(tmp / "in" / "targets_aug.jsonl", [row(T1A, "s")])
            with self.assertRaises(SR.ReserveError):
                SR.filter_bank_dir(tmp / "in", tmp / "out2", RES)
            with self.assertRaises(SR.ReserveError):
                SR.filter_bank_dir(tmp / "nothing", tmp / "out3", RES)


# ------------------------------------------------------------------------------------------------ 3. audit_inputs
class AuditTests(unittest.TestCase):
    def test_a_clean_input_set_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = audit(make_inputs(tmp, (T1A, T2A, T3A)))
            self.assertTrue(r["ok"], r)
            self.assertEqual(sum(len(v) for v in r["leaks"].values()), 0)
            self.assertEqual(sum(r["unresolved"].values()), 0)
            self.assertEqual(r["counts"]["inputs_audited"], ["bank", "calibration", "images", "label_store", "metric_cache"])
            self.assertGreater(min(r["counts"]["examined"].values()), 0)

    def test_the_unfiltered_input_set_is_flagged_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = audit(make_inputs(tmp, ALL_LOGS))
            self.assertFalse(r["ok"])
            for inp in ("bank", "metric_cache", "label_store", "images", "calibration"):
                self.assertTrue(r["leaks"][inp], inp)
            # the D9 cache lists every part of a reserved directory, twins included
            self.assertIn(f"{R1A}/a1__lane_rank_1.pkl.xz", r["leaks"]["metric_cache"])
            self.assertIn(f"{R1A}/a1.pkl.xz", r["leaks"]["metric_cache"])

    def test_planted_leaked_bank_row_rank0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            self.assertTrue(audit(root)["ok"])
            with open(root / "bank" / "targets_rank0.jsonl", "ab") as f:
                f.write(json.dumps(row(R1B, "planted", 0)).encode() + b"\n")
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(r["leaks"]["bank"], [f"{R1B}/planted|rank0"])

    def test_planted_leaked_twin_rank1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            with open(root / "bank" / "targets_rank1.jsonl", "ab") as f:
                f.write(json.dumps(row(R2A, "tw", 1, LR1)).encode() + b"\n")
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(r["leaks"]["bank"], [f"{R2A}/tw|rank1|lane_rank=1"])

    def test_planted_d9_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            touch(root / "mc" / R1A / "planted.pkl.xz", b"p")
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(r["leaks"]["metric_cache"], [f"{R1A}/planted.pkl.xz"])
            shutil.rmtree(root / "mc" / R1A)
            touch(root / "mc" / R2A / "tok__lane_rank_2.pkl.xz", b"p")                          # a TWIN part in a reserved log's directory
            self.assertEqual(audit(root)["leaks"]["metric_cache"], [f"{R2A}/tok__lane_rank_2.pkl.xz"])
            shutil.rmtree(root / "mc" / R2A)
            (root / "mc" / R2A).mkdir()                                                          # an EMPTY reserved directory is still a reserved key
            self.assertEqual(audit(root)["leaks"]["metric_cache"], [f"{R2A}/"])

    def test_planted_label_store_shard_and_stale_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            write_aux_shard(root / "store", R1A, ("a1", "b2", "c3"))
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(r["leaks"]["label_store"], [f"{R1A}/a1", f"{R1A}/b2", f"{R1A}/c3"])
            (root / "store" / "shards" / f"{R1A}.aux1").unlink()
            self.assertTrue(audit(root)["ok"])
            write_manifest(root / "store", (T1A, T2A, T3A, R2A))                                 # the shard moved but the manifest (and so its store_sha256) still lists it
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(r["leaks"]["label_store"], [f"manifest:{R2A}"])

    def test_planted_image_zip_frame_dir_and_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            touch(root / "pix" / f"{R1A}_CAM_L0.zip", b"z")
            touch(root / "pix" / "navtrain_current_2" / R2A / "CAM_F0" / "h.jpg", b"j")
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(r["leaks"]["images"], [f"dir:navtrain_current_2/{R2A}", f"zip:{R1A}_CAM_L0.zip"])
            # the other two forms an image index takes: an iterable of references and a {basename: path} index (build_targets._IMAGE_INDEX)
            refs = [f"/data/navtrain_pixels/navtrain_current_1/{T1A}/CAM_F0/h1.jpg", f"/data/pix/{R1A}_CAM_F0.zip::abcd.jpg"]
            r = audit(root, image_index=refs)
            self.assertEqual(r["leaks"]["images"], [f"ref:{refs[1]}"])
            r = audit(root, image_index={"h1.jpg": refs[0], "abcd.jpg": f"/x/navtrain_current_3/{R2A}/CAM_B0/abcd.jpg"})
            self.assertEqual(len(r["leaks"]["images"]), 1)
            self.assertFalse(r["ok"])

    def test_planted_calibration_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A, R1B))
            r = audit(root)
            self.assertEqual(r["leaks"]["calibration"], [f"calib:{R1B}"])
            for form in ({R1B: 1, T1A: 1}, {"logs": {R1B: 1}}, [R1B, T1A]):                     # a path, the table, a {log: rig} dict, an iterable
                self.assertEqual(audit(root, calib_rows=form)["leaks"]["calibration"], [f"calib:{R1B}"])

    def test_logless_keys_are_unresolved_never_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            self.assertTrue(audit(root)["ok"])
            with open(root / "bank" / "targets_rank1.jsonl", "ab") as f:
                f.write(json.dumps({"token": "x", "rank": 1}).encode() + b"\n")                  # no log_name
                f.write(b"{broken\n")
            touch(root / "mc" / "stray.pkl.xz", b"p")                                           # a file at the cache root: no log
            touch(root / "mc" / "L0" / "t.pkl.xz", b"p")                                         # a directory that is not a log name
            write_aux_shard(root / "store", T1A + "x", ("a",))                                    # a shard file that is not a log
            write_aux_shard(root / "store", mk(S_TR2, 8, 80), ("a",), header_log=T2A)             # header log != file name
            touch(root / "pix" / "oddly_named.zip", b"z")                                         # a zip that names no log
            (root / "calib.json").write_text(json.dumps({"fields": [], "logs": {"": {}, "L9": {}, T1A: {}}}), encoding="utf-8")
            r = audit(root)
            self.assertFalse(r["ok"])
            self.assertEqual(sum(len(v) for v in r["leaks"].values()), 0)                          # not a leak: UNRESOLVED
            self.assertEqual(r["unresolved"]["bank"], 2)
            self.assertEqual(r["unresolved"]["metric_cache"], 2)
            self.assertEqual(r["unresolved"]["label_store"], 2)
            self.assertEqual(r["unresolved"]["images"], 1)
            self.assertEqual(r["unresolved"]["calibration"], 2)
            self.assertTrue(r["counts"]["unresolved_examples"]["bank"])

    def test_an_empty_or_missing_input_is_not_a_clean_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            (root / "empty_mc").mkdir()
            r = audit(root, metric_cache_dir=root / "empty_mc")
            self.assertFalse(r["ok"])
            self.assertEqual(r["counts"]["empty_inputs"], ["metric_cache"])
            self.assertEqual(sum(len(v) for v in r["leaks"].values()), 0)
            with self.assertRaises(SR.ReserveError):
                audit(root, metric_cache_dir=root / "nope")
            with self.assertRaises(SR.ReserveError):
                audit(root, label_store=root / "mc")                                             # not a store (no shards/)
            with self.assertRaises(SR.ReserveError):
                SR.audit_inputs(RES, bank_files=[])
            with self.assertRaises(SR.ReserveError):
                SR.audit_inputs(RES, bank_files=[root / "bank" / "absent.jsonl"])
            with self.assertRaises(SR.ReserveError):
                SR.audit_inputs(frozenset(), bank_files=[root / "bank" / "targets_rank0.jsonl"])
            r = SR.audit_inputs(RES, bank_files=[root / "bank" / "targets_rank0.jsonl", root / "bank" / "targets_rank1.jsonl"])             # bank alone is a valid audit
            self.assertTrue(r["ok"] and r["counts"]["inputs_audited"] == ["bank"])

    def test_leak_lists_are_capped_but_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp, (T1A, T2A, T3A))
            with open(root / "bank" / "targets_rank0.jsonl", "ab") as f:
                for i in range(30):
                    f.write(json.dumps(row(R1A, f"p{i:02d}")).encode() + b"\n")
            r = audit(root, max_report=5)
            self.assertEqual(len(r["leaks"]["bank"]), 5)
            self.assertEqual(r["counts"]["leak_totals"]["bank"], 30)
            self.assertFalse(r["ok"])


# ------------------------------------------------------------------------------------------------ 4. calibration, moving the reserve out of a shared root, the label store
class MoveTests(unittest.TestCase):
    def test_filter_calib(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tab = {"fields": ["fx"], "n_logs": 4, "distinct_rigs_exact": {"CAM_F0": 99}, "unreadable": [],
                   "logs": {R1A: {"CAM_F0": [1.0]}, T1A: {"CAM_F0": [1.0]}, T2A: {"CAM_F0": [2.0]}, R2A: {"CAM_F0": [3.0]}}}
            (tmp / "c.json").write_text(json.dumps(tab), encoding="utf-8")
            r = SR.filter_calib(tmp / "c.json", tmp / "o" / "c.json", RES)
            out = json.loads((tmp / "o" / "c.json").read_text())
            self.assertEqual((r["logs_in"], r["logs_out"], r["excluded"]), (4, 2, 2))
            self.assertEqual(sorted(out["logs"]), sorted([T1A, T2A]))
            self.assertEqual((out["n_logs"], out["distinct_rigs_exact"], out["fields"]), (2, {"CAM_F0": 2}, ["fx"]))
            self.assertEqual(out["reserve_filtered"]["excluded_logs"], 2)
            tab["logs"]["L0"] = {}
            (tmp / "bad.json").write_text(json.dumps(tab), encoding="utf-8")
            with self.assertRaises(SR.UnresolvedLog):
                SR.filter_calib(tmp / "bad.json", tmp / "o2.json", RES)
            self.assertFalse((tmp / "o2.json").exists())

    def test_move_cache_and_images_then_restore_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp)
            before = tree(root / "mc"), tree(root / "pix")
            self.assertFalse(audit(root)["ok"])
            d = SR.move_reserved("cache", root / "mc", root / "mc_reserve", RES, dry_run=True)
            self.assertEqual((d["n_planned"], d["n_moved"]), (3, 0))
            self.assertEqual(tree(root / "mc"), before[0])                                      # a dry run moves nothing
            self.assertFalse((root / "mc_reserve").exists())
            c = SR.move_reserved("cache", root / "mc", root / "mc_reserve", RES)
            i = SR.move_reserved("images", root / "pix", root / "pix_reserve", RES)
            self.assertEqual((c["n_moved"], c["n_already"], i["n_moved"]), (3, 0, 3 + 1))        # 3 log dirs; 3 frame dirs + 1 zip (R1A's)
            self.assertTrue((root / "mc_reserve" / R1A / "a1.pkl.xz").is_file() and not (root / "mc" / R1A).exists())
            self.assertTrue((root / "pix_reserve" / f"{R1A}_CAM_F0.zip").is_file() and (root / "pix" / ".done_1").is_file())        # the .done_N markers stay
            moved = audit(root, bank_files=[root / "bank" / "targets_rank0.jsonl"], label_store=None, calib_rows=None)
            self.assertTrue(not moved["leaks"]["metric_cache"] and not moved["leaks"]["images"], moved)
            self.assertEqual(sorted(SR.session_of(p.name) for p in (root / "mc").iterdir()), sorted({S_TR1, S_TR2, S_TR3}))
            again = SR.move_reserved("cache", root / "mc", root / "mc_reserve", RES)            # idempotent: an interrupted chain re-runs safely, and the record still holds all 3
            self.assertEqual((again["n_planned"], again["n_moved"], again["n_already"]), (3, 0, 3))
            back = SR.restore_moved(root / "mc_reserve")
            back2 = SR.restore_moved(root / "pix_reserve")
            self.assertEqual((back["restored"], back2["restored"]), (3, 4))
            self.assertEqual((tree(root / "mc"), tree(root / "pix")), (before[0], before[1]))   # the restore reproduces the tree exactly
            # the reverse of a plant: an item at BOTH places refuses rather than overwriting
            shutil.copytree(root / "mc" / R1A, root / "mc_reserve" / R1A)
            with self.assertRaises(SR.ReserveError):
                SR.move_reserved("cache", root / "mc", root / "mc_reserve", RES)

    def test_a_root_with_an_unresolvable_entry_is_refused_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp)
            touch(root / "mc" / "stray.pkl.xz", b"p")
            snap = tree(root / "mc")
            with self.assertRaises(SR.UnresolvedLog):
                SR.move_reserved("cache", root / "mc", root / "mc_reserve", RES)
            self.assertEqual(tree(root / "mc"), snap)
            self.assertFalse((root / "mc_reserve").exists())
            touch(root / "pix" / "oddly_named.zip", b"z")
            with self.assertRaises(SR.UnresolvedLog):
                SR.move_reserved("images", root / "pix", root / "pix_reserve", RES)
            with self.assertRaises(SR.ReserveError):
                SR.move_reserved("cache", root / "absent", root / "x", RES)
            with self.assertRaises(SR.ReserveError):
                SR.move_reserved("bank", root / "mc", root / "x", RES)

    def test_split_label_store_then_refinalise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_inputs(tmp)
            (root / "store" / "run_meta_0.json").write_text("{}", encoding="utf-8")
            r = SR.split_label_store(root / "store", root / "store_reserve", RES)
            self.assertEqual(r["n_moved"], 3)
            self.assertEqual(sorted(p.name for p in (root / "store_reserve" / "shards").iterdir()), sorted(f"{lg}.aux1" for lg in (R1A, R1B, R2A)))
            self.assertTrue((root / "store_reserve" / "run_meta_0.json").is_file())
            clean = touch(root / "clean_bank.jsonl", json.dumps(row(T1A, "a1")).encode() + b"\n")
            store_audit = lambda s: SR.audit_inputs(RES, bank_files=[clean], label_store=s)
            r = store_audit(root / "store")
            self.assertFalse(r["ok"])                                                           # the training store's manifest still lists the moved shards: caught
            self.assertEqual(sorted(r["leaks"]["label_store"]), sorted(f"manifest:{lg}" for lg in (R1A, R1B, R2A)))
            if optional("aux_label_store.finalize", _has_numpy(), "numpy not importable here"):
                sys.path.insert(0, str(HERE))
                import aux_label_store as AL
                man = AL.finalize(root / "store")                                               # the real finalize rewrites the manifest from the shards that are left
                AL.finalize(root / "store_reserve")
                self.assertEqual(man["counts"]["logs"], 3)
                r = store_audit(root / "store")
                self.assertTrue(r["ok"], r)
                self.assertEqual(r["counts"]["examined"]["label_store"], 3 + 3)                  # 3 shards + 3 manifest entries
                self.assertFalse(store_audit(root / "store_reserve")["ok"])                      # the reserve store IS all reserve
            bad = make_inputs(Path(tmp) / "b")
            write_aux_shard(bad / "store", mk(S_TR2, 8, 80), ("a",), header_log=R1A)            # a shard filed under a training name but headed with a reserved log
            with self.assertRaises(SR.UnresolvedLog):
                SR.split_label_store(bad / "store", bad / "sr", RES)
            self.assertFalse((bad / "sr").exists())


def _has_numpy():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------------------------------ 5. real data: the rank-0 exclusion, measured
class RealDataTests(unittest.TestCase):
    def test_rank0_exclusion_reproduces_10141_from_the_token_table(self):
        if not optional("token-table rank-0 count", TOKEN_TABLE.is_file(), f"{TOKEN_TABLE} not shipped"):
            return
        self.assertEqual(hashlib.sha256(TOKEN_TABLE.read_bytes()).hexdigest(), "c7456c882b3d297c5ba202947d82b850e640a5b08648a3f4dea3c7c3dd176cd1")
        res = SR.load_reserve(FIXTURE)
        c = SR.count_from_token_table(TOKEN_TABLE, res)
        self.assertEqual((c["table_rows"], c["attributed"], c["logless"]), (103288, 103037, 251))
        self.assertEqual((c["n_excluded"], c["n_remaining"], c["sessions_hit"]), (10141, 92896, 16))
        doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(c["per_session"], {x["session"]: x["tokens"] for x in doc["sessions"]})      # every session's count matches the draw's own

    def test_the_filter_on_a_real_shaped_rank0_bank_excludes_exactly_10141(self):
        """The filter itself (not just a table count) over a bank with the 3-C bank's rank-0 shape: one row per attributed token, REAL tokens and logs, token-table cross-checked."""
        if not optional("filter over the real rank-0 shape", TOKEN_TABLE.is_file(), f"{TOKEN_TABLE} not shipped"):
            return
        res = SR.load_reserve(FIXTURE)
        tl = SR.token_log_map(TOKEN_TABLE)
        self.assertEqual(len(tl), 103037)
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"log_name": lg, "token": t, "step": 0, "rank": 0} for t, lg in tl.items()]
            write_jsonl(Path(tmp) / "targets_rank0.jsonl", rows)
            r = SR.filter_bank(Path(tmp) / "targets_rank0.jsonl", Path(tmp) / "out.jsonl", res, token_log=tl)
            self.assertEqual((r["n_in"], r["n_out"], r["n_excluded"], r["n_unresolved"]), (103037, 92896, 10141, 0))
            self.assertEqual((r["n_token_checked"], r["n_token_unmapped"]), (103037, 0))
            a = SR.audit_inputs(res, bank_files=[Path(tmp) / "out.jsonl"])
            self.assertTrue(a["ok"] and a["counts"]["examined"]["bank"] == 92896, a["counts"])
            a = SR.audit_inputs(res, bank_files=[Path(tmp) / "targets_rank0.jsonl"])
            self.assertFalse(a["ok"])
            self.assertEqual(a["counts"]["leak_totals"]["bank"], 10141)
            # a rank-1 twin shaped like the 3-C bank's (same token, parent's log) of a reserved scene is excluded with it; one of a training scene stays
            tok_r = next(t for t, lg in tl.items() if SR.session_of(lg) in res)
            tok_t = next(t for t, lg in tl.items() if SR.session_of(lg) not in res)
            write_jsonl(Path(tmp) / "r1.jsonl", [row(tl[tok_r], tok_r, 1, LR1), row(tl[tok_t], tok_t, 1, LR1)])
            r1 = SR.filter_bank(Path(tmp) / "r1.jsonl", Path(tmp) / "r1.out", res, token_log=tl)
            self.assertEqual((r1["n_in"], r1["n_out"], r1["n_excluded"]), (2, 1, 1))

    def test_pins_agree_with_the_token_table_free_constants(self):
        self.assertEqual(SR.RESERVE_N_SESSIONS, 16)
        self.assertEqual(SR.RESERVE_N_LOGS, 116)
        self.assertEqual(SR.RESERVE_SCENES, 10141)
        self.assertEqual(SR.BANK_RANK0_SCENES, 103037)
        self.assertEqual(SR.RESERVE_FILE_SHA256[:8], "53f3bcd4")


# ------------------------------------------------------------------------------------------------ 5b. the keying claims, pinned against the shipped source
class SourceKeyingTests(unittest.TestCase):
    """The audit is only as good as its map of how each input is keyed.  These patterns are the claims of session_reserve.py's docstring; a builder that re-keys a twin, a cache path, a shard or a
    container makes the corresponding pattern vanish and this test RED, instead of the audit going blind to a new key path."""
    CLAIMS = [
        ("build_teacher_rollouts.py", r'"token": sc\.scenario_name,\s*\n\s*"log_name": sc\.log_name', "a bank row (rank 0 AND every twin built by make_row) carries the scenario's own log_name and token"),
        ("augment_search.py", r"row = BR\.make_row\(sc, planner, controller, 0, cams, cam_ts, 1\)", "a lane/goal twin is made from the SAME scenario object as its parent token"),
        ("slow_twins.py", r'tw = \{k: v for k, v in row\.items\(\) if k != "traj_world"\}', "a slow twin is a copy of its source row (so of its log_name and token)"),
        ("train.py", r'idx\.setdefault\(\(r\.get\("log_name", ""\), r\.get\("token", ""\), int\(r\.get\("step", 0\)\)\)', "the trainer groups a scene and its twins by (log_name, token, step)"),
        ("train.py", r'fn = f"\{log_name\}_\{c\}\.zip"', "the image containers are <log>_<CAM>.zip"),
        ("train.py", r'CT\.vectors\(self\.calib_table, r\.get\("log_name", ""\), cfg\.cameras\)', "calibration is looked up by the row's log_name"),
        ("online_labeller.py", r'return Path\(root\) / log_name / f"\{token\}\{twin_suffix\(aug\)\}\.pkl\.xz"', "the D9 part of a token AND of its twin lives in <root>/<log>/"),
        ("build_metric_cache.py", r'by\.setdefault\(r\["log_name"\], set\(\)\)\.add\(\(r\["token"\]', "the cache builder keys a twin by its row's own log_name"),
        ("aux_label_store.py", r'keys_by_log\[head\["log"\]\] = \[e\["t"\] for e in head\["entries"\]\]', "a label-store shard is one log, its entries are tokens"),
        ("aux_label_store.py", r'rel = "shards/" \+ os\.path\.basename\(p\)', "the store manifest lists shards/<log>.aux1"),
        ("build_targets.py", r'elif f\.endswith\("\.zip"\) and "_CAM_" in f:', "a container is recognised by _CAM_ in its name"),
    ]

    def test_the_keying_claims_hold_in_the_shipped_source(self):
        present = [(f, rx, why) for f, rx, why in self.CLAIMS if (HERE / f).is_file()]
        if not optional("source keying claims", len(present) == len(self.CLAIMS), f"{len(self.CLAIMS) - len(present)} of the source files are not shipped here"):
            return
        for f, rx, why in self.CLAIMS:
            self.assertRegex((HERE / f).read_text(encoding="utf-8"), rx, f"{f}: {why}")

    def test_the_claims_can_go_red(self):
        """A pattern that matches everything proves nothing: each claim must FAIL on the source with its line removed."""
        if not optional("source keying claims (red arm)", all((HERE / f).is_file() for f, _rx, _w in self.CLAIMS), "source files not shipped"):
            return
        for f, rx, why in self.CLAIMS:
            text = (HERE / f).read_text(encoding="utf-8")
            m = re.search(rx, text)
            self.assertIsNotNone(m, f)
            self.assertIsNone(re.search(rx, text.replace(m.group(0), "")), (f, why))


# ------------------------------------------------------------------------------------------------ 6. the CLI of p6_prep (the chain's only interface to the module)
class PrepCliTests(unittest.TestCase):
    def run_prep(self, *argv):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = PP.main(list(argv))
        return rc, buf.getvalue().strip().splitlines()

    def test_every_subcommand_prints_one_marker_and_the_right_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            R = str(FIXTURE)
            rc, out = self.run_prep("reserve-verify", "--reserve", R)
            self.assertEqual((rc, len(out)), (0, 1)); self.assertTrue(out[0].startswith("ZZRESERVE_OK sessions=16 logs=116 sha256=53f3bcd4"), out)
            mut = touch(tmp / "mut.json", FIXTURE.read_bytes().replace(b"veh-35", b"veh-36", 1))
            rc, out = self.run_prep("reserve-verify", "--reserve", str(mut))
            self.assertEqual(rc, 1); self.assertTrue(out[0].startswith("ZZRESERVE_BAD ReserveError:"), out)
            real_r, real_t = sorted(SR.reserve_logs(FIXTURE))[0], "2021.05.12.19.36.12_veh-35_00005_00204"
            write_bank(tmp / "bank", [row(real_r, "a"), row(real_t, "b")], [row(real_r, "a", 1, GH), row(real_t, "b", 1, LR1)])
            rc, out = self.run_prep("reserve-filter", "--reserve", R, "--bank", str(tmp / "bank"), "--out", str(tmp / "fb"))
            self.assertEqual(rc, 0); self.assertRegex(out[0], r"^ZZRESERVE_FILTER_OK rows_in=4 rows_out=2 excluded=2 unresolved=0 targets_sha256=[0-9a-f]{16}$")
            tt = tmp / "tokens.tsv.gz"                                                              # the token table's two columns the cross-check reads (token, log)
            with gzip.open(tt, "wt", encoding="utf-8", newline="") as f:
                f.write(f"token\tlog\na\t{real_r}\nb\t{real_t}\nnolog\t\n")
            rc, out = self.run_prep("reserve-filter", "--reserve", R, "--bank", str(tmp / "bank"), "--out", str(tmp / "fbt"), "--token-table", str(tt))
            self.assertEqual(rc, 0); self.assertRegex(out[0], r"^ZZRESERVE_FILTER_OK rows_in=4 rows_out=2 excluded=2 unresolved=0 ")
            self.assertEqual(json.loads((tmp / "fbt" / "reserve_filter_report.json").read_text())["token_log_checked"], True)
            with gzip.open(tt, "wt", encoding="utf-8", newline="") as f:                           # the table puts token b in the RESERVED log: the row that says otherwise is refused
                f.write(f"token\tlog\na\t{real_r}\nb\t{real_r}\n")
            rc, out = self.run_prep("reserve-filter", "--reserve", R, "--bank", str(tmp / "bank"), "--out", str(tmp / "fbt2"), "--token-table", str(tt))
            self.assertEqual(rc, 1); self.assertTrue(out[0].startswith("ZZRESERVE_FILTER_BAD AttributionConflict:"), out)
            write_jsonl(tmp / "bank2" / "targets_rank0.jsonl", [row(real_t, "b"), {"token": "x"}])
            rc, out = self.run_prep("reserve-filter", "--reserve", R, "--bank", str(tmp / "bank2"), "--out", str(tmp / "fb2"))
            self.assertEqual(rc, 1); self.assertTrue(out[0].startswith("ZZRESERVE_FILTER_BAD UnresolvedLog:"), out)
            self.assertFalse((tmp / "fb2" / "targets_rank0.jsonl").exists())
            (tmp / "calib.json").write_text(json.dumps({"fields": [], "logs": {real_r: {"c": [1]}, real_t: {"c": [1]}}}), encoding="utf-8")
            rc, out = self.run_prep("reserve-calib", "--reserve", R, "--calib", str(tmp / "calib.json"), "--out", str(tmp / "calib_p7.json"))
            self.assertEqual((rc, out[0]), (0, "ZZRESERVE_CALIB_OK logs_in=2 logs_out=1 excluded=1"))
            touch(tmp / "mc" / real_r / "a.pkl.xz", b"p"); touch(tmp / "mc" / real_t / "b.pkl.xz", b"p")
            rc, out = self.run_prep("reserve-move", "--reserve", R, "--kind", "cache", "--root", str(tmp / "mc"), "--to", str(tmp / "mcr"), "--dry-run")
            self.assertEqual((rc, out[0]), (0, "ZZRESERVE_MOVE_OK kind=cache moved=0 already=0 planned=1 dry_run=1"))
            rc, out = self.run_prep("reserve-move", "--reserve", R, "--kind", "cache", "--root", str(tmp / "mc"), "--to", str(tmp / "mcr"))
            self.assertEqual((rc, out[0]), (0, "ZZRESERVE_MOVE_OK kind=cache moved=1 already=0 planned=1"))
            rc, out = self.run_prep("reserve-move", "--to", str(tmp / "mcr"), "--restore")
            self.assertEqual((rc, out[0]), (0, "ZZRESERVE_RESTORE_OK restored=1"))
            rc, out = self.run_prep("reserve-move", "--reserve", R, "--kind", "cache", "--root", str(tmp / "mc"), "--to", str(tmp / "mcr"))
            rc, out = self.run_prep("reserve-audit", "--reserve", R, "--bank", str(tmp / "fb"), "--cache", str(tmp / "mc"), "--calib", str(tmp / "calib_p7.json"), "--out", str(tmp / "audit.json"))
            self.assertEqual(rc, 0); self.assertTrue(out[0].startswith("ZZRESERVE_AUDIT_OK inputs=bank,calibration,metric_cache"), out)
            self.assertTrue(json.loads((tmp / "audit.json").read_text())["ok"])
            rc, out = self.run_prep("reserve-audit", "--reserve", R, "--bank", str(tmp / "bank"), "--calib", str(tmp / "calib.json"))
            self.assertEqual(rc, 1); self.assertEqual(out[0], "ZZRESERVE_AUDIT_LEAK leaks=bank:2,calibration:1 unresolved= empty=")
            rc, out = self.run_prep("reserve-audit", "--reserve", R, "--bank", str(tmp / "nothing-here"))
            self.assertEqual(rc, 1); self.assertTrue(out[0].startswith("ZZRESERVE_AUDIT_BAD "), out)

    def test_existing_subcommands_are_unchanged(self):
        """The P6 path: the original docstring subcommands still parse and run (the full old behaviour is the tip's own test_p6_chain, run in this tree)."""
        with tempfile.TemporaryDirectory() as tmp:
            write_bank(Path(tmp) / "b", [row("L0", "t0"), row("L1", "t1")], [row("L0", "t0", 1, {"kind": "lane_rank", "value": 2})])
            rc, out = self.run_prep("lane-ranks", "--bank", str(Path(tmp) / "b"))
            self.assertEqual((rc, out), (0, ["ZZLANE_RANKS 2"]))             # `L0` / `L1` are not nuPlan names: the P6 subcommands never go through session_of, so the tip's tests keep working
        import ast
        src = Path(PP.__file__).read_text(encoding="utf-8")
        tree_ = ast.parse(src)
        top_imports = [n for n in tree_.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse([n for n in top_imports if "session_reserve" in ast.dump(n)], "p6_prep must import session_reserve lazily: the P6 path never loads it")


# ------------------------------------------------------------------------------------------------ 7. the REAL p6_pod.sh in the sandbox
def _sandbox_tools():
    try:
        import test_p6_chain as TC
        import shutil as _sh
        if _sh.which("bash") is None:
            return None
        return TC
    except Exception:                                                              # numpy / bash absent: reported by optional()
        return None


def p7_static(text):
    """Static properties of section 5a / 7a of p6_pod.sh (violation names; [] = fine)."""
    v = []
    code = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("#"))
    m = re.search(r'^if \[ -n "\$\{P7_RESERVE:-\}" \]; then\n(.*?)^fi\n', code, re.S | re.M)
    if not m:
        return ["P7_BLOCK_MISSING"]
    block = m.group(1)
    outside = code.replace(m.group(0), "")
    outside = re.sub(r'^if \[ -n "\$\{P7_RESERVE:-\}" \]; then p7_audit; log [^\n]*$', "", outside, flags=re.M)
    if re.search(r"P7_|p7_|reserve-", outside):
        v.append("P7_CODE_OUTSIDE_THE_GUARD")                                      # dormant means: unreachable unless P7_RESERVE is set
    i_guard = block.find("\n  resume_bank_check\n")
    i_filter, i_move = block.find("p6_prep.py reserve-filter"), block.find("p6_prep.py reserve-move")
    i_calib = block.find("p6_prep.py reserve-calib")
    if i_guard < 0:
        v.append("RESUME_GUARD_MISSING")
    elif not (0 <= i_guard < i_filter < i_move) or not (i_guard < i_calib):
        v.append("GUARD_NOT_BEFORE_ANY_MOVE")
    if not (0 <= i_filter < i_move):
        v.append("FILTER_NOT_BEFORE_MOVE")                                         # a refusing filter must stop the chain before anything moves
    if 'CBANK="${P7_SRC}_p7"' not in block:
        v.append("CBANK_NOT_REBOUND")
    if block.count("p7_audit") < 2 or "reserve-audit" not in block:
        v.append("AUDIT_MISSING")
    i_final = code.find('if [ -n "${P7_RESERVE:-}" ]; then p7_audit; log "P7 reserve audit (final)')
    if not (0 < i_final < code.find("\nARGS=") and i_final > code.find("stop_after lanes")):
        v.append("FINAL_AUDIT_NOT_BEFORE_ARGS")
    for c in re.findall(r'^\s*(?:nice -n [^ ]+ )?"\$PY" (?:p6_prep|aux_label_store)\.py [^\n]*$', block.replace("\\\n", " "), re.M):
        if "200>&-" not in c:
            v.append("FD200:" + c.strip()[:50])
    if "invalidate cache lanes lab mem egr launch_gate" not in block:
        v.append("DOWNSTREAM_NOT_INVALIDATED")
    if 'key_mark reserve "$P7_ID"' not in block or "key_done reserve" not in block:
        v.append("STAGE_NOT_MARKED")
    if "stop_after reserve" not in block:
        v.append("NO_STOP_AFTER")
    if 'lock_held "$DATA/refe_p6"' not in block:
        v.append("P6_LOCK_GUARD_MISSING")
    return v


class ChainStaticTests(unittest.TestCase):
    CHAIN = HERE / "p6_pod.sh"

    def test_static_properties_and_their_mutations(self):
        text = self.CHAIN.read_text(encoding="utf-8")
        self.assertEqual(p7_static(text), [])
        d = lambda t, needle: "\n".join(ln for ln in t.split("\n") if needle not in ln)
        muts = [("delete the resume-bank guard before the move", text.replace("\n  resume_bank_check\n", "\n"), "RESUME_GUARD_MISSING"),
                ("move before the filter", text.replace("p6_prep.py reserve-filter", "p6_prep.py reserve-moveX", 1), "FILTER_NOT_BEFORE_MOVE"),
                ("delete the final audit", d(text, 'p7_audit; log "P7 reserve audit (final)'), "FINAL_AUDIT_NOT_BEFORE_ARGS"),
                ("a reserve call outside the guard", text.replace('log "lanes ok"; stop_after lanes\n', 'log "lanes ok"; stop_after lanes\np6_prep_x reserve-audit\n'), "P7_CODE_OUTSIDE_THE_GUARD"),
                ("an audit child that keeps fd 200", text.replace("--out \"$P7D/audit.json\" > \"$P7D/audit.out\" 2>&1 200>&-", "--out \"$P7D/audit.json\" > \"$P7D/audit.out\" 2>&1"), "FD200"),
                ("nothing invalidated after the bank changed", text.replace("    invalidate cache lanes lab mem egr launch_gate\n    key_mark reserve", "    key_mark reserve"), "DOWNSTREAM_NOT_INVALIDATED"),
                ("the P6 lock guard deleted", d(text, 'lock_held "$DATA/refe_p6"'), "P6_LOCK_GUARD_MISSING")]
        for name, mt, want in muts:
            self.assertNotEqual(mt, text, name)
            got = p7_static(mt)
            self.assertTrue(any(g.startswith(want) for g in got), (name, want, got))

    def test_the_tips_own_static_checks_and_mutations_still_hold(self):
        TC = _sandbox_tools()
        if not optional("test_p6_chain static checks", TC is not None, "numpy / bash not available"):
            return
        text = self.CHAIN.read_text(encoding="utf-8")
        self.assertEqual(TC.check_chain(text), [])
        for name, mt, want in TC.mutations(text):
            self.assertTrue(any(g.startswith(want) for g in TC.check_chain(mt)), (name, want))
        self.assertIn("P7_", TC.SANDBOX_DROP_PREFIXES)                             # a P7 chain's G-SUITE must not leak P7_RESERVE into the sandboxed default-path chain

    def test_the_arguments_line_is_untouched(self):
        text = self.CHAIN.read_text(encoding="utf-8")
        am = re.findall(r'^ARGS="(.*)"$', text, re.M)
        self.assertEqual(len(am), 1)
        for needle in ("--targets $CBANK ", "--images $PIX ", "--calib $BANK/calib_table.json ", "--metric-cache $MC "):
            self.assertIn(needle, am[0])
        self.assertNotIn("P7", am[0])


def _norm_calls(sb):
    root = str(sb.root).replace("\\", "/")
    return [ln.replace(root, "<ROOT>") for ln in sb.calls_list()]


def _norm_log(sb):
    root = str(sb.root).replace("\\", "/")
    out = []
    for ln in sb.chain_log().splitlines():
        if re.match(r"^\d+\+\d+ records (in|out)$", ln) or "bytes copied" in ln:             # `dd`'s own timing line (the stage-0 write test)
            continue
        ln = re.sub(r"^\S+ ", "", ln)
        ln = re.sub(r"md5 [0-9a-f]+", "md5 <MD5>", ln).replace(root, "<ROOT>")
        out.append(ln)
    return out


def _data_snapshot(sb):
    """(relative path -> sha256) of the deterministic data the chain leaves: banks, caches, tables (not logs / stage markers, which carry timestamps)."""
    out = {}
    for sub in ("refe_navtrain", "navtrain_pixels"):
        for p in sorted((sb.data / sub).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(sb.data)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class ChainBehaviourTests(unittest.TestCase):
    """The REAL p6_pod.sh under bash with stubbed tools (the harness of test_p6_chain.py)."""
    ENV = {"P6_NAVSIM_LOGS_URL": "", "P6_YAW_WRAP": "off"}

    def test_default_path_is_identical_to_the_tips_chain_when_p7_reserve_is_unset(self):
        TC = _sandbox_tools()
        if not optional("sandbox chain (P6 default path)", TC is not None, "numpy / bash not available"):
            return
        new_text = (HERE / "p6_pod.sh").read_text(encoding="utf-8")
        results = []
        base_dir = os.environ.get("P7WIRE_BASE_DIR")                               # the tip's refe/ (pristine); absent on the pod
        scripts = [("new", new_text)]
        if optional("tip p6_pod.sh comparison", bool(base_dir) and (Path(base_dir) / "p6_pod.sh").is_file(), "P7WIRE_BASE_DIR not set: only the 'no reserve call' half runs"):
            scripts.append(("tip", (Path(base_dir) / "p6_pod.sh").read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in scripts:
                root = Path(tmp) / "sb"                                            # the SAME path for every script: the ARGS sha and the logged paths must be comparable
                if root.exists():
                    shutil.rmtree(root)
                root.mkdir()
                sb = TC.Sandbox(root, text)
                rc, out = sb.run(dict(self.ENV))
                self.assertTrue(rc == 0 and "ZZP6_DONE" in sb.chain_log(), (name, rc, out[-400:], sb.chain_log()[-400:]))
                results.append((name, _norm_calls(sb), _norm_log(sb), _data_snapshot(sb), sorted(p.name for p in (sb.data / "refe_p6").iterdir())))
        name, calls, log, data, files = results[0]
        self.assertFalse([c for c in calls if "reserve" in c.lower() or "P7" in c], "the default path called a reserve subcommand")
        self.assertFalse([ln for ln in log if "P7" in ln], "the default path logged a P7 line")
        if len(results) == 2:
            (_n1, c1, l1, d1, f1), (_n2, c2, l2, d2, f2) = results
            self.assertEqual(c1, c2, "the call log differs from the tip's chain")
            self.assertEqual(l1, l2, "the chain log differs from the tip's chain")
            self.assertEqual(d1, d2, "the data the chain leaves differs from the tip's chain")
            self.assertEqual(f1, f2)
            self.assertGreater(len(c1), 20)

    # ---- the P7 path
    def _prepare(self, tmp, script_text=None, bank_extra=None, label_store=False, ship=True):
        TC = _sandbox_tools()
        (Path(tmp) / "sb").mkdir()
        sb = TC.Sandbox(Path(tmp) / "sb", script_text)
        if ship:                                                                       # the shipped tree must carry the module (the chain refuses without it: test (e) below)
            shutil.copy2(HERE / "session_reserve.py", sb.pkg / "refe" / "session_reserve.py")
        d = sb.data / "refe_navtrain"
        res = sorted(SR.reserve_logs(FIXTURE))
        self.r_logs = [res[0], res[-1]]                                              # two reserved logs (different sessions)
        self.t_logs = ["2021.05.12.19.36.12_veh-35_00005_00204", "2021.06.09.14.50.36_veh-26_00832_00905", "2021.08.11.12.00.00_veh-38_00100_00200", "2021.07.21.10.10.10_veh-12_00001_00002"]
        for lg in self.t_logs:
            assert SR.session_of(lg) not in SR.load_reserve(FIXTURE), lg
        logs = self.r_logs + self.t_logs
        r0 = [row(lg, t, command=1) for lg in logs for t in ("aa", "bb")]
        r1 = [row(self.r_logs[0], "aa", 1, GH), row(self.r_logs[0], "aa", 1, LR1), row(self.t_logs[0], "aa", 1, GH), row(self.t_logs[0], "aa", 1, LR1), row(self.t_logs[2], "bb", 1, LR1)]
        r1 += list(bank_extra or ())
        self.cbank = d / "train_grow_cmd_3c"
        write_bank(self.cbank, r0, r1)
        self.n_r0, self.n_r1 = len(r0), len(r1)
        (d / "train_grow" / "calib_table.json").write_text(json.dumps({"fields": ["fx"], "n_logs": 6, "logs": {lg: {"CAM_F0": [1.0]} for lg in logs}}), encoding="utf-8")
        TC.make_parts_tree(d / "metric_cache", [(lg, t) for lg in logs for t in ("aa", "bb")], twin_k=1)
        for lg in logs:
            touch(sb.data / "navtrain_pixels" / "navtrain_current_1" / lg / "CAM_F0" / "h.jpg", b"j")
        touch(sb.data / "navtrain_pixels" / f"{logs[0]}_CAM_F0.zip", b"z")
        touch(sb.data / "navtrain_pixels" / f"{logs[2]}_CAM_F0.zip", b"z")
        (sb.data / "refe_p6" / "stage").mkdir(parents=True, exist_ok=True)
        (sb.data / "refe_p6" / "stage" / "commands.ok").write_text("x")               # the 3-C flow (P6_CBANK set): the commands stage is taken as done, exactly as on the pod
        self.store = None
        if label_store:
            self.store = sb.data / "aux_store"
            for lg in logs:
                write_aux_shard(self.store, lg, ("aa", "bb"))
            write_manifest(self.store, logs)
            fp = (sb.stub / "fakepy.py")
            fp.write_text(fp.read_text(encoding="utf-8").replace('("p6_prep.py", "annotate_commands.py", "p6_overlap.py")', '("p6_prep.py", "annotate_commands.py", "p6_overlap.py", "aux_label_store.py")'), encoding="utf-8")
        return sb

    def _env(self, sb, **over):
        e = dict(self.ENV)
        e.update({"P6_CBANK": str(self.cbank).replace("\\", "/"), "P7_RESERVE": str(FIXTURE).replace("\\", "/")})
        if self.store is not None:
            e["P7_LABEL_STORE"] = str(self.store).replace("\\", "/")
        e.update(over)
        return e

    def _untouched(self, sb):
        d = sb.data
        r = SR.load_reserve(FIXTURE)
        self.assertEqual(len([p for p in (d / "refe_navtrain" / "metric_cache").iterdir() if SR.session_of(p.name) in r]), 2, "the metric cache was touched")
        self.assertTrue((d / "navtrain_pixels" / "navtrain_current_1" / self.r_logs[0]).is_dir() and not (d / "navtrain_pixels_reserve").exists(), "the image root was touched")
        self.assertFalse((d / "refe_navtrain" / "metric_cache_reserve").exists())
        self.assertFalse((d / "refe_navtrain" / "train_grow_cmd_3c_p7").exists(), "a filtered bank appeared")

    def test_p7_reserve_leaves_every_input_and_the_arguments_name_the_copies(self):
        TC = _sandbox_tools()
        if not optional("sandbox chain (P7 reserve stage)", TC is not None and _has_numpy(), "numpy / bash not available"):
            return
        with tempfile.TemporaryDirectory() as tmp:
            sb = self._prepare(tmp, label_store=True)
            src_sha = {p.name: sha(p) for p in self.cbank.glob("targets_rank*.jsonl")}
            src_calib = sha(sb.data / "refe_navtrain" / "train_grow" / "calib_table.json")
            rc, out = sb.run(self._env(sb))
            log = sb.chain_log()
            self.assertTrue(rc == 0 and "ZZP6_DONE" in log, (rc, out[-500:], log[-800:]))
            res = SR.load_reserve(FIXTURE)
            d = sb.data
            p7 = d / "refe_navtrain" / "train_grow_cmd_3c_p7"
            rows = [json.loads(x) for fp in sorted(p7.glob("targets_rank*.jsonl")) for x in fp.read_text().splitlines()]
            self.assertEqual(len(rows), (self.n_r0 + self.n_r1) - 2 * 2 - 2, "rank-0: 2 reserved logs x 2 tokens; rank-1: the 2 twins of the reserved log")
            self.assertEqual(sum(1 for r in rows if r["rank"] == 1), self.n_r1 - 2)
            self.assertFalse([r for r in rows if SR.session_of(r["log_name"]) in res])
            rep = json.loads((p7 / "reserve_filter_report.json").read_text())
            self.assertEqual((rep["n_excluded"], rep["by_rank_excluded"]), (6, {"0": 4, "1": 2}))
            self.assertEqual({p.name: sha(p) for p in self.cbank.glob("targets_rank*.jsonl")}, src_sha, "the source bank must be untouched")
            self.assertEqual(sha(d / "refe_navtrain" / "train_grow" / "calib_table.json"), src_calib, "the source calibration table must be untouched")
            calib = json.loads((d / "refe_p6" / "p7_reserve" / "calib" / "calib_table.json").read_text())
            self.assertEqual((len(calib["logs"]), calib["n_logs"]), (4, 4))
            self.assertEqual(sorted(p.name for p in (d / "refe_navtrain" / "metric_cache").iterdir()), sorted(self.t_logs))
            self.assertEqual(sorted(p.name for p in (d / "refe_navtrain" / "metric_cache_reserve").iterdir() if p.is_dir()), sorted(self.r_logs))
            self.assertFalse([p for p in (d / "navtrain_pixels").rglob("*") if any(lg in str(p) for lg in self.r_logs)])
            self.assertTrue((d / "navtrain_pixels_reserve" / "navtrain_current_1" / self.r_logs[0]).is_dir())
            self.assertEqual(sorted(p.stem for p in (self.store / "shards").glob("*.aux1")), sorted(self.t_logs))
            self.assertEqual(sorted(p.stem for p in (Path(str(self.store) + "_reserve") / "shards").glob("*.aux1")), sorted(self.r_logs))
            man = json.loads((self.store / "manifest.json").read_text())
            self.assertEqual(sorted(k.split("/")[-1][:-5] for k in man["files"]), sorted(self.t_logs))     # the REAL finalize ran on the training store
            audit_json = json.loads((d / "refe_p6" / "p7_reserve" / "audit.json").read_text())
            self.assertTrue(audit_json["ok"], audit_json)
            self.assertEqual(audit_json["counts"]["inputs_audited"], ["bank", "calibration", "images", "label_store", "metric_cache"])
            c = sb.calls_list()
            self.assertTrue(sb.stage("reserve"))
            builds = [x for x in c if x.startswith("build_metric_cache.py build")]
            self.assertTrue(builds and all(f"--targets {str(p7).replace(chr(92), '/')}" in x for x in builds), builds)      # the cache stage builds from the FILTERED bank
            full = next(x for x in c if x.startswith("train.py") and "--steps" not in x and "--preflight" not in x)
            self.assertIn(f"--targets {str(p7).replace(chr(92), '/')} ", full + " ")
            self.assertIn(f"--calib {str(d).replace(chr(92), '/')}/refe_p6/p7_reserve/calib/calib_table.json ", full + " ")
            gate = next(x for x in c if x.startswith("launch_gate.py run"))
            self.assertIn("train_grow_cmd_3c_p7", gate)                                            # the gate is minted over the same argv
            self.assertEqual(len([x for x in c if x.startswith("p6_prep.py reserve-audit")]), 2)    # 5a and 7a
            self.assertEqual(len([x for x in c if x.startswith("p6_prep.py reserve-filter")]), 1)
            # a second start: nothing is filtered or moved again, the audit still runs, the chain still ends DONE
            n_before = len(c)
            rc, out = sb.run(self._env(sb))
            new = sb.calls_list()[n_before:]
            self.assertEqual(rc, 0, out[-400:])
            self.assertFalse([x for x in new if x.startswith(("p6_prep.py reserve-filter", "p6_prep.py reserve-move", "p6_prep.py reserve-calib", "p6_prep.py reserve-labelstore", "aux_label_store.py"))], new)
            self.assertTrue([x for x in new if x.startswith("p6_prep.py reserve-audit")])
            # a RESUME of the P7 run: the run recorded the filtered bank; P6_CBANK is then that path (verified, not re-filtered)
            (d / "refe_runs" / "p6_paper_faithful").mkdir(parents=True, exist_ok=True)
            (d / "refe_runs" / "p6_paper_faithful" / "config.json").write_text(TC.rb_config(str(p7).replace("\\", "/")))
            n_before = len(sb.calls_list())
            rc, out = sb.run(self._env(sb, P6_CBANK=str(p7).replace("\\", "/")))
            new = sb.calls_list()[n_before:]
            self.assertEqual(rc, 0, (out[-400:], sb.chain_log()[-500:]))
            self.assertFalse([x for x in new if x.startswith(("p6_prep.py reserve-filter", "p6_prep.py reserve-move"))], new)
            self.assertIn("already names a filtered bank", sb.chain_log())

    def test_a_leak_planted_after_the_fact_is_refused_before_ARGS(self):
        TC = _sandbox_tools()
        if not optional("sandbox chain (P7 planted leaks)", TC is not None, "numpy / bash not available"):
            return
        with tempfile.TemporaryDirectory() as tmp:
            sb = self._prepare(tmp)
            rc, out = sb.run(self._env(sb, P6_STOP_AFTER="reserve"))
            self.assertEqual(rc, 0, (out[-400:], sb.chain_log()[-600:]))
            p7 = sb.data / "refe_navtrain" / "train_grow_cmd_3c_p7"
            # (a) a reserved ROW planted into the filtered bank (a rank-1 twin)
            with open(p7 / "targets_rank1.jsonl", "ab") as f:
                f.write(json.dumps(row(self.r_logs[0], "aa", 1, LR1)).encode() + b"\n")
            mark = len(sb.chain_log())
            rc, out = sb.run(self._env(sb))
            self.assertNotEqual(rc, 0)
            self.assertIn("REFUSED: P7 reserve audit FAILED", sb.chain_log()[mark:])
            self.assertEqual(len([x for x in sb.calls_list() if x.startswith("train.py")]), 0, "the trainer must not start")
            # (b) a reserved D9 part appearing later (a cache stage that rebuilt it): the FINAL audit (7a) catches it even though 5a was clean
            write_jsonl(p7 / "targets_rank1.jsonl", [r for r in map(json.loads, (p7 / "targets_rank1.jsonl").read_text().splitlines()) if r["log_name"] not in self.r_logs])
            rc, out = sb.run(self._env(sb, P6_STOP_AFTER="reserve"))
            self.assertEqual(rc, 0, (out[-400:], sb.chain_log()[-500:]))
            stub = sb.stub / "fakepy.py"
            stub.write_text(stub.read_text(encoding="utf-8").replace('put(out / r["log_name"] / (r["token"] + ".pkl.xz")', 'put(out / os.environ.get("STUB_LEAK_LOG", r["log_name"]) / (r["token"] + ".pkl.xz")'), encoding="utf-8")
            mark = len(sb.chain_log())
            rc, out = sb.run(self._env(sb, STUB_LEAK_LOG=self.r_logs[1]))
            self.assertNotEqual(rc, 0)
            self.assertIn("REFUSED: P7 reserve audit FAILED", sb.chain_log()[mark:])
            self.assertTrue((sb.data / "refe_navtrain" / "metric_cache" / self.r_logs[1]).is_dir())           # the planted part IS the cause
            self.assertEqual(len([x for x in sb.calls_list() if x.startswith("train.py")]), 0)

    def test_refusals_leave_every_input_untouched(self):
        TC = _sandbox_tools()
        if not optional("sandbox chain (P7 refusals)", TC is not None, "numpy / bash not available"):
            return
        res_raw = FIXTURE.read_bytes()
        # (a) a logless rank-1 row in the bank: the FILTER refuses, before anything moves
        with tempfile.TemporaryDirectory() as tmp:
            sb = self._prepare(tmp, bank_extra=[{"token": "zz", "step": 0, "rank": 1, "aug": GH}])
            rc, out = sb.run(self._env(sb))
            self.assertNotEqual(rc, 0)
            self.assertIn("REFUSED: P7 reserve: the bank filter refused", sb.chain_log())
            self._untouched(sb)
            self.assertFalse(sb.stage("reserve"))
        # (b) a reserve file with one byte changed: refused at the verify, before anything is created
        with tempfile.TemporaryDirectory() as tmp:
            sb = self._prepare(tmp)
            b = bytearray(res_raw); b[100] ^= 1
            mut = touch(Path(tmp) / "reserve_mut.json", bytes(b))
            rc, out = sb.run(self._env(sb, P7_RESERVE=str(mut).replace("\\", "/")))
            self.assertNotEqual(rc, 0)
            self.assertIn("REFUSED: P7 reserve file refused", sb.chain_log())
            self._untouched(sb)
        # (c) a run directory that was STARTED on the P6 bank (a P6 run): refused by the resume-bank guard before a single file moves
        with tempfile.TemporaryDirectory() as tmp:
            sb = self._prepare(tmp)
            (sb.data / "refe_runs" / "p6_paper_faithful").mkdir(parents=True, exist_ok=True)
            (sb.data / "refe_runs" / "p6_paper_faithful" / "config.json").write_text(TC.rb_config(str(self.cbank).replace("\\", "/")))
            rc, out = sb.run(self._env(sb))
            self.assertNotEqual(rc, 0)
            self.assertIn("ZZP6_RESUME_BANK_MISMATCH", sb.chain_log() + out)
            self._untouched(sb)
            # ... and the same run directory WITHOUT P7_RESERVE is the P6 resume it always was (the control: the guard is the only difference)
            env = self._env(sb)
            del env["P7_RESERVE"]
            rc, out = sb.run({**env, "P6_STOP_AFTER": "recover"})
            self.assertEqual(rc, 0, (out[-300:], sb.chain_log()[-300:]))
        # (e) the shipped tree does not carry session_reserve.py: refused before anything is read
        with tempfile.TemporaryDirectory() as tmp:
            sb = self._prepare(tmp, ship=False)
            rc, out = sb.run(self._env(sb))
            self.assertNotEqual(rc, 0)
            self.assertIn("session_reserve.py is not shipped", sb.chain_log())
            self._untouched(sb)
        # (d) a P7 chain (its OWN $P6 dir) while the P6 chain holds its lock: refused, nothing moved
        with tempfile.TemporaryDirectory() as tmp:
            text = (HERE / "p6_pod.sh").read_text(encoding="utf-8")
            p7_text = text.replace("P6=$DATA/refe_p6;", "P6=$DATA/refe_p7;", 1)
            self.assertNotEqual(p7_text, text)
            sb = self._prepare(tmp, script_text=p7_text)
            (sb.data / "refe_p7" / "stage").mkdir(parents=True, exist_ok=True)
            (sb.data / "refe_p7" / "stage" / "commands.ok").write_text("x")
            (sb.data / "refe_p6").mkdir(parents=True, exist_ok=True)
            (sb.data / "refe_p6" / "chain.lock").write_text("")
            rc, out = sb.run(self._env(sb, STUB_HELD="refe_p6"))
            self.assertNotEqual(rc, 0)
            log7 = (sb.data / "refe_p7" / "chain.log").read_text(encoding="utf-8")
            self.assertIn("the P6 chain at", log7)
            self._untouched(sb)
            # control: the same copy with the lock FREE runs the stage
            rc, out = sb.run(self._env(sb, P6_STOP_AFTER="reserve"))
            self.assertEqual(rc, 0, (out[-300:], (sb.data / "refe_p7" / "chain.log").read_text()[-400:]))
            self.assertTrue((sb.data / "refe_navtrain" / "train_grow_cmd_3c_p7").is_dir())


# ------------------------------------------------------------------------------------------------ runner
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (UnitTests, ReserveFileTests, FilterTests, AuditTests, MoveTests, RealDataTests, SourceKeyingTests, PrepCliTests, ChainStaticTests, ChainBehaviourTests):
        for name in loader.getTestCaseNames(cls):
            if only is None or only in f"{cls.__name__}.{name}":
                suite.addTest(cls(name))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    n = res.testsRun
    if res.wasSuccessful():
        print(f"ZZSESSRES_PASS {n}  (optional checks run: {sorted(set(OPTIONAL['ran']))}; ABSENT: {sorted(set(OPTIONAL['absent']))})")
        return 0
    print(f"ZZSESSRES_FAIL {len(res.failures) + len(res.errors)}/{n}: " + ", ".join(t.id().rsplit(".", 2)[-2] + "." + t.id().rsplit(".", 1)[-1] for t, _ in res.failures + res.errors))
    return 1


if __name__ == "__main__":
    sys.exit(main())
