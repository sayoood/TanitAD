#!/usr/bin/env python3
"""P5-C Amendment A3 label path: C1's labels are computed from the RJ5 teacher trajectory of each P7 bank row, and keyed EXACTLY as the trainer looks
labels up today (REFe agent, 2026-11-15; PI 2026-11-14, GAC D-REFE-PI-2026-11-15-1).

WHY THIS SHAPE (the smallest change).  The trainer's one reader (`refe/traj_label_table.py`, A2 pin d65e01c4) keys a label by
"log|token|rank|sha256(the BANK row's own traj)".  The pinned dual labeller (`a1/p5c_dual_labels.py`, A1 pin 50b9f0db) computes Tc20s -> P5-R(lam*) from
whatever `traj` its input bank rows carry, and writes the sha of THAT traj.  So nothing pinned is edited:
  feed     real bank + RJ5 join  ->  a FEED bank (same rows, `traj` := the row's RJ5 traj where one exists, else the bank traj) + a per-row map
  (the pinned labeller runs on the feed bank, unchanged: P5C_BANKDIR=<feed dir>)
  rekey    labeller records (keyed by the FED traj's sha)  ->  records keyed by the BANK traj's sha, each carrying its provenance; the pinned gates
           (`a1/p5c_table_gates.py`) and the pinned table builder then run on the REAL bank, unchanged
  provenance  the per-row sidecar (label source, RJ5 traj sha, roll platform, R0 control status) + the gates' meta with an "a3" block, for
           `traj_label_table.py build --meta-json` (so the provenance is sha-bound into the table)
  repair-combine  the A3 RK0 input: raw sim8 of the BANK traj (C0's target) against the RJ5-based label's sim8, on the SAME fixed-rule rows
  fallback A3.3 (PI 2026-11-16): after rekey, the ONE ruled row whose P5-R label fails S2 takes its own Tc20s label (record field p5r_fallback = tb5_s2_tc20s; the P5-R
           label is kept as p5r_discarded).  Allow-list only, at most MAX_LABEL_FALLBACK_ROWS rows; any other row failing S2 is left alone (and TB-5 then FAILS it)
  tb0r     A3's cross-environment check on RJ5 inputs: the pod's RJ5-based labels against the dev box's (same pinned labeller, 16,133 local rows),
           with the A1 tolerances (R-A 1e-9 on Tc20s; R-B R-TOL-1/2/3 on P5-R), by calling the pinned gate functions

Per-row provenance (feed map, rekeyed records, sidecar):
  label_source     "rj5" | "bank_fallback"            (the brief's two values; nothing else is ever written)
  fallback_reason  null (rj5) | "missing" (rank-0 key with no RJ5 roll in the join) | "rank1" (rank-1 rows: no RJ5 roll exists for them)
  label_traj_sha256  sha of the traj the label is computed from (the RJ5 traj, or the bank traj for a fallback)
  rj5_source / roll_platform  the join's source label and its DECLARED platform ("devbox" | "pod"); null for a fallback
  r0_log_status    "exact" | "diverged" | "uncontrolled" (TB-8, disclosed, never gated) and r0_row_dxy_m / r0_log_max_dxy_m

Fail-closed.  A refusal prints "ZZRJ5_REFUSE <reason>" and exits 2 (integrity) or 3 (coverage below the committed bar).  Never silently filled:
  * a join row whose bank key is not the bank row's own key (log|token|0|sha of the bank traj) -> REFUSED (wrong-key override)
  * a join row for a (log, token) the bank does not have, a rank != 0 join row, a duplicate, a "conflict", an RJ5 traj whose sha is not the recorded one,
    a bad shape / non-finite value, or a yaw gap >= pi against the bank traj -> REFUSED
  * a bank rank-0 key ABSENT from the join (a join about another bank) -> REFUSED; only an explicit status "missing" gets the declared fallback
  * a source without a declared platform -> REFUSED
  * rekey: a record whose traj sha is not the FED traj's (e.g. the labeller saw the bank traj), a record for a key not in the map, a duplicate,
    a torn line, or a map row with NO record -> REFUSED

Stdlib only, except `tb0r` (imports the pinned gates module, numpy).  Deterministic: sorted keys, gzip mtime 0, no wall clock in any output.

  python p5c_rj5_path.py feed --bank-dir B --join J.jsonl.gz --expect-join-sha256 H --platform local=devbox --platform A=devbox --platform B=devbox
         [--platform C=pod] --r0-rows local=DIR --r0-rows A=DIR ... --min-rj5-coverage 0.99 --out-dir F
  python p5c_rj5_path.py rekey --feed-map F/feed_map.jsonl.gz --expect-feed-map-sha256 H --labels OUT/labels_s*of*.jsonl --out R.jsonl
  python p5c_rj5_path.py repair-combine --bank-repair repair_rank0_bank.json --feed-repair repair_rank0_feed.json --out repair_rank0_a3.json
  python p5c_rj5_path.py provenance --feed-map F/feed_map.jsonl.gz --rekey-manifest R.manifest.json --gates-meta G/table_meta_fields.json
         --out-meta G/table_meta_fields_a3.json --out-sidecar C1_provenance.jsonl.gz [--extra NAME=FILE ...]
  python p5c_rj5_path.py fallback --rekeyed R.jsonl --out R_fb.jsonl --allow-row TOKEN:RJ5_TRAJ_SHA256     (A3.3; writes R_fb.jsonl.manifest.json)
  python p5c_rj5_path.py tb0r --gates-module a1/p5c_table_gates.py --feed-bank F --pod-labels OUT/labels_s*of*.jsonl --dev-labels rj5_labels_local.jsonl.gz
         --expect-n 16133 --out tb0r.json [--tc20s-tol 1e-9]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import struct
import sys
from array import array

SCHEMA = "p5c-a3-rj5-path-v1"
JOIN_SCHEMA = "rj5-bank-join-v1"
PLATFORMS = ("devbox", "pod")
LABEL_SOURCES = ("rj5", "bank_fallback")
FALLBACK_REASONS = ("missing", "rank1")
LABEL_FALLBACK_REASON = "tb5_s2_tc20s"   # A3.3 (PI 2026-11-16): the ONE recorded reason a P5-R row may carry its own Tc20s label (it lives in the sidecar's fallback_reason, never in the feed map)
MAX_LABEL_FALLBACK_ROWS = 1              # A3.3 cap, by construction: the ruled row only
R0_BAR_M = 1e-6                      # the R0 identity bar of rj5_bank_join.py (TF-N2 C-R0): a re-roll within 1e-6 m (xy) of the bank traj is "exact"
MAX_DYAW = math.pi                   # traj_label_table._check_label's yaw-branch rule, applied to the RJ5 traj before any label exists
TC20S_TOL = 1e-9                     # A1 R-A


class Refused(Exception):
    def __init__(self, reason, code=2):
        super().__init__(reason)
        self.code = code


# ------------------------------------------------------------------ hashing / io
def traj_sha256(traj) -> str:
    """== refe/traj_label_table.traj_sha256 == p5c_dual_labels.traj_sha (float64 little-endian, row-major)."""
    h = hashlib.sha256()
    for pose in traj:
        for v in pose:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")


def read_jsonl(p, what):
    """yield parsed lines; a torn line is a REFUSAL (an incomplete shard must never be read as a complete one)."""
    with _open(p) as f:
        for n, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                raise Refused(f"{what} {p}: line {n} is torn (not JSON)")


def flat(traj):
    return array("d", [float(v) for p in traj for v in p])


def unflat(a):
    return [[a[i], a[i + 1], a[i + 2]] for i in range(0, len(a), 3)]


def dxy_max(a, b):
    return max(math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]) for i in range(0, len(a), 3))


def _gz_write_lines(path, lines):
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as g:
            for o in lines:
                g.write((json.dumps(o, sort_keys=True) + "\n").encode("utf-8"))
    return sha256_file(path)


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    return sha256_file(path)


def bank_files(bank_dir):
    fs = sorted(glob.glob(os.path.join(bank_dir, "targets_rank*.jsonl")))      # the trainer's own glob (traj_label_table.read_bank_rows)
    names = [os.path.basename(p) for p in fs]
    if "targets_rank0.jsonl" not in names:
        raise Refused(f"{bank_dir}: no targets_rank0.jsonl")
    return fs


def _check_traj(t, where):
    if not isinstance(t, list) or not t or any(not isinstance(p, list) or len(p) != 3 for p in t):
        raise Refused(f"{where}: traj is not a [T, 3] list")
    for p in t:
        for v in p:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise Refused(f"{where}: traj holds a non-finite or non-numeric value {p!r}")


# ------------------------------------------------------------------ feed
def parse_platforms(specs):
    out = {}
    for s in specs or []:
        if "=" not in s:
            raise Refused(f"--platform wants SOURCE=devbox|pod, got {s!r}")
        k, v = s.split("=", 1)
        if v not in PLATFORMS:
            raise Refused(f"--platform {s!r}: platform must be one of {list(PLATFORMS)}")
        if k in out and out[k] != v:
            raise Refused(f"--platform {k} declared twice with different platforms")
        out[k] = v
    return out


def load_r0(specs, bank0):
    """R0 control rows (arm "R0", rj5_roll.py output rj5_*.jsonl) -> ({(log, token): dxy}, {log: max dxy}).  dxy is computed HERE against the real bank row
    (max over poses of the xy distance), never read from the rows' own bank_dmax.  A control row for a key the bank does not have is a REFUSAL."""
    row_dxy, n_files = {}, 0
    for s in specs or []:
        if "=" not in s:
            raise Refused(f"--r0-rows wants SOURCE=DIR, got {s!r}")
        _, d = s.split("=", 1)
        files = sorted(glob.glob(os.path.join(d, "rj5_*.jsonl")))
        if not files:
            raise Refused(f"--r0-rows {s!r}: no rj5_*.jsonl under {d}")
        for fp in files:
            n_files += 1
            for r in read_jsonl(fp, "R0 rows"):
                if r.get("arm") != "R0" or "traj" not in r:
                    continue
                k = (r["log_name"], r["token"])
                if k not in bank0:
                    raise Refused(f"R0 control row {k} (file {os.path.basename(fp)}) is not a bank rank-0 key")
                _check_traj(r["traj"], f"R0 row {k}")
                dx = dxy_max(flat(r["traj"]), bank0[k]["traj"])
                if k in row_dxy and abs(row_dxy[k] - dx) > 0.0:
                    raise Refused(f"R0 control row {k} appears twice with different trajectories")
                row_dxy[k] = dx
    log_max = {}
    for (lg, _), dx in row_dxy.items():
        log_max[lg] = max(log_max.get(lg, 0.0), dx)
    return row_dxy, log_max, n_files


def read_bank(bank_dir):
    """-> (rows in file order [(file basename, dict)], bank0 {(log, token): {"sha", "traj" (flat), "idx"}})"""
    rows, bank0 = [], {}
    for fp in bank_files(bank_dir):
        base = os.path.basename(fp)
        for r in read_jsonl(fp, "bank"):
            for k in ("log_name", "token", "rank", "traj"):
                if k not in r:
                    raise Refused(f"bank {base}: a row lacks {k!r}")
            _check_traj(r["traj"], f"bank row {r['log_name']}|{r['token']}|{r['rank']}")
            rk = int(r["rank"])
            slim = {"log_name": r["log_name"], "token": r["token"], "rank": rk, "traj": flat(r["traj"]), "sha": traj_sha256(r["traj"])}
            for k in ("sim_hz", "step"):
                if k in r:
                    slim[k] = r[k]
            if rk == 0:
                key = (r["log_name"], r["token"])
                if key in bank0:
                    raise Refused(f"bank: duplicate rank-0 key {key}")
                bank0[key] = {"sha": slim["sha"], "traj": slim["traj"], "idx": len(rows)}
            rows.append((base, slim))           # slim copy (the pod bank rows carry many more fields; only these reach the labeller)
    return rows, bank0


def read_join(path, bank0):
    """-> {(log, token): {"status", "source", "traj" (list), "sha"}} for every bank rank-0 key.  Every integrity rule of the module docstring."""
    out = {}
    for d in read_jsonl(path, "join"):
        for k in ("log_name", "token", "rank", "bank_key", "bank_traj_sha256", "status"):
            if k not in d:
                raise Refused(f"join row lacks {k!r}: {str(d)[:120]}")
        key = (d["log_name"], d["token"])
        where = f"join row {key[0]}|{key[1]}"
        if int(d["rank"]) != 0:
            raise Refused(f"{where}: rank {d['rank']} (RJ5 rolls exist for rank-0 keys only)")
        if key in out:
            raise Refused(f"{where}: duplicate key in the join")
        b = bank0.get(key)
        if b is None:
            raise Refused(f"{where}: (log, token) is not a rank-0 key of this bank: wrong-key override REFUSED")
        want_key = f"{key[0]}|{key[1]}|0|{b['sha']}"
        if d["bank_key"] != want_key or d["bank_traj_sha256"] != b["sha"]:
            raise Refused(f"{where}: its bank key/sha ({str(d['bank_traj_sha256'])[:12]}...) is not this bank row's own ({b['sha'][:12]}...): "
                          f"wrong-key override REFUSED (the label would be keyed to another traj)")
        st = d["status"]
        if st == "rj5":
            t = d.get("rj5_traj")
            _check_traj(t, where)
            if len(t) * 3 != len(b["traj"]):
                raise Refused(f"{where}: RJ5 traj has {len(t)} poses, the bank traj {len(b['traj']) // 3}")
            sha = traj_sha256(t)
            if sha != d.get("rj5_traj_sha256"):
                raise Refused(f"{where}: sha256 of rj5_traj {sha[:12]}... != its recorded rj5_traj_sha256 {str(d.get('rj5_traj_sha256'))[:12]}...")
            bt = b["traj"]
            worst = max(abs(float(t[i][2]) - bt[3 * i + 2]) for i in range(len(t)))
            if worst >= MAX_DYAW:
                raise Refused(f"{where}: |RJ5 yaw - bank yaw| reaches {worst:.4f} rad >= pi (a wrap-branch mismatch; the table reader would refuse its label)")
            src = d.get("source")
            if not src:
                raise Refused(f"{where}: status rj5 without a source label")
            out[key] = {"status": "rj5", "source": src, "traj": flat(t), "sha": sha}
        elif st == "missing":
            if d.get("rj5_traj") is not None:
                raise Refused(f"{where}: status missing but an rj5_traj is present")
            out[key] = {"status": "missing", "source": None, "traj": None, "sha": None}
        elif st == "conflict":
            raise Refused(f"{where}: status conflict (two different RJ5 trajs for one key): never resolved silently")
        else:
            raise Refused(f"{where}: unknown status {st!r}")
    absent = [k for k in bank0 if k not in out]
    if absent:
        raise Refused(f"{len(absent):,} bank rank-0 keys are ABSENT from the join (e.g. {absent[:2]}): the join is not about this bank. "
                      f"Only an explicit status 'missing' gets the declared fallback")
    return out


def feed(bank_dir, join_path, expect_join_sha, platform_specs, r0_specs, min_cov, out_dir):
    if not expect_join_sha:
        raise Refused("--expect-join-sha256 is required: the RJ5 source must be pinned")
    jsha = sha256_file(join_path)
    if jsha != expect_join_sha:
        raise Refused(f"join {join_path}: sha256 {jsha[:12]}... != the pinned {expect_join_sha[:12]}...")
    plat = parse_platforms(platform_specs)
    if not r0_specs:
        raise Refused("--r0-rows is required (TB-8: the per-log R0 control status is a disclosed column of every C1 row)")
    if min_cov is None or not (0.0 <= min_cov <= 1.0):
        raise Refused("--min-rj5-coverage is required, in [0, 1] (the committed launch bar)")
    rows, bank0 = read_bank(bank_dir)
    join = read_join(join_path, bank0)
    srcs = sorted({v["source"] for v in join.values() if v["status"] == "rj5"})
    undeclared = [s for s in srcs if s not in plat]
    if undeclared:
        raise Refused(f"join sources {undeclared} have no declared roll platform (--platform SOURCE=devbox|pod): a platform is declared, never guessed")
    row_dxy, log_max, n_r0_files = load_r0(r0_specs, bank0)
    n0 = len(bank0)
    n_rj5 = sum(1 for v in join.values() if v["status"] == "rj5")
    cov = n_rj5 / n0 if n0 else 0.0
    os.makedirs(out_dir, exist_ok=True)
    base_man = {"schema": SCHEMA, "step": "feed", "join_file": os.path.basename(join_path), "join_sha256": jsha, "bank_files_sha256": {os.path.basename(p): sha256_file(p) for p in bank_files(bank_dir)},
                "n_bank_rows": len(rows), "n_bank_rank0_keys": n0, "n_rj5": n_rj5, "rj5_coverage_rank0": cov, "min_rj5_coverage": min_cov, "platforms": plat,
                "r0_rows_files": n_r0_files, "n_r0_rows": len(row_dxy)}
    if cov < min_cov:
        base_man["refused"] = f"RJ5 coverage of rank-0 keys {cov:.6f} < the committed {min_cov}"
        _dump_json(os.path.join(out_dir, "feed_manifest.json"), base_man)
        raise Refused(base_man["refused"], code=3)
    fbank = os.path.join(out_dir, "bank")
    os.makedirs(fbank, exist_ok=True)
    handles = {}
    counts = {}
    maprows = []
    try:
        for base, r in rows:
            if base not in handles:
                handles[base] = open(os.path.join(fbank, base), "w", encoding="utf-8", newline="\n")
            rk = int(r["rank"])
            key = (r["log_name"], r["token"])
            o = {"log_name": r["log_name"], "token": r["token"], "rank": rk}
            for k in ("sim_hz", "step"):
                if k in r:
                    o[k] = r[k]
            bsha = r["sha"]
            m = {"log_name": r["log_name"], "token": r["token"], "rank": rk, "bank_traj_sha256": bsha}
            if rk == 0 and join[key]["status"] == "rj5":
                j = join[key]
                o["traj"] = unflat(j["traj"])
                m.update(label_source="rj5", fallback_reason=None, label_traj_sha256=j["sha"], rj5_source=j["source"], roll_platform=plat[j["source"]])
            else:
                o["traj"] = unflat(r["traj"])
                m.update(label_source="bank_fallback", fallback_reason="rank1" if rk != 0 else "missing", label_traj_sha256=bsha, rj5_source=None, roll_platform=None)
            lg = r["log_name"]
            m["r0_log_status"] = ("diverged" if log_max[lg] > R0_BAR_M else "exact") if lg in log_max else "uncontrolled"
            m["r0_log_max_dxy_m"] = log_max.get(lg)
            m["r0_row_dxy_m"] = row_dxy.get(key) if rk == 0 else None
            handles[base].write(json.dumps(o) + "\n")
            ck = f"{m['label_source']}|{m['fallback_reason']}|rank{rk}"
            counts[ck] = counts.get(ck, 0) + 1
            maprows.append(m)
    finally:
        for h in handles.values():
            h.close()
    # self-check: the fed traj's sha, re-read from disk, is the map's label_traj_sha256 on every row (float repr round-trip, nothing re-serialised wrongly)
    i = 0
    for fp in sorted(glob.glob(os.path.join(fbank, "targets_rank*.jsonl"))):
        for r in read_jsonl(fp, "feed bank"):
            m = maprows[i]
            if (r["log_name"], r["token"], int(r["rank"])) != (m["log_name"], m["token"], m["rank"]) or traj_sha256(r["traj"]) != m["label_traj_sha256"]:
                raise Refused(f"feed self-check: row {i} on disk does not carry the traj the map declares")
            i += 1
    if i != len(maprows):
        raise Refused(f"feed self-check: {i} rows on disk, {len(maprows)} in the map")
    maprows_sorted = sorted(maprows, key=lambda m: (m["log_name"], m["token"], m["rank"]))
    map_sha = _gz_write_lines(os.path.join(out_dir, "feed_map.jsonl.gz"), maprows_sorted)
    r0c = {}
    for m in maprows:
        if m["rank"] == 0:
            r0c[m["r0_log_status"]] = r0c.get(m["r0_log_status"], 0) + 1
    man = dict(base_man, counts=dict(sorted(counts.items())), r0_status_rank0_rows=dict(sorted(r0c.items())), r0_logs={"exact": sum(1 for v in log_max.values() if v <= R0_BAR_M),
               "diverged": sum(1 for v in log_max.values() if v > R0_BAR_M)}, feed_map_sha256=map_sha,
               feed_bank_files_sha256={os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(fbank, "targets_rank*.jsonl")))},
               n_rank0_bank_fallback=counts.get("bank_fallback|missing|rank0", 0))
    _dump_json(os.path.join(out_dir, "feed_manifest.json"), man)
    return man


# ------------------------------------------------------------------ rekey
def load_map(path, expect_sha=None):
    if expect_sha is not None:
        got = sha256_file(path)
        if got != expect_sha:
            raise Refused(f"feed map {path}: sha256 {got[:12]}... != the pinned {expect_sha[:12]}...")
    mp = {}
    for m in read_jsonl(path, "feed map"):
        k = (m["log_name"], m["token"], int(m["rank"]))
        if k in mp:
            raise Refused(f"feed map: duplicate row {k}")
        if m.get("label_source") not in LABEL_SOURCES:
            raise Refused(f"feed map row {k}: label_source {m.get('label_source')!r} is not one of {list(LABEL_SOURCES)}")
        mp[k] = m
    return mp


PROV_FIELDS = ("label_traj_sha256", "label_source", "fallback_reason", "rj5_source", "roll_platform", "r0_log_status", "r0_log_max_dxy_m", "r0_row_dxy_m")


def rekey(map_path, label_paths, out_path, expect_map_sha=None):
    mp = load_map(map_path, expect_map_sha)
    seen = set()
    counts = {}
    n = 0
    files = []
    tmp = out_path + ".partial"
    with open(tmp, "w", encoding="utf-8", newline="\n") as out:
        for fp in label_paths:
            files.append((os.path.basename(fp), sha256_file(fp)))
            for r in read_jsonl(fp, "labeller records"):
                k = (r["log"], r["token"], int(r["rank"]))
                m = mp.get(k)
                if m is None:
                    raise Refused(f"labeller record {k} is not a row of the feed map")
                if k in seen:
                    raise Refused(f"labeller record {k} appears twice (a resumed shard): REFUSED, never deduplicated silently")
                seen.add(k)
                if r.get("traj_sha256") != m["label_traj_sha256"]:
                    raise Refused(f"labeller record {k}: computed from a traj with sha {str(r.get('traj_sha256'))[:12]}..., the feed declares {m['label_traj_sha256'][:12]}... "
                                  f"({m['label_source']}): the labeller did not see the fed traj")
                o = dict(r)
                o["traj_sha256"] = m["bank_traj_sha256"]
                for f_ in PROV_FIELDS:
                    o[f_] = m.get(f_)
                out.write(json.dumps(o, sort_keys=True) + "\n")
                n += 1
                ok = "p5r" in r and "p5r_error" not in r
                ck = f"{m['label_source']}|{m['fallback_reason']}|rank{k[2]}|{'p5r' if ok else 'p5r_error'}"
                counts[ck] = counts.get(ck, 0) + 1
    missing = [k for k in mp if k not in seen]
    if missing:
        os.remove(tmp)
        raise Refused(f"{len(missing):,} feed rows have NO labeller record (e.g. {missing[:2]}): never filled silently")
    os.replace(tmp, out_path)
    man = {"schema": SCHEMA, "step": "rekey", "feed_map_sha256": sha256_file(map_path), "label_files_sha256": dict(files), "n_records": n,
           "counts": dict(sorted(counts.items())), "out": os.path.basename(out_path), "out_sha256": sha256_file(out_path)}
    _dump_json(out_path + ".manifest.json", man)
    return man


# ------------------------------------------------------------------ label fallback (A3.3)
def _is_hex64(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def label_fallback(rekeyed_path, out_path, allow):
    """The ONE ruled row whose P5-R label fails S2 takes its own Tc20s label.  allow = [(token, label_traj_sha256), ...] (the RJ5 traj sha), at most MAX_LABEL_FALLBACK_ROWS entries, rank 0 only.
    A candidate = a record with a P5-R label whose p5r_S2_yaw_branch is not true.  A candidate ON the allow-list is rewritten (p5r := tc20s, p5r_discarded := the old label, p5r_fallback := the reason);
    a candidate NOT on it is left untouched and counted (the gates then FAIL it).  More than MAX_LABEL_FALLBACK_ROWS rewrites is REFUSED.  With an empty allow-list the output is the input, record for record."""
    allow = [tuple(a) for a in allow]
    if len(allow) > MAX_LABEL_FALLBACK_ROWS:
        raise Refused(f"fallback: {len(allow)} allowed rows > the cap {MAX_LABEL_FALLBACK_ROWS} (A3.3 rules exactly one row; any other row needs a new ruling)")
    for tok, h in allow:
        if not tok or not _is_hex64(h):
            raise Refused(f"fallback: allow-list entry ({tok!r}, {h!r}) is not (token, 64-hex RJ5 traj sha256)")
    aset = set(allow)
    man_path = rekeyed_path + ".manifest.json"
    rman = json.load(open(man_path, encoding="utf-8"))
    if rman.get("out_sha256") != sha256_file(rekeyed_path):
        raise Refused(f"fallback: {rekeyed_path} is not the file its rekey manifest describes")
    applied, unlisted, n = [], [], 0
    tmp = out_path + ".partial"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as out:
            for r in read_jsonl(rekeyed_path, "rekeyed records"):
                n += 1
                cand = r.get("p5r") is not None and "p5r_error" not in r and r.get("p5r_S2_yaw_branch") is not True
                key = f"{r['log']}|{r['token']}|{r['rank']}|{r['traj_sha256']}"
                if cand and int(r["rank"]) == 0 and (r["token"], r.get("label_traj_sha256")) in aset:
                    if len(applied) >= MAX_LABEL_FALLBACK_ROWS:
                        raise Refused(f"fallback: more than {MAX_LABEL_FALLBACK_ROWS} rows would take the Tc20s label (A3.3 cap)")
                    tc = r.get("tc20s")
                    try:
                        ok = tc is not None and len(tc) == 20 and all(len(x) == 3 and all(math.isfinite(float(v)) for v in x) for x in tc)
                    except (TypeError, ValueError):
                        ok = False
                    if not ok:
                        raise Refused(f"fallback: the ruled row {key} has no finite [20][3] Tc20s label to fall back to")
                    o = dict(r)
                    o["p5r_discarded"] = r["p5r"]
                    o["p5r"] = r["tc20s"]
                    o["p5r_fallback"] = LABEL_FALLBACK_REASON
                    applied.append({"table_key": key, "log_name": r["log"], "token": r["token"], "rank": int(r["rank"]), "reason": LABEL_FALLBACK_REASON, "label_traj_sha256": r.get("label_traj_sha256"),
                                    "bank_traj_sha256": r["traj_sha256"], "discarded_p5r_S2_yaw_branch": r.get("p5r_S2_yaw_branch"), "discarded_p5r_S1_finite": r.get("p5r_S1_finite")})
                    out.write(json.dumps(o, sort_keys=True) + "\n")
                    continue
                if cand:
                    unlisted.append(key)
                out.write(json.dumps(r, sort_keys=True) + "\n")
    except Refused:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out_path)
    man = {"schema": SCHEMA, "step": "label-fallback", "feed_map_sha256": rman["feed_map_sha256"], "source_rekeyed_sha256": rman["out_sha256"], "source_rekey_manifest_sha256": sha256_file(man_path),
           "label_files_sha256": rman.get("label_files_sha256"), "n_records": n, "counts": rman["counts"], "out": os.path.basename(out_path), "out_sha256": sha256_file(out_path),
           "fallback_reason": LABEL_FALLBACK_REASON, "cap": MAX_LABEL_FALLBACK_ROWS, "allow": [list(a) for a in allow], "n_fallback": len(applied), "fallback_rows": applied,
           "n_unlisted_s2_candidates": len(unlisted), "unlisted_s2_candidates_head": unlisted[:20]}
    _dump_json(out_path + ".manifest.json", man)
    return man


# ------------------------------------------------------------------ repair-combine
def repair_combine(bank_rep_path, feed_rep_path, out_path):
    """RK0 under A3: r_0 = sim8-fail(T) - sim8-fail(label) with T = the BANK traj (C0's target) and the label = the RJ5-based label, on the SAME
    fixed-rule rows.  bank_rep = p5c_dual_labels.py repair on the real bank; feed_rep = the same command on the feed bank."""
    b = json.load(open(bank_rep_path, encoding="utf-8"))
    f = json.load(open(feed_rep_path, encoding="utf-8"))
    for k in ("rank", "rule", "n_requested"):
        if b.get(k) != f.get(k):
            raise Refused(f"repair reports differ in {k!r}: bank {b.get(k)!r} vs feed {f.get(k)!r}")
    if b.get("n_errors") or f.get("n_errors"):
        raise Refused(f"repair reports carry errors (bank {b.get('n_errors')}, feed {f.get('n_errors')}): the sample is not the fixed-rule sample")
    br = {(r["log"], r["token"]): r for r in b["rows"]}
    fr = {(r["log"], r["token"]): r for r in f["rows"]}
    if set(br) != set(fr) or len(br) != len(b["rows"]) or len(fr) != len(f["rows"]):
        raise Refused(f"repair row sets differ: bank {len(br)} rows, feed {len(fr)}, common {len(set(br) & set(fr))} (the sample rule must pick the same rows)")
    rows = []
    for k in sorted(br):
        x, y = br[k], fr[k]
        rows.append({"log": k[0], "token": k[1], "raw_sim8": bool(x["raw_sim8"]), "tc20s_sim8": bool(y["tc20s_sim8"]), "p5r_sim8": bool(y["p5r_sim8"]),
                     "p5r_sim8_scored_alone": y.get("p5r_sim8_scored_alone"), "rj5_raw_sim8": bool(y["raw_sim8"]), "bank_p5r_sim8": bool(x["p5r_sim8"])})
    n = len(rows)
    mean = lambda key: (sum(int(r[key]) for r in rows) / n) if n else None
    res = {k: f[k] for k in ("rank", "rule", "n_requested")}
    res.update(n=n, n_errors=0, raw_sim8_fail_rate=mean("raw_sim8"), tc20s_sim8_fail_rate=mean("tc20s_sim8"), p5r_sim8_fail_rate=mean("p5r_sim8"),
               rj5_raw_sim8_fail_rate=mean("rj5_raw_sim8"), bank_p5r_sim8_fail_rate=mean("bank_p5r_sim8"),
               a3_combined={"schema": SCHEMA, "raw_sim8": "the BANK traj (C0's target)", "tc20s_sim8/p5r_sim8": "the RJ5-based labels",
                            "bank_repair_sha256": sha256_file(bank_rep_path), "feed_repair_sha256": sha256_file(feed_rep_path)},
               rows=rows)
    _dump_json(out_path, res)
    return res


# ------------------------------------------------------------------ provenance
def provenance(map_path, rekey_manifest_path, gates_meta_path, out_meta, out_sidecar, extras=None):
    mp = load_map(map_path)
    rman = json.load(open(rekey_manifest_path, encoding="utf-8"))
    if rman.get("feed_map_sha256") != sha256_file(map_path):
        raise Refused("rekey manifest is about another feed map")
    gm = json.load(open(gates_meta_path, encoding="utf-8"))
    if not isinstance(gm, dict) or "a3" in gm:
        raise Refused(f"{gates_meta_path}: not a gates meta object, or it already carries an 'a3' block")
    fbrows = {f["table_key"]: f for f in rman.get("fallback_rows", [])}        # A3.3: absent in a pre-A3.3 manifest = the old behaviour, byte for byte
    side = []
    for (lg, tok, rk), m in sorted(mp.items()):
        tk = f"{lg}|{tok}|{rk}|{m['bank_traj_sha256']}"
        row = dict({f_: m.get(f_) for f_ in PROV_FIELDS}, table_key=tk)
        if tk in fbrows:
            row["fallback_reason"] = fbrows[tk]["reason"]
            row["p5r_label_replaced_by"] = "tc20s"
        side.append(row)
    n_fb_used = sum(1 for r in side if r["table_key"] in fbrows)
    if n_fb_used != len(fbrows):
        raise Refused(f"provenance: {len(fbrows) - n_fb_used} label-fallback rows of the manifest are not rows of the feed map")
    side_sha = _gz_write_lines(out_sidecar, side)
    c = {}
    for m in mp.values():
        ck = f"{m['label_source']}|{m['fallback_reason']}|rank{m['rank']}"
        c[ck] = c.get(ck, 0) + 1
    plat = {}
    for m in mp.values():
        if m["roll_platform"]:
            plat[f"{m['rj5_source']}={m['roll_platform']}"] = plat.get(f"{m['rj5_source']}={m['roll_platform']}", 0) + 1
    a3 = {"schema": SCHEMA, "feed_map_sha256": rman["feed_map_sha256"], "rekey_manifest_sha256": sha256_file(rekey_manifest_path), "rekeyed_records_sha256": rman["out_sha256"],
          "provenance_sidecar": os.path.basename(out_sidecar), "provenance_sidecar_sha256": side_sha, "rows_by_label_source": dict(sorted(c.items())),
          "rj5_rows_by_source_platform": dict(sorted(plat.items())), "extras_sha256": {}}
    for s in extras or []:
        name, p = s.split("=", 1)
        a3["extras_sha256"][name] = sha256_file(p)
    if fbrows:
        a3["label_fallback_rows"] = {"n": len(fbrows), "reasons": sorted({f["reason"] for f in fbrows.values()}), "table_keys": sorted(fbrows)}
    gm["a3"] = a3
    _dump_json(out_meta, gm)
    return a3


# ------------------------------------------------------------------ tb0r (A3 TB-0 on RJ5 inputs; imports the pinned gates)
def _import_gates(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("p5c_table_gates_pinned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tb0r(gates_path, feed_bank, pod_label_paths, dev_label_paths, expect_n, out_path, tol=TC20S_TOL):
    G = _import_gates(gates_path)
    bank = G.load_bank([feed_bank])
    pod, _, pod_dup = G.load_label_recs(pod_label_paths)
    dev, _, dev_dup = G.load_label_recs(dev_label_paths)
    if pod_dup or dev_dup:
        raise Refused(f"duplicate label records (pod {pod_dup}, dev {dev_dup})")
    M_p = G.merge_p5r(bank, pod)
    M_t = G.merge(bank, pod)
    ctl = {t: {"label": d["p5r"], "sim8": bool(d["p5r_sim8"]), "learn": None, "raw_learn": None} for t, d in dev.items() if "p5r" in d and "p5r_error" not in d}
    ok_p, det_p = G._p5r_vs_control(bank, M_p, ctl, None)
    g_t = G.gate_tb0(bank, M_t, dev, tol, pod)
    n_dev = len(dev)
    n_dev_sha_bad = 0
    idx = {G.triple(bank["log"][i], bank["token"][i], bank["rank"][i]): i for i in range(len(bank["log"]))}
    for t, d in dev.items():
        i = idx.get(t)
        if i is None or d.get("traj_sha256") != bank["sha"][i]:
            n_dev_sha_bad += 1
    cover_ok = n_dev == expect_n and n_dev_sha_bad == 0 and det_p["n_common_rows"] == len(ctl) and g_t["n"] == sum(1 for d in dev.values() if "tc20s" in d and "error" not in d)
    status = "PASS" if (ok_p and g_t["status"] == "PASS" and cover_ok) else "FAIL"
    res = {"id": "TB-0r", "status": status, "schema": SCHEMA, "bar": f"A3 TB-0 on RJ5 inputs: Tc20s max abs diff <= {tol:g} (R-A) AND P5-R R-TOL-1/2/3 (R-B) per rank, on every dev-box RJ5 row "
                                                                      f"(n == {expect_n}, every dev row's traj sha == the fed traj's)",
           "n_dev_records": n_dev, "n_dev_traj_sha_not_fed": n_dev_sha_bad, "coverage_ok": cover_ok, "p5r": {"ok": ok_p, **det_p},
           "tc20s": {"status": g_t["status"], "value": g_t["value"], "n": g_t["n"], **g_t["detail"]},
           "gates_module_sha256": sha256_file(gates_path)}
    _dump_json(out_path, res)
    return res


# ------------------------------------------------------------------ cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("feed")
    f.add_argument("--bank-dir", required=True)
    f.add_argument("--join", required=True)
    f.add_argument("--expect-join-sha256", required=True)
    f.add_argument("--platform", action="append", default=[])
    f.add_argument("--r0-rows", action="append", default=[])
    f.add_argument("--min-rj5-coverage", type=float, required=True)
    f.add_argument("--out-dir", required=True)
    r = sp.add_parser("rekey")
    r.add_argument("--feed-map", required=True)
    r.add_argument("--expect-feed-map-sha256", required=True)
    r.add_argument("--labels", nargs="+", required=True)
    r.add_argument("--out", required=True)
    c = sp.add_parser("repair-combine")
    c.add_argument("--bank-repair", required=True)
    c.add_argument("--feed-repair", required=True)
    c.add_argument("--out", required=True)
    p = sp.add_parser("provenance")
    p.add_argument("--feed-map", required=True)
    p.add_argument("--rekey-manifest", required=True)
    p.add_argument("--gates-meta", required=True)
    p.add_argument("--out-meta", required=True)
    p.add_argument("--out-sidecar", required=True)
    p.add_argument("--extra", action="append", default=[])
    t = sp.add_parser("tb0r")
    t.add_argument("--gates-module", required=True)
    t.add_argument("--feed-bank", required=True)
    t.add_argument("--pod-labels", nargs="+", required=True)
    t.add_argument("--dev-labels", nargs="+", required=True)
    t.add_argument("--expect-n", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--tc20s-tol", type=float, default=TC20S_TOL, help="R-A tolerance of the Tc20s part (default %g, the pre-A3.3 behaviour; A3.3 passes 1e-8)" % TC20S_TOL)
    fb = sp.add_parser("fallback")
    fb.add_argument("--rekeyed", required=True)
    fb.add_argument("--out", required=True)
    fb.add_argument("--allow-row", action="append", default=[], metavar="TOKEN:RJ5_TRAJ_SHA256")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "feed":
            res = feed(a.bank_dir, a.join, a.expect_join_sha256, a.platform, a.r0_rows, a.min_rj5_coverage, a.out_dir)
            print(json.dumps({k: res[k] for k in ("n_bank_rows", "n_rj5", "rj5_coverage_rank0", "counts", "feed_map_sha256")}))
            print("ZZRJ5_FEED_OK")
        elif a.cmd == "rekey":
            res = rekey(a.feed_map, a.labels, a.out, a.expect_feed_map_sha256)
            print(json.dumps({k: res[k] for k in ("n_records", "counts", "out_sha256")}))
            print("ZZRJ5_REKEY_OK")
        elif a.cmd == "repair-combine":
            res = repair_combine(a.bank_repair, a.feed_repair, a.out)
            print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
            print("ZZRJ5_REPAIR_OK")
        elif a.cmd == "provenance":
            res = provenance(a.feed_map, a.rekey_manifest, a.gates_meta, a.out_meta, a.out_sidecar, a.extra)
            print(json.dumps(res))
            print("ZZRJ5_PROVENANCE_OK")
        elif a.cmd == "fallback":
            res = label_fallback(a.rekeyed, a.out, [tuple(x.split(":", 1)) if ":" in x else (x, "") for x in a.allow_row])
            print(json.dumps({k: res[k] for k in ("n_records", "n_fallback", "n_unlisted_s2_candidates", "out_sha256")}))
            print("ZZRJ5_FALLBACK_OK")
        else:
            res = tb0r(a.gates_module, a.feed_bank, a.pod_labels, a.dev_labels, a.expect_n, a.out, a.tc20s_tol)
            print(json.dumps({"status": res["status"], "coverage_ok": res["coverage_ok"], "tc20s_max": res["tc20s"]["value"], "p5r_failed": res["p5r"]["failed_criteria"]}))
            print("ZZTB0R_" + res["status"])
            return 0 if res["status"] == "PASS" else 1
        return 0
    except Refused as e:
        print(f"ZZRJ5_REFUSE {e}")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
