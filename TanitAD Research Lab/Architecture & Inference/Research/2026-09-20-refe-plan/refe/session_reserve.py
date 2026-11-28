#!/usr/bin/env python3
"""P7-WIRE-RESERVE -- remove the P7 session reserve (R-A, `raw/2026-10-22-p7-session-reserve/raw/reserve_sessions.json`) from EVERY log-keyed training input, and PROVE it.

Pure standard library, no torch / numpy at import (the launch gate, the pod chain and the tests all import it).  MM_RULINGS.md Q7; SESSION_RESERVE.md sections 3 and 4; SPEC_P7.md section 2.

THE UNIT.  A SESSION is `log.rsplit("_", 2)[0]` (`2021.06.07.17.46.49_veh-35_02607_03120` -> `2021.06.07.17.46.49_veh-35`).  The reserve is a set of 16 sessions (116 logs).  EXCLUSION IS BY SESSION,
never by bank row, because the aux label store holds 103,139 frames against the bank's 103,037 (a DB-resolved token with no bank row can still reach it).

THE ONE RULE THAT MAKES "RESOLVED" MEAN SOMETHING.  `session_of` REFUSES a string that is not a nuPlan log name.  The bare `rsplit` does not: `"L0".rsplit("_", 2)[0]` is `"L0"`, a "session" that is
in no reserve, so a logless / mangled key would read as a CLEAN key.  That is the zero-is-a-claim-about-the-probe trap, and it is why every function below raises `UnresolvedLog` (or reports
`unresolved`) instead of passing a key it could not attribute.

INTERFACE (the P7 launch gate's G-RESERVE calls exactly these names):
  session_of(log) -> str
  load_reserve(path) -> frozenset[str]                         sha256-pinned (RESERVE_FILE_SHA256); refuses on ANY mismatch
  filter_bank(rank_jsonl_in, out_path, reserve) -> {n_in, n_out, n_excluded, n_unresolved, sha256_out, ...}
  audit_inputs(reserve, *, bank_files, metric_cache_dir=None, label_store=None, image_index=None, calib_rows=None) -> {ok, leaks, unresolved, counts}
Also: filter_bank_dir (a whole bank directory + the trainer-identity digest), assert_rows_clear (the consumer-side refusal), filter_calib, move_reserved / restore_moved (D9 cache and image roots), split_label_store, reserve_logs,
count_from_token_table (the rank-0 count straight from the token table), is_reserved.

HOW EACH INPUT IS KEYED (read from the tip, `file:line` in RESULT.md):
  bank rows ........ `row["log_name"]` + `row["token"]`; a twin (rank >= 1) is built from the SAME scenario object as its parent (`build_teacher_rollouts.make_row(sc, ...)`, `augment_search.py`), so its
                     `log_name` / `token` ARE the parent's; `slow_twins.make_twin` copies the parent row.  Attribution of a twin is therefore its own `log_name`, and `train.py` groups scenes by
                     (log_name, token, step).  `filter_bank(..., token_log=)` additionally cross-checks every row against a token -> log map (the token table) and refuses a conflict.
  D9 metric cache .. `<root>/<log>/<token>[__lane_rank_K].pkl.xz` (`online_labeller.parts_path`): the LOG is the directory, twins live in their parent's log directory.
  aux label store .. `<store>/shards/<log>.aux1` (header JSON `log` + `entries[].t` = token) and `manifest.json` `files["shards/<log>.aux1"]` (`aux_label_store.write_shard / finalize`).
  image roots ...... `<root>/<log>_<CAM>.zip` containers and loose `<root>/**/<log>/<CAM>/<hash>.jpg` (`train.FrameStore`), and refs `<zp>::<member>` / `.../<log>/<CAM>/<hash>.jpg`.
  calibration ...... `calib_table.json["logs"][<log>]` (`calib_table.vectors`).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

# ------------------------------------------------------------------------------------------------ pinned constants
RESERVE_SCHEMA = "p7-session-reserve-v1"
# sha256 of reserve_sessions.json AS COMMITTED (LF line endings).  A CRLF checkout (core.autocrlf=true on the dev box) is the SAME text: `load_reserve` also accepts the CRLF -> LF normalised bytes
# (the JSON is pure ASCII with escaped control characters, so a raw CR can only be a line ending).  MEASURED 2026-10-22: `git archive` on the dev box exports it as 19,971 B / sha 4f5b1f1d... instead
# of the blob's 19,121 B / 53f3bcd4...; any other byte change refuses.
RESERVE_FILE_SHA256 = "53f3bcd4684513f0ee765fbcfbaf583b587f58c97e9c8915f4e4df3ae3e32f29"
RESERVE_SESSIONS_SHA256 = "b16dc83386ad02ce95ed54826bce93ea9e3259ff8696168ffbf836940a711fab"
RESERVE_LOGS_SHA256 = "c651d1f6080134c1d3922c336122c0f7c61bfc08017ecfd8a9b55f470cc8c2f0"
RESERVE_N_SESSIONS = 16
RESERVE_N_LOGS = 116
RESERVE_SCENES = 10141                   # rank-0 bank scenes in the reserve (MEASURED from the token table: count_from_token_table)
BANK_RANK0_SCENES = 103037               # the 3-C bank's rank-0 scenes (launch_gate.P6_BANK_TUPLES comment)

# every nuPlan navtrain / navtest log name (MEASURED: 1,192 + 136 of 1,192 + 136, and all 103,037 attributed table rows, match)
_LOG_RX = re.compile(r"\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+")
_ZIP_RX = re.compile(r"(?P<log>.+)_CAM_[A-Za-z0-9]+\.zip")
TWIN_SEP = "__"


class ReserveError(Exception):
    """A refusal of this module (a leak, a mismatch, a bad input).  Never swallowed by a caller that wants to proceed."""


class UnresolvedLog(ReserveError):
    """A key whose log cannot be resolved to a session.  It is REFUSED, never dropped and never treated as 'not in the reserve'."""


class AttributionConflict(ReserveError):
    """A bank row whose `log_name` disagrees with the token table's log for its token."""


# ------------------------------------------------------------------------------------------------ the unit
def session_of(log) -> str:
    """The session of a nuPlan log name.  Refuses anything that is not one (`UnresolvedLog`)."""
    if not isinstance(log, str) or _LOG_RX.fullmatch(log) is None:
        raise UnresolvedLog(f"not a nuPlan log name: {log!r}")
    return log.rsplit("_", 2)[0]


def _as_reserve(reserve) -> frozenset:
    if isinstance(reserve, (str, bytes)) or not hasattr(reserve, "__iter__"):
        raise ReserveError(f"the reserve must be a set of session names (got {type(reserve).__name__}); use load_reserve(path)")
    r = frozenset(reserve)
    if not r:
        raise ReserveError("the reserve is EMPTY: excluding nothing is a no-op that would read as a clean pass")
    for s in r:
        if not isinstance(s, str) or _LOG_RX.fullmatch(s + "_00000_00000") is None:
            raise ReserveError(f"{s!r} is not a session name (<date>_veh-NN)")
    return r


def is_reserved(log, reserve) -> bool:
    """True iff the log's session is in the reserve.  Raises `UnresolvedLog` for a key that is not a log."""
    return session_of(log) in _as_reserve(reserve)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _digest(lines) -> str:
    """`draw_reserve.digest`: sha256 of the sorted lines joined by LF, plus a final LF."""
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ the reserve file
def _read_pinned(path):
    raw = Path(path).read_bytes()
    got = sha256_bytes(raw)
    if got != RESERVE_FILE_SHA256:
        lf = raw.replace(b"\r\n", b"\n")
        if sha256_bytes(lf) != RESERVE_FILE_SHA256:
            raise ReserveError(f"{path}: sha256 {got} != the pinned {RESERVE_FILE_SHA256} (R-A, reserve_sessions.json): the reserve is not the registered draw")
        raw = lf
    return raw


def _load(path):
    raw = _read_pinned(path)
    doc = json.loads(raw.decode("utf-8"))
    sessions = [x["session"] for x in doc["sessions"]]
    logs = [lg["log"] for x in doc["sessions"] for lg in x["logs"]]
    why = []
    if doc.get("schema") != RESERVE_SCHEMA:
        why.append(f"schema {doc.get('schema')!r}")
    if len(sessions) != RESERVE_N_SESSIONS or len(set(sessions)) != RESERVE_N_SESSIONS:
        why.append(f"{len(sessions)} sessions (distinct {len(set(sessions))}) != {RESERVE_N_SESSIONS}")
    if len(logs) != RESERVE_N_LOGS or len(set(logs)) != RESERVE_N_LOGS:
        why.append(f"{len(logs)} logs != {RESERVE_N_LOGS}")
    if _digest(sessions) != RESERVE_SESSIONS_SHA256 or doc["reserve"].get("sessions_sha256") != RESERVE_SESSIONS_SHA256:
        why.append("sessions digest != the pinned sessions_sha256")
    if _digest(logs) != RESERVE_LOGS_SHA256 or doc["reserve"].get("logs_sha256") != RESERVE_LOGS_SHA256:
        why.append("logs digest != the pinned logs_sha256")
    if not why:
        for lg in logs:
            if session_of(lg) not in set(sessions):
                why.append(f"log {lg} belongs to no listed session")
                break
    if why:
        raise ReserveError(f"{path}: the reserve file is internally inconsistent: " + "; ".join(why))
    return frozenset(sessions), frozenset(logs), doc


def load_reserve(path) -> frozenset:
    """The reserve's SESSION names.  Refuses (`ReserveError`) unless the file's sha256 equals RESERVE_FILE_SHA256 (CRLF-normalised bytes accepted) and its content is consistent with the pinned digests."""
    return _load(path)[0]


def reserve_logs(path) -> frozenset:
    """The reserve's LOG names (the draw's 116), after the same verification as `load_reserve`."""
    return _load(path)[1]


def _reserve_sha_info(path) -> dict:
    s, lg, doc = _load(path)
    return {"file_sha256": RESERVE_FILE_SHA256, "sessions": len(s), "logs": len(lg), "sessions_sha256": RESERVE_SESSIONS_SHA256, "logs_sha256": RESERVE_LOGS_SHA256}


# ------------------------------------------------------------------------------------------------ the bank
def _digest_files(paths) -> str:
    """== train._digest_files (the trainer's `targets_sha256`): sha256 over (basename, bytes) of each file in the given order."""
    h = hashlib.sha256()
    for q in paths:
        h.update(os.path.basename(str(q)).encode("utf-8"))
        with open(q, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


def _aug_tag(row) -> str:
    a = row.get("aug")
    if isinstance(a, dict) and a.get("kind") is not None:
        return f"|{a.get('kind')}={a.get('value')}"
    return ""


def _row_key(row) -> str:
    return f"{row.get('log_name')}/{row.get('token')}|rank{row.get('rank', 0)}{_aug_tag(row)}"


def _iter_json_rows(path):
    """(lineno, row | None, why | None) for each non-blank line of a JSONL file, read as bytes (never re-encoded)."""
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                yield lineno, raw, None, "unparseable JSON"
                continue
            if not isinstance(row, dict):
                yield lineno, raw, None, "not a JSON object"
                continue
            yield lineno, raw, row, None


def filter_bank(rank_jsonl_in, out_path, reserve, *, token_log=None) -> dict:
    """Write `rank_jsonl_in` to `out_path` WITHOUT the rows whose log's session is in the reserve (rank-0 rows AND their twins: a twin carries its parent's `log_name`).

    Every surviving row is copied byte for byte, in order.  A row whose `log_name` cannot be resolved (missing, empty, not a log name) or a line that is not a JSON object is REFUSED: `UnresolvedLog` is
    raised after the whole file was scanned (so the message carries the count and the first line numbers) and NOTHING is written.  `n_unresolved` is therefore 0 in every returned dict.
    `token_log` (optional, {token: log}): a row whose token is in the map must carry that log (else `AttributionConflict`); `n_token_checked` / `n_token_unmapped` say how many rows it covered.
    Returns {n_in, n_out, n_excluded, n_unresolved, sha256_out, ...}: also by-rank counts, the excluded sessions and the input's sha256."""
    reserve = _as_reserve(reserve)
    src, dst = Path(rank_jsonl_in), Path(out_path)
    if not src.is_file():
        raise ReserveError(f"{src}: no such bank file")
    if src.resolve() == dst.resolve():
        raise ReserveError("filter_bank refuses to rewrite its input in place")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    n_in = n_out = n_exc = n_checked = n_unmapped = 0
    bad, conflicts = [], []
    by_in, by_out, by_exc, exc_sessions = {}, {}, {}, {}
    h = hashlib.sha256()
    try:
        with open(tmp, "wb") as fo:
            for lineno, raw, row, why in _iter_json_rows(src):
                n_in += 1
                if row is None:
                    bad.append((lineno, why))
                    continue
                log = row.get("log_name")
                try:
                    sess = session_of(log)
                except UnresolvedLog:
                    bad.append((lineno, f"log_name {log!r}"))
                    continue
                if token_log is not None:
                    want = token_log.get(row.get("token"))
                    if want is None:
                        n_unmapped += 1
                    else:
                        n_checked += 1
                        if want != log:
                            conflicts.append((lineno, f"token {row.get('token')} is in {want} (token table), the row says {log}"))
                            continue
                rk = str(int(row.get("rank", 0)))
                by_in[rk] = by_in.get(rk, 0) + 1
                if sess in reserve:
                    n_exc += 1
                    by_exc[rk] = by_exc.get(rk, 0) + 1
                    exc_sessions[sess] = exc_sessions.get(sess, 0) + 1
                    continue                                                        # THE EXCLUSION: a reserved row is not written
                data = raw if raw.endswith(b"\n") else raw + b"\n"
                fo.write(data)
                h.update(data)
                n_out += 1
                by_out[rk] = by_out.get(rk, 0) + 1
        if bad:
            raise UnresolvedLog(f"{src}: {len(bad)} of {n_in} rows REFUSED (no resolvable log): " + "; ".join(f"line {i}: {w}" for i, w in bad[:5]))
        if conflicts:
            raise AttributionConflict(f"{src}: {len(conflicts)} rows disagree with the token table: " + "; ".join(f"line {i}: {w}" for i, w in conflicts[:3]))
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    try:
        os.replace(tmp, dst)                                                 # after the `with` closed the file: Windows refuses to rename an open file
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"n_in": n_in, "n_out": n_out, "n_excluded": n_exc, "n_unresolved": 0, "sha256_out": h.hexdigest(), "sha256_in": sha256_file(src),
            "by_rank_in": by_in, "by_rank_out": by_out, "by_rank_excluded": by_exc, "excluded_sessions": dict(sorted(exc_sessions.items())),
            "n_token_checked": n_checked, "n_token_unmapped": n_unmapped, "in": str(src), "out": str(dst)}


def filter_bank_dir(bank_dir, out_dir, reserve, *, token_log=None, reserve_info=None) -> dict:
    """filter_bank over every `targets_rank*.jsonl` of a bank directory, same file names (the trainer reads exactly that glob).  Refuses a stray `targets_*.jsonl` (the trainer refuses it too: it would be a silently
    dropped arm) and a directory with no rank file.  Writes `reserve_filter_report.json` beside the files and returns it.  `targets_sha256` is the trainer's identity digest of the OUTPUT files."""
    reserve = _as_reserve(reserve)
    src, dst = Path(bank_dir), Path(out_dir)
    files = sorted(src.glob("targets_rank*.jsonl"))
    if not files:
        raise ReserveError(f"{src}: no targets_rank*.jsonl")
    stray = sorted(set(src.glob("targets_*.jsonl")) - set(files))
    if stray:
        raise ReserveError(f"{src}: {[s.name for s in stray]} do not match targets_rank*.jsonl and would be silently dropped")
    dst.mkdir(parents=True, exist_ok=True)
    per, tot = {}, {"n_in": 0, "n_out": 0, "n_excluded": 0, "n_unresolved": 0}
    by_in, by_out, by_exc, sess = {}, {}, {}, {}
    for fp in files:
        r = filter_bank(fp, dst / fp.name, reserve, token_log=token_log)
        per[fp.name] = {k: r[k] for k in ("n_in", "n_out", "n_excluded", "sha256_in", "sha256_out")}
        for k in tot:
            tot[k] += r[k]
        for d, s in ((by_in, r["by_rank_in"]), (by_out, r["by_rank_out"]), (by_exc, r["by_rank_excluded"]), (sess, r["excluded_sessions"])):
            for k, v in s.items():
                d[k] = d.get(k, 0) + v
    rep = {"schema": "p7-reserve-filter-report-v1", **tot, "by_rank_in": by_in, "by_rank_out": by_out, "by_rank_excluded": by_exc, "excluded_sessions": dict(sorted(sess.items())),
           "files": per, "source_targets_sha256": _digest_files(files), "targets_sha256": _digest_files(sorted(dst.glob("targets_rank*.jsonl"))),
           "reserve": reserve_info or {"sessions": len(reserve)}, "source_bank": str(src), "out_bank": str(dst),
           "token_log_checked": token_log is not None}
    (dst / "reserve_filter_report.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return rep


def assert_rows_clear(rows, reserve, *, what: str = "rows") -> int:
    """For a CONSUMER (the trainer's TargetBank, a downstream arm's loader): raise `ReserveError` if any row's `log_name` is in the reserve, `UnresolvedLog` if one resolves to nothing.  Returns the
    number of rows checked (so a caller can assert it is non-zero).  This is the trainer-side twin of `audit_inputs`; SPEC_P7 section 2 asks for the refusal string to live in `train.py`, which this
    module does not edit."""
    reserve = _as_reserve(reserve)
    n, leaked, bad = 0, [], []
    for r in rows:
        n += 1
        try:
            if session_of(r.get("log_name")) in reserve:
                leaked.append(_row_key(r))
        except UnresolvedLog:
            bad.append(_row_key(r))
    if bad:
        raise UnresolvedLog(f"{len(bad)} of {n} {what} name no resolvable log, e.g. {bad[:3]}")
    if leaked:
        raise ReserveError(f"{len(leaked)} of {n} {what} belong to the P7 session reserve, e.g. {leaked[:3]}: the reserve may not be trained on")
    return n


# ------------------------------------------------------------------------------------------------ the calibration table
def _calib_logs(calib_rows) -> list:
    """The log names a calibration input holds: a path to calib_table.json, the loaded table, a {log: ...} dict, or an iterable of names."""
    if isinstance(calib_rows, (str, os.PathLike)):
        calib_rows = json.loads(Path(calib_rows).read_text(encoding="utf-8"))
    if isinstance(calib_rows, dict):
        if isinstance(calib_rows.get("logs"), dict):
            return list(calib_rows["logs"])
        return list(calib_rows)
    return list(calib_rows)


def filter_calib(calib_in, calib_out, reserve) -> dict:
    """`calib_table.json` WITHOUT the reserve's logs (same `fields` / per-log rigs; `n_logs` and `distinct_rigs_exact` recomputed, provenance added).  A log that cannot be resolved is REFUSED."""
    reserve = _as_reserve(reserve)
    tab = json.loads(Path(calib_in).read_text(encoding="utf-8"))
    if not isinstance(tab.get("logs"), dict):
        raise ReserveError(f"{calib_in}: no `logs` table")
    keep, drop = {}, []
    for lg, rig in tab["logs"].items():
        if session_of(lg) in reserve:
            drop.append(lg)
        else:
            keep[lg] = rig
    out = {k: v for k, v in tab.items() if k not in ("logs", "n_logs", "distinct_rigs_exact")}
    chans = sorted({ch for rig in keep.values() if isinstance(rig, dict) for ch in rig})
    out["n_logs"] = len(keep)
    out["distinct_rigs_exact"] = {ch: len({tuple(rig[ch]) for rig in keep.values() if ch in rig}) for ch in chans}
    out["logs"] = keep
    out["reserve_filtered"] = {"excluded_logs": len(drop), "logs_in": len(tab["logs"]), "source": str(calib_in), "source_sha256": sha256_file(calib_in)}
    dst = Path(calib_out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8", newline="\n")
    os.replace(tmp, dst)
    return {"logs_in": len(tab["logs"]), "logs_out": len(keep), "excluded": len(drop), "excluded_logs": sorted(drop), "sha256_out": sha256_file(dst)}


# ------------------------------------------------------------------------------------------------ the image roots
def _scan_images_dir(root, max_depth: int = 4):
    """[(kind, relpath, log | None)] for the per-log containers and log directories under an image root.  `<log>_<CAM>.zip` files and directories NAMED as a log are the units; a log directory is not
    descended into (its frames are all that log's) and neither is anything deeper than `max_depth`.  A `.zip` that does not parse as `<log>_<CAM>.zip` has log None (unresolved).  Other files (`.done_N`) are ignored."""
    root = Path(root)
    out = []

    def walk(d: Path, depth: int):
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as e:
            raise ReserveError(f"{d}: cannot list ({e})")
        for e in entries:
            rel = os.path.relpath(e.path, root).replace("\\", "/")
            if e.is_dir():
                if _LOG_RX.fullmatch(e.name):
                    out.append(("dir", rel, e.name))
                elif depth < max_depth:
                    walk(Path(e.path), depth + 1)
            elif e.name.endswith(".zip"):
                m = _ZIP_RX.fullmatch(e.name)
                out.append(("zip", rel, m.group("log") if m and _LOG_RX.fullmatch(m.group("log")) else None))

    walk(root, 1)
    return out


def _log_of_ref(ref):
    """The log an image reference names: `<container>.zip::<member>` (the container's `<log>_<CAM>.zip`) or a path with a log-named component.  None if it names none."""
    if not isinstance(ref, str):
        return None
    head = ref.split("::", 1)[0].replace("\\", "/")
    if "::" in ref:
        m = _ZIP_RX.fullmatch(head.rsplit("/", 1)[-1])
        if m and _LOG_RX.fullmatch(m.group("log")):
            return m.group("log")
    for part in head.split("/"):
        if _LOG_RX.fullmatch(part):
            return part
    return None


# ------------------------------------------------------------------------------------------------ moving the reserve out of a shared root (reversible)
MOVED_MANIFEST = "reserve_moved.json"


def _plan_moves(kind, root, reserve):
    root = Path(root)
    if not root.is_dir():
        raise ReserveError(f"{root}: no such directory")
    if kind == "cache":
        plan = []
        for e in sorted(os.scandir(root), key=lambda e: e.name):
            if not e.is_dir():
                raise UnresolvedLog(f"{root}: {e.name!r} is a file at the root of the metric cache (every entry is <log>/<token>.pkl.xz)")
            if session_of(e.name) in reserve:
                plan.append(e.name)
        return plan
    if kind == "images":
        items = _scan_images_dir(root)
        bad = [rel for (_k, rel, lg) in items if lg is None]
        if bad:
            raise UnresolvedLog(f"{root}: {len(bad)} container(s) that name no log, e.g. {bad[:3]}")
        return [rel for (_k, rel, lg) in items if session_of(lg) in reserve]
    raise ReserveError(f"unknown kind {kind!r} (cache | images)")


def move_reserved(kind, root, to, reserve, *, dry_run: bool = False) -> dict:
    """MOVE (never copy) the reserve's per-log entries out of `root` into `to` (same relative paths): kind `cache` = the `<log>/` directories of the D9 metric cache, kind `images` = the
    `<log>_<CAM>.zip` containers and `<log>/` frame directories.  `to/reserve_moved.json` records every item ever moved (a re-run MERGES into it, so an idempotent re-run never forgets what a
    restore must bring back) and is written BEFORE the first move.  An item already at its destination is counted (`n_already`), not moved again; one present at BOTH places, or recorded but
    present at NEITHER, is refused.  `restore_moved(to)` undoes it.  A root with an entry that names no log is refused untouched."""
    reserve = _as_reserve(reserve)
    root, to = Path(root), Path(to)
    plan = _plan_moves(kind, root, reserve)
    mf = to / MOVED_MANIFEST
    old = []
    if mf.is_file():
        prev = json.loads(mf.read_text(encoding="utf-8"))
        if prev.get("kind") != kind or Path(prev.get("root", "")) != root:
            raise ReserveError(f"{mf} records another move ({prev.get('kind')} from {prev.get('root')}), not {kind} from {root}")
        old = list(prev.get("items", []))
    items = sorted(set(old) | set(plan))
    rep = {"schema": "p7-reserve-moved-v1", "kind": kind, "root": str(root), "to": str(to), "n_planned": len(items), "items": items, "reserve_sessions": sorted(reserve)}
    for rel in items:
        here, there = (root / rel).exists(), (to / rel).exists()
        if here and there:
            raise ReserveError(f"{rel} exists in both {root} and {to}: refusing to overwrite")
        if not here and not there and not dry_run:
            raise ReserveError(f"{rel} is recorded as reserve but is in neither {root} nor {to}")
    if dry_run:
        return {**rep, "dry_run": True, "n_moved": 0, "n_already": sum(1 for rel in items if not (root / rel).exists() and (to / rel).exists())}
    to.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    n_moved = n_already = 0
    for rel in items:
        src, dst = root / rel, to / rel
        if not src.exists():
            n_already += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n_moved += 1
    return {**rep, "dry_run": False, "n_moved": n_moved, "n_already": n_already}


def restore_moved(to) -> dict:
    """Undo `move_reserved`: move every item named in `to/reserve_moved.json` back under its `root`."""
    to = Path(to)
    man = json.loads((to / MOVED_MANIFEST).read_text(encoding="utf-8"))
    root, n = Path(man["root"]), 0
    for rel in man["items"]:
        src, dst = to / rel, root / rel
        if not src.exists():
            continue
        if dst.exists():
            raise ReserveError(f"{rel} exists in both {to} and {root}: refusing to overwrite")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n += 1
    return {"restored": n, "root": str(root)}


# ------------------------------------------------------------------------------------------------ the aux label store
_AUX_MAGIC = b"AUXSDF1\n"


def _read_aux_header(path) -> dict:
    """The JSON header of an `.aux1` shard (stdlib; == aux_label_store.read_shard(..., with_blobs=False)[0])."""
    with open(path, "rb") as f:
        if f.read(len(_AUX_MAGIC)) != _AUX_MAGIC:
            raise ValueError("bad magic")
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError("truncated header length")
        (hl,) = struct.unpack("<I", raw)
        body = f.read(hl)
        if len(body) != hl:
            raise ValueError("truncated header")
    return json.loads(body.decode("utf-8"))


def split_label_store(store, reserve_store, reserve, *, dry_run: bool = False) -> dict:
    """Move the reserve's shards `<store>/shards/<log>.aux1` into `<reserve_store>/shards/` (an evaluation-only store); copy `run_meta_*.json` so `aux_label_store.py finalize` works on both.  The caller then
    RE-FINALISES both stores (the training store's manifest still lists the moved shards until it is: `audit_inputs` flags that as a leak).  A shard whose file name and header log disagree is refused."""
    reserve = _as_reserve(reserve)
    shards = Path(store) / "shards"
    if not shards.is_dir():
        raise ReserveError(f"{shards}: no such directory")
    moves, bad = [], []
    for e in sorted(os.scandir(shards), key=lambda e: e.name):
        if not e.name.endswith(".aux1"):
            bad.append(e.name)
            continue
        stem = e.name[: -len(".aux1")]
        try:
            head = _read_aux_header(e.path)
            if head.get("log") != stem:
                bad.append(f"{e.name} (header log {head.get('log')!r})")
                continue
            if session_of(stem) in reserve:
                moves.append(e.name)
        except (ValueError, UnresolvedLog, OSError) as ex:
            bad.append(f"{e.name} ({type(ex).__name__})")
    if bad:
        raise UnresolvedLog(f"{shards}: {len(bad)} shard(s) with no attributable log, e.g. {bad[:3]}")
    if not dry_run:
        out = Path(reserve_store) / "shards"
        out.mkdir(parents=True, exist_ok=True)
        for nm in moves:
            if (out / nm).exists() and (shards / nm).exists():
                raise ReserveError(f"{nm} exists in both stores")
            if (shards / nm).exists():
                shutil.move(str(shards / nm), str(out / nm))
        for p in sorted(Path(store).glob("run_meta_*.json")):
            shutil.copy2(p, Path(reserve_store) / p.name)
    return {"n_moved": len(moves), "shards": moves, "dry_run": dry_run}


# ------------------------------------------------------------------------------------------------ the audit
def audit_inputs(reserve, *, bank_files, metric_cache_dir=None, label_store=None, image_index=None, calib_rows=None, max_report: int = 2000) -> dict:
    """Prove that no log-keyed training input holds a key from the reserve and that every key's log RESOLVES.

    bank_files        paths to `targets_rank*.jsonl` (rank-0 AND rank-1): each row's `log_name`.  Required, non-empty.
    metric_cache_dir  the D9 root `<log>/<token>[__twin].pkl.xz`: each directory name is a log, each file inside a leaked directory is a leaked key; a FILE at the root names no log (unresolved).
    label_store       the aux label store: `shards/<log>.aux1` (file name AND header log must agree and resolve), each entry of a reserved shard a leaked key; `manifest.json`'s `files` listing is audited too.
    image_index       a directory (the image root: `<log>_<CAM>.zip` containers + log-named directories), a {basename: path} index, or an iterable of image references.
    calib_rows        a path to calib_table.json, the loaded table, a {log: rig} mapping, or an iterable of log names.
    Returns {ok, leaks: {input: [keys]}, unresolved: {input: n}, counts: {...}}.  `ok` is False on ANY leak, ANY unresolved key, and on an audited input that examined ZERO keys (an empty
    directory passed as the cache is a mis-wired probe, not a clean input: `counts["empty_inputs"]`).  A path that does not exist RAISES (`ReserveError`).  `leaks` lists are sorted and capped at
    `max_report`; `counts["leak_totals"]` carries the true numbers."""
    reserve = _as_reserve(reserve)
    bank_files = [Path(p) for p in (bank_files or ())]
    if not bank_files:
        raise ReserveError("audit_inputs: bank_files is empty (the bank is the one input that is never optional)")
    leaks, unres, examined, examples = {}, {}, {}, {}

    def note_bad(inp, what):
        unres[inp] = unres.get(inp, 0) + 1
        ex = examples.setdefault(inp, [])
        if len(ex) < 10:
            ex.append(what)

    def check_log(inp, log, key):
        examined[inp] = examined.get(inp, 0) + 1
        try:
            sess = session_of(log)
        except UnresolvedLog:
            note_bad(inp, f"{key}: no resolvable log ({log!r})")
            return
        if sess in reserve:
            leaks.setdefault(inp, []).append(key)

    leaks["bank"], unres["bank"], examined["bank"] = [], 0, 0
    for p in bank_files:
        if not p.is_file():
            raise ReserveError(f"{p}: no such bank file")
        for lineno, _raw, row, why in _iter_json_rows(p):
            if row is None:
                examined["bank"] += 1
                note_bad("bank", f"{p.name}:{lineno}: {why}")
                continue
            check_log("bank", row.get("log_name"), _row_key(row))

    if metric_cache_dir is not None:
        root = Path(metric_cache_dir)
        if not root.is_dir():
            raise ReserveError(f"{root}: no such metric-cache directory")
        leaks["metric_cache"], unres["metric_cache"], examined["metric_cache"] = [], 0, 0
        for e in sorted(os.scandir(root), key=lambda e: e.name):
            if not e.is_dir():
                examined["metric_cache"] += 1
                note_bad("metric_cache", f"{e.name}: a file at the cache root has no log")
                continue
            before = len(leaks["metric_cache"])
            check_log("metric_cache", e.name, e.name + "/")
            if len(leaks["metric_cache"]) > before:
                leaks["metric_cache"].pop()                                  # replaced by the directory's files, below
                files = sorted(f.name for f in os.scandir(e.path))
                leaks["metric_cache"] += [f"{e.name}/{nm}" for nm in files] or [e.name + "/"]

    if label_store is not None:
        store = Path(label_store)
        shards = store / "shards"
        if not shards.is_dir():
            raise ReserveError(f"{shards}: no such label-store directory")
        leaks["label_store"], unres["label_store"], examined["label_store"] = [], 0, 0
        for e in sorted(os.scandir(shards), key=lambda e: e.name):
            if not e.name.endswith(".aux1"):
                examined["label_store"] += 1
                note_bad("label_store", f"{e.name}: not a shard")
                continue
            stem = e.name[: -len(".aux1")]
            try:
                head = _read_aux_header(e.path)
            except (ValueError, OSError) as ex:
                examined["label_store"] += 1
                note_bad("label_store", f"{e.name}: unreadable header ({type(ex).__name__})")
                continue
            if head.get("log") != stem:
                examined["label_store"] += 1
                note_bad("label_store", f"{e.name}: header log {head.get('log')!r} != the file name")
                continue
            before = len(leaks["label_store"])
            check_log("label_store", stem, stem)
            if len(leaks["label_store"]) > before:
                leaks["label_store"].pop()
                leaks["label_store"] += [f"{stem}/{x.get('t')}" for x in head.get("entries", [])] or [stem]
        man = store / "manifest.json"
        if man.is_file():
            try:
                files = json.loads(man.read_text(encoding="utf-8")).get("files", {})
            except ValueError:
                files = None
                examined["label_store"] += 1
                note_bad("label_store", "manifest.json: unparseable")
            for rel in sorted(files or {}):
                nm = rel.rsplit("/", 1)[-1]
                log = nm[: -len(".aux1")] if nm.endswith(".aux1") else nm
                check_log("label_store", log, f"manifest:{log}")            # a stale manifest still listing a reserved shard IS a leak: its store_sha256 would pin it

    if image_index is not None:
        leaks["images"], unres["images"], examined["images"] = [], 0, 0
        if isinstance(image_index, (str, os.PathLike)):
            root = Path(image_index)
            if not root.is_dir():
                raise ReserveError(f"{root}: no such image root")
            for kind, rel, lg in _scan_images_dir(root):
                if lg is None:
                    examined["images"] += 1
                    note_bad("images", f"{rel}: a container that names no log")
                else:
                    check_log("images", lg, f"{kind}:{rel}")
        else:
            refs = list(image_index.values()) if isinstance(image_index, dict) else list(image_index)
            for ref in refs:
                check_log("images", _log_of_ref(ref), f"ref:{ref}")

    if calib_rows is not None:
        leaks["calibration"], unres["calibration"], examined["calibration"] = [], 0, 0
        for lg in _calib_logs(calib_rows):
            check_log("calibration", lg, f"calib:{lg}")

    totals = {k: len(v) for k, v in leaks.items()}
    capped = {k: sorted(v)[:max_report] for k, v in leaks.items()}
    empty = sorted(k for k, n in examined.items() if n == 0)
    ok = not any(totals.values()) and not any(unres.values()) and not empty
    return {"ok": bool(ok), "leaks": capped, "unresolved": unres,
            "counts": {"reserve_sessions": len(reserve), "inputs_audited": sorted(examined), "examined": examined, "leak_totals": totals,
                       "empty_inputs": empty, "unresolved_examples": examples, "bank_files": [p.name for p in bank_files]}}


# ------------------------------------------------------------------------------------------------ the token table (rank-0 counts, token -> log)
def _table_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            yield r


def token_log_map(table_path) -> dict:
    """{token: log} for the table rows that carry a log (103,037 of 103,288)."""
    return {r["token"]: r["log"] for r in _table_rows(table_path) if r.get("log")}


def count_from_token_table(table_path, reserve) -> dict:
    """The rank-0 exclusion, MEASURED from the token table alone (one row per scene; the 3-C bank's rank-0 scenes are its attributed rows): how many of the attributed scenes lie in the reserve."""
    reserve = _as_reserve(reserve)
    rows = attributed = excl = 0
    per = {}
    for r in _table_rows(table_path):
        rows += 1
        if not r.get("log"):
            continue
        attributed += 1
        s = session_of(r["log"])
        if s in reserve:
            excl += 1
            per[s] = per.get(s, 0) + 1
    return {"table_rows": rows, "attributed": attributed, "logless": rows - attributed, "n_excluded": excl, "n_remaining": attributed - excl, "per_session": dict(sorted(per.items())),
            "sessions_hit": len(per)}


# ------------------------------------------------------------------------------------------------ CLI
def _cli_filter(a) -> int:
    reserve = load_reserve(a.reserve)
    tl = token_log_map(a.token_table) if a.token_table else None
    rep = filter_bank_dir(a.bank, a.out, reserve, token_log=tl, reserve_info=_reserve_sha_info(a.reserve))
    print(json.dumps({k: rep[k] for k in ("n_in", "n_out", "n_excluded", "n_unresolved", "by_rank_in", "by_rank_out", "by_rank_excluded", "targets_sha256")}, sort_keys=True))
    return 0


def _cli_audit(a) -> int:
    reserve = load_reserve(a.reserve)
    bank = sorted(Path(a.bank).glob("targets_rank*.jsonl")) if Path(a.bank).is_dir() else [Path(a.bank)]
    r = audit_inputs(reserve, bank_files=bank, metric_cache_dir=a.cache, label_store=a.label_store, image_index=a.images, calib_rows=a.calib)
    print(json.dumps(r, sort_keys=True)[:4000])
    return 0 if r["ok"] else 1


def _cli_counts(a) -> int:
    reserve = load_reserve(a.reserve)
    r = count_from_token_table(a.token_table, reserve)
    print(json.dumps({k: v for k, v in r.items() if k != "per_session"}, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--reserve", required=True)
    p = sub.add_parser("counts", help="the rank-0 exclusion from the token table")
    p.add_argument("--reserve", required=True)
    p.add_argument("--token-table", required=True)
    p = sub.add_parser("filter", help="a bank directory -> its reserve-free copy (+ reserve_filter_report.json); this is also the rank-1 measurement")
    p.add_argument("--reserve", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-table", default=None, help="cross-check every row's log against the token table (a twin must carry its parent's log)")
    p = sub.add_parser("audit")
    p.add_argument("--reserve", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--label-store", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--calib", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "verify":
            print(json.dumps(_reserve_sha_info(a.reserve), sort_keys=True))
            return 0
        return {"counts": _cli_counts, "filter": _cli_filter, "audit": _cli_audit}[a.cmd](a)
    except ReserveError as e:
        print(f"ZZRESERVE_REFUSED {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
