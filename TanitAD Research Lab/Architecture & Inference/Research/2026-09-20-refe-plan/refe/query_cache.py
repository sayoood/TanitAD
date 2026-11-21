"""Exact memoisation of the nuPlan DB queries a teacher rollout repeats -- data-prep speed lever #2.

⛔ MEASURED 2026-09-23 (cProfile, 12 navtrain frames, dev box, map cache ON): 210 s of 319 s (66 %)
of a cached rollout is sqlite -- `execute_many` row fetch 128.5 s (118,720 rows), cursor.execute
61.2 s, sqlite3.connect 15.0 s, connection.close 7.8 s. 118,720 of those rows come from
`get_tracked_objects_for_lidarpc_token_from_db`, i.e. the per-lidar_pc box query that
`get_future_tracked_objects` issues once per future sample at EVERY simulation step, so the same
lidar_pc's boxes are read again and again; and every query opens + parses + closes its own
connection (nuplan's query_session is deliberately stateless).

EXACT BY CONSTRUCTION -- the rows this produces must equal the ones already banked:
  * the navtrain DBs are READ-ONLY during data prep, so a query's rows are a pure function of
    (db_file, query_text, parameters);
  * a sqlite3.Row is immutable and holds only immutable values (int/float/str/bytes), and it stays
    valid after its connection closes, so a hit re-yields the very rows the DB returned, in order;
  * the parsers downstream (`_parse_tracked_object_row` ...) run again on every hit, so stateful
    helpers such as `get_unique_incremental_track_id` see the identical call sequence;
  * a result is stored only when the caller EXHAUSTED the generator (a caller that stops early,
    e.g. `next(...)`, gets the original lazy behaviour and nothing is cached);
  * parameters that cannot be keyed exactly bypass the cache.
A miss runs on a persistent per-thread connection opened exactly as nuplan opens it (same
`sqlite3.connect(db_file)`, same row_factory) -- it only skips the reconnect + schema re-parse.
REFE_QUERY_CACHE_VERIFY=1 re-runs the original query on every hit and raises on any difference
(`diag_query_cache.py`).
"""
from __future__ import annotations

import collections
import os
import sqlite3
import threading

_INSTALLED = False
STATS = {"hit": 0, "miss": 0, "bypass": 0, "partial": 0, "evict": 0, "verified": 0,
         "one_hit": 0, "one_miss": 0}
MAX_ROWS = int(os.environ.get("REFE_QUERY_CACHE_ROWS", "400000"))
MAX_CONN = 4
_MISSING = object()


def _key(query_text, params, db_file):
    try:
        if params is None:
            p = None
        else:
            p = tuple(bytes(v) if isinstance(v, (bytearray, memoryview)) else v for v in params)
        k = (db_file, query_text, p)
        hash(k)
        return k
    except TypeError:
        return None


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    from nuplan.database.nuplan_db import nuplan_scenario_queries as Q
    from nuplan.database.nuplan_db import query_session as S
    orig_many, orig_one = S.execute_many, S.execute_one
    verify = os.environ.get("REFE_QUERY_CACHE_VERIFY", "0") == "1"
    cache: "collections.OrderedDict" = collections.OrderedDict()
    size = [0]
    tls = threading.local()

    def _conn(db_file):
        conns = getattr(tls, "conns", None)
        if conns is None:
            conns = tls.conns = collections.OrderedDict()
        c = conns.get(db_file)
        if c is None:
            c = sqlite3.connect(db_file)          # exactly nuplan's call
            c.row_factory = sqlite3.Row
            conns[db_file] = c
            while len(conns) > MAX_CONN:
                conns.popitem(last=False)[1].close()
        else:
            conns.move_to_end(db_file)
        return c

    def _run(query_text, params, db_file):
        cur = _conn(db_file).cursor()
        try:
            cur.execute(query_text, params)       # exactly nuplan's call
            for row in cur:
                yield row
        finally:
            cur.close()

    def _cost(v):
        return max(len(v), 1) if isinstance(v, tuple) else 1

    def _store(k, rows):
        cache[k] = rows
        size[0] += _cost(rows)
        while size[0] > MAX_ROWS and len(cache) > 1:
            _, old = cache.popitem(last=False)
            size[0] -= _cost(old)
            STATS["evict"] += 1

    def _check(query_text, params, db_file, rows):
        fresh = list(orig_many(query_text, params, db_file))
        same = len(fresh) == len(rows) and all(
            tuple(a) == tuple(b) and a.keys() == b.keys() for a, b in zip(fresh, rows))
        if not same:
            raise AssertionError(f"query cache MISMATCH on {db_file}: {query_text[:80]!r}")
        STATS["verified"] += 1

    def execute_many(query_text, query_parameters, db_file):
        k = _key(query_text, query_parameters, db_file)
        if k is None:
            STATS["bypass"] += 1
            yield from orig_many(query_text, query_parameters, db_file)
            return
        hit = cache.get(k)
        if hit is not None:
            cache.move_to_end(k)
            STATS["hit"] += 1
            if verify:
                _check(query_text, query_parameters, db_file, hit)
            yield from hit
            return
        STATS["miss"] += 1
        rows = []
        STATS["partial"] += 1                    # undone below only if the caller exhausts us
        for r in _run(query_text, query_parameters, db_file):
            rows.append(r)
            yield r
        STATS["partial"] -= 1
        if verify:                               # the persistent-connection path is checked too
            _check(query_text, query_parameters, db_file, rows)
        _store(k, tuple(rows))

    def execute_one(query_text, query_parameters, db_file):
        k = _key(query_text, query_parameters, db_file)
        if k is None:
            STATS["bypass"] += 1
            return orig_one(query_text, query_parameters, db_file)
        k = ("__one__",) + k
        hit = cache.get(k, _MISSING)
        if hit is not _MISSING:
            cache.move_to_end(k)
            STATS["one_hit"] += 1
            if verify:
                fresh = orig_one(query_text, query_parameters, db_file)
                if (fresh is None) != (hit is None) or (
                        fresh is not None and (tuple(fresh) != tuple(hit) or fresh.keys() != hit.keys())):
                    raise AssertionError(f"query cache MISMATCH (one) on {db_file}: {query_text[:80]!r}")
                STATS["verified"] += 1
            return hit
        STATS["one_miss"] += 1
        it = _run(query_text, query_parameters, db_file)
        try:
            result = next(it, None)
            if result is not None and next(it, None) is not None:
                raise RuntimeError("execute_one query returned multiple rows.")   # nuplan's contract
        finally:
            it.close()
        if verify:
            fresh = orig_one(query_text, query_parameters, db_file)
            if (fresh is None) != (result is None) or (
                    fresh is not None and (tuple(fresh) != tuple(result) or fresh.keys() != result.keys())):
                raise AssertionError(f"query cache MISMATCH (one, miss) on {db_file}: {query_text[:80]!r}")
            STATS["verified"] += 1
        _store(k, result)
        return result

    execute_many.__wrapped__ = orig_many
    execute_one.__wrapped__ = orig_one
    # nuplan_scenario_queries imported the names at module load: patch its globals (every query
    # helper looks them up there) and the defining module for anything imported later
    Q.execute_many, Q.execute_one = execute_many, execute_one
    S.execute_many, S.execute_one = execute_many, execute_one
    _INSTALLED = True
    return True


def summary() -> str:
    s = STATS
    n = s["hit"] + s["miss"]
    m = s["one_hit"] + s["one_miss"]
    return (f"query cache: many {100.0 * s['hit'] / max(n, 1):.1f} % hits ({s['hit']:,}/{n:,}), "
            f"one {100.0 * s['one_hit'] / max(m, 1):.1f} % ({s['one_hit']:,}/{m:,}), "
            f"evicted {s['evict']:,}, bypass {s['bypass']:,}, partial {s['partial']:,}, "
            f"verified {s['verified']:,}")
