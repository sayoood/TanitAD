#!/usr/bin/env python3
"""nt_tokens.py -- H-NT-DECIDE-1 arm A1's training token list ``TOKENS_A1.json`` (E1 owns it).  numpy + stdlib; any python.

Universe: the OFFICIAL navtrain scene filter (navsim v1.1 tree 3e8291b, sha256 c37fea56...), 103,288 tokens / 1,192 logs.
Held out BY LOG (PREREG_NT_DECIDE section 2; consecutive 2 Hz tokens share 13 of 14 frames, so excluding by token is not enough):
  * all 38 selection-scan logs = ``NAVTRAIN_SAMPLE_TOKENS.json["per_log"][*]["log"]`` (sample sha 9dd68698...);
  * every navtest log = ``navtest.yaml`` ``log_names``.
Expected (DESIGN section 3, MEASURED arithmetic): 99,824 tokens / 1,154 logs.

Every token is located in its OpenScene trainval log pickle (``navsim_logs/trainval/<log>.pkl``, read with the r8ns agent's
allow-listed ``navtrain_common.load_log_pickle`` -- IMPORTED, not copied) and must own a devkit scene window
(``navtrain_common.scene_window``: 4 history + 10 future frames, roadblock ids at t0).  Any token not found exactly once, or
without a window, is a REFUSAL (no file written).

The output carries the fields ``export_navtrain_inputs.py`` (``load_sample``) requires -- ``status: FIXED``, ``tokens``, ``count``,
``sha256_sorted_tokens``, ``token_log`` -- so the r8ns agent's devkit exporter runs on it UNCHANGED.

Exclusion proof (K3, positive assertion): ``train_logs & selection_logs == set()`` and ``train_logs & navtest_logs == set()`` are
evaluated and printed with both set sizes; token-level intersections with the sample tokens and the navtest tokens likewise.
``--self-test`` runs the deliberate-regression arm: adding ONE selection log to the train log set must make the guard REFUSE.

Markers: ``ZZTOK_DONE tokens=<n> logs=<m> sha=<sha16>`` / ``ZZTOK_REFUSED_<why>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

R8NS_NAVTRAIN_CODE = "/home/nvidia/r8ns/code/navtrain"
V11 = "/home/nvidia/navsim-thor/src/navsim-v11"
FILTER_DIR = f"{V11}/navsim/planning/script/config/common/train_test_split/scene_filter"
NAVTEST_FILTER_SHA256 = "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd"   # MEASURED 2026-10-21 on Thor
SAMPLE_SHA256_SORTED = "9dd68698659ec73c131b677120c45fa6f385ea058d4a055f7378f4c13f46d707"
EXPECTED = {"n_tokens": 99824, "n_logs": 1154, "n_selection_logs": 38}


class Refused(Exception):
    pass


def exclusion_guard(train_logs: set, selection_logs: set, navtest_logs: set) -> dict:
    """POSITIVE assertion: both intersections are computed and must be the EMPTY SET.  Raises :class:`Refused` otherwise."""
    i_sel = set(train_logs) & set(selection_logs)
    i_nt = set(train_logs) & set(navtest_logs)
    rep = {"n_train_logs": len(train_logs), "n_selection_logs": len(selection_logs), "n_navtest_logs": len(navtest_logs),
           "train_and_selection": sorted(i_sel), "train_and_navtest": sorted(i_nt),
           "train_and_selection_is_empty_set": i_sel == set(), "train_and_navtest_is_empty_set": i_nt == set()}
    if i_sel != set() or i_nt != set():
        raise Refused(f"exclusion_violated sel={len(i_sel)} navtest={len(i_nt)}")
    return rep


def self_test(train_logs: set, selection_logs: set, navtest_logs: set) -> dict:
    """K3 deliberate-regression arm: one selection log (and, separately, one navtest log) added -> the guard MUST refuse."""
    out = {}
    for name, extra in (("add_one_selection_log", sorted(selection_logs)[0]), ("add_one_navtest_log", sorted(navtest_logs)[0])):
        try:
            exclusion_guard(set(train_logs) | {extra}, selection_logs, navtest_logs)
            out[name] = "NOT_REFUSED"          # the guard is blind -> the caller refuses the whole build
        except Refused:
            out[name] = "REFUSED"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default="/home/nvidia/r8ns/navtrain/navsim_logs/trainval")
    ap.add_argument("--sample", default="/home/nvidia/r8ns/navtrain/NAVTRAIN_SAMPLE_TOKENS.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-out", required=True, help="per-token log position index (json.gz) for the E1 writer")
    a = ap.parse_args(argv)
    sys.path.insert(0, R8NS_NAVTRAIN_CODE)
    import navtrain_common as N            # the r8ns agent's helpers, imported unchanged
    t0 = time.time()
    try:
        nt = N.parse_scene_filter(f"{FILTER_DIR}/navtrain.yaml")
        ts = N.parse_scene_filter(f"{FILTER_DIR}/navtest.yaml")
        if nt["file_sha256"] != N.NAVTRAIN_FILTER_SHA256:
            raise Refused("navtrain_filter_sha256")
        if ts["file_sha256"] != NAVTEST_FILTER_SHA256:
            raise Refused("navtest_filter_sha256")
        hdr = nt["header"]
        sample = json.load(open(a.sample, encoding="utf-8"))
        if sample.get("sha256_sorted_tokens") != SAMPLE_SHA256_SORTED or N.sha256_tokens(sample["tokens"]) != SAMPLE_SHA256_SORTED:
            raise Refused("sample_sha256")
        selection_logs = {r["log"] for r in sample["per_log"]}
        if len(selection_logs) != EXPECTED["n_selection_logs"]:
            raise Refused(f"selection_logs_{len(selection_logs)}")
        navtest_logs = set(ts["log_names"])
        navtest_tokens = set(ts["tokens"])
        nav_tokens = set(nt["tokens"])
        if len(nav_tokens) != len(nt["tokens"]):
            raise Refused("navtrain_filter_duplicate_tokens")
        # ---- locate every navtrain token in its log (scan every navtrain log pickle once) ----
        where, n_frames_log, dup = {}, {}, []
        for li, ln in enumerate(sorted(set(nt["log_names"]))):
            frames, stub = N.load_log_pickle(os.path.join(a.logs_root, f"{ln}.pkl"))
            if stub:
                raise Refused(f"unpickler_stubbed_{ln}")
            n_frames_log[ln] = len(frames)
            for pos, f in enumerate(frames):
                tok = f["token"]
                if tok in nav_tokens:
                    if tok in where:
                        dup.append(tok)
                    win = N.scene_window(frames, pos, hdr["num_history_frames"], hdr["num_future_frames"], hdr["has_route"])
                    where[tok] = {"log": ln, "pos": pos, "n_log_frames": len(frames), "window_ok": win is not None,
                                  "next_is_navtrain_token": (pos + 1 < len(frames) and frames[pos + 1]["token"] in nav_tokens),
                                  "next_token": frames[pos + 1]["token"] if pos + 1 < len(frames) else None}
            if (li + 1) % 100 == 0:
                print(f"[tokens] {li + 1} logs scanned, {len(where)} tokens located, {time.time() - t0:.0f} s", flush=True)
        if dup:
            raise Refused(f"token_in_two_places_{len(dup)}")
        missing = nav_tokens - set(where)
        if missing:
            raise Refused(f"navtrain_tokens_not_found_{len(missing)}")
        no_win = sorted(t for t, w in where.items() if not w["window_ok"])
        if no_win:
            raise Refused(f"tokens_without_devkit_window_{len(no_win)}")
        sel_logs_in_filter = selection_logs & set(nt["log_names"])
        # ---- the A1 pool: exclusion BY LOG ----
        pool = sorted(t for t, w in where.items() if w["log"] not in selection_logs and w["log"] not in navtest_logs)
        train_logs = {where[t]["log"] for t in pool}
        guard = exclusion_guard(train_logs, selection_logs, navtest_logs)
        st = self_test(train_logs, selection_logs, navtest_logs)
        if any(v != "REFUSED" for v in st.values()):
            raise Refused("guard_self_test_blind")
        tok_sel = set(pool) & set(sample["tokens"])
        tok_nt = set(pool) & navtest_tokens
        if tok_sel != set() or tok_nt != set():
            raise Refused("token_level_overlap")
        held_sel_tokens = sum(1 for t, w in where.items() if w["log"] in selection_logs)
        print(f"[tokens] K3 exclusion: |train_logs|={len(train_logs)}  |selection_logs|={len(selection_logs)}  "
              f"|navtest_logs|={len(navtest_logs)}  train&selection={sorted(guard['train_and_selection'])}  "
              f"train&navtest={sorted(guard['train_and_navtest'])}  (both == set(): "
              f"{guard['train_and_selection_is_empty_set'] and guard['train_and_navtest_is_empty_set']})", flush=True)
        print(f"[tokens] token-level: |pool|={len(pool)} & |sample tokens|={len(sample['tokens'])} -> {len(tok_sel)};  "
              f"& |navtest tokens|={len(navtest_tokens)} -> {len(tok_nt)};  self-test {st}", flush=True)
        sha = N.sha256_tokens(pool)
        doc = {"what": "H-NT-DECIDE-1 arm A1 training token list (E1, DataFlyWheel)", "status": "FIXED",
               "rule": "navtrain filter tokens, minus every token of the 38 selection-scan logs and of every navtest log (by LOG)",
               "count": len(pool), "n_logs": len(train_logs), "sha256_sorted_tokens": sha,
               "expected": EXPECTED, "matches_expected": len(pool) == EXPECTED["n_tokens"] and len(train_logs) == EXPECTED["n_logs"],
               "tokens": pool, "token_log": {t: where[t]["log"] for t in pool}, "log_names": sorted(train_logs),
               "universe": {"navtrain_filter": f"{FILTER_DIR}/navtrain.yaml", "navtrain_filter_sha256": nt["file_sha256"],
                            "navtrain_header": hdr, "n_navtrain_tokens": len(nav_tokens), "n_navtrain_logs": len(set(nt["log_names"])),
                            "navtest_filter": f"{FILTER_DIR}/navtest.yaml", "navtest_filter_sha256": ts["file_sha256"],
                            "n_navtest_tokens": len(navtest_tokens), "n_navtest_logs": len(navtest_logs)},
               "held_out": {"selection_logs": sorted(selection_logs), "n_selection_logs": len(selection_logs),
                            "n_selection_logs_in_navtrain_filter": len(sel_logs_in_filter),
                            "n_navtrain_tokens_in_selection_logs": held_sel_tokens,
                            "selection_sample_sha256_sorted_tokens": SAMPLE_SHA256_SORTED,
                            "navtest_logs": "navtest.yaml log_names (see universe)"},
               "K3_exclusion": {**guard, "token_level": {"pool_and_sample_tokens": len(tok_sel), "pool_and_navtest_tokens": len(tok_nt)},
                                "deliberate_regression": st},
               "map_locations_note": "not read here (the exporter records map_name per token)",
               "tool": "nt_tokens.py", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=0)
        os.replace(tmp, a.out)
        import gzip
        idx = {t: where[t] for t in pool}
        with gzip.open(a.index_out + ".tmp", "wt", encoding="utf-8") as fh:
            json.dump({"tokens_sha256_sorted": sha, "index": idx}, fh)
        os.replace(a.index_out + ".tmp", a.index_out)
        print(f"ZZTOK_DONE tokens={len(pool)} logs={len(train_logs)} sha={sha[:16]} matches_expected={doc['matches_expected']} "
              f"wall={time.time() - t0:.0f}s", flush=True)
        return 0
    except Refused as r:
        print(f"ZZTOK_REFUSED_{r}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
