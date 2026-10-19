"""make_windows_navfeed.py -- the PRE-REGISTERED window selection for the nav-feed experiment (PREREG_NAV_FEED.md section 3).

CPU, numpy only, reads two BANKED inputs, writes ``raw/windows_navfeed.json`` (episode_index + sha12 keyed: no raw clip id,
no clip-id prefix, no stable_episode_id ever leaves this script).

  inputs   * the archived step40000ema_unc seed-0 dump (the windows the battery rolled; its ``decisions.nav_cmd`` IS the clip-level
             token that was fed):  C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s0
           * the v9 EVAL release the model was trained against (md5 6b5c7f207cffc3b7eb3cd527fd433599):
             E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz

  rule     MISMATCH = every window whose FED (clip-level) token != the v9 per-frame token at (sid, k = ws + 2)   [1,157 expected, NAV_TRACE sec 3]
           CONTROL  = 300 windows whose fed token == the v9 per-frame token, STRATIFIED by that token (100 FOLLOW, 100 LEFT, 100 RIGHT),
                      drawn with numpy.random.default_rng(20261020): for token c in (0, 1, 2), in that order, from the candidates sorted
                      by (episode_index, ws), ``rng.choice(n_candidates, size=100, replace=False)``; the union is then sorted.

The join key is the trace agent's, with the positive per-window assertion ``row__k == k`` (never an offset assumption):
k = ws + 2, and ``ws + 2`` is stated independently by the dump manifest's ``grid.ws_is`` (RAW frame = ws + 2).

usage   python make_windows_navfeed.py [--out raw/windows_navfeed.json] [--check-expected]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ARCHIVE_DUMP = "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s0"
V9_EVAL = "E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz"
V9_EVAL_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"
RAW_OFFSET = 2
CONTROL_SEED = 20261020
CONTROL_PER_TOKEN = 100
TOKEN_NAME = {0: "FOLLOW", 1: "LEFT", 2: "RIGHT"}
# LITERALS from NAV_TRACE.md sec 3 (MEASURED by the trace agent on the same two inputs); never derived from this script
EXPECT = {"n_windows": 4754, "n_episodes": 139, "n_mismatch": 1157,
          "matched_by_v9_token": {0: 2909, 1: 158, 2: 530},
          "confusion_v9_row_fed_col": [[2909, 280, 739], [26, 158, 29], [76, 7, 530]]}


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def v9_token_lookup(z):
    sid, row0, nr, k0 = z["clip__sid"], z["clip__row0"], z["clip__n_rows"], z["clip__k0"]
    k, tok = z["row__k"].astype(np.int64), z["row__nav_token"]
    by_sid = {int(s): i for i, s in enumerate(sid)}

    def at(s, kk):
        i = by_sid[int(s)]
        j = int(kk) - int(k0[i])
        if j < 0 or j >= int(nr[i]):
            raise SystemExit(f"[make_windows_navfeed] frame {kk} outside clip sid-index {i}")
        r = int(row0[i]) + j
        if int(k[r]) != int(kk):                      # positive key assertion
            raise SystemExit(f"[make_windows_navfeed] release row {r} holds k={int(k[r])}, expected {kk}")
        if not np.isfinite(tok[r]):
            raise SystemExit("[make_windows_navfeed] non-finite v9 nav token")
        return int(tok[r])
    return at, by_sid


def collect(dump: str, v9: str):
    z = np.load(v9, allow_pickle=False)
    at, by_sid = v9_token_lookup(z)
    man = json.load(open(Path(dump) / "manifest.json", encoding="utf-8"))
    ws_is = man["grid"]["ws_is"]
    if f"ws + {RAW_OFFSET}" not in ws_is:
        raise SystemExit(f"[make_windows_navfeed] the dump manifest does not state RAW = ws + {RAW_OFFSET}: {ws_is!r}")
    rows = []          # (episode_index, ws, fed, v9)
    sha = {}
    for e in man["episodes"]:
        fi, s = int(e["file_index"]), int(e["episode_id"])
        dec = np.load(Path(dump) / "decisions" / f"ep{fi:03d}.npz", allow_pickle=False)
        epz = np.load(Path(dump) / f"ep{fi:03d}.npz", allow_pickle=False)
        ws = dec["ws"].astype(np.int64)
        assert np.array_equal(epz["ws"].astype(np.int64), ws)
        fed = dec["nav_cmd"].astype(np.int64)
        sha[int(e["episode_index"])] = str(z["clip__sha12"][by_sid[s]])
        # independent check of the identity: sha12 of the manifest clip id == the release's sha12 for that sid
        assert hashlib.sha256(str(e["clip_id"]).encode()).hexdigest()[:12] == sha[int(e["episode_index"])]
        for w, f in zip(ws, fed):
            rows.append((int(e["episode_index"]), int(w), int(f), at(s, int(w) + RAW_OFFSET)))
    return rows, sha, man


def select(rows):
    a = np.array(rows, dtype=np.int64)                # columns: ep, ws, fed, v9
    conf = np.zeros((3, 3), dtype=np.int64)
    for _e, _w, f, v in a:
        conf[v, f] += 1
    mism = a[a[:, 2] != a[:, 3]]
    matched = a[a[:, 2] == a[:, 3]]
    rng = np.random.default_rng(CONTROL_SEED)
    ctrl = []
    cand_n = {}
    for c in (0, 1, 2):
        cand = matched[matched[:, 3] == c]
        cand = cand[np.lexsort((cand[:, 1], cand[:, 0]))]          # sorted by (episode_index, ws)
        cand_n[c] = int(len(cand))
        if len(cand) < CONTROL_PER_TOKEN:
            raise SystemExit(f"[make_windows_navfeed] only {len(cand)} matched windows with v9 token {c}")
        pick = rng.choice(len(cand), size=CONTROL_PER_TOKEN, replace=False)
        ctrl.append(cand[pick])
    ctrl = np.concatenate(ctrl)
    ctrl = ctrl[np.lexsort((ctrl[:, 1], ctrl[:, 0]))]
    mism = mism[np.lexsort((mism[:, 1], mism[:, 0]))]
    return a, mism, ctrl, conf, cand_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=ARCHIVE_DUMP)
    ap.add_argument("--v9", default=V9_EVAL)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "raw" / "windows_navfeed.json"))
    ap.add_argument("--check-expected", action="store_true", help="exit non-zero unless every count equals the NAV_TRACE literal")
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args()
    if not a.skip_md5:
        got = md5_file(a.v9)
        if got != V9_EVAL_MD5:
            raise SystemExit(f"[make_windows_navfeed] v9 md5 {got} != {V9_EVAL_MD5}")
    rows, sha, man = collect(a.dump, a.v9)
    allw, mism, ctrl, conf, cand_n = select(rows)
    ok = {"n_windows": len(allw) == EXPECT["n_windows"],
          "n_episodes": len(sha) == EXPECT["n_episodes"],
          "n_mismatch": len(mism) == EXPECT["n_mismatch"],
          "matched_by_v9_token": {c: cand_n[c] for c in (0, 1, 2)} == EXPECT["matched_by_v9_token"],
          "confusion": conf.tolist() == EXPECT["confusion_v9_row_fed_col"]}
    eps = {}
    for key, arr in (("mismatch", mism), ("control", ctrl)):
        for e, w, f, v in arr:
            d = eps.setdefault(str(int(e)), {"episode_index": int(e), "sha12": sha[int(e)], "mismatch_ws": [], "control_ws": [],
                                             "mismatch_fed": [], "mismatch_v9": [], "control_v9": []})
            d[f"{key}_ws"].append(int(w))
            if key == "mismatch":
                d["mismatch_fed"].append(int(f))
                d["mismatch_v9"].append(int(v))
            else:
                d["control_v9"].append(int(v))
    both = {(int(e), int(w)) for e, w, _f, _v in mism} & {(int(e), int(w)) for e, w, _f, _v in ctrl}
    out = {
        "schema": "navfeed-windows/1",
        "rule": __doc__.split("rule     ")[1].split("The join key")[0].strip(),
        "control_seed": CONTROL_SEED, "control_per_token": CONTROL_PER_TOKEN, "raw_offset": RAW_OFFSET,
        "inputs": {"dump": a.dump, "dump_manifest_md5": md5_file(str(Path(a.dump) / "manifest.json")),
                   "v9_eval": a.v9, "v9_eval_md5": V9_EVAL_MD5},
        "counts": {"n_windows_source": int(len(allw)), "n_episodes_source": len(sha),
                   "n_mismatch": int(len(mism)), "n_mismatch_episodes": len({int(e) for e in mism[:, 0]}),
                   "n_control": int(len(ctrl)), "n_control_episodes": len({int(e) for e in ctrl[:, 0]}),
                   "mismatch_by_v9_token": {TOKEN_NAME[c]: int((mism[:, 3] == c).sum()) for c in (0, 1, 2)},
                   "mismatch_by_fed_token": {TOKEN_NAME[c]: int((mism[:, 2] == c).sum()) for c in (0, 1, 2)},
                   "control_by_token": {TOKEN_NAME[c]: int((ctrl[:, 3] == c).sum()) for c in (0, 1, 2)},
                   "matched_candidates_by_token": {TOKEN_NAME[c]: cand_n[c] for c in (0, 1, 2)},
                   "confusion_v9_row_fed_col": conf.tolist(), "mismatch_control_overlap": len(both)},
        "matches_nav_trace_literals": ok,
        "episodes": [eps[k] for k in sorted(eps, key=int)],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"counts": out["counts"], "matches_nav_trace_literals": ok, "out": a.out,
                      "out_sha256": sha256_file(a.out)}, indent=1))
    bad = (not all(v for v in ok.values())) or len(both) != 0
    if a.check_expected and bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
