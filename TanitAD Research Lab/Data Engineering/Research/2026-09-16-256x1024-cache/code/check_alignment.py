"""G4 -- the rebuilt payload is still on the SAME time grid as the labels.

Raising the azimuth resolution must not move a single row. Two independent
banked artifacts pin the grid and both are checked EXACTLY (no tolerance):

1. The SAM3 map GT (``D:/Projects/TanitAD-artifacts/sam3-maps-eval/``). Its own
   meta states the grid it was built on, verbatim:
       "v2ep EPISODE grid: t_query = linspace(t_cam[0], t_cam[-1],
        int(span_s * 10)), frame = first camera frame at or after t_query
        (v2_compressed._resampled); axis0 = raw v2ep frame index"
   so the grid is recomputed here from the STAGED timestamps.parquet by the
   deployed code path and required to equal ``t_query_us`` / ``cam_frame_idx`` /
   ``t_img_us`` exactly.

2. The banked bevhead token index
   ``C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/index.npz``:
   per clip, ``max(raw_frame)+1 == n_frames`` and
   ``stacked_row == raw_frame - (n_stack-1)``.

⛔ The grid is recomputed with ``pandas``/``numpy`` exactly as
``v2_compressed._resampled`` spells it -- same searchsorted, same clip, same
linspace -- because a second spelling of the time grid is the same class of
defect as a second spelling of the projection.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

STACK = os.environ.get("TANITAD_STACK", r"C:/Users/Admin/tanitad-snap-20260915/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import torch                                                         # noqa: E402
from tanitad.data.physicalai import TARGET_HZ                        # noqa: E402

N_STACK = 3


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def grid_for(ts_parquet):
    """The episode time grid, spelled exactly as v2_compressed._resampled does."""
    ts = pd.read_parquet(ts_parquet)
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), 4)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)
    return t_frames, t_query, frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    R = {"n_clips": len(ids)}
    fails = []

    # ---- 1. the SAM3 map time grid -------------------------------------------
    rows, no_map = [], []
    for cid in ids:
        s = sha12(cid)
        mp = os.path.join(a.maps, f"{s}.sam3mapgt.npz")
        pp = os.path.join(a.cache, f"{cid}.v2ep.pt")
        if not os.path.exists(mp):
            no_map.append(s)
            continue
        if not os.path.exists(pp):
            fails.append(f"{s}:no_payload")
            continue
        z = np.load(mp, allow_pickle=True)
        t_frames, t_query, fidx = grid_for(
            os.path.join(a.root, "r0", "camera_front_wide",
                         f"{cid}.timestamps.parquet"))
        d = torch.load(pp, map_location="cpu", weights_only=False)
        n = int(d["jpeg_len"].shape[0])
        dq = float(np.abs(t_query - z["t_query_us"]).max()) if len(t_query) == len(
            z["t_query_us"]) else float("inf")
        di = (int(np.abs(fidx - z["cam_frame_idx"]).max())
              if len(fidx) == len(z["cam_frame_idx"]) else -1)
        dt = (int(np.abs(t_frames[fidx].astype(np.int64)
                         - z["t_img_us"]).max())
              if len(fidx) == len(z["t_img_us"]) else -1)
        ok = (n == len(z["t_img_us"]) and dq == 0.0 and di == 0 and dt == 0)
        rows.append({"clip_sha12": s, "n_frames_payload": n,
                     "n_rows_map": int(len(z["t_img_us"])),
                     "t_query_absmax": dq, "cam_frame_idx_absmax": di,
                     "t_img_us_absmax": dt, "pass": bool(ok)})
        if not ok:
            fails.append(f"{s}:map_grid")
    R["map_grid"] = {"checked": len(rows), "no_map_file": no_map,
                     "n_pass": sum(r["pass"] for r in rows), "clips": rows}
    print(f"G4.1 SAM3 map grid: {sum(r['pass'] for r in rows)}/{len(rows)} exact "
          f"(t_query, cam_frame_idx, t_img_us all absmax 0); "
          f"{len(no_map)} clip(s) have no map file", flush=True)

    # ---- 2. the banked bevhead token index -----------------------------------
    z = np.load(a.tokens, allow_pickle=True)
    tok_s12 = [str(x) for x in z["clip_sha12"]]
    want = [sha12(c) for c in ids]
    R["tokens"] = {"index": os.path.abspath(a.tokens),
                   "n_clip_sha12": len(tok_s12), "n_rows": int(len(z["raw_frame"])),
                   "set_equal_to_built": sorted(tok_s12) == sorted(want),
                   "order_equal_to_ids_file": tok_s12 == want}
    if sorted(tok_s12) != sorted(want):
        fails.append("tokens:clip_set")
    print(f"G4.2 token index: {len(tok_s12)} clip_sha12, "
          f"set==built {R['tokens']['set_equal_to_built']}, "
          f"order==ids file {R['tokens']['order_equal_to_ids_file']}", flush=True)

    ordn = np.asarray(z["clip_ordinal"]); raw = np.asarray(z["raw_frame"])
    st = np.asarray(z["stacked_row"])
    trows, bad = [], 0
    for o, s in enumerate(tok_s12):
        sel = ordn == o
        cid = next((c for c in ids if sha12(c) == s), None)
        pp = os.path.join(a.cache, f"{cid}.v2ep.pt") if cid else ""
        if not pp or not os.path.exists(pp):
            fails.append(f"{s}:no_payload_for_tokens"); bad += 1; continue
        n = int(torch.load(pp, map_location="cpu",
                           weights_only=False)["jpeg_len"].shape[0])
        r, k = raw[sel], st[sel]
        ok = (int(r.max()) + 1 == n and int(len(r)) == n - (N_STACK - 1)
              and bool((k == r - (N_STACK - 1)).all()) and int(r.min()) == N_STACK - 1)
        trows.append({"clip_sha12": s, "n_frames_payload": n, "n_token_rows": int(len(r)),
                      "raw_frame_min": int(r.min()), "raw_frame_max": int(r.max()),
                      "pass": bool(ok)})
        if not ok:
            fails.append(f"{s}:token_rows"); bad += 1
    R["tokens"]["per_clip"] = trows
    R["tokens"]["n_pass"] = sum(t["pass"] for t in trows)
    print(f"G4.3 token row index: {sum(t['pass'] for t in trows)}/{len(trows)} "
          f"clips satisfy max(raw_frame)+1 == n_frames and "
          f"stacked_row == raw_frame-{N_STACK-1}", flush=True)

    R["failures"] = fails
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ','.join(fails[:10])}"
          f" -> {a.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
