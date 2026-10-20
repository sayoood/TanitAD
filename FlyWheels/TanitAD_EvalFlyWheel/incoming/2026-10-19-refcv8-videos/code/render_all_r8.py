"""render_all_r8.py -- render the WHOLE eval139 set of one banked tree, one episode after another (CPU, ONE render process, resumable).

    set OMP_NUM_THREADS=4  &  CUDA_VISIBLE_DEVICES=
    python render_all_r8.py --tree step48852ema_unc --out-dir E:/refcv8_eval_kit/videos/final48852 [--deploy-npz <DEPLOY8_POS_*.npz>] [--heads-dir <dir>] [--episodes 0,1,2]
    python render_all_r8.py --tree step48852ema_pf  --out-dir E:/refcv8_eval_kit/videos/final48852_pf [--heads-dir <pf heads dir>]     (THE binding read, 2026-10-20)

* PER-FRAME tree (*_pf): the NAV label is the token the roll WROTE as fed (+ "clip nav (not fed)"); the RT layer is the REGISTERED per-frame table or OFF
  (on-screen note) -- --deploy-npz auto never falls back to the clip-fed 30k file on a pf tree; a clip-fed / unrecorded heads dump is REFUSED per episode.

* An episode whose mp4 already exists (and is non-empty) is SKIPPED, so a crash / reboot / the overnight CPU re-roll can interrupt it and it resumes.
* Before every episode it waits until free commit >= --min-free-gb (default 5): the dev box is shared with the battery chain and the CPU re-rolls.
* Never touches the GPU lock and never starts a GPU job.  Writes `<out-dir>/RENDER_LOG.jsonl` (one JSON line per episode: frames, wall, controls, map-GT flag).
* ~18 s wall per episode on this box (MEASURED, ep114 201 frames, 4 threads) -> ~45 min for 139 episodes.  Disk: ~8-14 MB per episode (CRF 18) -> ~1.5 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_data as D  # noqa: E402
import render_r8_video as R  # noqa: E402


def free_commit_gb() -> float:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)"],
                         capture_output=True, text=True).stdout.strip().replace(",", ".")
    try:
        return float(out)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deploy-npz", default=D.AUTO, help="auto: per-clip tree -> 30k EMA (labelled); *_pf tree -> registered per-frame table or RT OFF")
    ap.add_argument("--arch-root", default=str(D.ARCH))
    ap.add_argument("--pf-registered", default=str(D.PF_REGISTERED))
    ap.add_argument("--pf-pkg", default=str(D.PF_PKG))
    ap.add_argument("--heads-dir", default=None)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    B = D.EvalBank(a.tree, deploy_npz=a.deploy_npz, arch_root=a.arch_root, pf_registered=a.pf_registered, pf_pkg=a.pf_pkg)
    print(json.dumps({"tree": a.tree, "nav_feed": B.feed_mode, "rt": B.rt_note or f"ON ({B.deploy_label})", "rt_provenance": B.rt.get("provenance")}), flush=True)
    eps = B.indices() if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    log = out / "RENDER_LOG.jsonl"
    for ci in eps:
        mp4 = out / f"ep{ci:03d}.mp4"
        if mp4.exists() and mp4.stat().st_size > 0:
            continue
        waited = 0
        while True:
            fg = free_commit_gb()
            if fg != fg or fg >= a.min_free_gb:               # NaN (probe failed) is treated as 'go' but recorded
                break
            time.sleep(30)
            waited += 30
        t0 = time.time()
        try:
            part = out / f"ep{ci:03d}.part.mp4"                  # ffmpeg picks the container from the extension: keep .mp4 last
            res = R.render_episode(B, ci, str(part), a.tree, crf=a.crf, progress=False, heads_dir=a.heads_dir)
            os.replace(str(part), str(mp4))
            rec = {"episode": ci, "ok": True, **{k: res[k] for k in ("frames", "seconds", "map_gt", "windows", "controls", "nav_feed",
                                                                      "n_windows_fed_token_differs_from_clip", "rt", "heads_feed")}}
        except SystemExit as e:
            rec = {"episode": ci, "ok": False, "error": str(e)[:300]}
        except Exception as e:                                # one bad episode must not stop the set; it is LOGGED, never swallowed
            rec = {"episode": ci, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        rec.update(wall_s=round(time.time() - t0, 1), waited_s=waited, free_commit_gb=fg)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    done = sum(1 for ci in eps if (out / f"ep{ci:03d}.mp4").exists())
    print(f"[render_all] {done}/{len(eps)} episodes present in {out}")


if __name__ == "__main__":
    main()
