"""STEP 2 (DEV BOX) — drive the 86-clip re-run in resumable chunks.

⛔ THE DEV BOX IS THE RESUME AUTHORITY, NOT THE VM AND NOT HF. Free-Colab reclaims
the T4 without warning (MEASURED three times during the 115 run), and this task may
not push to HuggingFace, so neither of the two places p3 could lean on is available.
The done-set therefore lives here, is recomputed BY CONTENT before every chunk with
`s2_lab_lib.content_census_local` — the SAME predicate the far-side census applies,
because `census_records` is now shared — and a reclaim costs one chunk.

The predicate, unchanged from the 115 run:
    liveness control present AND zero error entries
    AND schema_version >= 2 AND engine.confidence_threshold == 0.25

⚠️ A CHUNK THAT ADDS NO COMPLETE CLIPS IS A STALL, NOT PROGRESS. The loop counts
them and ABORTS after `--max-stalls`, rather than spinning until the timeout and
reporting a partial run as though it had merely been interrupted.

usage:
  python f2_drive86.py --aug120 <dir> --v2-dir <dir> --bank <dir> \
                       --session tanitad-floor86 [--chunk 22] [--ship]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
COLAB = r"C:\Users\Admin\venvs\colab\Scripts\colab.exe"
SHIM = os.path.join(REPO, "colab", "win_shims")
HERE = os.path.dirname(os.path.abspath(__file__))
SHIP = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                    "Implementation", "incoming", "2026-08-16-sam3-extraction-v2",
                    "code", "ship.py")
sys.path.insert(0, os.path.join(REPO, "colab"))
import s2_lab_lib as L                                            # noqa: E402

CONF, SCHEMA = 0.25, 2


def colab(*args, timeout=None):
    """⚠️ PYTHONUTF8=1 + MSYS_NO_PATHCONV=1 are REQUIRED, not hygiene: colab-cli
    0.6.0 opens the script with the locale codec (cp1252 here) and any file
    carrying a ⛔/⚠️ dies before a line reaches the VM; and MSYS rewrites a
    POSIX remote path into `C:/Program Files/Git/content/…`, which the VM's
    contents API answers with a 500."""
    env = dict(os.environ, PYTHONPATH=SHIM, MSYS_NO_PATHCONV="1",
               PYTHONUTF8="1")
    p = subprocess.run([COLAB, *args], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p


def census(bank, want):
    return L.content_census_local(bank, want=set(want),
                                  require_schema=SCHEMA, require_conf=CONF)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug120", required=True)
    ap.add_argument("--v2-dir", required=True, help="the banked 115")
    ap.add_argument("--bank", required=True, help="local durable bank for the 86")
    ap.add_argument("--session", default="tanitad-floor86")
    ap.add_argument("--chunk", type=int, default=22)
    ap.add_argument("--exec-timeout", type=float, default=3000.0)
    ap.add_argument("--max-stalls", type=int, default=2)
    ap.add_argument("--ship", action="store_true", help="ship the closure first")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw",
                                                  "f2_run86.json"))
    a = ap.parse_args(argv)
    os.makedirs(a.bank, exist_ok=True)

    # ---- the population, recomputed rather than inherited ------------------ #
    cohort = {c["clip_id"] for c in json.load(
        open(os.path.join(a.aug120, "merged", "ph0_v2.json"),
             encoding="utf-8"))["clips"]}
    have_v2 = {os.path.basename(p)[:-5]
               for p in glob.glob(os.path.join(a.v2_dir, "*.json"))}
    the86 = sorted(cohort - have_v2)
    assert len(cohort) == 201, f"cohort moved: {len(cohort)}"
    assert len(have_v2) == 115, f"v2 leg moved: {len(have_v2)}"
    assert len(the86) == 86, f"residual is {len(the86)}, not 86"
    assert not (set(the86) & have_v2), "residual overlaps the v2 leg"
    print(f"[pop] cohort {len(cohort)} · v2 {len(have_v2)} · residual "
          f"{len(the86)} (recomputed, not read from a manifest)")

    if a.ship:
        print(f"[ship] {SHIP} -> {a.session}")
        rc = subprocess.run([sys.executable, SHIP, a.session],
                            env=dict(os.environ, PYTHONUTF8="1")).returncode
        if rc:
            print(f"SHIP_FAILED rc={rc}")
            return rc

    log = []
    stalls = 0
    t_run = time.time()
    while True:
        cen = census(a.bank, the86)
        done = set(cen["complete_clips"])
        todo = [c for c in the86 if c not in done]
        print(f"[resume] complete {len(done)}/86 · todo {len(todo)} · "
              f"det {cen['n_det_total']} scene {cen['n_scene_det_total']} "
              f"errors {sum(cen['error_census'].values())}")
        if not todo:
            print("F2_ALL_COMPLETE")
            break
        chunk = todo[:a.chunk]
        spec = os.path.join(tempfile.gettempdir(), "todo86.json")
        with open(spec, "w", encoding="utf-8") as fh:
            json.dump({"todo": chunk, "chunk": len(log)}, fh)
        r = colab("upload", "-s", a.session, spec, "/content/todo86.json")
        if r.returncode:
            print("UPLOAD_FAILED", (r.stderr or r.stdout)[-800:])
            return r.returncode
        print(f"[exec] chunk {len(log)} · {len(chunk)} clips · "
              f"timeout {a.exec_timeout:.0f}s", flush=True)
        t0 = time.time()
        r = colab("exec", "-s", a.session, "-f",
                  os.path.join(HERE, "f1_run86.py"),
                  "--timeout", str(a.exec_timeout),
                  timeout=a.exec_timeout + 600)
        tail = ((r.stdout or "") + (r.stderr or ""))
        print(tail[-3000:])
        # ⚠️ THE CLIENT'S EXIT IS NOT THE RUN'S VERDICT. A killed `colab exec`
        # client does NOT kill the kernel (MEASURED, 115 run §6.2) and a live
        # kernel keeps banking. So the pull happens either way and the CENSUS
        # decides — never the return code.
        pulled = 0
        tgz = os.path.join(tempfile.gettempdir(), "out86.tgz")
        if os.path.exists(tgz):
            os.remove(tgz)
        rd = colab("download", "-s", a.session, "/content/out86.tgz", tgz)
        if rd.returncode or not os.path.exists(tgz):
            print("[pull] FAILED", (rd.stderr or rd.stdout)[-500:])
        else:
            with tarfile.open(tgz) as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = os.path.basename(m.name)
                    if not name.endswith(".json"):
                        continue
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    with open(os.path.join(a.bank, name), "wb") as fh:
                        fh.write(src.read())
                    pulled += 1
            print(f"[pull] {pulled} json from {os.path.getsize(tgz)} B tar")
        cen2 = census(a.bank, the86)
        gained = len(set(cen2["complete_clips"])) - len(done)
        log.append({"chunk": len(log), "requested": len(chunk),
                    "wall_s": round(time.time() - t0, 1),
                    "rc": r.returncode, "pulled_json": pulled,
                    "complete_after": cen2["n_complete"], "gained": gained})
        print(f"[chunk {len(log)-1}] +{gained} complete -> "
              f"{cen2['n_complete']}/86 in {time.time()-t0:.0f}s")
        if gained <= 0:
            stalls += 1
            print(f"⚠️ STALL {stalls}/{a.max_stalls} — chunk added no complete "
                  f"clips (rc={r.returncode})")
            if stalls >= a.max_stalls:
                print("F2_STALLED")
                break
        else:
            stalls = 0

    cen = census(a.bank, the86)
    zero = cen["zero_det_clips"]
    out = {
        "class": "MEASURED", "n_target": 86,
        "wall_s_total": round(time.time() - t_run, 1),
        "chunks": log, "pushed_to_hf": False,
        "census": {k: v for k, v in cen.items()
                   if k not in ("zero_det_clips", "complete_clips")},
        "zero_split": {
            "empty_scene_control_live":
                sum(1 for z in zero if z["liveness_live"]),
            "dead_control": sum(1 for z in zero if not z["liveness_live"]),
            "clips": zero},
        "residual": sorted(set(the86) - set(cen["complete_clips"])),
        "PASS": bool(cen["pass_"] and cen["n_complete"] == 86),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[f2] complete {cen['n_complete']}/86 · det {cen['n_det_total']} · "
          f"scene {cen['n_scene_det_total']} · live {cen['liveness_live']} · "
          f"errors {sum(cen['error_census'].values())} · PASS {out['PASS']}")
    print("F2_PASS" if out["PASS"] else "F2_INCOMPLETE")
    print("F2_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
