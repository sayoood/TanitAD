"""The two FIX-5 tests Thor SKIPPED (``resnet34.a1_in1k`` is not in Thor's HF cache), run on the
dev box where the checkpoint IS cached, on a byte-faithful tree of the LANDED tip.

    python run_r34_fix5_tests.py <tree> <outdir> [--boxes-done <marker>]

Master Mind 2026-09-26: "run them ON THE TIP (ab436ee) on the dev box, where resnet34 is cached,
after BOXES_DONE ... and under your floors; report pass/fail with junit. They must run before any
refcv7 launch."

Gates, in order:
1. the BOXES_DONE marker exists (the map-video render owns the box until then);
2. the CPU-chain RAM floor -- 3 consecutive samples 30 s apart read >= 7.5 GiB available, and our
   child is killed below 6.5 GiB. The unit is GiB (``ullAvailPhys / 2**30``), the same unit the
   programme's other guards print as "GB" (the NavSim scorer's ``psutil ... / 2**30``).
On a wait timeout it does NOT launch: it records NOT_RUN and exits 3 (the older runners launched
after their wait budget ran out, which is a floor violation dressed as a start).

A SKIP is not a pass: exit 0 only when BOTH named tests are PASSED in the junit XML; the verdict is
read from the junit file (the artifact), never from pytest's exit status.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite_bounded import (KILL_GB, PY, START_GAP_S, START_GB,  # noqa: E402
                               START_SAMPLES, free_gb)

TESTS = ("test_G2_GREEN_the_real_checkpoint_passes_and_fingerprints_as_measured",
         "test_G2_R3_stem_intact_layer4_RANDOM_now_goes_RED")
FILE = "tests/test_guard_blind_spots_fix5.py"
CKPT_DIR = Path("C:/Users/Admin/.cache/huggingface/hub/models--timm--resnet34.a1_in1k")
BOXES_WAIT_S, RAM_WAIT_S, RUN_TIMEOUT_S = 12 * 3600, 6 * 3600, 1800


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    tree, outdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    marker = Path(sys.argv[sys.argv.index("--boxes-done") + 1]) if "--boxes-done" in sys.argv \
        else Path("C:/Users/Admin/qland/work/mapvid/BOXES_DONE")
    outdir.mkdir(parents=True, exist_ok=True)
    summ = outdir / "r34_fix5_summary.json"
    rec: dict = {
        "what": "the two FIX-5 tests Thor skipped (resnet34.a1_in1k not cached there), on the dev box",
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "evidence_class": f"MEASURED ({socket.gethostname()}, {platform.system()}, CPU only)",
        "tree": str(tree).replace("\\", "/"),
        "base_commit": (tree / ".gate_base_commit").read_text(encoding="utf-8").strip()
        if (tree / ".gate_base_commit").exists() else None,
        "tests": {t: None for t in TESTS}, "gates": {"boxes_done_marker": str(marker)},
        "started": now()}

    def bank(verdict: str) -> None:
        rec["verdict"], rec["finished"] = verdict, now()
        summ.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"ZZR34 {verdict} ZZ", flush=True)

    # the checkpoint timm will read (offline): revision + file sha256, so "the cached file" is a fact
    rev = (CKPT_DIR / "refs" / "main").read_text(encoding="utf-8").strip()
    f = CKPT_DIR / "snapshots" / rev / "model.safetensors"
    rec["checkpoint"] = {"repo": "timm/resnet34.a1_in1k", "revision": rev, "path": str(f),
                         "size": f.stat().st_size, "sha256": sha256(f)}

    tp = str(tree).replace("\\", "/")
    env = dict(os.environ, PYTHONPATH=f"{tp}/stack;{tp}/stack/scripts;{tp}/taniteval",
               OMP_NUM_THREADS="4", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="")
    chk = subprocess.run([PY, "-c", "import tanitad, tanitad.models.timm_trunk as t; "
                          "print(tanitad.__file__); print(t.__file__)"],
                         env=env, capture_output=True, text=True, cwd=str(tree / "stack"))
    where = chk.stdout.split()
    rec["tanitad_file"], rec["timm_trunk_file"] = (where + [None, None])[:2]
    if len(where) != 2 or not all(w.replace("\\", "/").lower().startswith(tp.lower()) for w in where):
        rec["preflight_stderr"] = chk.stderr[-800:]
        bank("PREFLIGHT_REFUSED: tanitad does not resolve inside the tree")
        return 2

    # gate 1: BOXES_DONE
    w0 = time.time()
    while not marker.exists():
        if time.time() - w0 > BOXES_WAIT_S:
            bank(f"NOT_RUN: {marker} did not appear within {BOXES_WAIT_S // 3600} h")
            return 3
        time.sleep(30)
    rec["gates"]["boxes_done_seen"] = now()

    # gate 2: the RAM floor (3 consecutive samples, 30 s apart)
    w0, ok, samples = time.time(), 0, []
    while ok < START_SAMPLES:
        g = free_gb()
        samples = (samples + [round(g, 2)])[-6:]
        ok = ok + 1 if g >= START_GB else 0
        if ok >= START_SAMPLES:
            break
        if time.time() - w0 > RAM_WAIT_S:
            rec["gates"]["last_samples_gib"] = samples
            bank(f"NOT_RUN: RAM floor ({START_GB} GiB x{START_SAMPLES}) not met within "
                 f"{RAM_WAIT_S // 3600} h")
            return 3
        time.sleep(START_GAP_S)
    rec["gates"].update({"ram_floor_passed": now(), "start_samples_gib": samples[-START_SAMPLES:],
                         "floor": f"start >= {START_GB} GiB x{START_SAMPLES} ({START_GAP_S:.0f} s "
                                  f"apart), kill < {KILL_GB} GiB"})

    junit = outdir / "r34_fix5.junit.xml"
    log = outdir / "r34_fix5.log"
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfEs", f"--junitxml={junit}",
           *[f"{FILE}::{t}" for t in TESTS]]
    rec["pytest_cmd"] = cmd
    t0, status, min_free = time.time(), None, free_gb()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=str(tree / "stack"), env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
        rec["child_pid"] = p.pid
        while p.poll() is None:
            time.sleep(2.0)
            g = free_gb()
            min_free = min(min_free, g)
            if g < KILL_GB:
                p.kill(); p.wait(); status = "RAM_ABORT"; break
            if time.time() - t0 > RUN_TIMEOUT_S:
                p.kill(); p.wait(); status = "TIMEOUT"; break
    rec["pytest"] = {"rc": p.returncode, "seconds": round(time.time() - t0, 1),
                     "min_free_gib": round(min_free, 2), "status": status or "EXITED"}
    if status:
        bank(f"NOT_COMPLETED: {status}")
        return 4
    if not junit.exists():
        bank("NO_JUNIT: pytest wrote no junit file -- inadmissible, whatever its exit status")
        return 5
    rec["tests"].update(parse_junit(junit, TESTS))
    ok_all, sts = verdict_of(rec["tests"])
    bank("PASS: both passed" if ok_all else f"FAIL: {dict(zip(TESTS, sts))}")
    return 0 if ok_all else 1


def parse_junit(junit: Path, names) -> dict:
    """{name: {"status": passed|failed|error|skipped, "message", "time_s"}} for the NAMED tests
    found in the junit file; a name that is absent is simply not in the result."""
    out = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        name = tc.get("name")
        if name not in names:
            continue
        kids = {c.tag: (c.get("message") or "")[:300] for c in tc
                if c.tag in ("failure", "error", "skipped")}
        st = ("failed" if "failure" in kids else "error" if "error" in kids
              else "skipped" if "skipped" in kids else "passed")
        out[name] = {"status": st, "message": next(iter(kids.values()), ""),
                     "time_s": float(tc.get("time") or 0.0)}
    return out


def verdict_of(tests: dict) -> tuple[bool, list]:
    """True only when EVERY named test is present AND passed -- a skip or an absence is not a pass."""
    sts = [v["status"] if v else "ABSENT" for v in tests.values()]
    return all(s == "passed" for s in sts), sts


if __name__ == "__main__":
    sys.exit(main())
