#!/usr/bin/env python3
"""(b) Does the LIVE command line + `--yaw-loss plain --declare-change yaw_loss` resume cleanly from a checkpoint the
UNPATCHED trainer wrote -- with `--declare-change scorer_mode` still on the command line, as pod_train_v2.sh's OP_FLAGS
carry it? Real `train.main` on temporary copies (tiny shim, CPU), through the real resume path:

  1  UNPATCHED, fixed scorer, --grow: starts, checkpoints, halts (rc 7)
  2  UNPATCHED resume with the live OP_FLAGS (--scorer-mode onpolicy ... --declare-change scorer_mode): the 2026-09-26
     switch replayed; checkpoints with scorer_mode onpolicy in its identity; halts (rc 7) -- the live run's state
  3  PATCHED --preflight: identity differs ONLY by the declared yaw_loss; writes nothing (rc 0, PREFLIGHT_OK)
  4  PATCHED WITHOUT --declare-change yaw_loss: refused (rc 4) -- the negative control
  5  PATCHED with both declares: resumes and completes (rc 0); ONE declared_change event naming only --yaw-loss;
     "MEASURE M6" printed; every later checkpoint's meta carries measures.yaw_loss == "plain"
  6  PATCHED resume again from 5's checkpoint (no --declare-change yaw_loss needed any more): identity matches
    python test_switch_resume.py            -> ZZSWITCH_RESUME_OK / ZZSWITCH_RESUME_FAIL
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFE = HERE.parents[2] / "refe"
sys.path.insert(0, str(REFE))
import selftest_measures as SM  # noqa: E402  make_bank / targs / the child trainer shim


def main() -> int:
    import torch
    tmp = Path(tempfile.mkdtemp(prefix="switch_resume_"))
    orig, pat = tmp / "orig", tmp / "patched"
    for d in (orig, pat):
        d.mkdir()
        for f in REFE.glob("*.py"):
            shutil.copy2(f, d / f.name)
    for pch in ("measures_hooks.patch", "measures_consumers.patch"):
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-p1", str(REFE / pch)], cwd=str(pat), check=True)
    bank, op, empty = tmp / "bank", tmp / "op", tmp / "empty"
    empty.mkdir()
    SM.make_bank(bank, 8, [8.0, 12.0, 15.0], [0.0, 0.02, -0.03], rank1_every=2, onpolicy=op)
    run = tmp / "run"
    grow = ["--grow", "--epochs", "3", "--grow-scenes", "8", "--ckpt-every-steps", "1"]
    OPF = ["--scorer-mode", "onpolicy", "--onpolicy-targets", str(op), "--declare-change", "scorer_mode"]
    M6 = ["--yaw-loss", "plain", "--declare-change", "yaw_loss"]
    n = [0]

    def child(root, argv):
        n[0] += 1
        spec = tmp / f"spec{n[0]}.json"
        out = tmp / f"child{n[0]}.out"
        spec.write_text(json.dumps({"root": str(root), "out": str(out), "argv": argv}))
        lf = tmp / f"child{n[0]}.log"
        with open(lf, "w", encoding="utf-8") as fh:
            subprocess.run([sys.executable, str(REFE / "selftest_measures.py"), "--child", "trainer", "--spec", str(spec)],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=str(root),
                           env=dict(os.environ, OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8"))
        return torch.load(out, weights_only=False)["rc"], lf.read_text(encoding="utf-8")

    def args(*extra):
        return SM.targs(bank, run, empty, *grow, *extra, batch=2, accum=1)
    res = {}
    res["1_unpatched_start"] = child(orig, args("--halt-after-steps", "1"))
    res["2_unpatched_onpolicy_switch"] = child(orig, args("--resume", *OPF, "--halt-after-steps", "2"))
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    mode_in_ckpt = ck["identity"]["args"].get("scorer_mode")
    before = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
    res["3_patched_preflight"] = child(pat, args("--resume", *OPF, *M6, "--preflight"))
    after = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
    res["4_patched_undeclared"] = child(pat, args("--resume", *OPF, "--yaw-loss", "plain"))
    res["5_patched_switch"] = child(pat, args("--resume", *OPF, *M6, "--halt-after-steps", "4"))
    ev = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
    dec = [e for e in ev if e.get("event") == "declared_change"]
    ck5 = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    res["6_patched_resume_again"] = child(pat, args("--resume", *OPF, "--yaw-loss", "plain"))
    fin = torch.load(run / "model_final.pt", map_location="cpu", weights_only=False)
    checks = {
        "1 unpatched start halts (rc 7)": res["1_unpatched_start"][0] == 7,
        "2 unpatched on-policy switch replayed (rc 7, scorer_mode onpolicy in the checkpoint)":
            res["2_unpatched_onpolicy_switch"][0] == 7 and mode_in_ckpt == "onpolicy",
        "3 preflight OK, only the declared yaw_loss differs, nothing written":
            res["3_patched_preflight"][0] == 0 and "PREFLIGHT_OK" in res["3_patched_preflight"][1]
            and "--yaw-loss: 'wrapped' -> 'plain'" in res["3_patched_preflight"][1] and before == after,
        "4 without --declare-change yaw_loss: REFUSED (rc 4)":
            res["4_patched_undeclared"][0] == 4 and "REFUSING TO RESUME" in res["4_patched_undeclared"][1],
        "5 switch resumes (rc 7 at the test halt), M6 active, one declared change naming only yaw_loss":
            res["5_patched_switch"][0] == 7 and "MEASURE M6" in res["5_patched_switch"][1]
            and len(dec) >= 2 and dec[-1]["changes"] == ["--yaw-loss: 'wrapped' -> 'plain'"],
        "5 the switched checkpoint's meta carries measures.yaw_loss plain":
            (ck5["meta"].get("measures") or {}).get("yaw_loss") == "plain",
        "6 resume again needs no declare (identity matches), completes (rc 0), final meta plain":
            res["6_patched_resume_again"][0] == 0 and (fin["meta"].get("measures") or {}).get("yaw_loss") == "plain",
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  (declared_change events: {[d['changes'] for d in dec]}; temp {tmp})")
    ok = all(checks.values())
    if ok:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZSWITCH_RESUME_OK" if ok else "ZZSWITCH_RESUME_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
