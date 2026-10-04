"""Eval steps 3-4 on the DEV BOX for a seam + score produced elsewhere (a Colab VM): the resume-from-seam mode.

A NEW wrapper beside `eval/eval_checkpoint.py`, not an edit of it (that file is run by the live snapshot pipeline).
It imports eval_checkpoint only for its constants and its `relabel`/`run` helpers, and runs the SAME two commands
eval_checkpoint's steps 3-4 run, in the same interpreter and cwd:
  3. `parse_navtest6.py`  (TanitAD venv, EV6 dir)  floors + paired intervals on the SAME tokens
  4. `families6.py`       (TanitAD venv, EV6 dir)  the four families (strategic UNAVAILABLE by design)
Steps 1-2 are replaced by their VM outputs, which must already have passed their own gates: the seam's report must
say ZZSEAM_OK conditions (rows == tokens asked, frame control <= 1 mm) and the score's counts.json status PASS.

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe colab/navtest/resume_from_seam.py --name sub200_ep016_colab \
        --seam <vm seam.npz> --csv <vm score csv> --tokens <A1_sub200_tokens.json> --device "Colab L4" \
        --gate "P0 G-S/G-P" --out <dir>  [--prev ep015 ...]
Writes <out>/<name>.json -- the same point schema as eval_checkpoint.py plus `device`, `vm_gpu`, `parity_gate` --
and <out>/<name>/{3_parse,4_families}.{json,log}. It writes into the learning-curve's points/ directory ONLY when
--out points there; P0 does not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, f"{PKG}/eval")
import eval_checkpoint as EC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seam", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--vm-gpu", default=None)
    ap.add_argument("--gate", required=True, help="the parity gate that licenses this device's points")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", nargs="*", default=[])
    a = ap.parse_args()
    t0 = time.time()
    pdir = os.path.join(a.out, a.name)
    os.makedirs(pdir, exist_ok=True)
    out = {"name": a.name, "tokens": a.tokens, "seam_path": a.seam, "csv_path": a.csv, "device": a.device,
           "vm_gpu": a.vm_gpu, "parity_gate": a.gate, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "steps_1_2": "produced off the dev box; admitted on their own gates below"}
    rep_p = os.path.splitext(a.seam)[0] + ".report.json"
    seam = json.load(open(rep_p, encoding="utf-8"))
    ok_seam = seam["rows"] == seam["tokens_asked"] and seam["frame_control"]["max_m"] <= seam["frame_control"]["bar_m"]
    cnt_p = os.path.splitext(a.csv)[0] + ".counts.json"
    score = json.load(open(cnt_p, encoding="utf-8"))
    out["seam"], out["score"] = seam, {k: score.get(k) for k in (
        "label", "status", "log_successful", "log_failed", "csv_valid_rows", "C1_max_abs_delta", "summary_x100_4dp",
        "wall_s", "failures")}
    if not ok_seam or score.get("status") != "PASS":
        out["verdict"] = "SEAM_FAILED" if not ok_seam else "SCORE_FAILED"
        json.dump(out, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
        print(f"ZZRESUME_FAIL {a.name} {out['verdict']}")
        return 1
    par = os.path.join(pdir, "3_parse.json")
    extra = [f"R6_{p}={EC.DATA}/score/refe_{p}/refe_{p}.csv" for p in a.prev
             if os.path.exists(f"{EC.DATA}/score/refe_{p}/refe_{p}.csv")]
    cmd = [EC.TANITAD_PY, "parse_navtest6.py", "--arm-csv", os.path.abspath(a.csv), "--tokens", a.tokens,
           "--label", f"REFe-{a.name}", "--out", par] + (["--extra"] + extra if extra else [])
    EC.run(cmd, EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), os.path.join(pdir, "3_parse.log"))
    fam = os.path.join(pdir, "4_families.json")
    EC.run([EC.TANITAD_PY, "families6.py", "--seam", os.path.abspath(a.seam), "--inputs", EC.EXPORT, "--stage", "1",
            "--label", f"REFe-{a.name}", "--out", fam, "--n-boot", "2000"],
           EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), os.path.join(pdir, "4_families.log"))
    for k, p in (("floors", par), ("families", fam)):
        out[k] = EC.relabel(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else None
    out["seconds"] = round(time.time() - t0, 1)
    out["verdict"] = "OK" if out["floors"] is not None and out["families"] is not None else "STEP_3_4_INCOMPLETE"
    json.dump(out, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
    print(f"ZZRESUME_{'OK' if out['verdict'] == 'OK' else 'FAIL'} {a.name} "
          f"PDMS={score['summary_x100_4dp']['PDMS']} device={a.device}")
    return 0 if out["verdict"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
