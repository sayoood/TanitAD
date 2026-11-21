"""PREREG_V8F_HIST's reader on the REAL banked M2b evals (zero GPU, dev box). Each proof is a NEW directory (nothing is overwritten or deleted):

  python real_proofs.py <repo root> <out root> [<K = 10 lead block npz>]     (PYTHONPATH must reach <repo root>/stack and <repo root>/taniteval)
  The lead block defaults to `Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz` under <repo root>; name it when the export left it out.

  1. identity   the banked M2b side against ITSELF: `--check-inputs` -- B is a 4-frame run at the auto hmax 16 (both seeds), the PREREG's known values reproduce
                (F, LAMBDA(M2b), the S5 excess, the latcons RMSE 0.686 SD of M2b_s0), the eval grid has the committed D4 counts (2,399 windows, 141 at t = 3, 141 at t = 7,
                2,117 with t >= 11), and every contrast of the paired instrument reads exactly +0 [0, 0] (the reader refuses otherwise).
  2. refusals   the same inputs with ONE premise broken, each a NEW staging directory, each a rc 2 REFUSAL (not a pass, not a crash):
                (a) B's argv names `--hist 8`                   -> "is not a 4-frame run"
                (b) B's two seeds disagree about hmax            -> "disagree about hmax"
                (c) a banked B eval file removed                 -> "is missing"
                (d) B's argv names `--window-start 3` (Amendment 1: B is the banked M2b, which predates the flag) -> "predates the flag"
The eval staging copies the banked M2b files of the five suffixes ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacorc-all-goal": M2b's tacorc predates tac_code, so its GT constraints come from the all-goal oracle) from where they were banked."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
LEAD = ["--lead-block", str(Path(sys.argv[3]).resolve())] if len(sys.argv) > 3 else []
PK = ROOT / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = PK / "raw"
spec = importlib.util.spec_from_file_location("v8f_hist_read_real", PK / "code" / "v8f_hist_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
SRC = {"": "full/eval", "-tacabs": "2026-10-22-cz/eval", "-tacorc": "2026-10-20-tac/eval", "-tacorc-all-latcons": "2026-10-21-tacgrp2/eval", "-tacorc-all-goal": "2026-10-21-tacgrp2/eval"}


def stage_banked(dst):
    ev, ru = dst / "eval", dst / "runs"
    ev.mkdir(parents=True, exist_ok=False)
    ru.mkdir(parents=True, exist_ok=False)
    for s in (0, 1):
        t = f"M2b_s{s}"
        for suf, d in SRC.items():
            for ext in ("npz", "json"):
                p = RAW / d / f"eval_{t}{suf}.{ext}"
                if p.exists() or ext == "npz":
                    shutil.copyfile(p, ev / p.name)
        (ru / t).mkdir()
        for f in ("summary.json", "argv.json", "selfcheck.json", "train_log.jsonl"):
            shutil.copyfile(RAW / "full/runs" / t / f, ru / t / f)
    shutil.copyfile(RAW / "full/eval/floors_perwindow.npz", ev / "floors_perwindow.npz")
    return ev, ru


def run(ev, ru, out, tag):
    rc = R.main(["--check-inputs", "--eval-dir", str(ev), "--runs-dir", str(ru), "--out", str(out), "--dk-n-boot", "500"] + LEAD)
    print(f"[{tag}] rc={rc}", flush=True)
    return rc


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    # ---- 1. identity
    d1 = OUT / "identity"
    ev, ru = stage_banked(d1)
    assert run(ev, ru, d1 / "check_inputs.json", "identity") == 0
    ci = json.loads((d1 / "check_inputs.json").read_text(encoding="utf-8"))
    cells = [(s, k, v) for s, blk in ci["identity_control"].items() for k, v in blk.items() if isinstance(v, dict) and "delta" in v]
    assert cells and all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for _, _, v in cells), "an identity cell is not exactly +0 [0, 0]"
    wc = ci["window_counts"]
    print("[identity] window_counts", wc)
    print(f"[identity] F {ci['F']:.4f} m | b_hmax {ci['b_hmax']} b_hist {ci['b_hist']} | b_steps {ci['b_steps']} b_n_windows {ci['b_n_windows']}")
    print("[identity] lambda_b", ci["lambda_b"], "| excess_p10", ci["excess_p10"])
    print("[identity] latcons RMSE(B) slot-pooled", ci["latcons_rmse_slot_b"], "window-weighted", ci["latcons_rmse_window_b"])
    print(f"[identity] {len(cells)} identity contrasts across 2 seeds, every one exactly +0 [0, 0]")
    assert ci["b_hmax"] == 16 and ci["b_hist"] == 4
    assert all(v == 19164 for v in ci["b_steps"].values()) and all(v == 306566 for v in ci["b_n_windows"].values())
    # ---- 2. refusals, each from a fresh staging directory
    d2 = OUT / "refuse_hist8"
    ev, ru = stage_banked(d2)
    j = json.loads((ru / "M2b_s0" / "argv.json").read_text(encoding="utf-8"))
    j["argv"] = list(j.get("argv", [])) + ["--hist", "8"]
    (ru / "M2b_s0" / "argv.json").write_text(json.dumps(j), encoding="utf-8")
    assert run(ev, ru, d2 / "check_inputs.json", "refuse hist 8") == 2
    d3 = OUT / "refuse_hmax"
    ev, ru = stage_banked(d3)
    j = json.loads((ru / "M2b_s1" / "argv.json").read_text(encoding="utf-8"))
    j["argv"] = list(j.get("argv", [])) + ["--hmax", "18"]
    (ru / "M2b_s1" / "argv.json").write_text(json.dumps(j), encoding="utf-8")
    assert run(ev, ru, d3 / "check_inputs.json", "refuse hmax disagree") == 2
    d4 = OUT / "refuse_missing"
    ev, ru = stage_banked(d4)
    (ev / "eval_M2b_s1-tacabs.npz").rename(ev / "eval_M2b_s1-tacabs.npz.removed")        # renamed aside, never deleted
    assert run(ev, ru, d4 / "check_inputs.json", "refuse missing eval") == 2
    d5 = OUT / "refuse_window_start"
    ev, ru = stage_banked(d5)
    j = json.loads((ru / "M2b_s0" / "argv.json").read_text(encoding="utf-8"))
    j["argv"] = list(j.get("argv", [])) + ["--window-start", "3"]
    (ru / "M2b_s0" / "argv.json").write_text(json.dumps(j), encoding="utf-8")
    assert run(ev, ru, d5 / "check_inputs.json", "refuse B naming --window-start") == 2
    print("ALL REAL PROOFS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
