"""PREREG_V8F_PGT's reader on the REAL banked M2b evals (zero GPU, dev box). Three proofs, each a NEW directory (nothing is overwritten or deleted):

  python real_proofs.py <repo root> <out root>        (PYTHONPATH must reach <repo root>/stack and <repo root>/taniteval)

  1. identity     the banked M2b side against ITSELF: `--check-inputs` -- F, LAMBDA(M2b), the S5 excess reproduce the PREREG's known values, and the
                  PRIMARY, M-LEAK, M-DK, present-slot, oracle-ADE6 and every family cell read exactly +0 [0, 0] (the reader refuses otherwise).
  2. regression   a SYNTHETIC M2b-pgt whose plans are M2b's PLUS a forward shift at 2 s (the along-track of the standard plan and of the all-latcons
                  plan ramp to +D at step 10): M-LEAK and M-DK must read WORSE, separated; the PRIMARY is not met; the verdict is not SUCCESS.
  3. positive     a SYNTHETIC M2b-pgt whose plans move 60 % of the way to the GT (standard plan) and whose latcons leak is cut to 40 %: the SAME code
                  must be able to read SUCCESS (so the reader is not structurally incapable of a pass).
The synthetic arms are built from the banked M2b arrays ONLY to exercise the real reader on real shapes; their files carry arm names M2b-pgt and are
NOT a result of any trained model (their output files are named SYNTHETIC_*). Their runs carry the declared schedule in argv, levers and a train_log, as a real run's do."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
PK = ROOT / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = PK / "raw"
spec = importlib.util.spec_from_file_location("v8f_pgt_read_real", PK / "code" / "v8f_pgt_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
DK, WC = R._modules()
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
                    try:
                        shutil.copyfile(p, ev / p.name)
                    except OSError:
                        if ext == "npz":
                            raise
        (ru / t).mkdir()
        for f in ("summary.json", "argv.json", "selfcheck.json", "train_log.jsonl"):
            shutil.copyfile(RAW / "full/runs" / t / f, ru / t / f)
    shutil.copyfile(RAW / "full/eval/floors_perwindow.npz", ev / "floors_perwindow.npz")
    return ev, ru


def ramp(d, k=30, n=10):
    return d * np.minimum(1.0, (np.arange(k) + 1) / n)


def synth_pgt(dst, gt, std_fn, lat_fn):
    """M2b-pgt_s{0,1} built from the banked M2b files: std_fn / lat_fn map (pred, orc_pred, gt) -> the arm's standard / all-latcons plan."""
    ev, ru = dst / "eval", dst / "runs"
    for s in (0, 1):
        zs = {suf: dict(np.load(ev / f"eval_M2b_s{s}{suf}.npz")) for suf in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacorc-all-goal")}
        orc = zs["-tacorc"]["pred"].astype(np.float64)
        std = std_fn(zs[""]["pred"].astype(np.float64), orc, gt)
        lat = lat_fn(zs["-tacorc-all-latcons"]["pred"].astype(np.float64), orc, gt)

        def ade(p):
            return np.linalg.norm(p.astype(np.float32).astype(np.float64) - gt, axis=-1).mean(1)
        z = dict(zs[""])
        z["pred"], z["ade6"] = std.astype(np.float32), ade(std)
        np.savez(ev / f"eval_M2b-pgt_s{s}.npz", **z)
        shutil.copyfile(ev / f"eval_M2b_s{s}.json", ev / f"eval_M2b-pgt_s{s}.json")
        z = dict(zs["-tacabs"])
        z["pred"], z["ade6"] = std.astype(np.float32), ade(std)
        np.savez(ev / f"eval_M2b-pgt_s{s}-tacabs.npz", **z)
        shutil.copyfile(ev / f"eval_M2b_s{s}-tacabs.json", ev / f"eval_M2b-pgt_s{s}-tacabs.json")
        z = dict(zs["-tacorc-all-goal"])                 # the NEW-format GT oracle: tac_code = the GT code (here: all-goal's, whose constraint block IS the GT)
        z["pred"], z["ade6"] = orc.astype(np.float32), ade(orc)
        np.savez(ev / f"eval_M2b-pgt_s{s}-tacorc.npz", **z)
        z = dict(zs["-tacorc-all-latcons"])
        z["pred"], z["ade6"] = lat.astype(np.float32), ade(lat)
        np.savez(ev / f"eval_M2b-pgt_s{s}-tacorc-all-latcons.npz", **z)
        for suf in ("-tacorc", "-tacorc-all-latcons"):
            (ev / f"eval_M2b-pgt_s{s}{suf}.json").write_text(json.dumps({"tac_manipulation": {"mode": "oracle"}}), encoding="utf-8")
        d = ru / f"M2b-pgt_s{s}"
        d.mkdir()
        sm = json.loads((ru / f"M2b_s{s}" / "summary.json").read_text(encoding="utf-8"))
        sm["levers"] = {**sm["levers"], "p_gt_start": 0.5, "p_gt_end": 0.0}
        (d / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
        av = json.loads((ru / f"M2b_s{s}" / "argv.json").read_text(encoding="utf-8"))
        a2 = list(av["argv"]) + ["--p-gt-start", "0.5", "--p-gt-end", "0.0"]
        a2[a2.index("--out") + 1] = a2[a2.index("--out") + 1].replace("M2b_s", "M2b-pgt_s")
        (d / "argv.json").write_text(json.dumps({"argv": a2, "sha": "synthetic"}), encoding="utf-8")
        sc = json.loads((ru / f"M2b_s{s}" / "selfcheck.json").read_text(encoding="utf-8"))
        sc["levers"] = {**sc["levers"], "p_gt_start": 0.5, "p_gt_end": 0.0}
        (d / "selfcheck.json").write_text(json.dumps(sc), encoding="utf-8")
        with open(d / "train_log.jsonl", "w", encoding="utf-8") as f:
            for st in range(50, sm["steps"] + 1, 50):
                f.write(json.dumps({"step": st, "p_gt": R.schedule(st - 1, sm["steps"], 0.5, 0.0)}) + "\n")


def run(dst, stage, tag):
    out = dst / f"pgt_read_{stage}.json"
    a = ["--stage", stage, "--eval-dir", str(dst / "eval"), "--runs-dir", str(dst / "runs"), "--out", str(out), "--dk-n-boot", "500"]
    rc = R.main(a)
    r = json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None
    print(f"[{tag}] rc={rc}")
    return rc, r


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    A = type("A", (), dict(gt_npz=None, eval_grid=str(RAW / "eval_grid_windows.json"), eval_kin=str(RAW / "eval141_kin.npz"), eval_join=str(RAW / "join_eval141.npz")))
    gt = DK.ground_truth(A)["gt"]
    # ---- 1. identity
    d1 = OUT / "identity"
    stage_banked(d1)
    a = ["--check-inputs", "--eval-dir", str(d1 / "eval"), "--runs-dir", str(d1 / "runs"), "--out", str(d1 / "check_inputs.json"), "--dk-n-boot", "500"]
    rc = R.main(a)
    print("[identity] rc", rc)
    assert rc == 0
    ic = json.loads((d1 / "check_inputs.json").read_text(encoding="utf-8"))["identity_control"]
    cells = {k for s in ic.values() for k in s}
    assert all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for s in ic.values() for v in s.values())
    print(f"[identity] {len(cells)} contrasts x {len(ic)} seeds, every one exactly +0 [0, 0]:", sorted(cells))
    # ---- 2. deliberate regression: M2b's plans + a forward shift at 2 s
    d2 = OUT / "regression"
    stage_banked(d2)
    D_STD, D_LAT = 2.0, 0.5
    synth_pgt(d2, gt, lambda p, orc, g: p + np.stack([ramp(D_STD), np.zeros(30)], -1)[None], lambda p, orc, g: p + np.stack([ramp(D_LAT), np.zeros(30)], -1)[None])
    rc, r = run(d2, "final", "regression")
    assert rc == 0
    for s in ("s0", "s1"):
        row = r["rows"][s]
        assert row["mleak"]["delta"]["point"] > 0 and row["mleak"]["delta"]["separated"], f"{s}: M-LEAK must read WORSE, separated"
        assert row["mdk"]["delta"]["point"] > 0 and row["mdk"]["delta"]["separated"], f"{s}: M-DK must read WORSE, separated"
        assert not row["judge"]["mechanism"] and not row["judge"]["passed"]
    assert r["VERDICT"] == "FAIL-NO-MECHANISM"
    print("[regression]", r["reading"])
    # ---- 3. positive control: the same code CAN read SUCCESS
    d3 = OUT / "positive"
    stage_banked(d3)
    synth_pgt(d3, gt, lambda p, orc, g: g + 0.4 * (p - g), lambda p, orc, g: orc + 0.4 * (p - orc))
    rc, r = run(d3, "final", "positive")
    assert rc == 0
    print("[positive]", r["reading"])
    for tag, d in (("regression", d2), ("positive", d3)):
        shutil.copyfile(d / "pgt_read_final.json", OUT / f"SYNTHETIC_pgt_read_{tag}.json")
    shutil.copyfile(d1 / "check_inputs.json", OUT / "check_inputs_identity.json")
    print("OK")


main()
