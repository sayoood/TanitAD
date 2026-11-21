"""PREREG_V8F_GRID's reader on the REAL banked M2b evals (zero GPU, dev box). Each proof is a NEW directory (nothing is overwritten or deleted):

  python real_proofs.py <repo root> <out root>        (PYTHONPATH must reach <repo root>/stack and <repo root>/taniteval)

  1. identity (base M2b)      the banked M2b side against ITSELF: `--check-inputs` -- F, LAMBDA(M2b), the S5 excess reproduce the PREREG's known values (via PGT's own
                              check), the PGT stand-in read ties B's ADE6 to what the PGT reader recorded for the banked M2b, and the PRIMARY, M-LEAK, M-DK, present-slot,
                              M-LAT, oracle-ADE6 and every family cell read exactly +0 [0, 0] (the reader refuses otherwise). Also the latcons RMSE of M2b_s0: 0.686 SD.
  2. regression (both bases)  a SYNTHETIC G = B's plans PLUS a forward shift at 2 s (+2.0 m on the standard plan), T's codes unchanged: the PRIMARY must FAIL, M-LAT must read
                              no fall, the verdict is FAIL-NO-MECHANISM.
  3. success (both bases)     a SYNTHETIC G whose plans move 60 % of the way to the GT and whose T latcons error shrinks 40 %: the SAME code must read SUCCESS.
  4. mechanism without drive  a SYNTHETIC G with B's plans exactly and T's latcons error shrunk 30 %: PRIMARY fails (delta exactly 0), M-LAT met -> FAIL-MECHANISM-OK.
  Both base branches run: base M2b (the banked evals) and base M2b-pgt (a SYNTHETIC M2b-pgt derived from the banked arrays by PREREG_V8F_PGT's own proof recipe, so that the
  PGT stand-in read -- raw/2026-10-22-pgt/SYNTHETIC_pgt_read_positive.json, VERDICT SUCCESS -- ties to it).
The synthetic arms are built from the banked arrays ONLY to exercise the real reader on real shapes; they carry the names M2b-pgt / M2b-8x20 / M2b-pgt-8x20 and are NOT a
result of any trained model (their output files are named SYNTHETIC_*)."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
PK = ROOT / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = PK / "raw"
spec = importlib.util.spec_from_file_location("v8f_grid_read_real", PK / "code" / "v8f_grid_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
DK, WC = R._modules()
SRC = {"": "full/eval", "-tacabs": "2026-10-22-cz/eval", "-tacorc": "2026-10-20-tac/eval", "-tacorc-all-latcons": "2026-10-21-tacgrp2/eval", "-tacorc-all-goal": "2026-10-21-tacgrp2/eval"}
C0, NC = WC.C0, WC.N_CONS
LAT = [C0 + i for i in WC.GROUPS["latcons"]]
STEPS = 19164
A_ = type("A", (), dict(gt_npz=None, eval_grid=str(RAW / "eval_grid_windows.json"), eval_kin=str(RAW / "eval141_kin.npz"), eval_join=str(RAW / "join_eval141.npz")))
GT = DK.ground_truth(A_)["gt"]


def ade(p):
    return np.linalg.norm(np.asarray(p).astype(np.float32).astype(np.float64) - GT, axis=-1).mean(1)


def ramp(d, k=30, n=10):
    return d * np.minimum(1.0, (np.arange(k) + 1) / n)


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


def template(ev, s):
    """The banked M2b seed s in the NEW format: the GT-code oracle carries tac_code (= all-goal's constraint block IS the GT), pred = the GT-code plan."""
    zs = {suf: dict(np.load(ev / f"eval_M2b_s{s}{suf}.npz")) for suf in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacorc-all-goal")}
    orc = zs["-tacorc"]["pred"].astype(np.float64)
    return {"std": zs[""]["pred"].astype(np.float64), "orc": orc, "lat": zs["-tacorc-all-latcons"]["pred"].astype(np.float64), "z": zs}


def emit_evals(ev, tag, T, std, orc, lat, tcons_fn, ckpt=None):
    """Eval files of one synthetic arm. `tcons_fn(code, gt_code, present)` maps T's own code (absmask eval) to the arm's."""
    z = T["z"]
    gt_code = np.asarray(z["-tacorc-all-goal"]["tac_code"], np.float64)
    present = (~np.asarray(z["-tacorc-all-goal"]["tac_oracle_fallback"], bool))[:, None] & (gt_code != 0.0)
    tcode = tcons_fn(np.asarray(z["-tacabs"]["tac_code"], np.float64), gt_code, present)
    out = {"": dict(z[""]), "-tacabs": dict(z["-tacabs"]), "-tacorc": dict(z["-tacorc-all-goal"]), "-tacorc-all-latcons": dict(z["-tacorc-all-latcons"])}
    out[""]["pred"], out[""]["ade6"] = std.astype(np.float32), ade(std)
    out["-tacabs"].update(pred=std.astype(np.float32), ade6=ade(std), tac_code=tcode)
    out["-tacorc"].update(pred=orc.astype(np.float32), ade6=ade(orc))
    lc = np.asarray(z["-tacorc-all-latcons"]["tac_code"], np.float64).copy()
    lc[:, LAT] = tcode[:, LAT]
    out["-tacorc-all-latcons"].update(pred=lat.astype(np.float32), ade6=ade(lat), tac_code=lc)
    for suf, d in out.items():
        np.savez(ev / f"eval_{tag}{suf}.npz", **d)
    return gt_code, present


def emit_eval_jsons(ev, tag, src_json, ckpt=None):
    for suf, body in (("", None), ("-tacabs", {"tac_manipulation": {"mode": "absmask"}}), ("-tacorc", {"tac_manipulation": {"mode": "oracle"}}),
                      ("-tacorc-all-latcons", {"tac_manipulation": {"mode": "oracle"}})):
        j = dict(json.loads(Path(src_json).read_text(encoding="utf-8"))) if body is None else dict(body)
        if ckpt:
            j["ckpt"] = ckpt
        (ev / f"eval_{tag}{suf}.json").write_text(json.dumps(j), encoding="utf-8")


def sched(step, total, p0, p1):
    return p0 + (p1 - p0) * min(max(step / total, 0.0), 1.0)


def emit_run_pgt(ru, s):
    """M2b-pgt_s<s>: M2b's run files plus the two lever flags, the built schedule and a train_log on the 0.5 -> 0.0 schedule."""
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
            f.write(json.dumps({"step": st, "p_gt": sched(st - 1, sm["steps"], 0.5, 0.0)}) + "\n")


def emit_run_g(ru, base, s):
    """<base>_s<s>_8x20: the B run's files with --grid pool8x20, the parameter table (+120 x 256), the checkpoint's grid / n_tok / pos rows and the inherited schedule."""
    d = ru / f"{base}_s{s}_8x20"
    d.mkdir()
    b = ru / f"{base}_s{s}"
    sm = json.loads((b / "summary.json").read_text(encoding="utf-8"))
    sc = json.loads((b / "selfcheck.json").read_text(encoding="utf-8"))
    p0, p1 = R.DOSE[base]
    sha = f"synthetic-g-{base}-{s}"
    for rec in (sm, sc):
        lv = dict(rec["levers"])
        lv["params"] = {**lv["params"], "adapter": lv["params"]["adapter"] + 120 * 256}
        lv["n_params_trainable"] = lv["n_params_trainable"] + 120 * 256
        lv["p_gt_start"], lv["p_gt_end"] = p0, p1
        rec["levers"], rec["argv_sha"] = lv, sha
    sc["info"] = {**sc.get("info", {}), "lru": 43}
    sm["seconds"] = sm["seconds"] * 1.7
    (d / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
    (d / "selfcheck.json").write_text(json.dumps(sc), encoding="utf-8")
    av = json.loads((b / "argv.json").read_text(encoding="utf-8"))["argv"]
    av = list(av)
    av[av.index("--grid") + 1] = "pool8x20"
    av[av.index("--out") + 1] = av[av.index("--out") + 1] + "_8x20"
    (d / "argv.json").write_text(json.dumps({"argv": av, "sha": sha}), encoding="utf-8")
    shutil.copyfile(b / "train_log.jsonl", d / "train_log.jsonl")
    torch.save({"model": {"pos": torch.zeros(160, 256)}, "cfg": {"n_tok": 160}, "grid": "pool8x20", "argv_sha": sha, "step": sm["steps"], "arm": "M2b", "seed": s}, d / "ckpt_last.pt")


def stage_base(dst, base, pgt_stand_in):
    """The B side of one proof world: banked M2b (base M2b) or the synthetic M2b-pgt (base M2b-pgt); and the PGT stand-in read next to it."""
    ev, ru = stage_banked(dst)
    pgt = dst / "pgt"
    pgt.mkdir()
    if base == "M2b":
        shutil.copyfile(RAW / "2026-10-22-pgt/SYNTHETIC_pgt_read_regression.json", pgt / "pgt_read.json")      # final read, VERDICT FAIL-NO-MECHANISM: B = M2b
    else:
        for s in (0, 1):
            T = template(ev, s)
            g = GT
            std = g + 0.4 * (T["std"] - g)
            lat = T["orc"] + 0.4 * (T["lat"] - T["orc"])
            emit_evals(ev, f"M2b-pgt_s{s}", T, std, T["orc"], lat, lambda code, gc, pr: code)
            emit_eval_jsons(ev, f"M2b-pgt_s{s}", ev / f"eval_M2b_s{s}.json")
            emit_run_pgt(ru, s)
        shutil.copyfile(RAW / "2026-10-22-pgt/SYNTHETIC_pgt_read_positive.json", pgt / "pgt_read.json")        # final read, VERDICT SUCCESS: B = M2b-pgt
    return ev, ru, pgt


def shrink_lat(frac):
    def f(code, gc, pr):
        out = code.copy()
        cols = LAT
        out[:, cols] = np.where(pr[:, cols], gc[:, cols] + (1.0 - frac) * (code[:, cols] - gc[:, cols]), code[:, cols])
        return out
    return f


def synth_g(dst, base, std_fn, tcons_fn, lat_fn=None):
    ev, ru = dst / "eval", dst / "runs"
    for s in (0, 1):
        Tb = template(ev, s) if base == "M2b" else None
        if base == "M2b":
            T = Tb
            b_std, b_lat = T["std"], T["lat"]
        else:                                                 # B = the synthetic M2b-pgt: its plans are what G is compared with
            zb = {suf: np.load(ev / f"eval_M2b-pgt_s{s}{suf}.npz") for suf in ("", "-tacorc-all-latcons")}
            T = template(ev, s)
            b_std, b_lat = zb[""]["pred"].astype(np.float64), zb["-tacorc-all-latcons"]["pred"].astype(np.float64)
        std = std_fn(b_std, T["orc"], GT)
        lat = (lat_fn or (lambda p, orc, g: p))(b_lat, T["orc"], GT)
        emit_evals(ev, f"{base}-8x20_s{s}", T, std, T["orc"], lat, tcons_fn)
        emit_eval_jsons(ev, f"{base}-8x20_s{s}", ev / f"eval_{'M2b' if base == 'M2b' else 'M2b'}_s{s}.json", ckpt=f"D:\\tanitad\\runs\\v8f_mini_full\\{base}_s{s}_8x20\\ckpt_last.pt")
        emit_run_g(ru, base, s)


def run(dst, base, stage, tag):
    out = dst / f"grid_read_{stage}.json"
    a = ["--stage", stage, "--base", base, "--pgt-dir", str(dst / "pgt"), "--eval-dir", str(dst / "eval"), "--runs-dir", str(dst / "runs"), "--out", str(out), "--dk-n-boot", "500"]
    rc = R.main(a)
    r = json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None
    print(f"[{tag}] rc={rc}", flush=True)
    return rc, r


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    keep = {}
    for base in ("M2b", "M2b-pgt"):
        # ---- 0. the base rule on the stand-in PGT reads
        d0 = OUT / f"identity_{base}"
        ev, ru, pgt = stage_base(d0, base, None)
        a = ["--decide-base", "--pgt-dir", str(pgt)]
        assert R.main(a) == 0
        assert R.decide_base(pgt)["base"] == base
        # ---- 1. identity: the B side alone
        rc = R.main(["--check-inputs", "--base", base, "--pgt-dir", str(pgt), "--eval-dir", str(ev), "--runs-dir", str(ru), "--out", str(d0 / "check_inputs.json"), "--dk-n-boot", "500"])
        print(f"[identity {base}] rc", rc, flush=True)
        assert rc == 0
        ci = json.loads((d0 / "check_inputs.json").read_text(encoding="utf-8"))
        cells = {k for s in ci["identity_control"].values() for k in s}
        assert all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for s in ci["identity_control"].values() for k, v in s.items() if isinstance(v, dict) and "delta" in v)
        print(f"[identity {base}] {len(cells)} contrasts x 2 seeds, every one exactly +0 [0, 0]; latcons RMSE(B) {ci['latcons_rmse_slot_b']} (window-weighted {ci['latcons_rmse_window_b']})", flush=True)
        keep[f"check_inputs_{base}"] = d0 / "check_inputs.json"
        # ---- 2. deliberate regression: G = B's plans + a forward shift at 2 s; T's codes unchanged
        d2 = OUT / f"regression_{base}"
        stage_base(d2, base, None)
        synth_g(d2, base, lambda p, orc, g: p + np.stack([ramp(2.0), np.zeros(30)], -1)[None], lambda code, gc, pr: code)
        rc, r = run(d2, base, "final", f"regression {base}")
        assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM", r and r["VERDICT"]
        for s in ("s0", "s1"):
            assert not r["rows"][s]["judge"]["passed"] and r["rows"][s]["primary"]["point"] > 0 and r["rows"][s]["mlat"]["fall"] == 0.0
        print(f"[regression {base}]", r["reading"], flush=True)
        keep[f"regression_{base}"] = d2 / "grid_read_final.json"
        # ---- 3. success: plans 60 % toward the GT, T's latcons error -40 %
        d3 = OUT / f"success_{base}"
        stage_base(d3, base, None)
        synth_g(d3, base, lambda p, orc, g: g + 0.4 * (p - g), shrink_lat(0.4))
        rc, r = run(d3, base, "final", f"success {base}")
        assert rc == 0 and r["VERDICT"] == "SUCCESS" and r["mechanism"] is True, r and r["VERDICT"]
        print(f"[success {base}]", r["reading"], flush=True)
        keep[f"success_{base}"] = d3 / "grid_read_final.json"
        # ---- 4. mechanism without drive: B's plans exactly, T's latcons error -30 %
        d4 = OUT / f"mechanism_only_{base}"
        stage_base(d4, base, None)
        synth_g(d4, base, lambda p, orc, g: p, shrink_lat(0.3))
        rc, r = run(d4, base, "final", f"mechanism-only {base}")
        assert rc == 0 and r["VERDICT"] == "FAIL-MECHANISM-OK" and r["mechanism"] is True, r and r["VERDICT"]
        print(f"[mechanism-only {base}]", r["reading"], flush=True)
        keep[f"mechanism_only_{base}"] = d4 / "grid_read_final.json"
    for k, p in keep.items():
        shutil.copyfile(p, OUT / (("SYNTHETIC_" if not k.startswith("check_inputs") else "") + f"grid_read_{k}.json"))
    print("OK")


main()
