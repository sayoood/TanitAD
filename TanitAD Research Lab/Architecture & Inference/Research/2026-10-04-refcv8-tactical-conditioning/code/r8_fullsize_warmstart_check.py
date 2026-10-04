"""refcv8 at FULL SIZE, zero GPU (dev box CPU): does the refcv8 smoke argv build, load refcv7-50,400 through the
trainer's warm-start rule, pass G-DVB, join v9 on eval139 through the loader, and reproduce refcv7's plan?

Steps (MEASURED, written to raw/r8_fullsize_warmstart.json):
1. m7 = refcv7_loader.build_model(refcv7 config.json, ckpt_50400.pt, strict=True) -- the refcv7 reference.
2. for each refcv8 variant -- A: every seam on but allocation (n_alloc 0); B: + allocation M 32, emission OFF:
   m8 = build_model(config with the refcv8 argv, same ckpt, strict=False), then `refcv8_train.warm_start_from` on the
   same file (the TRAINER's rule: every source key strict, every missing key a refcv8 seam); G-DVB mismatches.
3. the eval139 dataset through the LOADER with the refcv8 argv (the v9 EVAL join attached by `build_eval_dataset`).
4. N windows, batch 1, eval mode, `torch.manual_seed(1000 + i)` before each forward: m7 vs m8 on the SAME batch.
   A: traj / sel_idx / anchor_traj / sel_score_v3 bit-identical. B: traj / sel_idx / base fan bit-identical, base
   scores within 1e-5 (the GEMM-shape rounding SPEC_WPB I-W states).

Run (from the package):  PYTHONPATH=<extract+fix>/stack;<...>/taniteval python code/r8_fullsize_warmstart_check.py
    --stack <extract+fix>/stack --n 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = "D:/refcv6_eval_kit"
CKPT_DIR = Path("D:/refcv7_eval_kit/ckpt")
V9 = "D:/Projects/TanitAD-artifacts/v9labels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--variants", default="A_seams_no_alloc,B_alloc32_emit_off")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.environ.setdefault("REFCV6_KIT", KIT)
    os.environ.setdefault("REFCV6_REPO", "D:/Projects/TanitAD")
    stack = Path(a.stack)
    sys.path.insert(0, str(stack / "scripts"))
    import torch
    import tanitad
    if str(Path(tanitad.__file__).resolve()).upper().startswith("G:"):
        raise SystemExit("tanitad imported from G:")
    from tanitad.eval import refcv7_loader as L
    from tanitad.train import refcv8_train as RT
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "6"))))
    t0 = time.time()
    cfg7_rec = json.loads((CKPT_DIR / "config.json").read_text(encoding="utf-8"))
    ck = str(CKPT_DIR / "ckpt_50400.pt")
    out = {"what": "refcv8 full-size warm start, zero GPU (dev box CPU), MEASURED", "ckpt": ck, "variants": {}}
    m7, cfg7, args7, rec7 = L.build_model(cfg7_rec, ck, device="cpu", strict=True)
    m7.eval()
    out["refcv7"] = {"step": rec7.get("state_dict", {}).get("step"), "dvb": rec7.get("declared_vs_built", {})
                     .get("mismatches")}
    smoke = json.loads((stack / "ops/runs.d/refcv8-wpb-smoke.argv.json").read_text(encoding="utf-8"))["argv"]

    def setf(argv, f, v):
        argv = list(argv)
        if f in argv:
            i = argv.index(f)
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                j += 1
            argv[i:j] = [f] + list(v)
        else:
            argv += [f] + list(v)
        return argv

    def drop(argv, f):
        return [t for t in argv if t != f]

    base = setf(smoke, "--r8-v9-labels", [f"{V9}/v9_labels_train.npz"])
    base = setf(base, "--r8-v9-labels-eval", [f"{V9}/v9_labels_eval139.npz"])
    base = setf(base, "--join-defect-masks", [str(stack / "tanitad/configs/refcv8_join_label_defects.json")])
    base = setf(base, "--init-from", [ck])
    def drop_flag(argv, f):                              # a flag AND its value(s)
        if f not in argv:
            return list(argv)
        i = argv.index(f)
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        return argv[:i] + argv[j:]

    variants = {
        "A_seams_no_alloc": setf(drop_flag(drop(setf(base, "--r8-n-alloc", ["0"]), "--r8-alloc-emit"),
                                           "--r8-alloc-emit-start"), "--w-r8-alloc-l1", ["0"]),
        "B_alloc32_emit_off": drop_flag(drop(base, "--r8-alloc-emit"), "--r8-alloc-emit-start"),
        # P9 (MM ruling Q1): the LAUNCH argv as written (emission scheduled from S_emit) -> I-2 at step 0
        "C_launch_argv_I2": list(base),
        # its deliberate regression: the same argv emitting from step 0 (the start removed, no warm-start pin here)
        "D_REGRESSION_emit_at_step0_I2": drop_flag(base, "--r8-alloc-emit-start"),
    }
    # an n_alloc 0 build may not keep L_sat (it acts on allocated candidates only)
    variants["A_seams_no_alloc"] = setf(variants["A_seams_no_alloc"], "--w-r8-sat", ["0"])
    keep = set(a.variants.split(","))
    for name, argv in variants.items():
        if name not in keep:
            continue
        tv = time.time()
        c8 = copy.deepcopy(cfg7_rec)
        c8["argv"] = argv
        if name.startswith("D_"):                         # the trainer pin refuses this argv WITH --init-from; the
            c8["argv"] = drop_flag(argv, "--init-from")      # loader is given the same ckpt and loads it below
        m8, cfg8, args8, rec8 = L.build_model(c8, ck, device="cpu", strict=False)
        ws = RT.warm_start_from(m8, ck)
        v_emit = {"loader": rec8.get("r8_emit"), "emits_at_step0": bool(RT.emits_at_step0(m8.cfg.refcv8))}
        m8.eval()
        v = {"missing_from_loader": len(rec8["state_dict"]["missing"]),
             "unexpected_from_loader": len(rec8["state_dict"].get("unexpected", [])),
             "all_missing_are_refcv8_seams": all(RT.is_refcv8_key(k) for k in rec8["state_dict"]["missing"]),
             "warm_start_from": ws, "dvb_mismatches": rec8.get("declared_vs_built", {}).get("mismatches"),
             "r8_n_params": int(getattr(m8, "r8_n_params", 0))}
        e_ds, e_eps, drec = L.build_eval_dataset(m8, cfg8, args8, c8)
        v["eval_v9_join"] = drec.get("r8_v9", {}).get("join")
        tr = L.trainer()
        cap = {}
        real8, real7 = type(m8).forward, type(m7).forward

        def hooked(self, *aa, **kw):
            o = real8(self, *aa, **kw)
            cap[id(self)] = {k: o[k].detach().clone() for k in ("traj", "sel_idx", "anchor_traj", "sel_score_v3")
                             if k in o and torch.is_tensor(o[k])}
            if "r8_n_base" in o:
                cap[id(self)]["n_base"] = int(o["r8_n_base"])
            return o
        type(m8).forward = hooked
        diffs = []
        try:
            idx = [int(i) for i in torch.linspace(0, len(e_ds) - 1, a.n).round().tolist()]
            for i in idx:
                batch = torch.utils.data.default_collate([e_ds[i]])
                with torch.no_grad():
                    torch.manual_seed(1000 + i)
                    tr.compute_losses_v3(m7, batch, "cpu", mode=args7.mode)
                    torch.manual_seed(1000 + i)
                    tr.compute_losses_v3(m8, batch, "cpu", mode=args8.mode)
                o7, o8 = cap[id(m7)], cap[id(m8)]
                nb = o8.get("n_base", o7["anchor_traj"].shape[1])
                row = {"window": i, "traj_equal": bool(torch.equal(o7["traj"], o8["traj"])),
                       "max_abs_dtraj_m": float((o7["traj"] - o8["traj"]).abs().max()),
                       "max_abs_dfan_base_m": float((o7["anchor_traj"] - o8["anchor_traj"][:, :nb]).abs().max()),
                       "sel_idx_equal": bool(torch.equal(o7["sel_idx"], o8["sel_idx"])),
                       "picked_extra": bool((o8["sel_idx"] >= nb).any()),
                       "base_fan_equal": bool(torch.equal(o7["anchor_traj"], o8["anchor_traj"][:, :nb])),
                       "base_score_max_abs_diff": float((o7["sel_score_v3"] - o8["sel_score_v3"][:, :nb]).abs().max())}
                diffs.append(row)
        finally:
            type(m8).forward = real8
        v["windows"] = diffs
        if "_I2" in name:
            i2 = RT.i2_identity_row(diffs, emit_at_step0=v_emit["emits_at_step0"])
            v["I2"], v["emission"] = i2, v_emit
            crit = i2["PASS"]
            v["criterion"] = "I-2 literals (MM ruling Q1): sel_idx 100 %, max |dtraj| <= 1e-3 m, |d base score| <= 1e-4"
        else:
            crit = (all(r["traj_equal"] and r["sel_idx_equal"] and r["base_fan_equal"] for r in diffs)
                    and all(r["base_score_max_abs_diff"] <= (0.0 if name.startswith("A") else 1e-5) for r in diffs))
            v["criterion"] = ("bit-identical" if name.startswith("A") else
                              "plan + base fan bit-identical, base scores <= 1e-5")
        v["identity_PASS"] = bool(crit)
        v["wall_s"] = round(time.time() - tv, 1)
        out["variants"][name] = v
        print(f"[r8-fullsize] {name}: identity {'PASS' if crit else 'FAIL'}; missing {v['missing_from_loader']} "
              f"(all seams {v['all_missing_are_refcv8_seams']}); dvb {v['dvb_mismatches']}; join {v['eval_v9_join']}; "
              f"{v['wall_s']} s", flush=True)
        del m8
    out["wall_s"] = round(time.time() - t0, 1)
    p = HERE.parent / "raw" / f"r8_fullsize_warmstart{a.tag}.json"
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[r8-fullsize] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
