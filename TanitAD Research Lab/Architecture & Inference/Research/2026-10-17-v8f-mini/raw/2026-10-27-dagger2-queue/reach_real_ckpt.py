#!/usr/bin/env python3
"""REACHABILITY, part 1 (REAL checkpoints, REAL argv.json, no corpus): does the REAL code of the repo tip 8e121ea8 ACCEPT DAG's checkpoint as the bank generator and as --init-ckpt, and DAG's argv.json as --m2b-argv?

Everything here is the tip's own code (stack/scripts/v8f_dagger_bank.py, v8f_mini_train.py, tanitad/*) imported from an extracted copy of the tip (REACH_TREE), on the dev-box CPU:

  R1  the generator's own loader  v8f_dagger_bank.load_ckpt(DAG ckpt)         -- cfg gate, strict state-dict load
  R2  the trainer's own loader    v8f_mini_train.load_init_ckpt(DAG ckpt, model built from DAG's argv.json)  -- INIT_CKPT_MATCH gate, strict load, read-back of every tensor
  R3  the generator's TRAIN list parser: TR.build_parser().parse_args(DAG argv.json["argv"]) -- accepted; and the namespace differs from M2b's ONLY in fine-tune / DAgger / out flags (no data flag moves,
      so build_data(ns) rebuilds the SAME 4,512-clip TRAIN list)
  R4  the checkpoint's record: DAG's cfg == M2b's on every architecture key, the state dicts have the same keys and shapes
  R5  the bytes: each DAG ckpt / argv.json hashes to the value v8f-dagger/M2b_s{S}_dag/SHA256SUMS.txt records

Control (must read the OPPOSITE): a doctored config (d changed) must be REFUSED by load_init_ckpt, and a state dict with a key removed must be REFUSED by the generator's strict load.

    CUDA_VISIBLE_DEVICES=-1 python reach_real_ckpt.py [out.json]
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
TREE = Path(os.environ.get("REACH_TREE", r"C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\dag2q_work\tree"))
DAGDIR = Path(os.environ.get("DAG_RUNS", r"E:\Projects\TanitAD-artifacts\v8f-dagger"))
M2BDIR = Path(os.environ.get("M2B_RUNS", r"C:\Users\Admin\box2_ship\logs\v8f_mini\runs"))
for p in (TREE / "stack" / "scripts", TREE / "stack", TREE / "taniteval"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
import v8f_dagger_bank as GEN  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(2)
RES = {"cases": []}


def case(name, ok, detail=""):
    RES["cases"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:500]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:260]}]" if detail else ""), flush=True)


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sums(seed) -> dict:
    out = {}
    for line in (DAGDIR / f"M2b_s{seed}_dag" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, n = line.split(" ", 1)
        out[n.strip().lstrip("*")] = h
    return out


def main():
    for seed in (0, 1):
        d = DAGDIR / f"M2b_s{seed}_dag"
        ck_p, av_p = d / "ckpt_last.pt", d / "argv.json"
        S = sums(seed)
        case(f"R5 s{seed}: DAG ckpt_last.pt hashes to SHA256SUMS.txt's value", sha(ck_p) == S["ckpt_last.pt"], S["ckpt_last.pt"][:16])
        case(f"R5 s{seed}: DAG argv.json hashes to SHA256SUMS.txt's value", sha(av_p) == S["argv.json"], S["argv.json"][:16])
        # ---- R1: the generator's loader
        model, cfg, ck = GEN.load_ckpt(str(ck_p), torch.device("cpu"))
        case(f"R1 s{seed}: v8f_dagger_bank.load_ckpt accepts DAG's checkpoint (cfg gate kin_mode=yaw, strict load_state_dict)", cfg.kin_mode == "yaw" and len(ck["model"]) > 0, f"{len(ck['model'])} tensors")
        rows = GEN.check_readable_rows(cfg)
        case(f"R1 s{seed}: the generator's readable-rows guard passes on DAG's cfg", rows["first_row_read"] >= 1, rows)
        # ---- R4: against the banked M2b
        mk = torch.load(M2BDIR / f"M2b_s{seed}" / "ckpt_last.pt", map_location="cpu", weights_only=False)
        diff = {k: (mk["cfg"].get(k), ck["cfg"].get(k)) for k in sorted(set(mk["cfg"]) | set(ck["cfg"])) if mk["cfg"].get(k) != ck["cfg"].get(k)}
        arch = {k: v for k, v in diff.items() if k in TR.INIT_CKPT_MATCH}
        case(f"R4 s{seed}: DAG's cfg equals M2b's on EVERY architecture key (INIT_CKPT_MATCH); the keys that differ are training levers only", not arch, f"arch diffs {arch}; lever diffs {diff}")
        sk, mk_sd = ck["model"], mk["model"]
        shp = lambda sd: {k: tuple(v.shape) for k, v in sd.items()}   # noqa: E731
        case(f"R4 s{seed}: DAG's state dict has exactly M2b's keys and shapes", shp(sk) == shp(mk_sd), f"{len(sk)} vs {len(mk_sd)} tensors")
        meta = {k: (type(v).__name__, (v if isinstance(v, (int, float, str)) else None)) for k, v in ck.items() if k not in ("model", "opt", "optimizer", "cfg")}
        case(f"R4 s{seed}: the checkpoint's own record (arm / seed / step / argv_sha) is readable", ck.get("arm") == "M2b" and ck.get("seed") == seed, meta)
        # ---- R3: DAG's argv.json through the trainer's parser, against M2b's
        dag_argv = json.loads(av_p.read_text(encoding="utf-8"))["argv"]
        m2b_argv = json.loads((M2BDIR / f"M2b_s{seed}" / "argv.json").read_text(encoding="utf-8"))["argv"]
        ns_d, ns_m = TR.build_parser().parse_args(dag_argv), TR.build_parser().parse_args(m2b_argv)
        moved = sorted(k for k in vars(ns_d) if getattr(ns_d, k) != getattr(ns_m, k, None))
        allowed = {"out", "init_ckpt", "steps", "lr", "warmup", "p_gt_start", "p_gt_end", "ego_perturb_p", "dagger_p", "dagger_bank", "dagger_bank_sha256", "dagger_seed"}
        case(f"R3 s{seed}: the trainer's parser ACCEPTS DAG's argv.json['argv'] and it differs from M2b's only in out / init / steps / lr / warmup / p_gt / ego-perturb / dagger flags (NO data flag moves)",
             set(moved) <= allowed, f"moved {moved}")
        data_keys = ("cache_dir", "tracks", "join_train", "v9_train", "v9_train_md5", "lead_train", "speed_train", "exclude", "data_frac", "max_clips", "grid", "kin_mode", "batch", "seed", "fit_join", "synthetic", "hist", "window_start", "hmax", "lru")
        case(f"R3 s{seed}: every data-selecting flag is equal, so build_data(ns) rebuilds M2b's TRAIN list (4,512 clips)", all(getattr(ns_d, k) == getattr(ns_m, k) for k in data_keys if hasattr(ns_d, k)), {k: getattr(ns_d, k) for k in ("exclude", "data_frac", "max_clips", "grid")})
        case(f"R3 s{seed}: DAG's argv is not a --synthetic run (the generator's guard)", not ns_d.synthetic)
        # ---- R2: the trainer's init loader, model built from DAG's own argv
        cfg_t = TR.make_config(ns_d)
        prior = M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))
        mdl = M.V8FMini(cfg_t, prior)
        rec = TR.load_init_ckpt(str(ck_p), mdl, cfg_t, apply=True)
        case(f"R2 s{seed}: v8f_mini_train.load_init_ckpt accepts DAG's checkpoint as --init-ckpt: sha256 recorded, applied, 0 tensors differ after the load", rec["sha256"] == S["ckpt_last.pt"] and rec["applied"] and rec["n_mismatch_after_load"] == 0 and rec["n_tensors"] == len(sk),
             {k: rec[k] for k in ("sha256", "applied", "n_tensors", "n_mismatch_after_load", "source_arm", "source_seed", "source_step")})
        # ---- controls (must refuse)
        bad_cfg = copy.deepcopy(ck)
        bad_cfg["cfg"]["d"] = int(bad_cfg["cfg"]["d"]) * 2
        tmp = TREE.parent / f"_ctl_badcfg_s{seed}.pt"
        torch.save(bad_cfg, tmp)
        try:
            TR.load_init_ckpt(str(tmp), M.V8FMini(cfg_t, prior), cfg_t, apply=True)
            refused = False
        except ValueError as e:
            refused = "config mismatch" in str(e)
        case(f"CONTROL s{seed}: a DAG checkpoint with d doubled is REFUSED by load_init_ckpt (the probe can fail)", refused)
        bad_sd = copy.deepcopy(ck)
        bad_sd["model"].pop(next(iter(bad_sd["model"])))
        torch.save(bad_sd, tmp)
        try:
            GEN.load_ckpt(str(tmp), torch.device("cpu"))
            refused = False
        except RuntimeError as e:
            refused = "Missing key" in str(e) or "missing" in str(e).lower()
        case(f"CONTROL s{seed}: a DAG checkpoint with one tensor removed is REFUSED by the generator's strict load (the probe can fail)", refused)
        tmp.unlink()
    RES["n_pass"] = sum(c["ok"] for c in RES["cases"])
    RES["n_cases"] = len(RES["cases"])
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps(RES, indent=1), encoding="utf-8")
    print(f"reach_real_ckpt: {RES['n_pass']}/{RES['n_cases']} pass")
    return 0 if RES["n_pass"] == RES["n_cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
