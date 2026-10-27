#!/usr/bin/env python3
"""REACHABILITY part 1 (CLAUDE.md "built, tested, and unreachable from its caller"): the REAL DAG_s checkpoints (v8f-dagger/M2b_s{0,1}_dag/ckpt_last.pt, read-only) through the tip 3b7b404e's REAL parser -> make_config ->
V8FMini -> load_init_ckpt, for the PRODUCTION argv the staged queue prints for A / REP / B at both seeds. CPU, no GPU. Part 2 (reach_synthetic.py) runs the trainer's gate and a short training; this part is the ONLY place the real
checkpoints meet the tip trainer before box 2 does. It answers: does the start state the chain pins (and asserts on box 2) LOAD into the model the DAG-ORC argv builds, strictly and read back, and what does the tip's
config check (INIT_CKPT_MATCH, with the F1 `o_interface` hunk) actually compare for these two files?

    CUDA_VISIBLE_DEVICES=-1 python reach_real_ckpt.py [out.json]
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagorc_lib as L  # noqa: E402

TREE = Path(os.environ.get("REACH_TREE", r"F:\TanitAD-scratch\dagorc_tree"))
WORK = Path(os.environ.get("REACH_WORK", r"F:\TanitAD-scratch\dagorc_work\reach_real"))
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(2)
res = {"cases": [], "t0": time.time()}
PRIOR_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_prior.pt"


def case(name, ok, detail=""):
    res["cases"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:700]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:400]}]" if detail else ""), flush=True)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    # the repo's frozen P10 prior (a tracked file at the tip; the model's prior.* keys have its shapes)
    pp = WORK / "p10_prior.pt"
    r = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", f"3b7b404e:{PRIOR_REL}"], capture_output=True)
    assert r.returncode == 0 and len(r.stdout) > 1000, r.stderr[:200]
    pp.write_bytes(r.stdout)
    prior = M.load_p10(str(pp))
    mu, sd = np.zeros(26, np.float32), np.ones(26, np.float32)      # the buffers are overwritten by the strict load; only their shapes matter here
    pins = {}
    for s in (0, 1):
        d = {}
        for ln in (L.DAG_DIR / f"M2b_s{s}_dag" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            h, n = ln.split(" ", 1)
            d[n.strip().lstrip("*")] = h
        pins[s] = d["ckpt_last.pt"]
    cks = {}
    for s in (0, 1):
        path = L.DAG_DIR / f"M2b_s{s}_dag" / "ckpt_last.pt"
        case(f"R0 DAG_s{s}: the banked checkpoint hashes to the pinned sha256 (SHA256SUMS.txt)", sha_file(path) == pins[s], pins[s][:16])
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cks[s] = (path, ck)
        has_oi = "o_interface" in ck["cfg"]
        case(f"R0 DAG_s{s}: a v8f_mini_train checkpoint (model + cfg dicts); records whether its cfg carries `o_interface` (the F1 hunk reads that key only when present)", isinstance(ck.get("model"), dict) and isinstance(ck.get("cfg"), dict),
             f"o_interface in cfg: {has_oi} ({ck['cfg'].get('o_interface')!r}); arm {ck.get('arm')} seed {ck.get('seed')} step {ck.get('step')}; {len(ck['model'])} tensors")
        res.setdefault("o_interface_in_cfg", {})[str(s)] = has_oi
    for s in (0, 1):
        path, ck = cks[s]
        for arm in ("dagorc", "dagorcrep", "dagorc01"):
            tl = L.trainer_line(L.dry(f"traindag {arm} {s}"))
            toks = L.train_argv(tl)
            try:
                a = TR.build_parser().parse_args(toks)
                parsed = True
            except SystemExit:
                parsed = False
            case(f"R1 {arm} s{s}: the production training argv parses with the tip trainer's own parser", parsed)
            if not parsed:
                continue
            p = {"dagorc": 0.3, "dagorcrep": 0.0, "dagorc01": 0.1}[arm]
            case(f"R1 {arm} s{s}: the parsed namespace carries the registered levers (ego_perturb_p {p}, dagger_p 0, init-ckpt tail M2b_s{s}_dag, no bank)",
                 a.ego_perturb_p == p and a.dagger_p == 0.0 and a.init_ckpt.replace("\\", "/").endswith(f"/M2b_s{s}_dag/ckpt_last.pt") and not a.dagger_bank and a.steps == 3000 and a.seed == (10 + s if arm == "dagorcrep" else s),
                 {k: getattr(a, k) for k in ("ego_perturb_p", "dagger_p", "init_ckpt", "steps", "seed")})
            cfg = TR.make_config(a)
            model = M.V8FMini(cfg, prior, mu, sd)
            t0 = time.time()
            try:
                rec = TR.load_init_ckpt(str(path), model, cfg, apply=True)
                err = None
            except Exception as e:      # noqa: BLE001
                rec, err = None, f"{type(e).__name__}: {str(e)[:300]}"
            case(f"R2 {arm} s{s}: load_init_ckpt(DAG_s{s}) into the model this argv builds: strict load, every tensor read back equal, sha256 = the pin",
                 rec is not None and rec["applied"] is True and rec["n_mismatch_after_load"] == 0 and rec["sha256"] == pins[s] and rec["n_tensors"] == len(ck["model"]), err or {k: rec[k] for k in ("applied", "n_mismatch_after_load", "n_tensors", "source_arm", "source_step")} | {"sec": round(time.time() - t0, 1)})
            compared = [k for k in TR.INIT_CKPT_MATCH if k in ck["cfg"]]
            skipped = [k for k in TR.INIT_CKPT_MATCH if k not in ck["cfg"]]
            res.setdefault("config_fields", {})[f"{arm}_s{s}"] = {"compared": compared, "absent_in_checkpoint": skipped}
    # G-DVB coverage of every flag the production argv uses (the brief: "every flag used has a G-DVB entry"); the trainer's own gate refuses a parser flag without an entry, this lists them for these argv
    P = TR.build_parser()
    want_kind = {"ego_perturb_p": "ego:ego_perturb_p", "ego_perturb_vscale": "ego:ego_perturb_vscale", "ego_perturb_vmax": "ego:ego_perturb_vmax", "ego_perturb_adrop": "ego:ego_perturb_adrop",
                 "ego_perturb_yawrate_std": "ego:ego_perturb_yawrate_std", "ego_perturb_seed": "ego:ego_perturb_seed", "init_ckpt": "data:init_ckpt", "dagger_p": "dagger:dagger_p"}
    used = {}
    for s in (0, 1):
        for arm in ("dagorc", "dagorcrep", "dagorc01"):
            toks = L.train_argv(L.trainer_line(L.dry(f"traindag {arm} {s}")))
            for x in toks:
                if x.startswith("--"):
                    act = P._option_string_actions.get(x)
                    used.setdefault(x, act.dest if act else None)
    nokind = sorted(f for f, d in used.items() if d is None or d not in TR.FLAG_CHECKS)
    case(f"R5 every one of the {len(used)} flags the six production argv use is a parser flag WITH a FLAG_CHECKS (G-DVB) entry", not nokind and len(used) > 20, nokind or sorted(used)[:60])
    bad_kind = {d: TR.FLAG_CHECKS.get(d) for d in want_kind if TR.FLAG_CHECKS.get(d) != want_kind[d]}
    case("R5 the DAG-ORC levers' entries are the ones traced: ego_perturb_{p,vscale,vmax,adrop,yawrate_std,seed} -> ego:, init_ckpt -> data:init_ckpt, dagger_p -> dagger:dagger_p (read back from the BUILT objects)", not bad_kind, bad_kind)
    res["flags_used"] = {f: used[f] for f in sorted(used)}
    # deliberate regressions: the tip's config check must REFUSE a model that is not the checkpoint's
    path, ck = cks[0]
    toks = L.train_argv(L.trainer_line(L.dry("traindag dagorc 0")))
    for nm, edit in (("a different width (--d 48)", lambda t: t[:t.index("--d") + 1] + ["48"] + t[t.index("--d") + 2:] if "--d" in t else t + ["--d", "48"]),
                     ("a different grid (--grid pool8x20)", lambda t: [("pool8x20" if x == "pool4x10" else x) for x in t]),
                     ("a different history (--hist 6)", lambda t: t[:t.index("--hist") + 1] + ["6"] + t[t.index("--hist") + 2:] if "--hist" in t else t + ["--hist", "6"])):
        a = TR.build_parser().parse_args(edit(list(toks)))
        try:
            cfg = TR.make_config(a)
            model = M.V8FMini(cfg, prior, mu, sd)
            TR.load_init_ckpt(str(path), model, cfg, apply=True)
            refused = False
            why = "ACCEPTED"
        except Exception as e:      # noqa: BLE001
            refused, why = isinstance(e, (ValueError, RuntimeError)), f"{type(e).__name__}: {str(e)[:200]}"
        case(f"R3 deliberate regression: DAG_s0 into {nm}: REFUSED", refused, why)
    # a checkpoint whose own cfg is doctored (one architecture field) must be refused, and the unmodified copy accepted (the control)
    d2 = WORK / "doctored"
    d2.mkdir(exist_ok=True)
    ck2 = copy.deepcopy(ck)
    ck2["cfg"]["n_q"] = ck2["cfg"]["n_q"] + 1
    torch.save(ck2, d2 / "ckpt_last.pt")
    a = TR.build_parser().parse_args(list(toks))
    cfg = TR.make_config(a)
    try:
        TR.load_init_ckpt(str(d2 / "ckpt_last.pt"), M.V8FMini(cfg, prior, mu, sd), cfg, apply=True)
        ref, why = False, "ACCEPTED"
    except Exception as e:      # noqa: BLE001
        ref, why = isinstance(e, ValueError), f"{type(e).__name__}: {str(e)[:200]}"
    case("R3 deliberate regression: a checkpoint whose cfg.n_q is doctored (+1) is REFUSED with a ValueError", ref, why)
    shutil.rmtree(d2, ignore_errors=True)
    # F1: what the o_interface key does for these files. A checkpoint WITHOUT the key cannot be told 'cut' from 'full' by the config check -- recorded, not hidden.
    has_oi = res["o_interface_in_cfg"]["0"]
    a = TR.build_parser().parse_args(toks + ["--o-interface", "cut"])
    cfg = TR.make_config(a)
    try:
        TR.load_init_ckpt(str(path), M.V8FMini(cfg, prior, mu, sd), cfg, apply=False)
        acc = True
    except ValueError:
        acc = False
    case(f"R4 F1 hunk on the REAL file: DAG_s0 {'carries' if has_oi else 'does NOT carry'} cfg.o_interface; a 'cut' model {'is refused' if not acc else 'is ACCEPTED (the check reads the key only when present; DAG_s was trained at full -- DAG-ORC names no --o-interface, default full)'}",
         (acc is False) if has_oi else True, f"o_interface in cfg {has_oi}; cut-model load refused: {not acc}")
    res["passed"] = all(c["ok"] for c in res["cases"])
    res["sec"] = round(time.time() - res["t0"])
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reach_real_ckpt.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    n = sum(1 for c in res["cases"] if c["ok"])
    print(f"RESULT: {n}/{len(res['cases'])} cases pass ({res['sec']} s) -> {out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
