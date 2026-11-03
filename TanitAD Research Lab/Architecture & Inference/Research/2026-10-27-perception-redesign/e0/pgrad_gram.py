"""E0j / C11 -- P-GRAD with a GRAM output, on the refcv9 E1 perception rig (REFCV9_DESIGN 2.2, 4.1 E0j, 4.2 C11).

THIS IS A COPY IN THE REFCV9 PACKAGE of the refcv8 `pgrad.py` idea (the refcv8 file is left untouched: its tip blob lacks the three fixes
the stranded E: draft carries -- TF32 off, requires_grad re-enable after the eval loader froze everything, and parts -- and it targets
refcv7's six-term loss; here the rig loss is exactly {map, box3d, agent}). It stores, per parameter group, the BLOCK MATRIX
K[(b,s),(c,t)] = <v(b,s), v(c,t)> over the batches b of the draw and the vectors s,t in {map, box3d, agent, direct}, where
v(b, term) = grad of the UNWEIGHTED term (w = 1) and v(b, direct) = ONE backward of the weighted total sum_j w2_j L_j (the
linearity control). `gram_core` turns K into the exact projected share for any weight vector; `e0j_read.py` does the reading.

THE BUILD (what the model IS -- the E1 arm M1/M2 step 0, seed 0)
  argv = stack/ops/runs.d/r9e1_C0.argv.json  [+ `--r9-map-banded` and `--map-hires-near-lift-m 10`   for --stage banded]
         (the banded head REFUSES near-lift 20: `MapHiresConfig.__post_init__`; the M1 smoke argv carries 10), built by the
         tip's own eval loader (`refcv7_loader.build_model`, G-DVB against that argv), weights = refcv8 EMA-48,852 through the
         rig's own predicate (planner/tactical keys dropped; under banded the dropped `_map_hires.refine.cls.*` and the NEW
         `_map_hires.banded.*` keys are NAMED, anything else missing/unexpected REFUSES). The banded head is therefore a FRESH
         seed-0 init -- the gradient is read at the E1 starting point, NOT at a trained head (disclosed in the card).
  --stage baseline8: the same rig WITHOUT `--r9-map-banded` (refcv8's 8-way 10 cm CE, near-lift 20): the matched-instrument
         "refcv8 own w = 1" reading to sit beside the banded one.
THE WINDOWS: pgrad's rule -- 139 TRAIN clips (TRAIN-DIAG cache), sorted by sha12, 8 evenly spaced windows each (1,112), a seeded
         numpy default_rng(0) draw of n_batches*batch (64) without replacement. The same rule as the refcv8 P-GRAD.
MEASUREMENT CHOICES (disclosed): trunk in FP32 (memory_levers bf16 off); TF32 OFF for matmul AND cuDNN (the stranded draft MEASURED
         a 6.2e-4 linearity error with TF32 on -- a precision floor, not a gradient property; G-SMOKE-L4's 4.2e-4 is the same class);
         cuDNN deterministic; `model.train()` (BN is frozen + folded as trained); ONE forward per batch, then one
         `torch.autograd.grad` per term with retain_graph; `.grad` is never touched; manual_seed(1000 + batch).
CONTROLS (a failure VOIDS the reading -- `e0j_read.py`): L linearity (direct backward at w2 vs the Gram, every batch + pooled;
         |share diff| <= 1e-4 AND ||sum w2 g - G2|| / ||G2|| <= 1e-4), Z (0.0 x box3d reads EXACTLY 0 in every group; batch 0),
         T (the trainer's own `losses["loss"]` gradient == sum_j w_cfg_j g_j: no hidden term; batch 0) and a value check
         |loss - sum_j w_cfg_j L_j| on every batch, S (sum_i s_i = 1), Gram symmetric PSD.
Env (caller, before import): PYTHONPATH=<tree>/stack;<tree>/taniteval, REFCV6_KIT=/home/nvidia,
         REFCV6_REMAP_OVERRIDES=<train-diag remap>. Prints sha12 / counts only; NO raw clip id is ever written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gram_core  # noqa: E402

TERMS = ("map", "box3d", "agent")
LOSS_KEY = {"map": "map_hires", "box3d": "box3d", "agent": "agent"}      # the UNWEIGHTED terms compute_losses_perc_rig returns
KINDS = TERMS + ("direct",)


def stage_argv(argv: list, stage: str) -> list:
    """C0 argv -> the stage's argv. banded: + --r9-map-banded, near-lift 20 -> 10 (both REQUIRED together). baseline8: C0 as is."""
    a = list(argv)
    if stage == "baseline8":
        if "--r9-map-banded" in a:
            raise SystemExit("[pgrad-gram] baseline8 argv already carries --r9-map-banded")
        return a
    if stage != "banded":
        raise SystemExit(f"[pgrad-gram] unknown stage {stage!r}")
    if "--r9-map-banded" in a:
        raise SystemExit("[pgrad-gram] the base argv already carries --r9-map-banded; pass the C0 argv")
    i = a.index("--map-hires-near-lift-m")
    a[i + 1] = "10"
    j = a.index("--r9-perc-rig")
    a.insert(j + 1, "--r9-map-banded")
    return a


def make_step0_ckpt(src: str, dst: str, banded: bool) -> dict:
    """The refcv8 EMA -> the rig step-0 state dict, by the rig's own predicate; under banded also drop `_map_hires.refine.cls.*`."""
    import torch
    from tanitad.refs import refcv9_perc_rig as R
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model"]
    keep = {k: v for k, v in sd.items() if not R.is_skipped_key(k)}
    unknown = [k for k in keep if not R.is_built_key(k)]
    if unknown:
        raise SystemExit(f"[pgrad-gram] {len(unknown)} source key(s) neither built nor skipped by the rig, e.g. {unknown[:3]}")
    dropped = []
    if banded:
        dropped = sorted(k for k in keep if k.startswith("_map_hires.refine.cls."))
        for k in dropped:
            del keep[k]
    torch.save({"model": keep, "step": ck.get("step"), "kind": "rig_step0_from_" + str(ck.get("kind", "unknown")),
                "source": str(src)}, dst)
    return {"src": src, "dst": dst, "n_source_keys": len(sd), "n_kept": len(keep), "banded_dropped": dropped,
            "step": ck.get("step"), "kind": ck.get("kind")}


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("banded", "baseline8"), required=True)
    ap.add_argument("--ckpt", required=True, help="refcv8 EMA-48,852 (ckpt_ema_48852.pt)")
    ap.add_argument("--base-config", required=True,
                    help="a rig run's config.json (the stamps the loader checks): banded -> r9c1-M1-smoke, baseline8 -> r9dump/rig_config.json")
    ap.add_argument("--argv-json", required=True, help="stack/ops/runs.d/r9e1_C0.argv.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w2", type=float, nargs=3, default=(4.0, 0.5, 2.0), metavar=("MAP", "BOX3D", "AGENT"),
                    help="the SECOND weight vector for the direct-backward linearity control (non-unit on every term)")
    ap.add_argument("--l-control", type=int, default=1, help="1 = direct backward at w2 on every batch (0 = skip: baseline8)")
    ap.add_argument("--z-batches", type=int, default=1)
    ap.add_argument("--t-batches", type=int, default=1)
    ap.add_argument("--scratch", default="/home/nvidia/r8ns/r9gram/scratch")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    t0 = time.time()
    dev = a.device
    # TF32 OFF everywhere (precision floor of the linearity control) and deterministic cuDNN.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    from tanitad.eval import refcv7_loader as L
    L.bootstrap()
    tr = L.trainer()
    import tanitad
    print(f"[pgrad-gram] tanitad from {tanitad.__file__}", flush=True)
    from tanitad.train import grad_conflict as gcf

    base = L.load_config(a.base_config)
    c0 = json.load(open(a.argv_json, encoding="utf-8"))
    config = copy.deepcopy(base)
    config["argv"] = stage_argv(c0["argv"], a.stage)
    banded = a.stage == "banded"
    os.makedirs(a.scratch, exist_ok=True)
    step0 = make_step0_ckpt(a.ckpt, str(Path(a.scratch) / f"rig_step0_{a.stage}.pt"), banded)
    model, cfg, args, mrec = L.build_model(config, step0["dst"], device=dev, strict=not banded)
    sdrec = mrec["state_dict"]
    if banded:
        bad_m = [k for k in sdrec["missing"] if not k.startswith("_map_hires.banded.")]
        bad_u = [k for k in sdrec["unexpected"] if not k.startswith("_map_hires.refine.cls.")]
        if bad_m or bad_u or not sdrec["missing"]:
            raise SystemExit(f"[pgrad-gram] banded load: missing outside the banded head {bad_m[:3]}, unexpected outside refine.cls "
                             f"{bad_u[:3]}, or no banded key missing at all ({len(sdrec['missing'])})")
    assert bool(getattr(args, "r9_map_banded", False)) == banded
    assert bool(getattr(getattr(model, "_map_hires", None), "banded", None) is not None) == banded, "banded head built/declared mismatch"
    assert bool(getattr(model.core, "_perc_rig", False)), "the model is not the perception rig"
    w_cfg = {"map": float(model._w_map_hires), "box3d": float(model._w_box3d), "agent": float(model._w_agent)}
    print(f"[pgrad-gram] built stage={a.stage} step {sdrec['step']} weights {w_cfg} missing {len(sdrec['missing'])} "
          f"unexpected {len(sdrec['unexpected'])} ({time.time() - t0:.0f}s)", flush=True)

    enc = model.core.encoder
    lv = getattr(getattr(enc, "trunk", enc), "memory_levers", None)
    if lv is None:
        lv = enc.memory_levers
    levers_before = dict(lv)
    lv["bf16"] = False                                       # FP32 trunk
    # ---- dataset: TRAIN-DIAG windows (the remap override points --eval-cache at the 139-clip train cache) ---------------- #
    args_ds = copy.copy(args)
    departures = list(mrec.get("departures", []))
    if not getattr(args_ds, "eval_labels", None):
        fb = L.PATH_REMAP.get("--eval-labels")
        departures.append(f"argv names no --eval-labels (the rig has no tactical labels); the loader's dataset needs them to build -> {fb} (no perception target reads them)")
        args_ds.eval_labels = fb
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args_ds, config)
    if hasattr(e_ds, "newest_row_only"):
        e_ds.newest_row_only = bool(e_ds.u8_frames)          # the trainer's own rig setting (refc_v3_train.py:10616)
    sha_of_eid = dict(getattr(e_ds, "_vis1_sha12", None) or {})
    if not sha_of_eid:
        from tanitad.data.v2_dataset import load_or_build_manifest, stable_episode_id
        sha_of_eid = {int(stable_episode_id(str(c))): L.sha12(str(c))
                      for c in load_or_build_manifest(args.eval_cache, verbose=False)["clip_id"]}
    by_ep = {}
    for i, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    grid = []
    for e_i in sorted(by_ep, key=lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)]):
        ts = sorted(by_ep[e_i])
        for j in range(8):
            grid.append(ts[int((j + 0.5) * len(ts) / 8)][1])
    n_pick = a.n_batches * a.batch
    rng = np.random.default_rng(a.seed)
    pick = [int(grid[k]) for k in rng.choice(len(grid), min(n_pick, len(grid)), replace=False)]
    win = [(sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)], int(e_ds.index[i][1])) for i in pick]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, pick), batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, drop_last=False)
    print(f"[pgrad-gram] dataset {len(e_eps)} episodes {len(e_ds)} windows, grid {len(grid)}, draw {len(pick)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- parameter groups (DISJOINT; whole trunk = the sum of the trunk groups) ------------------------------------------- #
    pref = ("core.encoder.", "_map_hires.lift.", "_map_hires.encoder.", "_perception.bev_pool.")
    for n, p in model.named_parameters():
        if n.startswith(pref):
            p.requires_grad_(True)                           # the eval loader froze everything; a gradient reaches the groups via activations
    groups, params = {}, []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("core.encoder."):
            g = gcf.default_group_of(n[len("core.encoder."):])
        elif n.startswith("_map_hires.lift."):
            g = "lift"
        elif n.startswith("_map_hires.encoder."):
            g = "map_encoder"
        elif n.startswith("_perception.bev_pool."):
            g = "planner_bev_pool"
        else:
            continue
        groups.setdefault(g, []).append(len(params))
        params.append(p)
    if not params:
        raise SystemExit("[pgrad-gram] no parameter requires grad -- refusing to measure the empty set")
    gsizes = {g: int(sum(params[i].numel() for i in ix)) for g, ix in groups.items()}
    trunk_groups = sorted(g for g in groups if g not in ("lift", "map_encoder", "planner_bev_pool"))
    print(f"[pgrad-gram] groups {gsizes}", flush=True)

    def vec(grads, g):
        return torch.cat([(grads[i] if grads[i] is not None else torch.zeros_like(params[i])).reshape(-1).float().cpu()
                          for i in groups[g]])

    def gradvecs(term_tensor):
        if term_tensor is None or not getattr(term_tensor, "requires_grad", False):
            return {g: torch.zeros(gsizes[g]) for g in groups}
        gr = torch.autograd.grad(term_tensor, params, retain_graph=True, allow_unused=True)
        out = {g: vec(gr, g) for g in groups}
        del gr
        return out

    w2 = dict(zip(TERMS, a.w2))
    V = {g: [] for g in groups}                              # per group: list over batches of [4, n_g] float32 (CPU)
    per_batch, ctl = [], {"Z_max_abs": 0.0, "Z_n": 0, "T_max_rel_err": 0.0, "T_n": 0, "value_max_rel": 0.0}
    model.train()
    for bi, b in enumerate(dl):
        tb = time.time()
        torch.manual_seed(1000 + bi)
        if dev.startswith("cuda"):
            torch.cuda.manual_seed_all(1000 + bi)
        losses = tr.compute_losses_v3(model, b, dev, mode=args.mode, ablate_frames=args.ablate_frames, log_metrics=False)
        T = {t: losses.get(LOSS_KEY[t]) for t in TERMS}
        present = {t: (T[t] is not None and getattr(T[t], "requires_grad", False)) for t in TERMS}
        vals = {t: (float(T[t].detach()) if T[t] is not None else None) for t in TERMS}
        tot = float(losses["loss"].detach())
        wsum = sum(w_cfg[t] * vals[t] for t in TERMS if vals[t] is not None)
        vrel = abs(tot - wsum) / max(1.0, abs(tot))
        ctl["value_max_rel"] = max(ctl["value_max_rel"], vrel)
        gv = {t: gradvecs(T[t]) for t in TERMS}
        row = {"batch": bi, "windows": win[bi * a.batch:(bi + 1) * a.batch], "term_present": present, "loss_values": vals,
               "total_value": tot, "value_rel_err": vrel,
               "n_map_labelled": float(losses.get("map_hires_n_labelled", float("nan"))),
               "n_map_windows": float(losses.get("map_hires_n_windows", float("nan"))),
               "n_agent_labelled": float(losses.get("agent_n_labelled", float("nan")))}
        # ---- L: ONE backward of the weighted total at w2 ------------------------------------------------------------------ #
        direct = None
        if a.l_control:
            tl = [w2[t] * T[t] for t in TERMS if present[t]]
            direct = gradvecs(sum(tl[1:], tl[0])) if tl else {g: torch.zeros(gsizes[g]) for g in groups}
        for g in groups:
            rows = [gv[t][g] for t in TERMS] + [direct[g] if direct is not None else torch.zeros(gsizes[g])]
            V[g].append(torch.stack(rows))
        # ---- Z: a weight-0 term reads EXACTLY 0 in every group ------------------------------------------------------------- #
        if bi < a.z_batches:
            zt = next((0.0 * T[t] for t in ("box3d", "map", "agent") if present[t]), None)
            zv = gradvecs(zt) if zt is not None else {g: torch.zeros(gsizes[g]) for g in groups}
            row["Z_max_abs"] = float(max(float(zv[g].abs().max()) for g in groups))
            ctl["Z_max_abs"] = max(ctl["Z_max_abs"], row["Z_max_abs"])
            ctl["Z_n"] += 1
            del zv
        # ---- T: the trainer's OWN total vs the three weighted terms (a hidden loss term shows here) ------------------------- #
        if bi < a.t_batches:
            tv = gradvecs(losses["loss"])
            rel = {}
            for g in groups:
                s = sum(w_cfg[t] * gv[t][g].double() for t in TERMS)
                tg = tv[g].double()
                rel[g] = float((s - tg).norm() / tg.norm()) if float(tg.norm()) > 0 else 0.0
            row["T_rel_err"] = rel
            ctl["T_max_rel_err"] = max(ctl["T_max_rel_err"], max(rel.values()))
            ctl["T_n"] += 1
            del tv
        row["wall_s"] = round(time.time() - tb, 1)
        per_batch.append(row)
        print(f"[pgrad-gram] ZZ{bi + 1}-{a.n_batches}ZZ {row['wall_s']:.0f}s present {present} value_rel {vrel:.1e} "
              f"nmap {row['n_map_labelled']:.0f}/{row['n_map_windows']:.0f} nagent {row['n_agent_labelled']:.0f} "
              f"cuda_max {(torch.cuda.max_memory_allocated() / 1e9) if dev.startswith('cuda') else 0:.1f} GB", flush=True)
        del losses, T, gv, direct

    # ---- the block matrix K per group: float64 dots in chunks ------------------------------------------------------------ #
    nb = len(per_batch)
    Kd = {}
    kdev = dev if dev.startswith("cuda") else "cpu"
    for g in groups:
        Kd[g] = gram_core.block_matrix(V[g], device=kdev).tolist()      # rows: batch-major (b, kind in KINDS)
        V[g] = None
    controls_raw = {"Z_zero_weight_max_abs": ctl["Z_max_abs"], "Z_n_batches": ctl["Z_n"], "T_hidden_term_max_rel_err": ctl["T_max_rel_err"],
                    "T_n_batches": ctl["T_n"], "value_check_max_rel": ctl["value_max_rel"]}
    rec = {"probe": "E0j / C11 P-GRAD Gram", "schema": "e0j-gram-v1", "stage": a.stage,
           "tier": "OPEN-LOOP PERCEPTION DIAGNOSTIC (gradient census at the E1 step-0; no plan scored; four families N/A)",
           "terms": list(TERMS), "kinds": list(KINDS), "w2": w2, "w_configured": w_cfg, "n_batches": nb, "batch": a.batch,
           "groups": {g: {"n_params": gsizes[g]} for g in groups}, "trunk_groups": trunk_groups, "K": Kd,
           "ckpt": a.ckpt, "ckpt_md5": md5_file(a.ckpt), "step0_ckpt": {k: v for k, v in step0.items() if k != "banded_dropped"},
           "banded_dropped_source_keys": len(step0["banded_dropped"]),
           "state_dict": {"strict": sdrec["strict"], "n_missing": len(sdrec["missing"]), "n_unexpected": len(sdrec["unexpected"]),
                          "missing_prefixes_ok": banded},
           "argv": config["argv"], "argv_json": a.argv_json, "base_config": a.base_config,
           "window_draw_sha256": hashlib.sha256(json.dumps(win).encode()).hexdigest(), "n_windows_grid": len(grid),
           "seed_draw": a.seed, "forward_seed": "torch.manual_seed(1000 + batch)",
           "measurement_choices": {"fp32_trunk": True, "trunk_levers_as_built": levers_before, "tf32": False, "cudnn_deterministic": True,
                                   "train_mode": True, "device": dev, "head_init": "fresh seed-0 init (banded) / refcv8 EMA weights (baseline8)"},
           "controls_raw": controls_raw, "per_batch": per_batch, "loader_departures": departures,
           "cuda_max_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev.startswith("cuda") else None,
           "wall_s": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "pgrad_gram.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, out / "pgrad_gram.json")
    print(f"[pgrad-gram] DONE stage={a.stage} Z {controls_raw['Z_zero_weight_max_abs']} T {controls_raw['T_hidden_term_max_rel_err']:.2e} "
          f"value {controls_raw['value_check_max_rel']:.1e} wall {rec['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
