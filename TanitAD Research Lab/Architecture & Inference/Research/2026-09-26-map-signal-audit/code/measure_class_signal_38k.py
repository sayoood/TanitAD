"""Map-signal audit, task 3: MEASURED per-class loss and gradient share of refcv6's map loss at step 38,000.

What runs (and nothing else):
  * the model is built by the EvalFlyWheel battery loader (`refcv6_loader.py`, tip blob md5 6b7d07ac...,
    the one the 35k map video used), STRICT load of `D:/refcv6_eval_kit/ckpt_final/ckpt_step38000.pt`
    (md5 5a2e7222..., verified here) under its own `config.json` (md5 a3193a46..., verified);
  * the forward is the trainer's own `compute_losses_v3`, stopped by a forward hook as soon as
    `model(...)` returns (the render tool's pattern) -- so `fmap_s16`, the lift geometry, the map logits
    and `map_valid` are exactly what the trainer's loss would have read;
  * the perception branch is then RE-RUN from a detached copy of `fmap_s16` with autograd on
    ACTIVATIONS only (no parameter gradient, no optimiser, nothing written to the model):
        f (= fmap_s16) -> lift -> bev -> map_branch -> logits
    and the map loss is decomposed per class c:  L = sum_c L_c,  L_c = sum_cells p_c (-log q_c) / n
    over the trainer's supervised cells (map_seen AND the lift's map_valid, refc_v3_train.py:4472-4481).
    For each c: dL_c/d logits, dL_c/d bev (the lift output) and dL_c/d f (what the TRUNK receives).
Departures (recorded in the output): CPU fp32 trunk (bf16/NHWC levers OFF -- CPU autocast-bf16 is
~500x slower, battery smoke_cpu.py:39-50), `--trunk-compile` dropped by the loader, no future frames
decoded, agent/3-D joins not attached (loss-only targets the forward never reads).
Controls (must read known values, asserted): constant-logit model = log 9 exactly and mass-share
attribution; prior-logit model = H(mass); GT-as-logits ~ the entropy floor; the re-run logits equal the
in-forward logits; the re-run loss equals `refcv6_perception_branch.map_loss_row`. Plus the GT-SHUFFLED
control (each window scored against another window's labels, a fixed derangement).

RAM rules (the dev box is shared tonight): refuses to start unless >= 7.5 GB free on 3 samples 30 s
apart; a watchdog thread aborts THIS process (os._exit) if free RAM drops below 4.0 GB. GPU only if
--device cuda is passed explicitly (the caller checks the GPU gate). Clip ids never leave the process
(sha12 only).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
TREE = Path(os.environ.get("MSA_TREE", "C:/Users/Admin/msa_tree_2210"))
LOADER = TREE / "refcv6_loader_tip.py"
LOADER_MD5 = "6b7d07aceac9fd7002cebb4acc1be5ed"
KIT = Path("D:/refcv6_eval_kit")
CKPT = KIT / "ckpt_final/ckpt_step38000.pt"
CKPT_MD5 = "5a2e7222a9f5f8c7aa7bf38ef4698d8a"
CONFIG = KIT / "ckpt_final/config.json"
CONFIG_MD5 = "a3193a4685ce0d07a6ae6b89fdc994a8"
GT_DIR = KIT / "data/sam3_gt_eval_thor137"
#: dev box: the kit (flat sha12 GT, argv remapped to the kit). Thor (--site thor): the run's own paths.
GT_LAYOUT = "flat_sha12"
GT_ROOT = None
REMAP = None                         # None = the loader's PATH_REMAP (kit); {} = the run's own argv paths
C9 = ["seen-no-class", "drivable", "lane/road line", "crosswalk", "arrow/text", "non-drivable edge",
      "hatched", "sidewalk/verge", "not-seen"]
ENRICH = [3, 4, 6, 5]                # crosswalk, arrow/text, hatched, edge: 2 windows each
START_FREE_GB, ABORT_FREE_GB = 7.5, 4.0


def free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:                      # Thor's tanitad-train venv has no psutil (no pip installs)
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 2 ** 20
        raise


def ram_gate(log) -> list:
    s = []
    for k in range(3):
        g = free_gb()
        s.append(round(g, 2))
        log(f"[gate] free RAM sample {k + 1}/3: {g:.2f} GB")
        if g < START_FREE_GB:
            raise SystemExit(f"[gate] free RAM {g:.2f} GB < {START_FREE_GB} GB -- refusing to start")
        if k < 2:
            time.sleep(30)
    return s


def watchdog(log, stop: threading.Event, rec: dict):
    lo = 1e9
    while not stop.is_set():
        g = free_gb()
        lo = min(lo, g)
        rec["min_free_gb_seen"] = round(lo, 2)
        if g < ABORT_FREE_GB:
            log(f"[watchdog] free RAM {g:.2f} GB < {ABORT_FREE_GB} GB -- ABORTING this process")
            sys.stdout.flush()
            os._exit(3)
        stop.wait(2.0)


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def load_loader():
    if md5_file(LOADER) != LOADER_MD5:
        raise SystemExit(f"[probe] {LOADER} md5 != {LOADER_MD5}")
    os.environ["REFCV6_REPO"] = str(TREE)
    os.environ["REFCV6_KIT"] = str(KIT)
    spec = importlib.util.spec_from_file_location("refcv6_loader_msa", str(LOADER))
    L = importlib.util.module_from_spec(spec)
    sys.modules["refcv6_loader_msa"] = L
    spec.loader.exec_module(L)
    L.bootstrap()
    import tanitad
    here = os.path.normcase(os.path.abspath(tanitad.__file__))
    if not here.startswith(os.path.normcase(os.path.abspath(str(TREE)))):
        raise SystemExit(f"[probe] tanitad from {here}, not from {TREE}")
    return L


class _Captured(Exception):
    pass


def forward_like_trainer(tr, model, batch, device, *, mode, ablate_frames, seed=0):
    """render_refcv6_map_video.py:886-904, verbatim in behaviour."""
    import torch
    box = {}

    def hook(_m, _inp, out):
        box["out"] = out
        raise _Captured()
    h = model.register_forward_hook(hook)
    torch.manual_seed(int(seed))
    try:
        with torch.no_grad():
            tr.compute_losses_v3(model, batch, device, mode=mode, ablate_frames=ablate_frames)
        raise SystemExit("[probe] compute_losses_v3 returned without the forward hook firing")
    except _Captured:
        pass
    finally:
        h.remove()
    return box["out"]


def per_class_decomposition(torch, F, br, f16, grid, valid, frac, seen, mvalid, tag: str) -> dict:
    """Re-run lift -> map branch from f16 (activations only) and split the map loss per class."""
    with torch.enable_grad():
        f = f16.detach().clone().float().requires_grad_(True)
        bev = br.lift(f, grid, valid)
        mb = br.map_branch(bev)
        z = mb["map_logits"]
        m = (seen & mvalid)                                      # [B, X, Y]
        n = int(m.sum())
        p = frac / frac.sum(1, keepdim=True).clamp_min(1e-6)
        logq = F.log_softmax(z, dim=1)
        mf = m.unsqueeze(1).to(z.dtype)
        Lc = -(p * logq * mf).sum(dim=(0, 2, 3)) / max(n, 1)    # [C]
        L = Lc.sum()
        gz, gb, gf = torch.autograd.grad(L, [z, bev, f], retain_graph=True)
        rows = []
        for c in range(z.shape[1]):
            gzc, gbc, gfc = torch.autograd.grad(Lc[c], [z, bev, f], retain_graph=True)
            rows.append({
                "class": C9[c], "loss_c": float(Lc[c]),
                "grad_logits_norm": float(gzc.norm()),
                "grad_logits_proj_share": float((gzc * gz).sum() / (gz * gz).sum()),
                "grad_lift_out_norm": float(gbc.norm()),
                "grad_lift_out_proj_share": float((gbc * gb).sum() / (gb * gb).sum()),
                "grad_trunk_fmap_s16_norm": float(gfc.norm()),
                "grad_trunk_fmap_s16_proj_share": float((gfc * gf).sum() / (gf * gf).sum()),
                # the TOTAL loss's gradient on logit channel c (what the head row for c receives)
                "total_grad_on_logit_channel_c_norm": float(gz[:, c].norm()),
            })
        q = logq.exp().detach()
    out = {"tag": tag, "n_cells": n, "loss": float(L),
           "grad_logits_norm": float(gz.norm()), "grad_lift_out_norm": float(gb.norm()),
           "grad_trunk_fmap_s16_norm": float(gf.norm()), "per_class": rows}
    # label/prediction statistics on the supervised cells
    with torch.no_grad():
        lab = frac.argmax(1)[m]
        pred = z.detach().argmax(1)[m]
        qs = q.permute(1, 0, 2, 3)[:, m]                       # [C, n]
        ps = p.permute(1, 0, 2, 3)[:, m]
        for c, r in enumerate(out["per_class"]):
            r["mass"] = float(ps[c].sum() / max(n, 1))
            r["n_cells_argmax_label"] = int((lab == c).sum())
            r["n_cells_argmax_pred"] = int((pred == c).sum())
            inter = int(((lab == c) & (pred == c)).sum())
            union = int(((lab == c) | (pred == c)).sum())
            r["iou_argmax"] = (inter / union) if union else None
            hi = ps[c] >= 0.5
            r["n_cells_p_ge_0.5"] = int(hi.sum())
            r["mean_q_where_p_ge_0.5"] = float(qs[c][hi].mean()) if bool(hi.any()) else None
            r["mean_q_where_p_0"] = float(qs[c][ps[c] == 0].mean()) if bool((ps[c] == 0).any()) else None
            r["max_q"] = float(qs[c].max()) if n else None
            r["frac_label_cells_where_pred_prob_is_top"] = (
                float((qs.argmax(0)[lab == c] == c).float().mean()) if bool((lab == c).any()) else None)
            # the analytic-table definition (analytic_class_signal.py): A_c = sum p_c ||q - e_c|| / n
            e_norm = torch.sqrt(((qs ** 2).sum(0) - 2 * qs[c] + 1.0).clamp_min(0.0))
            r["attrib_A"] = float((ps[c] * e_norm).sum() / max(n, 1))
            r["own_pull_P"] = float((ps[c] * (1 - qs[c])).sum() / max(n, 1))
        tA = sum(r["attrib_A"] for r in out["per_class"])
        tP = sum(r["own_pull_P"] for r in out["per_class"])
        for r in out["per_class"]:
            r["attrib_A_share"] = r["attrib_A"] / tA if tA else None
            r["own_pull_P_share"] = r["own_pull_P"] / tP if tP else None
    return out


def controls_constant(torch, F, frac, seen, mvalid) -> dict:
    """Constant logits through the SAME arithmetic as the probe: the analytic values must come back.

    uniform z = 0:  L = log C;  L_c = mass_c log C;  dL/dz = (1/C - p) m / n  (every cell, every class);
                    ||dL_c/dz||_F = sqrt((1-1/C)^2 + (C-1)/C^2) * sqrt(sum_cells p_c^2) / n.
    prior   z = log(mass):  L = H(mass).
    """
    m = seen & mvalid
    n = int(m.sum())
    C = frac.shape[1]
    p = frac / frac.sum(1, keepdim=True).clamp_min(1e-6)
    mf = m.unsqueeze(1).to(p.dtype)
    mass = (p * mf).sum(dim=(0, 2, 3)) / n
    res = {"n_cells": n}
    for name, zc in (("uniform", torch.zeros(C, dtype=p.dtype, device=p.device)),
                     ("prior", torch.log(mass.clamp_min(1e-12)))):
        z = zc.view(1, -1, 1, 1).expand_as(p).clone().requires_grad_(True)
        logq = F.log_softmax(z, dim=1)
        Lc = -(p * logq * mf).sum(dim=(0, 2, 3)) / n
        L = Lc.sum()
        (gz,) = torch.autograd.grad(L, [z], retain_graph=True)
        want = (math.log(C) if name == "uniform"
                else float(-(mass * torch.log(mass.clamp_min(1e-12))).sum()))
        r = {"loss": float(L), "want": want, "abs_err": abs(float(L) - want)}
        if name == "uniform":
            r["max_abs_Lc_minus_mass_logC"] = float((Lc - mass * math.log(C)).abs().max())
            g_want = (1.0 / C - p) * mf / n
            r["max_abs_grad_minus_analytic"] = float((gz - g_want).abs().max())
            k = math.sqrt((1 - 1 / C) ** 2 + (C - 1) / C ** 2)
            errs = []
            for c in range(C):
                (gzc,) = torch.autograd.grad(Lc[c], [z], retain_graph=True)
                want_c = k * float(torch.sqrt(((p[:, c] ** 2) * m).sum())) / n
                errs.append(abs(float(gzc.norm()) - want_c) / max(want_c, 1e-30))
            r["max_rel_err_per_class_grad_norm"] = max(errs)
        res[name] = r
    # GT-as-logits: z = log(p + 1e-6) -> L ~ entropy floor
    z = torch.log(p + 1e-6)
    logq = F.log_softmax(z, dim=1)
    L = float(-(p * logq * m.unsqueeze(1)).sum() / n)
    H = float(-(p * torch.log(p.clamp_min(1e-12)) * m.unsqueeze(1)).sum() / n)
    res["gt_as_logits"] = {"loss": L, "entropy_floor": H, "abs_err": abs(L - H)}
    return res


def select_windows(L, tr, model, cfg, targs, config, log) -> tuple:
    """The fixed rule (declared before any model output was read):
    NEUTRAL: the 8 smallest-sha12 eval clips whose every eval window has a SAM3 frame; per clip the
             median window (by t).
    ENRICHED: for crosswalk, arrow/text, hatched, non-drivable edge in that order: the 2 windows with the
             largest soft label mass of the class on supervised cells (seen AND the clip's lift
             map_valid), from clips not already chosen, ties by (sha12, t).  GT only; no model output.
    """
    import argparse as _ap
    import numpy as np
    import torch
    from tanitad.data import v2_dataset as v2d
    from tanitad.data import semantic_map_gt as smg

    class MapWindows(tr.V3Dataset):
        u8_frames = True

        def _window_u8(self, i: int) -> dict:
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w],
                    "future_frames": torch.zeros(0, dtype=torch.uint8),
                    "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                    "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                    "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}

    def clip_of(ep):
        fp = ep.frames
        return Path(fp._cache.files[fp._clip]).name.split(".")[0]

    s_ds, s_eps, _ = L.build_eval_dataset(model, cfg, targs, config, with_perception_targets=False,
                                          dataset_cls=MapWindows)
    man = v2d.load_or_build_manifest(targs.eval_cache, verbose=False)
    ns = sorted(set(int(x) for x in man["n_stack"]))
    assert len(ns) == 1, ns
    n_stack = ns[0]
    W = int(cfg.core.window)
    by_ep: dict = {}
    for wi, (e_i, t) in enumerate(s_ds.index):
        by_ep.setdefault(e_i, []).append((int(t), wi))
    cands = {}
    for e_i, ep in enumerate(s_eps):
        cid = clip_of(ep)
        s12 = sha12(cid)
        # the kit's flat sha12 copy on the dev box; the canonical corpus layout on Thor
        gp = (GT_DIR / f"{s12}.sam3mapgt.npz" if GT_LAYOUT == "flat_sha12"
              else smg.gt_path(GT_ROOT, cid))
        if not gp.is_file() or e_i not in by_ep:
            continue
        gt = smg.open_path(gp, cid)
        raws = [t + W - 1 + n_stack - 1 for t, _ in by_ep[e_i]]
        if max(raws) >= gt.n_frames:
            continue
        lv = model._lift_bank.geometry(int(ep.episode_id))[1].any(0).cpu().numpy()
        u8 = gt.cart_u8()
        rows = []
        for t, wi in sorted(by_ep[e_i]):
            r = t + W - 1 + n_stack - 1               # refc_v3_train.py:3185 + perception_targets raw_frames
            a = u8[r].astype(np.int32)
            seen = 2 * (255 - a[8]) >= 255
            p = a.astype(np.float64) / 255.0
            p = p / np.maximum(p.sum(0, keepdims=True), 1e-6)
            mm = seen & lv
            rows.append((t, wi, [float(p[c][mm].sum()) for c in range(9)]))
        cands[s12] = {"cid": cid, "e_i": e_i, "rows": rows}
        del gt
    order = sorted(cands)
    chosen = []
    for s12 in order[:8]:
        rows = sorted(cands[s12]["rows"])
        t, wi, mass = rows[len(rows) // 2]
        chosen.append({"group": "neutral", "sha12": s12, "t": t, "wi": wi, "why": "median window",
                       "mass_cells": mass})
    used = {c["sha12"] for c in chosen}
    for c in ENRICH:
        pool = sorted(((-r[2][c], s12, r[0], r[1], r[2]) for s12 in order if s12 not in used
                       for r in cands[s12]["rows"]))
        k = 0
        for negm, s12, t, wi, mass in pool:
            if s12 in used:
                continue
            chosen.append({"group": "enriched", "sha12": s12, "t": t, "wi": wi,
                           "why": f"max mass of {C9[c]}", "mass_cells": mass})
            used.add(s12)
            k += 1
            if k == 2:
                break
    keep = {cands[c["sha12"]]["cid"] for c in chosen}
    sel = {"rule": select_windows.__doc__.strip(), "n_candidate_clips": len(cands), "n_stack": n_stack,
           "window": W, "chosen": chosen}
    log(f"[select] {len(cands)} candidate clips; chosen {[(c['group'][0], c['sha12'], c['t']) for c in chosen]}")
    return sel, keep, MapWindows, clip_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=16)
    ap.add_argument("--no-gate", action="store_true", help="tests only; never on the shared box")
    ap.add_argument("--site", choices=("devbox", "thor"), default="devbox",
                    help="thor: the run's own checkpoint/config/argv paths and the canonical GT layout")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    global RAW, CKPT, CONFIG, GT_LAYOUT, GT_ROOT, REMAP
    if a.site == "thor":
        run = Path("/home/nvidia/refcv6_run/runs/refcv6-r101-s0")
        CKPT, CONFIG = run / "ckpt.pt", run / "config.json"
        GT_LAYOUT, GT_ROOT, REMAP = "canonical", Path("/home/nvidia/data/sam3_corpus"), {}
    if a.out_dir:
        RAW = Path(a.out_dir)
    RAW.mkdir(parents=True, exist_ok=True)
    logf = open(RAW / "measure_class_signal_38k.log", "a", encoding="utf-8")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    rec: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": a.device, "site": a.site,
                 "ckpt_path": str(CKPT), "config_path": str(CONFIG), "gt_layout": GT_LAYOUT,
                 "argv_remap": "run's own paths" if REMAP == {} else "eval kit (loader PATH_REMAP)"}
    if not a.no_gate:
        rec["ram_gate_samples_gb"] = ram_gate(log)
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(log, stop, rec), daemon=True).start()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(a.threads))
    t0 = time.time()
    for p, want in ((CKPT, CKPT_MD5), (CONFIG, CONFIG_MD5)):
        got = md5_file(p)
        if got != want:
            raise SystemExit(f"[probe] {p} md5 {got} != {want}")
    rec["ckpt"] = {"path": str(CKPT), "md5": CKPT_MD5}
    rec["config"] = {"path": str(CONFIG), "md5": CONFIG_MD5}
    L = load_loader()
    tr = L.trainer()
    import tanitad
    rec["tanitad_file"] = tanitad.__file__
    rec["tree_tip_sha"] = (TREE / "TIP_SHA.txt").read_text().strip() if (TREE / "TIP_SHA.txt").exists() else None
    config = L.load_config(CONFIG)
    _tl = torch.load

    def _tl_mmap(f, *aa, **kk):
        if str(f) == str(CKPT):
            kk.setdefault("mmap", True)
        return _tl(f, *aa, **kk)
    torch.load = _tl_mmap
    try:
        model, cfg, targs, mrec = L.build_model(config, str(CKPT), device=a.device, remap=REMAP)
    finally:
        torch.load = _tl
    sd = mrec["state_dict"]
    if sd["missing"] or sd["unexpected"] or int(sd.get("step") or -1) != 38000:
        raise SystemExit(f"[probe] strict load not clean / wrong step: {sd['missing'][:3]} "
                         f"{sd['unexpected'][:3]} step {sd.get('step')}")
    if not mrec["param_breakdown"]["equal"]:
        raise SystemExit("[probe] param_breakdown differs from config.json")
    levers_changed = []
    if a.device == "cpu":
        for mod in model.modules():
            lv = getattr(mod, "memory_levers", None)
            if isinstance(lv, dict) and "bf16" in lv:
                if lv.get("bf16") or lv.get("channels_last"):
                    levers_changed.append(type(mod).__name__)
                lv["bf16"] = False
                lv["channels_last"] = False
    _tr = [m_ for m_ in model.modules() if type(m_).__name__ == "TimmResNetTrunk"]
    rec["model"] = {"strict_missing": 0, "strict_unexpected": 0, "step": 38000,
                    "param_breakdown_equal": True, "build_s": mrec.get("build_s"),
                    "map_lift_valid_mask": bool(getattr(model, "_map_lift_valid_mask", True)),
                    "equalize_bottom_rows": {
                        "argv": int(getattr(targs, "equalize_bottom_rows", 0) or 0),
                        "trunk_built": (int(getattr(getattr(_tr[0], "cfg", None), "equalize_bottom_rows", 0) or 0)
                                        if _tr else None),
                        "lift_bank": int(getattr(model._lift_bank, "equalize_bottom_rows", 0) or 0)},
                    "cpu_fp32_levers_switched_off_on": levers_changed}
    log(f"[probe] model built in {time.time() - t0:.0f} s; {json.dumps(rec['model'])}")
    mode = getattr(targs, "mode", "diffusion")
    ablate = bool(getattr(targs, "ablate_frames", False))
    # ---- window selection (GT + geometry only) -------------------------------------------------- #
    import argparse as _ap2
    targs_sel = _ap2.Namespace(**vars(targs))
    targs_sel.agent_join = None
    targs_sel.join3d = None
    sel, keep_cids, MapWindows, clip_of = select_windows(L, tr, model, cfg, targs_sel, config, log)
    json.dump(sel, open(RAW / "measure_class_signal_38k_selection.json", "w", encoding="utf-8"), indent=1)
    # ---- the dataset with map targets, ONLY the chosen clips ------------------------------------- #
    from tanitad.data import v2_dataset as v2d
    _orig = v2d.build_v2_providers

    def _only(*aa, **kk):
        eps = _orig(*aa, **kk)
        k = [e for e in eps if clip_of(e) in keep_cids]
        if len(k) != len(keep_cids):
            raise SystemExit(f"[probe] provider filter kept {len(k)} of {len(keep_cids)}")
        return k
    v2d.build_v2_providers = _only
    try:
        e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, targs_sel, config,
                                                 with_perception_targets=True, dataset_cls=MapWindows)
    finally:
        v2d.build_v2_providers = _orig
    # map (sha12, t) -> index in the filtered dataset
    idx_of = {}
    for wi, (e_i, t) in enumerate(e_ds.index):
        idx_of[(sha12(clip_of(e_eps[e_i])), int(t))] = wi
    br = model._perception
    rows = []
    batches = []
    for k, ch in enumerate(sel["chosen"][:a.max_windows]):
        wi = idx_of[(ch["sha12"], ch["t"])]
        item = e_ds[wi]
        batch = __import__("torch").utils.data.default_collate([item])
        if not bool(batch["map_label"][0]):
            raise SystemExit(f"[probe] window {ch['sha12']}@{ch['t']} has no map label")
        tw = time.time()
        out = forward_like_trainer(tr, model, batch, a.device, mode=mode, ablate_frames=ablate, seed=0)
        pout = out["perception"]
        f16 = out["fmap_s16"].detach()
        grid, valid = model._lift_bank.for_episodes(batch["map_ep"], device=a.device)
        frac = batch["map_frac"].to(a.device).float()
        seen = batch["map_seen"].to(a.device).bool()
        mvalid = pout["map_valid"].detach().bool()
        dec = per_class_decomposition(torch, F, br, f16, grid, valid, frac, seen, mvalid, "true_gt")
        # control: re-run logits == in-forward logits; loss == map_loss_row
        with torch.enable_grad():
            z_re = br.map_branch(br.lift(f16.float(), grid, valid))["map_logits"].detach()
        dec["ctl_rerun_logits_max_abs_diff"] = float((z_re - pout["map_logits"].detach().float()).abs().max())
        mrow = tr._perc.map_loss_row(pout["map_logits"].detach().float(), frac, seen, lift_valid=mvalid)
        dec["ctl_map_loss_row"] = float(mrow["loss"])
        dec["ctl_map_loss_row_abs_diff"] = abs(float(mrow["loss"]) - dec["loss"])
        dec["ctl_constant"] = controls_constant(torch, F, frac, seen, mvalid)
        dec.update({"group": ch["group"], "sha12": ch["sha12"], "t": ch["t"], "why": ch["why"],
                    "forward_s": round(time.time() - tw, 1)})
        rows.append(dec)
        batches.append({"f16": f16.cpu(), "grid": grid.cpu(), "valid": valid.cpu(), "frac": frac.cpu(),
                        "seen": seen.cpu(), "mvalid": mvalid.cpu(), "group": ch["group"]})
        del out, pout
        log(f"[probe] {k + 1}/{min(len(sel['chosen']), a.max_windows)} {ch['group']} {ch['sha12']}@{ch['t']}: "
            f"loss {dec['loss']:.4f} (map_loss_row {dec['ctl_map_loss_row']:.4f}), n {dec['n_cells']}, "
            f"rerun diff {dec['ctl_rerun_logits_max_abs_diff']:.2e}, {dec['forward_s']} s, "
            f"free {free_gb():.2f} GB")
    # ---- GT-shuffled control: window i's features/geometry vs window j's labels (derangement within group)
    shuf = []
    for grp in ("neutral", "enriched"):
        ids = [i for i, b in enumerate(batches) if b["group"] == grp]
        for a_i, i in enumerate(ids):
            j = ids[(a_i + 1) % len(ids)]
            b, bj = batches[i], batches[j]
            d = per_class_decomposition(torch, F, br, b["f16"].to(a.device), b["grid"].to(a.device),
                                        b["valid"].to(a.device), bj["frac"].to(a.device),
                                        bj["seen"].to(a.device), b["mvalid"].to(a.device), f"shuffled {i}<-{j}")
            d.update({"group": grp, "features_of": rows[i]["sha12"], "labels_of": rows[j]["sha12"]})
            shuf.append(d)
    rec["windows"] = rows
    rec["gt_shuffled"] = shuf
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["free_gb_end"] = round(free_gb(), 2)
    rec["departures"] = [
        "CPU fp32 trunk (bf16/NHWC levers switched off; as-trained was bf16 on CUDA)" if a.device == "cpu"
        else "GPU as trained (bf16 trunk)", "--trunk-compile dropped (loader)",
        "future frames not decoded; agent/3-D joins not attached (loss-only)",
        "forward = compute_losses_v3 stopped by a forward hook after model(...) returns",
        "per-class gradients are ACTIVATION gradients from a re-run of lift -> map branch on a detached "
        "fmap_s16; no parameter gradient, no optimiser"]
    stop.set()
    json.dump(rec, open(RAW / "measure_class_signal_38k.json", "w", encoding="utf-8"), indent=1)
    log(f"[probe] done in {rec['elapsed_s']} s; min free RAM seen {rec.get('min_free_gb_seen')} GB")


if __name__ == "__main__":
    main()
