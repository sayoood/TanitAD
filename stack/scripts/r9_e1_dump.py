#!/usr/bin/env python3
"""r9_e1_dump.py -- the pack / raster PRODUCER for a RIG (or refcv8) checkpoint (refcv9 E1; BRIEF_E1_DUMP).

    python r9_e1_dump.py --ckpt <ckpt> --config <run config.json> --windows <DEV_HOLD_SPLIT.json> [--split dev|hold]
                         --s2-windows <S2 {clip: [ws]}> --out <dir>

It writes, in EXACTLY the formats ``r9_e1_eval.py`` reads and the banked producers wrote:
  * ``<out>/perc_det.pkl``   PERC8's dump (``perc_pass_r7.py`` -> ``perc_pass_r8.py``): ``[{"sha12", "ws", "box3d": [pack]}, ...]``
                             (one record per window; every head ``detection_metrics.HEADS`` that produced a pack is stored);
  * ``<out>/rast/e<e_i:03d>_w<ws:04d>.npz``  MAPSEL's rasters (``mapsel_roll.py``): uint8 planes ``pdrv``, ``pedge``, ``psw``,
                             ``code`` (1000 x 600, 10 cm, the model's OWN planes under its declared prior-corrected rule) and ``gt``
                             (the SAM3 GT code plane, 255 = unobserved) when the window carries a 10 cm label;
  * ``<out>/dump.json``      the record (checkpoint md5 / step / kind, config md5, window-set digest, pose-sync state, the
                             future-frame control, timing). sha12 only -- NO raw clip id is ever written.

REUSE, NOT RE-IMPLEMENTATION. The banked producers are scripts, not importable modules, so the pieces are VERBATIM COPIES held
equal to the originals by ``tests/test_r9_e1_dump.py`` (the way ``r9_e1_eval.py`` pins ``census_one`` / ``nearfield_prec``):
  ``make_perc_dataset_cls``, ``_perc_tensors``           <- perc_pass_r7.py  (PERC8 = r8_repoint + this)
  ``LawOnly`` / ``LawOnlyDevice`` / ``patch_frames_to_device`` / ``collate``  <- g0_refcv7.py
  ``own_planes``                                          <- mapsel_roll.py   (+ the savez layout of its ``MapSelProxy``)
Inference only: no_grad, eval mode (the loader freezes every parameter), vision-only inputs, the future frames are ZEROS. The
weights are the checkpoint's own: an EMA file (``kind: ema_model_only``, e.g. ``ckpt_ema_48852.pt`` / a run's ``model_ema.pt``) is
the EMA; a full ``ckpt.pt`` carries only the raw weights (its ``ema`` block holds no tensors) and is RECORDED as raw (``--require-ema``
refuses it).

POSE CLOCK. PERC8 and MAPSEL both ran with ``R8_NO_POSE_SYNC=1`` (unshifted pose clock). The tip loader enables the pose-sync
sidecar whenever the run's argv names one, so this script nulls it by default (``--pose-sync off``) to read the banked producers'
clock; ``--pose-sync on`` is the run's own eval clock. The state is recorded.

``compare`` sub-command (the known-answer reads): ``r9_e1_dump.py compare --packs-a A --packs-b B [--rast-a D --rast-b D]``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

SCHEMA = "r9e1-dump-v1"
DRV_CODES = (1, 2, 3, 4, 6)          # == mapsel_roll.DRV_CODES (pinned by the test through own_planes' behaviour)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ================================================================================================================ VERBATIM COPIES
# ---- perc_pass_r7.py ------------------------------------------------------------------------------------------------------------
def make_perc_dataset_cls(tr, law_ahead: int):
    """The trainer's V3Dataset; the future frames are ZEROS (never decoded). See the module docstring."""
    class PercWindows(tr.V3Dataset):
        def _window_u8(self, i: int) -> dict:          # refb_train.py:220-231, future frames zeroed
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            fr = ep.frames[t:t + w]
            return {
                "frames": fr,
                "actions": ep.actions[t:t + w],
                "future_frames": torch.zeros((law_ahead,) + tuple(fr.shape[1:]), dtype=fr.dtype),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }
    return PercWindows


def _perc_tensors(out) -> dict:
    p = out.get("perception") or {}
    d = {}
    if p.get("map_hires_logits") is not None:
        d["map_hires_logits"] = p["map_hires_logits"]
    bs = p.get("box_slots") or {}
    for k in ("presence_logit", "box", "cls_logits", "cz"):
        if torch.is_tensor(bs.get(k)):
            d[f"box_slots.{k}"] = bs[k]
    return d


# ---- g0_refcv7.py ---------------------------------------------------------------------------------------------------------------
class LawOnly:
    def __init__(self, u8_frame, law_idx):
        self.u8 = u8_frame
        self.law_idx = int(law_idx)


class LawOnlyDevice:
    def __init__(self, dev_frame, law_idx):
        self.x = dev_frame
        self.law_idx = law_idx

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)
                and int(key[1]) == self.law_idx):
            raise KeyError(f"LawOnlyDevice holds only [:, {self.law_idx}], asked {key}")
        return self.x


_PATCHED = {}


def patch_frames_to_device(tr):
    if "f2d" in _PATCHED:
        return
    orig = tr.frames_to_device

    def f(x, device):
        if isinstance(x, LawOnly):
            return LawOnlyDevice(orig(x.u8, device), x.law_idx)
        return orig(x, device)
    tr.frames_to_device = f
    _PATCHED["f2d"] = orig


def collate(e_ds, idx_list, law_idx):
    import torch.utils.data as tud
    items = [e_ds[i] for i in idx_list]
    eb = tud.default_collate(items)
    ff = eb["future_frames"]
    eb["future_frames"] = LawOnly(ff[:, law_idx].clone(), law_idx)
    del ff
    return eb


# ---- refcv9 C5 lever 4 (--r9-box-perframe-queries): the per-row rig camera, staged BEFORE every forward --------------------------
def stage_perframe_queries(model, eb, device) -> bool:
    """Stage the clip's own rig camera on the 300-slot seed module (``box_c5_flag.stage``, ONE-SHOT) BEFORE a forward.

    ONLY a built model that carries per-frame queries (``model._perception.box_pfq``) is touched: for every other arm this is a pure attribute probe that
    returns False, so the default dump's argv, forward and outputs are byte-identical to the tip's.  The batch is the collated eval window, whose
    ``agent_ep`` the eval dataset emits alongside the agent join (``refcv7_loader.build_eval_dataset`` with ``with_perception_targets=True``); a pfq model
    given a batch WITHOUT it REFUSES inside ``stage`` (SystemExit), and a forward that nothing staged RAISES in ``PerFrameQueries.forward`` (RuntimeError):
    the contract is fail-loud, never a silent global-anchor-only dump.  ``compute_losses_v3`` stages too (same call, idempotent overwrite); this call is the
    dump's OWN guarantee, so the dump no longer depends on which trainer function it happens to drive."""
    br = getattr(model, "_perception", None)
    if br is None or getattr(br, "box_pfq", None) is None:
        return False
    from tanitad.train import box_c5_flag as _bc5
    _bc5.stage(model, eb, device)
    return True


# ---- mapsel_roll.py -------------------------------------------------------------------------------------------------------------
#: the counters ``own_planes`` writes (mapsel_roll's ``ST["c"]``); the dump reads them back into dump.json
ST = {"c": {"calls": 0, "rule_seen": set(), "decide_checked": 0, "decide_mismatch_cells": 0}}


def own_planes(model, logits_r):
    """logits_r [1, 8, H, W] -> (pdrv_u8, pedge_u8, psw_u8, code_u8) numpy [H, W], under the model's DECLARED rule: p^ = softmax(z - log w) (w = the loss's frozen class weights),
    code = argmax(z - log w) (== map_head_hires.decide('prior_corrected', w), asserted on a sample), pdrv = sum of p^ over the DRIVABLE codes."""
    from tanitad.models import map_head_hires as MH
    cw = model._map_hires_class_weight
    rule = str(model._map_hires.cfg.decision_rule)
    ST["c"]["rule_seen"].add(rule)
    if rule != "prior_corrected" or cw is None:
        raise RuntimeError(f"[mapsel] the declared decision rule is {rule!r} / class weights {'absent' if cw is None else 'present'}: this arm is defined for prior_corrected + weights")
    z = logits_r.detach().float()
    s = z - MH._decision_logw(cw, z).view(1, -1, 1, 1)
    p = torch.softmax(s, 1)[0]
    code = s.argmax(1)[0]
    if ST["c"]["decide_checked"] < 25 or ST["c"]["calls"] % 100 == 0:
        ref = MH.decide(z, rule, cw)[0]
        ST["c"]["decide_checked"] += 1
        ST["c"]["decide_mismatch_cells"] += int((ref != code).sum())
    pdrv = p[list(DRV_CODES)].sum(0)
    q = lambda t: t.mul(255.0).round().clamp_(0, 255).to(torch.uint8).cpu().numpy()   # noqa: E731
    return q(pdrv), q(p[EDGE_CODE]), q(p[SIDEWALK_CODE]), code.to(torch.uint8).cpu().numpy()


EDGE_CODE = 5                        # == mapsel_roll.EDGE_CODE
SIDEWALK_CODE = 7                    # == mapsel_roll.SIDEWALK_CODE


# ================================================================================================================ the dump's own parts
def raster_name(e_i: int, ws: int) -> str:
    """MAPSEL's file name: ``fp = RAST / f"e{e_i:03d}_w{ws:04d}.npz"`` (mapsel_roll.MapSelProxy.__call__)."""
    return f"e{int(e_i):03d}_w{int(ws):04d}.npz"


def write_raster(rast_dir, e_i: int, ws: int, planes: tuple, gt) -> int:
    """MAPSEL's write, line for line: ``arrs`` = pdrv/pedge/psw/code (+ ``gt`` when the window carries a label), written to
    ``<name>.tmp.npz`` with ``savez_compressed`` then ``os.replace``d. Returns the byte size."""
    pdrv, pedge, psw, code = planes[:4]
    arrs = {"pdrv": pdrv, "pedge": pedge, "psw": psw, "code": code}
    if len(planes) > 4 and planes[4] is not None:     # refcv9 C1: the head's own `unobs` plane (uint8; absent for refcv8)
        arrs["unobs"] = np.asarray(planes[4], np.uint8)
    if gt is not None:
        arrs["gt"] = np.asarray(gt, np.uint8)
    rast = Path(rast_dir)
    rast.mkdir(parents=True, exist_ok=True)
    fp = rast / raster_name(e_i, ws)
    tmp = fp.with_name(fp.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    n = os.path.getsize(tmp)
    os.replace(tmp, fp)
    return n


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def make_record(sha12: str, ws: int, el: dict, heads) -> dict:
    """PERC8's per-window record (perc_pass_r7.py main()): ``{"sha12", "ws", <head>: pack}`` for every head that produced a pack."""
    rec = {"sha12": sha12, "ws": int(ws)}
    for hd in heads:
        pk = el.get(f"_det_pack_{hd}")
        if pk:
            rec[hd] = pk
    return rec


def write_packs(path, det) -> str:
    with open(path, "wb") as f:
        pickle.dump(det, f, protocol=4)
    return sha256_file(path)


# ---------------------------------------------------------------------------------------------------------------- windows
def load_grid(path) -> dict:
    g = json.load(open(path, encoding="utf-8"))
    if not isinstance(g, dict) or not g:
        raise SystemExit(f"[r9-e1-dump] {path}: not a non-empty {{clip: [ws]}} object")
    return {str(c): [int(w) for w in ws] for c, ws in g.items()}


def check_split_against_grid(split_obj, split_name, grid_sha256, grid_sha12_counts):
    """E0i's DEV_HOLD_SPLIT.json stamps the S2 grid it was built from (``inputs.s2_windows_json_sha256``) and the per-episode S2
    window count (``windows_per_episode[sha12][0]``). A grid that is not that file REFUSES; a count that differs REFUSES."""
    if not isinstance(split_obj, dict):
        return {"checked": False, "why": "the window file is not a split object"}
    stamp = ((split_obj.get("inputs") or {}).get("s2_windows_json_sha256"))
    if stamp and grid_sha256 and stamp != grid_sha256:
        raise SystemExit(f"[r9-e1-dump] the S2 grid's sha256 {grid_sha256[:12]} != the split's stamped s2_windows_json_sha256 "
                         f"{stamp[:12]}: not the grid this split was cut from")
    out = {"checked": True, "grid_sha256_matches_stamp": bool(stamp), "count_mismatch": []}
    blk = split_obj.get(str(split_name).upper()) or {}
    wpe = blk.get("windows_per_episode") or {}
    for s12, cols in wpe.items():
        if s12 not in grid_sha12_counts:
            raise SystemExit(f"[r9-e1-dump] split {split_name}: clip {s12} is not in the S2 grid")
        if int(cols[0]) != int(grid_sha12_counts[s12]):
            out["count_mismatch"].append([s12, int(cols[0]), int(grid_sha12_counts[s12])])
    if out["count_mismatch"]:
        raise SystemExit(f"[r9-e1-dump] split {split_name}: S2 window counts differ from the grid for "
                         f"{len(out['count_mismatch'])} clip(s), e.g. {out['count_mismatch'][:3]}")
    if blk.get("n_windows_s2") is not None and sum(int(c[0]) for c in wpe.values()) != int(blk["n_windows_s2"]):
        raise SystemExit(f"[r9-e1-dump] split {split_name}: n_windows_s2 {blk['n_windows_s2']} != the sum of its episodes")
    return out


def select_clips(grid: dict, sha12_of, windows_path=None, split=None) -> tuple:
    """-> (clips [full ids of the grid in the chosen set], sha12 list, split record). ``windows_path`` None = every clip of the grid."""
    by_sha = {sha12_of(c): c for c in grid}
    if len(by_sha) != len(grid):
        raise SystemExit("[r9-e1-dump] two grid clips share a sha12")
    counts = {s: len(grid[c]) for s, c in by_sha.items()}
    grid_sha = None
    if windows_path is None:
        return sorted(grid), sorted(by_sha), {"set": "ALL", "n_clips": len(grid)}
    from r9_e1_eval import load_window_list
    want = load_window_list(windows_path, split)
    miss = [s for s in want if s not in by_sha]
    if miss:
        raise SystemExit(f"[r9-e1-dump] {len(miss)} listed clip(s) are not in the S2 grid, e.g. {miss[:3]}")
    try:
        obj = json.loads(Path(windows_path).read_text(encoding="utf-8"))
    except ValueError:
        obj = None
    return [by_sha[s] for s in want], want, {"set": str(split or "list").upper(), "n_clips": len(want), "split_obj": obj, "counts": counts}


def window_order(grid, clips, ci: dict, index_pos: dict, W: int) -> list:
    """[(clip, ws, dataset index j)] in the banked producers' order: clips by manifest index, windows in the grid's order. A
    window the eval index does not hold REFUSES (never silently dropped)."""
    order = []
    for c in sorted(clips, key=lambda c: ci[c]):
        for w0 in grid[c]:
            key = (ci[c], int(w0) - (W - 1))
            if key not in index_pos:
                raise SystemExit(f"[r9-e1-dump] window ws={w0} of clip {key[0]} is not in the eval index")
            order.append((c, int(w0), index_pos[key]))
    return order


def fit_window_order(index, clip_ids, sha12_of, wanted, K: int, W: int) -> list:
    """The FIT split's windows (``--fit-k-per-ep``): ``run_diag.py``'s train_diag picker (2026-10-21-refcv8-perc8/code/run_diag.py, the `else` branch of
    the window-set block) VERBATIM IN ITS ARITHMETIC, restricted to the ``wanted`` clips (a set of sha12): episodes in sha12 order, and per episode the
    ``K`` windows at ``ts[int((j + 0.5) * len(ts) / K)]`` of its sorted dataset starts ``ts``. ``index`` = the dataset's ``(episode, t)`` list, ``clip_ids`` the
    manifest's clip ids indexed by episode. -> [(clip id, ws = t + W - 1, dataset index)] -- the same triple ``window_order`` returns. The picks come from
    the DATASET'S OWN index, so no window can be one the dataset does not hold."""
    by_ep = {}
    for i, (e_i, t) in enumerate(index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    order = []
    for e_i in sorted(by_ep, key=lambda e: sha12_of(clip_ids[e])):
        c = clip_ids[e_i]
        if sha12_of(c) not in wanted:
            continue
        ts = sorted(by_ep[e_i])
        for j in range(int(K)):
            t, i = ts[int((j + 0.5) * len(ts) / int(K))]
            order.append((c, int(t) + W - 1, i))
    return order


def order_digest(order, sha12_of) -> str:
    """sha256 over the (sha12, ws) sequence: two dumps on the SAME windows have the same digest."""
    h = hashlib.sha256()
    for c, w0, _j in order:
        h.update(f"{sha12_of(c)}:{w0};".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- the rig step-0 checkpoint
def make_rig_step0_ckpt(src, dst) -> dict:
    """KA-STEP0: the refcv8 EMA with the planner / tactical keys DROPPED by the rig's own predicate -- the exact set
    ``refcv9_perc_rig.warm_start_perc_rig`` loads (it refuses any other key). No training step has been taken."""
    from tanitad.refs import refcv9_perc_rig as R
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    sd = ck["model"]
    keep = {k: v for k, v in sd.items() if not R.is_skipped_key(k)}
    unknown = [k for k in keep if not R.is_built_key(k)]
    if unknown:
        raise SystemExit(f"[r9-e1-dump] {len(unknown)} source key(s) are neither built nor skipped by the rig, e.g. {unknown[:3]}")
    out = {"model": keep, "step": ck.get("step"), "kind": "rig_step0_from_" + str(ck.get("kind", "unknown")),
           "source": str(src)}
    torch.save(out, str(dst))
    return {"src": str(src), "dst": str(dst), "n_source_keys": len(sd), "n_kept": len(keep), "n_dropped": len(sd) - len(keep),
            "step": ck.get("step"), "kind": ck.get("kind"), "dst_sha256": sha256_file(dst)}


# ================================================================================================================ the dump
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the run's config.json (its argv builds the model)")
    ap.add_argument("--windows", help="E0i DEV_HOLD_SPLIT.json (or a sha12 list); omitted = every clip of the S2 grid")
    ap.add_argument("--split", default=None, help="dev | hold (needed when --windows is a split object)")
    ap.add_argument("--s2-windows", default=os.environ.get("R9_S2_WINDOWS", "/home/nvidia/r8ns/perc8/inputs/s2_windows.json"),
                    help="the S2 window grid {clip: [ws]} (sha256 9943f3a2... = E0i's stamp)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", help="cpu = SMOKE ONLY (code-path check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-sync", choices=("off", "on"), default="off",
                    help="off = the banked producers' clock (R8_NO_POSE_SYNC=1); on = the run's own eval clock")
    ap.add_argument("--law-control", type=int, default=2, help="windows rolled with the REAL future frame too (perc_pass_r7's control)")
    ap.add_argument("--rig-step0-from", help="a FULL (refcv8) EMA checkpoint: build the rig from it with NO training step (KA-STEP0)")
    ap.add_argument("--require-ema", action="store_true", help="refuse a checkpoint that is not an EMA file")
    ap.add_argument("--fit-k-per-ep", type=int, default=0,
                    help="FIT DUMP (the map-threshold fit's rasters): K windows per clip at run_diag's evenly spaced train_diag positions, for the TRAIN clips "
                         "listed in --windows (a sha12 list) that are present in the dataset cache; needs --exclude-split; --s2-windows is not read. 0 = off "
                         "(every other behaviour is unchanged)")
    ap.add_argument("--exclude-split", help="FIT dump: E0i DEV_HOLD_SPLIT.json -- the dump REFUSES a FIT clip that is in DEV or HOLD")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th window of the selection (SMOKE / sample)")
    ap.add_argument("--max-windows", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--rast-every", type=int, default=1, help="write a raster for every n-th kept window")
    ap.add_argument("--no-rasters", action="store_true")
    ap.add_argument("--no-packs", action="store_true")
    return ap


def run_dump(a) -> int:
    t_all = time.time()
    stack = HERE.parent
    for p in (str(stack), str(stack.parent / "taniteval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tanitad.eval import refcv7_loader as L
    L.bootstrap()
    tr = L.trainer()
    device = a.device
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = L.load_config(a.config)
    ckpt_path = a.ckpt
    step0 = None
    if a.rig_step0_from:
        step0 = make_rig_step0_ckpt(a.rig_step0_from, out_dir / "rig_step0_ckpt.pt")
        ckpt_path = str(out_dir / "rig_step0_ckpt.pt")
        print(f"[dump] rig step-0 checkpoint: kept {step0['n_kept']} of {step0['n_source_keys']} keys", flush=True)
    ck_head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_kind = str(ck_head.get("kind", "full_checkpoint (raw weights; its ema block holds no tensors)" if "ema" in ck_head else "unknown"))
    is_ema_file = "ema_model_only" in ck_kind
    if a.require_ema and not is_ema_file:
        raise SystemExit(f"[r9-e1-dump] --require-ema: {ckpt_path} is {ck_kind!r}, not an EMA file")
    del ck_head
    model, cfg, args, mrec = L.build_model(config, ckpt_path, device)
    is_rig = bool(getattr(getattr(model, "core", model), "_perc_rig", False))
    W = int(cfg.core.window)
    LA = int(tr.LAW_AHEAD)
    law_idx = LA - 1
    # ---- the eval dataset ------------------------------------------------------------------------------------------ #
    args_ds = copy.copy(args)
    departures = []
    if a.pose_sync == "off" and getattr(args_ds, "pose_sync_sidecar", None):
        departures.append("pose-sync sidecar NOT enabled (PERC8 / MAPSEL ran R8_NO_POSE_SYNC=1): the unshifted pose clock")
        args_ds.pose_sync_sidecar = None
    if not getattr(args_ds, "eval_labels", None):
        fallback = L.PATH_REMAP.get("--eval-labels")
        departures.append(f"the run's argv names no --eval-labels (the rig carries no tactical labels); the loader's eval dataset needs "
                         f"them to build, so the kit's refcv8 eval labels {fallback} are attached (no perception target reads them)")
        args_ds.eval_labels = fallback
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args_ds, config, with_perception_targets=True,
                                             dataset_cls=make_perc_dataset_cls(tr, LA))
    from tanitad.data.v2_dataset import load_or_build_manifest
    clip_ids = [str(c) for c in load_or_build_manifest(args_ds.eval_cache, verbose=False)["clip_id"]]
    ci = {c: i for i, c in enumerate(clip_ids)}
    fit_rec = None
    if a.fit_k_per_ep:
        if not a.windows or not a.exclude_split:
            raise SystemExit("[r9-e1-dump] --fit-k-per-ep needs --windows <FIT clip sha12 list> and --exclude-split <DEV_HOLD_SPLIT.json>")
        if a.split:
            raise SystemExit("[r9-e1-dump] --fit-k-per-ep takes a plain sha12 list in --windows: --split names DEV / HOLD, which a FIT dump never reads")
        from r9_e1_eval import load_window_list
        want = load_window_list(a.windows)
        evalset = set(load_window_list(a.exclude_split, "dev")) | set(load_window_list(a.exclude_split, "hold"))
        clash = sorted(set(want) & evalset)
        if clash:
            raise SystemExit(f"[r9-e1-dump] {len(clash)} FIT clip(s) are in DEV / HOLD of {Path(a.exclude_split).name} (e.g. {clash[:3]}): a FIT dump never reads eval")
        by_sha12 = {L.sha12(c): c for c in clip_ids}
        miss = [s for s in want if s not in by_sha12]
        if miss:
            raise SystemExit(f"[r9-e1-dump] {len(miss)} FIT clip(s) are not in the dataset cache {getattr(args_ds, 'eval_cache', '?')} (e.g. {miss[:3]}): "
                             f"point --eval-cache (REFCV6_REMAP_OVERRIDES) at the TRAIN cache that holds them")
        grid, grid_sha = None, None
        clips = [by_sha12[s] for s in want]
        order = fit_window_order(e_ds.index, clip_ids, L.sha12, set(want), a.fit_k_per_ep, W)
        split_rec = {"set": "FIT", "n_clips": len(want)}
        fit_rec = {"k_per_ep": int(a.fit_k_per_ep), "n_clips": len(want), "clips_sha12": sorted(want),
                   "exclude_split": Path(a.exclude_split).name, "exclude_split_sha256": sha256_file(a.exclude_split),
                   "disjoint_from_dev_and_hold": True,
                   "rule": "TRAIN clips; K windows per clip at ts[int((j + 0.5) * len(ts) / K)] of the dataset's sorted starts (run_diag.py train_diag)"}
    else:
        grid = load_grid(a.s2_windows)
        grid_sha = sha256_file(a.s2_windows)
        clips, sel_sha12, split_rec = select_clips(grid, L.sha12, a.windows, a.split)
        if split_rec.get("split_obj") is not None and a.split:
            split_rec["check"] = check_split_against_grid(split_rec["split_obj"], a.split, grid_sha, split_rec["counts"])
        split_rec.pop("split_obj", None)
        split_rec.pop("counts", None)
        pos = {tuple(int(v) for v in x): j for j, x in enumerate(e_ds.index)}
        unknown = [c for c in clips if c not in ci]
        if unknown:
            raise SystemExit(f"[r9-e1-dump] {len(unknown)} selected clip(s) are not in the eval cache")
        order = window_order(grid, clips, ci, pos, W)
    order = order[::max(1, int(a.every))]
    if a.max_windows:
        order = order[:a.max_windows]
    digest = order_digest(order, L.sha12)
    print(f"[dump] model built ({'RIG' if is_rig else 'full'}), {len(order)} windows ({split_rec['set']}), digest {digest[:12]}", flush=True)
    # ---- the forward ------------------------------------------------------------------------------------------------ #
    import torch as _t
    patch_frames_to_device(tr)
    cap = {}
    orig_fwd = model.forward

    def fwd(*aa, **kk):
        out = orig_fwd(*aa, **kk)
        cap["out"] = out
        return out
    model.forward = fwd
    dm = tr._det_metrics
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))

    pfq_staged = {"n": 0}

    def fwd_losses(eb):
        cap.clear()
        if stage_perframe_queries(model, eb, device):            # C5 lever 4: before EVERY forward (control windows included); a no-op for every other arm
            pfq_staged["n"] += 1
        return tr.compute_losses_v3(model, eb, device, mode=mode, ablate_frames=abl)

    # ---- the future-frame control (perc_pass_r7's LAW-frame control; a rig has no law term, so only equality is required) ---- #
    law_ctl = []
    with _t.no_grad():
        for (c, w0, j) in order[:max(0, a.law_control)]:
            eb0 = collate(e_ds, [j], law_idx)
            e_i, t = e_ds.index[j]
            real = e_ds.episodes[e_i].frames[t + W:t + W + LA][law_idx]
            eb1 = dict(eb0)
            eb1["future_frames"] = LawOnly(real[None].clone(), law_idx)
            _t.manual_seed(a.seed)
            el0 = fwd_losses(eb0)
            p0 = {k: v.detach().clone() for k, v in _perc_tensors(cap["out"]).items()}
            _t.manual_seed(a.seed)
            el1 = fwd_losses(eb1)
            p1 = _perc_tensors(cap["out"])
            same = {k: bool(_t.equal(p0[k], p1[k])) for k in p0}
            has_law = "law" in el0
            law0, law1 = float(el0.get("law", float("nan"))), float(el1.get("law", float("nan")))
            law_ctl.append({"sha12": L.sha12(c), "ws": w0, "perception_equal": same, "has_law_term": has_law,
                            "law_zero_frame": law0, "law_real_frame": law1, "law_differs": (law0 != law1) if has_law else None})
            del el0, el1, eb0, eb1
    ctl_ok = bool(law_ctl) and all(x["perception_equal"] and all(x["perception_equal"].values())
                                   and (x["law_differs"] is True or not x["has_law_term"]) for x in law_ctl)
    if a.law_control and not ctl_ok:
        model.forward = orig_fwd
        json.dump({"tool": "r9_e1_dump.py", "law_control": law_ctl, "status": "REFUSED"},
                  open(str(out_dir / "dump.json.REFUSED"), "w", encoding="utf-8"), indent=1)
        raise SystemExit(f"[r9-e1-dump] future-frame control FAILED (a zero future frame changed a perception output, or the swap "
                         f"did not take): {law_ctl}")
    print(f"[dump] future-frame control: {'PASS' if ctl_ok else 'SKIPPED'} on {len(law_ctl)} windows", flush=True)
    # ---- the loop ---------------------------------------------------------------------------------------------------- #
    det, written, n_box, n_rast, n_gt, rast_bytes = [], [], 0, 0, 0, 0
    tim = {"wait_decode": 0.0, "forward": 0.0, "planes": 0.0}
    dec_pool = ThreadPoolExecutor(max_workers=1)
    pend = deque()
    it = iter(order)

    def submit_next():
        nxt = next(it, None)
        if nxt is not None:
            pend.append((nxt, dec_pool.submit(collate, e_ds, [nxt[2]], law_idx)))

    rast_dir = out_dir / "rast"
    _t.manual_seed(a.seed)
    if device.startswith("cuda"):
        _t.cuda.manual_seed_all(a.seed)
    t0 = time.time()
    submit_next()
    submit_next()
    try:
        with _t.no_grad():
            n = 0
            while pend:
                (c, w0, j), fut = pend.popleft()
                tw = time.time()
                eb = fut.result()
                tim["wait_decode"] += time.time() - tw
                submit_next()
                s12 = L.sha12(c)
                tf = time.time()
                el = fwd_losses(eb)
                pout = cap["out"].get("perception") or {}
                tim["forward"] += time.time() - tf
                rec = make_record(s12, w0, el, dm.HEADS)
                if rec.get("box3d"):
                    n_box += 1
                det.append(rec)
                lg = pout.get("map_hires_logits")
                mb_ = pout.get("map_banded")             # refcv9 C1 (--r9-map-banded): the banded channels + unobs, no 8-way logits
                if not a.no_rasters and (lg is not None or mb_ is not None) and n % max(1, int(a.rast_every)) == 0:
                    tp_ = time.time()
                    ST["c"]["calls"] += 1
                    if mb_ is None:
                        planes = own_planes(model, lg[0:1])
                    else:        # the ONE adapter (tanitad/eval/map_planes.py): the 4 MAPSEL planes + `unobs`
                        from tanitad.eval import map_planes as _MPL
                        _cfg = model._map_hires.cfg
                        planes = _MPL.to_u8(_MPL.planes_from_banded(
                            {k: [t[0:1] for t in v] for k, v in mb_.items()}, x_max_m=float(_cfg.x_max_m), y_half_m=float(_cfg.y_half_m)))
                    lab = eb.get("map_fine_label")
                    gt = None
                    if lab is not None and bool(lab.reshape(-1)[0]) and eb.get("map_fine") is not None:
                        gt = eb["map_fine"][0].cpu().numpy()
                        n_gt += 1
                    e_i, t = (int(x) for x in e_ds.index[j])
                    rast_bytes += write_raster(rast_dir, e_i, t + W - 1, planes, gt)
                    written.append(raster_name(e_i, t + W - 1))
                    n_rast += 1
                    tim["planes"] += time.time() - tp_
                del el, eb
                cap.clear()
                if n % 200 == 0:
                    print(f"[dump] {n + 1}/{len(order)} windows ({time.time() - t0:.0f} s; box3d {n_box}, rasters {n_rast}; "
                          f"{ {k: round(v, 1) for k, v in tim.items()} })", flush=True)
                n += 1
    finally:
        model.forward = orig_fwd
        dec_pool.shutdown(wait=True)
    packs_path = out_dir / "perc_det.pkl"
    packs_sha = None
    if not a.no_packs:
        packs_sha = write_packs(packs_path, det)
    names_digest = hashlib.sha256("\n".join(sorted(written)).encode()).hexdigest()
    meta = {"tool": "r9_e1_dump.py", "schema": SCHEMA, "rig": is_rig, "ckpt": str(a.ckpt), "ckpt_used": str(ckpt_path),
            "ckpt_md5": L.md5_file(ckpt_path), "ckpt_kind": ck_kind, "weights_are_ema": is_ema_file, "step": mrec["state_dict"]["step"],
            "rig_step0": step0, "config": str(a.config), "config_md5": L.md5_file(a.config),
            "split": split_rec, "n_windows": len(order), "n_clips": len(clips), "order_digest": order_digest(order, L.sha12),
            "n_eval_windows_in_dataset": len(e_ds), "window_W": W, "s2_grid_sha256": grid_sha,
            "pose_sync": a.pose_sync, "pose_sync_record": drec.get("pose_sync"), "departures": departures + list(mrec.get("departures", [])),
            "declared_vs_built": mrec.get("declared_vs_built"), "param_breakdown_equal": (mrec.get("param_breakdown") or {}).get("equal"),
            "future_frame_control": {"pass": ctl_ok, "windows": law_ctl,
                                     "rule": "zero future frame: every perception tensor torch.equal to the real-future-frame forward"
                                             " (and, where a law term exists, the term differs)"},
            "n_box3d_windows": n_box, "n_rasters": n_rast, "n_rasters_with_gt": n_gt, "raster_bytes": rast_bytes,
            "raster_names_sha256": names_digest, "packs": None if a.no_packs else {"path": str(packs_path), "sha256": packs_sha},
            "map_decision": {"rule_seen": sorted(ST["c"]["rule_seen"]), "decide_checked": ST["c"]["decide_checked"],
                             "decide_mismatch_cells": ST["c"]["decide_mismatch_cells"]},
            "timing_s": {k: round(v, 1) for k, v in tim.items()}, "loop_s": round(time.time() - t0, 1),
            "wall_s": round(time.time() - t_all, 1), "seed": a.seed, "device": device,
            "cuda_max_memory_allocated_gib": (round(_t.cuda.max_memory_allocated() / 2 ** 30, 3) if _t.cuda.is_available() and _t.cuda.is_initialized() else None),
            "SMOKE_RESTRICTED": bool(a.max_windows) or int(a.every) > 1 or int(a.rast_every) > 1,
            "torch": _t.__version__, "tree_commit": _tree_commit(stack.parent)}
    if fit_rec is not None:                 # only a FIT dump carries the block: every other dump.json is unchanged
        meta["fit"] = fit_rec
    if pfq_staged["n"]:
        _pfq = model._perception.box_pfq
        meta["perframe_queries"] = {"staged_forwards": pfq_staged["n"], "n_seeds": int(_pfq.n_seeds), "n_global": int(_pfq.n_global),
                                    "staged_before": "every forward (control windows + main loop)"}
    (out_dir / "dump.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print("[dump] done: " + json.dumps({k: meta[k] for k in ("rig", "n_windows", "n_box3d_windows", "n_rasters", "loop_s", "wall_s")}), flush=True)
    return 0


def _tree_commit(root: Path):
    for name in ("TREE_COMMIT.txt",):
        p = Path(root) / name
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()[:80]
    return None


# ================================================================================================================ compare
PACK_KEYS = ("logit", "xy", "cls", "cls_corr", "gt_xy", "gt_cls", "pos", "ign", "hidden", "matched", "exempt")


def _max_abs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    if a.dtype == bool or b.dtype == bool:
        return float((a != b).sum())
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def compare_packs(det_a, det_b, head="box3d") -> dict:
    """Window-by-window pack equality over the (sha12, ws) keys both dumps hold, key by key: max abs diff, count of windows whose
    arrays are bit-identical, shape disagreements. The GT keys (``gt_xy`` / ``gt_cls`` / ``pos``) must be EXACTLY equal (same labels);
    the prediction keys (``logit`` / ``xy`` / ``cls``) carry the float noise."""
    A = {(str(r["sha12"]), int(r["ws"])): r[head][0] for r in det_a if r.get(head)}
    B = {(str(r["sha12"]), int(r["ws"])): r[head][0] for r in det_b if r.get(head)}
    common = sorted(set(A) & set(B))
    out = {"n_a": len(A), "n_b": len(B), "n_common": len(common), "only_a": len(set(A) - set(B)), "only_b": len(set(B) - set(A)),
           "keys": {}, "n_windows_bit_identical": 0, "n_windows_shape_mismatch": 0}
    for k in PACK_KEYS:
        out["keys"][k] = {"max_abs_diff": 0.0, "n_windows_differing": 0, "n_shape_mismatch": 0, "n_missing": 0}
    for key in common:
        same = True
        shape_bad = False
        for k in PACK_KEYS:
            pa, pb = A[key].get(k), B[key].get(k)
            rec = out["keys"][k]
            if pa is None or pb is None:
                if (pa is None) != (pb is None):
                    rec["n_missing"] += 1
                    same = False
                continue
            d = _max_abs(pa, pb)
            if d is None:
                rec["n_shape_mismatch"] += 1
                shape_bad = True
                same = False
                continue
            rec["max_abs_diff"] = max(rec["max_abs_diff"], d)
            if d > 0:
                rec["n_windows_differing"] += 1
                same = False
        out["n_windows_bit_identical"] += int(same)
        out["n_windows_shape_mismatch"] += int(shape_bad)
    return out


def compare_rasters(dir_a, dir_b, names=None, n=None, tol_prob=2) -> dict:
    """Plane-by-plane raster equality on the names both directories hold (``n`` evenly spaced when given). ``gt`` and ``code`` are
    class planes: EXACT. ``pdrv`` / ``pedge`` / ``psw`` are probabilities quantised to uint8: a max abs diff <= ``tol_prob`` counts is
    float noise (declared), anything larger is a different forward."""
    da, db = Path(dir_a), Path(dir_b)
    common = sorted(set(os.listdir(da)) & set(os.listdir(db)))
    common = [f for f in common if f.endswith(".npz") and not f.endswith(".tmp.npz")]
    if names is not None:
        keep = set(names)
        common = [f for f in common if f in keep]
    if n and len(common) > n:
        idx = np.unique(np.linspace(0, len(common) - 1, int(n)).round().astype(int))
        common = [common[i] for i in idx]
    out = {"n_common": len(common), "tol_prob_counts": int(tol_prob), "planes": {}, "files_with_missing_plane": 0}
    for p in ("pdrv", "pedge", "psw", "code", "gt"):
        out["planes"][p] = {"max_abs_diff": 0, "n_files_differing": 0, "n_cells_differing": 0, "n_cells": 0, "n_files": 0}
    for f in common:
        za, zb = np.load(da / f), np.load(db / f)
        miss = False
        for p in out["planes"]:
            ina, inb = p in za.files, p in zb.files
            if ina != inb:
                miss = True
                continue
            if not ina:
                continue
            xa, xb = za[p].astype(np.int32), zb[p].astype(np.int32)
            rec = out["planes"][p]
            rec["n_files"] += 1
            if xa.shape != xb.shape:
                rec["n_files_differing"] += 1
                rec["max_abs_diff"] = max(rec["max_abs_diff"], 255)
                continue
            d = np.abs(xa - xb)
            nd = int((d > 0).sum())
            rec["n_cells"] += int(d.size)
            rec["n_cells_differing"] += nd
            rec["n_files_differing"] += int(nd > 0)
            rec["max_abs_diff"] = max(rec["max_abs_diff"], int(d.max()) if d.size else 0)
        out["files_with_missing_plane"] += int(miss)
    ok = all(out["planes"][p]["max_abs_diff"] == 0 for p in ("code", "gt")) and \
        all(out["planes"][p]["max_abs_diff"] <= tol_prob for p in ("pdrv", "pedge", "psw")) and out["files_with_missing_plane"] == 0
    out["pass"] = bool(ok and len(common) > 0)
    out["rule"] = ("PASS iff code and gt are exactly equal, pdrv / pedge / psw differ by <= tol_prob counts of 255, and no file lacks a "
                   "plane the other has")
    return out


def run_compare(argv) -> int:
    ap = argparse.ArgumentParser(prog="r9_e1_dump.py compare")
    ap.add_argument("--packs-a")
    ap.add_argument("--packs-b")
    ap.add_argument("--rast-a")
    ap.add_argument("--rast-b")
    ap.add_argument("--rast-n", type=int, default=0, help="compare n evenly spaced common rasters (0 = all)")
    ap.add_argument("--rast-names", help="text file of raster file names to keep")
    ap.add_argument("--tol-prob", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"tool": "r9_e1_dump.py compare", "schema": SCHEMA}
    if a.packs_a and a.packs_b:
        with open(a.packs_a, "rb") as f:
            da = pickle.load(f)
        with open(a.packs_b, "rb") as f:
            db = pickle.load(f)
        res["packs"] = compare_packs(da, db)
        res["packs_a"], res["packs_b"] = {"path": a.packs_a, "sha256": sha256_file(a.packs_a)}, {"path": a.packs_b, "sha256": sha256_file(a.packs_b)}
    if a.rast_a and a.rast_b:
        names = None if not a.rast_names else [ln.strip() for ln in Path(a.rast_names).read_text().splitlines() if ln.strip()]
        res["rasters"] = compare_rasters(a.rast_a, a.rast_b, names, a.rast_n or None, a.tol_prob)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "packs" else {kk: vv for kk, vv in v.items() if kk != "keys"}) for k, v in res.items()}, indent=1)[:3000])
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return run_compare(argv[1:])
    return run_dump(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
