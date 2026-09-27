#!/usr/bin/env python
"""G-MAP-OVERFIT -- can the 10 cm map path learn EVERY SAM3 class? (refcv7 NEW-2)

``SPEC_REFCV7.md`` §8 (A3) item 3, protocol and literals pre-registered by the
map-signal audit: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-26-map-signal-audit/raw/PREREG_G_MAP_OVERFIT.md`` (machine-readable:
``raw/gmo_spec.json``). This harness implements it; it invents no number.

WHAT RUNS (the prereg's §3, literal): the refcv7 LAUNCH map path -- the resnet101
trunk (frozen BN, the launch levers, the C26 equalisation), its stride-8 tap, the NEW-2
branch with the launch ``MapHiresConfig`` -- trained ONLY on the 10 cm loss
(``hires_map_ce``: hard CE, ignore 255, the lift-valid narrowing, the FROZEN launch
class weights) over the 16 fixed TRAIN frames, AdamW (lr 1e-3 constant, betas 0.9 /
0.999, eps 1e-8, weight decay from the spec), batch 4, N steps, seed 0. Scored cells
(§4) = ``fine_codes != 255`` AND lift-valid AND rig x in the GATED band; IoU POOLED
over the frames; the verdict uses the DECLARED decision rule (§5) and both rules are
reported.

ARMS (all from ONE shared init):

* ``healthy``      -- the MAIN arm (branch + the trunk's stem..stride-8 stage);
* ``lane_w0``      -- R1: the lane weight 0 in the LOSS (the decision keeps the launch
                      weights: R1 tests the learning signal, not the decision rule);
                      must FAIL on lane;
* ``s8_zeros``     -- R2: ``zeros_like(fmap_s8)`` enters the branch (the F3-whitelist
                      class); the trunk stays in the optimiser and gets no map gradient;
                      must FAIL on ALL FIVE thin classes (``must_fail_all``);
* ``s8_detached``  -- informative: ``fmap_s8.detach()`` (ImageNet stride-8 features can
                      still memorise 16 frames, so it may PASS -- not gated);
* ``frozen_trunk`` -- informative: the branch alone trains (identical to
                      ``s8_detached`` here, since the trunk has no other loss).
* ``near_zeros``   -- NEW-2 R2 (SPEC_REFCV7 A12): the 0.1 m near lift samples
                      ``zeros_like(fmap_s8)`` while the 0.25 m path reads the real map,
                      i.e. the NEW-2 arm plus a skip that can only learn constants; gated
                      by an A12 spec's ``must_fail`` row (non-drivable edge). Needs
                      ``--near-lift-m > 0``.

THE NEAR LIFT (``--near-lift-m``, NEW-2 R2 / A12): 0 (default) = the NEW-2 branch as landed
(the prereg's MAIN); > 0 = the branch built with ``MapHiresConfig.near_lift_x_m``, every
other literal unchanged. Its zero-initialised skip is built LAST, so the record also carries
the branch fingerprint WITHOUT it (``branch_init_sha256_without_near``), which must equal the
NEW-2 MAIN's for the same seed.

CONTROLS on the same scored cells (§7): C1 drivable everywhere (IoU_drivable =
n_drivable / n_scored to 1e-9, every other class 0, and it FAILS §6); C2 logits =
20 · one-hot(label) (IoU 1.0 for every class, under the declared rule); C3 uniform
weights (the prior-corrected and raw decisions bit-identical on the MAIN arm's logits).

VERDICT (§8): PASS iff the MAIN arm passes §6 (presence >= min_cells for all 8
classes -- else INCONCLUSIVE => FAIL; the IoU bars; each class's per-cell CE final <=
ratio x step 0; a finite loss at every step), every gated regression arm fails as
required, and C1-C3 reproduce. Anything else is FAIL. Exit 0 = PASS, 2 = FAIL.

THE FRAME AXIS: v2ep payloads carry no per-frame timestamps (MEASURED 2026-09-26: the
payload's ``frame`` dict holds height / width / f_ref / projection only). With
``--cam-ts-dir`` (the corpus's ``<clip_id>[.camera_front_wide_120fov].timestamps.parquet``)
each label is time-checked at 1 ms (``ClipMapGT.check_times`` on
``episode_frame_times_us``, the prereg's §2 / §9a.6 guard); without it only the
reader's POSE-CONTENT check runs (``check_pose_alignment``: the GT pose must follow the
v2ep pose on the same frame index within 5 cm; INCONCLUSIVE refuses) -- and then the
verdict CANNOT be PASS (``time_guard_1ms`` is a gate condition). The record names the
guard that ran. Clip ids never printed or written: sha12 only.

THE EXTENT (SPEC_REFCV7 §11.2 / §12, A6/A7): the LAUNCH map is 100 m x +-30 m (1000 x
600 at 0.1 m; ``--x-max-m`` / ``--y-half-m``), read through the fine reader at that
extent (``/3`` GT; a ``/2`` file is refused outside 60 m x +-16 m). The gated band stays
the prereg's (``0_20``); every other 20 m band is reported. The launch decoder
recomputes in backward (``--grad-ckpt on``, §12 item 4). ``refcv6_head_05m`` (R3,
informative) is NOT RUN: A6 removed the 0.5 m head (§8: "if R3 was run").
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                        # stack/

from tanitad.data import semantic_map_gt_fine as F              # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7, MapExtent  # noqa: E402
from tanitad.models import map_head_hires as H                  # noqa: E402

ARMS = ("healthy", "lane_w0", "s8_zeros", "s8_detached", "frozen_trunk", "near_zeros")
INFORMATIVE_KNOWN = ("s8_detached", "frozen_trunk", "refcv6_head_05m")
LANE = 2
DRIVABLE = 1
THIN = ("lane", "crosswalk", "arrow", "edge", "hatched")
#: R3 of the prereg: informative, and NOT buildable in refcv7 (A6 removed the head)
NOT_RUN_WHY = {"refcv6_head_05m": "refcv7 A6 (SPEC_REFCV7 §11.1) removed the 0.5 m "
                                  "map head; PREREG §8 gates R3 only 'if R3 was run'"}


# --------------------------------------------------------------------------- #
# the spec                                                                     #
# --------------------------------------------------------------------------- #
def _expand_frames(frames) -> list:
    """``{"clip_sha12", "raw_frame"}`` rows, or the audit frame-set entries
    ``{"clip_sha12", "raw_v2ep_frames": [...]}``. Anything else is refused."""
    rows = []
    for e in frames:
        if "raw_frame" in e:
            rows.append({"clip_sha12": str(e["clip_sha12"]), "raw_frame": int(e["raw_frame"])})
        elif "raw_v2ep_frames" in e:
            rows += [{"clip_sha12": str(e["clip_sha12"]), "raw_frame": int(f)}
                     for f in e["raw_v2ep_frames"]]
        else:
            raise SystemExit(f"⛔ frame entry without raw_frame / raw_v2ep_frames: "
                             f"{sorted(e)}")
    if not rows:
        raise SystemExit("⛔ the spec names no frame")
    return rows


def _placeholder(v) -> bool:
    return isinstance(v, str) and v.strip().startswith("<")


def load_spec(path, *, class_weights: str | None = None,
              decision_rule: str | None = None, band_keys=H.BAND_KEYS) -> dict:
    """Read + validate the spec. The prereg's placeholders (``class_weights``,
    ``thresholds.decision_rule``) MUST be filled by the caller (``--class-weights``,
    ``--decision-rule``); nothing is defaulted."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("frames", "steps", "thresholds"):
        if k not in s:
            raise SystemExit(f"⛔ spec has no {k!r}: the prereg supplies it, this "
                             f"harness does not invent it")
    s["frames"] = _expand_frames(s["frames"])
    th = s["thresholds"]
    if not isinstance(th, dict) or not th.get("iou"):
        raise SystemExit("⛔ spec.thresholds.iou is empty: no literal per-class bar")
    for cls in th["iou"]:
        if cls not in H.CLASS_KEYS:
            raise SystemExit(f"⛔ threshold class {cls!r} not in {H.CLASS_KEYS}")
    if th.get("band", "0_20") not in tuple(band_keys) + ("all",):
        raise SystemExit(f"⛔ thresholds.band {th.get('band')!r}")
    if "min_cells" not in th:
        raise SystemExit("⛔ thresholds.min_cells missing: the presence floor is the "
                         "prereg's §6.1, never waived")
    rule = decision_rule if decision_rule is not None else th.get("decision_rule")
    if rule is None or _placeholder(rule):
        raise SystemExit("⛔ no declared decision rule: pass --decision-rule (the launch "
                         "config's MapHiresConfig.decision_rule)")
    if rule not in H.DECISION_RULES:
        raise SystemExit(f"⛔ decision rule {rule!r} not in {H.DECISION_RULES}")
    th["decision_rule"] = rule
    cw = class_weights if class_weights is not None else s.get("class_weights", "uniform")
    if cw is None or _placeholder(cw):
        raise SystemExit("⛔ no class weights: pass --class-weights <the LAUNCH weights "
                         "json> (a dry-run file is refused)")
    s["class_weights"] = cw
    for arm, must in (s.get("must_fail") or {}).items():
        if arm not in ARMS or arm == "healthy":
            raise SystemExit(f"⛔ must_fail names {arm!r}")
        for cls in must:
            if cls not in th["iou"]:
                raise SystemExit(f"⛔ {arm} must fail {cls!r}, which has no threshold")
    for arm in (s.get("must_fail_all") or {}):
        if arm not in (s.get("must_fail") or {}):
            raise SystemExit(f"⛔ must_fail_all names {arm!r}, which has no must_fail row")
    for arm in s.get("informative_arms") or []:
        if arm not in INFORMATIVE_KNOWN:
            raise SystemExit(f"⛔ unknown informative arm {arm!r}")
    return s


def class_weights_of(spec: dict, extent: MapExtent | None = None
                     ) -> tuple[torch.Tensor, dict]:
    if spec["class_weights"] == "uniform":
        return torch.ones(8), {"uniform": True, "sha256": None}
    # refuses a dry run, and weights counted on another extent
    return H.load_class_weights(spec["class_weights"], extent=extent)


# --------------------------------------------------------------------------- #
# the data                                                                     #
# --------------------------------------------------------------------------- #
def load_frames(spec: dict, v2_cache: Path, gt_root: Path, extrinsics: Path,
                frame, cfg: H.MapHiresConfig, equalize_bottom_rows: int,
                cam_ts_dir: Path | None = None) -> dict:
    """-> ``{"x": uint8 [N, 9, H, W], "codes": uint8 [N, *cfg.out_hw], "grid",
    "valid", "sha12", "raw_frame", "axis_guard"}``. READ-ONLY on every input; the GT is
    read at the branch config's DECLARED extent."""
    import torchvision.io as tvio
    from tanitad.data.v2_dataset import MANIFEST_NAME, _list_clips, stable_episode_id
    man = torch.load(Path(v2_cache) / MANIFEST_NAME, map_location="cpu",
                     weights_only=False)
    if man.get("files") != _list_clips(str(v2_cache)):
        raise SystemExit("⛔ the v2 manifest is stale: the clip set is not the cache's")
    by12 = {F.sha12(c): str(c) for c in man.get("clip_id") or []}
    sys.path.insert(0, str(HERE.parent))
    import refc_v3_train as T                                   # noqa: E402
    _single, table = T._read_rig_extrinsics(str(extrinsics))
    if table is None:
        raise SystemExit("⛔ the extrinsics file is not a per-clip table")
    store = F.FineMapGTStore(Path(gt_root), max_open=2, extent=cfg.extent)
    xs, cs, s12s, rfs, cids, guards = [], [], [], [], [], {}
    for row in spec["frames"]:
        s12, rf = str(row["clip_sha12"]), int(row["raw_frame"])
        if s12 not in by12:
            raise SystemExit(f"⛔ frame clip sha12 {s12} is not in the v2 cache")
        cid = by12[s12]
        gt = store.open(cid)
        d = torch.load(Path(v2_cache) / f"{cid}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        if s12 not in guards:
            guards[s12] = _axis_guard(gt, d, cid, cam_ts_dir)
        if guards[s12]["kind"] == "time_1ms":
            gt.check_times([rf], [guards[s12]["t_img_us"][rf]])
        codes = gt.read([rf]).codes[0]
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(d["jpeg_len"].to(torch.int64), 0)])
        if rf < 2 or rf >= int(d["jpeg_len"].shape[0]):
            raise SystemExit(f"⛔ raw frame {rf} of {s12}: no full 3-frame stack")
        ims = [tvio.decode_image(d["jpeg_buf"][int(offs[i]):int(offs[i + 1])],
                                 mode=tvio.ImageReadMode.RGB) for i in (rf - 2, rf - 1, rf)]
        xs.append(torch.cat(ims, 0))
        cs.append(torch.from_numpy(codes.copy()))
        s12s.append(s12)
        rfs.append(rf)
        cids.append(cid)
    bank = H.HiresLiftGeometryBank({c: table[c] for c in set(cids)}, frame=frame,
                                   cfg=cfg, equalize_bottom_rows=equalize_bottom_rows)
    grid, valid = bank.for_episodes([int(stable_episode_id(c)) for c in cids])
    return {"x": torch.stack(xs), "codes": torch.stack(cs), "grid": grid,
            "valid": valid, "sha12": s12s, "raw_frame": rfs,
            "axis_guard": {k: {kk: vv for kk, vv in v.items() if kk != "t_img_us"}
                           for k, v in guards.items()}}


def _axis_guard(gt, payload, cid: str, cam_ts_dir) -> dict:
    """The frame-axis guard for one clip: 1 ms time check when camera timestamps are
    available, else the pose-content check (must be ALIGNED)."""
    if cam_ts_dir is not None:
        import pandas as pd
        from tanitad.data.semantic_map_gt import episode_frame_times_us
        cands = [Path(cam_ts_dir) / f"{cid}.timestamps.parquet",
                 Path(cam_ts_dir) / f"{cid}.camera_front_wide_120fov.timestamps.parquet"]
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            raise SystemExit(f"⛔ [{gt.clip_sha12}] no camera timestamps parquet under "
                             f"--cam-ts-dir: the 1 ms guard cannot run")
        ts = pd.read_parquet(hit)
        tcol = next(k for k in ts.columns if "time" in k.lower())
        g = episode_frame_times_us(ts[tcol].to_numpy())
        if len(g["t_img_us"]) != gt.n_frames:
            raise SystemExit(f"⛔ [{gt.clip_sha12}] episode grid {len(g['t_img_us'])} "
                             f"frames vs GT {gt.n_frames}")
        return {"kind": "time_1ms", "t_img_us": g["t_img_us"]}
    rep = gt.check_pose_alignment(payload["poses"].numpy())     # raises when off
    if rep["verdict"] != "ALIGNED":
        raise SystemExit(f"⛔ [{gt.clip_sha12}] the pose-content frame-axis check is "
                         f"{rep['verdict']} (a near-stationary clip cannot identify the "
                         f"axis): pass --cam-ts-dir for the 1 ms time check")
    return {"kind": "pose_content_5cm", "aligned_max_m": rep["aligned_max_m"],
            "shift_median_m": rep["shift_median_m"]}


# --------------------------------------------------------------------------- #
# statistics                                                                   #
# --------------------------------------------------------------------------- #
def _band_slice(band: str, band_keys=H.BAND_KEYS):
    if band == "all":
        return slice(None)
    b = tuple(band_keys).index(band)
    return slice(b, b + 1)


def summarise(tot: dict, band: str, band_keys=H.BAND_KEYS) -> dict:
    """Per-class pooled IoU (declared + raw), n and mean CE in the gated band, plus
    every band of the extent for the record."""
    sl = _band_slice(band, band_keys)
    out = {"iou": {}, "iou_raw": {}, "n": {}, "ce_mean": {}, "by_band": {}}
    for c, name in enumerate(H.CLASS_KEYS):
        n = float(tot["n"][c, sl].sum())
        u = float(tot["union"][c, sl].sum())
        ur = float(tot["unionraw"][c, sl].sum())
        out["n"][name] = int(n)
        out["iou"][name] = float(tot["inter"][c, sl].sum()) / u if u > 0 else None
        out["iou_raw"][name] = float(tot["interraw"][c, sl].sum()) / ur if ur > 0 else None
        out["ce_mean"][name] = float(tot["ce"][c, sl].sum()) / n if n > 0 else None
        out["by_band"][name] = {
            bk: {"n": int(tot["n"][c, b]),
                 "iou": (float(tot["inter"][c, b]) / float(tot["union"][c, b])
                         if float(tot["union"][c, b]) > 0 else None),
                 "iou_raw": (float(tot["interraw"][c, b]) / float(tot["unionraw"][c, b])
                             if float(tot["unionraw"][c, b]) > 0 else None)}
            for b, bk in enumerate(band_keys)}
    return out


def _signal_totals(batches, class_weight, rule) -> dict:
    tot = None
    for logits, codes, lv in batches:
        sig = H.per_class_signal(logits, codes, class_weight=class_weight,
                                 lift_valid=lv, decision_rule=rule)
        part = {k: sig[k].cpu() for k in ("n", "inter", "union", "interraw",
                                          "unionraw", "ce")}
        tot = part if tot is None else {k: tot[k] + part[k] for k in part}
    return tot


def _s8_params(trunk) -> list:
    names = []
    for n, _m in trunk.net.items():
        names.append(n)
        if n == trunk.s8_module:
            break
    return [p for n in names for p in trunk.net[n].parameters()]


def _forward(trunk, branch, data, idx, device, arm):
    x = data["x"][idx].to(device).float() / 255.0
    xn = trunk.normalise(x)
    s8 = trunk.s8_from_normalised(xn, torch.arange(x.shape[0]))
    if arm == "s8_zeros":
        s8 = torch.zeros_like(s8)                   # BOTH lifts (the near lift reads s8 too)
    elif arm in ("s8_detached", "frozen_trunk"):
        s8 = s8.detach()
    kw = {}
    if arm == "near_zeros":                         # NEW-2 R2: only the near lift sees zeros
        kw["near_source"] = torch.zeros_like(s8)
    out = branch(s8, data["grid"][idx].to(device), data["valid"][idx].to(device), **kw)
    return out["map_hires_logits"], H.lift_valid_to_fine(out["map_hires_lift_valid"])


def evaluate(trunk, branch, data, class_weight, rule, device, batch, arm,
             keep_logits: bool = False):
    trunk.eval()
    branch.eval()
    kept = []

    def batches():
        with torch.no_grad():
            for i in range(0, data["x"].shape[0], batch):
                idx = torch.arange(i, min(i + batch, data["x"].shape[0]))
                lg, lv = _forward(trunk, branch, data, idx, device, arm)
                if keep_logits:
                    kept.append((lg.cpu(), lv.cpu()))
                yield lg, data["codes"][idx].to(device), lv
    tot = _signal_totals(batches(), class_weight, rule)
    return tot, kept


def run_arm(arm: str, trunk0, branch0, data: dict, spec: dict, class_weight,
            device) -> dict:
    """Train one arm from the SHARED init; return step-0 and final statistics."""
    rule = spec["thresholds"]["decision_rule"]
    band = spec["thresholds"].get("band", "0_20")
    bks = tuple(branch0.cfg.band_keys)
    trunk = copy.deepcopy(trunk0).to(device)
    branch = copy.deepcopy(branch0).to(device)
    w_eval = class_weight.clone().to(device)          # the DECISION keeps launch weights
    w_loss = class_weight.clone().to(device)
    if arm == "lane_w0":
        w_loss[LANE] = 0.0
    trunk_trains = arm in ("healthy", "lane_w0", "s8_zeros", "near_zeros")
    params = list(branch.parameters()) + (_s8_params(trunk) if trunk_trains else [])
    trunk_before = [p.detach().clone() for p in _s8_params(trunk)]
    opt = torch.optim.AdamW(params, lr=float(spec.get("lr", 1e-3)), betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=float(spec.get("weight_decay", 0.0)))
    steps, bs = int(spec["steps"]), int(spec.get("batch", 4))
    every = int(spec.get("eval_every", max(1, steps // 10)))
    n = data["x"].shape[0]
    g = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    tot0, _ = evaluate(trunk, branch, data, w_eval, rule, device, bs, arm)
    step0 = summarise(tot0, band, bks)
    curve, finite, t0 = [], True, time.time()
    on_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        trunk.train()
        branch.train()
        idx = torch.randperm(n, generator=g)[:bs]
        lg, lv = _forward(trunk, branch, data, idx, device, arm)
        r = H.hires_map_ce(lg, data["codes"][idx].to(device), class_weight=w_loss,
                           lift_valid=lv)
        if not bool(torch.isfinite(r["loss"].detach())):
            finite = False
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        opt.step()
        if step % every == 0 or step == steps:
            tot, _ = evaluate(trunk, branch, data, w_eval, rule, device, bs, arm)
            sm = summarise(tot, band, bks)
            curve.append({"step": step, "train_loss": float(r["loss"].detach()),
                          "iou": sm["iou"], "iou_raw": sm["iou_raw"],
                          "ce_mean": sm["ce_mean"],
                          "s_per_step": round((time.time() - t0) / step, 4)})
            print(json.dumps({"arm": arm, "step": step,
                              "train_loss": curve[-1]["train_loss"],
                              "iou": sm["iou"]}), flush=True)
    totf, logits = evaluate(trunk, branch, data, w_eval, rule, device, bs, arm,
                            keep_logits=(arm == "healthy"))
    final = summarise(totf, band, bks)
    moved = sum(float((p.detach().cpu() - q.cpu()).abs().sum())
                for p, q in zip(_s8_params(trunk), trunk_before))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if on_cuda else None
    return {"arm": arm, "step0": step0, "final": final, "curve": curve,
            "loss_finite_every_step": finite, "trunk_s8_abs_change": moved,
            "w_loss": w_loss.cpu().tolist(), "w_decision": w_eval.cpu().tolist(),
            "s_per_step": round((time.time() - t0) / max(steps, 1), 4),
            "peak_mem_GiB": None if peak is None else round(peak, 3),
            "_logits": logits}


# --------------------------------------------------------------------------- #
# controls (on the SAME scored cells)                                          #
# --------------------------------------------------------------------------- #
def controls(data: dict, healthy_logits, class_weight, rule, band,
             band_keys=H.BAND_KEYS) -> dict:
    """C1 / C2 / C3 of the prereg §7, each reproducing a KNOWN value."""
    sl = _band_slice(band, band_keys)
    lv_all = torch.cat([lv for _lg, lv in healthy_logits])
    codes = data["codes"]
    sup = (codes != 255) & lv_all
    sup_b = torch.zeros_like(sup)
    rows = {bk: (i * 200, min((i + 1) * 200, int(codes.shape[1])))
            for i, bk in enumerate(band_keys)}
    for bk in ([band] if band != "all" else list(rows)):
        a, b = rows[bk]
        sup_b[:, a:b] = sup[:, a:b]
    # C1: drivable everywhere -- a decision-level control (no logits involved)
    n_sc = int(sup_b.sum())
    n_drv = int(((codes == DRIVABLE) & sup_b).sum())
    c1_iou = {}
    for c, name in enumerate(H.CLASS_KEYS):
        gc = (codes == c) & sup_b
        pc = sup_b if c == DRIVABLE else torch.zeros_like(sup_b)
        u = int((gc | pc).sum())
        c1_iou[name] = (int((gc & pc).sum()) / u) if u else None
    c1_ok = (n_sc > 0 and abs((c1_iou["drivable"] or 0.0) - n_drv / n_sc) <= 1e-9
             and all(v in (None, 0.0) for k, v in c1_iou.items() if k != "drivable"))
    # C2: logits = 20 * one-hot(label), decided under the DECLARED rule
    lab = torch.where(codes == 255, torch.zeros_like(codes), codes).long()
    z = 20.0 * torch.nn.functional.one_hot(lab, 8).permute(0, 3, 1, 2).float()
    sig = H.per_class_signal(z, codes, class_weight=class_weight, lift_valid=lv_all,
                             decision_rule=rule)
    c2 = {}
    for c, name in enumerate(H.CLASS_KEYS):
        u = float(sig["union"][c, sl].sum())
        c2[name] = float(sig["inter"][c, sl].sum()) / u if u else None
    c2_ok = (all(v in (None, 1.0) for v in c2.values())
             and any(v == 1.0 for v in c2.values()))
    # C3: uniform weights => the two decision rules are bit-identical (MAIN's logits)
    c3_ok = bool(healthy_logits) and all(
        torch.equal(H.decide(lg, "prior_corrected", torch.ones(8)), H.decide(lg, "raw"))
        for lg, _lv in healthy_logits)
    return {"C1_constant_drivable": {"iou": c1_iou, "n_scored": n_sc, "n_drivable": n_drv,
                                     "expected_iou_drivable": (n_drv / n_sc) if n_sc else None,
                                     "reproduced": bool(c1_ok)},
            "C2_gt_as_logits": {"iou": c2, "reproduced": bool(c2_ok)},
            "C3_rule_identity_w_ones": {"reproduced": bool(c3_ok)}}


# --------------------------------------------------------------------------- #
# the verdict                                                                  #
# --------------------------------------------------------------------------- #
def _passes_bars(final: dict, th: dict) -> dict:
    return {cls: (final["iou"].get(cls) is not None
                  and final["iou"][cls] >= float(bar))
            for cls, bar in th["iou"].items()}


def verdict(results: dict, spec: dict, ctrl: dict | None,
            axis_guard: dict | None = None) -> dict:
    th = spec["thresholds"]
    min_cells = int(th["min_cells"])
    ce_max = th.get("per_class_ce_final_over_step0_max")
    main = results.get("healthy")
    out = {"band": th.get("band", "0_20"), "decision_rule": th["decision_rule"]}
    if main is None:
        out.update(MAIN={"verdict": "NOT RUN"}, regression_arms={},
                   controls_reproduced=False, G_MAP_OVERFIT="FAIL")
        return out
    n = main["final"]["n"]
    short = {c: v for c, v in n.items() if v < min_cells}
    bars = _passes_bars(main["final"], th)
    ce_ok = {}
    if ce_max is not None:
        for c in H.CLASS_KEYS:
            a, b = main["step0"]["ce_mean"].get(c), main["final"]["ce_mean"].get(c)
            ce_ok[c] = bool(a is not None and b is not None and a > 0
                            and b <= float(ce_max) * a)
    main_ok = (not short and all(bars.values()) and all(ce_ok.values())
               and main["loss_finite_every_step"])
    out["MAIN"] = {"presence_short": short, "inconclusive": bool(short),
                   "bars": bars, "ce_ratio_ok": ce_ok,
                   "loss_finite_every_step": main["loss_finite_every_step"],
                   "verdict": ("INCONCLUSIVE" if short else "PASS" if main_ok else "FAIL")}
    reg = {}
    for arm, classes in (spec.get("must_fail") or {}).items():
        if arm not in results:
            reg[arm] = {"ran": False, "failed_as_required": False}
            continue
        passed = _passes_bars(results[arm]["final"], th)
        failed = [c for c in classes if not passed[c]]
        need_all = bool((spec.get("must_fail_all") or {}).get(arm, False))
        ok = (len(failed) == len(classes)) if need_all else bool(failed)
        reg[arm] = {"ran": True, "must_fail": list(classes), "failed": failed,
                    "rule": "all" if need_all else "any", "failed_as_required": ok}
    out["regression_arms"] = reg
    out["informative"] = {a: ({"ran": True, "bars": _passes_bars(results[a]["final"], th)}
                              if a in results else
                              {"ran": False, "why": NOT_RUN_WHY.get(a, "not in --arms")})
                          for a in (spec.get("informative_arms") or [])}
    # ⛔ PREREG §2 / §9a.6: every label time-checked at 1 ms. The pose-content check is
    # a weaker guard (5 cm on the GT pose); a run that only had it cannot PASS.
    kinds = sorted({str(g.get("kind")) for g in (axis_guard or {}).values()})
    out["time_guard"] = {"kinds": kinds,
                         "time_1ms_on_every_clip": bool(axis_guard) and kinds == ["time_1ms"]}
    c_ok = ctrl is not None and all(v["reproduced"] for v in ctrl.values())
    c1_fails = ctrl is not None and not all(
        (ctrl["C1_constant_drivable"]["iou"].get(c) or 0.0) >= float(bar)
        for c, bar in th["iou"].items())
    out["controls"] = ctrl
    out["controls_reproduced"] = bool(c_ok and c1_fails)
    gate = (out["MAIN"]["verdict"] == "PASS"
            and all(r["failed_as_required"] for r in reg.values())
            and out["controls_reproduced"]
            and out["time_guard"]["time_1ms_on_every_clip"])
    out["G_MAP_OVERFIT"] = "PASS" if gate else "FAIL"
    return out


def _fingerprint(module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _fingerprint_without(module, prefix: str) -> str:
    """:func:`_fingerprint` over the state_dict minus the keys under ``prefix`` -- the NEW-2
    R2 branch's shared parameters, comparable with the NEW-2 MAIN's fingerprint."""
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        if k.startswith(prefix):
            continue
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--v2-cache", required=True, type=Path)
    ap.add_argument("--gt-root", required=True, type=Path)
    ap.add_argument("--extrinsics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output DIRECTORY")
    ap.add_argument("--class-weights", default=None,
                    help="the LAUNCH weights json (fills the spec's placeholder)")
    ap.add_argument("--decision-rule", default=None, choices=H.DECISION_RULES,
                    help="the launch config's declared rule (fills the placeholder)")
    ap.add_argument("--frameset", type=Path, default=None,
                    help="the prereg's gmo_frameset.json: its md5 must equal the spec's")
    ap.add_argument("--cam-ts-dir", type=Path, default=None)
    ap.add_argument("--launch-commit", default="")
    ap.add_argument("--launch-argv-sha256", default="")
    ap.add_argument("--arms", default="healthy,lane_w0,s8_zeros,s8_detached")
    ap.add_argument("--x-max-m", type=float, default=EXTENT_REFCV7.x_max_m,
                    help="the LAUNCH map extent ahead (SPEC_REFCV7 §12: 100)")
    ap.add_argument("--y-half-m", type=float, default=EXTENT_REFCV7.y_half_m,
                    help="the LAUNCH map extent to each side (SPEC_REFCV7 §12: 30)")
    ap.add_argument("--near-lift-m", type=float, default=0.0,
                    help="NEW-2 R2 (SPEC_REFCV7 A12): the 0.1 m near lift over x in [0, M) m "
                         "(MapHiresConfig.near_lift_x_m); 0 = the NEW-2 branch as landed")
    ap.add_argument("--grad-ckpt", choices=("on", "off"), default="on",
                    help="the launch decoder's checkpointing (SPEC_REFCV7 §12 item 4)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    extent = MapExtent(float(a.x_max_m), float(a.y_half_m))
    spec = load_spec(a.spec, class_weights=a.class_weights, decision_rule=a.decision_rule,
                     band_keys=extent.band_keys)
    if a.frameset is not None:
        md5 = hashlib.md5(a.frameset.read_bytes()).hexdigest()
        if spec.get("frameset_md5") and md5 != spec["frameset_md5"]:
            raise SystemExit(f"⛔ frameset md5 {md5} != the spec's {spec['frameset_md5']}")
    arms = [x for x in a.arms.split(",") if x]
    for x in arms:
        if x not in ARMS:
            raise SystemExit(f"⛔ unknown arm {x!r}")
    for x in list(spec.get("must_fail") or {}) + ["healthy"]:
        if x not in arms:
            raise SystemExit(f"⛔ the gated arm {x!r} is not in --arms")
    if "near_zeros" in arms and not float(a.near_lift_m) > 0.0:
        raise SystemExit("⛔ the near_zeros arm needs --near-lift-m > 0: without a near lift "
                         "it is MAIN itself and cannot be a regression arm")
    if spec.get("near_lift_m") is not None \
            and float(spec["near_lift_m"]) != float(a.near_lift_m):
        raise SystemExit(f"⛔ the spec registers near_lift_m {spec['near_lift_m']}, "
                         f"--near-lift-m is {a.near_lift_m}")
    from tanitad.models import timm_trunk as TT
    from tanitad.models.trunk_shapes import frame_for_width
    tsp = spec.get("trunk") or {}
    hw = tuple(tsp.get("image_hw", (416, 1024)))
    eq = int(tsp.get("equalize_bottom_rows", 43))
    torch.manual_seed(int(spec.get("seed", 0)))
    trunk = TT.TimmResNetTrunk(TT.TimmTrunkConfig(
        model_name=str(tsp.get("model", "resnet101.a1_in1k")),
        pretrained=bool(tsp.get("pretrained", True)), frames=3, image_hw=hw,
        frozen_bn=True, equalize_bottom_rows=eq,
        chunk_ckpt=int(tsp.get("chunk_ckpt", 0) or 0), bf16=bool(tsp.get("bf16", False)),
        channels_last=bool(tsp.get("channels_last", False)),
        fold_bn=bool(tsp.get("fold_bn", False))))
    tap = trunk.enable_s8_tap()
    cw, cw_stamp = class_weights_of(spec, extent)
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(extent.x_max_m),
                            y_half_m=float(extent.y_half_m),
                            grad_ckpt=(a.grad_ckpt == "on"),
                            near_lift_x_m=float(a.near_lift_m),
                            decision_rule=spec["thresholds"]["decision_rule"],
                            class_weights_sha256=cw_stamp.get("sha256") or "")
    branch = H.MapHiresBranch(hcfg, d_image=trunk.s8_dim, image_hw=trunk.s8_shape)
    fp = {"trunk_init_sha256": _fingerprint(trunk),
          "branch_init_sha256": _fingerprint(branch)}
    if getattr(branch, "near", None) is not None:
        fp["branch_init_sha256_without_near"] = _fingerprint_without(branch, "near.")
    frame = frame_for_width(int(hw[1]), int(hw[0]))
    data = load_frames(spec, a.v2_cache, a.gt_root, a.extrinsics, frame, hcfg, eq,
                       cam_ts_dir=a.cam_ts_dir)
    results = {arm: run_arm(arm, trunk, branch, data, spec, cw, a.device) for arm in arms}
    ctrl = controls(data, results["healthy"]["_logits"], cw,
                    spec["thresholds"]["decision_rule"],
                    spec["thresholds"].get("band", "0_20"), extent.band_keys)
    v = verdict(results, spec, ctrl, data["axis_guard"])
    for r in results.values():
        r.pop("_logits", None)
    a.out.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "tanitad.g_map_overfit_record/1",
           "spec_sha256": hashlib.sha256(a.spec.read_bytes()).hexdigest(),
           "frameset_md5": spec.get("frameset_md5"),
           "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
           "launch_commit": a.launch_commit, "launch_argv_sha256": a.launch_argv_sha256,
           "class_weights": cw_stamp, "decision_rule": spec["thresholds"]["decision_rule"],
           "extent": extent.as_dict(), "branch_config": hcfg.as_dict(),
           "near_lift_m": float(a.near_lift_m),
           "tap": tap, "trunk_levers": {k: tsp.get(k) for k in
                                        ("chunk_ckpt", "bf16", "channels_last", "fold_bn")},
           "fingerprints": fp, "branch_params": branch.param_breakdown(),
           "n_frames": len(data["sha12"]), "frames_sha12": data["sha12"],
           "raw_frames": data["raw_frame"], "axis_guard": data["axis_guard"],
           "optimiser": {"name": "AdamW", "lr": float(spec.get("lr", 1e-3)),
                         "betas": [0.9, 0.999], "eps": 1e-8,
                         "weight_decay": float(spec.get("weight_decay", 0.0)),
                         "batch": int(spec.get("batch", 4)), "steps": int(spec["steps"]),
                         "seed": int(spec.get("seed", 0))},
           "device": a.device, "verdict": v, "results": results}
    (a.out / "g_map_overfit.json").write_text(json.dumps(rec, indent=1, default=float),
                                              encoding="utf-8")
    print(json.dumps({k: v[k] for k in ("MAIN", "regression_arms", "controls_reproduced",
                                         "time_guard", "G_MAP_OVERFIT")},
                     indent=1, default=str))
    return 0 if v["G_MAP_OVERFIT"] == "PASS" else 2


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
