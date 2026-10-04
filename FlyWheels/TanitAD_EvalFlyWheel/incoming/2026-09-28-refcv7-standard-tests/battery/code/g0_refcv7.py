"""GATE G0 (SPEC.md §2) -- the refcv7 loader reproduces refcv7's OWN in-run eval row.

    python g0_refcv7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_1500.pt \
        --config D:/refcv7_eval_kit/ckpt/config.json --metrics <metrics.jsonl copy> \
        --seeds 0,1,2,3,4,5,6,7 --out <g0.json>

The trainer's own `compute_losses_v3`, `_eval_row_from_acc` and `detection_metrics.summarise`
(`refc_v3_train.py:9658-9714`) on the loader's eval dataset over the SAME 128 windows in the SAME 8
batches of 16 (`inrun_eval_perm`). Recorded departures: (1) the forward is micro-batched INSIDE the
single model call (`microbatch.py`); (2) only the LAW frame of `future_frames` reaches the device;
(3) `--trunk-compile` is off (loader); (4) the DDIM draw is seeded per inference seed; (5) the
parameters carry their TRAINING requires_grad flags (the in-run eval's state), restored from the
built model's `declare_grad_unreachable` declarations -- the loader leaves them all False.

Ported from the refcv6 package's `reproduce_inrun_eval.py` (2026-09-23 battery), refcv7 changes only.
Every artifact is asserted by the CALLER on the JSON this writes, never on this process's exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tanitad.eval import refcv7_loader as L  # noqa: E402

L.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402

# --------------------------------------------------------------------------------------- #
# term classes -- SPEC §2, by NAME, written before any value was read                      #
# --------------------------------------------------------------------------------------- #
EXCLUDED_EXACT = {"eval_goal_gate_grad", "eval_step"}
COUNT_EXACT = {"eval_batches", "eval_windows", "eval_slot_valid_frac", "eval_tac_label_rows",
               "eval_tac_label_v7", "eval_nav_injected", "eval_ego_injected",
               "eval_box3d_visible_filter"}
MODEL_DEP_COUNT_SUFFIX = ("_n_conf", "_tp@gate", "_n_ignore_masked_slots", "_n_presence_exempt")
DET_TOKENS = ("_prec@gate", "_rec@gate", "_f1@gate", "_ap2m", "_auroc_", "_cls_acc_tp",
              "_det_ap", "_det_map", "_conf_ratio", "_centre_err_p50")
MATCHED_BASE = {"agent_centre", "agent_cls", "agent_presence", "agent_size", "agent_yaw",
                "box3d", "box3d_centre", "box3d_cls", "box3d_h", "box3d_occ", "box3d_presence",
                "box3d_presence_focal", "box3d_rates", "box3d_size", "box3d_yaw", "box3d_z",
                "box3d_vis1"}
#: t(0.995, K-1) keyed by K (the number of seeds). K = 24 added by SPEC AMENDMENT A5: t(0.995, 23) = 2.807.
T_995 = {2: 63.657, 3: 9.925, 4: 5.841, 5: 4.604, 6: 4.032, 7: 3.707, 8: 3.499, 9: 3.355,
         10: 3.250, 12: 3.106, 16: 2.947, 24: 2.807}
#: the seeds G0 was REGISTERED with (SPEC §2): "as registered" and A2 are always judged on these only
REGISTERED_SEEDS = tuple(range(8))
#: SPEC AMENDMENT A5: the STOCHASTIC class is judged on K >= 24 seeds at milestones
A5_MIN_SEEDS = 24


def term_class(k: str) -> str:
    """SPEC §2's table. ⚠️ A standalone `n` / `npos` TOKEN marks a count (a raw substring test
    matches `lon_...`; the refcv6 G0 found that bug before its first output)."""
    if k in EXCLUDED_EXACT or "_calib_" in k:
        return "EXCLUDED"
    s = k[len("eval_"):] if k.startswith("eval_") else k
    if k.endswith("_conf_ratio_alarm"):
        return "DERIVED"                      # judged through conf_ratio (SPEC §2)
    if k.endswith(MODEL_DEP_COUNT_SUFFIX):
        return "DETECTION"
    toks = s.split("_")
    if (k in COUNT_EXACT or "n" in toks or "npos" in toks or s.startswith("agent_rows_")
            or s.startswith("n_map_hires_cells")):
        return "COUNT"
    if any(t in k for t in DET_TOKENS) and (s.startswith("agent_") or s.startswith("box3d_")):
        return "DETECTION"
    if (s in MATCHED_BASE or s.startswith("agent_presence_layer") or s.startswith("agent_layer")
            or s.startswith("box3d_presence_layer") or s.startswith("box3d_layer")):
        return "MATCHED"
    for pre in ("map_hires_inter_", "map_hires_union_", "map_hires_interraw_",
                "map_hires_unionraw_"):
        if s.startswith(pre):
            return "MAP10_COUNTS"
    if s.startswith("map_hires_iou_") or s.startswith("map_hires_iouraw_"):
        return "MAP10_IOU"
    return "SMOOTH_OR_STOCHASTIC"


# --------------------------------------------------------------------------------------- #
# the LAW-only future frame (compute_losses_v3 reads fut_frames[:, LAW_AHEAD-1] only, :4685) #
# --------------------------------------------------------------------------------------- #
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


def make_g0_dataset_cls(tr, law_ahead: int):
    class G0Windows(tr.V3Dataset):
        def _window_u8(self, i: int) -> dict:          # refb_train.py:220-231, future cut to LAW
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {
                "frames": ep.frames[t:t + w],
                "actions": ep.actions[t:t + w],
                "future_frames": ep.frames[t + w:t + w + law_ahead],
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }
    return G0Windows


def collate(e_ds, idx_list, law_idx):
    import torch.utils.data as tud
    items = [e_ds[i] for i in idx_list]
    eb = tud.default_collate(items)
    ff = eb["future_frames"]
    eb["future_frames"] = LawOnly(ff[:, law_idx].clone(), law_idx)
    del ff
    return eb


class MmapBatches:
    """SPEC AMENDMENT A5 item 5: the collated batches written ONCE to disk, read back through a
    file-backed mmap on every use (indexing or iteration). A batch file that already exists is reused
    (same checkpoint-md5-keyed directory), so a crashed G0 re-collates nothing. `LawOnly` is stored as
    its uint8 tensor + index and rebuilt on load. The content is bit-identical to the old in-RAM list
    (`g0_diag_r7.py`'s `s0` arm reproduces G0's stored seed-0 row through the same path)."""

    def __init__(self, root: Path, makers):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = []
        for i, make in enumerate(makers):
            p = self.root / f"batch_{i}.pt"
            if not p.exists():
                eb = make()
                ser = {k: ({"__lawonly__": v.u8, "law_idx": v.law_idx} if isinstance(v, LawOnly) else v)
                       for k, v in eb.items()}
                torch.save(ser, str(p) + ".part")
                Path(str(p) + ".part").replace(p)
                del eb, ser
            self.paths.append(p)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(str(self.paths[i]), map_location="cpu", weights_only=False, mmap=True)
        return {k: (LawOnly(v["__lawonly__"], v["law_idx"])
                    if isinstance(v, dict) and "__lawonly__" in v else v) for k, v in d.items()}

    def __iter__(self):
        return (self[i] for i in range(len(self.paths)))

    def remove(self):
        for p in self.paths:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            self.root.rmdir()
        except OSError:
            pass


def sub_batch(eb, n, rows=None):
    rows = list(range(n)) if rows is None else rows
    o = {}
    for k, v in eb.items():
        if torch.is_tensor(v) and v.dim() >= 1:
            o[k] = v[rows]
        elif isinstance(v, LawOnly):
            o[k] = LawOnly(v.u8[rows], v.law_idx)
        elif isinstance(v, (list, tuple)):
            o[k] = type(v)(v[i] for i in rows)
        else:
            o[k] = v
    return o


# --------------------------------------------------------------------------------------- #
# requires_grad AS TRAINED (SPEC §2)                                                        #
# --------------------------------------------------------------------------------------- #
def training_flags(model) -> dict:
    """{param name: requires_grad as the TRAINER holds it}: True, except parameters under a module
    that carries `_gradreach.GRAD_UNREACHABLE_FLAG` (`declare_grad_unreachable` froze them)."""
    from tanitad.models import _gradreach as _gr
    dead = _gr.grad_unreachable_prefixes(model)
    out = {}
    for n, _p in model.named_parameters():
        out[n] = not any(n == pre or n.startswith(pre + ".") for pre in dead)
    return {"flags": out, "unreachable_prefixes": dead}


def set_flags(model, flags: dict):
    for n, p in model.named_parameters():
        p.requires_grad_(bool(flags[n]))


def buffer_digest(model) -> str:
    h = hashlib.sha256()
    for n, b in model.named_buffers():
        if b is None:
            continue
        h.update(n.encode())
        h.update(b.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------- #
# the in-run eval, replayed                                                                 #
# --------------------------------------------------------------------------------------- #
def scalar_items(el: dict) -> dict:
    out = {}
    for k, v in el.items():
        if torch.is_tensor(v) and v.ndim == 0:
            out[k] = float(v.detach())
        elif isinstance(v, (int, float, bool)) and not str(k).startswith("_"):
            out[k] = float(v)
    return out


def run_eval(tr, model, batches, device, mode, abl, seed, batch_size):
    """refc_v3_train.py:9658-9747 with the RNG seeded per inference seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    acc, nb, per_batch, det_packs = {}, 0, [], {}
    with torch.no_grad():
        for eb in batches():
            el = tr.compute_losses_v3(model, eb, device, mode=mode, ablate_frames=abl)
            row = {}
            for k, v in el.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                    row[k] = float(v.detach())
                elif isinstance(v, (int, float, bool)):
                    acc[k] = acc.get(k, 0.0) + float(v)
                    row[k] = float(v)
            for hd in tr._det_metrics.HEADS:
                if el.get(f"_det_pack_{hd}"):
                    det_packs.setdefault(hd, []).extend(el[f"_det_pack_{hd}"])
            per_batch.append(row)
            nb += 1
            del el
    erow = tr._eval_row_from_acc(acc, nb, model)
    for hd, pk in det_packs.items():
        for dk, dv in tr._det_metrics.summarise(pk, hd).items():
            erow[dk] = (None if (isinstance(dv, float) and dv != dv) else round(float(dv), 5))
    erow.update(eval_batches=nb, eval_windows=nb * int(batch_size))
    return erow, per_batch, {hd: len(v) for hd, v in det_packs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--micro", default="2,2,3,3,3,3")
    ap.add_argument("--mutations", default="m1,m2,m4")
    ap.add_argument("--diagnostic-arms", default="eps0",
                    help="REPORTED, never judged (2026-10-04 diagnosis): 'eps0' = seed-0 replay with the "
                         "DDIM draw zeroed -- tells whether the in-run row sits at the noise-free decode")
    ap.add_argument("--skip-wrapper-control", action="store_true")
    ap.add_argument("--n-batches", type=int, default=0, help="PROBE ONLY: fewer batches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-cache", default=None,
                    help="dir for the collated batches (SPEC A5 item 5; default next to --out)")
    ap.add_argument("--keep-batch-cache", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    tr = L.trainer()
    device = "cuda"
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "g0_refcv7.py", "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest()
           if spec.exists() else None, "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
           "config": a.config, "config_md5": L.md5_file(a.config), "micro": a.micro,
           "seeds": a.seeds, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "tanitad_file": __import__("tanitad").__file__, "started": time.strftime("%FT%T"),
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn": torch.backends.cudnn.allow_tf32}}
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        json.dump(rec, open(out_p.with_suffix(".partial.json"), "w", encoding="utf-8"), indent=1,
                  default=str)

    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
    rec["model"] = {k: mrec.get(k) for k in ("state_dict", "param_breakdown",
                                             "anchor_file_vs_ckpt_buffers", "declared_vs_built",
                                             "stamp_checks", "trunk_memory_levers_built",
                                             "trunk_memory_levers_run", "departures", "mode",
                                             "decoder_steps", "sampler", "build_s", "argv_remap")}
    step = int(a.step if a.step is not None else mrec["state_dict"]["step"])
    rec["step"] = step
    print(f"[g0] model built in {mrec['build_s']} s; strict missing="
          f"{len(mrec['state_dict']['missing'])} unexpected={len(mrec['state_dict']['unexpected'])} "
          f"step={step}; param_breakdown equal={mrec['param_breakdown']['equal']}", flush=True)
    # ---- requires_grad as trained ---------------------------------------------------- #
    tf = training_flags(model)
    loader_flags = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
    rec["requires_grad"] = {"loader_true": sum(loader_flags.values()),
                            "trained_true": sum(tf["flags"].values()),
                            "n_params_tensors": len(tf["flags"]),
                            "unreachable_prefixes": tf["unreachable_prefixes"]}
    # ---- the in-run row -------------------------------------------------------------- #
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"[g0] {len(ev)} eval rows at step {step} in {a.metrics}")
    inrun = ev[0]
    rec["inrun_row_n_keys"] = len(inrun)
    # ---- eval dataset + the fixed subset --------------------------------------------- #
    t_ds = time.time()
    law_idx = int(tr.LAW_AHEAD) - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    rec["dataset"] = {k: drec.get(k) for k in ("n_episodes", "n_windows", "label_clock", "labels",
                                               "nav", "max_speed_v6", "agent_join", "map_fine",
                                               "join3d", "vis1", "camera_coverage")}
    B = int(args.batch)
    NB = int(a.n_batches or args.eval_batches)
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), B)
    rec["perm_sha256"] = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    rec["n_batches"] = NB
    if a.n_batches:
        rec["PROBE_ONLY_fewer_batches"] = True
    print(f"[g0] eval dataset: {drec['n_episodes']} episodes -> {drec['n_windows']} windows "
          f"({time.time() - t_ds:.0f}s); subset {len(perm)}; batches {NB}", flush=True)
    t_c = time.time()
    # SPEC AMENDMENT A5 item 5 (memory, not numbers): the collated batches live on DISK and are read
    # back through a file-backed mmap (no commit charge) instead of ~3.9 GB of RAM; bit-identical.
    cached = MmapBatches(Path(a.batch_cache) if a.batch_cache else
                         out_p.parent / f"g0_batch_cache_{rec['ckpt_md5'][:8]}",
                         [lambda i=i: collate(e_ds, perm[i * B:(i + 1) * B], law_idx)
                          for i in range(NB)])
    rec["batch_cache"] = str(cached.root)
    rec["collate_s"] = round(time.time() - t_c, 1)
    print(f"[g0] collated {NB} batches in {rec['collate_s']} s (disk cache {cached.root})", flush=True)

    def batches():
        return iter(cached)
    patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    sizes = [int(x) for x in a.micro.split(",")]
    set_flags(model, tf["flags"])
    bd0 = buffer_digest(model)
    torch.cuda.reset_peak_memory_stats()
    bank()
    # ---- requires_grad control: one 2-window forward, eps zeroed, both flag states ----- #
    orig_randn_like = torch.randn_like
    try:
        torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
        sb = sub_batch(cached[0], 2)
        res = {}
        for nm, fl in (("trained_flags", tf["flags"]), ("loader_flags", loader_flags)):
            set_flags(model, fl)
            torch.manual_seed(0)
            with torch.no_grad():
                res[nm] = scalar_items(tr.compute_losses_v3(model, sb, device, mode=mode,
                                                            ablate_frames=abl))
        set_flags(model, tf["flags"])
        diff = {k: (res["trained_flags"][k], res["loader_flags"].get(k))
                for k in res["trained_flags"] if res["trained_flags"][k] != res["loader_flags"].get(k)}
        rec["requires_grad_control"] = {"batch": 2, "ddim_eps": "zeros", "n_terms":
                                        len(res["trained_flags"]), "n_differ": len(diff),
                                        "differ_first20": dict(list(diff.items())[:20]),
                                        "max_rel_all": max((abs(a_ - b_) / max(abs(a_), 1e-12)
                                                            for a_, b_ in diff.values()
                                                            if b_ is not None), default=0.0),
                                        "differ_all_rel": {k: abs(a_ - b_) / max(abs(a_), 1e-12)
                                                           for k, (a_, b_) in diff.items()
                                                           if b_ is not None}}
        print(f"[g0] requires_grad control: {len(diff)} of {len(res['trained_flags'])} terms "
              f"differ between trained and loader flags (dev box)", flush=True)
    finally:
        torch.randn_like = orig_randn_like
    bank()
    # ---- the seeds -------------------------------------------------------------------- #
    mb = MicroBatchForward(model, sizes).install()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    by_seed = {}
    for s in seeds:
        t_s = time.time()
        erow, pb, npk = run_eval(tr, model, batches, device, mode, abl, s, B)
        dg = buffer_digest(model)
        by_seed[s] = {"row": erow, "per_batch": pb, "det_packs": npk,
                      "wall_s": round(time.time() - t_s, 1), "buffers_unchanged": dg == bd0}
        print(f"[g0] seed {s}: eval_loss {erow.get('eval_loss')} eval_traj {erow.get('eval_traj')} "
              f"({time.time() - t_s:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB,"
              f" buffers unchanged {dg == bd0})", flush=True)
        rec["by_seed"] = {str(k): v for k, v in by_seed.items()}
        bank()
    rec["merge_rules_flagged"] = mb.flagged()
    rec["merge_rules_all"] = mb.rules
    rec["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    # ---- mutations (M1 must FAIL; M2 / M4 power probes) ------------------------------ #
    muts = [m.strip() for m in a.mutations.split(",") if m.strip()]
    rec["mutations"] = {}
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    for m in muts:
        t_m = time.time()
        info = {}
        try:
            if m == "m1":
                old = int(getattr(trunk.cfg, "equalize_bottom_rows", 0) or 0)
                calls0 = int(getattr(trunk, "equalize_calls", 0))
                object.__setattr__(trunk.cfg, "equalize_bottom_rows", 0)
                info = {"what": "trunk bottom-row equalisation DROPPED (FIX-3 defect, "
                                "D-REFCV6-EQUALIZE-DROPPED)", "equalize_rows_as_built": old}
                erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                info["equalize_calls_during_mutation"] = int(getattr(trunk, "equalize_calls", 0)) - calls0
                object.__setattr__(trunk.cfg, "equalize_bottom_rows", old)
            elif m == "m2":
                br = model._map_hires
                old = str(br.cfg.decision_rule)
                object.__setattr__(br.cfg, "decision_rule", "raw")
                info = {"what": "map decision rule prior_corrected -> raw (the declared rule "
                                "dropped)", "rule_as_built": old}
                erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                object.__setattr__(br.cfg, "decision_rule", old)
            elif m == "m4":
                dec = model.core.decoder
                old = dec.residual_prior
                from tanitad.models import kinematic_prior as _kp
                dec.residual_prior = _kp.RESIDUAL_PRIOR_OFF
                info = {"what": "NEW-1 residual prior OFF at eval", "prior_as_built": str(old)}
                try:
                    erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                finally:
                    dec.residual_prior = old
            else:
                raise SystemExit(f"unknown mutation {m}")
            info.update(seed=0, row=erow_m, wall_s=round(time.time() - t_m, 1))
        except SystemExit:
            raise
        except Exception as exc:                        # noqa: BLE001 -- a loud failure IS a detection
            info.update(raised=f"{type(exc).__name__}: {str(exc)[:400]}")
        rec["mutations"][m] = info
        print(f"[g0] mutation {m}: {info.get('raised') or 'ran'} ({info.get('wall_s')} s)", flush=True)
        bank()
    # ---- diagnostic arms: REPORTED, NEVER JUDGED (not mutations; no verdict reads them) ------- #
    rec["diagnostic_arms"] = {}
    for dname in [x.strip() for x in a.diagnostic_arms.split(",") if x.strip()]:
        t_d = time.time()
        if dname != "eps0":
            rec["diagnostic_arms"][dname] = {"raised": "unknown diagnostic arm"}
            continue
        _orig = torch.randn_like
        torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
        try:
            erow_d, pb_d, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
            rec["diagnostic_arms"][dname] = {
                "what": "seed-0 replay with the DDIM eps zeroed (refc.py:2802); REPORTED, never judged",
                "row": erow_d, "per_batch_traj": [r.get("traj") for r in pb_d],
                "wall_s": round(time.time() - t_d, 1)}
        except Exception as exc:                        # noqa: BLE001 -- diagnostic only
            rec["diagnostic_arms"][dname] = {"raised": f"{type(exc).__name__}: {str(exc)[:300]}"}
        finally:
            torch.randn_like = _orig
        print(f"[g0] diagnostic {dname}: eval_traj "
              f"{(rec['diagnostic_arms'][dname].get('row') or {}).get('eval_traj')} "
              f"(in-run {inrun.get('eval_traj')})", flush=True)
        bank()
    info_bd = buffer_digest(model)
    rec["buffers_unchanged_after_all"] = info_bd == bd0
    mb.remove()
    # ---- wrapper control (SPEC §2, the refcv6 A2 form) --------------------------------- #
    if not a.skip_wrapper_control:
        import wrapper_probe_r7 as W
        rec["wrapper_control"] = W.probe(tr, model, cached[0], device, mode, abl, trunk,
                                         sub_batch_fn=sub_batch)
        print(f"[g0] wrapper clause: {rec['wrapper_control'].get('clause')} "
              f"({rec['wrapper_control'].get('clause_condition')})", flush=True)
    # SPEC AMENDMENT A5 item 3: as registered (seeds 0..7) FIRST, then A2 (seeds 0..7), then A5 (all
    # seeds, >= 24) -- the gate. With fewer than 24 seeds there is no A5 verdict and A2 gates (pre-A5).
    reg = {s: v for s, v in by_seed.items() if int(s) in REGISTERED_SEEDS}
    rec["verdict_as_registered"] = judge(inrun, reg, rec)
    rec["verdict_A2"] = judge(inrun, reg, rec, amend="A2")
    if len(by_seed) >= A5_MIN_SEEDS:
        rec["verdict"] = judge(inrun, by_seed, rec, amend="A5")      # SPEC AMENDMENT A5 gates milestones
    else:
        rec["verdict"] = rec["verdict_A2"]
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["finished"] = time.strftime("%FT%T")
    json.dump(rec, open(out_p, "w", encoding="utf-8"), indent=1, default=str)
    if not a.keep_batch_cache:
        cached.remove()                      # derived, re-creatable; D: is short of space
    v = rec["verdict"]
    print(json.dumps({"G0_as_registered": rec["verdict_as_registered"]["G0"],
                      "reasons_as_registered": rec["verdict_as_registered"]["reasons"][:10],
                      **{k: v[k] for k in ("G0", "amendment", "reasons", "n_terms", "by_class_counts",
                                           "medians", "mutation_detection")}},
                     indent=1, default=str)[:6000])
    print(f"[g0] wrote {a.out} ({rec['wall_s']} s)")


# --------------------------------------------------------------------------------------- #
# the verdict                                                                               #
# --------------------------------------------------------------------------------------- #
def _isnull(v):
    return v is None or (isinstance(v, float) and v != v)


def tol_ok(cls: str, k: str, x: float, y: float) -> tuple[bool, float, str]:
    """(ok, deviation, tol string) for a DETERMINISTIC comparison of reproduction y vs in-run x."""
    d = abs(y - x)
    rel = d / max(abs(x), 1e-12)
    if cls == "COUNT":
        exact_key = k.startswith("eval_map_hires_") or k.startswith("eval_n_map_hires")
        tol = 1e-6 * max(1.0, abs(x)) if exact_key else 1e-5 + 1e-9
        return d <= tol, d, "exact (5 dp)"
    if cls == "DETECTION":
        if k.endswith(MODEL_DEP_COUNT_SUFFIX):
            return (rel <= 0.05 or d <= 2.0), rel, "rel<=5% or abs<=2"
        if k.endswith("_conf_ratio") or k.endswith("_centre_err_p50"):
            return rel <= 0.05, rel, "rel<=5%"
        return d <= 0.02, d, "abs<=0.02"
    if cls == "MATCHED":
        return rel <= 0.15, rel, "rel<=15%"
    if cls == "MAP10_COUNTS":
        return d <= max(0.02 * abs(x), 25.0), rel, "abs<=max(2%|x|, 25 cells)"
    if cls == "MAP10_IOU":
        return d <= 0.01, d, "abs<=0.01"
    if cls == "SMOOTH":
        ok = (d <= 1e-3) if abs(x) < 0.1 else (rel <= 0.01)
        return ok, rel, "abs<=1e-3 (|x|<0.1) else rel<=1%"
    raise ValueError(cls)


#: SPEC AMENDMENT A2 (registered 2026-09-28 before any milestone number): a DETECTION key computed on
#: fewer than A2_MIN_SUPPORT ground-truth events is REPORTED with its n and never gates (one event there
#: moves the metric by more than any fixed tolerance: the step-1,500 G0's only OUT term was an AP on
#: n_pos = 1); at or above it the tolerance is max(0.02, 2/n) -- "two discrete events".
A2_MIN_SUPPORT = 30
_BANDS = r"(all|0_20|20_40|40_60)"


def detection_support(k: str, inrun: dict):
    """The number of GT events a per-class / per-band DETECTION key is computed on, read from the
    in-run row's own count keys; None for a POOLED key (it keeps the registered tolerance)."""
    import re
    s = k[len("eval_"):] if k.startswith("eval_") else k
    m = re.match(r"(agent|box3d)_det_ap[0-9p]+_(.+)_" + _BANDS + r"$", s)
    if m and m.group(2) != "all":
        return float(inrun.get(f"eval_{m.group(1)}_det_npos_{m.group(2)}_{m.group(3)}") or 0.0)
    m = re.match(r"(agent|box3d)_det_map[0-9p]+_" + _BANDS + r"$", s)
    if m:
        pre = f"eval_{m.group(1)}_det_npos_"
        suf = f"_{m.group(2)}"
        ns = [float(v) for kk, v in inrun.items() if kk.startswith(pre) and kk.endswith(suf)
              and not kk.startswith(pre + "all_") and v]
        return min(ns) if ns else 0.0
    m = re.match(r"(agent|box3d)_rec@gate_(.+)$", s)
    if m:
        return float(inrun.get(f"eval_{m.group(1)}_npos_{m.group(2)}") or 0.0)
    return None


def judge(inrun, by_seed, rec, amend: str | None = None):
    seeds = sorted(by_seed)
    K = len(seeds)
    terms = sorted(k for k in inrun if k.startswith("eval_"))
    res, reasons, cls_count = {}, [], {}
    for k in terms:
        c = term_class(k)
        x = inrun[k]
        vals = [by_seed[s]["row"].get(k, "__MISSING__") for s in seeds]
        r = {"inrun": x, "class": c}
        if c in ("EXCLUDED", "DERIVED"):
            r["verdict"] = c
            res[k] = r
            cls_count[c] = cls_count.get(c, 0) + 1
            continue
        if any(v == "__MISSING__" for v in vals):
            r["verdict"] = "MISSING"
            reasons.append(f"{k}: not produced by the reproduction")
            res[k] = r
            continue
        if _isnull(x) or any(_isnull(v) for v in vals):
            ok = _isnull(x) and all(_isnull(v) for v in vals)
            r.update(seed_values=[None if _isnull(v) else v for v in vals], cls=c + "/UNDEFINED",
                     verdict="OK" if ok else "OUT")
            if not ok:
                reasons.append(f"{k} [UNDEFINED rule] in-run {x} vs seeds {r['seed_values'][:3]}")
            res[k] = r
            cls_count["UNDEFINED"] = cls_count.get("UNDEFINED", 0) + 1
            continue
        vals = [float(v) for v in vals]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if K >= 2 else 0.0
        spread = (max(vals) - min(vals)) / max(abs(mean), 1e-12)
        if c == "SMOOTH_OR_STOCHASTIC":
            c = "STOCHASTIC" if spread > 1e-5 else "SMOOTH"
        r.update(mean=mean, sd=sd, rel_spread=spread, cls=c, seed_values=vals)
        x = float(x)
        if c == "STOCHASTIC":
            if K not in T_995:
                raise ValueError(f"no t(0.995, {K - 1}) registered; add it to T_995 (never default)")
            t = T_995[K]
            half = t * sd * math.sqrt(1 + 1 / K) + 0.01 * abs(mean)
            ok = (mean - half) <= x <= (mean + half)
            r.update(pi_lo=mean - half, pi_hi=mean + half, tol=f"99% PI t={t} +1% |mean|")
        else:
            if spread > 1e-5 and c in ("COUNT", "DETECTION", "MAP10_COUNTS", "MAP10_IOU", "MATCHED"):
                r["note"] = "seed spread > 1e-5 in a deterministic class: judged on the seed mean"
            ok, dev, tol = tol_ok(c, k, x, mean)
            sup = detection_support(k, inrun) if (amend in ("A2", "A5") and c == "DETECTION") else None
            if sup is not None:
                r["support_n"] = sup
                if sup < A2_MIN_SUPPORT:
                    r.update(dev=abs(mean - x), tol=f"REPORTED (support n={sup:g} < {A2_MIN_SUPPORT})",
                             cls="DETECTION_LOWSUPPORT", verdict="REPORTED")
                    cls_count["DETECTION_LOWSUPPORT"] = cls_count.get("DETECTION_LOWSUPPORT", 0) + 1
                    res[k] = r
                    continue
                t2 = max(0.02, 2.0 / sup)
                ok, dev, tol = abs(mean - x) <= t2, abs(mean - x), f"abs<=max(0.02, 2/n)={t2:.4f}"
            r.update(dev=dev, tol=tol)
            if c in ("SMOOTH", "MATCHED", "MAP10_COUNTS") and abs(x) > 0:
                r["rel_dev"] = abs(mean - x) / abs(x)
            if c == "DETECTION" and not k.endswith(MODEL_DEP_COUNT_SUFFIX) \
                    and not k.endswith(("_conf_ratio", "_centre_err_p50")):
                r["abs_dev01"] = abs(mean - x)
        r["verdict"] = "OK" if ok else "OUT"
        if not ok:
            reasons.append(f"{k} [{c}] in-run {x} vs seeds mean {mean:.6g} (sd {sd:.3g})")
        cls_count[c] = cls_count.get(c, 0) + 1
        res[k] = r
    med = {}
    sm = [v["rel_dev"] for v in res.values() if v.get("cls") == "SMOOTH" and "rel_dev" in v
          and abs(float(v["inrun"])) >= 0.1]
    mt = [v["rel_dev"] for v in res.values() if v.get("cls") == "MATCHED" and "rel_dev" in v]
    mc = [v["rel_dev"] for v in res.values() if v.get("cls") == "MAP10_COUNTS" and "rel_dev" in v
          and abs(float(v["inrun"])) >= 1000]
    dt = [v["abs_dev01"] for v in res.values() if v.get("cls") == "DETECTION" and "abs_dev01" in v]
    med["SMOOTH"] = statistics.median(sm) if sm else None
    med["MATCHED"] = statistics.median(mt) if mt else None
    med["MAP10_COUNTS"] = statistics.median(mc) if mc else None
    med["DETECTION_abs01"] = statistics.median(dt) if dt else None
    if med["SMOOTH"] is not None and med["SMOOTH"] > 0.002:
        reasons.append(f"SMOOTH median rel dev {med['SMOOTH']:.4f} > 0.002")
    if med["MATCHED"] is not None and med["MATCHED"] > 0.05:
        reasons.append(f"MATCHED median rel dev {med['MATCHED']:.4f} > 0.05")
    if med["MAP10_COUNTS"] is not None and med["MAP10_COUNTS"] > 0.002:
        reasons.append(f"MAP10 median rel dev {med['MAP10_COUNTS']:.4f} > 0.002")
    if med["DETECTION_abs01"] is not None and med["DETECTION_abs01"] > 0.005:
        reasons.append(f"DETECTION median abs dev {med['DETECTION_abs01']:.4f} > 0.005")
    md = rec["model"]
    if md["state_dict"]["missing"] or md["state_dict"]["unexpected"]:
        reasons.append("strict load not clean")
    if not md["param_breakdown"]["equal"]:
        reasons.append("param_breakdown differs from config.json")
    anc = md.get("anchor_file_vs_ckpt_buffers") or {}
    anc_bad = [k for k, v in anc.items() if isinstance(v, dict) and v.get("max_abs_diff") != 0.0]
    if anc_bad:
        reasons.append(f"anchor file != ckpt anchor buffers: {anc_bad}")
    if (md.get("declared_vs_built") or {}).get("mismatches"):
        reasons.append("loader G-DVB mismatches")
    if not all(v.get("buffers_unchanged") for v in by_seed.values()):
        reasons.append("model buffers changed during an eval pass")
    wc = rec.get("wrapper_control")
    if wc is None:
        reasons.append("wrapper control not run")
    elif wc.get("clause") != "PASS":
        reasons.append(f"wrapper clause {wc.get('clause')}: {wc.get('reasons')}")
    # mutation detection: >= 1 non-COUNT term outside its class tolerance vs the IN-RUN value
    det = {}
    for m, info in (rec.get("mutations") or {}).items():
        if info.get("raised"):
            det[m] = {"detected": True, "how": "raised", "n_terms_out": None}
            continue
        row = info.get("row") or {}
        outs = []
        for k, r in res.items():
            c = r.get("cls", "")
            # ⭐ a term already OUT under the unmutated reproduction cannot be MOVED outside by a
            # mutation (SPEC §2's "move ... outside"); counting it inflated M1 (corrected 2026-09-28,
            # recorded in SPEC AMENDMENT A2)
            if c in ("COUNT",) or r.get("verdict") != "OK" or c.endswith("/UNDEFINED"):
                continue
            y = row.get(k)
            x = r["inrun"]
            if _isnull(y) or _isnull(x):
                continue
            if c == "STOCHASTIC":
                bad = not (r["pi_lo"] <= float(y) <= r["pi_hi"])
            elif r.get("support_n") is not None:
                bad = abs(float(y) - float(x)) > max(0.02, 2.0 / float(r["support_n"]))
            else:
                bad = not tol_ok(c, k, float(x), float(y))[0]
            if bad:
                outs.append({"term": k, "class": c, "inrun": x, "mutated": y})
        det[m] = {"detected": bool(outs), "n_terms_out": len(outs), "first10": outs[:10]}
    m1 = det.get("m1")
    if m1 is None:
        reasons.append("M1 not run")
    blind = [m for m, d in det.items() if not d["detected"]]
    only_m1 = bool(m1) and not m1["detected"]
    if only_m1:
        reasons.append("M1 (trunk equalisation dropped) stayed inside every tolerance: the gate "
                       "has no power -> VOID")
    g0 = "PASS" if not reasons else "FAIL"
    if only_m1 and all(r.startswith("M1") for r in reasons):
        g0 = "VOID"
    return {"G0": g0, "amendment": amend or "as registered", "reasons": reasons,
            "n_terms": len(res), "by_class_counts": cls_count,
            "medians": med, "mutation_detection": det,
            "blind_spots_named": [f"{m}: not detected by G0 (a wiring defect this gate cannot see)"
                                  for m in blind if m != "m1"],
            "terms": res}


if __name__ == "__main__":
    main()
