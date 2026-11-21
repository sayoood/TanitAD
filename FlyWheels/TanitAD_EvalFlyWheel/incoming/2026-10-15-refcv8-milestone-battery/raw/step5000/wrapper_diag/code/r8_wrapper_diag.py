"""r8_wrapper_diag.py -- CPU-only discriminating experiment for the refcv8 step-5,000 G0 wrapper-clause FAIL.

Question: is `merge defect: cascade / r8_subscore` (wrapper_probe_r7.py) (a) a MERGE-RULE defect of microbatch.py,
(b) a genuine BATCH COUPLING in the refcv8 forward, or (c) neither -- the probe's arms see DIFFERENT DDIM noise per row,
because the probe zeroes only `torch.randn_like` while refcv8 draws its base eps with `torch.randn((b, _nb)+...)`
(refc.py:3036) and its extra-candidate eps / source-anchor choice from `R8Generator.randn_eval / rand_eval`
(refcv8_conditioning.py:202-208), seeded off the global RNG state.

Two noise modes, same 4 rows (= the probe's batch: first 4 windows of in-run eval batch 0, perm sha checked):
  live : exactly the probe's patch (randn_like -> zeros); every other draw live
  zero : ALSO torch.randn (no generator kwarg) -> zeros, R8Generator.randn_eval -> zeros, R8Generator.rand_eval -> 0.5
Arms per mode: plain(seed 0), repeat(seed 0), seed1 (plain, seed 1), micro13, micro31, permuted (rows reversed),
and in zero mode also each row ALONE (batch 1) and the W1 / W2 deliberate merge regressions.
Controls that must read known values: zero/seed1 == zero/plain bit-for-bit (noise fully neutralised);
live/repeat == live/plain bit-for-bit (determinism); trunk outputs (pooled, fmap_s16) per-row-identical in every arm.

CPU only (CUDA_VISIBLE_DEVICES=""), OMP_NUM_THREADS=6, P3 settings (trunk fp32, NCHW). Banks raw/diag_arms.json after
every arm. Assert on that JSON, never on the exit code.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code_r8"))
import r8_repoint  # noqa: E402

L = r8_repoint.apply()
import torch  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import g0_refcv7 as G  # noqa: E402
import wrapper_probe_r7 as W  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402
from tanitad.refs import refcv8_conditioning as r8c  # noqa: E402

CKPT = "E:/refcv8_eval_kit/ckpt/ckpt_5000.pt"
CONFIG = "E:/refcv8_eval_kit/ckpt/config.json"
PERM_SHA_G0 = "99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975"
RAW = HERE / "raw"
CACHE = HERE / "cache" / "batch4.pt"
OUT = RAW / "diag_arms.json"
N = 4
SAVE_KEYS = ("out.traj", "out.anchor_traj", "out.sel_score", "out.sel_score_v3", "out.r8_sub_logits",
             "out.anchor_logits", "out.r8_alloc", "out.r8_hyp_alloc", "out.pooled")
REC: dict = {"tool": "r8_wrapper_diag.py", "started": time.strftime("%FT%T"), "ckpt": CKPT, "config": CONFIG,
             "n_rows": N, "torch": torch.__version__, "omp": os.environ["OMP_NUM_THREADS"],
             "cuda_available": torch.cuda.is_available(), "arms": {}, "errors": {}}


def bank():
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    json.dump(REC, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    tmp.replace(OUT)


def flatten(o, path="out"):
    if torch.is_tensor(o):
        yield path, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{path}.{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flatten(v, f"{path}[{i}]")


class Capture:
    """Outermost wrapper on model.forward: records every merged `out` compute_losses_v3 receives."""

    def __init__(self, model):
        self.model, self.outs, self.prev = model, [], None

    def install(self):
        self.prev = self.model.__dict__.get("forward")
        inner = self.model.forward

        def f(*a, **k):
            o = inner(*a, **k)
            self.outs.append(o)
            return o
        self.model.forward = f
        return self

    def remove(self):
        if self.prev is None:
            del self.model.forward
        else:
            self.model.forward = self.prev


@contextmanager
def noise(mode):
    o_rl, o_rn = torch.randn_like, torch.randn
    o_re, o_ue = r8c.R8Generator.randn_eval, r8c.R8Generator.rand_eval
    torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)          # the probe's own patch (wrapper_probe_r7:141)
    if mode == "zero":
        def randn0(*size, **kk):
            if kk.get("generator") is not None:
                return o_rn(*size, **kk)                                  # seeded fixed draws (e.g. refc_v3:3122) kept
            shp = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
            return torch.zeros(tuple(shp), device=kk.get("device"), dtype=kk.get("dtype") or torch.float32)
        torch.randn = randn0
        r8c.R8Generator.randn_eval = lambda self, shape, device, site, dtype=torch.float32: torch.zeros(
            tuple(shape), device=device, dtype=dtype)
        r8c.R8Generator.rand_eval = lambda self, shape, device, site: torch.full(tuple(shape), 0.5, device=device)
    try:
        yield
    finally:
        torch.randn_like, torch.randn = o_rl, o_rn
        r8c.R8Generator.randn_eval, r8c.R8Generator.rand_eval = o_re, o_ue


def load_or_build_batch(tr, model, cfg, args, config):
    if CACHE.exists():
        d = torch.load(str(CACHE), map_location="cpu", weights_only=False)
        eb = {k: (G.LawOnly(v["__lawonly__"], v["law_idx"]) if isinstance(v, dict) and "__lawonly__" in v else v)
              for k, v in d["batch"].items()}
        REC["batch_meta"] = d["meta"]
        return eb
    t0 = time.time()
    e_ds, _eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                            dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    B = int(args.batch)
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), B)
    sha = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    law_idx = int(tr.LAW_AHEAD) - 1
    eb = G.collate(e_ds, perm[0:N], law_idx)
    meta = {"perm_sha256": sha, "perm_sha_matches_g0": sha == PERM_SHA_G0, "rows_window_index": perm[0:N],
            "batch_B": B, "law_idx": law_idx, "build_s": round(time.time() - t0, 1),
            "episode_id": [int(x) for x in eb["episode_id"]] if torch.is_tensor(eb.get("episode_id")) else None,
            "pose_sync": drec.get("pose_sync", {}).get("n_covered") if isinstance(drec.get("pose_sync"), dict) else None}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ser = {k: ({"__lawonly__": v.u8, "law_idx": v.law_idx} if isinstance(v, G.LawOnly) else v) for k, v in eb.items()}
    torch.save({"batch": ser, "meta": meta}, str(CACHE))
    REC["batch_meta"] = meta
    del e_ds
    return eb


def main():
    t_all = time.time()
    tr = L.trainer()
    config = L.load_config(CONFIG)
    model, cfg, args, mrec = L.build_model(config, CKPT, "cpu")
    sd = mrec["state_dict"]
    REC["model"] = {"step": sd.get("step"), "strict": sd.get("strict"), "missing": len(sd.get("missing") or []),
                    "unexpected": len(sd.get("unexpected") or []), "training": bool(model.training),
                    "ckpt_md5": L.md5_file(CKPT)}
    tf = G.training_flags(model)
    G.set_flags(model, tf["flags"])
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    W._set_condition(W.CONDITIONS["P3_fp32_det"], trunk)
    REC["trunk_levers"] = dict(trunk.memory_levers)
    eb4 = load_or_build_batch(tr, model, cfg, args, config)
    G.patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    REC["mode"], REC["ablate_frames"] = mode, abl
    bank()
    print(f"[diag] setup {time.time() - t_all:.0f}s; batch meta {REC['batch_meta']}", flush=True)

    refs: dict = {}

    def run(name, nmode, seed, rows=None, sizes=None, cls=MicroBatchForward):
        t0 = time.time()
        rows = list(range(N)) if rows is None else rows
        sb = G.sub_batch(eb4, len(rows), rows=rows)
        nr = len(rows)
        w = cap = None
        try:
            with noise(nmode):
                torch.manual_seed(seed)
                if sizes:
                    w = cls(model, sizes).install()
                cap = Capture(model).install()
                with torch.no_grad():
                    d = tr.compute_losses_v3(model, sb, "cpu", mode=mode, ablate_frames=abl)
        except Exception as exc:                                    # noqa: BLE001 -- recorded, run continues
            REC["errors"][name] = f"{type(exc).__name__}: {str(exc)[:400]}\n{traceback.format_exc()[-1500:]}"
            print(f"[diag] {name} RAISED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            bank()
            return None
        finally:
            if cap is not None:
                cap.remove()
            if w is not None:
                w.remove()
        scal = W._scalars(d)
        del d
        out = cap.outs[-1]
        per_row = {p: t.detach().float().cpu().clone() for p, t in flatten(out)
                   if t.dim() >= 1 and t.shape[0] == nr}
        rec = {"noise": nmode, "seed": seed, "rows": rows, "sizes": sizes, "n_forward_calls": len(cap.outs),
               "wall_s": round(time.time() - t0, 1), "scalars": scal,
               "rules_flagged": (w.flagged() if w is not None else None)}
        REC["arms"][name] = rec
        torch.save({k: v for k, v in per_row.items() if k in SAVE_KEYS or k.startswith("out.layer_u0_hat")},
                   str(RAW / "arms" / f"{name}.pt"))
        cap.outs.clear()
        print(f"[diag] {name}: {rec['wall_s']}s  cascade={scal.get('cascade')} r8_subscore={scal.get('r8_subscore')} "
              f"traj={scal.get('traj')}", flush=True)
        return per_row

    def compare(name, cur, ref, row_map):
        """row_map[i] = the ref row that cur row i must equal. Per key max|diff| and rel to max|ref|."""
        res = {}
        for k, t in cur.items():
            if k not in ref:
                continue
            r = ref[k][row_map]
            if r.shape != t.shape:
                res[k] = {"shape_mismatch": [list(t.shape), list(r.shape)]}
                continue
            fin = torch.isfinite(r) & torch.isfinite(t)
            dd = (t - r).abs()[fin]
            mx = float(dd.max()) if dd.numel() else 0.0
            scale = float(r.abs()[fin].max()) if fin.any() else 0.0
            nonfin_mismatch = int((torch.isfinite(r) != torch.isfinite(t)).sum())
            res[k] = {"max_abs": mx, "rel": mx / max(scale, 1e-12), "nonfinite_mismatch": nonfin_mismatch}
        REC["arms"][name]["vs_ref"] = res
        top = sorted(((v.get("rel", 0.0), k) for k, v in res.items() if "rel" in v), reverse=True)[:8]
        REC["arms"][name]["top_rel"] = [[k, r] for r, k in top]
        print(f"[diag]   {name} top rel: {[(k, f'{r:.2e}') for r, k in top[:5]]}", flush=True)

    ident = list(range(N))
    rev = list(range(N))[::-1]
    # ---- priority 1: zero-noise arms (the discriminating test) ---- #
    refs["zero"] = run("zero_plain", "zero", 0)
    bank()
    for nm, kw in (("zero_seed1", dict(seed=1)), ("zero_micro13", dict(seed=0, sizes=[1, 3])),
                   ("zero_micro31", dict(seed=0, sizes=[3, 1])), ("zero_perm", dict(seed=0, rows=rev))):
        cur = run(nm, "zero", kw.pop("seed"), **kw)
        if cur is not None and refs["zero"] is not None:
            compare(nm, cur, refs["zero"], rev if nm == "zero_perm" else ident)
        bank()
    # ---- priority 2: live-noise arms (the probe's own conditions, on CPU) ---- #
    refs["live"] = run("live_plain", "live", 0)
    bank()
    for nm, kw in (("live_repeat", dict(seed=0)), ("live_seed1", dict(seed=1)),
                   ("live_micro13", dict(seed=0, sizes=[1, 3])), ("live_micro31", dict(seed=0, sizes=[3, 1])),
                   ("live_perm", dict(seed=0, rows=rev))):
        cur = run(nm, "live", kw.pop("seed"), **kw)
        if cur is not None and refs["live"] is not None:
            compare(nm, cur, refs["live"], rev if nm == "live_perm" else ident)
        bank()
    # ---- priority 3: each row ALONE, zero noise ---- #
    for i in range(N):
        cur = run(f"zero_row{i}", "zero", 0, rows=[i])
        if cur is not None and refs["zero"] is not None:
            compare(f"zero_row{i}", cur, refs["zero"], [i])
        bank()
    # ---- priority 4: the deliberate merge regressions under zero noise (power) ---- #
    for nm, cls in (("zero_W1", W.W1MergeFirstOnly), ("zero_W2", W.W2ReversedCat)):
        cur = run(nm, "zero", 0, sizes=[1, 3], cls=cls)
        if cur is not None and refs["zero"] is not None:
            compare(nm, cur, refs["zero"], ident)
        bank()
    REC["finished"] = time.strftime("%FT%T")
    REC["wall_s"] = round(time.time() - t_all, 1)
    bank()
    print(f"[diag] DONE {REC['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
