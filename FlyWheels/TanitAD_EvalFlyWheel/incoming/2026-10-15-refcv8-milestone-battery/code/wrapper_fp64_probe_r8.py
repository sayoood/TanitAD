"""wrapper_fp64_probe_r8.py -- STEP-GENERIC copy (2026-10-18) of raw/step20000/wrapper_fp64/wrapper_fp64_probe.py, the W-R8-2
re-check (PI 2026-10-17).  Step / checkpoint / output dir come from R8_FP64_STEP (default 20000), R8_FP64_CKPT (default
E:/refcv8_eval_kit/ckpt/ckpt_<step>.pt) and R8_FP64_OUTDIR (default <pkg>/raw/step<step>/wrapper_fp64); nothing else changed
except that the lat_tac GPU row is optional (at another step a different term may be the one over the bar; the W-R8-2 verdict
reads F64_n_terms_wrapper_gt_1e-9 over ALL terms + W1/W2).  Original docstring:

wrapper_fp64_probe.py -- is the refcv8 step-20,000 G0 wrapper-clause miss (`lat_tac` 1.0647e-5 > 1e-5) fp32
reduction-order noise or a real micro-batch coupling?  CPU only.

    python -B wrapper_fp64_probe.py --arm F32|F64 --out raw/arm_F32.json
    python -B wrapper_fp64_probe.py --combine --out result.json         (reads raw/arm_F32.json + raw/arm_F64.json)

Both arms run the UNMODIFIED `wrapper_probe_r8fix.probe` (P3, micro [1,3]/[3,1], the cached 4 rows of the G0 in-run batch).
F64 differs ONLY by: model.double(), torch.set_default_dtype(float64), every floating batch tensor and the ingested frames
cast to float64.  Nothing in the package `code/` or the refcv7 battery code is edited; the probe is wrapped from here.
Extras (secondary): per-arm capture of `lat_logits_tac` and of every row-batched output tensor (diff vs plain), each row
ALONE vs in the batch, an independent float64 CE from the captured logits, and a TorchFunctionMode dtype audit (F64 plain arm).
Assert on the json, never on the exit code.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import ctypes
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # NOT "": on Windows an empty value DELETES the variable (MEASURED here: is_available() stayed True)
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PKG = HERE.parent                           # .../2026-10-15-refcv8-milestone-battery (this copy lives in code/)
STEP = int(os.environ.get("R8_FP64_STEP", "20000"))
CKPT = os.environ.get("R8_FP64_CKPT", f"E:/refcv8_eval_kit/ckpt/ckpt_{STEP}.pt")
OUT_DIR = Path(os.environ.get("R8_FP64_OUTDIR", str(PKG / f"raw/step{STEP}/wrapper_fp64")))
CONFIG = "E:/refcv8_eval_kit/ckpt/config.json"
BATCH_CACHE = "C:/Users/Admin/qland/work/r8_wrapper_diag/cache/batch4.pt"
PERM_SHA_G0 = "99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975"
MIN_FREE_COMMIT_GB = 4.0
ARM_ORDER = ["plain", "repeat", "permuted", "micro13", "micro31", "W1", "W2"]
N = 4


def free_commit_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2**30


def watchdog(out_path: Path):
    def run():
        while True:
            f = free_commit_gb()
            if f < MIN_FREE_COMMIT_GB:
                out_path.with_suffix(".ABORTED_LOW_COMMIT.txt").write_text(f"free commit {f:.2f} GB < {MIN_FREE_COMMIT_GB}\n")
                os._exit(3)
            time.sleep(8)
    threading.Thread(target=run, daemon=True).start()


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), 1e-12)


def main_arm(arm: str, out_p: Path, rows_only: bool = False) -> int:
    t0 = time.time()
    rec: dict = {"tool": "wrapper_fp64_probe.py", "arm": arm, "started": time.strftime("%FT%T"), "ckpt": CKPT, "ok": False,
                 "omp_num_threads": os.environ["OMP_NUM_THREADS"], "free_commit_gb_at_start": round(free_commit_gb(), 2)}
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        rec["wall_s"] = round(time.time() - t0, 1)
        tmp = out_p.with_suffix(".part")
        tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, out_p)

    if rec["free_commit_gb_at_start"] < 8.0:
        rec["error"] = f"free commit {rec['free_commit_gb_at_start']} GB < 8 GB at start"
        bank()
        return 2
    watchdog(out_p)
    try:
        sys.path.insert(0, str(PKG / "code"))
        import r8_repoint
        L = r8_repoint.apply()
        import torch
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        import g0_refcv7 as G
        import wrapper_probe_r7 as W
        import wrapper_probe_r8fix as FIX
        torch.cuda.synchronize = lambda *a, **k: None
        rec.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(), tanitad_file=__import__("tanitad").__file__,
                   threads=torch.get_num_threads())
        assert not torch.cuda.is_available(), "CUDA must be hidden for this CPU-only experiment"
        tr = L.trainer()
        config = L.load_config(CONFIG)
        model, cfg, args, mrec = L.build_model(config, CKPT, "cpu")
        sd = mrec["state_dict"]
        rec["model"] = {"step": sd.get("step"), "missing": len(sd.get("missing") or []), "unexpected": len(sd.get("unexpected") or [])}
        assert sd.get("step") == STEP, (sd.get("step"), STEP)
        G.set_flags(model, G.training_flags(model)["flags"])
        from tanitad.models.timm_trunk import TimmResNetTrunk
        trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
        d = torch.load(BATCH_CACHE, map_location="cpu", weights_only=False)
        assert d["meta"]["perm_sha256"] == PERM_SHA_G0 and d["meta"]["rows_window_index"] == [20658, 368, 10637, 7314], d["meta"]
        eb = {k: (G.LawOnly(v["__lawonly__"], v["law_idx"]) if isinstance(v, dict) and "__lawonly__" in v else v)
              for k, v in d["batch"].items()}
        rec["batch"] = {"path": BATCH_CACHE, "perm_sha256": d["meta"]["perm_sha256"], "rows_window_index": d["meta"]["rows_window_index"]}
        G.patch_frames_to_device(tr)
        mode = getattr(args, "mode", "diffusion")
        abl = bool(getattr(args, "ablate_frames", False))

        f64 = arm == "F64"
        FLOAT_NOOP: collections.Counter = collections.Counter()
        TO_REDIRECT: collections.Counter = collections.Counter()
        if f64:
            model.double()
            torch.set_default_dtype(torch.float64)
            for k, v in list(eb.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    eb[k] = v.double()
            f2d_prev = tr.frames_to_device

            def f2d64(x, device):
                r = f2d_prev(x, device)
                if hasattr(r, "x") and torch.is_tensor(r.x):                    # LawOnlyDevice
                    r.x = r.x.double()
                    return r
                return r.double() if torch.is_tensor(r) and r.is_floating_point() else r
            tr.frames_to_device = f2d64
            # The model has explicit `.float()` downcasts (e.g. timm_trunk._s8_backbone ends `out.float().contiguous()`), which
            # would make a float64 tensor float32 and crash `F.conv2d` against float64 weights (MEASURED, first F64 attempt).
            # In the F64 arm `.float()` on an already-float64 tensor is a NO-OP (counted per call site and reported);
            # on a float32 tensor it is an exact upcast to float64, on a non-floating tensor it is unchanged, so the uint8 frames still go through the original float32 ingest and the F64
            # arm sees exactly the SAME input values as F32 (then upcast).
            _orig_float = torch.Tensor.float

            def _float(self, *a, **k):
                if self.dtype != torch.uint8:                # float64 stays; float32 / bool / int are cast to float64 (exact); uint8 frames keep the
                    fr = sys._getframe(1)                    # original float32 ingest, then f2d64 upcasts them
                    FLOAT_NOOP[f"{fr.f_code.co_filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{fr.f_lineno}:{fr.f_code.co_name}"] += 1
                    return self if self.dtype == torch.float64 else _orig_to(self, torch.float64)
                return _orig_float(self, *a, **k)
            torch.Tensor.float = _float
            # Same for an explicit `.to(torch.float32)` (e.g. refc.py:2809 `self.time_mlp(t.reshape(-1).to(torch.float32))`, MEASURED to
            # crash F.linear on the second F64 attempt): in the F64 arm a cast to exactly torch.float32 becomes a cast to float64.
            _orig_to = torch.Tensor.to

            def _to(self, *a, **k):
                hit = any(x is torch.float32 for x in a) or any(v is torch.float32 for v in k.values())
                if hit:
                    fr = sys._getframe(1)
                    TO_REDIRECT[f"{fr.f_code.co_filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{fr.f_lineno}:{fr.f_code.co_name}"] += 1
                    a = tuple(torch.float64 if x is torch.float32 else x for x in a)
                    k = {kk: (torch.float64 if v is torch.float32 else v) for kk, v in k.items()}
                return _orig_to(self, *a, **k)
            torch.Tensor.to = _to
            # And factory calls with an explicit dtype=torch.float32 (e.g. map_head_hires.py:648 `torch.arange(.., dtype=torch.float32)`,
            # MEASURED to crash F.grid_sample on the third/fourth F64 attempt): redirected to float64 in this arm.
            def _wrap_factory(name):
                f0 = getattr(torch, name)

                def g(*a, **k):
                    if k.get("dtype") is torch.float32 or any(x is torch.float32 for x in a):
                        fr = sys._getframe(1)
                        TO_REDIRECT[f"{name}@{fr.f_code.co_filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{fr.f_lineno}:{fr.f_code.co_name}"] += 1
                        k = {kk: (torch.float64 if v is torch.float32 else v) for kk, v in k.items()}
                        a = tuple(torch.float64 if x is torch.float32 else x for x in a)
                    return f0(*a, **k)
                setattr(torch, name, g)
            for _nm in ("arange", "zeros", "ones", "full", "tensor", "as_tensor", "empty", "linspace", "eye", "zeros_like", "ones_like",
                        "full_like", "empty_like", "rand", "randn"):
                _wrap_factory(_nm)
        rec["param_dtypes"] = dict(collections.Counter(str(p.dtype) for p in model.parameters()))
        rec["buffer_dtypes"] = dict(collections.Counter(str(b.dtype) for b in model.buffers()))
        rec["default_dtype"] = str(torch.get_default_dtype())
        bank()

        # ---- capture machinery: wraps tr.compute_losses_v3 so each probe arm's MERGED model output is recorded ---- #
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
            def __init__(self, m):
                self.m, self.outs, self.prev = m, [], None

            def install(self):
                self.prev = self.m.__dict__.get("forward")
                inner = self.m.forward

                def f(*a, **k):
                    o = inner(*a, **k)
                    self.outs.append(o)
                    return o
                self.m.forward = f
                return self

            def remove(self):
                if self.prev is None:
                    del self.m.forward
                else:
                    self.m.forward = self.prev

        from torch.overrides import TorchFunctionMode

        class Audit(TorchFunctionMode):
            """Logs ops that return a non-float64 floating tensor while the run is meant to be float64."""

            def __init__(self):
                super().__init__()
                self.counts = collections.Counter()
                self.n_ops = 0

            def __torch_function__(self, func, types, a=(), kw=None):
                kw = kw or {}
                out = func(*a, **kw)
                self.n_ops += 1
                if torch.is_tensor(out) and out.is_floating_point() and out.dtype != torch.float64:
                    flat = list(a) + list(kw.values())
                    ins = [t for x in flat for t in (x if isinstance(x, (list, tuple)) else [x]) if torch.is_tensor(t) and t.is_floating_point()]
                    kind = "DOWNCAST(f64->%s)" % str(out.dtype).replace("torch.", "") if any(t.dtype == torch.float64 for t in ins) \
                        else "ISLAND(%s)" % str(out.dtype).replace("torch.", "")
                    fr, site = sys._getframe(2), "?"
                    while fr is not None:
                        fn = fr.f_code.co_filename.replace("\\", "/")
                        if "/tanitad/" in fn or "/scripts/" in fn or "refc" in fn.rsplit("/", 1)[-1]:
                            site = f"{fn.rsplit('/', 1)[-1]}:{fr.f_lineno}:{fr.f_code.co_name}"
                            break
                        fr = fr.f_back
                    self.counts[(kind, getattr(func, "__name__", str(func)), site)] += 1
                return out

        calls: list = []
        plain_ref: dict = {}
        orig_cl = tr.compute_losses_v3
        lat_label = {}

        def wrapped(m, sb, device, **kw):
            idx = len(calls)
            name = ARM_ORDER[idx] if idx < len(ARM_ORDER) else f"extra{idx}"
            cap = Capture(m).install()
            aud = Audit() if (f64 and name == "plain") else None
            t1 = time.time()
            try:
                with (aud if aud is not None else contextlib.nullcontext()):
                    dd = orig_cl(m, sb, device, **kw)
            finally:
                cap.remove()
            out = cap.outs[-1]
            ent = {"arm": name, "wall_s": round(time.time() - t1, 1),
                   "term_dtypes": dict(collections.Counter(str(v.dtype) for v in dd.values() if torch.is_tensor(v) and v.ndim == 0)),
                   "lat_tac_dtype": str(dd["lat_tac"].dtype) if torch.is_tensor(dd.get("lat_tac")) else None,
                   "lat_tac_value": float(dd["lat_tac"]) if "lat_tac" in dd else None,
                   "lat_logits_tac_dtype": str(out["lat_logits_tac"].dtype) if "lat_logits_tac" in out else None}
            if "lat_v7" in sb:
                lat_label["lat_v7"] = sb["lat_v7"].tolist()
            ent["lat_label_rows"] = sb["lat_v7"].tolist() if "lat_v7" in sb else None
            rows_n = N
            ent["rows_in_call"] = rows_n
            if "lat_logits_tac" in out:
                ent["lat_logits_tac"] = out["lat_logits_tac"].detach().double().cpu().tolist()
            per = {p: t.detach().clone() for p, t in flatten(out) if t.dim() >= 1 and t.shape[0] == (rows_n or -1) and t.is_floating_point()}
            if name == "plain":
                plain_ref.update(per)
            elif name in ("repeat", "permuted", "micro13", "micro31") and plain_ref:
                rmap = list(range(N))[::-1] if name == "permuted" else list(range(N))
                rows = {}
                for p, t in per.items():
                    if p not in plain_ref or plain_ref[p].shape != t.shape:
                        continue
                    r = plain_ref[p][rmap].double()
                    tt = t.double()
                    fin = torch.isfinite(r) & torch.isfinite(tt)
                    dd_ = (tt - r).abs()
                    mx = float(dd_[fin].max()) if fin.any() else 0.0
                    sc = float(r.abs()[fin].max()) if fin.any() else 0.0
                    rows[p] = (mx / max(sc, 1e-300), mx, sc)
                top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:10]
                ent["out_tensor_diff_vs_plain"] = {"n_keys": len(rows), "top10_by_rel": [[k, v[0], v[1], v[2]] for k, v in top],
                                                    "n_keys_exactly_equal": sum(1 for v in rows.values() if v[1] == 0.0)}
                if "out.lat_logits_tac" in per:
                    dl = (per["out.lat_logits_tac"].double() - plain_ref["out.lat_logits_tac"][rmap].double()).abs()
                    ent["lat_logits_tac_absdiff_per_row_max"] = dl.reshape(N, -1).max(dim=1).values.tolist()
            if aud is not None:
                ent["audit_n_ops"] = aud.n_ops
                ent["audit_nonfloat64_events"] = [[k[0], k[1], k[2], c] for k, c in sorted(aud.counts.items(), key=lambda kv: -kv[1])[:60]]
                ent["audit_n_distinct_sites"] = len(aud.counts)
                ent["audit_total_events"] = int(sum(aud.counts.values()))
            calls.append(ent)
            return dd

        tr.compute_losses_v3 = wrapped
        if not rows_only:
            # ---- the registered measurement: the fixed probe, unmodified, P3 only ---- #
            W.CONDITIONS = {"P3_fp32_det": W.CONDITIONS["P3_fp32_det"]}
            t1 = time.time()
            r = FIX.probe(tr, model, eb, "cpu", mode, abl, trunk, sub_batch_fn=G.sub_batch)
            cond = r["conditions"]["P3_fp32_det"]
            if "analysis" not in cond:
                raise RuntimeError(f"P3 produced no analysis: {cond.get('status')}")
            an = cond["analysis"]
            rec["probe_wall_s"] = round(time.time() - t1, 1)
            rec["clause_unamended"] = r.get("clause")
            rec["raw_scalars"] = cond["raw"]
            rec["analysis"] = {k: v for k, v in an.items() if k != "terms"}
            rec["terms"] = {k: {x: v[x] for x in ("plain", "floor", "wrapper", "bar", "ok")} for k, v in an["terms"].items()}
            rec["calls"] = calls
            rec["float_noop_sites"] = [[k, c] for k, c in FLOAT_NOOP.most_common(40)]
            rec["float_noop_total"] = int(sum(FLOAT_NOOP.values()))
            rec["to_f32_redirected_sites"] = [[k, c] for k, c in TO_REDIRECT.most_common(40)]
            rec["to_f32_redirected_total"] = int(sum(TO_REDIRECT.values()))
            raw = cond["raw"]
            lt = {a: raw[a].get("lat_tac") for a in ARM_ORDER if a in raw}
            rec["lat_tac"] = {"values": lt, "wrapper_rel_probe": an["terms"].get("lat_tac", {}).get("wrapper"),
                              "abs_diff_micro13": abs(lt["plain"] - lt["micro13"]), "abs_diff_micro31": abs(lt["plain"] - lt["micro31"]),
                              "rel_micro13": rel(lt["plain"], lt["micro13"]), "rel_micro31": rel(lt["plain"], lt["micro31"]),
                              "rel_repeat": rel(lt["plain"], lt["repeat"]), "rel_permuted": rel(lt["plain"], lt["permuted"])}
            # all terms, ranked, and the F64 1e-9 question
            ranked = sorted(an["terms"].items(), key=lambda kv: -kv[1]["wrapper"])
            rec["top12_by_wrapper"] = [[k, v["wrapper"], v["floor"], v["plain"]] for k, v in ranked[:12]]
            rec["n_terms"] = an["n_terms"]
            rec["n_terms_wrapper_gt_1e-9"] = sum(1 for _, v in ranked if v["wrapper"] > 1e-9)
            rec["terms_wrapper_gt_1e-9"] = [[k, v["wrapper"]] for k, v in ranked if v["wrapper"] > 1e-9][:60]
            bank()

            # ---- independent CE from the captured plain logits (float64), against the arm's own scalar ---- #
            plain_call = calls[0]
            if plain_call.get("lat_logits_tac") and plain_call.get("lat_label_rows"):
                lg = torch.tensor(plain_call["lat_logits_tac"], dtype=torch.float64)
                lb = torch.tensor(plain_call["lat_label_rows"])
                ce_all = torch.nn.functional.cross_entropy(lg, lb, ignore_index=-100, reduction="none")
                valid = lb != -100
                ce_mean = float(ce_all[valid].mean()) if bool(valid.any()) else float("nan")
                p = torch.softmax(lg, dim=-1)
                rec["lat_tac_crosscheck"] = {"labels": lb.tolist(), "n_valid_rows": int(valid.sum()), "ce_per_row_f64": ce_all.tolist(),
                                             "ce_mean_f64_from_captured_logits": ce_mean, "scalar_plain": lt["plain"],
                                             "abs_diff_scalar_vs_independent": abs(lt["plain"] - ce_mean),
                                             "p_true_per_row": [float(p[i, int(lb[i])]) if int(lb[i]) >= 0 else None for i in range(N)],
                                             "max_abs_logit_per_row": lg.abs().max(dim=1).values.tolist()}
        else:
            # rows-only: the P3 condition (fp32 trunk, NCHW, deterministic) set explicitly, then ONE plain 4-row forward for the reference
            # (the probe restores the as-built bf16/NHWC trunk levers when it returns, so an after-the-probe row-alone arm is NOT P3).
            W._set_condition(W.CONDITIONS["P3_fp32_det"], trunk)
            tr.compute_losses_v3 = wrapped
            with FIX.neutralise_eval_draws():
                torch.manual_seed(0)
                with torch.no_grad():
                    wrapped(model, G.sub_batch(eb, N), "cpu", mode=mode, ablate_frames=abl)
            rec["calls"] = calls
        plain_call = calls[0]
        rec["rows_alone_condition"] = ("P3_fp32_det (fp32 trunk, NCHW, deterministic)" if rows_only else
                                       "AS-BUILT trunk levers (bf16, NHWC) after the probe restored them: NOT P3, INVALID as a coupling test")
        rec["trunk_levers_at_rows_alone"] = dict(trunk.memory_levers)
        # ---- secondary: each row ALONE vs in the batch (logit level), neutralised draws ---- #
        solo = {}
        tr.compute_losses_v3 = orig_cl
        for i in range(N):
            sb = G.sub_batch(eb, 1, rows=[i])
            cap = None
            try:
                with FIX.neutralise_eval_draws():
                    torch.manual_seed(0)
                    cap = Capture(model).install()
                    with torch.no_grad():
                        dd = orig_cl(model, sb, "cpu", mode=mode, ablate_frames=abl)
                out = cap.outs[-1]
            finally:
                if cap is not None:
                    cap.remove()
            row = {"lat_tac_single_row_scalar": float(dd["lat_tac"])}
            if "lat_logits_tac" in out:
                a = out["lat_logits_tac"].detach().double()[0]
                b = torch.tensor(plain_call["lat_logits_tac"], dtype=torch.float64)[i]
                row["lat_logits_tac_max_absdiff_vs_batch"] = float((a - b).abs().max())
                row["lat_logits_tac_rel_vs_batch"] = float((a - b).abs().max() / b.abs().max())
            per1 = {p: t.detach().double() for p, t in flatten(out) if t.dim() >= 1 and t.shape[0] == 1 and t.is_floating_point()}
            worst = []
            for p, t in per1.items():
                if p in plain_ref and plain_ref[p].shape[1:] == t.shape[1:]:
                    r_ = plain_ref[p][i:i + 1].double()
                    fin = torch.isfinite(r_) & torch.isfinite(t)
                    mx = float((t - r_).abs()[fin].max()) if fin.any() else 0.0
                    sc = float(r_.abs()[fin].max()) if fin.any() else 0.0
                    worst.append((mx / max(sc, 1e-300), p, mx))
            worst.sort(reverse=True)
            row["out_tensors_n"] = len(worst)
            row["out_tensors_top5_rel_vs_batch"] = [[p, rr, mx] for rr, p, mx in worst[:5]]
            row["out_tensors_max_rel_vs_batch"] = worst[0][0] if worst else None
            solo[f"row{i}"] = row
            del dd, out, per1
        rec["rows_alone"] = solo
        rec["ok"] = True
    except Exception as exc:                                    # noqa: BLE001 -- the json is the result
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        rec["traceback_tail"] = traceback.format_exc()[-2500:]
    rec["free_commit_gb_at_end"] = round(free_commit_gb(), 2)
    bank()
    print(f"fp64probe[{arm}]:{'ok' if rec['ok'] else 'FAIL ' + str(rec.get('error'))}", flush=True)
    return 0 if rec["ok"] else 1


def combine(out_p: Path) -> int:
    raw = OUT_DIR / "raw"
    a32 = json.load(open(raw / "arm_F32.json", encoding="utf-8"))
    a64 = json.load(open(raw / "arm_F64.json", encoding="utf-8"))
    assert a32.get("ok") and a64.get("ok"), (a32.get("error"), a64.get("error"))
    gpu = json.load(open(PKG / f"raw/step{STEP}/wrapper_clause_r8fix_gpu.json", encoding="utf-8"))
    lat32, lat64 = a32["lat_tac"], a64["lat_tac"]
    r32 = max(lat32["rel_micro13"], lat32["rel_micro31"])
    r64 = max(lat64["rel_micro13"], lat64["rel_micro31"])
    gpu_lat = next((x for x in gpu["amended"]["top8_by_wrapper"] if x["term"] == "lat_tac"), {"wrapper": None, "bar": None})
    # does the lat_tac path itself run in float64 in the F64 arm?
    lat64_path_f64 = (a64["calls"][0]["lat_tac_dtype"] == "torch.float64" and a64["calls"][0]["lat_logits_tac_dtype"] == "torch.float64")
    f32_same_order = 1e-6 <= r32 <= 1e-4
    r64_small = r64 <= 1e-9
    ratio = (r32 / r64) if r64 > 0 else float("inf")
    if not lat64_path_f64:
        reading = "INCONCLUSIVE (lat_tac path not float64 in the F64 arm)"
    elif (r64_small or ratio >= 1000.0) and f32_same_order:
        reading = "R-NUMERICS"
    elif r64 >= 1e-7:
        reading = "R-COUPLING"
    else:
        reading = "INCONCLUSIVE"
    res = {"tool": "wrapper_fp64_probe.py --combine", "reading": reading, "ckpt": CKPT, "step": STEP,
           "gpu_lat_tac_wrapper_rel": gpu_lat["wrapper"], "gpu_bar": gpu_lat["bar"],
           "F32_cpu": {"lat_tac_wrapper_rel": r32, **lat32}, "F64_cpu": {"lat_tac_wrapper_rel": r64, **lat64},
           "ratio_F32_over_F64": ratio, "criteria": {"F64_rel_le_1e-9": r64_small, "F64_ge_1000x_below_F32": ratio >= 1000.0,
                                                    "F32_rel_in_[1e-6,1e-4]": f32_same_order, "F64_rel_ge_1e-7": r64 >= 1e-7,
                                                    "lat_tac_path_is_float64_in_F64_arm": lat64_path_f64},
           "F64_n_terms": a64["n_terms"], "F64_n_terms_wrapper_gt_1e-9": a64["n_terms_wrapper_gt_1e-9"],
           "F64_terms_wrapper_gt_1e-9": a64["terms_wrapper_gt_1e-9"], "F64_top12": a64["top12_by_wrapper"],
           "F32_n_terms_wrapper_gt_1e-5": sum(1 for v in a32["terms"].values() if v["wrapper"] > 1e-5),
           "F32_top12": a32["top12_by_wrapper"], "F32_max_wrapper_rel": a32["analysis"]["max_wrapper_rel"],
           "F64_max_wrapper_rel": a64["analysis"]["max_wrapper_rel"],
           "F64_W1_W2_detected": [a64["analysis"].get("W1_detected"), a64["analysis"].get("W2_detected")],
           "F32_W1_W2_detected": [a32["analysis"].get("W1_detected"), a32["analysis"].get("W2_detected")],
           "F64_param_dtypes": a64["param_dtypes"], "F64_buffer_dtypes": a64["buffer_dtypes"]}
    # ---- extras: same quantity in ulps, F32 vs F64 plain, logit-level diffs, rows alone under P3 ---- #
    def ulp32(x):
        return 2.0 ** (math.floor(math.log2(abs(x))) - 23)
    v32, v64 = lat32["values"], lat64["values"]
    res["lat_tac_ulps"] = {
        "F32_micro31_minus_plain_abs": abs(v32["micro31"] - v32["plain"]), "F32_ulp_of_value": ulp32(v32["plain"]),
        "F32_diff_in_ulps": abs(v32["micro31"] - v32["plain"]) / ulp32(v32["plain"]),
        "F64_micro_minus_plain_abs": abs(v64["micro13"] - v64["plain"]), "F64_ulp_of_value": math.ulp(v64["plain"]),
        "F64_diff_in_ulps": abs(v64["micro13"] - v64["plain"]) / math.ulp(v64["plain"]),
        "eps_ratio_2^29": 2.0 ** 29}
    res["lat_tac_plain_F32_vs_F64"] = {"F32": v32["plain"], "F64": v64["plain"], "abs": abs(v32["plain"] - v64["plain"]),
                                       "rel_to_F64": abs(v32["plain"] - v64["plain"]) / v64["plain"],
                                       "note": "same input values (fp32-ingested frames upcast); F64 constants given as Python decimals are float64-exact"}
    l32 = a32["calls"][0]["lat_logits_tac"]
    l64 = a64["calls"][0]["lat_logits_tac"]
    res["plain_lat_logits_F32_vs_F64_max_abs"] = max(abs(x - y) for r1, r2 in zip(l32, l64) for x, y in zip(r1, r2))
    res["lat_logits_tac_absdiff_per_row_max"] = {arm: {c["arm"]: c.get("lat_logits_tac_absdiff_per_row_max") for c in aj["calls"]
                                                       if c.get("lat_logits_tac_absdiff_per_row_max") is not None}
                                                 for arm, aj in (("F32", a32), ("F64", a64))}
    res["lat_tac_crosscheck"] = {"F32": a32.get("lat_tac_crosscheck"), "F64": a64.get("lat_tac_crosscheck")}
    res["out_tensor_diffs_micro_vs_plain"] = {arm: {c["arm"]: {"n_keys": c["out_tensor_diff_vs_plain"]["n_keys"],
                                                              "top5_by_rel": c["out_tensor_diff_vs_plain"]["top10_by_rel"][:5]}
                                                   for c in aj["calls"] if c["arm"] in ("micro13", "micro31")}
                                             for arm, aj in (("F32", a32), ("F64", a64))}
    res["audit_F64_plain"] = {k: a64["calls"][0].get(k) for k in ("audit_n_ops", "audit_total_events", "audit_n_distinct_sites",
                                                                 "audit_nonfloat64_events")}
    res["F64_patches"] = {"float_noop_total": a64["float_noop_total"], "float_noop_top": a64["float_noop_sites"][:10],
                          "to_f32_redirected_total": a64["to_f32_redirected_total"], "to_f32_redirected_top": a64["to_f32_redirected_sites"][:12]}
    for arm in ("F32", "F64"):
        rp = raw / f"arm_{arm}_rows_P3.json"
        if rp.exists():
            rj = json.load(open(rp, encoding="utf-8"))
            res.setdefault("rows_alone_P3", {})[arm] = {"ok": rj.get("ok"), "error": rj.get("error"), "condition": rj.get("rows_alone_condition"),
                                                        "levers": rj.get("trunk_levers_at_rows_alone"), "rows": rj.get("rows_alone")}
    out_p.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("reading", "criteria", "gpu_lat_tac_wrapper_rel", "ratio_F32_over_F64", "lat_tac_ulps")}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F32", "F64"])
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--rows-only", action="store_true", help="only the row-alone arms, under P3 (writes its own json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(combine(Path(a.out)) if a.combine else main_arm(a.arm, Path(a.out), a.rows_only))
