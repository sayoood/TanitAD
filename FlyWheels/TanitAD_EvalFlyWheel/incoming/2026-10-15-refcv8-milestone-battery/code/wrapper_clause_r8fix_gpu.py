"""wrapper_clause_r8fix_gpu.py -- the GPU re-run of the G0 WRAPPER CLAUSE with the amendment-W-R8-1 instrument (EvalFlyWheel 2026-10-16).

    python wrapper_clause_r8fix_gpu.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_5000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --out <raw/step5000/wrapper_clause_r8fix_gpu.json> [--g0-json <g0.json for the reproduction control>] [--device cuda]

Same rows as G0 (collate of the first in-run batch, the probe takes its first 4 windows), same P3 settings (fp32 trunk + NCHW, TF32 off,
deterministic), same micro sizes [1,3].  THREE reads, so the verdict is not only a number from a new instrument:
  1. `control_unfixed`  -- the UNFIXED `wrapper_probe_r7.probe` on this harness.  It must REPRODUCE the banked G0 terms
                           (`cascade` 1.235e-01, `r8_subscore` 1.642e-02 on the step-5,000 G0): if it does not, this harness is not G0's
                           and the fixed number below is not comparable (reported, never silently accepted).
  2. `fixed_as_returned` -- `wrapper_probe_r8fix.probe` (all four draw sites neutralised).
  3. `amended`           -- (2) with `r8_spd_crc` removed from the clause's terms (W-R8-1 item 2: a CRC of the batch, changes under any
                           reordering by design).  THIS is the registered reading: top-level `CLAUSE`.
CPU smoke: `--device cpu --batch-cache <diag batch4.pt>` (the CPU diagnosis cache) runs the same code path without the GPU.
Assert on the json (`CLAUSE`, `ok`), never on the exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DROP = ("r8_spd_crc",)
KEYS = ("cascade", "r8_subscore", "traj", "r8_alloc_l1", "sel_v3", "loss")


def clause_of(an: dict) -> str:
    """Exactly the last lines of `wrapper_probe_r7.probe` (PASS / VOID / FAIL)."""
    w12 = an["W1_detected"] and an["W2_detected"]
    return "PASS" if (not an["terms_over_bar"] and w12) else "VOID" if not w12 else "FAIL"


def amend(cond_rec: dict, drop=DROP) -> dict:
    """Re-run `wrapper_probe_r7._analyse` over the probe's RAW arm dicts with `drop` removed from every arm (pure given the raw)."""
    import wrapper_probe_r7 as W
    raw = {arm: {k: v for k, v in d.items() if k not in drop} for arm, d in cond_rec["raw"].items()}
    an = W._analyse(raw, include_mutations=True)
    an["clause"] = clause_of(an)
    an["dropped_terms"] = list(drop)
    return an


def summarise(an: dict) -> dict:
    t = an["terms"]
    top = sorted(t.items(), key=lambda kv: -kv[1]["wrapper"])[:8]
    return {"clause": an.get("clause"), "n_terms": an["n_terms"], "max_wrapper_rel": an["max_wrapper_rel"],
            "max_floor_rel": an["max_floor_rel"], "terms_over_bar": an["terms_over_bar"],
            "W1_detected": an["W1_detected"], "W2_detected": an["W2_detected"], "n_W2_detected_terms": len(an.get("W2_detected_terms") or []),
            "W1_detected_terms": an.get("W1_detected_terms"),
            "key_terms": {k: {x: t[k][x] for x in ("plain", "floor", "wrapper", "bar", "ok")} for k in KEYS if k in t},
            "top8_by_wrapper": [{"term": k, "wrapper": v["wrapper"], "floor": v["floor"], "bar": v["bar"], "ok": v["ok"]} for k, v in top]}


def reproduction(control_an: dict, g0_json: str | None) -> dict:
    if not g0_json or not os.path.exists(g0_json):
        return {"checked": False, "why": "no --g0-json"}
    g = json.load(open(g0_json, encoding="utf-8"))
    ref = (((g.get("wrapper_control") or {}).get("conditions") or {}).get("P3_fp32_det") or {}).get("analysis", {}).get("terms")
    if not ref:
        return {"checked": False, "why": "g0.json has no wrapper_control P3 terms"}
    out, worst = {}, 0.0
    for k in ("cascade", "r8_subscore", "traj", "loss"):
        if k in ref and k in control_an["terms"]:
            a, b = ref[k]["wrapper"], control_an["terms"][k]["wrapper"]
            rel = abs(a - b) / max(abs(a), 1e-12) if a else abs(b)
            out[k] = {"g0_wrapper": a, "now_wrapper": b, "g0_floor": ref[k]["floor"], "now_floor": control_an["terms"][k]["floor"], "rel_diff": rel}
            worst = max(worst, rel)
    return {"checked": True, "terms": out, "max_rel_diff": worst,
            "reproduces_g0": bool(out) and worst < 1e-3,
            "rule": "unfixed probe on this harness vs the banked G0 P3 terms, rel diff < 1e-3 on cascade/r8_subscore/traj/loss"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g0-json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-cache", default=None, help="CPU smoke only: a cached 4-row batch (the diag's batch4.pt)")
    ap.add_argument("--no-control", action="store_true", help="skip the unfixed-probe reproduction control")
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    rec: dict = {"tool": "wrapper_clause_r8fix_gpu.py", "started": time.strftime("%FT%T"), "ckpt": a.ckpt, "device": a.device,
                 "amendment": "W-R8-1", "ok": False, "CLAUSE": None, "probe_conditions": ["P3_fp32_det"]}
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        rec["wall_s"] = round(time.time() - t0, 1)
        tmp = out_p.with_suffix(".part")
        tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, out_p)

    try:
        import r8_repoint
        L = r8_repoint.apply()
        import torch
        import g0_refcv7 as G
        import wrapper_probe_r7 as W
        import wrapper_probe_r8fix as FIX
        torch.cuda.synchronize = (lambda *x, **k: None) if a.device == "cpu" else torch.cuda.synchronize
        rec.update(torch=torch.__version__, tanitad_file=__import__("tanitad").__file__,
                   gpu=(torch.cuda.get_device_name(0) if a.device != "cpu" and torch.cuda.is_available() else None))
        if a.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        h = hashlib.md5()
        with open(a.ckpt, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        rec["ckpt_md5"] = h.hexdigest()
        tr = L.trainer()
        config = L.load_config(a.config)
        model, cfg, args, mrec = L.build_model(config, a.ckpt, a.device)
        sd = mrec["state_dict"]
        rec["model"] = {"step": sd.get("step"), "missing": len(sd.get("missing") or []), "unexpected": len(sd.get("unexpected") or [])}
        rec["step"] = sd.get("step")
        G.set_flags(model, G.training_flags(model)["flags"])
        from tanitad.models.timm_trunk import TimmResNetTrunk
        trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
        law_idx = int(tr.LAW_AHEAD) - 1
        if a.batch_cache:
            d = torch.load(a.batch_cache, map_location="cpu", weights_only=False)
            eb = {k: (G.LawOnly(v["__lawonly__"], v["law_idx"]) if isinstance(v, dict) and "__lawonly__" in v else v)
                  for k, v in d["batch"].items()}
            rec["batch"] = {"source": "cache", "path": a.batch_cache, "meta": d.get("meta")}
        else:
            e_ds, _eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                                    dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
            B = int(args.batch)
            perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), B)
            sha = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
            eb = G.collate(e_ds, perm[0:B], law_idx)
            rec["batch"] = {"source": "collate(first in-run batch)", "perm_sha256": sha, "rows_window_index_first4": perm[0:4], "B": B}
        G.patch_frames_to_device(tr)
        mode = getattr(args, "mode", "diffusion")
        abl = bool(getattr(args, "ablate_frames", False))
        W.CONDITIONS = {"P3_fp32_det": W.CONDITIONS["P3_fp32_det"]}      # the clause is judged under P3 only
        bank()
        runs = ([] if a.no_control else [("control_unfixed", W.probe)]) + [("fixed", FIX.probe)]
        res = {}
        for name, fn in runs:
            t1 = time.time()
            if a.device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            r = fn(tr, model, eb, a.device, mode, abl, trunk, sub_batch_fn=G.sub_batch)
            cond = r["conditions"].get("P3_fp32_det") or {}
            if "analysis" not in cond:
                raise RuntimeError(f"{name}: P3 did not produce an analysis ({cond.get('status')})")
            res[name] = cond
            rec[f"{name}_wall_s"] = round(time.time() - t1, 1)
            if a.device != "cpu":
                rec[f"{name}_cuda_max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            print(f"[wrapfix] {name}: clause {r.get('clause')} max wrapper {cond['analysis']['max_wrapper_rel']:.3e} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        if "control_unfixed" in res:
            an_c = res["control_unfixed"]["analysis"]
            an_c["clause"] = clause_of(an_c)
            rec["control_unfixed"] = summarise(an_c)
            rec["control_reproduces_g0"] = reproduction(an_c, a.g0_json)
        an_f = res["fixed"]["analysis"]
        an_f["clause"] = clause_of(an_f)
        rec["fixed_as_returned"] = summarise(an_f)
        an_a = amend(res["fixed"])
        rec["amended"] = summarise(an_a)
        rec["CLAUSE"] = an_a["clause"]
        rec["max_wrapper_rel"] = an_a["max_wrapper_rel"]
        rec["bars"] = {k: v["bar"] for k, v in an_a["terms"].items() if k in KEYS}
        rec["eval_draws_neutralised"] = ["torch.randn_like", "torch.randn (generator-less)", "R8Generator.randn_eval", "R8Generator.rand_eval"]
        rec["reading"] = ("CLAUSE is the amended (W-R8-1) reading on the GPU: PASS means every shared loss term's micro-batched forward "
                          "equals the plain one within max(3*floor, 1e-5) AND the deliberate regressions W1/W2 are detected. "
                          "It does NOT turn the step-5,000 G0 verdict (FAIL as coded) into PASS by itself: the Master Mind registers that.")
        rec["ok"] = True
    except Exception as exc:                                    # noqa: BLE001 -- the json is the result
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    bank()
    print(f"wrapfix:{'ok:' + str(rec['CLAUSE']) if rec['ok'] else 'fail:' + str(rec.get('error'))}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
