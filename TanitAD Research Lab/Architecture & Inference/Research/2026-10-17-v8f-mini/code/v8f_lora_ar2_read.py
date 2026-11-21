"""PREREG_V8F_LORA: the AR-2 chain's reader -- what the chain (``box2_ops/chain_v8f_lora_ar2.sh``) asks of the files it pulled from box 2. Pure python, no GPU, no box 2.

    --check-declared   the declared train clips are the FIRST n of B's list (the cache manifest in sha12 order, --exclude 540:600 applied) and the declared eval clips
                       are the first of the eval manifest in sha12 order; every declared row is a production row
    --identity DIR     judge ``identity_lora.json`` (+ ``identity_lora_mutant.json``): PASS only if the control PASSED, was QUALIFYING (cuda, every frame, >= 2 train and
                       >= 2 eval clips, no autocast), covered exactly the declared clips, and the mutant FAILED (a control the mutant cannot break is blind)
    --smoke DIR        read ``smoke_report.json``: the AR-2 checks, the measured step time and peak memory, the projected hours of one 19,164-step seed at effective batch 64
                       (ESTIMATED from the measured per-frame costs) and the venue rule of PREREG_V8F_LORA proof 5 (box 2 iff <= 36 h)
    --compare-frames  built frames == banked production frames (tensor by tensor) for the declared clips

rc 0 = ok, 1 = a check failed (the reason is printed and in the json), 2 = bad arguments / missing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

VENUE_HOURS = 36.0                 # PREREG_V8F_LORA proof 5
FULL_STEPS = 19_164
TAU_DEVICE = 1e-3
LORA_TRAINABLE = 3_145_728
SMOKE_STEPS = 200
EXPECTED_ROW = {"source": "local256", "enc_dtype": "bf16"}


def sha12(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def list_sha256(clips) -> str:
    """The trainer's ``train_sha256``: newline-joined clip ids in sha12 order."""
    return hashlib.sha256(chr(10).join(sorted(clips, key=sha12)).encode("utf-8")).hexdigest()


def read_rows(path) -> dict:
    rows = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            rows[d["clip"]] = d
    return rows


def first_train_clips(rows: dict, n: int, exclude=(540, 600)) -> list:
    """B's list = the manifest's clips in sha12 order minus ``--exclude lo:hi`` of that order; the first ``n`` (``select_train_clips`` at fraction 1.0 + ``--max-clips n``)."""
    order = sorted(rows, key=sha12)
    ex = set(order[exclude[0]:exclude[1]])
    return [c for c in order if c not in ex][:n]


def check_declared(train_rows: dict, eval_rows: dict, train_ids, eval_ids, n_train: int, exclude=(540, 600)) -> list:
    bad = []
    want = first_train_clips(train_rows, n_train, exclude)
    if list(train_ids) != want:
        bad.append(f"the declared train clips {[c[:8] for c in train_ids]} are not the first {n_train} of B's list in sha12 order {[c[:8] for c in want]}")
    if len(train_rows) != 4572:
        bad.append(f"the train manifest has {len(train_rows)} rows, production has 4,572")
    if len(eval_rows) != 141:
        bad.append(f"the eval manifest has {len(eval_rows)} rows, production has 141")
    ew = sorted(eval_rows, key=sha12)[:len(eval_ids)]
    if list(eval_ids) != ew:
        bad.append(f"the declared eval clips {[c[:8] for c in eval_ids]} are not the first {len(eval_ids)} of the eval manifest in sha12 order {[c[:8] for c in ew]}")
    for split, ids, rows in (("train", train_ids, train_rows), ("eval", eval_ids, eval_rows)):
        for c in ids:
            r = rows.get(c)
            if r is None:
                bad.append(f"{split} clip {c[:8]} has no manifest row")
                continue
            for k, v in EXPECTED_ROW.items():
                if r.get(k) != v:
                    bad.append(f"{split} clip {c[:8]}: {k} = {r.get(k)!r}, production rows read {v!r}")
    return bad


def frames_equal(a, b) -> list:
    """Two v2ep files, tensor by tensor (the raw zip bytes can differ by the member names ``torch.save`` takes from the file name). -> differences (empty = equal)."""
    import torch
    x = torch.load(a, map_location="cpu", weights_only=False)
    y = torch.load(b, map_location="cpu", weights_only=False)
    bad = []
    if set(x) != set(y):
        return [f"keys differ: {sorted(set(x) ^ set(y))}"]
    for k in sorted(x):
        u, v = x[k], y[k]
        if isinstance(u, torch.Tensor):
            if u.dtype != v.dtype or u.shape != v.shape or not torch.equal(u, v):
                bad.append(f"tensor {k} differs")
        elif u != v:
            bad.append(f"{k}: {u!r} != {v!r}")
    return bad


# --------------------------------------------------------------------------------------------- the identity control
def judge_identity(real: dict, mutant: dict, train_ids, eval_ids, expect_driver_sha=None) -> dict:
    bad = []
    if real.get("verdict") != "PASS":
        bad.append(f"the identity control did not PASS: verdict {real.get('verdict')!r}; " + "; ".join(real.get("fail_reasons") or [real.get("refused", "no reason recorded")]))
    if not real.get("qualifying"):
        bad.append("the reading is not qualifying: " + "; ".join(real.get("qualifying_reasons") or ["no qualifying field"]))
    if real.get("mutant") is not None:
        bad.append("the control file is a mutant run")
    got = set((real.get("clips") or {}))
    want = set(train_ids) | set(eval_ids)
    if got != want:
        bad.append(f"the control covered {sorted(c[:8] for c in got)}, the declared clips are {sorted(c[:8] for c in want)}")
    if real.get("tau_device") != TAU_DEVICE:
        bad.append(f"tau_device {real.get('tau_device')!r} != {TAU_DEVICE}")
    if real.get("weights_sha256") is None or real.get("weights_sha256") != real.get("builder_model_sha256"):
        bad.append("the encoder weights' sha256 is not the cache builder's")
    if real.get("builder_unchanged") is not True:
        bad.append("the cache builder's encode_episode is not the one the mirror was pinned against")
    for c, v in (real.get("clips") or {}).items():
        r = v.get("manifest_row_matched") or {}
        if r.get("source") != "local256" or r.get("enc_dtype") != "bf16":
            bad.append(f"{c[:8]}: the matched manifest row is not a production row ({r.get('source')!r}, {r.get('enc_dtype')!r})")
        if v.get("n_compared") != v.get("n_frames"):
            bad.append(f"{c[:8]}: {v.get('n_compared')} of {v.get('n_frames')} frames compared")
    if expect_driver_sha and real.get("driver_sha256") != expect_driver_sha:
        bad.append(f"the control ran driver {str(real.get('driver_sha256'))[:12]}, the tip's is {expect_driver_sha[:12]}")
    mut_worst = None
    if mutant is None:
        bad.append("no mutant reading: the control is not evidence without one")
    else:
        mut_worst = mutant.get("worst_rel_l2")
        if not mutant.get("mutant"):
            bad.append("the mutant file is not a mutant run")
        if mutant.get("mutant_caught") is not True:
            bad.append(f"the mutant {mutant.get('mutant')!r} was NOT caught (verdict {mutant.get('verdict')!r}, worst rel-L2 {mut_worst}): the instrument is blind to a changed preprocessing constant")
        if set((mutant.get("clips") or {})) != want:
            bad.append("the mutant did not cover the declared clips")
        if expect_driver_sha and mutant.get("driver_sha256") != expect_driver_sha:
            bad.append("the mutant ran another driver than the tip's")
    return {"ok": not bad, "problems": bad, "worst_rel_l2": real.get("worst_rel_l2"), "worst_clip": real.get("worst_clip"), "mutant": (mutant or {}).get("mutant"),
            "mutant_worst_rel_l2": mut_worst, "mutant_caught": (mutant or {}).get("mutant_caught"),
            "per_clip_rel_l2": {c[:8]: v.get("rel_l2") for c, v in (real.get("clips") or {}).items()},
            "per_clip_frac_bitwise_equal": {c[:8]: v.get("frac_bitwise_equal") for c, v in (real.get("clips") or {}).items()}, "device": real.get("cuda") or real.get("device")}


# --------------------------------------------------------------------------------------------- the AR-2 smoke
def venue_of(hours, device_is_cuda: bool) -> str:
    if not device_is_cuda or hours is None:
        return "n/a (not measured on a GPU)"
    return ("BOX 2 (projected <= 36 h per seed; needs the PI's go, one job, safe mode, the temperature guard)" if hours <= VENUE_HOURS
            else f"A40 under AR-1 to AR-4 (plan 13.5): box 2 projects {hours:.1f} h per seed > {VENUE_HOURS:g} h")


def read_smoke(rep: dict, steps: list, declared_train, selfcheck: dict | None = None) -> dict:
    bad = []
    v = rep.get("verdict") or {}
    mb = rep.get("measured_batch1") or {}
    pr = rep.get("projection_ESTIMATED") or {}
    sc = rep.get("selfcheck") or {}
    lv = ((sc.get("levers") or {}).get("encoder")) or {}
    info = (selfcheck or {}).get("info") or {}
    is_cuda = str(rep.get("device", "")).startswith("cuda")
    if not v.get("AR2_gradient_and_finite_checks_pass"):
        bad.append("the AR-2 gradient / finite checks did not pass: " + json.dumps({k: x for k, x in v.items() if isinstance(x, bool) and not x}))
    if rep.get("rc") != 0:
        bad.append(f"the trainer returned rc {rep.get('rc')}")
    if rep.get("amp") is not False or "--amp" in (rep.get("trainer_argv") or []):
        bad.append("autocast was ON: Amendment 2 pins --amp OFF (the builder's pure-bf16 numerics)")
    if rep.get("cpu_fp32_mechanics_proof"):
        bad.append("this is the CPU mechanics proof, not the box-2 smoke")
    if not is_cuda:
        bad.append(f"device {rep.get('device')!r} is not a cuda device")
    if v.get("n_steps_run") != SMOKE_STEPS:
        bad.append(f"{v.get('n_steps_run')} steps ran, the smoke is {SMOKE_STEPS}")
    if str(lv.get("backbone_dtype")) != "torch.bfloat16" or lv.get("checkpointing") is not True or lv.get("lora_trainable") != LORA_TRAINABLE:
        bad.append(f"the built encoder is not bf16 + gradient checkpointing + {LORA_TRAINABLE:,} LoRA parameters: {lv.get('backbone_dtype')}, ckpt {lv.get('checkpointing')}, {lv.get('lora_trainable')}")
    if info:
        if info.get("n_clips") != len(declared_train) or info.get("train_sha256") != list_sha256(declared_train):
            bad.append(f"the smoke trained on {info.get('n_clips')} clips (train_sha256 {str(info.get('train_sha256'))[:12]}), not the declared {len(declared_train)} ({list_sha256(declared_train)[:12]})")
        if info.get("frames_missing") not in (0, None):
            bad.append(f"{info.get('frames_missing')} declared clips had no frames")
    else:
        bad.append("no selfcheck.json: the smoke's clip set could not be checked against the declared one")
    ip = rep.get("identity_probe_tau_device")
    # PREREG_V8F_LORA Amendment 2 binds the ON-THE-FLY B = 0 path to tau_device; the training path (16-frame micro-batches) IS that path, so its
    # probe gates AR-2 (added by the MM 2026-10-25: the driver's control encodes in the builder's batches of 8 and cannot speak for 16)
    if not ip or ip.get("rel_l2") is None:
        bad.append("no training-path identity probe: the 16-frame micro-batch path was never compared with the cache")
    elif not (float(ip["rel_l2"]) <= TAU_DEVICE):
        bad.append(f"the TRAINING path's identity probe misses tau_device: rel-L2 {float(ip['rel_l2']):.3e} > {TAU_DEVICE:g} "
                   "(train with the builder's batch of 8, or find why the micro-batch changes the numerics, before any training)")
    steps_t = [s["step_wall_s"] for s in steps] if steps else []
    out = {"ok": not bad, "problems": bad, "device": rep.get("device"), "cuda": rep.get("cuda"), "amp": rep.get("amp"), "n_steps_run": v.get("n_steps_run"),
           "evidence_class": {"step_time_peak_memory": "MEASURED (the smoke, this run)", "projected_hours": "ESTIMATED from the measured per-frame costs (not run)"},
           "step_time_batch1_s": {"median": mb.get("step_wall_s_median"), "iter_median": mb.get("iter_s_median"), "n_in_medians": mb.get("n_steps_in_medians"),
                                  "first": steps_t[0] if steps_t else None, "max": max(steps_t) if steps_t else None},
           "phases_median_s": {"encode_nograd": mb.get("phase1_encode_nograd_s_median"), "heads_fwd_bwd": mb.get("heads_fwd_bwd_s_median"), "encode_grad_bwd": mb.get("phase2_encode_grad_s_median")},
           "unique_frames_per_step_median": mb.get("unique_frames_per_step_median"),
           "peak_memory_gib": {"allocated_max": mb.get("peak_alloc_gib_max"), "reserved_max": mb.get("peak_reserved_gib_max"),
                               "heads_alone_at_batch_64_allocated": ((pr.get("t_heads_b64") or {}).get("peak_alloc_gib_max")),
                               "note": "the smoke's peak is the encoder at a 16-frame micro-batch with batch-1 heads; the real step also holds the batch-64 heads and the features of every unique frame; the two are NOT measured together"},
           "loss_first_to_last": mb.get("loss_first_to_last"),
           "identity_probe_tau_device": ip,
           "projection_ESTIMATED": {"hours_one_seed": pr.get("hours_one_seed_ESTIMATED"), "steps": pr.get("steps", FULL_STEPS), "step_s_at_batch_64": pr.get("step_s_b64_ESTIMATED"),
                                    "U_unique_frames_per_window": pr.get("U_unique_frames_per_window_b64"), "census_scope": pr.get("census_scope"),
                                    "assumes_micro_batch_frames": pr.get("assumes_micro_batch_frames"), "formula": pr.get("formula"), "not_in_the_model": pr.get("not_in_the_model"),
                                    "assumes": "effective batch 64 reached by gradient caching in 16-frame encoder micro-batches (--accum 64); box 2's 8 GB card"},
           "venue_rule_proof5": venue_of(pr.get("hours_one_seed_ESTIMATED"), is_cuda)}
    if pr.get("hours_one_seed_ESTIMATED") is None:
        out["ok"] = False
        out["problems"].append("the smoke report carries no projection")
    return out


def summary_line(ident: dict, smoke: dict) -> str:
    ip = smoke.get("identity_probe_tau_device") or {}
    st, pk, pj = smoke.get("step_time_batch1_s", {}), smoke.get("peak_memory_gib", {}), smoke.get("projection_ESTIMATED", {})
    f = lambda x, fmt: "n/a" if x is None else format(x, fmt)
    return (f"identity PASS worst rel-L2 {f(ident.get('worst_rel_l2'), '.2e')} ({(ident.get('worst_clip') or '')[:8]}), mutant {ident.get('mutant')} caught at {f(ident.get('mutant_worst_rel_l2'), '.2e')} | "
            f"smoke {smoke.get('n_steps_run')} steps: step {f(st.get('median'), '.2f')} s, peak {f(pk.get('allocated_max'), '.2f')} GiB alloc / {f(pk.get('reserved_max'), '.2f')} reserved, "
            f"smoke-path identity probe rel-L2 {f(ip.get('rel_l2'), '.2e')} ({'<=' if ip.get('pass') else '>'} 1e-3), "
            f"projected {f(pj.get('hours_one_seed'), '.1f')} h/seed (ESTIMATED) -> {smoke.get('venue_rule_proof5')}")


# --------------------------------------------------------------------------------------------- cli
def _json(p):
    return json.load(open(p, encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-declared", action="store_true")
    ap.add_argument("--identity", default=None, help="dir with identity_lora.json and identity_lora_mutant.json")
    ap.add_argument("--smoke", default=None, help="dir with smoke_report.json (+ run/selfcheck.json)")
    ap.add_argument("--compare-frames", action="store_true")
    ap.add_argument("--manifest-train")
    ap.add_argument("--manifest-eval")
    ap.add_argument("--train-clips", nargs="*", default=[])
    ap.add_argument("--eval-clips", nargs="*", default=[])
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--exclude", default="540:600")
    ap.add_argument("--expect-driver-sha", default=None)
    ap.add_argument("--built", help="--compare-frames: dir of freshly built <clip>.v2ep.pt (train/ and eval/ subdirs)")
    ap.add_argument("--banked", nargs="*", default=[], help="--compare-frames: dirs holding the banked production <clip>.v2ep.pt (searched in order)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.exclude.split(":"))
    res, bad = {}, []
    try:
        if a.check_declared:
            bad += check_declared(read_rows(a.manifest_train), read_rows(a.manifest_eval), a.train_clips, a.eval_clips, a.n_train, (lo, hi))
            res["check_declared"] = {"ok": not bad, "problems": list(bad)}
        if a.compare_frames:
            diffs = {}
            for split, ids in (("train", a.train_clips), ("eval", a.eval_clips)):
                for c in ids:
                    cand = [Path(d) / f"{c}.v2ep.pt" for d in a.banked if (Path(d) / f"{c}.v2ep.pt").is_file()]
                    if not cand:
                        diffs[c] = ["no banked production frames file found in " + ", ".join(a.banked)]
                        continue
                    diffs[c] = frames_equal(Path(a.built) / split / f"{c}.v2ep.pt", cand[0])
            res["compare_frames"] = {"differences": {c[:8]: d for c, d in diffs.items()}, "ok": not any(diffs.values())}
            bad += [f"{c[:8]}: {x}" for c, d in diffs.items() for x in d]
        ident = smoke = None
        if a.identity:
            d = Path(a.identity)
            real = _json(d / "identity_lora.json")
            mutant = _json(d / "identity_lora_mutant.json") if (d / "identity_lora_mutant.json").is_file() else None
            ident = judge_identity(real, mutant, a.train_clips, a.eval_clips, a.expect_driver_sha)
            res["identity"] = ident
            bad += ident["problems"]
        if a.smoke:
            d = Path(a.smoke)
            rep = _json(d / "smoke_report.json")
            steps = [json.loads(x) for x in open(d / "smoke_steps.jsonl", encoding="utf-8") if x.strip()] if (d / "smoke_steps.jsonl").is_file() else []
            sc = _json(d / "run" / "selfcheck.json") if (d / "run" / "selfcheck.json").is_file() else None
            smoke = read_smoke(rep, steps, a.train_clips, sc)
            res["smoke"] = smoke
            bad += smoke["problems"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}")
        return 2
    res["ok"] = not bad
    res["problems"] = bad
    if ident and smoke and ident["ok"]:
        res["summary"] = summary_line(ident, smoke)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    if "summary" in res:
        print(res["summary"])
    for b in bad:
        print("PROBLEM:", b)
    print("OK" if not bad else "NOT OK")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
