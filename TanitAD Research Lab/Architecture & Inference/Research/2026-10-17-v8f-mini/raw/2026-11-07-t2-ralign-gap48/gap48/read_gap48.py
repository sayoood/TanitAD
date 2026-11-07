"""Reads a pulled GAP48 run on the dev box. DESCRIPTIVE: it reports, it decides nothing. CPU only.

    python -I read_gap48.py --pull C:/Users/Admin/t2v8f_pulls/gap48/pull_gap48 [--manifest build/MANIFEST_GAP48.json] [--out-dir .] [--allow-stub]

Writes RESULT_GAP48.json and RESULT_GAP48.md into --out-dir. Everything is recomputed from the banked token tensors (fp16 [n,40,1024], the exact image of the fp8 tokens); the cosines the box
recorded are compared with the recomputed ones and a mismatch is a named failure.

What it reads:
  1. render-vs-real, per scene and pooled: flattened, uncentered cosine of the pooled [40,1024] tokens (`t2a_v8f_encoder.cosine`'s statistic, as GAPSCALE) and the relative L2. Quantiles: numpy linear.
  2. the pinned eval-141 mean (`refs/blind_mean_eval141.pt` inside stage1/t2v8f_stage1.tar, tensor-bytes sha256 cd6a7ab5...a2fcb9, asserted): cosine(render tokens, mean) against cosine(real tokens, mean),
     paired per tick. GAPSCALE could not measure the first because render tokens were not banked.
  3. the yardstick of gapscale/GAPSCALE.json beside it (real-vs-real across clips, within a clip at lags, real vs the mean, stage 1's two-scene render-vs-real).
IDENTITY GUARD: a scene whose render tokens equal its real tokens bit for bit on every tick is flagged IDENTITY (the harness encoded the same frame twice) and excluded from the pooled statistics;
if any scene is flagged the run status is IDENTITY_DETECTED. A run made with the stub renderer or with the test hook is labelled and is not a measurement of the NuRec render.
ESTIMATOR: the interval on the pooled median is a SCENE-cluster bootstrap (resample the scenes, seed 0, 2000 draws): it answers "would another draw of scenes say this?". The renderer is deterministic
(stage-1 T5, bit-equal), the encoder is fixed, and there is no training run in the loop, so the other two variances (training, inference sampling) do not apply to this measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
MEAN_SHA = "cd6a7ab5e3c5450ec3528889b04e06e5605fea30ea9ff02a0e36d9ba47a2fcb9"
MEAN_MEMBER = "pkg/refs/blind_mean_eval141.pt"
STAGE1_TAR = f"{T2}/stage1/t2v8f_stage1.tar"
GAPSCALE = f"{T2}/gapscale/GAPSCALE.json"
IDENT_TOL = 1e-9
RECOMPUTE_TOL = 1e-9
BOOT_N, BOOT_SEED = 2000, 0


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cos(a, b):
    """Flattened, uncentered cosine in float64 (the statistic of t2a_v8f_encoder.cosine)."""
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def rel_l2(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def q(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return {"n": int(x.size), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}


def load_mean(stage1_tar=STAGE1_TAR):
    import torch
    with tarfile.open(stage1_tar) as tf:
        b = tf.extractfile(MEAN_MEMBER).read()
    m = torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
    got = hashlib.sha256(m.numpy().tobytes()).hexdigest()
    if got != MEAN_SHA or tuple(m.shape) != (40, 1024) or str(m.dtype) != "torch.float16":
        raise SystemExit(f"ZZ GAP48-MEAN-NOT-PINNED sha256={got} shape={tuple(m.shape)} dtype={m.dtype} ZZ")
    return m.numpy(), got


def boot_ci(vals_by_scene, stat=np.median, n=BOOT_N, seed=BOOT_SEED):
    """Scene-cluster bootstrap of a statistic of the pooled per-tick values: resample SCENES with replacement."""
    rng = np.random.default_rng(seed)
    keys = list(vals_by_scene)
    if len(keys) < 2:
        return None
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        out[i] = stat(np.concatenate([vals_by_scene[keys[j]] for j in pick]))
    return {"lo95": float(np.percentile(out, 2.5)), "hi95": float(np.percentile(out, 97.5)), "n_draws": n, "seed": seed, "estimator": "scene-cluster bootstrap; answers: would another draw of scenes say this"}


def read(pull, manifest_path, stage1_tar=STAGE1_TAR, gapscale=GAPSCALE, allow_stub=False):
    import torch
    pull = pull.replace("\\", "/")
    rj = json.load(open(f"{pull}/GAP48_RESULT.json", encoding="utf-8"))
    man = json.load(open(manifest_path, encoding="utf-8"))
    mean, mean_sha = load_mean(stage1_tar)
    marks = {n: os.path.isfile(f"{pull}/{n}") for n in ("GAP48.done", "GAP48.fail", "GAP48.running")}
    problems, scenes, flags = [], {}, []
    if rj.get("stub") and not allow_stub:
        problems.append("STUB_RESULT: made with the stub renderer and stub encoder; not a measurement of the NuRec render (use --allow-stub for a rehearsal read)")
    if rj.get("test_hook"):
        problems.append("TEST_HOOK_RESULT: made with the job's test seam set; not a measurement")
    if rj.get("manifest_sha256") != hashlib.sha256(open(manifest_path, "rb").read()).hexdigest():
        problems.append("MANIFEST_MISMATCH: the result was made against another manifest than the one given")
    if rj.get("condition") != "objects":
        flags.append(f"CONDITION_{rj.get('condition')}: not the objects condition")
    want = [s["scene"] for s in man["scenes"]][: rj["n_scenes_expected"]]
    for sid in want:
        r = (rj.get("scenes") or {}).get(sid)
        if not r or r.get("status") != "DONE":
            problems.append(f"SCENE_NOT_BANKED {sid[:8]} {None if not r else r.get('reason')}")
            continue
        p = f"{pull}/{r['tokens_file']}"
        if not os.path.isfile(p) or sha256_file(p) != r["tokens_sha256"]:
            problems.append(f"TOKENS_SHA_MISMATCH {sid[:8]}")
            continue
        t = torch.load(p, map_location="cpu", weights_only=True)
        R, Q = t["render"].numpy(), t["real"].numpy()
        if R.shape != Q.shape or R.shape[1:] != (40, 1024) or R.dtype != np.float16:
            problems.append(f"TOKENS_SHAPE {sid[:8]} {R.shape} {R.dtype}")
            continue
        c = np.array([cos(R[i], Q[i]) for i in range(len(R))])
        l2 = np.array([rel_l2(R[i], Q[i]) for i in range(len(R))])
        if np.abs(c - np.asarray(r["cosine"])).max() > RECOMPUTE_TOL or np.abs(l2 - np.asarray(r["rel_l2"])).max() > RECOMPUTE_TOL:
            problems.append(f"RECOMPUTE_MISMATCH {sid[:8]} max|dcos|={np.abs(c - np.asarray(r['cosine'])).max():.3e}")
        bit_equal = np.array([np.array_equal(R[i], Q[i]) for i in range(len(R))])
        ident = bool(bit_equal.all()) and bool(np.all(c >= 1.0 - IDENT_TOL))
        cr = np.array([cos(R[i], mean) for i in range(len(R))])
        cq = np.array([cos(Q[i], mean) for i in range(len(R))])
        scenes[sid] = {"scene8": sid[:8], "idx": r["idx"], "where": r["where"], "n_pairs": int(len(R)), "identity": ident, "n_bit_equal_ticks": int(bit_equal.sum()), "cos": c, "rel_l2": l2,
                       "cos_render_mean": cr, "cos_real_mean": cq, "seconds": r.get("seconds"), "actors": r.get("actors")}
    ident_scenes = sorted(s["scene8"] for s in scenes.values() if s["identity"])
    for s_ in scenes.values():
        if 0 < s_["n_bit_equal_ticks"] < s_["n_pairs"]:
            flags.append(f"PARTIAL_IDENTITY {s_['scene8']}: {s_['n_bit_equal_ticks']} of {s_['n_pairs']} ticks have render tokens equal to the real tokens bit for bit")
    use = {k: v for k, v in scenes.items() if not v["identity"]}
    status = ("IDENTITY_DETECTED" if ident_scenes else "STUB_OR_HOOK_NOT_A_MEASUREMENT" if (rj.get("stub") or rj.get("test_hook")) else "INCOMPLETE" if problems else "OK")
    out = {"stage": "gap48", "status": status, "problems": problems, "flags": flags, "identity_scenes": ident_scenes, "markers": marks, "stub": bool(rj.get("stub")), "test_hook": bool(rj.get("test_hook")),
           "condition": rj.get("condition"), "tick_step": rj.get("tick_step"), "n_scenes_expected": rj["n_scenes_expected"], "n_scenes_read": len(scenes), "n_scenes_used": len(use), "mean_sha256": mean_sha,
           "manifest_sha256": rj.get("manifest_sha256"), "scene_list_sha256": man.get("scene_list_sha256"), "evidence": "MEASURED (ours): GAP48_RESULT.json and tokens/*.pt of the pull; recomputed here",
           "estimator": "pooled quantiles are over every (scene, tick) pair; the interval is a scene-cluster bootstrap (would another draw of SCENES say this). No training or inference sampling in the loop."}
    if use:
        allc = np.concatenate([v["cos"] for v in use.values()])
        alll = np.concatenate([v["rel_l2"] for v in use.values()])
        allr = np.concatenate([v["cos_render_mean"] for v in use.values()])
        allq = np.concatenate([v["cos_real_mean"] for v in use.values()])
        diff = allr - allq
        sm = np.array([np.median(v["cos"]) for v in use.values()])
        out["render_vs_real"] = {"cosine_pairs": q(allc), "cosine_pairs_median_ci": boot_ci({k: v["cos"] for k, v in use.items()}), "cosine_scene_medians": q(sm), "rel_l2_pairs": q(alll),
                                 "n_scenes_median_below_0p8": int((sm < 0.8).sum())}
        out["render_vs_mean"] = {"cos_render_to_mean": q(allr), "cos_real_to_mean": q(allq), "paired_diff_render_minus_real": q(diff),
                                 "paired_diff_median_ci": boot_ci({k: v["cos_render_mean"] - v["cos_real_mean"] for k, v in use.items()}),
                                 "frac_pairs_render_closer_to_mean": float((diff > 0).mean()), "n_scenes_render_closer_median": int(sum(np.median(v["cos_render_mean"] - v["cos_real_mean"]) > 0 for v in use.values()))}
    out["per_scene"] = {k: {"scene8": v["scene8"], "idx": v["idx"], "where": v["where"], "n_pairs": v["n_pairs"], "identity": v["identity"], "cosine": q(v["cos"]), "rel_l2_median": float(np.median(v["rel_l2"])),
                            "cos_render_mean_median": float(np.median(v["cos_render_mean"])), "cos_real_mean_median": float(np.median(v["cos_real_mean"])), "seconds_total": (v["seconds"] or {}).get("total")}
                        for k, v in scenes.items()}
    if os.path.isfile(gapscale):
        g = json.load(open(gapscale, encoding="utf-8"))
        out["yardstick"] = {"source": "gapscale/GAPSCALE.json", "stage1_render_vs_real_median": g["stage1_render_vs_real_median"], "real_across_clips": g["real_across_clips"],
                            "real_within_clip_lag_ticks": g["real_within_clip_lag_ticks"], "real_frame_vs_eval_mean": g["real_frame_vs_eval_mean"]}
        if use:
            med = out["render_vs_real"]["cosine_pairs"]["median"]
            rows = {"across clips": g["real_across_clips"]["median"], **{f"same clip, lag {k}": v["median"] for k, v in g["real_within_clip_lag_ticks"].items()}}
            nearest = min(rows, key=lambda k: abs(rows[k] - med))
            out["yardstick"]["nearest_row_to_gap48_median"] = {"row": nearest, "median": rows[nearest], "gap48_median": med}
    secs = [v["seconds"]["total"] for v in scenes.values() if v.get("seconds") and v["seconds"].get("total") is not None]
    out["timing"] = {"scene_seconds": q(secs), "sum_seconds": float(np.sum(secs)) if secs else None}
    return out


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def render_md(o):
    L = [f"# RESULT_GAP48: the NuRec render gap on the read scenes (descriptive; decides nothing)", ""]
    L += [f"**Status: {o['status']}.** Scenes read {o['n_scenes_read']} of {o['n_scenes_expected']}, used {o['n_scenes_used']}. Condition `{o['condition']}`, every {o['tick_step']}th tick. Evidence class: {o['evidence']}.", ""]
    if o["problems"] or o["identity_scenes"] or o["flags"]:
        L += ["## Read before the numbers", ""]
        L += [f"* IDENTITY: scene(s) {', '.join(o['identity_scenes'])} have render tokens equal to the real tokens bit for bit on every tick (the harness encoded one frame twice); excluded from every pooled figure." if o["identity_scenes"] else ""]
        L += [f"* {x}" for x in o["problems"] + o["flags"]]
        L += [""]
    if "render_vs_real" in o:
        rv, rm = o["render_vs_real"], o["render_vs_mean"]
        c, ci, sm = rv["cosine_pairs"], rv["cosine_pairs_median_ci"], rv["cosine_scene_medians"]
        L += ["## 1. Render vs the real frame at the same tick (flattened uncentered cosine of the [40,1024] tokens)", "",
              "| pooled over | n | median | p10 | p90 | min | max |", "|---|---|---|---|---|---|---|",
              f"| all (scene, tick) pairs | {c['n']} | {f4(c['median'])} | {f4(c['p10'])} | {f4(c['p90'])} | {f4(c['min'])} | {f4(c['max'])} |",
              f"| scene medians | {sm['n']} | {f4(sm['median'])} | {f4(sm['p10'])} | {f4(sm['p90'])} | {f4(sm['min'])} | {f4(sm['max'])} |", "",
              f"Median of the pairs, 95 % scene-cluster bootstrap interval: [{f4(ci and ci['lo95'])}, {f4(ci and ci['hi95'])}] ({ci and ci['n_draws']} draws, seed {ci and ci['seed']}). Scenes with a median below 0.8: {rv['n_scenes_median_below_0p8']} of {sm['n']}.",
              f"Relative L2 (pairs): median {f4(rv['rel_l2_pairs']['median'])}, p10 {f4(rv['rel_l2_pairs']['p10'])}, p90 {f4(rv['rel_l2_pairs']['p90'])}.", "",
              "## 2. Distance to the pinned eval-141 mean (paired per tick)", "",
              "| quantity | n | median | p10 | p90 |", "|---|---|---|---|---|"]
        for lab, key in (("cosine(render tokens, mean)", "cos_render_to_mean"), ("cosine(real tokens, mean)", "cos_real_to_mean"), ("paired difference, render minus real", "paired_diff_render_minus_real")):
            x = rm[key]
            L.append(f"| {lab} | {x['n']} | {f4(x['median'])} | {f4(x['p10'])} | {f4(x['p90'])} |")
        d = rm["paired_diff_median_ci"]
        L += ["", f"Median paired difference, 95 % scene-cluster interval: [{f4(d and d['lo95'])}, {f4(d and d['hi95'])}]. The render is closer to the mean than its real frame in {100 * rm['frac_pairs_render_closer_to_mean']:.1f} % of pairs "
              f"and in {rm['n_scenes_render_closer_median']} of {c and sm['n']} scenes (scene median). Mean tensor sha256 `{o['mean_sha256'][:8]}...{o['mean_sha256'][-6:]}` (asserted).", ""]
    if "yardstick" in o:
        y = o["yardstick"]
        L += ["## 3. Yardstick, beside (gapscale/GAPSCALE.json: real frames of the eval-141 cache, same statistic)", "", "| pair | median [p10, p90] |", "|---|---|"]
        if "render_vs_real" in o:
            c = o["render_vs_real"]["cosine_pairs"]
            L.append(f"| **GAP48: render vs its real frame** | **{f4(c['median'])} [{f4(c['p10'])}, {f4(c['p90'])}]** |")
        s1 = y["stage1_render_vs_real_median"]
        L.append(f"| stage 1 render vs real, two scenes (no actors attached) | {', '.join(f'{k} {f4(v)}' for k, v in s1.items())} |")
        for k, v in y["real_within_clip_lag_ticks"].items():
            L.append(f"| real vs the same clip, lag {k} | {f4(v['median'])} [{f4(v['p10'])}, {f4(v['p90'])}] |")
        a, m = y["real_across_clips"], y["real_frame_vs_eval_mean"]
        L += [f"| real vs a frame of a different clip | {f4(a['median'])} [{f4(a['p10'])}, {f4(a['p90'])}] |", f"| real vs the eval-141 mean | {f4(m['median'])} [{f4(m['p10'])}, {f4(m['p90'])}] |", ""]
        if "nearest_row_to_gap48_median" in y:
            n = y["nearest_row_to_gap48_median"]
            L += [f"The GAP48 median ({f4(n['gap48_median'])}) sits nearest the row \"{n['row']}\" ({f4(n['median'])}). Descriptive placement only; no threshold is attached to it.", ""]
    ps = o.get("per_scene") or {}
    if ps:
        L += ["## 4. Per scene", "", "| scene | root | pairs | cosine median [p10, p90] | rel-L2 median | render-to-mean | real-to-mean | identity |", "|---|---|---|---|---|---|---|---|"]
        for v in sorted(ps.values(), key=lambda v: v["idx"]):
            c = v["cosine"]
            L.append(f"| {v['scene8']} | {v['where']} | {v['n_pairs']} | {f4(c['median'])} [{f4(c['p10'])}, {f4(c['p90'])}] | {f4(v['rel_l2_median'])} | {f4(v['cos_render_mean_median'])} | {f4(v['cos_real_mean_median'])} | {'IDENTITY' if v['identity'] else ''} |")
        L.append("")
    t = o.get("timing") or {}
    if t.get("sum_seconds") is not None:
        L += [f"Box-2 scene wall time (sum over scenes): {t['sum_seconds'] / 60:.1f} min, per scene median {t['scene_seconds']['median']:.1f} s.", ""]
    L += ["## Estimator and limits", "", o["estimator"], "",
          "One renderer build per scene (deterministic, stage-1 T5 bit-equal), the frozen in-loop DINOv3-L encoder, no policy in the loop: this measures the render, not an arm. "
          "It decides nothing by itself; the placement against the yardstick is a description.", ""]
    return "\n".join(x for x in L if x is not None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--manifest", default=f"{HERE}/build/MANIFEST_GAP48.json")
    ap.add_argument("--stage1-tar", default=STAGE1_TAR)
    ap.add_argument("--gapscale", default=GAPSCALE)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--allow-stub", action="store_true")
    a = ap.parse_args(argv)
    o = read(a.pull, a.manifest, a.stage1_tar, a.gapscale, a.allow_stub)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(o, open(f"{a.out_dir}/RESULT_GAP48.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    open(f"{a.out_dir}/RESULT_GAP48.md", "w", encoding="utf-8", newline="\n").write(render_md(o) + "\n")
    print(json.dumps({"status": o["status"], "n_scenes_used": o["n_scenes_used"], "problems": o["problems"][:3]}))
    return 0 if o["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
