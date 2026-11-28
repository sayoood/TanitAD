#!/usr/bin/env python3
"""YAXN P2 AXNOISE chain (PREREG_AXD.md section 6, IMPL_P2.md). Idempotent (a finished stage leaves stages/<name>.json); LAST ACTION writes YAXN/markers/axn_<tag>.DONE|FAILED (json).
    python axn_chain.py --tag smoke       # tiny: 4-step trainings, 8-token CPU seams, no GPU lock, + RED seams; mechanics only
    python axn_chain.py --tag full        # K1 200 steps x seeds 0,1, 923 tokens, GPU + shared lock (never forced), MemoryError retry
Stages: pins -> disk -> gates_axn (CPU, with RED arms) -> per seed [train -> compose (K1-K5) -> identity (G-CKPT)] -> [smoke: RED seams] -> per seed [seam (lock, retry) -> harness -> families6 -> perwindow] -> analyze -> cleanup.
Strictly sequential: this chain never runs two training / GPU / eval processes at once. Host memory is shared with other sessions: every subprocess stage waits (bounded) for headroom and retries (bounded, logged) on a MemoryError.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axn_common as N  # noqa: E402

A, C = N.A, N.C
import ax_eval_arm as AE  # noqa: E402  (YAX, read-only: seam_env, GL = the shared dev-box GPU lock, run_score, run_families, run_perwindow, HP)
import axn_analyze as AZ  # noqa: E402  (imports axd_analyze; never imports axn_seam, which patches axd_common)

SMOKE_TOKENS = Path("C:/Users/Admin/yaxd_run/tokens/tokens_red.json")        # YAXD's 8-token A5 log, 3 CRUISE
RED_SEAMS = ("idt_perturbed", "transform_not_applied", "memo_stale")


class Ctx:
    def __init__(self, tag):
        self.tag, self.smoke = tag, tag == "smoke"
        self.root = N.RUN_BASE / tag
        for d in ("stages", "logs", "arms", "ckpt", "identity", "eval", "analysis", "tokens", "red"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def log(self, m):
        line = f"[{N.now()}] {m}"
        print(line, flush=True)
        with open(self.root / "PIPELINE.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def stage(self, name, fn):
        mp = self.root / "stages" / f"{name}.json"
        if mp.exists():
            return N.jload(mp)
        self.log(f"stage {name}: start")
        t0 = time.time()
        res = {**(fn() or {}), "seconds": round(time.time() - t0, 1), "finished": N.now()}
        N.jdump(res, mp)
        self.log(f"stage {name}: done in {res['seconds']} s")
        return res


def marker(tag, kind, **kw):
    N.MARKERS.mkdir(exist_ok=True)
    N.jdump(dict(kw, kind=kind, t=N.now(), pid=os.getpid()), N.MARKERS / f"axn_{tag}.{kind}")


def fresh(p: Path):
    if p.exists():
        os.rename(p, str(p) + f".stale_{int(time.time())}")


def sh_retry(ctx, name, argv, token, env=None, cwd=None, attempts=3, expect_rc=0):
    """run a subprocess; retry (bounded, logged) only when its tail shows a host-memory failure."""
    for k in range(1, attempts + 1):
        N.wait_mem(ctx.log, min_phys=3000, min_virt=4500, max_wait_s=600)
        rc, txt = N.sh(argv, ctx.root / "logs" / f"{name}.attempt{k}.log", env=env, cwd=cwd)
        if rc == expect_rc and (token is None or token in txt):
            shutil.copyfile(ctx.root / "logs" / f"{name}.attempt{k}.log", ctx.root / "logs" / f"{name}.log")
            return txt
        if k < attempts and N.is_mem_failure(txt):
            ctx.log(f"RETRY {name}: attempt {k} hit a memory failure (rc {rc}); retrying after a pause")
            with open(ctx.root / "logs" / "retries.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": N.now(), "stage": name, "attempt": k, "rc": rc}) + "\n")
            time.sleep(60 * k)
            continue
        raise RuntimeError(f"{name}: rc={rc} (want {expect_rc}) token {token!r} {'present' if token and token in txt else 'ABSENT'}; tail: {txt[-1500:]}")


def check_pins(ctx):
    bad = []
    pin = json.loads((HERE / "CODE_PIN_AXN.json").read_text())
    for f in N.CODE + N.PINNED_DOCS:
        if N.sha256_file(HERE / f) != pin.get(f):
            bad.append(f)
    for line in (N.YAX / "CODE_PIN.sha256").read_text(encoding="utf-8").splitlines():             # dependencies: YAX and YAXD, read-only
        if line.strip() and not line.startswith("#"):
            sha, rel = line.split(None, 1)
            if N.sha256_file(N.YAX / rel.strip().lstrip("*")) != sha:
                bad.append("YAX/" + rel.strip())
    for f, sha in json.loads((N.YAXD / "CODE_PIN_AXD.json").read_text()).items():
        if N.sha256_file(N.YAXD / f) != sha:
            bad.append("YAXD/" + f)
    if N.sha256_file(A.PREREG) != (N.YAXD / "PREREG_AXD.sha256").read_text().split()[0]:
        bad.append("PREREG_AXD.md")
    ok = {"base_u400": N.sha256_file(C.U400) == C.U400_SHA, "cache_manifest": N.sha256_file(C.CACHE / "manifest.json") == C.CACHE_MANIFEST_SHA, "frame_manifest": N.sha256_file(C.MANIFEST) == C.MANIFEST_SHA,
          "select_K1": N.jload(N.YAX / "run_full" / "stages" / "select.json")["chosen_X1"] == N.RECIPE["name"], "yax_X1_X2_done": (N.YAX / "markers" / "yax_full.DONE").exists(),
          "controls_present": all((N.YAX_EVAL / "eval" / n / "arm.json").exists() for n in N.CONTROL.values()), "p1_json_present": N.P1_JSON.exists()}
    if bad or not all(ok.values()):
        raise RuntimeError(f"REFUSING: pin mismatch {bad}; checks {ok} (code fix = re-pin + AMENDMENT)")
    ctx.log("pins OK")
    return {"pins": ok}


# ------------------------------------------------------------------ gates_axn: sigma 0 == AXON bit for bit, sigma 1 != AXON, determinism, RED arms
def states_equal(pa, pb) -> tuple[bool, int]:
    import torch
    a = torch.load(pa, map_location="cpu", weights_only=False)["state"]
    b = torch.load(pb, map_location="cpu", weights_only=False)["state"]
    if sorted(a) != sorted(b):
        return False, -1
    return all(torch.equal(a[k], b[k]) for k in a), sum(0 if torch.equal(a[k], b[k]) else 1 for k in a)


def comparator_sees_one_ulp(pa) -> bool:
    """positive control for states_equal: one ulp moved in one tensor of the reference must be seen."""
    import torch
    a = torch.load(pa, map_location="cpu", weights_only=False)["state"]
    k = next(k for k, v in a.items() if v.is_floating_point() and v.numel())
    b = {kk: v.clone() for kk, v in a.items()}
    f = b[k].view(-1)
    f[0] = torch.nextafter(f[0], torch.tensor(float("inf"), dtype=f.dtype))
    return not all(torch.equal(a[kk], b[kk]) for kk in a)


def gates_axn(ctx):
    d = ctx.root / "gates_axn"
    fresh(d)
    d.mkdir(parents=True)
    base = ["--cache", C.CACHE, "--manifest", C.MANIFEST, "--ax-policy", "real", "--steps", 5, "--lr", "5e-5", "--seed", 0] + N.MINI_TRAIN
    runs = {}

    def train(name, script, extra):
        out = d / name
        sh_retry(ctx, f"gates_axn_{name}", [N.PY, script, "--out", out] + base + extra, token="ZZAXFT_OK", env=N.CPU_ENV)
        return out / "final.pt"
    runs["axon_a"] = train("axon_a", N.YAX / "ax_ft.py", [])
    runs["axon_b"] = train("axon_b", N.YAX / "ax_ft.py", [])
    runs["sigma0"] = train("sigma0", HERE / "axd_ft.py", ["--ax-noise-sigma", 0.0])
    runs["sigma1"] = train("sigma1", HERE / "axd_ft.py", ["--ax-noise-sigma", 1.0])
    det, _ = states_equal(runs["axon_a"], runs["axon_b"])
    s0, s0n = states_equal(runs["axon_a"], runs["sigma0"])
    s1, s1n = states_equal(runs["axon_a"], runs["sigma1"])
    import torch
    w0 = torch.load(runs["axon_a"], map_location="cpu", weights_only=False)["state"]["ego_enc.0.weight"]
    w1 = torch.load(runs["sigma1"], map_location="cpu", weights_only=False)["state"]["ego_enc.0.weight"]
    ax1 = json.load(open(d / "sigma1" / "axnoise.json"))
    res = {"determinism_AXON_ref_a_eq_b": det, "G_SIGMA0_bit_identical_to_AXON": s0, "sigma0_n_tensors_differing": s0n, "G_NOISE_REACH_sigma1_differs": (not s1) and not torch.equal(w0, w1),
           "sigma1_n_tensors_differing": s1n, "comparator_sees_1ulp(positive control)": comparator_sees_one_ulp(runs["axon_a"]), "sigma1_post_gates": {k: ax1[k] for k in ("G_NOISE_N", "G_NOISE_STD_TRAIN", "G_HONEST_VAL")}}
    red = {}
    for nm, sig, key in (("noise_not_applied", 1.0, "AXN_G_NOISE_STD"), ("noise_scaled", 1.0, "AXN_G_NOISE_STD"), ("sigma0_offset", 0.0, "AXN_G_NOISE_STD"), ("val_noised", 1.0, "AXN_G_HONEST_GATE")):
        out = d / f"RED_{nm}"
        sh_retry(ctx, f"gates_axn_RED_{nm}", [N.PY, HERE / "axd_ft.py", "--out", out, "--ax-noise-sigma", sig, "--axn-regress", nm, "--gates-only"] + base, token="ZZAXFT_GATE_RED", env=N.CPU_ENV, expect_rc=3)
        gj = N.jload(out / "gates.json")
        red[nm] = {"rc": 3, "gate": key, "gate_pass_value": gj[key]["pass"], "RED": gj[key]["pass"] is False and gj["pass"] is False}
    res["RED_arms"] = red
    res["pass"] = bool(det and s0 and res["G_NOISE_REACH_sigma1_differs"] and res["comparator_sees_1ulp(positive control)"] and all(r["RED"] for r in red.values())
                       and all(v["pass"] for v in res["sigma1_post_gates"].values()))
    N.jdump(res, ctx.root / "analysis" / "gates_axn.json")
    if not res["pass"]:
        raise RuntimeError(f"gates_axn RED: {json.dumps(res)[:1500]}")
    return {"pass": True}


# ------------------------------------------------------------------ training / compose / identity (CPU)
def train_seed(ctx, s):
    out = ctx.root / "arms" / f"K1_AXNOISE_s{s}"
    if (out / "final.pt").exists():
        return {"cached": True}
    fresh(out)
    argv = [N.PY, HERE / "axd_ft.py", "--ax-noise-sigma", N.SIGMA, "--cache", C.CACHE, "--manifest", C.MANIFEST, "--ax-policy", "real", "--seed", s,
            "--steps", 4 if ctx.smoke else N.RECIPE["steps"], "--lr", N.RECIPE["lr"], "--out", out] + (N.SMOKE_TRAIN if ctx.smoke else [])
    sh_retry(ctx, f"train_s{s}", argv, token="ZZAXN_OK", env=N.CPU_ENV)
    return {"out": str(out)}


def compose_seed(ctx, s):
    ck = ctx.root / "ckpt" / f"AXN_s{s}.pt"
    fin = ctx.root / "arms" / f"K1_AXNOISE_s{s}" / "final.pt"
    if not ck.exists():
        N.require_disk(ctx.root, ctx.log)
        sh_retry(ctx, f"ckpt_s{s}", [N.PY, N.YAX / "ax_ckpt_tools.py", "build", "--final", fin, "--out", ck], token="ZZAXCKPT_OK", env=N.CPU_ENV)
    sh_retry(ctx, f"identity_s{s}", [N.PY, N.YAX / "ax_ckpt_tools.py", "identity", "--ckpt", ck, "--final", fin, "--out", ctx.root / "identity" / f"AXN_s{s}.json", "--frames", 128 if ctx.smoke else 512],
             token="ZZAXID_OK", env=N.CPU_ENV)
    return {"ckpt": str(ck)}


# ------------------------------------------------------------------ eval: seam (lock, MemoryError retry) -> harness -> families6 -> perwindow
def seam_cmd(ctx, name, ckpt, arms_out, tokens, a0_out, regress):
    return [N.PY, HERE / "axn_seam.py", "--arms-out", arms_out, "--regress", regress, "--", "--ckpt", ckpt, "--frames", C.D / "frames", "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
            "--export", C.export_path(), "--tokens", tokens, "--out", a0_out, "--arm", f"REFe_{name}", "--rule", "navsim_v1_lane", "--device", "cpu" if ctx.smoke else "cuda"]


def run_seam(ctx, label, cmd, arms_out, max_attempts=5, give_up_h=24.0):
    """The memory-retry the brief asks for: up to `max_attempts` runs of the seam; a failure whose tail shows a host-memory / allocation failure is retried (logged to logs/retries.jsonl) after a
    bounded wait for headroom, the GPU lock RELEASED during the wait and re-acquired per attempt (never forced); any other failure raises at once."""
    meta = Path(arms_out) / "seam_meta.json"
    if meta.exists() and json.load(open(meta)).get("rc") == 0:
        return {"skipped": True}
    env = dict(AE.seam_env(), **({"CUDA_VISIBLE_DEVICES": "-1"} if ctx.smoke else {}))
    for k in range(1, max_attempts + 1):
        fresh(meta) if meta.exists() else None
        N.require_disk("C:/", ctx.log, min_gib=2.0, max_wait_s=1800)
        N.wait_mem(ctx.log)
        body = None if ctx.smoke else AE.GL.acquire_blocking(give_up_h * 3600, owner="yaxn_ax_eval", log=ctx.log)
        ctx.log(f"seam {label} attempt {k}: " + (f"GPU lock taken {body}" if body else "CPU, no lock (smoke)"))
        t0 = time.time()
        try:
            rc, txt = N.sh(cmd, ctx.root / "logs" / f"seam_{label}.attempt{k}.log", env=env, cwd=str(C.EVAL))
        finally:
            if body:
                AE.GL.release_gpu_lock(AE.GL.LOCK, body)
        ok = "ZZAXDSEAM OK" in txt[-600:]
        ctx.log(f"seam {label} attempt {k}: rc {rc} ok {ok} in {time.time() - t0:.0f} s" + (" (lock released)" if body else ""))
        if ok:
            return {"attempts": k, "seconds": round(time.time() - t0, 1)}
        if N.is_mem_failure(txt) and k < max_attempts:
            ctx.log(f"RETRY seam {label}: memory failure on attempt {k}; waiting for headroom, then re-queueing on the lock")
            with open(ctx.root / "logs" / "retries.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": N.now(), "stage": f"seam_{label}", "attempt": k, "rc": rc, "tail": txt[-300:]}) + "\n")
            time.sleep(120)
            continue
        raise RuntimeError(f"seam {label} failed (attempt {k}/{max_attempts}, rc {rc}, memory failure: {N.is_mem_failure(txt)}): {txt[-1500:]}")
    raise RuntimeError("unreachable")


def eval_seed(ctx, s, tokens, toks):
    name = f"AXN_s{s}"
    outd = ctx.root / "eval" / name
    outd.mkdir(parents=True, exist_ok=True)
    if (outd / "arm.json").exists():
        return {"cached": True}
    AE.HP.require(C.HARNESS, caller=C.EVAL / "score_navtest_refe.py")
    res = {"name": name, "started": N.now(), "seam": run_seam(ctx, name, seam_cmd(ctx, name, ctx.root / "ckpt" / f"{name}.pt", ctx.root / f"seam_arms_s{s}", tokens, outd / "seam.npz", "none"), ctx.root / f"seam_arms_s{s}")}
    res["score"] = AE.run_score(name, tokens, outd, toks, tag=f"axn{ctx.tag}")
    res["families"] = AE.run_families(name, outd, toks)
    AE.run_perwindow(outd, tokens)
    res["finished"] = N.now()
    N.jdump(res, outd / "arm.json")
    return {"scored": res["score"]["status"].get("status")}


def compose_axon(ctx, s):
    """AMENDMENT_1: re-compose AXON_s from YAX's banked K1 decoder tensors, gated like the AXNOISE checkpoints (K1-K5 at build, identity = G-CKPT)."""
    ck = ctx.root / "ckpt" / f"AXON_s{s}.pt"
    if not ck.exists():
        N.require_disk(ctx.root, ctx.log)
        sh_retry(ctx, f"ckpt_axon_s{s}", [N.PY, N.YAX / "ax_ckpt_tools.py", "build", "--final", N.AXON_FINAL[s], "--out", ck], token="ZZAXCKPT_OK", env=N.CPU_ENV)
    sh_retry(ctx, f"identity_axon_s{s}", [N.PY, N.YAX / "ax_ckpt_tools.py", "identity", "--ckpt", ck, "--final", N.AXON_FINAL[s], "--out", ctx.root / "identity" / f"AXON_s{s}.json", "--frames", 128 if ctx.smoke else 512],
             token="ZZAXID_OK", env=N.CPU_ENV)
    return ck


def axon_repro(a0_npz, s) -> dict:
    """reported, never voiding: does the re-composed AXON_s plan the same as the banked X1_AXON_s seam (same tokens)?"""
    import numpy as np
    z, b = np.load(a0_npz), np.load(N.YAX_EVAL / "eval" / N.CONTROL[s] / "seam.npz")
    bi = {str(t): i for i, t in enumerate(b["token"])}
    d = [float(np.abs(z["poses"][i].astype(np.float64) - b["poses"][bi[str(t)]].astype(np.float64)).max()) for i, t in enumerate(z["token"])]
    return {"n": len(d), "bit_identical_to_banked_X1_AXON_seam": all(x == 0.0 for x in d), "max_abs_m": max(d)}


def ref_axon_seed(ctx, s, tokens):
    """AMENDMENT_1: the in-run gain reference. AXON_s through axn_seam (A0 + IDT + IDTm + IDT gain arms only), same tokens, same lock/retry; no scoring (the PDMS control is the banked X1_AXON_s eval)."""
    name = f"AXON_s{s}"
    arms_out = ctx.root / f"seam_arms_axon_s{s}"
    ad = ctx.root / "ref_axon" / name
    ad.mkdir(parents=True, exist_ok=True)
    res = {}
    done = (arms_out / "seam_meta.json").exists() and json.load(open(arms_out / "seam_meta.json")).get("rc") == 0
    if not done:
        ck = compose_axon(ctx, s)
        res["seam"] = run_seam(ctx, name, seam_cmd(ctx, name, ck, arms_out, tokens, ad / "A0.npz", "none"), arms_out)
        res["ckpt_removed_after_seam"] = N.remove_ckpt(ctx.tag, "AXON", s)
    res["repro"] = axon_repro(ad / "A0.npz", s)
    N.jdump(res["repro"], ad / "repro.json")
    return res


def red_seams(ctx, tokens):
    """smoke only: each seam-level RED arm must turn ITS gate red on the real seam (the P1 gates, now on the AXNOISE checkpoint)."""
    import axd_analyze as AN
    out = {}
    for r in RED_SEAMS:
        rd = ctx.root / "red" / r
        rd.mkdir(parents=True, exist_ok=True)
        run_seam(ctx, f"RED_{r}", seam_cmd(ctx, "AXN_s0", ctx.root / "ckpt" / "AXN_s0.pt", rd / "seam_arms", tokens, rd / "A0.npz", r), rd / "seam_arms")
        rec, _ = AZ.load_records_dir(rd / "seam_arms")
        red = {"transform_not_applied": not AN.gate_reach_idt_gain(rec)["PASS"], "idt_perturbed": not AN.gate_idt(rec)["PASS"], "memo_stale": not AN.gate_memo(rec)["PASS"]}[r]
        out[r] = red
        ctx.log(f"RED seam {r}: its gate reads RED = {red}")
    if not all(out.values()):
        raise RuntimeError(f"a seam RED arm did not turn its gate red: {out}")
    return {"red_seams": out}


def run(a):
    ctx = Ctx(a.tag)
    marker(a.tag, "STARTED", root=str(ctx.root))
    for m in ("DONE", "FAILED"):
        (N.MARKERS / f"axn_{a.tag}.{m}").unlink(missing_ok=True)
    if ctx.smoke:
        tokens_src = SMOKE_TOKENS
    else:
        tokens_src = C.A5_TOKENS
    tokens = ctx.root / "tokens" / "tokens.json"
    if not tokens.exists():
        shutil.copyfile(tokens_src, tokens)
    toks = sorted(C.jload(tokens)["tokens"])
    ctx.log(f"tag {a.tag}: {len(toks)} tokens; E: {N.free_gib(ctx.root):.1f} GiB free, C: {N.free_gib('C:/'):.1f} GiB free")
    N.jdump(check_pins(ctx), ctx.root / "analysis" / "pins.json")                 # never cached: every (re)launch re-checks the pins
    N.require_disk(ctx.root, ctx.log)
    ctx.stage("gates_axn", lambda: gates_axn(ctx))
    for s in N.SEEDS:                                                      # all CPU work first, so the GPU lock wait (hours, possibly) is the last thing the chain does
        ctx.stage(f"train_s{s}", lambda s=s: train_seed(ctx, s))
        ctx.stage(f"compose_s{s}", lambda s=s: compose_seed(ctx, s))
    for s in N.SEEDS:
        ctx.stage(f"eval_s{s}", lambda s=s: eval_seed(ctx, s, tokens, toks))
    for s in N.SEEDS:
        ctx.stage(f"ref_axon_s{s}", lambda s=s: ref_axon_seed(ctx, s, tokens))
    if ctx.smoke:
        ctx.stage("red_seams", lambda: red_seams(ctx, tokens))
    rc, txt = N.sh([N.PY, HERE / "axn_analyze.py", "--root", ctx.root, "--tokens", tokens] + (["--smoke"] if ctx.smoke else []), ctx.root / "logs" / "analyze.log", env={"PYTHONIOENCODING": "utf-8"})
    line = next((l for l in txt.splitlines() if l.startswith("ZZAXNVERDICT")), None)
    if rc != 0 or not line:
        raise RuntimeError(f"analyze failed rc {rc}: {txt[-1500:]}")
    p2 = N.jload(ctx.root / "analysis" / "P2.json")
    ctx.log(line)
    mech = {k: r["gates_red"] for k, r in p2["seeds"].items()}
    if ctx.smoke and any(mech.values()):
        raise RuntimeError(f"smoke: a mechanical gate is red: {mech}")
    for s in N.SEEDS:                                                      # composed checkpoints are 1.29 GB: keep them only for a CANDIDATE (MM needs them for the closed loop); final.pt re-composes any of them
        if ctx.smoke or p2["verdict"] != "CANDIDATE":
            N.remove_ckpt(ctx.tag, "AXN", s)
        N.remove_ckpt(ctx.tag, "AXON", s)
    marker(a.tag, "DONE", verdict=p2["verdict"], per_seed=p2["per_seed_verdict"], gates_red=mech, root=str(ctx.root))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=("smoke", "full"))
    a = ap.parse_args()
    try:
        return run(a)
    except Exception:
        why = traceback.format_exc()[-3500:]
        print(why, flush=True)
        marker(a.tag, "FAILED", why=why)
        return 1


if __name__ == "__main__":
    sys.exit(main())
