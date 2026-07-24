"""TanitEval — runner CLI.

  python -m taniteval.runner run --model refa-dinov2 [--episodes 40]
  python -m taniteval.runner run-all [--episodes 40]
  python -m taniteval.runner ab --a refa-dinov2 --b refa-ijepa
  python -m taniteval.runner regression [--update-golden]
  python -m taniteval.runner driving --model flagship-30k
  python -m taniteval.runner driving-all
  python -m taniteval.runner closedloop --model flagship-30k [--episodes 40]
  python -m taniteval.runner closedloop-all
  python -m taniteval.runner closedloop-report
  python -m taniteval.runner report

This module is THE ONE canonical entrypoint — every eval axis is a subcommand
here (open-loop ADE/miss via `run`, the beyond-ADE TanitEval-v2 suite inline +
`driving`, imagination-in-the-loop `closedloop`, `hierarchy`, `generalize`,
`pathspeed`, `efficiency`). It pins the CLEAN held-out val split and refuses the
leaky one at the data layer (see taniteval.data.list_val_episodes).

Each run writes results/<key>.json (benchmark + the inline `efficiency` and
`driving` blocks) + results/windows_<key>.pt (raw per-window predictions — the
substrate for A/B, viz, and the CPU-only TanitEval v2 tier-0 backfill)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

from taniteval import ab as abmod  # noqa: E402
from taniteval import bench, data, loaders, rollout  # noqa: E402
from taniteval.registry import MODELS  # noqa: E402

RES = Path("/root/taniteval/results")
VAL = f"/root/valdata/{data.CLEAN_VAL}"   # single source of truth (data.py)


def _entry(key):
    e = [m for m in MODELS if m["key"] == key]
    assert e, f"unknown model {key}; known: {[m['key'] for m in MODELS]}"
    return e[0]


def run_one(key, episodes=40, device="cuda"):
    e = _entry(key)
    t0 = time.time()
    L = loaders.load(e, device)
    arch = e["arch"]
    # Direct-trajectory-head arms (own their trajectory surface, no grounded
    # operative rollout): REF-B planner heads + REF-C anchored-diffusion decoder.
    direct_head = arch in ("refb", "refc")
    if not L["traj_capable"] and not direct_head:
        print(f"[run] {key}: no trajectory surface — profiling-only",
              flush=True)
        return None
    files = data.list_val_episodes(VAL, episodes)
    assert files, f"no val episodes under {VAL}"
    # LEAKAGE GUARD: exclude val episodes present in this model's train set.
    if e.get("train_ids"):
        from tanitad.data.mixing import load_episode
        train_ids = set(Path(e["train_ids"]).read_text().split())
        keep = [f for f in files
                if str(load_episode(str(f), mmap=True).episode_id)
                not in train_ids]
        dropped = len(files) - len(keep)
        if dropped:
            print(f"[guard] {key}: DROPPED {dropped}/{len(files)} val eps "
                  f"(train-set leakage); {len(keep)} clean remain", flush=True)
        files = keep
        assert len(files) >= 8, (
            f"only {len(files)} leak-free val eps — refuse a decision-grade "
            f"number; evaluate this model on its own disjoint val instead")
    if L["feed"] == "frames":
        eps = data.load_frames(files)
    else:
        eps = data.load_features(files, L["feed"], device)
    if arch == "refb":
        from taniteval import refb_eval
        win = refb_eval.collect(L["model"], eps, device,
                                speed_input=bool(e.get("speed_input")),
                                yaw_input=bool(e.get("yaw_input")))
    elif arch == "refc":
        from taniteval import refc_eval
        win = refc_eval.collect(L["model"], eps, device,
                                speed_input=bool(e.get("speed_input")),
                                mode=e.get("mode", "diffusion"))
    else:
        win = rollout.collect(L["model"], L["step_readout"], eps, device,
                              speed_input=bool(e.get("speed_input")),
                              yaw_input=bool(e.get("yaw_input")),
                              dyn_input=bool(e.get("dyn_input")))
    res = bench.run(win)
    if win.get("method"):
        res["method"] = win["method"]
    res["model"] = {k: e.get(k) for k in
                    ("key", "name", "arch", "encoder", "speed_input", "hf")}
    res["ckpt_step"] = L["step"]
    # EFFICIENCY IS A DEFAULT AXIS (2026-07-20): every accuracy run also reports
    # what one planning step COSTS — latency/stage-breakdown/FLOPs/memory/params
    # in the SAME results JSON. Cheap (batch 1, fp32, ~10 s) so it never
    # discourages a full eval; never fatal to the accuracy number.
    try:
        from taniteval import efficiency
        res["efficiency"] = efficiency.quick(e, L, eps[0], device)
    except Exception as ex:
        res["efficiency"] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        print(f"[eff] {key}: efficiency panel FAILED: {res['efficiency']['error']}",
              flush=True)
    # DRIVING CAPABILITY IS A DEFAULT AXIS (2026-07-21): ADE is one column, not
    # the verdict. Every accuracy run also reports the TanitEval v2 tier-0 set —
    # cruise quality, transient response, the along/cross split, progress, path
    # geometry, heading by curvature, curvature sign, kinematic strata — each
    # with an episode-cluster bootstrap and a PAIRED test against BOTH trivial
    # floors (CV and hold-v0). CPU-only over the windows already in memory
    # (~2.4 s at B=2000), so it never competes for the GPU and never delays an
    # eval; never fatal to the accuracy number.
    try:
        from taniteval import driving
        res["driving"] = driving.quick(win, arm=key)
    except Exception as ex:
        res["driving"] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        print(f"[driving] {key}: driving panel FAILED: "
              f"{res['driving']['error']}", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    RES.mkdir(parents=True, exist_ok=True)
    rollout.save_windows(win, RES / f"windows_{key}.pt")
    (RES / f"{key}.json").write_text(json.dumps(res, indent=2, default=str))
    eff = res.get("efficiency", {})
    if "plan_step" in eff:
        print(f"[run] {key} efficiency: plan step "
              f"p50={eff['plan_step']['p50_ms']:.2f} ms "
              f"p99={eff['plan_step']['p99_ms']:.2f} ms "
              f"({eff['realtime']['budget_used_pct_p99']:.0f}% of the 100 ms "
              f"budget) · {eff.get('flops', {}).get('gflops', '—')} GFLOPs · "
              f"{eff['memory']['peak_alloc_mb']:.0f} MB peak · "
              f"{eff['params']['total_params_m']:.1f} M params", flush=True)
    dv = res.get("driving", {})
    if "verdict" in dv:
        v, hl = dv["verdict"], dv["headline"]
        print(f"[run] {key} driving (TanitEval v2 tier-0): "
              f"along {hl['long_abs_2s_m']['mean']:.3f} / cross "
              f"{hl['lat_abs_2s_m']['mean']:.3f} m · speed MAE "
              f"{hl['speed_mae_mps']['mean']:.3f} vs hold-v0 "
              f"{dv['floor_values']['holdv0']['speed_mae_mps']['value']:.3f} "
              f"m/s · straight heading "
              f"{dv.get('by_curvature', {}).get('straight', {}).get('model_heading_mae_deg', '—')}° "
              f"· win lives: {v['where_the_win_lives']} · tracks speed > CV: "
              f"{v['tracks_speed_better_than_cv']}", flush=True)
    hm = res["heldout"]["model"]
    print(f"[run] {key} step={L['step']} n={res['n_windows']} "
          f"ade@2s={hm['ade_0_2s']['mean']:.3f}±{hm['ade_0_2s']['ci95']:.3f} "
          f"fde={hm['fde@2s']['mean']:.3f} miss@2m={hm['miss_rate@2m']['mean']:.3f} "
          f"tms={hm['tms_openloop']['mean']:.3f} ({res['wall_s']}s)", flush=True)
    return res


def run_imagination(key, episodes=12, device="cuda"):
    """Imagination panel: isolate the vision/imagination contribution of a
    world-model arm (vision x action ablation + latent fidelity)."""
    from taniteval import imagination
    e = _entry(key)
    L = loaders.load(e, device)
    if not L["traj_capable"]:
        print(f"[imag] {key}: not rollout-capable (planner) — skip", flush=True)
        return None
    files = data.list_val_episodes(VAL, 40)
    if e.get("train_ids"):                              # reuse the leakage guard
        from tanitad.data.mixing import load_episode
        tid = set(Path(e["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    files = files[:episodes]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    res = imagination.run(L["model"], L["step_readout"], eps, device,
                          speed_input=bool(e.get("speed_input")),
                          yaw_input=bool(e.get("yaw_input")),
                          dyn_input=bool(e.get("dyn_input")))
    res["model"] = {k: e.get(k) for k in ("key", "name", "arch", "encoder")}
    res["ckpt_step"] = L["step"]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"imag_{key}.json").write_text(json.dumps(res, indent=2))
    lf = res["latent_fidelity"]
    print(f"[imag] {key} step={L['step']}: vision_use={res['vision_use_pct']}% "
          f"imagination={res['imagination_pct']}% "
          f"latent={lf['real'] if lf else 'n/a'} "
          f"(Δ{lf['vision_gain'] if lf else '—'}) — {res['verdict']}",
          flush=True)
    return res


def run_hierarchy(key, episodes=40, device="cuda", stride=8):
    """Hierarchy panel (H26): is the operative->tactical->strategic conditioning
    cascade load-bearing (each layer's conditioning helps the layer it
    conditions, layers cohere) or decorative? Cross-layer ablation + consistency
    + H18 grounded-vs-ungrounded. Needs a trained 4-brain arm (flagship/REF-A)."""
    from taniteval import hierarchy
    e = _entry(key)
    L = loaders.load(e, device)
    # G1 (2026-07-20): this guard used to skip SILENTLY, which is why REF-B and
    # REF-C have no H26 numbers — and H26 is the program's core-goal proof. A
    # future arm whose tactical brain is not literally `tactical_policy` (v3.5
    # plans an `AnchoredDiffusionDecoder`) would be skipped the same way and the
    # missing panel would look like a passing one. Fail LOUD and name the brain.
    _model = L["model"]
    if not L["traj_capable"] or getattr(_model, "tactical_policy", None) is None:
        _known = [n for n in ("tactical_policy", "tactical_pred", "decoder",
                              "planner", "tactical") if getattr(_model, n, None)
                  is not None]
        _brains = ", ".join(f"{n}={type(getattr(_model, n)).__name__}"
                            for n in _known) or "none found"
        print(f"[hier] {key}: SKIPPED — no H26 hierarchy numbers for this arm.\n"
              f"[hier]   reason: traj_capable={L['traj_capable']}, "
              f"tactical_policy={'present' if getattr(_model, 'tactical_policy', None) is not None else 'ABSENT'}\n"
              f"[hier]   tactical-ish attributes on {type(_model).__name__}: {_brains}\n"
              f"[hier]   ⚠️  A SKIP IS NOT A PASS. `hierarchy.py` currently supports only\n"
              f"[hier]       (tactical_policy + strategic_policy) arms — see hierarchy.py:225.\n"
              f"[hier]       Generalising it is REQUIRED before v3.5's Gate H can run.",
              flush=True)
        return None
    files = data.list_val_episodes(VAL, 40)
    if e.get("train_ids"):                              # reuse the leakage guard
        from tanitad.data.mixing import load_episode
        tid = set(Path(e["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    files = files[:episodes]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    res = hierarchy.run(L["model"], L["step_readout"], eps, device,
                        speed_input=bool(e.get("speed_input")), max_eps=episodes,
                        stride=stride, yaw_input=bool(e.get("yaw_input")),
                        dyn_input=bool(e.get("dyn_input")))
    res["model"] = {k: e.get(k) for k in ("key", "name", "arch", "encoder")}
    res["ckpt_step"] = L["step"]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"hier_{key}.json").write_text(json.dumps(res, indent=2, default=str))
    if res.get("skipped"):
        print(f"[hier] {key}: skipped ({res['skipped']})", flush=True)
        return res
    th = res["thesis_read"]["A_conditioning_helps_conditioned_layer"]
    print(f"[hier] {key} step={L['step']} n={res['n_windows']}: "
          f"{th['n_of_3_seams_beneficial']}/3 seams beneficial — {th['verdict']} | "
          f"man~traj κ={res['consistency']['maneuver_vs_trajectory']['kappa']} | "
          f"H18 grounded {res['h18_grounded_vs_ungrounded']['grounded_op_rollout_ade_2s']}"
          f"/{res['h18_grounded_vs_ungrounded']['ungrounded_tactical_head_ade_2s']}m",
          flush=True)
    return res


def run_ab(a, b):
    wa = rollout.load_windows(RES / f"windows_{a}.pt")
    wb = rollout.load_windows(RES / f"windows_{b}.pt")
    r = abmod.compare(wa, wb, a, b)
    (RES / f"ab_{a}_vs_{b}.json").write_text(json.dumps(r, indent=2))
    print(f"[ab] {a} {r['ade_a']:.3f} vs {b} {r['ade_b']:.3f} | "
          f"B-win {r['win_rate_b']:.1%} | dCI {r['delta_ci95']} | "
          f"verdict: {r['verdict']}", flush=True)
    return r


def regression(update=False, tol_frac=0.08):
    """Golden-value regression: every stored result within tol of golden."""
    gpath = RES / "golden.json"
    keys = ["ade_0_2s", "fde@2s", "miss_rate@2m"]
    current = {}
    for f in RES.glob("*.json"):
        if f.name.startswith(("ab_", "golden", "run_")):
            continue
        d = json.loads(f.read_text())
        if "heldout" in d:
            current[f.stem] = {k: d["heldout"]["model"][k]["mean"] for k in keys}
    if update or not gpath.exists():
        gpath.write_text(json.dumps(current, indent=2))
        print(f"[regression] golden updated ({len(current)} models)", flush=True)
        return True
    golden, ok = json.loads(gpath.read_text()), True
    for mk, vals in golden.items():
        for k, gv in vals.items():
            cv = current.get(mk, {}).get(k)
            if cv is None:
                print(f"[regression] MISSING {mk}.{k}", flush=True); ok = False
            elif cv > gv * (1 + tol_frac):
                print(f"[regression] REGRESSED {mk}.{k}: {cv:.4f} > golden "
                      f"{gv:.4f} (+{tol_frac:.0%})", flush=True); ok = False
    print(f"[regression] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser("taniteval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True)
    r.add_argument("--episodes", type=int, default=40)
    ra = sub.add_parser("run-all"); ra.add_argument("--episodes", type=int,
                                                    default=40)
    c = sub.add_parser("ab"); c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    im = sub.add_parser("imagination"); im.add_argument("--model", required=True)
    im.add_argument("--episodes", type=int, default=12)
    ima = sub.add_parser("imag-all"); ima.add_argument("--episodes", type=int,
                                                       default=12)
    hi = sub.add_parser("hierarchy"); hi.add_argument("--model", required=True)
    hi.add_argument("--episodes", type=int, default=40)
    hi.add_argument("--stride", type=int, default=8)
    hia = sub.add_parser("hier-all"); hia.add_argument("--episodes", type=int,
                                                       default=40)
    g = sub.add_parser("regression"); g.add_argument("--update-golden",
                                                     action="store_true")
    ge = sub.add_parser("generalize"); ge.add_argument("--model", required=True)
    ge.add_argument("--episodes", type=int, default=40)
    ge.add_argument("--corpus", default="physicalai")
    gea = sub.add_parser("gen-all"); gea.add_argument("--episodes", type=int,
                                                      default=40)
    # NEW panel (adf2): decoupled longitudinal/lateral planning-quality metrics
    # — thin dispatch only; all logic lives in taniteval.pathspeed (own module).
    ps = sub.add_parser("pathspeed"); ps.add_argument("--model", required=True)
    ps.add_argument("--episodes", type=int, default=40)
    psa = sub.add_parser("pathspeed-all"); psa.add_argument("--episodes",
                                                            type=int, default=40)
    # NEW panel (2026-07-20): inference efficiency — the DEPLOYMENT axis.
    # `run` already emits a cheap batch-1 fp32 read into results/<key>.json;
    # these commands are the FULL version (precision sweep + throughput).
    ef = sub.add_parser("efficiency"); ef.add_argument("--model", required=True)
    ef.add_argument("--precision", default="fp32",
                    help="comma list: fp32,tf32,amp16 (applied IDENTICALLY to "
                         "every arm — never let it drift between arms)")
    ef.add_argument("--batch", type=int, default=1)
    ef.add_argument("--iters", type=int, default=200)
    ef.add_argument("--warmup", type=int, default=30)
    ef.add_argument("--no-throughput", action="store_true")
    efa = sub.add_parser("eff-all")
    efa.add_argument("--precision", default="fp32")
    efa.add_argument("--batch", type=int, default=1)
    efa.add_argument("--iters", type=int, default=200)
    efa.add_argument("--warmup", type=int, default=30)
    efa.add_argument("--no-throughput", action="store_true")
    # NEW panel (2026-07-21): DRIVING CAPABILITY — TanitEval v2 tier-0. Emitted
    # inline by every `run`; these commands recompute it OFFLINE from the
    # persisted windows_<key>.pt (CPU-only, no GPU, no model load), which is why
    # the backfill can populate the whole leaderboard without touching a pod.
    dr = sub.add_parser("driving"); dr.add_argument("--model", required=True)
    dr.add_argument("--n-boot", type=int, default=2000)
    dra = sub.add_parser("driving-all")
    dra.add_argument("--n-boot", type=int, default=2000)
    # CLOSED-LOOP: imagination-in-the-loop rollout (the open-loop-ADE-does-not-
    # predict-closed-loop axis). Thin dispatch — logic lives in
    # taniteval.closedloop (own module, own clean-split leak guard).
    cl = sub.add_parser("closedloop"); cl.add_argument("--model", required=True)
    cl.add_argument("--episodes", type=int, default=40)
    cla = sub.add_parser("closedloop-all")
    cla.add_argument("--episodes", type=int, default=40)
    sub.add_parser("closedloop-report")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        run_one(a.model, a.episodes)
    elif a.cmd == "run-all":
        for m in MODELS:
            try:
                run_one(m["key"], a.episodes)
            except Exception as e:
                print(f"[run-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "ab":
        run_ab(a.a, a.b)
    elif a.cmd == "imagination":
        run_imagination(a.model, a.episodes)
    elif a.cmd == "imag-all":
        for m in MODELS:
            try:
                run_imagination(m["key"], a.episodes)
            except Exception as e:
                print(f"[imag-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "hierarchy":
        run_hierarchy(a.model, a.episodes, stride=a.stride)
    elif a.cmd == "hier-all":
        for m in MODELS:
            try:
                run_hierarchy(m["key"], a.episodes)
            except Exception as e:
                print(f"[hier-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "regression":
        ok = regression(update=a.update_golden)
        sys.exit(0 if ok else 1)
    elif a.cmd == "generalize":
        from taniteval import generalization
        generalization.run_and_save(a.model, episodes=a.episodes,
                                    corpus=a.corpus)
    elif a.cmd == "gen-all":
        from taniteval import generalization
        for m in MODELS:
            try:
                generalization.run_and_save(m["key"], episodes=a.episodes)
            except Exception as e:
                print(f"[gen-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "pathspeed":
        from taniteval import pathspeed
        pathspeed.run_and_save(a.model, episodes=a.episodes)
    elif a.cmd == "pathspeed-all":
        from taniteval import pathspeed
        for m in MODELS:
            try:
                pathspeed.run_and_save(m["key"], episodes=a.episodes)
            except Exception as e:
                print(f"[pathspeed-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "efficiency":
        from taniteval import efficiency
        efficiency.run_and_save(
            a.model, precisions=tuple(p.strip()
                                      for p in a.precision.split(",")),
            batch=a.batch, iters=a.iters, warmup=a.warmup,
            throughput=not a.no_throughput)
    elif a.cmd == "eff-all":
        from taniteval import efficiency
        precs = tuple(p.strip() for p in a.precision.split(","))
        for m in MODELS:
            try:
                efficiency.run_and_save(m["key"], precisions=precs,
                                        batch=a.batch, iters=a.iters,
                                        warmup=a.warmup,
                                        throughput=not a.no_throughput)
            except Exception as e:
                print(f"[eff-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "driving":
        from taniteval import driving
        driving.run_and_save(a.model, res_dir=RES, n_boot=a.n_boot)
    elif a.cmd == "driving-all":
        from taniteval import driving
        driving.run_all(RES, n_boot=a.n_boot)
    elif a.cmd == "closedloop":
        from taniteval import closedloop
        closedloop.run_and_save(a.model, episodes=a.episodes)
    elif a.cmd == "closedloop-all":
        from taniteval import closedloop
        for m in MODELS:
            try:
                closedloop.run_and_save(m["key"], episodes=a.episodes)
            except Exception as e:
                print(f"[closedloop-all] {m['key']} FAILED: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    elif a.cmd == "closedloop-report":
        import closedloop_report
        closedloop_report.main()
    elif a.cmd == "report":
        from taniteval import report
        print(report.build(RES), flush=True)


if __name__ == "__main__":
    main()
