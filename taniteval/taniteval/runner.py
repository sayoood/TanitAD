"""TanitEval — runner CLI.

  python -m taniteval.runner run --model refa-dinov2 [--episodes 40]
  python -m taniteval.runner run-all [--episodes 40]
  python -m taniteval.runner ab --a refa-dinov2 --b refa-ijepa
  python -m taniteval.runner regression [--update-golden]
  python -m taniteval.runner report

Each run writes results/<key>.json (benchmark) + results/windows_<key>.pt
(raw per-window predictions — the substrate for A/B and viz)."""
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
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"


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
    res["wall_s"] = round(time.time() - t0, 1)
    RES.mkdir(parents=True, exist_ok=True)
    rollout.save_windows(win, RES / f"windows_{key}.pt")
    (RES / f"{key}.json").write_text(json.dumps(res, indent=2, default=str))
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
    if not L["traj_capable"] or getattr(L["model"], "tactical_policy", None) is None:
        print(f"[hier] {key}: not a trained 4-brain arm (no policy/rollout) — skip",
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
    elif a.cmd == "report":
        from taniteval import report
        print(report.build(RES), flush=True)


if __name__ == "__main__":
    main()
