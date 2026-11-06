"""Entry for the T2 re-perception rollout of the v8F-mini planner (AGENT C). `t2a_run.py` pattern: register the arms, replace the kit's `run_rollout` by a
dispatcher, then run the kit's `closedloop_drive.main()`.

    python t2v8f_run.py --scene-dir S --arm C0_s0 --condition objects --t2-cond T2 --starts 60,80 --out O --t2-scene-idx 0 --t2-scene-id <id>
    python t2v8f_run.py matrix --scene-dir S --scene-idx 0 --scene-id <id> --arms A,B --t2-conds T2,RR --agents objects,empty --starts 60 --out O [--inproc]
    python t2v8f_run.py score --root O --out SCORE.json [--tracks 0=/path/sequence_tracks.json]      (dev box: runs the scorer in its OWN process)

Arms (card section 3): C0_s0 C0_s1 DAGOACC_s0 DAGOACC_s1 P10 ZERO BLIND. Agents (the kit's --condition): objects | empty. T2 condition (--t2-cond): T2 | RR | logged.
Every rollout is `t2v8f_exec.run_t2_rollout` (PLAYBACK executor); the record is JSON-safe in the scorer's input format (`t2v8f_lead.rows_for_records`).
Environment: T2A_KIT (the T2-A kit: tree/ + t2a/), T2V8F_TREE (the tip tree), T2V8F_V9_DIR, T2V8F_TOOL, T2A_STUB=1 (CPU stub renderer, forces CPU).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if os.environ.get("T2A_STUB") == "1" or os.environ.get("T2A_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ARMS = ("C0_s0", "C0_s1", "DAGOACC_s0", "DAGOACC_s1", "P10", "ZERO", "BLIND")
CKPT_FILE = {"C0_s0": "C0_s0.pt", "C0_s1": "C0_s1.pt", "DAGOACC_s0": "DAGOACC_s0.pt", "DAGOACC_s1": "DAGOACC_s1.pt", "P10": "C0_s0.pt", "ZERO": "C0_s0.pt", "BLIND": "C0_s0.pt"}
TIER = "T2 re-perception rollout (never pooled with T1-kin)"
PLUMBING = "PLUMBING ONLY: this rollout is never read for H1 or H2"
STATE: dict = {}


def kit_dir():
    return os.environ.get("T2A_KIT", "C:/Users/Admin/t2a_scratch/box2_kit").replace("\\", "/")


def setup():
    """Tip tree first on sys.path (asserts `tanitad` resolves inside it), then the kit's renderer code appended AFTER it, then the kit module and the dispatcher."""
    if "cd" in STATE:
        return STATE["cd"]
    sys.path.insert(0, HERE)
    import t2a_v8f_policy as P
    P.bootstrap(os.environ.get("T2V8F_TREE"))
    kit = kit_dir()
    for p in (f"{kit}/tree/stack/experiments/alpasim-gsplat", f"{kit}/tree/stack/experiments/nurec-gsplat", f"{kit}/tree/taniteval/tools", f"{kit}/t2a"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    import closedloop_drive as cd
    if not hasattr(cd, "EXTRA_ARMS"):
        raise SystemExit("[t2v8f_run] closedloop_drive.py lacks EXTRA_ARMS: the kit overlay was not applied")
    import t2v8f_exec as X
    import t2v8f_kit as K
    if os.environ.get("T2V8F_TOOL"):
        X.TOOL = os.environ["T2V8F_TOOL"]                       # box 2: the tool ships in the package; the sha256 pin is unchanged
    if os.environ.get("T2V8F_V9_DIR"):
        K.V9_DIR = os.environ["T2V8F_V9_DIR"]
    STATE.update(cd=cd, P=P, X=X, K=K, orig=cd.run_rollout, nav={})
    cd.run_rollout = dispatch
    for a in ARMS:
        cd.EXTRA_ARMS[a] = {"build": build_arm, "needs_ckpt": False, "add_args": add_args}
    if os.environ.get("T2A_STUB") == "1":
        import stub_gsplat_renderer
        stub_gsplat_renderer.install()
        fake_attach()
    else:
        try:
            import cv2  # noqa: F401
        except ImportError:                                      # box 2 has no OpenCV: the objects path (falsify_actors) gets the numpy / PIL / PyAV helpers
            import gsplat_renderer
            import t2v8f_nocv2 as N
            STATE["nocv2"] = N.install()
    return cd


def fake_attach():
    """CPU stub only: the stub renderer has no scene and no actors, so `--condition objects` attaches nothing (the rehearsal covers the control flow, not the render)."""
    import types
    m = types.ModuleType("actor_map")
    m.attach_actors_verified = lambda r, sd: {"verdict": "ACCEPTED", "n_mapped": 0, "stub": True}
    sys.modules["actor_map"] = m


def add_args(ap):
    g = ap.add_argument_group("t2v8f")
    g.add_argument("--t2-cond", default="T2", choices=("T2", "RR", "logged"))
    g.add_argument("--t2-ckpt-root", default=None, help="dir holding C0_s0.pt C0_s1.pt DAGOACC_s0.pt DAGOACC_s1.pt (default: the artifact tree via t2v8f_kit.arm_ckpts)")
    g.add_argument("--t2-scene-idx", type=int, default=0)
    g.add_argument("--t2-scene-id", default=None)
    g.add_argument("--t2-encoder", default="dinov3", choices=("dinov3", "stub"))
    g.add_argument("--t2-hub", default=None, help="HF hub dir holding models--facebook--dinov3-vitl16-pretrain-lvd1689m")
    g.add_argument("--t2-device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")
    g.add_argument("--t2-schedule", default="periodic", choices=("periodic", "fresh"), help="the T1-kin tool's default is periodic")
    g.add_argument("--t2-blind-mean", default=None, help="torch file holding the fp16 [40,1024] BLIND mean")
    g.add_argument("--t2-plumbing", action="store_true", help="stamp every record: plumbing only")


def resolve_ckpt(arm, root):
    if root:
        return os.path.join(root, CKPT_FILE[arm]).replace("\\", "/")
    p = STATE["K"].arm_ckpts()[arm]
    if p is None:
        raise SystemExit(f"arm {arm}: its checkpoint is absent (beside, not built)")
    return p


class EntryPolicy:
    """Built lazily per scene: the nav table is a function of the scene's logged route, so the dispatcher installs it. Delegates to a NavPolicy."""

    def __new__(cls, *a, **k):
        K = STATE["K"]

        class _E(K.NavPolicy):
            T2V8F = True

            @property
            def f_eff(self):
                e = getattr(self.inner.inner if hasattr(self.inner, "inner") else self.inner, "encoder", None)
                return None if e is None else (getattr(e, "canon", {}) or {}).get("f_eff")
        return _E(*a, **k)


def build_arm(args):
    import numpy as np
    import torch
    P, K = STATE["P"], STATE["K"]
    arm = args.arm
    STATE["args"] = args
    enc = None
    if arm != "BLIND":
        if args.t2_encoder == "stub":
            from t2a_v8f_encoder import StubEncoder
            enc = StubEncoder()
        else:
            import t2a_v8f_encoder as E
            hub = args.t2_hub or os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", "D:/tanitad/hf"), "hub")
            cands = [hub, os.environ.get("HF_HOME", "")]
            snap = None
            for h in cands:
                try:
                    snap = E.find_local_snapshot(h)
                    break
                except Exception:
                    continue
            if snap is None:
                raise SystemExit(f"DINOv3-L snapshot not found under {cands}")
            enc = E.InLoopEncoder(snap, "bf16", args.t2_device)
    inner = P.V8FMiniT2Policy(resolve_ckpt(arm, args.t2_ckpt_root), encoder=enc, arm="p10" if arm == "P10" else "model", schedule=args.t2_schedule,
                              device=args.t2_device)
    if arm == "ZERO":
        inner = K.ZeroPolicy(inner)
    if arm == "BLIND":
        if not args.t2_blind_mean:
            raise SystemExit("BLIND needs --t2-blind-mean")
        inner = K.BlindPolicy(inner, torch.load(args.t2_blind_mean, map_location="cpu"))
    pol = EntryPolicy(inner, np.zeros((0, 5), np.float32))
    pol.name = arm
    return pol


def json_safe(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def dispatch(transport, renderer, policy, intr, start_frame, n_steps, gt_T, gt_ts_us, **kw):
    """The kit's `run_rollout` slot. v8F arms -> `run_t2_rollout` (nav from the builder's own functions on the scene's logged route); any other policy -> the kit's original."""
    import numpy as np
    if not getattr(policy, "T2V8F", False):
        return STATE["orig"](transport, renderer, policy, intr, start_frame, n_steps, gt_T, gt_ts_us, **kw)
    X, K, a = STATE["X"], STATE["K"], STATE["args"]
    gt = [np.asarray(T, np.float64) for T in gt_T]
    ts = np.asarray(gt_ts_us, np.float64)
    idx = int(a.t2_scene_idx)
    if idx not in STATE["nav"]:
        STATE["nav"][idx] = K.nav_table(gt, ts)[0]
        np.savez(os.path.join(a.out, f"scene_{idx}.npz"), gt_T=np.stack(gt), gt_ts_us=ts)
    table = STATE["nav"][idx]
    policy.table = np.asarray(table, np.float32)
    f0 = int(start_frame)
    if f0 + 60 >= len(gt):
        raise ValueError(f"start {f0} + 60 ticks does not fit the scene ({len(gt)} ticks)")
    Ts = renderer.cam.T_sensor_rig
    md5s = []                                                    # md5 of every rendered frame, in call order: the T5 determinism gate compares two runs on it

    def render(T, t):
        if a.arm == "BLIND":
            return None
        img = transport.render(T @ Ts, t)
        md5s.append(hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest())
        return img
    rec = X.run_t2_rollout(policy, render, gt, ts, f0, intr, nav_row=table[f0], condition=a.t2_cond)
    calls = [{"phase": c["phase"], "tau": c.get("tau"), "t_us": c["t_us"], "x": float(c["pose"][0, 3]), "y": float(c["pose"][1, 3]), "z": float(c["pose"][2, 3]),
              "yaw": X.yaw_of(c["pose"])} for c in rec["calls"]]
    out = {k: rec[k] for k in ("f0", "t0_step", "raw_off", "plans", "extra", "track", "cl", "g", "v0", "policy_track_equals_executor")}
    out.update({"scene": idx, "scene_id": a.t2_scene_id, "arm": a.arm, "tcond": a.t2_cond, "agents": a.condition, "tier": TIER, "calls": calls, "n_renders": len(calls),
                "start_frame": f0, "n_steps": int(n_steps), "steps": [], "render_ms": float(np.mean(transport.render_ms[-len(calls):])) if transport.render_ms else 0.0,
                "frame_md5": md5s, "schedule": a.t2_schedule, "encoder": a.t2_encoder, "purpose": PLUMBING if a.t2_plumbing else "deciding candidate"})
    out = json_safe(out)
    with open(os.path.join(a.out, "records.jsonl"), "a", encoding="utf-8", newline="\n") as f:      # banked incrementally, one start per line
        f.write(json.dumps(out) + "\n")
    return out


def rollout_main(argv):
    cd = setup()
    old = sys.argv
    sys.argv = ["t2v8f_run.py"] + list(argv)
    try:
        cd.main()
    finally:
        sys.argv = old


def matrix_main(argv):
    """One rollout set per (agents, T2 condition, arm) in its own out dir with a DONE / FAIL marker (a subprocess each, so VRAM is released; --inproc for the CPU rehearsal)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-idx", type=int, required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--t2-conds", default="T2,RR")
    ap.add_argument("--agents", default="objects")
    ap.add_argument("--starts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inproc", action="store_true")
    ap.add_argument("--timeout", type=float, default=7200.0)
    a, rest = ap.parse_known_args(argv)
    bad = []
    for ag in a.agents.split(","):
        for tc in a.t2_conds.split(","):
            for arm in a.arms.split(","):
                d = f"{a.out}/{a.scene_id[:8]}/{ag}/{tc}/{arm}"
                os.makedirs(d, exist_ok=True)
                if os.path.exists(d + "/DONE"):
                    continue
                for m in ("FAIL",):
                    if os.path.exists(f"{d}/{m}"):
                        os.replace(f"{d}/{m}", f"{d}/{m}.prev")
                av = ["--scene-dir", a.scene_dir, "--arm", arm, "--condition", ag, "--starts", a.starts, "--steps", "60", "--out", d, "--t2-cond", tc, "--t2-scene-idx",
                      str(a.scene_idx), "--t2-scene-id", a.scene_id] + rest
                try:
                    if a.inproc:
                        STATE.pop("nav", None)
                        STATE["nav"] = {}
                        rollout_main(av)
                        rc = 0
                    else:
                        rc = subprocess.run([sys.executable, "-I", f"{HERE}/t2v8f_run.py"] + av, timeout=a.timeout).returncode
                except SystemExit as e:
                    rc = 1 if e.code else 0
                except Exception as e:                     # noqa: BLE001
                    rc = 1
                    open(f"{d}/error.txt", "w", encoding="utf-8").write(repr(e))
                n = sum(1 for _ in open(f"{d}/records.jsonl")) if os.path.exists(f"{d}/records.jsonl") else 0
                want = len(a.starts.split(","))
                ok = rc == 0 and n == want
                open(f"{d}/{'DONE' if ok else 'FAIL'}", "w").write(f"rc={rc} records={n}/{want}\n")
                if not ok:
                    bad.append(d)
    print(json.dumps({"failed": bad}))
    return 1 if bad else 0


def score_main(argv):
    """The scorer in its OWN process (it binds the reader tree): reads every records.jsonl under --root, scores each (arm, T2 cond, agents) group with the reader's own rows,
    and writes the summary. Contrasts X1 / X2 / BLIND are computed when the groups exist; with the 2-scene stage 1 they are plumbing, not a read."""
    import argparse
    import glob
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracks", action="append", default=[], help="IDX=path to sequence_tracks.json")
    a = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    import t2v8f_lead as L
    import t2v8f_score as S
    groups, scene_npz = {}, {}
    for f in sorted(glob.glob(f"{a.root}/**/records.jsonl", recursive=True)):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            groups.setdefault((r["arm"], r["tcond"], r["agents"]), []).append(r)
        for z in glob.glob(os.path.dirname(f) + "/scene_*.npz"):
            scene_npz[int(os.path.basename(z)[6:-4])] = z
    tr = {int(k.split("=")[0]): k.split("=", 1)[1] for k in a.tracks}
    scenes = {i: L.SceneLog(np.load(z)["gt_T"], np.load(z)["gt_ts_us"], L.load_tracks(tr[i]) if i in tr else None) for i, z in scene_npz.items()}
    res, rows_by = {"tier": S.TIER, "groups": {}, "contrasts": {}}, {}
    for key, recs in sorted(groups.items()):
        recs.sort(key=lambda r: (r["scene"], r["f0"]))
        rows, missing, info = L.rows_for_records(recs, scenes, "|".join(key))
        rows_by[key] = (rows, info)
        res["groups"]["|".join(key)] = {"n_starts": len(recs), "mean": {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in rows.items()},
                                        "n_valid": S.n_valid(rows), "missing": missing, "pair_keys": info["keys"], "purpose": sorted({r["purpose"] for r in recs})}
    for s in (0, 1):
        k_t2, k_p10, k_rr, k_bl = (f"C0_s{s}", "T2", "objects"), ("P10", "T2", "objects"), (f"C0_s{s}", "RR", "objects"), ("BLIND", "T2", "objects")
        if k_t2 in rows_by and k_p10 in rows_by and k_rr in rows_by:
            try:
                S.pair_check(rows_by[k_t2][1]["keys"], rows_by[k_p10][1]["keys"])
                S.pair_check(rows_by[k_t2][1]["keys"], rows_by[k_rr][1]["keys"])
                res["contrasts"][f"seed{s}"] = json_safe(S.x_contrasts(rows_by[k_t2][0], rows_by[k_p10][0], rows_by[k_rr][0], rows_by[k_t2][1]["eid"],
                                                                      rows_blind=rows_by[k_bl][0] if k_bl in rows_by else None))
            except Exception as e:                       # noqa: BLE001
                res["contrasts"][f"seed{s}"] = {"error": repr(e)[:300]}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({"groups": len(res["groups"]), "contrasts": list(res["contrasts"])}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "matrix":
        sys.exit(matrix_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score_main(sys.argv[2:])
    else:
        rollout_main(sys.argv[1:])
