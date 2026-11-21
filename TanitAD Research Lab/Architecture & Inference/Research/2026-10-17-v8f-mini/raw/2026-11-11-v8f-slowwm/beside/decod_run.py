"""SLOWWM beside B1 (decodability) runner, one spec per arm like ``read/ia0_slowwm.py``:
  python decod_run.py [--smoke] [--force] [--train-dir DIR] [--out DIR] "<arm>|<run_dir>|<basis_npz>|<ubasis_npz>" ...
  -> <out>/decod_<arm>.json (resumable per arm; ``smoke_`` prefix with --smoke) and <out>/DECOD_B1.json (every arm read so far). Default <out> = $BESIDE_OUT.
Arms: R35H_s0 / R35H_s1 (adapter: basis and ubasis empty), SLOW_s0 / SLOW_s1 / SLOW_rep, PCA-B_s0 / PCA-B_s1. Tier T0, BESIDE (no bar).

Eval side: ``lp_slowwm.prepare_arm_read`` (every refusal of the read package). Train side: the WMACT TRAIN clip set (``wmact/latents_train.pt`` windows: 141 clips x 17 starts) through the SAME route
(``basis_clip_z1`` + ``lp_lib.build_ctx``) on per-frame inputs of those train clips: ``--train-dir`` holds ``frames_train_s{0,1}.npz`` (``decod_build_train.py``) and, for a basis arm, ``ubasis_train_{slow,pca}.npz``.
The train clips live on box 2: this runner never reaches there. Without ``--train-dir`` only ``--smoke`` runs, on an episode-disjoint half split of the eval windows (machinery check, NOT the protocol).
A non-smoke row is written only when ``CONTROLS_B1B2.json`` says all_pass and (i), (ii) and their red arms also read right on this arm's own arrays."""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decod_lib as DL  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

S, LR, LP, SH, P = DL.S, DL.LR, DL.LP, DL.SH, DL.P


def train_ctx(name, run_dir, basis, ubasis, train_dir):
    """The train-clip window set of one arm through the read package's route. Raises on any mismatch (adapter hash, basis kind / sha, clip list)."""
    L = S.load_checked(name, run_dir, basis, ubasis)
    fr = LR.load_eval_frames(f"{train_dir}/frames_train_s{L['frames_seed']}.npz")
    if fr["adapter_sha"] != L["adapter_sha"]:
        raise LR.LadderMismatch(f"{name}: train frames adapter hash {fr['adapter_sha'][:12]} != the arm's {L['adapter_sha'][:12]}")
    if L["kind"] != "adapter":
        SH.attach_basis_frames(fr, L["B"], f"{train_dir}/ubasis_train_{L['kind']}.npz")
    ev_tr = P.load_latents(LP.TRAIN, "M2b_s0")
    missing = sorted(set(ev_tr["eid"]) - set(fr["clips"]))
    if missing:
        raise LR.LadderMismatch(f"{name}: {len(missing)} WMACT train clips are not in the train frames file")
    with S.basis_clip_z1():
        ctx = LP.build_ctx(L["model"].wm, fr, name, ev_tr, ev_tr, LP.td3_defined(L["built"]))
    return ctx


def read_arm(name, run_dir, basis=None, ubasis=None, smoke=False, train_dir=None):
    t0 = time.time()
    DL.wait_ram()
    k, tracks = DL.k_steps(), DL.load_tracks()
    ctx_ev = S.prepare_arm_read(name, run_dir, basis, ubasis)
    ev = DL.arrays_from_ctx(ctx_ev, tracks, k)
    if train_dir:
        tr = DL.arrays_from_ctx(train_ctx(name, run_dir, basis, ubasis, train_dir), tracks, k)
        src = f"WMACT train clip set (latents_train.pt windows), frames from {train_dir}"
    elif smoke:
        tr, ev = DL.pseudo_split(ev)
        src = "SMOKE: episode-disjoint half split of the eval-141 windows (not the protocol's train set)"
    else:
        raise RuntimeError("no --train-dir: only --smoke may run without the train clips")
    ctl = DL.controls_on_arrays(tr, ev)
    if not ctl["pass"] and not smoke:
        raise RuntimeError(f"{name}: B1 controls (i)/(ii) or their red arms did not read right on this arm's arrays: {json.dumps(ctl)[:800]}")
    rows = DL.b1_rows(tr, ev)
    return {"protocol": "B1 decodability (BESIDE_RULES_SLOWWM.md)", "tier": "T0", "role": "BESIDE, no bar", "name": name, "basis_kind": ctx_ev.basis_kind, "basis_sha256": ctx_ev.basis_sha or None, "smoke": bool(smoke),
            "latent": f"level-1 z1 rolled from the c1 context under the logged actions, step {k} (= 2.0 s)", "k_steps": k, "train_source": src, "targets": rows, "controls_on_this_arm": ctl,
            "controls_file_all_pass": DL.controls_passed(), "probe": {"ridge_alpha_grid": "10**linspace(-3,3,13)", "cv": "5-fold grouped by clip on train", "standardise": "train mean/std", "scored": "once on eval-141"},
            "bootstrap": {"estimator": "episode-cluster bootstrap over eval-141 episodes, probe held fixed (answers 'another draw of episodes' only)", "B": DL.B_BOOT, "seed": DL.SEED},
            "yaw_mask": f"speeds at NOW+2.0 s and 0.2 s earlier both > MIN_DS_MPS {DL.MIN_DS_MPS} (taniteval.four_families)", "free_gb_at_start": round(LR.free_gb(), 2), "wall_s": DL.stamp(t0)}


def main(argv):
    smoke, force = "--smoke" in argv, "--force" in argv
    opts = {}
    rest = []
    it = iter(argv)
    for a in it:
        if a in ("--train-dir", "--out"):
            opts[a] = next(it)
        elif not a.startswith("--"):
            rest.append(a)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    out_dir = opts.get("--out", DL.OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    if not smoke and not DL.controls_passed():
        print("REFUSED: CONTROLS_B1B2.json does not say all_pass: no arm row is written", flush=True)
        return 2
    rc = 0
    for spec in rest:
        f = spec.split("|")
        name, run_dir = f[0], f[1]
        basis, ub = (f[2] if len(f) > 2 else ""), (f[3] if len(f) > 3 else "")
        out = os.path.join(out_dir, ("smoke_" if smoke else "") + f"decod_{name}.json")
        if os.path.exists(out) and not force:
            print(f"[{name}] done already, skipped", flush=True)
            continue
        try:
            r = read_arm(name, run_dir, basis or None, ub or None, smoke, opts.get("--train-dir"))
            json.dump(r, open(out + ".tmp", "w", encoding="utf-8"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)
            os.replace(out + ".tmp", out)
            print(f"[{name}] B1 read in {r['wall_s']} s -> {out}", flush=True)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            open(os.path.join(out_dir, f"FAILED_decod_{name}.txt"), "w", encoding="utf-8").write(f"{type(e).__name__}: {str(e)[:2000]}\n")
            rc = 1
    summ = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("decod_") and fn.endswith(".json"):
            z = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
            summ[z["name"]] = {"basis_kind": z["basis_kind"], "tier": "T0", **{t: {"R2": v["eval"]["R2"], "MAE": v["eval"]["MAE"], "n_windows": v["eval"]["n_windows"]} for t, v in z["targets"].items()}}
    json.dump({"b1_summary": summ, "note": "BESIDE, no bar; z1 spaces differ per arm, R2 and MAE are read per arm"}, open(os.path.join(out_dir, "DECOD_B1.json"), "w", encoding="utf-8"), indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
