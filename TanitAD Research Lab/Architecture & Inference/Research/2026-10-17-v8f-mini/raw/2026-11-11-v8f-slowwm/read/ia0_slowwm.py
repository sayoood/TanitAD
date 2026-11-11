"""SLOWWM IA-0 read (BESIDE, never deciding: PREREG_V8F_SLOWWM "Read"): skill against Copy-Last per level at the band ends (L1 2.0 s, L2 6.0 s, L3 12 s), through the ladder readers (``ladder_read/lr_lib.py``
``gather`` / ``ia0_arm`` / ``level_passes``, unchanged). z1 lives in a different coordinate space per arm, so these rows compare each arm against ITS OWN Copy-Last and Oracle, never across arms; a slow basis also makes
Copy-Last strong by construction. For a basis arm the two lr_lib entry points that touch z1's input are dispatched to the hook for the duration of the read (``lr_lib.z_in`` -> the arm's ``ubasis``;
``lr_lib.encode_clip`` -> ``wm.z1_from_basis``); an adapter arm (R35H) runs lr_lib's own functions. Tier T0.
Usage: python ia0_slowwm.py [--smoke] [--force] "<name>|<run_dir>|<basis_npz>|<ubasis_npz>" ...   -> results/ia0_<name>.json (resumable) and IA0_SLOWWM.json (all arms read so far)."""
import contextlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lp_slowwm as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

L = S.LR
SH = S.SH
BAND = {1: "2", 2: "6", 3: "12"}       # seconds, the rows' keys in ia0_arm ("{h:g}")


@contextlib.contextmanager
def basis_read():
    """Dispatch lr_lib's z_in and encode_clip on the model: a basis model reads ``fr['ubasis']`` through ``z1_from_basis``; every other model gets lr_lib's own function. Undone on exit."""
    o_z, o_e = L.z_in, L.encode_clip

    def z_in(model, fr, ci):
        return SH.z_in_basis(model, fr, ci) if SH.is_basis(model) else o_z(model, fr, ci)

    def encode_clip(model, zin, act, contrast=None):
        if not SH.is_basis(model):
            return o_e(model, zin, act, contrast)
        wm = model.wm
        a = torch.as_tensor(act).clone()
        if contrast is not None:
            ch, dv = contrast
            a[..., ch] += dv
        z1 = wm.z1_from_basis(torch.as_tensor(zin)[None])
        a_sc = wm.scale_actions(a[None])[:, :-1]
        enc = wm.stack.encode(z1, a_sc)
        return enc.z, [a_sc] + [enc.act[l] for l in (1, 2)]
    L.z_in, L.encode_clip = z_in, encode_clip
    try:
        yield
    finally:
        L.z_in, L.encode_clip = o_z, o_e


def band_end(ia0):
    """{level: the band-end row's skill, interval, Copy-Last / Oracle context} from ``ia0_arm``'s rows."""
    out = {}
    for lv, key in BAND.items():
        r = ia0[str(lv)]["rows"].get(key, {})
        out[str(lv)] = {"horizon_s": float(key), "readable": r.get("readable"), "skill": r.get("point"), "lo": r.get("lo"), "hi": r.get("hi"), "above_copy_last": r.get("above"), "below_copy_last": r.get("below"),
                        "healthy": bool(ia0[str(lv)]["health"].get("separated_two_part"))}
    return out


def read_arm_ia0(name, run_dir, basis_npz=None, ubasis_npz=None, smoke=False):
    t0 = time.time()
    free = L.memory_guard()
    X = S.load_checked(name, run_dir, basis_npz, ubasis_npz)
    model, fr = X["model"], X["fr"]
    with basis_read():
        cells, states, cum, plan, built_ctx = L.gather(model, fr)
        W = L.IL.boot_weights(np.arange(len(fr["clips"])))
        worst = L.beside_copy_selfcheck(cells, W)
        if worst != 0.0:
            raise RuntimeError(f"Copy-Last against itself is not exactly 0.0 (max |value| {worst})")
        ia0 = L.ia0_arm(cells, plan, cum, W, "built", smoke)
        ia0_ctx1 = L.ia0_arm(cells, plan, cum, W, "ctx1", smoke)
    return {"tier": "T0", "role": "BESIDE, no bar", "name": name, "basis_kind": X["kind"], "basis_sha256": X["B"]["sha"] if X["B"] else None, "smoke": bool(smoke), "built": X["built"], "argv_checked": X["checked"],
            "copy_vs_itself_max_abs": worst, "n_clips": len(fr["clips"]), "cum_stride": cum, "ctx_built": built_ctx, "ia0": ia0, "ia0_planning_context1": ia0_ctx1, "passes": L.level_passes(ia0),
            "band_end": band_end(ia0), "band_end_context1": band_end(ia0_ctx1), "bootstrap": {"estimator": "episode-cluster bootstrap over the 141 eval clips", "B": L.B_BOOT, "seed": 0},
            "free_gb_at_start": round(free, 2), "wall_s": round(time.time() - t0, 1)}


def main(argv):
    smoke, force = "--smoke" in argv, "--force" in argv
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    res_dir = os.environ.get("SLOWWM_IA0_DIR", os.path.join(HERE, "results"))
    os.makedirs(res_dir, exist_ok=True)
    rc = 0
    for spec in [a for a in argv if not a.startswith("--")]:
        f = spec.split("|")
        name, run_dir = f[0], f[1]
        basis, ub = (f[2] if len(f) > 2 else ""), (f[3] if len(f) > 3 else "")
        out = os.path.join(res_dir, ("smoke_" if smoke else "") + f"ia0_{name}.json")
        if os.path.exists(out) and not force:
            print(f"[{name}] done already, skipped", flush=True)
            continue
        try:
            r = read_arm_ia0(name, run_dir, basis or None, ub or None, smoke)
            json.dump(r, open(out + ".tmp", "w", encoding="utf-8"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            os.replace(out + ".tmp", out)
            print(f"[{name}] IA-0 read in {r['wall_s']} s -> {out}", flush=True)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            open(os.path.join(res_dir, f"FAILED_ia0_{name}.txt"), "w", encoding="utf-8").write(f"{type(e).__name__}: {str(e)[:2000]}\n")
            rc = 1
    summ = {}
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("ia0_") and fn.endswith(".json"):
            z = json.load(open(os.path.join(res_dir, fn), encoding="utf-8"))
            summ[z["name"]] = {"basis_kind": z["basis_kind"], "tier": "T0", "band_end": z["band_end"], "band_end_context1": z["band_end_context1"]}
    json.dump({"band_end_summary": summ, "note": "BESIDE, no bar; each arm against its own Copy-Last (z1 spaces differ per arm)"}, open(os.environ.get("SLOWWM_IA0_SUMMARY", os.path.join(HERE, "IA0_SLOWWM.json")), "w", encoding="utf-8"), indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
