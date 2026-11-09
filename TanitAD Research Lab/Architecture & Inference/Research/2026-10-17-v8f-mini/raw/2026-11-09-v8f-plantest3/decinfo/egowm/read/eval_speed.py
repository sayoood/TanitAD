"""EGOWM read layer, part 0: the MEASURED speed of the 141 eval clips, and how well integrating the logged acceleration reproduces it (deliverable 3).

Source of the speed (the same one the trainer uses on the --wm-zcache path, ``v8f_wm_windows._kin_small``): ``kin3[:, 0]`` of ``ClipStore.get_kin(clip)`` in kin_mode ``yaw`` = the speed of the clip's NATIVE pose
track at raw frame 2j (``ego_channels_from_poses``), cache step j. Row 0 is NaN by construction (``yaw_kin_from_track``); the read layer fills it with row 1 and says so (``filled_row0``).
The eval frame cache (``ladder_read/cache/frames_s*.npz``) carries NO speed, only (a, yaw rate) per frame; the speed is joined here by clip name and checked for length.

CLI:  python eval_speed.py build   -> eval141_speed.npz (+ sha256 line)
      python eval_speed.py kinematics -> KINEMATICS_EGOWM.json / .md
CPU only. Needs LADDER_READ_CODE_ROOT = the patched EGOWM tree (the default below)."""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/egowm/tree")
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(THEORY, "ladder_read"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lr_lib as L  # noqa: E402

SPEED_NPZ = os.path.join(HERE, "eval141_speed.npz")
FRAMES0 = f"{L.SCR}/cache/frames_s0.npz"
DT = 0.2


def build(out=SPEED_NPZ):
    """{clips, T, v (concatenated, NaN at row 0 of each clip)} for the eval clips in the frame cache's order."""
    fr = L.load_eval_frames(FRAMES0)
    tracks = L.D.load_native_tracks(f"{L.SF}/inputs/native_tracks.pt")
    store = L.D.ClipStore(f"{L.SF}/cache", "eval", lru=2, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    vs = []
    for c, t in zip(fr["clips"], fr["T"]):
        kin = store.get_kin(c)["kin3"]
        if len(kin) != t:
            raise RuntimeError(f"{c}: kin3 has {len(kin)} rows, the frame cache {t}")
        vs.append(kin[:, 0].astype(np.float32))
    np.savez(out, clips=np.array(fr["clips"]), T=np.array(fr["T"]), v=np.concatenate(vs))
    return out


def load(path=SPEED_NPZ, fill_row0=True):
    """-> {clip: v [T] float32}. fill_row0: row 0 (NaN, no backward pose) := row 1, so that an anchor at frame 0 is defined; the number of rows filled is returned."""
    z = np.load(path, allow_pickle=False)
    cut = np.cumsum(z["T"])[:-1]
    out, n_fill = {}, 0
    for c, v in zip([str(x) for x in z["clips"]], np.split(z["v"], cut)):
        v = v.copy()
        if fill_row0 and not np.isfinite(v[0]) and np.isfinite(v[1]):
            v[0] = v[1]
            n_fill += 1
        out[c] = v
    return out, n_fill


def floored_integral(v0, acc, dt=DT):
    """Hand integration (numpy loop, independent of the trainer's closed form): speeds at offsets 0..len(acc), floored at 0 at every step."""
    s = [float(v0)]
    for a in acc:
        s.append(max(0.0, s[-1] + dt * float(a)))
    return np.array(s)


def kinematics(shift=0):
    """MAE (m/s) of the E-KIN speed at +2.0 s (10 steps) and +6.0 s (30 steps) against the measured speed there, anchors = every frame whose frames up to the target are ``ok`` and whose speed is measured.
    ``shift`` 0 = the trainer's rule (acc = act[f, 0] for the transition f -> f+1); 1 = beside, act[f+1, 0]. Also: E-NOW (hold) and the unfloored sum. Per clip then pooled."""
    fr = L.load_eval_frames(FRAMES0)
    vd, _ = load(fill_row0=False)
    res = {}
    for h_s, n in ((2.0, 10), (6.0, 30)):
        per_clip = {"kin": [], "kin_unfloored": [], "now": []}
        pooled = {"kin": [], "kin_unfloored": [], "now": []}
        signed = []
        n_anchor, n_clip = 0, 0
        for c, a, ok in zip(fr["clips"], fr["act"], fr["ok"]):
            v = vd[c]
            T = len(v)
            m = {"kin": [], "kin_unfloored": [], "now": []}
            for f0 in range(1, T - n):
                if not (np.isfinite(v[f0]) and np.isfinite(v[f0 + n]) and ok[f0:f0 + n + 1].all()):
                    continue
                acc = a[f0 + shift:f0 + shift + n, 0]
                if len(acc) < n:
                    continue
                kin = floored_integral(v[f0], acc)[n]
                unf = v[f0] + DT * float(acc.sum())
                m["kin"].append(abs(kin - v[f0 + n]))
                m["kin_unfloored"].append(abs(unf - v[f0 + n]))
                m["now"].append(abs(v[f0] - v[f0 + n]))
                signed.append(kin - v[f0 + n])
            if m["kin"]:
                n_clip += 1
                for k in m:
                    per_clip[k].append(float(np.mean(m[k])))
                    pooled[k] += m[k]
                n_anchor += len(m["kin"])
        res[f"{h_s:g}s"] = {"n_anchor": n_anchor, "n_clips": n_clip,
                            **{f"{k}_mae_per_clip_mean": float(np.mean(per_clip[k])) for k in per_clip},
                            **{f"{k}_mae_per_clip_median": float(np.median(per_clip[k])) for k in per_clip},
                            **{f"{k}_mae_per_clip_max": float(np.max(per_clip[k])) for k in per_clip},
                            **{f"{k}_mae_pooled": float(np.mean(pooled[k])) for k in pooled},
                            "kin_signed_bias_pooled": float(np.mean(signed))}
    return res


def main(argv):
    if argv and argv[0] == "build":
        p = build()
        print(p, hashlib.sha256(open(p, "rb").read()).hexdigest())
        return 0
    if argv and argv[0] == "kinematics":
        out = {"tier": "T0 measurement beside the card (no bar)", "evidence_class": "MEASURED (this script, eval141 clips of the M2b cache)",
               "speed_source": "ClipStore.get_kin(clip)['kin3'][:, 0], kin_mode yaw, native pose track at raw frame 2j (stored in eval141_speed.npz)",
               "accel_source": "frames_s0.npz act[:, 0] = nan_to_num(cache 'a'), the trainer's act[..., 0]",
               "rule": "E-KIN: v(h) = max(0, v(h-1) + 0.2 * act[f0+h-1, 0]) from the measured v(f0); error against measured v(f0+n)",
               "trainer_rule_shift0": kinematics(0), "beside_shift1": kinematics(1)}
        json.dump(out, open(os.path.join(HERE, "KINEMATICS_EGOWM.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
