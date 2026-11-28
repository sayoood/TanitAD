"""D-REFCV4B-ASTAR-GEOMETRY -- the contamination MEASURED ON THE REAL ARM.

The full `refcv3_arm.py` re-roll is BLOCKED at HEAD by an unrelated drift
refusal (`tac_goal_tok_head`), so this measures the half of `oracle_sel` that
does not need a forward pass: the ANCHOR SELECTION itself.

`a_star` depends on exactly three things -- the decoder's `anchors` /
`anchor_controls` buffers (both PERSISTENT, so both come straight out of the
real checkpoint), each window's measured `v0`, and each window's GT waypoints.
None of them needs the transformer. So the SELECTION can be scored on the real
trained vocabulary and the real 141-clip eval cache with no forward at all.

⛔ WHAT THIS CANNOT SETTLE. `oracle_sel` is the a_star anchor's LEARNED
REFINEMENT (`out["anchor_traj"][a_star]`), and the refinement needs the
forward. So this measures the selection defect at ANCHOR level and is a
necessary condition for the ceiling assertion, not the headline number.
Reported as exactly that.

Both bindings are fed to the SHIPPED `_oracle_anchor_index`, and the bank comes
from the REAL `roll_bank` -- no second convention anywhere.
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"
CFG = r"C:\Users\Admin\refcv4b_final\config.json"
EPS = r"C:\Users\Admin\tanitad-data\refav1-eval141\eps"
N_CLIPS = int(os.environ.get("N_CLIPS", "12"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "roll", "real_arm_astar.json")

sys.path[:0] = [os.path.join(REPO, "stack"),
                os.path.join(REPO, "stack", "scripts"), REPO]

import importlib.util
spec = importlib.util.spec_from_file_location(
    "_arm", os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
arm = importlib.util.module_from_spec(spec)
sys.modules["_arm"] = arm
spec.loader.exec_module(arm)

from tanitad.refs import refc
from tanitad.data.v2_dataset import build_v2_providers, load_or_build_manifest


cfgj = json.load(io.open(CFG, encoding="utf-8"))
HZ = tuple(int(h) for h in cfgj["horizons"])
N_ANCH = int(cfgj["anchors"]["shape"][0])
S = len(HZ)
print("arm            : refcv4b @ 40284  (v0_conditioned=True, units=alat)")
print("horizons       :", HZ)
print("n_anchors      :", N_ANCH)

# ---- the REAL trained vocabulary, straight out of the checkpoint ----------
t = time.time()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model"] if "model" in ck else ck
A_KEY = "core.decoder.anchors"
C_KEY = "core.decoder.anchor_controls"
assert A_KEY in sd, sorted(k for k in sd if "anchor" in k)[:10]
anchors = sd[A_KEY].float().clone()                  # [N, S, 2] @ ref speed
controls = sd[C_KEY].float().clone()                 # [N, 2] (accel, a_lat)
print("ckpt loaded    : %.1fs   anchors %s  controls %s"
      % (time.time() - t, tuple(anchors.shape), tuple(controls.shape)))
print("controls col1  : [%.3f, %.3f] m/s^2 (LATERAL ACCEL, not curvature)"
      % (float(controls[:, 1].min()), float(controls[:, 1].max())))

# ---- a decoder that carries them, so the REAL roll_bank is what runs -------
dec = refc.AnchoredDiffusionDecoder(
    feat_dim=64, n_steps=S, d_meas=8, d_ctx=8, tac_latent_dim=8,
    anchors=torch.zeros(N_ANCH, S, 2), cfg=refc.DecoderConfig(),
    hierarchy=False, graft_maneuver=False, graft_target_latent=False,
    grounded_selector=False, horizons=HZ, v0_conditioned=True,
    ref_speed_ms=10.0, control_units="alat", alat_v_floor_ms=4.0,
    kappa_cap=0.12)
with torch.no_grad():
    dec.anchors.copy_(anchors)
    dec.anchor_controls.copy_(controls)

# sanity: the checkpoint's `anchors` really is this family at the ref speed
ref_roll = dec.roll_bank(None, None, 1, torch.float32)[0]
d_ref = float((ref_roll - anchors).abs().max())
print("anchors == roll_bank(ref_speed) max|d| = %.4g  (small => the buffer is "
      "the reference-speed family, as refc.py documents)" % d_ref)

# ---- the REAL eval windows ------------------------------------------------
man = load_or_build_manifest(EPS, verbose=False)
eps = build_v2_providers([EPS], lru_size=8, verbose=False)[:N_CLIPS]
print("clips          : %d of %d" % (len(eps), len(man["files"])))

import refb_labels  # stack/scripts/refb_labels.py (on sys.path via stack/scripts)

rows = []
for ei, ep in enumerate(eps):
    poses = ep.poses.float()                          # [T, 4] (x, y, yaw, v)
    T = poses.shape[0]
    for t0 in range(0, T - max(HZ) - 1, 10):          # stride 10 frames
        fut = poses[t0 + 1:t0 + 1 + max(HZ)]
        if fut.shape[0] < max(HZ):
            continue
        pose_last = poses[t0:t0 + 1]
        fut_ext = torch.zeros(1, max(HZ), 4)
        fut_ext[0, :fut.shape[0]] = fut
        tgt = refb_labels.waypoint_targets(pose_last, fut_ext, HZ)   # [1, S, 2]
        v0 = float(poses[t0, 3])
        sv = torch.ones(S)
        good = dec.roll_bank(torch.tensor([v0]), None, 1, torch.float32)
        bad = dec.anchors[None].float()
        a_ok = arm._oracle_anchor_index(good, tgt, sv)
        a_bad = arm._oracle_anchor_index(bad, tgt, sv)
        # anchor-level error IN THE DECODED GEOMETRY (what a refinement starts
        # from under EITHER binding)
        e_ok = float((good[0, a_ok] - tgt[0]).pow(2).sum(-1).sqrt().mean())
        e_bad = float((good[0, a_bad] - tgt[0]).pow(2).sum(-1).sqrt().mean())
        # and at the 2 s slot (grid "2s" = V3_HORIZONS index 3 -> 20 ticks)
        i2 = HZ.index(20)
        e2_ok = float((good[0, a_ok, i2] - tgt[0, i2]).pow(2).sum().sqrt())
        e2_bad = float((good[0, a_bad, i2] - tgt[0, i2]).pow(2).sum().sqrt())
        rows.append((ei, t0, v0, a_ok, a_bad, e_ok, e_bad, e2_ok, e2_bad))

r = np.array([x[2:] for x in rows], dtype=np.float64)
ep_id = np.array([x[0] for x in rows])
v0s, aok, abad = r[:, 0], r[:, 1].astype(int), r[:, 2].astype(int)
eok, ebad, e2ok, e2bad = r[:, 3], r[:, 4], r[:, 5], r[:, 6]
n = len(rows)
diff = int((aok != abad).sum())

print()
print("=" * 72)
print("MEASURED on the REAL refcv4b vocabulary, REAL eval windows")
print("=" * 72)
print("n windows                       : %d  (%d clips, stride 10)" % (n, len(eps)))
print("v0 range                        : %.2f - %.2f m/s (mean %.2f)"
      % (v0s.min(), v0s.max(), v0s.mean()))
print("a_star CHANGED by the fix       : %d / %d  = %.2f %%"
      % (diff, n, 100.0 * diff / n))
print()
print("ANCHOR-level error, full 6 s horizon (mean L2 over the 8 slots)")
print("  CORRECT binding (out.anchor_bank) : %.4f m" % eok.mean())
print("  DEFECT  binding (decoder.anchors) : %.4f m" % ebad.mean())
print("  contamination                     : %+.4f m" % (ebad - eok).mean())
print("  windows the defect made WORSE     : %d / %d" % (int((ebad > eok + 1e-9).sum()), n))
print("  windows the defect made BETTER    : %d / %d" % (int((ebad < eok - 1e-9).sum()), n))
print()
print("ANCHOR-level error at the 2 s slot (the reported grid)")
print("  CORRECT binding                   : %.4f m" % e2ok.mean())
print("  DEFECT  binding                   : %.4f m" % e2bad.mean())
print("  contamination                     : %+.4f m" % (e2bad - e2ok).mean())

# episode-cluster bootstrap on the PAIRED per-window delta (the programme's
# estimator; overlapping_holdout_se is never used)
rng = np.random.default_rng(20260906)
uep = np.unique(ep_id)
d2 = e2bad - e2ok
boot = []
for _ in range(2000):
    pick = rng.choice(uep, size=len(uep), replace=True)
    boot.append(np.concatenate([d2[ep_id == e] for e in pick]).mean())
lo, hi = np.percentile(boot, [2.5, 97.5])
print("  paired episode-cluster CI (2 s)   : [%+.4f, %+.4f]  (%d episodes, "
      "2000 resamples)" % (lo, hi, len(uep)))
print("  separated from zero               :", bool(lo > 0 or hi < 0))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({
    "arm": "refcv4b@40284", "n_windows": n, "n_clips": len(eps),
    "n_episodes": int(len(uep)),
    "astar_changed": diff, "astar_changed_frac": diff / n,
    "anchor_ade_6s_correct": float(eok.mean()),
    "anchor_ade_6s_defect": float(ebad.mean()),
    "anchor_ade_2s_correct": float(e2ok.mean()),
    "anchor_ade_2s_defect": float(e2bad.mean()),
    "paired_delta_2s": float(d2.mean()),
    "paired_ci_2s": [float(lo), float(hi)],
    "estimator": "paired episode-cluster bootstrap, 2000 resamples",
    "tier": "T0 (oracle selection; anchor level, no forward pass)",
    "ckpt": CKPT, "episodes": EPS,
}, io.open(OUT, "w", encoding="utf-8"), indent=2)
print()
print("banked ->", OUT)
