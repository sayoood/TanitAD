"""WP-I — what representation actually carries driving SEMANTICS, at power?

⛔⛔ THE DEFECT THIS RUN REPAIRS FIRST, BEFORE ANY ARM IS COMPARED.
The `vru_ahead` target used by P4-3c, WP-H and WP-J is built from the agent join,
whose visibility flag `occ` is computed at the SENSOR's 120 deg frustum. The
ENCODER is fed the canonical SQUARE frame -- `physicalai.as_frame(None, 256,
F_REF)` returns `CanonicalFrame(h=256, w=256, f_ref=266.0, projection='pinhole')`,
i.e. a half-angle of atan(128/266) = 25.70 deg, HFOV 51.4 deg. The join's own
docstring warns about exactly this ("PASS THE ENCODER'S FRAME INSTEAD when the
consumer was fed a centred sub-frame"). MEASURED on this join:

    48.5 % of forward `automobile` and 46.7 % of forward `person` cuboids lie
    OUTSIDE the encoder's field of view;
    30.4 % of `vru<20m` (NEAR) positive windows and 8.5 % of `vru<60m` (ALL)
    positive windows are positives for an agent the encoder NEVER SAW.

⇒ every arm was being asked to decode agents absent from its input, and the NEAR
band -- the one WP-H expected to be EASIER -- carries 3.6x more of that label
noise than ALL. So `NEAR - ALL` was partly measuring the label frame, not
resolution. Both gated and ungated targets are reported here so the size of the
artifact is visible rather than asserted.

⭐ THE QUESTION THE PI ASKED, made measurable: can a strong pretrained backbone
inject scene semantics that REF-C's own trunk does not carry? FOUR arms, all on
the SAME windows and the SAME 2x4 spatial pooling grid (the WP-J lesson: a floor
is only a floor if it has the same representational affordances as the arm):

    pixels     raw 9-channel frame, pooled 2x4              the matched floor
    rand       the SAME ResNet architecture, RANDOM weights the architecture floor
    trunk      REF-C's trained encoder                      ours
    dinov3     DINOv3 ViT-L/16, frozen, 3 sub-frames        the strong teacher

The `rand` arm is what separates "a deep conv stack pooled to 2x4" from "features
REF-C's training produced". If rand ~= trunk, REF-C's training added no semantics.

⛔ CONTROLS -- mandatory; this probe family produced four documented failures on
2026-08-22 and six specification defects in this campaign:
  * CONSTANT-only      must read EXACTLY 0.5000 (the no-information value)
  * SHUFFLED-target    must collapse to ~0.5000
  * PC-PIXEL (brightness > median)  the PIXEL arm must win -- validates the floor
  * PC-TRUNK (REF-C's own selected-anchor lateral sign)  the TRUNK arm must win --
                       validates the trunk extraction. A positive control that
                       only ONE arm can pass is what makes the panel readable;
                       WP-J's `lead<25m` control was passed by BOTH arms and so
                       validated nothing.
  * n, d, base rate and episode count printed for every row
  * LEAVE-ONE-EPISODE-OUT: every scored prediction is out-of-fold w.r.t. its
    episode, so the split is episode-disjoint AND all episodes are scored.
  * lambda is chosen INSIDE each fold's training data only, never on the scored
    episode.
  * interval = episode-cluster bootstrap over the episodes, PAIRED for arm
    differences (taniteval doctrine). warning: 15 clusters is few; the interval
    is reported with its n and is NOT a substitute for a replicate arm.

Tier: T0. NON-PARITY pilot corpus. Evidence class: MEASURED (ours).
"""
import importlib.util
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEAR_M, FAR_M = 20.0, 60.0
GRID = (2, 4)
HALF = math.atan(128.0 / 266.0)          # the ENCODER's half-angle, not the sensor's
MAX_PER_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 120
VRU = {"person", "rider", "stroller", "animal"}

# ---------------------------------------------------------------- models ----
from tanitad.refs import refc as _refc                                   # noqa: E402


def _refc_encoder(trained):
    m = _refc.RefCModel(_refc.refc_config())
    if trained:
        ck = torch.load(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt",
                        map_location="cpu", weights_only=False)["model"]
        r = m.load_state_dict(ck, strict=False)
        assert set(r.missing_keys) == {"decoder.anchor_controls"}, r.missing_keys
        assert not r.unexpected_keys, r.unexpected_keys
    return m.to(DEV).eval()


model = _refc_encoder(True)
torch.manual_seed(0)
rand_model = _refc_encoder(False)          # SAME architecture, untrained weights

from transformers import DINOv3ViTModel                                  # noqa: E402

DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
dino = DINOv3ViTModel.from_pretrained(DINO_ID, dtype=torch.float16).to(DEV).eval()
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
PATCH = int(dino.config.patch_size)
print("[wpi] dinov3 params %.1f M - patch %d - hidden %d"
      % (sum(p.numel() for p in dino.parameters()) / 1e6, PATCH,
         dino.config.hidden_size), flush=True)


def pool(x, grid=GRID):
    """[C,H,W] -> flat vector on the SHARED spatial grid. Every arm uses this."""
    return torch.nn.functional.adaptive_avg_pool2d(x.float(), grid).flatten()


@torch.no_grad()
def dino_feat(frame9):
    """frame9: [9,H,W] in [0,1] = 3 stacked RGB sub-frames.

    AFFORDANCE PARITY: the trunk is fed all 9 channels (3 sub-frames), so the
    teacher is too -- one ViT pass per sub-frame, concatenated. Feeding it only
    the last frame would hand the trunk a temporal advantage and reproduce the
    WP-J defect in the time axis.
    """
    gh, gw = frame9.shape[-2] // PATCH, frame9.shape[-1] // PATCH
    outs = []
    for k in range(3):
        img = frame9[3 * k:3 * k + 3].unsqueeze(0)
        img = ((img - IMNET_MEAN) / IMNET_STD).half()
        h = dino(pixel_values=img).last_hidden_state[0]          # [pre + P, D]
        tok = h[h.shape[0] - gh * gw:].reshape(gh, gw, -1).permute(2, 0, 1)
        outs.append(pool(tok))
    return torch.cat(outs)


# ------------------------------------------------------------------ data ----
import collections                                                       # noqa: E402
import io as _io                                                         # noqa: E402

raw = collections.defaultdict(dict)
for line in _io.open(O + "/pilot_val_agents.jsonl", encoding="utf-8"):
    d = json.loads(line)
    raw[d["clip_id"]][d["frame_idx"]] = d["agents"]

src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    O + "/pilot_val_agents.jsonl", seed=0, lru=6)

FEATS = {"pixels": [], "rand": [], "trunk": [], "dinov3": []}
TGT = collections.defaultdict(list)
EPS = []
rng = np.random.default_rng(1234)
t0 = time.time()
with torch.no_grad():
    for si, stem in enumerate(src.stems):
        key = next((k for k in raw if k in stem or stem in k), None)
        if key is None:
            continue
        T = int(src._ep(stem)["poses"].shape[0])
        lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
        idx = sorted(rng.choice(np.arange(lo, hi),
                                size=min(MAX_PER_EP, hi - lo), replace=False))
        for t in idx:
            t = int(t)
            ags = raw[key].get(t)
            if ags is None:
                continue
            w = src.window(stem, t)
            if w is None:
                continue
            b = P.collate([w], DEV)
            f9 = b["frames"][0, -1]                       # [9,256,256] in [0,1]
            fm = model.encoder(b["frames"][:, -1])
            fm = fm[0] if isinstance(fm, (tuple, list)) else fm
            fr = rand_model.encoder(b["frames"][:, -1])
            fr = fr[0] if isinstance(fr, (tuple, list)) else fr
            FEATS["pixels"].append(pool(f9).cpu().numpy())
            FEATS["trunk"].append(pool(fm[0]).cpu().numpy())
            FEATS["rand"].append(pool(fr[0]).cpu().numpy())
            FEATS["dinov3"].append(dino_feat(f9).cpu().numpy())

            def seen(a):
                return a["cx"] > 0 and abs(math.atan2(a["cy"], a["cx"])) <= HALF

            vru_n = any(a.get("cls") in VRU and 0 < a["cx"] < NEAR_M
                        and abs(a["cy"]) <= 4.0 for a in ags)
            vru_f = any(a.get("cls") in VRU and 0 < a["cx"] < FAR_M
                        and abs(a["cy"]) <= 4.0 for a in ags)
            vru_n_fov = any(a.get("cls") in VRU and 0 < a["cx"] < NEAR_M
                            and abs(a["cy"]) <= 4.0 and seen(a) for a in ags)
            vru_f_fov = any(a.get("cls") in VRU and 0 < a["cx"] < FAR_M
                            and abs(a["cy"]) <= 4.0 and seen(a) for a in ags)
            n_seen = sum(1 for a in ags if seen(a))
            lead = min([a["cx"] for a in ags
                        if a["cx"] > 0 and abs(a["cy"]) <= 2.0] or [FAR_M])
            out = model(b["frames"], None, b["v0"], steps=2)
            sel = int(out["sel_score"][0].argmax())
            lat = float(out["anchor_traj"][0, sel, -1, 1])
            TGT["vru<20m RAW (ungated)"].append(float(vru_n))
            TGT["vru<60m RAW (ungated)"].append(float(vru_f))
            TGT["vru<20m IN-FOV"].append(float(vru_n_fov))
            TGT["vru<60m IN-FOV"].append(float(vru_f_fov))
            TGT["crowd: >=3 agents in FOV"].append(float(n_seen >= 3))
            TGT["lead < 25 m"].append(float(lead < 25.0))
            TGT["PC-PIXEL brightness"].append(float(f9.mean()))
            TGT["PC-TRUNK sel lateral>0"].append(float(lat > 0))
            EPS.append(stem)
        print("[wpi] ep %d/%d %s: %d windows (%.0f s)"
              % (si + 1, len(src.stems), stem, len(EPS), time.time() - t0),
              flush=True)

E = np.array(EPS)
UE = sorted(set(EPS))
X = {k: np.asarray(v, np.float64) for k, v in FEATS.items()}
Y = {k: np.asarray(v, np.float64) for k, v in TGT.items()}
Y["PC-PIXEL brightness"] = (Y["PC-PIXEL brightness"] >
                            np.median(Y["PC-PIXEL brightness"])).astype(float)
N = len(E)
print("\nwindows n=%d - episodes %d - %s"
      % (N, len(UE), " - ".join("d[%s]=%d" % (k, v.shape[1])
                                for k, v in X.items())), flush=True)


# ------------------------------------------------------------- estimator ----
def auc(s, y):
    """Rank AUC with TIES AVERAGED. A constant predictor MUST return 0.5000."""
    p, n = y.sum(), (1 - y).sum()
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1, dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


LAMBDAS = (1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)


def loeo_scores(F, y):
    """Leave-one-EPISODE-out ridge; returns out-of-fold scores for every row.

    Solved in the DUAL (Gram) form because d >> n for three of the four arms:
    w = A^T (A A^T + lam I)^-1 y is algebraically identical to the primal ridge
    and costs n^3 instead of d^3, so no arm is penalised for its width. One
    eigendecomposition per fold serves every lambda.

    ⛔⛔ THE DEFECT THE SHUFFLED CONTROL CAUGHT, AND THE FIX.
    Version 1 pooled every fold's raw out-of-fold score into one AUC. Each fold
    fits a DIFFERENT lambda on a DIFFERENT training set, so its scores carry a
    fold-specific OFFSET and SCALE. Pooling them makes the ranking partly a
    comparison BETWEEN episodes, i.e. a channel that encodes episode identity --
    and when a shuffled label happens to be commoner in some episodes than
    others, that channel alone produces signal. MEASURED at n=60: the
    SHUFFLED-target control read **0.7736** where it must read ~0.5000.

    ⇒ Each fold's held-out scores are standardised by the mean and sd of that
    fold's own TRAINING predictions -- statistics computed without ever touching
    the held-out episode, so the normalisation cannot leak. Fold scores then
    live on one scale and the pooled AUC is a genuine ranking again.
    ⚠️ Consequence to state in the writeup, not hide: this removes signal that
    lives PURELY in a per-episode offset. What survives is within-fold ranking
    on a shared scale, which is the conservative reading.
    """
    out = np.zeros(len(y))
    for ep in UE:
        te = E == ep
        tr = ~te
        A, B = F[tr], F[te]
        mu, sd = A.mean(0), A.std(0) + 1e-8
        A, B = (A - mu) / sd, (B - mu) / sd
        ytr = y[tr]
        inner = np.arange(len(A)) % 4 != 0          # lambda picked INSIDE the fold
        Ai = A[inner]
        ev2, Q2 = np.linalg.eigh(Ai @ Ai.T)
        Q2ty = Q2.T @ ytr[inner]
        Xin = A[~inner] @ Ai.T
        best, bl = -np.inf, LAMBDAS[-1]
        for lam in LAMBDAS:
            al = Q2 @ (Q2ty / (ev2 + lam))
            a = auc(Xin @ al, ytr[~inner])
            if a == a and a > best:
                best, bl = a, lam
        G = A @ A.T
        evals, Q = np.linalg.eigh(G)
        alpha = Q @ ((Q.T @ ytr) / (evals + bl))
        s_tr = G @ alpha                       # TRAIN-only calibration statistics
        m, s = s_tr.mean(), s_tr.std() + 1e-12
        out[te] = ((B @ A.T) @ alpha - m) / s
    return out


def ep_boot(sc_a, sc_b, y, iters=2000, seed=0):
    """PAIRED episode-cluster bootstrap on the AUC difference (a - b)."""
    r = np.random.default_rng(seed)
    idx = {e: np.where(E == e)[0] for e in UE}
    d = []
    for _ in range(iters):
        pick = np.concatenate([idx[UE[i]] for i in
                               r.integers(0, len(UE), len(UE))])
        if y[pick].sum() in (0, len(pick)):
            continue
        d.append(auc(sc_a[pick], y[pick]) - auc(sc_b[pick], y[pick]))
    d = np.array([x for x in d if x == x])
    if not len(d):
        return float("nan"), float("nan")
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


ARMS = ["pixels", "rand", "trunk", "dinov3"]
res, SC = {}, {}
print("\n%-28s%7s%s" % ("target", "base", "".join("%10s" % a for a in ARMS)),
      flush=True)
for name, y in Y.items():
    SC[name] = {a: loeo_scores(X[a], y) for a in ARMS}
    row = {a: auc(SC[name][a], y) for a in ARMS}
    res[name] = {"base_rate": float(y.mean()), "auc": row, "n_pos": int(y.sum())}
    print("%-28s%7.3f%s" % (name, y.mean(),
                            "".join("%10.4f" % row[a] for a in ARMS)), flush=True)

# ---- the controls that must read known values -------------------------------
# ⛔ THE SHUFFLED CONTROL IS RUN ON *EVERY* ARM, not just one. A leak that lives
# in one representation's geometry (e.g. the widest arm overfitting a fold) is
# invisible to a control run only on another arm -- and it was the widest arm
# (dinov3, d=24576) that read systematically BELOW 0.5 in the smoke run, which
# is the signature of exactly such an arm-specific artifact.
const = np.ones((N, 1))
y0 = Y["vru<60m IN-FOV"]
a_const = auc(loeo_scores(const, y0), y0)
ysh = y0.copy()
np.random.default_rng(7).shuffle(ysh)
sh_by_arm = {a: auc(loeo_scores(X[a], ysh), ysh) for a in ARMS}
a_sh = sh_by_arm["trunk"]
print("\n  CONSTANT-only control     %.4f   <- must be 0.5000 EXACTLY" % a_const)
print("  SHUFFLED-target control, per arm  (each must be ~0.5000):")
for a in ARMS:
    flag = "" if abs(sh_by_arm[a] - 0.5) < 0.12 else "   <== FAIL"
    print("      %-8s %.4f%s" % (a, sh_by_arm[a], flag))
print("  feature health (std of the pooled features, must be > 0, no NaN):")
for a in ARMS:
    print("      %-8s d=%-6d std %.4g  nan %d  const-cols %d"
          % (a, X[a].shape[1], float(X[a].std()),
             int(np.isnan(X[a]).sum()), int((X[a].std(0) < 1e-12).sum())))

pcp = res["PC-PIXEL brightness"]["auc"]
pct = res["PC-TRUNK sel lateral>0"]["auc"]
print("  PC-PIXEL   pixels %.4f vs trunk %.4f   <- PIXELS must win"
      % (pcp["pixels"], pcp["trunk"]))
print("  PC-TRUNK   trunk  %.4f vs pixels %.4f   <- TRUNK must win"
      % (pct["trunk"], pct["pixels"]))
ctrl_ok = (abs(a_const - 0.5) < 1e-6
           and all(abs(v - 0.5) < 0.12 for v in sh_by_arm.values())
           and pcp["pixels"] > pcp["trunk"] and pct["trunk"] > pct["pixels"])

# ---- the decision-grade comparisons, with paired episode-cluster CIs ---------
print("\n%-44s%9s%24s" % ("paired difference (AUC)", "delta", "CI95"), flush=True)
cmps = []
for tname in ("vru<60m IN-FOV", "vru<20m IN-FOV", "crowd: >=3 agents in FOV"):
    for a, b in (("trunk", "pixels"), ("trunk", "rand"),
                 ("dinov3", "trunk"), ("dinov3", "pixels")):
        y = Y[tname]
        d = res[tname]["auc"][a] - res[tname]["auc"][b]
        lo, hi = ep_boot(SC[tname][a], SC[tname][b], y)
        sep = "SEPARATED" if (lo == lo and (lo > 0 or hi < 0)) else "overlaps 0"
        cmps.append({"target": tname, "a": a, "b": b, "delta": d,
                     "ci": [lo, hi], "separated": sep == "SEPARATED"})
        print("%-22s%-22s%+9.4f   [%+.4f, %+.4f] %s"
              % (tname[:20], a + " - " + b, d, lo, hi, sep), flush=True)

fov_shift = {}
for t in ("vru<20m RAW (ungated)", "vru<60m RAW (ungated)"):
    g = t.replace(" RAW (ungated)", " IN-FOV")
    fov_shift[t] = {a: res[g]["auc"][a] - res[t]["auc"][a] for a in ARMS}
print("\n  FOV-GATE EFFECT (in-FOV target minus raw target), per arm:")
for t, v in fov_shift.items():
    print("    %-24s%s" % (t.split()[0], " ".join("%s %+0.4f" % (a, v[a])
                                                  for a in ARMS)))

json.dump({"n": N, "episodes": len(UE), "grid": list(GRID),
           "encoder_half_angle_deg": math.degrees(HALF),
           "d": {k: int(v.shape[1]) for k, v in X.items()},
           "results": res, "constant_control": a_const,
           "shuffled_control": a_sh, "shuffled_by_arm": sh_by_arm,
           "controls_ok": bool(ctrl_ok),
           "comparisons": cmps, "fov_gate_effect": fov_shift,
           "_tier": "T0; NON-PARITY pilot; leave-one-episode-out; "
                    "paired episode-cluster bootstrap",
           "_dino": DINO_ID},
          open(O + "/wpi_semantic_floor.json", "w"), indent=1)
print("\n-> wpi_semantic_floor.json")
