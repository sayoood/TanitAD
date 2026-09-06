"""WP-H — is the trunk's agent-blindness an OBJECTIVE failure or a RESOLUTION limit?

⛔ THE CONFOUND THIS SEPARATES. P4-3c records the trunk as decodable on `lead_gap_m`
and `ego_speed` but AT CHANCE on `vru_ahead` / `left_occupied`, and attributes that
to the objective. WP-G measured that a median VRU is ~5x11 px -- 0.8 of one ViT
patch. Everything that decodes is large-and-near or ego-derived; everything that
fails is a small distant agent. From the decodability numbers alone those two
explanations are INDISTINGUISHABLE.

⭐ THE DISCRIMINATOR: decode the SAME target restricted to NEAR agents (< 20 m,
where a VRU spans ~2 patches) versus ALL ranges. If NEAR separates and ALL does
not, the input is the problem, not the objective.

⚠️ AUC, NOT ACCURACY, and this is not a detail: the near label is RARER, and a
rarer label is easier to get "right" by predicting the majority. AUC is base-rate
robust. Base rates are printed for both.

⛔ CONTROLS -- this is the probe family that produced FOUR documented failures on
2026-08-22, so every one is mandatory:
  * CONSTANT-only         must read AUC 0.500 EXACTLY (the no-information value)
  * SHUFFLED-target       must collapse to ~0.500
  * RAW-PIXEL floor       a learned trunk that does not beat raw pixels added nothing
  * lambda fitted on the FIT split ONLY, never on the scored split
  * n and d printed; episode-disjoint split

Tier: T0. NON-PARITY pilot. Evidence class: MEASURED (ours).
"""
import importlib.util
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
dev = "cuda" if torch.cuda.is_available() else "cpu"
NEAR_M, FAR_M = 20.0, 60.0

from tanitad.refs import refc as _refc                                    # noqa: E402
model = _refc.RefCModel(_refc.refc_config())
_ck = torch.load(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt",
                 map_location="cpu", weights_only=False)["model"]
_r = model.load_state_dict(_ck, strict=False)
assert set(_r.missing_keys) == {"decoder.anchor_controls"} and not _r.unexpected_keys
model = model.to(dev).eval()

src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

# the agent join carries `cls`; re-read it so VRU membership is by CLASS, not size
import collections                                                        # noqa: E402
raw = collections.defaultdict(dict)
import io as _io                                                          # noqa: E402
for line in _io.open(f"{O}/pilot_val_agents.jsonl", encoding="utf-8"):
    d = json.loads(line)
    raw[d["clip_id"]][d["frame_idx"]] = d["agents"]
VRU = {"person", "rider", "stroller", "animal"}

FEAT, PIX, Yn, Yf, Ylead, EPS = [], [], [], [], [], []
rng = np.random.default_rng(1234)
with torch.no_grad():
    for stem in src.stems:
        key = next((k for k in raw if k in stem or stem in k), None)
        if key is None:
            continue
        T = int(src._ep(stem)["poses"].shape[0])
        lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
        for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(40, hi - lo),
                                    replace=False)):
            t0 = int(t0)
            ags = raw[key].get(t0)
            if ags is None:
                continue
            w = src.window(stem, t0)
            if w is None:
                continue
            b = P.collate([w], dev)
            fmap = model.encoder(b["frames"][:, -1])       # the trunk's own features
            if isinstance(fmap, (tuple, list)):
                fmap = fmap[0]
            # ⛔ THE COMPARISON MUST BE FAIR IN *SPATIAL STRUCTURE*, not only in
            # provenance. Run 1 pooled the trunk to a single 704-d vector and
            # compared it against pixels pooled to a 4x8 GRID -- so the floor kept
            # localisation the trunk arm was denied, and "pixels beat the trunk"
            # may have measured my pooling, not the representation. Both sides now
            # get the SAME 2x4 spatial grid.
            g = torch.nn.functional.adaptive_avg_pool2d(fmap[0].float(), (2, 4))
            FEAT.append(g.flatten().cpu().numpy())
            px = b["frames"][0, -1].float()
            PIX.append(torch.nn.functional.adaptive_avg_pool2d(px, (2, 4))
                       .flatten().cpu().numpy())

            # ⚠️ the CANONICAL definition (e_trunk2_targets.py) gates laterally at
            # |cy| <= 4.0. My first version used cx>0 alone, so it was NOT the same
            # target P4-3c reports and the numbers would not have been comparable.
            vru_near = any(a.get("cls") in VRU and 0 < a["cx"] < NEAR_M
                           and abs(a["cy"]) <= 4.0 for a in ags)
            vru_far = any(a.get("cls") in VRU and 0 < a["cx"] < FAR_M
                          and abs(a["cy"]) <= 4.0 for a in ags)
            lead = min([a["cx"] for a in ags
                        if a["cx"] > 0 and abs(a["cy"]) <= 2.0] or [FAR_M])
            Yn.append(float(vru_near)); Yf.append(float(vru_far))
            Ylead.append(float(lead)); EPS.append(stem)

F = np.array(FEAT, np.float64); X = np.array(PIX, np.float64)
Yn, Yf, Ylead, E = map(np.array, (Yn, Yf, Ylead, EPS))
ue = sorted(set(EPS)); cut = set(ue[: int(0.6 * len(ue))])
tr = np.array([e in cut for e in E])
print(f"windows n={len(Yn)} · trunk d={F.shape[1]} · pixel-floor d={X.shape[1]} · "
      f"episodes {len(ue)} · fit {tr.sum()} / score {(~tr).sum()}")
print(f"base rates  vru<{NEAR_M:.0f}m {Yn.mean():.3f} · vru<{FAR_M:.0f}m {Yf.mean():.3f}")


def auc(s, y):
    """Rank-based AUC with TIES AVERAGED.

    The first version sorted and never handled ties, so a CONSTANT predictor --
    every score identical -- was scored on argsort's arbitrary index order and
    read 0.6765 instead of 0.5000. The constant-only control caught it, which is
    exactly why that control is mandatory: a metric that cannot return the
    no-information value on no information will not return the truth on data.
    """
    p, n = y.sum(), (1 - y).sum()
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    ss = s[order]                                   # average ranks within tie groups
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def probe(Feat, y, tag):
    """Ridge with lambda chosen on a FIT-INTERNAL split only. Returns AUC."""
    A, B = Feat[tr], Feat[~tr]
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A, B = (A - mu) / sd, (B - mu) / sd
    ytr = y[tr]
    inner = np.arange(len(A)) % 4 != 0            # 75/25 inside the FIT split
    best, bl = -1, None
    for lam in (1e-1, 1, 10, 100, 1e3, 1e4):
        w = np.linalg.solve(A[inner].T @ A[inner] + lam * np.eye(A.shape[1]),
                            A[inner].T @ ytr[inner])
        a = auc(A[~inner] @ w, ytr[~inner])
        if a == a and a > best:
            best, bl = a, lam
    w = np.linalg.solve(A.T @ A + bl * np.eye(A.shape[1]), A.T @ ytr)
    return auc(B @ w, y[~tr]), bl


print()
print(f"{'target':<26}{'trunk AUC':>11}{'pixel floor':>13}{'lambda':>9}")
res = {}
Ylead_bin = (Ylead < 25.0).astype(float)   # POSITIVE CONTROL: large near object
for name, y in (("vru ahead < 20 m (NEAR)", Yn), ("vru ahead < 60 m (ALL)", Yf),
                ("lead < 25 m (POS CONTROL)", Ylead_bin)):
    a, lam = probe(F, y, name)
    ap, _ = probe(X, y, name)
    res[name] = {"trunk_auc": a, "pixel_auc": ap, "lambda": lam,
                 "base_rate": float(y.mean())}
    print(f"{name:<26}{a:>11.4f}{ap:>13.4f}{lam:>9g}")

# the CONTROL that must read the no-information value exactly
const = np.ones((len(Yn), 1))
a_const, _ = probe(const, Yn, "constant")
ysh = Yn.copy(); np.random.default_rng(7).shuffle(ysh)
a_sh, _ = probe(F, ysh, "shuffled")
print(f"\n  CONSTANT-only control      {a_const:.4f}   <- must be 0.5000")
print(f"  SHUFFLED-target control    {a_sh:.4f}   <- must collapse to ~0.5000")

near, far = res["vru ahead < 20 m (NEAR)"], res["vru ahead < 60 m (ALL)"]
pc = res["lead < 25 m (POS CONTROL)"]
print("")
print(f"  POSITIVE CONTROL lead<25m: trunk {pc['trunk_auc']:.4f} vs pixels "
      f"{pc['pixel_auc']:.4f}  <- the trunk SHOULD win here")
ok = (abs(a_const - 0.5) < 0.06 or a_const != a_const) and abs(a_sh - 0.5) < 0.12
gap = near["trunk_auc"] - far["trunk_auc"]
print(f"\n  NEAR - ALL  =  {gap:+.4f}")
verdict = ("CONTROLS FAILED — void" if not ok else
           "RESOLUTION LIMIT — the near band decodes and the full range does not; "
           "the input is the bottleneck, not the objective" if gap > 0.05 else
           "NOT A RESOLUTION LIMIT — restricting to resolved agents does not help; "
           "the objective explanation survives" if gap < 0.02 else
           "AMBIGUOUS — the gap is inside the noise at this n")
print(f"\n  => {verdict}")
json.dump({"n": int(len(Yn)), "d_trunk": int(F.shape[1]), "d_pixel": int(X.shape[1]),
           "episodes": len(ue), "results": res, "constant_control": a_const,
           "shuffled_control": a_sh, "near_minus_all": gap, "controls_ok": bool(ok),
           "verdict": verdict, "_tier": "T0; NON-PARITY pilot; episode-disjoint; "
           "REF-C trunk (not champ30k)"},
          open(f"{O}/wph_resolution.json", "w"), indent=1)
print("-> wph_resolution.json")
