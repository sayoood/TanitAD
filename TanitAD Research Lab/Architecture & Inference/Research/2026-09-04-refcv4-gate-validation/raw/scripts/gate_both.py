"""The refcv4 anchor gate, both vocabularies, one artifact, one surface.

Surface = the banked per-window dump of the refcv3 step-40,284 open-loop suite:
n = 4,823 windows / 141 episodes, K = 4 instants (0.5/1.0/1.5/2.0 s), ego frame
at t0, metres.  This is the SAME surface `ha` = 0.2996 m was measured on, so the
comparison is paired rather than cross-surface.
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "refcv3_40284_dump")

eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
acc = {k: [] for k in ("g", "ha", "ha0", "os", "oracle_sel", "v0")}
EPI = []
for i, f in enumerate(eps):
    z = np.load(os.path.join(DUMP, f))
    for k in acc:
        acc[k].append(z[k])
    EPI.append(np.full(z["g"].shape[0], i, np.int64))
D = {k: np.concatenate(v).astype(np.float64) for k, v in acc.items()}
EPI = np.concatenate(EPI)
G = D["g"]


def ade(p):
    return np.linalg.norm(p - G, axis=-1).mean(1)


def boot(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(EPI)
    by = [np.where(EPI == e)[0] for e in uq]
    d = a - b
    out = np.array([d[np.concatenate([by[p] for p in
                                      rng.integers(0, len(uq), len(uq))])].mean()
                    for _ in range(n_boot)])
    return float(d.mean()), *(float(x) for x in np.percentile(out, [2.5, 97.5]))


def oiv(path):
    A = torch.load(path, map_location="cpu", weights_only=True).double().numpy()[:, :4]
    dm = np.linalg.norm(G[:, None] - A[None], axis=-1).mean(2)
    i = dm.argmin(1)
    r = A[i] - G
    return (dm[np.arange(len(i)), i], np.abs(r[..., 0]).mean(1),
            np.abs(r[..., 1]).mean(1), len(np.unique(i)))


HA = ade(D["ha"])
rows = {}
for nm, p in (("refcv3_synthetic", "refcv3_anchors.pt"),
              ("refcv4_new", "anchors_pod.pt")):
    a, al, la, nd = oiv(os.path.join(HERE, p))
    d, lo, hi = boot(a, HA)
    rows[nm] = {"oracle_in_vocab_ade_m": float(a.mean()),
                "along_mae_m": float(al.mean()), "lat_mae_m": float(la.mean()),
                "distinct_anchors_used": nd,
                "vs_ha_delta_m": d, "vs_ha_ci95": [lo, hi],
                "beats_ha": bool(hi < 0)}

art = {
    "artifact_kind": "tanitad.refcv4_anchor_gate_recompute",
    "generated_utc": "2026-09-04",
    "evidence_class": "MEASURED (ours)",
    "tier": "T0 — a vocabulary/geometry property, not a driving number",
    "surface": {
        "source": "taniteval/results/refcv3-40284-openloop-dump.tar.gz",
        "windows": int(len(G)), "episodes": int(len(eps)),
        "instants_s": [0.5, 1.0, 1.5, 2.0], "frame": "ego at t0", "units": "m",
        "window_stride": 5},
    "estimator": "paired episode-cluster bootstrap, n_boot 2000, seed 0, "
                 "cluster = episode",
    "controls": {
        "panel_reproduced_from_dump": {k: float(ade(D[k]).mean())
                                       for k in ("ha", "ha0", "os", "oracle_sel")},
        "panel_published": {"ha": 0.2996, "ha0": 0.6723, "os": 0.4419,
                            "oracle_sel": 0.3668},
        "zero_path_no_information_ade_m": float(np.linalg.norm(G, axis=-1).mean()),
        "ha0_equals_constant_velocity_max_abs_err_m": float(np.abs(
            D["ha0"] - np.stack([D["v0"][:, None] * np.array([.5, 1, 1.5, 2]),
                                 np.zeros((len(G), 4))], -1)).max())},
    "gate_as_issued": {"threshold_m": 0.2996, "quantity": "oracle-in-vocabulary "
                       "(raw min-over-anchors ADE, no model)"},
    "arms": rows,
    "why_the_threshold_is_mis_scoped": {
        "refcv3_raw_oracle_in_vocab_m": rows["refcv3_synthetic"]["oracle_in_vocab_ade_m"],
        "refcv3_refined_oracle_sel_m": float(ade(D["oracle_sel"]).mean()),
        "refinement_gain_m": float(rows["refcv3_synthetic"]["oracle_in_vocab_ade_m"]
                                   - ade(D["oracle_sel"]).mean()),
        "mechanism": "refc.py:1400 `x = anchors[None] + offset`; offset is a "
                     "free-form linear head (refc.py `_decode`: "
                     "`self.offset_head(q).reshape(b, n, n_steps, 2)`), "
                     "unclamped and unscaled. The emitted fan `anchor_traj` is "
                     "the REFINED fan, so the raw vocabulary is an "
                     "INITIALISATION, not a ceiling.",
        "consequence": "the 0.2996 bar was justified by `oracle_sel` (0.3668), "
                       "a MODEL-INCLUSIVE quantity, and applied to the raw "
                       "MODEL-FREE quantity. Applied that way it would also "
                       "have aborted refcv3, whose raw vocabulary reads 1.0838 "
                       "and whose refined oracle nonetheless reads 0.3668."},
}
json.dump(art, open(os.path.join(HERE, "REFCV4_ANCHOR_GATE.json"), "w"), indent=1)
for k, v in rows.items():
    print(f"{k:<20s} OIV {v['oracle_in_vocab_ade_m']:.4f}  "
          f"along {v['along_mae_m']:.4f}  lat {v['lat_mae_m']:.4f}  "
          f"vs ha {v['vs_ha_delta_m']:+.4f} [{v['vs_ha_ci95'][0]:+.4f}, "
          f"{v['vs_ha_ci95'][1]:+.4f}]  beats_ha={v['beats_ha']}")
print("refinement gain (refcv3): "
      f"{art['why_the_threshold_is_mis_scoped']['refinement_gain_m']:.4f} m")
print("wrote REFCV4_ANCHOR_GATE.json")
