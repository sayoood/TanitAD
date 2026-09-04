"""E-SEED-2b — the SAME panel on the register's OWN `n_agents` definition.

⛔ WHY THIS RUN EXISTS. E-SEED-2's first pass defined `n_agents` as the raw count
of every `obstacle.offline` cuboid in the frame -- ALL classes, no field-of-view
filter (mean 18.17, max 83). That asks a 120-deg FRONT-CAMERA encoder to count
agents it cannot see: `build_obstacle_join.visibility_occ` MEASURES that a cuboid
is out of view iff `|atan2(cy, cx)| > 60 deg`, and it is bit-identical to
`bev_raster.fov_mask`. Frozen DINOv3 -- the reference arm, which the register
reads at +0.2754 -- came back at -0.0687 on that target, i.e. BELOW the constant
control, which is a property of the LABEL, not of the representation. That is the
E-DEC-23 failure repeated (`lead_gap_m`'s 80 m sentinel) and it is caught the same
way: by looking at the label before drawing a conclusion from it.

⭐ THE DEFINITION IS NOT RE-IMPLEMENTED HERE. `tanitad.data.psg_targets` is the
register's own instrument and it states (module docstring) that `n_agents` IS the
sum of the PSG count channel. This script IMPORTS `frame_target` and sums channel
0, so the target is the programme's, not mine.

Three targets are scored side by side so the definitional sensitivity is visible
rather than chosen:
  n_agents_psg  sum_col log1p(count_col) over the 8 in-FOV azimuth columns  <- canonical
  n_agents_fov  raw count of cuboids with occ == 0 (in view of the camera)
  n_agents_all  raw count of every cuboid            <- pass 1's target, kept

Features are the CACHED ones from `e_seed2_panel.py`; nothing is re-extracted, so
the arms are bit-identical across the two label sets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "clone_scripts"))
import e_seed2_panel as P                                         # noqa: E402

BANK = P.BANK
AGENTS = Path(r"C:\Users\Admin\tanitad-caches\val40-obstacle-20260818"
              r"\join\val40_agents.jsonl")


def build_labels() -> tuple[dict, dict]:
    from tanitad.data.psg_targets import frame_target
    meta = json.loads((BANK / "bank_meta.json").read_text(encoding="utf-8"))
    clip_ids = meta["clip_ids"]
    want = {c: i for i, c in enumerate(clip_ids)}
    per: dict[tuple[int, int], tuple[float, float, float]] = {}
    with open(AGENTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ci = want.get(r["clip_id"])
            if ci is None:
                continue
            ag = r.get("agents", [])
            tgt = frame_target(ag)                       # [8, 2]
            per[(ci, int(r["frame_idx"]))] = (
                float(tgt[:, 0].sum()),
                float(sum(1 for a in ag if int(a.get("occ", 1)) == 0)),
                float(len(ag)))
    rows = np.load(BANK / "rows.npy")
    out = np.empty((len(rows), 3), dtype=np.float64)
    for i, (ci, t) in enumerate(rows):
        out[i] = per[(int(ci), int(t))]
    return {"n_agents_psg": out[:, 0], "n_agents_fov": out[:, 1],
            "n_agents_all": out[:, 2]}, meta


def main() -> int:
    targets, meta = build_labels()
    L = np.load(BANK / "labels.npy")
    clip = np.load(BANK / "rows.npy")[:, 0]
    # cross-check: pass 1's label must reproduce n_agents_all exactly
    assert np.allclose(L[:, 0], targets["n_agents_all"]), "label join drifted"
    targets["speed"] = L[:, 1].astype(np.float64)
    for k, v in targets.items():
        print(f"[label] {k:13s} mean={v.mean():+.4f} std={v.std():.4f} "
              f"min={v.min():.3f} max={v.max():.3f}")

    res = {"meta": {**{k: v for k, v in meta.items() if k != "clip_ids"},
                    "arms": P.ARMS, "pca_k": P.PCA_K, "n_boot": P.N_BOOT,
                    "k_outer": P.K_OUTER, "k_inner": P.K_INNER,
                    "estimator": ("4-fold clip-disjoint out-of-fold prediction; "
                                  "pooled R2; clip-cluster bootstrap over 24 "
                                  "clips, 2000 draws; paired for contrasts"),
                    "n_agents_psg_definition":
                        ("sum over the 8 in-FOV azimuth columns of "
                         "log1p(count) -- tanitad.data.psg_targets.frame_target "
                         "channel 0, IMPORTED not re-implemented"),
                    "evidence_class": "MEASURED (ours; dev-box RTX 4060)"},
           "arms": {}, "preds": {}}
    for arm in P.ARMS:
        G = np.load(BANK / f"feat_{arm}.npy")
        S_ = np.load(BANK / f"featsp_{arm}.npy")
        blk = res["arms"].setdefault(arm, {})
        for space, F in (("global", G), ("spatial4x4", S_)):
            sb = blk.setdefault(space, {})
            for tname, y in targets.items():
                pred, lams, edges = P.oof_predict(F, y, clip)
                sb[tname] = {
                    "r2": P._r2(y, pred), "ci95": P.boot_ci(y, pred, clip),
                    "rho": float(np.corrcoef(y, pred)[0, 1]),
                    "lambdas": [float(x) for x in lams],
                    "lambda_at_grid_edge": bool(any(edges)),
                    "n": int(len(y)), "d_ambient": int(F.shape[1]),
                    "d_probe": P.PCA_K, "n_clusters": int(len(np.unique(clip)))}
                res["preds"][f"{arm}::{space}::{tname}"] = pred
            print(f"[{arm}/{space}] " + " | ".join(
                f"{t} {sb[t]['r2']:+.4f} [{sb[t]['ci95'][0]:+.3f},"
                f"{sb[t]['ci95'][1]:+.3f}]" for t in targets), flush=True)

    pairs = [("dino_hf", "scratch", "INSTRUMENT VALIDITY: can the probe see it?"),
             ("dino_hf", "pixel", "does DINOv3 beat the raw-input floor?"),
             ("seed_asis", "scratch", "is the seed AS WIRED better than random?"),
             ("seed_imnet", "scratch", "is the REPAIRED seed better than random?"),
             ("seed_imnet", "seed_asis", "does input normalisation help?"),
             ("dino_hf", "seed_imnet", "how much does the repaired seed lose?"),
             ("dino_hf", "seed_asis", "how much does the seed AS WIRED lose?"),
             ("seed_imnet", "pixel", "does the repaired seed beat raw input?"),
             ("seed_asis", "pixel", "does the seed as wired beat raw input?"),
             ("seed_imnet", "seed_imnet_pos0", "does the random `pos` help?")]
    con = {}
    for space in ("global", "spatial4x4"):
        for tname, y in targets.items():
            for a, b, q in pairs:
                con[f"{space}::{tname}::{a}-{b}"] = {
                    "question": q,
                    **P.paired_ci(y, res["preds"][f"{a}::{space}::{tname}"],
                                  res["preds"][f"{b}::{space}::{tname}"], clip)}
    res["contrasts"] = con
    res["controls"] = {
        "constant_only_r2": 0.0,
        "note": ("a predictor emitting the scored-set mean reads R2 = 0.0 "
                 "EXACTLY; every R2 above is against that same SST"),
        "deliberate_regression_arm": "scratch (random-init ViTEncoder)",
        "raw_input_floor_arm": "pixel"}
    (BANK / "eseed2b_panel.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "preds"}, indent=2),
        encoding="utf-8")
    np.savez_compressed(BANK / "eseed2b_preds.npz",
                        **{k: np.asarray(v) for k, v in res["preds"].items()},
                        clip=clip, **{k: v for k, v in targets.items()})
    print("\nwrote", BANK / "eseed2b_panel.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
