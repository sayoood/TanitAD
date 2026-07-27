#!/usr/bin/env python3
"""E-H2 GATE — everything that must PASS before any swept cell is quotable.

Four gates, in the order a wrong answer would do the most damage:

G1  BIT-IDENTITY.  ``graft_lambda = 1, graft_tau = 1`` reproduces the SHIPPED
    ``_factor_grafts`` arithmetic bit-for-bit.  The reference is the pre-change
    expression re-implemented here, not the new code compared to itself.  A
    non-identical baseline would invalidate all 42 cells.
G2  COMMITTED BARS.  0.8563 / 0.2505 / 0.6423 / 0.8781 / 0.7620 recomputed from
    the staged per-window tensors, plus the paired ``H_graft(64) - flat`` delta
    re-run end-to-end on the decision-grade estimator.
G3  NO TRUNCATION.  The deployment path contains no ``q``: no top-k, no mask, no
    -inf, at any (lambda, tau).  Verified by source scan AND by execution.
G4  base_rank SEMANTICS.  The staged ``base_rank`` is
    ``[as-trained pick] ++ [anchor index order]``, NOT a score ranking -- checked
    on all 881 rows, because a V5 label depends on it.

CPU only.  Inputs are staged artifacts + ``stack/``.  No pod, no GPU, no network.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
sys.path.insert(0, str(_REPO / "stack"))
sys.path.insert(0, str(_REPO / "taniteval"))
from taniteval.ci import (episode_cluster_bootstrap,               # noqa: E402
                          paired_episode_cluster_bootstrap)
from tanitad.models.flagship_v4 import (N_DIST, N_LAT, N_LON,      # noqa: E402
                                        FlagshipV4Head, V4Config)
from tanitad.models.metric_dynamics import grad_scale              # noqa: E402

_V5 = (_REPO / "TanitAD Research Hub" / "Architecture & Inference" /
       "Implementation" / "incoming" / "2026-07-26-v5-imagination-selection")
HIER = _V5 / "raw" / "v5_hier_windows.pt"
RED = _V5 / "raw" / "v5_v4_windows_reduced.pt"
COMMITTED = {"produced|F_flat": 0.8563, "produced|O_oracle_in_fan": 0.2505,
             "produced|F_base_only": 0.8781, "neutral|F_flat": 0.7620,
             "oracle|F_flat": 0.6423, "oracle|F_base_only": 0.6615}
B = 2000


# --------------------------------------------------------------------- G1 ---
def _small() -> V4Config:
    from tanitad.refs.refc import DecoderConfig
    cfg = V4Config()
    cfg.state_dim, cfg.readout_grid, cfg.d_cell, cfg.window = 64, 4, 4, 4
    cfg.horizons, cfg.imag_read = (1, 2, 3, 4), (1, 2)
    cfg.n_anchors, cfg.d_token, cfg.d_meas, cfg.n_probes = 12, 16, 8, 2
    cfg.factor_hidden = 8
    cfg.decoder = DecoderConfig(d=16, n_heads=2, layers=2, ff_mult=2,
                                aux_hidden=16, diffusion_steps=2, noise_std=0.1)
    return cfg


def _batch(cfg, b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"states": torch.randn(b, cfg.window, cfg.state_dim, generator=g),
            "v0": torch.rand(b, generator=g) * 20 + 3,
            "imagined": torch.randn(b, cfg.n_probes * len(cfg.imag_read),
                                    cfg.state_dim, generator=g),
            "vt_band": torch.randint(0, 23, (b,), generator=g),
            "route": torch.randint(0, 4, (b,), generator=g),
            "route_graded": torch.randn(b, generator=g)}


def _run(head, b, **kw):
    return head(b["states"], b["v0"], imagined=b["imagined"],
                vt_band=b["vt_band"], route=b["route"],
                route_graded=b["route_graded"], vt_speed=b["v0"], **kw)


def gate_bit_identity() -> dict:
    cfg = _small()
    cfg.seam_fail = 100.0
    head = FlagshipV4Head(cfg).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():                      # zero-init grafts hide a broken λ
        for lin in (head.lat_to_anchor, head.lon_to_anchor, head.dist_to_anchor):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.05)
    b = _batch(cfg)
    out = _run(head, b)

    tokens = head.build_tokens(grad_scale(b["states"], 1.0), b["imagined"])
    m, _, _ = head.condition(b["v0"], b["vt_band"], b["route"], b["route_graded"])
    refined = head.decoder(tokens, m,
                           steps=cfg.decoder.diffusion_steps)["refined_logits"]
    lsm = torch.log_softmax
    graft = (head.lat_to_anchor(lsm(out["lat_logits"], dim=-1))
             + head.lon_to_anchor(lsm(out["lon_logits"], dim=-1))
             + head.dist_to_anchor(lsm(out["dist_logits"], dim=-1)))
    ratio = graft.norm(dim=-1) / refined.norm(dim=-1).clamp_min(1e-9)
    shipped = refined + graft * (cfg.seam_clamp
                                 / ratio.clamp_min(cfg.seam_clamp))[:, None]
    ident = bool(torch.equal(out["refined_logits"], shipped))

    # λ=0 must reproduce the PRE-graft score exactly (the F_base_only arm)
    head.cfg.graft_lambda = 0.0
    l0 = bool(torch.equal(_run(head, b)["refined_logits"], refined))
    head.cfg.graft_lambda = 1.0
    return {
        "lambda1_tau1_bit_identical_to_shipped": ident,
        "max_abs_diff": float((out["refined_logits"] - shipped).abs().max()),
        "lambda0_reproduces_pre_graft_score_exactly": l0,
        "PASS": ident and l0,
        "_why": "x/1.0 and 1.0*x are EXACT in IEEE-754, so the two divisions and "
                "one multiply the knobs add are provably no-ops at the defaults. "
                "Asserted, not assumed.",
        "_also_pinned_in": "stack/tests/test_flagship_v4.py::"
                           "test_lambda_one_tau_one_is_bit_identical_to_the_"
                           "shipped_graft_path",
    }


# --------------------------------------------------------------------- G3 ---
def gate_no_truncation() -> dict:
    """`q` must not exist in the deployed selector, at ANY prior strength."""
    # Scan the SELECTION PATH ITSELF, not the whole module: a whole-file scan
    # false-positives on the two `masked_fill` calls in `condition()`, which drop
    # the GOAL TOKEN during training and have nothing to do with the candidate
    # set. Scanning the wrong scope would have produced a fake failure here.
    import inspect
    from tanitad.models.flagship_v15 import FlagshipV15Head
    sel_path = {
        "FlagshipV4Head._factor_grafts":
            inspect.getsource(FlagshipV4Head._factor_grafts),
        "FlagshipV15Head.select": inspect.getsource(FlagshipV15Head.select),
    }
    banned = ["topk", "top_k", "hierarchical_pick", "masked_fill", "scatter_",
              "-inf", "argsort", "[:q]", "[:, :q]"]
    hits = {w: {k: v.count(w) for k, v in sel_path.items() if w in v}
            for w in banned}
    hits = {w: v for w, v in hits.items() if v}
    _accounted = {
        "masked_fill x2 in FlagshipV15Head.condition": "goal-token dropout during "
        "TRAINING (vt_band -> VT_DROPPED, route -> ROUTE_DROPPED). Masks the "
        "CONDITION, never the candidate set; not in the selection path.",
    }
    cfg = _small()
    cfg.seam_clamp, cfg.seam_fail = 1.0e6, 1.0e9
    head = FlagshipV4Head(cfg).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for lin in (head.lat_to_anchor, head.lon_to_anchor, head.dist_to_anchor):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.05)
    b = _batch(cfg)
    runs = {}
    for lam, tau in ((0.0, 1.0), (1.0, 1.0), (8.0, 1.0), (1.0, 0.1), (8.0, 0.1)):
        head.cfg.graft_lambda, head.cfg.graft_tau = lam, tau
        o = _run(head, b)
        runs[f"lam{lam}_tau{tau}"] = {
            "n_candidates_scored": int(o["sel_score"].shape[1]),
            "all_finite": bool(torch.isfinite(o["sel_score"]).all()),
            "pick_is_flat_argmax": bool(torch.equal(
                o["sel_idx"], o["sel_score"].argmax(dim=1)))}
    ok = (not hits
          and all(v["n_candidates_scored"] == cfg.n_anchors
                  and v["all_finite"] and v["pick_is_flat_argmax"]
                  for v in runs.values()))
    return {"selection_path_scanned": sorted(sel_path),
            "banned_token_hits_in_the_selection_path": hits,
            "hits_elsewhere_accounted_for": _accounted,
            "executed": runs, "PASS": ok,
            "_finding": "`q` was NEVER in the deployment path — it exists only in "
                        "the E-V5-2 MEASUREMENT harness "
                        "(v5_hierarchical_select.py::hierarchical_pick). "
                        "'Delete q from the deployment path' is therefore a "
                        "VERIFICATION plus a regression guard, not a deletion.",
            "_guard": "stack/tests/test_flagship_v4.py::"
                      "test_the_selector_never_truncates_the_candidate_set"}


# --------------------------------------------------------------------- G2 ---
def gate_committed_bars() -> dict:
    H = torch.load(HIER, map_location="cpu", weights_only=False)
    R = torch.load(RED, map_location="cpu", weights_only=False)
    eid = np.asarray([str(int(e)) for e in R["ep"]])
    got, ok = {}, True
    for k, want in COMMITTED.items():
        if k not in H:
            got[k] = {"committed": want, "recomputed": None,
                      "note": "not in the staged dump"}
            continue
        v = float(H[k].double().mean())
        hit = abs(round(v, 4) - want) < 5e-5
        ok = ok and hit
        got[k] = {"committed": want, "recomputed": round(v, 4), "match": hit}
    paired = paired_episode_cluster_bootstrap(
        H["produced|H_graft_q64"].double().numpy(),
        H["produced|F_flat"].double().numpy(), eid, n_boot=B)
    flat_ci = episode_cluster_bootstrap(
        H["produced|F_flat"].double().numpy(), eid, n_boot=B)
    align = bool(torch.allclose(H["produced|F_flat"],
                                R["ade_by_arm"]["A0_as_trained"], atol=1e-6))
    return {"bars": got, "windows": int(H["produced|F_flat"].numel()),
            "episodes": int(len(np.unique(eid))),
            "H_graft_q64_minus_flat_paired": paired,
            "produced_F_flat_interval": flat_ci,
            "hier_and_reduced_dumps_are_the_SAME_881_windows": align,
            "PASS": bool(ok and align),
            "_estimator": "episode-cluster bootstrap, B=%d, unit = episode "
                          "cluster. NEVER overlapping_holdout_se." % B}


# --------------------------------------------------------------------- G4 ---
def gate_base_rank_semantics() -> dict:
    """``base_rank`` is ``[pick] ++ [anchor index order]`` -- verify on all rows."""
    R = torch.load(RED, map_location="cpu", weights_only=False)
    br, sel0 = R["base_rank"], R["ref_sel_idx"]
    W, N = br.shape
    order = torch.arange(N).repeat(W, 1)
    cat = torch.cat([sel0[:, None], order], dim=1)
    keep = torch.ones(W, N + 1, dtype=torch.bool)
    keep.scatter_(1, (sel0 + 1)[:, None], False)
    keep[:, 0] = True
    reconstructed = cat[keep].reshape(W, N)
    rows = int((br == reconstructed).all(dim=1).sum())
    # a real score ranking would NOT be monotone in anchor index after column 0
    tail_is_index_order = int((br[:, 1:].diff(dim=1) > 0).all(dim=1).sum())
    err4 = R["fan_err4"]
    n1 = float(err4.gather(1, br[:, :1]).mean())
    return {"rows_matching_[pick]++[index order]": rows, "n_rows": W,
            "rows_whose_tail_is_strictly_increasing_anchor_index":
                tail_is_index_order,
            "column0_reproduces_F_flat": round(n1, 4),
            "PASS": bool(rows == W),
            "_correction": "`base_rank` is NOT a rank. V5 section 5.2's label "
                           "'top-n by the as-trained base ranking' is wrong; the "
                           "columns are '[the as-trained pick] ++ [anchors in "
                           "INDEX order]'. Its CONCLUSIONS survive — index order "
                           "is still a nested family — but the label does not.",
            "_root_cause_class": "a tensor's semantics taken from its NAME "
                                 "rather than from its construction site "
                                 "(v5_cost_curve.py:109-122)"}


def main() -> None:
    t0 = time.time()
    R = {"_experiment": "E-H2 gate — bit-identity, committed bars, no-truncation, "
                        "base_rank semantics",
         "_evidence_class": "MEASURED (ours)", "_host": platform.node(),
         "_python": platform.python_version(), "_torch": torch.__version__,
         "_device": "cpu (dev box; no pod contacted, no GPU used)"}
    R["G1_bit_identity"] = gate_bit_identity()
    R["G2_committed_bars"] = gate_committed_bars()
    R["G3_no_truncation"] = gate_no_truncation()
    R["G4_base_rank_semantics"] = gate_base_rank_semantics()
    R["ALL_PASS"] = all(R[k]["PASS"] for k in R if k.startswith("G"))
    R["_wallclock_s"] = round(time.time() - t0, 1)
    out = _HERE.parent.parent / "raw" / "eh2_gate.json"
    out.write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps({k: (v["PASS"] if isinstance(v, dict) and "PASS" in v else v)
                      for k, v in R.items() if k.startswith(("G", "ALL"))},
                     indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
