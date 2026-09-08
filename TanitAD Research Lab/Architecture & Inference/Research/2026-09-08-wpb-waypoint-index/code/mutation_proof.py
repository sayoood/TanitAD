"""refcv5 WP-B / ``E-WP-INDEX-1`` — THE MUTATION PROOF.

⛔⛔ **A GUARD THAT CANNOT GO RED PROVES NOTHING.** This script runs the WP-B
guard set twice over: once on the TRUE implementation, where every guard must
read GREEN, and once per MUTATION, where the historical defect is reintroduced
and at least one guard must read RED.

⭐ **The mutations are the real defects, not synthetic ones.** ``M1`` is the
defect the whole work package is written against ("index by array position
instead of by geometry"); ``M2`` is the range/bearing transposition from the
``anchors.pt`` units retraction, one axis over; ``M7`` is a bug that was
actually present in the first draft of ``apply_radius_gate`` and was caught by
``G9`` before it shipped.

⛔ The guards assert **LITERALS** — ``6.0``, ``sqrt(436) - 20``, ``e**6``,
``[3, 3, 0]`` — computed by hand or by ``math``, never read back from the code
under test. Re-running a producer's own derivation and finding agreement
measures determinism, not correctness (`CLAUDE.md`, `e4af94f`).

Usage::

    python mutation_proof.py --out <dir>

writes ``mutation_log.json`` (the machine record) and ``MUTATION_LOG.md``
(the readable one) into ``<dir>``, and exits non-zero if the true
implementation is not all-GREEN or if any mutation fails to turn a guard RED.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import traceback

import torch

from tanitad.refs import refc_wp_index as wi

# --------------------------------------------------------------------------
# the scene (identical to stack/tests/test_wp_index.py — one definition, two
# consumers, so a change to the scene cannot make one of them silently stale)
# --------------------------------------------------------------------------
STRAIGHT = torch.tensor([[[[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]]]])
LEFT = torch.tensor([[[[5.0, 0.5], [10.0, 1.5], [15.0, 3.5], [20.0, 6.0]]]])
AGENTS = torch.tensor([[[20.0, 0.0], [20.0, 6.0], [5.0, 0.0]]])


# --------------------------------------------------------------------------
# THE GUARDS — every expectation a literal
# --------------------------------------------------------------------------
def g1_analytic_metric_literals():
    g = wi.waypoint_agent_geometry(STRAIGHT, AGENTS)
    assert g["d_min"].reshape(-1).tolist() == [0.0, 6.0, 0.0]
    assert g["s_star"].reshape(-1).tolist() == [3, 3, 0]
    assert g["tau"].reshape(-1).tolist() == [1.0, 1.0, 0.0]
    assert g["lat"].reshape(-1).tolist() == [0.0, 6.0, 0.0]
    assert g["lon"].reshape(-1).tolist() == [0.0, 0.0, 0.0]


def g2_polar_literals_from_math():
    g = wi.waypoint_agent_geometry(STRAIGHT, AGENTS)
    r = math.sqrt(436.0)
    assert abs(float(g["d_range"][0, 0, 1]) - (r - 20.0)) < 1e-4
    assert abs(float(g["cos_db"][0, 0, 1]) - 20.0 / r) < 1e-6
    assert abs(float(g["sin_db"][0, 0, 1]) - 6.0 / r) < 1e-6


def g3_the_addressed_slot_moves_with_the_plan():
    gs = wi.waypoint_agent_geometry(STRAIGHT, AGENTS)
    gl = wi.waypoint_agent_geometry(LEFT, AGENTS)
    assert int(gs["d_min"].argmin()) == 0
    assert int(gl["d_min"].argmin()) == 1


def g4_attention_ratio_is_exp_of_the_gap():
    cfg = wi.WaypointIndexConfig(enable=True, hidden=4, scale_m=10.0)
    head = wi.WaypointIndexBias(cfg, n_heads=1)
    with torch.no_grad():
        head.mlp[0].weight.zero_(); head.mlp[0].bias.zero_()
        head.mlp[0].weight[0, 0] = 100.0
        head.mlp[2].weight.zero_(); head.mlp[2].bias.zero_()
        head.mlp[2].weight[0, 0] = -1.0
    rel, _ = wi.build_relation(STRAIGHT, AGENTS[:, :2], cfg)
    bias = head(rel)
    gap = float(bias[0, 0, 0] - bias[0, 0, 1])
    assert gap > 0.0
    with torch.no_grad():
        head.mlp[2].weight[0, 0] = -1.0 * (6.0 / gap)
    w = torch.softmax(head(rel)[0, 0], dim=-1)
    assert abs(float(w[0] / w[1]) - 403.4287934927351) < 0.5


def g5_const_control_is_identical_across_anchors():
    wp = torch.randn(2, 5, 4, 2) * 10.0
    pos = torch.randn(2, 7, 2) * 10.0
    cfg = wi.WaypointIndexConfig(enable=True, mode="const")
    rel, _ = wi.build_relation(wp, pos, cfg)
    head = wi.WaypointIndexBias(cfg, n_heads=2)
    with torch.no_grad():
        torch.nn.init.normal_(head.mlp[-1].weight, std=1.0)
        torch.nn.init.normal_(head.mlp[-1].bias, std=1.0)
    bias = head(rel).reshape(2, 2, 5, 7)
    for n in range(5):
        assert torch.equal(bias[:, :, n], bias[:, :, 0])
    rel_g, _ = wi.build_relation(wp, pos, wi.WaypointIndexConfig(enable=True))
    bias_g = head(rel_g).reshape(2, 2, 5, 7)
    assert not torch.equal(bias_g[:, :, 1], bias_g[:, :, 0])


def g6_shuffle_is_the_partner_rows_address_and_refuses_batch_one():
    torch.manual_seed(0)
    wp = torch.randn(4, 3, 4, 2) * 10.0
    pos = torch.randn(4, 5, 2) * 10.0
    rel_s, _ = wi.build_relation(
        wp, pos, wi.WaypointIndexConfig(enable=True, mode="shuffle",
                                        shuffle_seed=3))
    wp_p, perm = wi.shuffle_waypoints(wp, 3)
    rel_ref, _ = wi.build_relation(wp_p, pos,
                                   wi.WaypointIndexConfig(enable=True))
    assert torch.equal(rel_s, rel_ref)
    assert not torch.equal(perm, torch.arange(4))
    try:
        wi.build_relation(torch.randn(1, 3, 4, 2), torch.randn(1, 5, 2),
                          wi.WaypointIndexConfig(enable=True, mode="shuffle"))
    except ValueError:
        return
    raise AssertionError("shuffle at B=1 did not refuse")


def g7_detach_severs_the_graph():
    wp = (torch.randn(2, 3, 4, 2) * 10.0).requires_grad_(True)
    pos = (torch.randn(2, 5, 2) * 10.0).requires_grad_(True)
    rel, _ = wi.build_relation(wp, pos, wi.WaypointIndexConfig(enable=True))
    assert rel.requires_grad
    rel.sum().backward()
    assert float(wp.grad.abs().sum()) > 0.0
    assert float(pos.grad.abs().sum()) > 0.0
    wp2 = (torch.randn(2, 3, 4, 2) * 10.0).requires_grad_(True)
    pos2 = (torch.randn(2, 5, 2) * 10.0).requires_grad_(True)
    rel2, _ = wi.build_relation(
        wp2, pos2, wi.WaypointIndexConfig(enable=True, detach=True))
    assert rel2.requires_grad is False and rel2.grad_fn is None
    assert wp2.grad is None and pos2.grad is None


def g8_degenerate_scene_is_finite():
    rel, d = wi.build_relation(torch.zeros(1, 2, 4, 2), torch.zeros(1, 3, 2),
                               wi.WaypointIndexConfig(enable=True))
    assert bool(torch.isfinite(rel).all()) and bool(torch.isfinite(d).all())


def g9_radius_gate_never_empties_a_row_and_counts_padding():
    bias = torch.zeros(2, 2, 3)
    d_min = torch.tensor([[[50.0, 60.0, 70.0], [1.0, 60.0, 70.0]],
                          [[50.0, 60.0, 0.5], [2.0, 3.0, 4.0]]])
    pad = torch.tensor([[False, False, True], [False, False, True]])
    out = wi.apply_radius_gate(bias, d_min, 8.0, 1, pad)
    fin = torch.isfinite(out)
    assert fin[0, 0].tolist() == [True, True, True]
    assert fin[1, 0].tolist() == [True, True, True]      # padding-aware
    assert fin[0, 1].tolist() == [True, False, False]
    assert bool(fin.any(dim=-1).all())


def g10_bias_shape_is_BH_N_M():
    cfg = wi.WaypointIndexConfig(enable=True)
    rel, _ = wi.build_relation(torch.randn(2, 5, 4, 2), torch.randn(2, 7, 2),
                               cfg)
    assert tuple(rel.shape) == (2, 5, 7, 8)
    assert tuple(wi.WaypointIndexBias(cfg, 3)(rel).shape) == (6, 5, 7)


def g11_output_layer_is_zero_init():
    head = wi.WaypointIndexBias(wi.WaypointIndexConfig(enable=True), 4)
    assert float(head.mlp[-1].weight.abs().sum()) == 0.0
    assert float(head.mlp[-1].bias.abs().sum()) == 0.0


def g12_first_waypoint_heading_is_measured_from_the_EGO_ORIGIN():
    """⛔ THE HOLE THE MUTATION PROOF FOUND. Every other guard uses a
    STRAIGHT plan, whose per-step heading is constant — so borrowing step 1's
    heading for step 0 changes nothing and ``M6`` went undetected.

    The discriminating scene is a TURNING plan with the closest approach at
    ``s* = 0``. ``LEFT``'s first step is ``(0,0) -> (5, 0.5)``; an agent at
    ``(5, 1.5)`` is ``1.0`` m from waypoint 0 and 5.0 m from waypoint 1, so
    ``s* = 0`` and ``delta = (0, 1)``. Then, with ``|(5, 0.5)| = sqrt(25.25)``:

        lon = delta . h      = 0.5 / sqrt(25.25)
        lat = delta . h_perp = 5.0 / sqrt(25.25)

    Borrowing step 1's heading ``(5, 1.0)/sqrt(26)`` gives ``0.19612`` /
    ``0.98058`` instead — both plausible, neither equal to the literal.
    """
    agent = torch.tensor([[[5.0, 1.5]]])
    g = wi.waypoint_agent_geometry(LEFT, agent)
    r = math.sqrt(25.25)
    assert int(g["s_star"]) == 0
    assert abs(float(g["d_min"]) - 1.0) < 1e-5
    assert abs(float(g["lon"]) - 0.5 / r) < 1e-5
    assert abs(float(g["lat"]) - 5.0 / r) < 1e-5


GUARDS = {
    "G1_analytic_metric_literals": g1_analytic_metric_literals,
    "G2_polar_literals_from_math": g2_polar_literals_from_math,
    "G3_addressed_slot_moves_with_the_plan": g3_the_addressed_slot_moves_with_the_plan,
    "G4_attention_ratio_is_exp_of_the_gap": g4_attention_ratio_is_exp_of_the_gap,
    "G5_const_control_identical_across_anchors": g5_const_control_is_identical_across_anchors,
    "G6_shuffle_identity_and_batch1_refusal": g6_shuffle_is_the_partner_rows_address_and_refuses_batch_one,
    "G7_detach_severs_the_graph": g7_detach_severs_the_graph,
    "G8_degenerate_scene_is_finite": g8_degenerate_scene_is_finite,
    "G9_radius_gate_row_and_padding": g9_radius_gate_never_empties_a_row_and_counts_padding,
    "G10_bias_shape_BH_N_M": g10_bias_shape_is_BH_N_M,
    "G11_output_layer_zero_init": g11_output_layer_is_zero_init,
    "G12_first_waypoint_heading_from_origin": g12_first_waypoint_heading_is_measured_from_the_EGO_ORIGIN,
}


# --------------------------------------------------------------------------
# THE MUTATIONS — each reintroduces a REAL defect
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _patch(name, value):
    old = getattr(wi, name)
    setattr(wi, name, value)
    try:
        yield
    finally:
        setattr(wi, name, old)


def m1_index_by_array_position():
    """⛔ THE DEFECT THE WHOLE WORK PACKAGE IS WRITTEN AGAINST: address the
    agent by its ARRAY POSITION instead of by the trajectory's geometry."""
    def geom(wp, pos):
        b, n = wp.shape[0], wp.shape[1]
        m = pos.shape[1]
        idx = torch.arange(m, dtype=wp.dtype).reshape(1, 1, m).expand(b, n, m)
        z = torch.zeros(b, n, m, dtype=wp.dtype)
        return {"d_min": idx.contiguous(),
                "s_star": torch.zeros(b, n, m, dtype=torch.long),
                "tau": z, "lon": z, "lat": z, "d_range": z,
                "cos_db": z + 1.0, "sin_db": z}
    return _patch("waypoint_agent_geometry", geom)


def m2_swap_x_and_y():
    """⛔ The range/bearing transposition — the `anchors.pt` units family."""
    real = wi.waypoint_agent_geometry

    def geom(wp, pos):
        return real(wp, pos.flip(-1))
    return _patch("waypoint_agent_geometry", geom)


def m3_ignore_the_detach_flag():
    """⛔ The detached CONTROL silently becomes the treatment."""
    real = wi.build_relation

    def build(wp, pos, cfg):
        if cfg.detach:
            cfg = wi.WaypointIndexConfig(
                enable=cfg.enable, hidden=cfg.hidden, scale_m=cfg.scale_m,
                mode=cfg.mode, const_xy=cfg.const_xy, detach=False,
                radius_m=cfg.radius_m, shuffle_seed=cfg.shuffle_seed)
        return real(wp, pos, cfg)
    return _patch("build_relation", build)


def m4_transpose_the_bias():
    """⛔ ``[B*H, M, N]`` instead of ``[B*H, N, M]`` — a SILENT mis-index
    whenever N == M, which is why the guard scene uses N != M."""
    class Bias(wi.WaypointIndexBias):
        def forward(self, rel):
            b, n, m, _ = rel.shape
            out = self.mlp(rel)
            return out.permute(0, 3, 2, 1).reshape(b * self.n_heads, m, n)
    return _patch("WaypointIndexBias", Bias)


def m5_nonzero_output_init():
    """⛔ The zero-init IS the removability guarantee, not decoration."""
    class Bias(wi.WaypointIndexBias):
        def __init__(self, cfg, n_heads):
            super().__init__(cfg, n_heads)
            torch.nn.init.normal_(self.mlp[-1].weight, std=0.5)
            torch.nn.init.normal_(self.mlp[-1].bias, std=0.5)
    return _patch("WaypointIndexBias", Bias)


def m6_drop_the_origin_prepend():
    """⛔ The heading off-by-one: waypoint 0 borrows waypoint 1's step."""
    def geom(wp, pos):
        b, n, s, _ = wp.shape
        m = pos.shape[1]
        pos = pos.to(wp.dtype)
        delta = pos[:, None, None, :, :] - wp[:, :, :, None, :]
        dist = torch.linalg.vector_norm(delta, dim=-1)
        d_min, s_star = dist.min(dim=2)
        step = wp[:, :, 1:, :] - wp[:, :, :-1, :]          # NO origin
        step = torch.cat([step[:, :, :1], step], dim=2)
        head = step / torch.linalg.vector_norm(
            step, dim=-1, keepdim=True).clamp_min(1e-6)
        idx = s_star[..., None].expand(b, n, m, 2)
        h = torch.gather(head, 2, idx)
        w_star = torch.gather(wp, 2, idx)
        d_star = pos[:, None, :, :] - w_star
        lon = (d_star * h).sum(-1)
        lat = -d_star[..., 0] * h[..., 1] + d_star[..., 1] * h[..., 0]
        r_a = torch.linalg.vector_norm(pos, dim=-1)[:, None, :]
        r_w = torch.linalg.vector_norm(w_star, dim=-1)
        den = (r_a * r_w).clamp_min(1e-6)
        a_x, a_y = pos[:, None, :, 0], pos[:, None, :, 1]
        return {"d_min": d_min, "s_star": s_star,
                "tau": s_star.to(wp.dtype) / float(max(s - 1, 1)),
                "lon": lon, "lat": lat,
                "d_range": r_a.expand_as(r_w) - r_w,
                "cos_db": (w_star[..., 0] * a_x + w_star[..., 1] * a_y) / den,
                "sin_db": (w_star[..., 0] * a_y - w_star[..., 1] * a_x) / den}
    return _patch("waypoint_agent_geometry", geom)


def m7_radius_gate_ignores_padding():
    """⛔ THE BUG THAT WAS ACTUALLY IN THE FIRST DRAFT. A query whose only
    in-radius slot is PADDING looks non-empty to a radius-only test, and every
    unit of attention mass then lands on a slot that does not exist."""
    def gate(bias, d_min, radius_m, n_heads, pad=None):
        if radius_m <= 0.0:
            return bias
        b, n, m = d_min.shape
        drop = d_min > float(radius_m)
        empty = drop.all(dim=-1)                    # <- padding NOT considered
        if bool(empty.any()):
            drop = drop & ~empty[..., None]
        drop = drop[:, None].expand(b, int(n_heads), n, m).reshape(
            b * int(n_heads), n, m)
        return bias.masked_fill(drop, float("-inf"))
    return _patch("apply_radius_gate", gate)


def m8_shuffle_accepts_batch_one():
    """⛔ The control silently becomes the treatment at B = 1."""
    def shuf(wp, seed=0):
        b = wp.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(b, generator=gen).to(wp.device)
        return wp.index_select(0, perm), perm
    return _patch("shuffle_waypoints", shuf)


def m9_no_denominator_clamp():
    """⛔ NaN on a padded slot at the origin — and NaN + (-inf) is NaN, so one
    padded slot poisons the whole attention row including the live keys."""
    def geom(wp, pos):
        b, n, s, _ = wp.shape
        m = pos.shape[1]
        pos = pos.to(wp.dtype)
        delta = pos[:, None, None, :, :] - wp[:, :, :, None, :]
        dist = torch.linalg.vector_norm(delta, dim=-1)
        d_min, s_star = dist.min(dim=2)
        wp0 = torch.cat([wp.new_zeros(b, n, 1, 2), wp], dim=2)
        step = wp0[:, :, 1:, :] - wp0[:, :, :-1, :]
        head = step / torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        idx = s_star[..., None].expand(b, n, m, 2)
        h = torch.gather(head, 2, idx)
        w_star = torch.gather(wp, 2, idx)
        d_star = pos[:, None, :, :] - w_star
        r_a = torch.linalg.vector_norm(pos, dim=-1)[:, None, :]
        r_w = torch.linalg.vector_norm(w_star, dim=-1)
        den = r_a * r_w                                   # <- NO clamp
        a_x, a_y = pos[:, None, :, 0], pos[:, None, :, 1]
        return {"d_min": d_min, "s_star": s_star,
                "tau": s_star.to(wp.dtype) / float(max(s - 1, 1)),
                "lon": (d_star * h).sum(-1),
                "lat": -d_star[..., 0] * h[..., 1] + d_star[..., 1] * h[..., 0],
                "d_range": r_a.expand_as(r_w) - r_w,
                "cos_db": (w_star[..., 0] * a_x + w_star[..., 1] * a_y) / den,
                "sin_db": (w_star[..., 0] * a_y - w_star[..., 1] * a_x) / den}
    return _patch("waypoint_agent_geometry", geom)


MUTATIONS = {
    "M1_index_by_array_position": m1_index_by_array_position,
    "M2_swap_x_and_y": m2_swap_x_and_y,
    "M3_ignore_the_detach_flag": m3_ignore_the_detach_flag,
    "M4_transpose_the_bias": m4_transpose_the_bias,
    "M5_nonzero_output_init": m5_nonzero_output_init,
    "M6_drop_the_origin_prepend": m6_drop_the_origin_prepend,
    "M7_radius_gate_ignores_padding": m7_radius_gate_ignores_padding,
    "M8_shuffle_accepts_batch_one": m8_shuffle_accepts_batch_one,
    "M9_no_denominator_clamp": m9_no_denominator_clamp,
}


def run_guards() -> dict:
    out = {}
    for name, fn in GUARDS.items():
        torch.manual_seed(1234)
        try:
            fn()
            out[name] = {"verdict": "GREEN", "error": None}
        except Exception as exc:                                  # noqa: BLE001
            out[name] = {"verdict": "RED",
                         "error": f"{type(exc).__name__}: {exc}"[:400]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    baseline = run_guards()
    rec = {"torch": torch.__version__,
           "python": sys.version.split()[0],
           "n_guards": len(GUARDS), "n_mutations": len(MUTATIONS),
           "baseline": baseline, "mutations": {}}

    ok = all(v["verdict"] == "GREEN" for v in baseline.values())
    problems = []
    if not ok:
        problems.append("baseline is not all-GREEN: "
                        + ", ".join(k for k, v in baseline.items()
                                    if v["verdict"] == "RED"))

    for mname, mfn in MUTATIONS.items():
        with mfn():
            res = run_guards()
        red = sorted(k for k, v in res.items() if v["verdict"] == "RED")
        rec["mutations"][mname] = {"red": red, "detail": res}
        if not red:
            problems.append(f"{mname} turned NO guard red -- the guard set "
                            f"cannot see this defect")

    rec["problems"] = problems
    with open(os.path.join(args.out, "mutation_log.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)

    lines = [
        "# WP-B mutation proof — `E-WP-INDEX-1`", "",
        f"`torch {torch.__version__}` · `python {rec['python']}` · "
        f"**n_guards = {len(GUARDS)}** · **n_mutations = {len(MUTATIONS)}** · "
        "dev-box CPU (RTX 4060 box, CPU path), float32.", "",
        "## Baseline — the TRUE implementation", "",
        "| guard | verdict |", "|---|---|",
    ]
    for k, v in baseline.items():
        lines.append(f"| `{k}` | **{v['verdict']}** |")
    lines += ["", "## Mutations — each reintroduces a REAL defect", "",
              "⛔ A guard that cannot go RED proves nothing. Every row below "
              "must name at least one RED guard.", "",
              "| mutation | guards that went RED |", "|---|---|"]
    for k, v in rec["mutations"].items():
        red = ", ".join(f"`{r}`" for r in v["red"]) or "**NONE — DEFECT**"
        lines.append(f"| `{k}` | {red} |")
    lines += ["", "## Verdict", "",
              ("**PASS** — baseline all-GREEN and every mutation turns at "
               "least one guard RED." if not problems
               else "**FAIL**\n\n" + "\n".join(f"* {p}" for p in problems)),
              ""]
    with open(os.path.join(args.out, "MUTATION_LOG.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                             # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
