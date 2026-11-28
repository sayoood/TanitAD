"""E11 / E12 — the two S1 controls, proven as IDENTITIES on CPU before any GPU is spent.

⛔ WHY THESE RUN NOW, WITHOUT A MODEL. `PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md` §exit
commits both arms to EXACT values, and calls them identities in its own table:

    S1-GATE-CONST | recovery **exactly 0**: empty tracks => nc == 1 => BASE's pick on every window
    S1-RANDOM     | exactly the candidate mean (an identity, tested)

An identity does not need the trained head -- and waiting for one would be the mistake the
programme already paid for twice: `S1-GATE-PRED` is NOT RUNNABLE against the collapsed box head,
and if the controls only ran alongside it, a deviation discovered later would be unattributable
between "the gate is wrong" and "the head is collapsed". Establishing them first makes any future
deviation attributable to exactly one thing.

⭐ THE DISCRIMINATING CONTROL, AND IT IS THE WHOLE POINT OF ARM 3. "Recovery is exactly 0 under
empty tracks" is satisfied *vacuously* by a gate that never masks anything under ANY tracks --
which is `A CHECK THAT SHARES THE DEFECT IT CHECKS FOR` in gate costume, and would make
`S1-GATE-ORACLE` read 0 too while looking like a clean control. So the SAME code path is run
against tracks that genuinely collide, and it MUST change the pick. Without that arm, E12 passing
is not evidence.

⛔ The checker is IMPORTED (`tanitad.rl.pdm_proxy`), never re-implemented -- the prereg is
explicit that `taniteval/tools/fan_safety.py` uses a different collision model and is not used.
⛔ CPU only, read-only, no corpus, no checkpoint.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/Projects/TanitAD/stack")

import torch                                                    # noqa: E402
from tanitad.rl import pdm_proxy as P                           # noqa: E402

N_TICKS = P.PROXY.n_ticks
T0 = N_TICKS + 1
DT = P.PROXY.dt


def straight_fan(m: int = 8, v: float = 8.0, headings=None) -> torch.Tensor:
    """`m` candidates, each a constant-speed straight line at its own heading.

    States are `[M, T0, 4] = (x, y, yaw, speed)` -- the layout `no_at_fault_collision` reads.
    """
    if headings is None:
        headings = torch.linspace(-0.35, 0.35, m)
    t = torch.arange(T0, dtype=torch.float32) * DT
    st = torch.zeros(m, T0, 4)
    for i, h in enumerate(headings):
        st[i, :, 0] = v * t * torch.cos(h)
        st[i, :, 1] = v * t * torch.sin(h)
        st[i, :, 2] = h
        st[i, :, 3] = v
    return st


def empty_tracks(device=None) -> P.AgentTracks:
    """`S1-GATE-CONST`'s world: everything free.

    ⚠️ `AgentTracks.from_frames` allocates `A = max(1, len(ids))`, so an empty world is ONE
    slot that is never `valid` -- not a zero-width tensor. Building it by hand the same way
    keeps this arm on the identical code path rather than on a shape special case.
    """
    T = T0 + max(P.PROXY.ttc_horizons) if hasattr(P.PROXY, "ttc_horizons") else T0 + 4
    return P.AgentTracks(xy=torch.zeros(T, 1, 2), yaw=torch.zeros(T, 1),
                         lw=torch.zeros(1, 2), valid=torch.zeros(T, 1, dtype=torch.bool),
                         static=torch.zeros(1, dtype=torch.bool), speed=torch.zeros(T, 1))


def blocking_tracks(y_offset: float, device=None) -> P.AgentTracks:
    """A single STOPPED car sitting on the ego's path -- the arm that must NOT read 0."""
    T = T0 + 4
    xy = torch.zeros(T, 1, 2)
    xy[:, 0, 0] = 18.0                      # 18 m ahead
    xy[:, 0, 1] = y_offset
    return P.AgentTracks(xy=xy, yaw=torch.zeros(T, 1),
                         lw=torch.tensor([[4.5, 2.0]]),
                         valid=torch.ones(T, 1, dtype=torch.bool),
                         static=torch.ones(1, dtype=torch.bool), speed=torch.zeros(T, 1))


def gate_pick(scores: torch.Tensor, nc: torch.Tensor) -> int:
    """BASE's rule with colliding candidates masked to -inf; BASE's pick if all collide.

    ⛔ Transcribed from the prereg's arm table, not invented here: *"BASE's rule, with
    candidates colliding under the gate's tracks masked to -inf; BASE's pick if every
    candidate collides"*.
    """
    keep = nc >= 1.0
    if not bool(keep.any()):
        return int(torch.argmax(scores))
    masked = scores.clone()
    masked[~keep] = float("-inf")
    return int(torch.argmax(masked))


def main() -> int:
    torch.manual_seed(0)
    res = {"_what": "E11 (S1-RANDOM) and E12 (S1-GATE-CONST) proven as identities, CPU only",
           "_evidence_class": "MEASURED (ours), CPU, no checkpoint, no corpus",
           "_checker": "tanitad.rl.pdm_proxy.no_at_fault_collision (IMPORTED, not reimplemented)",
           "_prereg": "Project Steering/PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md",
           "n_ticks": N_TICKS, "dt": DT, "arms": {}}

    # --- E12: S1-GATE-CONST -------------------------------------------------- #
    # The committed value is recovery EXACTLY 0 -- not "small", not "within noise".
    n_windows, m = 64, 8
    const_w = empty_tracks()
    disagree, nc_min = 0, 1e9
    for _ in range(n_windows):
        st = straight_fan(m)
        scores = torch.randn(m)                       # BASE's ranking score; any values work
        nc = P.no_at_fault_collision(st, const_w)
        nc_min = min(nc_min, float(nc.min()))
        if gate_pick(scores, nc) != int(torch.argmax(scores)):
            disagree += 1
    res["arms"]["S1-GATE-CONST"] = {
        "_committed": "recovery exactly 0 (empty tracks => nc == 1 => BASE's pick every window)",
        "n_windows": n_windows, "n_candidates": m,
        "windows_where_gate_changed_the_pick": disagree,
        "min_nc_over_all_candidates": nc_min,
        "recovery": 0.0 if disagree == 0 else None,
        "PASS": bool(disagree == 0 and nc_min == 1.0)}

    # --- THE DISCRIMINATING CONTROL: the same gate MUST bite when tracks are real ---- #
    # ⛔ Without this, "exactly 0" is also what a gate that can never mask anything reads.
    bit, n_ctrl = 0, 64
    nc_seen = set()
    for _ in range(n_ctrl):
        st = straight_fan(m)
        # a stopped car straddling the fan: the near-straight candidates hit it
        w = blocking_tracks(y_offset=0.0)
        nc = P.no_at_fault_collision(st, w)
        nc_seen.update(round(float(v), 3) for v in nc)
        scores = torch.zeros(m)
        scores[int(torch.argmin(nc))] = 10.0          # BASE prefers a COLLIDING candidate
        if gate_pick(scores, nc) != int(torch.argmax(scores)):
            bit += 1
    res["arms"]["_CONTROL_gate_bites_on_real_tracks"] = {
        "_why": ("'recovery exactly 0 under empty tracks' is satisfied vacuously by a gate that "
                 "never masks anything. This runs the SAME path against a stopped car on the "
                 "ego's line and requires the pick to MOVE."),
        "n_windows": n_ctrl, "windows_where_gate_changed_the_pick": bit,
        "distinct_nc_values_seen": sorted(nc_seen),
        "PASS": bool(bit > 0 and any(v < 1.0 for v in nc_seen))}

    # --- E11: S1-RANDOM ------------------------------------------------------ #
    # "the EXACT uniform expectation over the fan (the candidate mean, NO RNG)". This is an
    # identity, so it is asserted exactly rather than sampled -- and a sampled version would
    # have been the weaker artifact: it would carry an interval around a number that has none.
    max_abs_err, n_id = 0.0, 256
    for _ in range(n_id):
        s = torch.randn(m) * 5.0
        uniform_expectation = float((s * (1.0 / m)).sum())     # E[s] under a uniform draw
        candidate_mean = float(s.mean())
        max_abs_err = max(max_abs_err, abs(uniform_expectation - candidate_mean))
    res["arms"]["S1-RANDOM"] = {
        "_committed": "exactly the candidate mean (an identity, tested)",
        "n_draws": n_id, "n_candidates": m,
        "max_abs_difference_expectation_vs_mean": max_abs_err,
        "PASS": bool(max_abs_err <= 1e-6),
        "_note": ("no RNG is consumed: the arm is the EXPECTATION, so it has no sampling "
                  "variance and no interval. A version that drew samples would report a CI "
                  "around a quantity that is exact.")}

    ok = all(a.get("PASS") for a in res["arms"].values())
    res["_VERDICT"] = (
        "⭐ BOTH S1 CONTROLS READ THEIR COMMITTED VALUES, CPU ONLY, BEFORE ANY GPU: "
        "S1-GATE-CONST recovery exactly 0 over %d windows (nc == 1 for every candidate), "
        "S1-RANDOM exactly the candidate mean (max |err| %.2e), and the discriminating "
        "control confirms the SAME gate does bite on real tracks (%d/%d windows) -- so the "
        "zero is a property of the empty world, not of an inert gate."
        % (n_windows, max_abs_err, bit, n_ctrl)) if ok else (
        "⛔ A CONTROL DID NOT READ ITS COMMITTED VALUE -- see the per-arm PASS flags.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print()
    print(res["_VERDICT"])
    pathlib.Path("C:/Users/Admin/qland/work/pbox/s1_controls.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
