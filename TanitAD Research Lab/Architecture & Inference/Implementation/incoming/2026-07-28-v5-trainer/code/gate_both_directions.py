"""BOTH DIRECTIONS on the mid-run held-out gate — the numbers behind the tests.

⛔ A guard that cannot fail is worse than none (class C13). The suite asserts the
behaviour; this probe records the NUMBERS, so the claim "the gate caught it" is
quotable against an artifact rather than against a passing test.

Three records:

  1. STABLE arm  — 4 probes of an unchanged deployable surface: no stop.
  2. DEGRADED arm — after the incumbent probe the planner starts drifting
     2 m/s sideways (~4 m off the logged path at 2 s: out of the lane and not
     returning). Two consecutive separated-worse probes must STOP the run.
  3. ⚠️ THE FALSE DIRECTION I ALMOST SHIPPED — a SLOWDOWN (the first
     degradation tried) makes the composite go UP, because
     ``pseudosim.score_windows`` gives a barely-moving plan ``recovery = NaN``
     by construction ("standing still is not recovery"). Recorded because a
     failing test that fails for the wrong reason is the same defect one layer
     down, and this one was caught only by looking at the sign.

Everything travels through the REAL ``pseudo_evaluate`` -> composite -> paired
episode-cluster bootstrap. Nothing stubs ``observe``.

Run:  python gate_both_directions.py --out ../raw/gate_both_directions_<date>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[6]
for p in (str(REPO / "stack"), str(REPO / "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.train import heldout_gate as HG                     # noqa: E402


class _World(torch.nn.Module):
    def encode_window(self, frames):
        b = frames.shape[0]
        m = frames.reshape(b, -1).mean(-1)
        return m[:, None, None] * torch.ones(b, 4, 8)


class _Planner(torch.nn.Module):
    def __init__(self, horizons=tuple(range(1, 21))):
        super().__init__()
        self.cfg = type("C", (), {"horizons": horizons})()
        self.drift = 0.0          # m/s lateral departure
        self.decay = 1.0          # scale on forward motion

    def forward(self, states, v0, **kw):
        s = len(self.cfg.horizons)
        t = torch.arange(1, s + 1, dtype=torch.float32) * 0.1
        c = (states[:, -1, 0] - 0.5) * 2.0
        x = v0[:, None].float() * t[None] * self.decay
        y = c[:, None] * t[None] ** 2 + self.drift * t[None]
        return {"wp_seq": torch.stack([x, y], dim=-1)}


class _Ep:
    def __init__(self, T=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.frames = torch.rand(T, 1, 32, 32, generator=g)
        x = torch.arange(T).float() * 0.5
        self.poses = torch.stack([x, torch.zeros(T), torch.zeros(T),
                                  torch.full((T,), 5.0)], dim=-1)


def _gate():
    return HG.HeldoutGate(HG.HeldoutGateConfig(
        every=1, patience=2, n_boot=400, episodes=4, stride=8, batch=8))


def _row(rec):
    return {"step": rec["step"], "primary_value": rec["primary_value"],
            "n_windows": rec["n_windows"], "n_episodes": rec["n_episodes"],
            "separated_worse": rec.get("separated_worse"),
            "separated_better": rec.get("separated_better"),
            "worse_streak": rec.get("worse_streak"),
            "stop": rec.get("stop"), "paired": rec.get("paired")}


def run_arm(mutate=None, n_probes=3) -> dict:
    g, eps = _gate(), [_Ep(seed=i) for i in range(4)]
    world, head = _World(), _Planner()
    rows = []
    for s in range(n_probes):
        if s == 1 and mutate is not None:
            mutate(head)
        rows.append(_row(g.probe(s, world, head, eps, device="cpu")))
    return {"probes": rows, "stopped": bool(g.stop),
            "stop_reason": g.stop_reason,
            "admitted_components": g._pinned_admitted}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    stable = run_arm(None, n_probes=4)
    drifting = run_arm(lambda h: setattr(h, "drift", 2.0), n_probes=3)
    slowed = run_arm(lambda h: setattr(h, "decay", 0.1), n_probes=2)

    rec = {
        "primary": HG.PRIMARY_NAME,
        "refused_primary": HG.REFUSED_PRIMARY,
        "estimator": "paired episode-cluster bootstrap (B=400, unit = held-out "
                     "episode) — taniteval.ci.paired_episode_cluster_bootstrap",
        "evidence_class": "MEASURED (ours; synthetic held-out episodes, dev box)",
        "1_STABLE_arm_is_not_stopped": stable,
        "2_DEGRADED_arm_IS_stopped": drifting,
        "3_the_false_direction": {
            **slowed,
            "⚠️": "A SLOWDOWN makes the composite go UP. pseudosim gives a "
                  "barely-moving plan recovery=NaN by construction (the "
                  "progress-matched denominator, added because 'standing still "
                  "is not recovery'), so the surviving components favour it. "
                  "The first draft of the failing test used this and would have "
                  "PASSED-as-red for the wrong reason if the sign had not been "
                  "checked.",
        },
        "ALL_PASS": bool(
            not stable["stopped"]
            and drifting["stopped"]
            and drifting["probes"][1]["separated_worse"]
            and drifting["probes"][2]["separated_worse"]
            and HG.PRIMARY_NAME in (drifting["stop_reason"] or "")),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    return 0 if rec["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
