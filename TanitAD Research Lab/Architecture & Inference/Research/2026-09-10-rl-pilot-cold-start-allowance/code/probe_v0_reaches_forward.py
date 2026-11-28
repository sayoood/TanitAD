"""PROBE: does ``v0`` change the pilot's production forward, and does the RL
conditioning contract know?

⛔ WHY THIS EXISTS. Found while closing PI DECISION QUEUE **item 8** (the cold
start's missing ``anchor_controls``); it is evidence for **item 7** (the RL
pilot's adapter), which the queue records as *"pending an agent verdict"*.

The claim under test is ``refcv3_adapter.RefCConditioningContract``'s: it
resolves, per build, which conditioning channels this model *"was TRAINED
with"*, and refuses a batch that drops one. Its ``v0`` predicate is
``anchors.v0_conditioned`` OR ``sel_reach_clamp`` (``refcv3_adapter.py:240``),
and ``refc.refc_config()`` leaves **both False**.

⭐ THE HYPOTHESIS THIS PROBE REFUTED, RECORDED BECAUSE IT WAS WRONG AND CHEAP TO
CHECK. I first reasoned — from the contract's own stated reason, which cites
``refc.py:3097`` → ``:2128`` → ``:1722`` — that with both flags False ``v0``
must be entirely unused, i.e. the pilot's build is speed-blind. **MEASURED: it
is not.** ``v0`` changes ``anchor_traj``. So the contract is not requiring a
channel that does nothing; it is FAILING TO REQUIRE a channel that does
something, through a path its predicates do not cover.

⭐ THE PATH THE PREDICATES MISS, read from source: ``refc.py:3199-3200`` builds
the measurement encoder's input UNCONDITIONALLY —

    v = zeros(b, 1) if v0 is None else (v0 / 10.0).reshape(b, 1)

— with no reference to either flag. And ``refc.py:3204-3206`` derives ``keep``
(the X15 ego-validity channel) from ``v0 is not None``, so a batch that simply
OMITS ``v0`` does not merely lose the speed: it asserts ``keep = 0``. The same
file calls that the *"X15 zero-collision"* — a withheld speed made
indistinguishable from a genuinely stationary car.

⚠️ WHAT THIS PROBE DOES **NOT** CLAIM. It does not claim the contract is wrong
about the ACTION SPACE: with ``v0_conditioned=False``, ``roll_bank``
(``refc.py:1816``) really does return the stored ``anchors`` unchanged, so the
vocabulary is untouched. The finding is narrower and it is about SCOPE: the
guard's refusal is keyed on the action-space path only, while the forward has a
second, always-live consumer, so a dropped ``v0`` changes the POLICY with
nothing raising.

⛔ It changes no behaviour. Deciding whether ``v0`` should become REQUIRED is
item 7's call and belongs to that stream — this only supplies the measurement.

Controls (both mandatory; a probe with neither is unfalsifiable):
  * a **v0-conditioned** build with a real control vocabulary MUST show a
    difference — if it does not, the probe is broken, not the model;
  * ``eval()`` and ``torch.no_grad()`` throughout, and the SAME frames tensor
    for every arm, so the only thing varying is ``v0``. ⚠️ ``ego_dropout``
    defaults to 0.5 and is TRAINING-only, so a probe left in train mode would
    read a dropout draw as a speed effect.
"""
from __future__ import annotations

import argparse
import io
import json

import torch

from tanitad.refs import refc
from tanitad.rl.config import PostTrainConfig
from tanitad.rl.refcv3_adapter import make_refcv3_sample_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    cfg = refc.refc_config()
    m = refc.RefCModel(cfg).eval()          # ⛔ eval: ego_dropout is train-only
    b = a.batch
    w, ch, px = int(cfg.window), int(cfg.encoder.in_channels), \
        int(cfg.encoder.image_size)
    g = torch.Generator().manual_seed(a.seed + 1)
    frames = torch.rand(b, w, ch, px, px, generator=g)

    def fwd(v0):
        with torch.no_grad():
            return m(frames, None, v0, steps=2)["anchor_traj"]

    lo = fwd(torch.full((b,), 5.0))
    hi = fwd(torch.full((b,), 25.0))
    none = fwd(None)
    same = fwd(torch.full((b,), 5.0))

    rep = {
        "_what": "does v0 change the production forward, and does the RL "
                 "conditioning contract require it?",
        "_evidence_class": "MEASURED (ours; this run)",
        "config": {"v0_conditioned": bool(cfg.anchors.v0_conditioned),
                   "sel_reach_clamp": bool(cfg.sel_reach_clamp),
                   "graft_lan": bool(cfg.graft_lan),
                   "ego_dropout": float(cfg.ego_dropout)},
        "anchor_traj_shape": list(lo.shape),
        # ⭐ THE DETERMINISM CONTROL: the same v0 twice must be BIT-IDENTICAL,
        # or every "difference" below is just sampling noise.
        "control_determinism_bitwise_equal": bool(torch.equal(lo, same)),
        "v0_5_vs_25_max_abs_m": float((lo - hi).abs().max()),
        "v0_5_vs_DROPPED_max_abs_m": float((lo - none).abs().max()),
    }

    # ---- the contract's own answer ---------------------------------------- #
    pcfg = PostTrainConfig(method="grpo", group_size=2, steps=1, batch=b,
                           trainable_prefixes=("decoder",), w_imitation=0.0)
    sample_fn = make_refcv3_sample_fn(m, pcfg, build_ctx=lambda bt, out=None: {},
                                      strict_conditioning=True)
    contract = sample_fn.conditioning_contract
    batch = {"frames": frames, "v0": torch.full((b,), 5.0)}
    contract.check(batch)                    # ⛔ resolve BEFORE reading the map
    assert contract.resolutions == 1
    reqmap = contract.requirements
    assert reqmap, "empty requirement map = a FAILED READ, not a finding"
    rep["contract"] = {
        "requirement_map": reqmap,
        "required": sorted(contract.required),
        "v0_required": bool(reqmap.get("v0")),
    }
    try:
        contract.check({"frames": frames})   # v0 DROPPED
        rep["contract"]["drop_v0_verdict"] = "PASSED — no refusal"
    except Exception as exc:                 # noqa: BLE001
        rep["contract"]["drop_v0_verdict"] = f"REFUSED: {type(exc).__name__}"

    # ---- CONTROL: a v0-conditioned build MUST move ------------------------ #
    cfg2 = refc.refc_config()
    cfg2.anchors.v0_conditioned = True
    torch.manual_seed(a.seed)
    m2 = refc.RefCModel(cfg2).eval()
    with torch.no_grad():
        m2.decoder.anchor_controls.copy_(
            torch.randn(m2.decoder.anchor_controls.shape,
                        generator=torch.Generator().manual_seed(7)) * 0.3)
        p = m2(frames, None, torch.full((b,), 5.0), steps=2)["anchor_traj"]
        q = m2(frames, None, torch.full((b,), 25.0), steps=2)["anchor_traj"]
    rep["control_v0_conditioned_max_abs_m"] = float((p - q).abs().max())
    rep["control_v0_conditioned_differs"] = bool(not torch.equal(p, q))

    ok = (rep["control_determinism_bitwise_equal"]
          and rep["control_v0_conditioned_differs"])
    rep["controls_valid"] = ok
    rep["verdict"] = (
        "v0 REACHES the forward through the measurement encoder "
        "(refc.py:3199) on a build where the contract does NOT require it"
        if (ok and rep["v0_5_vs_DROPPED_max_abs_m"] > 0
            and not rep["contract"]["v0_required"])
        else "INCONCLUSIVE — read the controls")

    print(json.dumps(rep, indent=1, sort_keys=True))
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="").write(
            json.dumps(rep, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
