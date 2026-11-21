"""PER-HEAD GRADIENT REACH with the BEV path live — MEASURED, never asserted.

PI RULING 2026-09-17 (R2/R3): the tactical behaviour decoder reads the MAP, and
its loss MAY shape the shared trunk. This instrument answers the only question
that matters about that wiring:

    Does a loss computed on the TACTICAL DECODER'S OUTPUT ALONE reach the
    ResNet trunk THROUGH THE BEV BRANCH?

⭐ THE ISOLATION, AND WHY IT NEEDS NO AGENT-OFF ARM. The loss is built from
``tacv6_*`` logits and NOTHING else — no planner term, no map term, no box
term. The BEV lift's only input is ``fmap_s16``, the trunk's own stride-16 map.
So a NON-ZERO ``grad_abs_sum`` on ``lift`` and ``bev_encoder`` under that loss
is a proof of reach that does not depend on separating two paths in one sum:
those two modules are not in the graph at all unless the tactical loss put them
there.

⛔ AND IT MUST BE ABLE TO READ ZERO. Three controls, each of which must read
EXACTLY 0.0 on the BEV modules:

  ``agent_only``   ``d_bev = 0``            — the pre-ruling arm. The BEV
                                              modules exist (``w_map`` is live)
                                              and the tactical loss cannot see
                                              them.
  ``bev_detached`` ``--tac-decoder-bev-detach`` — R3's ablation. The tokens
                                              reach the decoder; the gradient
                                              does not reach the trunk.
  ``planner_only`` no tactical term at all  — the same model, a loss that is
                                              not tactical. Reads 0 on the BEV
                                              modules and NON-ZERO on the
                                              trunk, so "0 everywhere" cannot
                                              be mistaken for a working probe.

Without the third, a rig that silently failed to backward at all would report
``0.0`` on the controls and look like a clean negative — the
``constant-control`` rule of CLAUDE.md, one instrument over.

⚠️ CPU-ONLY by construction (``CUDA_VISIBLE_DEVICES=""``): the GPU is held by an
RL arm. Nothing here is a capability claim and no metric family is computed —
this is a WIRING result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve()
while ROOT.name and not (ROOT / "stack").is_dir():
    ROOT = ROOT.parent
for _p in (str(ROOT / "stack"), str(ROOT / "stack" / "scripts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T                                        # noqa: E402
from tanitad.models import refcv6_perception_branch as PB        # noqa: E402
from tanitad.refs import refc_v3 as v3                           # noqa: E402

#: ⛔ NO GEOMETRY IS WRITTEN DOWN AS A BELIEF. The frame goes in as argv and
#: every shape downstream is read off the built objects. 256 x 1024 and not the
#: PI's 408 x 1024 for one MEASURED reason, reported as a blocker rather than
#: worked around: `TimmTrunkConfig.__post_init__` refuses any axis 32 does not
#: divide, and 408 % 32 == 8.
IMAGE_HW = ("256", "1024")

BASE = ["--arm", "hier", "--size", "tiny", "--out", "X",
        "--v7-labels", "labels.jsonl.gz",
        "--agents", "head", "--w-agent", "1.0",
        "--agent-join", "join.jsonl.xz", "--agent-join-verify", "off",
        "--tac-decoder-v6", "--w-tac-v6", "1.0",
        "--trunk", "timm", "--image-hw", *IMAGE_HW,
        "--no-trunk-pretrained",
        "--w-map", "1.0", "--map-gt-root", "m",
        "--agent-rig-camera", "extrinsics"]

ARMS = {
    "agent_only":   [],                                  # the pre-ruling arm
    "bev":          ["--tac-decoder-d-bev", "96"],       # the ruling
    "bev_detached": ["--tac-decoder-d-bev", "96",
                     "--tac-decoder-bev-detach"],        # R3's ablation
}


def _extrinsics(tmp: Path, n: int) -> Path:
    """A per-clip mount-pose table. ⛔ Per clip, never one camera: the lift
    back-projects through the road plane and the corpus's MEASURED mount height
    spans 1.2131-1.6672 m over 554 distinct values in 2,400 clips."""
    p = tmp / "extr.json"
    p.write_text(json.dumps({
        "clip%02d" % i: {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                         "tx": 2.05, "ty": 0.0, "tz": 1.30 + 0.01 * i}
        for i in range(n)}), encoding="utf-8")
    return p


def build(arm: str, tmp: Path, n_clips: int = 4):
    """A REAL model through the REAL pin — never a hand-assembled config."""
    argv = BASE + ["--agent-rig-extrinsics", str(_extrinsics(tmp, n_clips))] \
        + ARMS[arm]
    args = T.build_parser().parse_args(argv)
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    # an unrelated v3 precondition the trainer sets on its own launch path;
    # identical on every arm here, so it cannot separate them.
    cfg.core.graft_target_latent = True
    model = v3.RefCV3Model(cfg)
    pcfg = PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0)
    model._perception = PB.build_perception_branch(model, pcfg)
    model._w_map, model._w_box3d = 1.0, 0.0
    frame = PB.frame_for_model(model)
    bank = PB.LiftGeometryBank(
        {("clip%02d" % i): e for i, e in enumerate(
            T._read_rig_extrinsics(str(_extrinsics(tmp, n_clips)))[1].values())},
        frame=frame, stride=int(pcfg.stride))
    return model, cfg, args, bank


def tactical_loss(out: dict) -> torch.Tensor:
    """⛔ THE TACTICAL TERM AND NOTHING ELSE. No planner, no map, no box — the
    whole isolation argument rests on this function reading exactly four
    tensors, all of them the decoder's own output."""
    keys = ("tacv6_goal_logits", "tacv6_lat_logits", "tacv6_lon_logits",
            "tacv6_goal_conf")
    have = [k for k in keys if torch.is_tensor(out.get(k))]
    if not have:
        raise SystemExit(
            "[grad-reach] the forward emitted no `tacv6_*` logits: the "
            "behaviour decoder did not run, so a 0.0 reading below would be "
            "about the RIG, not about the wiring.")
    return sum((out[k].float() ** 2).mean() for k in have), have


def planner_loss(out: dict) -> torch.Tensor:
    """The THIRD control's loss: real, non-tactical, and it must move the
    trunk. Without it, "0.0 on the BEV modules" is indistinguishable from "this
    script never ran a backward"."""
    return (out["traj"].float() ** 2).mean()


def per_behaviour_bev_sensitivity(out: dict) -> dict | None:
    """⭐ THE BRIEF'S OWN QUESTION: *do map-derived behaviours (lane keeping,
    corridor offset) now receive gradient?*

    For each of the 22 v7 tactical goal tokens ``j``, this takes
    ``d(goal_logits[:, j].sum()) / d(bev_tokens)`` and reports its absolute
    sum. A token whose logit does not depend on the map reads EXACTLY 0.

    ⛔ THIS IS A WIRING MEASUREMENT, NOT A CAPABILITY ONE. A non-zero here says
    the map CAN shape that behaviour's logit — that evidence now exists where
    under agent-only it structurally did not. It says NOTHING about whether the
    head learns to use it; that needs a trained arm and a per-class result, and
    no GPU arm was run.
    """
    from tanitad.models.vocab_v7 import TACTICAL_GOAL_TOKENS_V7
    p = out.get("perception") or {}
    bt = p.get("bev_tokens")
    gl = out.get("tacv6_goal_logits")
    # ⛔ THREE DIFFERENT ABSENCES, NAMED. A bare `None` would collapse "no map
    # was fed" (agent-only: the structural state R2 lifted) into "a map was fed
    # with its graph cut" (R3's ablation) into "the decoder did not run" — and
    # those are three different findings. Reporting one string for all three is
    # how an absence claim becomes unreadable.
    if not torch.is_tensor(gl):
        return {"_status": "NO-DECODER-OUTPUT"}
    if not torch.is_tensor(bt):
        return {"_status": "NO-BEV-TOKENS-FED (agent-only)"}
    if not bt.requires_grad:
        return {"_status": "BEV-TOKENS-DETACHED (R3 ablation): no gradient "
                           "path from any behaviour logit to the map"}
    n = min(int(gl.shape[-1]), len(TACTICAL_GOAL_TOKENS_V7))
    rows = {}
    for j in range(n):
        g, = torch.autograd.grad(gl[..., j].sum(), bt, retain_graph=True,
                                 allow_unused=True)
        rows[TACTICAL_GOAL_TOKENS_V7[j]] = (
            0.0 if g is None else float(g.detach().abs().sum()))
    return rows


def run(arm: str, tmp: Path, seed: int, use_planner_loss: bool = False) -> dict:
    torch.manual_seed(seed)
    model, cfg, args, bank = build(arm, tmp)
    model.train()
    b, w = 2, int(cfg.core.window)
    enc = cfg.core.encoder
    h, ww = enc.image_hw()
    frames = torch.rand(b, w, int(enc.in_channels), int(h), int(ww))
    # ⛔ The bank keys on `stable_episode_id(clip_id)`, NOT on a row index. A
    # positional id would refuse (it did, first run) — and if it had silently
    # fallen back to a default camera instead, every BEV cell below would have
    # been filled through the wrong road plane while the numbers still looked
    # healthy. The refusal is the right behaviour; this reads the real ids.
    eps = sorted(bank.clip_of)[:b]
    grid, valid = bank.for_episodes(eps)
    # ⚠️ NO `ego_state=`: this build does not carry the v4 lever and the forward
    # REFUSES the tensor rather than dropping it. The behaviour decoder's
    # condition then carries zeros in its v0/a0 slots — which changes nothing
    # about GRADIENT REACH, the only question this instrument asks, and is
    # identical on all four arms.
    out = model(frames, nav_cmd=torch.zeros(b, dtype=torch.long),
                v0=torch.tensor([5.0, 8.0]),
                perception_grid=grid, perception_valid=valid)
    if use_planner_loss:
        loss, terms = planner_loss(out), ["traj"]
    else:
        loss, terms = tactical_loss(out)
    per_tok = (None if use_planner_loss
               else per_behaviour_bev_sensitivity(out))
    model.zero_grad(set_to_none=True)
    loss.backward()
    rep = PB.grad_reach_report(model)
    p = out.get("perception") or {}
    return {
        "arm": arm,
        "loss_terms": terms,
        "loss": float(loss.detach()),
        "perception_keys": sorted(p),
        "bev_tokens_shape": (list(p["bev_tokens"].shape)
                             if torch.is_tensor(p.get("bev_tokens")) else None),
        "bev_token_dim": int(model._perception.bev_token_dim),
        "n_scene": (out["tacv6_n_scene"].tolist()
                    if torch.is_tensor(out.get("tacv6_n_scene")) else None),
        "decoder_sources": list(cfg.tac_decoder_cfg.sources),
        "bev_detached": bool(cfg.tac_decoder_bev_detach),
        "bev_tokens_fed": p.get("bev_tokens_fed"),
        "per_behaviour_dlogit_dbev_abs": per_tok,
        "grad": {k: {"grad_abs_sum": v["grad_abs_sum"],
                     "n_params": v["n_params"],
                     "n_params_with_grad": v["n_params_with_grad"]}
                 for k, v in rep.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    rows = [run(k, tmp, a.seed) for k in ARMS]
    rows.append({**run("bev", tmp, a.seed, use_planner_loss=True),
                 "arm": "bev_PLANNER_LOSS_CONTROL"})

    by = {r["arm"]: r for r in rows}
    # ⛔ `map_head` IS NOT ON THIS PATH, and putting it here was my own error on
    # the first run — the predicate read False on a correctly wired rig.
    # `MapHead` is a SIBLING consumer of `bev_feats`, not an ancestor of the
    # decoder's tokens: `BEVMapBranch.forward` returns `{bev_feats, map_logits}`
    # and the tokens are pooled from `bev_feats`. So a tactical-only loss MUST
    # leave the map head at exactly 0, and does. The modules the tactical loss
    # must move are the two that sit BETWEEN the trunk and the tokens.
    bev_mods = ("lift", "bev_encoder")

    def ga(arm, mod):
        return by[arm]["grad"].get(mod, {}).get("grad_abs_sum")

    verdict = {
        # ⭐ THE CLAIM: a tactical-only loss moves the BEV branch, hence the
        # trunk. ⛔ Asserted as a POSITIVE number, and the three controls below
        # are what make it readable.
        "tactical_loss_reaches_bev_branch": all(
            (ga("bev", m) or 0.0) > 0.0 for m in bev_mods),
        "tactical_loss_reaches_trunk_on_bev_arm": (ga("bev", "trunk") or 0.0) > 0.0,
        # CONTROL 1 — the pre-ruling arm cannot see the BEV modules.
        "agent_only_bev_grad_is_exactly_zero": all(
            ga("agent_only", m) == 0.0 for m in bev_mods),
        # CONTROL 2 — R3's ablation cuts the path.
        "detached_bev_grad_is_exactly_zero": all(
            ga("bev_detached", m) == 0.0 for m in bev_mods),
        # CONTROL 3 — the probe can read a non-zero when it should.
        "planner_loss_moves_trunk": (
            ga("bev_PLANNER_LOSS_CONTROL", "trunk") or 0.0) > 0.0,
        "planner_loss_leaves_bev_branch_at_zero": all(
            ga("bev_PLANNER_LOSS_CONTROL", m) == 0.0 for m in bev_mods),
        # ⭐ the decoder ATTENDED to them, not merely received them
        "n_scene_rises_with_bev": (
            by["bev"]["n_scene"] is not None
            and by["agent_only"]["n_scene"] is not None
            and min(by["bev"]["n_scene"]) > max(by["agent_only"]["n_scene"])),
        # ⭐ THE SHARPEST NUMBER IN THE TABLE, and it was not the one I set out
        # to measure. `n_params_with_grad` on the TRUNK is how much of the
        # backbone the tactical loss can touch. The planner reads stride 32, so
        # its gradient never enters the stride-16-only parameters; the BEV lift
        # reads stride 16, so it does. The DIFFERENCE is the part of the trunk
        # that R3 newly opens to the tactical layer.
        "trunk_params_with_grad": {
            a: by[a]["grad"]["trunk"]["n_params_with_grad"] for a in by},
        "trunk_params_opened_by_bev_path": (
            by["bev"]["grad"]["trunk"]["n_params_with_grad"]
            - by["agent_only"]["grad"]["trunk"]["n_params_with_grad"]),
        # ⛔ The map head must read EXACTLY 0 under a tactical-only loss on
        # every arm — it is a sibling of the tokens, not an ancestor. Stated as
        # a positive expectation so a future change that wired it in would be
        # VISIBLE rather than quietly welcome.
        "map_head_is_zero_under_tactical_loss_on_every_arm": all(
            r["grad"]["map_head"]["grad_abs_sum"] == 0.0
            for r in rows if "map_head" in r["grad"]),
        # ⭐ THE BRIEF'S QUESTION, ANSWERED PER TOKEN. `FOLLOW_LANE` is lane
        # keeping and `CORRIDOR_OFFSET` is the corridor offset — the two
        # behaviours §E8 named as UNINTERPRETABLE under agent-only.
        # ⛔ REACHABILITY, NOT SELECTIVITY. At initialisation the 22 values are
        # near-uniform (an untrained cross-attention attends roughly evenly), so
        # this says the map CAN shape each behaviour's logit — not that any head
        # has learned to use it. That needs a trained arm; none was run.
        "n_behaviours_with_a_map_gradient": (
            None if not isinstance(
                by["bev"].get("per_behaviour_dlogit_dbev_abs"), dict)
            or "_status" in by["bev"]["per_behaviour_dlogit_dbev_abs"] else
            sum(1 for v in by["bev"]["per_behaviour_dlogit_dbev_abs"].values()
                if v > 0.0)),
        "map_derived_behaviours_reachable": {
            k: by["bev"].get("per_behaviour_dlogit_dbev_abs", {}).get(k)
            for k in ("FOLLOW_LANE", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR")},
    }
    # ⛔ A head with parameters and grad_abs_sum EXACTLY 0 is the
    # `tac_goal_tok_head` class (11,286 params at 0 for 40,284 steps). Named,
    # per arm, so the report cannot quietly omit one.
    verdict["heads_at_exactly_zero"] = {
        r["arm"]: sorted(k for k, g in r["grad"].items()
                         if g["n_params"] > 0 and g["grad_abs_sum"] == 0.0)
        for r in rows}
    doc = {"arms": rows, "verdict": verdict, "image_hw": list(IMAGE_HW),
           "device": "cpu", "torch": torch.__version__}
    Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
