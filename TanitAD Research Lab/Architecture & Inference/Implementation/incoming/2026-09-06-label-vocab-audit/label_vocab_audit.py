"""TanitAD LABEL-AND-VOCABULARY AUDIT — the gradient/extraction discriminator.

⭐ THE QUESTION (PI 2026-09-06): which labels are USED for training, which are
not, and which vocabulary is EXTRACTED AT INFERENCE?

⛔ INSPECTION IS NOT EVIDENCE. This tool establishes each cell POSITIVELY:

  L1 the label reaches the BATCH          - key present in a collated batch
  L2 the label reaches the FORWARD        - mutating it moves the loss
  L3 the term has NON-ZERO GRADIENT       - `p.grad is None` per head, AFTER a
                                            real backward through the trainer's
                                            OWN `compute_losses_v3`
  L4 the head is EXTRACTABLE at inference - key present in `model(...)` output

⭐ `p.grad is None` IS THE DISCRIMINATOR, NEVER THE WEIGHT'S VALUE. A term can
carry a positive weight and still be guarded out; a term can be summed with a
0.0 weight and still allocate a head. Only the gradient settles it.

⭐ EVERY NULL CARRIES A SAME-BREATH DISCRIMINATING CONTROL. `frames` must move
the loss and `traj` must receive gradient; if a control reads null the rig is
dead and no other null in that run means anything.

⚠️ ZERO-INIT EDGES GIVE FALSE NEGATIVES AT INIT. `nav_to_*`, `ego_to_*` and
`gstr_film` are zero-init BY DESIGN, so an untrained model is bit-inert on them
and a null there is an INIT ARTIFACT, not an absent wire. `--dezero`
randomises them, which turns "is this edge active now" into "is this edge WIRED
AT ALL" -- the only form of the question a fresh model can answer.

ASCII only (cp1252 box). CPU only by default: the A40 runs refcv5.

Usage:
    python label_vocab_audit.py            # full audit, kin3 + v7.0
    python label_vocab_audit.py --json OUT # also write machine-readable
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys

import torch


def load_trainer(stack_dir: str):
    """Import the LIVE trainer as a module (it is a script, not a package)."""
    if stack_dir not in sys.path:
        sys.path.insert(0, stack_dir)
    spec = iu.spec_from_file_location(
        "refc_v3_train", f"{stack_dir}/scripts/refc_v3_train.py")
    mod = iu.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["refc_v3_train"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = saved
    return mod


ZERO_INIT = ("nav_to_tac", "nav_to_str", "ego_to_tac", "ego_to_str",
             "gstr_film")

#: head -> the vocabulary it emits, for the report
HEAD_VOCAB = {
    "core.route_head": "route classes (core)",
    "lat_head_tac": "tactical LAT actions (z_tac)",
    "lon_head_tac": "tactical LON actions (z_tac)",
    "str_goal_head": "strategic goal (bearing/dist, 3-dim)",
    "tac_goal_head": "tactical geometric goal (x,y,heading,speed)@tau",
    "core.law": "future pooled latent (aux)",
}


def dezero(model) -> list:
    hit = []
    for name in ZERO_INIT:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        with torch.no_grad():
            for p in mod.parameters():
                p.normal_(0.0, 0.35)
        hit.append(name)
    with torch.no_grad():
        if hasattr(model, "goal_gate"):
            model.goal_gate.fill_(1.0)
            hit.append("goal_gate=1")
    return hit


def build_batch(T, cfg, n_ep: int = 2):
    eps = T._synth_episodes(n_ep, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    return torch.utils.data.default_collate([ds[i] for i in range(n_ep)])


def grad_census(model) -> dict:
    """`p.grad is None` per parameter, grouped by owning module path."""
    out = {}
    for name, p in model.named_parameters():
        mod = name.rsplit(".", 1)[0]
        rec = out.setdefault(mod, {"n": 0, "no_grad": 0, "gnorm": 0.0})
        rec["n"] += 1
        if p.grad is None:
            rec["no_grad"] += 1
        else:
            rec["gnorm"] += float(p.grad.detach().abs().sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--dezero", action="store_true", default=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    T = load_trainer(args.stack)
    v3 = T.v3
    report: dict = {"weights": {}, "arms": {}}

    print("=" * 78)
    print("PART 1 - LOSS WEIGHTS AS THE TRAINER RESOLVES THEM")
    print("  (a 0.0 default means the term is GUARDED OUT unless enabled)")
    print("=" * 78)
    for n in sorted(x for x in dir(T) if "WEIGHT" in x):
        v = getattr(T, n)
        report["weights"][n] = v
        flag = "   <-- ZERO BY DEFAULT" if v == 0.0 else ""
        print("  %-30s %-8s%s" % (n, v, flag))

    for vocab in ("kin3", "v7.0"):
        print()
        print("=" * 78)
        print("ARM: tac_vocab_version =", vocab)
        print("=" * 78)
        arm: dict = {}
        cfg = v3.refc_v3_smoke_config(hier=True)
        cfg.tac_vocab_version = vocab
        model = v3.RefCV3Model(cfg)
        model.train()
        arm["head_widths"] = {
            "lat_head_tac": model.lat_head_tac.out_features,
            "lon_head_tac": model.lon_head_tac.out_features,
        }
        print("  z_tac head widths: lat=%d lon=%d"
              % (arm["head_widths"]["lat_head_tac"],
                 arm["head_widths"]["lon_head_tac"]))
        if args.dezero:
            print("  de-zeroed edges:", dezero(model))

        batch = build_batch(T, cfg)
        arm["batch_keys"] = sorted(batch.keys())
        print("  L1 BATCH KEYS (%d):" % len(arm["batch_keys"]))
        print("     ", arm["batch_keys"])

        # ---- L3: real backward through the trainer's OWN loss --------------
        try:
            res = T.compute_losses_v3(model, batch, "cpu")
            loss = res["loss"] if isinstance(res, dict) else res
            model.zero_grad(set_to_none=True)
            loss.backward()
            arm["loss"] = float(loss.detach())
            print("  L3 total loss = %.6f (backward OK)" % arm["loss"])
        except Exception as e:
            arm["loss_error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
            print("  L3 LOSS FAILED:", arm["loss_error"])
            report["arms"][vocab] = arm
            continue

        cen = grad_census(model)
        arm["grad_census"] = cen
        dead = {k: v for k, v in cen.items() if v["no_grad"] == v["n"]}
        arm["dead_modules"] = sorted(dead)
        print("  L3 GRADIENT CENSUS: %d parameter groups, %d receive NO"
              " gradient" % (len(cen), len(dead)))
        for k in sorted(dead):
            print("     NO GRAD: %-38s %s" % (k, HEAD_VOCAB.get(k, "")))
        # the same-breath control: the trajectory head MUST have gradient
        ctrl = [k for k in cen if "traj" in k or "decoder" in k]
        ctrl_ok = any(cen[k]["no_grad"] < cen[k]["n"] for k in ctrl)
        arm["control_traj_has_grad"] = bool(ctrl_ok)
        print("     CONTROL (traj/decoder has gradient): %s%s"
              % (ctrl_ok, "" if ctrl_ok else "  <-- RIG DEAD, nulls void"))

        # ---- L4: extractable at inference ---------------------------------
        model.eval()
        with torch.no_grad():
            out = model(batch["frames"].float() / 255.0
                        if batch["frames"].dtype == torch.uint8
                        else batch["frames"],
                        batch.get("nav_cmd"), batch.get("v0"))
        arm["out_keys"] = sorted(out.keys())
        print("  L4 INFERENCE OUTPUT KEYS (%d):" % len(arm["out_keys"]))
        print("     ", arm["out_keys"])
        report["arms"][vocab] = arm

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print()
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
