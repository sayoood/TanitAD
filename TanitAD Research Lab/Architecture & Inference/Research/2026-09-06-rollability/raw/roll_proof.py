"""D-ROLL-1 -- REAL-CHECKPOINT ROLLABILITY PROOF.

Loads every banked v7.0 checkpoint through ``refcv3_arm``'s OWN rebuild path
(`refc_v3_train.build_parser` + `_pin_trainer_cfg` on the recorded `argv`), runs
the config cross-check, and reports missing/unexpected state_dict key counts.

Bar: 0 missing / 0 unexpected, and the `param_breakdown` cross-check PASSES.

ASCII-only output: the console is cp1252 and a non-ASCII `print` is fatal on the
success path (MEASURED 2026-09-06 -- every v6 preflight refusal died in its own
`print`, exiting 1 with a traceback instead of 2 with the reason).
"""
from __future__ import annotations

import json
import os
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WT = r"C:\Users\Admin\tanitad-wt"
for p in (os.path.join(WT, "stack"), WT):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPTS = [
    ("refcv4b@40284", r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"),
    ("refcv3-rlmin@40284", r"C:\Users\Admin\rl_refcv3_min\base\ckpt_step40284_frozen.pt"),
    ("refcv3-ol@30000", r"C:\Users\Admin\run_refcv3_ol\ckpt\ckpt_30000.pt"),
    ("refcv3-viz@40284", r"C:\Users\Admin\run_refcv3_viz\ckpt\ckpt_step40284_frozen.pt"),
]


_ARM = None


def _arm():
    """⛔ IMPORT BY PATH, never as ``taniteval.tools.refcv3_arm``.

    ``refcv3_arm._bootstrap_paths`` EVICTS every ``taniteval.*`` entry from
    ``sys.modules`` to kill the namespace shadow -- including its OWN
    in-progress entry -- so a package import of it dies with
    ``KeyError: 'taniteval.tools.refcv3_arm'`` inside ``_load_unlocked``.
    This is the module's own ``_load_by_path`` convention, reused.
    """
    global _ARM
    if _ARM is None:
        import importlib.util
        p = os.path.join(WT, "taniteval", "tools", "refcv3_arm.py")
        spec = importlib.util.spec_from_file_location("refcv3_arm_for_roll", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["refcv3_arm_for_roll"] = mod
        spec.loader.exec_module(mod)
        _ARM = mod
    return _ARM


def one(name: str, ckpt: str) -> dict:
    import torch
    arm = _arm()
    from tanitad.refs import refc_v3 as v3

    side = os.path.join(os.path.dirname(ckpt), "config.json")
    with open(side, encoding="utf-8") as fh:
        config = json.load(fh)

    rec = {"arm": name, "ckpt": ckpt, "config_json": side,
           "recorded_total": (config.get("param_breakdown") or {}).get("total"),
           "recorded_has_tacgoal_line":
               "tac_goal_tok_head" in (config.get("param_breakdown") or {}),
           "tac_vocab_version": config.get("tac_vocab_version")}

    cfg, targs, src = arm.rebuild_config(config)
    rec["rebuilt_from"] = src
    rec["cfg_tac_goal_tok_head"] = bool(getattr(cfg, "tac_goal_tok_head", None))

    model = v3.RefCV3Model(cfg)
    rec["rebuilt_has_head"] = model.tac_goal_tok_head is not None
    bd = v3.param_breakdown_v3(model)
    rec["rebuilt_total"] = int(bd["total"])
    rec["rebuilt_has_tacgoal_line"] = "tac_goal_tok_head" in bd

    # THE GUARD, unmodified. A conflict raises SystemExit.
    try:
        arm.cross_check_config(config, cfg, model)
        rec["cross_check"] = "PASS"
    except SystemExit as ex:
        rec["cross_check"] = "REFUSED"
        rec["cross_check_msg"] = str(ex)[:600]

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    res = model.load_state_dict(ck["model"], strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    dec = getattr(model.core, "decoder", None)
    inert = sorted(k for k in missing if k.endswith(".anchor_controls")
                   and dec is not None
                   and not bool(getattr(dec, "anchor_v0_cond", False)))
    rec.update({
        "ckpt_step": ck.get("step"),
        "n_missing": len(missing), "n_unexpected": len(unexpected),
        "missing_keys": missing[:12], "unexpected_keys": unexpected[:12],
        "tolerated_inert_buffers": inert,
        "n_missing_after_documented_tolerance": len([k for k in missing
                                                     if k not in inert]),
    })
    # same-breath NON-ZERO control: the checkpoint really was read and really
    # does carry weights, so a 0/0 cannot come from an empty state_dict.
    rec["control_n_ckpt_tensors"] = len(ck["model"])
    rec["control_n_model_params"] = sum(1 for _ in model.parameters())
    del ck, model
    return rec


def main() -> int:
    out, bad = [], 0
    for name, ck in CKPTS:
        if not os.path.exists(ck):
            out.append({"arm": name, "ckpt": ck, "status": "INCONCLUSIVE-ABSENT"})
            bad += 1
            continue
        try:
            r = one(name, ck)
        except Exception as ex:                       # noqa: BLE001
            r = {"arm": name, "ckpt": ck, "status": "ERROR",
                 "error": f"{type(ex).__name__}: {ex}"[:800],
                 "tb": traceback.format_exc()[-1200:]}
            bad += 1
        else:
            ok = (r["cross_check"] == "PASS" and r["n_missing"] == 0
                  and r["n_unexpected"] == 0 and r["control_n_ckpt_tensors"] > 0)
            r["status"] = "ROLLABLE" if ok else "NOT-ROLLABLE"
            bad += 0 if ok else 1
        out.append(r)
        print("[%-20s] %-13s cross_check=%-8s missing=%-4s unexpected=%-4s "
              "rebuilt_total=%s recorded_total=%s ckpt_tensors=%s"
              % (r["arm"], r["status"], r.get("cross_check", "-"),
                 r.get("n_missing", "-"), r.get("n_unexpected", "-"),
                 r.get("rebuilt_total", "-"), r.get("recorded_total", "-"),
                 r.get("control_n_ckpt_tensors", "-")))
        sys.stdout.flush()

    dst = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\_roll\roll_proof.json"
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"checkpoints": out, "n_failing": bad}, fh, indent=1)
    print("WROTE", dst, "n_failing =", bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
