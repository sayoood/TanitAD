"""P6-READ-DRYRUN Q1: does the harness's model loader build P6's architecture from `ckpt_last.pt`, strictly?

CPU only, no GPU lock. The calls under test are the planner's OWN (refe/planner.py:344-359, tip 1664143):
    cfg = REFeConfig.for_backbone("vitl16"); cfg = ckpt_io.config_for_checkpoint(cfg, ckpt)
    model = REFe(cfg); ckpt_io.load_for_inference(model, ckpt, backbone="vitl16", meta_out=meta)
plus an INDEPENDENT key audit (so the planner's own refusals are not the only witness) and negative controls that must RAISE.

  python -I p6_load_audit.py --ckpt <ckpt_last.pt> --out raw/load_audit.json [--sim-final <path to write a trainer-format FULL file>]

`--sim-final` writes model_final.pt-format bytes with the TRAINER'S OWN call (train.py:1902-1904):
    ckpt_io.atomic_save({"format": FORMAT_FULL, "model": full_state(model), "meta": dict(meta, step=.., epoch=.., finished_at=..)}, path)
from the model the partial just built, then loads it back through the same loader. It is a format rehearsal, NOT P6's final weights.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6dry_common as C  # noqa: E402

C.setup_env()
import torch  # noqa: E402

import ckpt_io  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402


def raises(fn, *exc):
    try:
        fn()
    except exc or (Exception,) as e:
        return True, f"{type(e).__name__}: {str(e)[:200]}"
    return False, "DID NOT RAISE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-final", default=None)
    ap.add_argument("--config-json", default=None, help="the run's config.json (params / identity / source_sha256 cross-check)")
    a = ap.parse_args()
    R: dict = {"ckpt": a.ckpt, "torch": torch.__version__, "gates": {}}
    t0 = time.time()
    R["ckpt_bytes"] = os.path.getsize(a.ckpt)
    R["ckpt_sha256"] = ckpt_io.sha256_file(a.ckpt)

    # ---- A. the config the planner builds
    cfg0 = REFeConfig.for_backbone("vitl16")
    cfg = ckpt_io.config_for_checkpoint(cfg0, a.ckpt)
    d0, d1 = dataclasses.asdict(cfg0), dataclasses.asdict(cfg)
    R["cfg_changed_by_ckpt_meta"] = {k: [d0[k], d1[k]] for k in d1 if d0[k] != d1[k]}
    R["cfg_key_fields"] = {k: d1[k] for k in ("img_h", "img_w", "n_cameras", "cameras", "n_proposals", "n_score_components",
                                              "ego_dim", "nav_mode", "n_goal_points", "detach_scorer_context", "undistort")}

    # ---- B. the model the planner builds
    model = REFe(cfg).eval()
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    R["params"] = {"total": tot, "trainable": trn}
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = sd.get("meta", {})
    R["file_format"] = sd.get("format")
    R["file_meta"] = {k: meta[k] for k in meta}
    part = sd["model_partial"]
    if a.config_json:
        cj = json.load(open(a.config_json, encoding="utf-8"))
        R["run_config_params"] = cj["params"]
        R["gates"]["params_equal_run_config"] = (cj["params"]["total"] == tot and cj["params"]["trainable"] == trn)
        R["gates"]["frozen_sha256_meta_equals_run_config_identity"] = (meta.get("frozen_sha256") == cj["identity"]["frozen_sha256"])

    # ---- C. independent key audit (no load yet): the partial's keys must be EXACTLY trainable params + buffers, shapes/dtypes equal
    ms = model.state_dict()
    nonpersistent = sorted(({n for n, _ in model.named_buffers()}) - set(ms))     # e.g. backbone.img_mean / img_std: never in a state_dict, so never in a partial
    want = (ckpt_io.trainable_names(model) | {n for n, _ in model.named_buffers()}) & set(ms)
    have = set(part)
    ms = model.state_dict()
    shape_bad = [k for k in have & set(ms) if tuple(part[k].shape) != tuple(ms[k].shape) or part[k].dtype != ms[k].dtype]
    R["key_audit"] = {"partial_keys": len(have), "expected_trainable_plus_buffers": len(want),
                      "in_file_not_in_model": sorted(have - set(ms))[:10], "expected_but_absent_in_file": sorted(want - have)[:10],
                      "shape_or_dtype_mismatch": shape_bad[:10], "n_frozen_params_not_in_file": len(ckpt_io.frozen_names(model)),
                      "partial_numel": sum(t.numel() for t in part.values()), "non_persistent_buffers_never_in_a_state_dict": nonpersistent}
    R["gates"]["partial_keys_exactly_trainable_plus_buffers"] = (have == want and not shape_bad)

    # ---- D. the planner's own loader call
    t1 = time.time()
    meta_out: dict = {}
    fmt = ckpt_io.load_for_inference(model, a.ckpt, map_location="cpu", backbone="vitl16", meta_out=meta_out)
    R["load_for_inference"] = {"format": fmt, "seconds": round(time.time() - t1, 1), "meta_out_keys": sorted(meta_out)}
    R["gates"]["planner_loader_accepts_partial"] = (fmt == ckpt_io.FORMAT_PARTIAL)
    R["gates"]["per_sample_calib_read_from_meta"] = bool(meta_out.get("per_sample_calib"))
    ms2 = model.state_dict()
    R["gates"]["every_partial_tensor_landed_bit_equal"] = all(torch.equal(ms2[k], part[k]) for k in part)
    R["gates"]["fingerprint_equals_meta"] = (ckpt_io.frozen_fingerprint(model) == meta.get("frozen_sha256"))
    R["frozen_fingerprint"] = meta.get("frozen_sha256")

    # ---- E. independent strict=False audit: missing == frozen set EXACTLY, unexpected == []
    m2 = REFe(cfg).eval()
    missing, unexpected = m2.load_state_dict(part, strict=False)
    R["strict_false_audit"] = {"missing": len(missing), "unexpected": list(unexpected),
                               "missing_equals_frozen_set": set(missing) == ckpt_io.frozen_names(m2),
                               "missing_not_frozen": sorted(set(missing) - ckpt_io.frozen_names(m2))[:10]}
    R["gates"]["no_unexpected_keys"] = (len(unexpected) == 0)
    R["gates"]["missing_keys_are_exactly_the_frozen_trunk"] = R["strict_false_audit"]["missing_equals_frozen_set"]
    del m2

    # ---- F. negative controls: each MUST raise
    neg = {}
    drop = dict(part); k0 = sorted(k for k in part if k.startswith("cmd_embed"))[0]; drop.pop(k0)
    neg["partial_missing_a_trained_key"] = raises(lambda: ckpt_io.load_partial(REFe(cfg).eval(), drop, meta.get("frozen_sha256")), ValueError)
    extra = dict(part); extra["bogus.weight"] = torch.zeros(1)
    neg["partial_with_an_unexpected_key"] = raises(lambda: ckpt_io.load_partial(REFe(cfg).eval(), extra, meta.get("frozen_sha256")), ValueError)
    neg["partial_with_wrong_trunk_fingerprint"] = raises(lambda: ckpt_io.load_partial(model, part, "0" * 64), ValueError)
    cfg_goal = copy.deepcopy(cfg); cfg_goal.nav_mode = "goal"
    neg["command_partial_into_a_goal_mode_model"] = raises(lambda: ckpt_io.load_partial(REFe(cfg_goal).eval(), part, None), ValueError, RuntimeError)
    R["negative_controls"] = {k: {"raised": v[0], "msg": v[1]} for k, v in neg.items()}
    R["gates"]["all_negative_controls_raise"] = all(v[0] for v in neg.values())

    # ---- G. the FULL format (what model_final.pt is): written with the trainer's own call, loaded back through the planner's loader
    if a.sim_final:
        full = ckpt_io.full_state(model)
        t2 = time.time()
        ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": full,
                             "meta": dict(meta, step=10075, epoch=25, finished_at="REHEARSAL")}, a.sim_final)
        R["sim_final"] = {"path": a.sim_final, "bytes": os.path.getsize(a.sim_final), "write_s": round(time.time() - t2, 1)}
        del full
        c3 = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone("vitl16"), a.sim_final)
        m3 = REFe(c3).eval()
        fmt3 = ckpt_io.load_for_inference(m3, a.sim_final, map_location="cpu", backbone="vitl16", meta_out={})
        s3 = m3.state_dict()
        R["sim_final"].update({"format_loaded": fmt3, "nav_mode_from_meta": c3.nav_mode,
                               "bit_equal_to_partial_built_model": all(torch.equal(s3[k], ms2[k]) for k in ms2) and set(s3) == set(ms2)})
        R["gates"]["full_format_strict_load_ok"] = (fmt3 == ckpt_io.FORMAT_FULL and c3.nav_mode == "command"
                                                    and R["sim_final"]["bit_equal_to_partial_built_model"])
        # the meta hook is load-bearing: the same FULL file into a goal-mode config must fail the strict load
        c4 = REFeConfig.for_backbone("vitl16")
        neg2 = raises(lambda: REFe(c4).eval().load_state_dict(torch.load(a.sim_final, map_location="cpu", weights_only=False)["model"]), RuntimeError)
        R["sim_final"]["goal_mode_strict_load_raises"] = {"raised": neg2[0], "msg": neg2[1]}
        R["gates"]["full_format_into_goal_mode_raises"] = neg2[0]

    R["seconds"] = round(time.time() - t0, 1)
    R["ALL_GATES_PASS"] = all(R["gates"].values())
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"gates": R["gates"], "ALL_GATES_PASS": R["ALL_GATES_PASS"], "seconds": R["seconds"]}, indent=1))
    return 0 if R["ALL_GATES_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
