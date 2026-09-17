"""Are the ImageNet weights REALLY loaded at 256 x 1024 -- proven by mutation?

``timm_trunk._assert_pretrained_loaded`` is the guard. A guard that has only
ever seen the fixed code has not been tested, so this exercises it three ways:

* CONTROL      the shipped build must PASS and the measured stem statistic is
               reported beside the pinned band, not just "ok";
* MUTATION A   He-reinitialise the stem (what a silent ``pretrained`` failure
               leaves behind) -- the guard must go RED;
* MUTATION B   build the SAME trunk with ``pretrained=False`` and call the
               guard -- it must go RED. This is the arm that distinguishes
               "the guard checks the weights" from "the guard checks a flag".

Also reported: the backbone's stride-16 / stride-32 channel counts and grids at
the cache's own geometry, read from timm's ``feature_info``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WT = Path(os.environ.get("TANITAD_WT", Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(WT / "stack"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tanitad.models import timm_trunk as tt  # noqa: E402
from tanitad.models.trunk_shapes import TrunkSpec  # noqa: E402


def stem_stat(net):
    name, stem = tt._find_stem(net)
    return name, float(stem.weight.detach().abs().sum()), stem


def probe(model_name: str, frame) -> dict:
    row: dict = {"model_name": model_name}
    trunk = tt.build_timm_trunk(in_channels=9, model_name=model_name,
                                mode="shared", fuse="concat1x1",
                                fuse_identity_init=True,
                                image_hw=(frame.height, frame.width))
    prov = trunk.provenance()
    row["provenance"] = prov
    net = trunk.net if hasattr(trunk, "net") else trunk
    name, got, stem = stem_stat(net)
    pinned = tt._PINNED_STATS.get(model_name)
    row["stem"] = {
        "module": name, "abs_sum": got, "pinned": pinned,
        "rtol": tt._CONV1_ABS_SUM_RTOL,
        "band": None if pinned is None else
        [pinned * (1 - tt._CONV1_ABS_SUM_RTOL),
         pinned * (1 + tt._CONV1_ABS_SUM_RTOL)],
        "he_init_reference_in_docstring": 875.0,
    }
    # CONTROL
    try:
        tt._assert_pretrained_loaded(net, model_name)
        row["CONTROL_shipped_build"] = "PASS"
    except Exception as exc:
        row["CONTROL_shipped_build"] = "RED: %s" % exc

    # MUTATION A -- He-reinitialise the stem in place
    saved = stem.weight.detach().clone()
    with torch.no_grad():
        nn.init.kaiming_normal_(stem.weight, mode="fan_out",
                                nonlinearity="relu")
    row["MUT_A_he_reinit_stem"] = {
        "abs_sum_after": float(stem.weight.detach().abs().sum())}
    try:
        tt._assert_pretrained_loaded(net, model_name)
        row["MUT_A_he_reinit_stem"]["guard"] = "GREEN -- THE GUARD IS INERT"
    except Exception as exc:
        row["MUT_A_he_reinit_stem"]["guard"] = "RED (correct): %s" % str(exc)[:200]
    with torch.no_grad():
        stem.weight.copy_(saved)

    # MUTATION B -- a genuinely untrained build of the same architecture
    import timm
    cold = timm.create_model(model_name, pretrained=False, features_only=True,
                             out_indices=(2, 3))
    n2, got2, _ = stem_stat(cold)
    row["MUT_B_pretrained_false"] = {"stem_abs_sum": got2}
    try:
        tt._assert_pretrained_loaded(cold, model_name)
        row["MUT_B_pretrained_false"]["guard"] = "GREEN -- THE GUARD IS INERT"
    except Exception as exc:
        row["MUT_B_pretrained_false"]["guard"] = \
            "RED (correct): %s" % str(exc)[:200]

    spec = TrunkSpec.from_timm(model_name.split(".")[0], frame)
    row["feature_info"] = {
        "perception": {"stride": spec.perception.stride,
                       "channels": spec.perception.channels,
                       "hw": list(spec.perception.hw)},
        "planner": {"stride": spec.planner.stride,
                    "channels": spec.planner.channels,
                    "hw": list(spec.planner.hw)}}
    return row


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e2e_1024 import frame_of_cache
    cache = os.environ.get(
        "E2E_CACHE", "D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl")
    fr = frame_of_cache(cache)
    res = {"evidence_class": "MEASURED",
           "frame": {"height": fr.height, "width": fr.width,
                     "f_ref": fr.f_ref, "projection": fr.projection},
           "arms": {}}
    for name in ("resnet101.a1_in1k", "resnet34.a1_in1k"):
        res["arms"][name] = probe(name, fr)
        print("[pretrained] %s control=%s mutA=%s mutB=%s"
              % (name, res["arms"][name]["CONTROL_shipped_build"],
                 res["arms"][name]["MUT_A_he_reinit_stem"]["guard"][:30],
                 res["arms"][name]["MUT_B_pretrained_false"]["guard"][:30]),
              flush=True)
    out = Path(__file__).resolve().parents[1] / "raw" / "pretrained_guard.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: {"stem": v["stem"],
                          "control": v["CONTROL_shipped_build"],
                          "mutA": v["MUT_A_he_reinit_stem"],
                          "mutB": v["MUT_B_pretrained_false"],
                          "feature_info": v["feature_info"]}
                      for k, v in res["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
