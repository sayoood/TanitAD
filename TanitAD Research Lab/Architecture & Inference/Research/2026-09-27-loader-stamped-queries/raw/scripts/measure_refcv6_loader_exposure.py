"""MEASURE the I3 exposure of the refcv6 battery loader (``refcv6_loader.build_model``) on a post-A9 tree.

Usage:  python measure_refcv6_loader_exposure.py <tree root>   (CPU only, no data, ~10 s)

It builds synthetic refc records the way ``refc_v3_train.train`` builds them: the agent head (``--agents head``),
the box head (``--w-box3d 1``), and an argv SILENT on ``--agent-queries``. Records are made at 100 queries
(pre-A9, stamps without the A9/A6/A7/A14 fields) and at 300 (post-A9). Each is loaded through
(a) ``taniteval/tools/refcv3_arm.load_model``, the reference rule, and (b) the vendored battery loader
``stack/tanitad/eval/refcv6_loader.build_model`` (strict). The vendored body is the audit blob 11808258...,
byte-identical to the EvalFlyWheel battery copy. It prints one JSON line per (record, loader) with the
outcome and, on failure, every ``size mismatch`` line.
"""
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
for p in (ROOT / "stack", ROOT / "stack" / "scripts", ROOT / "taniteval", ROOT / "taniteval" / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ["REFCV6_REPO"] = str(ROOT)          # the vendored loader resolves ITS stack from this, at import
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("refc_v3_train_i3_measure", str(ROOT / "stack/scripts/refc_v3_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402

ARGV = ["--arm", "hier", "--smoke", "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k",
        "--no-trunk-pretrained", "--trunk-in-channels", "9", "--image-hw", "256", "640",
        "--agents", "head", "--agent-join", "join.jsonl", "--w-agent", "1.0", "--w-box3d", "1.0",
        "--out", "x"]
A9_AGENT_KEYS = ("presence_loss", "presence_prior", "deep_supervision", "vis1")
A9_PERCEPTION_KEYS = A9_AGENT_KEYS + ("query_select", "bev_source", "planner_crop_m")


def write_record(run: Path, queries: int, pre_a9: bool) -> Path:
    args = T.build_parser().parse_args(ARGV)
    args.agent_queries = queries
    cfg = T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)
    pcfg = PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=queries)
    model._perception = PB.build_perception_branch(model, pcfg)
    agents, perc = dict(cfg.core.agents.as_dict()), dict(pcfg.as_dict())
    if pre_a9:
        for k in A9_AGENT_KEYS:
            agents.pop(k, None)
        for k in A9_PERCEPTION_KEYS:
            perc.pop(k, None)
        agents["n_queries_default_upstream"] = queries
    perc["branch_params"] = model._perception.param_breakdown()
    config = {"argv": list(ARGV), "arm": "hier", "image_hw": list(cfg.core.encoder.image_hw()),
              "horizons": list(cfg.core.trajectory.horizons), "seams": {"agents": agents},
              "refcv6_perception": perc,
              "param_breakdown": {k: int(v) for k, v in v3.param_breakdown_v3(model).items()}}
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model": model.state_dict(), "opt": {}}, run / "ckpt.pt")
    (run / "config.json").write_text(json.dumps(config, default=str), encoding="utf-8")
    return run / "ckpt.pt"


def counts(model):
    return [int(model.core.agent_head.queries.shape[1]), int(model._perception.box_dec.queries.shape[1])]


spec = importlib.util.spec_from_file_location("refcv3_arm_i3_measure", str(ROOT / "taniteval/tools/refcv3_arm.py"))
A = importlib.util.module_from_spec(spec)
sys.modules["refcv3_arm_i3_measure"] = A
spec.loader.exec_module(A)
L = importlib.import_module("tanitad.eval.refcv6_loader")
print(json.dumps({"tree": str(ROOT), "vendored_loader": str(Path(L.__file__).resolve()),
                  "loader_STACK": str(L.STACK), "N_QUERIES_DEFAULT": int(PB.N_QUERIES_DEFAULT),
                  "parser_default_agent_queries": int(T.build_parser().get_default("agent_queries"))}))
tmp = Path(tempfile.mkdtemp(prefix="i3_measure_"))
for tag, q, pre in (("pre_A9_100", 100, True), ("post_A9_300", 300, False)):
    ck = write_record(tmp / tag, q, pre)
    config = json.loads((ck.parent / "config.json").read_text(encoding="utf-8"))
    try:
        m, _c, _t, prov = A.load_model(str(ck), None, "cpu", False)
        sd = prov["state_dict_load"]
        rec = {"ok": True, "counts_built": counts(m), "missing": len(sd["missing_keys"]),
               "unexpected": len(sd["unexpected_keys"])}
    except SystemExit as e:
        rec = {"ok": False, "refusal": str(e).splitlines()[0][:200]}
    print(json.dumps({"record": tag, "loader": "refcv3_arm.load_model", **rec}))
    try:
        m, _c, _a, r = L.build_model(config, str(ck), device="cpu", strict=True)
        rec = {"ok": True, "counts_built": counts(m), "missing": len(r["state_dict"]["missing"]),
               "unexpected": len(r["state_dict"]["unexpected"])}
    except RuntimeError as e:
        rec = {"ok": False, "error": type(e).__name__,
               "size_mismatch": [ln.strip() for ln in str(e).splitlines() if "size mismatch" in ln]}
    print(json.dumps({"record": tag, "loader": "tanitad.eval.refcv6_loader.build_model", **rec}))
