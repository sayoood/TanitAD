"""D-REFCV6-EQUALIZE-DROPPED audit: which attributes does `_pin_trainer_cfg` set that the
config's dataclass does NOT declare, and which of them survive to the built model?

Run against a CLEAN tree (git archive of the tip, optionally + the fix overlay):

    TREE=C:/Users/Admin/dvb_tip_0926a python audit_adhoc_attrs.py OUT.json

Three measurements, all on CPU, no checkpoint, no download (HF_HUB_OFFLINE=1):

1. RUNTIME census. The live run's own argv (refcv6-r101-s0 `config.json`) is parsed with the
   trainer's own parser and pinned through `_pin_trainer_cfg`, with and without `--image-hw`.
   Every dataclass instance in the resulting config tree is walked and every key of its
   `__dict__` that is NOT a declared field is reported (path, value, whether the rebuild kept it).
2. STATIC census. Every `cfg.<...>.<attr> = ...` / `core.<...>.<attr> = ...` assignment in the
   pin helpers is listed with its line, and checked against the declared fields of the class the
   base path resolves to on a freshly pinned config. A static list is needed because a runtime
   census only sees the branches the chosen argv reaches.
3. The TRUNK as built: `refc.build_encoder(cfg.core.encoder)` on a small timm trunk
   (`resnet34.a1_in1k`, random init: the census is about the CONFIG, not the weights), then
   `trunk.cfg.equalize_bottom_rows`, one `normalise` and `equalize_calls`.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
TREE = Path(os.environ["TREE"]).resolve()
sys.path[:0] = [str(TREE / "stack"), str(TREE / "stack" / "scripts"), str(TREE / "taniteval")]

import torch  # noqa: E402
import tanitad  # noqa: E402

_tf = Path(tanitad.__file__).resolve()
assert str(_tf).lower().startswith(str(TREE).lower()), (
    f"tanitad imported from {_tf}, NOT from the tree under test {TREE}")
import refc_v3_train as tr  # noqa: E402
from tanitad.refs import refc, refc_v3 as v3  # noqa: E402

LIVE_CONFIG = Path(os.environ.get("LIVE_CONFIG", "D:/refcv6_eval_kit/ckpt/config.json"))
PIN_HELPERS = ("_pin_trainer_cfg", "_pin_refcv5_seams", "_pin_refcv6_tactical", "_pin_refcv7")


def walk(obj, path="cfg", out=None, seen=None):
    """-> list of (path, class, [undeclared keys]) for every dataclass in the tree."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        declared = {f.name for f in dataclasses.fields(obj)}
        extra = sorted(k for k in vars(obj) if k not in declared)
        out.append((path, type(obj).__name__, extra))
        for k, v in vars(obj).items():
            walk(v, f"{path}.{k}", out, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]", out, seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}[{k!r}]", out, seen)
    return out


def pinned(argv):
    args = tr.build_parser().parse_args(argv)
    base = v3.refc_v3_sized_config(args.size, hier=args.arm == "hier")
    return args, tr._pin_trainer_cfg(base, args)


def runtime_census(argv):
    _, cfg = pinned(argv)
    rows = walk(cfg)
    return {p: {"class": c, "undeclared": {k: repr(getattr(eval_path(cfg, p), k))
                                           for k in extra}}
            for p, c, extra in rows if extra}


def eval_path(cfg, p):
    obj = cfg
    for part in p.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def static_census(cfg_ref):
    """Every attribute assignment in the pin helpers, with the declared-field verdict."""
    src_file = inspect.getsourcefile(tr)
    rows = []
    for name in PIN_HELPERS:
        fn = getattr(tr, name, None)
        if fn is None:
            continue
        src, start = inspect.getsourcelines(fn)
        tree = ast.parse("".join(src).replace("\r\n", "\n"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                if not isinstance(t, ast.Attribute):
                    continue
                chain = []
                cur = t
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if not isinstance(cur, ast.Name):
                    continue
                chain.append(cur.id)
                chain.reverse()                          # e.g. ['cfg','core','encoder','x']
                root, *mid, attr = chain
                if root not in ("cfg", "core", "hc", "args"):
                    continue
                line = start + node.lineno - 1
                owner = None
                if root == "args":
                    owner_cls, declared = "argparse.Namespace", None
                else:
                    try:
                        obj = cfg_ref if root == "cfg" else (
                            cfg_ref.core if root == "core" else cfg_ref.refcv7_head_cfg)
                        for m in mid:
                            obj = getattr(obj, m)
                        owner = obj
                        owner_cls = type(obj).__name__
                        declared = (attr in {f.name for f in dataclasses.fields(obj)}
                                    if dataclasses.is_dataclass(obj) else None)
                    except Exception as e:                # pragma: no cover
                        owner_cls, declared = f"<unresolved: {e!r}>", None
                rows.append({"helper": name, "line": line,
                             "file": Path(src_file).name,
                             "target": ".".join(chain), "owner_class": owner_cls,
                             "declared_field": declared,
                             "under_encoder": mid[:2] == ["core", "encoder"]
                             or (root == "core" and mid[:1] == ["encoder"])})
    return rows


def trunk_measurement(argv):
    args, cfg = pinned(argv)
    enc = refc.build_encoder(cfg.core.encoder)
    trunk = next(m for m in enc.modules() if type(m).__name__ == "TimmResNetTrunk")
    k = int(cfg.core.encoder.in_channels) // 3
    h, w = cfg.core.encoder.image_hw()
    x = torch.rand(1, 3 * k, h, w)
    y = trunk.normalise(x)
    zero_ref = (torch.zeros_like(x[..., -43:, :]) - trunk._mean) / trunk._std
    return {
        "argv_equalize_bottom_rows": int(args.equalize_bottom_rows),
        "cfg_has_attr": hasattr(cfg.core.encoder, "trunk_equalize_bottom_rows"),
        "cfg_value": getattr(cfg.core.encoder, "trunk_equalize_bottom_rows", "MISSING"),
        "declared_field": "trunk_equalize_bottom_rows" in {
            f.name for f in dataclasses.fields(cfg.core.encoder)},
        "trunk_cfg_equalize_bottom_rows": int(trunk.cfg.equalize_bottom_rows),
        "trunk_image_hw": list(trunk.cfg.image_hw),
        "equalize_calls_after_one_normalise": int(getattr(trunk, "equalize_calls", 0)),
        "norm_calls": int(trunk.norm_calls),
        "bottom43_zeroed_in_01_domain": bool(torch.allclose(y[..., -43:, :], zero_ref,
                                                             atol=1e-6)),
    }


class _RecordingDC:
    """Proxy for the trainer's `_dc` (= `dataclasses`) that records what `replace` DROPS."""

    def __init__(self, real):
        self._real = real
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def replace(self, obj, **kw):
        new = self._real.replace(obj, **kw)
        before, after = set(vars(obj)), set(vars(new))
        declared = {f.name for f in self._real.fields(obj)}
        self.calls.append({
            "class": type(obj).__name__, "changes": sorted(kw),
            "dropped": {k: repr(vars(obj)[k]) for k in sorted(before - after)},
            "undeclared_before": sorted(before - declared),
            "undeclared_after": sorted(after - declared)})
        return new


def rebuild_interception(argv):
    rec = _RecordingDC(tr._dc)
    real = tr._dc
    tr._dc = rec
    try:
        pinned(argv)
    finally:
        tr._dc = real
    return rec.calls


#: Thor path -> local eval-kit copy, for the files `_pin_trainer_cfg` itself opens (the rig
#: extrinsics) and the anchor artifact. The same table as the battery loader's `PATH_REMAP`
#: (`C:/Users/Admin/ev6_battery/code/refcv6_loader.py`), written out here so this audit imports
#: nothing from another stream.
KIT = Path(os.environ.get("REFCV6_KIT", "D:/refcv6_eval_kit"))
PATH_REMAP = {
    "--anchors": str(KIT / "data/anchors/refc_anchors_6s_v0cond_alat_117.pt"),
    "--agent-rig-extrinsics": str(KIT / "data/refcv6_train_eval139_extrinsics.json"),
    "--agent-join": str(KIT / "data/joins/b1_train_plus_eval_agents.jsonl.xz"),
    "--join3d": str(KIT / "data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"),
    "--map-gt-root": str(KIT / "data/sam3_gt_eval_thor137"),
    "--speed-max-sidecar-v6-eval": str(KIT / "data/refcv6_speed_max_v8_eval.jsonl"),
}


def remap(argv):
    out = list(argv)
    for flag, local in PATH_REMAP.items():
        if flag in out:
            out[out.index(flag) + 1] = local
    out[out.index("--out") + 1] = str(Path(os.environ.get("TMP", ".")) / "dvb_audit_out")
    return out


def main(out_path: str) -> None:
    live = json.loads(LIVE_CONFIG.read_text(encoding="utf-8"))
    live_argv = remap(list(live["argv"]))
    # a SMALL trunk for the build measurement: the census is about the config path
    small = [a for a in live_argv]
    small[small.index("--trunk-name") + 1] = "resnet34.a1_in1k"
    small += ["--no-trunk-pretrained"]
    for flag in ("--trunk-compile", "--trunk-chunk-ckpt", "--trunk-bf16",
                 "--trunk-channels-last", "--trunk-fold-bn", "--trunk-dedup-frames",
                 "--trunk-frozen-bn"):
        if flag in small:
            k = small.index(flag)
            del small[k:k + (2 if flag == "--trunk-chunk-ckpt" else 1)]
    _, cfg_ref = pinned(live_argv)
    res = {
        "tree": str(TREE), "tanitad_file": str(_tf),
        "trainer_file": str(Path(tr.__file__).resolve()),
        "live_config": str(LIVE_CONFIG),
        "live_argv": live_argv,
        "rebuild_calls_on_live_argv": rebuild_interception(live_argv),
        "runtime_undeclared_on_final_cfg_live_argv": runtime_census(live_argv),
        "static_assignments": static_census(cfg_ref),
        "trunk_as_built_WITH_image_hw_416x1024": trunk_measurement(small),
    }
    Path(out_path).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("static_assignments", "live_argv")}, indent=1))
    und = [r for r in res["static_assignments"] if r["declared_field"] is False]
    print("STATIC: %d assignments, %d to an UNDECLARED field:" % (
        len(res["static_assignments"]), len(und)))
    for r in und:
        print("  %s:%d  %s  (on %s)" % (r["file"], r["line"], r["target"], r["owner_class"]))
    enc = [r for r in res["static_assignments"] if r["under_encoder"]]
    print("STATIC: %d assignments under cfg.core.encoder:" % len(enc))
    for r in enc:
        print("  %s:%d  %s  declared=%s" % (r["file"], r["line"], r["target"],
                                           r["declared_field"]))


if __name__ == "__main__":
    main(sys.argv[1])
