"""cfg_parity_probe_r8.py -- CPU, NO MODEL: does the clamp tree (de4e15b) build the SAME config as the launch tree (fb47a58) from refcv8's own argv?   NEW file.

    # one process per tree (module names collide otherwise); each dumps the parsed args + the pinned model config as JSON
    R8_PROBE_TREE=C:/lgs/Lfb47a58376            PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval'           python cfg_parity_probe_r8.py dump --out fb47a58.json
    R8_PROBE_TREE=E:/refcv8_eval_kit/tree_de4e15b PYTHONPATH='E:/refcv8_eval_kit/tree_de4e15b/stack;E:/refcv8_eval_kit/tree_de4e15b/taniteval' python cfg_parity_probe_r8.py dump --out de4e15b.json
    python cfg_parity_probe_r8.py diff fb47a58.json de4e15b.json --out CFG_PARITY.json

WHY.  K2 (flag-OFF bit parity, GPU) asks whether the clamp-carrying tree rolls the banked refcv8 numbers.  Everything that could break it BEFORE the first forward is a CONFIG
difference: a new parser default, a new field, a changed pin.  The loader builds the model from `config.json['argv']` through the trainer's parser and `_pin_trainer_cfg`
(`refcv7_loader._build_model`).  This probe replays exactly those two steps (no model, no checkpoint, no data file) in each tree and diffs the result.  It cannot see a
change inside a forward; it CAN refute a config-level cause of a K2 failure, and it names every parser/config field the commit added.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CONFIG = "E:/refcv8_eval_kit/ckpt/config.json"


def _plain(o, depth=0):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _plain(getattr(o, f.name), depth + 1) for f in dataclasses.fields(o)}
    if isinstance(o, dict):
        return {str(k): _plain(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v, depth + 1) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return repr(o)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        out[prefix] = json.dumps(d, default=str)
    else:
        out[prefix] = d
    return out


def cmd_dump(a) -> int:
    tree = os.environ.get("R8_PROBE_TREE")
    if not tree:
        raise SystemExit("set R8_PROBE_TREE to the tree this process imports from")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import r8_repoint
    r8_repoint.LAUNCH_TREE = Path(tree)
    from r8clamp_common import BATTERY_CODE
    r8_repoint.BATTERY_CODE = Path(BATTERY_CODE)
    r8_repoint.configure_env()
    from tanitad.eval import refcv7_loader as L
    L.bootstrap()
    import tanitad
    if not Path(tanitad.__file__).resolve().as_posix().lower().startswith(Path(tree).resolve().as_posix().lower()):
        raise SystemExit(f"tanitad imported from {tanitad.__file__}, not {tree}")
    tr = L.trainer()
    from tanitad.refs import refc_v3 as v3
    config = L.load_config(CONFIG)
    args, argv, arec = L.parse_args(config)
    cfg = tr._pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    res = {"tree": tree, "tanitad_file": tanitad.__file__, "loader_file": L.__file__, "n_argv_tokens": len(argv),
           "args": _plain(vars(args)), "cfg": _plain(cfg)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"out": a.out, "n_args": len(res["args"]), "n_cfg_leaves": len(flatten(res["cfg"]))}))
    return 0


def cmd_diff(a) -> int:
    A, B = json.loads(Path(a.a).read_text(encoding="utf-8")), json.loads(Path(a.b).read_text(encoding="utf-8"))
    out = {"a": a.a, "b": a.b, "a_tree": A["tree"], "b_tree": B["tree"], "same_argv_tokens": A["n_argv_tokens"] == B["n_argv_tokens"]}
    for sect in ("args", "cfg"):
        fa, fb = flatten(A[sect]), flatten(B[sect])
        out[sect] = {"only_in_a": sorted(set(fa) - set(fb)), "only_in_b": {k: fb[k] for k in sorted(set(fb) - set(fa))},
                     "changed": {k: {"a": fa[k], "b": fb[k]} for k in sorted(set(fa) & set(fb)) if fa[k] != fb[k]},
                     "n_a": len(fa), "n_b": len(fb)}
    ch = out["args"]["changed"] | out["cfg"]["changed"]
    out["verdict"] = ("CONFIG-IDENTICAL on every field both trees declare; the commit only ADDS the fields listed in only_in_b" if not ch and not out["args"]["only_in_a"]
                      and not out["cfg"]["only_in_a"] else "CONFIG DIFFERS -- read `changed` / `only_in_a`: a K2 failure may have a config cause")
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "args_changed": len(out["args"]["changed"]), "cfg_changed": len(out["cfg"]["changed"]),
                      "args_added": len(out["args"]["only_in_b"]), "cfg_added": len(out["cfg"]["only_in_b"])}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    f = sub.add_parser("diff")
    f.add_argument("a")
    f.add_argument("b")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_dump(a) if a.cmd == "dump" else cmd_diff(a)


if __name__ == "__main__":
    sys.exit(main())
