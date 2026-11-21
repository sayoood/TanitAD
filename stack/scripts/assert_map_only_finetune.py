"""Post-run assert for a ``--r8-train-only-map`` run (PI decision D-R9-MAPHEAD-PRECISION, 2026-10-26): every tensor
OUTSIDE the three map-only modules must be BIT-IDENTICAL between the source checkpoint and the fine-tuned one -- in the
RAW weights AND in the EMA weights, buffers (BatchNorm running statistics, ``num_batches_tracked``, the class-prior
buffers, the anchors) included. Prints which map-only tensors changed and by how much.

    python stack/scripts/assert_map_only_finetune.py --src ckpt_ema_48852.pt --dst <run>/ckpt.pt [--json out.json]

EXIT: 0 = every non-map tensor is bit-identical in every view of ``--dst``; 1 = a non-map tensor differs, or the key
sets differ, or a view is missing; 2 = a file cannot be read as a trainer checkpoint.

THE FORMATS (read, never assumed -- ``stack/scripts/refc_v3_train.py``):
  * model-only files (``_save_ema_model`` / the milestone ``ckpt_<step>.pt`` / ``model_ema.pt`` / ``ckpt_0.pt``):
    ``{"model": state_dict, "step", ...}``; ``kind == "ema_model_only"`` marks an EMA file -> ONE view, ``ema`` or
    ``raw``;
  * the rolling ``ckpt.pt`` (the loop's save / ``_ckpt_payload``): ``{"model": RAW state_dict, "opt", "step",
    "data_pos", "param_names", "ema": {"schema": "weight_ema/1", "decay", "num_updates", "shadow": {name: tensor}}}``
    -> TWO views: ``raw`` = ``model``; ``ema`` = ``model`` with every shadowed entry replaced by its shadow cast to
    the live dtype (exactly ``WeightEMA.model_state_dict``).
  The REFERENCE is the ``--src`` view named by ``--src-view`` (default: ``ema`` when the file has one, else ``raw``;
  the shipped refcv8 planner is the EMA one).

⛔ INDEPENDENT of the trainer by construction: the map-only module paths are a LITERAL here (the PI's three, the same
literal the trainer's ``map_only_ft.MAP_ONLY_MODULES`` carries -- a test holds the two equal), and bit equality is a
comparison of raw BYTES (a -0.0 differs from a +0.0), not ``torch.equal``.

THE r9 FORM (refcv9 WARM START, ``--r9-train-only <sets>``; MM 2026-10-27):

    python stack/scripts/assert_map_only_finetune.py --src ckpt_ema_48852.pt --dst <run>/ckpt.pt \
        --sets planner,tactical,map [--also-frozen core.decoder.offset_head ...] [--require-each-module-moved]

  * ``--sets``: the TRAINED modules are the union of the named sets -- :data:`TRAIN_ONLY_SETS`, a LITERAL here (a test
    holds it equal to ``map_only_ft.TRAIN_ONLY_SETS``). Every tensor OUTSIDE them -- the PERCEPTION -- must be
    bit-identical in every view, exactly as the map-only form demands of the planner.
  * ``--also-frozen PREFIX`` (repeatable): a subtree INSIDE a set that must ALSO be bit-identical (the build's
    declared grad-unreachable subtrees: the launch gate's ``map_only["declared_frozen"]`` literal). It is checked as
    frozen and named in the output.
  * the output NAMES the changed tensors PER SET and per module (``per_set``); ``--require-each-module-moved`` fails
    a view in which some module of the sets has NO changed tensor (the tool alone passes a run that trained nothing).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

#: the PI's three map-only modules (D-R9-MAPHEAD-PRECISION), dot-boundary prefixes of state_dict keys
MAP_ONLY_MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
#: ``--r9-train-only``'s sets (refcv9 warm start), in their canonical order -- LITERALS (the trainer's
#: ``map_only_ft.TRAIN_ONLY_SETS`` is held equal by a test, never read)
TRAIN_ONLY_SETS = {
    "planner": ("core.decoder",),
    "tactical": ("phi_tac", "nav_to_tac", "lat_head_tac", "lon_head_tac", "tac_goal_head", "tac_latent_proj",
                 "tac_decoder_v6", "tac_behaviour_gate_v6", "core.tactical_trunk", "core.lat_head", "core.lon_head"),
    "map": MAP_ONLY_MODULES}
EMA_SCHEMA = "weight_ema/1"


def _under(key: str, prefixes) -> bool:
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def is_map_only(key: str) -> bool:
    return _under(key, MAP_ONLY_MODULES)


def parse_sets(spec: str) -> tuple:
    """``"planner,map"`` -> ``("planner", "map")`` in the canonical order; refuses an empty, unknown or repeated name."""
    parts = [x.strip() for x in str(spec).split(",")]
    bad = [x for x in parts if x not in TRAIN_ONLY_SETS]
    if not parts or bad or len(set(parts)) != len(parts):
        raise ValueError(f"--sets {spec!r}: the sets are {list(TRAIN_ONLY_SETS)}, each named once (bad: {bad})")
    return tuple(x for x in TRAIN_ONLY_SETS if x in parts)


def tensor_bytes(t: torch.Tensor) -> bytes:
    x = t.detach().to("cpu").reshape(-1)
    dense = torch.empty(x.shape, dtype=x.dtype)
    dense.copy_(x)
    return dense.view(torch.uint8).numpy().tobytes() if dense.numel() else b""


def same_bits(a, b) -> bool:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return a == b
    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and tensor_bytes(a) == tensor_bytes(b))


def _load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:                                     # noqa: BLE001 -- legacy (non-zip) files cannot mmap
        return torch.load(path, map_location="cpu", weights_only=False)


def views(path: str) -> tuple[dict, dict]:
    """-> ({view name: state_dict}, record). Raises ValueError on a file that is not a trainer checkpoint."""
    st = _load(path)
    if not isinstance(st, dict) or not isinstance(st.get("model"), dict) or not st["model"]:
        raise ValueError(f"{path}: no `model` state_dict -- not a refc_v3_train checkpoint")
    sd = st["model"]
    rec = {"path": str(path), "step": st.get("step"), "kind": st.get("kind"),
           "top_keys": sorted(k for k in st if k != "model"), "n_keys": len(sd)}
    if "ema" in st:
        ema = st["ema"]
        if not isinstance(ema, dict) or ema.get("schema") != EMA_SCHEMA or not isinstance(ema.get("shadow"), dict):
            raise ValueError(f"{path}: `ema` is not a {EMA_SCHEMA} state")
        sh = ema["shadow"]
        unknown = sorted(k for k in sh if k not in sd)
        if unknown:
            raise ValueError(f"{path}: the EMA shadows keys the model does not have, e.g. {unknown[:3]}")
        ema_sd = {k: (sh[k].to(dtype=v.dtype) if k in sh else v) for k, v in sd.items()}
        rec.update(ema_decay=ema.get("decay"), ema_num_updates=ema.get("num_updates"), n_shadow=len(sh),
                   n_shadow_outside_map_only=sum(1 for k in sh if not is_map_only(k)))
        return {"raw": sd, "ema": ema_sd}, rec
    return ({"ema": sd} if st.get("kind") == "ema_model_only" else {"raw": sd}), rec


def compare(ref: dict, got: dict, sets: tuple | None = None, also_frozen: tuple = (),
            require_moved: bool = False) -> dict:
    """One view against the reference -> {"ok", "missing", "extra", "non_map_changed", "map_changed", ...}. With
    ``sets`` the trained modules are the sets' union (minus ``also_frozen``), every other key is FROZEN (the
    perception), and ``per_set`` names the changed tensors of each set and module."""
    mods = MAP_ONLY_MODULES if sets is None else tuple(m for s_ in sets for m in TRAIN_ONLY_SETS[s_])
    also = tuple(also_frozen)

    def trained(k):
        return _under(k, mods) and not _under(k, also)
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    non_map, map_changed, n_non_map, n_map = [], [], 0, 0
    for k in sorted(set(ref) & set(got)):
        a, b = ref[k], got[k]
        if trained(k):
            n_map += 1
            if not same_bits(a, b):
                row = {"key": k, "shape": list(getattr(b, "shape", []))}
                if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.is_floating_point():
                    d = (b.double() - a.double())
                    row["max_abs_diff"] = float(d.abs().max()) if d.numel() else 0.0
                    na = float(a.double().norm())
                    row["rel_norm_change"] = (float(d.norm()) / na) if na > 0 else math.inf
                map_changed.append(row)
        else:
            n_non_map += 1
            if not same_bits(a, b):
                non_map.append({"key": k,
                                "dtype": [str(getattr(a, "dtype", type(a))), str(getattr(b, "dtype", type(b)))],
                                "shape": [list(getattr(a, "shape", [])), list(getattr(b, "shape", []))]})
    res = {"ok": not (missing or extra or non_map), "n_non_map_checked": n_non_map, "n_map_only": n_map,
           "missing": missing[:20], "n_missing": len(missing), "extra": extra[:20], "n_extra": len(extra),
           "non_map_changed": non_map[:50], "n_non_map_changed": len(non_map),
           "map_changed": map_changed, "n_map_changed": len(map_changed)}
    if sets is None:
        return res
    keys = sorted(set(ref) & set(got))
    changed = {r["key"] for r in map_changed}
    per_set, unmoved = {}, []
    for s_ in sets:
        pm = {}
        for m in TRAIN_ONLY_SETS[s_]:
            km = [k for k in keys if _under(k, (m,)) and trained(k)]
            pm[m] = {"n_tensors": len(km), "n_changed": sum(1 for k in km if k in changed)}
            if not pm[m]["n_changed"]:
                unmoved.append(m)
        per_set[s_] = {"modules": list(TRAIN_ONLY_SETS[s_]), "per_module": pm,
                       "n_tensors": sum(v["n_tensors"] for v in pm.values()),
                       "changed": [r["key"] for r in map_changed if _under(r["key"], TRAIN_ONLY_SETS[s_])],
                       "n_changed": sum(v["n_changed"] for v in pm.values())}
    also_rows = {a: sum(1 for k in keys if _under(k, (a,))) for a in also}
    res.update({"sets": list(sets), "per_set": per_set, "also_frozen": also_rows,
                "n_frozen_checked": n_non_map, "n_frozen_changed": len(non_map), "frozen_changed": non_map[:50],
                "modules_unmoved": unmoved})
    if require_moved and unmoved:
        res["ok"] = False
    return res


def run(src: str, dst: str, src_view: str | None = None, dst_views=None, sets: tuple | None = None,
        also_frozen: tuple = (), require_moved: bool = False) -> dict:
    sv, srec = views(src)
    if src_view is None:
        src_view = "ema" if "ema" in sv else "raw"
    if src_view not in sv:
        raise ValueError(f"--src {src} has views {sorted(sv)}, not {src_view!r}")
    dv, drec = views(dst)
    want = list(dst_views) if dst_views else sorted(dv)
    out = {"tool": "assert_map_only_finetune.py", "map_only_modules": list(MAP_ONLY_MODULES),
           "src": srec, "src_view": src_view, "dst": drec, "views": {}}
    if sets is not None:
        out.update({"sets": list(sets), "trained_modules": [m for s_ in sets for m in TRAIN_ONLY_SETS[s_]],
                    "also_frozen": list(also_frozen), "require_each_module_moved": bool(require_moved)})
    ok = True
    for name in want:
        if name not in dv:
            out["views"][name] = {"ok": False, "error": f"--dst has no `{name}` view (has {sorted(dv)})"}
            ok = False
            continue
        r = (compare(sv[src_view], dv[name]) if sets is None else
             compare(sv[src_view], dv[name], sets, tuple(also_frozen), require_moved))
        out["views"][name] = r
        ok = ok and r["ok"]
    out["ok"] = bool(ok)
    return out


def _print(res: dict) -> None:
    print(f"[assert-map-only] src {res['src']['path']} (view `{res['src_view']}`, step {res['src']['step']}) -> "
          f"dst {res['dst']['path']} (step {res['dst']['step']}, views {sorted(res['views'])})")
    for name, r in res["views"].items():
        if "error" in r:
            print(f"  [{name}] FAIL: {r['error']}")
            continue
        print(f"  [{name}] non-map tensors checked {r['n_non_map_checked']}, changed {r['n_non_map_changed']}; "
              f"keys missing {r['n_missing']}, extra {r['n_extra']}; map-only tensors {r['n_map_only']}, changed "
              f"{r['n_map_changed']} -> {'PASS' if r['ok'] else 'FAIL'}")
        for row in r["non_map_changed"][:20]:
            print(f"    NON-MAP CHANGED: {row['key']} {row['dtype']} {row['shape']}")
        for k in r["missing"][:10]:
            print(f"    MISSING in dst: {k}")
        for k in r["extra"][:10]:
            print(f"    EXTRA in dst: {k}")
        if "per_set" not in r:
            for row in r["map_changed"]:
                print(f"    map changed: {row['key']} {row['shape']} max|d|={row.get('max_abs_diff')} "
                      f"rel={row.get('rel_norm_change')}")
            continue
        print(f"    perception (every tensor outside the sets {r['sets']}) + also-frozen {r['also_frozen']}: "
              f"{r['n_frozen_checked']} checked, {r['n_frozen_changed']} changed")
        for s_, e in r["per_set"].items():
            print(f"    set `{s_}`: {e['n_changed']} of {e['n_tensors']} tensors changed; per module "
                  f"{ {m: v['n_changed'] for m, v in e['per_module'].items()} }")
            for k in e["changed"]:
                print(f"      {s_} changed: {k}")
        if r["modules_unmoved"]:
            print(f"    {'FAIL' if res.get('require_each_module_moved') else 'NOTE'}: modules with NO changed tensor: "
                  f"{r['modules_unmoved']}")
    print(f"[assert-map-only] {'PASS' if res['ok'] else 'FAIL'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the source checkpoint (the --init-from file, ckpt_ema_48852.pt)")
    ap.add_argument("--dst", required=True, help="the fine-tuned checkpoint (ckpt.pt, model_ema.pt, ckpt_<step>.pt)")
    ap.add_argument("--src-view", choices=("raw", "ema"), default=None,
                    help="the reference view of --src (default: ema when present, else raw)")
    ap.add_argument("--dst-view", choices=("raw", "ema"), action="append", default=None,
                    help="restrict to these --dst views (default: every view the file has)")
    ap.add_argument("--json", default=None, help="also write the full result here")
    ap.add_argument("--sets", default=None, metavar="SET[,SET]",
                    help="--r9-train-only's sets (planner / tactical / map): the trained modules are their union and "
                         "every other tensor (the perception) must be bit-identical. Absent: the map-only form")
    ap.add_argument("--also-frozen", action="append", default=None, metavar="PREFIX",
                    help="with --sets: a subtree INSIDE a set that must ALSO be bit-identical (declared frozen)")
    ap.add_argument("--require-each-module-moved", action="store_true",
                    help="with --sets: FAIL a view in which some module of the sets has no changed tensor")
    a = ap.parse_args(argv)
    if a.sets is None and (a.also_frozen or a.require_each_module_moved):
        print("[assert-map-only] ERROR: --also-frozen / --require-each-module-moved need --sets", file=sys.stderr)
        return 2
    try:
        sets = None if a.sets is None else parse_sets(a.sets)
        res = run(a.src, a.dst, a.src_view, a.dst_view, sets, tuple(a.also_frozen or ()),
                  bool(a.require_each_module_moved))
    except (ValueError, OSError, RuntimeError) as e:
        print(f"[assert-map-only] ERROR: {e}", file=sys.stderr)
        return 2
    _print(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
