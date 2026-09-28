#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume a PRE-FREEZE refcv7 ``ckpt.pt`` into the FROZEN build (restart item 2).

WHY THIS IS NEEDED. The freeze (`apply_freeze.py`) takes 20 tensors out of the optimiser. The
model state dict is unaffected (``requires_grad`` is not serialised: a strict load just works),
but ``torch.optim.Optimizer.load_state_dict`` matches parameters BY POSITION inside each group, and
both of the live run's AdamW groups now have fewer entries. The trainer's resume
(`refc_v3_train.train`: ``opt.load_state_dict(state["opt"])``) therefore REFUSES a pre-freeze
checkpoint with ``ValueError: loaded state dict contains a parameter group that doesn't match the
size of optimizer's group`` -- loudly, which is correct. This tool rewrites ``ckpt["opt"]`` so the
frozen optimiser loads it, and nothing else.

THE MAPPING IS DERIVED, THEN PROVEN.
* The OLD optimiser's order is `param_groups_dd`'s: every ``requires_grad`` parameter in
  ``named_parameters()`` order, split by the ``core.encoder.`` prefix into [encoder, head]. On the
  pre-freeze build ``requires_grad`` = the frozen build's ``requires_grad`` OR "inside a module
  the freeze declared" (`model._bypass_declared`).
* It is PROVEN against the checkpoint, never assumed: both saved group lengths must equal the
  derived ones, and EVERY saved state entry's ``exp_avg`` / ``exp_avg_sq`` shape must equal the
  shape of the parameter the mapping assigns to it (808 entries on the live run, 788 with state).
* ⛔ A dropped entry that CARRIES STATE is REFUSED: it would mean the "dead" tensor did receive a
  gradient at some step, i.e. the freeze would change the run. (MEASURED on the live ckpt at step
  1,500: all 20 dropped entries carry none.)
* The result is LOADED into a real AdamW built by the trainer's own `build_optimizer` on the frozen
  model before anything is written.

The model is built by the trainer ITSELF (`launch_gate.run_trainer_until(T, argv, "model")`, the
gate's G-EVAL capture: real `main()` -> `train()`, stopped at the model build, before any training
data is read), from the tree that will run.

Usage (on the restart host, the FROZEN tree, training stopped):
  python ckpt_freeze_convert.py --tree <frozen tree> --argv-file <launch argv json>
         --ckpt-in <out>/ckpt.pt --ckpt-out <out>/ckpt.pt.frozen [--pathmap <file>]
then move ckpt.pt aside (keep it) and put ckpt.pt.frozen in its place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ENC_PREFIX = "core.encoder."


class ConvertRefused(SystemExit):
    pass


def old_and_new_orders(model) -> dict:
    """-> {"names", "new": {"encoder": [...], "head": [...]}, "old": {...}, "dropped": [...]}."""
    declared = dict(getattr(model, "_bypass_declared", {}) or {})
    if not declared:
        raise ConvertRefused("[convert] the model declared NO bypassed module -- this is not the "
                             "frozen build (apply_freeze.py), nothing to convert")

    def under(n: str) -> bool:
        return any(n == p or n.startswith(p + ".") for p in declared)

    names, new_rg, old_rg = [], {}, {}
    for n, p in model.named_parameters():
        names.append(n)
        new_rg[n] = bool(p.requires_grad)
        old_rg[n] = bool(p.requires_grad) or under(n)
    dropped = [n for n in names if old_rg[n] and not new_rg[n]]

    def split(rg):
        return {"encoder": [n for n in names if rg[n] and n.startswith(ENC_PREFIX)],
                "head": [n for n in names if rg[n] and not n.startswith(ENC_PREFIX)]}
    return {"names": names, "declared": declared, "new": split(new_rg), "old": split(old_rg),
            "dropped": dropped}


def convert_opt_state(opt_sd: dict, model, *, allow_state_drop: bool = False) -> tuple[dict, dict]:
    """-> (new optimiser state dict, record). Pure: reads ``opt_sd`` and the model, writes nothing."""
    orders = old_and_new_orders(model)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    groups = opt_sd["param_groups"]
    state = opt_sd["state"]
    gnames = [g.get("name") for g in groups]
    if gnames != ["encoder", "head"]:
        raise ConvertRefused(f"[convert] saved param groups are {gnames}, not ['encoder', 'head'] "
                             f"(the dd recipe) -- REFUSED")
    rec: dict = {"declared": orders["declared"], "dropped": orders["dropped"],
                 "saved_group_sizes": [len(g["params"]) for g in groups],
                 "old_group_sizes": [len(orders["old"]["encoder"]), len(orders["old"]["head"])],
                 "new_group_sizes": [len(orders["new"]["encoder"]), len(orders["new"]["head"])]}
    if rec["saved_group_sizes"] == rec["new_group_sizes"] and not orders["dropped"]:
        raise ConvertRefused("[convert] nothing to do")
    if rec["saved_group_sizes"] == rec["new_group_sizes"]:
        raise ConvertRefused(f"[convert] the checkpoint's groups already have the FROZEN sizes "
                             f"{rec['new_group_sizes']} -- converted already? REFUSED")
    if rec["saved_group_sizes"] != rec["old_group_sizes"]:
        raise ConvertRefused(f"[convert] saved group sizes {rec['saved_group_sizes']} != the "
                             f"derived pre-freeze sizes {rec['old_group_sizes']} -- the checkpoint "
                             f"was not written by this build's pre-freeze twin. REFUSED")
    # old index -> name (torch numbers params sequentially across groups, in group order)
    old_names = orders["old"]["encoder"] + orders["old"]["head"]
    saved_idx = [i for g in groups for i in g["params"]]
    if len(saved_idx) != len(old_names) or len(set(saved_idx)) != len(saved_idx):
        raise ConvertRefused("[convert] saved param indices are not a clean enumeration")
    idx2name = dict(zip(saved_idx, old_names))
    # ⛔ PROOF of the mapping: every state entry's moment tensors have the mapped param's shape
    bad_shape = []
    for i, st in state.items():
        n = idx2name.get(i)
        if n is None:
            bad_shape.append((i, "no param"))
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and tuple(st[k].shape) != shapes[n]:
                bad_shape.append((i, n, k, tuple(st[k].shape), shapes[n]))
    rec["n_state_entries"] = len(state)
    rec["n_shape_checked"] = len(state) - len([b for b in bad_shape if b[1] == "no param"])
    if bad_shape:
        raise ConvertRefused(f"[convert] {len(bad_shape)} state entries do not fit the derived "
                             f"mapping (first: {bad_shape[:3]}) -- REFUSED")
    dropped_idx = [i for i, n in idx2name.items() if n in set(orders["dropped"])]
    with_state = [idx2name[i] for i in dropped_idx if i in state]
    rec["dropped_with_state"] = with_state
    if with_state and not allow_state_drop:
        raise ConvertRefused(f"[convert] {len(with_state)} dropped tensor(s) CARRY AdamW state "
                             f"({with_state[:4]}): they received a gradient on some step, so "
                             f"freezing them is NOT bit-identical for this run. REFUSED "
                             f"(--allow-state-drop records and proceeds)")
    name2old = {n: i for i, n in idx2name.items()}
    new_names = orders["new"]["encoder"] + orders["new"]["head"]
    new_state, new_groups, k = {}, [], 0
    for g, key in zip(groups, ("encoder", "head")):
        ng = {kk: vv for kk, vv in g.items() if kk != "params"}
        ng["params"] = []
        for n in orders["new"][key]:
            ng["params"].append(k)
            oi = name2old[n]
            if oi in state:
                new_state[k] = state[oi]
            k += 1
        new_groups.append(ng)
    assert k == len(new_names)
    rec["n_state_entries_out"] = len(new_state)
    rec["n_state_dropped"] = len(state) - len(new_state)
    return {"state": new_state, "param_groups": new_groups}, rec


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def build_frozen_model(tree: str, argv: list[str]):
    """The trainer's OWN build, stopped before any data is read (the launch gate's G-EVAL capture)."""
    sys.path[:0] = [str(Path(tree) / "stack"), str(Path(tree) / "stack" / "scripts"),
                    str(Path(tree) / "taniteval")]
    import tanitad
    tf = os.path.normcase(os.path.abspath(tanitad.__file__))
    if not tf.startswith(os.path.normcase(os.path.abspath(str(Path(tree) / "stack")))):
        raise ConvertRefused(f"[convert] tanitad imported from {tf}, not from {tree}")
    import launch_gate as LG
    import refc_v3_train as T
    cap = LG.run_trainer_until(T, list(argv), "model")
    return T, cap["model"], cap["args"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="the FROZEN tree (apply_freeze.py applied)")
    ap.add_argument("--argv-file", required=True, help="the launch argv JSON (object['argv'] or list)")
    ap.add_argument("--pathmap", default=None,
                    help="optional launch-gate path map (dev-box tests); Thor passes none")
    ap.add_argument("--ckpt-in", required=True)
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--allow-state-drop", action="store_true")
    ap.add_argument("--record", default=None, help="write the JSON record here too")
    a = ap.parse_args(argv)
    obj = json.loads(Path(a.argv_file).read_text(encoding="utf-8"))
    run_argv = list(obj["argv"] if isinstance(obj, dict) else obj)
    if a.pathmap:
        sys.path.insert(0, str(Path(a.tree) / "stack" / "scripts"))
        import launch_gate as LG
        pm = LG.parse_path_map([l.strip() for l in Path(a.pathmap).read_text(encoding="utf-8")
                                .splitlines() if l.strip() and not l.startswith("#")])
        out = []
        for f, vals in LG.flag_pairs(run_argv):
            out.append(f)
            out += [LG.map_path(v, pm) if (f != "--out" and LG._looks_like_path(v)) else v
                    for v in vals]
        run_argv = out
    # the model build never needs the compile wrapper (the launch gate's model job drops it too:
    # it wraps the backbone CALL only -- module tree and state_dict unchanged)
    dropped_flags = [f for f in ("--trunk-compile",) if f in run_argv]
    run_argv = [t for t in run_argv if t not in dropped_flags]
    import torch
    T, model, args = build_frozen_model(a.tree, run_argv)
    ck_in = Path(a.ckpt_in)
    state = torch.load(ck_in, map_location="cpu", weights_only=False)
    # the model state must load STRICTLY into the frozen build (requires_grad is not serialised)
    model.load_state_dict(state["model"], strict=True)
    new_opt, rec = convert_opt_state(state["opt"], model, allow_state_drop=a.allow_state_drop)
    # ⛔ PROOF: the trainer's own optimiser on the frozen model loads it
    opt = T.build_optimizer(model, args)
    opt.load_state_dict(new_opt)
    rec.update({"ckpt_in": str(ck_in), "ckpt_in_sha256": _sha256(ck_in), "step": int(state["step"]),
                "tree": str(a.tree), "argv_file": str(a.argv_file),
                "build_dropped_flags": dropped_flags,
                "loaded_into_build_optimizer": True, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                         time.gmtime())})
    state["opt"] = new_opt
    state["freeze_convert"] = rec
    ck_out = Path(a.ckpt_out)
    if ck_out.exists():
        raise ConvertRefused(f"[convert] {ck_out} exists -- REFUSED (never overwrite a checkpoint)")
    torch.save(state, ck_out)
    rec["ckpt_out_sha256"] = _sha256(ck_out)
    txt = json.dumps(rec, indent=1, default=str)
    if a.record:
        Path(a.record).write_text(txt, encoding="utf-8")
    print(txt)
    print(f"ZZCONVERT-OK-{rec['n_state_entries_out']}-{len(rec['dropped'])}ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
