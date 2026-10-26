"""Map-only fine-tune: ``--r8-train-only-map`` (PI decision D-R9-MAPHEAD-PRECISION, 2026-10-26).

THE DECISION (PI, 2026-10-26, recommended option (a), cap 6 Thor-hours): fine-tune ONLY the map-only modules of the
shipped refcv8 model -- ``refine`` + ``near`` + ``near_refine`` of the 10 cm map head, 105,032 parameters -- with the
planner BIT-IDENTICAL, asserted, and a launch-gate route for the freeze-all-but-map trainer flag.

WHAT IS MAP-ONLY, AND WHY EXACTLY THESE THREE
    ``MapHiresBranch.forward`` (``tanitad/models/map_head_hires.py``) computes ``feats = encoder(lift(fmap_s8))`` and
    returns it as ``map_hires_bev`` -- the planner's pooled BEV reads THAT (``refc_v3.py``: ``br(fmap_s16,
    bev_hires=hi["map_hires_bev"])``). ``refine`` / ``near`` / ``near_refine`` only turn ``feats`` (+ the stride-8 map,
    for the near lift) into the 10 cm logits; nothing downstream of the planner reads the logits in a build without
    the drivable critic (``--r8-critic-drivable``, absent from the refcv8-r101-s0 48,852 config and REFUSED here).
    ``grad_reach_report_hires`` already names them map-only; ``lift`` / ``encoder`` / the trunk's stride-8 stage are
    SHARED with the planner and stay FROZEN.

    MEASURED 2026-10-26 on ``E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`` (md5 b4364029b68f29a33eeb4aebbbbd002b):
    refine 20,936 + near 65,600 + near_refine 18,496 = 105,032 values in 12 leaf groups (== the launch gate's
    ``_NT_MAP_ONLY_GROUPS``, read independently off the NT smoke's ``ckpt_0.pt``).

WHAT "FROZEN" MUST MEAN -- every route by which a frozen tensor can move is closed, and then CHECKED
    * parameters: ``requires_grad=False`` and absent from the optimizer (one AdamW group, ``map_only``);
    * running statistics: every norm with running stats OUTSIDE the map-only modules is pinned to eval and its
      ``train`` is replaced (the ``timm_trunk._freeze_bn_`` idiom: ``model.train()`` recurses every step);
    * trainer-side buffer writes: ``core.update_tactical_prior`` (the lat/lon class-prior EMA) is skipped under the
      flag (``compute_losses_v3`` reads :data:`MARKER_ATTR`);
    * the weight EMA: built with ``exclude=ema_exclude``, so it shadows the map-only tensors ONLY. A frozen tensor's
      EMA is itself; shadowing it would still move it -- ``lerp(x, x, w)`` turns a -0.0 into +0.0 -- and the
      exported EMA model would not be bit-identical;
    * the warm start reads the SHIPPED EMA file (``kind == "ema_model_only"``), so raw == EMA == the shipped model at
      step 0 for every tensor, asserted bit for bit;
    * :class:`FrozenStateGuard` re-reads every non-map tensor (parameters AND buffers) against the init file at every
      logged / saved step and REFUSES on the first drifted byte.

EVERY EXPECTATION IS A LITERAL: the three module paths, the 105,032 / 12-group refcv8 count, the init file's kind.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

__all__ = ["SCHEMA", "FLAG", "DEST", "MAP_ONLY_MODULES", "MARKER_ATTR", "GROUP_NAME", "INIT_KIND",
           "REFCV8_MAP_ONLY_NUMEL", "REFCV8_MAP_ONLY_GROUPS", "REFCV8_EMA_48852_MD5", "MapOnlyError",
           "is_map_only", "is_on", "pin_args", "apply_freeze", "pin_frozen_norms", "leaf_groups",
           "build_optimizer", "optimizer_problems", "bit_equal", "load_source", "verify_against_source",
           "FrozenStateGuard", "ema_exclude", "ema_problems", "grad_dependent_tensors", "dvb_check",
           "register_dvb_levers"]

SCHEMA = "map_only_ft/1"
FLAG = "--r8-train-only-map"
DEST = "r8_train_only_map"
#: the map-only modules, by attribute path on the BUILT model (dot-boundary prefixes: ``_map_hires.near`` does NOT
#: cover ``_map_hires.near_refine``)
MAP_ONLY_MODULES: tuple = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
#: the freeze record the BUILT model carries (read BY NAME by ``declared_vs_built.check_grad_unreachable``)
MARKER_ATTR = "_tanitad_map_only_ft"
#: the optimizer's one parameter group under the flag
GROUP_NAME = "map_only"
#: the ``--init-from`` file the flag warm-starts from: ``refc_v3_train._save_ema_model``'s layout
INIT_KIND = "ema_model_only"
#: refcv8-r101-s0 (48,852 config) -- MEASURED off ckpt_ema_48852.pt, see the module docstring
REFCV8_MAP_ONLY_NUMEL = 105_032
REFCV8_MAP_ONLY_GROUPS: dict = {
    "_map_hires.refine.inp.0": 2_048, "_map_hires.refine.inp.1": 64, "_map_hires.refine.conv1": 9_216,
    "_map_hires.refine.norm1": 64, "_map_hires.refine.conv2": 9_216, "_map_hires.refine.norm2": 64,
    "_map_hires.refine.cls": 264,
    "_map_hires.near.lift": 32, "_map_hires.near.lift.proj": 65_568,
    "_map_hires.near_refine.0.c1": 9_216, "_map_hires.near_refine.0.n1": 64, "_map_hires.near_refine.0.c2": 9_216}
#: the shipped refcv8 EMA model (E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt, ckpt_48852.md5)
REFCV8_EMA_48852_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"


class MapOnlyError(SystemExit):
    """A refusal of the map-only fine-tune (a ``SystemExit``, like every other trainer refusal)."""


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def is_map_only(name: str) -> bool:
    """True for a parameter / buffer / module path inside one of :data:`MAP_ONLY_MODULES`."""
    return any(_under(str(name), p) for p in MAP_ONLY_MODULES)


def ema_exclude(name: str) -> bool:
    """``WeightEMA(exclude=...)``: every tensor OUTSIDE the map-only modules is frozen, so its EMA is itself."""
    return not is_map_only(name)


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


# ================================================================================================================= #
# argv                                                                                                              #
# ================================================================================================================= #
def pin_args(args) -> None:
    """The flag's refusals, before config.json and before any model is built. A no-op when the flag is off.

    Needs: ``--refcv8``, ``--map-hires on`` with the near lift AND a near refine block (the PI named all three
    modules), a positive ``--w-map-hires`` (else nothing trains), ``--opt dd`` (the recipe the map head was trained
    with: AdamW, wd 1e-4, the head rate), and ``--init-from`` (the shipped EMA model).
    Refuses every switch that would read or write a FROZEN tensor or measure a gradient that cannot exist:
    the B1 guard (box head frozen; no head/encoder group to cut), the gradient-share instrument and the conflict
    detector (both read trunk / BEV gradients; the trunk and the BEV are frozen), BN recalibration (rewrites the
    frozen trunk statistics), and the drivable critic (the planner would READ the map head -- the map-only modules
    would stop being map-only)."""
    if not is_on(args):
        return
    bad = []
    if not bool(getattr(args, "refcv8", False)):
        bad.append("--refcv8 is not set: the map-only fine-tune is defined on the refcv8 build")
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: there is no 10 cm map head to fine-tune")
    if not float(getattr(args, "map_hires_near_lift_m", 0.0) or 0.0) > 0.0:
        bad.append("--map-hires-near-lift-m is 0: the `near` module (one of the three the PI named) is not built")
    if not int(getattr(args, "map_hires_near_refine_blocks", 0) or 0) >= 1:
        bad.append("--map-hires-near-refine-blocks is 0: the `near_refine` module (one of the three) is not built")
    if not float(getattr(args, "w_map_hires", 0.0) or 0.0) > 0.0:
        bad.append("--w-map-hires is 0: the map-only modules would receive no gradient -- nothing would train")
    if str(getattr(args, "opt", "adam")) != "dd":
        bad.append("--opt is not dd: the map head was trained in refcv8's AdamW head group (wd 1e-4, the head rate)")
    if not getattr(args, "init_from", None):
        bad.append("--init-from is not set: the fine-tune starts from the SHIPPED refcv8 EMA model "
                   "(ckpt_ema_48852.pt), or the frozen planner is not the shipped planner")
    if getattr(args, "b1_guard_ref_ap2m", None) is not None:
        bad.append("--b1-guard-ref-ap2m: the B1 guard cuts the head LR on the EMA box3d AP -- the box head is FROZEN "
                   "(its AP cannot move) and there is no head/encoder group to cut")
    if int(getattr(args, "grad_share_every", 0) or 0) > 0:
        bad.append("--grad-share-every > 0: the instrument measures the trunk / bev025 / bev_pool groups, which are "
                   "FROZEN -- it would read nothing (and its fp32 replay is a full extra forward)")
    if str(getattr(args, "conflict_detector", "auto") or "auto") != "off":
        bad.append("--conflict-detector is not `off`: it measures planner-vs-perception gradient conflicts on the "
                   "trunk, which is FROZEN -- pass --conflict-detector off")
    if int(getattr(args, "conflict_every", 1) or 1) != 1:
        bad.append("--conflict-every is set: the conflict detector is off under the flag, so the cadence would be "
                   "stamped and read by nothing -- drop it")
    if int(getattr(args, "trunk_bn_recalib", 0) or 0) > 0:
        bad.append("--trunk-bn-recalib: it REWRITES the frozen trunk's BatchNorm statistics")
    if bool(getattr(args, "r8_critic_drivable", False)):
        bad.append("--r8-critic-drivable: the planner's selection would READ the map head, so the three modules "
                   "would no longer be map-only and the planner could not stay bit-identical in function")
    if bad:
        raise MapOnlyError(f"[map-only] ⛔ {FLAG} refused:\n  - " + "\n  - ".join(bad))


# ================================================================================================================= #
# the freeze                                                                                                        #
# ================================================================================================================= #
def _resolve(model, path: str):
    node = model
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def leaf_groups(names_numel: Iterable[tuple]) -> dict:
    """``{leaf group: numel}`` -- the leaf group of ``a.b.weight`` is ``a.b`` (the gate's ``grad_table`` rule)."""
    out: dict = {}
    for n, k in names_numel:
        g = n.rsplit(".", 1)[0] if "." in n else n
        out[g] = out.get(g, 0) + int(k)
    return out


def pin_frozen_norms(model) -> int:
    """Pin every norm with RUNNING statistics outside the map-only modules to eval, permanently (its ``train`` is
    replaced: ``nn.Module.train`` recurses, and the trainer calls ``model.train()`` after every eval). -> how many."""
    k = 0
    for name, m in model.named_modules():
        if is_map_only(name):
            continue
        has_stats = (isinstance(m, nn.modules.batchnorm._BatchNorm)
                     or (isinstance(m, nn.modules.instancenorm._InstanceNorm)
                         and bool(getattr(m, "track_running_stats", False))))
        if not has_stats:
            continue
        m.eval()
        m.train = (lambda mode=True, _m=m: _m)   # noqa: E731 -- deliberate (timm_trunk._freeze_bn_)
        k += 1
    return k


def apply_freeze(model) -> dict:
    """Freeze everything but the three map-only modules; pin the frozen norms; stamp the record on the model.
    -> the record ``config.json["map_only_ft"]`` carries. REFUSES a build that lacks any of the three modules, and a
    parameter TIED across the boundary (one tensor that is both map-only and planner)."""
    missing = [p for p in MAP_ONLY_MODULES if _resolve(model, p) is None]
    if missing:
        raise MapOnlyError(f"[map-only] ⛔ {FLAG}: the built model has no {missing} -- the PI's three map-only "
                           f"modules must all be built (--map-hires on, --map-hires-near-lift-m > 0, "
                           f"--map-hires-near-refine-blocks >= 1)")
    owners: dict = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        owners.setdefault(id(p), set()).add(is_map_only(n))
    tied = [n for n, p in model.named_parameters() if len(owners[id(p)]) > 1]
    if tied:
        raise MapOnlyError(f"[map-only] ⛔ {FLAG}: {len(tied)} parameter(s) are TIED across the map-only boundary, "
                           f"e.g. {tied[:3]} -- training them would move the planner")
    trainable, n_frozen, numel_frozen = [], 0, 0
    for n, p in model.named_parameters():
        want = is_map_only(n)
        p.requires_grad_(want)
        if want:
            trainable.append((n, int(p.numel())))
        else:
            n_frozen += 1
            numel_frozen += int(p.numel())
    if not trainable:
        raise MapOnlyError(f"[map-only] ⛔ {FLAG}: the map-only modules hold no parameter")
    pinned = pin_frozen_norms(model)
    rec = {"schema": SCHEMA, "flag": FLAG, "modules": list(MAP_ONLY_MODULES),
           "trainable": [n for n, _ in trainable],
           "n_trainable_tensors": len(trainable),
           "n_trainable_params": int(sum(k for _, k in trainable)),
           "trainable_leaf_groups": leaf_groups(trainable),
           "n_frozen_tensors": n_frozen, "n_frozen_params": int(numel_frozen),
           "frozen_norms_pinned": int(pinned),
           "buffer_writes_skipped": ["core.update_tactical_prior (lat/lon class-prior EMA)"],
           "ema": "shadows the map-only tensors only (WeightEMA exclude=ema_exclude)"}
    setattr(model, MARKER_ATTR, rec)
    return rec


# ================================================================================================================= #
# the optimizer                                                                                                     #
# ================================================================================================================= #
def build_optimizer(model, args):
    """refcv8's recipe for the map head -- AdamW, ``--weight-decay`` (1e-4), the HEAD rate ``--lr`` -- as ONE group
    ``map_only`` holding exactly the trainable parameters (the map-only ones, :func:`apply_freeze`)."""
    if getattr(model, MARKER_ATTR, None) is None:
        raise MapOnlyError(f"[map-only] ⛔ {FLAG}: build_optimizer before apply_freeze -- the optimizer would hold "
                           f"the whole model")
    params = [p for _n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    stray = [n for n in names if not is_map_only(n)]
    if stray or not params:
        raise MapOnlyError(f"[map-only] ⛔ {FLAG}: the trainable set is not the map-only modules "
                           f"(outside: {stray[:3]}, n trainable {len(params)})")
    opt = torch.optim.AdamW([{"params": params, "lr": float(args.lr),
                              "weight_decay": float(getattr(args, "weight_decay", 1e-4)),
                              "name": GROUP_NAME}], lr=float(args.lr))
    print(f"[map-only] opt AdamW wd={getattr(args, 'weight_decay', 1e-4)} group `{GROUP_NAME}` lr={float(args.lr):.3e} "
          f"({len(params)} tensors, {sum(int(p.numel()) for p in params)} params); everything else FROZEN", flush=True)
    return opt


def optimizer_problems(opt, model, args) -> list:
    """What makes ``opt`` NOT the map-only recipe ([] = it is): AdamW, one group named ``map_only`` at ``--lr`` with
    ``--weight-decay``, holding exactly the map-only parameters (by identity), each trainable; no other parameter
    trainable."""
    bad = []
    if type(opt).__name__ != "AdamW":
        bad.append(f"optimizer is {type(opt).__name__}, the recipe is AdamW")
    if [g.get("name") for g in opt.param_groups] != [GROUP_NAME]:
        bad.append(f"param groups {[g.get('name') for g in opt.param_groups]} != ['{GROUP_NAME}']")
    want = {id(p) for n, p in model.named_parameters() if is_map_only(n)}
    held = [p for g in opt.param_groups for p in g["params"]]
    if {id(p) for p in held} != want or len(held) != len(want):
        bad.append(f"the optimizer holds {len(held)} tensors, the map-only modules {len(want)} (not the same set)")
    if any(not p.requires_grad for p in held):
        bad.append("a tensor in the optimizer is frozen")
    live = [n for n, p in model.named_parameters() if p.requires_grad and not is_map_only(n)]
    if live:
        bad.append(f"{len(live)} parameter(s) OUTSIDE the map-only modules are trainable, e.g. {live[:3]}")
    for g in opt.param_groups:
        if abs(float(g.get("weight_decay", 0.0)) - float(getattr(args, "weight_decay", 1e-4))) > 1e-12:
            bad.append(f"group weight_decay {g.get('weight_decay')} != {getattr(args, 'weight_decay', 1e-4)}")
        lr0 = float(g.get("initial_lr", g.get("lr")))
        if abs(lr0 - float(args.lr)) > 1e-15:
            bad.append(f"group peak lr {lr0} != --lr {args.lr}")
    return bad


# ================================================================================================================= #
# bit identity: the source, the guard, the EMA                                                                      #
# ================================================================================================================= #
_INT_VIEW = {torch.float32: torch.int32, torch.float64: torch.int64, torch.float16: torch.int16,
             torch.bfloat16: torch.int16}


def bit_equal(a, b) -> bool:
    """Same dtype, same shape, same BITS (a -0.0 differs from +0.0; a NaN equals itself bit for bit)."""
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return False
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    a, b = a.detach(), b.detach()
    if b.device != a.device:
        b = b.to(a.device)
    iv = _INT_VIEW.get(a.dtype)
    if iv is not None:
        return bool(torch.equal(a.view(iv), b.view(iv)))
    return bool(torch.equal(a, b))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_source(path: str) -> tuple:
    """-> (model state_dict on CPU, record). REFUSES anything but the trainer's model-only EMA file
    (``kind == "ema_model_only"``): a raw / full checkpoint's weights are NOT the shipped (EMA) planner."""
    p = Path(str(path))
    if not p.is_file():
        raise MapOnlyError(f"[map-only] ⛔ --init-from {path}: no such file")
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    kind = state.get("kind") if isinstance(state, dict) else None
    if kind != INIT_KIND or not isinstance(state.get("model"), dict):
        raise MapOnlyError(f"[map-only] ⛔ --init-from {path}: kind {kind!r}, the flag needs the SHIPPED EMA model "
                           f"(kind {INIT_KIND!r}, e.g. ckpt_ema_48852.pt): a raw or full checkpoint's planner is not "
                           f"the shipped planner")
    rec = {"path": str(p), "md5": _md5(str(p)), "kind": kind, "step": state.get("step"),
           "ema_decay": state.get("ema_decay"), "num_updates": state.get("num_updates"),
           "n_keys": len(state["model"])}
    return state["model"], rec


def verify_against_source(model, src: dict, *, fresh: bool) -> dict:
    """Every key of the model's state_dict against the source file, bit for bit: the NON-map keys always (raw == the
    shipped EMA model), the map-only keys too on a FRESH start (a resume has trained them). REFUSES a key set that
    differs (the build is not the source's architecture) and any differing tensor."""
    sd = model.state_dict()
    if set(sd) != set(src):
        raise MapOnlyError(f"[map-only] ⛔ the model and --init-from have different keys (model-only "
                           f"{sorted(set(sd) - set(src))[:3]}, file-only {sorted(set(src) - set(sd))[:3]}): the "
                           f"build is not the shipped model's architecture")
    diff = [k for k, v in sd.items() if (fresh or not is_map_only(k)) and not bit_equal(v, src[k])]
    if diff:
        raise MapOnlyError(f"[map-only] ⛔ {len(diff)} tensor(s) differ from --init-from "
                           f"({'fresh start' if fresh else 'resume: frozen tensors'}), e.g. {diff[:4]} -- the frozen "
                           f"planner is not the shipped planner")
    n_map = sum(1 for k in sd if is_map_only(k))
    return {"checked_keys": (len(sd) if fresh else len(sd) - n_map), "n_keys": len(sd), "n_map_only_keys": n_map,
            "fresh": bool(fresh), "result": "bit-identical"}


class FrozenStateGuard:
    """The frozen tensors' REFERENCE = the init file's non-map tensors (CPU). :meth:`changed` re-reads every frozen
    parameter AND buffer of the live model against it; :meth:`assert_ok` REFUSES on the first drifted byte."""

    def __init__(self, src: dict):
        self.ref = {k: v.detach().to("cpu") for k, v in src.items() if not is_map_only(k)}
        self.n_checks = 0

    def changed(self, model) -> list:
        sd = model.state_dict()
        out = [k for k in self.ref if k not in sd]
        out += [k for k, v in self.ref.items() if k in sd and not bit_equal(sd[k].detach().cpu(), v)]
        self.n_checks += 1
        return out

    def assert_ok(self, model, where: str) -> None:
        bad = self.changed(model)
        if bad:
            raise MapOnlyError(f"[map-only] ⛔ {where}: {len(bad)} FROZEN tensor(s) changed, e.g. {bad[:4]} -- the "
                               f"planner is no longer the shipped planner; refusing to continue or save")


def ema_problems(ema, model, src: dict | None = None, *, fresh: bool = False) -> list:
    """The EMA under the flag ([] = sound): it shadows EXACTLY the map-only tensors; on a fresh start every shadow
    equals the live tensor AND the source (raw == EMA == the shipped model at step 0)."""
    bad = []
    keys = set(getattr(ema, "shadow", {}) or {})
    outside = sorted(k for k in keys if not is_map_only(k))
    if outside:
        bad.append(f"the EMA shadows {len(outside)} tensor(s) outside the map-only modules, e.g. {outside[:3]}")
    want = {n for n, p in model.named_parameters() if is_map_only(n) and p.requires_grad}
    if not want <= keys:
        bad.append(f"the EMA does not shadow {sorted(want - keys)[:3]}")
    if fresh:
        sd = model.state_dict()
        for k in sorted(keys):
            s = ema.shadow[k]
            live = sd.get(k)
            if live is None or not bit_equal(s, live.detach().to(dtype=s.dtype)):
                bad.append(f"EMA[{k}] != live at the fresh start")
            elif src is not None and (k not in src or not bit_equal(s, src[k].to(dtype=s.dtype))):
                bad.append(f"EMA[{k}] != --init-from at the fresh start")
    return bad


# ================================================================================================================= #
# the dependency probe                                                                                              #
# ================================================================================================================= #
def grad_dependent_tensors(obj: Any, prefix: str = "out") -> list:
    """Names of every tensor in a (nested) output that REQUIRES GRAD. Under the freeze only the map-only modules are
    trainable, so a tensor that requires grad depends on them: every planner output must be absent from this list."""
    out = []
    if torch.is_tensor(obj):
        if obj.requires_grad:
            out.append(prefix)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += grad_dependent_tensors(v, f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += grad_dependent_tensors(v, f"{prefix}[{i}]")
    return out


# ================================================================================================================= #
# G-DVB                                                                                                             #
# ================================================================================================================= #
def dvb_check(model, args) -> list:
    """G-DVB ``built`` lever: declared ``--r8-train-only-map`` <=> the BUILT model's trainable set is EXACTLY the
    parameters under the three literal module paths (all three built, each parameter trainable), and the freeze
    record the model carries counts what is really trainable."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = is_on(args)
    rec = getattr(model, MARKER_ATTR, None)
    if not callable(getattr(model, "named_parameters", None)):
        return []
    if not declared:
        if rec is not None:
            return [Mismatch(FLAG, False, "a map-only freeze", f"model.{MARKER_ATTR}",
                             "the build froze everything but the map head and argv does not declare it")]
        return []
    out = []
    if rec is None:
        return [Mismatch(FLAG, True, "no map-only freeze", f"model.{MARKER_ATTR}",
                         "argv declares the map-only fine-tune but the built model was not frozen")]
    missing = [p for p in MAP_ONLY_MODULES if _resolve(model, p) is None]
    if missing:
        out.append(Mismatch(FLAG, list(MAP_ONLY_MODULES), f"missing {missing}", "model", "a map-only module is not built"))
    tr = [(n, int(p.numel())) for n, p in model.named_parameters() if p.requires_grad]
    outside = [n for n, _ in tr if not is_map_only(n)]
    if outside:
        out.append(Mismatch(FLAG, "trainable = the map-only modules only", f"{len(outside)} trainable outside",
                            "model.named_parameters()", f"e.g. {outside[:3]}"))
    frozen_map = [n for n, p in model.named_parameters() if is_map_only(n) and not p.requires_grad]
    if frozen_map:
        out.append(Mismatch(FLAG, "every map-only parameter trainable", f"{len(frozen_map)} frozen",
                            "model.named_parameters()", f"e.g. {frozen_map[:3]}"))
    numel = sum(k for _, k in tr)
    if int(rec.get("n_trainable_params", -1)) != numel:
        out.append(Mismatch(FLAG, f"record n_trainable_params {rec.get('n_trainable_params')}", numel,
                            f"model.{MARKER_ATTR}", "the freeze record does not count what is trainable"))
    return out


def register_dvb_levers(register: Callable) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
