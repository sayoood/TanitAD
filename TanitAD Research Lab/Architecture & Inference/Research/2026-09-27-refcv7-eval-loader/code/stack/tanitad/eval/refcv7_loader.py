"""refcv7 EVAL LOADER -- the model and the held-out eval dataset, built EXACTLY as refcv7's
`refc_v3_train.train()` builds them.

Architecture & Inference, 2026-09-27 (the refcv7 launch gate's G-EVAL, SPEC_REFCV7 sec. 2).

WHY THIS FILE EXISTS (and why `refcv6_loader.py` beside it is not enough)
------------------------------------------------------------------------
`stack/tanitad/eval/refcv6_loader.py` is the box-head audit's loader, VENDORED byte for byte (its
provenance is pinned by `tests/test_g_box_overfit.py`); it replays refcv6's `train()` as it stood at
287d72e. MEASURED 2026-09-27 by the Master Mind's G-EVAL dry run on the intended refcv7 argv, its
STRICT load fails on `Unexpected key(s) ... "_map_hires.lift.unobserved", ...`. Diffing the code
lines of train()'s build region at 287d72e against the tip (37086c3) names EVERY block it lacks --
the list is the diff, not a memory of the SPEC:

  (1) train():7932-7934  the banked nav-compliance tau file, verified against the float, its
                         sha256 stamped onto the CONFIG (`nav_compliance_tau_sha256`);
  (2) train():7998       `model._vis1` (refcv7 A9 R3 -- VIS-1 travels on the model);
  (3) train():8005-8069  NEW-2 + A6/A7: the 10 cm map branch -- the trunk's stride-8 tap, the class
                         weights at the DECLARED extent, `MapHiresConfig` (extent, grad ckpt, NEAR
                         LIFT, NEAR REFINE BLOCKS, weights sha256, decision rule), the branch, the
                         0.25 m `HiresLiftGeometryBank`, `_w_map_hires` / `_map_hires_class_weight`
                         (+ stamp);
  (4) train():8070-8130  the perception branch as refcv7 builds it: ALSO built under
                         `--bev-source map_hires_pool`, with `bev_source`, `planner_crop_m`, the four
                         A9 slot-refinement fields (`_slot_refine_kwargs`) and the A14 `query_select`;
  (5) train():8740-8748  the eval split's G3 call with train()'s exact arguments (`synthetic`,
                         `max_unverified_frac`);
  (6) train():8871-8879  the eval split's 10 cm FINE map store (`enable_map_hires`) instead of the
                         0.5 m `MapGTStore` under `--map-hires on`;
  (7) train():8894-8895  the VIS-1 sidecar on the eval split.

Everything else refcv7 adds is pinned onto the CONFIG inside `_pin_trainer_cfg` and therefore
rebuilt by calling the trainer's OWN pin, as the refcv6 loader already did: the residual prior
(`--residual-prior`, :500), `--graft-tac8-prior` / `--graft-nav-compliance` /
`--speed-ceiling-filter` (FIX-4, `_pin_refcv6_tactical` :1006 / :1036 / :1044), and the AGENT
head's A9 fields and 300 queries (`_pin_refcv5_seams` :1234-1245). ⚠️ The BOX head's A9 fields and
its `query_select` are NOT pinned -- train() sets them at 8088-8089, block (4) here.

⛔ INDEPENDENCE. This file REPLAYS train()'s lines; it never calls `train()`. The launch gate's
G-EVAL builds the reference by running the REAL `train()` (captured at its first data-source call)
and compares the two bit for bit (state_dict, `_w_*` attributes, param_breakdown, the forward on a
fixed real eval batch). A loader that called `train()` itself could never disagree with it.

DOCUMENTED DEPARTURES (each is also appended to `record["departures"]`)
  * `--trunk-compile` is REMOVED from argv (no Triton on Windows); it wraps the backbone CALL in
    `torch.compile` and does not touch the module tree or the state_dict.
  * Thor data paths are REMAPPED to the eval kit, per flag, recorded -- and the argv's basename
    is CHECKED against the kit file it is remapped to (refused when they differ, unless the
    operator's RECORDED override names the flag).
  * The TRAIN split is never built. The train-split constants the eval reads come from the run's
    own `config.json`, never re-derived: `agent_pad`, the tactical-goal `pos_weight` / class mask,
    the withheld-bank mode.
  * The TRAIN-split sampler, the P0 calibration loader, the optimiser, the conflict detector and
    the BN recalibration are training instruments built after the capture point; none is built.
  * The checkpoint is loaded STRICTLY; the kit's anchor file must equal the checkpoint's own anchor
    buffers (recorded), and every refcv7 block's config.json stamp is held against the rebuilt
    branch (refused on contradiction).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

LOADER = "refcv7_loader"


# ---------------------------------------------------------------------------------------- #
# environment -- the gate's contract is REFCV6_*; REFCV7_* is accepted, and a CONFLICT refuses #
# ---------------------------------------------------------------------------------------- #
def _env(suffix: str, default=None):
    """``REFCV7_<suffix>`` or ``REFCV6_<suffix>`` (the launch gate sets the latter). ⛔ Both set to
    DIFFERENT values refuses: a stale REFCV7_* in the shell must never silently override the
    gate's choice (or the reverse)."""
    a, b = os.environ.get(f"REFCV7_{suffix}"), os.environ.get(f"REFCV6_{suffix}")
    if a and b and os.path.normcase(os.path.abspath(a)) != os.path.normcase(os.path.abspath(b)):
        raise SystemExit(f"[{LOADER}] REFCV7_{suffix}={a!r} and REFCV6_{suffix}={b!r} disagree; "
                         f"unset one (the launch gate sets REFCV6_{suffix})")
    return a or b or default


#: the tree whose trainer is replayed; default = the tree this file lives in (stack/tanitad/eval/)
REPO = Path(_env("REPO", str(Path(__file__).resolve().parents[3])))
STACK = REPO / "stack"
SCRIPTS = STACK / "scripts"
TANITEVAL = REPO / "taniteval"
#: the eval kit; its data/ mirrors Thor's /home/nvidia/data/ layout
KIT = Path(_env("KIT", "D:/refcv6_eval_kit"))


def _map_gt_root_local() -> str:
    """The 10 cm SAM3 GT root in the kit. The dev-box kit carries the EVAL subset of Thor's
    ``sam3_gt_v3`` as ``sam3_gt_v3_eval`` (pathmap_refcv7.txt); a Thor 'kit' (KIT=/home/nvidia)
    carries ``sam3_gt_v3`` itself. The choice is recorded in the remap record."""
    sub = KIT / "data" / "sam3_gt_v3_eval"
    full = KIT / "data" / "sam3_gt_v3"
    return str(full if (not sub.exists() and full.exists()) else sub)


#: flag -> (the basename the run's argv must name, the kit path it is remapped to)
PATH_REMAP_SPEC: dict[str, tuple[str, str]] = {
    "--anchors": ("refc_anchors_6s_v0cond_alat_117.pt",
                  str(KIT / "data/anchors/refc_anchors_6s_v0cond_alat_117.pt")),
    "--eval-cache": ("refcv6-b1-416x1024-eval139", str(KIT / "data/refcv6-b1-416x1024-eval139")),
    "--eval-labels": ("s2_labels_v8_eval.jsonl.gz",
                      str(KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")),
    "--speed-max-sidecar-v6-eval": ("refcv6_speed_max_v8_eval.jsonl",
                                    str(KIT / "data/refcv6_speed_max_v8_eval.jsonl")),
    "--agent-join": ("b1_train_plus_eval_agents.jsonl.xz",
                     str(KIT / "data/joins/b1_train_plus_eval_agents.jsonl.xz")),
    "--agent-rig-extrinsics": ("refcv6_train_eval139_extrinsics.json",
                               str(KIT / "data/refcv6_train_eval139_extrinsics.json")),
    # refcv7 NEW-2: the /3 fine codes at the declared extent (NOT refcv6's sam3_gt_eval_thor137 /2 maps)
    "--map-gt-root": ("sam3_gt_v3", _map_gt_root_local()),
    "--join3d": ("b1_train_plus_eval_agents_3d.jsonl.xz",
                 str(KIT / "data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz")),
    "--clip-clock-sidecar": ("refcv6_clip_clock_sidecar.jsonl",
                             str(KIT / "data/refcv6_clip_clock_sidecar.jsonl")),
    # refcv7 (sha256-verified against Thor by the Master Mind, 2026-09-27)
    "--vis1-sidecar": ("vis1_sidecar_refcv6b1_train4369_eval139.npz",
                       str(KIT / "data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz")),
    "--nav-compliance-tau-file": ("nav_compliance_tau_train.json",
                                  str(KIT / "data/refcv7/nav_compliance_tau_train.json")),
    "--map-hires-class-weights": ("map_hires_class_weights_train_100x30.json",
                                  str(KIT / "data/refcv7/map_hires_class_weights_train_100x30.json")),
}
#: Thor path -> kit path, per flag (the refcv6 loader's name, kept for callers)
PATH_REMAP: dict[str, str] = {f: p for f, (_, p) in PATH_REMAP_SPEC.items()}
#: a RECORDED override (JSON {flag: local path}) -- e.g. a train-episode roll. Unset changes nothing.
REMAP_OVERRIDES_FILE = _env("REMAP_OVERRIDES")
OVERRIDDEN: dict[str, str] = {}
if REMAP_OVERRIDES_FILE:
    with open(REMAP_OVERRIDES_FILE, encoding="utf-8") as _fh:
        OVERRIDDEN = {str(k): str(v) for k, v in json.load(_fh).items()}
    PATH_REMAP.update(OVERRIDDEN)
#: flags whose TRAIN-side file is not in the kit and is not read by anything this module builds
TRAIN_ONLY_PATHS = ("--v2-cache", "--v7-labels", "--speed-max-sidecar-v6", "--out",
                    "--eval-window-dump")
#: removed from argv, with the reason
DROP_FLAGS = {"--trunk-compile": "no Triton on Windows; torch.compile wraps the backbone CALL "
                                 "only -- module tree and state_dict identical "
                                 "(LAUNCH_READINESS_FIXES.md sec. 10b)"}


def bootstrap() -> None:
    """sys.path exactly as the harnesses need it; assert the stack resolves to THIS repo."""
    for p in (str(STACK), str(TANITEVAL), str(SCRIPTS), str(TANITEVAL / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never reach the internet
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import tanitad                                           # noqa: F401
    here = os.path.normcase(os.path.abspath(tanitad.__file__))
    want = os.path.normcase(os.path.abspath(str(STACK)))
    if not here.startswith(want):
        raise SystemExit(f"[{LOADER}] tanitad imported from {here}, not from {want} -- an "
                         f"editable install elsewhere is shadowing the tree under test")


_TRAINER = None


def trainer():
    """`refc_v3_train.py` by path (it is a script) -- the SAME object refcv3_arm uses if loaded."""
    global _TRAINER
    if _TRAINER is None:
        bootstrap()
        name = "refc_v3_train_for_arm"          # refcv3_arm's name: ONE module object, not two
        if name in sys.modules:
            _TRAINER = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / "refc_v3_train.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            _TRAINER = mod
    return _TRAINER


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha12(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def _norm(x):
    """JSON round-trip, so a tuple and a list (and 1 and 1.0 inside a dict) compare as the record
    would carry them."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


# ---------------------------------------------------------------------------------------- #
# argv                                                                                      #
# ---------------------------------------------------------------------------------------- #
def _basename(p: str) -> str:
    return os.path.basename(str(p).replace("\\", "/").rstrip("/"))


def remap_argv(argv: list[str], remap: dict | None = None) -> tuple[list[str], dict]:
    """The run's argv with Thor paths -> kit paths and `--trunk-compile` removed.

    ⛔ The remap is BY FLAG, so it is CHECKED BY NAME: when a flag is remapped to its DEFAULT kit
    file, the argv must name that file's basename (`PATH_REMAP_SPEC`). A run whose argv names
    another anchors file / class-weights file / tau file than the kit carries is REFUSED, never
    silently evaluated against the kit's file. An operator override (`REFCV6_REMAP_OVERRIDES`, or
    a caller's `remap=`) is exempt and recorded as such."""
    explicit = remap is not None
    remap = dict(PATH_REMAP if remap is None else remap)
    out = []
    rec = {"remapped": {}, "dropped": {}, "train_only_kept_unread": {}, "kit": str(KIT),
           "overrides_file": REMAP_OVERRIDES_FILE}
    bad = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in DROP_FLAGS:
            rec["dropped"][a] = DROP_FLAGS[a]
            i += 1
            continue
        if a in remap and i + 1 < len(argv):
            thor, local = argv[i + 1], remap[a]
            src = ("caller remap" if explicit else
                   "recorded override" if a in OVERRIDDEN else "kit default")
            ent = {"thor": thor, "local": local, "source": src}
            spec = PATH_REMAP_SPEC.get(a)
            if src == "kit default" and spec is not None:
                ent["basename_expected"] = spec[0]
                if _basename(thor) != spec[0]:
                    bad.append(f"{a}: the argv names {_basename(thor)!r}, the kit's file for this "
                               f"flag is {spec[0]!r}")
            rec["remapped"][a] = ent
            out += [a, local]
            i += 2
            continue
        if a in TRAIN_ONLY_PATHS and i + 1 < len(argv):
            rec["train_only_kept_unread"][a] = argv[i + 1]
        out.append(a)
        i += 1
    if bad:
        raise SystemExit(f"[{LOADER}] the run's argv names files the kit does not carry -- "
                         f"refusing to evaluate the kit's file in their place (pass a RECORDED "
                         f"override, REFCV6_REMAP_OVERRIDES, to map them explicitly):\n  - "
                         + "\n  - ".join(bad))
    if any(e["source"] == "kit default" for e in rec["remapped"].values()) and not KIT.is_dir():
        raise SystemExit(f"[{LOADER}] the eval kit {KIT} does not exist on this host. Pass the "
                         f"launch gate's --eval-kit (REFCV6_KIT): on the dev box D:/refcv6_eval_kit; "
                         f"on Thor /home/nvidia, whose data/ IS the kit layout")
    return out, rec


def parse_args(config: dict, remap: dict | None = None):
    """-> (args, argv_local, remap record). ⭐ Mirrors `main()` (refc_v3_train.py:11357-11367):
    the parser AND the command line ride on the Namespace (`_ew_parser` / `_ew_argv`), so the
    effective-weight audit `check_effective_weights` sees what `train()` sees."""
    tr = trainer()
    argv, rec = remap_argv(list(config["argv"]), remap)
    ap = tr.build_parser()
    args = ap.parse_args(argv)
    args._ew_parser = ap
    args._ew_argv = list(argv)
    return args, argv, rec


# ---------------------------------------------------------------------------------------- #
# the MODEL -- refc_v3_train.train() lines 7893-8290 + 8483-8497 + 8962, replayed            #
# (line refs = the tip 37086c399cf2, stack/scripts/refc_v3_train.py)                         #
# ---------------------------------------------------------------------------------------- #
def trunk_rows_as_trained(tr, cfg, config: dict):
    """-> (cfg, record). ⛔ D-REFCV6-EQUALIZE-DROPPED (SPEC_REFCV7 FIX-3): rebuild the TRUNK as
    trained, never as declared -- `trunk_equalize_rows_as_trained(config)` reads the post-fix stamp
    `seams.trunk_equalize_bottom_rows` (a refcv7 record carries it; for such a record the value is
    the pin's own). A pre-FIX record is handled exactly as `refcv6_loader` does."""
    enc = cfg.core.encoder
    if not (hasattr(tr, "trunk_equalize_rows_as_trained")
            and hasattr(enc, "trunk_equalize_bottom_rows")):
        raise SystemExit(f"[{LOADER}] this tree predates FIX-3 (no declared trunk equalize "
                         f"field) -- it cannot be a refcv7 tree")
    declared = int(enc.trunk_equalize_bottom_rows)
    rows, why = tr.trunk_equalize_rows_as_trained(config)
    enc.trunk_equalize_bottom_rows = int(rows)
    return cfg, {"status": "set as trained", "declared_by_pin": declared,
                 "as_trained": int(rows), "why": why}


def build_model(config: dict, ckpt_path: str, device: str = "cuda", remap: dict | None = None,
                strict: bool = True):
    """-> (model, cfg, args, record). Mirrors train() block by block.

    ⛔ RNG-NEUTRAL: train() calls `torch.manual_seed(args.seed)` before building (replayed, so module
    init is the run's); the build runs inside `torch.random.fork_rng`, so the caller's CPU and CUDA
    generator states are exactly what they were before the call (refcv6_loader's MEASURED
    2026-09-24 incident: a bare seed reset every 'inference seed' to seed 0)."""
    import torch
    devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devs):
        return _build_model(config, ckpt_path, device, remap, strict)


def _build_model(config: dict, ckpt_path: str, device: str = "cuda", remap: dict | None = None,
                 strict: bool = True):
    import torch
    tr = trainer()
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    t0 = time.time()
    args, argv, arec = parse_args(config, remap)
    rec: dict = {"loader": LOADER, "repo": str(REPO), "argv_local": argv, "argv_remap": arec,
                 "departures": []}
    rec["departures"] += [f"argv {k} REMOVED: {v}" for k, v in arec["dropped"].items()]
    rec["departures"].append(
        f"Thor data paths REMAPPED to the eval kit {KIT} for "
        f"{sorted(arec['remapped'])} (per flag, basename-checked; recorded in argv_remap)")
    # ---- train():7896-7902 -- the argument refusals (no data read by any of them) -------- #
    tr._check_nav_from_v7_args(args)
    tr._check_max_speed_args(args)
    tr._check_goal_point_args(args)
    tr.check_effective_weights(args)
    # ---- train():7908 -- the anchor artifact FIRST (units resolved onto args for the pin) - #
    art = tr._read_anchor_artifact(args)
    # (train():7917-7921 -- DataLoader tensor sharing: no model effect; train():7922 device)
    # ---- train():7924-7927 -- the seed (only model INIT consumes it; every parameter is then
    # loaded strictly), and the cuDNN setting ------------------------------------------------ #
    torch.manual_seed(args.seed)
    if bool(getattr(args, "cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
    # ---- train():7928-7930 -- the config through the trainer's OWN pin. refcv7's residual
    # prior, --graft-tac8-prior, --graft-nav-compliance, --speed-ceiling-filter, the agent seam's
    # A9 fields and query count are all pinned HERE (`_pin_refcv7`, `_pin_refcv6_tactical`). -- #
    # ⭐ loader-side, BEFORE the pin: the agent head's query count AS TRAINED (refcv7 A9 R4
    # moved the parser default 100 -> 300; `agent_queries_as_trained` is the trainer's helper for
    # exactly this re-parse). For a refcv7 record it is the parser's own value -- recorded.
    _aq = getattr(tr, "agent_queries_as_trained", None)
    if _aq is not None and str(getattr(args, "agents", "off")) != "off":
        _n_q, _why_q = _aq(config, args)
        rec["agent_queries"] = {"parser": int(getattr(args, "agent_queries")),
                                "as_trained": None if _n_q is None else int(_n_q), "why": _why_q}
        if _n_q is not None and int(_n_q) != int(args.agent_queries):
            args.agent_queries = int(_n_q)
            rec["departures"].append(
                f"--agent-queries: parser default {rec['agent_queries']['parser']} -> the "
                f"record's {int(_n_q)} ({_why_q})")
    cfg = tr._pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    # ⛔ FIX-3: the TRUNK as trained (loader-side; see trunk_rows_as_trained)
    cfg, rec["trunk_equalize_as_trained"] = trunk_rows_as_trained(tr, cfg, config)
    # ---- train():7932-7934 -- SPEC_REFCV7 sec. 7 (A2): the banked tau file, held against the
    # float BEFORE anything is built; its sha256 is a DECLARED config field --------------- #
    _navc = tr._verify_navc_tau_file(args)
    if _navc is not None:
        cfg.nav_compliance_tau_sha256 = _navc["sha256"]
    rec["nav_compliance_tau_file"] = _navc
    # ---- train():7939-7945 -- the registered hier-vs-flat delta (a config-pair gate; no weight
    # is read). Replayed so a refusal the trainer makes, the loader makes too ------------- #
    delta = v3.config_delta(
        tr._pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=True), args),
        tr._pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=False), args))
    if set(delta) != tr.REGISTERED_DELTA_KEYS:
        raise SystemExit(f"[{LOADER}] config delta {sorted(delta)} != registered "
                         f"{sorted(tr.REGISTERED_DELTA_KEYS)} (train() refuses this too)")
    # ---- train():7946-7948 --------------------------------------------------------------- #
    tr._check_anchor_artifact_against_cfg(art, cfg, args)
    if args.graft_lan or args.goal_str:
        cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
    # ---- train():7950 -- the model (ImageNet init is overwritten by the strict load below;
    # HF_HUB_OFFLINE keeps it local) -------------------------------------------------------- #
    model = v3.RefCV3Model(cfg).to(device)
    # ---- train():7955-7957 --------------------------------------------------------------- #
    model._w_agent = float(getattr(args, "w_agent", tr.AGENT_WEIGHT_DEFAULT))
    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
    # ---- train():7971-7986 -- H-BOXCLS-1: the class weight, from its banked artifact ------ #
    model._cls_class_weight, model._cls_class_weight_stamp = None, None
    _cwmode = str(getattr(args, "agent_cls_weight", "off"))
    if _cwmode != "off":
        _cwname, _cwline = tr.CLS_WEIGHT_CHOICES[_cwmode]
        _cwline, _cwpop, _cwsrc = tr._cls_weight_expectation(args, _cwline)
        _cw, _cws = tr._agent_slots.load_cls_class_weight(
            _cwname, expect_corpus_line=_cwline, target_population=_cwpop)
        model._cls_class_weight = _cw.to(device)
        model._cls_class_weight_stamp = _cws
        rec["cls_weight"] = {"mode": _cwmode, "digest": _cws.get("digest"),
                             "corpus_line": _cws.get("corpus_line")}
    # ---- train():7987-7995 --------------------------------------------------------------- #
    model._w_map = float(getattr(args, "w_map", 0.0) or 0.0)
    model._w_box3d = float(getattr(args, "w_box3d", 0.0) or 0.0)
    model._map_lift_valid_mask = bool(getattr(args, "map_lift_valid_mask", True))
    model._box3d_visible_filter = bool(getattr(args, "box3d_visible_filter", True))
    # ---- train():7998 -- refcv7 A9 R3: VIS-1 on the model (refcv6 loader: ABSENT) -------- #
    model._vis1 = bool(getattr(args, "slot_vis1", False))
    # ---- train():8005-8069 -- refcv7 NEW-2 + A6/A7: THE MAP AT 10 cm (refcv6 loader: ABSENT).
    # BUILT FIRST -- before the perception branch, whose A6 planner pool reads this branch's
    # 0.25 m encoder (its width and grid are read off the BUILT branch). Off => nothing. ----- #
    rec["map_hires"] = build_map_hires_block(tr, model, args, device)
    # ---- train():8070-8130 -- the perception branch AS refcv7 BUILDS IT (refcv6 loader: built
    # only behind w_map/w_box3d, with neither bev_source, planner_crop_m, the A9 fields nor
    # query_select) ------------------------------------------------------------------------- #
    rec["perception"] = build_perception_block(tr, model, args, device)
    # ---- train():8140-8149 -- refcv6 sec. 6: a conflict detector that can never read anything
    # is refused (a refusal only; the detector itself is a training instrument) ------------ #
    if tr._conflict_override(args) is True and not tr._conflict_aux_weights(model):
        raise SystemExit(f"[{LOADER}] --conflict-detector on with NO live perception weight "
                         f"(train() refuses this too)")
    # ---- train():8156-8162 -- tactical-goal carriers (pos_weight/mask filled from the STAMP
    # below; train() fills them from the loaded TRAIN split at 8483-8497) ------------------ #
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model._w_tac_v6 = float(getattr(args, "w_tac_v6", 0.0) or 0.0)
    # ---- train():8165-8195 -- refcv7 (DrivoR-T) weights + the two built-vs-weight refusals - #
    model._w_r7_wta = float(getattr(args, "w_r7_wta", 0.0) or 0.0)
    model._w_r7_scorer = float(getattr(args, "w_r7_scorer", 0.0) or 0.0)
    model._r7_n_perturb = int(getattr(args, "r7_n_perturb", 0) or 0)
    model._r7_nav_tau_rad = float(getattr(args, "r7_nav_tau_rad", 0.0) or 0.0)
    model._r7_calls = 0
    if (getattr(model, "refcv7_wta", None) is not None) != (model._w_r7_wta > 0.0):
        raise SystemExit(f"[{LOADER}] refcv7 heads built/weight disagree")
    if (getattr(model, "tac_decoder_v6", None) is not None) != (model._w_tac_v6 > 0.0):
        raise SystemExit(f"[{LOADER}] tac_decoder_v6 built/weight disagree")
    # ---- train():8196-8208 --------------------------------------------------------------- #
    model._w_u0 = float(getattr(args, "w_u0", tr.U0_WEIGHT_DEFAULT))
    model._w_goal_point = float(getattr(args, "goal_point_w", tr.GOAL_POINT_WEIGHT_DEFAULT))
    model._rig_camera, _cam_stamp = tr._build_rig_camera(cfg, args)
    rec["rig_camera"] = _cam_stamp
    # ---- train():8211 -- the ground-prior probe, REPLAYED (refcv6 loader: a departure). It
    # returns at once unless --agent-w-ground > 0, draws from its OWN generator and reads the
    # model without writing it; replaying it means a refusal train() makes, the loader makes. #
    rec["ground_prior_probe"] = tr.assert_ground_prior_is_supervised(model, args)
    # ---- train():8221-8290 -- the anchor vocabulary, with train()'s two refusals ---------- #
    if args.anchors:
        anc = art.anchors.to(device)
        _ctrl = None if art.controls is None else art.controls.to(device)
        _want = bool(getattr(args, "anchor_v0_conditioned", False))
        if _want and _ctrl is None:
            raise SystemExit(f"[{LOADER}] v0-conditioned build but the anchor file has no controls")
        if _ctrl is not None and not _want:
            raise SystemExit(f"[{LOADER}] the anchor file carries controls but "
                             f"--anchor-v0-conditioned was not given (train() refuses this too)")
        model.core.decoder.load_anchors(anc, _ctrl)
        if _ctrl is not None and not bool(((_ctrl[:, 0] == 0) & (_ctrl[:, 1] == 0)).any()):
            raise SystemExit(f"[{LOADER}] the control grid has no {{a=0, kappa=0}} node "
                             f"(train() refuses this too)")
        rec["anchors"] = {"path": args.anchors, "shape": list(anc.shape),
                          "controls_shape": None if _ctrl is None else list(_ctrl.shape),
                          "control_units": art.control_units,
                          "sha256_16": hashlib.sha256(anc.detach().float().cpu().numpy()
                                                      .tobytes()).hexdigest()[:16]}
    else:
        rec["departures"].append("no --anchors: the SYNTHETIC default vocabulary, as train() "
                                 "builds it (train() warns loudly)")
    # ---- (train():8342 -- refuse_eval_clips_in_train: the FIRST data read; the launch gate's
    # G-EVAL captures the trainer's reference model HERE) ---------------------------------- #
    # ---- train():8483-8497 -- the TRAIN-split tactical-goal constants, from the run's STAMP -- #
    tgs = config.get("tac_goal_stats") or {}
    if model._w_tac_goal > 0.0 or model._w_tac_v6 > 0.0:
        pw = tgs.get("pos_weight")
        census = tgs.get("census")
        if not pw or census is None:
            raise SystemExit(f"[{LOADER}] config.json carries no tac_goal_stats.pos_weight/census "
                             f"-- the eval loss's goal term cannot be reproduced")
        mask_rep = tr._tac_goal_head.mask_report(census)
        want_trainable = list(tgs.get("trainable") or [])
        if list(mask_rep["trainable"]) != want_trainable:
            raise SystemExit(f"[{LOADER}] mask_report over the STAMPED census gives trainable "
                             f"{mask_rep['trainable']} but the stamp says {want_trainable}")
        model._tac_goal_pos_weight = torch.tensor(pw, dtype=torch.float32)
        model._tac_goal_class_mask = torch.tensor(mask_rep["mask"], dtype=torch.float32)
        rec["tac_goal_from_stamp"] = {"pos_weight": pw, "mask": list(mask_rep["mask"]),
                                      "n_trainable": int(mask_rep["n_trainable"]),
                                      "source": "config.json tac_goal_stats (train split)"}
        rec["departures"].append(
            "tactical-goal pos_weight / class mask from config.json tac_goal_stats (the TRAIN "
            "split's own constants; train():8483-8497 derives them from the loaded split)")
    # ---- train():8962 -- the withheld bank (mode `fixed` reads no episodes) --------------- #
    wb = config.get("withheld_bank") or {}
    if str(getattr(args, "withheld_bank", "fixed")) != "fixed":
        raise SystemExit(f"[{LOADER}] withheld bank mode != fixed needs the train episodes")
    rec["withheld_bank"] = tr._apply_withheld_bank(model, args, [], device)
    if wb and wb.get("mode") != rec["withheld_bank"]["mode"]:
        raise SystemExit(f"[{LOADER}] withheld bank {rec['withheld_bank']} != stamp {wb}")
    # ---- the STRICT load (train():8979-8982 loads a checkpoint the same way -- strict, and
    # BEFORE its G-DVB check at 8995) -------------------------------------------------------- #
    anc_before = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k.startswith("core.decoder.anchor")}
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        raise SystemExit(f"[{LOADER}] {ckpt_path} has no 'model' key")
    res = model.load_state_dict(ck["model"], strict=strict)
    rec["state_dict"] = {"strict": strict, "missing": list(getattr(res, "missing_keys", [])),
                         "unexpected": list(getattr(res, "unexpected_keys", [])),
                         "n_keys": len(ck["model"]),
                         "n_map_hires_keys": sum(1 for k in ck["model"] if k.startswith("_map_hires.")),
                         "ckpt_keys": sorted(k for k in ck if k != "model"),
                         "step": ck.get("step")}
    # ⭐ CONTROL: the kit's anchor file must equal the checkpoint's own anchor buffers
    anc_ctl = {}
    for k, v in anc_before.items():
        w = model.state_dict()[k]
        anc_ctl[k] = {"shape": list(v.shape),
                      "max_abs_diff": float((v.float() - w.float().to(v.device)).abs().max())
                      if v.numel() else 0.0}
    rec["anchor_file_vs_ckpt_buffers"] = anc_ctl
    del ck
    # ---- train():8995 -- G-DVB: ARGV against the BUILT model (the loader's OWN build) ----- #
    _dvb = tr._dvb
    _bad = list(_dvb.check(model, args, tr.build_parser()))
    rec["declared_vs_built"] = {"mismatches": [str(b) for b in _bad],
                                "registry_entries": len(_dvb.REGISTRY)}
    if _bad:
        raise SystemExit(f"[{LOADER}] G-DVB: the REPLAYED build is not what argv declares "
                         f"({len(_bad)} mismatch(es)):\n  - " + "\n  - ".join(str(b) for b in _bad))
    # ---- loader-side controls: every refcv7 block's config.json STAMP against the rebuild -- #
    rec["stamp_checks"] = stamp_checks(model, config, rec, tr)
    # ---- eval-time settings -------------------------------------------------------------- #
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bd = v3.param_breakdown_v3(model)
    stamped = config.get("param_breakdown") or {}
    rec["param_breakdown"] = {"rebuilt": {k: int(v) for k, v in bd.items()},
                              "config_json": stamped,
                              "equal": ({k: int(v) for k, v in bd.items()}
                                        == {k: int(v) for k, v in stamped.items()})}
    enc = model.core.encoder
    rec["trunk_memory_levers_built"] = dict(getattr(getattr(enc, "trunk", enc), "memory_levers", {})
                                            or getattr(enc, "memory_levers", {}) or {})
    rec["trunk_memory_levers_run"] = config.get("trunk_memory_levers")
    rec["mode"] = getattr(args, "mode", "diffusion")
    rec["decoder_steps"] = int(cfg.core.decoder.diffusion_steps) if rec["mode"] == "diffusion" else 0
    rec["sampler"] = str(getattr(cfg.core.decoder, "sampler", "none"))
    rec["build_s"] = round(time.time() - t0, 1)
    return model, cfg, args, rec


def map_hires_config(tr, args, class_weights_sha256: str):
    """train():8019-8027 -- `MapHiresConfig` exactly as train() declares it: the DECLARED extent,
    grad checkpointing, the NEW-2 R2 near lift (`--map-hires-near-lift-m`), the R3 near refine
    blocks (`--map-hires-near-refine-blocks`), the class weights' sha256 and the decision rule."""
    _mhr = tr._mhr
    _hext = _mhr.declared_extent(args)
    return _mhr.MapHiresConfig(
        w_map_hires=float(args.w_map_hires),
        x_max_m=float(_hext.x_max_m), y_half_m=float(_hext.y_half_m),
        grad_ckpt=bool(_mhr.declared_grad_ckpt(args)),
        near_lift_x_m=float(_mhr.declared_near_lift_m(args)),
        near_refine_blocks=int(_mhr.declared_near_refine_blocks(args)),
        class_weights_sha256=str(class_weights_sha256),
        decision_rule=str(getattr(args, "map_hires_decision_rule", _mhr.DECISION_RULES[0])))


def build_map_hires_block(tr, model, args, device) -> dict | None:
    """train():8005-8069 -- refcv7 NEW-2 + A6/A7, the 10 cm map branch, on ``model``; -> the record
    (``None`` with ``--map-hires off``, when NOTHING is built: no tap, no module, no key)."""
    _mhr, _perc = tr._mhr, tr._perc
    # train():8005-8010 -- the carriers, always
    model._w_map_hires = 0.0
    model._map_hires = None
    model._lift_bank_hires = None
    model._map_hires_class_weight = None
    model._map_hires_class_weight_stamp = None
    if not tr._map_hires_on(args):                                      # train():8011
        return None
    _hext = _mhr.declared_extent(args)                                  # train():8012
    _s8 = model.core.encoder.enable_s8_tap()                            # train():8013
    # train():8017-8018 -- the weights FIRST: their sha256 is a DECLARED field of the branch
    # config; counted at THIS extent (another extent is refused by the reader)
    _hcw, _hcws = _mhr.load_class_weights(args.map_hires_class_weights, extent=_hext)
    _hcfg = map_hires_config(tr, args, _hcws["sha256"])                 # train():8019-8027
    model._map_hires = _mhr.build_map_hires_branch(model, _hcfg).to(device)   # train():8028
    model._w_map_hires = float(args.w_map_hires)                        # train():8029
    model._map_hires_class_weight = _hcw.to(device)                     # train():8030
    model._map_hires_class_weight_stamp = _hcws                         # train():8031
    # train():8032-8042 -- the PER-CLIP 0.25 m lift bank, with the C26 rows
    _hpe, _htable = tr._read_rig_extrinsics(str(getattr(args, "agent_rig_extrinsics", "")))
    if _htable is None:
        raise SystemExit(f"[{LOADER}] --map-hires on needs a PER-CLIP extrinsics table "
                         f"(train() refuses this too)")
    model._lift_bank_hires = _mhr.HiresLiftGeometryBank(
        _htable, frame=_perc.frame_for_model(model), cfg=_hcfg,
        equalize_bottom_rows=int(getattr(args, "equalize_bottom_rows", 0) or 0))
    # train():8043-8069 -- the stamp (its model-defining fields; paths omitted)
    return {**_hcfg.as_dict(), "extent": _hext.as_dict(), "trunk_tap": _s8,
            "bev_source": str(getattr(args, "bev_source", _perc.BEV_SOURCES[0])),
            "planner_crop_m": [float(v) for v in (getattr(args, "bev_planner_crop_m", None)
                                                  or _mhr.PLANNER_CROP_DEFAULT)],
            "branch_params": model._map_hires.param_breakdown(),
            "class_weights_sha256": _hcws["sha256"],
            "lift_valid_mask": bool(model._map_lift_valid_mask),
            "lift_bank_n_clips": len(model._lift_bank_hires),
            "equalize_bottom_rows": int(getattr(args, "equalize_bottom_rows", 0) or 0)}


def build_perception_block(tr, model, args, device) -> dict | None:
    """train():8070-8130 -- the perception branch as refcv7 builds it, on ``model``; -> the record.
    ⭐ refcv7 A6: under ``--bev-source map_hires_pool`` it is the CONSUMER side (the planner pool, the
    BEV tokens, the box head) and is built even at ``w_map == w_box3d == 0``."""
    _mhr, _perc = tr._mhr, tr._perc
    model._perception = None                                            # train():8072-8074
    model._lift_bank = None
    _bev_src = str(getattr(args, "bev_source", _perc.BEV_SOURCES[0]) or _perc.BEV_SOURCES[0])
    if not (model._w_map > 0.0 or model._w_box3d > 0.0 or _bev_src == "map_hires_pool"):
        return None                                                     # train():8080
    _pcfg = _perc.PerceptionBranchConfig(                               # train():8081-8085
        w_map=model._w_map, w_box3d=model._w_box3d, bev_source=_bev_src,
        planner_crop_m=tuple(float(v) for v in (getattr(args, "bev_planner_crop_m", None)
                                                or _mhr.PLANNER_CROP_DEFAULT)))
    # train():8088-8089 -- refcv7 A9: the box head's four refinement fields, from the SAME mapping
    # the agent head uses; A14: its query selection -- a declared frozen-dataclass replace
    _pcfg = tr._dc.replace(_pcfg, **tr._slot_refine_kwargs(args),
                           query_select=str(getattr(args, "slot_query_select", "learned")
                                            or "learned"))
    model._perception = _perc.build_perception_branch(model, _pcfg).to(device)   # train():8090
    _pframe = _perc.frame_for_model(model)                              # train():8091
    if model._w_map > 0.0:                                              # train():8092-8104
        _pe, _ptable = tr._read_rig_extrinsics(str(getattr(args, "agent_rig_extrinsics", "")))
        if _ptable is None:
            raise SystemExit(f"[{LOADER}] --w-map > 0 needs a PER-CLIP extrinsics table")
        model._lift_bank = _perc.LiftGeometryBank(
            _ptable, frame=_pframe, stride=int(_pcfg.stride),
            equalize_bottom_rows=int(getattr(args, "equalize_bottom_rows", 0) or 0))
    return {**_pcfg.as_dict(),                                          # train():8105-8127
            "equalize_bottom_rows": int(getattr(args, "equalize_bottom_rows", 0) or 0),
            "branch_params": model._perception.param_breakdown(),
            "fmap_s16_channels": int(model.core.encoder.s16_dim),
            "fmap_s16_hw": list(model.core.encoder.s16_shape),
            "frame": {"height": int(_pframe.height), "width": int(_pframe.width),
                      "projection": str(_pframe.projection), "f_ref": float(_pframe.f_ref)},
            "lift_bank_n_clips": 0 if model._lift_bank is None else len(model._lift_bank)}


def stamp_checks(model, config: dict, rec: dict, tr) -> dict:
    """⭐ A control that must read a KNOWN value: each refcv7 block the run STAMPED into its
    `config.json` is held against the block this loader rebuilt. ABSENT stamps are recorded as
    absent (a record written for another argv, or a unit rig); a PRESENT stamp that CONTRADICTS the
    rebuild REFUSES -- the loader would otherwise evaluate a model the weights never were.
    ⚠️ Only model-defining fields are compared: file PATHS differ by host by design, and the
    coverage censuses are the dataset's, not the model's."""
    out: dict = {}
    bad: list[str] = []

    def cmp(name, want, have):
        w, h = _norm(want), _norm(have)
        out[name] = {"config_json": w, "rebuilt": h, "equal": w == h}
        if w != h:
            bad.append(f"{name}: config.json {str(w)[:200]} vs rebuilt {str(h)[:200]}")

    # NEW-2: the 10 cm branch
    mh, st = rec.get("map_hires"), config.get("map_hires")
    if st is None and "map_hires" in config:
        if mh is not None:
            bad.append("map_hires: config.json says null (--map-hires off) but the argv builds it")
        out["map_hires"] = "stamped null"
    elif isinstance(st, dict):
        if mh is None:
            bad.append("map_hires: config.json stamps a 10 cm branch the argv does not build")
        else:
            keys = [k for k in (tr._mhr.MapHiresConfig(w_map_hires=1.0).as_dict()) if k in st]
            cmp("map_hires.config", {k: st[k] for k in keys}, {k: mh[k] for k in keys})
            for k in ("trunk_tap", "branch_params", "bev_source", "planner_crop_m",
                      "lift_valid_mask"):
                if k in st:
                    cmp(f"map_hires.{k}", st[k], mh[k])
            if isinstance(st.get("class_weights"), dict) and "sha256" in st["class_weights"]:
                cmp("map_hires.class_weights.sha256", st["class_weights"]["sha256"],
                    mh["class_weights_sha256"])
            out["map_hires.lift_bank_n_clips"] = {"config_json": st.get("lift_bank_n_clips"),
                                                  "rebuilt": mh.get("lift_bank_n_clips"),
                                                  "compared": False}
    else:
        out["map_hires"] = "not in config.json"
    # A6/A9/A14: the perception branch
    pr, st = rec.get("perception"), config.get("refcv6_perception")
    if st is None and "refcv6_perception" in config:
        if pr is not None:
            bad.append("refcv6_perception: config.json says null but the argv builds the branch")
        out["refcv6_perception"] = "stamped null"
    elif isinstance(st, dict):
        if pr is None:
            bad.append("refcv6_perception: config.json stamps a branch the argv does not build")
        else:
            for k in sorted(pr):
                if k in st and k != "lift_bank_n_clips":
                    cmp(f"refcv6_perception.{k}", st[k], pr[k])
    else:
        out["refcv6_perception"] = "not in config.json"
    # A2: the banked tau file (config.json seams.nav_compliance_tau_file = {path, sha256, tau})
    nv = rec.get("nav_compliance_tau_file")
    st = ((config.get("seams") or {}).get("nav_compliance_tau_file")
          if isinstance(config.get("seams"), dict) else None)
    if isinstance(st, dict) and nv is not None:
        for k in ("sha256", "tau"):
            if k in st:
                cmp(f"seams.nav_compliance_tau_file.{k}", st[k], nv[k])
    elif isinstance(st, dict) and nv is None:
        bad.append("seams.nav_compliance_tau_file stamped but the argv names no tau file")
    else:
        out["seams.nav_compliance_tau_file"] = "not in config.json"
    # A9 R4: the agent head's query count
    ag = ((config.get("seams") or {}).get("agents") if isinstance(config.get("seams"), dict)
          else None)
    have_q = getattr(getattr(model.cfg.core, "agents", None), "queries", None)
    if isinstance(ag, dict) and "queries" in ag and have_q is not None:
        cmp("seams.agents.queries", ag["queries"], int(have_q))
    # FIX-3: the trunk rows (the stamp was READ to build the trunk; this proves it reached it)
    st_rows = ((config.get("seams") or {}).get("trunk_equalize_bottom_rows")
               if isinstance(config.get("seams"), dict) else None)
    if st_rows is not None:
        cmp("seams.trunk_equalize_bottom_rows", st_rows,
            int(model.cfg.core.encoder.trunk_equalize_bottom_rows))
    out["n_compared"] = sum(1 for v in out.values() if isinstance(v, dict) and "equal" in v)
    if bad:
        raise SystemExit(f"[{LOADER}] config.json CONTRADICTS the rebuilt model -- refusing rather "
                         f"than guessing which one describes the weights:\n  - "
                         + "\n  - ".join(bad))
    return out


def _find_levers(model) -> dict:
    for m in model.modules():
        lv = getattr(m, "memory_levers", None)
        if isinstance(lv, dict) and lv:
            return dict(lv)
    return {}


# ---------------------------------------------------------------------------------------- #
# the HELD-OUT eval dataset -- train() lines 8720-8920, replayed                             #
# ---------------------------------------------------------------------------------------- #
def build_eval_dataset(model, cfg, args, config: dict, *, with_perception_targets: bool = True,
                       dataset_cls=None):
    """-> (e_ds, e_eps, record). `with_perception_targets=False` skips the agent / map / 3-D / VIS-1
    joins (only the LOSS reads them; the forward reads `map_ep`'s episode id, which the caller
    passes itself). `dataset_cls` lets a caller pass a V3Dataset subclass -- the window CONTRACT is
    the trainer's either way."""
    tr = trainer()
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import build_v2_providers, load_or_build_manifest, stable_episode_id
    rec: dict = {}
    # train():8369-8376 -- the dataset class and its constructor kwargs (the train side's)
    want_lan = bool(args.graft_lan or args.goal_str)
    base_cls = tr.lan_dataset_class(tr.V3Dataset) if want_lan else tr.V3Dataset
    dcls = dataset_cls or base_cls
    if dataset_cls is not None and not issubclass(dataset_cls, tr.V3Dataset):
        raise SystemExit(f"[{LOADER}] dataset_cls must subclass the trainer's V3Dataset")
    kw = dict(window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    if want_lan:
        kw["lan_cfg"] = tr.DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                                         min_lead_m=args.lan_min_lead_m)
    # train():8724 -- the SAME provider call the train side uses
    e_eps = build_v2_providers([args.eval_cache], lru_size=args.v2_lru)
    # (train():8725-8731 -- the train/eval overlap refusal needs the TRAIN cache; not built here)
    e_ds = dcls(e_eps, **kw)
    e_ds.u8_frames = bool(getattr(args, "u8_batches", False))
    nav_on = bool(getattr(args, "nav_from_v7", False))
    if not args.eval_labels:
        raise SystemExit(f"[{LOADER}] the run's argv has no --eval-labels")
    # train():8734-8748 -- labels, the A16 label clock, and G3 with train()'s exact arguments
    e_lab, e_man = v7l.load_v7_labels(args.eval_labels, allow_oracle_nav=True)
    e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}
    e_ds.v7_dt = 0.1
    rec["label_clock"] = e_ds.enable_clip_clock(getattr(args, "clip_clock_sidecar", None))
    g3 = e_ds.assert_label_clock_true(
        getattr(args, "clip_clock_sidecar", None), split="eval",
        synthetic=bool(getattr(args, "synth_episodes", 0)),
        max_unverified_frac=getattr(args, "label_clock_max_unverified", None),
        exclude_unverified_tactical=True)
    by_sid = {int(stable_episode_id(str(c))): sha12(str(c))
              for c in load_or_build_manifest(args.eval_cache, verbose=False)["clip_id"]}
    if isinstance(g3, dict):
        g3 = dict(g3)
        g3["tactical_excluded_sha12"] = sorted(by_sid.get(int(s), f"sid:{s}")
                                               for s in g3.get("tactical_excluded_sids") or [])
    rec["label_clock"]["g3"] = g3
    rec["labels"] = {"path": args.eval_labels, "md5": e_man.md5, "n_records": e_man.n_records}
    # train():8751-8758 -- the nav source (the TRAIN normaliser for --nav-args is not in a run
    # record; refused rather than fitted on the split it scores)
    if nav_on:
        rec["nav"] = e_ds.enable_nav_from_v7(e_man)
        if getattr(args, "nav_args", False):
            raise SystemExit(f"[{LOADER}] --nav-args needs the TRAIN normaliser; not in this run")
    # train():8783-8797
    e_ds.ego_history = bool(getattr(args, "ego_history", False))
    e_ds.r7_agent_future = float(getattr(args, "w_r7_scorer", 0.0) or 0.0) > 0.0
    if (float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0
            or float(getattr(args, "w_tac_v6", 0.0) or 0.0) > 0.0):
        e_ds.tac_goal_targets = True
        e_ds.tac_goal_negatives = str(getattr(args, "tac_goal_negatives", "measured"))
    # train():8802-8816 -- the ceilings
    if getattr(args, "max_speed_input", False):
        raise SystemExit(f"[{LOADER}] E16 --max-speed-input is not a refcv7 arm")
    if getattr(args, "max_speed_input_v6", False):
        rec["max_speed_v6"] = e_ds.enable_max_speed_v6(
            str(getattr(args, "speed_max_sidecar_v6_eval", None) or args.speed_max_sidecar_v6),
            e_man)
    # train():8820-8863 -- the agent join, with the TRAIN split's pad (from the stamp)
    if with_perception_targets and getattr(args, "agent_join", None):
        from train_p8_occupancy import JoinFileReader as _JFR
        pad = int(((config.get("agent_join_stats") or {}).get("train") or {}).get("agent_pad", 0))
        if pad <= 0:
            raise SystemExit(f"[{LOADER}] config.json carries no train agent_pad")
        t_j = time.time()
        _e_rd = _JFR(args.agent_join, episode_ids={int(e.episode_id) for e in e_eps},
                     with_rates=not bool(getattr(args, "agent_join_no_rates", False)),
                     with_track_ids=bool(getattr(args, "join3d", None)))
        if str(getattr(args, "bev_aux", "off")) != "off":
            raise SystemExit(f"[{LOADER}] --bev-aux is not a refcv7 arm")
        rec["agent_join"] = e_ds.enable_agent_join(
            _e_rd, pad=pad, allow_legacy_ids=bool(getattr(args, "agent_join_allow_legacy_ids", False)))
        rec["agent_join"]["pad_source"] = "config.json agent_join_stats.train.agent_pad"
        rec["agent_join"]["load_s"] = round(time.time() - t_j, 1)
    # train():8868-8892 -- the map target (10 cm under --map-hires on) and the 3-D join
    if with_perception_targets and (getattr(args, "map_gt_root", None)
                                    or getattr(args, "join3d", None)):
        _e_clip, _e_ns = tr._clip_table_for_caches([args.eval_cache])
        if getattr(args, "map_gt_root", None) and tr._map_hires_on(args):
            rec["map_fine"] = e_ds.enable_map_hires(
                tr._sem_fine.FineMapGTStore(Path(args.map_gt_root),
                                            max_open=int(getattr(args, "map_lru", 4) or 4),
                                            extent=tr._mhr.declared_extent(args)),
                _e_clip, _e_ns, min_coverage=getattr(args, "map_min_coverage", None))
        elif getattr(args, "map_gt_root", None):
            rec["map_gt"] = e_ds.enable_map_gt(
                tr._perception_targets.MapGTStore(Path(args.map_gt_root),
                                                  max_open=int(getattr(args, "map_lru", 4) or 4)),
                _e_clip, _e_ns, min_coverage=getattr(args, "map_min_coverage", None))
        if getattr(args, "join3d", None) and e_ds.agent_join is not None:
            if e_ds.map_clip_of_ep is None:
                e_ds.map_clip_of_ep, e_ds.map_n_stack = _e_clip, _e_ns
            t_j = time.time()
            rec["join3d"] = e_ds.enable_join3d(tr.require_join3d(
                tr._agent_cuboid.open_join3d(args.join3d, clips=set(_e_clip.values())),
                args.join3d, len(set(_e_clip.values())), split="eval"))
            rec["join3d"]["load_s"] = round(time.time() - t_j, 1)
    # train():8894-8895 -- refcv7 A9 R3: the VIS-1 sidecar on the EVAL split (train() reads the
    # sidecar object once, at 8685, for the train split)
    if with_perception_targets and getattr(args, "slot_vis1", False):
        _vis1_sc = tr._vis1.VIS1Sidecar(args.vis1_sidecar)
        rec["vis1"] = {"sidecar": _vis1_sc.stamp(),
                       "eval": e_ds.enable_vis1(_vis1_sc, split="eval")}
    # ⭐ loader-side, and a REFUSAL: every eval episode must have its camera in the model's banks.
    # train() measures the RIG camera against its TRAIN episodes (train():9000,
    # `assert_rig_camera_covers`); the eval split is a different clip set, so it is measured here
    # -- the 0.25 m lift bank would otherwise refuse mid-eval, at the first uncovered window.
    if model is not None:
        ids = [int(e.episode_id) for e in e_eps]
        cov = {}
        for nm in ("_lift_bank_hires", "_lift_bank"):
            bank = getattr(model, nm, None)
            if bank is not None:
                c = bank.coverage(ids)
                cov[nm] = c
                if c["n_missing"]:
                    raise SystemExit(f"[{LOADER}] model.{nm} covers {c['n_covered']}/{c['n']} eval "
                                     f"episodes -- the lift has no camera for the rest")
        cam = getattr(model, "_rig_camera", None)
        if cam is not None and hasattr(cam, "coverage"):
            c = cam.coverage(ids)
            cov["_rig_camera"] = {k: v for k, v in c.items() if k != "missing_sample"}
            if c.get("n_missing") and not bool(getattr(args, "agent_rig_extrinsics_allow_partial",
                                                       False)):
                raise SystemExit(f"[{LOADER}] the rig camera bank covers {c['n_covered']}/{c['n']} "
                                 f"eval episodes")
        rec["camera_coverage"] = cov
    rec["n_episodes"] = len(e_eps)
    rec["n_windows"] = len(e_ds)
    return e_ds, e_eps, rec


def inrun_eval_perm(e_ds, eval_batches: int, batch: int) -> list[int]:
    """train():8905-8907 verbatim: the FIXED deterministic-random subset the in-run eval scores."""
    import torch
    g_ev = torch.Generator().manual_seed(12345)
    n_need = eval_batches * batch
    return torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()


def load_config(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
