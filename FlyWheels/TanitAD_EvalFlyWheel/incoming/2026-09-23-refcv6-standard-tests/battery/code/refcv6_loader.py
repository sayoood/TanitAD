"""refcv6 LOADER — the model and the held-out eval dataset, built EXACTLY as the trainer builds them.

EvalFlyWheel · refcv6 standard tests · battery stream · 2026-09-23.

WHY THIS FILE EXISTS (and why `taniteval/tools/refcv3_arm.py::load_model` is not enough on its own)
----------------------------------------------------------------------------------------------------
refcv3_arm.py already rebuilds a REF-C checkpoint through the trainer's `build_parser` +
`_pin_trainer_cfg`, and rebuilds refcv6's perception branch from the `config.json` stamp. Read
against `refc_v3_train.train()` for THIS run it misses three things, each MEASURED from source:

  (1) the BEV lift bank is built WITHOUT `equalize_bottom_rows` (refcv3_arm.py:1101 passes
      `heights_m`; refc_v3_train.py:6826-6830 passes `equalize_bottom_rows=43` for this run);
  (2) its roll never feeds `v_max_ms` (the v6 max-speed input) and never calls
      `core.set_ego_window` (the ego-history GRU) -- refc_v3.py:1937 and refc.py:4413 REFUSE
      both omissions on a refcv6 build, so the full arm cannot be rolled by it at all
      (the E9 run of 2026-09-19 was a `--size tiny` A3 checkpoint without those channels);
  (3) the model attributes train() sets on the MODEL (not the config) that the loss and the
      forward read: `_w_*` weights, `_cls_class_weight` (b1), `_rig_camera`, the tactical-goal
      `pos_weight` / class mask, the withheld-bank mode, the loaded anchor controls.

So this module replays train()'s model- and eval-dataset construction line for line (every
block cites the trainer line it mirrors), with exactly these documented departures:

  * `--trunk-compile` is REMOVED from argv (no Triton on Windows). It wraps the backbone call
    in `torch.compile` and does not touch the module tree or the state_dict
    (LAUNCH_READINESS_FIXES.md §10b); every other lever (chunk 8, frozen + folded BN,
    bf16 + NHWC, frame dedup) is KEPT, so the dev box runs the run's own arithmetic minus
    Inductor's fusion.
  * Thor data paths in argv are REMAPPED to the local eval kit (recorded, per flag).
  * The TRAIN split is never built. The four train-split constants the eval reads are taken
    from the run's own `config.json` stamp, never re-derived:
      - `agent_pad` 397                         (config.json agent_join_stats.train.agent_pad)
      - tactical-goal `pos_weight` [22]          (config.json tac_goal_stats.pos_weight)
      - tactical-goal class mask [22]            (tanitad mask_report over the STAMPED census,
                                                  cross-checked against the stamped trainable list)
      - withheld-bank mode `fixed`               (config.json withheld_bank)
  * The checkpoint is loaded STRICTLY; `load_anchors` from the kit's anchor file must equal
    the checkpoint's own anchor buffers bit for bit (a control that must read a known value).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------------------- #
# paths                                                                                     #
# ---------------------------------------------------------------------------------------- #
REPO = Path(os.environ.get("REFCV6_REPO", "C:/Users/Admin/ev6"))
STACK = REPO / "stack"
SCRIPTS = STACK / "scripts"
TANITEVAL = REPO / "taniteval"
KIT = Path(os.environ.get("REFCV6_KIT", "D:/refcv6_eval_kit"))

#: Thor path -> kit path. Every remap is recorded in the load record.
PATH_REMAP = {
    "--anchors": str(KIT / "data/anchors/refc_anchors_6s_v0cond_alat_117.pt"),
    "--eval-cache": str(KIT / "data/refcv6-b1-416x1024-eval139"),
    "--eval-labels": str(KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"),
    "--speed-max-sidecar-v6-eval": str(KIT / "data/refcv6_speed_max_v8_eval.jsonl"),
    "--agent-join": str(KIT / "data/joins/b1_train_plus_eval_agents.jsonl.xz"),
    "--agent-rig-extrinsics": str(KIT / "data/refcv6_train_eval139_extrinsics.json"),
    "--map-gt-root": str(KIT / "data/sam3_gt_eval_thor137"),
    "--join3d": str(KIT / "data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"),
}
#: flags whose TRAIN-side file is not in the kit and is not read by anything this module builds
TRAIN_ONLY_PATHS = ("--v2-cache", "--v7-labels", "--speed-max-sidecar-v6", "--out")
#: removed from argv, with the reason
DROP_FLAGS = {"--trunk-compile": "no Triton on Windows; torch.compile wraps the backbone CALL "
                                 "only -- module tree and state_dict identical "
                                 "(LAUNCH_READINESS_FIXES.md §10b)"}


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
        raise SystemExit(f"[refcv6_loader] tanitad imported from {here}, not from {want} -- "
                         f"an editable install elsewhere is shadowing the tree under test")


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


def sha12(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------- #
# argv                                                                                      #
# ---------------------------------------------------------------------------------------- #
def remap_argv(argv: list[str], remap: dict | None = None) -> tuple[list[str], dict]:
    """The run's argv with Thor paths -> kit paths and `--trunk-compile` removed."""
    remap = dict(PATH_REMAP if remap is None else remap)
    out, rec = [], {"remapped": {}, "dropped": {}, "train_only_kept_unread": {}}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in DROP_FLAGS:
            rec["dropped"][a] = DROP_FLAGS[a]
            i += 1
            continue
        if a in remap and i + 1 < len(argv):
            rec["remapped"][a] = {"thor": argv[i + 1], "local": remap[a]}
            out += [a, remap[a]]
            i += 2
            continue
        if a in TRAIN_ONLY_PATHS and i + 1 < len(argv):
            rec["train_only_kept_unread"][a] = argv[i + 1]
        out.append(a)
        i += 1
    return out, rec


def parse_args(config: dict, remap: dict | None = None):
    tr = trainer()
    argv, rec = remap_argv(list(config["argv"]), remap)
    args = tr.build_parser().parse_args(argv)
    return args, argv, rec


# ---------------------------------------------------------------------------------------- #
# the MODEL -- refc_v3_train.train() lines 6711-6990 + 7618, replayed                       #
# ---------------------------------------------------------------------------------------- #
def build_model(config: dict, ckpt_path: str, device: str = "cuda", remap: dict | None = None,
                strict: bool = True):
    """-> (model, cfg, args, record). Mirrors train() block by block (line refs = ev6 == 287d72e).

    ⛔ RNG-NEUTRAL. train() calls `torch.manual_seed(args.seed)` before building (replayed below,
    so module init is the run's). Left bare, that RESET THE CALLER'S RNG: refcv3_arm.run_dump seeds
    `--infer-seed` and THEN calls load_model, so every "inference seed" became seed 0 and a DDIM
    replicate read bit-identical to float noise (MEASURED 2026-09-24 on the CPU smoke: seed 0 vs 1,
    max |os path diff| 4.5e-8 m). The build now runs inside `torch.random.fork_rng`, so the
    caller's CPU and CUDA generator states are exactly what they were before the call."""
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
    rec: dict = {"argv_local": argv, "argv_remap": arec, "departures": []}
    # train():6714-6720 -- the argument refusals (no data read by any of them)
    tr._check_nav_from_v7_args(args)
    tr._check_max_speed_args(args)
    tr._check_goal_point_args(args)
    tr.check_effective_weights(args)
    # train():6726 -- the anchor artifact FIRST (units resolved onto args for the pin)
    art = tr._read_anchor_artifact(args)
    # train():6743 -- the seed (only model INIT consumes it; every parameter is then loaded)
    torch.manual_seed(args.seed)
    if bool(getattr(args, "cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
    # train():6747-6749 -- the config through the trainer's own pin
    cfg = tr._pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    # (the delta check at 6754-6762 is a launch gate on a config PAIR; it reads no weights)
    tr._check_anchor_artifact_against_cfg(art, cfg, args)
    if args.graft_lan or args.goal_str:
        cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
    # ⚠️ ImageNet init is overwritten by the strict load below; HF_HUB_OFFLINE keeps it local.
    model = v3.RefCV3Model(cfg).to(device)
    # train():6773-6775
    model._w_agent = float(getattr(args, "w_agent", tr.AGENT_WEIGHT_DEFAULT))
    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
    # train():6790-6808 -- the class weight
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
    # train():6809-6816
    model._w_map = float(getattr(args, "w_map", 0.0) or 0.0)
    model._w_box3d = float(getattr(args, "w_box3d", 0.0) or 0.0)
    model._map_lift_valid_mask = bool(getattr(args, "map_lift_valid_mask", True))
    model._box3d_visible_filter = bool(getattr(args, "box3d_visible_filter", True))
    model._perception = None
    model._lift_bank = None
    # train():6819-6832 -- the perception branch + the per-clip lift bank WITH equalize_bottom_rows
    if model._w_map > 0.0 or model._w_box3d > 0.0:
        _perc = tr._perc
        _pcfg = _perc.PerceptionBranchConfig(w_map=model._w_map, w_box3d=model._w_box3d)
        model._perception = _perc.build_perception_branch(model, _pcfg).to(device)
        _pframe = _perc.frame_for_model(model)
        if model._w_map > 0.0:
            _pe, _ptable = tr._read_rig_extrinsics(str(getattr(args, "agent_rig_extrinsics", "")))
            if _ptable is None:
                raise SystemExit("[refcv6_loader] --w-map > 0 needs a PER-CLIP extrinsics table")
            model._lift_bank = _perc.LiftGeometryBank(
                _ptable, frame=_pframe, stride=int(_pcfg.stride),
                equalize_bottom_rows=int(getattr(args, "equalize_bottom_rows", 0) or 0))
        rec["perception"] = {
            "cfg": _pcfg.as_dict(),
            "branch_params": model._perception.param_breakdown(),
            "lift_bank_n_clips": 0 if model._lift_bank is None else len(model._lift_bank),
            "equalize_bottom_rows": int(getattr(args, "equalize_bottom_rows", 0) or 0),
            "frame": {"height": int(_pframe.height), "width": int(_pframe.width),
                      "projection": str(_pframe.projection), "f_ref": float(_pframe.f_ref)}}
    # train():6878-6887 -- tactical-goal carriers (pos_weight/mask filled from the STAMP below)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model._w_tac_v6 = float(getattr(args, "w_tac_v6", 0.0) or 0.0)
    # train():6890-6895 -- refcv7 (off here; the same built-vs-weight refusal)
    model._w_r7_wta = float(getattr(args, "w_r7_wta", 0.0) or 0.0)
    model._w_r7_scorer = float(getattr(args, "w_r7_scorer", 0.0) or 0.0)
    model._r7_n_perturb = int(getattr(args, "r7_n_perturb", 0) or 0)
    model._r7_nav_tau_rad = float(getattr(args, "r7_nav_tau_rad", 0.0) or 0.0)
    model._r7_calls = 0
    if (getattr(model, "refcv7_wta", None) is not None) != (model._w_r7_wta > 0.0):
        raise SystemExit("[refcv6_loader] refcv7 heads built/weight disagree")
    if (getattr(model, "tac_decoder_v6", None) is not None) != (model._w_tac_v6 > 0.0):
        raise SystemExit("[refcv6_loader] tac_decoder_v6 built/weight disagree")
    # train():6932-6943
    model._w_u0 = float(getattr(args, "w_u0", tr.U0_WEIGHT_DEFAULT))
    model._w_goal_point = float(getattr(args, "goal_point_w", tr.GOAL_POINT_WEIGHT_DEFAULT))
    model._rig_camera, _cam_stamp = tr._build_rig_camera(cfg, args)
    rec["rig_camera"] = _cam_stamp
    rec["departures"].append(
        "train():6946 `assert_ground_prior_is_supervised` (a startup gradient probe) is NOT run: "
        "it measures a training property and writes nothing the forward or the loss reads")
    # train():6957-6985 -- the anchor vocabulary
    if args.anchors:
        anc = art.anchors.to(device)
        _ctrl = None if art.controls is None else art.controls.to(device)
        if bool(getattr(args, "anchor_v0_conditioned", False)) and _ctrl is None:
            raise SystemExit("[refcv6_loader] v0-conditioned build but the anchor file has no controls")
        model.core.decoder.load_anchors(anc, _ctrl)
        rec["anchors"] = {"path": args.anchors, "shape": list(anc.shape),
                          "controls_shape": None if _ctrl is None else list(_ctrl.shape),
                          "control_units": art.control_units,
                          "sha256_16": hashlib.sha256(anc.detach().float().cpu().numpy()
                                                      .tobytes()).hexdigest()[:16]}
    # ---- the TRAIN-split constants, from the run's own stamp ------------------------- #
    tgs = config.get("tac_goal_stats") or {}
    if model._w_tac_goal > 0.0 or model._w_tac_v6 > 0.0:
        pw = tgs.get("pos_weight")
        census = tgs.get("census")
        if not pw or census is None:
            raise SystemExit("[refcv6_loader] config.json carries no tac_goal_stats.pos_weight/"
                             "census -- the eval loss's goal term cannot be reproduced")
        mask_rep = tr._tac_goal_head.mask_report(census)
        want_trainable = list(tgs.get("trainable") or [])
        if list(mask_rep["trainable"]) != want_trainable:
            raise SystemExit(f"[refcv6_loader] mask_report over the STAMPED census gives trainable "
                             f"{mask_rep['trainable']} but the stamp says {want_trainable}")
        model._tac_goal_pos_weight = torch.tensor(pw, dtype=torch.float32)
        model._tac_goal_class_mask = torch.tensor(mask_rep["mask"], dtype=torch.float32)
        rec["tac_goal_from_stamp"] = {"pos_weight": pw, "mask": list(mask_rep["mask"]),
                                      "n_trainable": int(mask_rep["n_trainable"]),
                                      "source": "config.json tac_goal_stats (train split)"}
    # train():7618 -- the withheld bank (mode `fixed` reads no episodes)
    wb = config.get("withheld_bank") or {}
    if str(getattr(args, "withheld_bank", "fixed")) != "fixed":
        raise SystemExit("[refcv6_loader] withheld bank mode != fixed needs the train episodes")
    rec["withheld_bank"] = tr._apply_withheld_bank(model, args, [], device)
    if wb and wb.get("mode") != rec["withheld_bank"]["mode"]:
        raise SystemExit(f"[refcv6_loader] withheld bank {rec['withheld_bank']} != stamp {wb}")
    # ---- the STRICT load --------------------------------------------------------------- #
    anc_before = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k.startswith("core.decoder.anchor")}
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        raise SystemExit(f"[refcv6_loader] {ckpt_path} has no 'model' key")
    res = model.load_state_dict(ck["model"], strict=strict)
    rec["state_dict"] = {"strict": strict, "missing": list(getattr(res, "missing_keys", [])),
                         "unexpected": list(getattr(res, "unexpected_keys", [])),
                         "n_keys": len(ck["model"]), "ckpt_keys": sorted(k for k in ck if k != "model"),
                         "step": ck.get("step")}
    # ⭐ CONTROL: the kit's anchor file must equal the checkpoint's own anchor buffers.
    anc_ctl = {}
    for k, v in anc_before.items():
        w = model.state_dict()[k]
        anc_ctl[k] = {"shape": list(v.shape),
                      "max_abs_diff": float((v.float() - w.float().to(v.device)).abs().max())
                      if v.numel() else 0.0}
    rec["anchor_file_vs_ckpt_buffers"] = anc_ctl
    del ck
    # ---- eval-time settings -------------------------------------------------------------- #
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bd = v3.param_breakdown_v3(model)
    stamped = config.get("param_breakdown") or {}
    rec["param_breakdown"] = {"rebuilt": {k: int(v) for k, v in bd.items()},
                              "config_json": stamped,
                              "equal": {k: int(v) for k, v in bd.items()} == {k: int(v) for k, v in stamped.items()}}
    enc = model.core.encoder
    rec["trunk_memory_levers_built"] = dict(getattr(getattr(enc, "trunk", enc), "memory_levers", {}) or
                                            getattr(enc, "memory_levers", {}) or {})
    rec["trunk_memory_levers_run"] = config.get("trunk_memory_levers")
    rec["mode"] = getattr(args, "mode", "diffusion")
    rec["decoder_steps"] = int(cfg.core.decoder.diffusion_steps) if rec["mode"] == "diffusion" else 0
    rec["sampler"] = str(getattr(cfg.core.decoder, "sampler", "none"))
    rec["build_s"] = round(time.time() - t0, 1)
    return model, cfg, args, rec


def _find_levers(model) -> dict:
    for m in model.modules():
        lv = getattr(m, "memory_levers", None)
        if isinstance(lv, dict) and lv:
            return dict(lv)
    return {}


# ---------------------------------------------------------------------------------------- #
# the HELD-OUT eval dataset -- train() lines 7092-7101 + 7394-7560, replayed                 #
# ---------------------------------------------------------------------------------------- #
def build_eval_dataset(model, cfg, args, config: dict, *, with_perception_targets: bool = True,
                       dataset_cls=None):
    """-> (e_ds, e_eps, record). `with_perception_targets=False` skips the agent / map / 3-D joins
    (only the LOSS reads them; the forward reads `map_ep`'s episode id, which the battery passes
    itself). `dataset_cls` lets the battery pass a V3Dataset subclass that skips future-frame
    decoding -- the window CONTRACT is the trainer's either way."""
    import torch
    tr = trainer()
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import build_v2_providers, stable_episode_id
    rec: dict = {}
    want_lan = bool(args.graft_lan or args.goal_str)
    base_cls = tr.lan_dataset_class(tr.V3Dataset) if want_lan else tr.V3Dataset
    dcls = dataset_cls or base_cls
    if dataset_cls is not None and not issubclass(dataset_cls, tr.V3Dataset):
        raise SystemExit("[refcv6_loader] dataset_cls must subclass the trainer's V3Dataset")
    kw = dict(window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    if want_lan:
        kw["lan_cfg"] = tr.DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                                         min_lead_m=args.lan_min_lead_m)
    e_eps = build_v2_providers([args.eval_cache], lru_size=args.v2_lru)
    e_ds = dcls(e_eps, **kw)
    e_ds.u8_frames = bool(getattr(args, "u8_batches", False))
    nav_on = bool(getattr(args, "nav_from_v7", False))
    if not args.eval_labels:
        raise SystemExit("[refcv6_loader] the run's argv has no --eval-labels")
    e_lab, e_man = v7l.load_v7_labels(args.eval_labels, allow_oracle_nav=True)
    e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}
    e_ds.v7_dt = 0.1
    rec["labels"] = {"path": args.eval_labels, "md5": e_man.md5, "n_records": e_man.n_records}
    if nav_on:
        rec["nav"] = e_ds.enable_nav_from_v7(e_man)
        if getattr(args, "nav_args", False):
            raise SystemExit("[refcv6_loader] --nav-args needs the TRAIN normaliser; not in this run")
    e_ds.ego_history = bool(getattr(args, "ego_history", False))
    e_ds.r7_agent_future = float(getattr(args, "w_r7_scorer", 0.0) or 0.0) > 0.0
    if (float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0
            or float(getattr(args, "w_tac_v6", 0.0) or 0.0) > 0.0):
        e_ds.tac_goal_targets = True
        e_ds.tac_goal_negatives = str(getattr(args, "tac_goal_negatives", "measured"))
    if getattr(args, "max_speed_input", False):
        raise SystemExit("[refcv6_loader] E16 max-speed input is not this arm")
    if getattr(args, "max_speed_input_v6", False):
        rec["max_speed_v6"] = e_ds.enable_max_speed_v6(
            str(getattr(args, "speed_max_sidecar_v6_eval", None) or args.speed_max_sidecar_v6), e_man)
    if with_perception_targets and getattr(args, "agent_join", None):
        from train_p8_occupancy import JoinFileReader as _JFR
        pad = int(((config.get("agent_join_stats") or {}).get("train") or {}).get("agent_pad", 0))
        if pad <= 0:
            raise SystemExit("[refcv6_loader] config.json carries no train agent_pad")
        t_j = time.time()
        _e_rd = _JFR(args.agent_join, episode_ids={int(e.episode_id) for e in e_eps},
                     with_rates=not bool(getattr(args, "agent_join_no_rates", False)),
                     with_track_ids=bool(getattr(args, "join3d", None)))
        if str(getattr(args, "bev_aux", "off")) != "off":
            raise SystemExit("[refcv6_loader] bev_aux is not this arm")
        rec["agent_join"] = e_ds.enable_agent_join(
            _e_rd, pad=pad, allow_legacy_ids=bool(getattr(args, "agent_join_allow_legacy_ids", False)))
        rec["agent_join"]["pad_source"] = "config.json agent_join_stats.train.agent_pad"
        rec["agent_join"]["load_s"] = round(time.time() - t_j, 1)
    if with_perception_targets and (getattr(args, "map_gt_root", None) or getattr(args, "join3d", None)):
        _e_clip, _e_ns = tr._clip_table_for_caches([args.eval_cache])
        if getattr(args, "map_gt_root", None):
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
    rec["n_episodes"] = len(e_eps)
    rec["n_windows"] = len(e_ds)
    return e_ds, e_eps, rec


def inrun_eval_perm(e_ds, eval_batches: int, batch: int) -> list[int]:
    """train():7561-7563 verbatim: the FIXED deterministic-random subset the in-run eval scores."""
    import torch
    g_ev = torch.Generator().manual_seed(12345)
    n_need = eval_batches * batch
    return torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()


def load_config(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
