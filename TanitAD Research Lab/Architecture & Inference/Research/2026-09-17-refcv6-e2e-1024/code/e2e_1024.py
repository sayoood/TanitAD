"""refcv6 SPEC-v2 END-TO-END on the 139 B1 eval clips at 256 x 1024 (CPU).

The PI asked for the pipeline to be validated on the 139 eval clips. The
existing `stack/tests/test_refcv6_perception_realdata.py` validates the DATA
and the GEOMETRY; it pools RAW PIXELS as a trunk stand-in and never runs a
trunk, a tactical layer or a planner. This runs the model.

⛔ THE MODEL IS BUILT THROUGH THE TRAINER'S OWN ASSEMBLY, never re-implemented:
``refc_v3_train.build_parser`` -> ``refc_v3_train._pin_trainer_cfg`` ->
``refc_v3.RefCV3Model``, and the batch comes from the trainer's own
``build_v2_providers`` -> ``V3Dataset`` and is scored by the trainer's own
``compute_losses_v3``. Where an arm CANNOT go through that path, the script
says so in its own output under ``assembly`` / ``cli_reachable`` rather than
forking the assembly silently.

⛔ Clip ids appear ONLY as sha12 (``tanitad.data.semantic_map_gt.sha12``).
⛔ No geometry literal: every shape is read from the payload
   (``v2_dataset.stored_frame_of``) or from timm's ``feature_info``
   (``trunk_shapes.TrunkSpec``).

Arms (``--arm``):
  assembly    build the model, dump provenance + seam stamp + a CLI-reachability
              audit of every SPEC-v2 seam.
  identity    bit-identity at 256x1024 against the PRE-refcv6 files materialised
              FROM GIT, with every refcv6 flag off.
  forward     per-clip real forward + backward through ``compute_losses_v3``;
              every loss, every shape, per-module ``grad_abs_sum``, RSS, wall time.
  perception  per-clip BEV lift -> map head (real SAM3 GT) and 3-D box head
              (real cuboid join), each backwarded SEPARATELY so the trunk
              gradient is attributable per head.
  tactical    the SPEC §4 behaviour decoder + §5 max-speed one-hot, switched on
              at the CONFIG because the trainer's CLI has no flag for either.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WT = Path(os.environ.get("TANITAD_WT", Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(WT / "stack"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


# --------------------------------------------------------------------------- #
# the trainer, imported as a module (it is a script, not a package member)     #
# --------------------------------------------------------------------------- #
def load_trainer():
    p = WT / "stack" / "scripts" / "refc_v3_train.py"
    spec = importlib.util.spec_from_file_location("refc_v3_train", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def rss_gb() -> float:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024.0 ** 3)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# the argv the trainer itself would be launched with                          #
# --------------------------------------------------------------------------- #
def frame_of_cache(cache: str):
    """The CanonicalFrame the cache was BUILT at -- read, never assumed.

    ⛔ This is why no geometry literal appears in this file: `--image-hw` is
    fed from the payload's own `frame` dict via `v2_dataset.stored_frame_of`.
    """
    import glob as _g
    from tanitad.data.v2_dataset import stored_frame_of
    p = sorted(_g.glob(os.path.join(cache, "*.v2ep.pt")))[0]
    return stored_frame_of(torch.load(p, map_location="cpu",
                                      weights_only=False))


def trainer_argv(cache: str, *, trunk_name: str, out: str,
                 v7_labels: str | None, anchors: str | None,
                 agents: str = "off",
                 agent_join: str | None = None) -> list[str]:
    """The SPEC-v2 launch line, minus anything the CLI does not expose."""
    fr = frame_of_cache(cache)
    a = [
        "--arm", "hier", "--size", "small",
        # §2 / §10.2 — the ImageNet trunk, resnet101 primary
        "--trunk", "timm", "--trunk-name", trunk_name,
        "--trunk-mode", "shared", "--trunk-fuse", "concat1x1",
        # §10.3 — ego history as an input
        "--ego-history",
        # §1 — the strategic layer is deactivated for this experiment
        "--no-strategic",
        # §10.1 — the geometry, READ OFF THE CACHE, never a literal
        "--image-hw", str(int(fr.height)), str(int(fr.width)),
        # §3 — F1..F9 (F7 needs an eval-consumer change that is not ours;
        # F8 is the deliberate-regression arm, so neither is in the primary)
        "--sampler", "ddim",
        "--f1-random-t", "--f2-dd-step", "--f3-per-layer",
        "--f4-adaln", "--f5-focal", "--f6-w-u0-zero", "--f9-assert-vocab",
        "--v2-cache", cache,
        "--out", out,
    ]
    if agents != "off":
        # ⛔ SPEC §4's tactical decoder attends to the AGENT SLOTS and the BEV
        # tokens. `RefCV3Model` never supplies `bev_tokens` (MEASURED: no
        # caller in stack/ outside tests passes it), so agent slots are the
        # ONLY scene the decoder can be given -- and with `--agents off`
        # `refc.py:4125` refuses the hook outright.
        a += ["--agents", agents]
        if agent_join:
            # ⛔ `--agents head` with `--w-agent 0` is REFUSED by the trainer
            # ("a detector that is never supervised ... would read as 'agent
            # tokens do not help' -- a REFUTATION manufactured by a missing
            # loss"). So the detector is supervised from the banked B1 EVAL
            # 2-D join, which is what puts `agent_box` in the batch.
            a += ["--agent-join", agent_join, "--w-agent", "0.05"]
    if v7_labels:
        a += ["--v7-labels", v7_labels]
    if anchors:
        # ⛔ NO --anchor-control-units: the artifact declares its own
        # (`control_units: alat`), and passing the flag would stamp the run
        # `control_units_source: cli-override-legacy-file`.
        a += ["--anchors", anchors, "--anchor-v0-conditioned",
              "--n-anchors", "117"]
    return a


def cache_dir(args) -> str:
    """``--v2-cache`` is ``nargs="+"``; this harness passes exactly one."""
    c = args.v2_cache
    return str(c[0] if isinstance(c, (list, tuple)) else c)


def build_args(trainer, argv: list[str]):
    ap = trainer.build_parser()
    args = ap.parse_args(argv)
    args._ew_parser = ap
    args._ew_argv = list(argv)
    return ap, args


def build_model(trainer, args, *, cfg_overrides: dict | None = None):
    """The trainer's OWN assembly prologue, in the trainer's own order.

    Every call here is a function defined in ``refc_v3_train`` or in
    ``tanitad.refs.refc_v3`` -- nothing is reconstructed locally.
    """
    from tanitad.refs import refc_v3 as v3
    trainer._check_nav_from_v7_args(args)
    trainer._check_max_speed_args(args)
    trainer._check_goal_point_args(args)
    trainer.check_effective_weights(args)
    art = trainer._read_anchor_artifact(args)
    cfg = trainer._pin_trainer_cfg(
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    trainer._check_anchor_artifact_against_cfg(art, cfg, args)
    applied = {}
    for k, v in (cfg_overrides or {}).items():
        applied[k] = [getattr(cfg, k, None), v]
        setattr(cfg, k, v)
    model = v3.RefCV3Model(cfg)
    model._w_agent = float(getattr(args, "w_agent",
                                   trainer.AGENT_WEIGHT_DEFAULT))
    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model._w_u0 = float(getattr(args, "w_u0", trainer.U0_WEIGHT_DEFAULT))
    model._w_goal_point = float(getattr(args, "goal_point_w",
                                        trainer.GOAL_POINT_WEIGHT_DEFAULT))
    model._rig_camera, _ = trainer._build_rig_camera(cfg, args)
    if args.anchors and art is not None:
        model.core.decoder.load_anchors(
            art.anchors, None if art.controls is None else art.controls)
    return cfg, model, art, applied


# --------------------------------------------------------------------------- #
# the data, through the trainer's own loader                                  #
# --------------------------------------------------------------------------- #
def build_dataset(trainer, cfg, args):
    from tanitad.data import parity
    from tanitad.data.v2_dataset import build_v2_providers
    par = parity.assert_v2_parity_cache(args.v2_cache, label="e2e-1024",
                                        require=False)
    eps = build_v2_providers(args.v2_cache, lru_size=int(args.v2_lru))
    ds = trainer.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                           channels=cfg.core.encoder.in_channels)
    ds.u8_frames = bool(getattr(args, "u8_batches", False))
    join = {}
    # ---- the trainer's OWN v7.2 join block (train(), `if args.v7_labels:`) --
    if args.v7_labels:
        import tanitad.data.v7_labels as v7l
        from tanitad.data.v2_dataset import stable_episode_id
        labels, manifest = v7l.load_v7_labels(args.v7_labels,
                                              allow_oracle_nav=True)
        by_sid = {stable_episode_id(l.clip_id): l for l in labels}
        ds.v7_by_sid = by_sid
        ds.v7_dt = 0.1
        hit = sum(1 for e in eps if int(e.episode_id) in by_sid)
        join = {"n_records": len(labels), "joined": hit,
                "n_episodes": len(eps), "frac": hit / max(len(eps), 1),
                "release": getattr(manifest, "release", None)}
    # ⭐ refcv6 §2b — the trainer sets this in the SAME block; without it the
    # model refuses the batch ("--ego-history but the batch carries no
    # pose_hist"), which is the correct refusal and the reason it is here.
    ds.ego_history = bool(getattr(args, "ego_history", False))
    # ---- the trainer's OWN agent-join block (train(), `if args.agent_join`) -
    # ⛔ `JoinFileReader` keys on `frame_idx`, the EPISODE index space
    # (`train_p8_occupancy.py:246`) -- NOT the RAW v2ep index `AgentJoin3D`
    # uses. Both spaces are emitted by the SAME builder under DIFFERENT names
    # (`build_b1_agent_join.py:15-32`); see raw/join_frame_key.json.
    if getattr(args, "agent_join", None):
        sys.path.insert(0, str(WT / "stack" / "scripts"))
        from train_p8_occupancy import JoinFileReader
        rd = JoinFileReader(args.agent_join,
                            episode_ids={int(e.episode_id) for e in eps},
                            with_rates=True)
        join["agent_join"] = {"n_records": rd.n_records, "n_clips": rd.n_clips,
                              "filtered_out": rd.n_records_filtered_out,
                              "max_agents_per_frame": rd.max_agents_per_frame}
        join["agent_join_census"] = ds.enable_agent_join(
            rd, pad=int(getattr(args, "agent_pad", 0)))
    return eps, ds, par, join


def clip_ids_by_episode(cache: str, eps) -> dict:
    """``episode index -> clip_id``, from the cache's own manifest sidecar."""
    man = torch.load(os.path.join(cache, "_v2manifest.pt"), map_location="cpu",
                     weights_only=False)
    ids = man.get("episode_uid") or man.get("episode_id")
    by_uid = {int(u): str(c) for u, c in zip(ids, man["clip_id"])}
    out, missing = {}, []
    for i, e in enumerate(eps):
        c = by_uid.get(int(e.episode_id))
        if c is None:
            missing.append(int(e.episode_id))
        out[i] = c
    if missing:
        raise SystemExit("[e2e] %d episodes have no clip_id in the manifest: "
                         "%s" % (len(missing), missing[:5]))
    return out


def one_window_per_episode(ds, n_eps: int) -> list[tuple[int, int]]:
    """(episode_index, dataset_index) — the MIDDLE window of each episode, so
    the sample is not the clip's first frames for every clip."""
    by_ep: dict[int, list[int]] = {}
    for i, (e, _t) in enumerate(ds.index):
        by_ep.setdefault(int(e), []).append(i)
    eps = sorted(by_ep)
    if n_eps and n_eps < len(eps):
        # ⛔ EVENLY SPACED, never the first N: the cache is written in clip-id
        # order, so "the first 45" is a slice of the id space, not a sample of
        # the corpus.
        step = len(eps) / float(n_eps)
        eps = [eps[int(i * step)] for i in range(n_eps)]
    out = []
    for e in eps:
        idxs = by_ep[e]
        out.append((e, idxs[len(idxs) // 2]))
    return out


# --------------------------------------------------------------------------- #
# arm: assembly                                                               #
# --------------------------------------------------------------------------- #
SPEC_SEAMS = {
    # SPEC section -> (config attribute or probe, the CLI flag that reaches it)
    "S2_timm_trunk": "--trunk / --trunk-name",
    "S2b_ego_history": "--ego-history",
    "S3_F1_random_t": "--f1-random-t",
    "S3_F2_dd_step": "--f2-dd-step",
    "S3_F3_per_layer": "--f3-per-layer",
    "S3_F4_adaln": "--f4-adaln",
    "S3_F5_focal": "--f5-focal",
    "S3_F6_w_u0_zero": "--f6-w-u0-zero",
    "S3_F7_n_noise": "--f7-samples-per-anchor",
    "S3_F8_waypoint_noise": "--f8-flat-noise",
    "S4_tactical_behaviour_decoder": None,
    "S5_max_speed_onehot_4value": None,
    "S6_map_head": None,
    "S6_box3d_head": None,
    "S10_1_image_1024": "--image-hw",
}


def arm_assembly(trainer, args, cfg, model, raw: Path) -> dict:
    from tanitad.models.trunk_shapes import TrunkSpec
    from tanitad.data.v2_dataset import stored_frame_of
    from tanitad.refs import refc_v3 as v3

    ap = args._ew_parser
    dests = {a.dest for a in ap._actions}
    opts = set()
    for a in ap._actions:
        opts.update(a.option_strings)

    enc = model.core.encoder
    prov = enc.provenance() if hasattr(enc, "provenance") else None
    stamp = trainer._seam_stamp(cfg, args)

    # what the payload says, read not assumed
    import glob
    p = sorted(glob.glob(os.path.join(cache_dir(args), "*.v2ep.pt")))[0]
    d = torch.load(p, map_location="cpu", weights_only=False)
    fr = stored_frame_of(d)
    spec = TrunkSpec.from_timm(cfg.core.encoder.trunk_name.split(".")[0], fr)

    reach = {}
    for seam, flag in SPEC_SEAMS.items():
        if flag is None:
            reach[seam] = {"cli_flag": None, "reachable": False}
        else:
            first = flag.split()[0]
            reach[seam] = {"cli_flag": flag, "reachable": first in opts}
    # the two config fields that exist but have no flag
    reach["S4_tactical_behaviour_decoder"]["config_field"] = "RefCV3Config.tac_decoder_v6"
    reach["S4_tactical_behaviour_decoder"]["field_exists"] = hasattr(cfg, "tac_decoder_v6")
    reach["S4_tactical_behaviour_decoder"]["arg_dest_present"] = "tac_decoder_v6" in dests
    reach["S5_max_speed_onehot_4value"]["config_field"] = "RefCV3Config.max_speed_onehot_v6"
    reach["S5_max_speed_onehot_4value"]["field_exists"] = hasattr(cfg, "max_speed_onehot_v6")
    reach["S5_max_speed_onehot_4value"]["arg_dest_present"] = "max_speed_onehot_v6" in dests
    # the perception heads: are they anywhere in the built model?
    mod_srcs = set()
    for _n, m in model.named_modules():
        mod_srcs.add(type(m).__module__ + "." + type(m).__name__)
    reach["S6_map_head"]["in_built_model"] = any(
        s.endswith(("bev_encoder.BEVMapBranch", "bev_encoder.MapHead",
                    "bev_lift.BEVLift")) for s in mod_srcs)
    reach["S6_box3d_head"]["in_built_model"] = any(
        s.endswith(("box3d_head.Box3DSlotDecoder", "box3d_head.Box3DMemory"))
        for s in mod_srcs)

    out = {
        "evidence_class": "MEASURED",
        "worktree": str(WT),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=WT,
                                   capture_output=True, text=True).stdout.strip(),
        "argv": list(args._ew_argv),
        "payload_frame": {"height": fr.height, "width": fr.width,
                          "f_ref": fr.f_ref, "projection": fr.projection},
        "encoder_image_hw": list(cfg.core.encoder.image_hw()),
        "encoder_in_channels": int(cfg.core.encoder.in_channels),
        "window": int(cfg.core.window),
        "trunk_provenance": prov,
        "trunk_spec_from_timm": {
            "perception_stride": spec.perception.stride,
            "perception_hw": list(spec.perception.hw),
            "perception_channels": spec.perception.channels,
            "planner_stride": spec.planner.stride,
            "planner_hw": list(spec.planner.hw),
            "planner_channels": spec.planner.channels,
        },
        "seam_stamp": stamp,
        "param_breakdown": v3.param_breakdown_v3(model),
        "n_params_total": int(sum(p.numel() for p in model.parameters())),
        "spec_seam_cli_reachability": reach,
        "n_cli_dests": len(dests),
    }
    (raw / "assembly.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# arm: identity — the pre-refcv6 files, materialised FROM GIT                 #
# --------------------------------------------------------------------------- #
PRE_REFCV6 = "4d74774"          # last commit before refcv6 entered refc.py


def arm_identity(raw: Path, pre_ref: str = PRE_REFCV6) -> dict:
    """With every refcv6 flag off, does the 256x1024 forward reproduce the
    pre-refcv6 model bit for bit?

    The baseline is materialised FROM GIT into a private package so the claim
    is against the committed pre-refcv6 code, never against a description of it.
    """
    import shutil
    import tempfile
    from tanitad.refs import refc_v3 as v3

    tmp = Path(tempfile.mkdtemp(prefix="pre_refcv6_"))
    res: dict = {"evidence_class": "MEASURED", "baseline_commit": pre_ref,
                 "materialised": [], "blocked": None}
    try:
        # a whole private copy of the tanitad package at the baseline commit
        files = subprocess.run(["git", "ls-tree", "-r", "--name-only",
                                pre_ref, "stack/tanitad"],
                               cwd=WT, capture_output=True, text=True)
        names = [f for f in files.stdout.splitlines() if f.endswith(".py")]
        if len(names) < 50:
            res["blocked"] = (f"git ls-tree returned only {len(names)} files "
                              f"for stack/tanitad at {pre_ref} -- SHORT READ, "
                              f"inadmissible (CLAUDE.md ls-tree trap)")
            return res
        for n in names:
            blob = subprocess.run(["git", "show", f"{pre_ref}:{n}"], cwd=WT,
                                  capture_output=True)
            if blob.returncode != 0:
                res["blocked"] = f"git show failed for {n}"
                return res
            dst = tmp / Path(n).relative_to("stack")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(blob.stdout)
        res["materialised"] = [len(names)]

        sys.path.insert(0, str(tmp))
        saved = {k: v for k, v in sys.modules.items()
                 if k == "tanitad" or k.startswith("tanitad.")}
        for k in list(saved):
            del sys.modules[k]
        try:
            import tanitad as old_pkg
            assert str(Path(old_pkg.__file__).parent) == str(tmp / "tanitad"), \
                f"the baseline import resolved to {old_pkg.__file__}"
            from tanitad.refs import refc_v3 as old_v3
            old_mod = old_v3
        finally:
            pass

        def build(mod, seed=7):
            torch.manual_seed(seed)
            cfg = mod.refc_v3_sized_config("small", hier=True)
            cfg.core.encoder = type(cfg.core.encoder)(
                in_channels=cfg.core.encoder.in_channels,
                image_size=256, image_width=1024,
                base_width=cfg.core.encoder.base_width,
                blocks=cfg.core.encoder.blocks)
            torch.manual_seed(seed)
            m = mod.RefCV3Model(cfg)
            return m.eval(), cfg

        a, cfg_a = build(old_mod)
        # restore the live package
        for k in list(sys.modules):
            if k == "tanitad" or k.startswith("tanitad."):
                del sys.modules[k]
        sys.path.remove(str(tmp))
        sys.modules.update(saved)
        from tanitad.refs import refc_v3 as new_v3
        b, cfg_b = build(new_v3)

        sa, sb = a.state_dict(), b.state_dict()
        res["state_dict_keys_equal"] = bool(sa.keys() == sb.keys())
        res["only_in_pre"] = sorted(set(sa) - set(sb))[:20]
        res["only_in_new"] = sorted(set(sb) - set(sa))[:20]
        res["n_params_pre"] = int(sum(p.numel() for p in a.parameters()))
        res["n_params_new"] = int(sum(p.numel() for p in b.parameters()))
        bad = [k for k in sa if k in sb and not torch.equal(sa[k], sb[k])]
        res["n_weight_mismatch"] = len(bad)
        res["weight_mismatch_sample"] = bad[:10]

        h, w = cfg_b.core.encoder.image_hw()
        res["forward_image_hw"] = [h, w]
        # ⚠️ B = 1: a 256x1024 forward through the in-repo CNN holds ~19 GB
        # of activations at B = 1 on this box (MEASURED in the forward arm),
        # and two models are alive here. B is not part of the claim.
        B = 1
        torch.manual_seed(11)
        frames = torch.rand(B, cfg_b.core.window,
                            cfg_b.core.encoder.in_channels, h, w)
        v0 = torch.tensor([4.0])
        with torch.no_grad():
            torch.manual_seed(99)
            oa = a(frames, v0=v0)
            torch.manual_seed(99)
            ob = b(frames, v0=v0)
        res["out_keys_equal"] = bool(set(oa) == set(ob))
        res["out_only_in_pre"] = sorted(set(oa) - set(ob))
        res["out_only_in_new"] = sorted(set(ob) - set(oa))
        deltas = {}
        for k in sorted(set(oa) & set(ob)):
            va, vb = oa[k], ob[k]
            if torch.is_tensor(va) and torch.is_tensor(vb):
                if va.shape != vb.shape:
                    deltas[k] = "shape %s vs %s" % (tuple(va.shape),
                                                    tuple(vb.shape))
                else:
                    deltas[k] = float((va.float() - vb.float()).abs().max())
        res["per_key_max_abs_delta"] = deltas
        num = [v for v in deltas.values() if isinstance(v, float)]
        res["bit_identical"] = bool(
            res["state_dict_keys_equal"] and res["n_weight_mismatch"] == 0
            and res["out_keys_equal"] and num and max(num) == 0.0)
        res["max_abs_delta_overall"] = max(num) if num else None
    except Exception as exc:              # a blocked identity IS a result
        import traceback
        res["blocked"] = "%s: %s" % (type(exc).__name__, exc)
        res["traceback"] = traceback.format_exc()[-3000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (raw / "identity.json").write_text(json.dumps(res, indent=1),
                                       encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# arm: forward — the real 139-clip pass                                       #
# --------------------------------------------------------------------------- #
#: the TERMS (not the diagnostics) `compute_losses_v3` can emit on this arm.
#: A term that is EXACTLY 0.0 or ABSENT is reported, never silently accepted.
LOSS_TERMS = {"traj", "cls", "lat", "lon", "law", "route", "sel_v3",
              "goal_tac", "lat_tac", "lon_tac", "u0", "gstr", "goal_str",
              "agent", "bev_aux", "tac_goal", "goal_point"}
#: emitted alongside them and NOT losses -- a zero here means nothing.
DIAGNOSTIC_KEYS = {"anchor_acc", "goal_gate", "goal2s_err_m",
                   "goal_score_absmean", "slot_valid_frac", "echo_ratio"}

PROBE_MODULES = [
    "core.encoder",              # THE TRUNK
    "core.decoder",              # the diffusion planner
    "core.tactical_trunk",
    "core.measurement",
    "core.law_head",
    "core.ego_hist",             # refcv6 §2b
    "core.strategic",            # must be DEAD under --no-strategic
    "core.route_head",           # must be DEAD (no route head, PI)
    "phi_tac",
    "tac_goal_head",
    "lat_head_tac",
    "lon_head_tac",
    "scorer",
    "tac_latent_proj",
    # ⭐ SPEC §4 / §5. `_grad_probe_row` emits `gp_<name>_found = 0.0` for a
    # module that does not exist, so on the arms where these are NOT built the
    # row says "absent", never "zero gradient" -- the two must not look alike.
    "tac_decoder_v6",
    "tac_behaviour_gate_v6",
    "core.agent_head",
]


def arm_forward(trainer, args, cfg, model, ds, pairs, raw: Path,
                log) -> dict:
    from tanitad.data.semantic_map_gt import sha12
    rows = []
    model.train()
    t_all = time.time()
    for n, (e_i, d_i) in enumerate(pairs):
        t0 = time.time()
        item = ds[d_i]
        batch = torch.utils.data.default_collate([item])
        clip_id = getattr(ds.eps[e_i], "clip_id", None) if hasattr(ds, "eps") \
            else None
        row = {"episode_index": int(e_i), "dataset_index": int(d_i)}
        if clip_id:
            row["clip_sha12"] = sha12(str(clip_id))
        shapes = {k: list(v.shape) for k, v in batch.items()
                  if torch.is_tensor(v)}
        row["batch_shapes"] = shapes
        model.zero_grad(set_to_none=True)
        losses = trainer.compute_losses_v3(model, batch, "cpu",
                                           mode="diffusion")
        scal = {k: float(v.detach()) for k, v in losses.items()
                if torch.is_tensor(v) and v.ndim == 0}
        row["losses"] = scal
        row["nonfinite"] = sorted(k for k, v in scal.items()
                                  if not np.isfinite(v))
        row["exactly_zero"] = sorted(k for k, v in scal.items() if v == 0.0)
        # the trainer's OWN total is `loss`; there is no `total` key. Summing
        # the dict would double-count every term already inside it.
        key = "loss" if "loss" in losses else "total"
        row["total_key"] = key
        row["total"] = scal.get(key)
        row["loss_terms_nonzero"] = sorted(
            k for k in scal if k in LOSS_TERMS and scal[k] != 0.0)
        row["loss_terms_zero"] = sorted(
            k for k in scal if k in LOSS_TERMS and scal[k] == 0.0)
        row["loss_terms_absent"] = sorted(LOSS_TERMS - set(scal))
        row["diagnostics"] = {k: scal[k] for k in scal if k in DIAGNOSTIC_KEYS}
        losses[key].backward()
        row["grad"] = trainer._grad_probe_row(model, PROBE_MODULES)
        row["t_s"] = time.time() - t0
        row["rss_gb"] = rss_gb()
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print("[forward] %d/%d ep=%d t=%.1fs rss=%.2fGB %s=%.4f"
              % (n + 1, len(pairs), e_i, row["t_s"], row["rss_gb"], key,
                 row["total"] if row["total"] is not None else float("nan")),
              flush=True)
        del losses, batch, item
        model.zero_grad(set_to_none=True)
        gc.collect()
    out = {"evidence_class": "MEASURED", "n_clips": len(rows),
           "wall_s": time.time() - t_all, "rows": rows}
    (raw / "forward.json").write_text(json.dumps(out, indent=1),
                                      encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# arm: perception — map head and 3-D box head on the REAL stride-16 map       #
# --------------------------------------------------------------------------- #
def arm_perception(trainer, args, cfg, model, eps, ds, pairs, raw: Path,
                   log, map_dir: Path, calib: Path, join3d: Path) -> dict:
    """⛔ THE HEADS ARE ASSEMBLED HERE, NOT BY THE TRAINER.

    `RefCV3Model` contains no BEVLift, no BEVMapBranch and no Box3D* module
    (asserted in `arm_assembly`), and `compute_losses_v3` has no map or box
    term. The heads therefore cannot be exercised through the trainer's
    assembly, and this arm is explicitly labelled HARNESS-ASSEMBLED. What it
    can still establish is whether the SEAM the trainer's model DOES emit
    (`out["fmap_s16"]`) carries the heads: shapes, finite non-trivial losses,
    and gradient back to the trunk.
    """
    from tanitad.data.agent_cuboid_gt import open_join3d, zh_for_frame
    from tanitad.data.bev_raster import GRID_DEFAULT
    from tanitad.data.perception_targets import MapGTStore, collate_map_targets
    from tanitad.data.physicalai import _load_chunk_extrinsics
    from tanitad.data.semantic_map_gt import sha12
    from tanitad.data.v2_dataset import stored_frame_of
    from tanitad.models.agent_slots import match_slots, targets_from_join
    from tanitad.models.bev_encoder import (BEVEncoderConfig, BEVMapBranch,
                                            map_soft_ce)
    from tanitad.models.bev_lift import BEVLift, build_lift_geometry
    from tanitad.models.box3d_head import (Box3DMemory, Box3DSlotDecoder,
                                           box3d_set_loss, zh_targets)
    from tanitad.models.trunk_shapes import TrunkSpec

    extr = _load_chunk_extrinsics(str(calib / "sensor_extrinsics.parquet"))
    store = MapGTStore(map_dir, max_open=2)
    have_map = {p.name[:-len(".sam3mapgt.npz")]
                for p in map_dir.glob("*.sam3mapgt.npz")}
    j3 = open_join3d(str(join3d)) if join3d and Path(join3d).exists() else None

    import glob
    p0 = sorted(glob.glob(os.path.join(cache_dir(args), "*.v2ep.pt")))[0]
    fr = stored_frame_of(torch.load(p0, map_location="cpu",
                                    weights_only=False))
    spec = TrunkSpec.from_timm(cfg.core.encoder.trunk_name.split(".")[0], fr)
    s16_c, s16_hw = spec.perception.channels, spec.perception.hw

    torch.manual_seed(0)
    d_bev = 96
    lift = BEVLift(d_in=s16_c, d_out=d_bev, feat_hw=s16_hw)
    branch = BEVMapBranch(BEVEncoderConfig(d_in=d_bev, d_out=d_bev))
    # ⛔ THE HEAD IS SIZED TO THE SPEC §6 PRE-REGISTERED PARAMETER BAND
    # (2 M, 4 M), NOT to what is convenient. A first attempt at d_model 128 /
    # depth 2 / 64 queries built 749,591 params and `Box3DSlotDecoder.__init__`
    # REFUSED it (MEASURED, logs/perception.log of the first attempt) -- "a
    # bigger head stops measuring what the LATENT carries", the AgentSlotDecoder
    # rule. The band is honoured by taking the module's OWN defaults
    # (d_model 256, depth 3, 8 heads, 100 queries -> 3,643,159 params);
    # `enforce_band=False` exists for shape tests and is deliberately NOT used,
    # because switching a guard off to get past it is moving the goalpost.
    d_model = 256
    mem = Box3DMemory(d_image=s16_c, d_bev=d_bev, d_model=d_model,
                      image_hw=s16_hw)
    box = Box3DSlotDecoder(d_memory=d_model, n_memory=mem.n_tokens,
                           d_model=d_model)
    heads = {"lift": lift, "map_branch": branch, "box_mem": mem,
             "box_dec": box}

    # ⛔ THE LAZY v2 PROVIDER CARRIES NO `clip_id` (attrs: actions, episode_id,
    # frames, maneuvers, poses). The identity comes from the cache's OWN
    # `_v2manifest.pt` sidecar, joined on `episode_uid`/`episode_id` rather
    # than on list ORDER, and every episode must resolve or the arm refuses.
    ep_clip = clip_ids_by_episode(cache_dir(args), eps)

    equiv_proof: dict = {"checked": None}
    rows, t_all = [], time.time()
    for n, (e_i, d_i) in enumerate(pairs):
        t0 = time.time()
        cid = ep_clip.get(e_i)
        row = {"episode_index": int(e_i), "dataset_index": int(d_i),
               "clip_sha12": sha12(str(cid)) if cid else None}
        if cid is None or sha12(str(cid)) not in have_map:
            row["skipped"] = "no SAM3 map for this clip"
            rows.append(row); log.write(json.dumps(row) + "\n"); log.flush()
            continue
        if cid not in extr:
            row["skipped"] = "no extrinsics for this clip"
            rows.append(row); log.write(json.dumps(row) + "\n"); log.flush()
            continue

        item = ds[d_i]
        batch = torch.utils.data.default_collate([item])
        frames = trainer.frames_to_device(batch["frames"], "cpu")
        model.zero_grad(set_to_none=True)
        for m in heads.values():
            m.zero_grad(set_to_none=True)
        # ⭐⭐ THE STRIDE-16 MAP IS COMPUTED EXACTLY AS `refc.py:3867-3869`
        # COMPUTES IT: one trunk pass over ALL `b*w` window positions, then
        # `[:, -1]`. Only the decoder/planner half of the model is skipped.
        #
        # ⛔ A CHEAPER VERSION OF THIS WAS TRIED AND THE GUARD CAUGHT IT.
        # `forward_features(frames[:, -1])` -- one window position, 1/W of the
        # activations -- looked exactly equivalent, and is NOT: the trunk runs
        # in TRAIN mode, so its BatchNorm normalises over the batch, and a
        # batch of 1 frame is not a batch of `b*w = 8`. MEASURED max |delta|
        # **72.7779** (logs of the first attempt). The equivalence proof below
        # is what turned a plausible 8x saving into a measured refusal.
        enc = model.core.encoder
        bb, ww = frames.shape[:2]
        s16_all = enc.forward_features(
            frames.reshape(bb * ww, *frames.shape[2:]))[0]
        fm = s16_all.reshape(bb, ww, *s16_all.shape[1:])[:, -1]
        if equiv_proof.get("checked") is None:
            with torch.no_grad():
                # the SAME one-shot ego-window handoff the trainer does
                # (`refc_v3_train.py:2439-2446`); without it the core refuses.
                if getattr(model.core, "ego_hist", None) is not None:
                    ph = batch["pose_hist"]
                    model.core.set_ego_window(ph, int(ph.shape[1]))
                full = model(frames, v0=batch.get("v0"))["fmap_s16"]
                same = bool(torch.equal(full.detach(), fm.detach()))
                dmax = float((full.detach() - fm.detach()).abs().max())
                shp = list(full.shape)
            equiv_proof.update(checked=True, bit_identical=same,
                               max_abs_delta=dmax, shape=shp)
            del full
            gc.collect()
            if not same:
                raise SystemExit(
                    "[e2e] the single-position stride-16 map is NOT the "
                    "model's own `fmap_s16` (max |delta| %g) -- refusing to "
                    "score a different tensor than the model uses" % dmax)
        row["fmap_s16_shape"] = list(fm.shape)
        row["fmap_s16_requires_grad"] = bool(fm.requires_grad)

        geo = build_lift_geometry(extr[cid], frame=fr, stride=16,
                                  grid=GRID_DEFAULT)
        bev = lift(fm, geo.grid.unsqueeze(0), geo.valid.unsqueeze(0))
        row["bev_shape"] = list(bev.shape)
        br = branch(bev)
        row["map_logits_shape"] = list(br["map_logits"].shape)
        row["bev_feats_shape"] = list(br["bev_feats"].shape)

        # ---- map loss on the REAL SAM3 target -----------------------------
        # ⛔ THE WINDOW'S *NOW* IS `t + W - 1`, NOT `t`. `refc.py:3869` reads
        # the stride-16 map at window position `-1`, and the trainer reads the
        # v7.2 tactical labels and the agent join at the SAME `t + w - 1`
        # (`refc_v3_train.py:2168-2172`). Using `t` would label the frame 0.7 s
        # before the one the trunk saw.
        t_row = int(ds.index[d_i][1])
        w_idx = t_row + int(cfg.core.window) - 1
        row["window_t"] = t_row
        row["window_now_row"] = w_idx
        tgt = collate_map_targets(store, [(cid, w_idx)],
                                  n_stack=int(cfg.core.encoder.in_channels
                                              // 3))
        ml = map_soft_ce(br["map_logits"], tgt.frac, tgt.seen)
        row["map"] = {"loss": float(ml["loss"].detach()),
                      "n_cells": int(ml["n_cells"]),
                      "label_grid": list(tgt.frac.shape),
                      "seen_frac": float(tgt.seen.float().mean())}
        model.zero_grad(set_to_none=True)
        ml["loss"].backward(retain_graph=True)
        row["map"]["grad"] = trainer._grad_probe_row(model, ["core.encoder"])

        # ---- box loss on the REAL cuboids, z/h from the REAL 3-D join -----
        row["box"] = {}
        t_s = float(tgt.t_img_us[0]) / 1e6
        raw_f = int(tgt.raw_frame[0])
        row["box"]["t_img_us"] = int(tgt.t_img_us[0])
        row["box"]["raw_frame"] = raw_f
        obst = Path(os.environ.get(
            "TANITAD_OBSTACLE_DIR",
            "C:/Users/Admin/tanitad-data/physicalai/labels/"
            "obstacle_offline_b1eval"))
        try:
            from tanitad.data.agent_cuboid_gt import (cuboids_at_time,
                                                      read_clip_cuboids,
                                                      track_ids_at_time)
            obs = read_clip_cuboids(str(obst / f"{cid}.parquet"))
            cub = cuboids_at_time(obs, t_s, clip_id=str(cid))   # [A, 8]
            tids = track_ids_at_time(obs, t_s)
            cls_by_track = {}
            if "label_class" in obs:
                for t, c in zip(np.asarray(obs["track_id"]).astype(str),
                                np.asarray(obs["label_class"]).astype(str)):
                    cls_by_track.setdefault(str(t), str(c))
            row["box"]["n_cuboids"] = int(cub.shape[0])
        except Exception as exc:
            cub, tids = None, []
            row["box"]["cuboid_error"] = "%s: %s" % (type(exc).__name__, exc)
        if cub is not None and cub.shape[0] > 0:
            t2 = targets_from_join(
                cub[:, :6].astype(np.float32),
                classes=[cls_by_track.get(t) for t in tids]
                if cls_by_track else None)
            # z/h from THE JOIN (the artifact the PI named) ...
            # ⛔⛔ TWO INDEX SPACES MEET HERE AND THEY DIFFER BY n_stack - 1.
            # `AgentJoin3D` indexes `int(rec["frame"])`
            # (`agent_cuboid_gt.py:353`) and `build_b1_agent_join.py:1` says
            # that key is the **RAW v2ep frame**; the trainer's own agent seam
            # passes the EPISODE index `t + w - 1`
            # (`refc_v3_train.py:2172`), which the same builder emits under the
            # DIFFERENT name `frame_idx`. MEASURED here (raw/join_frame_key.
            # json): sweeping the offset, mean |dcz| against the parquet's own
            # `center_z` is 0.0123 m at RAW+0 and 0.076-0.171 m at every other
            # offset in [-3, +3] -- so the RAW index is the key, by 6.2x.
            cz, h3, mask = zh_for_frame(str(cid), raw_f, tids, join3d=j3)
            row["box"]["n_zh_from_join"] = int(mask.sum())
            # ... cross-checked against the PARQUET's own cz/h, which is a
            # DIFFERENT FILE reached by a DIFFERENT index (time, not frame).
            # ⛔ Agreement is the evidence that the two index spaces line up;
            # a disagreement is reported, never averaged away.
            if int(mask.sum()) > 0:
                dz = np.abs(np.asarray(cz)[mask] - cub[mask, 6])
                dh = np.abs(np.asarray(h3)[mask] - cub[mask, 7])
                row["box"]["join_vs_parquet_max_abs_dz"] = float(dz.max())
                row["box"]["join_vs_parquet_max_abs_dh"] = float(dh.max())
            t3 = zh_targets(t2, cz, h3, mask=mask)
            row["box"]["n_z_targets"] = int(t3["zh_mask"].sum())
            memtok = mem(fm, br["bev_feats"])
            row["box"]["memory_tokens"] = list(memtok.shape)
            pred = box(memtok)
            mt = match_slots(pred, t3)
            bl = box3d_set_loss(pred, t3, match=mt)
            row["box"]["loss_total"] = float(bl["total"].detach())
            row["box"]["n"] = {k: int(v) for k, v in bl["n"].items()}
            row["box"]["terms"] = {k: float(v.detach()) for k, v in bl.items()
                                   if torch.is_tensor(v) and v.ndim == 0}
            # ⛔ THE CONTROL THAT MUST READ A KNOWN VALUE: same prediction,
            # z/h WITHHELD. `n["z"]` must be 0 and `loss_z` EXACTLY 0.0, or
            # the z/h terms are not actually consumed.
            wo = box3d_set_loss(pred, zh_targets(t2), match=mt)
            row["box"]["control_no_zh"] = {
                "n_z": int(wo["n"]["z"]), "loss_z": float(wo["loss_z"]),
                "total": float(wo["total"].detach()),
                "delta_total": float(bl["total"].detach()
                                     - wo["total"].detach())}
            model.zero_grad(set_to_none=True)
            bl["total"].backward()
            row["box"]["grad"] = trainer._grad_probe_row(model,
                                                         ["core.encoder"])
        row["t_s"] = time.time() - t0
        row["rss_gb"] = rss_gb()
        rows.append(row)
        log.write(json.dumps(row) + "\n"); log.flush()
        print("[perception] %d/%d ep=%d map=%.4f box=%s t=%.1fs"
              % (n + 1, len(pairs), e_i,
                 row.get("map", {}).get("loss", float("nan")),
                 row.get("box", {}).get("loss_total"), row["t_s"]), flush=True)

    out = {"evidence_class": "MEASURED",
           "heads_assembled_by": "THE HARNESS, not the trainer "
                                 "(RefCV3Model contains no map/box head)",
           "head_params": {k: int(sum(p.numel() for p in m.parameters()))
                           for k, m in heads.items()},
           "single_position_equivalence": equiv_proof,
           "n_clips": len(rows), "wall_s": time.time() - t_all, "rows": rows}
    (raw / "perception.json").write_text(json.dumps(out, indent=1),
                                         encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("assembly", "identity", "forward", "perception",
                             "tactical"))
    ap.add_argument("--cache", default="D:/Projects/TanitAD-artifacts/"
                                       "v2ep-eval139-256x1024cyl")
    ap.add_argument("--trunk-name", default="resnet101.a1_in1k")
    ap.add_argument("--clips", type=int, default=0, help="0 = all 139")
    ap.add_argument("--v7-labels", default=None)
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--agent-join", default=None,
                    help="the B1 EVAL 2-D obstacle join; the SPEC §4 tactical "
                         "decoder attends to AGENT SLOTS, and the trainer "
                         "refuses an unsupervised detector.")
    a = ap.parse_args(argv)

    pkg = Path(__file__).resolve().parents[1]
    raw = Path(a.raw) if a.raw else pkg / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer()
    out_dir = str(pkg / "_runout")
    argvt = trainer_argv(a.cache, trunk_name=a.trunk_name, out=out_dir,
                         v7_labels=a.v7_labels, anchors=a.anchors,
                         agents=("head" if a.arm == "tactical" else "off"),
                         agent_join=(a.agent_join if a.arm == "tactical"
                                     else None))
    _ap, args = build_args(trainer, argvt)

    if a.arm == "identity":
        r = arm_identity(raw)
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("per_key_max_abs_delta",
                                       "traceback")}, indent=1))
        return 0

    overrides = {}
    if a.arm == "tactical":
        # ⛔ SPEC §5's 4-value max-speed one-hot is NOT in this override, and
        # the reason is MEASURED, not assumed. Forcing
        # `max_speed_onehot_v6=True` builds the seam, and the forward then
        # REFUSES every batch: "this build is --max-speed-input but no
        # v_max_ms reached the forward while a nav token did"
        # (`refc_v3.py:1833`). The only supplier of `v_max_ms` is
        # `V3Dataset.enable_max_speed`, which the trainer calls ONLY under
        # `--max-speed-input` -- and that flag builds E16's CONTINUOUS 8-step
        # seam, which `RefCV3Model.__init__` refuses to run alongside the
        # refcv6 one-hot ("BOTH max-speed channels are on ... Pick one.").
        # ⇒ §5 has neither a CLI flag nor a data channel; it is unreachable
        # end to end. Evidence: raw/tactical_maxspeed_refusal.log.
        overrides = {"tac_decoder_v6": True}
    cfg, model, art, applied = build_model(trainer, args,
                                           cfg_overrides=overrides)
    if applied:
        print("[e2e] CONFIG OVERRIDES (the CLI has no flag for these): %s"
              % applied, flush=True)

    if a.arm == "assembly":
        r = arm_assembly(trainer, args, cfg, model, raw)
        print(json.dumps(r["spec_seam_cli_reachability"], indent=1))
        print("params", r["n_params_total"])
        return 0

    eps, ds, par, join = build_dataset(trainer, cfg, args)
    print('[e2e] v7 join: %s' % json.dumps(join), flush=True)
    n = a.clips or len(eps)
    pairs = one_window_per_episode(ds, n)
    print("[e2e] %d episodes, %d windows total, running %d clips"
          % (len(eps), len(ds), len(pairs)), flush=True)

    suffix = ("_" + a.tag) if a.tag else ""
    logp = raw.parent / "logs" / f"{a.arm}{suffix}.jsonl"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with open(logp, "w", encoding="utf-8") as log:
        if a.arm in ("forward", "tactical"):
            r = arm_forward(trainer, args, cfg, model, ds, pairs, raw, log)
            if a.arm == "tactical":
                (raw / "tactical.json").write_text(
                    json.dumps({"config_overrides": applied, **r}, indent=1),
                    encoding="utf-8")
                (raw / "forward.json").unlink(missing_ok=True)
        elif a.arm == "perception":
            r = arm_perception(
                trainer, args, cfg, model, eps, ds, pairs, raw, log,
                Path(os.environ.get("TANITAD_SAM3_MAP_DIR",
                                    "D:/Projects/TanitAD-artifacts/"
                                    "sam3-maps-eval")),
                Path(os.environ.get("TANITAD_CALIB_DIR",
                                    "D:/Projects/TanitAD-artifacts/"
                                    "hf-corpus-aug-20260915/stage/calibration")),
                Path(os.environ.get("TANITAD_AGENT_JOIN3D",
                                    "D:/Projects/TanitAD-artifacts/"
                                    "b1-agent-join-3d-20260917/"
                                    "b1eval_agents_3d.jsonl.xz")))
    print("[e2e] done -> %s" % raw, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
