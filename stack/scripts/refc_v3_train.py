"""REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm.

Design: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-08-18-refc-v3-design/REFC_V3_DESIGN.md`` (edge list E1..E12).
Experiment: ``PREREG_REFC_V3.md`` (E-V3DOM-1 — arms v3-H / v3-F, 3 seeds,
both outcomes committed). ⛔ Registered BEFORE any training step; do not launch
while Thor trains.

WHAT THIS SCRIPT IS AND IS NOT
============================================================================
A THIN COMPOSITION over ``refc_train.py``'s measured machinery — the same
fail-loud dataset stack (``RouteV21Dataset``: v2.1 labels, the measured label
choice), the same optimizer convention (Adam lr 1e-4 / warmup / cosine — the
arm's point), the same loss weights for every SHARED surface (imported, not
copied, so the two trainers cannot drift apart silently). What is NEW here and
only here:

  * ``V3Dataset`` — the PARITY-PRESERVING 6 s horizon. ``max_horizon`` stays 20
    for window ENUMERATION (``_contract.py:120`` re-selects windows otherwise —
    the canonical 406,099 must stay bit-identical), and steps 21..60 arrive by
    CLAMP + per-step validity mask. Tactical goal labels via
    ``refb_labels.goal_tac_targets`` (E4.1 layout, clamp+valid by contract).
  * ``compute_losses_v3`` — masked trajectory/assignment losses over the 8-slot
    horizon (exact-zero gradient at invalid slots, pinned by
    ``tests/test_refc_v3.py``), the factored tactical CE on BOTH decision
    surfaces (core aux — identical in both arms — and the z_tac heads on the H
    arm), the masked goal losses, and the survivor-set selection CE.
  * ``--arm hier|flat`` — the dominance pair. The config delta between the two
    is DERIVED and REFUSED if it is not exactly the registered lever set
    (C122's rule, enforced at build, not documented).
  * ``--preflight`` — builds everything, pins the delta, runs the C115
    freeze-history gate and the E11 mini intervention audit, runs one synthetic
    forward+loss, and EXITS. The launch line runs this first; a flag that
    parses and then dies mid-run is the class that cost ~3.1 GPU-days.
  * ``--synth-episodes N`` — CI-only synthetic corpus so the smoke test runs
    end-to-end with zero data. REFUSED together with ``--data-root``.
  * ``--u8-batches`` (2026-09-02, next-relaunch arm; default OFF) — the
    DataLoader ships ``frames``/``future_frames`` as uint8 and
    :func:`frames_to_device` applies the contract's ``/255`` on the device.
    In-flight batches — the ``/dev/shm`` cost that pinned refcv3's 50 GB
    cgroup at its cap: 6 workers x 3.30 GB fp32 = 19.8 GB, MEASURED — drop
    4x; the loss is bit-identical (``tests/test_refc_v3_u8_batches.py``).

Done-marker discipline: on completion this trainer writes ``summary.json`` with
``{"done": true}`` IN THE SAME RUN — the v5f supervisor resurrection (a
finished run relaunched for 2 days) is the reason.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The preflight's own banners carry ⛔/✅. On a cp1252 console (the Windows dev
# box, where a preflight is MEANT to be run before spending a GPU day) printing
# them raises UnicodeEncodeError — MEASURED: the run below died on the "✅ PASS"
# line AFTER every check had passed, so a PASSING preflight exited non-zero with
# a traceback. A gate whose success path crashes is a false negative.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already utf-8, or not a TTY
        pass

import refb_labels  # noqa: E402
from refc_train import (  # noqa: E402  — SHARED surfaces, imported not copied
    ANCHOR_CLS_WEIGHT, LAT_WEIGHT, LAW_AHEAD, LAW_WEIGHT, LON_WEIGHT,
    ROUTE_WEIGHT, TRAJ_WEIGHT, RouteV21Dataset, lan_dataset_class,
)
from refb_train import load_cached_episodes  # noqa: E402

from tanitad.data.lan import LanConfig as DataLanConfig  # noqa: E402

from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_tactical as tac  # noqa: E402
from tanitad.data import v7_labels as v7l
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.models import vocab_v7  # noqa: E402
from tanitad.refs import refb  # noqa: E402
from tanitad.refs import refc_agents as _refc_agents  # noqa: E402
from tanitad.models import kinematic as kin  # noqa: E402

# --- v3-only loss weights (everything shared is imported above) --------------
#: tactical goal regression (E8) — sized with the route/maneuver aux family;
#: the goal is a 12-number readout, not a second trajectory head.
GOAL_TAC_WEIGHT = 0.5
#: strategic goal (E3) — the ROUTE aux family's weight, deliberately: it is the
#: same kind of signal (leak-guarded LAN label) on a different head.
GOAL_STR_WEIGHT = ROUTE_WEIGHT
#: survivor-set selection CE (E9) — parallel to ANCHOR_CLS_WEIGHT: it is the
#: same "which candidate" question asked of the blended score.
SEL_V3_WEIGHT = 1.0

# ---- refcv5 weights (all default 0.0 -> the loss is bit-unchanged) --------
# ⛔ EVERY ONE OF THESE IS A DECLARED DECISION, NOT A DEFAULT, and every one
# defaults to 0.0 so that ADDING the seams to the code cannot change a run that
# does not ask for them. `--w-agent` / `--w-u0` are the switches.
#
# ⚠️ The units differ per term BY CONSTRUCTION (metres, nats, m/s^2), which is
# why they are named and stamped rather than folded into one number -- the same
# rule `SLOT_LOSS_W` and `V6LossWeights.w_select` carry.
AGENT_WEIGHT_DEFAULT = 0.0        # WP-6: the GT-supervised detection set loss
U0_WEIGHT_DEFAULT = 0.0           # WP-4: the x0 loss, in CONTROL space
#: the registered dominance lever set — build REFUSES any other delta (C122).
REGISTERED_DELTA_KEYS = {"hier", "core.graft_target_latent"}
#: ⭐ v4 adds its own lever set on top of the hier/flat pair. Both arms of a
#: v4 run carry the SAME ego pins, so the hier-vs-flat delta is UNCHANGED - the
#: v4 levers are checked against refcv3 separately (`REGISTERED_DELTA_KEYS_V4`,
#: `tests/test_refc_v4.py::test_registered_delta_is_pinned`), which keeps the
#: two questions ("what does the hierarchy cost?" and "what does v4 change?")
#: from being answered by one confounded diff.

MILESTONES = (5000, 15000, 20000, 30000)
MAX_H_EXT = max(v3.V3_HORIZONS)            # 60 — fetched by clamp, never enum

# --- --nav-from-v7: the nav SOURCE switch (E-ARCH-NAVSRC-1, 2026-09-02) ------
#: v7.2 nav token -> legacy ``refb.NAV_COMMANDS`` name, POSITION-pinned: the
#: model's nav one-hot / embedding rows are ordered by ``NAV_COMMANDS``
#: (refc.py:2024, refc_v3.py:164) while the v7.2 ids enumerate
#: ``vocab_v7.NAV_COMMAND_TOKENS``. This is the SAME mapping
#: ``refav1_loader._NAV_TOKEN_TO_LEGACY`` pins (equality asserted by
#: tests/test_refc_v3_nav_from_v7.py). It is a local copy ON PURPOSE: the
#: trainer must not acquire an import-time dependency on a loader that is
#: shipped to pods separately; :func:`assert_nav_token_alignment` re-checks
#: the position pin at dataset init and in the preflight, so drift is loud.
NAV_TOKEN_TO_LEGACY = {"NAV_FOLLOW_ROAD": "follow", "NAV_TURN_L": "left",
                       "NAV_TURN_R": "right"}
#: the config.json stamps — an arm must be identifiable from its own artifacts
NAV_FROM_V7_DERIVATION = ("v7.2 nav_command token (oracle, provenance "
                          "ego-future; allow_oracle_nav=True)")
NAV_V1_DERIVATION = "refb_labels.nav_command (v1, unchanged)"


def _pin_trainer_cfg(cfg: v3.RefCV3Config, args) -> v3.RefCV3Config:
    """The trainer's OWN pins on a freshly built config (both arms, all paths).

    * ``tac_vocab_version = "kin3"`` — THE documented contract this trainer
      never wired (found 2026-09-01): ``RefCV3Config``'s field comment says
      *"the kinematic trainer passes 'kin3' explicitly because its labels are
      the 3x3 kinematic classes"*, and the E6 comment says *"The trainer PINS
      its version"* — but nothing pinned it, so the v7.0 default (PI mandate
      2026-08-27) built 8-wide z_tac heads that ``compute_losses_v3``
      supervised with 3-class kinematic labels. That is EXACTLY the *"train
      silently WRONG classes"* case the width refusal names, invisible to it
      because the refusal checked only the CORE's head — and
      ``derive_man5_logprobs`` (a [B,3]x[B,3] contract) then read v7.0's
      LANE_CHANGE_L/R slots as turn_left/right into the H19 prior, an improper
      AND mis-labelled 5-way. MEASURED before the pin: preflight PASSED with
      ``tac_heads 14,364`` and ``lat_tac``/``lon_tac`` init-CE at ln(8).
      The v7.0 head is the go-forward space and NEEDS v7 labels
      (``SPEC_V7_LABEL_TRAINER_WIRING.md`` — label artifacts not yet
      deliverable), so a flag would be a dead switch today; pin, don't offer.
    * ``--image-hw`` — build the encoder at the CORPUS's geometry (e.g.
      256 640 for the B1 ``*.v2ep.pt`` cache). Param count is UNCHANGED (the
      trunk is fully convolutional; feat_dim = base_width*8 regardless), so
      the registered capacity ledger still holds; only compute changes.
      The delta gate is unaffected: both arms are pinned identically, and
      ``config_delta`` is derived from configs built through this same helper.
    """
    # ⭐ The vocabulary follows THE LABELS, not a hardcode. With --v7-labels
    # the released v7.2 tactical vocabulary (8x8) supervises the heads (PI
    # 2026-09-02, MANDATORY); otherwise the kinematic 3x3 derivation does, and
    # pinning it here is what stopped an 8-wide build training on 3 classes.
    cfg.tac_vocab_version = ("v7.0" if getattr(args, "v7_labels", None)
                             else "kin3")
    if args.image_hw:
        h, w = (int(args.image_hw[0]), int(args.image_hw[1]))
        enc = cfg.core.encoder
        cfg.core.encoder = refc.CNNEncoderConfig(
            in_channels=enc.in_channels, image_size=h,
            image_width=None if w == h else w,
            base_width=enc.base_width, blocks=enc.blocks)
    # ---- ⭐⭐ REF-C v4 pins (E11' + E14 + X15) ------------------------
    # Applied to BOTH arms identically, exactly like every other pin here, so
    # `config_delta` stays the derived instrument it is: the v4 lever set is
    # registered in REGISTERED_DELTA_KEYS_V4 and checked against the SAME
    # helper's output, never against a hand-written list of intentions.
    if getattr(args, "ego_state_inject", False):
        cfg.ego_state_inject = True
        cfg.core.ego_valid_channel = True     # precondition, not an option
    if getattr(args, "echo_base", False):
        cfg.echo_base = True
    if getattr(args, "ego_valid_channel", False):
        cfg.core.ego_valid_channel = True
    if getattr(args, "ego_dropout", None) is not None:
        cfg.core.ego_dropout = float(args.ego_dropout)
    # ---- ⛔ S2 REACH CLAMP, RE-DERIVED FOR THE HORIZON ACTUALLY PLANNED OVER
    # `refc.py:652` DERIVES `horizon_s` from `max(trajectory.horizons)`, so at
    # V3_HORIZONS the band is a*6.0, not a*2.0. The inherited `sel_accel_max =
    # 2.5` therefore opens it to +-15.0 m/s on a corpus whose v0 mean is
    # 5.24 m/s -- near-vacuous. MEASURED 2026-09-04 on the refcv4 anchor
    # vocabulary over 19,602 held-out eval / 635,331 train windows
    # (`refc_anchors_6s_b1train_128.pt.json`, `clamp6s.json`):
    #     a=2.5 kills 18.02 %   a=2.0 kills 26.23 %   a=1.5 kills 38.18 %
    # and the GT-deletion rate -- the criterion that actually binds, because a
    # band that removes the trajectory the ego FLEW is wrong, not conservative
    # -- is eval 0.000 % at a>=1.5 and train 0.007 % at a=2.0 vs 0.048 % at
    # a=1.5. dADE on the survivors is +0.00000 m at every a>=1.25, so the
    # "inert on ADE" property the 2 s statistic claimed does still hold here.
    # ⚠️ The 72.08 %/77.28 % figures in `refc_v3.py` are 2 s statistics and are
    # NOT reproduced at 6 s -- they must never be quoted for this band.
    # Applied to BOTH arms, so `config_delta` (hier vs flat) is unchanged.
    if getattr(args, "sel_accel_max", None) is not None:
        cfg.core.sel_accel_max = float(args.sel_accel_max)
    # ⭐ refcv4-b — the v0-CONDITIONED vocabulary. Applied to BOTH arms, so
    # `config_delta` (hier vs flat) is unchanged and C122 still passes.
    if getattr(args, "n_anchors", None):
        # ⚠️ The vocabulary SIZE is a property of the built vocabulary, not of
        # the model family: a v0-conditioned (accel, curvature) product grid is
        # odd x odd, so it can never be exactly 128. Applied to BOTH arms.
        cfg.core.anchors.n_anchors = int(args.n_anchors)
    if getattr(args, "anchor_v0_conditioned", False):
        cfg.core.anchors.v0_conditioned = True
        cfg.core.anchors.ref_speed_ms = float(
            getattr(args, "anchor_ref_speed", 10.0))
        # ⛔ `--anchor-control-units` now DEFAULTS TO None: the artifact is the
        # authority on what its own `controls` column MEANS, and the flag is
        # an explicit OVERRIDE for a legacy file that declares nothing.
        # `_read_anchor_artifact` resolves the two BEFORE this pin runs and
        # writes the resolved value back into `args`; a None here means no
        # artifact was read (a fixed-path build), where the value is inert.
        # MEASURED 2026-09-04: the live refcv4b anchors.pt declares nothing,
        # and the same bytes read as curvature give 396 g at 36 m/s (104/117
        # over mu = 0.7) against the true 0.31 g (0/117) -- tanitad.refs.
        # anchor_meta carries the incident and the rule.
        cfg.core.anchors.control_units = str(
            getattr(args, "anchor_control_units", None) or "kappa")
        # The artifact's own derivation constants travel with it: a file
        # rolled under one kappa cap / speed floor must be re-rolled under the
        # SAME ones by the decoder, or the checkpoint-visible `anchors` buffer
        # and the per-window bank are two different vocabularies under one
        # name. Adopted here (both arms, so the hier/flat delta is unchanged)
        # and stamped into config.json by `_anchor_stamp`.
        _am = getattr(args, "_anchor_artifact_meta", None) or {}
        if _am.get("kappa_cap") is not None:
            cfg.core.anchors.kappa_cap = float(_am["kappa_cap"])
        if _am.get("alat_v_floor") is not None:
            cfg.core.anchors.alat_v_floor_ms = float(_am["alat_v_floor"])
    _pin_refcv5_seams(cfg, args)
    return cfg


def _pin_refcv5_seams(cfg, args) -> None:
    """refcv5 WP-4 / WP-6 — install the sampler and the agent seam from argv.

    ⛔ EVERY seam here is OFF by default, and OFF means NOT CONSTRUCTED. A run
    that does not pass these flags builds a model that is bit-identical to
    refcv4b, RNG draw order included, so an existing checkpoint keeps loading
    strictly. That is checked by a test, not asserted here.
    """
    core = cfg.core
    # --- WP-4: the control-space DDIM sampler ---------------------------- #
    sampler = str(getattr(args, "sampler", "none"))
    core.decoder.sampler = sampler
    core.decoder.sampler_space = str(getattr(args, "sampler_space", "control"))
    core.decoder.sampler_train_t_max = int(getattr(args, "sampler_train_t_max",
                                                   50))
    core.decoder.sampler_infer_t = int(getattr(args, "sampler_infer_t", 8))
    core.decoder.sampler_steps = int(getattr(args, "sampler_steps", 2))
    core.decoder.sampler_groups = int(getattr(args, "sampler_groups", 1))
    if sampler == "ddim" and not core.anchors.v0_conditioned:
        raise SystemExit(
            "[v3] ⛔ --sampler ddim needs a v0-CONDITIONED vocabulary. The "
            "sampler's state IS the control sequence; a fixed-path bank "
            "carries anchor_controls of all zeros, so the anchored Gaussian "
            "would be centred on 'do nothing' — a plausible-looking WRONG "
            "experiment. Pass an --anchor-file built with controls.")
    if sampler == "ddim" and float(getattr(args, "w_u0", 0.0)) <= 0.0:
        raise SystemExit(
            "[v3] ⛔ --sampler ddim with --w-u0 0 trains the sampler with NO "
            "loss on its own prediction: `control_head` is zero-init, so it "
            "would stay at exactly zero and the arm would silently be the "
            "anchored Gaussian with no denoiser at all — and it would look "
            "like a trained sampler in every log. Pass --w-u0 > 0, or run "
            "--sampler none.")
    # --- WP-6: the agent seam -------------------------------------------- #
    if getattr(args, "agents", "off") != "off":
        acfg = _refc_agents.AgentSeamConfig(
            enable=True,
            oracle=(args.agents == "oracle"),
            oracle_sigma_range_m=float(getattr(args, "agent_sigma_range", 0.0)),
            oracle_miss_rate=float(getattr(args, "agent_miss_rate", 0.0)),
            queries=int(getattr(args, "agent_queries", 32)),
            w_project=float(getattr(args, "agent_w_project", 0.0)),
            w_ground=float(getattr(args, "agent_w_ground", 0.0)),
            presence_hard=bool(getattr(args, "agent_presence_hard", False)))
        core.agents = acfg
        core.decoder.cross_agent = True
        if args.agents == "head" and float(getattr(args, "w_agent", 0.0)) <= 0.0:
            raise SystemExit(
                "[v3] ⛔ --agents head with --w-agent 0 builds a detector that "
                "is never supervised. Its tokens would be noise, the "
                "zero-init gate would have no reason to open, and the arm "
                "would read as 'agent tokens do not help' — a REFUTATION "
                "manufactured by a missing loss. Pass --w-agent > 0, or use "
                "--agents oracle (which needs no detector loss).")


# ============================================================================
# Dataset — parity-preserving 6 s
# ============================================================================

def assert_nav_token_alignment() -> None:
    """⛔ The position pin: ``vocab_v7.NAV_COMMAND_TOKENS[i]`` must name
    ``refb.NAV_COMMANDS[i]`` through :data:`NAV_TOKEN_TO_LEGACY`, and the core
    must one-hot against the same tuple (``refc.NAV_COMMANDS``). Verified, not
    assumed — a silent re-ordering would land the token on the wrong nav row
    and train on a plausible wrong signal."""
    toks = tuple(vocab_v7.NAV_COMMAND_TOKENS)
    legacy = tuple(refb.NAV_COMMANDS)
    for i, tok in enumerate(toks):
        want = NAV_TOKEN_TO_LEGACY.get(tok)
        have = legacy[i] if i < len(legacy) else None
        if want is None or want != have:
            raise AssertionError(
                f"[v3 nav-from-v7] nav index alignment broken: "
                f"NAV_COMMAND_TOKENS[{i}]={tok!r} maps to {want!r} but "
                f"refb.NAV_COMMANDS[{i}]={have!r} — the v7.2 token would land "
                f"on the wrong nav row")
    if tuple(refc.NAV_COMMANDS) != legacy:
        raise AssertionError(
            f"[v3 nav-from-v7] refc.NAV_COMMANDS {tuple(refc.NAV_COMMANDS)} != "
            f"refb.NAV_COMMANDS {legacy} — the core one-hots nav_cmd against "
            f"its own tuple (refc.py:2024); the two must stay identical")
    if legacy.index("follow") != refb_labels.NAV_FOLLOW:
        raise AssertionError(
            f"[v3 nav-from-v7] 'follow' sits at {legacy.index('follow')} in "
            f"refb.NAV_COMMANDS but refb_labels.NAV_FOLLOW is "
            f"{refb_labels.NAV_FOLLOW} — the unlabeled default would not be "
            f"'follow'")


def _check_nav_from_v7_args(args) -> None:
    """Refuse AT START a --nav-from-v7 launch that would die or mislead later.

    * without ``--v7-labels`` there is no token to read — the flag would be a
      dead switch that LOOKS switched;
    * with an in-training eval (``--eval-cache`` + ``--eval-every``) but no
      ``--eval-labels``, train and eval would be fed DIFFERENT nav sources and
      every eval row would compare a v7.2-fed model against the v1 input —
      an eval that silently measures something else.
    ``getattr`` throughout: test rigs build partial Namespaces."""
    if not getattr(args, "nav_from_v7", False):
        return
    if not getattr(args, "v7_labels", None):
        raise SystemExit("[v3] ⛔ --nav-from-v7 needs --v7-labels: the nav "
                         "token is read from the v7.2 label records, and "
                         "without them the flag would be a dead switch that "
                         "looks switched.")
    if (getattr(args, "eval_cache", None) and getattr(args, "eval_every", 0)
            and not getattr(args, "eval_labels", None)):
        raise SystemExit("[v3] ⛔ --nav-from-v7 with an in-training eval needs "
                         "--eval-labels: the eval dataset must see the SAME "
                         "nav source as training, or every eval row compares "
                         "against the v1 derivation.")


class V3Dataset(RouteV21Dataset):
    #: clip-stable-id -> V7Label, or None for the kin3 path. Set by the trainer
    #: rather than passed through the ctor, because the base class owns the
    #: signature and widening it would touch every RouteV21 consumer.
    v7_by_sid: dict | None = None
    v7_dt: float = 0.1
    #: --nav-from-v7 (E-ARCH-NAVSRC-1): when True, ``nav_cmd``/``nav_valid``
    #: are OVERRIDDEN per window from the clip's v7.2 ``nav_command`` token
    #: (see :meth:`enable_nav_from_v7`). ⛔ Default False keeps the v1
    #: derivation (``refb_labels.nav_command`` via ``FailLoudWindowDataset``)
    #: byte-identical — the live run resumes through this class.
    nav_from_v7: bool = False
    v7_manifest = None
    _nav_by_sid: dict | None = None
    nav_from_v7_stats: dict | None = None

    """RouteV21Dataset + clamped/masked 6 s future + E4.1 tactical goals.

    ``max_horizon`` MUST stay at the caller's 20: enumeration parity. The
    extended future is fetched here per item from the episode's own poses."""

    def enable_nav_from_v7(self, manifest) -> dict:
        """Switch THIS dataset's ``nav_cmd`` INPUT to the clip's v7.2 token.

        Per-clip constant (the release's ``t0_constant`` semantics — token
        and args are window-independent, ``v7_labels.NavEmitter``), resolved
        once here so ``__getitem__`` is a lookup. Requires the v7.2 join
        (``v7_by_sid``) and the loaded labels' ``LabelManifest`` carrying
        ``allow_oracle_nav=True``: ``v7_labels.oracle_nav`` is the ONLY route
        to the token and it checks the MANIFEST, so the permission and the
        run's recorded config cannot disagree. Returns the per-clip stats
        that go into config.json — the token distribution over THIS
        dataset's clips and the count of clips without a record.
        """
        if self.v7_by_sid is None:
            raise ValueError(
                "[v3] --nav-from-v7 needs the v7.2 join (v7_by_sid is None) — "
                "the trainer sets it from --v7-labels / --eval-labels first")
        if not getattr(manifest, "allow_oracle_nav", False):
            raise v7l.OracleNavRefused(
                "[v3] --nav-from-v7 reads an ORACLE nav (provenance ego-future) "
                "— load_v7_labels(..., allow_oracle_nav=True) is required so "
                "the manifest carries the stamp")
        assert_nav_token_alignment()
        nav_by_sid: dict[int, int] = {}
        for sid, lab in self.v7_by_sid.items():
            nav = v7l.oracle_nav(lab, manifest) or {}      # ⭐ the gate
            tok = nav.get("token")
            if tok not in NAV_TOKEN_TO_LEGACY:
                raise ValueError(
                    f"[v3] ⛔ clip {lab.clip_id!r}: nav token {tok!r} has no "
                    f"legacy mapping (known: {sorted(NAV_TOKEN_TO_LEGACY)}) — "
                    f"vocabulary drift is a different experiment, refused")
            nav_by_sid[sid] = refb.NAV_COMMANDS.index(NAV_TOKEN_TO_LEGACY[tok])
        counts = {name: 0 for name in NAV_TOKEN_TO_LEGACY.values()}
        missing = 0
        for ep in self.episodes:
            idx = nav_by_sid.get(int(ep.episode_id))
            if idx is None:
                missing += 1
            else:
                counts[refb.NAV_COMMANDS[idx]] += 1
        n = len(self.episodes)
        if n and missing == n:
            raise ValueError(
                f"[v3] ⛔ --nav-from-v7 joined ZERO of {n} clips — wrong label "
                f"blob for this corpus (md5={manifest.md5})")
        self._nav_by_sid = nav_by_sid
        self.v7_manifest = manifest
        self.nav_from_v7 = True
        self.nav_from_v7_stats = {
            "n_clips": n, **counts, "missing": missing,
            "derivation": NAV_FROM_V7_DERIVATION,
            "unlabeled_default": "nav_cmd=0 ('follow') + nav_valid=False "
                                 "(refav1_loader convention)",
            "label_md5": manifest.md5,
            "allow_oracle_nav": bool(manifest.allow_oracle_nav)}
        # ASCII-only by design: this prints at dataset init.
        print(f"[v3] nav_from_v7: follow {counts['follow']} / left "
              f"{counts['left']} / right {counts['right']}, missing {missing} "
              f"(of {n} clips; md5={manifest.md5})", flush=True)
        return self.nav_from_v7_stats

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        T = ep.poses.shape[0]
        idx = torch.arange(t + w, t + w + MAX_H_EXT)
        item["future_poses_ext"] = ep.poses[idx.clamp(max=T - 1)]   # [60, 4]
        item["future_valid_ext"] = idx <= (T - 1)                   # [60] bool
        g, gv = refb_labels.goal_tac_targets(ep.poses, t + w - 1,
                                             v3.GOAL_TAU_STEPS)
        item["goal_tac"] = g                                        # [K, 4]
        item["goal_tac_valid"] = gv                                 # [K] bool
        # ---- v7.2 tactical labels (PI 2026-09-02: MANDATORY) --------------
        # ⭐ Joined on `stable_episode_id(clip_id)` — the v7.2 clip index names
        # it "the ONLY admissible join key", and `LazyV2Episode` carries the
        # stable id (not the clip_id string), so this is the join the artifacts
        # were built for rather than a string match we invented.
        # The window's NOW is the last OBSERVED frame: t + w - 1 at dt.
        if self.v7_by_sid is not None:
            lab = self.v7_by_sid.get(int(ep.episode_id))
            if lab is None:
                lat_v7 = lon_v7 = v7l.IGNORE_ID       # clip has no record
            else:
                lat_v7, lon_v7 = v7l.tactical_class_ids(
                    lab, (t + w - 1) * self.v7_dt)
            item["lat_v7"] = torch.tensor(lat_v7, dtype=torch.long)
            item["lon_v7"] = torch.tensor(lon_v7, dtype=torch.long)
        # ---- --nav-from-v7: the nav INPUT from the clip's v7.2 token -------
        # ⛔ OVERRIDES the v1 nav_cmd/nav_valid the base chain assigned
        # (refb_train.py:220-222, refb_labels.nav_command). `route_target` /
        # `route_valid` are the aux TARGET and stay untouched — RouteV21Dataset
        # documents that separation. A clip with NO record feeds NAV_FOLLOW +
        # nav_valid=False — refav1_loader's convention (index 0 is the
        # codebase's unlabeled default, refb.py:60) — counted at init and
        # stamped into config.json.
        if self.nav_from_v7:
            nav_idx = self._nav_by_sid.get(int(ep.episode_id))
            if nav_idx is None:
                item["nav_cmd"] = torch.tensor(refb_labels.NAV_FOLLOW,
                                               dtype=torch.long)
                item["nav_valid"] = torch.tensor(False)
            else:
                item["nav_cmd"] = torch.tensor(nav_idx, dtype=torch.long)
                item["nav_valid"] = torch.tensor(True)
        return item


def _synth_episodes(n: int, cfg: refc.RefCConfig, seed: int = 0,
                    min_frames: int = 40):
    """CI-only synthetic corpus (unicycle drives, tiny frames). NEVER a
    substitute for the parity cache — refused alongside --data-root.

    ``min_frames`` exists because the DEFAULT corpus is too SHORT to carry a
    LAN route and that is not obvious from reading it: T=40 at 2-8 m/s is
    ~8-32 m of total path, while the shortest LAN anchor sits at 20 m ARC-LENGTH
    *beyond* a leak guard of ~2 s x v + 5 m. MEASURED (D-LAN-COV, this package):
    every anchor of every window is masked on the default corpus, so a LAN
    preflight built on it would compute ``goal_str`` over an all-invalid label
    and report a green 0.0 — a loss that exists and cannot learn. The LAN arm
    therefore asks for a corpus long enough for the question to be answerable.
    """
    import types
    g = torch.Generator().manual_seed(seed)
    h, wpx = cfg.encoder.image_hw()
    eps = []
    for e in range(n):
        T = max(min_frames, 40) + 8 * e
        v = 2.0 + 6.0 * torch.rand((), generator=g)
        yr = (torch.rand((), generator=g) - 0.5) * 0.4
        yaw = torch.cumsum(torch.full((T,), float(yr)) * 0.1, dim=0)
        xy = torch.cumsum(
            torch.stack([v * torch.cos(yaw), v * torch.sin(yaw)], -1) * 0.1, 0)
        poses = torch.cat([xy, yaw[:, None],
                           torch.full((T, 1), float(v))], dim=-1)
        eps.append(types.SimpleNamespace(
            frames=(torch.rand(T, cfg.encoder.in_channels, h, wpx,
                               generator=g) * 255).to(torch.uint8),
            actions=torch.zeros(T, 2),
            poses=poses,
            episode_id=f"synth-{e:03d}"))
    return eps


# ============================================================================
# Losses — masked 6 s + hierarchy terms
# ============================================================================

def frames_to_device(x: torch.Tensor, device) -> torch.Tensor:
    """THE ONE frame-ingest point of this trainer (training AND held-out eval).

    ``--u8-batches`` ships frames as uint8; here they become the float32 [0,1]
    the encoder expects (``refc.py:994`` — a bare Conv2d stem, no
    normalisation anywhere inside the model) by ``x.float().div_(<0-dim
    tensor 255.0 on x.device>)``: the SAME map as
    ``tanitad.data._contract.to_float_frames`` (``x.float().div(255.0)``,
    ``_contract.py:56``), the division IN PLACE on the fresh fp32 tensor (one
    full-size fp32 allocation fewer on the device — 2.36 GB at the live
    future-frames shape) and by a TENSOR divisor so that CUDA performs a true
    division — bit-identical to the CPU contract over all 256 uint8 values on
    BOTH devices (pinned, with the scalar form's 1-ulp drift as the control,
    in ``tests/test_refc_v3_u8_batches.py``). A float input passes through
    untouched, exactly as ``to_float_frames`` passes it: with the flag OFF
    (the live run, the preflight, every existing dump) this is
    ``x.to(device)`` and nothing else.
    """
    x = x.to(device)
    if x.dtype == torch.uint8:
        # ⛔ A 0-dim DEVICE-TENSOR divisor, NOT the Python scalar 255.0.
        # MEASURED 2026-09-02 (torch 2.11.0+cu128, RTX 4060, exhaustive over
        # the 256 uint8 values): with a Python scalar, CUDA's div kernel takes
        # the multiply-by-reciprocal fast path and 126/256 values land 1 ulp
        # (5.96e-8) off the CPU contract — `to_float_frames` itself, run on
        # CUDA, would silently drift from HEAD's inputs. With a tensor divisor
        # the kernel is a true IEEE division: 0/256 off, on CPU and on CUDA.
        return x.float().div_(torch.full((), 255.0, device=x.device,
                                         dtype=torch.float32))
    return x


def compute_losses_v3(model: v3.RefCV3Model, batch: dict, device: str,
                      mode: str = "diffusion",
                      ablate_frames: bool = False) -> dict:
    cfg = model.cfg
    core = cfg.core
    # --u8-batches: uint8 in flight -> float32 [0,1] HERE, on the device, by
    # the contract's own /255 (frames_to_device). Float batches pass through.
    frames = frames_to_device(batch["frames"], device)
    fut_frames = frames_to_device(batch["future_frames"], device)
    # ⛔⛔ THE DELIBERATE-REGRESSION LEVER (rig only, `--ablate-frames`).
    # Replace the OBSERVED window with a scalar constant, so every sample sees
    # the SAME information-free image. The arm keeps every ego channel and
    # therefore can ONLY solve the task by echoing its own dynamics — it is an
    # echo BY CONSTRUCTION, and `TanitAD_ValidateAIDesign` §2 requires it: if
    # the anti-echo gate does not FAIL this arm, a PASS on the real arm means
    # nothing.
    # ⚠️ CONSTANT, NOT ZERO, and not noise. Zeros push the trunk's activation
    # statistics off-distribution, so a failure could be read as "the encoder
    # broke" rather than "the scene carries nothing"; fresh noise per step
    # would make the input a random variable the model can average out, which
    # is a DIFFERENT ablation. A constant removes exactly the information and
    # nothing else. ⛔ `fut_frames` is deliberately NOT ablated: it is the
    # LAW auxiliary's TARGET, and ablating a target changes the objective
    # rather than the input — a second lever hidden inside the first.
    if ablate_frames:
        frames = torch.full_like(frames, float(frames.mean()))
    fut_ext = batch["future_poses_ext"].to(device)          # [B, 60, 4]
    fut_valid = batch["future_valid_ext"].to(device)        # [B, 60] bool
    pose_last = batch["pose_last"].to(device)
    nav_cmd = batch["nav_cmd"].to(device)
    nav_valid = batch["nav_valid"].to(device)
    route_tgt = batch["route_target"].to(device)
    goal_tac = batch["goal_tac"].to(device)                 # [B, K, 4]
    goal_valid = batch["goal_tac_valid"].to(device)         # [B, K] bool
    lan = batch["lan"].to(device) if "lan" in batch else None
    v0 = pose_last[:, 3]
    b = frames.shape[0]
    steps = core.decoder.diffusion_steps if mode == "diffusion" else 0

    # ⭐⭐ REF-C v4 (E11'): the MEASURED ego state at t0. Derived from
    # `pose_last` + `actions[:, -1]`, both of which sit at the LAST OBSERVED
    # frame and both of which the window contract ALREADY returns - no new
    # dataset field and no cache rebuild. `future_poses_ext` is NOT read here
    # and must not be: that is the edge E11' pins interventionally.
    ego_state = None
    if getattr(cfg, "ego_state_inject", False):
        ego_state = v3.ego_state_from_batch(
            {"pose_last": pose_last, "actions": batch["actions"]},
            device=device)

    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan,
                ego_state=ego_state)

    # ---- trajectory target over the 8-slot 6 s horizon, masked -------------
    traj_tgt = refb_labels.waypoint_targets(pose_last, fut_ext,
                                            core.trajectory.horizons)
    slot_valid = torch.stack([fut_valid[:, h - 1]
                              for h in core.trajectory.horizons], dim=1)
    sv = slot_valid.to(traj_tgt.dtype)                      # [B, S]
    # ⛔ THE TARGET MUST BE MEASURED AGAINST THE BANK THAT WAS ACTUALLY
    # DECODED. With a v0-conditioned vocabulary `decoder.anchors` is the family
    # rolled at the REFERENCE speed, not this window's fan, so scoring `a_star`
    # against it would supervise the anchor classifier on a geometry the model
    # never emitted — silently, and with `anchor_acc` still reading plausibly.
    # `out["anchor_bank"]` is [B, N, S, 2] and is EXACTLY `x0`; for a fixed
    # vocabulary it is `anchors[None].expand(...)`, so this line is unchanged
    # arithmetic there (verified bit-identical, 2026-09-04).
    anchors = out["anchor_bank"].to(traj_tgt.dtype)          # [B, N, S, 2]
    dist = (((traj_tgt[:, None] - anchors) ** 2).sum(-1)
            * sv[:, None]).sum(-1)                          # [B, N] valid-only
    a_star = dist.argmin(dim=1)
    ar = torch.arange(b, device=device)
    loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    recon = out["anchor_traj"][ar, a_star]                  # [B, S, 2]
    denom = (sv.sum() * 2).clamp_min(1.0)
    loss_traj = (((recon - traj_tgt).abs().sum(-1)) * sv).sum() / denom

    # ---- factored tactical CE (2 s labels; the shared aux surface) ---------
    # ⭐ TWO SUPERVISION SOURCES, ONE REFUSAL EACH (PI 2026-09-02 made v7.2
    # MANDATORY for this launch). `lat_v7` present => the released v7.2
    # tactical vocabulary (8x8), joined per window by stable episode id;
    # absent => the runtime kin3 derivation (3x3) from poses. The head width
    # must match ITS OWN source, and a mismatch is REFUSED, never truncated --
    # a wider head fed narrower labels trains silently wrong classes, which is
    # exactly the defect found in this trainer on 2026-09-01.
    # ⛔⛔ TWO TACTICAL SURFACES, TWO VOCABULARIES — AND THE CORE'S IS FIXED.
    # MEASURED 2026-09-02: the core's heads are `nn.Linear(aux_hidden,
    # N_LAT_MAN)` with `N_LAT_MAN = tac.N_LAT` a MODULE-LEVEL CONSTANT
    # (refc.py:146,1742) — the REF-C core is STRUCTURALLY kin3, and
    # `tac_vocab_version` only sizes v3's OWN z_tac heads. Widening the core
    # would also have to redefine `derive_man5_logprobs` (a push-forward
    # DEFINED on the 3x3 kinematic vocabulary) and the H19 `lat_to_anchor`
    # prior — a change to the shared core, not a wiring job.
    # ⇒ v7.2 supervises the surface it can actually reach: the HIERARCHY's
    # tactical decision heads (z_tac). The core's legacy aux head keeps its
    # kinematic derivation. Both are trained; they are NOT the same label set,
    # and the launch record says so rather than implying one vocabulary.
    lat_k, lon_k = tac.window_factored_labels(pose_last, fut_ext[:, :20])
    if (out["lat_logits"].shape[-1] != tac.N_LAT
            or out["lon_logits"].shape[-1] != tac.N_LON):
        raise RuntimeError(
            f"CORE head/label mismatch: heads are "
            f"{out['lat_logits'].shape[-1]}x{out['lon_logits'].shape[-1]} vs "
            f"kinematic {tac.N_LAT}x{tac.N_LON}. The core surface is kin3 by "
            f"construction (refc.py N_LAT_MAN); only the v3 z_tac heads follow "
            f"tac_vocab_version.")
    loss_lat = F.cross_entropy(out["lat_logits"], lat_k)
    loss_lon = F.cross_entropy(out["lon_logits"], lon_k)
    # C-REFCV3-EVAL-PRIOR-LEAK (2026-09-02): the in-training eval reuses this function, and an
    # unconditional update EMA'd the HELD-OUT split's label marginals into core.lat/lon_log_prior
    # (buffers that shape decodes via logit_adjust). The v1 trainer gates the same call on
    # model.training (refc_train.py:521); so does this one now.
    if model.training:
        model.core.update_tactical_prior(lat_k, lon_k)
    # the z_tac surface's labels: v7.2 when supplied, else the same kin3
    use_v7 = "lat_v7" in batch
    if use_v7:
        lat_t, lon_t = batch["lat_v7"].to(device), batch["lon_v7"].to(device)
        n_lat_expect, n_lon_expect = (len(v7l.HEADS["tac_lat"]),
                                      len(v7l.HEADS["tac_lon"]))
        src = f"v7.2 ({n_lat_expect}x{n_lon_expect})"
    else:
        lat_t, lon_t = lat_k, lon_k
        n_lat_expect = n_lon_expect = tac.N_LAT
        src = f"kin3 ({tac.N_LAT}x{tac.N_LON})"
    # ⛔ -100 marks a window this clip's single record does not describe. CE
    # ignores those rows; an ALL-ignored batch is NaN, not zero, so it is
    # SKIPPED (the same guard refav1 needed -- measured, not assumed).
    lat_ok = int((lat_t != v7l.IGNORE_ID).sum())
    lon_ok = int((lon_t != v7l.IGNORE_ID).sum())
    loss_lat_tac = torch.zeros((), device=device)
    loss_lon_tac = torch.zeros((), device=device)
    if cfg.hier:                       # the H arm's z_tac decision surface
        # the SAME refusal for the v3 heads — the core-only check above let an
        # 8-wide v7.0 build train silently against kin3 labels (2026-09-01).
        if (out["lat_logits_tac"].shape[-1] != n_lat_expect
                or out["lon_logits_tac"].shape[-1] != n_lon_expect):
            raise RuntimeError(
                f"z_tac head/label vocabulary mismatch: lat "
                f"{out['lat_logits_tac'].shape[-1]} / lon "
                f"{out['lon_logits_tac'].shape[-1]} vs {src} -- the model was "
                f"built with tac_vocab_version={model.tac_vocab_version!r}. "
                f"The z_tac heads and the core heads MUST share one source; "
                f"this check mirrors the core one because a v3 build once "
                f"passed the core check and trained 8-wide z_tac heads "
                f"against 3-class labels (2026-09-01).")
        loss_lat_tac = (F.cross_entropy(out["lat_logits_tac"], lat_t) if lat_ok
                        else torch.zeros((), device=device))
        loss_lon_tac = (F.cross_entropy(out["lon_logits_tac"], lon_t) if lon_ok
                        else torch.zeros((), device=device))

    # ---- route aux (v2.1 masked CE — refc_train's convention) --------------
    # ⛔⛔ FIXED 2026-09-02, MEASURED ON THE A40: this masked by `nav_valid`,
    # which is a DIFFERENT VALIDITY FIELD than the target it guards. v2.1's
    # `route_target` is ROUTE_UNKNOWN (= 3, deliberately OUTSIDE the 3-class CE
    # range) exactly where `route_valid` is False — and a window can carry a
    # VALID nav command with an UNKNOWN route. Those rows survived the
    # nav_valid mask and hit the CE, producing
    #   `nll_loss_forward_reduce_cuda_kernel_2d: t >= 0 && t < n_classes`
    # — an async device-side assert that surfaces at the NEXT CUDA call
    # (conv2d), so the traceback blames the wrong line entirely.
    # ⚠️ WHY IT SURVIVED EVERY SMOKE: it is window-frequency dependent. The
    # 2-step batch-2 smoke and the batch-4 run never drew an offending window;
    # batch 20 hit it inside 14 steps. A 30 k launch would have died hours in.
    # ⇒ mask on `route_valid` when the v2.1 labeler supplies it (mirroring
    # `refc_train.py`'s shared path verbatim), and keep the FAIL-LOUD check —
    # an UNKNOWN surviving the mask is a labeler contract violation, never
    # something to clamp to `straight`.
    if "route_valid" in batch:
        mask = batch["route_valid"].to(device)
        if bool(mask.any()):
            tgt_v = route_tgt[mask]
            n_route = out["route_logits"].shape[-1]
            if int(tgt_v.max()) >= n_route:
                raise ValueError(
                    f"ROUTE_UNKNOWN survived the valid mask (max target "
                    f"{int(tgt_v.max())} >= n_route {n_route}) — the v2.1 "
                    f"contract is route<3 <=> valid=True")
            loss_route = F.cross_entropy(out["route_logits"][mask], tgt_v)
        else:                        # no judgeable window in this batch
            loss_route = torch.zeros((), device=device)
    else:
        mask = nav_valid
        loss_route = (F.cross_entropy(out["route_logits"][mask],
                                      route_tgt[mask])
                      if bool(mask.any())
                      else torch.zeros((), device=device))

    # ---- LAW aux (0.5 s pooled-latent target, no_grad encode) --------------
    with torch.no_grad():
        law_tgt = model.core.encode_pooled(fut_frames[:, LAW_AHEAD - 1]
                                           .to(device))
    loss_law = F.mse_loss(out["law_pred"], law_tgt)

    loss = (TRAJ_WEIGHT * loss_traj + ANCHOR_CLS_WEIGHT * loss_cls
            + LAW_WEIGHT * loss_law + ROUTE_WEIGHT * loss_route
            + (LAT_WEIGHT / 2.0) * (loss_lat + loss_lat_tac)
            + (LON_WEIGHT / 2.0) * (loss_lon + loss_lon_tac))
    # ⛔ THE /2 IS THE FIX, NOT A TYPO. `refc_train.py:83-91` states in writing
    # that the TOTAL tactical aux pressure is held at EXACTLY MANEUVER_WEIGHT
    # (0.10) so an arm differs from the base run in STRUCTURE, not in loss
    # budget. This trainer supervises TWO tactical surfaces -- the core's pooled
    # kin3 heads (`loss_lat`/`loss_lon`) and the v7.2 z_tac heads
    # (`loss_lat_tac`/`loss_lon_tac`) -- so the unhalved form spends
    # 0.05*2 + 0.05*2 = 0.20, i.e. DOUBLE the documented budget, which refcv3
    # shipped with. Halving restores 0.025*4 = 0.10 = MANEUVER_WEIGHT.

    extra: dict = {}
    # ⭐ WHICH label set trained the tactical decision surface, and how many
    # rows survived the in-band mask, IN EVERY LOG ROW. A run that silently
    # fell back to kin3 (missing --v7-labels, or a corpus whose clips have no
    # record) would otherwise look identical to a v7.2 run in the log.
    extra["tac_label_v7"] = 1.0 if use_v7 else 0.0
    extra["tac_label_rows"] = float(lat_ok)
    if cfg.hier:
        # E8 — masked goal regression (each level trains by its OWN label).
        loss_goal = v3.masked_goal_loss(out["g_tac"], goal_tac, goal_valid)
        loss = loss + GOAL_TAC_WEIGHT * loss_goal
        extra["goal_tac"] = loss_goal
        with torch.no_grad():
            e2 = (out["g_tac"][:, model._tau_slot_2s(), :2]
                  - goal_tac[:, model._tau_slot_2s(), :2]).norm(dim=-1)
            m2 = goal_valid[:, model._tau_slot_2s()]
            extra["goal2s_err_m"] = (e2[m2].mean() if bool(m2.any())
                                     else torch.zeros((), device=device))
        # E3 — strategic goal off the leak-guarded LAN label (train-only, E12).
        if lan is not None:
            bearing_t, dist_t, valid_t = refc.RefCModel.goal_targets(
                lan, core.lan.k)
            loss_gstr = v3.strategic_goal_loss(out["g_str"], bearing_t,
                                               dist_t, valid_t)
            loss = loss + GOAL_STR_WEIGHT * loss_gstr
            extra["goal_str"] = loss_gstr
        # E9 — survivor-set CE on the blended score (the gates' ONLY gradient:
        # argmax has none; the goal head is firewalled by detach).
        fan_err = (((out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1)
                    * sv[:, None]).sum(-1)
                   / sv.sum(-1, keepdim=True).clamp_min(1.0))     # [B, N]
        loss_sel = v3.selection_ce(out["sel_score_v3"], fan_err.detach(),
                                   out.get("reach_keep"))
        loss = loss + SEL_V3_WEIGHT * loss_sel
        extra["sel_v3"] = loss_sel
        extra["goal_gate"] = model.goal_gate.detach()
        # ⭐ CAVEAT-B (PI 2026-09-02): the gate value alone cannot distinguish
        # "has not opened YET" from "will never open". Emit the score scale it
        # multiplies and its own gradient, so the 30 k read answers the
        # question with data instead of a 14-step glance.
        if "goal_score_absmean" in out:
            extra["goal_score_absmean"] = out["goal_score_absmean"]
        if model.goal_gate.grad is not None:
            extra["goal_gate_grad"] = model.goal_gate.grad.detach().abs()
        # E13 telemetry: nav actually reached the tactical/strategic states on
        # this batch. A conditioning edge that silently no-ops (nav_cmd=None
        # everywhere) is the advertised-but-inert defect; this makes it visible.
        if "nav_injected" in out:
            extra["nav_injected"] = float(bool(out["nav_injected"]))
        # ⭐⭐ THE ANTI-ECHO READOUT (E11'/E14). The model COMPUTES these
        # every forward and the trainer was DROPPING them, so the design's own
        # promise — *"is it echoing? is read off the LOG rather than inferred
        # from an eval three days later"* — was not kept by the code that
        # writes the log. Same class as the `tac_label_v7` drop noted below:
        # a diagnostic that does not survive to the log is not a diagnostic.
        #   echo_ratio = |g_tac_delta| / |echo_base| = how much goal VISION
        #   contributes beyond the ego self-extrapolation. A run that sits at
        #   ~0 has learned nothing the ego state did not already say.
        # ⚠️ It is a RATIO OF MAGNITUDES, not a verdict: a large delta can
        # still be wrong, and a small one can still be right if `ha0_ext` is
        # already near-optimal. The verdict is `tanitad.eval.echo_gate`; this
        # is the live tell that says whether to keep paying for the run.
        for _k in ("echo_ratio", "echo_base_absmean", "g_tac_delta_absmean"):
            if _k in out:
                extra[_k] = out[_k]
        # E11' telemetry, the mirror of `nav_injected`: did the ego block
        # actually reach the goal path on this batch? An edge that silently
        # no-ops reads as "the hypothesis failed".
        if "ego_injected" in out:
            extra["ego_injected"] = float(bool(out["ego_injected"]))
        if "ego_keep_frac" in out:
            extra["ego_keep_frac"] = out["ego_keep_frac"]
    # ⭐ H-EGO-LIT-4 diagnostics: the model's OWN 2 s speed vs the GT 2 s
    # speed, split by the withholding draw. `withheld_speed_mae` is the A0
    # log the pre-registration reads the `pred` warm-up step N from ("the
    # step at which the withheld-row speed MAE first drops below 2.5 m/s").
    # A key is absent (not NaN) on a batch with no rows in that regime.
    if "bank_speed_pred" in out and "ego_keep" in out:
        _slot = model._tau_slot_2s()
        _gv = goal_valid[:, _slot]
        _err = (out["bank_speed_pred"].float()
                - goal_tac[:, _slot, 3].float()).abs()
        _keep = out["ego_keep"].float() > 0.5
        for _tag, _m in (("withheld", (~_keep) & _gv), ("kept", _keep & _gv)):
            _n = int(_m.sum())
            extra[f"{_tag}_rows"] = float(_n)
            if _n:
                extra[f"{_tag}_speed_mae"] = _err[_m].mean().detach()
        extra["bank_speed_pred_mean"] = out["bank_speed_pred"].float().mean()

    # ---- refcv5 WP-4: the x0 loss, in CONTROL space ----------------------
    # ⭐ This is the ONE term that is not expressible in metres, which is why
    # `u0_hat` has to leave the decoder at all. The target is the GT path's own
    # control sequence through the programme's inverse map -- so the sampler is
    # supervised in the space it samples in, not in the space it is read out in.
    w_u0 = float(getattr(model, "_w_u0", 0.0))
    if w_u0 > 0.0 and "u0_hat" in out:
        u_gt = kin.unicycle_controls_from_path_varstep(
            traj_tgt, kin.slot_dts(core.trajectory.horizons))       # [B, S, 2]
        if core.anchors.control_units == "alat":
            # ⛔ THE TARGET MUST BE IN THE VOCABULARY'S UNITS. The inverse map
            # returns CURVATURE; the bank's controls are LATERAL ACCELERATION
            # when `control_units == "alat"`. Comparing the two directly is the
            # units error `anchor_meta.py` exists to prevent -- a_lat = v^2 k
            # differs from k by a factor of ~1300 at 36 m/s, and BOTH tables
            # look plausible.
            v_ref = v0.clamp_min(core.anchors.alat_v_floor_ms) ** 2
            u_gt = torch.stack([u_gt[..., 0], u_gt[..., 1] * v_ref[:, None]],
                               dim=-1)
        a_idx = a_star[:, None, None, None].expand(b, 1, u_gt.shape[1], 2)
        u_sel = out["u0_hat"].gather(1, a_idx).squeeze(1)           # [B, S, 2]
        norm = torch.tensor(core.decoder.control_norm, device=device,
                            dtype=u_sel.dtype)
        loss_u0 = (((u_sel - u_gt) / norm).abs().sum(-1) * sv).sum() / denom
        loss = loss + w_u0 * loss_u0
        extra["u0"] = loss_u0

    # ---- refcv5 WP-6: the GT-supervised detection set loss ----------------
    # ⛔ `obstacle.offline` is a TRAIN-TIME LABEL. It enters HERE, in the loss,
    # and nowhere in the forward -- which is the vision-only rule enforced by
    # where the tensor is read, not by a comment.
    w_agent = float(getattr(model, "_w_agent", 0.0))
    if w_agent > 0.0 and "agent_box" not in batch:
        # ⛔⛔ REFUSE, DO NOT SKIP. MEASURED on the tiny rig 2026-09-05: with
        # `--agents head --w-agent 1.0` on a dataset that carries no
        # `obstacle.offline` join, this branch was a silent `and "agent_box" in
        # batch` no-op -- the run trained, converged, wrote a checkpoint, and
        # stamped `w_agent: 1.0` in config.json while THE DETECTOR WAS NEVER
        # SUPERVISED. The head would have been shaped only by the planner loss
        # through its token gate, and the arm would have read as "the learned
        # agent head does not help".
        # ⭐ This is the SAME failure the --w-agent 0 guard in
        # `_pin_refcv5_seams` refuses, one level down: there the loss weight is
        # missing, here the LABELS are. A guard on the flag alone is not
        # enough, because the flag was set correctly.
        raise SystemExit(
            "[v3] ⛔ --w-agent > 0 but the batch carries no `agent_box`: this "
            "dataset has no obstacle.offline join wired, so the detection loss "
            "would be SILENTLY SKIPPED and the run would stamp w_agent > 0 "
            "while training no detector. Wire the join into the dataset (see "
            "`train_p8_occupancy.JoinFileReader` + "
            "`agent_slots.targets_from_join`), or run --agents oracle / "
            "--w-agent 0.")
    if w_agent > 0.0 and "agent_slots" in out and "agent_box" in batch:
        tgt_ag = {"box": batch["agent_box"].to(device),
                  "yaw": batch["agent_yaw"].to(device),
                  "cls": batch["agent_cls"].to(device),
                  "valid": batch["agent_valid"].to(device),
                  "occ": batch.get("agent_occ",
                                   torch.full_like(batch["agent_yaw"], -1.0)
                                   ).to(device),
                  "rates": batch.get(
                      "agent_rates",
                      torch.zeros(*batch["agent_yaw"].shape, 3)).to(device),
                  "rates_mask": batch.get(
                      "agent_rates_mask",
                      torch.zeros_like(batch["agent_valid"])).to(device)}
        ag = _refc_agents.agent_losses(out["agent_slots"], tgt_ag,
                                       core.agents,
                                       cam=getattr(model, "_rig_camera", None))
        loss = loss + w_agent * ag["total"]
        for k_ag in ("presence", "cls", "centre", "size", "yaw", "project",
                     "ground"):
            if f"loss_{k_ag}" in ag:
                extra[f"agent_{k_ag}"] = ag[f"loss_{k_ag}"]
        # ⭐ n PER TERM, in the log row. A detection number without its n is
        # inadmissible, and `n_dropped` is how a too-small `n_queries` becomes
        # VISIBLE instead of silently flattering the head on crowded frames.
        extra["agent_n_target"] = float(ag["n"]["target"])
        extra["agent_n_matched"] = float(ag["n"]["matched"])
        extra["agent_n_dropped"] = float(ag["n"]["dropped"])

    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
            "route": loss_route, "lat": loss_lat, "lon": loss_lon,
            "lat_tac": loss_lat_tac, "lon_tac": loss_lon_tac,
            "anchor_acc": (out["anchor_logits"].argmax(1) == a_star)
            .float().mean(), "slot_valid_frac": sv.mean(), **extra}


# ============================================================================
# Preflight — everything that must be true BEFORE a GPU day
# ============================================================================

def _lan_arm_preflight(cfg, args) -> int:
    """Exercise the LAN → ``goal_str`` pathway BEFORE a GPU day.

    ⚠️ THIS ARM EXISTS BECAUSE THE PREFLIGHT COULD NOT SEE THE THING IT WAS
    BEING CITED FOR. MEASURED (D-LAN-PF, this package): ``preflight()`` builds
    a bare ``V3Dataset``, which emits no ``lan`` key, so ``compute_losses_v3``
    skips ``loss_gstr`` BY DESIGN and ``goal_str`` never appears in the loss
    dict at all — with or without ``--goal-str``. The quantity people read
    instead, ``route``, is the v2.1 NAV-derived CE masked by ``nav_valid``; it
    is bit-identical with and without every LAN flag and can never testify
    about LAN. Same class as C9/C13/C14: an instrument structurally unable to
    report the answer it is quoted for.

    Three checks, in the order that makes a failure diagnosable:
      1. the dataset actually emits ``lan`` at the pinned width;
      2. the LABEL is live — ``any_valid_frac`` > 0. A ``goal_str`` computed
         over an all-invalid label is 0.0 and looks like a pass;
      3. ``goal_str`` is PRESENT, FINITE and NON-ZERO.
    Then a DELIBERATE-REGRESSION control (programme 6.2): with the leak guard
    pushed to +inf the label must go dead and ``goal_str`` must collapse to
    0.0. A check that cannot fail is not evidence, so the control runs every
    time and its own failure fails the preflight.
    """
    k = len(args.lan_arclengths)
    lan_cfg = DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                            min_lead_m=args.lan_min_lead_m)
    cfg.core.lan = refc.LanConfig(k=k)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)

    # Long enough that the leak guard is a GUARD and not a wall — see
    # _synth_episodes.__doc__ for the arithmetic that sets this floor.
    # ⚠️ Derived from the ARC-LENGTHS ONLY, deliberately NOT from min_lead_m:
    # the regression control below cranks min_lead_m to +inf, and a corpus that
    # grew with it would change TWO things at once — the control would compare
    # a different corpus rather than isolate the guard. The cap keeps a
    # pathological --lan-arclengths from asking for an unallocatable corpus
    # (at 4000 frames the reachable path is ~800 m, so anchors beyond that
    # still read dead, which is the correct verdict and not an evasion).
    need_m = max(args.lan_arclengths) + 50.0
    min_frames = min(int(need_m / (2.0 * 0.1)) + 4 * cfg.core.window, 4000)
    eps = _synth_episodes(2, cfg.core, seed=0, min_frames=min_frames)

    def _batch(lc):
        ds = lan_dataset_class(V3Dataset)(
            eps, window=cfg.core.window, max_horizon=20,
            channels=cfg.core.encoder.in_channels, lan_cfg=lc)
        return (torch.utils.data.default_collate([ds[0], ds[1]]),
                ds.lan_stats(n=512, seed=0))

    batch, stats = _batch(lan_cfg)
    if "lan" not in batch:
        print("[v3-preflight] ⛔ FAIL: lan dataset emitted no `lan` key")
        return 6
    if tuple(batch["lan"].shape[1:]) != (k * 4,):
        print(f"[v3-preflight] ⛔ FAIL: lan width {tuple(batch['lan'].shape)} "
              f"!= pinned {k * 4}")
        return 6
    print(f"[v3-preflight] lan label coverage: {json.dumps(stats)}")
    if stats["any_valid_frac"] <= 0.0:
        print("[v3-preflight] ⛔ FAIL: every LAN anchor is masked — `goal_str` "
              "would be 0.0 over an ALL-INVALID label and read as a pass. The "
              "strategic goal head would train on nothing (C9 class).")
        return 6

    losses = compute_losses_v3(model, batch, "cpu", mode="diffusion")
    gs = losses.get("goal_str")
    if gs is None:
        print("[v3-preflight] ⛔ FAIL: `goal_str` absent from the loss dict — "
              "the LAN pathway did not fire even though `lan` was present.")
        return 6
    gs = float(gs.detach())
    if not (gs == gs and abs(gs) != float("inf")):
        print(f"[v3-preflight] ⛔ FAIL: goal_str non-finite ({gs})")
        return 6
    if gs == 0.0:
        print("[v3-preflight] ⛔ FAIL: goal_str is exactly 0.0 on a LIVE label "
              "— the strategic head carries no gradient.")
        return 6

    # ---- deliberate-regression control: the check must be ABLE to fail ------
    dead_cfg = DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                             min_lead_m=1e9)
    dead_batch, dead_stats = _batch(dead_cfg)
    dead_gs = float(compute_losses_v3(model, dead_batch, "cpu",
                                      mode="diffusion")["goal_str"].detach())
    if dead_stats["any_valid_frac"] != 0.0 or dead_gs != 0.0:
        print(f"[v3-preflight] ⛔ FAIL: the regression control did NOT go dead "
              f"(valid_frac={dead_stats['any_valid_frac']}, "
              f"goal_str={dead_gs}) — this preflight cannot detect a dead LAN "
              f"label, so its PASS means nothing.")
        return 6
    print(f"[v3-preflight] lan arm OK: goal_str={gs:.4f} live / "
          f"{dead_gs:.4f} under the +inf-guard control "
          f"(control able to fail: True)")
    return 0



def _read_anchor_artifact(args):
    """Read ``--anchors`` through :mod:`tanitad.refs.anchor_meta` and RESOLVE
    its control units against the explicit ``--anchor-control-units`` override.

    Returns ``None`` without ``--anchors``. Otherwise it (i) refuses -- before
    any data or GPU work -- a ``controls``-carrying file that declares no
    ``control_units`` unless the override is given (the LIVE refcv4b file is
    exactly that case and keeps loading through ``--anchor-control-units
    alat``), (ii) writes the resolved units back into
    ``args.anchor_control_units`` so ``_pin_trainer_cfg`` sees ONE value, and
    (iii) stashes the file's declared derivation constants in
    ``args._anchor_artifact_meta`` for the pin to adopt.
    """
    if not getattr(args, "anchors", None):
        return None
    from tanitad.refs import anchor_meta
    try:
        art = anchor_meta.read_anchor_artifact(
            args.anchors,
            cli_control_units=getattr(args, "anchor_control_units", None))
    except anchor_meta.AnchorUnitsError as e:
        raise SystemExit(f"[v3] ⛔ anchors: {e}")
    if art.controls is not None:
        args.anchor_control_units = art.control_units
    args._anchor_artifact_meta = dict(art.declared)
    print(f"[v3] anchors: artifact {anchor_meta.describe(art)}", flush=True)
    return art


def _check_anchor_artifact_against_cfg(art, cfg, args) -> None:
    """Refuse a file whose DECLARED horizon / dt / reference speed differ from
    what the decoder built from ``cfg`` will use. Undeclared fields (a legacy
    file) never fire; a declared one that differs is a second vocabulary
    wearing the first one's name, so it is refused rather than reconciled."""
    if art is None:
        return
    from tanitad.refs import anchor_meta
    hz = tuple(cfg.core.trajectory.horizons)
    bad = anchor_meta.mismatches(
        art, horizon_s=max(hz) * 0.1, dt=0.1,
        ref_speed_ms=(float(cfg.core.anchors.ref_speed_ms)
                      if art.controls is not None else None),
        kappa_cap=(float(cfg.core.anchors.kappa_cap)
                   if art.controls is not None else None),
        alat_v_floor=(float(cfg.core.anchors.alat_v_floor_ms)
                      if art.controls is not None else None))
    if bad:
        raise SystemExit(
            "[v3] ⛔ anchors: the artifact was built for different constants "
            "than this decoder would re-roll it under -- " + "; ".join(bad)
            + ". Rebuild the vocabulary for this trainer, or launch with the "
              "matching flags (--anchor-ref-speed for ref_speed_ms; kappa_cap "
              "and alat_v_floor are adopted from the file).")


def _seam_stamp(cfg, args) -> dict:
    """The hierarchy-seam booleans, serialised so a finished run can rebuild
    its own model config from its own record.

    MEASURED 2026-09-04 (`…/2026-09-04-refcv4b-seam-state/SEAM_STATE.md`):
    `hierarchy`, `graft_maneuver`, `factored_maneuver`, `graft_prior_center`,
    `lan_enable`, `goal_str` were absent from config.json at EVERY nesting
    level; the live arm was reconstructible only by knowing that refc_v3.py
    forces them. A run record that cannot rebuild its own model config is not
    a run record. `lan_enable` is the LAN LABEL pathway (`--goal-str` or
    `--graft-lan` builds it); `graft_lan` is the corridor as a MODEL INPUT,
    refused by E12 for every registered arm and stamped separately so the two
    can never be confused.
    """
    core = cfg.core
    return {
        "hier": bool(cfg.hier),
        "hierarchy": bool(core.hierarchy),
        "graft_maneuver": bool(core.graft_maneuver),
        "factored_maneuver": bool(core.factored_maneuver),
        "graft_prior_center": bool(core.graft_prior_center),
        "graft_target_latent": bool(core.graft_target_latent),
        "grounded_selector": bool(core.grounded_selector),
        "graft_imagination": bool(core.graft_imagination),
        "tactical_speed_input": bool(core.tactical_speed_input),
        "lan_enable": bool(getattr(args, "goal_str", False)
                           or getattr(args, "graft_lan", False)),
        "graft_lan": bool(core.graft_lan or getattr(args, "graft_lan", False)),
        "goal_str": bool(getattr(args, "goal_str", False)),
        # ⭐ H-EGO-LIT-4 (2026-09-05): the withheld-row bank policy. A run that
        # does not stamp it cannot say which of the four arms it was.
        "withheld_bank": str(getattr(args, "withheld_bank", "fixed")),
        "withheld_bank_warmup": int(getattr(args, "withheld_bank_warmup", 0)),
        "withheld_speed_max_ms": float(getattr(args, "withheld_speed_max",
                                               35.0)),
        # ⭐ refcv5 (2026-09-05). A run that does not stamp these cannot say
        # which arm it was -- the exact failure `SEAM_STATE.md` MEASURED for
        # six pre-existing seams, where the live arm was reconstructible only
        # by knowing what refc_v3.py forces.
        "agents": (core.agents.as_dict()
                   if getattr(core, "agents", None) is not None else None),
        "cross_agent": bool(getattr(core.decoder, "cross_agent", False)),
        "sampler": str(getattr(core.decoder, "sampler", "none")),
        "sampler_space": str(getattr(core.decoder, "sampler_space", "control")),
        "sampler_infer_t": int(getattr(core.decoder, "sampler_infer_t", 8)),
        "sampler_steps": int(getattr(core.decoder, "sampler_steps", 2)),
        "sampler_groups": int(getattr(core.decoder, "sampler_groups", 1)),
        "control_norm": list(getattr(core.decoder, "control_norm", (4.0, 3.0))),
        "w_agent": float(getattr(args, "w_agent", AGENT_WEIGHT_DEFAULT)),
        "w_u0": float(getattr(args, "w_u0", U0_WEIGHT_DEFAULT)),
    }


def _apply_withheld_bank(model, args, eps, device) -> dict:
    """Install ``--withheld-bank`` on the decoder and return its config stamp.

    ``pred``/``random``/``none`` need a v0-CONDITIONED vocabulary (there is
    nothing to roll otherwise) and are refused without one; ``pred`` also
    needs the hierarchy (the goal head is the speed source). ``random`` draws
    from the TRAINING marginal of v0 — every frame's speed over the training
    episodes, sub-sampled to <= 200k values — which is stamped by n / mean /
    std so the control's distribution is on record. The mode itself goes live
    in the train loop (warm-up), never here.
    """
    mode = str(getattr(args, "withheld_bank", "fixed"))
    dec = model.core.decoder
    stamp = {"mode": mode,
             "warmup_steps": int(getattr(args, "withheld_bank_warmup", 0)),
             "speed_max_ms": float(getattr(args, "withheld_speed_max", 35.0)),
             "random_pool": None}
    if mode != "fixed" and not getattr(dec, "anchor_v0_cond", False):
        raise SystemExit(
            f"[v3] ⛔ --withheld-bank {mode} needs a v0-CONDITIONED vocabulary "
            f"(--anchor-v0-conditioned with a controls-carrying --anchors); a "
            f"fixed-path vocabulary has no per-row roll to redirect.")
    if mode == "pred" and not model.cfg.hier:
        raise SystemExit("[v3] ⛔ --withheld-bank pred needs --arm hier: the "
                         "goal head (g_tac) is the speed source.")
    dec.anchor_withheld_speed_max = float(stamp["speed_max_ms"])
    if mode == "random":
        vals = []
        for ep in eps:
            po = getattr(ep, "poses", None)
            if po is None:
                continue
            vals.append(torch.as_tensor(po)[:, 3].to(torch.float32)
                        .reshape(-1).clone())
        pool = torch.cat(vals) if vals else torch.zeros(0)
        pool = pool[torch.isfinite(pool)].clamp_min(0.0)
        if pool.numel() == 0:
            raise SystemExit("[v3] ⛔ --withheld-bank random: no v0 values in "
                             "the training episodes to draw from")
        if pool.numel() > 200_000:
            g = torch.Generator().manual_seed(int(args.seed))
            pool = pool[torch.randperm(pool.numel(), generator=g)[:200_000]]
        dec.anchor_random_speed_pool = pool.to(device)
        stamp["random_pool"] = {
            "n": int(pool.numel()), "mean_ms": round(float(pool.mean()), 4),
            "std_ms": round(float(pool.std()), 4),
            "p10_p50_p90_ms": [round(float(q), 3) for q in
                               torch.quantile(pool, torch.tensor(
                                   [0.1, 0.5, 0.9]))],
            "source": "poses[:, 3] over every frame of the TRAINING episodes "
                      "(the marginal the withheld row's bank is drawn from; "
                      "independent of the row)"}
    # the decoder starts on the FIXED roll; the train loop flips it at
    # `warmup_steps` and logs `withheld_bank_active`.
    dec.anchor_withheld_bank = "fixed" if stamp["warmup_steps"] > 0 else mode
    print(f"[v3] withheld bank: mode={mode} warmup={stamp['warmup_steps']} "
          f"speed_max={stamp['speed_max_ms']} m/s "
          f"pool={stamp['random_pool']}", flush=True)
    return stamp


def _anchor_stamp(path, anchors, controls=None, units="kappa",
                  art=None) -> dict:
    """Record the anchor vocabulary BY CONTENT, not by the presence of a path.

    A run that was launched without ``--anchors`` silently carries
    ``refc.default_anchors`` (the SYNTHETIC bootstrap set), and that is
    indistinguishable from a data-driven run in every artifact refcv3 left
    behind.  The sha256 of the tensor actually installed in the decoder is what
    settles it, so `source` is derived from the tensor, never from the flag.
    """
    import hashlib
    a = anchors.detach().to("cpu", torch.float32).contiguous()
    h = hashlib.sha256(a.numpy().tobytes()).hexdigest()
    stamp = {"path": str(path) if path else None,
             "shape": list(a.shape),
             "sha256_installed": h,
             "source": "file" if path else "refc.default_anchors (SYNTHETIC)"}
    if controls is not None:
        c = controls.detach().to("cpu", torch.float32).contiguous()
        stamp["controls_shape"] = list(c.shape)
        stamp["controls_sha256_installed"] = hashlib.sha256(
            c.numpy().tobytes()).hexdigest()
        stamp["v0_conditioned"] = True
        stamp["control_units"] = units
        stamp["straight_ahead_control_present"] = bool(
            ((c[:, 0] == 0) & (c[:, 1] == 0)).any())
    else:
        stamp["v0_conditioned"] = False
    # ⭐ where the units CAME FROM, and what the file itself declares. For the
    # live refcv4b file this reads `cli-override-legacy-file` with every
    # declared field None -- the record says the operator supplied the units.
    if art is not None:
        stamp["control_units_source"] = art.control_units_source
        stamp["artifact_schema"] = art.meta.get("schema")
        stamp["artifact_declared"] = dict(art.declared)
        stamp["artifact_provenance"] = art.meta.get("provenance")
    if path:
        try:
            stamp["file_sha256"] = hashlib.sha256(
                open(path, "rb").read()).hexdigest()
        except OSError as e:
            stamp["file_sha256"] = f"unreadable: {e}"
    return stamp


def preflight(args) -> int:
    _check_nav_from_v7_args(args)          # no-op unless --nav-from-v7
    art = _read_anchor_artifact(args)      # None without --anchors
    print("[v3-preflight] building both arms + pinning the delta …")
    cfg_h = v3.refc_v3_sized_config(args.size, hier=True)
    cfg_f = v3.refc_v3_sized_config(args.size, hier=False)
    if args.smoke:
        cfg_h, cfg_f = (v3.refc_v3_smoke_config(True),
                        v3.refc_v3_smoke_config(False))
    cfg_h, cfg_f = _pin_trainer_cfg(cfg_h, args), _pin_trainer_cfg(cfg_f, args)
    delta = v3.config_delta(cfg_h, cfg_f)
    if set(delta) != REGISTERED_DELTA_KEYS:
        print(f"[v3-preflight] ⛔ FAIL: config delta {sorted(delta)} != "
              f"registered {sorted(REGISTERED_DELTA_KEYS)} — amend "
              f"PREREG_REFC_V3.md BEFORE launch.")
        return 2
    cfg = cfg_h if args.arm == "hier" else cfg_f
    _check_anchor_artifact_against_cfg(art, cfg, args)
    model = v3.RefCV3Model(cfg)
    bd = v3.param_breakdown_v3(model)
    print(f"[v3-preflight] arm={args.arm} params={bd}")
    torch.manual_seed(0)
    h, wpx = cfg.core.encoder.image_hw()
    frames = torch.rand(2, cfg.core.window, cfg.core.encoder.in_channels,
                        h, wpx)
    if cfg.hier:
        rep = v3.freeze_history_report(model, frames,
                                       v0=torch.tensor([3.0, 7.0]))
        print(f"[v3-preflight] freeze-history gate: {rep}")
        if not rep["pass"]:
            print("[v3-preflight] ⛔ FAIL: the C115 gate — the H arm is "
                  "flat-in-disguise; the experiment would be VOID (OUTCOME V).")
            return 3
        model.eval()
        if getattr(cfg, "ego_state_inject", False):
            # ================= REF-C v4 (E11') =================
            # ⭐⭐ THE REFUSED EDGE MOVED, SO THE PREFLIGHT MOVES WITH IT.
            # Under v3 this block pinned `v0 -> goal` as a REFUSED edge. E11'
            # admits it, so the v4 checks are: ego MUST reach the goal path
            # (else the arm is v3 wearing a v4 config, and a null result would
            # be about the wiring), frames MUST still reach the vision-borne
            # nodes, and the E14 base MUST be exactly the extrapolation.
            e_lo = v3.ego_state_at_t0(
                torch.zeros(2, cfg.core.window, 4),
                torch.zeros(2, cfg.core.window, 2))
            e_hi = torch.tensor([[9.0, 2.5, 0.45, 0.05, 1.0],
                                 [4.0, -1.5, -0.20, -0.05, 1.0]])
            with torch.no_grad():
                a = model(frames, v0=torch.tensor([0.0, 0.0]), ego_state=e_lo)
                bq = model(frames, v0=torch.tensor([9.0, 4.0]), ego_state=e_hi)
                c = model(torch.rand_like(frames),
                          v0=torch.tensor([0.0, 0.0]), ego_state=e_lo)
            # (1) ego REACHES the goal path. ⚠️ `z_tac`/`g_str` are
            # zero-init on the ego edge, so at step 0 only the E14 base can
            # carry it - which is exactly what `echo_base` is for. Without
            # `echo_base` this check is deferred to the trained gate, and the
            # preflight SAYS SO rather than passing silently.
            if getattr(cfg, "echo_base", False):
                if torch.equal(a["g_tac"], bq["g_tac"]):
                    print("[v3-preflight] ⛔ FAIL: ego does not move "
                          "g_tac (E11' DEAD - this is v3 wearing a v4 config)")
                    return 4
                want = v3.kinematic_goal_extrapolation(
                    e_hi[:, 0], e_hi[:, 1], e_hi[:, 3], cfg.goal_tau_seconds)
                if not torch.equal(bq["g_tac"], want):
                    print("[v3-preflight] ⛔ FAIL: E14 base is not the "
                          "kinematic extrapolation at init - the residual head "
                          "is not zero-init, so the arm does NOT start at "
                          "ha0_ext and every echo-ratio reading is off-scale")
                    return 4
            else:
                print("[v3-preflight] ⚠ ego->goal is zero-init and "
                      "UNVERIFIABLE at step 0 without --echo-base; the live "
                      "check is `echo_ratio` in the training log")
            # (2) frames STILL move the vision-borne nodes (C109: a probe that
            #     cannot fire proves nothing). ⛔ `g_tac` is deliberately
            #     EXCLUDED under `echo_base`: its residual head is zero-init, so
            #     it is vision-dead AT INIT BY DESIGN. That is the one node whose
            #     liveness the preflight genuinely cannot assert - and weakening
            #     the check to cover it would be exactly the vacuous pass C109
            #     names, so it is deferred to the logged `echo_ratio` instead.
            for k in ("g_str", "z_tac"):
                if torch.equal(a[k], c[k]):
                    print(f"[v3-preflight] ⛔ FAIL: frames do not move {k}"
                          f" - probe UNPOWERED, not clean (C109)")
                    return 4
            # (3) the ego block is WITHHELD honestly: keep=0 zeroes the base.
            e_off = e_hi.clone(); e_off[:, 4] = 0.0
            with torch.no_grad():
                d = model(frames, v0=torch.tensor([9.0, 4.0]),
                          ego_state=e_off)
            if getattr(cfg, "echo_base", False) and                     not torch.equal(d["echo_base"],
                                    torch.zeros_like(d["echo_base"])):
                print("[v3-preflight] ⛔ FAIL: keep=0 did not zero the "
                      "E14 base - a withheld ego block is leaking (X15)")
                return 4
            print("[v3-preflight] E11' OK: ego reaches the goal path, frames "
                  "still move g_str/z_tac, keep=0 withholds cleanly")
        else:
            # ================= REF-C v3 (E11, unchanged) =================
            with torch.no_grad():
                a = model(frames, v0=torch.tensor([0.0, 0.0]))
                bq = model(frames, v0=torch.tensor([9.0, 4.0]))
                c = model(torch.rand_like(frames), v0=torch.tensor([0.0, 0.0]))
            for k in ("g_str", "g_tac", "z_tac"):
                if not torch.equal(a[k], bq[k]):
                    print(f"[v3-preflight] ⛔ FAIL: v0 leaked into {k} "
                          f"(E11)")
                    return 4
                if torch.equal(a[k], c[k]):
                    print(f"[v3-preflight] ⛔ FAIL: frames do not move {k}"
                          f" - probe UNPOWERED, not clean (C109)")
                    return 4
        model.train()
    # one synthetic end-to-end loss step (fail here, not on the pod). The lan
    # LABEL pathway is exercised on the train path, not here — the hier loss
    # skips loss_gstr when `lan` is absent BY DESIGN, and the preflight's job
    # is the shared+goal surfaces.
    eps = _synth_episodes(2, cfg.core, seed=0)
    ds = V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                   channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    # ⛔ THE PREFLIGHT MUST EXERCISE THE LAUNCH CONFIG, NOT A NEIGHBOUR OF IT.
    # MEASURED 2026-09-02: with --v7-labels the model is built with 8-wide
    # z_tac heads, but the synthetic preflight corpus carries no v7 records, so
    # the loss fell back to kin3 and the vocabulary refusal fired — the gate
    # working correctly on a config nobody would launch. A preflight that can
    # only pass in a shape the real run never takes is not a preflight, so the
    # synthetic batch is given v7-SHAPED labels (one in-band row and one
    # ignored row, which also exercises the -100 path the real corpus produces
    # on ~76 % of windows).
    if getattr(args, "v7_labels", None):
        n_lat, n_lon = len(v7l.HEADS["tac_lat"]), len(v7l.HEADS["tac_lon"])
        batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
        batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID],
                                       dtype=torch.long)
        assert n_lat >= 1 and n_lon >= 1
    # --nav-from-v7: the launch feeds v7.2 tokens, so the preflight batch gets
    # v7-SHAPED nav too (one turn each way, both valid) and the position pin
    # runs HERE — a broken pin must fail the preflight, not the pod launch.
    if getattr(args, "nav_from_v7", False):
        try:
            assert_nav_token_alignment()
        except AssertionError as e:
            print(f"[v3-preflight] ⛔ FAIL: {e}")
            return 7
        batch["nav_cmd"] = torch.tensor(
            [refb.NAV_COMMANDS.index("left"), refb.NAV_COMMANDS.index("right")],
            dtype=torch.long)
        batch["nav_valid"] = torch.tensor([True, True])
    losses = compute_losses_v3(model, batch, "cpu", mode="diffusion")
    bad = [k for k, t in losses.items()
           if torch.is_tensor(t) and not bool(t.isfinite().all())]
    if bad:
        print(f"[v3-preflight] ⛔ FAIL: non-finite losses {bad}")
        return 5
    print(f"[v3-preflight] loss step OK: "
          f"{ {k: round(float(t.detach()), 4) for k, t in losses.items() if torch.is_tensor(t) and t.ndim == 0} }")
    # ⛔ `route` above is the v2.1 NAV-derived CE, NOT the LAN route. It is
    # masked by nav_valid and is bit-identical with and without every LAN flag
    # (MEASURED, D-LAN-PF) — reading it as evidence about LAN is a category
    # error the LAN arm below exists to prevent.
    if (args.goal_str or args.graft_lan) and cfg.hier:
        rc = _lan_arm_preflight(cfg, args)
        if rc:
            return rc
    print("[v3-preflight] ✅ PASS")
    return 0


# ============================================================================
# Train loop
# ============================================================================

def train(args) -> dict:
    # --nav-from-v7 (E-ARCH-NAVSRC-1): refuse a mis-specified switch BEFORE any
    # data or GPU work; a no-op with the flag off.
    _check_nav_from_v7_args(args)
    nav_on = bool(getattr(args, "nav_from_v7", False))
    # ⭐ THE ANCHOR ARTIFACT IS READ FIRST: a units-less legacy file is refused
    # before any data or GPU work, the resolved units and the file's own
    # derivation constants reach `_pin_trainer_cfg` below, and the decoder is
    # BUILT under the constants the file was rolled with.
    art = _read_anchor_artifact(args)
    # MEASURED on the A40 pod 2026-09-02: the FIRST real launch died within
    # seconds -- `DataLoader worker killed by signal: Bus error ... out of
    # shared memory`. B1 payloads are ~34 MB/clip and torch's DEFAULT tensor
    # sharing passes them through /dev/shm, which is 24 GB in this container;
    # 4 workers x an LRU of decoded clips exhausts it almost immediately.
    # `file_system` sharing moves those handles off /dev/shm at no cost here
    # (the payloads are already files on disk) -- the same fix the throughput
    # sweep needed. MUST be set before any worker forks, hence the very top.
    if getattr(args, "workers", 0) > 0:
        import torch.multiprocessing as _tmp
        _tmp.set_sharing_strategy("file_system")
        print(f"[v3] tensor sharing = file_system (workers={args.workers}; "
              f"/dev/shm 24 GB vs ~34 MB/clip payloads)", flush=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    cfg = _pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    # ⛔ The delta is derived AT THE SAME SIZE both arms run at. Deriving it at a
    # different rung would compare a config pair neither arm uses. The pair is
    # pinned through the same helper as the built arm, so the recorded delta is
    # the delta of the configs that actually train.
    delta = v3.config_delta(
        _pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=True), args),
        _pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=False), args))
    if set(delta) != REGISTERED_DELTA_KEYS:
        raise SystemExit(f"[v3] ⛔ config delta {sorted(delta)} != registered "
                         f"{sorted(REGISTERED_DELTA_KEYS)} — amend the prereg "
                         f"first (C122).")
    _check_anchor_artifact_against_cfg(art, cfg, args)
    if args.graft_lan or args.goal_str:
        cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))

    model = v3.RefCV3Model(cfg).to(device)
    # refcv5: the two loss weights travel ON the model, because
    # `compute_losses_v3(model, batch, device, ...)` has no `args`. PLAIN
    # attributes, never buffers -- they must not enter state_dict and change
    # checkpoint compatibility (the `_seam_conf` discipline).
    model._w_agent = float(getattr(args, "w_agent", AGENT_WEIGHT_DEFAULT))
    model._w_u0 = float(getattr(args, "w_u0", U0_WEIGHT_DEFAULT))
    model._rig_camera = None
    # ⛔ THE ANCHOR VOCABULARY IS LOAD-BEARING AND ITS ABSENCE WAS SILENT.
    # refcv3 trained without `--anchors` and nobody noticed for the whole run,
    # because this branch printed nothing and recorded nothing. MEASURED cost:
    # the synthetic fallback's oracle-in-vocabulary ADE is 0.9433 m, WORSE than a
    # single straight line (0.6780 m), against the data-driven set's 0.4369 m — so
    # the model's CEILING was below the trivial floor before training began.
    # The base trainer has always been loud here (`refc_train.py:918-923`); this
    # one was not. Both branches now speak, and the default is a WARNING, not a
    # default. See RETRACTION_LOG and `.../2026-09-04-refcv3-smoothness/`.
    if args.anchors:
        assert art is not None                       # read at the top of train()
        anc = art.anchors.to(device)
        _ctrl = None if art.controls is None else art.controls.to(device)
        # ⛔ BOTH DIRECTIONS ARE REFUSED, LOUDLY. A v0-conditioned build given a
        # controls-free file would silently fall back to fixed paths — the exact
        # vocabulary this arm exists to replace — and a fixed build given a
        # controls file would train against a bank rolled at one reference speed
        # while the file's author meant per-window. Neither failure is visible in
        # any artifact the run leaves behind, so neither is allowed to be silent.
        _want = bool(getattr(args, "anchor_v0_conditioned", False))
        if _want and _ctrl is None:
            raise SystemExit(
                "[v3] ⛔ --anchor-v0-conditioned given but "
                f"{args.anchors} carries no `controls` [N, 2]. The bank is "
                "rolled per window from those controls; there is nothing to "
                "roll.")
        if _ctrl is not None and not _want:
            raise SystemExit(
                f"[v3] ⛔ {args.anchors} carries `controls` [N, 2] (a "
                "v0-CONDITIONED vocabulary) but --anchor-v0-conditioned was NOT "
                "given. Its paths are the family rolled at the REFERENCE speed "
                "only, and training on them fixed would silently be a different "
                "experiment.")
        model.core.decoder.load_anchors(
            anc.to(device), None if _ctrl is None else _ctrl.to(device))
        try:
            import hashlib
            _sha = hashlib.sha256(
                anc.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
        except Exception:                      # provenance is best-effort, never fatal
            _sha = "unavailable"
        print(f"[v3] anchors: loaded {tuple(anc.shape)} from {args.anchors} "
              f"(sha256 {_sha})", flush=True)
        if _ctrl is not None:
            _ok = bool(((_ctrl[:, 0] == 0) & (_ctrl[:, 1] == 0)).any())
            _u = art.control_units
            print(f"[v3] anchors: v0-CONDITIONED, controls {tuple(_ctrl.shape)} "
                  f"(accel, {'lateral accel' if _u == 'alat' else 'curvature'}), "
                  f"units={_u} (source: {art.control_units_source}), rolled "
                  f"per window; withheld rows at "
                  f"{getattr(args, 'anchor_ref_speed', 10.0)} m/s. "
                  f"straight-ahead control present: {_ok}", flush=True)
            if not _ok:
                # ⚠️ RETRACTED 2026-09-05 (RETRACTION_LOG): this message used
                # to say "Rebuild with odd counts". An odd count is neither
                # necessary nor sufficient -- np.linspace(-4, 3, 13) has 13
                # nodes and NO zero (step 7/12: ..., -0.5, +0.0833, ...); only
                # a SYMMETRIC odd grid, or a re-centred one, contains 0.0.
                raise SystemExit(
                    "[v3] ⛔ the control grid does not contain {a=0, kappa=0} "
                    "EXACTLY. A linspace that does not pass through 0.0 -- an "
                    "even count on a symmetric range, or ANY count on an "
                    "asymmetric one such as np.linspace(-4, 3, 13) -- omits "
                    "it, and the set then reads 1.2768 m oracle-in-vocabulary "
                    "against 0.2610 — a 4.9x artifact that looks exactly like "
                    "a resolution finding. Rebuild so that 0.0 is a node of "
                    "BOTH axes (assert `np.any(grid == 0.0)`; re-centre if "
                    "the range is asymmetric).")
    else:
        print("[v3] ⛔ anchors: NO --anchors GIVEN — using the SYNTHETIC "
              "`default_anchors` fallback. Its oracle-in-vocabulary ADE was MEASURED "
              "at 0.9433 m, worse than a single straight line (0.6780 m) and 2.16x "
              "the data-driven vocabulary (0.4369 m). The model's ceiling is below "
              "the trivial floor. This is almost certainly NOT what you want.",
              flush=True)

    # data — raw epcache, v2 compressed cache, or CI-synthetic; exactly one
    n_src = sum(1 for s in (args.data_root, args.v2_cache,
                            args.synth_episodes) if s)
    if n_src != 1:
        raise SystemExit("[v3] pass exactly one of --data-root / --v2-cache / "
                         "--synth-episodes (the synthetic corpus is CI-only "
                         "and must never masquerade as a training cache)")
    v2_parity = None
    if args.synth_episodes:
        eps = _synth_episodes(args.synth_episodes, cfg.core, seed=args.seed)
    elif args.v2_cache:
        # the flagship's --v2-cache recipe (train_flagship4b.py), verbatim in
        # structure: membership guard BEFORE any GPU work, lazy LRU-bounded
        # providers (contract-identical episode surface), and the independent
        # provider-vs-guard count cross-check. `require=False` keeps
        # deliberately-non-parity corpora usable behind ONE loud line;
        # --require-parity turns the same check into a refusal.
        from tanitad.data import parity
        from tanitad.data.v2_dataset import build_v2_providers
        v2_parity = parity.assert_v2_parity_cache(
            args.v2_cache, label="v3 v2-cache", require=args.require_parity)
        eps = build_v2_providers(args.v2_cache, lru_size=args.v2_lru)
        if v2_parity.get("parity") and len(eps) != v2_parity["episodes_loaded"]:
            raise parity.ParityViolation(
                f"PARITY VIOLATION [v3 v2-cache]: the guard verified "
                f"{v2_parity['episodes_loaded']} clip files but the loader "
                f"built {len(eps)} providers — the _v2manifest.pt sidecar "
                f"disagrees with the directory; rebuild it "
                f"(build_v2_providers(..., rebuild=True)) and re-run.")
        if args.episodes:
            eps = eps[:args.episodes]
        print(f"[v3] {len(eps)} lazy v2 providers from {args.v2_cache} "
              f"(lru {args.v2_lru})")
    else:
        eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                              args.episodes)
        print(f"[v3] {len(eps)} episodes from {train_dir}")
    # ⛔ geometry is asserted against the EPISODES, not against the flag: a
    # 256x640 corpus fed to a 256x256 build would run (conv is size-agnostic)
    # and silently train a model whose config lies about its input.
    eh, ew = cfg.core.encoder.image_hw()
    fh, fw = int(eps[0].frames.shape[-2]), int(eps[0].frames.shape[-1])
    if (fh, fw) != (eh, ew):
        raise SystemExit(
            f"[v3] ⛔ geometry mismatch: encoder built for {eh}x{ew} but the "
            f"corpus emits {fh}x{fw} — pass --image-hw {fh} {fw} (params are "
            f"unchanged; only compute scales), or point at a matching cache.")
    want_lan = bool(args.graft_lan or args.goal_str)
    dcls = lan_dataset_class(V3Dataset) if want_lan else V3Dataset
    kw = dict(window=cfg.core.window, max_horizon=20,   # ⛔ parity: NEVER 60
              channels=cfg.core.encoder.in_channels)
    if want_lan:
        kw["lan_cfg"] = DataLanConfig(
            arclengths_m=tuple(args.lan_arclengths),
            min_lead_m=args.lan_min_lead_m)
    ds = dcls(eps, **kw)
    # --u8-batches: the dataset emits frames AS STORED (uint8); the device-side
    # /255 is frames_to_device in compute_losses_v3. Instance attribute — the
    # V3Dataset idiom; see FailLoudWindowDataset.u8_frames (refb_train.py).
    u8 = bool(getattr(args, "u8_batches", False))
    ds.u8_frames = u8
    nav_stats = eval_nav_stats = v7_manifest = None
    # ---- v7.2 label join (PI 2026-09-02: MANDATORY for this launch) --------
    if args.v7_labels:
        from tanitad.data.v2_dataset import stable_episode_id
        labels, manifest = v7l.load_v7_labels(args.v7_labels,
                                              allow_oracle_nav=True)
        by_sid = {stable_episode_id(l.clip_id): l for l in labels}
        ds.v7_by_sid = by_sid
        ds.v7_dt = 0.1
        v7_manifest = manifest.to_dict()      # md5 + the oracle stamp
        # ⛔ COVERAGE IS REPORTED, NOT ASSUMED. MEASURED 2026-09-02: the v7.2
        # train set joins 4,572/4,713 = 97.0 % of B1 but only 190/2,400 =
        # 7.9 % of the PARITY corpus — so the same flag on the wrong cache
        # silently supervises the tactical heads on ~8 % of clips. The launch
        # record must carry the number, and a low one must be loud.
        hit = sum(1 for e in eps if int(e.episode_id) in by_sid)
        frac = hit / max(len(eps), 1)
        print(f"[v3] v7.2 labels: {len(labels)} records, joined "
              f"{hit}/{len(eps)} episodes = {100 * frac:.1f} % "
              f"(release {manifest.release if hasattr(manifest, 'release') else '?'})")
        if frac < 0.5:
            raise SystemExit(
                f"[v3] ⛔ v7.2 label coverage {100 * frac:.1f} % — refusing to "
                f"launch. The PI made v7.2 supervision MANDATORY; below half "
                f"the corpus the tactical/strategic heads would train on a "
                f"minority of clips while the run LOOKED labelled. Check the "
                f"cache is B1 (97.0 %) and not the parity corpus (7.9 %).")
        # ---- --nav-from-v7 (E-ARCH-NAVSRC-1, PI 2026-09-02): the nav INPUT
        # from the record's token — the input refav1 already trains on. The
        # v1 derivation feeds `follow` (+invalid) on 94.6 % of B1 windows
        # (MEASURED, nav-source-agreement package); the two agree on 65.5 %.
        if nav_on:
            nav_stats = ds.enable_nav_from_v7(manifest)
    # launch-line P4: the run PRINTS its episode/window counts at start — the
    # only way a parity claim about the enumeration is checkable from the log.
    print(f"[v3] {len(eps)} episodes -> {len(ds)} windows "
          f"(window {cfg.core.window}, max_horizon 20, "
          f"image_hw {cfg.core.encoder.image_hw()}, "
          f"tac_vocab {cfg.tac_vocab_version})")
    # ---- held-out eval split (PI 2026-09-02: "an eval step at 100 step") ---
    # ⛔ THE LEAK THIS CLOSES, MEASURED: the v7.2 release splits 4,572 train /
    # 147 eval with ZERO intersection, but the raw B1 corpus is 4,713 = 4,572 +
    # 141 of those eval clips. Training on "all of B1" therefore trains on 141
    # eval clips, and every eval number off them would be optimistic and
    # inadmissible. The launch trains the 4,572 and evaluates the held-out set.
    eval_dl = None
    if args.eval_cache and args.eval_every:
        # the SAME provider call the train side uses (refc_v3_train:680) —
        # a second construction path would be a second contract.
        from tanitad.data.v2_dataset import build_v2_providers
        e_eps = build_v2_providers([args.eval_cache], lru_size=args.v2_lru)
        overlap = ({int(e.episode_id) for e in eps}
                   & {int(e.episode_id) for e in e_eps})
        if overlap:
            raise SystemExit(
                f"[v3] ⛔ REFUSING: {len(overlap)} episodes appear in BOTH the "
                f"train cache and the eval cache. A held-out split that is not "
                f"held out measures memorisation.")
        e_ds = dcls(e_eps, **kw)
        e_ds.u8_frames = u8     # the eval decodes in the MAIN process: 4x less there too
        if args.eval_labels:
            from tanitad.data.v2_dataset import stable_episode_id
            e_lab, e_man = v7l.load_v7_labels(args.eval_labels,
                                              allow_oracle_nav=True)
            e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}
            e_ds.v7_dt = 0.1
            # the in-training eval sees the SAME nav source as training
            # (refused above when --eval-labels is missing)
            if nav_on:
                eval_nav_stats = e_ds.enable_nav_from_v7(e_man)
        # FIXED **and REPRESENTATIVE** windows.
        # ⛔ shuffle=False ALONE IS A TRAP, and it bit this eval on its first
        # run: taking the first N windows takes them from the START of the
        # first episodes (window NOW ~0.7-2 s), and every v7.2 record is
        # anchored at t0 = 8.0 s with a +-2 s band — so `eval_lat_tac` read
        # exactly 0.0000, an eval that silently never measured the tactical
        # heads while reporting a clean number. ⇒ take a DETERMINISTIC RANDOM
        # subset (seeded, computed once) and iterate it in a fixed order: still
        # the same windows at every step, but drawn from the whole corpus.
        g_ev = torch.Generator().manual_seed(12345)
        n_need = args.eval_batches * args.batch
        perm = torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()
        eval_dl = torch.utils.data.DataLoader(
            torch.utils.data.Subset(e_ds, perm),
            batch_size=args.batch, shuffle=False,
            num_workers=0, drop_last=True)
        print(f"[v3] held-out eval: {len(e_eps)} episodes -> {len(e_ds)} "
              f"windows, {args.eval_batches} fixed batches every "
              f"{args.eval_every} steps", flush=True)

    # ⛔ IN-FLIGHT BATCHES ARE THE SHM COST, NOT THE CLIP CACHE. MEASURED
    # 2026-09-02: two launches died on `Bus error ... out of shared memory`
    # even with file_system sharing. One collated batch is
    #   20 x window 8 x 9ch x 256 x 640 x 4B ~= 940 MB
    # so workers x prefetch_factor batches are in flight at once: 4 x 2 = 8
    # ~= 8 GB against a 24 GB /dev/shm, plus the eval loader's own. The knob
    # that actually bounds this is prefetch_factor, and it had been left at
    # torch's default of 2.
    pf = {"prefetch_factor": args.prefetch_factor} if args.workers > 0 else {}
    if args.workers > 0:
        print(f"[v3] loader: workers={args.workers} prefetch={args.prefetch_factor}"
              f" -> <={args.workers * args.prefetch_factor} batches in flight",
              flush=True)
    # ⭐ The in-flight bytes are COMPUTED and PRINTED (and stamped into
    # config.json) so the shm budget is checkable from train.log: frames +
    # future_frames per collated batch = B x (window + max_horizon) x C x H x W
    # x (1 B uint8 | 4 B fp32). At the live shape (20 x 28 x 9 x 256 x 640)
    # that is 3.30 GB fp32 / 0.83 GB uint8 per batch — the two /dev/shm
    # segments MEASURED per worker on 2026-09-02 (2,359,296,064 B and
    # 943,718,464 B) are exactly these two tensors plus a 64 B header each.
    _h, _w = cfg.core.encoder.image_hw()
    frame_batch_bytes = (args.batch * (cfg.core.window + kw["max_horizon"])
                         * cfg.core.encoder.in_channels * _h * _w
                         * (1 if u8 else 4))
    n_flight = args.workers * args.prefetch_factor if args.workers > 0 else 1
    print(f"[v3] u8_batches={u8}: frames+future_frames = "
          f"{frame_batch_bytes / 1e9:.3f} GB per collated batch "
          f"({'uint8' if u8 else 'float32'}), <= {n_flight} in flight "
          f"-> ~{n_flight * frame_batch_bytes / 1e9:.2f} GB", flush=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        **pf,
        drop_last=True, persistent_workers=args.workers > 0)

    withheld_stamp = _apply_withheld_bank(model, args, eps, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = lambda s: (s + 1) / max(1, args.warmup) if s < args.warmup else \
        0.5 * (1.0 + math.cos(math.pi * (s - args.warmup)
                              / max(1, args.steps - args.warmup)))   # noqa: E731

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    ck = out_dir / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])          # strict — resume exact
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        print(f"[v3] resumed {args.arm} at step {step}")
    (out_dir / "config.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "argv": sys.argv[1:],
        "registered_delta": {k: [repr(a), repr(b)]
                             for k, (a, b) in delta.items()},
        "param_breakdown": v3.param_breakdown_v3(model),
        "goal_provenance": refc.RefCModel.goal_provenance(),
        "provenance_roles": model.provenance_roles(),   # v4: INSTANCE call —
        # the refused edge differs between v3 (v0) and v4 (future ego),
        # so a static declaration could not tell the two arms apart, which
        # is exactly the "declared provenance" failure the audit exists for.
        "horizons": list(cfg.core.trajectory.horizons),
        # ⭐⭐ THE SELECTION / VOCABULARY STAMP. MEASURED 2026-09-04: refcv3's
        # config.json recorded NEITHER, so its own artifact cannot answer
        # "which anchor vocabulary?" or "how wide was the reach band?" — and
        # the answer to the first was `default_anchors` (SYNTHETIC), whose
        # oracle-in-vocabulary ADE 0-2 s is 1.0882 m against 0.3796 m for the
        # data-driven set and 0.6843 m for a single straight line. A silent
        # fallback to the synthetic default is exactly what an unstamped run
        # cannot rule out afterwards, so it is stamped here by CONTENT
        # (sha256 + shape), not by the presence of a path.
        "anchors": _anchor_stamp(
            getattr(args, "anchors", None), model.core.decoder.anchors,
            model.core.decoder.anchor_controls
            if getattr(model.core.decoder, "anchor_v0_cond", False) else None,
            getattr(args, "anchor_control_units", None) or "kappa",
            art=art),
        # ⭐⭐ THE SEAM STAMP (2026-09-05). MEASURED 2026-09-04: none of the
        # hierarchy-seam booleans were in config.json at any nesting level,
        # so the live arm could not rebuild its own model config from its
        # own record. `_seam_stamp` names each one.
        "seams": _seam_stamp(cfg, args),
        # ⭐ H-EGO-LIT-4: the withheld-row bank policy, with the random
        # control's marginal on record.
        "withheld_bank": withheld_stamp,
        "selection": {
            "sel_reach_clamp": bool(cfg.core.sel_reach_clamp),
            "sel_accel_max": float(cfg.core.sel_accel_max),
            "horizon_s": float(cfg.core.selection().horizon_s),
            "band_ms": float(cfg.core.sel_accel_max
                             * cfg.core.selection().horizon_s),
            "note": ("horizon_s is DERIVED from max(horizons); the 72.08 % / "
                     "77.28 % kill statistics quoted elsewhere are 2 s numbers "
                     "and do NOT hold here (re-derived 2026-09-04)."),
        },
        "goal_tau_steps": list(cfg.goal_tau_steps),
        "admission_sigma_m": cfg.admission_sigma_m,
        # ⭐⭐ THE EGO STAMP (v4). refcv3 recorded NEITHER ego knob, so a
        # finished run cannot answer "was the speed masked?" from its own
        # artifact - and that answer changes what every longitudinal number
        # MEANS. A run's config.json is the only durable record it has.
        "ego": {
            "ego_state_inject": bool(getattr(cfg, "ego_state_inject", False)),
            "echo_base": bool(getattr(cfg, "echo_base", False)),
            "ego_valid_channel": bool(cfg.core.ego_valid_channel),
            "ego_dropout": float(cfg.core.ego_dropout),
            "tactical_speed_input": bool(cfg.core.tactical_speed_input),
            "channels": (["v0", "a_long", "yaw_rate", "curvature", "keep"]
                         if getattr(cfg, "ego_state_inject", False)
                         else ["v0"]),
            "derivation": ("t0 = last OBSERVED frame; a_long = corpus ax "
                           "(actions[:,-1,1]); curvature = tan(steer)/2.9 "
                           "(actions[:,-1,0]); yaw_rate = v0*curvature. NO "
                           "finite differencing, NO future read."),
            "wheelbase_mode": "const2p9",
        },
        # ⭐ THE RUNG, STAMPED. `--size` decides ~4x of the parameter count and
        # was recorded NOWHERE, so a finished run could not say which rung it
        # was — and with a VALIDATION-RIG rung now reachable by name, an
        # unstamped run is one whose numbers cannot be told apart from a
        # registered arm's. `rig_rung: true` is the refusal token: nothing
        # carrying it may enter MODEL_REGISTRY.md.
        "size": args.size,
        "rig_rung": bool(args.size in v3.V3_RIG_SIZES),
        # ⛔ the deliberate-regression stamp. A run whose frames were
        # ablated is a GATE CONTROL; its numbers are not the model's and the
        # artifact must say so on its own.
        "ablate_frames": bool(args.ablate_frames),
        # the 2026-09-01 B1-readiness fields — the config.json is the ONLY
        # durable record of what corpus/geometry/vocab a run actually used.
        "image_hw": list(cfg.core.encoder.image_hw()),
        "tac_vocab_version": cfg.tac_vocab_version,
        "v2_cache": args.v2_cache, "require_parity": bool(args.require_parity),
        "v2_parity": v2_parity,
        # --u8-batches (2026-09-02): the in-flight dtype and its per-batch
        # frame bytes — a relaunch arm must be identifiable from its record.
        "u8_batches": u8,
        "frame_batch_bytes_est": frame_batch_bytes,
        # --nav-from-v7 (E-ARCH-NAVSRC-1): the nav SOURCE is stamped EITHER
        # way, so an arm is identifiable from its own artifacts
        # (C-NAV-SOURCE-DIVERGENCE). `v7_labels` is the label manifest —
        # md5 + the allow_oracle_nav stamp — None without --v7-labels.
        "nav_from_v7": nav_on,
        "nav_cmd_derivation": (NAV_FROM_V7_DERIVATION if nav_on
                               else NAV_V1_DERIVATION),
        "nav_from_v7_stats": ({"train": nav_stats, "eval": eval_nav_stats}
                              if nav_on else None),
        "v7_labels": v7_manifest,
    }, indent=1), encoding="utf-8")

    log = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
    t0, model = time.time(), model.train()
    it = iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        # ⭐ H-EGO-LIT-4: the withheld bank goes live after the warm-up. Set
        # EVERY step (not once) so a resumed run lands in the right regime,
        # and logged so the record says which bank each step trained on.
        _wb_active = step >= args.withheld_bank_warmup
        model.core.decoder.anchor_withheld_bank = (
            args.withheld_bank if _wb_active else "fixed")
        losses = compute_losses_v3(model, batch, device, mode=args.mode,
                                   ablate_frames=args.ablate_frames)
        losses["withheld_bank_active"] = float(_wb_active)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            # ⚠️ The tensor filter silently DROPPED every plain-float
            # diagnostic — `tac_label_v7`, `tac_label_rows`, `nav_injected`
            # were emitted by the loss and never reached the log, so "did the
            # v7.2 labels actually train?" was answerable only by INFERRING it
            # from the CE magnitude (lat 1.22 ~ ln 3 vs lat_tac 2.13 ~ ln 8).
            # A diagnostic that does not survive to the log is not a
            # diagnostic. Scalars now pass through as themselves.
            row = {k: (round(float(v.detach()), 5) if torch.is_tensor(v)
                       else round(float(v), 5))
                   for k, v in losses.items()
                   if (torch.is_tensor(v) and v.ndim == 0)
                   or isinstance(v, (int, float, bool))}
            row.update(step=step, elapsed_s=round(time.time() - t0, 1),
                       lr=opt.param_groups[0]["lr"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[v3:{args.arm}] step {step} "
                  f"loss {row['loss']:.4f} traj {row['traj']:.4f}")
        # ---- checkpoint FIRST, eval second (C-REFCV3-EVAL-DEATH) ----------
        # ⛔ THE ORDER OF THESE TWO BLOCKS IS LOAD-BEARING. MEASURED 2026-09-02:
        # refcv3 died SILENTLY at --eval-every 500 boundaries (steps 2,000 /
        # 4,500 / 6,000 / 10,500 / 17,000; empty stderr): the pod's memory
        # cgroup (v1, 50 GB, oom_kill 24, max_usage == limit) SIGKILLs the
        # trainer while the in-training eval decodes its 20-window batches in
        # the MAIN process on top of the six workers' in-flight fp32 batches.
        # The eval used to run BEFORE the save, so a death inside it lost every
        # step since the previous boundary -- the 17,000 death left ckpt.pt at
        # 16,500. Saving first bounds the cost of an eval death to ZERO
        # training steps. Training itself is untouched (same steps, same
        # losses; the eval still runs at the same point), and the checkpoint
        # is identical either way on EVERY parameter and optimizer tensor
        # (pinned by tests/test_refc_v3_save_before_eval.py) -- with ONE
        # measured exception that is a pre-existing defect, not an effect of
        # the order: compute_losses_v3 calls core.update_tactical_prior()
        # UNCONDITIONALLY, so the eval's held-out LABELS EMA into the
        # lat/lon_log_prior buffers (2 of 192 model tensors). The old order
        # saved the post-eval buffers, this order saves the pre-eval ones;
        # the leak itself is escalated (gate that call on model.training),
        # and the test flips strict-xfail -> pass the day it lands. The only
        # other observable change is that the step-N eval row / [v3:eval]
        # line now come AFTER the ckpt.pt write; metrics.jsonl's own row
        # order is unchanged (the save writes no row) and nothing consumes
        # the cross-file order. The memory fix itself (uint8 in-flight
        # batches) is a separate change.
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step}, ck)
            # the one line that makes the order VERIFIABLE from train.log
            print(f"[v3:{args.arm}] ckpt step {step} -> {ck.name}",
                  flush=True)
        if step in MILESTONES:
            torch.save({"model": model.state_dict(), "step": step},
                       out_dir / f"ckpt_{step}.pt")
        # ---- held-out eval (PI 2026-09-02) --------------------------------
        # ⚠️ WHAT THIS IS AND IS NOT: an in-training MONITOR at T0 on the same
        # loss surface, over a FIXED set of held-out windows. It is NOT the
        # four-metric-family result and must never be quoted as one — the
        # binding families (longitudinal / lateral / tactical / strategic with
        # paired episode-cluster CIs) are a separate T1 job. What it buys is
        # the ability to see generalisation move DURING a 25 h run instead of
        # after it, and to catch train-only improvement early.
        if eval_dl is not None and (step % args.eval_every == 0
                                    or step == args.steps):
            model.eval()
            acc, nb_e, eval_err = {}, 0, None
            # ⚠️ FAIL LOUD, SURVIVE. An in-training eval is a DIAGNOSTIC; it
            # must never take the run down. A CUDA OOM or a decode error raised
            # in here used to propagate out of train() and END THE RUN. It is
            # now logged as an `eval_error` row + a `[v3:eval] FAILED` line
            # (traceback to stderr), the model goes back to train mode, and
            # training continues from the checkpoint written just above.
            # ⛔ A kernel SIGKILL -- the cgroup OOM killer of
            # C-REFCV3-EVAL-DEATH -- CANNOT be caught by any Python handler;
            # for that case the save-before-eval order above is the whole
            # protection.
            try:
                with torch.no_grad():
                    for eb in eval_dl:
                        if nb_e >= args.eval_batches:
                            break
                        el = compute_losses_v3(
                            model, eb, device, mode=args.mode,
                            ablate_frames=args.ablate_frames)
                        for k, v in el.items():
                            if torch.is_tensor(v) and v.ndim == 0:
                                acc[k] = acc.get(k, 0.0) + float(v.detach())
                            elif isinstance(v, (int, float, bool)):
                                acc[k] = acc.get(k, 0.0) + float(v)
                        nb_e += 1
            except Exception as exc:          # noqa: BLE001 (by design)
                import traceback            # local: this block is the boundary
                traceback.print_exc()
                eval_err = f"{type(exc).__name__}: {exc}"
            model.train()
            if eval_err is not None:
                log.write(json.dumps({"step": step, "eval_error": eval_err,
                                      "eval_batches_done": nb_e}) + chr(10))
                log.flush()
                # ASCII-safe on purpose: a UnicodeEncodeError raised while
                # REPORTING the failure would defeat the survival.
                print("[v3:eval] FAILED at step %d: %s -- training continues"
                      % (step, eval_err.encode("ascii", "backslashreplace")
                         .decode("ascii")), flush=True)
            elif nb_e:
                erow = {f"eval_{k}": round(v / nb_e, 5) for k, v in acc.items()}
                erow.update(step=step, eval_batches=nb_e,
                            eval_windows=nb_e * args.batch)
                log.write(json.dumps(erow) + chr(10))
                log.flush()
                print(f"[v3:eval] step {step} "
                      f"loss {erow.get('eval_loss', float('nan')):.4f} "
                      f"traj {erow.get('eval_traj', float('nan')):.4f} "
                      f"lat_tac {erow.get('eval_lat_tac', float('nan')):.4f}",
                      flush=True)
    # ⛔ the done-marker, SAME turn as completion (the v5f supervisor lesson).
    (out_dir / "summary.json").write_text(
        json.dumps({"done": True, "step": step, "arm": args.arm,
                    "seed": args.seed,
                    "wallclock_s": round(time.time() - t0, 1)}),
        encoding="utf-8")
    log.close()
    print(f"[v3:{args.arm}] DONE at {step} — summary.json written")
    return {"step": step}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size",
                    choices=tuple(v3.V3_SIZES) + tuple(v3.V3_RIG_SIZES),
                    default="small",
                    help="encoder rung. 'small' is the AS-REGISTERED arm "
                         "(62,930,419); 'xl' is the D-008 >=250M rung "
                         "(217,760,775). ⛔ The size axis moves the ENCODER "
                         "ONLY — the H-vs-F delta is identical at every rung "
                         "(pinned by tests). ⚠️ Anything but 'small' VOIDS the "
                         "registered cost line in PREREG_REFC_V3.md; amend "
                         "BEFORE any read, never after. ⛔ 'tiny' (16,989,725) "
                         "is the VALIDATION RIG rung (V3_RIG_SIZES), not a "
                         "member of the registered ladder: it exists so a "
                         "design change can be gated on the dev box before it "
                         "earns compute, and NOTHING measured at it is a model "
                         "claim or may enter MODEL_REGISTRY.md.")
    ap.add_argument("--arm", choices=("hier", "flat"), required=True,
                    help="v3-H (goal cascade) or v3-F (flat) — the dominance "
                         "pair; delta pinned to the registered lever set")
    ap.add_argument("--data-root", default=None,
                    help="raw epcache root of ep_*.pt (parity-guarded; the "
                         "physicalai-train-e438721ae894 convention)")
    ap.add_argument("--v2-cache", nargs="+", default=None,
                    help="v2 compressed cache dir(s) of *.v2ep.pt (e.g. the B1 "
                         "corpus physicalai-b1-w120-256x640cyl). Swaps the raw "
                         "loader for the lazy LRU-bounded v2 providers; the "
                         "window contract is identical. Pair with "
                         "--image-hw 256 640 for the B1 geometry — the run "
                         "REFUSES a geometry mismatch either way.")
    ap.add_argument("--prefetch-factor", type=int, default=1,
                    help="batches prefetched PER WORKER. One collated batch is "
                         "~940 MB at batch 20, so workers x prefetch bounds "
                         "shared-memory use; torch's default of 2 put 8 "
                         "batches in flight and exhausted a 24 GB /dev/shm.")
    ap.add_argument("--u8-batches", action="store_true",
                    help="ship frames/future_frames through the DataLoader as "
                         "uint8 and convert to float32 [0,1] ON THE DEVICE "
                         "(the dataset contract's own /255). Cuts every "
                         "in-flight batch 4x: at batch 20 / window 8 / 20 "
                         "future frames / 9ch / 256x640 one collated batch is "
                         "3.30 GB fp32 -> 0.83 GB uint8, so 6 workers x "
                         "prefetch 1 hold ~4.95 GB of /dev/shm instead of "
                         "~19.8 GB (the unreclaimable share that put refcv3's "
                         "50 GB cgroup at its cap and OOM-killed it at eval "
                         "steps, MEASURED 2026-09-02). Loss bit-identical "
                         "(pinned). Default OFF: the live run resumes "
                         "byte-identical; this is a next-relaunch arm.")
    ap.add_argument("--eval-cache", default=None,
                    help="HELD-OUT v2 cache (the v7.2 eval split). ⛔ MUST be "
                         "disjoint from --v2-cache: 141 of the 147 v7.2 eval "
                         "clips sit inside the raw B1 corpus, so training on "
                         "'all of B1' trains on the eval set (MEASURED "
                         "2026-09-02).")
    ap.add_argument("--eval-labels", default=None,
                    help="s2_labels_v7.2_eval.jsonl.gz for the eval split")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="run the held-out eval every N steps (0 = off)")
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="FIXED number of eval batches — the same windows "
                         "every time, so step-to-step deltas are the model "
                         "moving and not the sample moving")
    ap.add_argument("--v7-labels", default=None,
                    help="s2_labels_v7.2_*.jsonl.gz — the RELEASED tactical "
                         "vocabulary (8x8). Joined per window on "
                         "stable_episode_id(clip_id), the v7.2 index's own "
                         "'ONLY admissible join key'. Sets tac_vocab_version "
                         "to v7.0; without it the trainer derives kin3 (3x3) "
                         "from poses. Coverage is PRINTED and a run below "
                         "50 %% is REFUSED.")
    ap.add_argument("--v2-lru", type=int, default=6,
                    help="per-process LRU of decoded-payload clips for "
                         "--v2-cache. ⚠️ B1 payloads are ~34 MB/clip "
                         "(256x640 PNG) — the old '2-4 MB' sizing is 8-17x "
                         "low (MEASURED, V5F_SIGKILL.md), and under "
                         "shuffle the hit rate is ~lru/n_clips, so big "
                         "values buy RAM pressure, not throughput. 6 "
                         "matches the v6 chain's setting.")
    ap.add_argument("--require-parity", action="store_true",
                    help="REFUSE unless --v2-cache references a REGISTERED "
                         "corpus key (B1 is unregistered as of 2026-09-01: "
                         "expect the loud NON-PARITY line without this flag; "
                         "registering B1 is the cross-arm-comparability "
                         "instrument)")
    ap.add_argument("--image-hw", type=int, nargs=2, default=None,
                    metavar=("H", "W"),
                    help="build the encoder at this input geometry (B1: "
                         "256 640). Params unchanged (fully-conv trunk); "
                         "compute scales with area. Applies to --preflight "
                         "too, so the gate runs at the launch geometry.")
    ap.add_argument("--synth-episodes", type=int, default=0,
                    help="CI-ONLY synthetic corpus (mutually exclusive with "
                         "--data-root / --v2-cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--mode", choices=("classifier", "diffusion"),
                    default="diffusion")
    ap.add_argument("--anchors", default=None,
                    help="6 s anchor vocabulary (build_refc_anchors.py over "
                         "V3_HORIZONS; the model's synthetic default otherwise)")
    ap.add_argument("--n-anchors", type=int, default=None,
                    help="anchor vocabulary size. Needed when the built file is "
                         "not the 128 the size preset assumes; `load_anchors` "
                         "refuses a shape mismatch, so a wrong value is loud.")
    ap.add_argument("--anchor-v0-conditioned", action="store_true",
                    help="refcv4-b: the anchor file carries `controls` [N, 2] = "
                         "(accel, curvature) and the bank is ROLLED PER WINDOW "
                         "from that window's measured v0 instead of being a "
                         "fixed set of paths in absolute metres. MEASURED on the "
                         "4,823-window banked surface: the fixed-path set reads "
                         "0.3773 m oracle-in-vocabulary (+0.0777 [+0.0528, "
                         "+0.1044] SEPARATED WORSE than ha = 0.2996); the "
                         "v0-conditioned family at 117 candidates reads 0.2610 "
                         "(-0.0387 [-0.0652, -0.0104] BEATS ha). Requires a "
                         "`controls` entry in --anchors; refused without one.")
    ap.add_argument("--anchor-control-units", default=None,
                    choices=["kappa", "alat"],
                    help="⛔ EXPLICIT OVERRIDE ONLY (default None): the anchor "
                         "artifact is the authority on what its own `controls` "
                         "column MEANS, and a controls-carrying file that "
                         "declares no `control_units` is REFUSED unless this "
                         "flag names them (recorded in config.json as "
                         "control_units_source='cli-override-legacy-file'); "
                         "a file that declares a different unit than this "
                         "flag is refused too. MEASURED 2026-09-04: the live "
                         "refcv4b anchors.pt declared nothing, and its "
                         "lateral column read as curvature gives 396 g at "
                         "36 m/s against the true 0.31 g. "
                         "What channel 1 of the anchor controls MEANS. "
                         "'kappa' = curvature 1/m, integrated as supplied. "
                         "'alat' = LATERAL ACCELERATION m/s^2, with curvature "
                         "DERIVED per window as a_lat / max(v0, 4.0)^2 and "
                         "clamped. MEASURED: a constant-CURVATURE family is "
                         "unflyable at speed (104 of 117 anchors break a mu=0.7 "
                         "circle at v0 = 27 m/s, peak 3.96 g); under 'alat' the "
                         "control space IS the Kamm-circle space, and the same "
                         "117 anchors read 0.1987 m oracle-in-vocabulary "
                         "(-0.1009 [-0.1213, -0.0813] vs ha) with 0/117 over "
                         "mu = 0.7 and a 0.68 g peak.")
    ap.add_argument("--anchor-ref-speed", type=float, default=10.0,
                    help="m/s the bank is rolled at where the ego channel was "
                         "WITHHELD by --ego-dropout. Rolling a withheld row from "
                         "its true v0 would put the withheld channel into the "
                         "candidate GEOMETRY, which is a harder leak than the "
                         "ranking one S2 guards.")
    # ---- refcv5 WP-4 / WP-6 ----------------------------------------------
    g5 = ap.add_argument_group(
        "refcv5", "WP-4 (control-space sampler) and WP-6 (agent tokens). "
        "EVERY flag here defaults to OFF, and OFF means NOT CONSTRUCTED: a "
        "run that passes none of them is bit-identical to refcv4b.")
    g5.add_argument("--sampler", default="none", choices=["none", "ddim"],
                    help="E-DDA-3. 'ddim' replaces the metre-space truncated "
                         "denoise with an anchored Gaussian in CONTROL space, "
                         "so every sample re-rolls through the kinematic model "
                         "and is flyable by construction. Needs a "
                         "v0-conditioned vocabulary and --w-u0 > 0.")
    g5.add_argument("--sampler-space", default="control",
                    choices=["control", "metre"],
                    help="DELIBERATE REGRESSION. 'metre' is the DD-literal "
                         "arm that noises the WAYPOINTS; pre-registered to "
                         "FAIL the flyability gate. If it does NOT fail, the "
                         "instrument cannot see what it is cited for: VOID.")
    g5.add_argument("--sampler-train-t-max", type=int, default=50,
                    help="train timestep ~ U[0, t_max). DD uses 50.")
    g5.add_argument("--sampler-infer-t", type=int, default=8,
                    help="the truncation point at eval. DD uses 8.")
    g5.add_argument("--sampler-steps", type=int, default=2,
                    help="DDIM steps at inference. DD uses 2; "
                         "D-REFC-DDAUDIT-2 MEASURED that passes beyond 2 "
                         "collapse the fan spread 104 -> 42 m, so this is NOT "
                         "a capacity knob.")
    g5.add_argument("--sampler-groups", type=int, default=1,
                    help="G samples per anchor. Only G=1 is implemented; G>1 "
                         "refuses and names the three consumers that still "
                         "assume an N-wide fan.")
    g5.add_argument("--w-u0", type=float, default=U0_WEIGHT_DEFAULT,
                    help="weight on the x0 loss, in CONTROL space -- the only "
                         "term that supervises the sampler in the space it "
                         "samples in.")
    g5.add_argument("--agents", default="off",
                    choices=["off", "head", "oracle"],
                    help="E-AGT-*. 'head' = the LEARNED monocular 3D head "
                         "(the deliverable arm; vision-only at inference, "
                         "obstacle.offline as TRAIN-TIME labels). 'oracle' = "
                         "GROUND-TRUTH boxes fed at inference: the CEILING, "
                         "INADMISSIBLE as a capability claim, run ONCE to "
                         "price the mechanism before any detector GPU-day. If "
                         "the oracle does not separate on LONGITUDINAL AND "
                         "TACTICAL the whole mechanism is refused.")
    g5.add_argument("--w-agent", type=float, default=AGENT_WEIGHT_DEFAULT,
                    help="weight on the GT-supervised detection set loss.")
    g5.add_argument("--agent-queries", type=int, default=32,
                    help="detection queries. 32 is MEASURED, not inherited: "
                         "on the val40 join the in-field/decode-box per-frame "
                         "count has max 24, so 32 drops ZERO targets; 16 "
                         "(agent_slots' placeholder) drops on 2.16 pct of "
                         "frames, nearest sacrificed target at 38.5 m.")
    g5.add_argument("--agent-w-project", type=float, default=0.0,
                    help="weight on the IMAGE-PLANE term. A monocular head "
                         "supervised only in BEV metres is asked to regress "
                         "the one axis it cannot directly see, with no term "
                         "in the space it can.")
    g5.add_argument("--agent-w-ground", type=float, default=0.0,
                    help="weight on the road-plane range prior. Costs NO "
                         "label (rig z=0 IS the road plane, MEASURED), so it "
                         "also trains on the NO_LABEL frames past ~20 s.")
    g5.add_argument("--agent-sigma-range", type=float, default=0.0,
                    help="E-AGT-BUDGET: range-noise sigma (m) on the ORACLE "
                         "boxes. The sigma where separation dies IS the "
                         "detector specification.")
    g5.add_argument("--agent-miss-rate", type=float, default=0.0,
                    help="E-AGT-BUDGET: the miss rate on the ORACLE boxes.")
    g5.add_argument("--agent-presence-hard", action="store_true",
                    help="hard-mask sub-threshold slots instead of soft "
                         "scaling. Soft is the default BECAUSE a hard mask has "
                         "zero gradient to the presence head through the "
                         "planner loss.")
    ap.add_argument("--withheld-bank", default="fixed",
                    choices=list(refc.WITHHELD_BANK_MODES),
                    help="H-EGO-LIT-4: the speed a WITHHELD row's anchor bank "
                         "is rolled at (v0-conditioned vocabularies only). "
                         "fixed = --anchor-ref-speed (the shipped roll; "
                         "bit-identical to the pre-flag trainer); pred = the "
                         "model's OWN detached vision-only g_tac 2 s speed "
                         "(SparseDrive's pattern), clamped to [0, "
                         "--withheld-speed-max]; random = a draw from the "
                         "TRAINING marginal of v0 (the blindness CONTROL); "
                         "none = every row at the reference speed (the field's "
                         "speed-blind vocabulary). Stamped in config.json "
                         "under seams.withheld_bank and withheld_bank.")
    ap.add_argument("--withheld-bank-warmup", type=int, default=0,
                    help="steps trained on the FIXED withheld bank before "
                         "--withheld-bank pred/random goes live (the goal "
                         "head's speed is noise at step 0). Logged per step "
                         "as withheld_bank_active.")
    ap.add_argument("--withheld-speed-max", type=float, default=35.0,
                    help="clamp for --withheld-bank pred, m/s.")
    ap.add_argument("--sel-accel-max", type=float, default=None,
                    help="S2 reach-clamp bound in m/s^2. RE-DERIVE IT FOR THE "
                         "HORIZON: `horizon_s` is max(horizons)*0.1, so the "
                         "band is a*horizon_s. The 2.5 default is a 2 s number "
                         "(+-5 m/s); at 6 s it opens to +-15 m/s and kills "
                         "only 18.02 pct of a data-driven vocabulary. MEASURED "
                         "pick for V3_HORIZONS: 2.0 (+-12 m/s, kills 26.23 "
                         "pct, deletes 0.000 pct of eval / 0.007 pct of train "
                         "GT, dADE +0.00000 m).")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--goal-str", action="store_true",
                    help="train the strategic goal head (needs the lan LABEL "
                         "field — minted WITHOUT building the input pathway, "
                         "the refc_goal_config discipline)")
    ap.add_argument("--graft-lan", action="store_true",
                    help="supplied-corridor MODEL INPUT — ⛔ NOT part of any "
                         "registered v3 arm (E12); exists for diagnostics only")
    ap.add_argument("--lan-arclengths", type=float, nargs="+",
                    default=[10.0, 20.0, 40.0, 80.0])
    ap.add_argument("--lan-min-lead-m", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU config (CI)")
    ap.add_argument("--preflight", action="store_true",
                    help="build + pin delta + C115 gate + E11 audit + one "
                         "synthetic loss step, then exit")
    # ---- ⭐⭐ REF-C v4 (E11' + E14 + X15), PI 2026-09-03 -----------------
    # ⚠️ There was NO CLI flag for EITHER ego knob before this, and refcv3's
    # config.json stamped NEITHER - so a finished run cannot answer "was the
    # speed masked?" from its own record. v4 adds the flags AND the stamp, and
    # the stamp is the half that still matters in six months.
    ap.add_argument("--ego-state-inject", action="store_true",
                    help="E11prime: feed the MEASURED t0 ego state (v0, "
                         "a_long, yaw_rate, curvature) to the goal/tactical "
                         "path. Implies --ego-valid-channel (refused "
                         "otherwise).")
    ap.add_argument("--echo-base", action="store_true",
                    help="E14: g_tac = kinematic extrapolation + zero-init "
                         "residual, so the trivial solution is free and every "
                         "learned parameter is spent on the scene.")
    ap.add_argument("--ego-valid-channel", action="store_true",
                    help="X15: an explicit 'the ego block is present' bit. "
                         "MEASURED n=781,635 windows: v0 is EXACTLY 0.0 on "
                         "4.4531 pct, so with --ego-dropout 0.5 the input "
                         "reads zero on 52.227 pct of TRAIN samples of which "
                         "only 4.263 pct are a genuinely stopped car (22.5:1) "
                         "- while at eval, dropout being training-only, 100 "
                         "pct are genuine. The same token means something "
                         "23.5x different between train and eval without it.")
    ap.add_argument("--ablate-frames", action="store_true",
                    help="⛔⛔ RIG ONLY - THE DELIBERATE-REGRESSION LEVER. "
                         "Replace the observed window with a scalar constant, "
                         "so the arm keeps every ego channel and NO scene "
                         "information: an echo BY CONSTRUCTION. Required by "
                         "TanitAD_ValidateAIDesign section 2 - if the "
                         "anti-echo gate does not FAIL this arm, a PASS on the "
                         "real arm means nothing. It is stamped in config.json "
                         "as `ablate_frames`; a run carrying it is a GATE "
                         "CONTROL and never a model.")
    ap.add_argument("--ego-dropout", type=float, default=None,
                    help="override core.ego_dropout. Under v4 this is ONE "
                         "draw per sample, shared by the goal path and the "
                         "measurement encoder (never two).")
    ap.add_argument("--nav-from-v7", action="store_true",
                    help="feed the model's nav_cmd INPUT from the clip's v7.2 "
                         "nav_command token (oracle, provenance ego-future; "
                         "loaded with allow_oracle_nav=True and STAMPED in "
                         "config.json) instead of refb_labels.nav_command "
                         "(v1: net yaw over the next 15-25 s of future poses "
                         "— `follow`+invalid on 94.6 %% of B1 windows, "
                         "E-ARCH-NAVSRC-1 2026-09-02). The same input refav1 "
                         "trains on. Applies to the train AND the held-out "
                         "eval dataset (needs --eval-labels when the eval is "
                         "on). A clip without a record feeds follow + "
                         "nav_valid=False and is COUNTED. Default OFF = "
                         "byte-identical v1 behaviour (the live run resumes "
                         "through this file).")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.preflight:
        raise SystemExit(preflight(args))
    train(args)


if __name__ == "__main__":
    main()
