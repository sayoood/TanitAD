"""apply_v7f_chain.py -- the v7F PROFILE for `stack/scripts/v6_chain.py` (PI R1-R6, 2026-09-27).

Adds `--profile {v6,v7f}` to the v6 stage chain. `v6` (the DEFAULT) is byte-identical for every
existing command; `v7f` plans S-W -> S-T only and emits the PI's R1-R6 flags (option B, strategic
OFF), checking every emitted line against the launch gate's own `v7f` argv rules (imported, never
re-implemented) plus the chain's R5/data-path rows, which the gate does not encode.

ANCHOR-BASED: every anchor must match EXACTLY ONCE or the patch refuses and writes nothing. The
source must be the integrated v7F merge's file: sha256 prefix 918859e75df9c343 as the CRLF worktree
copy, which is byte-for-byte the D: repo's tip blob db2e599d... once LF-normalised (the merge did
not touch this file). EITHER EOL form is accepted, the file's EOL is PRESERVED (CRLF in -> CRLF out,
LF in -> LF out), and the result is pinned by its LF-normalised sha256 so both forms are verified.
The new test file is installed from `fix/stack/tests/` beside this script, in the SAME EOL as the
patched v6_chain.py, unless --no-test.

usage:
    python apply_v7f_chain.py [--stack C:/Users/Admin/v7f_merge/stack] [--no-test] [--check]
"""
import argparse
import hashlib
import sys
from pathlib import Path

SRC_SHA256_PREFIX = "918859e75df9c343"          # the CRLF worktree copy (v7f_merge, v7f_chain, D:)
SRC_LF_SHA256 = "db2e599dbbe9dd8c72856eeb3fe5a8d4277502afeb439c6b8a9260410e345df4"   # = D: HEAD blob
#: the patched file this script must reproduce, LF-normalised (EOL-agnostic)
OUT_LF_SHA256 = "657358f14304e4b5fcc7e42a652a798195804cbcebc0ee6803364045e344505a"
#: ...and its CRLF form (the working copy this was built and tested in)
OUT_CRLF_SHA256 = "b1bf626b5f0b466354ebf551cad95827802f294f77d2c8486a335d772f5a434a"
TEST_LF_SHA256 = "d84c963178ab003443317df64ec838e5992293555b27e7edc96c53162f1cea9a"
HERE = Path(__file__).resolve().parent
CRLF = chr(13) + chr(10)


def _lf_sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(CRLF.encode(), b"\n")).hexdigest()


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--stack", default="C:/Users/Admin/v7f_merge/stack")
ap.add_argument("--no-test", action="store_true", help="patch v6_chain.py only")
ap.add_argument("--check", action="store_true", help="verify the anchors and exit; write nothing")
args = ap.parse_args()

target = Path(args.stack) / "scripts" / "v6_chain.py"
raw = target.read_bytes()
src_sha = hashlib.sha256(raw).hexdigest()
if not (src_sha.startswith(SRC_SHA256_PREFIX) or _lf_sha(raw) == SRC_LF_SHA256):
    sys.exit(f"[apply_v7f_chain] REFUSED: {target} sha256 {src_sha[:16]} (LF-normalised "
             f"{_lf_sha(raw)[:16]}) is not the v7F merge's v6_chain.py ({SRC_SHA256_PREFIX} / LF "
             f"{SRC_LF_SHA256[:16]}); refusing to patch a different base.")
eol = CRLF if CRLF.encode() in raw else chr(10)
s = raw.decode("utf-8").replace(CRLF, chr(10))


def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n:
        sys.exit(f"[apply_v7f_chain] anchor matched {c} (want {n}) -- refusing:\n{old[:300]}")
    s = s.replace(old, new)


# =====================================================================================================
# 1. the module docstring: the v7f usage line
# =====================================================================================================
rep('''    python3 scripts/v6_chain.py run --dry --root /tmp/ladder   # CPU, ~1 min
"""''', '''    python3 scripts/v6_chain.py run --dry --root /tmp/ladder   # CPU, ~1 min

THE v7F PROFILE (``--profile v7f``; PI R1-R6, 2026-09-27)
=========================================================
S-W -> S-T ONLY, strategic OFF, ONE combined tactical+operative 6 s plan (option
B). Every line it emits is checked against the launch gate's own ``v7f`` argv
rules (imported from ``launch_gate.py``, never re-implemented) plus the chain's
R5 / data-path rows. ``--profile v6`` (the DEFAULT) is byte-identical to the
pre-profile chain for every existing command. See :data:`V7F_REQUIREMENTS`.

    python3 scripts/v6_chain.py plan --profile v7f --nav-labels <v7 blob>
        --s2-labels <v7.2 blob> --speed-max-sidecar-v6 <refcv6 sidecar .jsonl>
    python3 scripts/v6_chain.py run --dry --profile v7f --root <scratch> ...
"""''')

# =====================================================================================================
# 2. the D1 default, as ONE constant, before ChainConfig
# =====================================================================================================
rep('''@dataclass
class ChainConfig:
    """Everything the ladder needs that is not a per-step fact."""''',
    '''#: ⚠️ R4's MODE IS PI DECISION D1 (``detached`` vs ``e2e``) — OPEN. ``detached``
#: (the plan loss trains the zero-init port only, never the behaviour heads) is
#: the R1/R4 builder's recommendation and the launch gate's rehearsed value
#: (``PROFILES["v7f"]["pi_pending_values"]``). The gate mints PI-DECISION, never
#: PASS, until D1 is recorded, so this default does not decide it.
V7F_TAC_OP_COND_DEFAULT = "detached"


@dataclass
class ChainConfig:
    """Everything the ladder needs that is not a per-step fact."""''')

# =====================================================================================================
# 3. ChainConfig: the v7f fields, APPENDED LAST (and hidden from the v6 `plan` view)
# =====================================================================================================
rep('''    save_every: int = 250

    def path(self, *parts: str) -> str:''',
    '''    save_every: int = 250
    # ---- ⭐ the v7F PROFILE (PI R1-R6, 2026-09-27) ---------------------------
    # ⛔ APPENDED LAST and hidden from the v6 `plan` view (:func:`config_view`),
    # so every v6 command prints byte-identically. Under ``v6`` they are unused,
    # and a CLI that sets one is REFUSED — a v7f-only input on a v6 ladder would
    # be advertised and read by nothing.
    #: ``"v6"`` (DEFAULT, byte-identical) | ``"v7f"`` — :data:`V7F_REQUIREMENTS`.
    profile: str = "v6"
    #: R2 — the nav label blob ``--nav-cond`` reads, on BOTH stages. v7f: REQUIRED.
    nav_labels: str | None = None
    #: R3 — the v7.2 (``s2-geom-v7``) tactical label blob. v7f: REQUIRED.
    s2_labels: str | None = None
    #: R1 — the refcv6 max-speed sidecar. ⚠️ Its ``source_md5`` must equal the
    #: md5 of ``nav_labels`` (the trainer's ``read_speed_max_sidecar_v6`` refuses
    #: otherwise), so the two travel as a PAIR. v7f: REQUIRED.
    speed_max_sidecar: str | None = None
    #: R4's mode — PI decision D1, see :data:`V7F_TAC_OP_COND_DEFAULT`.
    tac_op_cond: str = V7F_TAC_OP_COND_DEFAULT

    def path(self, *parts: str) -> str:''')

# =====================================================================================================
# 4. the v7f profile: requirements, declared S-W geometry, config + argv adjudication, the plan
# =====================================================================================================
rep('''def assert_no_tilde(cfg: ChainConfig) -> dict:''',
    '''# ============================================================================
# ⭐ THE v7F PROFILE — the PI's BINDING R1-R6 (2026-09-27), as data
# ============================================================================
PROFILES: tuple[str, ...] = ("v6", "v7f")

#: The requirement every v7f refusal quotes. PI 2026-09-27, BINDING for the
#: first v7F experiments. The canonical compliant argv is the one the binding
#: launch gate's ``v7f`` profile rehearses end-to-end
#: (``tests/test_launch_gate_v7f.py::ST_ARGV``); :func:`v7f_argv_rules` runs that
#: gate's own rule function on every line this profile emits.
V7F_REQUIREMENTS: dict[str, str] = {
    "R1": "the max speed is an INPUT and a CAP the plan must not exceed "
          "(--max-speed-input-v6 --plan-vmax-cap --speed-max-sidecar-v6; S-T "
          "only — the trainer REFUSES R1 in S-W)",
    "R2": "the navigation command is an input to the tactical AND operative "
          "layers (--nav-cond --nav-labels), FROM S-W ON: nav's shared "
          "embedding trains in S-W and S-T may not INTRODUCE nav.* "
          "(STAGE_MAY_INTRODUCE)",
    "R3": "ALL tactical labels train the tactical layer (--w-tac-label-all 1.0 "
          "--goal-multilabel --s2-labels; S-T)",
    "R4": "the tactical layer conditions the operative planning "
          "(--tac-op-cond {mode}; S-T). 'off' is refused; the MODE is PI "
          "decision D1",
    "R5": "ONE combined tactical+operative 6 s trajectory = option B: "
          "--n-candidates 1 on EVERY stage (S-T's --init-from dies on the "
          "cand_queries shape otherwise), and --proposals query --selector none "
          "on S-T (SEL-1 refused a selector)",
    "R6": "the strategic layer is OFF: --strategic-off on every stage, NO "
          "--tac-goal-cond (V6Config REFUSES the pair), and NO S-S / S-J stage "
          "(S-S trains only layer_str; S-J trains every group but interp)",
}
#: the stages R6 removes from the ladder, refused BY NAME when requested
V7F_REFUSED_STAGES: tuple[str, ...] = ("S-S", "S-J")
#: R4's admissible modes. ``off`` is refused (R4: the behaviours must reach the
#: plan), exactly as the gate's ``forbidden_values_v6`` refuses it.
V7F_TAC_OP_COND_MODES: tuple[str, ...] = ("detached", "e2e")
#: R3's weight (PI: "ALL our tactical labels must be used to train the tactical
#: layer"). A constant, not a knob: a weight sweep would be a new arm.
V7F_W_TAC_LABEL_ALL = 1.0
#: ⭐ THE v7F S-W MODEL GEOMETRY — DECLARED, because v7f's S-W is a NEW run with
#: no ancestor to carry from (v6F's S-W is the live run and never trained nav,
#: so it cannot seed an R2 ladder). VERBATIM the geometry block of the launch
#: gate's rehearsed compliant argv (``test_launch_gate_v7f.py::ST_ARGV``).
#: ⛔ Without it a fresh S-W builds at the trainer's DEFAULTS — ``--horizons 1 2
#: 4``, which the trainer's preflight REFUSES on a fresh run (two dead heads),
#: and a 9-channel 384-wide encoder. S-T never types it: S-T CARRIES S-W's own
#: ``config.json`` (E1), so what S-T builds is what S-W actually trained.
#: ⚠️ Every OTHER geometry dest takes the trainer's default on the day S-W
#: launches — exactly as in the gate's argv. S-W's ``config.json`` records the
#: resolved values and S-T carries those, so the drift window is S-W's launch.
V7F_SW_GEOMETRY: tuple[str, ...] = (
    "--in-channels", "3", "--enc-dim", "768", "--enc-depth", "12",
    "--enc-heads", "12", "--frame-h", "256", "--frame-w", "640",
    "--horizons", "1")
#: What S-W carries on EVERY launch, the dry ladder included (the DIMS above are
#: replaced there by :data:`TINY_GEOMETRY`, which also carries ``--in-channels
#: 3``). ``--nav-cond``/``--nav-labels`` are added from the config.
#:   * ``--strategic-off`` — R6; the gate REQUIRES it on every v7f stage;
#:   * ``--goal-multilabel`` — R3's head structure (0 params, 0 keys). The
#:     gate's rehearsed S-W predecessor keeps it, and S-W carrying it makes S-T's
#:     carried geometry equal to its own flag (no declared change at the seam);
#:   * ``--newest-frame-only`` — feeds the newest 3-channel frame (the trainer
#:     refuses it without ``--in-channels 3``). Geometry, carried to S-T.
V7F_SW_LEVERS: tuple[str, ...] = ("--strategic-off", "--goal-multilabel",
                                  "--newest-frame-only")
#: the ChainConfig fields ONLY the v7f profile reads (hidden from the v6 view)
V7F_CONFIG_FIELDS: tuple[str, ...] = ("profile", "nav_labels", "s2_labels",
                                      "speed_max_sidecar", "tac_op_cond")
#: the v7f-only CLI dests (a v6 ladder REFUSES them rather than ignoring them)
V7F_CLI_DESTS: tuple[str, ...] = ("nav_labels", "s2_labels",
                                  "speed_max_sidecar", "tac_op_cond")
#: the data inputs, per stage, that the v7f argv must name (the gate's rules do
#: not require the PATHS — the trainer's preflight does, one launch later)
V7F_DATA_FLAGS: dict[str, tuple[str, ...]] = {
    "S-W": ("--nav-labels",),
    "S-T": ("--nav-labels", "--s2-labels", "--speed-max-sidecar-v6")}


def v7f_sw_dir(cfg: ChainConfig) -> str:
    """The v7f S-W directory — DISTINCT from the v6F ladder's by construction."""
    return f"v7f-SW-{cfg.sw_steps // 1000}k"


def v7f_config(**kw) -> ChainConfig:
    """A :class:`ChainConfig` with the v7f profile's DEFAULTS applied.

    ⛔ ``ChainConfig()`` defaults to the v6 ladder (``tac_goal_cond=True``, the
    fan of 8, ``sw_dir="v6F-SW-30k"``) and :func:`assert_v7f_config` REFUSES each
    of those under v7f by name — so a caller who flips only ``profile`` gets a
    refusal naming R6/R5/R2, never a silently mixed ladder. This is the one
    constructor that sets all of them together."""
    c = ChainConfig(profile="v7f", tac_goal_cond=False, n_candidates=1)
    for k, v in kw.items():
        if not hasattr(c, k):
            raise ChainRefusal(f"[chain] ⛔ ChainConfig has no field {k!r}")
        setattr(c, k, v)
    if "sw_dir" not in kw:
        c.sw_dir = v7f_sw_dir(c)
    return c


def config_view(cfg: ChainConfig) -> dict:
    """``asdict(cfg)`` for the ``plan`` output — WITHOUT the v7f-only fields on
    a v6 ladder, so the v6 ``plan`` prints byte-identically to the pre-profile
    chain."""
    d = asdict(cfg)
    if getattr(cfg, "profile", "v6") == "v6":
        for k in V7F_CONFIG_FIELDS:
            d.pop(k, None)
    return d


def assert_v7f_config(cfg: ChainConfig) -> dict:
    """⛔ Every v7f refusal decidable from the CONFIG alone, all named at once."""
    probs: list[str] = []
    if cfg.tac_goal_cond:
        probs.append(f"tac_goal_cond is ON — R6: {V7F_REQUIREMENTS['R6']}. The "
                     f"g_str->P_T port is the other strategic->tactical route; "
                     f"V6Config.__post_init__ refuses it together with "
                     f"--strategic-off, so the S-T line would die at BUILD.")
    if int(cfg.n_candidates) != 1:
        probs.append(f"n_candidates is {cfg.n_candidates} — R5: "
                     f"{V7F_REQUIREMENTS['R5']}.")
    if cfg.st_arms:
        probs.append(f"--st-arms {list(cfg.st_arms)} — R5 is option B (ONE "
                     f"plan, --selector none) and SEL-1 is REFUSED "
                     f"({SEL1_ADMISSION['fired']}).")
    if cfg.st_winner:
        probs.append(f"--st-winner {cfg.st_winner} — R6: there is no S-S/S-J "
                     f"to continue an S-T lineage, and no arm to choose.")
    if cfg.tac_op_cond not in V7F_TAC_OP_COND_MODES:
        probs.append(f"--tac-op-cond {cfg.tac_op_cond!r} — R4: "
                     f"{V7F_REQUIREMENTS['R4']}; admissible "
                     f"{list(V7F_TAC_OP_COND_MODES)}.")
    for flag, v, req in (("--nav-labels", cfg.nav_labels, "R2"),
                         ("--s2-labels", cfg.s2_labels, "R3"),
                         ("--speed-max-sidecar-v6", cfg.speed_max_sidecar,
                          "R1")):
        if not (v and str(v).strip()):
            probs.append(f"{flag} is REQUIRED — {req}: "
                         f"{V7F_REQUIREMENTS[req]}. A v7f line without it is "
                         f"refused by the trainer's preflight one launch later "
                         f"(or trains the lever on nothing).")
    base = posixpath.basename(posixpath.normpath(
        str(cfg.sw_dir).replace("\\\\", "/")))
    if base.startswith("v6F-"):
        probs.append(f"--sw-dir {cfg.sw_dir} is a v6F directory — R2: the v6F "
                     f"S-W never built nav.*, and S-T may not INTRODUCE nav.*, "
                     f"so a v7f S-T cannot continue it. The v7f S-W is its own "
                     f"run ({v7f_sw_dir(cfg)}).")
    if probs:
        raise ChainRefusal(
            "[chain] ⛔ the v7f profile REFUSES this configuration "
            "(PI R1-R6, 2026-09-27):\\n  - " + "\\n  - ".join(probs))
    return {"ok": True}


def assert_v7f_step_request(key: str | None) -> dict:
    """⛔ S-S / S-J under v7f are refused BY NAME (R6), not as "no such step"."""
    if key and key.split(":")[0] in V7F_REFUSED_STAGES:
        raise ChainRefusal(
            f"[chain] ⛔ {key} is REFUSED under --profile v7f — R6: "
            f"{V7F_REQUIREMENTS['R6']}. The first v7F experiments are S-W -> "
            f"S-T only; the strategic stages are out of scope until the PI "
            f"turns the strategic layer back on.")
    return {"ok": True}


def _gate_v7f():
    """The launch gate's ``v7f`` profile and its ONE argv-rule function —
    IMPORTED, never re-implemented. ``launch_gate`` imports only the standard
    library at module level, so this keeps the chain torch-free.
    ⛔ Unimportable is a REFUSAL: a v7f line the gate's rules never saw must not
    leave the chain looking checked."""
    try:
        import launch_gate as LG                          # noqa: PLC0415
    except Exception as e:                                # noqa: BLE001
        raise ChainRefusal(
            f"[chain] ⛔ launch_gate.py is not importable beside the chain "
            f"({type(e).__name__}: {e}); a v7f line is never emitted unchecked.")
    prof = (getattr(LG, "PROFILES", None) or {}).get("v7f")
    fn = getattr(LG, "profile_argv_rules", None)
    if prof is None or fn is None:
        raise ChainRefusal(
            "[chain] ⛔ launch_gate has no PROFILES['v7f'] / profile_argv_rules — "
            "the gate's v7f argv rules cannot be run on this line, and the "
            "chain does not re-implement them.")
    return LG, prof, fn


def v7f_argv_rules(step, argv) -> dict:
    """BOTH rule sets on one emitted argv, never raising.

    * ``gate_refusals`` / ``pi_pending`` — ``launch_gate.profile_argv_rules(
      PROFILES["v7f"], argv)``: R1-R4, R6 and SEL-1 as the BINDING gate states
      them (``pi_pending`` carries R4's mode = PI decision D1);
    * ``chain_refusals`` — the rows the gate does NOT encode: R5 (option B) and
      the per-stage data paths (:data:`V7F_DATA_FLAGS`).
    """
    LG, prof, fn = _gate_v7f()
    argv = list(argv)
    gate_refusals, pi_pending = fn(prof, argv)
    chain: list[str] = []
    fv = LG.flag_values
    if fv(argv, "--n-candidates") != ["1"]:
        chain.append(f"R5: --n-candidates must be 1 on EVERY stage; the argv "
                     f"has {fv(argv, '--n-candidates')}")
    if step.stage == "S-T":
        for flag, want in (("--proposals", ["query"]), ("--selector", ["none"])):
            if fv(argv, flag) != want:
                chain.append(f"R5: S-T must pass {flag} {want[0]} (option B); "
                             f"the argv has {fv(argv, flag)}")
    w_sel = fv(argv, "--w-select")
    if w_sel and w_sel != ["0"] and w_sel != ["0.0"]:
        chain.append(f"R5: --w-select {w_sel} — option B has no selector to "
                     f"train")
    for flag in V7F_DATA_FLAGS.get(step.stage, ()):
        v = fv(argv, flag)
        if not v or not str(v[0]).strip():
            chain.append(f"{flag} is REQUIRED on a v7f {step.stage} line")
    return {"stage": step.stage, "gate": "launch_gate.PROFILES['v7f'] via "
                                         "profile_argv_rules (imported)",
            "gate_refusals": list(gate_refusals),
            "pi_pending": list(pi_pending), "chain_refusals": chain,
            "ok": not gate_refusals and not chain}


def assert_v7f_argv(step, argv) -> dict:
    """⛔ No v7f line leaves the chain with a rule refusal. PI-pending (D1) is
    REPORTED, not refused — the PI's decision is not the chain's to make."""
    rep = v7f_argv_rules(step, argv)
    if not rep["ok"]:
        raise ChainRefusal(
            f"[chain] ⛔ the emitted v7f {step.key} line breaks the PI's R1-R6:"
            f"\\n  - " + "\\n  - ".join(rep["gate_refusals"]
                                      + rep["chain_refusals"])
            + "\\n  The launch gate would refuse this argv too; the chain does "
              "not emit it.")
    return rep


def _build_plan_v7f(cfg: ChainConfig) -> tuple[ChainStep, ...]:
    """The v7f ladder: S-W -> S-T, nothing above (R6). Pure, like build_plan.

    ⚠️ WHY TWO S-T EXTRAS OF THE v6 LADDER ARE HANDLED DIFFERENTLY UNDER OPTION B:
    * ``--plan-wta-eps`` is DROPPED. ``v6_loss_step`` builds the epsilon-relaxed
      loser term only ``if n > 1`` (``train_v6_staged.py``, the ``plan_wta_eps``
      block): at ``--n-candidates 1`` there is no losing candidate, so any
      epsilon > 0 is an advertised-but-inert flag. (It is a geometry dest, so
      S-T still carries S-W's own recorded value — the trainer default 0.0.)
    * ``--w-t1 1.0`` is KEPT. The tactical latent-prediction term trains
      ``layer_tac``'s dynamics and is IN FORCE in the gate's rehearsed S-T
      (G-LIVE terms plan/seam/t1) at the trainer default 1.0; stating it pins
      that value against a default drift (E1's rule). Under R6 it no longer
      feeds a g_str->P_T port (none is built).
    """
    assert_v7f_config(cfg)
    nav = ("--nav-cond", "--nav-labels", str(cfg.nav_labels))
    sw_extra = nav + V7F_SW_LEVERS + (
        () if (cfg.dry and cfg.tiny) else V7F_SW_GEOMETRY)
    st_extra = (("--proposals", "query", "--selector", "none", "--w-t1", "1.0")
                + nav
                + ("--strategic-off",
                   "--max-speed-input-v6", "--plan-vmax-cap",
                   "--speed-max-sidecar-v6", str(cfg.speed_max_sidecar),
                   "--w-tac-label-all", str(V7F_W_TAC_LABEL_ALL),
                   "--goal-multilabel", "--s2-labels", str(cfg.s2_labels),
                   "--tac-op-cond", str(cfg.tac_op_cond)))
    return (
        ChainStep(
            key="S-W", stage="S-W", arm=None, out=cfg.path(cfg.sw_dir),
            steps=cfg.sw_steps, lr=cfg.lr, selector="none", w_select=0.0,
            init_from_key=None, prev_gate_key=None, max_horizon=None,
            tac_goal_cond=False, extra=sw_extra,
            note="v7F WORLD STAGE — a NEW run, not the live v6F S-W. R2: nav "
                 "trains HERE (S-T may not introduce nav.*). R6: strategic "
                 "OFF. R5: --n-candidates 1, or S-T's --init-from dies on the "
                 "cand_queries shape. lambda_plan == 0; R1/R3/R4 are S-T levers "
                 "the trainer REFUSES here."),
        ChainStep(
            key="S-T", stage="S-T", arm=None,
            out=cfg.path(f"v7f-ST-{cfg.st_steps // 1000}k"),
            steps=cfg.st_steps, lr=cfg.lr, selector="none", w_select=0.0,
            init_from_key="S-W", prev_gate_key="S-W", max_horizon=60,
            tac_goal_cond=False, extra=st_extra,
            note="v7F PLANNER STAGE, option B (R5): ONE combined tactical+"
                 "operative 6 s plan (--n-candidates 1 --proposals query "
                 "--selector none). R1 max-speed input + cap, R3 all tactical "
                 "labels, R4 --tac-op-cond, R6 strategic OFF. ⚠️ The R4 mode "
                 "is PI DECISION D1 (open): the launch gate mints PI-DECISION, "
                 "never PASS, until it is recorded."),
    )


def v7f_summary(cfg: ChainConfig) -> dict:
    """What ``plan --profile v7f`` adds: the requirements, the open decision,
    and the data pairing the trainer will enforce. Static — ``plan`` must work
    off-pod, so no launch line (and no geometry source) is needed for it."""
    return {
        "requirements": V7F_REQUIREMENTS,
        "refused_stages": list(V7F_REFUSED_STAGES),
        "pi_decision_D1": {"flag": "--tac-op-cond", "value": cfg.tac_op_cond,
                           "admissible": list(V7F_TAC_OP_COND_MODES),
                           "status": "OPEN — the gate mints PI-DECISION, never "
                                     "PASS, until the PI records it"},
        "data": {"nav_labels": cfg.nav_labels, "s2_labels": cfg.s2_labels,
                 "speed_max_sidecar": cfg.speed_max_sidecar,
                 "_pairing": "the sidecar's source_md5 must equal md5("
                             "--nav-labels) — read_speed_max_sidecar_v6 refuses "
                             "otherwise, at the trainer's startup"},
        "argv_rules": "every emitted line passes launch_gate.profile_argv_rules("
                      "PROFILES['v7f']) + the chain's R5/data rows "
                      "(assert_v7f_argv) or is not emitted",
    }


def assert_no_tilde(cfg: ChainConfig) -> dict:''')

# =====================================================================================================
# 5. the tilde refusal covers v7f's three data paths (None under v6 -> unchanged)
# =====================================================================================================
rep('''           (("--root", cfg.root), ("--workdir", cfg.workdir),
            ("--train-cache", cfg.train_cache), ("--val-cache", cfg.val_cache))
           if "~" in str(v)}''',
    '''           (("--root", cfg.root), ("--workdir", cfg.workdir),
            ("--train-cache", cfg.train_cache), ("--val-cache", cfg.val_cache),
            # v7f's three data inputs — None on a v6 ladder, so v6 is unchanged
            ("--nav-labels", getattr(cfg, "nav_labels", None)),
            ("--s2-labels", getattr(cfg, "s2_labels", None)),
            ("--speed-max-sidecar-v6", getattr(cfg, "speed_max_sidecar", None)))
           if "~" in str(v)}''')

# =====================================================================================================
# 6. build_plan dispatches on the profile (v6 path untouched)
# =====================================================================================================
rep('''    """Resolve the ladder. Pure — no filesystem, no torch."""
    assert_no_tilde(cfg)
''', '''    """Resolve the ladder. Pure — no filesystem, no torch."""
    assert_no_tilde(cfg)
    profile = getattr(cfg, "profile", "v6")
    if profile not in PROFILES:
        raise ChainRefusal(f"[chain] ⛔ unknown --profile {profile!r}; have "
                           f"{list(PROFILES)}")
    if profile == "v7f":
        return _build_plan_v7f(cfg)
''')

# =====================================================================================================
# 7. a v7f S-W never carries an older record (its geometry is DECLARED)
# =====================================================================================================
rep('''    if not step.init_from_key and not cfg.geometry_from:
        return []              # S-W starts the ladder; it HAS no ancestor''',
    '''    # ⛔ v7f: S-W's geometry is DECLARED (V7F_SW_GEOMETRY, the gate's rehearsed
    # block) and --geometry-from is ONLY S-T's off-box stand-in for S-W's own
    # record. Carrying an older run's record into a fresh v7f S-W would also
    # carry its LEVER values (e.g. `--tac-op-cond off`, which the gate refuses)
    # into a stage that must not name them.
    if not step.init_from_key and (not cfg.geometry_from
                                   or getattr(cfg, "profile", "v6") == "v7f"):
        return []              # S-W starts the ladder; it HAS no ancestor''')

# =====================================================================================================
# 8. assert_may_launch reports the v7f rule verdict on the argv it adjudicates
# =====================================================================================================
rep('''    if cfg is not None:
        report["geometry"] = assert_geometry_carry(
            step, plan,
            trainer_argv(step, cfg, plan, allow_inconclusive=allow_inconclusive,
                         off_reason=off_reason), cfg)
    return report''',
    '''    if cfg is not None:
        argv = trainer_argv(step, cfg, plan,
                            allow_inconclusive=allow_inconclusive,
                            off_reason=off_reason)
        report["geometry"] = assert_geometry_carry(step, plan, argv, cfg)
        if getattr(cfg, "profile", "v6") == "v7f":
            # the verdict of the argv being launched, D1 included (trainer_argv
            # has already REFUSED any rule break, so this is the clean record)
            report["v7f_rules"] = v7f_argv_rules(step, argv)
    return report''')

# =====================================================================================================
# 9. trainer_argv: no v7f line leaves the chain unchecked
# =====================================================================================================
rep('''    return head + carried_geometry(
        step, cfg, plan, skip=_flags_declared(head + av)) + av''',
    '''    argv = head + carried_geometry(
        step, cfg, plan, skip=_flags_declared(head + av)) + av
    if getattr(cfg, "profile", "v6") == "v7f":
        # ⛔ v7f: EVERY emitted line — dry or real, launch_line, manifest or
        # run — passes the launch gate's own v7f argv rules (imported) plus the
        # chain's R5/data rows, or it is REFUSED here. This is the regression
        # guard for a step that was edited after build_plan (e.g. a
        # tac_goal_cond=True step emitting --tac-goal-cond).
        assert_v7f_argv(step, argv)
    return argv''')

# =====================================================================================================
# 10. _cfg_from_args: the v7f profile's defaults, and v7f-only flags refused on v6
# =====================================================================================================
rep('''    if getattr(a, "a40", False):
        cfg.batch, cfg.s_per_step = A40_BATCH, A40_S_PER_STEP
    return cfg''',
    '''    if getattr(a, "a40", False):
        cfg.batch, cfg.s_per_step = A40_BATCH, A40_S_PER_STEP
    profile = getattr(a, "profile", None) or "v6"
    given = {d: getattr(a, d, None) for d in V7F_CLI_DESTS
             if getattr(a, d, None) is not None}
    if profile != "v7f":
        if given:
            raise ChainRefusal(
                f"[chain] ⛔ {sorted(given)} are v7f-profile inputs, and this is "
                f"a --profile {profile} ladder, which would read none of them. "
                f"Pass --profile v7f, or drop them.")
        return cfg
    cfg.profile = "v7f"
    for d, v in given.items():
        setattr(cfg, d, v)
    if getattr(a, "sw_dir", None) is None:
        cfg.sw_dir = v7f_sw_dir(cfg)      # never the v6F live run's directory
    if getattr(a, "n_candidates", None) is None:
        cfg.n_candidates = 1              # R5 — overrides the dry fan of 3
    # R6: the g_str->P_T port is not a v7f lever. The CLI has no way to turn it
    # ON; a programmatic tac_goal_cond=True is refused by assert_v7f_config.
    cfg.tac_goal_cond = False
    return cfg''')

# =====================================================================================================
# 11. the CLI
# =====================================================================================================
rep('''    ap.add_argument("--unpaired-arm-reason", default="")
    return ap''',
    '''    ap.add_argument("--unpaired-arm-reason", default="")
    # ---- ⭐ the v7F profile (PI R1-R6, 2026-09-27) ----------------------------
    ap.add_argument("--profile", choices=PROFILES, default="v6",
                    help="'v6' (default): the v6F ladder, byte-identical. "
                         "'v7f': S-W -> S-T only, strategic OFF, ONE combined "
                         "6 s plan (option B); every line checked against the "
                         "launch gate's v7f argv rules. S-S/S-J are REFUSED (R6).")
    ap.add_argument("--nav-labels", default=None,
                    help="v7f R2 (REQUIRED): the nav label blob --nav-cond "
                         "reads, on BOTH stages")
    ap.add_argument("--s2-labels", default=None,
                    help="v7f R3 (REQUIRED): the v7.2 s2-geom-v7 tactical "
                         "label blob")
    ap.add_argument("--speed-max-sidecar-v6", dest="speed_max_sidecar",
                    default=None,
                    help="v7f R1 (REQUIRED): the refcv6 max-speed sidecar. "
                         "Its source_md5 must equal md5(--nav-labels).")
    ap.add_argument("--tac-op-cond", choices=("off",) + V7F_TAC_OP_COND_MODES,
                    default=None,
                    help=f"v7f R4 mode — PI DECISION D1 (open). Default "
                         f"{V7F_TAC_OP_COND_DEFAULT!r}; 'off' is REFUSED (R4).")
    return ap''')

# =====================================================================================================
# 12. main: S-S/S-J refused by name under v7f; plan view; wall-clock; manifests for S-W
# =====================================================================================================
rep('''    plan = build_plan(cfg)
    assert_plan(plan)
    steps = ([step_by_key(plan, a.step)] if a.step else list(plan))''',
    '''    plan = build_plan(cfg)
    assert_plan(plan)
    if cfg.profile == "v7f":
        for k in (a.step, a.stop_after):
            assert_v7f_step_request(k)
    steps = ([step_by_key(plan, a.step)] if a.step else list(plan))''')
rep('''        out = {"root": cfg.root, "config": asdict(cfg),''',
    '''        out = {"root": cfg.root, "config": config_view(cfg),''')
rep('''                   sum(s.steps for s in plan if s.key != "S-W")''',
    '''                   # v6: S-W is the LIVE run; v7f: S-W is a new run too
                   sum(s.steps for s in plan
                       if s.key != "S-W" or cfg.profile == "v7f")''')
rep('''                                  "S-W's rate applied to stages that have "
                                  "never run)"}
        print(json.dumps(out, indent=1))''',
    '''                                  "S-W's rate applied to stages that have "
                                  "never run)"}
        if cfg.profile == "v7f":
            out["v7f"] = v7f_summary(cfg)
        print(json.dumps(out, indent=1))''')
rep('''            if s.key == "S-W":
                continue          # the live run already has its own supervisor''',
    '''            if s.key == "S-W" and cfg.profile == "v6":
                continue          # the live run already has its own supervisor''')

out = s.replace(chr(10), eol).encode("utf-8")
out_sha = hashlib.sha256(out).hexdigest()
if _lf_sha(out) != OUT_LF_SHA256:
    sys.exit(f"[apply_v7f_chain] REFUSED: the patched text hashes (LF-normalised) {_lf_sha(out)}, "
             f"not the banked {OUT_LF_SHA256}; nothing written.")
if eol == CRLF and out_sha != OUT_CRLF_SHA256:
    sys.exit(f"[apply_v7f_chain] REFUSED: the CRLF bytes hash {out_sha}, not the banked "
             f"{OUT_CRLF_SHA256}; nothing written.")
if args.check:
    print(f"[apply_v7f_chain] --check: every anchor matched exactly once; would write "
          f"{target} ({'CRLF' if eol == CRLF else 'LF'}) sha256 {out_sha} "
          f"(LF-normalised {OUT_LF_SHA256})")
    sys.exit(0)
target.write_bytes(out)
print(f"[apply_v7f_chain] patched {target} ({'CRLF' if eol == CRLF else 'LF'}) sha256 {out_sha} "
      f"(LF-normalised {OUT_LF_SHA256})")

if not args.no_test:
    src_t = HERE / "fix" / "stack" / "tests" / "test_v6_chain_v7f.py"
    dst_t = Path(args.stack) / "tests" / "test_v6_chain_v7f.py"
    if not src_t.exists():
        sys.exit(f"[apply_v7f_chain] REFUSED: {src_t} is missing (pass --no-test to patch only)")
    tsrc = src_t.read_bytes()
    if _lf_sha(tsrc) != TEST_LF_SHA256:
        sys.exit(f"[apply_v7f_chain] REFUSED: {src_t} is not the banked test file "
                 f"(LF-normalised {_lf_sha(tsrc)[:16]} != {TEST_LF_SHA256[:16]})")
    # the SAME EOL as the patched v6_chain.py beside it
    tb = tsrc.replace(CRLF.encode(), b"\n").replace(b"\n", eol.encode())
    if dst_t.exists() and _lf_sha(dst_t.read_bytes()) != TEST_LF_SHA256:
        sys.exit(f"[apply_v7f_chain] REFUSED: {dst_t} exists with DIFFERENT content; not overwriting")
    dst_t.write_bytes(tb)
    print(f"[apply_v7f_chain] installed {dst_t} ({'CRLF' if eol == CRLF else 'LF'}) sha256 "
          f"{hashlib.sha256(tb).hexdigest()} (LF-normalised {TEST_LF_SHA256})")
