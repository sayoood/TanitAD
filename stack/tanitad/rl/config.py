"""Config for RL post-training — every knob recorded, none inferred.

⛔ WHY EVERY FIELD IS ON THE DATACLASS AND NOT AN ENV VAR OR A DEFAULT ARGUMENT
------------------------------------------------------------------------------
P4's SPEC §2 rule R2 exists because two arms once differed ONLY in an
environment variable and produced byte-identical `config.json`. This config is a
frozen dataclass with `to_dict()`, and `posttrain.py` writes it whole into the
run's `config.json`. There are no env-var knobs in this library, on purpose: if
a knob matters it is a field here, and if it is a field here it lands in the run
record.

Method selection is `method: grpo | dpo | awr` per the commission. Each carries
its data requirement in `REQUIREMENTS` below, because the honest answer for two
of them today is that we cannot supply the signal — and a config that lets you
select an unsupplied method without saying so is how a run produces a confident
meaningless number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .rewards import DEFAULT_WEIGHTS

METHODS = ("grpo", "dpo", "awr")

#: What each method NEEDS, and whether we can supply it today. Consulted by
#: `PostTrainConfig.validate()`, which REFUSES rather than warns.
REQUIREMENTS: dict[str, dict] = {
    "grpo": {
        "needs": "a scalar reward per sampled trajectory + >=2 samples per group",
        "have": True,
        "why": ("rule-based rewards (collision / feasibility / headway / comfort) "
                "are computable from geometry + GT scene facts, and the diffusion "
                "decoder can be sampled G times per anchor under exploration noise"),
    },
    "awr": {
        "needs": "a scalar reward + an advantage; weights an imitation loss by exp(A/beta)",
        "have": True,
        "why": ("advantage-weighted regression adds NO module and no state-dict "
                "key — it is the plan loss times exp(A/beta). Cheapest option "
                "value in the family"),
    },
    "dpo": {
        "needs": "PREFERENCE PAIRS over trajectories (a chosen and a rejected)",
        "have": False,
        "why": ("⛔ WE DO NOT HAVE THIS. A demonstration corpus carries one "
                "policy's output per state — there is no negative class. "
                "Manufacturing one by thresholding the cardinal GT distance "
                "into a binary was MEASURED to cost +0.0974 m (base) / "
                "+0.1670 m (XL), separated. Selecting `dpo` is refused until a "
                "preference-collection instrument exists (P4 backlog M-B4)."),
    },
}


@dataclass(frozen=True)
class PostTrainConfig:
    """Every knob of an RL post-training run."""

    # --- method ------------------------------------------------------------
    method: str = "grpo"
    #: samples per anchor group. ⛔ G=1 makes the centred advantage identically
    #: zero — `advantage.grpo_advantage` refuses it rather than no-op'ing.
    group_size: int = 4
    normalize: str = "none"          # "none" = Dr. GRPO-correct; "std" opt-in
    w_intra: float = 1.0
    w_inter: float = 1.0
    kl_coef: float = 0.0             # OFF: the IL loss is the anchor, not a KL
    awr_beta: float = 1.0

    # --- exploration -------------------------------------------------------
    #: DDv2 ablation: MULTIPLICATIVE exploration noise beat additive
    #: (90.1 vs 89.7 PDMS). Multiplicative scales with the offset magnitude, so
    #: it explores proportionally instead of swamping small offsets.
    noise_mode: str = "multiplicative"
    noise_scale: float = 0.1

    # --- hard safety VETO (a CONSTRAINT, applied OUTSIDE the advantage) -----
    #: ⛔ Separated from the `headway` RANKING term on the Master Mind's design
    #: ruling. One term that tried to be both saturated and went INERT (A0
    #: measured median spread 0.0000). A constraint pins a candidate; a ranking
    #: signal orders them. They are different objects.
    ttc_min_s: float = 1.5
    veto_value: float = -1.0

    # --- reward ------------------------------------------------------------
    reward_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS))
    dt: float = 0.1

    # --- optimisation ------------------------------------------------------
    lr: float = 1e-5                 # post-training: small, from a cold start
    steps: int = 1000
    batch: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    #: the imitation regulariser that keeps RL on the demonstration manifold
    #: (DDv2 keeps one; without it the policy drifts off-manifold and the
    #: rule-based reward alone cannot tell it that is bad)
    w_imitation: float = 1.0

    # --- what may move -----------------------------------------------------
    #: ⛔ MANDATORY OPTION, default ON. RL post-training fine-tunes the PLANNER
    #: HEAD. A trunk that moves under a rule-based reward invalidates every
    #: representation number measured on it, and re-opens the encoder question
    #: mid-experiment.
    freeze_trunk: bool = True

    #: ⛔ THE GENERATOR, NOT THE SELECTOR. ``core.decoder.*`` is refcv3's
    #: anchored truncated-diffusion decoder — the thing that PRODUCES the fan,
    #: and the thing DDv2's GRPO post-trains.
    #:
    #: ⚠️ CORRECTED 2026-08-29, caught by running ``select_trainable`` against a
    #: real ``RefCV3Model`` instead of a stand-in. The first version of this
    #: default was ``("scorer", "lat_head_tac", …)`` — which trains **`scorer`,
    #: the SELECTOR**, and leaves the decoder frozen. That is the exact
    #: inversion of this library's stated doctrine (``advantage.py``: *"this
    #: library targets the generator, never the selector"*), and it would have
    #: trained the one component DDv2 identifies as the weak link — selector
    #: over-reliance — using a reward built specifically to avoid it. It ran
    #: cleanly and reported 0.81 % trainable, so nothing would have failed.
    #: ⇒ Integration-check against the real model, not the mock.
    trainable_prefixes: tuple[str, ...] = ("core.decoder",)

    #: ⛔ Prefixes that must NEVER be trainable here, whatever else is set.
    #: ``select_trainable`` refuses if one of these ends up requiring grad.
    forbidden_prefixes: tuple[str, ...] = ("scorer",)

    #: Prefixes EXCLUDED from training even when a trainable prefix covers them
    #: — skipped and RECORDED, not refused. Needed for the v2.1 pilot, where the
    #: selector surface (``decoder.conf_head``, 257 params) lives INSIDE the
    #: decoder: "train the decoder" must not mean "train the selector".
    exclude_prefixes: tuple[str, ...] = ()

    #: Denoise steps the sampler asks the model for. 0 = classifier pass;
    #: the v2.1 REF-C deploys truncated diffusion at 2 (its own config).
    decoder_steps: int = 0

    # --- bookkeeping -------------------------------------------------------
    out_dir: str = ""
    run_name: str = "refcv3-rl-posttrain"
    save_every: int = 250

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        req = REQUIREMENTS[self.method]
        if not req["have"]:
            raise ValueError(
                f"method {self.method!r} is REFUSED: {req['why']} "
                f"(needs: {req['needs']})")
        if self.method == "grpo" and self.group_size < 2:
            raise ValueError(
                f"grpo needs group_size >= 2, got {self.group_size}: a group of "
                "one gives an identically-zero advantage and a SILENT no-op.")
        if self.normalize not in ("none", "std"):
            raise ValueError(f"normalize must be 'none'|'std', got {self.normalize!r}")
        if self.noise_mode not in ("multiplicative", "additive"):
            raise ValueError(f"noise_mode must be 'multiplicative'|'additive', "
                             f"got {self.noise_mode!r}")
        if not self.freeze_trunk:
            # allowed, but it must be a decision someone typed
            pass
        if not self.reward_weights:
            raise ValueError("a reward with no components is not a reward")
        if self.reward_weights.get("gt_similarity", 0.0) > 0.0 and self.w_imitation > 0.0:
            raise ValueError(
                "DOUBLE-COUNTED IMITATION SIGNAL: `gt_similarity` has weight "
                f"{self.reward_weights['gt_similarity']} INSIDE the reward while "
                f"w_imitation={self.w_imitation} adds an imitation loss OUTSIDE "
                "it. Beyond the double count, the in-reward copy sits inside a "
                "GROUP-RELATIVE ADVANTAGE computed across the candidates of one "
                "fan: it gives the highest advantage to whichever candidate is "
                "nearest the single logged expert path and pushes mass off every "
                "other mode — i.e. it is a FAN-COLLAPSE objective, destroying the "
                "raw-fan quality this method exists to buy (DDv2 holds 84.4 "
                "top-10 where DiffusionDrive falls to 75.3). Keep the imitation "
                "anchor OUTSIDE the advantage (L = L_RL + lambda*L_IL): set "
                "w_imitation and drop gt_similarity from reward_weights, or set "
                "w_imitation=0 and accept the fan-collapse risk deliberately.")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_requirements"] = REQUIREMENTS[self.method]
        d["_evidence_class"] = "MEASURED (ours; this run's own configuration)"
        d["_tier"] = ("training-side config; capability claims are T1 only "
                      "(EVAL_DOCTRINE.md)")
        return d
