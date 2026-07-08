"""One-lever-per-run bake-off harness + results-table generator (WP3, Arch backlog #2).

WHY THIS EXISTS
---------------
Phase 0 needs a *disciplined* way to attribute a gate delta to a single
architectural choice. Ad-hoc "change three things, rerun, eyeball the number"
is how false wins enter a codebase. This harness enforces **OFAT** (one factor
at a time): every variant is the baseline config with **exactly one** field
flipped, and the harness *verifies* that (``lever_diff``) before it will run the
variant. A lever whose ``apply`` touches more than its declared field is a bug
the harness refuses to hide.

INSTRUMENT DOCTRINE (D-004 / G-AI1)
-----------------------------------
No architecture change may be motivated by a gate that has not passed its
instrument rows. So:

* every ``Lever`` NAMES the D-gate(s) that would falsify it (``gates``) and the
  hypothesis it serves — a lever with no falsifying gate is rejected at import
  time by the test suite (``test_registry_g_ai1``);
* each variant is scored through the **existing** D1-D3 gate runner
  (``tanitad.eval.gates``), so a variant that "wins" on a gate whose instruments
  are BLOCKED contributes **no claim** — the table shows BLOCKED, never a green
  number. The harness measures; it never itself decides an architecture change.

EFFICIENCY NUMBERS (G-AI2)
--------------------------
The only efficiency figure this module reports is the **measured** parameter
count (``sum(p.numel())`` via the run function). FLOPs/decision and batch-1
latency are the FLOPs-ledger tool (backlog #5) and are deliberately *not*
synthesised here — measured and estimated numbers are never mixed.

SCOPE / HONESTY (P8)
--------------------
This is the *tool*. A **decision-grade** bake-off needs a TRAINED checkpoint:
on untrained/collapsed latents the D-gates are BLOCKED or meaningless and no
lever ranking may be read. The package proves the harness MECHANICS end-to-end
on the smoke model + synthetic latents (levers apply one-factor, the driver
aggregates across seeds with a CI, the gate runner is wired, the table renders)
— it makes **no** architecture claim. The decision-grade run is queued behind
the A40 Stage-0 checkpoint, exactly like ``p0-spectral-sizing``.

Standalone: ``tanitad`` must be importable (editable stack install). Proposed
target on integration: ``stack/tanitad/eval/bakeoff.py`` (+ test).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from tanitad.config import StackConfig
from tanitad.eval.gates import GateReport

# The three decode gates this harness knows how to attribute a lever to.
KNOWN_GATES = ("D1", "D2", "D3")


# --------------------------------------------------------------------------- #
# Lever registry                                                              #
# --------------------------------------------------------------------------- #
class PlannedLeverError(NotImplementedError):
    """Raised by a lever that needs model code that does not exist yet.

    The message points at the work package / backlog item that must land the
    mechanism before the lever can be swept. Encoding the roadmap this way keeps
    every planned lever G-AI1-complete (gate + hypothesis) without pretending it
    is runnable.
    """


@dataclass(frozen=True)
class Lever:
    """A single, gate-attributable, one-factor mutation of the stack config.

    ``apply`` MUST change exactly the field(s) named in ``isolates`` and nothing
    else — the driver asserts this with ``lever_diff`` before running. ``gates``
    lists the D-gate(s) whose PASS/FAIL would confirm/refute the lever (G-AI1).
    """

    name: str
    hypothesis: str                       # e.g. "H4/A4"
    gates: tuple[str, ...]                # G-AI1: falsifying gate(s)
    isolates: str                         # human-readable one-factor description
    apply: Callable[[StackConfig], StackConfig]
    fields: tuple[str, ...] = ()          # dotted config paths the lever changes
    implemented: bool = True              # False => apply raises PlannedLeverError
    rationale: str = ""                   # the bake-off / evidence motivating it

    def build(self, base: StackConfig) -> StackConfig:
        """Return the variant config (deep-copied base with the one factor flipped)."""
        if not self.implemented:
            raise PlannedLeverError(
                f"lever '{self.name}' needs model code that is not in the stack "
                f"yet: {self.rationale or self.isolates}")
        return self.apply(copy.deepcopy(base))


def _baseline_apply(cfg: StackConfig) -> StackConfig:
    return cfg


def baseline_lever() -> Lever:
    return Lever("baseline", "—", KNOWN_GATES, "unchanged reference config",
                 _baseline_apply, fields=(), rationale="reference arm")


def default_levers() -> list[Lever]:
    """Config-native levers — flip one existing ``StackConfig`` switch, runnable today.

    Each is a real Phase-0 design question whose answer a D-gate can arbitrate.
    """

    def _residual_off(c: StackConfig) -> StackConfig:
        c.predictor.residual = False
        return c

    def _change_weight_off(c: StackConfig) -> StackConfig:
        c.predictor.change_weighted = False
        return c

    def _global_pool(c: StackConfig) -> StackConfig:
        c.readout.grid = 1                      # 1x1 == global average pool
        return c

    def _narrow_readout(c: StackConfig) -> StackConfig:
        c.readout.d_readout = max(1, c.readout.d_readout // 2)
        return c

    def _single_horizon(c: StackConfig) -> StackConfig:
        c.predictor.horizons = (1,)             # drop MTP multi-horizon heads
        return c

    def _short_window(c: StackConfig) -> StackConfig:
        c.predictor.window = max(2, c.predictor.window // 2)
        return c

    def _tactical_off(c: StackConfig) -> StackConfig:
        c.tactical_pred = None                  # remove the tactical brain
        return c

    def _h15_off(c: StackConfig) -> StackConfig:
        c.h15.enabled = False                   # ablate the imagination field
        return c

    return [
        Lever("residual_off", "H4/A4", ("D1", "D3"),
              "predictor.residual: delta-prediction -> absolute head",
              _residual_off, fields=("predictor.residual",),
              rationale="A4 bake-off + Delta-JEPA (2606.31232) decode latent "
                        "differences; falsified if absolute head ties/wins D1/D3"),
        Lever("change_weight_off", "H4/A4", ("D1", "D3"),
              "predictor.change_weighted: change-weighted MSE -> plain MSE",
              _change_weight_off, fields=("predictor.change_weighted",),
              rationale="A4: change-weighting stops static content dominating the "
                        "latent loss; falsified if plain MSE matches D1/D3"),
        Lever("global_pool_readout", "H1/A7", ("D1",),
              "readout.grid: 4x4 spatial grid -> 1x1 global pool",
              _global_pool, fields=("readout.grid",),
              rationale="A7: spatial readout should decode ego position >= global "
                        "pooling (D1 has a built-in vs-pool ablation)"),
        Lever("narrow_readout", "H3", ("D1",),
              "readout.d_readout: halve per-cell projection dim",
              _narrow_readout, fields=("readout.d_readout",),
              rationale="capacity lever tied to p0-spectral-sizing: if the "
                        "transition spectrum is low-rank, D1 should be flat here"),
        Lever("single_horizon", "H5", ("D3",),
              "predictor.horizons: (1,2,4) -> (1,) (drop MTP heads)",
              _single_horizon, fields=("predictor.horizons",),
              rationale="H5/MTP: multi-horizon heads as training signal + decode "
                        "accelerator; falsified if single-horizon matches D3"),
        Lever("short_window", "H1", ("D1",),
              "predictor.window: halve causal history length",
              _short_window, fields=("predictor.window",),
              rationale="how much action history the operative path needs; D1 "
                        "decodability vs history length"),
        Lever("tactical_off", "H1", ("D2",),
              "tactical_pred: remove the maneuver-horizon brain",
              _tactical_off, fields=("tactical_pred",),
              rationale="does the tactical brain buy maneuver-horizon selection "
                        "(D2) over the operative path alone"),
        Lever("h15_off", "H15", ("D2",),
              "h15.enabled: ablate the imagination field",
              _h15_off, fields=("h15.enabled",),
              rationale="H15 imagination-for-selection ablation; D2 direction acc "
                        "should drop if imagination is load-bearing"),
    ]


def planned_levers() -> list[Lever]:
    """Levers that need NEW model code before they can be swept.

    They carry full G-AI1 metadata (gate + hypothesis) and a pointer to the WP
    that must land the mechanism. ``build`` raises ``PlannedLeverError`` — the
    driver lists them in a separate 'planned' section, never runs them.
    """

    def _needs_code(_c: StackConfig) -> StackConfig:                # pragma: no cover
        raise PlannedLeverError("model code not implemented")

    return [
        Lever("adaln_conditioning", "H1/H12", ("D1", "D3"),
              "predictor conditioning: FiLM -> AdaLN (pre-norm gain/shift)",
              _needs_code, fields=("predictor.cond",), implemented=False,
              rationale="AdaLN action-conditioning is reported to beat FiLM "
                        "(2512.24497; Delta-JEPA 2606.31232 injects actions via "
                        "AdaLN). Needs a CondBlock variant in predictor.py (WP-arch)"),
        Lever("rope_conditioning", "H1", ("D1", "D3"),
              "predictor attention: add rotary (RoPE) temporal position",
              _needs_code, fields=("predictor.rope",), implemented=False,
              rationale="RoPE on the causal window is standard in recent "
                        "action-conditioned latent predictors (Delta-JEPA, "
                        "OmniDreams 2606.03159); needs rotary embed in "
                        "OperativePredictor attention (WP-arch)"),
        Lever("kstep_rollout", "H5", ("D2", "D3"),
              "training: single-step -> recursive K-step rollout loss (K~4)",
              _needs_code, fields=("train.rollout_k",), implemented=False,
              rationale="multistep rollout as data-aug against compounding error; "
                        "Pareto optimum reported ~K=4 (2512.24497 real=6-step; "
                        "Delta-JEPA). Needs a rollout loop in train_worldmodel (WP3)"),
        Lever("tactical_moe_sigma", "H2/H8/H15", ("D2",),
              "tactical predictor: dense -> MoE routed on imagination epistemic sigma",
              _needs_code, fields=("tactical_pred.moe",), implemented=False,
              rationale="route the tactical/sensor MoE on ImaginationField "
                        "epistemic sigma (DriveMoE/GEMINUS + our H15 link) — gate "
                        "an expert only where imagination uncertainty is low; needs "
                        "an MoE module (WP4)"),
    ]


# --------------------------------------------------------------------------- #
# One-factor isolation: recursive dataclass diff                              #
# --------------------------------------------------------------------------- #
def _is_dc(x) -> bool:
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def lever_diff(base: StackConfig, variant: StackConfig, _prefix: str = "") -> list[str]:
    """Dotted config paths whose value differs between ``base`` and ``variant``.

    Recurses into nested dataclasses. A dataclass replaced by ``None`` (e.g.
    ``tactical_pred``) reports as a single changed path, not a deep diff.
    """
    changed: list[str] = []
    for f in dataclasses.fields(base):
        bv = getattr(base, f.name)
        vv = getattr(variant, f.name)
        path = f"{_prefix}{f.name}"
        if _is_dc(bv) and _is_dc(vv) and type(bv) is type(vv):
            changed.extend(lever_diff(bv, vv, path + "."))
        elif bv != vv:
            changed.append(path)
    return changed


# --------------------------------------------------------------------------- #
# Statistics: multi-seed mean +/- 95% CI                                       #
# --------------------------------------------------------------------------- #
def mean_ci95(xs: Sequence[float]) -> tuple[float, float]:
    """(mean, 95% CI half-width). Normal approximation 1.96*sd/sqrt(n).

    n<2 -> CI is NaN (a single seed carries no interval — Thursday's >=3-seed
    rule exists precisely because CARLA-class variance hides lever effects).
    Non-finite samples are dropped; if none remain, returns (nan, nan).
    """
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, 1.96 * math.sqrt(var) / math.sqrt(len(vals))


# --------------------------------------------------------------------------- #
# Run function contract                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class RunOutput:
    """What a ``run_fn(cfg, seed)`` returns for one variant/seed.

    ``reports`` maps gate id -> GateReport (from ``tanitad.eval.gates``).
    ``n_params`` is the MEASURED parameter count of the built model (G-AI2).
    """
    reports: Mapping[str, GateReport]
    n_params: int
    extra: dict = field(default_factory=dict)


RunFn = Callable[[StackConfig, int], RunOutput]

# A per-gate "key metric" to summarise across seeds (lower-is-better flag).
KEY_METRIC = {
    "D1": ("ade@1s", True),                 # metres, lower better
    "D2": ("direction_acc", False),         # fraction, higher better
    "D3": ("ratio", True),                  # imagined/oracle, lower better
}


# --------------------------------------------------------------------------- #
# Variant result + driver                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class VariantResult:
    lever: str
    hypothesis: str
    gates_targeted: tuple[str, ...]
    fields_changed: list[str]               # verified one-factor paths
    n_params: int | None                    # measured; None for planned levers
    status: dict[str, str]                  # gate -> PASS/FAIL/BLOCKED/MIXED
    admissible: dict[str, bool]             # gate -> all seeds admissible
    metric: dict[str, tuple[float, float]]  # gate -> (mean, ci95) of key metric
    seeds: list[int]
    planned: bool = False
    note: str = ""


def _aggregate(gate: str, reports: Sequence[GateReport]) -> tuple[str, bool, tuple]:
    """Collapse per-seed reports for one gate into (status, admissible, key metric)."""
    statuses = {r.status for r in reports}
    status = statuses.pop() if len(statuses) == 1 else "MIXED"
    admissible = all(r.admissible for r in reports)
    key, _lower = KEY_METRIC.get(gate, (None, True))
    if key is None:
        return status, admissible, (float("nan"), float("nan"))
    vals = [r.metrics.get(key) for r in reports if key in r.metrics]
    return status, admissible, mean_ci95(vals)


def run_bakeoff(levers: Sequence[Lever], run_fn: RunFn,
                seeds: Sequence[int] = (0, 1, 2),
                base_cfg: StackConfig | None = None,
                gates: Sequence[str] = KNOWN_GATES) -> list[VariantResult]:
    """Run the OFAT matrix: for each lever, build the one-factor variant and score
    it through ``run_fn`` across ``seeds``; aggregate per-gate status + key metric.

    Planned levers (needing model code) are recorded as PLANNED rows and never run.
    Raises ``ValueError`` if a runnable lever's ``apply`` is not truly one-factor.
    """
    base = base_cfg if base_cfg is not None else StackConfig()
    out: list[VariantResult] = []
    for lever in levers:
        if not lever.implemented:
            out.append(VariantResult(
                lever.name, lever.hypothesis, lever.gates, list(lever.fields),
                None, {g: "PLANNED" for g in gates}, {g: False for g in gates},
                {g: (float("nan"), float("nan")) for g in gates}, list(seeds),
                planned=True, note=lever.rationale))
            continue

        variant = lever.build(base)
        diff = lever_diff(base, variant)
        if lever.name != "baseline" and lever.fields:
            unexpected = [d for d in diff
                          if not any(d == f or d.startswith(f + ".") for f in lever.fields)]
            missing = [f for f in lever.fields
                       if not any(d == f or d.startswith(f + ".") for d in diff)]
            if unexpected or missing:
                raise ValueError(
                    f"lever '{lever.name}' is not one-factor: changed {diff}, "
                    f"declared {list(lever.fields)} "
                    f"(unexpected={unexpected}, missing={missing})")

        per_seed = [run_fn(variant, s) for s in seeds]
        n_params = per_seed[0].n_params
        status, adm, metric = {}, {}, {}
        for g in gates:
            reps = [ro.reports[g] for ro in per_seed if g in ro.reports]
            if not reps:
                status[g], adm[g], metric[g] = "N/A", False, (float("nan"), float("nan"))
                continue
            status[g], adm[g], metric[g] = _aggregate(g, reps)
        out.append(VariantResult(
            lever.name, lever.hypothesis, lever.gates, diff, n_params,
            status, adm, metric, list(seeds), planned=False, note=lever.rationale))
    return out


# --------------------------------------------------------------------------- #
# Results-table generator (markdown)                                           #
# --------------------------------------------------------------------------- #
def _fmt_metric(gate: str, mc: tuple[float, float]) -> str:
    mean, ci = mc
    if not math.isfinite(mean):
        return "—"
    key = KEY_METRIC.get(gate, (gate, True))[0]
    ci_txt = f"±{ci:.3f}" if math.isfinite(ci) else "±—"
    return f"{key}={mean:.3f}{ci_txt}"


def _delta_vs_baseline(results: Sequence[VariantResult], gate: str,
                       lever: str, base_mean: float) -> str:
    for r in results:
        if r.lever == lever:
            mean = r.metric.get(gate, (float("nan"),))[0]
            if not (math.isfinite(mean) and math.isfinite(base_mean)):
                return "—"
            lower_better = KEY_METRIC.get(gate, (None, True))[1]
            d = mean - base_mean
            arrow = ("↓better" if d < 0 else "↑worse") if lower_better \
                else ("↑better" if d > 0 else "↓worse")
            return f"{d:+.3f} {arrow if d != 0 else ''}".strip()
    return "—"


def render_table(results: Sequence[VariantResult],
                 gates: Sequence[str] = KNOWN_GATES,
                 title: str = "Bake-off results") -> str:
    """Render an OFAT results table as markdown.

    Columns per gate carry the gate STATUS (PASS/FAIL/BLOCKED — a BLOCKED cell
    is not a claim) and the key metric mean±CI. A leading G-AI1 column names the
    falsifying gate(s); a measured param column is the only efficiency figure
    (G-AI2). Planned levers get their own section with the WP pointer.
    """
    runnable = [r for r in results if not r.planned]
    planned = [r for r in results if r.planned]
    base = next((r for r in runnable if r.lever == "baseline"), None)

    lines: list[str] = [f"### {title}", ""]
    header = ["lever", "hyp", "target-gate(G-AI1)", "params(measured)"]
    for g in gates:
        header += [f"{g} status", f"{g} metric", f"{g} Δ-vs-base"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in runnable:
        row = [r.lever, r.hypothesis, "/".join(r.gates_targeted),
               f"{r.n_params:,}" if r.n_params is not None else "—"]
        for g in gates:
            base_mean = base.metric.get(g, (float("nan"),))[0] if base else float("nan")
            row += [r.status.get(g, "N/A"),
                    _fmt_metric(g, r.metric.get(g, (float("nan"), float("nan")))),
                    "ref" if r.lever == "baseline"
                    else _delta_vs_baseline(runnable, g, r.lever, base_mean)]
        lines.append("| " + " | ".join(row) + " |")

    if planned:
        lines += ["", "#### Planned levers (need model code — not runnable yet)", ""]
        lines.append("| lever | hyp | target-gate(G-AI1) | needs |")
        lines.append("|---|---|---|---|")
        for r in planned:
            lines.append(f"| {r.lever} | {r.hypothesis} | "
                         f"{'/'.join(r.gates_targeted)} | {r.note} |")

    lines += [
        "",
        "> **Doctrine (D-004/G-AI1):** each lever names the gate(s) that would "
        "falsify it; a BLOCKED cell contributes NO claim (instruments failed). "
        "D1–D3 are decode/instrument gates — necessary, not sufficient "
        "(arXiv 2512.24497); closed-loop D4–D6 arbitrate.",
        "> **Efficiency (G-AI2):** `params` are MEASURED. FLOPs/decision and "
        "batch-1 latency are the FLOPs-ledger tool (backlog #5) — not shown here, "
        "measured and estimated numbers are never mixed.",
    ]
    return "\n".join(lines)
