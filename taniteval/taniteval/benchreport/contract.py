"""Read a BUILD_PLAN §1 run directory — the ONLY module that knows the file layout and schema.

W1 owns the schema (``taniteval/taniteval/bench/schema/{bench_run,summary}.schema.json``, validated by
W1's own ``schema_check.validate``). Anything the renderer needs beyond the frozen v1 fields
(``statistics``, ``per_speed_band``, ``primary_arm``, ``label``, the inner layout of ``per_stage`` /
``paired``) is read here with an explicit fallback, and a missing field becomes a NAMED issue that the
report shows in a banner — never a silently empty panel.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def ptr(*parts) -> str:
    """JSON-pointer-style key (RFC 6901 escaping, no leading slash) — what ``data-k`` carries."""
    return "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


class ContractError(RuntimeError):
    """The run directory cannot be rendered at all (no summary / no arms)."""


@dataclass
class Issue:
    level: str          # "critical" | "serious" | "warning"
    what: str

    def as_dict(self) -> dict:
        return {"level": self.level, "what": self.what}


@dataclass
class Run:
    root: Path
    bench: dict
    summary: dict
    scenes: dict | None
    issues: list = field(default_factory=list)
    schema_status: dict = field(default_factory=dict)

    # ---- identity -------------------------------------------------------------------------
    @property
    def protocol(self) -> str:
        return self.summary.get("protocol") or self.bench.get("protocol") or "UNDECLARED"

    @property
    def benchmark(self) -> str:
        return self.summary.get("benchmark") or self.bench.get("benchmark") or "UNDECLARED"

    @property
    def run_id(self) -> str:
        return self.summary.get("run_id") or self.bench.get("run_id") or self.root.name

    @property
    def stamps(self) -> dict:
        return self.summary.get("stamps") or self.bench.get("stamps") or {}

    @property
    def evidence_class(self) -> str:
        return self.summary.get("evidence_class") or self.stamps.get("evidence_class") or "UNDECLARED"

    @property
    def devkit(self) -> dict:
        return self.bench.get("devkit") or {}

    @property
    def floors(self) -> list:
        return list(self.summary.get("floors") or [])

    @property
    def arm_names(self) -> list:
        return list((self.summary.get("arms") or {}).keys())

    def arm(self, name: str) -> dict:
        return self.summary["arms"][name]

    def label(self, name: str) -> str:
        return self.arm(name).get("label") or name

    @property
    def model_arms(self) -> list:
        return [a for a in self.arm_names if self.arm(a).get("kind") == "model"]

    @property
    def primary_arm(self) -> str | None:
        p = self.summary.get("primary_arm")
        if p in self.arm_names:
            return p
        m = self.model_arms
        return m[0] if m else (self.arm_names[0] if self.arm_names else None)

    @property
    def is_fixture(self) -> bool:
        return bool((self.summary.get("provenance") or {}).get("fixture"))

    @property
    def is_dry_run(self) -> bool:
        """``bench_run.status == DRY_RUN_PASSED`` — a PASS that produced no scores (W1 schema, 2026-09-20)."""
        return self.bench.get("status") == "DRY_RUN_PASSED"

    @property
    def higher_is_better(self) -> bool:
        hm = self.summary.get("headline_metric") or {}
        return bool(hm.get("higher_is_better", True))

    # ---- NORMALISATION: two real layouts, one reader ---------------------------------------
    # W1's producer and W5's fixture adapter name the same quantities differently. Everything the
    # renderer asks for comes back as ``(value, pointer)`` — the pointer is the path as it REALLY is
    # in this file, so `data-k` never claims a key the run does not have.
    def stages(self) -> list:
        seen = []
        for a in self.arm_names:
            ps = _dig(self.summary, "arms", a, "per_stage")
            if isinstance(ps, dict) and "status" not in ps:
                for st in ps:
                    if st not in seen and isinstance(ps[st], dict):
                        seen.append(st)
        return sorted(seen, key=lambda s: {"stage_two": 0, "stage_one": 1}.get(s, 2))

    def stage_score(self, arm: str, stage: str) -> tuple:
        """The stage's headline number: W5 `scene_mean` (uniform) or W1 `score` (the devkit row)."""
        for key in ("scene_mean", "score"):
            v = _dig(self.summary, "arms", arm, "per_stage", stage, key)
            if _num(v):
                return v, ptr("arms", arm, "per_stage", stage, key)
        return None, None

    def stage_n(self, arm: str, stage: str):
        return _dig(self.summary, "arms", arm, "per_stage", stage, "n")

    def stage_submetric(self, arm: str, stage: str, key: str) -> tuple:
        v = _dig(self.summary, "arms", arm, "per_stage", stage, "submetrics", key)
        if _num(v):
            return v, ptr("arms", arm, "per_stage", stage, "submetrics", key)
        suffix = {"stage_one": "_s1", "stage_two": "_s2"}.get(stage)
        if suffix:                                   # W1: one flat `submetrics.combined_row` with _s1/_s2
            v = _dig(self.summary, "arms", arm, "submetrics", "combined_row", key + suffix)
            if _num(v):
                return v, ptr("arms", arm, "submetrics", "combined_row", key + suffix)
        return None, None

    def stage_rate(self, arm: str, stage: str, block: str, key: str) -> tuple:
        v = _dig(self.summary, "arms", arm, "per_stage", stage, block, key)
        return (v, ptr("arms", arm, "per_stage", stage, block, key)) if _num(v) else (None, None)

    def logs(self) -> list:
        out = []
        for a in self.arm_names:
            pl = _dig(self.summary, "arms", a, "per_log")
            if not isinstance(pl, dict):
                continue
            inner = pl.get("logs") if isinstance(pl.get("logs"), dict) else pl
            for k, v in inner.items():
                if not str(k).startswith("_") and isinstance(v, dict) and k not in out:
                    out.append(k)
        return sorted(out)

    def per_log(self, arm: str, log: str, stage: str | None = None) -> tuple:
        """W5: `per_log.logs.<log>.value`; W1: `per_log.<log>.<stage>.score_mean`."""
        v = _dig(self.summary, "arms", arm, "per_log", "logs", log, "value")
        if _num(v):
            return v, ptr("arms", arm, "per_log", "logs", log, "value")
        for st in ([stage] if stage else ["stage_two", "stage_one"]):
            v = _dig(self.summary, "arms", arm, "per_log", log, st, "score_mean")
            if _num(v):
                return v, ptr("arms", arm, "per_log", log, st, "score_mean")
        return None, None

    def per_log_n(self, arm: str, log: str, stage: str | None = None):
        v = _dig(self.summary, "arms", arm, "per_log", "logs", log, "n")
        if _num(v):
            return v
        for st in ([stage] if stage else ["stage_two", "stage_one"]):
            v = _dig(self.summary, "arms", arm, "per_log", log, st, "n")
            if _num(v):
                return v
        return None

    def family_metric(self, arm: str, fam: str, path: str) -> tuple:
        """A family metric as (value | refusal-dict | None, pointer, scope_tag).

        Three real layouts: W5's fixture nests instrument blocks under ``scopes.<scope>``; W1 emits a
        flat ``metrics`` dict; some blocks hold the leaf directly. A leaf may also be a REFUSAL
        (``{status: UNAVAILABLE, reason, n, n_steps_total, min_ds_m}`` — W2's stationary-plan shape),
        which is returned as-is so the renderer can print the reason instead of a blank."""
        parts = path.split("/")
        fb = _dig(self.summary, "arms", arm, "families", fam)
        if not isinstance(fb, dict):
            return None, None, None
        scopes = fb.get("scopes")
        if isinstance(scopes, dict):
            for sc in sorted(scopes, key=lambda s: {"stage_one": 0, "stage_two": 1}.get(s, 2)):
                v = _dig(scopes.get(sc), *parts)
                if _num(v) or (isinstance(v, dict) and v.get("status")):
                    return v, ptr("arms", arm, "families", fam, "scopes", sc, *parts), sc
        for base in (("metrics",), ()):
            v = _dig(fb, *base, *parts)
            if _num(v) or (isinstance(v, dict) and v.get("status")):
                return v, ptr("arms", arm, "families", fam, *base, *parts), fb.get("scope_tag")
        return None, None, None

    def family_field(self, arm: str, fam: str, key: str) -> tuple:
        """A family-level annotation (``_cross_is``, ``_along_mae_m_for_context``, …) wherever it sits."""
        fb = _dig(self.summary, "arms", arm, "families", fam)
        if not isinstance(fb, dict):
            return None, None
        if key in fb:
            return fb[key], ptr("arms", arm, "families", fam, key)
        for holder in ("metrics",):
            if isinstance(fb.get(holder), dict) and key in fb[holder]:
                return fb[holder][key], ptr("arms", arm, "families", fam, holder, key)
        for sc, blk in (fb.get("scopes") or {}).items():
            if isinstance(blk, dict) and key in blk:
                return blk[key], ptr("arms", arm, "families", fam, "scopes", sc, key)
        return None, None

    def paired(self, arm: str, floor: str) -> dict:
        p = _dig(self.summary, "arms", arm, "paired", floor)
        return p if isinstance(p, dict) else {}

    def paired_stages(self, arm: str, floor: str) -> list:
        bs = self.paired(arm, floor).get("by_stage")
        return sorted(bs, key=lambda s: {"stage_two": 0, "stage_one": 1}.get(s, 2)) if isinstance(bs, dict) else []

    def paired_counts(self, arm: str, floor: str, stage: str | None = None) -> dict | None:
        """{wins, ties, losses} as (value, pointer) + the SCOPE they were counted over.

        ⛔ On a two-stage protocol the stages have DIFFERENT token sets, so a pooled count and a
        per-stage count are different quantities (W1, 2026-09-20). Whichever is returned carries its
        scope text, and the renderer prints that text beside it."""
        p = self.paired(arm, floor)
        base = ["arms", arm, "paired", floor]
        src, scope = p, p.get("_wtl_scope") or (f"per-scene, {p.get('scope')}" if p.get("scope") else
                                                "per-scene, scope not declared by the run")
        if stage is not None:
            src = _dig(p, "by_stage", stage) or {}
            base += ["by_stage", stage]
            scope = f"per-scene, {stage.replace('_', ' ')} only"
        if not all(_num(src.get(k)) for k in ("wins", "ties", "losses")):
            return None
        n = src.get("n_common", src.get("n"))
        n_key = "n_common" if "n_common" in src else ("n" if "n" in src else None)
        return {"wins": (src["wins"], ptr(*base, "wins")), "ties": (src["ties"], ptr(*base, "ties")),
                "losses": (src["losses"], ptr(*base, "losses")),
                "n": (n, ptr(*base, n_key)) if n_key and _num(n) else (None, None),
                "scope": scope, "pooled": stage is None}

    def paired_mean_delta(self, arm: str, floor: str, stage: str | None = None) -> tuple:
        p = self.paired(arm, floor)
        if stage is None:
            for key in ("scene_mean_delta",):
                if _num(p.get(key)):
                    return p[key], ptr("arms", arm, "paired", floor, key)
            return None, None
        b = _dig(p, "by_stage", stage) or {}
        for key in ("score_mean_delta", "scene_mean_delta"):
            if _num(b.get(key)):
                return b[key], ptr("arms", arm, "paired", floor, "by_stage", stage, key)
        return None, None

    def paired_stat_delta(self, arm: str, floor: str, name: str = "S2_EPDMS_u") -> tuple:
        p = self.paired(arm, floor)
        v = _dig(p, "statistic_deltas", name)
        if _num(v):
            return v, ptr("arms", arm, "paired", floor, "statistic_deltas", name)
        v = p.get(f"{name}_delta")
        if _num(v):
            return v, ptr("arms", arm, "paired", floor, f"{name}_delta")
        return None, None

    def paired_sub_deltas(self, arm: str, floor: str, stage: str | None = None) -> dict:
        p = self.paired(arm, floor)
        if stage is None:
            d = p.get("submetric_deltas")
            base = ["arms", arm, "paired", floor, "submetric_deltas"]
        else:
            d = _dig(p, "by_stage", stage, "submetric_mean_deltas") or _dig(p, "by_stage", stage, "submetric_deltas")
            base = ["arms", arm, "paired", floor, "by_stage", stage,
                    "submetric_mean_deltas" if _dig(p, "by_stage", stage, "submetric_mean_deltas") else "submetric_deltas"]
        if not isinstance(d, dict):
            return {}
        return {k: (v, ptr(*base, k)) for k, v in d.items() if _num(v)}

    def file(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = self.root / rel
        return p if p.exists() else None


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _schema_validate(doc: dict, name: str) -> tuple[str, list]:
    try:
        from taniteval.bench.schema_check import load_schema, validate      # W1's own checker
    except Exception as e:                                                    # noqa: BLE001
        return "UNVERIFIED", [f"W1 schema checker not importable ({type(e).__name__}: {e})"]
    try:
        errs = validate(doc, load_schema(name))
    except Exception as e:                                                    # noqa: BLE001
        return "UNVERIFIED", [f"schema validation raised {type(e).__name__}: {e}"]
    return ("VALID" if not errs else "INVALID"), errs


def load_run(root: str | Path) -> Run:
    root = Path(root)
    sp, bp = root / "summary.json", root / "bench_run.json"
    if not sp.is_file():
        raise ContractError(f"{sp} not found — not a §1 run directory")
    summary = _load_json(sp)
    if not isinstance(summary.get("arms"), dict) or not summary["arms"]:
        raise ContractError(f"{sp}: no arms")
    issues: list = []
    bench = {}
    if bp.is_file():
        bench = _load_json(bp)
    else:
        issues.append(Issue("serious", "bench_run.json missing — devkit SHA / ckpt / device unknown"))
    scenes = _load_json(root / "scenes.json") if (root / "scenes.json").is_file() else None
    run = Run(root=root, bench=bench, summary=summary, scenes=scenes, issues=issues)

    for name, doc in (("summary", summary), ("bench_run", bench)):
        if not doc:
            run.schema_status[name] = {"status": "ABSENT", "errors": []}
            continue
        st, errs = _schema_validate(doc, name)
        run.schema_status[name] = {"status": st, "errors": errs}
        if st == "INVALID":
            issues.append(Issue("critical", f"{name}.json violates W1 schema v1 ({len(errs)}): "
                                            + "; ".join(errs[:4])))
        elif st == "UNVERIFIED":
            issues.append(Issue("serious", f"{name}.json schema UNVERIFIED: {errs[0]}"))

    # ---- cross-field rules the schema cannot express (W1 contract) -------------------------
    arms = summary["arms"]
    floors = run.floors
    for f in floors:
        if f not in arms:
            issues.append(Issue("critical", f"floor {f!r} is listed but is not an arm"))
    # ⭐ DRY_RUN_PASSED is a PASS that scored nothing (W1/E1 2026-09-20: a passing dry run used to say
    # REFUSED — a failure word on a success). An arm with no headline is EXPECTED here, so the mandatory
    # floor check reports it as a note, never as a BLOCKING "NOT MEASURED".
    if run.is_dry_run:
        issues.append(Issue("good", "DRY RUN PASSED — the preflight succeeded and nothing was scored; "
                                    "every headline below is UNAVAILABLE by design, not by failure"))
    if run.benchmark in ("navsim_v2", "navsim_v1"):
        for f in ("STOP", "CV"):
            a = arms.get(f)
            if a is None:
                issues.append(Issue("good" if run.is_dry_run else "critical",
                                    f"MANDATORY floor {f} is absent"
                                    + (" — expected on a dry run (no arm was scored)" if run.is_dry_run
                                       else " — the run is INCOMPLETE")))
            elif (a.get("status") != "OK"
                  or ("value" not in (a.get("headline") or {})
                      and "value" not in ((a.get("statistics") or {}).get("S2_EPDMS_u") or {}))):
                why = (a.get("headline") or {}).get("reason") or a.get("status")
                issues.append(Issue("good" if run.is_dry_run else "critical",
                                    f"MANDATORY floor {f} carries no score"
                                    + (f" — expected on a dry run: {why}" if run.is_dry_run else
                                       f" — NOT MEASURED in this run (status {a.get('status')}): {why}")))
    for n, a in arms.items():
        pr = a.get("paired") or {}
        for f in floors:
            if f not in pr:
                issues.append(Issue("serious", f"arm {n} is not paired against floor {f}"))
    if run.stamps.get("tier") in (None, ""):
        issues.append(Issue("critical", "no tier stamp"))
    if not run.stamps.get("loop"):
        issues.append(Issue("critical", "no loop stamp"))
    if bench and summary.get("protocol") != bench.get("protocol"):
        issues.append(Issue("critical", f"protocol mismatch: summary {summary.get('protocol')!r} vs "
                                        f"bench_run {bench.get('protocol')!r}"))
    if summary.get("headline_metric", {}).get("column") == "pdm_score":
        issues.append(Issue("critical", "headline reads `pdm_score` — the EPDMS is the `score` column"))
    return run
