"""Load every input the leaderboard prints — and refuse inputs that would break a binding rule.

Nothing here computes a new statistic: values are READ (JSON pointers into raw artifacts, W1 bench
summaries, the published-results file) and only re-signed / scaled where the source's own
convention requires it (``invert`` for a floor-minus-model delta; x100 for a NavSim fraction).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # <repo>/taniteval/taniteval/leaderboard/sources.py
CONFIG = Path(__file__).with_name("leaderboard_sources.json")

WARMUP = "EPDMS_v2_warmup_two_stage"
NAVSIM_PROTOCOLS = {"EPDMS_v2_navhard_two_stage", "EPDMS_v2_warmup_two_stage", "EPDMS_v2_navtest_single_stage",
                    "EPDMS_v2_private_test_hard_two_stage", "PDMS_v1_navtest"}


class SourceError(RuntimeError):
    """An input is missing, malformed, or would make the page break a binding rule."""


def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SourceError(f"missing input: {path}") from e


def jptr(doc, pointer: str):
    """RFC-6901 JSON pointer (``~1`` = '/', ``~0`` = '~'). Raises SourceError naming the pointer."""
    cur = doc
    if pointer in ("", "/"):
        return cur
    for raw in pointer.lstrip("/").split("/"):
        tok = raw.replace("~1", "/").replace("~0", "~")
        try:
            cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise SourceError(f"pointer {pointer!r} fails at {tok!r}") from e
    return cur


def short_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def rel(path: Path, root: Path) -> str:
    """Repo-relative POSIX path; an input outside the repo (tests) keeps its absolute path."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def load_config(root: Path) -> dict:
    return load_json(CONFIG)


# ---------------------------------------------------------------- published (external) numbers
def load_published(root: Path, cfg: dict) -> dict:
    pub = load_json(root / cfg["published_results"])
    protos = pub["protocols"]
    seen = set()
    for r in pub["results"]:
        if r["id"] in seen:
            raise SourceError(f"published_results: duplicate id {r['id']}")
        seen.add(r["id"])
        if r["protocol"] not in protos:
            raise SourceError(f"published_results: {r['id']} has an unknown protocol tag {r['protocol']!r}")
        src = r.get("source", {})
        if r.get("evidence_class", pub.get("evidence_class_default")) == "PUBLISHED":
            if not (src.get("library_key") and src.get("table") and src.get("page")):
                raise SourceError(f"published_results: {r['id']} is PUBLISHED without library key + table + page")
        if r["protocol"] in NAVSIM_PROTOCOLS and r["protocol"] != "PDMS_v1_navtest":
            side = r.get("harness", {}).get("fix151")
            if side not in ("pre", "post", "unverified"):
                raise SourceError(f"published_results: {r['id']} (EPDMS) must state its #151 side")
            if r.get("comparison_admissible") and side != "post" and r["protocol"] != "EPDMS_v2_navtest_single_stage":
                raise SourceError(f"published_results: {r['id']} is admissible but not post-#151")
    return pub


# ---------------------------------------------------------------- registry
def load_registry(root: Path, cfg: dict) -> str:
    return (root / cfg["registry"]).read_text(encoding="utf-8")


# ---------------------------------------------------------------- internal (registry-anchored raw JSON)
def _interval(v: dict, invert: bool = False) -> dict:
    d, lo, hi = v["delta"], v["lo"], v["hi"]
    if invert:
        d, lo, hi = -d, -hi, -lo
    return {"delta": d, "lo": lo, "hi": hi, "separated": bool(v["separated"]),
            "fragile": bool(v.get("fragile", False)), "n": v.get("n") or v.get("n_windows")}


_FAMILY_OF_PREFIX = (("LON_", "LONGITUDINAL"), ("LAT_", "LATERAL"), ("TAC_", "TACTICAL"), ("STR_", "STRATEGIC"),
                     ("ade", "ADE"), ("fde", "ADE"))
_FAMILY_ORDER = {"ADE": 0, "LONGITUDINAL": 1, "LATERAL": 2, "TACTICAL": 3, "STRATEGIC": 4}


_HIGHER_IS_BETTER = re.compile(r"(correct|accuracy|kappa|eq_logged|(^|_)acc($|_))")


def _higher_is_better(name: str, m: dict) -> bool:
    """The artifact's own flag wins; otherwise a TOKEN-exact name rule. ⚠️ A substring rule ('_acc')
    matched `LON_accel_mae_mps2` and printed a separated LOSS as a win — pinned by a test."""
    if "higher_is_better" in m:
        return bool(m["higher_is_better"])
    return bool(_HIGHER_IS_BETTER.search(name))


def family_metrics(block: dict, invert: bool = False, flat: bool = False) -> list:
    """Every metric of a per-family paired block, per family, never pooled. A metric the artifact
    refused (``n``/``reason`` without a delta) comes back as a NOT MEASURED entry, never dropped."""
    out = []
    items = []
    if flat:
        for name, m in block.items():
            fam = next((f for p, f in _FAMILY_OF_PREFIX if name.startswith(p)), None)
            if fam and isinstance(m, dict):
                items.append((fam, name, m))
    else:
        for fam, mets in block.items():
            if isinstance(mets, dict):
                inner = mets.get("metrics", mets)
                for name, m in inner.items():
                    if isinstance(m, dict):
                        items.append((fam.upper(), name, m))
    for fam, name, m in items:
        if "delta" in m and m.get("lo") is not None:
            iv = _interval(m, invert)
            out.append({"family": fam, "metric": name, **iv, "higher_is_better": _higher_is_better(name, m),
                        "n": m.get("n") or m.get("n_windows")})
        elif "reason" in m or m.get("status") == "UNAVAILABLE":
            out.append({"family": fam, "metric": name, "not_measured": m.get("reason", "UNAVAILABLE"), "n": m.get("n", 0)})
    out.sort(key=lambda x: (_FAMILY_ORDER.get(x["family"], 9), x["metric"]))
    return out


def load_internal(root: Path, cfg: dict, registry_text: str) -> list:
    rows = []
    for s in cfg["internal"]:
        path = root / s["path"]
        doc = load_json(path)
        lvl = jptr(doc, s["level"])
        row = {k: s[k] for k in ("id", "claim", "tier", "loop", "grid_label", "metric", "registry_section")}
        row["rig"] = s.get("rig", "")
        row["cluster_word"] = s.get("cluster_word", "episodes")
        row["level"] = {"mean": lvl["mean"], "lo": lvl.get("lo"), "hi": lvl.get("hi")}
        row["n_windows"] = jptr(doc, s["n_windows"])
        row["n_clusters"] = jptr(doc, s["n_episodes"])
        row["source"] = {"path": rel(path, root), "sha12": short_sha(path)}
        row["registry_anchor_found"] = s["registry_anchor"] in registry_text
        row["margins"] = []
        for m in s["margins"]:
            mdoc = load_json(root / m["path"]) if m.get("path") else doc
            try:
                iv = _interval(jptr(mdoc, m["ptr"]), m.get("invert", False))
            except (KeyError, TypeError) as e:
                raise SourceError(f"internal row {s['id']}: margin pointer {m['ptr']!r} is not a paired interval ({e})") from e
            row["margins"].append({"floor": m["floor"], **iv,
                                   "source": rel(root / m["path"], root) if m.get("path") else row["source"]["path"]})
        row["families"] = []
        for f in s.get("families", []):
            fdoc = load_json(root / f["path"]) if f.get("path") else doc
            row["families"].append({"floor": f["floor"],
                                    "metrics": family_metrics(jptr(fdoc, f["ptr"]), f.get("invert", False), f.get("flat", False)),
                                    "source": rel(root / f["path"], root) if f.get("path") else row["source"]["path"]})
        row["families_note"] = s.get("families_note", "")
        row["strategic"] = None
        if s.get("strategic_probe"):
            sp = s["strategic_probe"]
            sdoc = load_json(root / sp["path"]) if sp.get("path") else doc
            cond = jptr(sdoc, sp["ptr"])
            row["strategic"] = {"kind": "probe", "source": rel(root / sp["path"], root) if sp.get("path") else row["source"]["path"],
                                "conditionings": {k: {x: cond[k].get(x) for x in ("status", "accuracy", "kappa", "n")}
                                                  for k in sorted(cond) if isinstance(cond[k], dict)}}
        elif s.get("strategic_computed"):
            sp = s["strategic_computed"]
            sc = jptr(load_json(root / sp["path"]), sp["ptr"])
            row["strategic"] = {"kind": "computed", "source": rel(root / sp["path"], root),
                                **{x: sc.get(x) for x in ("status", "route_acc", "route_acc_ci", "route_kappa", "n", "n_episodes",
                                                          "route_chance_1_over_3", "nav_echo_index")}}
        rows.append(row)
    return rows


# ---------------------------------------------------------------- our benchmark runs (W1 contract)
def w1_validate(summary: dict, where: str) -> None:
    """W1's own validator — the contract is ENFORCED on read, not assumed. Prefers
    ``taniteval.bench.contract.validate_summary`` (schema + the cross-field rules: every floor is an
    arm of kind floor, every arm paired against every floor, SELF on the diagonal, NavSim floors
    present); falls back to the plain schema check if the contract module is absent."""
    try:
        from taniteval.bench.contract import validate_summary
        errs = validate_summary(summary)
    except ImportError:
        try:
            from taniteval.bench.schema_check import load_schema, validate
        except Exception as e:  # the contract module is part of the build's inputs
            raise SourceError(f"cannot import W1's contract validator ({e}) — refusing to read {where}") from e
        errs = validate(summary, load_schema("summary"))
    if errs:
        raise SourceError(f"{where}: violates the W1 summary contract — {errs[:5]}")


def _tier_loop(stamps: dict) -> str:
    """`stamps.loop` is an object (W1 keys it per stage) — render it as prose, never as a repr."""
    loop = stamps.get("loop")
    if isinstance(loop, dict):
        loop = " · ".join(f"{k.replace('_', ' ')} {v}" for k, v in loop.items())
    return f"{stamps.get('tier', '?')} · loop {loop}"


def summary_rows(s: dict, where: str, root: Path) -> list:
    """Normalise one W1-schema summary into renderable rows, re-checking the binding rules."""
    w1_validate(s, where)
    arms = s["arms"]
    if s["benchmark"] in ("navsim_v2", "navsim_v1"):
        for f in ("STOP", "CV"):
            if f not in s["floors"] or f not in arms:
                raise SourceError(f"{where}: NavSim run without its mandatory {f} floor — refused")
    prov = s.get("provenance", {})
    dev = prov.get("devkit")
    if isinstance(dev, dict):                       # W1 writes {repo, sha, patches[]}
        # ⚠️ same rule as the checkpoint: when the producer recorded its own wording, PRINT THAT.
        # Composing "N patches" over a verbatim line drops what the line said (measured: "post-#151
        # + 3 local patches + E1 Windows loader patch" became "+ 1 patches").
        if dev.get("as_recorded"):
            dev = dev["as_recorded"]
        else:
            np_ = len(dev.get("patches") or [])
            dev = f"{dev.get('repo', '?')}@{dev.get('sha', '?')}" + (f" + {np_} patches" if np_ else "")
    ck = prov.get("ckpt")
    if isinstance(ck, dict):
        # ⛔ READ W1'S FIELD, NEVER RE-DERIVE IT. The identity is formatted ONCE, by the producer, into
        # `registry_key_display` (four states: the key · `sha256:…` · UNIDENTIFIED (path only) · no
        # checkpoint). Two producers formatting one row independently are two things that can disagree.
        # ⚠️ And distinguish ABSENT from PRESENT-BUT-NULL: `{"k": None}.get("k", "…")` returns None, so a
        # key that EXISTS silently bypasses a consumer's default — the exact defect W1 traced its own
        # report failure to. Both are reported loudly; neither is guessed from `registry_key`/`path`.
        if "registry_key_display" not in ck:
            ck = "⛔ UNIDENTIFIED — this artifact predates `provenance.ckpt.registry_key_display`; ask its producer to backfill"
        elif ck["registry_key_display"] is None:
            ck = "⛔ UNIDENTIFIED — `provenance.ckpt.registry_key_display` is present but NULL in the artifact"
        else:
            ck = ck["registry_key_display"]
    meta = {"protocol": s["protocol"], "run_id": s["run_id"], "source": where,
            "source_sha12": (prov.get("source_sha256") or "")[:12],
            "harness": dev or "—", "ckpt": ck or "—",
            "tier_loop": prov.get("tier_loop") or _tier_loop(s["stamps"]),
            "converted": bool(prov.get("not_produced_by_bench_cli")), "converted_from": prov.get("source"),
            "headline_name": s["headline_metric"]["name"], "claim_bearing": s["claim_bearing"]}
    rows = []
    for name in arms:
        a = arms[name]
        iv = a["interval"]
        if s["protocol"] == WARMUP and iv.get("status") != "UNAVAILABLE":
            raise SourceError(f"{where}: warmup arm {name} carries an interval — refused (7 log groups < 8)")
        h = a["headline"]
        # ⛔ NEVER MULTIPLY ON ASSUMPTION. `x100` exists only where the headline IS a 0–1 score (EPDMS /
        # PDMS). MEASURED 2026-09-20: the old `h.get("x100", h["value"] * 100)` fallback turned an
        # internal-T1 headline of **2.9098 m ADE** into "290.98" — a correct formula applied under the
        # wrong unit, which reads exactly like an answer. The native value is carried with its column.
        off = None if h.get("status") == "UNAVAILABLE" else h.get("x100")
        native = None if h.get("status") == "UNAVAILABLE" else h.get("value")
        s2 = a.get("statistics", {}).get("S2_EPDMS_u")
        vs_s2, vs_head = {}, {}
        for f in ("STOP", "CV"):
            p = (s2 or {}).get("paired", {}).get(f)
            vs_s2[f] = None if not p else {"delta_x100": p["delta_x100"], "w": p["wins"], "t": p["ties"], "l": p["losses"], "n": p["n"]}
            h2 = a["paired"].get(f, {})
            vs_head[f] = None if h2.get("status") != "OK" or h2.get("headline_delta") is None else {
                "delta_x100": h2["headline_delta"] * 100, "w": None, "t": None, "l": None, "n": h2["n_common"]}
        use_s2 = any(vs_s2.values())
        vs, vs_stat = (vs_s2, "S2-u") if use_s2 else (vs_head, "EPDMS")
        reason = h.get("reason") if off is None else None
        rows.append({**meta, "key": name, "label": a.get("label", name), "kind": a.get("role", a["kind"]),
                     "arm_status": a.get("status", "OK"),
                     "declared": ", ".join(a["declared_inputs"]), "official_x100": off,
                     # ⚠️ 'hybrid' means ONE specific refusal (a CV stand-in stage 1). Any other missing
                     # headline is printed with the artifact's own reason — never relabelled as HYBRID.
                     "headline_value": native, "headline_column": h.get("column") or s["headline_metric"].get("column"),
                     "split": s.get("split"), "arm_tier": a.get("tier"),
                     "tool_provenance": prov.get("tool_provenance") or {},
                     # the arm's own per-metric statistics and native-unit pairs (NOT the ×100 NavSim ones)
                     "stat_interval": (a.get("statistics", {}) or {}).get(h.get("column") or "", {}) or {},
                     "paired_native": a.get("paired") or {}, "fam": a.get("families") or {},
                     "official_reason": reason, "hybrid": bool(reason and "HYBRID" in reason),
                     # W-25: the devkit's printed combined row SURVIVES the refusal, under a name that
                     # says what it is. It is shown, never used as the arm's score.
                     "hybrid_row_x100": (a.get("statistics", {}).get("official_combined_row_HYBRID") or {}).get("x100"),
                     "s2u_x100": (s2.get("x100") if s2.get("x100") is not None
                                  else (s2["value"] * 100 if isinstance(s2.get("value"), (int, float)) else None))
                     if s2 else None,
                     "vs": vs, "vs_statistic": vs_stat,
                     "vs_s2": vs_s2, "vs_head": vs_head,
                     "n_stage2": (_stage(a, 2) or {}).get("n"),
                     "stage1": _stage_view(a, 1), "stage2": _stage_view(a, 2),
                     "interval": "UNAVAILABLE" if iv["status"] == "UNAVAILABLE" else iv,
                     "interval_reason": f"UNAVAILABLE — {iv.get('reason', '')}" if iv["status"] == "UNAVAILABLE" else "OK",
                     "primary_statistic": (s2 or {}).get("statistic", "")})
    return rows


def _stage(a: dict, i: int) -> dict:
    """W1 writes `per_stage.stage_one`; the legacy converter writes `per_stage.stage1`. Accept both —
    ⚠️ and never silently return {} for a name neither producer uses."""
    ps = a.get("per_stage") or {}
    for k in ((f"stage_{'one' if i == 1 else 'two'}", f"stage{i}")):
        if k in ps:
            return ps[k] or {}
    return {}


def _stage_view(a: dict, i: int) -> dict:
    s = _stage(a, i)
    v = s.get("score", s.get("official_stage_score"))
    return {"x100": None if v is None else v * 100, "n": s.get("n"), "submetrics": s.get("submetrics") or {}}


def bench_tree_state(root: Path, cfg: dict) -> dict:
    """{summary.json path: (size, mtime_ns)} for every W1 run the glob sees — the cheap fingerprint
    that tells a build whether its inputs moved while it was rendering."""
    out = {}
    for f in sorted(glob.glob(str(root / cfg["bench_glob"]))):
        st = os.stat(f)
        out[rel(Path(f), root)] = (st.st_size, st.st_mtime_ns)
    return out


def load_bench_runs(root: Path, cfg: dict) -> tuple:
    """Real W1 runs: taniteval/results/bench/<benchmark>/<split>/<run_id>/summary.json."""
    rows, summaries = [], []
    for f in sorted(glob.glob(str(root / cfg["bench_glob"]))):
        s = load_json(Path(f))
        rows += summary_rows(s, rel(Path(f), root), root)
        summaries.append(s)
    return rows, summaries


def legacy_summaries(root: Path, cfg: dict) -> tuple:
    """Convert pre-contract banked outputs into W1-schema summaries (validated before use)."""
    from .legacy import e2_to_summary
    notes, e1 = [], None
    for spec in cfg.get("legacy_bench", []):
        if spec["adapter"] == "json_pointer":
            path = root / spec["path"]
            doc = load_json(path)
            note = {"id": spec["id"], "label": spec["label"], "source": rel(path, root), "sha12": short_sha(path),
                    "protocol": spec.get("protocol"), "render": spec.get("render"),
                    "inherited_note": spec.get("inherited_note"),
                    **{k: jptr(doc, p) for k, p in spec["pointers"].items()}}
            for k in spec.get("scale_x100", []):             # artifacts that bank a FRACTION, printed ×100
                note[k] = None if note.get(k) is None else note[k] * 100
            if spec.get("scope_note"):                       # only when the artifact HAS one (no null noise)
                note["scope_note"] = spec["scope_note"]
            if note["render"] is None:                       # the E1 harness-validation control
                note["evidence"] = "MEASURED (E1 control C8)"
                e1 = note
            notes.append(note)
        elif spec["adapter"] == "devkit_csv":
            # the devkit's OWN per-scene CSV, read at its own summary row — not a number copied out of
            # a RESULT.md table (⚠️ "a summary is not a path": a report can go stale, the CSV cannot)
            import csv as _csv
            path = root / spec["path"]
            with open(path, encoding="utf-8", newline="") as fh:
                rows = list(_csv.DictReader(fh))
            key, col = spec.get("summary_row", "average_all_frames"), spec.get("column", "score")
            hit = [r for r in rows if (r.get("token") or "").strip() == key]
            if len(hit) != 1:
                raise SourceError(f"{rel(path, root)}: expected exactly one '{key}' row, found {len(hit)}")
            notes.append({"id": spec["id"], "label": spec["label"], "source": rel(path, root),
                          "sha12": short_sha(path), "protocol": spec.get("protocol"),
                          "render": spec.get("render"), "scope_note": spec.get("scope_note"),
                          "value_x100": float(hit[0][col]) * 100, "n": len(rows) - 1,
                          "column": col, "summary_row": key})
        elif spec["adapter"] == "w3_navtest_analysis":
            path = root / spec["path"]
            doc = load_json(path)
            arms = []
            for key, meta in spec["arms"].items():
                a = doc["arms"].get(key)
                if a is None:
                    raise SourceError(f"{rel(path, root)}: arm {key!r} named in the config is absent from the artifact")
                iv = a.get("interval") or {}
                arms.append({"key": key, "label": meta["label"], "kind": meta["kind"], "declared": meta["declared"],
                             "n": a["n"], "x100": a["x100"], "pdms_x100": a["x100"]["PDMS"],
                             "devkit_row": a.get("devkit_average_row_score"),
                             "lo": iv.get("lo"), "hi": iv.get("hi"), "estimator": iv.get("estimator"),
                             "n_clusters": iv.get("n_clusters"), "cluster_unit": iv.get("cluster_unit"),
                             "paired": a.get("paired_interval") or {},
                             "vs_paper": a.get("vs_paper_table1") or {},
                             "vs_lb": a.get("vs_hf_leaderboard_INHERITED") or {}})
            notes.append({"id": spec["id"], "render": "w3_navtest", "protocol": spec["protocol"],
                          "source": rel(path, root), "sha12": short_sha(path), "arms": arms,
                          "stamps": doc.get("stamps") or {}, "estimator_note": (doc.get("estimator") or {}).get("_note"),
                          "decomp": doc.get("STOP_decomposition") or {}, "c6": doc.get("C6_stop_prediction") or {},
                          "failed_prediction": spec.get("failed_prediction"),
                          "result_doc": spec.get("result_doc"), "spec_doc": spec.get("spec_doc")})
    out = []
    for spec in cfg.get("legacy_bench", []):
        if spec["adapter"] == "e2_scores_summary":
            out.append((spec, e2_to_summary(root, spec, e1)))
        elif spec["adapter"] not in ("json_pointer", "devkit_csv", "w3_navtest_analysis"):
            raise SourceError(f"unknown legacy adapter {spec['adapter']!r}")
    return out, notes


def load_legacy(root: Path, cfg: dict, w1_arms: set, w1_values: dict | None = None) -> tuple:
    """Legacy rows, minus any legacy RUN a real W1 run has fully reproduced.

    ⚠️ A W1 run whose arm headline is UNAVAILABLE (a failed or aborted run) does NOT supersede a
    measured legacy row — otherwise a failed run would silently delete the floors of a good one.

    ⛔ AND THE SUPERSEDE IS RUN-SCOPED, NOT ARM-SCOPED. MEASURED 2026-09-20: W1's first three real
    warmup runs carry ONLY the two devkit floors (STOP, CV), while E2's banked run carries those
    floors **plus** refcv4b's eight arms — whose Δ columns are paired against them WITHIN that run.
    Superseding arm-by-arm therefore deleted E2's STOP and CV rows and left eight arms quoting a
    `Δ vs STOP` against a floor the table no longer showed. A W1 run supersedes a legacy run only
    when it reports a value for EVERY arm key of that run; otherwise the legacy run is kept whole
    and each arm the W1 side also measured is REPORTED as an independent reproduction (with the
    agreement, which is a free cross-rig control — not silently dropped)."""
    rows, (conv, notes) = [], legacy_summaries(root, cfg)
    w1_values = w1_values or {}
    for spec, s in conv:
        lrows = summary_rows(s, f"{spec['path']} (converted by taniteval.leaderboard.legacy)", root)
        keys = {(r["protocol"], r["key"]) for r in lrows}
        if keys and keys <= w1_arms:
            notes.append(f"legacy run {spec['id']} superseded IN FULL by W1 bench runs "
                         f"({len(keys)} arms, every one reported with a value)")
            continue
        for r in lrows:
            k = (r["protocol"], r["key"])
            if k in w1_arms and r["official_x100"] is not None and k in w1_values:
                d = abs(w1_values[k] - r["official_x100"])
                notes.append({"id": f"repro::{spec['id']}::{r['key']}", "render": "repro",
                              "protocol": r["protocol"], "arm": r["key"], "legacy_run": spec["id"],
                              "legacy_x100": r["official_x100"], "w1_x100": w1_values[k], "abs_delta": d,
                              "agrees": d < 5e-4})
            rows.append(r)
    return rows, notes


def load_audit(root: Path, cfg: dict):
    p = root / cfg["currency_audit"]
    return (load_json(p), rel(p, root)) if p.exists() else (None, rel(p, root))
