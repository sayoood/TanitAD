"""Render the leaderboard data model to the BENCH markdown section and to HTML.

Every sentence that carries a number is a template filled from the loaded data model, so the page can
never drift from its inputs. Rules are enforced here (``SourceError``), not merely documented.

⭐ **THE RULE THIS MODULE IS BUILT ON: the producer's own wording wins. A renderer that rebuilds a fact
is a SECOND PRODUCER, and two producers of one fact are two things that can disagree.** Print what the
artifact says — ``provenance.ckpt.registry_key_display`` for the checkpoint, the arm's own ``reason``
for a refusal, the devkit's own recorded line, and a number in the units its artifact declares.
⛔ Three defects in one day came from breaking it in three costumes: a re-derived checkpoint key, an
assumed ``×100`` that rendered **2.9098 m as "290.98"**, and a composed ``+ 1 patches`` that replaced a
verbatim line naming three. ⇒ if you are formatting a fact the producer already formatted, read its
field instead.
"""
from __future__ import annotations

import html
import json
import re

from .sources import SourceError, WARMUP

MINUS = "\u2212"


# ------------------------------------------------------------------ formatting
def num(v, dp: int) -> str:
    if v is None:
        return "—"
    s = f"{abs(v):.{dp}f}"
    return (MINUS + s) if v < 0 and s.strip("0.") else s


def sgn(v, dp: int) -> str:
    if v is None:
        return "—"
    s = f"{abs(v):.{dp}f}"
    if not s.strip("0."):
        return f"{0:.{dp}f}"
    return ("+" if v > 0 else MINUS) + s


def ci(lo, hi, dp: int) -> str:
    """Interval of a DELTA (signed)."""
    return f"[{sgn(lo, dp)}, {sgn(hi, dp)}]" if lo is not None and hi is not None else ""


def lci(lo, hi, dp: int) -> str:
    """Interval of a LEVEL (unsigned)."""
    return f"[{num(lo, dp)}, {num(hi, dp)}]" if lo is not None and hi is not None else ""


def cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def row(cells) -> str:
    return "| " + " | ".join(cell(c) for c in cells) + " |"


def header(cols, align=None) -> list:
    align = align or ["---"] * len(cols)
    return [row(cols), "|" + "|".join(align) + "|"]


def verdict(m: dict, lower_is_better: bool = True) -> str:
    """Three-way, from the SIGN and the interval (page §0.5)."""
    if not m["separated"]:
        v = "TIE"
    else:
        model_better = (m["delta"] < 0) if lower_is_better else (m["delta"] > 0)
        v = "model WINS" if model_better else "⛔ floor WINS"
    return v + (" · FRAGILE" if m.get("fragile") else "")


def src_ref(r: dict) -> str:
    s = r["source"]
    if not s.get("library_key"):
        return s.get("table", "—")
    out = f"`{s['library_key']}` · {s['table']} · p.{s['page']}"
    # a SECOND primary printing the same number is the strongest cheap check there is — show it
    for c in r.get("cross_checks") or []:
        if c.get("library_key") and c["library_key"] != s["library_key"]:
            agree = "=" if c.get("value") == r.get("value") else "≠"
            out += f" · 2nd primary `{c['library_key']}` {c['table']} p.{c['page']} ({agree} {num(c.get('value'), r.get('decimals', 1))})"
        elif c.get("value") is not None and c["value"] != r.get("value"):
            out += f" · ⚠️ the SAME paper's {c['table']} p.{c['page']} prints {num(c['value'], r.get('decimals', 1))}"
    return out


def pub_value(r: dict, protocols: dict) -> str:
    dp = r.get("decimals", protocols[r["protocol"]].get("decimals", 1))
    return num(r["value"], dp)


# ------------------------------------------------------------------ guards
def assert_one_protocol(protocol: str, rows: list) -> None:
    bad = sorted({r["protocol"] for r in rows} - {protocol})
    if bad:
        raise SourceError(f"table {protocol}: rows from {bad} may not share its column (navsim.cross_protocol)")


def assert_no_warmup_ci(rows: list) -> None:
    for r in rows:
        if r["protocol"] == WARMUP and r.get("interval") not in (None, "UNAVAILABLE"):
            raise SourceError(f"warmup row {r.get('key') or r.get('id')} carries an interval — refused")


# ------------------------------------------------------------------ the 5-October position
def _m(row: dict, floor_prefix: str):
    for m in row["margins"]:
        if m["floor"].startswith(floor_prefix):
            return m
    return None


def _mtxt(m: dict, dp: int = 4) -> str:
    return f"{sgn(m['delta'], dp)} {ci(m['lo'], m['hi'], dp)} ({verdict(m)})"


def _headline_cell(a: dict, dp: int = 2) -> str:
    """A value, or the artifact's OWN reason for not having one. ⛔ Never relabel one refusal as
    another: HYBRID means a CV stand-in stage 1, a FAILED run means the scorer produced nothing."""
    if a["official_x100"] is not None:
        return num(a["official_x100"], dp)
    if a.get("headline_value") is not None:
        # a headline that is NOT a 0–1 score prints in its own units, named — never rescaled to look
        # like an EPDMS (that is how 2.9098 m once became "290.98")
        return f"{num(a['headline_value'], 4)} *(`{a.get('headline_column') or '?'}`)*"
    why = (a.get("official_reason") or "UNAVAILABLE").replace("\n", " ")
    if a.get("hybrid"):
        # W-25: the devkit's printed row survives the refusal, shown and labelled, never used as a score
        hv = a.get("hybrid_row_x100")
        return ("⛔ HYBRID — not this arm's own number" +
                (f" *(the runner printed {num(hv, dp)}; stand-in rows counted from the seam artifact's own "
                 f"`source` column — see the run note)*" if hv is not None else ""))
    short = why if len(why) <= 150 else why[:147] + "…"
    return f"⛔ UNAVAILABLE — {short}" + (f" (arm status {a['arm_status']})" if a.get("arm_status") not in (None, "OK") else "")


def cmd_mark(text: str) -> str:
    """Mark a CONDITIONING cell that carries NavSim's driving command — a ROUTE-LEVEL ORACLE (W2
    D-NAVSIM-ROUTE-LEAK). The marker rides on the row because the protocol note alone is a footnote,
    and a row is read alone far more often than its section header."""
    t = str(text or "")
    return t + " ⚠️ route oracle" if "command" in t.lower() and "route oracle" not in t else t


def _vs(v) -> str:
    """A paired point difference with its per-scene W/T/L (no interval: the estimator refuses one)."""
    if not v:
        return "—"
    wtl = f" · W/T/L {v['w']}/{v['t']}/{v['l']}" if v.get("w") is not None else ""
    return f"{sgn(v['delta_x100'], 2)}{wtl}"


def position_block(model: dict) -> list:
    I = {r["id"]: r for r in model["internal"]}
    P = {r["id"]: r for r in model["published"]["results"]}
    nav = [r for r in model["navsim"] if r["protocol"] == WARMUP]
    N = {r["key"]: r for r in nav}
    repro = next((n for n in model["notes"] if isinstance(n, dict) and n.get("id") == "e1_hf_reproduction"), None)

    beats_all = [r["claim"] for r in model["internal"] if r["margins"] and all(m["separated"] and m["delta"] < 0 for m in r["margins"])]
    beats_ha0 = [r for r in model["internal"] if r["loop"] == "OPEN" and (_m(r, "ha0 ") and _m(r, "ha0 ")["separated"] and _m(r, "ha0 ")["delta"] < 0)]
    L = ["## ⭐ 5 October position — what TanitAD can claim today *(generated)*", ""]
    if not beats_all:
        L.append("⛔ **No TanitAD arm beats every trivial floor it is read against — at any tier, in either loop.** "
                 "The table below is the evidence, one row per arm, each margin paired and on identical windows.")
    else:
        L.append("⭐ **Arms that beat every floor they are read against:** " + "; ".join(beats_all) + ".")
    if beats_ha0:
        parts = [f"{r['claim'].split(' — ')[0]} {sgn(_m(r, 'ha0 ')['delta'], 4)}" for r in beats_ha0]
        L.append(f"* ✅ **Admissible (T1 · OPEN):** {len(beats_ha0)} arms beat the straight-line floor `ha0` "
                 f"(paired Δ m: {', '.join(parts)}) — *better than driving straight at constant speed*, and no stronger claim.")
    if "refcv4b" in I:
        r = I["refcv4b"]
        L.append(f"* ⛔ **Not admissible — beating hold-action or the echo control.** Best B1 arm refcv4b: vs `ha0_ext` "
                 f"{_mtxt(_m(r, 'ha0_ext'))}, vs `ha` {_mtxt(_m(r, 'ha '))}.")
    if "refav1" in I:
        L.append(f"* ⛔ refav1's planner reproduces the `ha0` plan and loses to it: {_mtxt(_m(I['refav1'], 'ha0 '))}.")
    cl = [I[k] for k in ("closed_refcv3", "closed_refcbase", "closed_flagship") if k in I]
    if cl:
        bits = [f"{r['claim'].split(' — ')[0]} vs `cl_ha0` {_mtxt(_m(r, 'cl_ha0 '), 4)}" for r in cl]
        L.append(f"* **CLOSED loop exists but is one scene** ({cl[0]['grid_label']}): " + "; ".join(bits) + ".")
    # a run may report several rows per arm (a failed W1 run and a measured legacy one): take the one WITH a value
    N = {}
    for a in nav:
        cur = N.get(a["key"])
        if cur is None or (cur["official_x100"] is None and a["official_x100"] is not None):
            N[a["key"]] = a
    if nav and "STOP" in N and "CV" in N and N["STOP"]["official_x100"] is not None:
        stop, cv = N["STOP"], N["CV"]
        rep = f"; our CV reproduces the HF warmup row {repro['reference']} at {num(repro['local_x100'], 6)} (Δ {repro['delta']})" if repro else ""
        a1, a2 = N.get("A1_ego_cmd"), N.get("A2_vision_pure")
        s = (f"* **NAVSIM v2 warmup (official harness, post-#151{rep}):** doing nothing (STOP) scores "
             f"**{num(stop['official_x100'], 2)}** official two-stage EPDMS against CV's **{num(cv['official_x100'], 2)}**.")
        if a1 and a1["vs"].get("CV") and a1["vs"].get("STOP") and stop["official_x100"] is not None:
            s += (f" On the stage-2 statistic refcv4b A1 (ego + command) sits ABOVE CV ({_vs(a1['vs']['CV'])}; the "
                  f"pre-registered bar) and BELOW STOP ({_vs(a1['vs']['STOP'])})")
        if a2 and a2["vs"].get("STOP"):
            s += f"; the vision-pure arm plans a stop ({_vs(a2['vs']['STOP'])} vs STOP)"
        L.append(s + ". Point estimates only — the estimator refuses an interval (7 log groups < 8).")
    NH = "EPDMS_v2_navhard_two_stage"
    nh = {a["key"]: a for a in model["navsim"] if a["protocol"] == NH and a["official_x100"] is not None}
    if nh.get("STOP") and nh.get("CV"):
        st, cv = nh["STOP"], nh["CV"]
        L.append(f"* ⛔ **On the OFFICIAL column (navhard two-stage) our FLOORS are now measured — and doing nothing "
                 f"wins: STOP {num(st['official_x100'], 2)} {_iv(st).split(' · ')[0]} against CV "
                 f"{num(cv['official_x100'], 2)} {_iv(cv).split(' · ')[0]}, a factor of "
                 f"{num(st['official_x100'] / cv['official_x100'], 2)}×** (paired Δ "
                 f"{sgn((st['vs_head'].get('CV') or {}).get('delta_x100'), 4)} ×100; log-cluster bootstrap over "
                 f"{(st['interval'] or {}).get('n_clusters') if isinstance(st.get('interval'), dict) else '?'} clusters). "
                 f"⛔ **No TanitAD ARM is on this column yet** — these are the devkit floors, and the bar any arm must "
                 f"clear is **STOP**, not CV.")
    w3 = _note(model, "w3_navtest", "PDMS_v1_navtest")
    if w3:
        A = {a["key"]: a for a in w3["arms"]}
        st, cv, hu = A.get("STOP"), A.get("CV"), A.get("HUMAN")
        ego = next((r for r in P.values() if r["protocol"] == "PDMS_v1_navtest" and "Ego Status MLP" in r["system"]), None)
        if st and cv:
            L.append(f"* ⛔ **And CV is not a floor on NAVSIM v1 navtest either — MEASURED on the full split "
                     f"({st['n']:,} tokens/arm): STOP {num(st['pdms_x100'], 2)} "
                     f"[{num((st['lo'] or 0) * 100, 2)}, {num((st['hi'] or 0) * 100, 2)}] against CV "
                     f"{num(cv['pdms_x100'], 2)} [{num((cv['lo'] or 0) * 100, 2)}, {num((cv['hi'] or 0) * 100, 2)}], "
                     f"{num(st['pdms_x100'] / cv['pdms_x100'], 1)}×**"
                     + (f" — within {num(ego['value'] - st['pdms_x100'], 1)} points of the published blind "
                        f"ego-status-MLP baseline ({num(ego['value'], 1)})" if ego else "")
                     + (f"; the human ceiling reads {num(hu['pdms_x100'], 2)}" if hu else "")
                     + f". ⛔ **No TanitAD arm is on this column either**, and its bar is STOP.")
    if all(k in P for k in ("navhard.drivefuture", "navhard.drivor_134k_toad", "navhard.pdm_closed", "navhard.cv")):
        L.append(f"* **The official column — the external bar, with no TanitAD arm on it.** Learned "
                 f"{pub_value(P['navhard.drivefuture'], model['published']['protocols'])} (DriveFuture) · with test-time search "
                 f"{pub_value(P['navhard.drivor_134k_toad'], model['published']['protocols'])} · privileged "
                 f"{pub_value(P['navhard.pdm_closed'], model['published']['protocols'])} · CV floor "
                 f"{pub_value(P['navhard.cv'], model['published']['protocols'])}. Every leader consumes ego status at inference; "
                 f"our vision-only rule forbids it, so our entry is a paired ego / vision-pure design.")
    att = next((n for n in model["notes"] if isinstance(n, dict) and n.get("render") == "navhard_attempt"), None)
    if att and not nh:
        L.append(f"* ⏳ **The navhard runner has been started and has not yet produced a score:** the devkit CV agent's run "
                 f"ended `{att['status']}` in aggregation after {att['wall_s']} s and wrote no final-scores CSV "
                 f"(`{att['source']}`). It is on the board as UNAVAILABLE-with-a-reason, not omitted.")
    elif att:
        L.append(f"* ⚠️ **The first navhard attempt failed and is still on the board** (`{att['status']}` after "
                 f"{att['wall_s']} s, no final-scores CSV — `{att['source']}`). A later run succeeded; the failed "
                 f"attempt is kept because the cost of a result includes the runs that did not produce one.")
    L.append("* ⛔ **Cannot claim yet:** any external-benchmark score; a driving claim beyond one closed-loop scene; "
             "any lever effect robust to training seed (`H-ESTIM-SEED-1`, one seed per arm); STRATEGIC skill beyond a T1 route head.")
    L.append("")
    L += header(["claim", "tier · loop", "grid", "our number [CI]", "margin vs each floor (model − floor) [paired CI] → verdict", "registry · raw artifact"])
    for r in model["internal"]:
        lv = r["level"]
        margins = "<br>".join(f"{m['floor']}: {_mtxt(m)}" for m in r["margins"])
        reg = f"{r['registry_section']}{'' if r['registry_anchor_found'] else ' ⛔ anchor NOT found in the registry'}"
        L.append(row([r["claim"], f"{r['tier']} · {r['loop']}",
                      f"{r['grid_label']} · {r['n_windows']:,} windows / {r['n_clusters']} {r['cluster_word']}",
                      f"{num(lv['mean'], 4)} {lci(lv['lo'], lv['hi'], 4)}".strip(),
                      margins, f"{reg} · `{r['source']['path']}` (sha256 {r['source']['sha12']}…)"]))
    for key in ("A1_ego_cmd", "A2_vision_pure"):
        if key in N and N[key]["vs"].get("STOP"):
            a = N[key]
            cvtxt = f"<br>CV: {_vs(a['vs']['CV'])}" if a["vs"].get("CV") else "<br>CV: — (not computed by the source run)"
            L.append(row([f"{a['label']} — NAVSIM v2 warmup", a["tier_loop"], f"{a['n_stage2']} stage-2 scenes",
                          f"S2-EPDMS-u {num(a['s2u_x100'], 2)} (interval UNAVAILABLE)",
                          f"STOP: {_vs(a['vs']['STOP'])}{cvtxt} ({a['vs_statistic']} ×100, paired per scene, no interval)",
                          f"§4.6 · `{a['converted_from'] or a['source']}`"]))
    L += ["", "*Metric per row as stated in the source; lower is better for every ADE row; EPDMS is higher-is-better. "
          "`TIE` = the paired interval contains 0; `FRAGILE` = the artifact's own flag (the interval reaches back within 25 % of "
          "the point estimate). Estimator: paired episode-cluster bootstrap (closed loop: rollout-start clusters); "
          "⛔ `overlapping_holdout_se` nowhere.*", ""]
    return L


# ------------------------------------------------------------------ four families per internal arm
def _strategic_line(st: dict) -> str:
    if not st:
        return ""
    if st["kind"] == "probe":
        c = st["conditionings"]
        parts = [f"{k.replace('nav_', '')} {num(v['accuracy'], 4)} (κ {sgn(v['kappa'], 4)}, n {v['n']})"
                 for k, v in c.items() if v.get("accuracy") is not None]
        accs = {round(v["accuracy"], 4) for v in c.values() if v.get("accuracy") is not None}
        t, sh = c.get("nav_true", {}), c.get("nav_shuffled", {})
        if len(accs) == 1:
            read = "identical under every nav conditioning — nav-INSENSITIVE by construction (the shuffle has no power on it)"
        elif (t.get("accuracy") or 0) >= 0.999 and (sh.get("kappa") if sh.get("kappa") is not None else 1) < 0.1:
            read = "perfect under the true nav and at/below chance once it is shuffled — an ECHO of its nav input, not route skill"
        else:
            read = "moves with the nav conditioning — read the shuffle row, not the true-nav row"
        return (f"**STRATEGIC** — route-head probe (reads the observed window; not the trajectory-derived family, which a 2 s path "
                f"cannot carry): accuracy {' · '.join(parts)} → {read}. `{st['source']}`")
    ci_ = st.get("route_acc_ci")
    cis = f" [{num(ci_[0], 4)}, {num(ci_[1], 4)}]" if isinstance(ci_, (list, tuple)) and len(ci_) == 2 else ""
    return (f"**STRATEGIC** — route head: accuracy {num(st.get('route_acc'), 4)}{cis} vs chance {num(st.get('route_chance_1_over_3'), 4)}, "
            f"κ {num(st.get('route_kappa'), 4)}, n {st.get('n')} windows / {st.get('n_episodes')} episodes; nav echo index "
            f"{num(st.get('nav_echo_index'), 4)} (status {st.get('status')}). `{st['source']}`")


def families_block(model: dict) -> list:
    L = ["### The four families behind each row *(generated · per family, never pooled · paired, model − floor)*", "",
         "*A family the artifact could not compute is printed as NOT MEASURED with the artifact's own reason and n. Verdict "
         "direction follows each metric's own sense (errors: lower is better; accuracies / agreement: higher is better).*", ""]
    for r in model["internal"]:
        L += [f"#### {r['claim']} · {r['tier']} · loop {r['loop']}", ""]
        if r["families"]:
            floors = [f["floor"] for f in r["families"]]
            keys = []
            by = {}
            for f in r["families"]:
                for m in f["metrics"]:
                    k = (m["family"], m["metric"])
                    if k not in by:
                        keys.append(k)
                        by[k] = {}
                    by[k][f["floor"]] = m
            L += header(["family", "metric"] + [f"vs `{fl}` Δ [paired CI] → verdict" for fl in floors] + ["n"])
            for k in keys:
                cells, n = [], None
                for fl in floors:
                    m = by[k].get(fl)
                    if not m:
                        cells.append("—")
                    elif "not_measured" in m:
                        cells.append(f"⛔ NOT MEASURED — {m['not_measured']} (n {m['n']})")
                        n = n if n is not None else m["n"]
                    else:
                        cells.append(f"{sgn(m['delta'], 4)} {ci(m['lo'], m['hi'], 4)} → {verdict(m, lower_is_better=not m['higher_is_better'])}")
                        n = n if n is not None else m.get("n")
                L.append(row([k[0], f"`{k[1]}`"] + cells + [f"{n:,}" if isinstance(n, int) else (n or "—")]))
            L.append("")
            srcs = sorted({f["source"] for f in r["families"]})
            L.append("*Source: " + " · ".join(f"`{s}`" for s in srcs) + "*")
        if r["families_note"]:
            L.append(f"*{r['families_note']}.*")
        sl = _strategic_line(r["strategic"])
        if sl:
            L.append("")
            L.append(sl)
        L.append("")
    return with_lateral_qualifier(L)


# ------------------------------------------------------------------ external field, one table per protocol
def _tanitad_row(ncols: int, pending: str, label: str = "**TanitAD — any arm**") -> str:
    return row([label, "⛔ NOT MEASURED — pending: " + pending] + [""] * (ncols - 2))


def _note(model: dict, render_kind: str, proto: str) -> dict | None:
    return next((n for n in model["notes"] if isinstance(n, dict) and n.get("render") == render_kind
                 and n.get("protocol") == proto), None)


def _iv(a: dict) -> str:
    """A NavSim interval, printed ×100 like the score it accompanies, WITH its estimator and n."""
    iv = a.get("interval")
    if iv in (None, "UNAVAILABLE") or not isinstance(iv, dict):
        return "UNAVAILABLE"
    lo, hi = iv.get("lo"), iv.get("hi")
    if lo is None or hi is None:
        return "UNAVAILABLE"
    return (f"[{num(lo * 100, 2)}, {num(hi * 100, 2)}] · `{iv.get('estimator', '?')}` over "
            f"{iv.get('n_clusters', '?')} {iv.get('cluster_unit', 'cluster')} clusters")


def our_navsim_block(model: dict, proto: str) -> list:
    """OUR arms on a NavSim protocol, from real W1 runs — with the floors' own submetrics beside them,
    because on this benchmark the ranking of the two floors is the finding."""
    ours = [r for r in model["navsim"] if r["protocol"] == proto]
    if not ours:
        return []
    assert_one_protocol(proto, ours)
    L, order = [], {"floor": 0, "control": 1, "reference": 1, "model": 2, "ablation": 3}
    runs = {}
    for a in ours:
        runs.setdefault(a["run_id"], []).append(a)
    for run_id in sorted(runs):
        arms = runs[run_id]
        m0 = arms[0]
        # ⛔ the checkpoint identity is PRINTED AS THE ARTIFACT STATES IT (W1's `registry_key_display`);
        # this renderer does not compose a second wording for the same fact.
        L += [f"*Run `{run_id}` — checkpoint: {m0['ckpt']} · harness `{m0['harness']}` · {m0['tier_loop']}. "
              f"Source `{m0['source']}`.*", ""]
        cols = ["arm", "kind", "declared inputs", "official two-stage EPDMS ×100",
                "interval (estimator · clusters)", "stage 1 ×100 (n)", "stage 2 ×100 (n)",
                "Δ vs STOP", "Δ vs CV"]
        L += header(cols)
        for a in sorted(arms, key=lambda a: (order.get(a["kind"], 9), -(a["official_x100"] or 0), a["key"])):
            s1, s2 = a.get("stage1") or {}, a.get("stage2") or {}
            L.append(row([a["label"], a["kind"], cmd_mark(a["declared"]), _headline_cell(a), _iv(a),
                          f"{num(s1.get('x100'), 2)} ({s1.get('n', '—')})",
                          f"{num(s2.get('x100'), 2)} ({s2.get('n', '—')})",
                          "self" if a["key"] == "STOP" else _vs(a["vs_head"].get("STOP")),
                          "self" if a["key"] == "CV" else _vs(a["vs_head"].get("CV"))]))
        L.append("")
        # the submetric table for the floors: it is what makes the STOP > CV ordering readable
        fl = [a for a in arms if a["kind"] == "floor" and (a.get("stage1") or {}).get("submetrics")]
        if len(fl) >= 2:
            keys = sorted({k for a in fl for k in a["stage1"]["submetrics"]})
            L += [f"*Stage-1 sub-metrics of the floors (from the same run) — EPDMS multiplies "
                  f"{', '.join(k for k in keys if k != 'EP')} and weights EP, so a term near 1.0 is a term that "
                  f"does not punish the arm:*", ""]
            L += header(["arm"] + keys)
            for a in sorted(fl, key=lambda a: -(a["official_x100"] or 0)):
                L.append(row([a["label"]] + [num(a["stage1"]["submetrics"].get(k), 4) for k in keys]))
            L.append("")
    return L


def navhard_tables(model: dict, proto: str) -> list:
    pub = model["published"]
    rows = [r for r in pub["results"] if r["protocol"] == proto]
    assert_one_protocol(proto, rows)
    post = [r for r in rows if r["harness"]["fix151"] == "post" and r.get("comparison_admissible")]
    other = [r for r in rows if r not in post]
    cols = ["system", "EPDMS", "role", "source (key · table · p.)", "#151 side", "modality (authors' claim)", "ego status at inference", "comparison"]
    L = []
    # ⭐ THE FINDING GOES WHERE THE READER MEETS THE COLUMN, not in a footnote. Everything in this
    # sentence is computed from the run's own artifact, so it cannot go stale against the table below.
    ours = {a["key"]: a for a in model["navsim"] if a["protocol"] == proto}
    st, cv = ours.get("STOP"), ours.get("CV")
    if st and cv and st["official_x100"] and cv["official_x100"] and st["official_x100"] > cv["official_x100"]:
        sub = (st.get("stage1") or {}).get("submetrics") or {}
        cvsub = (cv.get("stage1") or {}).get("submetrics") or {}
        near1 = [k for k, v in sorted(sub.items()) if k != "EP" and v is not None and v >= 0.93]
        L += [f"⛔ **DOING NOTHING BEATS THE MOVING FLOOR HERE: STOP {num(st['official_x100'], 2)} vs "
              f"CV {num(cv['official_x100'], 2)} — a factor of {num(st['official_x100'] / cv['official_x100'], 2)}×** "
              f"(MEASURED, our run `{st['run_id']}`; paired Δ {sgn((st['vs_head'].get('CV') or {}).get('delta_x100'), 4)} ×100). "
              f"The mechanism is in the score's own shape: EPDMS **multiplies** its compliance terms, and a stopped "
              f"ego keeps {', '.join(f'{k} {num(sub[k], 4)}' for k in near1)} at stage 1, while only ego progress "
              f"punishes it (EP {num(sub.get('EP'), 4)} against CV's {num(cvsub.get('EP'), 4)}). "
              f"⇒ **\"beats CV\" is NOT evidence of driving on this benchmark. The bar for any TanitAD arm is STOP.**", ""]
    L += header(cols)
    for r in sorted(post, key=lambda r: (-r["value"], r["id"])):
        L.append(row([r["system"], pub_value(r, pub["protocols"]), r["role"], src_ref(r), "post", r.get("modality", "—"),
                      cmd_mark(r.get("ego_status", "—")), "✅" if r.get("comparison_admissible") else "⛔ " + r.get("inadmissible_reason", "")]))
    scored = any(a["protocol"] == proto and a["official_x100"] is not None for a in model["navsim"])
    for n in model["notes"]:                                  # our attempts on this protocol, with no score yet
        if isinstance(n, dict) and n.get("render") == "navhard_attempt" and n.get("protocol") == proto:
            st = str(n["status"]).upper()
            if scored:                                        # a LATER run succeeded: say so on the attempt row
                n = {**n, "inherited_note": (n.get("inherited_note") or "") +
                     " ⭐ SUPERSEDED: a later suite run on this protocol completed and is scored below; this row is "
                     "kept because a failed attempt is part of the record, not a number to hide."}
            what = (f"a run is IN PROGRESS (`status` {n['status']}, started {n['started']}, {n['wall_s']} s so far)"
                    if st in ("RUNNING", "STARTED") else
                    f"the official runner ended `{n['status']}` after {n['wall_s']} s ({n['started']} → {n['ended']})")
            L.append(row([f"**{n['label']}**",
                          f"⛔ UNAVAILABLE — {what}; no final-scores CSV, so no EPDMS exists yet. "
                          f"{n.get('inherited_note') or ''}",
                          "floor (CV)", f"`{n['source']}` (sha256 {n['sha12']}…)", f"post (devkit {n['devkit_sha'][:9]})",
                          "no camera", "devkit ego status", "⛔ no score"]))
    L.append(_tanitad_row(len(cols), pub["protocols"][proto]["tanitad_pending"]))
    L.append(_tanitad_row(len(cols), "a DIFFERENT protocol (`EPDMS_v2_warmup_two_stage`) — see its own table; never this column",
                          "TanitAD — warmup numbers"))
    # ---- OUR OWN rows, from real W1 runs
    our = our_navsim_block(model, proto)
    if our:
        L += ["", "#### Our own arms on this protocol *(MEASURED — real suite runs, not a published row)*", ""] + our
    ref = _note(model, "reference_arm", proto)
    if ref:
        L += [f"*⚖️ Reference arm — **{ref['label']}**: EPDMS ×100 **{num(ref.get('value_x100'), 4)}** over n = "
              f"{ref.get('n')} ({ref.get('arm', '')}). {ref.get('scope_note', '')} "
              f"Source `{ref['source']}` (sha256 {ref['sha12']}…).*", ""]
    rp = _note(model, "repro_published", proto)
    if rp:
        # ⛔ W1's self-retraction, 2026-09-20: it banked this reference as "HF leaderboard, 11.4, TRUNCATE,
        # Δ 0.0000" — merging TWO artifacts into one row, whose own fields then contradicted each other.
        # The two never disagreed; quoting one artifact's convention against the other's printed row
        # manufactures a FALSE AGREEMENT, which is worse than a mismatch because nothing looks wrong.
        L += [f"*⭐ Harness validation — **{rp['label']}**. Our CV reads **{num(rp['ours'], 4)}** ×100 "
              f"(stage 1 {num(rp['s1'], 4)} · stage 2 {num(rp['s2'], 4)}), worst sub-metric |Δ| "
              f"{num(rp['worst_sub'], 2)}. ⛔ **STATE THE ARTIFACT BEFORE THE CONVENTION — there are TWO external "
              f"publications here and they print the same underlying value differently:*", ""]
        L += header(["external artifact", "prints", "dp", "its rule", "our {:.4f} reads".format(rp["ours"]), "verdict"])
        L.append(row([f"{rp['src']}", num(rp["paper"], 1), "1", "**truncates**",
                      f"{num(rp['ours'], 4)} → {num(rp['paper'], 1)}",
                      f"✅ REPRODUCED_UNDER_TRUNCATION (Δ vs the printed digits {num(rp['d_paper'], 4)} under rounding)"]))
        L.append(row(["HF navhard leaderboard (INHERITED)", num(rp["lb"], 4), "4", "**rounds**",
                      f"{num(rp['ours'], 4)} → {num(rp['lb'], 4)}", f"✅ Δ **{num(rp['d_lb'], 4)}** — exact"]))
        L += ["", f"*⛔ **Quoting one of these against the other's row manufactures a FALSE AGREEMENT** — worse than a "
              f"mismatch, because nothing looks wrong (W1 self-retraction, 2026-09-20: a reference banked as "
              f"\"leaderboard, 11.4, truncate, Δ 0.0000\" contradicted its own fields). ⇒ *\"NavSim publishes X\"* is "
              f"not a claim until it says WHICH publication. "
              f"⛔ **The pre-registered test was written for ROUNDING and reads FALSE** against the paper "
              f"(Δ {num(rp['d_paper'], 4)}); the reason is the convention, not the harness — "
              f"**{rp['n_trunc']}/{rp['n_terms']} published terms of that paper reproduce EXACTLY under TRUNCATION, "
              f"only {rp['n_round']}/{rp['n_terms']} under rounding.** {rp['prereg']} "
              f"Source `{rp['source']}` (sha256 {rp['sha12']}…).*", ""]
    if other:
        L += ["", f"*Rows NOT in the comparison above — a different harness version (pre-#151) or an unresolved side/conflict:*", ""]
        L += header(["system", "EPDMS", "#151 side", "source (key · table · p.)", "why it is excluded"])
        for r in sorted(other, key=lambda r: (r["harness"]["fix151"], -r["value"], r["id"])):
            L.append(row([r["system"], pub_value(r, pub["protocols"]), r["harness"]["fix151"], src_ref(r),
                          r.get("inadmissible_reason", "—")]))
    return L


def warmup_tables(model: dict, proto: str) -> list:
    pub = model["published"]
    ours = [r for r in model["navsim"] if r["protocol"] == proto]
    assert_one_protocol(proto, ours)
    assert_no_warmup_ci(ours)
    L = []
    prow = [r for r in pub["results"] if r["protocol"] == proto]
    repro = next((n for n in model["notes"] if isinstance(n, dict) and n.get("id") == "e1_hf_reproduction"), None)
    if prow or repro:
        L += header(["reference", "EPDMS ×100", "source", "evidence"])
        for r in prow:
            L.append(row([r["system"], pub_value(r, pub["protocols"]), src_ref(r), r.get("evidence_class", "PUBLISHED")]))
        if repro:
            L.append(row([repro["label"], f"{num(repro['local_x100'], 6)} (Δ {repro['delta']} vs {repro['reference']}; reproduced: {repro['reproduced']})",
                          f"`{repro['source']}` (sha256 {repro['sha12']}…)", "MEASURED (E1, control C8)"]))
        L.append("")
    if not ours:
        return L + [_tanitad_row(4, pub["protocols"][proto]["tanitad_pending"])]
    runs = {}
    for a in ours:
        runs.setdefault(a["run_id"], []).append(a)
    order = {"floor": 0, "control": 1, "model": 2, "ablation": 3}
    # Runs that produced the SAME arms with the SAME numbers are one table + a reproducibility count.
    # ⭐ Identical repeats are evidence (the rig is deterministic), but three identical tables are noise.
    sig = {rid: json.dumps(sorted((a["key"], a["kind"], a["official_x100"], a["s2u_x100"], a["declared"],
                                   json.dumps(a["vs_s2"], sort_keys=True, default=str),
                                   json.dumps(a["vs_head"], sort_keys=True, default=str)) for a in arms),
                           sort_keys=True, default=str) for rid, arms in runs.items()}
    groups, seen = [], {}
    for rid in sorted(runs):
        if sig[rid] in seen:
            groups[seen[sig[rid]]][1].append(rid)
        else:
            seen[sig[rid]] = len(groups)
            groups.append((rid, [rid]))
    for run_id, rids in groups:
        arms = runs[run_id]
        m0 = arms[0]
        conv = (f" Converted from `{m0['converted_from']}` (sha256 {m0['source_sha12']}…) by "
                f"`taniteval.leaderboard.legacy` into the W1 summary schema and validated by W1's `schema_check` — "
                f"not produced by the bench CLI." if m0["converted"] else f" Source `{m0['source']}`.")
        if len(rids) > 1:
            L.append(f"*⭐ Reproducibility: {len(rids)} W1 runs produced these arms with IDENTICAL values — "
                     f"{', '.join('`' + x + '`' for x in rids)}. One table, {len(rids)} runs; the repeat is a "
                     f"determinism control on the rig, ⛔ NOT {len(rids)} independent samples of anything.*")
            L.append("")
        L.append(f"*Run `{run_id}` — checkpoint: {m0['ckpt']} · harness `{m0['harness']}` · {m0['tier_loop']} · interval "
                 f"{m0['interval_reason']}. Headline = the official two-stage combined EPDMS (the HF leaderboard's statistic), "
                 f"printed only where it is not HYBRID. `S2-EPDMS-u` = {m0.get('primary_statistic') or '—'} — ⛔ NOT a two-stage "
                 f"EPDMS; its Δ columns are paired per scene.{conv}*")
        L.append("")
        # ONE statistic per Δ column per run: the stage-2 paired statistic if the run has it, else the headline
        use_s2 = any(a["vs_s2"].get("STOP") or a["vs_s2"].get("CV") for a in arms)
        stat = "S2-EPDMS-u ×100, paired per scene · W/T/L" if use_s2 else "official EPDMS ×100 (difference of two scores)"
        cols = ["arm", "kind", "declared inputs", "official two-stage EPDMS ×100", "S2-EPDMS-u ×100 (stage 2 only)",
                f"Δ vs STOP ({stat})", f"Δ vs CV ({stat})", "interval"]
        L += header(cols)
        for a in sorted(arms, key=lambda a: (order.get(a["kind"], 9), -(a["s2u_x100"] or 0), a["key"])):
            vs = a["vs_s2"] if use_s2 else a["vs_head"]
            L.append(row([a["label"], a["kind"], cmd_mark(a["declared"]),
                          _headline_cell(a),
                          num(a["s2u_x100"], 2) if a["s2u_x100"] is not None else "—",
                          "self" if a["key"] == "STOP" else _vs(vs.get("STOP")),
                          "self" if a["key"] == "CV" else _vs(vs.get("CV")),
                          "UNAVAILABLE"]))
        L += ["", "*— = the pair was not computed by the source run (never imputed).*"]
        for why in dict.fromkeys(a.get("official_reason") or "" for a in arms if a.get("hybrid")):
            L.append(f"*⛔ Why those headlines are refused: {why}*")
        L.append("")
    for n in model["notes"]:                     # the human reference arms (stage 1 only — see the scope note)
        if isinstance(n, dict) and n.get("render") == "reference_arm" and n.get("protocol") == proto:
            L.append(f"*⚖️ Reference arm — **{n['label']}**: EPDMS ×100 **{num(n.get('value_x100'), 4)}** over "
                     f"n = {n.get('n')} scenes (the devkit's own `{n.get('summary_row', '—')}` row of "
                     f"`{n['source']}`, sha256 {n['sha12']}…). {n.get('scope_note', '')}*")
    if any(isinstance(n, dict) and n.get("render") == "reference_arm" and n.get("protocol") == proto for n in model["notes"]):
        L.append("")
    for n in model["notes"]:                     # a legacy floor that a real W1 run also measured
        if isinstance(n, dict) and n.get("render") == "repro" and n.get("protocol") == proto:
            L.append(f"*⭐ Cross-rig reproduction — arm `{n['arm']}`: the W1 bench CLI reads "
                     f"{num(n['w1_x100'], 6)} where the banked run `{n['legacy_run']}` reads "
                     f"{num(n['legacy_x100'], 6)} (|Δ| {n['abs_delta']:.2e}; "
                     f"{'AGREES' if n['agrees'] else '⛔ DISAGREES'}). Two producers, one number — the banked run "
                     f"is kept because the W1 runs carry ONLY these two floors, ⛔ none of the refcv4b arms.*")
    if any(isinstance(n, dict) and n.get("render") == "repro" and n.get("protocol") == proto for n in model["notes"]):
        L.append("")
    return L


def _tier_loop_txt(st: dict) -> str:
    loop = st.get("loop")
    loop = ", ".join(f"{k} {v}" for k, v in loop.items()) if isinstance(loop, dict) else str(loop or "-")
    return f"{st.get('tier', '-')} · loop {loop} · background {st.get('background', '-')}"


def w3_navtest_block(model: dict, proto: str) -> list:
    """OUR OWN navtest arms (W3, full split). The head finding is DERIVED from the artifact, exactly as
    the navhard one is: on navtest a stopped car scores ~3x CV, so CV is not a floor here either."""
    n = _note(model, "w3_navtest", proto)
    if not n:
        return []
    A = {a["key"]: a for a in n["arms"]}
    st, cv = A.get("STOP"), A.get("CV")
    L = []
    if st and cv and st["pdms_x100"] > cv["pdms_x100"]:
        d = n["decomp"]
        band = d.get("per_t0_speed_band") or {}
        lead_all = bool(band) and all(v.get("STOP_x100", 0) > v.get("CV_x100", 0) for v in band.values())
        p = (st.get("paired") or {}).get("CV") or {}
        pw = d.get("paired_STOP_minus_CV", {})
        free = ", ".join(f"{k} {num(st['x100'][k], 2)}" for k in ("NC", "DAC", "TTC") if k in st["x100"])
        ego = next((r for r in model["published"]["results"]
                    if r["protocol"] == proto and "Ego Status MLP" in r["system"]), None)
        L += [f"⛔ **ON navtest A STOPPED CAR SCORES {num(st['pdms_x100'], 2)} — "
              f"{num(st['pdms_x100'] / cv['pdms_x100'], 1)}x CV's {num(cv['pdms_x100'], 2)}"
              + (f", and within {num(ego['value'] - st['pdms_x100'], 1)} points of the published ego-status-MLP "
                 f"baseline ({num(ego['value'], 1)})" if ego else "") + ".** "
              f"Paired on identical tokens: {sgn((p.get('delta') or 0) * 100, 2)} "
              f"[{sgn((p.get('lo') or 0) * 100, 2)}, {sgn((p.get('hi') or 0) * 100, 2)}], "
              f"W/T/L {pw.get('W')}/{pw.get('T')}/{pw.get('L')}"
              + (", leading in EVERY t0 speed band" if lead_all else "") + ". "
              f"Mechanism, MEASURED: {free} make **{num(d.get('floor_5_over_12_x100'), 2)} points free** (5/12 of "
              f"PDMS); the braking coast earns a median {num(d.get('stop_progress_m_median'), 2)} m; and on "
              f"{d.get('n_max_compliant_progress_le_5m')}/{st['n']} tokens "
              f"({num(100 * (d.get('frac_max_compliant_progress_le_5m') or 0), 2)} %) the best compliant progress is "
              f"<= 5 m, so **EP = 1 by rule**. ⇒ **CV is not a floor on navtest either.**", ""]
    if n.get("failed_prediction"):
        L += [f"{n['failed_prediction']} *(`{n['spec_doc']}` §6; `{n['result_doc']}` §3.3)*", ""]
    if "HUMAN" in A:
        L += ["⭐ **HUMAN IS DEFINED ON THIS PROTOCOL** — navtest is SINGLE-STAGE, so the logged expert can be "
              "scored and the ceiling is a number. ⛔ On the v2 TWO-STAGE splits (navhard, warmup) the suite "
              "REFUSES that headline because every synthetic stage-2 scene has `num_future_frames = 0`; there the "
              "human is **UNDEFINED, not missing** — read the v2 refusal as an identity of the protocol, never as "
              "a gap in our measurement.", ""]
    L += [f"*Run — full split, {A['CV']['n']:,} tokens per arm · {_tier_loop_txt(n['stamps'])} · source "
          f"`{n['source']}` (sha256 {n['sha12']}…). {n.get('estimator_note') or ''}*", ""]
    cols = ["arm", "kind", "declared inputs", "PDMS x100", "CI95 (estimator · clusters)", "n",
            "NC", "DAC", "DDC", "TTC", "C", "EP", "Δ vs CV [paired CI]", "vs the published cell"]
    L += header(cols)
    order = {"floor": 0, "reference": 1, "model": 2}
    for a in sorted(n["arms"], key=lambda a: (order.get(a["kind"], 9), -a["pdms_x100"])):
        p = (a.get("paired") or {}).get("CV") or {}
        pv = a["vs_paper"].get("PDMS") or {}
        lb = a["vs_lb"].get("PDMS") or {}
        cellv = []
        if pv:
            cellv.append(f"paper {num(pv['published_x100'], 1)} -> **{pv['verdict']}**")
        if lb:
            cellv.append(f"leaderboard {num(lb['published_x100'], 4)} -> **{lb['verdict']}**")
        bad = [k for k, v in a["vs_paper"].items() if v.get("verdict") == "NOT REPRODUCED"]
        if bad:
            cellv.append("⛔ the ENTIRE gap is " + "; ".join(
                f"**{k} {num(a['vs_paper'][k]['got_x100'], 4)} vs {num(a['vs_paper'][k]['published_x100'], 1)}**"
                for k in bad))
        iv = (f"[{num((a['lo'] or 0) * 100, 2)}, {num((a['hi'] or 0) * 100, 2)}] · `{a['estimator']}` · "
              f"{a['n_clusters']} {a['cluster_unit']} clusters") if a["lo"] is not None else "UNAVAILABLE"
        dcv = ("self" if a["key"] == "CV" else
               ((f"{sgn((p.get('delta') or 0) * 100, 2)} [{sgn((p.get('lo') or 0) * 100, 2)}, "
                 f"{sgn((p.get('hi') or 0) * 100, 2)}]" + (" -> separated" if p.get("separated") else " -> TIE"))
                if p else "—"))
        L.append(row([a["label"], a["kind"], cmd_mark(a["declared"]), num(a["pdms_x100"], 4), iv, f"{a['n']:,}"]
                     + [num(a["x100"].get(k), 2) for k in ("NC", "DAC", "DDC", "TTC", "C", "EP")]
                     + [dcv, " · ".join(cellv) or "—"]))
    L.append("")
    return L


def print_convention_block(model: dict, proto: str) -> list:
    """The probe's OWN two verdicts, printed as the file carries them.

    ⛔ The file holds a STANDING verdict (the paper: UNSETTLED) and a POST-MEASUREMENT block (this one
    cell: SETTLED). They are different scopes, and collapsing them is how *"the paper truncates"*
    reached a relay: a claim true of one cell, phrased as a claim about the paper — the
    *true but wrong for the reader* class. Both are printed, each with its scope named."""
    n = _note(model, "print_convention", proto)
    if not n:
        return []
    t = n.get("tally") or {}
    L = [f"*⭐ **Printing convention — {n['question']}***", "",
         f"*⛔ **THE CELL IS SETTLED; THE PAPER IS NOT — and the page prints both scopes, because one cell "
         f"demonstrably truncated is not a convention.***", ""]
    L += header(["scope", "the artifact's own verdict", "what it rests on"])
    if n.get("pm_cell"):
        L.append(row(["**this CV cell** (post-measurement, 2026-09-20)", n["pm_cell"],
                      (n.get("pm_what") or "") + " ⚠️ ASSUMPTION, stated and not itself measured: " +
                      (n.get("pm_assumption") or "")]))
    L.append(row(["**the PAPER** (standing verdict, unchanged)", n["verdict"],
                  "tally over its leaderboard cells: " + " · ".join(f"**{k} {v}**" for k, v in t.items()) +
                  f"; the one test internal to the paper implies **{n['internal_implies']}** "
                  f"(*{n['internal_caveat']}*)" +
                  (f". {n['pm_paper']}" if n.get("pm_paper") else "")]))
    L += ["", f"*⚠️ **What changed is the cell's EVIDENCE CLASS, not a new measurement of the paper**: from "
          f"*print vs an INHERITED leaderboard value* to *print vs a value WE measured*. ⛔ And the counterfactual "
          f"is the honest scope — had the paper's CV been 20.6499, **both** conventions would print 20.6 and the "
          f"cell would say nothing. ⭐ What W3's measurement supersedes: W1's 8/8 and E1's 19/19 were both about "
          f"the PAPER alone; this one separates the two SOURCES. Prior: {n['prior']} "
          f"⇒ {n.get('pm_consequence') or n['consequence']} {n['consequence'] if n.get('pm_consequence') else ''} "
          f"Source `{n['source']}` (sha256 {n['sha12']}…)" +
          (f" — {n['pm_provenance'].rstrip('.')}" if n.get("pm_provenance") else "") + ".*", ""]
    return L


def v1_tables(model: dict, proto: str) -> list:
    pub = model["published"]
    rows = [r for r in pub["results"] if r["protocol"] == proto]
    assert_one_protocol(proto, rows)
    cols = ["system", "PDMS", "role", "encoder · data", "inputs (authors' claim)", "ego status", "source (key · table · p.)"]
    L = w3_navtest_block(model, proto)
    L += print_convention_block(model, proto)
    L += ["*The published field on this protocol:*", "", f"*{pub['protocols'][proto].get('ladder_note', '')}*", ""]
    L += header(cols)
    for r in sorted(rows, key=lambda r: (r["role"] == "context", r["role"] in ("privileged", "reference"), r["value"], r["id"])):
        enc = " · ".join(x for x in (r.get("encoder"), r.get("data")) if x) or "—"
        L.append(row([r["system"], pub_value(r, pub["protocols"]), r["role"], enc, r.get("modality", "—"),
                      cmd_mark(r.get("ego_status", "—")), src_ref(r)]))
    L.append(_tanitad_row(len(cols), pub["protocols"][proto]["tanitad_pending"]))
    return L


def navtest2_tables(model: dict, proto: str) -> list:
    pub = model["published"]
    rows = [r for r in pub["results"] if r["protocol"] == proto]
    assert_one_protocol(proto, rows)
    L = []
    groups = [("pre", "legacy EPDMS* (pre-#151)"), ("post", "corrected EPDMS (post-#151)"), ("unverified", "implementation NOT STATED by the source — out of comparison")]
    present = [(side, title) for side, title in groups if any(r["harness"]["fix151"] == side for r in rows)]
    for gi, (side, title) in enumerate(present):
        g = [r for r in rows if r["harness"]["fix151"] == side]
        L += [f"*{title}:*", ""]
        L += header(["system", "EPDMS", "role", "modality (authors' claim)", "source (key · table · p.)", "note"])
        for r in sorted(g, key=lambda r: (-r["value"], r["id"])):
            L.append(row([r["system"], pub_value(r, pub["protocols"]), r["role"], r.get("modality", "—"), src_ref(r),
                          r.get("notes") or r["harness"].get("basis", "")]))
        if gi == len(present) - 1:
            L.append(_tanitad_row(6, pub["protocols"][proto]["tanitad_pending"]).replace("⛔ NOT MEASURED — pending: ", "⛔ "))
        L.append("")
    return L


def nuscenes_tables(model: dict, proto: str) -> list:
    pub = model["published"]
    rows = [r for r in pub["results"] if r["protocol"] == proto]
    assert_one_protocol(proto, rows)
    meta = pub["protocols"][proto]
    dp = meta.get("decimals", 2)
    STD = ("l2_1s", "l2_2s", "l2_3s", "l2_avg", "col_1s", "col_2s", "col_3s", "col_avg")
    cols = ["system", "L2 1 s", "2 s", "3 s", "avg", "col % 1 s", "2 s", "3 s", "avg", "role",
            "ego status", "command", "source (key · table · p.)", "note"]
    L = []
    for k in ("harness_note", "gt_control_note", "command_note"):
        if meta.get(k):
            L += [f"*{meta[k]}*", ""]
    # one sub-table per metric IMPLEMENTATION: same convention, different code is still never one column
    impls = sorted({r.get("implementation", "—") for r in rows})
    for ii, impl in enumerate(impls):
        L += [f"*Implementation: {impl}*", ""] + header(cols)
        grp = sorted([r for r in rows if r.get("implementation", "—") == impl],
                     key=lambda r: (r["values"].get("l2_avg") is None, r["values"].get("l2_avg", 0.0), r["id"]))
        for r in grp:
            v = r["values"]
            note = r.get("notes", "")
            if not r.get("comparison_admissible", True):
                note = "⛔ NOT COMPARABLE — " + (r.get("inadmissible_reason") or note)
            # a value key outside the standard set is NEVER silently relabelled into a standard column
            extra = ", ".join(f"`{k}` {num(x, 3)}" for k, x in v.items() if k not in STD)
            L.append(row([r["system"]] + [num(v[k], dp) if v.get(k) is not None else "—" for k in STD]
                         + [r.get("role", "—"), r.get("ego_status", "—"), cmd_mark(r.get("command", "—")), src_ref(r),
                            (note + (" · " + extra if extra else "")).strip(" ·")]))
        if ii == len(impls) - 1:
            L.append(_tanitad_row(len(cols), meta["tanitad_pending"]))
        L.append("")
    L += ["*— = the source does not print that cell (never imputed). A cell outside the eight standard metrics is "
          "printed under `note` with its own key, never moved into a standard column.*", ""]
    return L


def b2d_tables(model: dict, proto: str) -> list:
    pub = model["published"]
    rows = [r for r in pub["results"] if r["protocol"] == proto]
    assert_one_protocol(proto, rows)
    dp = pub["protocols"][proto].get("decimals", 2)
    cols = ["system", "Driving Score", "Success Rate %", "role", "source (key · table · p.)", "note"]
    L = header(cols)
    for r in sorted(rows, key=lambda r: (-r["values"]["ds"], r["id"])):
        L.append(row([r["system"], num(r["values"]["ds"], dp), num(r["values"]["sr"], dp), r["role"], src_ref(r), r.get("notes", "")]))
    L.append(_tanitad_row(len(cols), pub["protocols"][proto]["tanitad_pending"]))
    L.append("")
    for nb in pub.get("not_banked", []):
        L.append(f"*Not banked, never compared: {nb['system']} — {nb['status']}.*")
    return L


RENDERERS = {
    "EPDMS_v2_navhard_two_stage": navhard_tables,
    "EPDMS_v2_warmup_two_stage": warmup_tables,
    "PDMS_v1_navtest": v1_tables,
    "EPDMS_v2_navtest_single_stage": navtest2_tables,
    "nuScenes_OL_L2_stp3": nuscenes_tables,
    "nuScenes_OL_L2_uniad": nuscenes_tables,
    "Bench2Drive_closed_loop": b2d_tables,
}


def external_field(model: dict, order: list) -> list:
    pub = model["published"]
    L = ["## 0E. External field — where TanitAD stands *(generated · one table per protocol, never merged)*", "",
         "*Every external number is re-read from a **banked PDF** (library key · table · PDF page; sha256 re-checked by "
         "`…/2026-09-19-leaderboard-currency/code/verify_published_against_pdfs.py`) and lives in "
         "`products/P7-TanitEval/benchmarks/published_results.json`. Modality and ego-status cells are **the authors' claims** — "
         "the NavSim server records no modality. ⚠️ Two EPDMS implementations exist (the #151 human-filter fix); our harness pin "
         "`autonomousvision/navsim@0a380a9` is post-fix, and a row's `#151 side` says what its source establishes. "
         "Evidence class PUBLISHED unless marked.*", ""]
    known = set(pub["protocols"])
    for i, proto in enumerate([p for p in order if p in known], 1):
        meta = pub["protocols"][proto]
        L += [f"### 0E.{i} `{proto}` — {meta['title']}", "", f"*{meta['definition']}*", ""]
        # the two protocol-level hazards that travel WITH the table, never in a footnote nobody re-reads
        for k in ("sample_overlap", "route_oracle", "stop_floor_rule"):
            if meta.get(k):
                L += [f"*{meta[k]}*", ""]
        L += RENDERERS[proto](model, proto)
        L.append("")
    return L


INTERNAL_T1 = "TanitAD_T1_refc_physicalai"

# ⛔ W2's qualifier, carried wherever a cross-track number is printed. It is not decoration: these are
# exactly the rows MODEL_REGISTRY.md quotes.
LATERAL_QUALIFIER = ("⚠️ **`cross_mae_m` is a lateral OFFSET at the MATCHED TIME INDEX, not a distance to the "
                     "path** (W2). A stopped arm and a moving arm can therefore tie on it *by construction* — "
                     "the stopped arm's offset stops growing because it stops moving, not because it is on the "
                     "path. ⛔ It may never be read as \"stays in the lane\".")


def with_lateral_qualifier(lines: list) -> list:
    """Append W2's cross-track qualifier to ANY block that prints a cross-track number.

    ⛔ It rides with the table rather than living in one section, because these rows are exactly the
    ones MODEL_REGISTRY.md quotes, and a qualifier a reader has to go and find is a qualifier that
    does not travel with the number."""
    if any(re.search(r"cross_(mae|bias)", str(l)) for l in lines) and LATERAL_QUALIFIER not in lines:
        return lines + ["", LATERAL_QUALIFIER, ""]
    return lines


def internal_t1_block(model: dict) -> list:
    """The programme's own internal standard, as its OWN table — ⛔ never merged with an EPDMS or PDMS
    column. Tier + loop per arm, the model-free floors beside every arm, the estimator and interval per
    row (including the artifact's own reason when it refuses one), and W2's lateral qualifier."""
    rows = [r for r in model["navsim"] if r["protocol"] == INTERNAL_T1]
    if not rows:
        return []
    assert_one_protocol(INTERNAL_T1, rows)
    L = [f"## 0D. TanitAD internal standard — `{INTERNAL_T1}` *(generated · its own table, never merged with an "
         f"EPDMS or PDMS column)*", "",
         "*The programme's primary internal surface: the four metric families with the paired episode-cluster "
         "bootstrap on PhysicalAI. ⛔ A number here is in METRES (or the metric's own unit) and shares no scale, "
         "no split and no estimator with any NavSim column — the two may not be compared.*", ""]
    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    order = {"floor": 0, "reference": 1, "control": 1, "model": 2, "ablation": 3}
    for run_id in sorted(runs):
        arms = runs[run_id]
        m0 = arms[0]
        prov = m0.get("tool_provenance") or {}
        if prov.get("verdict"):
            d = prov.get("discriminators") or {}
            L += [f"⛔ **The artifact declares its own scale: `{prov['verdict']}`** — "
                  f"{d.get('grid.n_episodes', '?')} episodes / {d.get('grid.n_windows', '?')} windows at model step "
                  f"{d.get('model.step', '?')}. ⇒ **this is a SMOKE grid, not a result**: every interval below is "
                  f"inadmissible under the RG-14 floor of 8 episodes, and the page prints it with the artifact's "
                  f"own refusal rather than omitting the row.", ""]
        L += [f"*Run `{run_id}` — checkpoint: {m0['ckpt']} · split `{m0['split']}` · {m0['tier_loop']}. "
              f"Headline `{m0['headline_column']}`. Source `{m0['source']}`.*", ""]
        floors = [a["key"] for a in sorted(arms, key=lambda a: a["key"]) if a["kind"] == "floor"]
        # the honest headline for THIS surface, derived from the run's own paired verdicts
        mods = [a for a in arms if a["kind"] == "model"]
        for mo in mods:
            wins = [k for k, pr in (mo.get("paired_native") or {}).items()
                    if (pr.get("interval") or {}).get("separated") and (pr.get("headline_delta") or 0) < 0]
            loses = [k for k, pr in (mo.get("paired_native") or {}).items()
                     if (pr.get("interval") or {}).get("separated") and (pr.get("headline_delta") or 0) > 0]
            if not wins:
                L += [f"⛔ **`{mo['label']}` separates from NO model-free floor on this run"
                      + (f" — and is separated WORSE than {', '.join('`' + k + '`' for k in loses)}" if loses else "")
                      + ".** The honest headline on this surface is that the arm does not beat doing nothing; "
                        "⚠️ and on a 2-episode grid even that is a smoke reading, not a result.", ""]
        model_free = [k for k in ("ha", "ha0", "ha0_ext") if any(a["key"] == k for a in arms)]
        cols = (["arm", "kind", "tier", "declared inputs", f"{m0['headline_column']} (mean)", "interval (estimator · n)"]
                + [f"Δ vs `{k}` [paired CI] → verdict" for k in model_free])
        L += header(cols)
        for a in sorted(arms, key=lambda a: (order.get(a["kind"], 9), a["headline_value"] if a["headline_value"] is not None else 9e9)):
            cells = [a["label"], a["kind"], a.get("arm_tier") or "—", cmd_mark(a["declared"]),
                     num(a["headline_value"], 4), _iv_native(a)]
            for k in model_free:
                cells.append("self" if a["key"] == k else _paired_native(a, k))
            L.append(row(cells))
        L += ["", f"*⛔ `—` is a pair the run did not compute; it is never imputed. Floors declared by the run: "
              f"{', '.join('`' + f + '`' for f in floors) or 'none'}.*", ""]
        fam = [a for a in arms if a.get("fam")]
        if fam:
            L += ["*The four families, per arm — ⛔ never pooled, and a family the artifact could not compute keeps "
                  "its own reason and n:*", ""]
            L += header(["arm", "family", "status", "n", "metrics (the artifact's own names and values)", "reason when absent"])
            for a in sorted(fam, key=lambda a: (order.get(a["kind"], 9), a["key"])):
                for f in ("longitudinal", "lateral", "tactical", "strategic"):
                    blk = (a["fam"] or {}).get(f)
                    if not blk:
                        continue
                    mets = ", ".join(f"`{k}` {num(v, 4) if isinstance(v, (int, float)) else v}"
                                     for k, v in (blk.get("metrics") or {}).items())
                    L.append(row([a["label"], f.upper(), blk.get("status", "—"), blk.get("n", "—"),
                                  mets or "—", blk.get("reason") or ""]))
    return with_lateral_qualifier(L)


def _iv_native(a: dict) -> str:
    """An interval in the metric's OWN units — with the artifact's refusal printed when it has one."""
    st = a.get("stat_interval") or {}
    if st.get("lo") is None:
        return "UNAVAILABLE"
    txt = f"[{num(st['lo'], 4)}, {num(st['hi'], 4)}] · `{st.get('estimator', '?')}` · {st.get('n_episodes', '?')} episodes"
    # the ARM'S OWN refusal, verbatim — not a sentence this renderer composes about it
    if a.get("interval") == "UNAVAILABLE" or isinstance(a.get("interval"), dict) and a["interval"].get("status") == "UNAVAILABLE":
        why = a.get("interval_reason") or "UNAVAILABLE"
        txt += f" — ⛔ INADMISSIBLE: {why.replace('UNAVAILABLE — ', '')}"
    return txt


def _paired_native(a: dict, floor: str) -> str:
    p = (a.get("paired_native") or {}).get(floor)
    if not p:
        return "—"
    iv = p.get("interval") or {}
    sep = iv.get("separated")
    adm = iv.get("admissible")
    v = f"{sgn(p.get('headline_delta'), 4)} {ci(iv.get('lo'), iv.get('hi'), 4)}".strip()
    if sep is None:
        return v + " (no interval)"
    # ⚠️ the winner is named by KEY, never as "model": three of the four arms on this surface are floors,
    # and "model WINS" on a floor-vs-floor pair is simply false.
    better = a["key"] if (p.get("headline_delta") or 0) < 0 else floor
    word = (("⬅ `%s` better" % better) if better == a["key"] else ("⛔ `%s` better" % better)) if sep else "TIE"
    return v + f" → {word}" + ("" if adm else " ⛔ *(interval INADMISSIBLE — the arm's own flag)*")


def orphan_runs_block(model: dict, order: list) -> list:
    """W1 runs on disk whose protocol has NO table on this page yet.

    ⛔ A run that exists and is rendered nowhere is the "built, tested and unreachable from its
    caller" class in leaderboard costume: the page would look complete while a real result sat
    unread. They are listed with their checkpoint identity so the omission is visible and countable."""
    known = (set(order) & set(model["published"]["protocols"])) | {INTERNAL_T1}
    rows = [r for r in model["navsim"] if r["protocol"] not in known]
    if not rows:
        return []
    L = ["### 0E.x Runs on disk with no table on this page *(generated)*", "",
         "*These W1 runs were read and validated, and their protocol has no rendered column yet — listed rather "
         "than dropped, because a result the page silently ignores is worse than one it says it cannot place.*", ""]
    L += header(["protocol", "run", "arm", "kind", "checkpoint (artifact's own `registry_key_display`)", "headline"])
    for r in sorted(rows, key=lambda r: (r["protocol"], r["run_id"], r["key"])):
        L.append(row([f"`{r['protocol']}`", f"`{r['run_id']}`", r["label"], r["kind"], r["ckpt"], _headline_cell(r)]))
    return L + [""]


def audit_block(model: dict) -> list:
    a, path = model["audit"], model["audit_path"]
    if not a:
        return [f"*Currency audit input `{path}` not found — no audit table rendered.*", ""]
    L = [f"## 0F. Currency audit {a.get('date', '')} *(generated from `{path}`)*", "", f"*{a.get('summary', '')}*", ""]
    L += header(["item", "registry", "raw artifact", "on this page before the audit", "status", "action"])
    for r in a["rows"]:
        L.append(row([r["item"], r.get("registry", "—"), r.get("raw_artifact", "—"), r.get("page_before", "—"), r["status"], r.get("action", "—")]))
    L.append("")
    return L


def canonical_digest(model: dict) -> str:
    import hashlib
    blob = json.dumps({k: model[k] for k in ("published", "internal", "navsim", "notes", "audit")}, sort_keys=True,
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def render_md(model: dict, cfg: dict) -> str:
    b, e = cfg["markers"]["begin"], cfg["markers"]["end"]
    L = [b, "<!-- GENERATED by `python -m taniteval.leaderboard build` (run from <repo>/taniteval). Do not hand-edit between "
         "the BENCH markers: edit the inputs and rebuild. Everything outside the markers is hand-written and is never touched. -->", "",
         "> ⭐ **THE RULE THIS SECTION IS BUILT ON — THE PRODUCER'S OWN WORDING WINS. A renderer that rebuilds a "
         "fact is a SECOND PRODUCER, and two producers of one fact are two things that can disagree.** So a cell "
         "here prints what the artifact SAYS, not a sentence composed about it: the checkpoint identity comes from "
         "`provenance.ckpt.registry_key_display` (computed once by `taniteval.bench.contract`), a refusal prints "
         "the artifact's own reason, a devkit line prints the producer's verbatim string, and a number keeps the "
         "UNITS its artifact declares. ⛔ Three defects in one day came from breaking this rule in three costumes: "
         "a re-derived checkpoint key, an assumed `×100` that printed **2.9098 m as \"290.98\"**, and a composed "
         "`+ 1 patches` that replaced a line naming three. ⇒ **if you find yourself formatting a fact the producer "
         "already formatted, read its field instead.**", ""]
    L += position_block(model)
    L += families_block(model)
    L += internal_t1_block(model)
    L += external_field(model, cfg["protocol_order"])
    L += orphan_runs_block(model, list(cfg["protocol_order"]) + [INTERNAL_T1])
    L += audit_block(model)
    L.append(f"*Inputs: {len(model['published']['results'])} published rows · {len(model['internal'])} internal rows · "
             f"{len(model['navsim'])} NAVSIM run rows · {len(model['audit']['rows']) if model['audit'] else 0} audit rows · "
             f"data-model digest `{canonical_digest(model)}` (changes only when an input value changes).*")
    for n in model["notes"]:
        if isinstance(n, str):
            L.append(f"*{n}*")
    L.append(e)
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ HTML
def _md_table_to_html(lines: list) -> str:
    out = ["<table>"]
    for i, ln in enumerate(lines):
        if i == 1:
            continue
        cells = [c.strip().replace("\\|", "|") for c in ln.strip().strip("|").split(" | ")]
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{html.escape(c).replace('&lt;br&gt;', '<br>')}</{tag}>" for c in cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def w5_charts(model: dict) -> str:
    """Leaderboard figures built with W5's OWN components (`taniteval.benchreport`) — imported, never
    edited. If that package is absent the page renders its tables only and says so."""
    try:
        from taniteval.benchreport.charts import hbar_chart, nice_domain
    except Exception:
        return ""
    out = []
    pub = model["published"]
    # 1 — the official column: every admissible published row, plus our own (score-less) attempt
    rows, vals = [], []
    for r in sorted([r for r in pub["results"]
                     if r["protocol"] == "EPDMS_v2_navhard_two_stage" and r.get("comparison_admissible")
                     and r["harness"]["fix151"] == "post"], key=lambda r: r["value"]):
        rows.append({"pub": r["id"], "label": r["system"], "value": r["value"], "fmt": "f1",
                     "cls": "m-s3" if r["role"] in ("floor", "privileged") else "m-s2",
                     "tip": f"{r['source']['library_key']} {r['source']['table']} p.{r['source']['page']}"})
        vals.append(r["value"])
    att = next((n for n in model["notes"] if isinstance(n, dict) and n.get("render") == "navhard_attempt"), None)
    if att:
        rows.append({"arm": "TanitAD-CV", "label": "TanitAD — CV run (no score)", "value": None, "fmt": "f1",
                     "refusal": f"UNAVAILABLE — runner exited {att['status']} in aggregation; no scores CSV"})
    if rows:
        out.append('<div class="card"><h3>NAVSIM v2 navhard two-stage — the official column</h3>'
                   + hbar_chart("fig-navhard", "navhard two-stage EPDMS (post-#151)", rows,
                                domain=nice_domain(vals, lo=0.0), unit="EPDMS ×100 — higher is better",
                                desc="Published rows re-read from banked PDFs; TanitAD has no score on this protocol yet.")
                   + "</div>")
    # 2 — our NAVSIM warmup arms on the stage-2 statistic, with the two mandatory floors as reference lines
    nav = [a for a in model["navsim"] if a["protocol"] == WARMUP and a["s2u_x100"] is not None]
    if nav:
        refs = tuple({"floor": a["key"], "label": a["key"], "value": a["s2u_x100"], "fmt": "f2"}
                     for a in nav if a["key"] in ("STOP", "CV"))
        rows2 = [{"arm": a["key"], "label": a["label"], "value": a["s2u_x100"], "fmt": "f2",
                  "cls": "m-s1" if a["kind"] == "model" else "m-s3", "tag": a["kind"]}
                 for a in sorted(nav, key=lambda a: a["s2u_x100"])]
        out.append('<div class="card"><h3>NAVSIM v2 warmup — S2-EPDMS-u (stage-2 statistic, not a two-stage EPDMS)</h3>'
                   + hbar_chart("fig-warmup", "warmup stage-2 statistic ×100", rows2,
                                domain=nice_domain([a["s2u_x100"] for a in nav], lo=0.0),
                                unit="S2-EPDMS-u ×100 — higher is better", refs=refs,
                                desc="No interval exists on warmup: 7 log groups < the RG-14 floor of 8.")
                   + "</div>")
    # 3 — our internal arms' headline level (each row carries its own floors in the table below)
    lv = [(r["claim"].split(" — ")[0] + f" · {r['tier']} · {r['loop']}", r["level"]["mean"]) for r in model["internal"]]
    if lv:
        rows3 = [{"arm": l, "label": l, "value": v, "fmt": "f4", "cls": "m-s1"} for l, v in sorted(lv, key=lambda x: x[1])]
        out.append('<div class="card"><h3>TanitAD internal arms — headline level (metre, lower is better)</h3>'
                   + hbar_chart("fig-internal", "internal headline ADE by arm", rows3,
                                domain=nice_domain([v for _, v in lv], lo=0.0), unit="ADE (m) — lower is better",
                                desc="Mixed tiers and corpora: read each row's grid, floors and margins in the table.")
                   + "</div>")
    return "".join(out)


def render_html(md_section: str, w5_html: str) -> str:
    """The leaderboard page. Uses W5's page shell (`taniteval.benchreport.page.page`) when importable,
    so the charts' palette, tooltips and print rules are W5's; otherwise a minimal self-contained shell."""
    body, buf = [], []

    def flush():
        if buf:
            body.append(_md_table_to_html(buf))
            buf.clear()
    for ln in md_section.splitlines():
        if ln.startswith("|"):
            buf.append(ln)
            continue
        flush()
        if ln.startswith("<!--") or not ln.strip():
            continue
        if ln.startswith("### "):
            body.append(f"<h3>{html.escape(ln[4:])}</h3>")
        elif ln.startswith("## "):
            body.append(f"<h2>{html.escape(ln[3:])}</h2>")
        else:
            body.append(f"<p>{html.escape(ln.lstrip('* ').rstrip('*'))}</p>")
    flush()
    css = (":root{--bg:#ffffff;--fg:#1c1c1e;--mut:#5b5b62;--line:#d9d9de;--head:#f3f3f6;--acc:#0b5cad}"
           "@media (prefers-color-scheme: dark){:root{--bg:#141417;--fg:#ececf0;--mut:#a3a3ad;--line:#34343b;--head:#1d1d22;--acc:#6cb2ff}}"
           "body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;"
           "max-width:1400px;padding:16px}h1{font-size:22px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid var(--line)}"
           "h3{font-size:15px;margin-top:20px;color:var(--acc)}p{color:var(--mut)}table{border-collapse:collapse;width:100%;"
           "display:block;overflow-x:auto;margin:8px 0 16px}th,td{border:1px solid var(--line);padding:4px 8px;vertical-align:top;"
           "font-size:12.5px}th{background:var(--head);text-align:left}")
    charts = w5_html or ("<p>Charts: W5's components (<code>taniteval/taniteval/benchreport</code>) were not importable at "
                         "build time — the tables below are the complete record.</p>")
    inner = ("<h1>TanitAD Leaderboard — generated section</h1>"
             "<p class=\"lede\">Generated by <code>python -m taniteval.leaderboard build</code> from the same inputs as the "
             "marker-delimited section of <code>Benchmarks &amp; Eval/LEADERBOARD.md</code>. One table per protocol; our rows "
             "carry tier, loop and their floors; an interval appears only where the estimator admits one.</p>"
             + charts + "\n" + "\n".join(body))
    # ⛔ ONE module name only. W5 renamed `html.py` → `page.py` precisely because a top-level module named
    # `html` SHADOWS THE STDLIB; keeping the old name as a fallback would quietly resurrect that hazard,
    # and an import that silently falls back cannot tell you the module is gone.
    try:
        import importlib
        page = getattr(importlib.import_module("taniteval.benchreport.page"), "page")
    except Exception:
        page = None
    if page is not None:
        return page("TanitAD Leaderboard", "<b>TanitAD leaderboard</b><span>generated · one table per protocol</span>",
                    inner, {"generator": "taniteval.leaderboard"})
    if True:
        return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                "content=\"width=device-width,initial-scale=1\"><title>TanitAD Leaderboard</title><style>" + css +
                "</style></head><body>\n" + inner + "\n</body></html>\n")
