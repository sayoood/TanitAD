#!/usr/bin/env python3
"""P5-C read instrument B8: the printed-twice renderer (PREREG_P5C section 7).

Renders `p5c_read.py`'s JSON as text and REFUSES (exit 2, nothing written) when
  (1) a number is printed without its tracked AND transparent twin and without the label "no tracker in this number";
  (2) any rendered text contains the banned wording ("drives better", "safer", "better driving" and synonyms, card 7: banned in every
      result, register row, report and paper text derived from the card, INCLUDING under TI);
  (3) a verdict carries other than exactly one transparent qualifier (TM / TM+C / TI / TU), or a qualifier whose condition the CIs do not meet.
Item shapes accepted by `render_number`:
  pair        {"tracked": ci, "transparent": ci, ["TD": ci]}        ci = {point, lo, hi}
  no tracker  {"no_tracker": true, "point"|"value": ...}           printed with the label
  mechanism   {"mechanism": true, "ci": ci, "definition": str}      the definition holds both scorers (card 8.1)
Anything else is refused.
  python p5c_render.py --read read.json --out report.md [--notes "free text, also checked"]
"""
from __future__ import annotations

import argparse
import json
import re
import sys

NO_TRACKER = "no tracker in this number"
QUALS = ("TM+C", "TM", "TI", "TU")
BANNED = [
    r"\bdr(?:ives?|iving|ove|iven)\s+(?:\w+\s+){0,2}?(?:better|well|safer|safely|smoother|smoothly|more\s+safely|more\s+smoothly)\b",
    r"\b(?:better|improved|superior|safer|safe|smoother|higher[- ]quality|more\s+reliable)\s+(?:driving|driver|drive)\b",
    r"\bsafer\b", r"\bsafely\b", r"\bmore\s+safe\b", r"\bsafety\s+(?:gain|improvement|benefit|win)s?\b",
    r"\bimprov\w*\s+(?:the\s+)?(?:driving|safety|driver)\b", r"\bdrives?\s+like\s+a\s+human\b",
    r"\bbetter\s+at\s+driving\b", r"\bdrive\s+(?:well|better)\b", r"\bdriving\s+(?:quality|skill)\s+(?:gain|improvement|is\s+better)\b",
]
_BANNED_RE = [re.compile(p, re.I) for p in BANNED]


class RenderRefused(Exception):
    pass


def banned_hits(text: str) -> list:
    return [m.group(0) for r in _BANNED_RE for m in r.finditer(text or "")]


def check_text(text: str, where: str = "text") -> str:
    h = banned_hits(text)
    if h:
        raise RenderRefused(f"banned wording in {where}: {sorted(set(h))} (card 7, PI condition 2)")
    return text


def fmt_ci(ci: dict, signed: bool = True, nd: int = 2) -> str:
    f = (lambda v: f"{v:+.{nd}f}") if signed else (lambda v: f"{v:.{nd}f}")
    return f"{f(ci['point'])} [{f(ci['lo'])}, {f(ci['hi'])}]"


def render_number(name: str, item: dict, signed: bool = True) -> str:
    if not isinstance(item, dict):
        raise RenderRefused(f"{name}: not an item")
    if item.get("no_tracker") is True:
        v = item.get("point", item.get("value"))
        if v is None:
            raise RenderRefused(f"{name}: no_tracker item carries no value")
        return f"{name}: {v} ({NO_TRACKER})"
    if item.get("mechanism") is True:
        if "ci" not in item or "definition" not in item:
            raise RenderRefused(f"{name}: mechanism item lacks ci / definition")
        return f"{name}: {fmt_ci(item['ci'], signed)} ({item['definition']})"
    if "tracked" in item and "transparent" in item:
        s = f"{name}: tracked {fmt_ci(item['tracked'], signed)} | transparent {fmt_ci(item['transparent'], signed)}"
        if "TD" in item:
            s += f" | tracker dependence TD {fmt_ci(item['TD'], signed)}"
        return s
    have = [k for k in ("tracked", "transparent") if k in item]
    raise RenderRefused(f"{name}: printed with only {have or 'neither scorer'} and no '{NO_TRACKER}' label: the tracked AND transparent twin is required")


def check_qualifier(res: dict) -> dict:
    """exactly ONE qualifier, and its condition must hold on this very read's CIs (section 7 table)."""
    q = res.get("qualifier")
    if not isinstance(q, dict) or q.get("code") not in QUALS:
        raise RenderRefused(f"verdict carries no single transparent qualifier from {QUALS}: {q!r}")
    h = res["d_headline"]
    td, dt = h["TD"], h["d_T_transparent"]
    ok = {"TM+C": td["lo"] > 0 and dt["hi"] < 0, "TM": td["lo"] > 0 and dt["lo"] <= 0 and not dt["hi"] < 0,
          "TI": dt["lo"] > 0, "TU": not (td["lo"] > 0 or dt["lo"] > 0)}
    if not ok[q["code"]]:
        raise RenderRefused(f"qualifier {q['code']} does not match the CIs (TD {fmt_ci(td)}, d_T {fmt_ci(dt)})")
    return q


def claim_block(res: dict) -> list:
    v = res["verdict"]["verdict"]
    core = v.split(":", 1)[-1]
    h = res["d_headline"]
    q = check_qualifier(res)
    nums = (f"tracked {fmt_ci(h['d_tracked'])}, transparent {fmt_ci(h['d_T_transparent'])}, tracker dependence TD {fmt_ci(h['TD'])}, qualifier {q['code']}")
    out = [f"VERDICT: {v}  ({res['verdict']['wording']})"]
    if core == "LEVER":
        out.append(f"CLAIM: C1's plans survive the benchmark's tracker (NAVSIM v1.1 LQR) better than C0's: {nums}.")
    elif core == "SEPARATED-SINGLE":
        out.append(f"CLAIM: separated, single seed, not yet a lever claim: {nums}.")
    else:
        out.append(f"DESCRIPTIVE ONLY (no claim): {nums}.")
    qt = q["text"]
    if q["code"] == "TM+C":
        qt = f"the gain exists only through NAVSIM's tracker; C1's plans are worse as paths by d_T {fmt_ci(h['d_T_transparent'])}"
    out.append(f"QUALIFIER {q['code']}: {qt}")
    if q["code"] == "TI":
        out.append("NOTE: TI says the paths score higher on a formula. It is not a closed-loop result.")
    if res.get("REHEARSAL"):
        out.insert(0, f"*** {res['REHEARSAL']['label']} -- {res['REHEARSAL']['claim']} ***")
    return out


def _nt(name: str, text: str) -> str:
    return render_number(name, {"no_tracker": True, "value": text})


def _acc(c: dict) -> str:
    a = c["accuracy"]
    k = c["kappa"]
    kp = "undefined" if k["point"] is None else f"{k['point']:.3f}"
    return f"{a['point']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] (n {c['n']}), kappa {kp}"


def render_tactical(t) -> list:
    """TACTICAL block (B12): per family, never pooled; a family that could not be computed prints its reason and n."""
    if not t:
        raise RenderRefused("the read carries no tactical block (it must say NOT COMPUTED, never be absent)")
    L = ["== TACTICAL family (reported, no decision authority; plan-side numbers, not tracker scores) =="]
    if t.get("rehearsal"):
        L.append("TACTICAL REHEARSAL: subset, not the registered read")
    ca = t["command_adherence"]
    if ca["status"] != "COMPUTED":
        L.append(_nt("command adherence", f"{ca['status']}: {ca.get('reason') or ca.get('invalid')} (n {ca.get('n', ca.get('n_tokens'))})"))
    else:
        for role, a in ca["arms"].items():
            L.append(_nt(f"{role} command adherence, turn-active (headline)", _acc(a["turn_active"])))
            L.append(_nt(f"{role} command adherence, all tokens", _acc(a["all_tokens"])))
            L.append(_nt(f"{role} turn-active confusion (rows LEFT, STRAIGHT, RIGHT command; cols plan)", str(a["turn_active"]["confusion_command_rows_plan_cols"])))
        for pair, v in ca["paired_C1_minus_x"].items():
            d = v["turn_active"]["accuracy_delta"]
            L.append(_nt(f"{pair} turn-active accuracy delta", f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] separated {d['separated']}"))
        fl = ca["random_proposal_floor"]
        if fl.get("status") == "NOT COMPUTED":
            L.append(_nt("random-proposal floor", f"NOT COMPUTED: {fl['reason']} (n {fl['n']}); the adherence read is INCOMPLETE without it"))
        else:
            f = fl["turn_active"]["floor_accuracy"]
            L.append(_nt("random-proposal floor, turn-active accuracy", f"{f['point']:.3f} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}]; floor >= actual: {fl['turn_active']['floor_ge_actual_point']}"))
        L.append(_nt("command adherence gates", ", ".join(f"{k}={'ok' if v['ok'] else ('not verified' if v['ok'] is None else 'FAILED')}" for k, v in ca["gates"].items())))
    for arm, s in t["selection_quality"].items():
        if s["status"] == "COMPUTED":
            lv = s["levels_x100"]
            L.append(_nt(f"{arm} selection quality x100 (random / actual / best-of-64)", " / ".join(f"{lv[k]['mean']:.2f}" for k in ("random", "actual", "best_of_64")) + f" (n {s['n_tokens']})"))
        else:
            L.append(_nt(f"{arm} selection quality", f"{s['status']}: {s.get('reason') or s.get('invalid')} (n {s.get('n', s.get('n_tokens'))})"))
    g = t["goal_setting"]
    L.append(_nt("tactical goal-setting", f"{g['status']}: {g['reason']} (n {g['n']})"))
    return L


def render_controls(res: dict) -> list:
    c = res.get("controls") or {}
    L = ["== read controls =="]
    for x in c.get("supplied", []):
        L.append(f"control {x.get('kind') or 'unnamed'} ({x['file']}): {'PASS' if x['pass'] else 'FAIL'}")
    if c.get("missing_kinds"):
        L.append("required controls MISSING: " + ", ".join(c["missing_kinds"]))
    s = res.get("sealed")
    if s:
        L.append(f"SEALED.json {s['sealed_sha256'][:12]}: {s['verdict']}; arms " + ", ".join(f"{a} {h[:12]}" for a, h in s["arms"].items()))
    return L


def render(res: dict, notes: str = "") -> str:
    lines = claim_block(res)
    lines += ["", f"n tokens {res['n_tokens']}, logs {res['n_logs']}; {res['estimator']['kind']}, {res['estimator']['draws']} draws, seed {res['estimator']['seed']}, "
              f"{res['estimator']['interval']}; answers: {res['estimator']['answers']}", f"units: {res['units']}", ""]
    lines.append("== levels (each printed tracked and transparent) ==")
    for arm, ms in res["levels"].items():
        for m, it in ms.items():
            lines.append(render_number(f"{arm} {m}", it, signed=False))
    for pair, ms in res["contrasts"].items():
        lines.append(f"== contrast {pair} ==")
        for m, it in ms.items():
            lines.append(render_number(f"{pair} {m}", it))
    lines.append("== rates ==")
    for arm, blk in res["rates"]["arms"].items():
        for k, it in blk.items():
            lines.append(render_number(f"{arm} {k}", it, signed=False))
    for pair, blk in res["rates"]["contrasts"].items():
        for k, it in blk.items():
            lines.append(render_number(f"{pair} {k}", it))
    lines.append("== guards and clauses ==")
    c = res["verdict"]["clauses"]
    lines.append("clauses: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    lines.append(f"B = {res['B']['value']} ({res['B']['arithmetic']}); 2F3 = {res['B']['2F3_beside']} beside it, decides nothing")
    lines.append(f"G2: {res['g2']['status']}; adverse (tracked-executed frame only): {[a['component'] for a in res['g2']['adverse']]}")
    lines += render_tactical(res.get("tactical"))
    lines += render_controls(res)
    lines.append(f"STRATEGIC family: {res['strategic_family']['status']} ({res['strategic_family']['reason']})")
    lines.append(f"goal 94.55: {res['goal']['status']}; deploy: {res['deploy']['outcome']}")
    lines.append(render_number("training-loss-like quantities", {"no_tracker": True, "value": "none in this read"}))
    if notes:
        lines += ["", "NOTES: " + notes]
    return check_text("\n".join(lines) + "\n", "the rendered report")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    res = json.load(open(a.read, encoding="utf-8"))
    try:
        txt = render(res, a.notes)
    except RenderRefused as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print("ZZP5CR_RENDER_OK", len(txt.splitlines()), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
