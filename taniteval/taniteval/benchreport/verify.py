"""Parse a rendered report back and check it against ``summary.json`` — INDEPENDENT of the renderer.

It shares NO code with ``render.py`` / ``charts.py``: its own HTML tree (stdlib ``html.parser``), its own
RFC-6901 pointer resolver, its own number formats (``%``-style, not the renderer's f-strings), its own
label lookup (straight from summary.json). What it proves, for EVERY number stamped ``data-k``:

1. the pointer resolves in summary.json to a number, and ``data-v`` equals it EXACTLY;
2. the visible text equals that value at the declared precision (``data-fmt``);
3. the number sits inside the row / group of the SAME arm its pointer names (``data-arm``), and that
   group's visible label (``data-role="arm-label"``) is THAT arm's label — a swapped label goes RED;
4. every ``data-refusal-k`` points at a block that really has no value (a rendered "UNAVAILABLE" can
   never hide a number summary.json holds);
5. every published row drawn carries the run's protocol tag and its exact published value;
6. coverage: every arm's headline (value or refusal), every OK paired W/T/L, every S2-EPDMS-u value
   and every arm × family is present.

    python -m taniteval.benchreport.verify <run_dir>
"""
from __future__ import annotations

import json
import math
import sys
from html.parser import HTMLParser
from pathlib import Path

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
         "track", "wbr"}
_SVG_SELF = {"path", "rect", "line", "circle", "polyline", "polygon", "ellipse", "stop", "use"}


class _Node:
    __slots__ = ("tag", "attrs", "children", "parent", "text")

    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.children, self.parent, self.text = tag, dict(attrs), [], parent, []

    def all_text(self) -> str:
        out = list(self.text)
        for c in self.children:
            if isinstance(c, _Node):
                out.append(c.all_text())
        return "".join(out)

    def own_visible_text(self) -> str:
        """Text of this node and its children, excluding <title> (tooltip) children."""
        out = list(self.text)
        for c in self.children:
            if isinstance(c, _Node) and c.tag != "title":
                out.append(c.own_visible_text())
        return "".join(out)


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", [], None)
        self.cur = self.root
        self.nodes: list = []

    def handle_starttag(self, tag, attrs):
        n = _Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        self.nodes.append(n)
        if tag not in _VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        n = _Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        self.nodes.append(n)

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.text.append(data)


def resolve(doc, pointer: str):
    """RFC 6901 without the leading slash. Raises KeyError / IndexError if absent."""
    node = doc
    for raw in pointer.split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(key)]
        else:
            node = node[key]
    return node


def expected_text(v: float, f: str) -> str:
    if f == "f4":
        return "%.4f" % v
    if f == "f3":
        return "%.3f" % v
    if f == "f2":
        return "%.2f" % v
    if f == "f1":
        return "%.1f" % v
    if f == "int":
        return "%d" % int(round(v))
    if f == "pct1":
        return "%.1f %%" % (v * 100.0)
    if f == "sgn4":
        return "%+.4f" % v
    if f == "sgn3":
        return "%+.3f" % v
    raise ValueError(f"unknown format {f!r}")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def _ancestor_with(node, attr):
    n = node
    while n is not None and n.tag != "#root":
        if attr in n.attrs:
            return n
        n = n.parent
    return None


def _find_label(group) -> str | None:
    stack = [group]
    while stack:
        n = stack.pop()
        if n.attrs.get("data-role") == "arm-label":
            return " ".join(n.own_visible_text().split())
        stack.extend(c for c in reversed(n.children) if isinstance(c, _Node))
    return None


def verify_report(index_html, summary_json, published_json=None) -> dict:
    html_text = Path(index_html).read_text(encoding="utf-8")
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    published = json.loads(Path(published_json).read_text(encoding="utf-8")) if published_json else None
    t = _Tree()
    t.feed(html_text)
    t.close()
    labels = {a: (b.get("label") or a) for a, b in summary["arms"].items()}
    errors, seen_k, seen_ref = [], set(), set()
    n_numbers = 0

    def err(msg):
        if len(errors) < 500:
            errors.append(msg)

    for n in t.nodes:
        k = n.attrs.get("data-k")
        if k:
            n_numbers += 1
            seen_k.add(k)
            try:
                val = resolve(summary, k)
            except (KeyError, IndexError, ValueError):
                err(f"{k}: pointer does not resolve in summary.json")
                continue
            if not _is_num(val):
                err(f"{k}: summary value is not a number ({val!r})")
                continue
            try:
                dv = float(n.attrs.get("data-v", "nan"))
            except ValueError:
                dv = float("nan")
            if dv != float(val):
                err(f"{k}: data-v {n.attrs.get('data-v')} != summary {val!r}")
            f = n.attrs.get("data-fmt")
            if f:
                txt = " ".join(n.own_visible_text().split())
                exp = expected_text(float(val), f)
                if txt != exp:
                    err(f"{k}: shows {txt!r}, summary says {exp!r}")
            parts = k.split("/")
            if parts[0] == "arms" and len(parts) > 1:
                arm = parts[1].replace("~1", "/").replace("~0", "~")
                grp = _ancestor_with(n, "data-arm")
                if grp is not None:
                    if grp.attrs["data-arm"] != arm:
                        err(f"{k}: sits in the group of arm {grp.attrs['data-arm']!r}")
                    lab = _find_label(grp)
                    if lab is not None and lab != labels.get(grp.attrs["data-arm"]):
                        err(f"{k}: its row is labelled {lab!r}, summary label of {grp.attrs['data-arm']!r} is "
                            f"{labels.get(grp.attrs['data-arm'])!r}")
        rk = n.attrs.get("data-refusal-k")
        if rk:
            seen_ref.add(rk)
            try:
                blk = resolve(summary, rk)
            except (KeyError, IndexError, ValueError):
                err(f"refusal {rk}: pointer does not resolve")
                blk = None
            if isinstance(blk, dict) and _is_num(blk.get("value")) and blk.get("admissible", True) is not False \
                    and blk.get("status") in (None, "OK"):
                err(f"refusal {rk}: rendered UNAVAILABLE but summary holds value {blk.get('value')!r}")
            if _is_num(blk):
                err(f"refusal {rk}: rendered UNAVAILABLE but summary holds the number {blk!r}")
        pid = n.attrs.get("data-pub")
        if pid and "data-pub-v" in n.attrs:
            if published is None:
                err(f"published row {pid} rendered but no published file given to the verifier")
            else:
                row = next((r for r in published.get("results", []) if r.get("id") == pid), None)
                if row is None:
                    err(f"published row {pid}: not in the published file")
                else:
                    if row.get("protocol") != summary.get("protocol"):
                        err(f"published row {pid}: protocol {row.get('protocol')!r} on a "
                            f"{summary.get('protocol')!r} report (cross-protocol)")
                    if float(n.attrs["data-pub-v"]) != float(row["value"]):
                        err(f"published row {pid}: {n.attrs['data-pub-v']} != {row['value']!r}")
                    if " ".join(n.own_visible_text().split()) != expected_text(float(row["value"]), "f4"):
                        err(f"published row {pid}: text {n.own_visible_text()!r}")

    # every labelled group must carry ITS arm's label (catches swaps in rows without numbers too)
    for n in t.nodes:
        arm = n.attrs.get("data-arm")
        if arm and arm in labels:
            direct = _find_label(n)
            if direct is not None and direct != labels[arm]:
                err(f"group data-arm={arm!r} is labelled {direct!r} (summary: {labels[arm]!r})")
        if n.attrs.get("data-role") == "arm-short":
            grp = _ancestor_with(n, "data-arm")
            short = " ".join(n.own_visible_text().split())
            if grp is None:
                err(f"short label {short!r} outside any arm group")
            else:
                want = labels.get(grp.attrs["data-arm"], grp.attrs["data-arm"]).split(" · ")[0]
                if short != want:
                    err(f"reference line of {grp.attrs['data-arm']!r} is labelled {short!r} (summary: {want!r})")

    # ---- coverage
    missing = []
    for arm, a in summary["arms"].items():
        h = a.get("headline") or {}
        if _is_num(h.get("value")):
            if f"arms/{arm}/headline/value" not in seen_k and f"arms/{arm}/headline/x100" not in seen_k:
                missing.append(f"arms/{arm}/headline")
        elif f"arms/{arm}/headline" not in seen_ref:
            missing.append(f"arms/{arm}/headline (refusal)")
        s2 = (a.get("statistics") or {}).get("S2_EPDMS_u") or {}
        if _is_num(s2.get("value")) and f"arms/{arm}/statistics/S2_EPDMS_u/value" not in seen_k:
            missing.append(f"arms/{arm}/statistics/S2_EPDMS_u/value")
        for fl, p in (a.get("paired") or {}).items():
            if isinstance(p, dict) and p.get("status") == "OK":
                # ⛔ require exactly the counts the run HOLDS — on a two-stage protocol they may live
                # per stage (different token sets), pooled, or both (W1, 2026-09-20)
                scopes = [(f"arms/{arm}/paired/{fl}", p)]
                for st, b in (p.get("by_stage") or {}).items():
                    if isinstance(b, dict):
                        scopes.append((f"arms/{arm}/paired/{fl}/by_stage/{st}", b))
                for base, blk in scopes:
                    for x in ("wins", "ties", "losses"):
                        if _is_num(blk.get(x)) and f"{base}/{x}" not in seen_k:
                            missing.append(f"{base}/{x}")
            elif isinstance(p, dict) and p.get("status") == "UNAVAILABLE" and f"arms/{arm}/paired/{fl}" not in seen_ref:
                missing.append(f"arms/{arm}/paired/{fl} (refusal)")
        for fam in ("longitudinal", "lateral", "tactical", "strategic"):
            fb = (a.get("families") or {}).get(fam)
            if fb is None:
                continue
            if f"arms/{arm}/families/{fam}/n" not in seen_k and f"arms/{arm}/families/{fam}" not in seen_ref:
                missing.append(f"arms/{arm}/families/{fam}")
    status = "PASS" if not errors and not missing else "FAIL"
    return {"status": status, "n_numbers_checked": n_numbers, "n_refusals_checked": len(seen_ref),
            "n_errors": len(errors), "errors": errors[:100], "coverage_missing": missing[:100],
            "summary_sha_note": "summary.json read directly; renderer code not imported"}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    run = Path(argv[0])
    pub = None
    meta_pub = None
    idx = run / "report" / "index.html"
    for line in idx.read_text(encoding="utf-8").split("<meta ")[1:]:
        if 'name="tanitad-published"' in line:
            meta_pub = line.split('content="', 1)[1].split('"', 1)[0]
    if meta_pub:
        pub = meta_pub
    res = verify_report(idx, run / "summary.json", pub)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
