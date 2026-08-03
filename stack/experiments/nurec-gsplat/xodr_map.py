#!/usr/bin/env python3
"""Minimal, self-contained ASAM OpenDRIVE reader — the part a junction probe needs.

Scope on purpose: reference-line geometry, lane widths, lane-section structure,
junction connections + laneLinks.  Nothing else.  It exists so the junction
measurements in ``junction_probe.py`` are re-derived from ``map.xodr`` rather
than inherited from another agent's JSON.

Geometry primitives supported: ``line``, ``arc``, ``spiral`` (numeric Fresnel),
``poly3``, ``paramPoly3``.  The night scene 00040136 uses only ``line``/``arc``,
but the HF survey has to read arbitrary scenes, so all five are implemented and
the spiral has a closed-form regression test (a spiral of zero curvature change
must equal an arc, and a spiral of zero curvature must equal a line).

No third-party dependency beyond numpy.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Geometry", "LaneSection", "Lane", "Road", "Junction", "OpenDriveMap",
    "load_xodr", "parse_geo_reference",
]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
@dataclass
class Geometry:
    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str
    params: dict = field(default_factory=dict)

    def eval(self, ds: float) -> tuple[float, float, float]:
        """Return (x, y, hdg) at arc length ``ds`` from this record's start."""
        ds = max(0.0, min(ds, self.length))
        k = self.kind
        if k == "line":
            return (self.x + ds * math.cos(self.hdg),
                    self.y + ds * math.sin(self.hdg),
                    self.hdg)
        if k == "arc":
            curv = self.params["curvature"]
            if abs(curv) < 1e-12:
                return (self.x + ds * math.cos(self.hdg),
                        self.y + ds * math.sin(self.hdg), self.hdg)
            r = 1.0 / curv
            dh = ds * curv
            # centre is at +90 deg from hdg for positive curvature
            cx = self.x - r * math.sin(self.hdg)
            cy = self.y + r * math.cos(self.hdg)
            return (cx + r * math.sin(self.hdg + dh),
                    cy - r * math.cos(self.hdg + dh),
                    self.hdg + dh)
        if k == "spiral":
            c0 = self.params["curvStart"]
            c1 = self.params["curvEnd"]
            return self._spiral_eval(ds, c0, c1)
        if k in ("poly3", "paramPoly3"):
            return self._poly_eval(ds)
        raise ValueError(f"unsupported geometry kind {k!r}")

    # -- helpers ----------------------------------------------------------
    def _spiral_eval(self, ds, c0, c1, n_sub: int = 64):
        """Numeric integration of a clothoid.

        Falls back exactly onto the line/arc closed forms when the curvature is
        constant, which is what the regression test asserts.
        """
        if abs(c1 - c0) < 1e-14:
            g = Geometry(self.s, self.x, self.y, self.hdg, self.length,
                         "arc", {"curvature": c0})
            return g.eval(ds)
        dk = (c1 - c0) / self.length
        n = max(4, int(n_sub * ds / max(self.length, 1e-9)) + 4)
        t = np.linspace(0.0, ds, n)
        hdg = self.hdg + c0 * t + 0.5 * dk * t * t
        # trapezoid on cos/sin
        x = self.x + np.trapezoid(np.cos(hdg), t)
        y = self.y + np.trapezoid(np.sin(hdg), t)
        return (float(x), float(y), float(hdg[-1]))

    def _poly_eval(self, ds):
        p = self.params
        if self.kind == "poly3":
            a, b, c, d = p["a"], p["b"], p["c"], p["d"]
            u, v = ds, a + b * ds + c * ds ** 2 + d * ds ** 3
            dv = b + 2 * c * ds + 3 * d * ds ** 2
            du = 1.0
        else:
            pr = p["pRange"]
            pmax = self.length if pr == "arcLength" else 1.0
            t = ds / self.length * pmax
            u = p["aU"] + p["bU"] * t + p["cU"] * t ** 2 + p["dU"] * t ** 3
            v = p["aV"] + p["bV"] * t + p["cV"] * t ** 2 + p["dV"] * t ** 3
            du = p["bU"] + 2 * p["cU"] * t + 3 * p["dU"] * t ** 2
            dv = p["bV"] + 2 * p["cV"] * t + 3 * p["dV"] * t ** 2
        ch, sh = math.cos(self.hdg), math.sin(self.hdg)
        return (self.x + u * ch - v * sh,
                self.y + u * sh + v * ch,
                self.hdg + math.atan2(dv, du))


def _poly_eval_coeffs(el, ds):
    a = float(el.get("a", 0)); b = float(el.get("b", 0))
    c = float(el.get("c", 0)); d = float(el.get("d", 0))
    return a + b * ds + c * ds ** 2 + d * ds ** 3


# --------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------
@dataclass
class Lane:
    lid: int
    type: str
    level: bool
    widths: list                      # list of (sOffset, a, b, c, d)
    pred: int | None = None
    succ: int | None = None

    def width(self, ds_section: float) -> float:
        if not self.widths:
            return 0.0
        rec = self.widths[0]
        for w in self.widths:
            if w[0] <= ds_section + 1e-9:
                rec = w
            else:
                break
        t = ds_section - rec[0]
        return rec[1] + rec[2] * t + rec[3] * t ** 2 + rec[4] * t ** 3


@dataclass
class LaneSection:
    s: float
    left: list = field(default_factory=list)     # ascending lid 1,2,3...
    right: list = field(default_factory=list)    # descending lid -1,-2,...
    center: list = field(default_factory=list)


@dataclass
class Road:
    rid: str
    name: str
    length: float
    junction: str                     # "-1" when not inside a junction
    geometries: list = field(default_factory=list)
    lane_offsets: list = field(default_factory=list)   # (s, a, b, c, d)
    sections: list = field(default_factory=list)
    pred: tuple | None = None         # (elementType, elementId, contactPoint)
    succ: tuple | None = None
    rtype: str = ""
    speed: str = ""

    @property
    def in_junction(self) -> bool:
        return self.junction not in ("-1", "", None)

    # -- reference line ---------------------------------------------------
    def ref_pose(self, s: float) -> tuple[float, float, float]:
        s = max(0.0, min(s, self.length))
        g = self.geometries[0]
        for cand in self.geometries:
            if cand.s <= s + 1e-9:
                g = cand
            else:
                break
        return g.eval(s - g.s)

    def lane_offset(self, s: float) -> float:
        if not self.lane_offsets:
            return 0.0
        rec = self.lane_offsets[0]
        for r in self.lane_offsets:
            if r[0] <= s + 1e-9:
                rec = r
            else:
                break
        ds = s - rec[0]
        return rec[1] + rec[2] * ds + rec[3] * ds ** 2 + rec[4] * ds ** 3

    def section_at(self, s: float) -> LaneSection:
        sec = self.sections[0]
        for c in self.sections:
            if c.s <= s + 1e-9:
                sec = c
            else:
                break
        return sec

    def lane_t_bounds(self, s: float, lid: int) -> tuple[float, float] | None:
        """Inner and outer signed lateral offset of lane ``lid`` at ``s``.

        Returns ``None`` when the lane is not present in the section at ``s``.
        """
        sec = self.section_at(s)
        ds = s - sec.s
        t = self.lane_offset(s)
        side = sec.left if lid > 0 else sec.right
        sgn = 1.0 if lid > 0 else -1.0
        inner = t
        for ln in side:
            w = ln.width(ds)
            outer = inner + sgn * w
            if ln.lid == lid:
                return (inner, outer)
            inner = outer
        return None

    def lane_center_xy(self, s: float, lid: int):
        b = self.lane_t_bounds(s, lid)
        if b is None:
            return None
        x, y, hdg = self.ref_pose(s)
        t = 0.5 * (b[0] + b[1])
        return (x - t * math.sin(hdg), y + t * math.cos(hdg), hdg, abs(b[1] - b[0]))

    def sample_lane(self, lid: int, step: float = 1.0):
        """(N,2) centreline + (N,) width for a lane, over the whole road."""
        n = max(2, int(math.ceil(self.length / step)) + 1)
        ss = np.linspace(0.0, self.length, n)
        pts, ws = [], []
        for s in ss:
            r = self.lane_center_xy(float(s), lid)
            if r is None:
                continue
            pts.append((r[0], r[1]))
            ws.append(r[3])
        if not pts:
            return np.zeros((0, 2)), np.zeros((0,))
        return np.asarray(pts, float), np.asarray(ws, float)

    def driving_lane_ids(self) -> list:
        out = []
        for sec in self.sections:
            for ln in list(sec.left) + list(sec.right):
                if ln.type == "driving" and ln.lid not in out:
                    out.append(ln.lid)
        return out


@dataclass
class Junction:
    jid: str
    name: str
    # connections: list of dicts {incomingRoad, connectingRoad, contactPoint, laneLinks:[(from,to)]}
    connections: list = field(default_factory=list)


@dataclass
class OpenDriveMap:
    header: dict
    roads: dict                       # rid -> Road
    junctions: dict                   # jid -> Junction

    def junction_roads(self, jid: str) -> list:
        return [r for r in self.roads.values() if r.junction == jid]

    def internal_road_ids(self) -> set:
        return {r.rid for r in self.roads.values() if r.in_junction}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _f(el, k, default=0.0):
    v = el.get(k)
    return default if v is None else float(v)


def load_xodr(path) -> OpenDriveMap:
    root = ET.parse(str(path)).getroot()
    h = root.find("header")
    header = dict(h.attrib) if h is not None else {}
    if h is not None:
        gr = h.find("geoReference")
        if gr is not None and gr.text:
            header["geoReference"] = gr.text.strip()

    roads = {}
    for rel in root.findall("road"):
        r = Road(rid=rel.get("id"), name=rel.get("name", ""),
                 length=_f(rel, "length"), junction=rel.get("junction", "-1"))
        tel = rel.find("type")
        if tel is not None:
            r.rtype = tel.get("type", "")
            sp = tel.find("speed")
            if sp is not None:
                r.speed = f"{sp.get('max','')} {sp.get('unit','')}".strip()
        lk = rel.find("link")
        if lk is not None:
            for tag, attr in (("predecessor", "pred"), ("successor", "succ")):
                e = lk.find(tag)
                if e is not None:
                    setattr(r, attr, (e.get("elementType"), e.get("elementId"),
                                      e.get("contactPoint")))
        pv = rel.find("planView")
        if pv is not None:
            for gel in pv.findall("geometry"):
                s = _f(gel, "s"); x = _f(gel, "x"); y = _f(gel, "y")
                hdg = _f(gel, "hdg"); length = _f(gel, "length")
                kind, params = None, {}
                for child in gel:
                    kind = child.tag
                    if kind == "arc":
                        params = {"curvature": _f(child, "curvature")}
                    elif kind == "spiral":
                        params = {"curvStart": _f(child, "curvStart"),
                                  "curvEnd": _f(child, "curvEnd")}
                    elif kind == "poly3":
                        params = {k: _f(child, k) for k in "abcd"}
                    elif kind == "paramPoly3":
                        params = {k: _f(child, k) for k in
                                  ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV")}
                        params["pRange"] = child.get("pRange", "normalized")
                    break
                if kind is None:
                    kind, params = "line", {}
                r.geometries.append(Geometry(s, x, y, hdg, length, kind, params))
        r.geometries.sort(key=lambda g: g.s)

        lanes_el = rel.find("lanes")
        if lanes_el is not None:
            for lo in lanes_el.findall("laneOffset"):
                r.lane_offsets.append((_f(lo, "s"), _f(lo, "a"), _f(lo, "b"),
                                       _f(lo, "c"), _f(lo, "d")))
            r.lane_offsets.sort(key=lambda t: t[0])
            for sel in lanes_el.findall("laneSection"):
                sec = LaneSection(s=_f(sel, "s"))
                for side_name, bucket in (("left", sec.left), ("center", sec.center),
                                          ("right", sec.right)):
                    side_el = sel.find(side_name)
                    if side_el is None:
                        continue
                    for lel in side_el.findall("lane"):
                        lid = int(lel.get("id"))
                        widths = []
                        for wel in lel.findall("width"):
                            widths.append((_f(wel, "sOffset"), _f(wel, "a"),
                                           _f(wel, "b"), _f(wel, "c"), _f(wel, "d")))
                        widths.sort(key=lambda t: t[0])
                        ln = Lane(lid=lid, type=lel.get("type", ""),
                                  level=(lel.get("level") == "true"), widths=widths)
                        lnk = lel.find("link")
                        if lnk is not None:
                            p = lnk.find("predecessor"); s_ = lnk.find("successor")
                            if p is not None:
                                ln.pred = int(p.get("id"))
                            if s_ is not None:
                                ln.succ = int(s_.get("id"))
                        bucket.append(ln)
                # inner-to-outer ordering
                sec.left.sort(key=lambda l: l.lid)
                sec.right.sort(key=lambda l: -l.lid)
                r.sections.append(sec)
            r.sections.sort(key=lambda s: s.s)
        if not r.sections:
            r.sections.append(LaneSection(s=0.0))
        roads[r.rid] = r

    junctions = {}
    for jel in root.findall("junction"):
        j = Junction(jid=jel.get("id"), name=jel.get("name", ""))
        for cel in jel.findall("connection"):
            links = [(int(l.get("from")), int(l.get("to")))
                     for l in cel.findall("laneLink")]
            j.connections.append({
                "id": cel.get("id"),
                "incomingRoad": cel.get("incomingRoad"),
                "connectingRoad": cel.get("connectingRoad"),
                "contactPoint": cel.get("contactPoint"),
                "laneLinks": links,
            })
        junctions[j.jid] = j
    return OpenDriveMap(header=header, roads=roads, junctions=junctions)


def parse_geo_reference(s: str) -> dict:
    """proj4 string -> dict.  Tolerates the malformed ``+=alt_0=0`` token that
    the DeepMap exporter emits (measured on 00040136)."""
    out, bad = {}, []
    for tok in s.split():
        if not tok.startswith("+"):
            bad.append(tok); continue
        body = tok[1:]
        if body.startswith("="):
            bad.append(tok); continue
        if "=" in body:
            k, v = body.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[body] = True
    out["_malformed"] = bad
    return out
