"""Traffic-rule knowledge and the country label (HICAP_V10_SPEC.md sec. 3.4, stream R).

Pieces
------
* `RuleBook`      versioned, APPEND-ONLY table of per-country rule records (`rulebook_v1.json`); ids never move, extension = new version.
* `COUNTRIES`     the FROZEN sorted country table; `country_id = COUNTRIES.index(name)` everywhere (labels, readout, retrieval).
* `RuleMemory`    country posterior p(c) -> membership weights w_i = sum_c p(c) 1[i in R_c]; relevance s_i = <W_q q_s2, W_r r_i>;
                  top-k (=8) by w_i * softmax(s_i); returns the retrieved rule embeddings, indices, scores, validity.
* `GatedMemory`   q' = q + tanh(g) * CrossAttn(q, retrieved rule tokens), g = 0 at init. rules=None and g=0 are BIT-EXACT no-ops
                  (same pattern as model.GatedAudio). The goal query is never written to (hard mask, not a learned gate).
* `CountryReadout` contrastive readout of the strategic query s1 against the country node embeddings ("driving in <country>: ..."),
                  temperature as in the HiCAP critics (theta = log(1/0.07) init, clamp at log 100).
* `rule_masks`    numeric prune rules -> a profile-level speed mask and a tactical-cell (lat x lon) mask, consistent with prune.py:
                  AND-ed after R1, only where the road type / signal state is known, never touching the guard set.

ADMISSIBILITY (CLAUDE.md BINDING 2026-08-03, both rules)
--------------------------------------------------------
* The country and the rules are NOT the output of a situation classifier. HiCAP has no situation-classifier head; the country
  posterior is a separate readout of the strategic query s1 trained against an upstream METADATA label (`data_collection.parquet`,
  the recording country), and the rule texts are curated statute knowledge, not a function of any scene label.
* What each input is computed from, at inference:
    - country posterior p(c):  s1 query <- fusion over (frozen-VLM camera taps, motion frames, ego v/a_lon/yaw-rate, nav, route).
                               PREDICTED arm (primary): vision-derived through the fused query. SUPPLIED arm (registered ablation):
                               a GNSS geofence one-hot (`CountryReadout.posterior(..., supplied=id)`); never the upstream label itself.
    - rule relevance s_i:       s2 query (same fusion) x cached frozen-text embeddings of the rule texts.
    - rule tokens:              cached frozen-text embeddings (constant per RuleBook version; no scene information).
    - rule masks:               country posterior + the measured ego state (v0) + OPTIONAL road type / signal state supplied by the
                                caller (when unknown the rule mask is a no-op, so nothing is guessed).
* The goal / route path NEVER reads the country posterior: GatedMemory writes only the s1/s2 level queries (the `goal` row is hard
  masked to zero update), and nothing in this module touches the goal head, nav, or route tokens.
* Leak check for the country label: the label is upstream metadata and is never an inference input except in the explicitly
  labelled SUPPLIED arm; both arms are reported (H-HC18). The scenario-classification vision-only rule is untouched: rules and
  country are not scenario labels and no ego signal enters the country readout except through the shared fused trunk (stated: the
  trunk is shared with every level; it carries no country information beyond what the cameras and motion show).
* Order inside one tick (to avoid circularity): fusion -> q -> country posterior from q_s1 (PRE-memory) -> retrieval with q_s2 ->
  GatedMemory(q) -> level heads. The rules retrieved from the country cannot feed back into the country posterior of the same tick.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vocab import FactoredVocab, LAT_V9, LON_V9, N_LAT, N_LON, STANDSTILL_V0

RULEBOOK_PATH = Path(__file__).with_name("rulebook_v1.json")
COUNTRIES: tuple[str, ...] = (
    "Austria", "Belgium", "Bulgaria", "Croatia", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain",
    "Sweden", "United States")                                     # FROZEN: sorted; id = index. Never reorder; append = new table version.
COUNTRY_ID = {c: i for i, c in enumerate(COUNTRIES)}

EFFECTS = ("prune", "prior", "memory")
EVIDENCE = ("PUBLISHED", "UNVERIFIED")
ROAD_TYPES = ("urban", "rural", "expressway", "motorway", "junction", "signalised_junction", "roundabout", "pedestrian_crossing", "any")
SPEED_ROADS = ("urban", "rural", "expressway", "motorway")
CATEGORIES = ("speed_limit", "right_turn_on_red", "priority_to_right", "roundabout", "emergency_vehicle", "pedestrian_crossing", "tram",
              "school_bus", "daytime_lights", "driving_side")
RTOR_MODES = ("permitted_after_stop_unless_signed", "green_arrow_sign_after_stop", "green_arrow_signal_after_stop", "prohibited")
REQUIRED = ("rule_id", "countries", "category", "road_type", "effect", "text", "args", "source", "evidence", "verification", "version_added")
LAT_TURN_R = LAT_V9.index("TURN_R")
LON_HALT = tuple(LON_V9.index(k) for k in ("HOLD", "STOP"))        # the only longitudinal classes compatible with "red means stop"


def validate_record(r: dict, countries=COUNTRIES) -> list[str]:
    """Schema check of one record; returns a list of human-readable errors (empty = valid)."""
    e = [f"missing {k}" for k in REQUIRED if k not in r]
    if e: return e
    if not isinstance(r["rule_id"], str) or not r["rule_id"]: e.append("rule_id must be a non-empty string")
    if not r["countries"] or any(c not in countries for c in r["countries"]): e.append(f"{r['rule_id']}: unknown country in {r['countries']}")
    if len(set(r["countries"])) != len(r["countries"]): e.append(f"{r['rule_id']}: duplicate country")
    if r["category"] not in CATEGORIES: e.append(f"{r['rule_id']}: category {r['category']}")
    if r["road_type"] not in ROAD_TYPES: e.append(f"{r['rule_id']}: road_type {r['road_type']}")
    if r["effect"] not in EFFECTS: e.append(f"{r['rule_id']}: effect {r['effect']}")
    if r["evidence"] not in EVIDENCE: e.append(f"{r['rule_id']}: evidence {r['evidence']}")
    if not isinstance(r["text"], str) or len(r["text"]) < 10: e.append(f"{r['rule_id']}: text too short")
    s = r["source"]
    if not isinstance(s, dict) or not s.get("title") or not str(s.get("url", "")).startswith("https://"): e.append(f"{r['rule_id']}: source needs title + https url")
    a = r["args"]
    if not isinstance(a, dict): e.append(f"{r['rule_id']}: args must be a dict")
    elif r["category"] == "speed_limit":
        if r["road_type"] not in SPEED_ROADS: e.append(f"{r['rule_id']}: speed rule on road_type {r['road_type']}")
        v = a.get("v_default_kmh", "missing")
        if v == "missing": e.append(f"{r['rule_id']}: speed rule needs v_default_kmh (null = no general limit)")
        elif v is not None and not (10 <= float(v) <= 200): e.append(f"{r['rule_id']}: implausible speed {v}")
        if v is None and r["effect"] == "prune": e.append(f"{r['rule_id']}: no numeric limit cannot be a prune rule")
        if a.get("v_ceiling_kmh") is not None and v is not None and a["v_ceiling_kmh"] < v: e.append(f"{r['rule_id']}: ceiling below default")
        if len(r["countries"]) != 1: e.append(f"{r['rule_id']}: speed rules are per country")
    elif r["category"] == "right_turn_on_red" and a.get("rtor") not in RTOR_MODES: e.append(f"{r['rule_id']}: rtor mode {a.get('rtor')}")
    if r["evidence"] == "PUBLISHED" and not s.get("url"): e.append(f"{r['rule_id']}: PUBLISHED without url")
    return e


@dataclass
class RuleBook:
    """Versioned, append-only rule table. `records[i]` is rule index i forever; a new rule is a new version (ids never move)."""
    records: list
    countries: tuple = COUNTRIES
    version: int = 1
    log: list = field(default_factory=list)
    _emb_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.countries = tuple(self.countries)
        if self.countries != COUNTRIES: raise ValueError("country table differs from the frozen COUNTRIES table")
        errs = [x for r in self.records for x in validate_record(r, self.countries)]
        ids = [r["rule_id"] for r in self.records]
        if len(ids) != len(set(ids)): errs.append("duplicate rule_id")
        if errs: raise ValueError("invalid RuleBook: " + "; ".join(errs[:10]))
        self._index = {r["rule_id"]: i for i, r in enumerate(self.records)}

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path = RULEBOOK_PATH) -> "RuleBook":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(d["countries"]) != COUNTRIES: raise ValueError(f"{path}: country table is not the frozen COUNTRIES table")
        return cls(records=d["records"], countries=tuple(d["countries"]), version=int(d["version"]))

    def to_json(self) -> dict:
        return dict(name="TanitAD HiCAP RuleBook", version=self.version, countries=list(self.countries), records=self.records,
                    log=self.log)

    # ---------------------------------------------------------------- table views
    def __len__(self): return len(self.records)
    @property
    def ids(self) -> list[str]: return [r["rule_id"] for r in self.records]
    def index(self, rule_id: str) -> int: return self._index[rule_id]
    def texts(self) -> list[str]: return [r["text"] for r in self.records]

    def membership(self) -> np.ndarray:
        """[n_countries, n_rules] bool: rule i applies in country c."""
        M = np.zeros((len(self.countries), len(self.records)), bool)
        for i, r in enumerate(self.records):
            for c in r["countries"]: M[COUNTRY_ID[c], i] = True
        return M

    def rules_for(self, country: str) -> list[dict]: return [r for r in self.records if country in r["countries"]]

    def speed_rule(self, country: str, road_type: str) -> dict | None:
        for r in self.records:
            if r["category"] == "speed_limit" and r["road_type"] == road_type and country in r["countries"]: return r
        return None

    def rtor_mode(self, country: str) -> str | None:
        for r in self.records:
            if r["category"] == "right_turn_on_red" and country in r["countries"]: return r["args"]["rtor"]
        return None

    # ---------------------------------------------------------------- append-only extension
    def add_rule(self, record: dict) -> "RuleBook":
        """Extension only through a NEW version: returns a new RuleBook; existing indices never move (NodeLibrary contract)."""
        if record.get("rule_id") in self._index: raise ValueError(f"{record.get('rule_id')} already in RuleBook v{self.version}; extension must be a new rule")
        r = dict(record); r.setdefault("version_added", self.version + 1)
        errs = validate_record(r, self.countries)
        if errs: raise ValueError("; ".join(errs))
        return RuleBook(self.records + [r], self.countries, self.version + 1, self.log + [("add", r["rule_id"])])

    # ---------------------------------------------------------------- country nodes and embeddings
    def country_summary(self, country: str) -> str:
        parts = []
        side = [r for r in self.rules_for(country) if r["category"] == "driving_side"]
        if side: parts.append(f"drive on the {side[0]['args']['side']}")
        sp = []
        for road in SPEED_ROADS:
            r = self.speed_rule(country, road)
            if r is None: continue
            a = r["args"]
            if a["v_default_kmh"] is not None: sp.append(f"{road} {a['v_default_kmh']:g} km/h")
            elif "v_advisory_kmh" in a: sp.append(f"{road} no general limit (advisory {a['v_advisory_kmh']:g} km/h)")
            elif "v_range_kmh" in a: sp.append(f"{road} {a['v_range_kmh'][0]:.0f}-{a['v_range_kmh'][1]:.0f} km/h by state")
            elif "v_typical_kmh" in a: sp.append(f"{road} signed, typically {a['v_typical_kmh']:g} km/h")
        if sp: parts.append(", ".join(sp))
        m = self.rtor_mode(country)
        if m: parts.append({"permitted_after_stop_unless_signed": "right turn on red allowed after stopping unless signed",
                            "green_arrow_sign_after_stop": "right turn on red only with a green-arrow sign after stopping",
                            "green_arrow_signal_after_stop": "right turn on red only on a green-arrow signal after stopping",
                            "prohibited": "no right turn on red"}[m])
        if any(r["category"] == "school_bus" for r in self.rules_for(country)): parts.append("stop for school buses with flashing red lights")
        return "; ".join(parts)

    def country_nodes(self) -> list[str]:
        """Definition strings of the country nodes, in COUNTRIES order: 'driving in <country>: <one-line summary>'."""
        return [f"driving in {c}: {self.country_summary(c)}" for c in self.countries]

    def embed_rules(self, embed_fn, key: str | None = None) -> np.ndarray:
        """[n_rules, d] frozen-text embeddings of the rule texts, cached per (RuleBook version, embedder key)."""
        k = ("rules", self.version, key or getattr(embed_fn, "__qualname__", repr(embed_fn)))
        if k not in self._emb_cache:
            self._emb_cache[k] = np.stack([np.asarray(embed_fn(t), np.float32).reshape(-1) for t in self.texts()])
        return self._emb_cache[k]

    def embed_countries(self, embed_fn, key: str | None = None) -> np.ndarray:
        """[n_countries, d] frozen-text embeddings of the country node definitions (cached like embed_rules)."""
        k = ("countries", self.version, key or getattr(embed_fn, "__qualname__", repr(embed_fn)))
        if k not in self._emb_cache:
            self._emb_cache[k] = np.stack([np.asarray(embed_fn(t), np.float32).reshape(-1) for t in self.country_nodes()])
        return self._emb_cache[k]


# ==================================================================== torch modules
class RuleMemory(nn.Module):
    """Retrieval: w = p(c) @ M (membership weight), s = <W_q q, W_r r>/sqrt(d_rel), score = w * softmax(s); top-k by score.
    Returns dict(tokens [B,k,d_rule], idx [B,k], score [B,k], valid [B,k]). `valid` is False for slots whose rule has zero membership
    weight (a country with fewer than k rules), so GatedMemory never attends to another country's rule."""
    def __init__(self, d: int, d_rule: int, membership: np.ndarray | torch.Tensor, k: int = 8, d_rel: int = 128):
        super().__init__()
        self.k = k; self.d_rel = d_rel
        self.wq = nn.Linear(d, d_rel); self.wr = nn.Linear(d_rule, d_rel)
        self.register_buffer("M", torch.as_tensor(np.asarray(membership), dtype=torch.float32), persistent=False)   # [C,R]

    def forward(self, q: torch.Tensor, p_country: torch.Tensor, rule_emb: torch.Tensor) -> dict:
        w = p_country.to(self.M.dtype) @ self.M                                  # [B,R]
        s = self.wq(q) @ self.wr(rule_emb).t() / math.sqrt(self.d_rel)            # [B,R]
        score = w * torch.softmax(s, -1)
        k = min(self.k, rule_emb.shape[0])
        top, idx = score.topk(k, -1)
        valid = w.gather(1, idx) > 0
        return dict(tokens=rule_emb[idx], idx=idx, score=top, valid=valid, weight=w)


class GatedMemory(nn.Module):
    """q' = q + tanh(g) * CrossAttn(q, proj(rule tokens)) on the s1/s2 level queries only; g = 0 at init.
    rules=None returns q itself; g = 0 returns q bit-exactly (0 * finite = 0). Attention logits get + log(score) so the retrieval
    projections receive gradient; invalid slots are masked; rows with no valid slot get a zero update (never NaN)."""
    def __init__(self, d: int, d_rule: int, heads: int = 8, levels=None, targets=("s1", "s2lat", "s2lon")):
        super().__init__()
        from .model import LEVELS
        levels = tuple(levels or LEVELS)
        if "goal" in targets: raise ValueError("admissibility: the rule memory must not write the goal query")
        self.proj = nn.Linear(d_rule, d); self.att = nn.MultiheadAttention(d, heads, batch_first=True); self.heads = heads
        self.g = nn.Parameter(torch.zeros(len(levels)))
        self.register_buffer("level_mask", torch.tensor([1.0 if l in targets else 0.0 for l in levels]), persistent=False)

    def forward(self, q: torch.Tensor, rules: dict | None) -> torch.Tensor:   # q [B,L,d]
        if rules is None: return q
        kv = self.proj(rules["tokens"])                                          # [B,k,d]
        valid = rules["valid"]
        bias = torch.log(rules["score"].clamp(min=1e-12)).masked_fill(~valid, float("-inf"))   # [B,k]
        any_valid = valid.any(-1)                                                # [B]
        bias = torch.where(any_valid[:, None], bias, torch.zeros_like(bias))    # keep softmax finite on empty rows
        B, L = q.shape[:2]
        mask = bias[:, None, None, :].expand(B, self.heads, L, -1).reshape(B * self.heads, L, -1)
        r = self.att(q, kv, kv, attn_mask=mask, need_weights=False)[0]
        r = torch.where(any_valid[:, None, None], r, torch.zeros_like(r))
        return q + (torch.tanh(self.g) * self.level_mask).view(1, -1, 1) * r


class CountryReadout(nn.Module):
    """Contrastive country readout of the strategic query: logits = tau * <norm(H(q_s1)), norm(G(t_c))>, t_c the cached frozen-text
    embedding of 'driving in <country>: ...'. tau = exp(theta), theta = log(1/0.07) at init, clamped at log 100 (as HiCAP.tau)."""
    def __init__(self, d: int, d_text: int, d_out: int | None = None, logit_init: float = math.log(1 / 0.07), logit_max: float = math.log(100.0)):
        super().__init__()
        d_out = d_out or d
        self.h = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_out))
        self.tower = nn.Sequential(nn.Linear(d_text, 2 * d), nn.GELU(), nn.Linear(2 * d, d_out))
        self.theta = nn.Parameter(torch.tensor(float(logit_init))); self.logit_max = logit_max

    def tau(self): return self.theta.clamp(max=self.logit_max).exp()

    def forward(self, q_s1: torch.Tensor, country_text: torch.Tensor) -> torch.Tensor:
        return self.tau() * F.normalize(self.h(q_s1), dim=-1) @ F.normalize(self.tower(country_text), dim=-1).t()

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over the country nodes; y = -100 is IGNORE (rows without a country label)."""
        keep = y >= 0
        return F.cross_entropy(logits[keep], y[keep]) if keep.any() else logits.sum() * 0.0

    def posterior(self, q_s1, country_text, supplied: torch.Tensor | None = None) -> torch.Tensor:
        """PREDICTED arm: softmax of the readout. SUPPLIED arm (GNSS geofence, registered ablation): one-hot of `supplied` ids."""
        if supplied is not None: return F.one_hot(supplied.long(), country_text.shape[0]).float()
        return torch.softmax(self(q_s1, country_text), -1)


# ==================================================================== prune route
def _norm_p(p_country) -> np.ndarray:
    p = np.asarray(p_country, np.float64).reshape(-1)
    if p.shape[0] != len(COUNTRIES): raise ValueError(f"p_country must have {len(COUNTRIES)} entries (COUNTRIES order)")
    s = p.sum()
    return p / s if s > 0 else p


def speed_allowed(vocab: FactoredVocab, v0: float, v_limit_kmh: float, tol: float = 0.10, eps: float = 0.05) -> np.ndarray:
    """Profile bits that respect a speed cap: max over the horizon of v0+dv <= max(cap, v0) + eps, cap = limit*(1+tol) in m/s.
    When the ego is already above the cap, profiles that do not accelerate beyond v0 stay (a rule never demands an impossible brake)."""
    cap = v_limit_kmh * (1.0 + tol) / 3.6
    vmax = (v0 + vocab.profiles).max(1)
    return vmax <= max(cap, v0) + eps


def rtor_cell_allowed(mode: str | None, v0: float, green_arrow: bool = False) -> np.ndarray:
    """[N_LAT,N_LON] cells allowed at a RED signal under one country's right-turn-on-red rule (only TURN_R cells are ever affected:
    straight/left at red are country-invariant and outside the rule book). Before a full stop every mode keeps only TURN_R x
    {HOLD, STOP}; after the stop: US-mode allows the turn, arrow modes need the arrow, 'prohibited' never allows it."""
    ok = np.ones((N_LAT, N_LON), bool)
    if mode is None: return ok
    stopped = v0 < STANDSTILL_V0
    turn_ok = stopped and (mode == "permitted_after_stop_unless_signed" or (mode in ("green_arrow_sign_after_stop", "green_arrow_signal_after_stop") and green_arrow))
    if not turn_ok:
        ok[LAT_TURN_R] = False; ok[LAT_TURN_R, list(LON_HALT)] = True
    return ok


def rule_masks(book: RuleBook, vocab: FactoredVocab, p_country, v0: float, r1_profile: np.ndarray | None = None,
               road_type: str | None = None, signal: str | None = None, green_arrow: bool = False, tol: float = 0.10,
               vote: float = 0.5, rtor_signed_prohibited: bool = False) -> dict:
    """Numeric prune rules -> masks consistent with prune.py semantics.

    profile [K_V] bool: R1 AND (speed rule), the speed rule active ONLY when `road_type` is known (one of SPEED_ROADS) and the
                        country's record has a numeric default; a profile is removed when the posterior mass of countries whose rule
                        removes it is >= `vote`. A rule never resurrects an R1-removed profile (AND), and if the AND would leave no
                        profile the R1 mask is returned unchanged (fail-open, flagged) -- decode.py has the same empty-set fallback.
    cell [N_LAT,N_LON] bool: right-turn-on-red rule, active ONLY when `signal == 'red'` is known; same posterior vote.
    The guard set is NOT a vocabulary entry and is never masked here (see `candidate_keep`)."""
    p = _norm_p(p_country)
    r1 = np.ones(vocab.K_V, bool) if r1_profile is None else np.asarray(r1_profile, bool)
    info = dict(speed_active=False, rtor_active=False, fallback=False, removed_profiles=0, removed_cells=0)
    remove_mass = np.zeros(vocab.K_V)
    if road_type in SPEED_ROADS:
        for ci, c in enumerate(COUNTRIES):
            if p[ci] <= 0: continue
            r = book.speed_rule(c, road_type)
            if r is None or r["effect"] != "prune" or r["args"]["v_default_kmh"] is None: continue
            info["speed_active"] = True
            remove_mass += p[ci] * ~speed_allowed(vocab, v0, float(r["args"]["v_default_kmh"]), tol)
    rule_ok = remove_mass < vote - 1e-12
    prof = r1 & rule_ok
    if r1.any() and not prof.any(): prof = r1.copy(); info["fallback"] = True
    info["removed_profiles"] = int((r1 & ~prof).sum())
    cell_mass = np.zeros((N_LAT, N_LON))
    if signal == "red":
        info["rtor_active"] = True
        for ci, c in enumerate(COUNTRIES):
            if p[ci] <= 0: continue
            m = "prohibited" if rtor_signed_prohibited else book.rtor_mode(c)
            cell_mass += p[ci] * ~rtor_cell_allowed(m, v0, green_arrow)
    cell = cell_mass < vote - 1e-12
    info["removed_cells"] = int((~cell).sum())
    return dict(profile=prof, cell=cell, info=info)


def candidate_keep(masks: dict, prof_idx: np.ndarray, lat_cls: np.ndarray, lon_cls: np.ndarray, is_guard: np.ndarray) -> np.ndarray:
    """Composed-candidate keep mask: an open candidate survives iff its profile bit and its (lat, lon) cell bit survive; guard-set
    candidates ALWAYS survive (prune.guard_set is injected after every mask)."""
    prof_idx = np.asarray(prof_idx); is_guard = np.asarray(is_guard, bool)
    ok = np.ones(len(prof_idx), bool)
    o = ~is_guard
    ok[o] = masks["profile"][prof_idx[o]] & masks["cell"][np.asarray(lat_cls)[o], np.asarray(lon_cls)[o]]
    return ok | is_guard
