"""Knowledge injection and the System-2 verifier (paper sec:vocab 'Node embeddings', sec:prune 'System-2 verifier').

Knowledge enters in three places, none of which touches the frozen backbone's weights:
 (1) TEXT-DEFINED NODES: every Level-1/2 node embedding is a_v = norm(G([t_v ; phi(args_v)])), t_v the frozen language tower's last-token
     state on the token's DEFINITION string (cached once). A new versioned token gets a meaningful initial embedding from its text alone.
 (2) PRIOR-ENCODED RULES (prune.py): physics, route consistency, hysteresis, conformal keep-sets.
 (3) LANGUAGE VERIFICATION: in uncertain ticks the top candidates are rendered as text and the CLM match score is added.
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

DEFINITIONS = {
    # lateral (v9 LAT9, usable)
    "LANE_KEEP": "keep the current lane and follow the road without turning",
    "TURN_L": "turn left at a junction",
    "TURN_R": "turn right at a junction",
    # longitudinal (v9 LON9)
    "HOLD": "stay stationary and hold position",
    "CREEP": "creep forward slowly at very low speed",
    "STOP": "brake to a stop",
    "FOLLOW": "follow the lead vehicle keeping a safe time gap",
    "DECELERATE": "slow down to a lower target speed",
    "ACCELERATE": "speed up to a higher target speed",
    "KEEP": "keep the current speed",
}
ABBR = {"L": "left", "R": "right"}


def define(token: str) -> str:
    """Definition string for any token (v9 classes from DEFINITIONS; strategic tokens such as PREPARE_TURN_R_FOLLOW_ROUTE are verbalised)."""
    if token in DEFINITIONS: return DEFINITIONS[token]
    w = [ABBR.get(x, x.lower()) for x in token.split("_")]
    return " ".join(w)


@dataclass
class NodeLibrary:
    """Versioned table of text-defined nodes for one level. `text` [n,d_text] are CACHED frozen-tower embeddings."""
    level: str
    tokens: list[str]
    text: np.ndarray
    args: np.ndarray
    version: int = 1
    log: list = field(default_factory=list)

    def tables(self): return self.text.astype(np.float32), self.args.astype(np.float32)

    def add_token(self, token: str, embed_fn, args=None, definition: str | None = None) -> "NodeLibrary":
        """Extension only through a NEW version (the frozen-vocabulary rule): returns a new library; ids of existing tokens never move."""
        if token in self.tokens: raise ValueError(f"{token} already in {self.level} v{self.version}; extension must be a new token")
        d = definition or define(token); t = np.asarray(embed_fn(d), np.float32).reshape(1, -1)
        a = np.zeros((1, self.args.shape[1]), np.float32) if args is None else np.asarray(args, np.float32).reshape(1, -1)
        lib = NodeLibrary(self.level, self.tokens + [token], np.vstack([self.text, t]), np.vstack([self.args, a]), self.version + 1, self.log + [(token, d)])
        return lib

    def nearest(self, model_tower, token: str) -> list[tuple[str, float]]:
        """Audit of the initialisation: cosine of a token's INITIAL node embedding (through the existing tower, no training) to the others."""
        with torch.no_grad():
            t, a = self.tables(); A = model_tower(torch.from_numpy(t), torch.from_numpy(a))
        i = self.tokens.index(token); c = (A @ A[i]).numpy()
        return sorted([(t, float(c[j])) for j, t in enumerate(self.tokens) if j != i], key=lambda x: -x[1])


def hashed_text_embedding(s: str, dim: int = 64) -> np.ndarray:
    """Deterministic bag-of-words embedding used ONLY in tests/smoke runs (similar strings -> similar vectors). Real runs use the frozen tower."""
    v = np.zeros(dim, np.float32)
    for w in re.findall(r"[a-z]+", s.lower()):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little")); v += rng.normal(size=dim).astype(np.float32)
    n = np.linalg.norm(v); return v / n if n else v


# ------------------------------------------------------------------ System-2 verifier
def render_candidate(lat: str, lon: str, v_target: float | None = None, dist_m: float | None = None) -> str:
    s = f"lat={lat}, lon={lon}"
    if v_target is not None: s += f", target {v_target:.0f} m/s"
    if dist_m is not None: s += f" within {dist_m:.0f} m"
    return s


@dataclass
class Verifier:
    """Fires only in uncertain ticks: top-two margin < m_min OR imagination heads disagree. Reads at most K2 survivors, adds lam * CLM match.
    `score_fn(texts) -> np.ndarray[len(texts)]` is injected: one extra weight pass of the VLM over the cached scene prefix + candidate text
    (backbone.py; ~58.6 ms for an 8 B body in bf16, estimated). Language never generates; it only scores."""
    m_min: float = 0.5
    k2: int = 8
    lam: float = 1.0
    fired: int = 0
    seen: int = 0

    def __call__(self, scores: torch.Tensor, texts: list[str], score_fn, disagree: bool = False):
        self.seen += 1
        top2 = scores.topk(min(2, len(scores))).values
        margin = float(top2[0] - top2[-1]) if len(scores) > 1 else 9.0
        if margin >= self.m_min and not disagree: return scores, False
        self.fired += 1
        idx = scores.topk(min(self.k2, len(scores))).indices
        add = torch.zeros_like(scores)
        m = torch.as_tensor(score_fn([texts[int(i)] for i in idx]), dtype=scores.dtype)
        add[idx] = self.lam * (m - m.mean())
        return scores + add, True

    @property
    def rate(self): return self.fired / max(self.seen, 1)
