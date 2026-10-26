"""Reasoning text as an OPTIONAL, ONE-WAY system output (v10 spec `Paper/HiCAP/design/HICAP_V10_SPEC.md` §3.3, stream T).

The command is produced by HiCAP's scorer (`decode.decide`); this module only EXPLAINS it. Nothing here feeds the control computation:
  * `DecisionView.from_decide` copies the executed decision into an immutable snapshot of Python primitives (no tensors, no reference
    to the decide() dict), so the channel cannot mutate or alias control state;
  * `ReasoningChannel` returns only a `SubmitReceipt` (admitted / reason / id -- nothing derived from text) at submit time, and
    `Explanation(text, audit)` records through `drain()`, which is a LOGGING / HMI sink, never read by the controller;
  * the decode reuses the KV cache of the latest refresh prefill (injected `generate_fn(prefix_handle, prompt, max_new_tokens)`) --
    no second vision pass -- and runs at <= 0.5 Hz inside the refresh-duty budget D = f_r (T_r + o) <= 0.5 of `sec_eff.tex`.

Duty arithmetic (ESTIMATED, weight-streaming floor, the method of `sec_eff.tex` Table `tab:backbones`): one decoded token streams every
LM weight once. Cosmos3-Edge 2.43 B (PUBLISHED card; the vision tower is inside that count, so this is an UPPER bound for the LM) at
0.7 x 273 GB/s: bf16 25.4 ms/token -> 48 tokens 1.22 s; int8 0.61 s; NVFP4 (4.5 bit) 0.34 s (`decode_cost_estimate`). At 0.5 Hz the
bf16 explanation alone is duty 0.61 > 0.5, so the admission control below is NOT cosmetic: with a 2 Hz refresh of ~65 ms (D_r ~ 0.13)
the leftover 0.37 admits bf16 explanations at <= ~0.30 Hz and NVFP4 at the full 0.5 Hz.

`ConsistencyMonitor` embeds the explanation with the same frozen text path as the node definitions (injected `embed_fn`) and maps it
to the nearest LAT9 / LON9 tactical node; disagreement with the executed node is LOGGED. Raising the System-2 verifier on disagreement
is a registered ABLATION only (`MonitorCfg.verifier_trigger_ablation`, default False) and can act on the NEXT tick at the earliest.

`parse_coc` reads both supervision sources: corpus `alpamayo/records.parquet` (nvidia/Alpamayo2-Super; tasks trajectory / meta_action
/ auto_labeling) and upstream `reasoning/ood_reasoning.parquet` (`events` JSON). Teacher text is INHERITED quality, not ground truth.

`TextChannelLoRAConfig` documents the optional explanation adapter (NOT trained here); `adapter_disabled(model)` is the guard that keeps
cached taps byte-identical: the adapter must be OFF during tap extraction (`backbone.FrozenTapBackbone.extract`).
"""
from __future__ import annotations
import json
import math
import queue
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .knowledge import render_candidate, define
from .vocab import LAT_V9, LON_V9, DT

MAX_NEW_TOKENS = 48            # spec §3.3
F_MAX_HZ = 0.5                 # spec §3.3
D_MAX = 0.5                    # sec_eff.tex: refresh duty D = f_r (T_r + o) <= 0.5
PROMPT_TEMPLATE = ("Scene (cached). Decision: {decision}. Applicable rules: {rules}. "
                   "In one sentence, explain why.")


# ===================================================================== decision snapshot + prompt
@dataclass(frozen=True)
class DecisionView:
    """Immutable, primitive-only copy of an executed decision. Built from `decide()` output; holds no tensor and no reference to it."""
    lat: str
    lon: str
    v_target: float | None = None
    cell: tuple = ()
    is_guard: bool = False
    confidence: float | None = None

    @staticmethod
    def from_decide(out: dict, lat_names: Sequence[str] = LAT_V9, lon_names: Sequence[str] = LON_V9, dt: float = DT) -> "DecisionView":
        """`out` = decode.decide(...) dict. v_target = speed over the last trajectory step (the profile's end speed). Read-only access."""
        ci, cj = (int(out["cell"][0]), int(out["cell"][1]))
        tr = np.asarray(out["traj"], np.float64)
        v = float(np.linalg.norm(tr[-1] - tr[-2]) / dt) if tr.ndim == 2 and len(tr) >= 2 else None
        conf = out.get("confidence")
        return DecisionView(lat=str(lat_names[ci]), lon=str(lon_names[cj]), v_target=v, cell=(ci, cj),
                            is_guard=bool(out.get("is_guard", False)), confidence=None if conf is None else float(conf))


def _rule_text(r) -> str:
    if isinstance(r, str): return r
    if isinstance(r, dict): return str(r.get("text", ""))
    return str(getattr(r, "text", r))


@dataclass(frozen=True)
class ExplanationPrompt:
    """Decision + retrieved rule texts -> the §3.3 prompt. `max_rules` caps the prompt's prefill length (template shows r1; r2)."""
    template: str = PROMPT_TEMPLATE
    max_rules: int = 2
    max_rule_chars: int = 160
    verbalise: bool = False            # append the node definitions in parentheses (off = the spec's exact template)

    def render(self, d: DecisionView, rules: Iterable = ()) -> str:
        dec = render_candidate(d.lat, d.lon, d.v_target)
        if self.verbalise: dec += f" ({define(d.lat)}; {define(d.lon)})"
        rs = [re.sub(r"\s+", " ", _rule_text(r)).strip().rstrip(".;")[: self.max_rule_chars] for r in rules]
        rs = [r for r in rs if r][: self.max_rules]
        return self.template.format(decision=dec, rules="; ".join(rs) if rs else "none")

    def n_rules_used(self, rules: Iterable) -> int:
        return min(self.max_rules, sum(1 for r in rules if _rule_text(r).strip()))


# ===================================================================== duty / cost model
def decode_cost_estimate(params_b: float = 2.43, bits: float = 16.0, n_tokens: int = MAX_NEW_TOKENS,
                         bw_gbs: float = 273.0, eff: float = 0.7) -> float:
    """ESTIMATED seconds to decode `n_tokens` (weight-streaming floor: every weight read once per token). Defaults: Cosmos3-Edge on Thor."""
    return n_tokens * params_b * 1e9 * bits / 8 / (eff * bw_gbs * 1e9)


@dataclass
class ChannelCfg:
    f_max_hz: float = F_MAX_HZ
    d_max: float = D_MAX               # total duty budget shared with the refresh (sec_eff)
    window_s: float = 10.0             # rolling window over which duty is accounted
    max_new_tokens: int = MAX_NEW_TOKENS
    est_call_s: float = 0.34           # prior for one call (NVFP4 estimate); replaced by an EMA of measured calls
    ema: float = 0.3
    refresh_duty_prior: float = 0.0    # used when the refresh loop does not report busy intervals (note_refresh)
    max_text_chars: int = 400
    mode: str = "inline"               # "inline" (synchronous, deterministic: tests / offline replay) | "thread" (one background worker)


@dataclass(frozen=True)
class SubmitReceipt:
    admitted: bool
    reason: str                        # ok | rate | duty | busy | disabled
    request_id: int


@dataclass(frozen=True)
class AuditRecord:
    request_id: int
    t_submit: float
    t_start: float
    t_end: float
    duration_s: float
    predicted_s: float
    decision: str                      # render_candidate(...) of the executed decision
    n_rules: int
    prompt: str
    refresh_id: Any
    token_age_s: float | None
    duty_at_admit: float
    error: str | None = None


@dataclass(frozen=True)
class Explanation:
    """The ONLY output of the channel: text + audit. Never consumed by the control path."""
    text: str
    audit: AuditRecord


class ReasoningChannel:
    """Asynchronous, rate-limited (<= f_max) explanation decoder with duty accounting against the refresh budget.

    generate_fn(prefix_handle, prompt, max_new_tokens) -> str  (reuses the refresh prefill KV cache; see `make_hf_generate_fn`).
    The refresh loop reports its busy intervals through `note_refresh(t_start, duration_s)`; otherwise `refresh_duty_prior` is used.
    Admission at time t: (1) t - t_last_start >= 1/f_max; (2) worker idle; (3) [refresh busy + explanation busy in (t-W, t]
    + predicted cost of this call] / W <= d_max. Rejections are counted per reason, never queued (latest-wins: a stale explanation
    is worthless)."""

    def __init__(self, generate_fn: Callable, cfg: ChannelCfg = ChannelCfg(), prompt: ExplanationPrompt = ExplanationPrompt(),
                 clock: Callable[[], float] = time.monotonic, enabled: bool = True):
        self.gen, self.cfg, self.prompt, self.clock, self.enabled = generate_fn, cfg, prompt, clock, enabled
        self._refresh: deque = deque()          # (start, end)
        self._expl: deque = deque()             # (start, end)
        self._pred = float(cfg.est_call_s)
        self._t_last: float | None = None
        self._busy = False
        self._next_id = 0
        self._out: deque = deque()
        self._lock = threading.Lock()
        self.counts = dict(ok=0, rate=0, duty=0, busy=0, disabled=0, error=0)
        self._q: queue.Queue | None = None
        if cfg.mode == "thread":
            self._q = queue.Queue(maxsize=1)
            threading.Thread(target=self._worker, daemon=True, name="hicap-reasoning").start()

    # ---- duty accounting
    def note_refresh(self, t_start: float, duration_s: float) -> None:
        with self._lock: self._refresh.append((float(t_start), float(t_start) + float(duration_s)))

    @staticmethod
    def _busy_in(iv: deque, lo: float, hi: float) -> float:
        return sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in iv)

    def duty(self, t: float | None = None) -> dict:
        """Measured duty in the window (t-W, t]: refresh, explanation, total (fractions of wall time)."""
        t = self.clock() if t is None else t; W = self.cfg.window_s; lo = t - W
        with self._lock:
            while self._refresh and self._refresh[0][1] < lo: self._refresh.popleft()
            while self._expl and self._expl[0][1] < lo: self._expl.popleft()
            r = self._busy_in(self._refresh, lo, t) / W if self._refresh else self.cfg.refresh_duty_prior
            e = self._busy_in(self._expl, lo, t) / W
        return dict(refresh=r, explanation=e, total=r + e)

    # ---- submission (control side calls this with a snapshot; gets back only a receipt)
    def submit(self, decision: DecisionView, rules: Iterable = (), prefix_handle: Any = None, refresh_id: Any = None,
               t_capture: float | None = None) -> SubmitReceipt:
        if not isinstance(decision, DecisionView):
            raise TypeError("submit() takes a DecisionView snapshot (DecisionView.from_decide), never the decide() dict")
        t = self.clock(); rid = self._next_id; self._next_id += 1
        if not self.enabled: return self._reject("disabled", rid)
        if self._t_last is not None and t - self._t_last < 1.0 / self.cfg.f_max_hz - 1e-9: return self._reject("rate", rid)
        if self._busy: return self._reject("busy", rid)
        d = self.duty(t)
        if d["total"] + self._pred / self.cfg.window_s > self.cfg.d_max + 1e-12: return self._reject("duty", rid)
        rules = list(rules)
        job = dict(rid=rid, t_submit=t, decision=render_candidate(decision.lat, decision.lon, decision.v_target),
                   prompt=self.prompt.render(decision, rules), n_rules=self.prompt.n_rules_used(rules), handle=prefix_handle,
                   refresh_id=refresh_id, age=None if t_capture is None else t - float(t_capture), duty=d["total"], pred=self._pred)
        self._busy = True; self._t_last = t; self.counts["ok"] += 1
        if self._q is not None: self._q.put(job)
        else: self._run(job)
        return SubmitReceipt(True, "ok", rid)

    def _reject(self, why: str, rid: int) -> SubmitReceipt:
        self.counts[why] += 1
        return SubmitReceipt(False, why, rid)

    def _run(self, job: dict) -> None:
        t0 = self.clock(); err = None
        try:
            txt = self.gen(job["handle"], job["prompt"], self.cfg.max_new_tokens)
            if not isinstance(txt, str): raise TypeError(f"generate_fn returned {type(txt).__name__}, expected str")
            txt = re.sub(r"<\|[^|]*\|>", "", txt).strip()[: self.cfg.max_text_chars]
        except Exception as e:                                   # noqa: BLE001 -- the channel must never take the controller down
            txt, err = "", f"{type(e).__name__}: {e}"; self.counts["error"] += 1
        t1 = self.clock(); dur = max(0.0, t1 - t0)
        with self._lock:
            self._expl.append((t0, t1))
            self._pred = (1 - self.cfg.ema) * self._pred + self.cfg.ema * dur
            self._out.append(Explanation(txt, AuditRecord(job["rid"], job["t_submit"], t0, t1, dur, job["pred"], job["decision"],
                                                          job["n_rules"], job["prompt"], job["refresh_id"], job["age"], job["duty"], err)))
        self._busy = False

    def _worker(self) -> None:
        while True:
            self._run(self._q.get())

    # ---- sink side (logging / HMI), never the controller
    def drain(self) -> list[Explanation]:
        with self._lock:
            out = list(self._out); self._out.clear()
        return out

    def wait_idle(self, timeout: float = 30.0) -> bool:
        t_end = time.monotonic() + timeout
        while self._busy and time.monotonic() < t_end: time.sleep(0.005)
        return not self._busy

    def stats(self) -> dict:
        return dict(counts=dict(self.counts), predicted_call_s=self._pred, duty=self.duty())


# ===================================================================== consistency monitor
@dataclass
class MonitorCfg:
    window: int = 200                  # rolling agreement window (explanations)
    margin: float = 0.0                # an axis disagrees only if sim(nearest) - sim(executed) > margin
    silent_margin: float = 0.0         # axis counted SILENT (no claim) if max_sim - mean_sim < silent_margin (0 = never silent)
    verifier_trigger_ablation: bool = False   # registered ablation only; default system never lets text touch control


@dataclass(frozen=True)
class MonitorRecord:
    lat_exec: str
    lon_exec: str
    lat_text: str
    lon_text: str
    lat_sims: tuple
    lon_sims: tuple
    lat_agree: bool | None             # None = silent on this axis
    lon_agree: bool | None
    agree: bool                        # no non-silent axis disagrees


class ConsistencyMonitor:
    """Embeds the explanation (injected `embed_fn`, the same frozen text path as the node definitions) and maps it to the nearest LAT9 /
    LON9 node (definitions from `knowledge.define`). Keeps rolling agreement rates; logs, never acts (except the registered ablation)."""

    def __init__(self, embed_fn: Callable[[str], Any], cfg: MonitorCfg = MonitorCfg(),
                 lat_tokens: Sequence[str] = LAT_V9, lon_tokens: Sequence[str] = LON_V9, definitions: dict | None = None):
        self.embed, self.cfg = embed_fn, cfg
        self.lat_tokens, self.lon_tokens = list(lat_tokens), list(lon_tokens)
        dfn = (lambda t: definitions[t]) if definitions else define
        self.A_lat = self._tab([dfn(t) for t in self.lat_tokens]); self.A_lon = self._tab([dfn(t) for t in self.lon_tokens])
        self.hist: deque = deque(maxlen=cfg.window)

    def _vec(self, s: str) -> np.ndarray:
        v = np.asarray(self.embed(s), np.float64).reshape(-1); n = np.linalg.norm(v)
        return v / n if n else v

    def _tab(self, texts): return np.stack([self._vec(t) for t in texts])

    def _axis(self, e, A, tokens, executed):
        sims = A @ e; j = int(np.argmax(sims))
        if self.cfg.silent_margin > 0 and float(sims.max() - sims.mean()) < self.cfg.silent_margin: return tokens[j], sims, None
        if executed not in tokens: return tokens[j], sims, None
        ok = float(sims[j] - sims[tokens.index(executed)]) <= self.cfg.margin
        return tokens[j], sims, ok

    def check(self, text: str, lat_exec: str, lon_exec: str) -> MonitorRecord:
        e = self._vec(text)
        lt, ls, la = self._axis(e, self.A_lat, self.lat_tokens, lat_exec)
        ot, os_, oa = self._axis(e, self.A_lon, self.lon_tokens, lon_exec)
        rec = MonitorRecord(lat_exec, lon_exec, lt, ot, tuple(float(x) for x in ls), tuple(float(x) for x in os_), la, oa,
                            agree=(la is not False) and (oa is not False))
        self.hist.append(rec)
        return rec

    def check_explanation(self, ex: Explanation, d: DecisionView) -> MonitorRecord:
        return self.check(ex.text, d.lat, d.lon)

    def rate(self) -> dict:
        """Rolling agreement rates with their n (silent axes excluded from that axis's n)."""
        def r(vals):
            v = [x for x in vals if x is not None]
            return dict(rate=(sum(v) / len(v)) if v else float("nan"), n=len(v))
        return dict(lat=r([h.lat_agree for h in self.hist]), lon=r([h.lon_agree for h in self.hist]),
                    joint=r([h.agree for h in self.hist]))

    def verifier_trigger(self, rec: MonitorRecord) -> bool:
        """Registered ablation ONLY: True raises the System-2 verifier on the NEXT tick. Default config always returns False."""
        return bool(self.cfg.verifier_trigger_ablation and not rec.agree)


# ===================================================================== CoC parsing (both sources)
_SPECIAL = re.compile(r"<\|[^|]*\|>")


@dataclass(frozen=True)
class CocItem:
    kind: str                          # cot | coc | meta_action | ood_event
    text: str
    meta: dict = field(default_factory=dict)


def _clean(s) -> str:
    return _SPECIAL.sub("", str(s)).strip() if s is not None else ""


def parse_meta_action(text: str) -> dict:
    """'Longitudinal: Gentle Deceleration.\\nLateral: Steer Right.\\nLane: Lane Keep.' -> {'Longitudinal': 'Gentle Deceleration', ...}."""
    out = {}
    for line in _clean(text).splitlines():
        if ":" in line:
            k, v = line.split(":", 1); out[k.strip()] = v.strip().rstrip(".").strip()
    return out


def _first(x):
    if isinstance(x, list): return x[0] if x else None
    return x


def parse_coc(raw_json, task: str | None = None) -> list[CocItem]:
    """Parse one reasoning record into CocItems. Accepts
      * records.parquet `raw_json` (Alpamayo2-Super): task 'meta_action' -> cot + meta_action (meta: parsed fields);
        'auto_labeling' -> coc (chain_of_causation); 'trajectory' -> cot. Other tasks (vqa, grounding_via_vqa) -> [].
        `task` may be omitted; the format is then detected from the keys.
      * ood_reasoning.parquet `events` (JSON list of {event_start_frame, event_start_timestamp, coc}) -> ood_event items.
    Empty / null texts are dropped (never emitted as empty strings)."""
    if not isinstance(raw_json, (str, bytes, list, dict)): return []          # None / NaN / pd.NA (9 OOD rows have null events)
    d = json.loads(raw_json) if isinstance(raw_json, (str, bytes)) else raw_json
    items: list[CocItem] = []
    if isinstance(d, list):                                                       # OOD events
        for ev in d:
            t = _clean(ev.get("coc"))
            if t: items.append(CocItem("ood_event", t, dict(event_start_frame=ev.get("event_start_frame"),
                                                           event_start_timestamp_us=ev.get("event_start_timestamp"))))
        return items
    if task is None:
        if any(k in d for k in ("figure_style", "pred_xyz_shape", "pred_xyz", "ade_vs_gt_m")): task = "trajectory"
        elif _first(d.get("meta_action")): task = "meta_action"
        elif d.get("cot_auto_labeling_json") or _first(d.get("cot_auto_labeling")): task = "auto_labeling"
        else: return []
    if task == "trajectory":
        t = _clean(_first(d.get("cot")) or _first(d.get("cots")))
        return [CocItem("cot", t)] if t else []
    if task == "meta_action":
        t = _clean(_first(d.get("cot")))
        if t: items.append(CocItem("cot", t))
        ma = _clean(_first(d.get("meta_action")))
        if ma: items.append(CocItem("meta_action", re.sub(r"\s*\n\s*", " ", ma), parse_meta_action(ma)))
        return items
    if task == "auto_labeling":
        js = _first(d.get("cot_auto_labeling_json"))
        if not isinstance(js, dict):
            try: js = json.loads(_clean(_first(d.get("cot_auto_labeling")) or _first(d.get("cot")) or "{}"))
            except (json.JSONDecodeError, TypeError): js = {}
        t = _clean(js.get("chain_of_causation")) if isinstance(js, dict) else ""
        return [CocItem("coc", t)] if t else []
    return []


# ===================================================================== teacher-text -> v9 class keyword table (explicit, documented)
# Alpamayo2-Super meta_action vocabulary (MEASURED on records.parquet, 4,729 rows): Longitudinal {Gentle/Strong Acceleration,
# Gentle/Strong Deceleration, Maintain Speed, Stop, Reverse}; Lateral {Go Straight, (Sharp) Steer Left/Right, Reverse Left/Right};
# Lane {Lane Keep, Left/Right Lane Change, Slightly Shift Left/Right, Turn Left/Right}. Each entry: (primary v9 class, compatible set).
# "compatible" = the v9 classes a correct teacher statement can co-occur with given v9's own definitions (SPEC §3.2/§3.5):
#   Stop: STOP if moving, HOLD if already stationary; Deceleration: DECELERATE, or STOP when braking ends at rest, or FOLLOW (behind a
#   slowing lead); Maintain Speed: KEEP, or FOLLOW (constant-gap following); Acceleration: ACCELERATE, or FOLLOW / CREEP (pulling away).
META_LON_TO_LON9 = {
    "Stop": ("STOP", {"STOP", "HOLD"}),
    "Gentle Deceleration": ("DECELERATE", {"DECELERATE", "STOP", "FOLLOW"}),
    "Strong Deceleration": ("DECELERATE", {"DECELERATE", "STOP", "FOLLOW"}),
    "Maintain Speed": ("KEEP", {"KEEP", "FOLLOW"}),
    "Gentle Acceleration": ("ACCELERATE", {"ACCELERATE", "FOLLOW", "CREEP"}),
    "Strong Acceleration": ("ACCELERATE", {"ACCELERATE", "FOLLOW", "CREEP"}),
    "Reverse": (None, set()),                                   # v9 masks reversing rows (IGNORE); excluded with reason
}
# Lane line is the manoeuvre statement (primary for LAT9); v9 lane-change labels are OFF in this release (lc_measurable = 0), so a
# teacher lane change can only be compared to LANE_KEEP -- counted as its own row of the confusion, never as agreement.
META_LANE_TO_LAT9 = {
    "Lane Keep": ("LANE_KEEP", {"LANE_KEEP"}),
    "Slightly Shift Left": ("LANE_KEEP", {"LANE_KEEP"}),
    "Slightly Shift Right": ("LANE_KEEP", {"LANE_KEEP"}),
    "Turn Left": ("TURN_L", {"TURN_L"}),
    "Turn Right": ("TURN_R", {"TURN_R"}),
    "Left Lane Change": ("LANE_CHANGE_L", {"LANE_CHANGE_L"}),
    "Right Lane Change": ("LANE_CHANGE_R", {"LANE_CHANGE_R"}),
}
# Lateral (steering) line: fallback only where the Lane line is absent, and reported as its own diagnostic confusion.
META_LATERAL_TO_LAT9 = {
    "Go Straight": ("LANE_KEEP", {"LANE_KEEP"}),
    "Steer Left": ("LANE_KEEP", {"LANE_KEEP", "TURN_L"}),
    "Steer Right": ("LANE_KEEP", {"LANE_KEEP", "TURN_R"}),
    "Sharp Steer Left": ("TURN_L", {"TURN_L"}),
    "Sharp Steer Right": ("TURN_R", {"TURN_R"}),
    "Reverse Left": (None, set()),
    "Reverse Right": (None, set()),
}


def meta_to_v9(fields: dict) -> dict:
    """Parsed meta_action fields -> dict(lon=(primary, compatible), lat=(primary, compatible, source_line)). Unknown strings -> None."""
    lon = META_LON_TO_LON9.get(fields.get("Longitudinal", ""), (None, set()))
    if fields.get("Lane") in META_LANE_TO_LAT9: lat = META_LANE_TO_LAT9[fields["Lane"]] + ("Lane",)
    elif fields.get("Lateral") in META_LATERAL_TO_LAT9: lat = META_LATERAL_TO_LAT9[fields["Lateral"]] + ("Lateral",)
    else: lat = (None, set(), None)
    return dict(lon=lon, lat=lat)


# ===================================================================== optional text-channel LoRA (documented, NOT trained here)
@dataclass(frozen=True)
class TextChannelLoRAConfig:
    """Optional explanation adapter on the frozen reasoner's TEXT channel only (spec §3.3). NOT trained in this repo yet.

    * Targets: attention + MLP projections of the LANGUAGE MODEL blocks only. Never the vision tower, never the vision-language merger /
      projector, never the embedding / LM-head matrices (tied heads would move the node-definition text embeddings).
    * Data: Alpamayo2-Super chain_of_causation (4,729 clips x 1 anchor t0 = 5.1 s) + NVIDIA OOD CoC (events of the 50 clips that overlap
      the corpus), from `DataEng/v10/reasoning/v10_coc.parquet`, conditioned on the EXECUTED decision rendered by `ExplanationPrompt`
      (never on the teacher's own meta-action: the text must explain HiCAP's decision, not the teacher's).
    * Metrics: text->node agreement (`ConsistencyMonitor`), decision consistency, a human audit sample; no BLEU-only claim.
    * INVARIANT: the adapter is DISABLED during tap extraction (`adapter_disabled`), so cached taps, node / rule / country text
      embeddings and verifier scores stay byte-identical to the frozen model's; it is enabled only inside the explanation decode.
    """
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    layers_scope: str = "language_model"          # module-name filter; anything under the vision tower / merger is excluded
    exclude_patterns: tuple = ("visual", "vision", "merger", "mm_projector", "embed_tokens", "lm_head")
    lr: float = 1e-4
    epochs: int = 2
    max_new_tokens: int = MAX_NEW_TOKENS
    condition_on: str = "executed_decision"
    disabled_during_tap_extraction: bool = True

    def module_ok(self, name: str) -> bool:
        """Whether a (fully qualified) module name may carry an adapter under this config."""
        if any(p in name for p in self.exclude_patterns): return False
        return self.layers_scope in name and name.split(".")[-1] in self.target_modules


@contextmanager
def adapter_disabled(model):
    """Disable any PEFT adapter for the duration (tap extraction, text-embedding of nodes / rules, verifier scoring).
    Supports peft.PeftModel (`disable_adapter()` context) and the transformers PEFT integration (`disable_adapters` / `enable_adapters`).
    A model without adapters passes through unchanged."""
    if hasattr(model, "disable_adapter") and callable(model.disable_adapter):
        with model.disable_adapter():
            yield model
        return
    has = getattr(model, "_hf_peft_config_loaded", False) and hasattr(model, "disable_adapters")
    if has: model.disable_adapters()
    try:
        yield model
    finally:
        if has: model.enable_adapters()


# ===================================================================== reference HF generate_fn (GPU only; NOT RUN)
@dataclass
class PrefixHandle:
    """What a refresh prefill must hand to the channel (backbone hook, see the stream-T report): the KV cache after the prefill, the
    prefill input ids, and the multimodal position inputs needed to continue the m-RoPE positions without re-encoding the image."""
    past_key_values: Any
    input_ids: Any                                  # [1, L] prefill ids (incl. image tokens)
    mm_kwargs: dict = field(default_factory=dict)   # e.g. image_grid_thw / rope_deltas (Qwen-VL family)
    refresh_id: Any = None
    t_capture: float | None = None


def make_hf_generate_fn(model, tokenizer, use_adapter: bool = False):
    """Reference generate_fn for a transformers VLM. STATUS: written against the transformers generate() API; NOT RUN (no GPU here).
    UNVERIFIED for Cosmos3-Edge: (1) continuing generation from a cached multimodal prefix without pixel_values requires the model to
    accept `image_grid_thw` / cached `rope_deltas` so the suffix gets the right m-RoPE positions; (2) the cache is CROPPED back to the
    prefix length after each call so the refresh cache is reusable (DynamicCache.crop), else it is deep-copied."""
    def gen(h: PrefixHandle, prompt: str, max_new_tokens: int) -> str:
        import copy
        import torch
        suffix = tokenizer("\n" + prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(h.input_ids.device)
        ids = torch.cat([h.input_ids, suffix], -1); L = h.input_ids.shape[-1]
        cache = h.past_key_values
        can_crop = hasattr(cache, "crop")
        pkv = cache if can_crop else copy.deepcopy(cache)
        ctx = _nullctx() if use_adapter else adapter_disabled(model)          # the explanation LoRA is used ONLY here, if trained
        try:
            with torch.no_grad(), ctx:
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=pkv, use_cache=True,
                                     max_new_tokens=max_new_tokens, do_sample=False, **h.mm_kwargs)
        finally:
            if can_crop: cache.crop(L)
        return tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
    return gen


@contextmanager
def _nullctx():
    yield
