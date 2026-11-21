"""PREREG_V8F_LISTEN -- the causal "who listens to what" grid for the three layers S, T, O of v8F-mini.

ZERO TRAINING. One input of ONE layer is replaced by a donor window's value; every other input of that layer, and every UPSTREAM
layer, reads the window's own (clean) values; the metric is read at that layer's own output.

HOW THE OVERRIDE REACHES THE LAYER (the point of this module)
  The model's own ``run_s`` / ``run_t`` / ``run_o`` are called VERBATIM. Each assembles its conditioning tokens from five producers:
  ``_ego`` (ego token), ``e_nav``, ``e_lim``, ``_scene`` (scene tokens + pad mask); ``run_t`` / ``run_o`` take the S checkpoints (``ck``)
  and ``run_o`` the T ``code`` as ARGUMENTS, and O's age token is built from ``b['a_t'], b['a_s']``. So:
    * ego / nav / limit / scene -> the producer is shadowed, for the duration of ONE ``run_*`` call, by a function returning the DONOR's
      token (computed by the SAME producer on the donor pass). ``_patched`` REFUSES the call unless every shadowed producer was called
      EXACTLY ONCE -- an override that never reached the layer cannot pass silently -- and always restores the model.
    * ck / code -> the argument is the donor's.      * age -> a shallow copy of the batch carries the donor-cycle ages.
  Nothing of the model's cond assembly is re-implemented, so "no override" is the model's own computation BIT FOR BIT (pinned by
  test_listen.py against ``forward_infer``).

  O's kinematic prior reads ``p10f`` directly (``self.prior(self._at(b['p10f'], i), ...)``), NOT through ``_ego``: the O x ego cell
  swaps ONLY the ego conditioning token and the prior keeps the window's own input (pinned).

THE DONOR (the eval's own: ``tanitad.models.hjepa.donor_indices``, a derangement over EPISODES)
  The donor item is built at the SAME nominal ages as the window (as ``v8f_mini_eval --tac shuffle`` does); the dataset clamps them with
  the donor's own room, so the donor's effective ages are its own. The donor pass is a clean ``forward_infer`` of the donor batch.
    scene   = the donor's ``_scene`` tokens + pad of its own layer tick           ego/nav/limit = the donor's token at its own tick
    ck      = the donor's own S output re-expressed in ITS frames (``ck_for_t`` for T, ``ck_for_o`` for O)
    code    = the donor's own T code (as ``--tac shuffle``)
    age     = the age token of the donor's CYCLE POSITION: the ages the donor window carries in the banked ``--ages cycle`` eval
              (index j = perm[i]: a_t = j % 3, a_s = a_t + (j // 3) % 5, clamped by the donor's room as the dataset does). With the donor
              item at the same ages this cell would be identically null; O's held checkpoints and code are unchanged.
  READ AS A PRE-DATA INTERPRETATION: the PREREG's words are "the age token of the donor's cycle position".

ONLY THE TIERS THE PREREG NAMES: T1-class open loop, DIAGNOSTIC (donor inputs are interventions, not driving).
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

# ---- code path: the exported tip (3fdb1b61); the dev box has a stale editable ``tanitad`` on G: that must never win
_HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("LISTEN_CODE", os.path.join(_HERE, "code"))
for _p in (os.path.join(CODE, "stack", "scripts"), os.path.join(CODE, "taniteval"), os.path.join(CODE, "stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import v8f_mini as M  # noqa: E402

_loaded_from = os.path.abspath(M.__file__)
if not _loaded_from.startswith(os.path.abspath(CODE)):
    raise ImportError(f"tanitad was imported from {_loaded_from}, not from the exported tip under {CODE} (a stale editable install?)")

DT = M.DT
LAYERS = ("S", "T", "O")
INPUTS = ("scene", "ego", "nav", "limit", "ck", "code", "age")
#: the inputs each layer's ``run_*`` reads (PREREG table). Everything else is "not an input" (hatched in the figure).
_READS = {"S": ("scene", "ego", "nav", "limit"),
          "T": ("scene", "ego", "nav", "limit", "ck"),
          "O": ("scene", "ego", "nav", "limit", "ck", "code", "age")}
#: position of each non-scene input in the layer's conditioning-token stack (``torch.stack([...], 1)`` in run_*)
TOKEN_POS = {"S": {"ego": 0, "nav": 1, "limit": 2},
             "T": {"ego": 0, "nav": 1, "limit": 2, "ck": 3},
             "O": {"ego": 0, "nav": 1, "limit": 2, "ck": 3, "code": 4, "age": 5}}
#: the model callable that PRODUCES an input's token (shadowed for one call). ck / code / age are not here: they are arguments / batch fields.
_PRODUCER = {"scene": "_scene", "ego": "_ego", "nav": "e_nav", "limit": "e_lim"}
_READER = {"S": "read_s", "T": "read_t", "O": "read_o"}


def reads(model, layer: str) -> tuple:
    """The inputs ``layer`` of THIS model reads (a navonly model builds no scene path into S and T)."""
    r = _READS[layer]
    if layer in ("S", "T") and model.cfg.navonly:
        r = tuple(x for x in r if x != "scene")
    return r


def _check_ov(model, layer: str, ov) -> tuple:
    ov = tuple(ov)
    bad = [x for x in ov if x not in reads(model, layer)]
    if bad:
        raise ValueError(f"{layer} does not read {bad}: it is 'not an input' there (reads {reads(model, layer)})")
    if len(set(ov)) != len(ov):
        raise ValueError(f"duplicate inputs in {ov}")
    return ov


# ------------------------------------------------------------------------------------------------ a clean pass + its held intermediates
@dataclass
class Pass:
    """A CLEAN ``forward_infer`` of one collated batch, with the intermediates the cells need (all recomputed by the model's own calls)."""
    b: dict
    out: dict
    adapt: torch.Tensor
    last_s: torch.Tensor
    last_t: torch.Tensor
    last_now: torch.Tensor
    ck_s: torch.Tensor          # S's output, in S's tick frame
    ck_for_t: torch.Tensor      # ... re-expressed in T's tick frame
    ck_for_o: torch.Tensor      # ... re-expressed in the NOW frame
    ck_ok: torch.Tensor
    code: torch.Tensor          # the code O reads (T's own, held)


@torch.no_grad()
def make_pass(model, b: dict) -> Pass:
    out = model.forward_infer(b)
    B, Fn = b["feats"].shape[0], b["feats"].shape[1]
    dev = b["feats"].device
    last_now = torch.full((B,), Fn - 1, dtype=torch.long, device=dev)
    ck_s = out["ckpt"]
    return Pass(b=b, out=out, adapt=out["adapt"], last_s=last_now - b["a_s"], last_t=last_now - b["a_t"], last_now=last_now,
                ck_s=ck_s, ck_for_t=M.transform_ckpt(ck_s, b["pose_s"], b["pose_t"]),
                ck_for_o=M.transform_ckpt(ck_s, b["pose_s"], b["pose_now"]), ck_ok=torch.ones(B, 3, device=dev), code=out["code"])


def _tick(P: Pass, layer: str) -> torch.Tensor:
    return {"S": P.last_s, "T": P.last_t, "O": P.last_now}[layer]


# ------------------------------------------------------------------------------------------------ the donor's value of one input
def donor_token(model, Dn: Pass, layer: str, name: str):
    """What ``layer`` would read for ``name`` in the DONOR window: the token the SAME producer builds on the donor pass, at the donor's
    own tick of that layer (``scene`` -> ``(tokens, pad)``)."""
    i, b = _tick(Dn, layer), Dn.b
    if name == "ego":
        return model._ego(model._at(b["p10f"], i))
    if name == "nav":
        return model.e_nav(model._at(b["nav"], i))
    if name == "limit":
        return model.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1))
    if name == "scene":
        return model._scene(Dn.adapt, b["frame_ok"], i)
    raise ValueError(f"{name!r} has no producer to shadow")


@contextlib.contextmanager
def _patched(model, repl: dict):
    """Shadow the model's token producers named in ``repl`` by functions returning the given values, for ONE ``run_*`` call.
    Refuses (RuntimeError) unless each was called exactly once, and always removes the shadows."""
    calls = {k: 0 for k in repl}

    def mk(k):
        def f(*a, **kw):
            calls[k] += 1
            return repl[k]
        return f

    for k in repl:
        object.__setattr__(model, k, mk(k))            # instance __dict__ wins over the class method AND over the _modules entry
    try:
        yield
    finally:
        for k in repl:
            model.__dict__.pop(k, None)
    bad = {k: n for k, n in calls.items() if n != 1}
    if bad:
        raise RuntimeError(f"override did not reach the layer: producer call counts {bad} (each must be exactly 1)")


def _repl(model, layer: str, ov, Dn: Pass | None) -> dict:
    ov = _check_ov(model, layer, ov)
    if ov and Dn is None:
        raise ValueError("an override needs a donor pass")
    return {_PRODUCER[x]: donor_token(model, Dn, layer, x) for x in ov if x in _PRODUCER}


# ------------------------------------------------------------------------------------------------ the three layers, one input replaced
@torch.no_grad()
def run_S(model, P: Pass, ov=(), Dn: Pass | None = None) -> torch.Tensor:
    """S's checkpoints [B, 3, 3] in S's tick frame."""
    with _patched(model, _repl(model, "S", ov, Dn)):
        return model.run_s(P.b, P.adapt, P.last_s)


@torch.no_grad()
def run_T(model, P: Pass, ov=(), Dn: Pass | None = None, ck_t: torch.Tensor | None = None) -> dict:
    """T's output dict. ``ck_t``: the S checkpoints in T's frame (default: the clean pass's); ``'ck'`` in ``ov`` -> the donor's."""
    ov = _check_ov(model, "T", ov)
    if ov and Dn is None:
        raise ValueError("an override needs a donor pass")
    ck = P.ck_for_t if ck_t is None else ck_t
    if "ck" in ov:
        ck = Dn.ck_for_t
    with _patched(model, _repl(model, "T", [x for x in ov if x != "ck"], Dn)):
        return model.run_t(P.b, P.adapt, P.last_t, ck, P.ck_ok)


@torch.no_grad()
def run_O(model, P: Pass, ov=(), Dn: Pass | None = None, ck_o=None, code=None, ages=None):
    """O's ``(traj60, p30)``. ``ck_o`` / ``code``: the upstream values O reads (default: the clean pass's); ``'ck'`` / ``'code'`` in ``ov``
    -> the donor's; ``'age'`` -> ``ages`` = ``(a_t, a_s)`` of the donor's cycle position."""
    ov = _check_ov(model, "O", ov)
    if ov and Dn is None:
        raise ValueError("an override needs a donor pass")
    ck = P.ck_for_o if ck_o is None else ck_o
    cd = P.code if code is None else code
    b = P.b
    if "ck" in ov:
        ck = Dn.ck_for_o
    if "code" in ov:
        cd = Dn.code
    if "age" in ov:
        if ages is None:
            raise ValueError("the age override needs the donor-cycle ages")
        b = dict(b)
        b["a_t"], b["a_s"] = ages
    with _patched(model, _repl(model, "O", [x for x in ov if x in _PRODUCER], Dn)):
        return model.run_o(b, P.adapt, P.last_now, ck, P.ck_ok, cd)


@torch.no_grad()
def run_chain(model, P: Pass, Dn: Pass | None = None, ov: dict | None = None, ages=None) -> dict:
    """The whole S -> T -> O chain with ``ov = {'S': (...), 'T': (...), 'O': (...)}`` inputs replaced at each layer and the change FLOWING
    DOWNSTREAM (the Q-NAV arms). ``ov`` empty = ``forward_infer`` bit for bit."""
    ov = ov or {}
    ck_s = run_S(model, P, ov.get("S", ()), Dn)
    ck_for_t = M.transform_ckpt(ck_s, P.b["pose_s"], P.b["pose_t"])
    tout = run_T(model, P, ov.get("T", ()), Dn, ck_t=ck_for_t)
    code = model._code_pred(tout)
    ck_for_o = M.transform_ckpt(ck_s, P.b["pose_s"], P.b["pose_now"])
    traj60, p30 = run_O(model, P, ov.get("O", ()), Dn, ck_o=ck_for_o, code=code, ages=ages)
    return {"ckpt": ck_s, "tout": tout, "code": code, "traj60": traj60, "traj30": M.subsample_02(traj60), "p10": p30}


def cell_inputs(model, layer: str, x: str) -> tuple:
    """The inputs replaced by cell ``(layer, x)``; ``x == 'all'`` is the row reference (every input the layer reads, the prior excepted)."""
    return reads(model, layer) if x == "all" else _check_ov(model, layer, (x,))


@torch.no_grad()
def cell(model, P: Pass, Dn: Pass, layer: str, x: str, ages=None):
    """Layer ``layer``'s output with input ``x`` from the donor; upstream layers are the CLEAN pass's. S -> ck; T -> output dict; O -> (traj60, p30)."""
    ov = cell_inputs(model, layer, x)
    if layer == "S":
        return run_S(model, P, ov, Dn)
    if layer == "T":
        return run_T(model, P, ov, Dn)
    return run_O(model, P, ov, Dn, ages=ages)


def clean_cell(model, P: Pass, layer: str):
    """The layer's own clean output on the clean pass, as the cells return it."""
    if layer == "S":
        return P.out["ckpt"]
    if layer == "T":
        return {k: P.out[k] for k in ("goal_logits", "lat_logits", "lon_logits", "cons_std", "goal_prob", "cons_pred")}
    return P.out["traj60"], P.out["p10"]


#: the Q-NAV arms: the donor's nav token injected at the named layer(s), downstream propagating
QNAV_ARMS = {"nav@S": {"S": ("nav",)}, "nav@T": {"T": ("nav",)}, "nav@O": {"O": ("nav",)},
             "nav@STO": {"S": ("nav",), "T": ("nav",), "O": ("nav",)}}


# ------------------------------------------------------------------------------------------------ donor assignment and ages
def donor_perm(N: int, eid, seed: int) -> np.ndarray:
    """The eval's own derangement over EPISODES (``hjepa.donor_indices``)."""
    from tanitad.models.hjepa import donor_indices
    return donor_indices(N, episode_ids=torch.as_tensor(np.asarray(eid)), seed=int(seed)).numpy()


def cycle_ages(i: int, mode: str = "cycle") -> tuple:
    """The NOMINAL ages the banked eval gives window index ``i`` (``v8f_mini_eval.cmd_run``)."""
    if mode == "cycle":
        a_t = i % 3
        return a_t, a_t + (i // 3) % 5
    return 0, 0


def effective_ages(ds, t: int, a_t: int, a_s: int) -> tuple:
    """``V8FMiniWindows.item``'s clamp of nominal ages by the window's room (pinned equal to the dataset's by a test)."""
    room = max(int(t) - ds.window_start, 0)
    a_t = int(min(a_t, room, ds.max_age_t))
    a_s = int(max(a_t, min(a_s, room, ds.max_age_s)))
    return a_t, a_s


def donor_cycle_ages(ds, wins, j: int, mode: str = "cycle") -> tuple:
    """The effective ages donor window ``j`` carries in the banked eval (its own cycle position)."""
    a_t, a_s = cycle_ages(j, mode)
    return effective_ages(ds, wins[j][1], a_t, a_s)


# ------------------------------------------------------------------------------------------------ structural-zero (deliberate regression) copies
def regress_copy(model, kind: str):
    """A deep copy that CANNOT depend on the input a cell probes, so that cell must read exactly 0.0.
      'nav_const'  : the nav embedding's weights are zeroed -> every layer's nav token is the same constant (O x nav must read 0.0)
      'tscene_off' : T BUILDS no scene path (``read_t.uses_scene = False``, the model's own navonly mechanism for T) (T x scene must read 0.0)"""
    import copy
    m = copy.deepcopy(model)
    if kind == "nav_const":
        with torch.no_grad():
            m.e_nav.weight.zero_()
    elif kind == "tscene_off":
        m.read_t.uses_scene = False
    else:
        raise ValueError(kind)
    return m.eval()


# ------------------------------------------------------------------------------------------------ collection over windows
CELLS = [(L, x) for L in LAYERS for x in _READS[L]] + [(L, "all") for L in LAYERS]


def cell_name(layer: str, x: str) -> str:
    return f"{layer}/{x}"


class Collector:
    """Accumulates, over batches, each cell's per-window result (S: checkpoints in the NOW frame; T: argmax lat / lon; O: ADE6 / ADE2),
    the clean pass, and the Q-NAV arms. ``gt`` [N, 30, 2] float64 is the EVAL context's (the scoring ground truth)."""

    def __init__(self, model, gt: np.ndarray, qnav: bool = True, cells=None):
        self.model, self.gt = model, np.asarray(gt, np.float64)
        self.cells = [(L, x) for (L, x) in (CELLS if cells is None else cells) if x == "all" or x in reads(model, L)]
        if cells is not None:
            missing = [c for c in cells if c not in self.cells]
            if missing:
                raise ValueError(f"cells {missing} are not inputs of this model")
        self.qnav = bool(qnav) and cells is None                          # the Q-NAV arms run only with the full grid
        self.rows: dict[str, dict[str, list]] = {}

    def _put(self, name: str, **kw):
        r = self.rows.setdefault(name, {})
        for k, v in kw.items():
            r.setdefault(k, []).append(np.asarray(v))

    @staticmethod
    def _np(t):
        return t.detach().float().cpu().numpy()

    def _ade(self, idx, traj30):
        d = np.linalg.norm(self._np(traj30).astype(np.float64) - self.gt[idx], axis=-1)
        return d.mean(1), d[:, :10].mean(1)

    def _s(self, name, P, ck):
        self._put(name, ck_now=self._np(M.transform_ckpt(ck, P.b["pose_s"], P.b["pose_now"])))

    def _t(self, name, tout):
        self._put(name, lat_pred=self._np(tout["lat_logits"]).argmax(-1), lon_pred=self._np(tout["lon_logits"]).argmax(-1))

    def _o(self, name, idx, traj60):
        a6, a2 = self._ade(idx, M.subsample_02(traj60))
        self._put(name, ade6=a6, ade2=a2)

    @torch.no_grad()
    def add_batch(self, idx, P: Pass, Dn: Pass, ages_d):
        """``idx``: the batch's window indices into ``gt``; ``P`` / ``Dn``: the clean passes of the windows and of their donors; ``ages_d``:
        the donor-cycle ages ``(a_t [B], a_s [B])`` (long tensors)."""
        m, idx = self.model, np.asarray(idx)
        # the clean pass, in every layer's own metric AND with the full outputs the reproduction control compares
        out = P.out
        self._s("clean/S", P, P.ck_s)
        self._t("clean/T", out)
        self._o("clean/O", idx, out["traj60"])
        self._put("clean/full", ckpt=self._np(out["ckpt"]), goal_prob=self._np(out["goal_prob"]), cons_pred=self._np(out["cons_pred"]),
                  lat_logits=self._np(out["lat_logits"]), lon_logits=self._np(out["lon_logits"]), traj30=self._np(out["traj30"]),
                  code=self._np(out["code"]))
        for (L, x) in self.cells:
            o = cell(m, P, Dn, L, x, ages=ages_d)
            n = cell_name(L, x)
            if L == "S":
                self._s(n, P, o)
            elif L == "T":
                self._t(n, o)
            else:
                self._o(n, idx, o[0])
        if self.qnav:
            for arm, ov in QNAV_ARMS.items():
                if arm == "nav@O":
                    continue                                                    # = the O x nav cell (S, T clean): read from it
                r = run_chain(m, P, Dn, ov, ages=ages_d)
                self._o(arm, idx, r["traj60"])

    def finish(self) -> dict:
        out = {n: {k: np.concatenate(v) for k, v in r.items()} for n, r in self.rows.items()}
        if self.qnav and "O/nav" in out:
            out["nav@O"] = {k: v.copy() for k, v in out["O/nav"].items()}
        return out


def collate_window_batch(model, ds, wins, perm, idxs, ages_mode: str = "cycle"):
    """Items for window indices ``idxs`` (cycle ages by window index, as the banked eval) and their donors ``perm[i]`` AT THE SAME NOMINAL
    AGES (as ``--tac shuffle``); returns ``(P, Dn, ages_d)``."""
    from tanitad.data import v8f_mini_data as D
    own, don, ad = [], [], []
    for i in idxs:
        ci, t, _, _ = wins[i]
        a_t, a_s = cycle_ages(i, ages_mode)
        own.append(ds.item(ci, t, a_t, a_s))
        j = int(perm[i])
        dci, dt_, _, _ = wins[j]
        don.append(ds.item(dci, dt_, a_t, a_s))
        ad.append(donor_cycle_ages(ds, wins, j, ages_mode))
    P = make_pass(model, D.collate(own))
    Dn = make_pass(model, D.collate(don))
    ages_d = (torch.as_tensor([a for a, _ in ad], dtype=torch.long), torch.as_tensor([s for _, s in ad], dtype=torch.long))
    return P, Dn, ages_d


def collect(model, ds, wins, gt, perm, batch: int = 64, ages_mode: str = "cycle", qnav: bool = True, progress=None,
            n_use: int | None = None, cells=None) -> dict:
    """All cells over the first ``n_use`` windows (default all) for one donor assignment ``perm`` over ALL of ``wins`` (identity perm = the
    identity control). ``gt`` is aligned with ``wins``."""
    model.eval()
    col = Collector(model, gt, qnav=qnav, cells=cells)
    N = len(wins) if n_use is None else int(n_use)
    for b0 in range(0, N, batch):
        idxs = list(range(b0, min(b0 + batch, N)))
        P, Dn, ages_d = collate_window_batch(model, ds, wins, perm, idxs, ages_mode)
        col.add_batch(idxs, P, Dn, ages_d)
        if progress:
            progress(b0, N)
    return col.finish()
