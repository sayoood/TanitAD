"""PREREG_V8F_LISTEN2 -- the three follow-ups to the who-listens-to-what grid. ZERO TRAINING; ``listen.py`` (a copy of LISTEN's, unmodified) is REUSED.

  E1 (PRIMARY)  route through the chain: a DONOR's nav token is injected at S ONLY (``LS.run_chain(.., {'S': ('nav',)})``: S's checkpoints change, T and
                O read them as usual, their own nav tokens untouched); T-lat / T-lon macro-F1 and O's ADE6 are read at the end of the chain.
                Per window the module ASSERTS (``torch.equal``) that T's and O's own ego / nav / limit tokens, O's age token and both scene paths are the
                clean pass's, bit for bit; and that S's nav token is the donor's.
  E2            a second intervention family: each input is replaced by the TRAINING-SET MEAN of that input (a ``MeanSrc`` built from HBLEND's 9,582
                training windows, per seed) instead of a donor. The replacement goes through the SAME shadow mechanism as LISTEN's donors
                (``LS._patched``: every shadowed producer must be called exactly once), so an override that never reaches the layer is refused.
  E3            O's coherence: ego + S checkpoints + T code from the SAME donor, together (``LS.run_O(.., ('ego', 'ck', 'code'))``; the prior keeps the
                window's own p10f; scene / nav / limit / age stay O's own), against each single swap.

READINGS OF THE PREREG TEXT THAT THE TEXT LEAVES OPEN (all fixed here and in ``read2.py``, in code, BEFORE any LISTEN2 number was computed):
  M1  "the training-set mean of that input": the mean over the 9,582 HBLEND training windows (``train_windows(hist, start)`` of the 141 fit clips, the
      eval's cycle ages by window index) of the input AS THE LAYER READS IT at inference (``forward_infer``): per layer, at that layer's own tick, the
      token its producer builds (ego / nav / limit), the scene tokens, S's checkpoints in the layer's frame (``ck_for_t`` for T, ``ck_for_o`` for O),
      T's code, and the age pair. Every producer except the scene's is LINEAR, so the mean token of an input IS the producer of the mean input; the
      scene mean is taken token by token over the windows whose frame is not padded, and the replacement scene is not padded (pad all False).
  M2  mean ages: the mean (a_t, a_s) in ticks, fed as FLOAT tensors (the model only ever calls ``.float()`` on them).
  M3  E2's "largest effect": the argmax over the row's input columns of the SIGNED delta (the harm direction of the PREREG's sign convention). With
      delta(L, all) > 0 this equals R1's argmax of n = delta(L,X) / delta(L,all); it is used because 'all' may be undefined under mean replacement.
  M4  E2 verdict: a row AGREES iff its argmax equals the grid's largest-n input at BOTH seeds; k = the number of agreeing rows; ROBUST iff k = 4,
      PARTLY ROBUST iff k in {2, 3}, NOT ROBUST iff k < 2. The per-seed counts are reported beside.
  M5  E3 "max over the three single swaps": the single swap with the largest MEAN ADE6 on the full set (per seed); E3 = joint - that swap, paired.
      (A per-window maximum would be an upward-biased comparator and is reported only as a descriptive row.)
  M6  E1 "above the donor-seed floor": |E1(s0, donor 0) - E1(s0, donor 1)| of the same metric, applied to BOTH seeds (LISTEN's R3).
  M7  E1 outcome precedence: ROUTE FLOWS DOWN iff E1 > 0, separated and above the floor at both seeds; else PARALLEL PATH iff E1 is not separated
      at either seed; else SEEDS DISAGREE.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

import listen as LS

M = LS.M
LAYERS = LS.LAYERS

#: the E1 / E3 cells recomputed next to the new arms (they are LISTEN's cells; recomputing them lets the run be checked against LISTEN's own arrays)
REF_CELLS = [("S", "nav"), ("T", "ck"), ("T", "nav"), ("O", "ego"), ("O", "ck"), ("O", "code")]
E1_NAME, E3_NAME = "e1/nav@S", "e3/ego+ck+code"
E3_TRIPLE = ("ego", "ck", "code")
E3_SINGLES = ("ego", "ck", "code")


# ------------------------------------------------------------------------------------------------ capturing what each Reader is handed
@contextlib.contextmanager
def capture(model):
    cap, hs = {}, []
    for nm in ("read_s", "read_t", "read_o"):
        def hook(mod, args, nm=nm):
            cap.setdefault(nm, []).append(tuple(None if a is None else a.detach().clone() for a in args))
        hs.append(getattr(model, nm).register_forward_pre_hook(hook))
    try:
        yield cap
    finally:
        for h in hs:
            h.remove()


def _eq(a, b) -> bool:
    return torch.equal(a, b)


# ------------------------------------------------------------------------------------------------ E1: nav at S only, the rest of the chain untouched
def own_cond_tokens(model, P: LS.Pass, layer: str) -> dict:
    """The ego / nav / limit tokens ``layer`` builds for the window's OWN inputs (the clean pass), by the model's own producers."""
    i, b = LS._tick(P, layer), P.b
    return {"ego": model._ego(model._at(b["p10f"], i)), "nav": model.e_nav(model._at(b["nav"], i)),
            "limit": model.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1))}


def own_age_token(model, P: LS.Pass):
    return model.e_age(torch.stack([P.b["a_t"].float() * LS.DT, P.b["a_s"].float() * LS.DT], -1))


@torch.no_grad()
def e1_arm(model, P: LS.Pass, Dn: LS.Pass, ages_d, check: bool = True):
    """The whole S -> T -> O chain with the DONOR's nav token at S only. Returns ``(chain result, number of windows asserted)``.
    Asserted per window, with ``torch.equal`` (raises AssertionError): S's nav token is the donor's; T's and O's ego / nav / limit tokens are the clean
    pass's; O's age token is the clean pass's; T's and O's scene tokens and pad are the clean pass's."""
    with capture(model) as cap:
        r = LS.run_chain(model, P, Dn, {"S": ("nav",)}, ages=ages_d)
    if not check:
        return r, 0
    tok_d = own_cond_tokens(model, Dn, "S")
    cs, _, _ = cap["read_s"][0]
    own_s = own_cond_tokens(model, P, "S")
    assert _eq(cs[:, LS.TOKEN_POS["S"]["nav"]], tok_d["nav"]), "E1: S's nav token is not the donor's"
    assert _eq(cs[:, LS.TOKEN_POS["S"]["ego"]], own_s["ego"]) and _eq(cs[:, LS.TOKEN_POS["S"]["limit"]], own_s["limit"]), "E1: S's ego/limit moved"
    for L in ("T", "O"):
        cond, scene, pad = cap[LS._READER[L]][0]
        own = own_cond_tokens(model, P, L)
        for nm in ("ego", "nav", "limit"):
            assert _eq(cond[:, LS.TOKEN_POS[L][nm]], own[nm]), f"E1: {L}'s own {nm} token is not bit-identical to clean"
        if L == "O":
            assert _eq(cond[:, LS.TOKEN_POS["O"]["age"]], own_age_token(model, P)), "E1: O's age token moved"
        sc, sp = model._scene(P.adapt, P.b["frame_ok"], LS._tick(P, L))
        assert _eq(scene, sc) and _eq(pad, sp), f"E1: {L}'s scene path moved"
    return r, int(P.b["feats"].shape[0])


# ------------------------------------------------------------------------------------------------ E3: ego + S checkpoints + T code from one donor
@torch.no_grad()
def e3_joint(model, P: LS.Pass, Dn: LS.Pass):
    """O's ``(traj60, p30)`` with ego, S checkpoints and T code from the SAME donor window; the prior, scene, nav, limit and age stay the window's own."""
    out = LS.run_O(model, P, E3_TRIPLE, Dn)
    assert _eq(out[1], P.out["p10"]), "E3: the kinematic prior moved (it must keep the window's own p10f)"
    return out


# ------------------------------------------------------------------------------------------------ E2: mean replacement
@dataclass
class Means:
    """Training-set means of every input as each layer reads it (float64 sums / counts reduced to float32 tensors)."""
    tok: dict            # (layer, 'ego'|'nav'|'limit') -> [d]
    scene: dict          # layer -> [hist*n_tok, d]
    ck: dict             # 'T' -> [3, 3], 'O' -> [3, 3]   (S's checkpoints in the layer's frame)
    code: torch.Tensor   # [CODE_DIM]
    age: torch.Tensor    # [2]  mean (a_t, a_s) in ticks
    n: int
    meta: dict


class MeanAccumulator:
    def __init__(self, model):
        self.model, self.n = model, 0
        self.s: dict = {}
        self.scene_sum, self.scene_cnt = {}, {}
        self.pad_any = {L: 0 for L in LAYERS}

    def _add(self, key, x):
        v = x.detach().double().sum(0)
        self.s[key] = self.s[key] + v if key in self.s else v

    @torch.no_grad()
    def add(self, P: LS.Pass):
        m = self.model
        B = P.b["feats"].shape[0]
        for L in LAYERS:
            for nm in ("ego", "nav", "limit"):
                self._add(("tok", L, nm), LS.donor_token(m, P, L, nm))
            if "scene" in LS.reads(m, L):
                tok, pad = LS.donor_token(m, P, L, "scene")
                valid = (~pad)[..., None].double()
                ws = (tok.double() * valid).sum(0)
                wc = valid.sum(0)
                self.scene_sum[L] = self.scene_sum[L] + ws if L in self.scene_sum else ws
                self.scene_cnt[L] = self.scene_cnt[L] + wc if L in self.scene_cnt else wc
                self.pad_any[L] += int(pad.any(1).sum())
        self._add(("ck", "T"), P.ck_for_t)
        self._add(("ck", "O"), P.ck_for_o)
        self._add(("code",), P.code)
        self._add(("age",), torch.stack([P.b["a_t"], P.b["a_s"]], -1).double())
        self.n += B

    def finish(self, meta=None) -> Means:
        f = lambda t: (t / self.n).float()                                                     # noqa: E731
        scene = {}
        for L, cnt in self.scene_cnt.items():
            assert bool((cnt > 0).all()), f"{L}: a scene token position was padded in EVERY training window"
            scene[L] = (self.scene_sum[L] / cnt).float()
        return Means(tok={(k[1], k[2]): f(v) for k, v in self.s.items() if k[0] == "tok"}, scene=scene,
                     ck={k[1]: f(v) for k, v in self.s.items() if k[0] == "ck"}, code=f(self.s[("code",)]), age=f(self.s[("age",)]), n=self.n,
                     meta={**(meta or {}), "pad_any_windows": dict(self.pad_any)})


def save_means(mn: Means, path):
    flat = {f"tok|{L}|{nm}": v.numpy() for (L, nm), v in mn.tok.items()}
    flat.update({f"scene|{L}": v.numpy() for L, v in mn.scene.items()})
    flat.update({f"ck|{L}": v.numpy() for L, v in mn.ck.items()})
    flat["code"], flat["age"], flat["n"] = mn.code.numpy(), mn.age.numpy(), np.array(mn.n)
    import json
    flat["meta"] = np.array(json.dumps(mn.meta, default=str))
    np.savez_compressed(path, **flat)


def load_means(path) -> Means:
    import json
    z = np.load(path, allow_pickle=False)
    t = lambda k: torch.as_tensor(z[k])                                                         # noqa: E731
    return Means(tok={(k.split("|")[1], k.split("|")[2]): t(k) for k in z.files if k.startswith("tok|")},
                 scene={k.split("|")[1]: t(k) for k in z.files if k.startswith("scene|")}, ck={k.split("|")[1]: t(k) for k in z.files if k.startswith("ck|")},
                 code=t("code"), age=t("age"), n=int(z["n"]), meta=json.loads(str(z["meta"])))


class MeanSrc:
    """The replacement values: the training-set means, expanded to the batch."""

    def __init__(self, mn: Means):
        self.mn = mn

    def tok(self, model, P, layer, name):
        return self.mn.tok[(layer, name)][None].expand(P.b["feats"].shape[0], -1).contiguous()

    def scene(self, model, P, layer):
        t = self.mn.scene[layer]
        B = P.b["feats"].shape[0]
        return t[None].expand(B, -1, -1).contiguous(), torch.zeros(B, t.shape[0], dtype=torch.bool)

    def ck(self, model, P, layer):
        return self.mn.ck[layer][None].expand(P.b["feats"].shape[0], -1, -1).contiguous()

    def code(self, model, P):
        return self.mn.code[None].expand(P.b["feats"].shape[0], -1).contiguous()

    def ages(self, model, P):
        a = self.mn.age[None].expand(P.b["feats"].shape[0], -1)
        return a[:, 0].contiguous(), a[:, 1].contiguous()


class OwnSrc:
    """The CONTROL source: the window's OWN value of every input, handed through the same replacement path (so a mean cell must read exactly 0.0)."""

    def tok(self, model, P, layer, name):
        return LS.donor_token(model, P, layer, name)

    def scene(self, model, P, layer):
        return LS.donor_token(model, P, layer, "scene")

    def ck(self, model, P, layer):
        return P.ck_for_t if layer == "T" else P.ck_for_o

    def code(self, model, P):
        return P.code

    def ages(self, model, P):
        return P.b["a_t"].float(), P.b["a_s"].float()


@torch.no_grad()
def repl_cell(model, P: LS.Pass, src, layer: str, x: str):
    """Layer ``layer``'s output with input ``x`` ('all' = every input the layer reads) replaced by ``src``'s value; upstream layers are the CLEAN pass's.
    S -> ck; T -> output dict; O -> (traj60, p30). The shadow mechanism and the exactly-once check are LISTEN's (``LS._patched``)."""
    ov = LS.cell_inputs(model, layer, x)
    repl = {}
    for nm in ov:
        if nm in ("ego", "nav", "limit"):
            repl[LS._PRODUCER[nm]] = src.tok(model, P, layer, nm)
        elif nm == "scene":
            repl[LS._PRODUCER["scene"]] = src.scene(model, P, layer)
    b = P.b
    if layer == "S":
        with LS._patched(model, repl):
            return model.run_s(b, P.adapt, P.last_s)
    if layer == "T":
        ck = src.ck(model, P, "T") if "ck" in ov else P.ck_for_t
        with LS._patched(model, repl):
            return model.run_t(b, P.adapt, P.last_t, ck, P.ck_ok)
    ck = src.ck(model, P, "O") if "ck" in ov else P.ck_for_o
    cd = src.code(model, P) if "code" in ov else P.code
    if "age" in ov:
        b = dict(b)
        b["a_t"], b["a_s"] = src.ages(model, P)
    with LS._patched(model, repl):
        return model.run_o(b, P.adapt, P.last_now, ck, P.ck_ok, cd)


# ------------------------------------------------------------------------------------------------ collection
class Collector2(LS.Collector):
    """mode 'donor': the clean pass, the reference cells, E1 (nav at S only, whole chain) and E3 (joint triple). 'mean' / 'own': the clean pass and the
    replacement cells (every input of every layer + the row reference) by ``MeanSrc`` / ``OwnSrc``."""

    def __init__(self, model, gt, mode: str, means: Means | None = None, check: bool = True):
        super().__init__(model, gt, qnav=False, cells=[])
        assert mode in ("donor", "mean", "own"), mode
        self.mode, self.check, self.n_asserted = mode, check, 0
        self.src = MeanSrc(means) if mode == "mean" else (OwnSrc() if mode == "own" else None)
        if mode == "mean" and means is None:
            raise ValueError("mean mode needs the means")
        self.rcells = [(L, x) for (L, x) in LS.CELLS if x == "all" or x in LS.reads(model, L)]
        self.ref = [(L, x) for (L, x) in REF_CELLS if x in LS.reads(model, L)]

    def _clean(self, idx, P):
        out = P.out
        self._s("clean/S", P, P.ck_s)
        self._t("clean/T", out)
        self._o("clean/O", idx, out["traj60"])
        self._put("clean/full", ckpt=self._np(out["ckpt"]), goal_prob=self._np(out["goal_prob"]), cons_pred=self._np(out["cons_pred"]),
                  lat_logits=self._np(out["lat_logits"]), lon_logits=self._np(out["lon_logits"]), traj30=self._np(out["traj30"]), code=self._np(out["code"]))

    def _cell(self, name, L, idx, P, o):
        if L == "S":
            self._s(name, P, o)
        elif L == "T":
            self._t(name, o)
        else:
            self._o(name, idx, o[0])

    @torch.no_grad()
    def add_batch(self, idx, P: LS.Pass, Dn: LS.Pass | None, ages_d):
        m, idx = self.model, np.asarray(idx)
        self._clean(idx, P)
        if self.mode in ("mean", "own"):
            for (L, x) in self.rcells:
                self._cell(LS.cell_name(L, x), L, idx, P, repl_cell(m, P, self.src, L, x))
            return
        for (L, x) in self.ref:
            self._cell(LS.cell_name(L, x), L, idx, P, LS.cell(m, P, Dn, L, x, ages=ages_d))
        r, n = e1_arm(m, P, Dn, ages_d, self.check)
        self.n_asserted += n
        self._s(E1_NAME, P, r["ckpt"])
        self._t(E1_NAME, r["tout"])
        self._o(E1_NAME, idx, r["traj60"])
        traj60, _ = e3_joint(m, P, Dn)
        self._o(E3_NAME, idx, traj60)


def collate_own(model, ds, wins, idxs, ages_mode: str = "cycle") -> LS.Pass:
    """The clean pass of window indices ``idxs`` alone (cycle ages by window index, as LISTEN and the banked eval)."""
    from tanitad.data import v8f_mini_data as D
    items = []
    for i in idxs:
        ci, t, _, _ = wins[i]
        a_t, a_s = LS.cycle_ages(i, ages_mode)
        items.append(ds.item(ci, t, a_t, a_s))
    return LS.make_pass(model, D.collate(items))


def collect2(model, ds, wins, gt, perm, mode: str, means: Means | None = None, batch: int = 64, progress=None, check: bool = True):
    """All LISTEN2 cells over every window of ``wins`` (donor mode: ``perm`` is the donor assignment; the identity permutation is the identity control)."""
    model.eval()
    col = Collector2(model, gt, mode, means, check)
    N = len(wins)
    for b0 in range(0, N, batch):
        idxs = list(range(b0, min(b0 + batch, N)))
        if mode == "donor":
            P, Dn, ages_d = LS.collate_window_batch(model, ds, wins, perm, idxs)
        else:
            P, Dn, ages_d = collate_own(model, ds, wins, idxs), None, None
        col.add_batch(idxs, P, Dn, ages_d)
        if progress:
            progress(b0, N)
    return col.finish(), col.n_asserted
