#!/usr/bin/env python
"""H-HJ-COLLAPSE-1 (RA-0) -- is refav1's action-deafness H-JEPA's slow-feature collapse?

Pre-registered in ``Project Steering/PREREG_HJ_STAGE0.md`` (stream S2, CPU only).  This is
the MEASUREMENT job: it streams over the first 40 sorted clip ids of the 141 v7.2 EVAL
episodes, one episode at a time, and writes one JSON per episode (+ a float64 moments file in
the work dir).  ``hj_collapse_readout.py`` aggregates.  Resumable per episode.

DESIGN DECISIONS (fixed BEFORE the first number was seen; file mtime is the audit trail)
-------------------------------------------------------------------------------------
Model      refav1-b1-v72-ep3-speed, ckpt step 21109, STRICT load into the repo's refa_v1.py
           (taniteval/tools/refav1_arm.py::load_model).  fp32 on CPU.  (Training ran bf16
           autocast on GPU; the dev-box eval dumps are fp32 GPU.  UNVERIFIED: fp32-CPU vs
           fp32-GPU agreement on these exact windows.)
Episodes   sorted(cache stems)[:40]  == first 40 sorted clip ids (stems ARE the clip ids; checked).
Windows    the arm's own loader RefAV1Windows (W=4, K_loader=30, grid 0.2 s), t = last observed
           cache index.  Three window sets, all subsets of the loader's windows:
             S_all  every window (stride 1)              -> rho of the ADAPTER latent
             S_10   (t-(W-1)) % 10 == 0  (every 2 s)     -> baseline rollouts: rho of predictions,
                                                            rho of the adapter on the SAME set
             S_40   (t-(W-1)) % 40 == 0  (the PREREG)    -> action-swap sensitivities + controls
Latent     z_t := adapter(std(feats))[:, -1] of the 4-frame window ending at t (the window-
           causal latent the model really uses; whole-episode encoding would leak t+1 through
           the non-causal depthwise temporal conv).  The rollout start state is
           `last` = model._last_state(field) (== field[:, -1] here: motion_inject is False).
rho        paper Eq. (10): rho = E_e[tr Var_t(z | e)] / tr Var(z), UNBIASED (ddof=1) within each
           episode and pooled (so iid z gives rho ~ 1 at any n).  Episode-cluster bootstrap.
           Levels: token-mean (1024-d) AND the full 640x1024 field, for: raw standardised
           DINOv3 feats (REFERENCE), adapter latent, operative predictions k in {1,5,10}
           (0.2/1.0/2.0 s), tactical start field (64 queries x 1024) and tactical predictions
           k in {1,5,10} (0.6/3.0/6.0 s).
sens_k(c)  mean over windows of  ||zhat_k(swap c) - zhat_k|| / ||zhat_k - z_0||,  L2 over the
           FULL flattened field.  zhat is the operative predictor rolled from `last` with the
           window's own `intent` (tactical policy output from observed window + nav; held FIXED,
           so the swap isolates the predictor's action pathway -- H-JEPA's definition).
           z_0 = `last`.  Teacher-forced (T0) recorded actions -- the in-distribution input.
Predictor input per step j is the model's own augment_actions row:
           x_j = (a_j, steer_j, v_j/30),  v_j = v0 + sum_{i<j} a_i*0.2   (refa_v1.py:1688-1731).
           (loader channel 1 is the road-wheel STEER ANGLE fed as 'kappa' -- REFAV1_ARM.md 3b;
           this run feeds the model exactly what it was trained on.)
Channel swaps (own window A, donor window D, both 3-wide augmented inputs, K steps):
  all      X = D                                   (donor's a, steer, v-profile)
  a        X = A with channel 0 := D[...,0]        (v channel stays OWN, i.e. NOT re-integrated)
  steer    X = A with channel 1 := D[...,1]
  v        X = A with channel 2 := D[...,2]        (donor's whole speed profile v0_d + sum a_d)
  a_reint  [IMPLEMENTED BUT NOT RUN beyond episode 00: dropped for CPU time (~8.5 s per rollout);
           its info is the sum of the `a` and `v` columns] controls (a_donor, steer_own) + v0_own
  v0       controls own, v0 := donor v0, then augment_actions  (constant offset on channel 2)
  self     X = A   (the CONTROL: must read exactly 0.0)
PRIMARY columns (named in the PREREG): all, a, steer, v.   v0 is SECONDARY (the task brief wrote
"v0/speed"; `v` is the PREREG speed column, `v0` the offset-only variant).
Donor: a fixed derangement of the 40 episodes (numpy default_rng(0).permutation, fixed points
repaired by swapping with the next slot); donor window = the donor's S_40 window with t nearest
to the own window's t.
Tactical level: tactical.rollout(tac0, acts_in[:, ::3][:, :10], intent) -- the same swaps on the
30-step action tensor, strided exactly as forward() does (refa_v1.py:1838).

PRE-REGISTERED DECISION RULE (copied, and its one uncovered cell resolved NOW):
  steer-sens(op, k=10) < 0.05 and v-sens >= 5 x steer-sens   -> O-COLLAPSE
  steer-sens(op, k=10) >= 0.10                                -> O-NOT
  0.05 <= steer-sens < 0.10                                   -> inconclusive (run both levers)
  steer-sens < 0.05 but v-sens < 5 x steer-sens (UNCOVERED by the PREREG) -> reported as
      "inconclusive (uncovered cell: steer-deaf but v not 5x)"; both levers run.
  The decision uses the POINT estimate (mean of per-window ratios, `v` = the channel swap); the
  episode-cluster CI is reported beside it and any threshold it straddles is stated.

EXPLORATORY ADDITIONS (added after a 1-episode shake-down, before the other 39 episodes ran;
NOT part of the PREREG decision): dose-response probes d_steer05 / d_a1 (episode 00 also has d_steer15) -- own input
plus a constant +0.05 rad steer, or +1 m/s^2 accel, over the horizon -- so the
per-unit gain can be read when the donor swap is small; ||z_0||; the same sens ratio measured AFTER the
model's linear decoder `to_enc` (the space the loss reads; added after the shake-down showed the
adapter-space rollout displacement is ~43 % of ||z_0|| per step, i.e. a large action-independent
term sits in the PREREG denominator); and the T0 WM-vs-persistence MSE
(arm wm_diagnostic_T0 definition) on S_10.

CONTROLS (same windows, same probe code):
  self-swap with the real model          -> 0.0 exactly (op: every 4th S_40 window; tac: all)
  identity-predictor stub (z_k = z_0)     -> numerator exactly 0 (denominator also 0: flagged)
  identity+fixed-drift stub               -> exactly 0.0 with a NON-degenerate denominator
  linear-action stub (z_k = z_0 + sum_j x_j W, fixed random W, sigma 0.02) -> > 0

Usage:  hj_collapse_diag.py --out-dir <work> [--episodes-n 40]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.environ.get("TANITAD_REPO", "E:/Projects/TanitAD")
sys.path.insert(0, os.path.join(_REPO, "taniteval", "tools"))

CKPT = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt"
CFG = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json"
CACHE = "C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval"
EPS = "C:/Users/Admin/tanitad-data/refav1-eval141/eps"
LABELS = "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"

KS = (1, 5, 10)                       # steps read (op: 0.2/1.0/2.0 s ; tac: 0.6/3.0/6.0 s)
K_OP = 10                             # operative steps rolled (2.0 s)
VARIANTS = ("self", "all", "a", "steer", "v", "v0")   # a_reint dropped after ep00 (speed trim)
STUB_VARIANTS = ("self", "all", "a", "steer", "v")
#: DOSE-RESPONSE probes (EXPLORATORY, not in the PREREG; added after the 1-episode shake-down
#: showed the donor steer swap is tiny: rms 0.005 rad).  own input + a CONSTANT offset on one
#: channel over the whole horizon: (channel, amount).  steer in road-wheel radians, a in m/s^2.
DOSES = {"d_steer05": (1, 0.05), "d_a1": (0, 1.0)}   # d_steer15 dropped after ep00 (speed trim)
STRIDE_P = 10                         # S_10
STRIDE_S = 40                         # S_40  (PREREG)
SELF_OP_EVERY = 4                     # operative self-swap on every 4th S_40 window
SPEED_SCALE = 30.0


def _p(*a):
    print(*a, flush=True)


class Mom:
    """Streaming first/second moments of a set of vectors (float64)."""
    __slots__ = ("n", "s", "ss")

    def __init__(self):
        self.n, self.s, self.ss = 0, None, 0.0

    def add(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self.s is None:
            self.s = np.zeros_like(x)
        self.s += x
        self.ss += float(x @ x)
        self.n += 1

    def within(self):
        if self.n < 2:
            return None
        return (self.ss - float(self.s @ self.s) / self.n) / (self.n - 1)


def derangement(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n)), "derangement repair failed"
    assert sorted(perm.tolist()) == list(range(n))
    return perm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episodes-n", type=int, default=40)
    ap.add_argument("--only", type=int, default=None, help="process only this episode index")
    ap.add_argument("--repo-raw", default=None,
                    help="repo raw/ dir; per-episode JSON is mirrored there")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    import tanitad
    _p(f"[env] tanitad={tanitad.__file__} torch={torch.__version__} "
       f"threads={torch.get_num_threads()}")
    if not os.path.normcase(tanitad.__file__).startswith(os.path.normcase(_REPO)):
        raise SystemExit(f"[diag] tanitad imported from {tanitad.__file__}, not {_REPO}")
    import refav1_arm as R

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "ep"), exist_ok=True)
    if a.repo_raw:
        os.makedirs(os.path.join(a.repo_raw, "ep"), exist_ok=True)

    t_load = time.time()
    model, cfg, prov = R.load_model(CKPT, CFG, "cpu", False)
    assert prov["step"] == 21109, f"ckpt step {prov['step']} != 21109"
    assert prov["state_dict_load"] == {"missing_keys": [], "unexpected_keys": []}
    assert cfg.speed_channel and cfg.a_dim == 2 and not cfg.motion_inject
    _p(f"[model] step={prov['step']} strict={prov['state_dict_load']} "
       f"load={time.time() - t_load:.1f}s")
    stride_tac = model._stride(cfg.tac_dt)
    assert stride_tac == 3 and cfg.tac_steps == 10 and cfg.op_dt == 0.2

    names_all = R.episode_names(CACHE)
    assert len(names_all) == 141, len(names_all)
    names = names_all[:a.episodes_n]

    class _A:  # the arm's loader args
        cache, episodes, labels, nav, lru = CACHE, EPS, LABELS, LABELS, 3
    ld = R.build_loader(_A, cfg, 30, names)
    assert ld.W == 4 and len(ld.names) == len(names)
    for nm in names:
        assert ld.clip_id[nm] == nm, f"stem {nm} != clip_id {ld.clip_id[nm]}"
    sel_all = {}
    for wi, (ei, t) in enumerate(ld.windows):
        sel_all.setdefault(ei, []).append((wi, t))
    perm = derangement(len(names), 0)
    s40_t = {ei: [t for (_, t) in ws if (t - (ld.W - 1)) % STRIDE_S == 0]
             for ei, ws in sel_all.items()}
    _p(f"[loader] {len(ld)} windows over {len(names)} episodes; S_40 windows/ep "
       f"min {min(len(v) for v in s40_t.values())} max {max(len(v) for v in s40_t.values())} "
       f"total {sum(len(v) for v in s40_t.values())}")
    meta = {"ckpt": CKPT, "ckpt_step": prov["step"], "strict": prov["state_dict_load"],
            "names": names, "donor_perm": perm.tolist(), "ks": list(KS),
            "variants": list(VARIANTS), "stride_p": STRIDE_P, "stride_s": STRIDE_S,
            "self_op_every": SELF_OP_EVERY, "tac_stride": stride_tac,
            "n_windows_loader": len(ld), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "dtype": "float32", "device": "cpu",
            "tanitad_file": tanitad.__file__}
    with open(os.path.join(a.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if a.repo_raw:
        with open(os.path.join(a.repo_raw, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)

    # fixed stub parameters (seeded; independent of the data)
    g = torch.Generator().manual_seed(1234)
    W_lin = torch.randn(3, 1024, generator=g) * 0.02
    delta_op = torch.randn(1, 640, 1024, generator=g) * 0.02
    delta_tac = torch.randn(1, cfg.tac_queries, 1024, generator=g) * 0.02

    # ------------------------------------------------------------------ helpers
    def aug(ctrl, v0):
        return model.augment_actions(ctrl, v0)

    def make_variant(name, own_c, own_v0, don_c, don_v0):
        A = aug(own_c, own_v0)
        D = aug(don_c, don_v0)
        if name == "self":
            return A.clone()
        if name == "all":
            return D.clone()
        if name in ("a", "steer", "v"):
            ch = {"a": 0, "steer": 1, "v": 2}[name]
            X = A.clone()
            X[..., ch] = D[..., ch]
            return X
        if name == "a_reint":
            c = own_c.clone()
            c[..., 0] = don_c[..., 0]
            return aug(c, own_v0)
        if name == "v0":
            return aug(own_c, don_v0)
        raise KeyError(name)

    def roll_op(last, X, intent):
        return model.operative.rollout(last, X[:, :K_OP], intent=intent)

    def roll_tac(tac0, X, intent):
        tac_a = X[:, ::stride_tac][:, :cfg.tac_steps]
        return model.tactical.rollout(tac0, tac_a, intent=intent)

    def sens_pack(pred_sw, pred_own, z0, enc=False):
        """[num, den] in the ADAPTER space (the PREREG metric).  With enc=True (operative level
        only) two more entries: the same ratio's num/den after the model's own linear decoder
        `to_enc` (the space the training loss actually reads) -- EXPLORATORY robustness."""
        out = {}
        E0 = model.to_enc(z0).double() if enc else None
        for k in KS:
            num = float((pred_sw[:, k - 1].double() - pred_own[:, k - 1].double()).norm())
            den = float((pred_own[:, k - 1].double() - z0.double()).norm())
            out[f"k{k}"] = [num, den]
            if enc:
                Eo = model.to_enc(pred_own[:, k - 1]).double()
                Es = model.to_enc(pred_sw[:, k - 1]).double()
                out[f"k{k}"] += [float((Es - Eo).norm()), float((Eo - E0).norm())]
        return out

    def stub_roll(kind, z0, X, delta, n_steps, strided):
        """Stub predictors on the SAME inputs/interfaces as the real rollout."""
        if strided:
            X = X[:, ::stride_tac][:, :cfg.tac_steps]
        else:
            X = X[:, :K_OP]
        if kind == "identity":
            return z0[:, None].expand(-1, X.shape[1], -1, -1).clone()
        if kind == "drift":
            steps = torch.arange(1, X.shape[1] + 1, dtype=z0.dtype).view(1, -1, 1, 1)
            return z0[:, None] + steps * delta[:, None]
        if kind == "linear":
            inc = (X @ W_lin)                               # [1,K,1024]
            return z0[:, None] + torch.cumsum(inc, dim=1)[:, :, None, :]
        raise KeyError(kind)

    def donor_inputs(don_name, t_d):
        key = (don_name, t_d)
        if key not in donor_cache:
            o = torch.load(os.path.join(EPS, f"{don_name}.v2ep.pt"), map_location="cpu",
                           weights_only=False)
            v = o["poses"][:, 3].float()
            kap = o["actions"][:, 0].float()
            c = ld._kin_actions(v, kap, t_d, 30)[None]       # [1,30,2]
            donor_cache[key] = (c, torch.tensor([float(v[2 * t_d])]))
        return donor_cache[key]

    donor_cache: dict = {}

    # ---------------------------------------------------------------- per episode
    ep_order = list(range(len(names))) if a.only is None else [a.only]
    t_job = time.time()
    n_done = 0
    for ei in ep_order:
        nm = names[ei]
        jpath = os.path.join(a.out_dir, "ep", f"{nm}.json")
        if os.path.exists(jpath):
            try:
                if json.load(open(jpath)).get("done"):
                    _p(f"[skip] {ei:02d} {nm} already done")
                    continue
            except Exception:
                pass
        t_ep = time.time()
        F, v_ep, kap_ep = ld._episode(nm)
        wins = sel_all[ei]
        ts = [t for (_, t) in wins]
        moms = {k: Mom() for k in (
            ["feat_tok", "feat_full", "adp_tok", "adp_full", "adp10_tok", "adp10_full",
             "tac0_tok", "tac0_full"]
            + [f"op{k}_{lv}" for k in KS for lv in ("tok", "full")]
            + [f"tacp{k}_{lv}" for k in KS for lv in ("tok", "full")])}
        win_out = []
        n_s40_seen = 0
        with torch.inference_mode():
            # ---- adapter latent, every window (batched; non-causal tmix stays window-local)
            fields_p = {}                 # S_10 windows: (field[:, -1], window feats batch idx)
            B = 8
            for b0 in range(0, len(ts), B):
                tb = ts[b0:b0 + B]
                x = torch.stack([F[t - ld.W + 1:t + 1].float() for t in tb])   # [b,W,N,d]
                xs = model.std(x)
                fld = model.encode(x)                                          # [b,W,N,d]
                for j, t in enumerate(tb):
                    z = fld[j, -1]                                             # [N,d]
                    zraw = xs[j, -1]
                    moms["adp_tok"].add(z.mean(0))
                    moms["adp_full"].add(z)
                    moms["feat_tok"].add(zraw.mean(0))
                    moms["feat_full"].add(zraw)
                    if (t - (ld.W - 1)) % STRIDE_P == 0:
                        moms["adp10_tok"].add(z.mean(0))
                        moms["adp10_full"].add(z)
                        fields_p[t] = fld[j].clone()                           # [W,N,d]
            # ---- baseline rollouts on S_10, swaps on S_40
            for (wi, t) in wins:
                if (t - (ld.W - 1)) % STRIDE_P != 0:
                    continue
                ld._order, ld._cursor = [wi], 0
                bt = ld.batch(1)
                act = bt["actions"]                                # [1,30,2]
                chk = ld._kin_actions(v_ep, kap_ep, t, 30)[None]
                assert torch.equal(act, chk), "own actions != _kin_actions"
                v0 = float(v_ep[2 * t])
                assert abs(float(bt["v0"][0]) - v0) < 1e-6
                v0_t = torch.tensor([v0])
                nav_t = bt["nav_cmd"]
                field_w = fields_p[t][None]                        # [1,W,N,d]
                last = model._last_state(field_w)                  # [1,N,d]
                pooled_win = field_w.mean(dim=-2)
                brains = model._run_brains(pooled_win, nav_t)
                intent = brains["intent"]
                tac0 = model._tac_field(last)                      # [1,64,d]
                X_own = aug(act, v0_t)                             # [1,30,3]
                P_op = roll_op(last, X_own, intent)                # [1,10,N,d]
                P_tac = roll_tac(tac0, X_own, intent)              # [1,10,64,d]
                moms["tac0_tok"].add(tac0[0].mean(0))
                moms["tac0_full"].add(tac0[0])
                for k in KS:
                    moms[f"op{k}_tok"].add(P_op[0, k - 1].mean(0))
                    moms[f"op{k}_full"].add(P_op[0, k - 1])
                    moms[f"tacp{k}_tok"].add(P_tac[0, k - 1].mean(0))
                    moms[f"tacp{k}_full"].add(P_tac[0, k - 1])
                rec = {"t": t, "v0": v0, "nav": int(nav_t[0]),
                       "z0n_op": float(last.double().norm()),
                       "z0n_tac": float(tac0.double().norm())}
                # T0 WM-vs-persistence MSE in the frozen target space (arm definition)
                tgt = model.std(bt["future_feats"][:, :K_OP])
                pe = model.to_enc(P_op)
                src = model.std(bt["feats"][:, -1:])
                rec["wm_mse"] = {
                    "model": {f"k{k}": float(((pe[:, k - 1] - tgt[:, k - 1]) ** 2).mean())
                              for k in KS},
                    "const": {f"k{k}": float(((src[:, 0] - tgt[:, k - 1]) ** 2).mean())
                              for k in KS},
                    "zero": {f"k{k}": float((tgt[:, k - 1] ** 2).mean()) for k in KS}}
                if (t - (ld.W - 1)) % STRIDE_S == 0:
                    # donor window
                    dn = int(perm[ei])
                    dname = names[dn]
                    cand = s40_t[dn]
                    assert cand, f"donor episode {dname} has no S_40 window"
                    t_d = min(cand, key=lambda x: (abs(x - t), x))
                    don_c, don_v0 = donor_inputs(dname, t_d)
                    rec["donor"] = {"name": dname, "ei": dn, "t": t_d,
                                    "v0": float(don_v0[0])}
                    rec["op"], rec["tac"], rec["pert"] = {}, {}, {}
                    rec["stub"] = {s: {} for s in ("identity", "drift", "linear")}
                    do_self_op = ((n_s40_seen % SELF_OP_EVERY) == 0)
                    n_s40_seen += 1
                    for var in VARIANTS:
                        X = make_variant(var, act, v0_t, don_c, don_v0)
                        dX = X - X_own
                        rec["pert"][var] = {
                            "op_rms": [float(dX[0, :K_OP, c].pow(2).mean().sqrt())
                                       for c in range(3)],
                            "tac_rms": [float(dX[0, ::stride_tac][:cfg.tac_steps, c]
                                              .pow(2).mean().sqrt()) for c in range(3)]}
                        if var != "self" or do_self_op:
                            Po = roll_op(last, X, intent)
                            rec["op"][var] = sens_pack(Po, P_op, last, enc=True)
                            if var == "self":
                                rec["op"][var]["maxabs"] = float(
                                    (Po - P_op).abs().max())
                                rec["op"][var]["bit_identical"] = bool(torch.equal(Po, P_op))
                        Pt = roll_tac(tac0, X, intent)
                        rec["tac"][var] = sens_pack(Pt, P_tac, tac0)
                        if var == "self":
                            rec["tac"][var]["bit_identical"] = bool(torch.equal(Pt, P_tac))
                        if var in STUB_VARIANTS:
                            for kind in ("identity", "drift", "linear"):
                                so = stub_roll(kind, last, X_own, delta_op, K_OP, False)
                                sx = stub_roll(kind, last, X, delta_op, K_OP, False)
                                st = stub_roll(kind, tac0, X_own, delta_tac, 10, True)
                                sy = stub_roll(kind, tac0, X, delta_tac, 10, True)
                                rec["stub"][kind][var] = {
                                    "op": sens_pack(sx, so, last),
                                    "tac": sens_pack(sy, st, tac0)}
                    for dn_, (ch_, amt_) in DOSES.items():
                        X = X_own.clone()
                        X[..., ch_] += amt_
                        dX = X - X_own
                        rec["pert"][dn_] = {
                            "op_rms": [float(dX[0, :K_OP, c].pow(2).mean().sqrt())
                                       for c in range(3)],
                            "tac_rms": [float(dX[0, ::stride_tac][:cfg.tac_steps, c]
                                              .pow(2).mean().sqrt()) for c in range(3)]}
                        rec["op"][dn_] = sens_pack(roll_op(last, X, intent), P_op, last, enc=True)
                        rec["tac"][dn_] = sens_pack(roll_tac(tac0, X, intent), P_tac, tac0)
                win_out.append(rec)
                _p(f"  [{ei:02d} {nm[:8]} t={t:3d}] "
                   f"{'S40 ' if 'donor' in rec else 'S10 '}"
                   f"{time.time() - t_ep:6.1f}s")
        # ---- persist: moments (work dir only) then the JSON commit marker
        mom_npz = {}
        rho_scalars = {}
        for k, m in moms.items():
            mom_npz[f"{k}__s"] = m.s
            mom_npz[f"{k}__ss"] = np.float64(m.ss)
            mom_npz[f"{k}__n"] = np.int64(m.n)
            rho_scalars[k] = {"n": m.n, "ss": m.ss, "within": m.within()}
        np.savez(os.path.join(a.out_dir, "ep", f"{nm}__moments.npz"), **mom_npz)
        out = {"done": True, "ei": ei, "name": nm, "donor_ei": int(perm[ei]),
               "n_windows_all": len(ts), "rho_scalars": rho_scalars,
               "windows": win_out, "wall_s": time.time() - t_ep}
        tmp = jpath + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, jpath)
        if a.repo_raw:
            with open(os.path.join(a.repo_raw, "ep", f"{nm}.json"), "w") as fh:
                json.dump(out, fh)
        n_done += 1
        _p(f"[done] {ei:02d}/{len(names)} {nm} windows={len(ts)} S10/S40="
           f"{len(win_out)}/{sum('donor' in r for r in win_out)} "
           f"{time.time() - t_ep:.0f}s  job={time.time() - t_job:.0f}s")

    # ------------------------------------------------ one-off tripwire (first window)
    trip = os.path.join(a.out_dir, "tripwire.json")
    if not os.path.exists(trip) and a.only in (None, 0):
        ei = 0
        nm = names[ei]
        F, v_ep, kap_ep = ld._episode(nm)
        wi, t = sel_all[ei][0]
        with torch.inference_mode():
            ld._order, ld._cursor = [wi], 0
            bt = ld.batch(1)
            v0_t = torch.tensor([float(v_ep[2 * t])])
            out = model(bt["feats"], bt["actions"][:, :30],
                        future_feats=bt["future_feats"][:, :30],
                        nav_cmd=bt["nav_cmd"], v0=v0_t)
            field_w = model.encode(bt["feats"])
            last = model._last_state(field_w)
            br = model._run_brains(field_w.mean(dim=-2), bt["nav_cmd"])
            X = aug(bt["actions"][:, :30], v0_t)
            P = roll_op(last, X, br["intent"])
            Pt = roll_tac(model._tac_field(last), X, br["intent"])
            d_op = float((P - out["op_pred"][:, :K_OP]).abs().max())
            d_tac = float((Pt - out["tac_pred"]).abs().max())
            scale = float(out["op_pred"][:, :K_OP].abs().max())
        with open(trip, "w") as fh:
            json.dump({"window": [nm, t], "max_abs_diff_op_vs_forward": d_op,
                       "max_abs_diff_tac_vs_forward": d_tac, "op_pred_absmax": scale}, fh)
        if a.repo_raw:
            import shutil
            shutil.copy(trip, os.path.join(a.repo_raw, "tripwire.json"))
        _p(f"[tripwire] op {d_op:.3e} tac {d_tac:.3e} (scale {scale:.2f})")
    _p(f"[ALL-DONE] processed {n_done} episodes in {time.time() - t_job:.0f}s")


if __name__ == "__main__":
    main()
