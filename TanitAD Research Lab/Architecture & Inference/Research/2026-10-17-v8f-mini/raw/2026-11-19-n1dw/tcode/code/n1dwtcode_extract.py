#!/usr/bin/env python3
"""n1dwtcode_extract.py -- PREREG_N1D_TCODE (sha256 4cec3917...): the PROBE-N1D-FEAT extractor (n1dwprobe_extract.py, unchanged captures) plus F_CODE, F_TTRUNK, F_CODE_LAT.

ORIGINAL DOCSTRING: PROBE-N1D-FEAT (PREREG_V8F_N1DW.md sec 10, sha256 59f2b7f9...): feature extractor.

Loads one banked N1D checkpoint on CPU, fp32, deterministic, builds the D1-HO windows exactly as the NAVRBD held-out dump does (``dump_rbd.sh``: the 0c3b374f tree's
``v8f_t1_eval_rbd.py`` machinery, ``--nav deadreckon --scene logged --schedule periodic --limit causal --nav-rb on --nav-rb-feat exitdist``), runs ONLY the NOW tick (tick 0 of the
T1-kin rollout, a_t = a_s = 0: nothing has been executed yet, so the dead-reckoned state IS the logged state at t0) through ``model.forward_infer`` and keeps, per window:

  F_TEXT  [N,15]  the nav_rb row the nav token receives at NOW:   ``v8f_mini.py:923-927`` (``_nav_tok``): ``self._at(b["nav_rb"], i)``  (input of ``e_nav_rb``, i = last_now)
  F_NAV   [N,d]   O's nav token after the ``e_nav_rb`` addition:    ``v8f_mini.py:926`` ``t = e_nav(nav) + e_nav_rb(nav_rb)``, the value ``_nav_tok`` RETURNS in its O call (``run_o``, ``v8f_mini.py:1041``)
  F_PLAN  [N,k]   the input of O's plan head at NOW:                ``v8f_mini.py:1048-1050`` ``h = self.read_o(cond, sc, pad).flatten(1)``; ``coef = self.head_o(h.float())`` -> the tensor ``h``
                  (captured by a forward-pre-hook on ``head_o``; this is the pooled O feature the trajectory is decoded from)
  F_VIS   [N,d]   mean over tokens of the NOW frame's adapted vision tokens: ``v8f_mini.py:1061`` ``adapt = self.adapter(feats) + self.pos`` ([B,F,n_tok,d]); ``adapt[:, Fn-1].mean(1)``
                  (Fn-1 = last_now, ``v8f_mini.py:1091``)
plus ``g`` (the GT 6 s track ``d["gt"][t0]``, the label's source), ``gt_ok``, ``ci`` (= episode id of the join, the dump's ``eid``), ``t0`` and ``plan0`` (the tick-0 plan, for the parity control).

usage: python n1dwprobe_extract.py --arm N1D_s0 --out DIR [--max-windows N] [--tag T]
Everything is read only; GPU never used (CUDA_VISIBLE_DEVICES is blanked here before torch is imported)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

TREE = os.environ.get("N1DWPROBE_TREE", "/root/strat1/navrbd/tree")     # verified blob-by-blob against 0c3b374f (blobs_0c3b374f.txt, 2630/2630)
for p in (TREE + "/stack/scripts", TREE + "/stack", TREE + "/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import v8f_t1_eval_rbd as RBD  # noqa: E402  (imports v8f_t1_eval_rb as R, v8f_t1_eval as T)

R, T = RBD.R, RBD.T
D, M = T.D, T.M

RUNS = os.environ.get("N1DWPROBE_RUNS", "/root/strat1/navrbd/runs")
STEP0 = "/root/strat1/navrbd_step0"
INP = STEP0 + "/out/inputs"
TRACKS = "/root/dscale10k/in/labels/native_tracks_datascale10k.pt"
JOIN = "/root/dscale10k/merged/join_X.npz"
CACHE = STEP0 + "/cache_view"
RBD_NPZ = "/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz"
GRID = INP + "/eval_grid_x_feasible.json"
NAV_NPZ = INP + "/navtext_nav_x.npz"
BATCH = 32


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Capture:
    """Records, per forward_infer call, the four tensors. Installed as wrappers/hooks on ONE model; no output of the model is changed."""

    def __init__(self, model):
        self.model = model
        self.nav_calls = []                      # (text_in, token) per _nav_tok call; run_t calls first, run_o second
        self.plan_in = None
        self.adapt = None
        self.ttrunk = []                         # TCODE: input of h_lat = h = gelu(trunk_t(read_t(...)))  (v8f_mini.py run_t, line 1026)
        self.code_pred = []                      # TCODE: _code_pred(tout), the value _forward puts in out["code"] when no override is active
        orig_cp = model._code_pred

        def code_pred(o):
            c = orig_cp(o)
            self.code_pred.append(c.detach().float().clone())
            return c
        model._code_pred = code_pred
        model.h_lat.register_forward_pre_hook(lambda mod, args: self.ttrunk.append(args[0].detach().float().clone()))
        orig_nav = model._nav_tok
        orig_enc = model._encode

        def nav_tok(b, i):
            tok = orig_nav(b, i)
            self.nav_calls.append((model._at(b["nav_rb"], i).detach().float().clone(), tok.detach().float().clone()))
            return tok

        def encode(b):
            a = orig_enc(b)
            self.adapt = a.detach().float().clone()
            return a
        model._nav_tok, model._encode = nav_tok, encode
        model.head_o.register_forward_pre_hook(lambda mod, args: setattr(self, "plan_in", args[0].detach().float().clone()))

    def reset(self):
        self.nav_calls, self.plan_in, self.adapt = [], None, None
        self.ttrunk, self.code_pred = [], []

    def take(self):
        assert len(self.nav_calls) == 3, f"expected run_s + run_t + run_o nav-token calls (v8f_mini.py:1001,1019,1045), got {len(self.nav_calls)}"
        text, tok = self.nav_calls[-1]               # the O call (run_s, run_t, run_o in _forward order)
        assert self.plan_in is not None and self.adapt is not None
        vis = self.adapt[:, -1].mean(1)              # NOW frame = index Fn-1 (last_now) of [B, F, n_tok, d]
        assert len(self.ttrunk) == 1 and len(self.code_pred) == 1, (len(self.ttrunk), len(self.code_pred))
        return text, tok, self.plan_in, vis, self.ttrunk[0], self.code_pred[0]


def run(arm, out_dir, max_windows=0, tag=None, repeat=False):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if R.tool_sha256() != R.TOOL_SHA256:
        raise SystemExit("REFUSED: the T1 tool differs from its pin")
    ckp = f"{RUNS}/{arm}/train/ckpt_last.pt"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    assert cfg.nav_rb == "on" and cfg.nav_rb_feat == "exitdist" and cfg.nav_source == "text" and cfg.kin_mode == "yaw", (cfg.nav_rb, cfg.nav_rb_feat, cfg.nav_source, cfg.kin_mode)
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.to(torch.device("cpu")).eval()
    # TCODE: derive the F-CODE-LAT columns from the SOURCE (never by hand) and print them
    lat_cols = list(range(M.N_GOAL, M.N_GOAL + M.N_LAT))
    c0 = M.N_GOAL + M.N_LAT + M.N_LON
    turn_names = M.CONS_NAMES[M.CI["turn_t_start_s"]:M.CI["turn_exit_psi_deg"] + 1]
    turn_cols = [c0 + M.CI[n] for n in turn_names]
    code_lat_cols = lat_cols + turn_cols
    assert M.CODE_DIM == 60 and len(turn_names) == 9 and len(code_lat_cols) == 14, (M.CODE_DIM, turn_names)
    print("F-CODE-LAT columns:", {"lateral_softmax": lat_cols, "turn_constraints": dict(zip(turn_names, turn_cols))}, "CODE_DIM", M.CODE_DIM, "N_GOAL/LAT/LON/CONS", M.N_GOAL, M.N_LAT, M.N_LON, M.N_CONS, flush=True)
    # TCODE: the code O reads at eval is T's own (no teacher forcing, no mix, no override, no cut): asserted from the config and from the call that forward_infer makes
    assert cfg.o_interface == "full", cfg.o_interface
    import inspect
    assert "self._forward(g, 0.0, False)" in " ".join(inspect.getsource(model.forward_infer).split()), "forward_infer no longer calls _forward(g, 0.0, False): teacher forcing / override would be reachable"
    cap = Capture(model)
    navsha, rbsha = sha256_file(NAV_NPZ), sha256_file(RBD_NPZ)
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(JOIN)
    J["nav"], J["nav_d"], _ = D.nav_text_arrays(J, NAV_NPZ, navsha, "eval")
    J[R.RB_KEY], _ = D.nav_rb_arrays(J, RBD_NPZ, rbsha, "eval", "real", "exitdist")
    store = D.ClipStore(CACHE, "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(GRID, J)
    n_all = len(wins)
    if max_windows and max_windows < len(wins):
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, max_windows).round().astype(int)]
    proxy = T._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    class A:                                                   # the namespace make_states_factory reads
        nav_rb, nav_rb_variant = "on", "follow"
    make_states = RBD.make_states_factory(T.make_states, A)
    build_item = R.build_item_factory(T.build_item)
    keys = frozenset(T.MODEL_KEYS) | {R.RB_KEY}
    a_t, a_s = T.ages_for(0, "periodic")
    assert (a_t, a_s) == (0, 0)
    rows = {k: [] for k in ("F_TEXT", "F_NAV", "F_PLAN", "F_VIS", "F_CODE", "F_TTRUNK", "F_CODE_LAT", "g", "gt_ok", "plan0", "ci", "t0")}
    t_start = time.time()
    for b0 in range(0, len(wins), BATCH):
        chunk = wins[b0:b0 + BATCH]
        sts = make_states(store, tracks, J, chunk, "deadreckon", "real", None)
        items = [build_item(ds, proxy, st, 0, a_t, a_s, "logged", "causal", None) for st in sts]
        full = D.collate(items)
        batch = {k: full[k] for k in keys}
        cap.reset()
        with torch.no_grad():
            out = model.forward_infer(batch)
        text, tok, plan_in, vis, ttrunk, code_pred = cap.take()
        assert torch.equal(out["code"].float(), code_pred), "out['code'] is not T's own _code_pred: an override / mix is active"
        # the card's F-TEXT: the rows the token receives == the batch's last history row
        assert torch.equal(text, batch[R.RB_KEY][:, -1].float()), "F-TEXT is not the NOW row of nav_rb"
        rows["F_TEXT"].append(text.numpy()); rows["F_NAV"].append(tok.numpy()); rows["F_PLAN"].append(plan_in.numpy()); rows["F_VIS"].append(vis.numpy())
        rows["F_CODE"].append(out["code"].float().cpu().numpy()); rows["F_TTRUNK"].append(ttrunk.numpy()); rows["F_CODE_LAT"].append(out["code"].float().cpu().numpy()[:, code_lat_cols])
        rows["plan0"].append(out["traj30"].float().cpu().numpy())
        for ci, t0, _, _ in chunk:
            d = store.get(str(J["clip"][ci]))
            rows["g"].append(np.asarray(d["gt"][t0], np.float64))
            rows["gt_ok"].append(bool(d["gt_ok"][t0]) if "gt_ok" in d else True)
            rows["ci"].append(int(ci)); rows["t0"].append(int(t0))
        if (b0 // BATCH) % 20 == 0:
            print(f"[{b0 + len(chunk)}/{len(wins)}] {time.time() - t_start:.0f}s", flush=True)
    arr = {k: np.concatenate(v) if k in ("F_TEXT", "F_NAV", "F_PLAN", "F_VIS", "F_CODE", "F_TTRUNK", "F_CODE_LAT", "plan0") else np.asarray(v) for k, v in rows.items()}
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{out_dir}/feat_{tag or arm}.npz"
    np.savez(fn, **arr)
    meta = {"arm": arm, "ckpt_sha256": sha256_file(ckp), "n_windows": int(len(wins)), "n_grid": int(n_all), "n_episodes": int(len(set(arr["ci"].tolist()))),
            "dims": {k: list(arr[k].shape[1:]) for k in ("F_TEXT", "F_NAV", "F_PLAN", "F_VIS", "F_CODE", "F_TTRUNK", "F_CODE_LAT")}, "code_lat_cols": code_lat_cols, "code_lat_names": ["lat_softmax_%d" % i for i in range(M.N_LAT)] + list(turn_names), "gt_ok_false": int((~arr["gt_ok"]).sum()),
            "inputs_sha256": {"grid": sha256_file(GRID), "join": sha256_file(JOIN), "navtext": navsha, "nav_rb": rbsha, "tracks": sha256_file(TRACKS)},
            "torch": torch.__version__, "threads": torch.get_num_threads(), "feat_file_sha256": sha256_file(fn)}
    json.dump(meta, open(fn + ".json", "w"), indent=1)
    print("wrote", fn, json.dumps(meta["dims"]), "elapsed", f"{time.time() - t_start:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    run(a.arm, a.out, a.max_windows, a.tag)


if __name__ == "__main__":
    main()
