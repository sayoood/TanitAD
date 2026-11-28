#!/usr/bin/env python3
"""v8f_t1_speedsub.py -- seam B of PREREG_V8F_SPEEDSUB_T1 section 3: the speed-cell substitution wrapper around the PINNED T1-kin tool (`v8f_t1_eval.py`, sha256 61e140ed...).

The tool is NOT edited. This wrapper patches `V8FMini` at CLASS level, then calls the tool's own `main()` with the banked argv, so the rollout, the executor, the floor arm, the dumps and the
analysis are the tool's. Three class-level patches, all inert outside a `forward_infer` call made by this wrapper:
  forward_infer  opens a per-call context (the batch, the tick, the window ids) around the ORIGINAL forward_infer and closes it;
  _code_pred     the ONE place T's code is built: inside the context it returns T's code with ONLY the five speed cells (code indices [46, 56, 57, 58, 59]) replaced; the rest is T's own tensor.
                 One forward pass: O reads the substituted code in the same call (`forward_diag`'s semantics, without its second pass; pinned bit for bit by a test);
  run_o          a pass-through RECORDER of the `code` argument O receives, so K-NONSPEED / K-REC are measured on what O READS, not on what the substitution intended.
The P10 floor arm (`floor_planner`) calls `model.prior` only and reaches none of the three.

MODES (PREREG section 3; `--mode` is this wrapper's flag, everything else is the tool's argv):
  identity    --limit causal   O reads T's own cells (the paired base of every contrast; control K-ID; bit-equal to the unpatched tool)
  egoc        --limit causal   O reads `mlp_ego_c(ego block of the tick item)` standardised by the checkpoint's cons_mu / cons_sd                       <- THE ARM
  egoshufc    --limit causal   O reads the egoc run's recorded cells of window perm[w] at the SAME tick, perm = default_rng(20261029).permutation(2399)  (deliberate regression)
  identity_l  --limit logged   beside: identity in the logged lane
  egol        --limit logged   beside: the banked `mlp_ego` (a205609a...) in the logged lane
A mode in the wrong limit lane, a missing / mismatched read-out, a shuffle source from another run, an `o_interface = cut` checkpoint (the substitution could not reach O), a non-CPU device and a
missing `--dump-ticks` are REFUSED before the first window.

RECORD (beside the tool's ticks file): `<dump-ticks>/speedsub_<tag>.npz` (+ `.json`), one row per (window, tick) in the order the tool wrote its tick rows: win, tick, age_t, v0_last, ego [R, 22],
own_std [R, 5] (T's own cells), used_std [R, 5] (what the substitution produced), o_std [R, 5] (the cells O READ), nonspeed_dev [R] (max |O's code - T's code| over the 55 other dims), src_win [R].
After the tool returns, the rows are checked against the tool's own ticks file (win, tick, age_t, v0): a mis-aligned record is rc 3.

    python v8f_t1_speedsub.py --mode egoc --readout <readout_c.pt> --readout-sha <sha> --expect-ckpt-sha <sha> run --ckpt ... --tag ... [the banked argv]
    python v8f_t1_speedsub.py pin --out pins.json ...
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SCHEMA = "v8f-speedsub-t1-rec-v1"
TOOL_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"
MODES = ("identity", "egoc", "egoshufc", "identity_l", "egol")
MODE_LIMIT = {"identity": "causal", "egoc": "causal", "egoshufc": "causal", "identity_l": "logged", "egol": "logged"}
SUBSTITUTING = ("egoc", "egoshufc", "egol")
READOUT_KEY = {"egoc": "mlp_ego_c", "egoshufc": "mlp_ego_c", "egol": "mlp_ego"}
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
SPEED_CODE_IDX = (46, 56, 57, 58, 59)
SHUF_SEED, SHUF_N = 20261029, 2399
N_CELLS, D_EGO = 5, 22
HORIZON = 30


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Refused(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")
        self.msg = msg


def registered_perm(n=SHUF_N, seed=SHUF_SEED):
    return np.random.default_rng(seed).permutation(n)


def perm_sha256(perm):
    return hashlib.sha256(np.asarray(perm).astype(np.int64).tobytes()).hexdigest()


def source_grid(rec, n_windows, n_ticks):
    """The egoshufc source: the egoc run's recorded `used_std` as a [n_windows, n_ticks, 5] grid. Every (window, tick) must be present exactly once."""
    win, tick, used = np.asarray(rec["win"]), np.asarray(rec["tick"]), np.asarray(rec["used_std"])
    if len(win) != n_windows * n_ticks:
        raise Refused(f"shuffle source holds {len(win)} rows, a full grid is {n_windows} x {n_ticks}")
    g = np.full((n_windows, n_ticks, N_CELLS), np.nan, np.float32)
    seen = np.zeros((n_windows, n_ticks), bool)
    for w, t, u in zip(win, tick, used):
        if not (0 <= w < n_windows and 0 <= t < n_ticks) or seen[w, t]:
            raise Refused(f"shuffle source: (window {w}, tick {t}) is out of the grid or repeated")
        g[w, t], seen[w, t] = u, True
    if not seen.all() or not np.isfinite(g).all():
        raise Refused("shuffle source: the grid is incomplete or holds non-finite cells")
    return g


class SpeedSubHook:
    """The class-level patches and their per-call state. One instance per rollout process."""

    def __init__(self, M, sr, mode, readout=None, src_grid=None, perm=None, n_ticks=3, idx=SPEED_CODE_IDX):
        if mode not in MODES:
            raise Refused(f"unknown mode {mode!r}; one of {MODES}")
        if mode in SUBSTITUTING and readout is None:
            raise Refused(f"mode {mode!r} needs a read-out")
        if mode == "egoshufc" and (src_grid is None or perm is None):
            raise Refused("mode 'egoshufc' needs the egoc run's recorded cells and the registered permutation")
        self.M, self.sr, self.mode, self.readout, self.idx = M, sr, mode, readout, tuple(idx)
        self.src, self.perm, self.n_ticks = src_grid, perm, int(n_ticks)
        self.non = [i for i in range(M.CODE_DIM) if i not in self.idx]
        self.ctx, self.installed, self.orig = None, False, {}
        self.k, self.base, self.chunk_b, self.rows = 0, 0, None, {k: [] for k in ("win", "tick", "age_t", "v0_last", "ego", "own_std", "used_std", "o_std", "nonspeed_dev", "src_win")}
        self.nonspeed_dev_max, self.used_dev_max = 0.0, 0.0
        self.mu, self.sd = None, None

    # ---- install / restore -------------------------------------------------------------------------------------
    def install(self):
        if self.installed or getattr(self.M.V8FMini, "_speedsub_installed", False):
            raise Refused("the V8FMini class is already patched")
        V, hook = self.M.V8FMini, self
        self.orig = {"forward_infer": V.forward_infer, "_code_pred": V._code_pred, "run_o": V.run_o}

        def forward_infer(model, batch):
            hook._begin(model, batch)
            try:
                out = hook.orig["forward_infer"](model, batch)
            finally:
                ctx, hook.ctx = hook.ctx, None
            hook._finish(ctx)
            return out

        def _code_pred(model, out):
            code = hook.orig["_code_pred"](model, out)
            return code if hook.ctx is None else hook._substitute(model, code)

        def run_o(model, b, adapt, last_o, ck_o, ck_ok, code):
            if hook.ctx is not None:
                hook.ctx["o_code"] = code.detach().clone()
            return hook.orig["run_o"](model, b, adapt, last_o, ck_o, ck_ok, code)

        V.forward_infer, V._code_pred, V.run_o = forward_infer, _code_pred, run_o
        V._speedsub_installed = True
        self.installed = True
        return self

    def restore(self):
        if not self.installed:
            return
        V = self.M.V8FMini
        V.forward_infer, V._code_pred, V.run_o = self.orig["forward_infer"], self.orig["_code_pred"], self.orig["run_o"]
        del V._speedsub_installed
        self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.restore()

    # ---- per call ----------------------------------------------------------------------------------------------
    def _begin(self, model, batch):
        if self.mode in SUBSTITUTING and getattr(model.cfg, "o_interface", "full") == "cut":
            raise Refused("the checkpoint's o_interface is 'cut': O reads a constant code, a substitution cannot reach it")
        if self.mu is None:
            ci = [self.M.CI[n] for n in SPEED_CELLS]
            self.mu, self.sd = model.cons_mu[ci].double().cpu().numpy(), model.cons_sd[ci].double().cpu().numpy()
        B = int(batch["p10f"].shape[0])
        tick = self.k % self.n_ticks
        if tick == 0:
            if self.k > 0:
                self.base += self.chunk_b
            self.chunk_b = B
        elif B != self.chunk_b:
            raise RuntimeError(f"call {self.k}: batch of {B} windows inside a chunk of {self.chunk_b}: the rollout structure is not (chunks x {self.n_ticks} ticks)")
        self.ctx = {"batch": batch, "tick": tick, "win": self.base + np.arange(B), "B": B}
        self.k += 1

    def _substitute(self, model, code):
        """Called from `_code_pred` inside the context: -> the code O is handed."""
        import torch
        ctx, sr, idx = self.ctx, self.sr, list(self.idx)
        ctx["own_code"] = code.detach().clone()
        ctx["ego"] = sr.ego_vectors(ctx["batch"])
        ctx["src_win"] = ctx["win"]
        if self.mode in ("identity", "identity_l"):
            return code                                                  # the ORIGINAL tensor: bit-equal to the unpatched tool by construction
        mu, sd = self.mu, self.sd
        if self.mode == "egoshufc":
            ctx["src_win"] = self.perm[ctx["win"]]
            std = self.src[ctx["src_win"], ctx["tick"]].astype(np.float64)
        else:
            std = sr.phys_to_std(self.readout.predict_ego_batch(ctx["batch"]), mu, sd)
        new = code.clone()
        new[:, idx] = torch.as_tensor(np.asarray(std), dtype=code.dtype, device=code.device)
        ctx["used_std"] = new[:, idx].detach().float().cpu().numpy()
        return new

    def _finish(self, ctx):
        if ctx is None or "own_code" not in ctx or "o_code" not in ctx:
            raise RuntimeError("forward_infer ran without T's code or O's input: the patched hooks were not reached")
        idx, own, o = list(self.idx), ctx["own_code"], ctx["o_code"]
        non_dev = (o[:, self.non] - own[:, self.non]).abs().amax(1).float().cpu().numpy()
        own_std, o_std = own[:, idx].float().cpu().numpy(), o[:, idx].float().cpu().numpy()
        used = ctx.get("used_std", own_std)
        r = self.rows
        r["win"].append(ctx["win"]); r["tick"].append(np.full(ctx["B"], ctx["tick"])); r["ego"].append(ctx["ego"].astype(np.float32))
        r["age_t"].append(ctx["batch"]["a_t"].cpu().numpy().reshape(-1))
        r["v0_last"].append(ctx["batch"]["p10f"][:, -1, 0].double().cpu().numpy())
        r["own_std"].append(own_std); r["used_std"].append(used); r["o_std"].append(o_std); r["nonspeed_dev"].append(non_dev); r["src_win"].append(ctx["src_win"])
        self.nonspeed_dev_max = max(self.nonspeed_dev_max, float(non_dev.max()))
        self.used_dev_max = max(self.used_dev_max, float(np.abs(o_std - used).max()))

    def arrays(self):
        out = {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in self.rows.items()}
        return out


def check_alignment(arrays, tz, n_ticks, n_windows=None):
    """The wrapper's rows against the TOOL's own tick rows (same order: chunk, tick, window): win, tick, age_t and the measured speed v0 must be equal. -> (ok, detail)."""
    det = {"rows": int(len(arrays["win"])), "ticks_file_rows": int(len(tz["win"]))}
    ok = det["rows"] == det["ticks_file_rows"] and det["rows"] > 0
    if n_windows is not None:
        det["expected_rows"] = int(n_windows) * int(n_ticks)
        ok = ok and det["rows"] == det["expected_rows"]
    if ok:
        det["win_equal"] = bool(np.array_equal(arrays["win"], tz["win"]))
        det["tick_equal"] = bool(np.array_equal(arrays["tick"], tz["tick"]))
        det["age_t_equal"] = bool(np.array_equal(arrays["age_t"], tz["age_t"]))
        det["v0_equal"] = bool(np.array_equal(arrays["v0_last"], tz["v0"]))
        ok = all(det[k] for k in ("win_equal", "tick_equal", "age_t_equal", "v0_equal"))
    return bool(ok), det


# ------------------------------------------------------------------------------------------------ the CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--readout", default="", help="the read-out container (readout_c.pt, or the banked readout.pt for egol)")
    ap.add_argument("--readout-sha", default="", help="its pinned sha256 (REFUSED on any difference)")
    ap.add_argument("--expect-ckpt-sha", default="", help="the checkpoint's pinned sha256")
    ap.add_argument("--shuffle-source", default="", help="egoshufc: the egoc run's speedsub_<tag>.npz")
    ap.add_argument("--tool-root", default=os.path.abspath(os.path.join(HERE, "..", "..")).replace("\\", "/"))
    ap.add_argument("--instr", default=f"{HERE}/v8f_mini_eval_speedread.py")
    ap.add_argument("--instr-sha", default="")
    ap.add_argument("--tool-sha", default=TOOL_SHA256)
    ap.add_argument("--perm-n", type=int, default=SHUF_N, help="egoshufc: the permutation length. 2,399 is the REGISTERED one; another value exists for the synthetic tests only and can never pass the reader's K-SHUF")
    return ap


def load_tool(root, expect_sha):
    p = f"{root}/stack/scripts/v8f_t1_eval.py"
    got = sha256_file(p)
    if expect_sha and got != expect_sha:
        raise Refused(f"the T1 tool {p} has sha256 {got[:12]}..., the pin is {expect_sha[:12]}...")
    return load_module(p, "v8f_t1_eval_pinned"), got


def run(argv):
    wa, rest = build_parser().parse_known_args(argv)
    t_start = time.time()
    tool, tool_sha = load_tool(wa.tool_root, wa.tool_sha)
    ta = tool.build_parser().parse_args(rest)
    if ta.mode != "run":
        raise Refused("this wrapper wraps the tool's `run` mode only")
    mode = wa.mode
    if MODE_LIMIT[mode] != ta.limit:
        raise Refused(f"mode {mode!r} is the --limit {MODE_LIMIT[mode]} lane, the tool argv says --limit {ta.limit}")
    if not ta.dump_ticks:
        raise Refused("--dump-ticks is required: the record is checked against the tool's own tick rows")
    if str(ta.device) != "cpu":
        raise Refused(f"--device {ta.device!r}: the registered rollouts are CPU (bit-comparable with the banked ones)")
    if mode == "egoshufc" and ta.max_windows and wa.perm_n == SHUF_N:
        raise Refused("egoshufc uses the registered permutation of 2,399 windows: no --max-windows")
    n_ticks = tool.H // ta.replan_steps
    sr = load_module(wa.instr, "sr_instr_wrapper")
    if wa.instr_sha and sha256_file(wa.instr) != wa.instr_sha:
        raise Refused(f"the SPEEDREAD instrument {wa.instr} is not the pinned one")
    M = tool.M
    cells_ci = [M.CI[n] for n in SPEED_CELLS]
    if tuple(34 + c for c in cells_ci) != SPEED_CODE_IDX or 34 != M.N_GOAL + M.N_LAT + M.N_LON:
        raise Refused(f"the speed cells are not at code indices {SPEED_CODE_IDX} in this model")
    if not set(sr.EGO_READ_KEYS) <= set(tool.MODEL_KEYS):
        raise Refused(f"the read-out reads {sorted(set(sr.EGO_READ_KEYS) - set(tool.MODEL_KEYS))}, outside the tool's model keys")
    readout, readout_sha = None, None
    if mode in SUBSTITUTING:
        if not wa.readout or not os.path.isfile(wa.readout):
            raise Refused(f"mode {mode!r} needs --readout (found {wa.readout!r})")
        readout_sha = sha256_file(wa.readout)
        if not wa.readout_sha or readout_sha != wa.readout_sha:
            raise Refused(f"read-out sha256 {readout_sha[:12]}... is not the pinned {wa.readout_sha[:12] or '(none given)'}")
        ros = sr.load_readouts(wa.readout)
        if READOUT_KEY[mode] not in ros:
            raise Refused(f"{wa.readout} holds {sorted(ros)}, mode {mode!r} reads {READOUT_KEY[mode]!r}")
        readout = ros[READOUT_KEY[mode]]
        if not readout.ego_only or readout.kind != "mlp_ego" or len(readout.xm) != D_EGO:
            raise Refused("the read-out is not an ego-only 22-column MLP")
    ckpt_sha = sha256_file(ta.ckpt)
    if wa.expect_ckpt_sha and ckpt_sha != wa.expect_ckpt_sha:
        raise Refused(f"checkpoint sha256 {ckpt_sha[:12]}... is not the pinned {wa.expect_ckpt_sha[:12]}...")
    src_grid, perm, perm_sha, src_meta = None, None, None, None
    if mode == "egoshufc":
        if not wa.shuffle_source or not os.path.isfile(wa.shuffle_source):
            raise Refused("egoshufc needs --shuffle-source (the egoc run's speedsub_<tag>.npz)")
        srec = np.load(wa.shuffle_source, allow_pickle=False)
        src_meta = json.loads(str(srec["meta_json"]))
        if src_meta.get("mode") != "egoc" or src_meta.get("ckpt_sha256") != ckpt_sha or src_meta.get("readout_sha256") != readout_sha:
            raise Refused("the shuffle source is not an egoc rollout of THIS checkpoint with THIS read-out")
        if int(src_meta.get("n_windows", -1)) != wa.perm_n:
            raise Refused(f"the shuffle source holds {src_meta.get('n_windows')} windows, the permutation has {wa.perm_n}")
        src_grid = source_grid({k: srec[k] for k in ("win", "tick", "used_std")}, wa.perm_n, n_ticks)
        perm = registered_perm(wa.perm_n)
        perm_sha = perm_sha256(perm)
    hook = SpeedSubHook(M, sr, mode, readout, src_grid, perm, n_ticks)
    with hook:
        rc = tool.main(rest)
    arrays = hook.arrays()
    tick_path = os.path.join(ta.dump_ticks, f"ticks_{ta.tag}.npz")
    tz = np.load(tick_path, allow_pickle=True)
    n_win = int(tz["n_windows"])
    ok, align = check_alignment(arrays, tz, n_ticks, n_win)
    if mode == "egoshufc":
        align["n_windows_equals_perm_n"] = n_win == wa.perm_n
        ok = ok and align["n_windows_equals_perm_n"]
    meta = {"schema": SCHEMA, "mode": mode, "tag": ta.tag, "limit": ta.limit, "replan_steps": ta.replan_steps, "n_ticks": n_ticks, "n_windows": n_win, "ckpt": ta.ckpt, "ckpt_sha256": ckpt_sha,
            "readout": wa.readout or None, "readout_sha256": readout_sha, "readout_key": READOUT_KEY.get(mode), "tool_sha256": tool_sha, "wrapper_sha256": sha256_file(os.path.abspath(__file__)),
            "instrument_sha256": sha256_file(wa.instr), "code_idx": list(SPEED_CODE_IDX), "cons_idx": cells_ci,
            "cons_mu": [float(v) for v in hook.mu], "cons_sd": [float(v) for v in hook.sd],
            "nonspeed_dev_max": hook.nonspeed_dev_max, "o_vs_used_dev_max": hook.used_dev_max, "n_calls": hook.k, "alignment_with_tool_ticks": align, "aligned": bool(ok),
            "perm_sha256": perm_sha, "perm_seed": SHUF_SEED if mode == "egoshufc" else None, "perm_n": len(perm) if perm is not None else None, "perm_n_fixed": int((perm == np.arange(len(perm))).sum()) if perm is not None else None,
            "shuffle_source": wa.shuffle_source or None, "shuffle_source_tag": (src_meta or {}).get("tag"), "tool_rc": int(rc), "wall_s": round(time.time() - t_start, 1)}
    os.makedirs(ta.dump_ticks, exist_ok=True)
    stem = os.path.join(ta.dump_ticks, f"speedsub_{ta.tag}")
    np.savez(stem + ".npz", meta_json=np.asarray(json.dumps(meta)), **arrays)
    json.dump(meta, open(stem + ".json", "w", encoding="utf-8", newline="\n"), indent=1)
    if not ok:
        print("ZZSPEEDSUB-T1-WRAP-STOP record not aligned with the tool's tick rows", align, flush=True)
        return 3
    print(f"ZZSPEEDSUB-T1-WRAP-DONE {mode} {ta.tag} rows={align['rows']} nonspeed_dev_max={hook.nonspeed_dev_max} wall={meta['wall_s']}s", flush=True)
    return int(rc)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
