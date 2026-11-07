"""town_run: the T1-kin rollout of PREREG_V8F_TOWN.md with the TWIN, a THIN WRAPPER around the pinned tool of record `v8f_t1_eval_ri.py` (sha256 0417c838...).
The tool is loaded BY FILE PATH after its sha256 holds and is not edited. The wrapper patches module names in the loaded object, and nothing else:
  1. `model_planner` / `rollout`: capture the model, the window dataset `ds` and the clip proxy (read-only handles; the originals are called unchanged).
  2. `TickRecorder` -> `TownRecorder` (a subclass): after the original hook has recorded its tick row (the tick file is UNCHANGED bit for bit), it evaluates T a second time on the LOGGED ego state at the same
     time index j and records both T outputs, the speeds, the GT labels of T's frame, the receipts. The twin batch = the own batch with ONLY `SWAP_KEYS` replaced by the logged clip's values; `KEEP_KEYS` are the same tensors.
     The twin is one extra forward_infer call; it reads the states and writes none of them, so the rollout is untouched (control G-ID).
  3. `--zero-t-ego` (the deliberate-regression arm): `V8FMini.run_t` is wrapped so that, for the T call only, `e_ego.weight` and `e_ego.bias` are 0 (restored in `finally`).
`--twin-mode mutant` (control mutant): the twin batch is NOT swapped (twin := own); the twin controls must then FAIL.
Everything else is the tool's `run` CLI: `town_run.py [--twin-mode real|mutant] [--zero-t-ego] --receipt-dir D -- <tool run flags>`. Requires `--dump-ticks --dump-heads --limit causal`.
Tier T1-kin for the own rows, T0 for the twin rows. CPU only."""
import os
import sys
import json
import hashlib
import argparse
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

TOOL = "F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py"
TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
SWAP_KEYS = ("p10f", "base", "pose_now", "pose_t", "pose_s", "limit", "limit_ok")     # the ego-derived inputs of the model
KEEP_KEYS = ("feats", "frame_ok", "nav", "a_t", "a_s")                                  # frames, route countdown, hold ages: the same tensors in the twin
LAT_N, LON_N, CONS_N, GOAL_N = 5, 7, 26, 22


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_tool():
    got = sha256_file(TOOL)
    if got != TOOL_SHA:
        raise SystemExit(f"REFUSED: the tool is {got}, not the pinned {TOOL_SHA}")
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri_pinned_town", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri_pinned_town"] = mod
    spec.loader.exec_module(mod)
    return mod


class Shared:
    model = ds = proxy = None
    counters = {"twin_calls": 0, "twin_rows": 0, "tick0_rows": 0, "tick0_swap_unequal": 0, "later_rows": 0, "later_swap_unequal": 0, "logged_p10f_mismatch": 0,
                "keep_key_not_same_object": 0, "run_t_zeroed_calls": 0, "limit_bucket_differs": 0, "mode": None, "zero_t_ego": False}


def twin_batch(tool, S, batch, states, mutant):
    """-> (twin batch, items). `items` = the logged items (their `lab_t` is the GT of T's frame)."""
    items = []
    prev = S.proxy.d
    for b, st in enumerate(states):
        S.proxy.d = st.base_log                                           # the LOGGED clip: rows <= j are the human's
        it = S.ds.item(st.ci, st.j, int(batch["a_t"][b]), int(batch["a_s"][b]))
        it["limit"], it["limit_ok"] = tool.causal_bucket(st.base_log["v"], st.j, tool.CAUSAL_STEPS)
        items.append(it)
    S.proxy.d = prev
    full = tool.D.collate(items)
    tb = dict(batch)
    if not mutant:
        for k in SWAP_KEYS:
            tb[k] = full[k].to(batch[k].device)
    return tb, items


def make_recorder(tool, S, mutant):
    import torch

    class TownRecorder(tool.TickRecorder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.tw = {}

        def _put(self, k, v):
            self.tw.setdefault(k, []).append(v)

        def __call__(self, n, batch, plan, states):
            super().__call__(n, batch, plan, states)                       # the tool's own tick row, first
            hd = self.planner.heads
            tb, items = twin_batch(tool, S, batch, states, mutant)
            with torch.no_grad():
                out = S.model.forward_infer(tb)
            C = S.counters
            C["twin_calls"] += 1
            f = lambda t: t.float().cpu().numpy()
            tw = {k: f(out[k]) for k in ("lat_logits", "lon_logits", "cons_pred", "goal_prob")}
            for b, st in enumerate(states):
                C["twin_rows"] += 1
                a_t = int(batch["a_t"][b])
                eq = all(bool(torch.equal(batch[k][b], tb[k][b])) for k in SWAP_KEYS)
                if n == 0:
                    C["tick0_rows"] += 1
                    C["tick0_swap_unequal"] += int(not eq)
                else:
                    C["later_rows"] += 1
                    C["later_swap_unequal"] += int(not eq)
                C["keep_key_not_same_object"] += sum(int(tb[k] is not batch[k]) for k in KEEP_KEYS)
                lp = np.nan_to_num(np.asarray(st.base_log["p10f"][st.j]), nan=0.0).astype(np.float32)
                C["logged_p10f_mismatch"] += int(not mutant and not np.array_equal(tb["p10f"][b, -1].numpy(), lp))
                C["limit_bucket_differs"] += int(not torch.equal(batch["limit"][b], tb["limit"][b]) or bool(batch["limit_ok"][b]) != bool(tb["limit_ok"][b]))
                lab = items[b]["lab_t"]
                P = self._put
                P("win", self.base + b); P("ci", st.ci); P("t0", st.t0); P("tick", n); P("j", st.j); P("age_t", a_t)
                P("own_lat", hd["lat_logits"][b]); P("own_lon", hd["lon_logits"][b]); P("own_cons", hd["cons_pred"][b]); P("own_goal", hd["goal_prob"][b])
                P("twin_lat", tw["lat_logits"][b]); P("twin_lon", tw["lon_logits"][b]); P("twin_cons", tw["cons_pred"][b]); P("twin_goal", tw["goal_prob"][b])
                P("v_own", float(batch["p10f"][b, -1, 0])); P("v_log", float(tb["p10f"][b, -1, 0]))
                P("v_own_T", float(batch["p10f"][b, -1 - a_t, 0])); P("v_log_T", float(tb["p10f"][b, -1 - a_t, 0]))
                P("limit_own", int(batch["limit"][b].argmax())); P("limit_log", int(tb["limit"][b].argmax()))
                P("limit_ok_own", bool(batch["limit_ok"][b])); P("limit_ok_log", bool(tb["limit_ok"][b]))
                P("swap_equal", eq)
                P("gt_has", bool(lab["has"])); P("gt_lat", int(lab["lat_cls"])); P("gt_lon", int(lab["lon_cls"]))
                P("gt_lat_al", np.asarray(lab["lat_al"], bool)); P("gt_lon_al", np.asarray(lab["lon_al"], bool)); P("gt_cons", np.asarray(lab["cons"], np.float32))

        def save(self, path, extra=None):
            res = super().save(path, extra)                                # the tool's tick file, unchanged
            tag = str((extra or {}).get("tag", "run"))
            arr = {k: np.asarray(v) for k, v in self.tw.items()}
            np.savez(os.path.join(os.path.dirname(os.path.abspath(path)), f"twin_{tag}.npz"), **arr)
            return res

    return TownRecorder


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--twin-mode", choices=("real", "mutant"), default="real")
    ap.add_argument("--zero-t-ego", action="store_true")
    ap.add_argument("--receipt-dir", required=True)
    a, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    for need in ("--dump-ticks", "--dump-heads"):
        if need not in rest:
            raise SystemExit(f"REFUSED: town_run needs {need}")
    if "--limit" not in rest or rest[rest.index("--limit") + 1] != "causal":
        raise SystemExit("REFUSED: the twin recomputes the causal speed bucket; run with --limit causal")
    import torch
    torch.set_num_threads(int(os.environ.get("TOWN_THREADS", "4")))
    tool = load_tool()
    S = Shared
    S.counters.update({"mode": a.twin_mode, "zero_t_ego": bool(a.zero_t_ego)})
    mutant = a.twin_mode == "mutant"
    orig_mp, orig_ro = tool.model_planner, tool.rollout

    def model_planner(model, device, *aa, **kk):
        S.model = model
        return orig_mp(model, device, *aa, **kk)

    def rollout(planner, states, ds, proxy, **kk):
        S.ds, S.proxy = ds, proxy
        return orig_ro(planner, states, ds, proxy, **kk)

    tool.model_planner, tool.rollout = model_planner, rollout
    tool.TickRecorder = make_recorder(tool, S, mutant)
    if a.zero_t_ego:
        orig_run_t = tool.M.V8FMini.run_t

        def run_t(self, b, adapt, last_t, ck_t, ck_ok):
            w, bi = self.e_ego.weight.data.clone(), self.e_ego.bias.data.clone()
            self.e_ego.weight.data.zero_(); self.e_ego.bias.data.zero_()
            try:
                S.counters["run_t_zeroed_calls"] += 1
                return orig_run_t(self, b, adapt, last_t, ck_t, ck_ok)
            finally:
                self.e_ego.weight.data.copy_(w); self.e_ego.bias.data.copy_(bi)
        tool.M.V8FMini.run_t = run_t
    tag = rest[rest.index("--tag") + 1]
    rc = tool.main(["run"] + rest)
    os.makedirs(a.receipt_dir, exist_ok=True)
    wr = {"tag": tag, "argv": rest, "tool": TOOL, "tool_sha256": TOOL_SHA, "wrapper_sha256": sha256_file(__file__), "tool_rc": rc, "swap_keys": SWAP_KEYS, "keep_keys": KEEP_KEYS,
          "threads": torch.get_num_threads(), **S.counters}
    if S.model is not None:
        wr["e_ego_weight_sha256_after"] = hashlib.sha256(S.model.e_ego.weight.detach().numpy().tobytes()).hexdigest()
    json.dump(wr, open(f"{a.receipt_dir}/receipt_{tag}.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    print(f"RECEIPT {tag}: twin_rows {S.counters['twin_rows']} tick0_swap_unequal {S.counters['tick0_swap_unequal']} later_swap_unequal {S.counters['later_swap_unequal']}/{S.counters['later_rows']} "
          f"logged_p10f_mismatch {S.counters['logged_p10f_mismatch']} zeroed_calls {S.counters['run_t_zeroed_calls']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
