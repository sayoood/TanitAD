"""tswap_run: the T1-kin rollout of PREREG_V8F_TSWAP.md, a THIN WRAPPER around the pinned tool of record `v8f_t1_eval_ri.py` (sha256 0417c838...), loaded BY FILE PATH after its sha256 holds; not edited.
Arms (IMPL_CHOICES_TSWAP.md section 2):
  own     the pinned tool; the wrapper installs nothing.
  tswap   the model's `forward_infer` is wrapped on the instance: the TWIN (TOWN's twin batch: the own batch with only SWAP_KEYS taken from the LOGGED clip) is run, and O reads the 60-dim code of the TWIN's T
          (`forward_diag(batch, code_override = out_twin["code"])`); everything else of the own call (S -> O checkpoints, ages, executor, state) is the own call's.
  mutant  the same plumbing with the twin NOT swapped (twin := own): must reproduce `own` bit for bit.
Tier T1-kin; the twin feeds the logged state (privileged): a diagnostic. CPU only.
usage: tswap_run.py --arm own|tswap|mutant --receipt-dir D -- <tool run flags>   (needs --dump-ticks --dump-heads --limit causal)"""
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
TREE = "F:/TanitAD-scratch/motion_c/ri/tree_ri"
SWAP_KEYS = ("p10f", "base", "pose_now", "pose_t", "pose_s", "limit", "limit_ok")     # TOWN's: the ego-derived inputs of the model
KEEP_KEYS = ("feats", "frame_ok", "nav", "a_t", "a_s")                                  # the same tensors in the twin
CODE_DIM = 60                                                                           # goal 22 + lat 5 + lon 7 + cons 26


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
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri_pinned_tswap", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri_pinned_tswap"] = mod
    spec.loader.exec_module(mod)
    return mod


class Shared:
    model = ds = proxy = states = None
    pending = None
    counters = {"twin_calls": 0, "rows": 0, "tick0_rows": 0, "tick0_swap_unequal": 0, "tick0_code_unequal": 0, "later_rows": 0, "later_swap_unequal": 0, "later_code_unequal": 0,
                "logged_p10f_mismatch": 0, "keep_key_not_same_object": 0, "code_dim_bad": 0, "arm": None}


def twin_batch(tool, S, batch, states, mutant):
    """TOWN's twin batch, verbatim (town_run.py:54-69). -> (twin batch, items)."""
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


def install_tswap(tool, S, model, mutant):
    import torch
    if "forward_infer" in model.__dict__:                                  # model_planner is called once per CHUNK of windows: wrap the instance ONCE (a second wrap would nest twin inside twin)
        return
    orig_fi = model.forward_infer                                          # bound method of the class, taken before the wrap

    def tswap_fi(batch):
        states = S.states
        if states is None or len(states) != batch["feats"].shape[0]:
            raise RuntimeError("tswap: the rollout's states do not match the batch (a forward_infer call outside the model's rollout)")
        tb, _ = twin_batch(tool, S, batch, states, mutant)
        with torch.no_grad():
            tw = orig_fi(tb)                                               # T (and S) on the LOGGED state
            code_tw = tw["code"]
            out = model.forward_diag(batch, code_tw)                       # the own call, O reads the twin's code
            code_own = model._code_pred(out)                               # T's own code in this very call (O did not read it)
        C = S.counters
        C["twin_calls"] += 1
        rows = {"code_own": code_own.float().cpu().numpy(), "code_read": out["code"].float().cpu().numpy(), "code_unequal": [], "swap_equal": [],
                "tw_lat": tw["lat_logits"].float().cpu().numpy(), "tw_lon": tw["lon_logits"].float().cpu().numpy(), "tw_goal": tw["goal_prob"].float().cpu().numpy(),
                "tw_cons_std": tw["cons_std"].float().cpu().numpy(), "v_own": [], "v_log": []}
        for b, st in enumerate(states):
            eq = all(bool(torch.equal(batch[k][b], tb[k][b])) for k in SWAP_KEYS)
            cu = not bool(torch.equal(out["code"][b], code_own[b]))
            C["rows"] += 1
            if st.j == st.t0:                                              # tick 0: the own state IS the logged state
                C["tick0_rows"] += 1; C["tick0_swap_unequal"] += int(not eq); C["tick0_code_unequal"] += int(cu)
            else:
                C["later_rows"] += 1; C["later_swap_unequal"] += int(not eq); C["later_code_unequal"] += int(cu)
            C["keep_key_not_same_object"] += sum(int(tb[k] is not batch[k]) for k in KEEP_KEYS)
            lp = np.nan_to_num(np.asarray(st.base_log["p10f"][st.j]), nan=0.0).astype(np.float32)
            C["logged_p10f_mismatch"] += int(not mutant and not np.array_equal(tb["p10f"][b, -1].numpy(), lp))
            C["code_dim_bad"] += int(code_tw.shape[-1] != CODE_DIM)
            rows["code_unequal"].append(cu); rows["swap_equal"].append(eq)
            rows["v_own"].append(float(batch["p10f"][b, -1, 0])); rows["v_log"].append(float(tb["p10f"][b, -1, 0]))
        S.pending = rows
        return out
    model.forward_infer = tswap_fi


def make_recorder(tool, S):
    class TswapRecorder(tool.TickRecorder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.tw = {}

        def __call__(self, n, batch, plan, states):
            super().__call__(n, batch, plan, states)                       # the tool's own tick row, first (unchanged)
            P = S.pending
            if P is None:
                raise RuntimeError("tswap recorder: no pending twin rows for this tick")
            for b, st in enumerate(states):
                for k, v in (("win", self.base + b), ("ci", st.ci), ("t0", st.t0), ("tick", n), ("j", st.j), ("age_t", int(batch["a_t"][b])), ("age_s", int(batch["a_s"][b])),
                             ("code_own", P["code_own"][b]), ("code_read", P["code_read"][b]), ("code_unequal", P["code_unequal"][b]), ("swap_equal", P["swap_equal"][b]),
                             ("tw_lat", P["tw_lat"][b]), ("tw_lon", P["tw_lon"][b]), ("tw_goal", P["tw_goal"][b]), ("tw_cons_std", P["tw_cons_std"][b]), ("v_own", P["v_own"][b]), ("v_log", P["v_log"][b])):
                    self.tw.setdefault(k, []).append(v)
            S.pending = None

        def save(self, path, extra=None):
            res = super().save(path, extra)                                # the tool's tick file, unchanged
            tag = str((extra or {}).get("tag", "run"))
            np.savez(os.path.join(os.path.dirname(os.path.abspath(path)), f"tswap_{tag}.npz"), **{k: np.asarray(v) for k, v in self.tw.items()})
            return res
    return TswapRecorder


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--arm", choices=("own", "tswap", "mutant"), required=True)
    ap.add_argument("--receipt-dir", required=True)
    a, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    for need in ("--dump-ticks", "--dump-heads"):
        if need not in rest:
            raise SystemExit(f"REFUSED: tswap_run needs {need}")
    if "--limit" not in rest or rest[rest.index("--limit") + 1] != "causal":
        raise SystemExit("REFUSED: the twin recomputes the causal speed bucket; run with --limit causal")
    import torch
    torch.set_num_threads(int(os.environ.get("TSWAP_THREADS", "4")))
    tool = load_tool()
    if not os.path.abspath(tool.M.__file__).replace("\\", "/").startswith(TREE):
        raise SystemExit(f"REFUSED: tanitad is imported from {tool.M.__file__}, not from {TREE}")
    S = Shared
    S.counters["arm"] = a.arm
    orig_mp, orig_ro = tool.model_planner, tool.rollout
    cfg_note = {}

    def model_planner(model, device, *aa, **kk):
        S.model = model
        cfg_note.update({"o_interface": model.cfg.o_interface, "detach_upper": bool(model.cfg.detach_upper), "code_mix": model.cfg.code_mix})
        if a.arm != "own":
            if model.cfg.o_interface != "full":
                raise SystemExit(f"REFUSED: o_interface is {model.cfg.o_interface!r}; O does not read T's code, the swap would be vacuous")
            install_tswap(tool, S, model, a.arm == "mutant")
        return orig_mp(model, device, *aa, **kk)

    def rollout(planner, states, ds, proxy, **kk):
        S.ds, S.proxy, S.states = ds, proxy, states
        return orig_ro(planner, states, ds, proxy, **kk)

    tool.model_planner, tool.rollout = model_planner, rollout
    if a.arm != "own":
        tool.TickRecorder = make_recorder(tool, S)
    tag = rest[rest.index("--tag") + 1]
    rc = tool.main(["run"] + rest)
    os.makedirs(a.receipt_dir, exist_ok=True)
    wr = {"tag": tag, "arm": a.arm, "argv": rest, "tool": TOOL, "tool_sha256": TOOL_SHA, "wrapper_sha256": sha256_file(__file__), "tool_rc": rc, "swap_keys": SWAP_KEYS, "keep_keys": KEEP_KEYS,
          "threads": torch.get_num_threads(), "tanitad_file": tool.M.__file__, **cfg_note, **S.counters}
    json.dump(wr, open(f"{a.receipt_dir}/receipt_{tag}.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    print(f"RECEIPT {tag}: arm {a.arm} rows {S.counters['rows']} tick0_swap_unequal {S.counters['tick0_swap_unequal']} tick0_code_unequal {S.counters['tick0_code_unequal']} "
          f"later_code_unequal {S.counters['later_code_unequal']}/{S.counters['later_rows']} logged_p10f_mismatch {S.counters['logged_p10f_mismatch']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
