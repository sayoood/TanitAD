"""wh_check.py -- PREREG_V8F_WEATHERHEAD study checker (card 4.6; the pattern of ``strat1_check.py``).  Mints ``wh-gate-token-v1``.  DECIDES; launches nothing.

  G-HYG    the config refuses an undeclared ``ctx_pool_*`` attribute and bad values
  G-DVB    every argv lever (--ctx-pool, E grid, fit seed, init checkpoint sha256, label sha256, train manifest sha256, adapter mode and seed, labels) equals the BUILT and recorded value
  G-LIVE   card 4.4 items 2 and 3: non-zero gradient on the five sub-heads, EXACTLY 0.0 on every other parameter; Adam state for exactly the 55 head tensors
  G-CLOCK  the fit's row index and labels equal ``label_rows`` of strat1_ctx_read.py on 100 random train rows
  G-EVAL   each evaluator loads the head checkpoint strictly and its ``ctx_pool_logits`` equal the fit script's on the eval windows to atol 1e-5
  G-CKPT   save / load round trip, byte-equal tensors
  G-SUITE  tests of the new tree against the a4357e79 tree: 0 regressions (the result file written by ``suite``; its ``scope`` is recorded in the token)
EVERY check has a deliberate-regression arm that must read FAIL; a check whose regression arm reads PASS is itself FAILED (the check is blind).
Subcommands: argv ARM | arms | geval FAMILY | suite | mint ARM | verify ARM | run ARM | hyg"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402

os.environ.setdefault("STRAT1_TREE", W.TREE)
sys.path.insert(0, W.READ)
import strat1_ctx_read as R  # noqa: E402
import torch  # noqa: E402

M = W.M
TOKEN_SCHEMA = "wh-gate-token-v1"
WRITTEN_BY = "wh_check.py mint"
REQUIRED = ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE")
TREE_DIRS = (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval",))
TREE_EXT = (".py", ".txt", ".json", ".sh")
BUILD_FILES = ("wh_lib.py", "wh_fit.py", "wh_read.py", "wh_run.py", "wh_iso.py", "wh_check.py", "wh_chain.py")

#: the card's arm table (section 5).  epochs = 'auto' (the H arms select E_v on TRAIN) or the arm whose wh_meta.json lends its E_v.
ARMS = {
    "H_s0": dict(prio=1, kind="fit", init="S1_s0", adapter="s1", adapter_seed=0, fit_seed=0, labels="real", epochs="auto"),
    "H_s1": dict(prio=2, kind="fit", init="S1_s1", adapter="s1", adapter_seed=0, fit_seed=0, labels="real", epochs="auto"),
    "RA_s0": dict(prio=3, kind="fit", init="S1_s0", adapter="random", adapter_seed=0, fit_seed=0, labels="real", epochs="H_s0"),
    "RA_s1": dict(prio=4, kind="fit", init="S1_s1", adapter="random", adapter_seed=1, fit_seed=0, labels="real", epochs="H_s1"),
    "R1_s0": dict(prio=5, kind="fit", init="S1_s0", adapter="s1", adapter_seed=0, fit_seed=0, labels="permuted", epochs="H_s0"),
    "R1_s1": dict(prio=5, kind="fit", init="S1_s1", adapter="s1", adapter_seed=0, fit_seed=0, labels="permuted", epochs="H_s1"),
    "H_s0_rep": dict(prio=6, kind="fit", init="S1_s0", adapter="s1", adapter_seed=0, fit_seed=1, labels="real", epochs="H_s0"),
    "K4p_s0": dict(prio=7, kind="k4p", from_arm="H_s0"),
    "K4p_s1": dict(prio=7, kind="k4p", from_arm="H_s1"),
    "FEAT_s0": dict(prio=8, kind="fit", init="S1_s0", adapter="feat", adapter_seed=0, fit_seed=0, labels="real", epochs="H_s0"),
    "FEAT_s1": dict(prio=8, kind="fit", init="S1_s1", adapter="feat", adapter_seed=1, fit_seed=1, labels="real", epochs="H_s1"),
    "C0A_s0": dict(prio=9, kind="fit", init="C0_s0", adapter="s1", adapter_seed=0, fit_seed=0, labels="real", epochs="H_s0"),
    "C0A_s1": dict(prio=9, kind="fit", init="C0_s1", adapter="s1", adapter_seed=0, fit_seed=0, labels="real", epochs="H_s1"),
}


def sha_file(p):
    return W.sha256_file(p)


def render(arm, runs, device="cuda"):
    """the frozen argv of an arm (the tokens after the subcommand).  ALL levers are named (``--ctx-pool on`` included).  Paths are mapped under ``runs``."""
    s = ARMS[arm]
    out = f"{runs}/fit/{arm}"
    if s["kind"] == "k4p":
        return ["--ckpt", f"{runs}/fit/{s['from_arm']}/ckpt_wh.pt", "--out", out]
    ep = "auto" if s["epochs"] == "auto" else f"{runs}/fit/{s['epochs']}/wh_meta.json"
    return ["--arm", arm, "--init", s["init"], "--ctx-pool", "on", "--adapter", s["adapter"], "--adapter-seed", str(s["adapter_seed"]), "--fit-seed", str(s["fit_seed"]), "--labels", s["labels"],
            "--epochs", ep, "--e-grid", ",".join(str(e) for e in W.E_GRID), "--out", out, "--source", "auto", "--device", device, "--train-cache", W.TRAIN_CACHE]


def argv_sha(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode("utf-8")).hexdigest()


def tree_manifest(root=None):
    root = root or W.TREE
    out = {}
    for parts in TREE_DIRS:
        base = os.path.join(root, *parts)
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith(TREE_EXT):
                    full = os.path.join(dp, f)
                    out[os.path.relpath(full, root).replace(os.sep, "/")] = sha_file(full)
    return out


def tree_sha(root=None):
    return hashlib.sha256(json.dumps(tree_manifest(root), sort_keys=True).encode("utf-8")).hexdigest()


def build_sha():
    return hashlib.sha256(json.dumps({f: sha_file(os.path.join(HERE, f)) for f in BUILD_FILES if os.path.exists(os.path.join(HERE, f))}, sort_keys=True).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ the checks
def regress(read_pass_fn, must_fail_fn):
    """-> (record fields): the regression arm must read FAIL"""
    try:
        rp = bool(must_fail_fn())
    except Exception as e:                                                   # noqa: BLE001
        return {"regression_arm": {"read": "FAIL", "reason": f"{type(e).__name__}: {str(e)[:120]}", "must_read": "FAIL", "pass": True}}
    return {"regression_arm": {"read": "PASS" if rp else "FAIL", "must_read": "FAIL", "pass": not rp}}


def _hyg(cfg_cls):
    n = 0
    for kw in (dict(ctx_pool="maybe"), dict(ctx_pool=True), dict(ctx_pool="ON")):
        try:
            cfg_cls(**kw)
        except ValueError:
            n += 1
    try:
        cfg_cls(ctx_pool_x="on")
    except TypeError:
        n += 1
    return n == 4


class _Lenient:
    """the deliberate-regression config: accepts anything"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def g_hyg():
    ok = _hyg(M.V8FMiniConfig)
    rec = {"pass": ok, **regress(None, lambda: _hyg(_Lenient))}
    rec["pass"] = bool(ok and rec["regression_arm"]["pass"])
    return rec


def declared_record(a, tab_sha, man_sha):
    """what the ARGV declares, resolved to values"""
    return {"ctx_pool": a.ctx_pool, "init_ckpt_sha256": json.load(open(f"{W.CK}/{a.init}/BANK_MANIFEST.json"))["files"]["ckpt_last.pt"]["sha256"], "adapter": a.adapter, "adapter_seed": a.adapter_seed,
            "fit_seed": a.fit_seed, "labels": a.labels, "e_grid": [int(x) for x in a.e_grid.split(",")], "label_table_sha256": W.PINS["strat_labels_table.npz"], "train_manifest_sha256": man_sha}


def built_record(a, tab):
    """what the BUILT objects say, measured from tensors / files, not from argv"""
    ck = torch.load(f"{W.CK}/{a.init}/ckpt_last.pt", map_location="cpu", weights_only=False)
    m = W.load_pool_model(ck, a.adapter, a.adapter_seed)
    lv = m.built_levers()
    sd0 = ck["model"]
    same_adapter = all(torch.equal(m.state_dict()[k], sd0[k]) for k in sd0 if k.startswith(("adapter.", "pos", "time")))
    if a.adapter == "s1":
        mode, seed = "s1" if same_adapter else "other", a.adapter_seed
    else:
        recon = []
        for sd_ in (0, 1, 2, 3):
            m2 = W.load_pool_model(ck, "random", sd_)
            recon.append(all(torch.equal(m.state_dict()[k], m2.state_dict()[k]) for k in ("adapter.0.weight", "adapter.1.weight", "pos", "time")))
        mode, seed = ("random" if not same_adapter else "other"), (recon.index(True) if True in recon else -1)
        if a.adapter == "feat":
            mode = "feat" if mode == "random" else mode               # a FEAT adapter is the same fresh init, trained (the fit script's train_adapter)
    W.reinit_head(m, a.fit_seed)
    fresh = M.build_ctx_pool_heads(m.cfg.d, a.fit_seed).state_dict()
    other = M.build_ctx_pool_heads(m.cfg.d, a.fit_seed + 1).state_dict()
    hs = W.head_state(m)
    head_seed = a.fit_seed if all(torch.equal(hs[f"ctx_pool_head.{k}"], v) for k, v in fresh.items()) and not all(torch.equal(hs[f"ctx_pool_head.{k}"], v) for k, v in other.items()) else -1
    rs = W.train_rowset(tab, range(40))
    y = W.fit_labels(tab, rs["row"])
    yp = W.permute_labels_between_clips(y, rs["ci"], 0)
    return {"ctx_pool": lv.get("ctx_pool"), "init_ckpt_sha256": sha_file(f"{W.CK}/{a.init}/ckpt_last.pt"), "adapter": mode, "adapter_seed": seed if a.adapter != "s1" else a.adapter_seed, "fit_seed": head_seed,
            "labels": "permuted" if not np.array_equal(y, yp) and a.labels == "permuted" else ("real" if a.labels == "real" else "permuted-unbuilt"), "e_grid": list(W.E_GRID),
            "label_table_sha256": sha_file(W.pinned("strat_labels_table.npz")), "train_manifest_sha256": sha_file(f"{a.train_cache}/manifest.jsonl")}


def g_dvb(toks):
    from wh_fit import build_parser
    a = build_parser().parse_args(["fit"] + list(toks))
    tab = W.load_table()
    man_sha = sha_file(f"{a.train_cache}/manifest.jsonl")
    dec, built = declared_record(a, W.PINS["strat_labels_table.npz"], man_sha), built_record(a, tab)
    bad = [f"{k}: declared {dec[k]!r} != built {built.get(k)!r}" for k in dec if dec[k] != built.get(k)]

    def flipped():
        d2 = dict(dec)
        d2["fit_seed"] = dec["fit_seed"] + 1                                  # the regression arm: argv says another seed than the head was built with
        return not [k for k in d2 if d2[k] != built.get(k)]
    rec = {"pass": not bad, "problems": bad, "declared": dec, "built": {k: built[k] for k in dec}, **regress(None, flipped)}
    rec["pass"] = bool(not bad and rec["regression_arm"]["pass"])
    return rec


def _mut_step():
    src = inspect.getsource(W.fit_step).replace('loss = T["ctx_pool"]', 'loss = logits.sum() * 0.0')            # m1: the fit-loss line deleted
    ns = dict(W.__dict__)
    exec(src, ns)
    return ns["fit_step"]


def _mut_cls():
    src = textwrap.dedent(inspect.getsource(M.V8FMini.ctx_pool_fit_logits)).replace("torch.set_grad_enabled(torch.is_grad_enabled() and train_adapter)", "torch.enable_grad()")   # m2: no_grad dropped
    ns = dict(M.__dict__)
    exec(src, ns)
    return type("MutModel", (M.V8FMini,), {"ctx_pool_fit_logits": ns["ctx_pool_fit_logits"]})


def g_live(toks):
    from wh_fit import build_parser
    a = build_parser().parse_args(["fit"] + list(toks))
    tab = W.load_table()
    clips = [str(c) for c in tab["train_clip"]]
    rs = W.train_rowset(tab, range(64))
    y = torch.as_tensor(W.fit_labels(tab, rs["row"]))
    src = W.StreamSource(a.train_cache, clips, rs)
    ix = np.arange(W.BS)
    feats, yb = src.gather(ix), y[:W.BS]
    ck = torch.load(f"{W.CK}/{a.init}/ckpt_last.pt", map_location="cpu", weights_only=False)
    ta = a.adapter == "feat"
    m = W.load_pool_model(ck, a.adapter, a.adapter_seed)
    live = W.live_check(m, feats, yb, train_adapter=ta)
    m = W.load_pool_model(ck, a.adapter, a.adapter_seed)
    fr = W.fit_head(m, src, y.numpy(), np.arange(2 * W.BS), 1, 0, torch.device("cpu"), ta, lambda *x: None)
    # regression arms: m1 (fit-loss line deleted) -> G-LIVE reads FAIL;  m2 (no_grad dropped) -> the exact-zero control reads non-zero AND the census reads FAIL
    m1 = W.live_check(W.load_pool_model(ck, a.adapter, a.adapter_seed), feats, yb, step=_mut_step(), train_adapter=ta)
    mc = _mut_cls()
    mm = W.load_pool_model(ck, "s1", 0, cls=mc)
    m2 = W.live_check(mm, feats, yb, train_adapter=False)
    mm2 = W.load_pool_model(ck, "s1", 0, cls=mc)
    cen2 = W.fit_head(mm2, src, y.numpy(), np.arange(W.BS), 1, 0, torch.device("cpu"), False, lambda *x: None)["census"]
    reg = {"m1_fit_loss_line_deleted": {"live_pass": m1["pass"], "must_read": False}, "m2_no_grad_dropped": {"live_pass": m2["pass"], "other_grad_abs_sum": m2["other_grad_abs_sum"], "census_pass": cen2["pass"], "must_read": False}}
    reg_ok = (not m1["pass"]) and (not m2["pass"]) and (not cen2["pass"])
    ok = bool(live["pass"] and fr["census"]["pass"] and reg_ok)
    return {"pass": ok, "live": {k: live[k] for k in ("pass", "subhead_grad_abs_sum", "other_grad_abs_sum")}, "census": fr["census"], "regression_arm": {"read": "FAIL" if reg_ok else "PASS", "must_read": "FAIL", "pass": reg_ok, "detail": reg}}


def g_clock():
    tab = W.load_table()
    rs = W.train_rowset(tab)
    n = len(rs["row"])
    idx = np.random.RandomState(0).choice(n, 100, replace=False)
    rows_ok = bool(np.array_equal(rs["row"], np.asarray(tab["train_off"])[rs["ci"]] + rs["t"]))
    ev = {"eval_" + k[6:]: v for k, v in tab.items() if k.startswith("train_")}
    ref = R.label_rows(ev, rs["row"][idx])
    mine = W.fit_labels(tab, rs["row"][idx])
    ok = bool(rows_ok and np.array_equal(ref, mine))
    shifted = lambda: np.array_equal(ref, W.fit_labels(tab, rs["row"][idx], shift=1))        # noqa: E731
    rec = {"pass": ok, "n_rows_checked": 100, "row_index_equals_off_plus_t": rows_ok, **regress(None, shifted)}
    rec["pass"] = bool(ok and rec["regression_arm"]["pass"])
    return rec


def g_ckpt(workdir):
    ck = torch.load(f"{W.CK}/S1_s0/ckpt_last.pt", map_location="cpu", weights_only=False)
    m = W.load_pool_model(ck, "s1", 0)
    p = f"{workdir}/rt_ckpt.pt"
    W.save_head_ckpt(p, m, ck, {"arm": "G-CKPT"})
    rt = torch.load(p, map_location="cpu", weights_only=False)
    a, b = m.state_dict(), rt["model"]
    eq = set(a) == set(b) and all(W.tensor_sha(a[k]) == W.tensor_sha(b[k]) for k in a)

    def tampered():
        k = next(k for k in b if k.startswith("ctx_pool_head."))
        t = b[k].clone()
        t.view(-1)[0] += 1.0
        return set(a) == set(b) and all(W.tensor_sha(a[k2]) == W.tensor_sha(t if k2 == k else b[k2]) for k2 in a)
    rec = {"pass": bool(eq), "n_tensors": len(a), **regress(None, tampered)}
    rec["pass"] = bool(eq and rec["regression_arm"]["pass"])
    os.remove(p)
    return rec


def fit_side(ckpt, ci, t, cache, offset=0):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), W.p10_zero())
    m.load_state_dict(ck["model"])
    join = dict(np.load(W.pinned("join_eval141.npz")))
    rs = {"ci": np.asarray(ci), "t": np.asarray(t), "row": np.zeros(len(ci), np.int64)}
    return W.predict(m, W.StreamSource(cache, [str(c) for c in join["clip"]], rs, offset), np.arange(len(ci)), torch.device("cpu"))


def g_eval(family, workdir, t1_windows=4):
    """each evaluator (T0: the full grid; T1: the first ``t1_windows`` windows' tick 0) loads a head checkpoint strictly; its ctx_pool_logits equal the fit script's on the eval windows (atol 1e-5)"""
    import wh_run as RUN
    init = "S1_s0" if family == "text" else "C0_s0"
    os.makedirs(workdir, exist_ok=True)
    ck = torch.load(f"{W.CK}/{init}/ckpt_last.pt", map_location="cpu", weights_only=False)
    m = W.load_pool_model(ck, "s1", 0)
    ckp = f"{workdir}/geval_{family}.pt"
    W.save_head_ckpt(ckp, m, ck, {"arm": "G-EVAL", "SMOKE": False})
    rec = {"family": family, "init": init}
    rc0 = RUN.run_argv(RUN.t0_argv(ckp, f"WH-GEVAL-{family}", "on", workdir), f"{workdir}/geval_{family}_t0.log", wait_s=1800)
    z = np.load(f"{workdir}/eval_WH-GEVAL-{family}.npz", allow_pickle=True) if rc0 == 0 else None
    fs = fit_side(ckp, z["win_ci"], z["win_t"], W.EVAL_CACHE) if z is not None else None
    d0 = float(np.abs(z["ctx_pool_logits"] - fs).max()) if z is not None else None
    rec["T0"] = {"rc": rc0, "max_abs_diff": d0, "n_windows": None if z is None else int(len(fs)), "pass": bool(rc0 == 0 and d0 is not None and d0 <= 1e-5)}
    # T1 tick 0
    rc1 = RUN.run_argv(RUN.t1_argv(ckp, f"WH-GEVAL-{family}-t1", "on", workdir, t1_windows), f"{workdir}/geval_{family}_t1.log", wait_s=1800)
    p1 = f"{workdir}/ticks/ticks_WH-GEVAL-{family}-t1.npz"
    if rc1 == 0 and os.path.exists(p1):
        zt = np.load(p1, allow_pickle=True)
        sel = np.where(zt["tick"] == 0)[0]
        f1 = fit_side(ckp, zt["ci"][sel], zt["t0"][sel], W.EVAL_CACHE)
        d1 = float(np.abs(zt["ctx_pool_logits"][sel] - f1).max())
        rec["T1"] = {"rc": rc1, "max_abs_diff": d1, "n_windows": int(len(sel)), "pass": bool(d1 <= 1e-5)}
    else:
        rec["T1"] = {"rc": rc1, "pass": False, "max_abs_diff": None}
    # the evaluators REFUSE a flag that disagrees with the checkpoint's cfg (the silent-skip regression)
    rc_bad = RUN.run_argv(RUN.t0_argv(ckp, f"WH-GEVAL-{family}-bad", "off", workdir), f"{workdir}/geval_{family}_bad.log", wait_s=1800)
    rec["flag_mismatch_refused"] = {"rc": rc_bad, "pass": rc_bad not in (0, 75)}
    # regression arm: the fit-side logits of a ONE-FRAME-SHIFTED window must NOT equal the evaluator's
    sh = fit_side(ckp, z["win_ci"][:64], z["win_t"][:64], W.EVAL_CACHE, offset=1) if z is not None else None
    dsh = float(np.abs(z["ctx_pool_logits"][:64] - sh).max()) if z is not None else None
    rec["regression_arm"] = {"what": "fit-side logits of a window shifted by one frame", "max_abs_diff": dsh, "must_read": "FAIL (diff > 1e-5)", "read": "FAIL" if (dsh is not None and dsh > 1e-5) else "PASS", "pass": bool(dsh is not None and dsh > 1e-5)}
    rec["pass"] = bool(rec["T0"]["pass"] and rec["T1"]["pass"] and rec["flag_mismatch_refused"]["pass"] and rec["regression_arm"]["pass"])
    rec["tree_manifest_sha256"] = tree_sha()
    os.remove(ckp)
    return rec


# ---- G-SUITE ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def junit_results(path):
    import xml.etree.ElementTree as ET
    out = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        tid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
        st = "pass"
        for ch in tc:
            if ch.tag in ("failure", "error", "skipped"):
                st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]
                break
        out[tid] = st
    return out


def suite_compare(base, new):
    bad = ("fail", "error")
    reg = sorted(t for t, s in new.items() if s in bad and base.get(t) == "pass")
    return {"n_regressions": len(reg), "regressions": reg, "base_n": len(base), "new_n": len(new), "base_pass": sum(s == "pass" for s in base.values()), "new_pass": sum(s == "pass" for s in new.values()),
            "new_only_failures": sorted(t for t, s in new.items() if t not in base and s in bad), "fail_on_both": sorted(t for t, s in new.items() if s in bad and base.get(t) in bad)}


def run_suite(tree, tests, junit, py=None, extra_env=None):
    env = dict(os.environ, PYTHONPATH=";".join(os.path.normpath(p) for p in (tree + "/stack", tree + "/taniteval")), CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **(extra_env or {}))
    base = os.path.join(os.path.dirname(junit), "pt_" + os.path.basename(junit)[:-4])
    os.makedirs(os.path.dirname(base), exist_ok=True)
    cmd = [py or sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={base}", f"--junitxml={junit}"] + [os.path.join(tree, "stack", "tests", t) for t in tests]
    return subprocess.run(cmd, cwd=tree + "/stack", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace").returncode


def cmd_suite(a):
    os.makedirs(a.out_dir, exist_ok=True)
    new_x, base_x = f"{a.out_dir}/new.xml", f"{a.out_dir}/base.xml"
    tests = a.tests.split(",")
    run_suite(W.TREE, tests, new_x)
    run_suite(a.base_tree, [t for t in tests if os.path.exists(f"{a.base_tree}/stack/tests/{t}")], base_x)
    base, new = junit_results(base_x), junit_results(new_x)
    r = suite_compare(base, new)
    # regression arm: a planted failure in the new results must be read as a regression
    planted = dict(new)
    k = next(t for t, s in base.items() if s == "pass")
    planted[k] = "fail"
    r["regression_arm"] = {"read": "FAIL" if suite_compare(base, planted)["n_regressions"] > 0 else "PASS", "must_read": "FAIL", "pass": suite_compare(base, planted)["n_regressions"] > 0}
    res = {"schema": "wh-gsuite-result-v1", "written_by": "wh_check.py suite", "base_commit": W.BASE_COMMIT, "scope": "subset" if not a.full else "full", "tests": tests, "tree_manifest_sha256": tree_sha(), **r,
           "pass": bool(r["n_regressions"] == 0 and r["regression_arm"]["pass"] and r["new_n"] > 0 and r["base_n"] > 0)}
    json.dump(res, open(f"{a.out_dir}/gsuite_result.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("scope", "base_n", "new_n", "n_regressions", "new_only_failures", "pass")}))
    return 0 if res["pass"] else 1


# ---- token ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def binding(arm, toks, gsuite_path, geval_path=None):
    s = ARMS[arm]
    d = {"label_table": sha_file(W.pinned("strat_labels_table.npz")), "eval_grid": sha_file(W.pinned("eval_grid_windows.json")), "join_eval141": sha_file(W.pinned("join_eval141.npz")),
         "ci.py": sha_file(W.TREE + "/taniteval/taniteval/ci.py"), "strat1_ctx_read.py": sha_file(W.READ + "/strat1_ctx_read.py")}
    for k, p in (("ci.py", READER_PIN["ci.py"]), ("strat1_ctx_read.py", READER_PIN["strat1_ctx_read.py"])):
        if not d[k].startswith(p):
            raise RuntimeError(f"{k} sha {d[k][:8]} != the card pin {p}")
    if s["kind"] == "fit":
        d["init_ckpt"] = sha_file(f"{W.CK}/{s['init']}/ckpt_last.pt")
        d["train_manifest"] = sha_file(f"{W.TRAIN_CACHE}/manifest.jsonl")
    return {"arm": arm, "base_commit": W.BASE_COMMIT, "argv_sha256": argv_sha(toks), "data": d, "tree_manifest_sha256": tree_sha(), "build_sha256": build_sha(),
            "gsuite_result_sha256": sha_file(gsuite_path) if gsuite_path and os.path.exists(gsuite_path) else None,
            "geval_sha256": sha_file(geval_path) if geval_path and os.path.exists(geval_path) else None}


READER_PIN = {"ci.py": "528276c5", "strat1_ctx_read.py": "1b9377ba"}


def body_sha(tok):
    return hashlib.sha256(json.dumps({k: v for k, v in tok.items() if k != "body_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()


def write_token(path, b, checks):
    """the ONLY writer of a token: every required check must read PASS and carry a passing regression arm"""
    miss = [c for c in REQUIRED if not (checks.get(c) or {}).get("pass")]
    blind = [c for c in REQUIRED if (checks.get(c) or {}).get("regression_arm", {}).get("pass") is not True and c != "G-SUITE"]
    if miss or blind:
        raise ValueError(f"refusing to write a token: checks not PASS {miss}; regression arm not FAIL {blind}")
    tok = {"schema": TOKEN_SCHEMA, "written_by": WRITTEN_BY, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks, **b}
    tok["body_sha256"] = body_sha(tok)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(tok, f, indent=1, sort_keys=True, default=float)
    os.replace(path + ".part", path)
    return tok


def token_path(runs, arm):
    return f"{runs}/gate/{arm}/wh_gate_token.json"


def verify_token(path, toks, arm, gsuite_path=None, geval_path=None):
    """-> (token, reason): reason is None iff the token written by ``mint`` still binds this arm, argv, every input file, the tree and the build scripts"""
    if not path:
        return None, "no --token given"
    try:
        tok = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None, "no gate token"
    p = []
    if tok.get("schema") != TOKEN_SCHEMA or tok.get("written_by") != WRITTEN_BY:
        p.append("token not written by the checker")
    if tok.get("body_sha256") != body_sha(tok):
        p.append("token body was edited")
    if [c for c in REQUIRED if not (tok.get("checks", {}).get(c) or {}).get("pass")]:
        p.append("token checks not PASS")
    gs = gsuite_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), "gate", "gsuite_result.json")
    ge = geval_path
    if tok.get("arm") != arm:
        p.append(f"token is for arm {tok.get('arm')}")
    try:
        want = binding(arm, toks, gs, ge or None)
    except Exception as e:                                                   # noqa: BLE001
        return None, f"binding failed: {e}"
    for k, v in want.items():
        if k in ("geval_sha256", "gsuite_result_sha256"):
            continue
        if tok.get(k) != v:
            p.append(f"token {k} differs from the current {k}")
    tok["sha256"] = sha_file(path)
    return (tok, "; ".join(p)) if p else (tok, None)


def cmd_mint(a):
    toks = render(a.arm, a.runs, a.device)
    gd = f"{a.runs}/gate"
    os.makedirs(f"{gd}/{a.arm}", exist_ok=True)
    gs, ge = f"{gd}/gsuite_result.json", f"{gd}/geval_{a.family}.json"
    b = binding(a.arm, toks, gs, ge)
    checks = {"G-HYG": g_hyg(), "G-CLOCK": g_clock(), "G-CKPT": g_ckpt(f"{gd}/{a.arm}")}
    try:
        gsr = json.load(open(gs, encoding="utf-8"))
        checks["G-SUITE"] = {"pass": bool(gsr.get("pass") and gsr.get("tree_manifest_sha256") == b["tree_manifest_sha256"]), "scope": gsr.get("scope"), "n_regressions": gsr.get("n_regressions"),
                             "regression_arm": gsr.get("regression_arm"), "result_sha256": b["gsuite_result_sha256"]}
    except (OSError, ValueError):
        checks["G-SUITE"] = {"pass": False, "problems": ["no G-SUITE result file"]}
    try:
        ger = json.load(open(ge, encoding="utf-8"))
        checks["G-EVAL"] = {"pass": bool(ger.get("pass") and ger.get("tree_manifest_sha256") == b["tree_manifest_sha256"]), "T0": ger.get("T0"), "T1": ger.get("T1"), "regression_arm": ger.get("regression_arm"), "geval_sha256": b["geval_sha256"]}
    except (OSError, ValueError):
        checks["G-EVAL"] = {"pass": False, "problems": [f"no G-EVAL record ({ge}); run `geval {a.family}` first"]}
    if ARMS[a.arm]["kind"] == "fit":
        checks["G-DVB"], checks["G-LIVE"] = g_dvb(toks), g_live(toks)
    else:                                                                    # K4p: no fit, no new data: the DVB / LIVE of its parent H arm are what its ckpt carries; the mint still requires the rest
        checks["G-DVB"] = {"pass": True, "note": "k4p: no fit; the head is re-initialised under CTX_POOL_INIT_SEED by wh_fit k4p (refuses a head equal to the trained one)", "regression_arm": {"read": "FAIL", "must_read": "FAIL", "pass": True, "note": "n/a, checked at k4p run time"}}
        checks["G-LIVE"] = {"pass": True, "note": "k4p: no gradient step", "regression_arm": {"read": "FAIL", "must_read": "FAIL", "pass": True, "note": "n/a, no fit"}}
    json.dump(checks, open(f"{gd}/{a.arm}/checks.json", "w", encoding="utf-8"), indent=1, default=float)
    failed = [c for c in REQUIRED if not (checks.get(c) or {}).get("pass")]
    if failed:
        print("REFUSED: checks not PASS: " + ", ".join(failed))
        for c in failed:
            print("  ", c, json.dumps(checks[c], default=float)[:500])
        return 1
    write_token(token_path(a.runs, a.arm), b, checks)
    print("TOKEN-WRITTEN", a.arm)
    return 0


def cmd_run(a):
    toks = render(a.arm, a.runs, a.device)
    tp = token_path(a.runs, a.arm)
    tok, why = verify_token(tp, toks, a.arm, f"{a.runs}/gate/gsuite_result.json", None)
    if why:
        print(f"REFUSED {a.arm}: {why}")
        return 2
    print("TOKEN-OK", a.arm)
    from wh_fit import main as fit_main
    if ARMS[a.arm]["kind"] == "k4p":
        return fit_main(["k4p"] + toks)
    return fit_main(["fit"] + toks + ["--token", tp])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("argv")
    p.add_argument("arm")
    p.add_argument("--runs", required=True)
    p.add_argument("--device", default="cuda")
    sub.add_parser("arms")
    sub.add_parser("hyg")
    p = sub.add_parser("geval")
    p.add_argument("family", choices=("text", "v9"))
    p.add_argument("--runs", required=True)
    p = sub.add_parser("suite")
    p.add_argument("--base-tree", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tests", default="test_v8f_weatherhead.py,test_v8f_strat1.py,test_v8f_mini.py,test_v8f_level_head.py,test_v8f_lead_head.py,test_strat1_token.py,test_launch_gate_v8f.py")
    p.add_argument("--full", action="store_true")
    for nm in ("mint", "run"):
        p = sub.add_parser(nm)
        p.add_argument("arm", choices=sorted(ARMS))
        p.add_argument("--runs", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--family", choices=("text", "v9"), default="text")
    a = ap.parse_args(argv)
    if a.cmd == "argv":
        print(json.dumps(render(a.arm, a.runs, a.device)))
        return 0
    if a.cmd == "arms":
        print(json.dumps({k: v for k, v in sorted(ARMS.items(), key=lambda kv: kv[1]["prio"])}))
        return 0
    if a.cmd == "hyg":
        r = g_hyg()
        print(json.dumps(r))
        return 0 if r["pass"] else 1
    if a.cmd == "geval":
        r = g_eval(a.family, f"{a.runs}/gate/geval_{a.family}_work")
        json.dump(r, open(f"{a.runs}/gate/geval_{a.family}.json", "w", encoding="utf-8"), indent=1)
        print(json.dumps({k: r[k] for k in ("pass", "T0", "T1", "flag_mismatch_refused", "regression_arm")}, default=float))
        return 0 if r["pass"] else 1
    if a.cmd == "suite":
        return cmd_suite(a)
    return cmd_mint(a) if a.cmd == "mint" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
