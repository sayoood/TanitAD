"""PREREG_V8F_SLOWWM controls 1, 2 and 4 (and the synthetic half of 3 via test_slow_basis.py), each with its deliberate-regression arm (a MUTATION of a copy of the tree that must make the named tests go RED).
CPU only. Usage:
    python controls_slowwm.py --tree <patched tree> --base <base tree (1faa1a06)> --work <scratch dir> --out <dir> [--smoke-dir <dir with the SMOKE fit/cache>]
    python controls_slowwm.py --dump-c1 <tree> <npz>                    (child of control 1)
Control 1 batch is PLANTED (seeded random zmean / action), the arguments are R35H s0's banked argv (`ladder_R35H_s0/argv.txt`); the real windows live on box 2. Control 3 on REAL clips is `slow_basis.py check3` (SMOKE here).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARGV0 = "C:/Users/Admin/t2v8f_pulls/ladder_R35H_s0/argv.txt"
PY = sys.executable
F_WIN = 91


def setup(tree):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, os.path.join(tree, "stack")), sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
    import tanitad
    assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(os.path.abspath(tree))), tanitad.__file__
    import v8f_mini_train as T
    from tanitad.models import v8f_mini as M
    assert os.path.normcase(os.path.abspath(T.__file__)).startswith(os.path.normcase(os.path.abspath(tree))), T.__file__
    return torch, T, M


def env_for(tree, taniteval=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = ";".join([os.path.join(tree, "stack", "scripts"), os.path.join(tree, "stack"), taniteval or os.path.join(tree, "taniteval")])
    e.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", ZC_REPO=tree)
    return e


def line_of(tree, rel, needle, nth=0):
    ls = open(os.path.join(tree, rel), encoding="utf-8").read().splitlines()
    hits = [i + 1 for i, l in enumerate(ls) if needle in l]
    return hits[nth] if len(hits) > nth else None


def planted_batch(torch, n=2):
    g = torch.Generator().manual_seed(3)
    return torch.randn(n, F_WIN, 256, generator=g), torch.randn(n, F_WIN, 2, generator=g)


# ------------------------------------------------------------------------------------------ control 1 (child): dump one off step
def dump_c1(tree, out):
    torch, T, M = setup(tree)
    argv = open(ARGV0, encoding="utf-8").read().split()
    a = T.build_parser().parse_args(argv)
    cfg = T.make_config(a)
    torch.manual_seed(0)
    m = M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)
    zm, act = planted_batch(torch)
    res = {"init/" + k: v.clone().numpy() for k, v in m.state_dict().items()}
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if n.startswith("wm.")], lr=1e-3)
    m.train()
    torch.manual_seed(7)
    terms = m.wm_only_loss({"zmean": zm, "act": act})["terms"]
    for k, v in terms.items():
        res["term/" + k] = v.detach().numpy()
    terms["total"].backward()
    for n, p in m.named_parameters():
        if n.startswith("wm."):
            res["grad/" + n] = (p.grad.clone().numpy() if p.grad is not None else np.zeros(1))
    opt.step()
    for k, v in m.state_dict().items():
        res["after/" + k] = v.clone().numpy()
    np.savez(out, **res)
    print("DUMP-OK", len(res), "wm_basis", getattr(cfg, "wm_basis", "(no such field: base tree)"), flush=True)


def control1_identity(tree, base, work):
    outs = {}
    for tag, t in (("patched", tree), ("base", base)):
        f = os.path.join(work, f"c1_{tag}.npz")
        r = subprocess.run([PY, os.path.abspath(__file__), "--dump-c1", t, f], env=env_for(t), capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            return {"pass": False, "error": r.stderr[-800:]}
        outs[tag] = dict(np.load(f))
    a, b = outs["patched"], outs["base"]
    ks = sorted(set(a) | set(b))
    diff = [k for k in ks if k not in a or k not in b or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    return {"pass": not diff, "n_arrays": len(ks), "n_unequal": len(diff), "unequal_first": diff[:5], "what": "init state dict, the WM-only terms, every wm.* gradient and every parameter after one AdamW step, patched tree (default argv: --wm-basis absent) vs base 1faa1a06, bit for bit (np.array_equal)",
            "batch": "PLANTED seeded random zmean [2, 91, 256] and action [2, 91, 2] (the real windows live on box 2); argv = R35H s0's banked argv"}


# ------------------------------------------------------------------------------------------ mutations (deliberate regressions, run on a COPY)
MUTS = [
    ("M1_default_slow", "control 1: the default of --wm-basis flips to slow", [("stack/scripts/v8f_mini_train.py", 'choices=("adapter", "slow", "pca"), default="adapter"', 'choices=("adapter", "slow", "pca"), default="slow"')],
     ["test_off_default_is_adapter_everywhere"]),
    ("M1b_cfg_default_slow", "control 1: the config default flips to slow", [("stack/tanitad/models/v8f_mini.py", 'wm_basis: str = "adapter"', 'wm_basis: str = "slow"')],
     ["test_off_default_is_adapter_everywhere"]),
    ("M2_make_config_line_deleted", "control 2: the ONE line that applies --wm-basis is deleted", [("stack/scripts/v8f_mini_train.py", "                           wm_basis=a.wm_basis,", "")],
     ["test_reach_argparse_to_namespace_to_config_to_model", "test_gate_passes_and_reads_the_built_lever_back"]),
    ("M3_loss_ignores_zbasis", "control 2: the loss no longer reads the cached zbasis batch key", [("stack/tanitad/models/v8f_mini.py", 'if "zbasis" in batch:', 'if False and "zbasis" in batch:')],
     ["test_reach_z1_to_loss_and_a_nonzero_gradient_on_proj_with_exact_zero_on_the_frozen_adapter", "test_reach_the_zbasis_key_is_what_the_loss_reads_not_zmean"]),
    ("M4_proj_detached", "control 2: z1 is detached from proj (no gradient reaches wm.proj)", [("stack/tanitad/models/v8f_mini.py", "        return self._z1_post(u)\n", "        return self._z1_post(u).detach()\n")],
     ["test_reach_z1_to_loss_and_a_nonzero_gradient_on_proj_with_exact_zero_on_the_frozen_adapter"]),
    ("M5_kind_mismatch_accepted", "control 2: the cache kind is no longer checked (the adapter / wrong-kind cache is accepted under --wm-basis slow)", [("stack/tanitad/data/v8f_wm_windows.py", '("kind", kind), ', "")],
     ["test_refuses_a_cache_of_the_wrong_kind_missing_manifest_wrong_dim_or_wrong_sha", "test_trainer_refuses_before_any_build"]),   # second line of defence (test_gate_refuses_a_cache_of_the_other_kind stays green): bad_wm_basis compares the BUILT dataset's manifest kind
    ("M6_trainer_refusal_removed", "control 2: the trainer's static refusal returns no problem", [("stack/scripts/v8f_mini_train.py", '    bm = os.path.join(a.wm_zcache, WW.BASIS_MANIFEST) if a.wm_zcache else ""\n', '    return []\n    bm = os.path.join(a.wm_zcache, WW.BASIS_MANIFEST) if a.wm_zcache else ""\n')],
     ["test_trainer_refuses_before_any_build"]),   # second line of defence (test_gate_refuses_a_basis_declared_without_the_windows_that_read_it stays green): the ladder gate refuses a --wm-zcache without wmonly windows
    ("M7_mean_path_not_refused", "control 2: a basis model reads a token mean silently (the adapter arm under a basis flag)", [("stack/tanitad/models/v8f_mini.py", '''        if self.wm_basis != "adapter":                                          # PREREG_V8F_SLOWWM: a token mean handed to a basis model would silently run the adapter arm
            raise ValueError(f"wm_basis {self.wm_basis}: z1 is built from the cached basis u (z1_from_basis), not from a token mean")
''', "")],
     ["test_off_adapter_model_refuses_a_basis_batch_and_a_basis_model_refuses_the_adapter_paths", "test_gate_passes_and_reads_the_built_lever_back"]),
]


def copy_for_mutation(tree, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for rel in ("stack/tanitad", "stack/scripts", "stack/tests/test_v8f_slowwm.py", "stack/pyproject.toml"):
        s, d = os.path.join(tree, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        (shutil.copytree(s, d) if os.path.isdir(s) else shutil.copy2(s, d))


def run_pytest(root, tests, basetemp, k=None, taniteval=None):
    args = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", basetemp, os.path.join(root, "stack", "tests", "test_v8f_slowwm.py")]
    if k:
        args += ["-k", k]
    r = subprocess.run(args, env=env_for(root, taniteval), capture_output=True, text=True, encoding="utf-8", cwd=root)
    out = r.stdout + r.stderr
    failed = sorted(set(re.findall(r"FAILED [^\n]*?::(test_\w+)", out)))
    errors = sorted(set(re.findall(r"ERROR [^\n]*?::(test_\w+)", out)))
    return r.returncode, failed, errors, out[-400:]


def mutation_arm(tree, work, tag, desc, edits, must_fail):
    dst = os.path.join(work, "mut", tag)
    copy_for_mutation(tree, dst)
    for rel, old, new in edits:
        p = os.path.join(dst, rel)
        s = open(p, encoding="utf-8", newline="").read()
        assert s.count(old) == 1, (tag, rel, s.count(old))
        open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    rc, failed, errors, tail = run_pytest(dst, None, os.path.join(work, "pt_" + tag), k=" or ".join(must_fail), taniteval=os.path.join(tree, "taniteval"))
    res = {"mutation": desc, "must_fail": must_fail, "failed": failed, "errors_not_failures": errors, "red": bool(set(must_fail) <= set(failed)), "pytest_rc": rc}
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(os.path.join(work, "pt_" + tag), ignore_errors=True)
    return res


# ------------------------------------------------------------------------------------------ control 2: reachability (hops with file:line) and the live gradient
def control2(tree):
    torch, T, M = setup(tree)
    import dataclasses
    hops = {
        "argparse --wm-basis": ("stack/scripts/v8f_mini_train.py", 'ap.add_argument("--wm-basis"'),
        "FLAG_CHECKS entry": ("stack/scripts/v8f_mini_train.py", '"wm_basis": "lever:wm_basis"'),
        "namespace -> config (the one line)": ("stack/scripts/v8f_mini_train.py", "wm_basis=a.wm_basis,"),
        "static refusal before any build": ("stack/scripts/v8f_mini_train.py", "_bp = basis_cache_problems(a)"),
        "config field": ("stack/tanitad/models/v8f_mini.py", 'wm_basis: str = "adapter"'),
        "config validation": ("stack/tanitad/models/v8f_mini.py", "if self.wm_basis not in WM_BASIS_KINDS:"),
        "model attribute": ("stack/tanitad/models/v8f_mini.py", "self.wm_basis = cfg.wm_basis"),
        "dataset gets the kind": ("stack/scripts/v8f_mini_train.py", "zcache=a.wm_zcache, basis=a.wm_basis)"),
        "dataset reads zbasis + manifest check": ("stack/tanitad/data/v8f_wm_windows.py", 'f["zbasis"]'),
        "collate": ("stack/tanitad/data/v8f_wm_windows.py", 'out["zbasis"] ='),
        "batch -> loss": ("stack/tanitad/models/v8f_mini.py", 'self.wm.forward_basis(batch["zbasis"]'),
        "forward_basis -> z1_from_basis -> _z1_post (proj, enc1) -> stack.loss": ("stack/tanitad/models/v8f_mini.py", "return self.stack.loss(self.z1_from_basis(u)"),
        "G-DVB read-back": ("stack/scripts/v8f_mini_train.py", "def bad_wm_basis("),
        "selfcheck levers.wm_basis_readback": ("stack/scripts/v8f_mini_train.py", 'levers["wm_basis_readback"] ='),
    }
    res = {"hops": {k: f"{rel}:{line_of(tree, rel, n)}" for k, (rel, n) in hops.items()}}
    argv = open(ARGV0, encoding="utf-8").read().split()
    out = {}
    for kind in ("slow", "pca"):
        a = T.build_parser().parse_args(argv + ["--wm-basis", kind])
        cfg = T.make_config(a)
        torch.manual_seed(0)
        m = M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)
        u, act = planted_batch(torch)
        torch.manual_seed(1)
        loss = m.wm_only_loss({"zbasis": u, "act": act})["terms"]
        loss["total"].backward()
        gproj = float(sum(p.grad.abs().sum() for n, p in m.named_parameters() if n.startswith("wm.proj.") and p.grad is not None))
        gctrl = float(sum(p.grad.abs().sum() for n, p in m.named_parameters() if p.grad is not None and not n.startswith("wm.")))
        gwm = float(sum(p.grad.abs().sum() for n, p in m.named_parameters() if n.startswith("wm.") and p.grad is not None))
        dead = [n for n, p in m.named_parameters() if n.startswith("wm.") and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
        out[kind] = {"argparse_namespace": a.wm_basis, "config": cfg.wm_basis, "model.wm.wm_basis": m.wm.wm_basis, "built_basis": m.wm.built_basis(), "grad_abs_sum_wm.proj": gproj, "grad_abs_sum_wm_total": gwm,
                     "control_grad_abs_sum_outside_wm_must_be_exactly_0.0": gctrl, "wm_params_with_zero_grad": dead, "loss_wm": float(loss["wm"]),
                     "pass": bool(gproj > 0.0 and gctrl == 0.0 and a.wm_basis == cfg.wm_basis == m.wm.wm_basis == kind)}
    res["live"] = out
    res["pass"] = all(v["pass"] for v in out.values())
    return res


# ------------------------------------------------------------------------------------------ control 4: the hash audit
def control4(tree, work, smoke_dir):
    """The builder's only kinematic dependence is the boolean ``ok`` validity mask (``isfinite`` of the kinematic features, as zcache.py). The track values themselves are cross-checked against the cache's pose_grid by ``ClipStore.get_kin``, so
    they cannot be edited on disk; the audit patches ``floors_kin.kin_features`` IN PROCESS: (a) its finite VALUES perturbed (+ 5 x N(0,1)) with the finite pattern kept -> the basis sha256 is unchanged; (b) its validity PATTERN changed (the last
    row of every clip NaN) -> the sha CHANGES (the audit has power; this is its must-go-red arm and shows exactly where ego enters). SMOKE inputs: the 141-clip eval cache, every second clip."""
    sys.path.insert(0, os.path.join(tree, "stack")), sys.path.insert(0, os.path.join(tree, "taniteval")), sys.path.insert(0, HERE)
    os.environ["ZC_REPO"] = tree
    import slow_basis as SB
    from taniteval import floors_kin as FK
    real = FK.kin_features
    SF = "E:/Projects/TanitAD-artifacts/sfresh_inputs"
    rng = np.random.default_rng(5)

    def values(k):
        x = real(k)
        return np.where(np.isfinite(x), x + 5.0 * rng.standard_normal(x.shape), x)

    def pattern(k):
        x = np.array(real(k), copy=True)
        x[-1] = np.nan
        return x
    res, shas, nfr = {}, {}, {}
    for tag, fn in (("real", real), ("values_perturbed", values), ("pattern_changed", pattern)):
        FK.kin_features = fn
        o = os.path.join(work, "c4_" + tag)
        SB.main(["fit", "--root", f"{SF}/cache", "--split", "eval", "--tracks", f"{SF}/inputs/native_tracks.pt", "--out", o, "--half"])
        rec = json.load(open(os.path.join(o, "FIT_RECORD.json")))
        shas[tag], nfr[tag] = rec["basis_sha256"], rec["meta"]["n_frames"]
    FK.kin_features = real
    res.update({"basis_sha256": shas, "n_fit_frames": nfr})
    res["values_perturbed_same_sha"] = shas["real"] == shas["values_perturbed"]
    res["pattern_changed_differs"] = shas["real"] != shas["pattern_changed"]
    res["pass"] = bool(res["values_perturbed_same_sha"] and res["pattern_changed_differs"])
    res["note"] = "SMOKE inputs; the grep half is test_builder_reads_no_ego_action_or_label_array in test_slow_basis.py"
    return res


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--dump-c1":
        return dump_c1(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    res = {"card_sha256": "7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca", "base_commit": "1faa1a068f9f46dcb9d230c6b05efc20dce2fc1c"}
    if a.only in (None, "1"):
        res["control1_off_identity"] = control1_identity(a.tree, a.base, a.work)
    if a.only in (None, "mut"):
        res["mutations"] = {t: mutation_arm(a.tree, a.work, t, d, e, mf) for t, d, e, mf in MUTS}
    if a.only in (None, "2"):
        res["control2_reachability"] = control2(a.tree)
    if a.only in (None, "4"):
        res["control4_echo_hash_audit"] = control4(a.tree, a.work, a.smoke_dir)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else {m: x["red"] for m, x in v.items()} if k == "mutations" else None) for k, v in res.items() if k.startswith(("control", "mutations"))}, indent=1))


if __name__ == "__main__":
    main()
