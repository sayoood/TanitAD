#!/usr/bin/env python3
"""Shared machinery for the DAG-OACC / FLAT-DAG composition checks.

Code under test = the tree at E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a (every .py of stack/ and taniteval/ re-hashed against git tip 0a820606: `verify_tree.py`, 1565/1565 equal).
The REAL trainer (`v8f_mini_train.py`, its in-process gate `--selfcheck-only`) is called as a library, exactly as dagq/real_gate_proof.py does; the world is SYNTHETIC (the real corpus lives on box 2 only),
but the model is built at the REAL dimensions (d 256, n_q 4, nb 12 ...) so the REAL checkpoints M2b_s0 / FLAT_s0 load strictly as `--init-ckpt` / bank generator `--ckpt`.

Mutants: env CG_MUTANT=<name> installs a one-line source mutation IN MEMORY (the file on disk is never touched) before the trainer is imported; the mutation must apply exactly once or the run aborts.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("E:/Projects/TanitAD-artifacts/oaccflat_check")
TREE = Path(os.environ.get("CG_TREE", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"))
WORK = Path(os.environ.get("CG_WORK", str(ROOT / "work")))
ART = Path("E:/Projects/TanitAD-artifacts")
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

M2B_CKPT = {0: "C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s0/ckpt_last.pt", 1: "C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s1/ckpt_last.pt"}
FLAT_CKPT = {0: str(ART / "v8f-mini-flat/M2b_s0_flat.pt"), 1: str(ART / "v8f-mini-flat/M2b_s1_flat.pt")}
SHA_M2B = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
SHA_FLAT_PREFIX = {0: "8496934a", 1: "6fdfc9b0"}
DAG_ARGV = {s: json.load(open(ART / f"v8f-dagger/M2b_s{s}_dag/argv.json", encoding="utf-8"))["argv"] for s in (0, 1)}
FLAT_ARGV = {s: json.load(open(ROOT / f"flat_runs/flat_s{s}_argv.json", encoding="utf-8"))["argv"] for s in (0, 1)}
W_ACC = 49.6969

# ---- in-memory source mutants (name -> (module, path under TREE, old, new)); each `old` must occur EXACTLY once
MUTANTS = {
    "acc_line": ("tanitad.models.v8f_mini", "stack/tanitad/models/v8f_mini.py",
                 'total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0 else total', "total = total"),
    "swap_apply": ("tanitad.data.v8f_dagger", "stack/tanitad/data/v8f_dagger.py",
                   "out.update(upd)                                                           # the ONE line that applies the swap: delete it and `out` is `batch`", "pass"),
    "cut_ck": ("tanitad.models.v8f_mini", "stack/tanitad/models/v8f_mini.py",
               "ck_for_o = self.ck_nom[None].expand(B, -1, -1)               # S's checkpoints for O = the nominal arcs ...", "pass"),
    "cut_code": ("tanitad.models.v8f_mini", "stack/tanitad/models/v8f_mini.py",
                 "code = torch.zeros_like(code)                                # T's code = zeros, AFTER every override (the teacher forcing and forward_diag's code_override)", "pass"),
    "loop_dag_apply": ("v8f_mini_train", "stack/scripts/v8f_mini_train.py", None, None),     # resolved at install time (the loop's `dag.apply` line)
}


def install_mutant(name: str):
    if not name:
        return None
    modname, rel, old, new = MUTANTS[name]
    path = TREE / rel
    src = path.read_text(encoding="utf-8")
    if old is None:                                                  # the training loop's call: find the single line `b = dag.apply(b, step)`
        import re
        hits = [m for m in re.finditer(r"^(?P<ind>[ \t]*)(?P<stmt>b\s*=\s*dag\.apply\(b,\s*step\)[^\n]*)$", src, flags=re.M)]
        assert len(hits) == 1, f"mutant {name}: expected exactly 1 loop line `b = dag.apply(b, step)`, found {len(hits)}"
        h = hits[0]
        new_src = src[:h.start()] + h.group("ind") + "pass" + src[h.end():]
        desc = h.group("stmt")[:80]
    else:
        assert src.count(old) == 1, f"mutant {name}: the target line occurs {src.count(old)} times in {rel}, expected exactly 1"
        new_src = src.replace(old, new)
        desc = old[:80]
    assert new_src != src
    if "." in modname:
        importlib.import_module(modname.rsplit(".", 1)[0])
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    exec(compile(new_src, str(path), "exec"), mod.__dict__)
    if "." in modname:
        setattr(sys.modules[modname.rsplit(".", 1)[0]], modname.rsplit(".", 1)[1], mod)
    return {"mutant": name, "module": modname, "file": rel, "deleted_or_replaced": desc}


MUT_RECORD = install_mutant(os.environ.get("CG_MUTANT", ""))

import torch  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
SYN_T = 90
_orig_world = D.make_synthetic_world
D.make_synthetic_world = lambda root, n_train=6, n_eval=3, T=70, seed=0, **kw: _orig_world(root, n_train=n_train, n_eval=n_eval, T=SYN_T, seed=seed, **kw)

res = {"cases": [], "t0": time.time(), "mutant": MUT_RECORD, "tree": str(TREE)}


def case(name, ok, detail=""):
    res["cases"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:900]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:400]}]" if detail else ""), flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ real argv -> the gate's argv on the synthetic world
DROP = {"--cache-dir", "--v9-train", "--v9-train-md5", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--exclude"}     # data paths: the synthetic world replaces them
MAPPED = {"--out", "--batch", "--device", "--init-ckpt", "--dagger-bank", "--dagger-bank-sha256"}


def pairs(argv):
    """['--a','1','--b','--c','2'] -> [('--a','1'),('--b',None),('--c','2')] (every flag of the trainer takes one value or none)."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        assert t.startswith("--"), f"unexpected token {t!r}"
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.append((t, argv[i + 1]))
            i += 2
        else:
            out.append((t, None))
            i += 1
    return out


def to_gate_argv(real_argv, out_dir, *, init_ckpt, bank=None, bank_sha=None, extra=(), drop_flags=()):
    """The REAL argv with ONLY the data/path/device/batch flags re-pointed at the synthetic world (and --synthetic, --threads 1 added).
    Every other flag is carried over token for token, in order. `extra` = [(flag, value)] inserted right after --grid (the position queue stages use for --w-acc / --o-interface)."""
    argv = []
    for f, v in pairs(real_argv):
        if f in drop_flags:
            continue
        if f in DROP:
            continue
        if f == "--out":
            v = str(out_dir)
        elif f == "--batch":
            v = "16"
        elif f == "--device":
            v = "cpu"
        elif f == "--init-ckpt":
            v = str(init_ckpt)
        elif f == "--dagger-bank":
            v = str(bank)
        elif f == "--dagger-bank-sha256":
            v = str(bank_sha)
        argv += [f] if v is None else [f, v]
        if f == "--grid":
            for ef, ev in extra:
                argv += [ef] if ev is None else [ef, ev]
    return argv + ["--synthetic", "--threads", "1"]


def carried_verbatim(real_argv, gate_argv, *, allowed_extra=()):
    """Assert every non-data flag of the real argv is in the gate argv with the same value, and that nothing else was added (besides --synthetic/--threads/allowed_extra)."""
    r, g = dict(pairs(real_argv)), dict(pairs(gate_argv))
    bad = []
    for f, v in r.items():
        if f in DROP:
            continue
        if f in MAPPED:
            if f not in g:
                bad.append(f"{f} missing")
            continue
        if g.get(f, "<absent>") != v:
            bad.append(f"{f}: real {v!r} gate {g.get(f, '<absent>')!r}")
    added = sorted(set(g) - set(r) - {"--synthetic", "--threads"} - set(allowed_extra))
    return bad, added


def gate(out, argv):
    """TR.main([... --selfcheck-only]) -> (rc or SystemExit text, selfcheck dict or None)."""
    full = list(argv) + ["--selfcheck-only"]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "gate_argv.json").write_text(json.dumps(full, indent=0), encoding="utf-8")
    sc_path = Path(out) / "selfcheck.json"
    if sc_path.exists():
        sc_path.unlink()
    try:
        rc = TR.main(full)
    except SystemExit as e:
        rc = e.code
    except Exception as e:  # noqa: BLE001
        rc = f"EXC {type(e).__name__}: {str(e)[:500]}"
    shutil.rmtree(Path(out) / "_synthetic", ignore_errors=True)       # the gate's own 29 MB synthetic world (E: is exFAT with 1 MiB clusters and ~3 GB free): selfcheck.json + gate_argv.json are the record
    return rc, (json.loads(sc_path.read_text(encoding="utf-8")) if sc_path.exists() else None)


# ------------------------------------------------------------------------------------------------ the synthetic world + a bank made by a REAL checkpoint through the REAL generator
def make_world(root, seed=0):
    root = Path(root)
    if not (root / "eval_grid_windows.json").exists():
        D.make_synthetic_world(str(root), n_train=6, n_eval=2, T=60, seed=seed)
    return root


def make_bank(world, ckpt, out_dir, n_shards=2, extra_stub=()):
    """v8f_dagger_queue_tool.py bank (the queue's own tool: N x `v8f_dagger_bank.py gen --shard K` + the real merge) from `ckpt` on the synthetic world. -> (bank dir, sha256 of bank.json, stdout, rc)."""
    out_dir = Path(out_dir)
    stub = out_dir.parent / (out_dir.name + "_m2b_argv.json")
    stub.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"argv": ["--arm", "M2b", "--seed", "0", "--out", "x", "--kin-mode", "yaw", "--batch", "8", *extra_stub]}, open(stub, "w"))
    tool = TREE / "stack" / "scripts" / "v8f_dagger_queue_tool.py"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, str(tool), "bank", "--n-shards", str(n_shards), "--out-dir", str(out_dir), "--poll-s", "1", "--", "--ckpt", str(ckpt), "--m2b-argv", str(stub),
           "--cache-dir", str(world), "--tracks", str(world / "native_tracks.pt"), "--join-train", str(world / "join_train.npz"), "--eval-grid", str(world / "eval_grid_windows.json"),
           "--batch", "8", "--threads", "1", "--device", "cpu", "--replay-frac", "0.1"]
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(TREE))
    log = p.stdout + "\n--stderr--\n" + p.stderr
    sha = None
    if p.returncode == 0 and (out_dir / "bank.json").exists():
        sha = sha256_file(out_dir / "bank.json")
    return out_dir, sha, log, p.returncode, time.time() - t


def finish(tag):
    res["seconds"] = time.time() - res["t0"]
    res["n_pass"] = sum(c["ok"] for c in res["cases"])
    res["n_cases"] = len(res["cases"])
    res["all_pass"] = res["n_pass"] == res["n_cases"] and res["n_cases"] > 0
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    name = tag + (("__MUT_" + res["mutant"]["mutant"]) if res.get("mutant") else "")
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"{name}: {res['n_pass']}/{res['n_cases']} pass in {res['seconds']:.0f}s", flush=True)
    return 0 if res["all_pass"] else 1


def dsum(ga, gb, pre=""):
    return float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))


# ------------------------------------------------------------------------------------------------ probes that sit BESIDE the trainer's own (called from inside the real gate flow)
COMPOSED = {}


def composed_probe(model, batch, cfg, dag, n_steps=400):
    """DAG-OACC: the acceleration term on a batch the DAgger swap has REPLACED. Run inside the real gate (monkeypatched after the trainer's own dagger_probe), on a deep copy.
      share      mean n_selected / n_eligible of `dag.apply(batch, step)` at the DECLARED p over n_steps steps (the loop's own call and seed schedule)
      on swapped acc(swapped) finite and > 0 and != acc(logged) (the swap reaches the acc term); total(w) - total(0) == w x acc on the swapped batch; head_o gradient the term gives, on the swapped batch
      controls   same loss twice at w = 0 differs by EXACTLY 0.0 (grads too); prior grad diff EXACTLY 0.0; the term's contribution at the BUILT weight when the BUILT weight is 0 is EXACTLY 0.0"""
    out = {}
    sel = el = rows = 0
    for step in range(n_steps):
        dag.apply(batch, step)
        sel += dag.last["n_selected"]
        el += dag.last["n_eligible"]
        rows += dag.last["n_rows"]
    out["share_at_declared_p"] = (sel / el) if el else None
    out["share_n_eligible_draws"] = el
    out["share_n_rows"] = rows
    out["p_declared"] = dag.p
    m = copy.deepcopy(model).cpu()
    m.train()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    built_w = float(cfg.w_acc)
    pert = dag.apply(batch, dag.PROBE_STEP, p=1.0)

    def run(b, w):
        m.cfg.w_acc = w
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261027)
        T = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"composed probe: loss term {k} is not finite")
        T["total"].backward()
        g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), float(T["acc"].detach()), g, gp

    w = built_w if built_w > 0.0 else 1.0           # the probe weight (the BUILT one; 1.0 when the build is w = 0, as the trainer's acc_probe does)
    t_w, acc_sw, g_w, gp_w = run(pert, w)
    t_0, acc_sw0, g_0, gp_0 = run(pert, 0.0)
    t_0b, _, g_0b, _ = run(pert, 0.0)
    t_l, acc_log, g_l, _ = run(batch, w)
    t_l0, _, g_l0, _ = run(batch, 0.0)
    t_b, _, g_b, _ = run(pert, built_w)               # the loss at the BUILT weight, as the loop evaluates it
    out.update({
        "built_w_acc": built_w, "w_probe": w,
        "acc_on_swapped": acc_sw, "acc_on_logged": acc_log, "acc_swapped_minus_logged": acc_sw - acc_log,
        "total_w_swapped": t_w, "total_0_swapped": t_0, "contribution_swapped": t_w - t_0, "w_x_acc_swapped": w * acc_sw,
        "head_o_grad_from_term_on_swapped": dsum(g_w, g_0, "head_o."), "read_o_grad_from_term_on_swapped": dsum(g_w, g_0, "read_o."),
        "head_o_grad_from_term_on_logged": dsum(g_l, g_l0, "head_o."),
        "swap_moves_total": t_0 - t_l0,
        "control_repeat_total_diff": abs(t_0b - t_0), "control_repeat_grad_diff": dsum(g_0, g_0b, ""),
        "control_prior_grad_diff": float((gp_w - gp_0).abs().sum()),
        "contribution_at_BUILT_weight": t_b - t_0, "head_o_grad_diff_at_BUILT_weight": dsum(g_b, g_0, "head_o."),
    })
    return out


_orig_dagger_probe = TR.dagger_probe


def dagger_probe_plus(model, batch, cfg, dag):
    r = _orig_dagger_probe(model, batch, cfg, dag)
    if dag.active:
        COMPOSED.clear()
        COMPOSED.update(composed_probe(model, batch, cfg, dag))
    return r


TR.dagger_probe = dagger_probe_plus


def grad_probe_cut(model, batch, cfg):
    """FLAT-DAG: gradient readings the trainer's gate does not take. On a deep copy, one backward of the trainer's own loss:
         e_code.weight   exact gradient sum (the cut hands O code = zeros: d/dW (W . 0) = 0 EXACTLY under cut; > 0 under full)
         e_code.bias     reported (a bias is a learned CONSTANT, still a constant under the cut)
         e_ck            same as e_code
         S / T groups    gradient the O-PLAN losses (ade/fde/acc) give them: computed by backward of ONLY the traj terms -> exactly 0.0 under cut (O does not read them), > 0 under full"""
    m = copy.deepcopy(model).cpu()
    m.train()
    named = {n: p for n, p in m.named_parameters() if p.requires_grad}
    for p in named.values():
        p.grad = None
    torch.manual_seed(20261027)
    T = m.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
    ade, fde = T["ade"], T["fde"]
    traj_loss = cfg.w_traj * (ade / 3.0 + 0.5 * fde / 6.0)
    grads = torch.autograd.grad(traj_loss, list(named.values()), allow_unused=True)
    g = {n: (x.detach().clone() if x is not None else torch.zeros_like(named[n])) for n, x in zip(named, grads)}
    pre = lambda *ps: float(sum(g[n].abs().sum() for n in g if any(n.startswith(q) for q in ps)))      # noqa: E731
    out = {"traj_loss_only": True, "e_code.weight": float(g["e_code.weight"].abs().sum()), "e_code.bias": float(g["e_code.bias"].abs().sum()),
           "e_ck.weight": float(g["e_ck.weight"].abs().sum()), "e_ck.bias": float(g["e_ck.bias"].abs().sum()),
           "S_groups(read_s,head_s)": pre("read_s", "head_s"), "T_groups(read_t,trunk_t,h_goal)": pre("read_t", "trunk_t", "h_goal"),
           "O_groups(read_o,head_o)": pre("read_o", "head_o")}
    return out
