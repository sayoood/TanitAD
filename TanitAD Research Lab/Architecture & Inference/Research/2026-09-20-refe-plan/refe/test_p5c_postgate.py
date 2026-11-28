#!/usr/bin/env python3
"""P5-C post-training gate G-ADAM (PREREG_P5C.md 10.4) -- tests of `p5c_postgate.py`.

  python test_p5c_postgate.py            everything, incl. the source-mutation harness
  python test_p5c_postgate.py --fast     no mutation harness
  python test_p5c_postgate.py --check N  one named check

CPU only, synthetic vits16 (the REAL train.py trains base -> C1 (table) and C0 for 3 steps; the gate then reads their ckpt_last.pt / model_final.pt), torch + numpy only (NO scipy).  Prints ZZP5CPOST_PASS <n>.
Pattern of probe_zerograd3.py: a registered parameter without Adam state never received a gradient.  Every positive has a negative twin: a state entry deleted, a first moment zeroed, a wrong step, a wrong
count, a trunk tensor moved, an arm that trained nothing, two identical arms; the controls must read EXACTLY 0.0.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import p5c_postgate as PG  # noqa: E402
import test_traj_label_table as TT  # noqa: E402
import traj_label_table as TLT  # noqa: E402

N = 0
STEPS = 3


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def _run(td: Path, bank: str, name: str, extra) -> Path:
    out = td / name
    argv = [sys.executable, str(HERE / "train.py")] + TT._argv_p6(bank, str(out))
    argv[argv.index("--steps") + 1] = str(STEPS)
    r = subprocess.run(argv + list(extra), capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(HERE), env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"))
    ok(r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:])
    return out


def _arms(td: Path):
    bank = TT.make_bank(td)
    tp = TT.make_table(td, TLT.read_bank_rows(bank))
    base = _run(td, bank, "base", ["--seed", "0"])
    b = str(base / "model_final.pt")
    c1 = _run(td, bank, "c1", ["--init-from", b, "--seed", "0", "--traj-label-table", tp])
    c0 = _run(td, bank, "c0", ["--init-from", b, "--seed", "0"])
    return b, c1, c0


def _cli(out, base, *extra):
    r = subprocess.run([sys.executable, str(HERE / "p5c_postgate.py"), str(out), "--base", base] + list(extra), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"))
    return r.returncode, r.stdout + r.stderr


def chk_postgate():
    td = Path(tempfile.mkdtemp(prefix="p5cpg_"))
    try:
        ok(PG.controls() == [], "the probe's own controls (exact 0.0 / > 0 / no state) hold")
        b, c1, c0 = _arms(td)
        cfg = json.loads((c1 / "config.json").read_text(encoding="utf-8"))
        n = cfg["params"]["trainable"]
        ck = torch.load(c1 / "ckpt_last.pt", map_location="cpu", weights_only=False)
        opt = ck["opt"]
        reg = [p for g in opt["param_groups"] for p in g["params"]]
        ok(PG.adam_numel(opt) == n and len(opt["state"]) == len(reg) > 10, "the real trainer's state covers exactly its printed trainable count")
        ok(PG.judge_adam(opt, STEPS, n) == [], PG.judge_adam(opt, STEPS, n))
        # --- negatives on the Adam state
        o = copy.deepcopy(opt)
        del o["state"][reg[3]]
        r = PG.judge_adam(o, STEPS, n)
        ok(any("NO Adam state" in x for x in r) and any("covers" in x for x in r), r)          # (the probe_zerograd3 pattern: registered 229, with state 228)
        o = copy.deepcopy(opt)
        o["state"][reg[5]]["exp_avg"].zero_()
        ok(any("EXACTLY ZERO first moment" in x for x in PG.judge_adam(o, STEPS, n)))
        o = copy.deepcopy(opt)
        o["state"][reg[5]]["exp_avg_sq"].zero_()
        ok(any("EXACTLY ZERO exp_avg_sq" in x for x in PG.judge_adam(o, STEPS, n)))
        o = copy.deepcopy(opt)
        o["state"][reg[2]]["step"] = torch.tensor(STEPS - 1.0)
        ok(any("step count" in x for x in PG.judge_adam(o, STEPS, n)))
        o = copy.deepcopy(opt)
        o["state"][reg[2]]["exp_avg"][0] = float("nan")
        ok(any("non-finite" in x for x in PG.judge_adam(o, STEPS, n)))
        ok(any("covers" in x for x in PG.judge_adam(opt, STEPS, n + 1)) and any("covers" in x for x in PG.judge_adam(opt, STEPS, n - 1)), "the count is EXACT, both directions")
        ok(any("step count" in x for x in PG.judge_adam(opt, STEPS + 1, n)), "the arm's step count is exact")
        ok(PG.judge_adam(None, STEPS, n) != [] and PG.judge_adam({}, STEPS, n) != [])
        # --- weights
        base_sd, f1, f0 = (PG._state(torch.load(p, map_location="cpu", weights_only=False)) for p in (b, c1 / "model_final.pt", c0 / "model_final.pt"))
        ok(PG.judge_weights(f1, base_sd) == [] and PG.judge_weights(f0, base_sd) == [], (PG.judge_weights(f1, base_sd), PG.judge_weights(f0, base_sd)))
        ok(PG.judge_pair(f1, f0) == [], "C1 and C0 differ in the traj head (the table changed the targets)")
        frozen = [k for k in f1 if PG.is_frozen(k) and f1[k].dtype.is_floating_point]
        ok(len(frozen) > 5 and not any(PG.is_frozen(k) for k in f1 if k.startswith("traj_head.")))
        bad_f = dict(f1)
        bad_f[frozen[0]] = f1[frozen[0]] + 1e-3
        ok(any("FROZEN-trunk" in x for x in PG.judge_weights(bad_f, base_sd)), "a moved trunk tensor is RED")
        ok(any("trained nothing" in x or "did not train" in x for x in PG.judge_weights(dict(base_sd), base_sd)), "an arm equal to its base is RED")
        ok(any("did not train" in x for x in PG.judge_weights({k: (base_sd[k] if k.startswith("traj_head.") else v) for k, v in f1.items()}, base_sd)), "a traj head equal to the base is RED")
        ok(any("do not differ" in x or "bit-identical" in x for x in PG.judge_pair(f1, dict(f1))), "two identical arms are RED")
        ok(any("names differ" in x for x in PG.judge_weights({k: v for k, v in f1.items() if k != frozen[0]}, base_sd)))
        # --- the CLI (what the chain runs), positive and negative twins
        rc, out = _cli(c1, b, "--steps", str(STEPS), "--n-params", str(n), "--other", str(c0))
        ok(rc == 0 and "ZZP5C_POSTGATE_OK" in out and "FAIL" not in out, out)
        rc, out = _cli(c1, b, "--steps", str(STEPS), "--n-params-from", str(c0 / "config.json"))
        ok(rc == 0 and "ZZP5C_POSTGATE_OK" in out, out)
        rc, out = _cli(c1, b, "--steps", str(STEPS + 1), "--n-params", str(n))
        ok(rc == 1 and "ZZP5C_POSTGATE_FAIL" in out and "step count" in out, out)
        rc, out = _cli(c1, b, "--steps", str(STEPS), "--n-params", str(n + 1))
        ok(rc == 1 and "ZZP5C_POSTGATE_FAIL" in out, out)
        rc, out = _cli(c1, b, "--steps", str(STEPS))
        ok(rc == 1 and "never defaulted" in out, out)
        rc, out = _cli(c1, str(c1 / "model_final.pt"), "--steps", str(STEPS), "--n-params", str(n))
        ok(rc == 1 and "ZZP5C_POSTGATE_FAIL" in out, "a run judged against ITSELF as the base trained nothing")
        rc, out = _cli(c1, b, "--steps", str(STEPS), "--n-params", str(n), "--other", str(c1))
        ok(rc == 1 and "arms do not differ" in out, out)
        # config.json says which base the run loaded: judged against ANOTHER base file (a perturbed copy) the sha256 does not match
        other_base = td / "other_base.pt"
        sd_o = torch.load(b, map_location="cpu", weights_only=False)
        k_ = sorted(sd_o["model"])[0]
        sd_o["model"][k_] = sd_o["model"][k_] + 1e-3
        torch.save(sd_o, other_base)
        rc, out = _cli(c1, str(other_base), "--steps", str(STEPS), "--n-params", str(n))
        ok(rc == 1 and "warm_start.sha256" in out, out)
        rc, out = _cli(td / "nope", b, "--steps", str(STEPS), "--n-params", str(n))
        ok(rc == 1 and "does not exist" in out, out)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_no_scipy():
    import re
    src = Path(__file__).read_text(encoding="utf-8") + (HERE / "p5c_postgate.py").read_text(encoding="utf-8")
    ok(re.search(r"^\s*(import|from)\s+scipy", src, re.M) is None)


CHECKS = {"postgate": chk_postgate, "no_scipy": chk_no_scipy}

MUTATIONS = [
    ("p5c_postgate.py", "        if float(m.abs().sum()) == 0.0:\n", "        if False:\n", "postgate"),                                  # a zero first moment passes
    ("p5c_postgate.py", "        if float(v.abs().sum()) == 0.0:\n", "        if False:\n", "postgate"),                                  # a zero exp_avg_sq passes
    ("p5c_postgate.py", "    if numel != n_params:\n", "    if False:\n", "postgate"),                                                  # the count is no longer exact
    ("p5c_postgate.py", "    if len(with_state) != len(registered):\n", "    if False:\n", "postgate"),                                       # a stateless registered parameter passes
    ("p5c_postgate.py", '        if float(st.get("step", -1)) != float(steps):\n', "        if False:\n", "postgate"),                                  # the step count is not judged
    ("p5c_postgate.py", "    if moved_frozen:\n", "    if False:\n", "postgate"),                                                      # a moved trunk passes
    ("p5c_postgate.py", "    elif all(torch.equal(final[k], base[k]) for k in th):\n", "    elif False:\n", "postgate"),                                # an untrained head passes
    ("p5c_postgate.py", "    if all(torch.equal(final_a[k], final_b[k]) for k in th):\n", "    if False:\n", "postgate"),                             # identical arms pass
    ("p5c_postgate.py", "        bad += judge_adam(ck.get(\"opt\"), a.steps, n)\n", "        pass\n", "postgate"),                                    # the CLI no longer judges the Adam state
    ("p5c_postgate.py", "            if ws_ != PGfile_sha256(a.base):\n", "            if False:\n", "postgate"),
    ("p5c_postgate.py", '    if 2 in sd["state"]:\n', "    if True:\n", "postgate"),
    ("p5c_postgate.py", '    if float(sd["state"][0]["exp_avg"].abs().sum()) != 0.0:\n', "    if True:\n", "postgate"),
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="p5cpgmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:60]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_p5c_postgate.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
            print(f"  mutation RED as required: {fn} [{old.strip()[:56]}] -> {check}", flush=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        run_mutations()
    print(f"ZZP5CPOST_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
