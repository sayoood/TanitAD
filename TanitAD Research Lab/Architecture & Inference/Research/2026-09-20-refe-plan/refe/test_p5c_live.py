#!/usr/bin/env python3
"""P5-C G-LIVE (f) warm-start identity and (h) lever isolation on the real model (PREREG_P5C.md 10.2), plus the argv half of (h).

  python test_p5c_live.py            everything, incl. the source-mutation harness (minutes, CPU)
  python test_p5c_live.py --fast     no mutation harness
  python test_p5c_live.py --check N  one named check (what the mutation harness runs on a mutated copy of this tree)

CPU only, synthetic vits16 banks, torch + numpy only (NO scipy).  Prints ZZP5CLIVE_PASS <n>.
(f) the build capture's step-0 digest of EVERY tensor == the base file's, the optimiser is fresh, the base file's sha256 == the registered pin; RED arms: a wrong checkpoint, a tensor nudged after the load
    (`warm_perturbed`), a load that never happens (`warm_not_loaded`), a stateful optimiser, a resume file, an unpinned base.
(h) argv(C1) - argv(C0) == {--traj-label-table, --out}; on the real TargetBank rows the table changes item[3] and nothing else, the model's forward is identical, the score loss and its gradient are bit-identical,
    the REAL wta_loss differs, and a control (raw label, same forward) reads EXACTLY 0.0.  Every positive has a negative twin; the MUTATIONS table breaks one line of a COPY of this tree and requires RED.
"""
from __future__ import annotations

import copy
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

import test_traj_label_table as TT  # noqa: E402
import traj_label_table as TLT  # noqa: E402
import p5c_gate as PG  # noqa: E402

N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def _argv(td, extra=()):
    import gate_checks as C
    argv = TT._argv_p6(str(td / "bank"), str(td / "o"), extra)
    return C._set(C._drop(argv, "--steps", 1), "--epochs", "1")


def _base(td, seed: int, name: str) -> str:
    """a FULL-format checkpoint of a fresh synthetic model built with `seed` (what `--init-from` strict-loads)."""
    import ckpt_io
    import gate_checks as C
    b = C.instrumented_run(_argv(td, ["--seed", str(seed)]), mode="build", work=td / f"wb_{name}")
    p = td / f"{name}.pt"
    torch.save({"format": ckpt_io.FORMAT_FULL, "model": ckpt_io.full_state(b.model), "meta": {}}, p)
    return str(p)


def _warm_build(td, base: str, arm=None):
    import gate_checks as C
    return C.instrumented_run(_argv(td, ["--init-from", base, "--seed", "1"]), mode="build", work=td / f"wb_{arm or 'ok'}", arm=arm)


def _profile(arm_name: str, sha):
    import launch_gate as L
    p = dict(L.p5c_profile(arm_name))
    p["init_from_sha256"] = sha
    return p


# ------------------------------------------------------------------------------------------------ (f)
def chk_warm_identity():
    td = Path(tempfile.mkdtemp(prefix="p5cl_f_"))
    try:
        TT.make_bank(td)
        base, other = _base(td, 0, "base"), _base(td, 2, "other")
        sha = PG.file_sha256(base)
        bf = PG.base_facts(base)
        ok(bf["file_sha256"] == sha and bf["format"] == "full" and len(bf["tensors"]) > 10)
        b = _warm_build(td, base)
        ok(b.step0 is not None and b.step0["opt_state_entries"] == 0 and b.step0["ckpt_last_present"] is False and set(b.step0["tensors"]) == set(bf["tensors"]))
        ok(PG.judge_warm_identity(b.step0, bf, sha) == [], PG.judge_warm_identity(b.step0, bf, sha))
        # a build WITHOUT --init-from has a different init (seed 1 vs 0): the identity must read RED against the base (the probe can tell two models apart)
        import gate_checks as C
        n0 = C.instrumented_run(_argv(td, ["--seed", "1"]), mode="build", work=td / "wb_noinit")
        ok(n0.step0 is None, "no --init-from: no step-0 capture (every other profile unchanged)")
        ok(any("no step-0 capture" in r for r in PG.judge_warm_identity(n0.step0, bf, sha)))
        # RED arms through the gate's own instrumented run
        for arm_name, needle in (("warm_not_loaded", "differ from the base"), ("warm_perturbed", "differ from the base")):
            bb = _warm_build(td, base, arm=arm_name)
            r = PG.judge_warm_identity(bb.step0, bf, sha)
            ok(r and any(needle in x for x in r), (arm_name, r))
            if arm_name == "warm_perturbed":
                ok(sum(1 for k in bf["tensors"] if bb.step0["tensors"][k] != bf["tensors"][k]) == 1, "exactly one tensor was nudged")
        # the wrong checkpoint: the build loaded `base`, the profile pins `other`'s sha -> pin row RED; and the digest of `other` differs from what was built
        of = PG.base_facts(other)
        r = PG.judge_warm_identity(b.step0, of, PG.file_sha256(base))
        ok(any("not the registered pin" in x for x in r) and any("differ from the base" in x for x in r), r)
        # pin missing / malformed
        ok(any("registers no base sha256" in x for x in PG.judge_warm_identity(b.step0, bf, None)))
        ok(any("registers no base sha256" in x for x in PG.judge_warm_identity(b.step0, bf, "abc")))
        # a stateful optimiser, a resume file, a missing tensor, an extra tensor
        ok(any("not a fresh AdamW" in x for x in PG.judge_warm_identity(dict(b.step0, opt_state_entries=3), bf, sha)))
        ok(any("resume" in x for x in PG.judge_warm_identity(dict(b.step0, ckpt_last_present=True), bf, sha)))
        t2 = dict(b.step0["tensors"])
        k0 = sorted(t2)[0]
        del t2[k0]
        ok(any("only in base" in x for x in PG.judge_warm_identity(dict(b.step0, tensors=t2), bf, sha)))
        t3 = dict(b.step0["tensors"], extra_tensor=["torch.float32", [1], "0" * 64])
        ok(any("only in model" in x for x in PG.judge_warm_identity(dict(b.step0, tensors=t3), bf, sha)))
        # the digest is over BYTES: one flipped bit in a tensor changes it, a -0.0 vs 0.0 too
        t = torch.zeros(4)
        t2_ = t.clone()
        t2_[0] = -0.0
        ok(PG.tensor_digest(t) != PG.tensor_digest(t2_) and PG.tensor_digest(t) == PG.tensor_digest(t.clone()))
        # through the gate's own function: pass with the pin, RED on each regression arm, RED on a wrong pin
        prof = _profile("C0", sha)
        ok(C.p5c_live_rows(b, b, prof)[0] == [], C.p5c_live_rows(b, b, prof)[0])
        bw = _warm_build(td, base, arm="warm_not_loaded")
        ok(any(x.startswith("(f)") for x in C.p5c_live_rows(bw, bw, prof)[0]))
        ok(any("not the registered pin" in x for x in C.p5c_live_rows(b, b, _profile("C0", PG.file_sha256(other)))[0]))
        ok(any("registers no base sha256" in x for x in C.p5c_live_rows(b, b, _profile("C0", None))[0]))
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ (h)
def _c1_build(td, tp, arm=None):
    import gate_checks as C
    return C.instrumented_run(_argv(td, ["--traj-label-table", tp]), mode="build", work=td / "wb_tab", arm=arm)


def chk_argv_isolation():
    import launch_gate as L
    c1, c0, c0r = L.p5c_argv("C1"), L.p5c_argv("C0"), L.p5c_argv("C0r")

    def flags(a):
        return [t for t in a if t.startswith("--")]

    def pairs(a):
        out, i = {}, 0
        while i < len(a):
            if a[i].startswith("--"):
                v = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
                out[a[i]] = v
                i += 2 if v is not None else 1
            else:
                i += 1
        return out
    p1, p0, pr = pairs(c1), pairs(c0), pairs(c0r)
    d10 = {k for k in set(p1) | set(p0) if p1.get(k) != p0.get(k)}
    ok(d10 == {"--traj-label-table", "--out"}, d10)
    ok("--traj-label-table" not in p0 and "--traj-label-table" in p1)
    d00r = {k for k in set(pr) | set(p0) if pr.get(k) != p0.get(k)}
    ok(d00r == {"--seed", "--out"}, d00r)
    ok(sorted(set(flags(c1)) - set(flags(c0))) == ["--traj-label-table"] and set(flags(c0)) <= set(flags(c1)), "C1's flag set is C0's plus the table flag, nothing else")
    # negative twin: any extra difference is seen (a different lr in C1)
    bad = list(c1)
    bad[bad.index("--lr") + 1] = "2e-4"
    ok({k for k in set(pairs(bad)) | set(p0) if pairs(bad).get(k) != p0.get(k)} != {"--traj-label-table", "--out"})
    # and the gate itself refuses an arm whose non-path lever moved
    prof = dict(L.PROFILES["refe_p5c"], init_from_sha256="a" * 64, traj_label_sha256="b" * 64, traj_label_rank_policy="all")
    ok(any("lr" in r for r in L.p5c_refusals(prof, bad)), L.p5c_refusals(prof, bad))


def chk_isolation_probe():
    td = Path(tempfile.mkdtemp(prefix="p5cl_h_"))
    try:
        bank = TT.make_bank(td)
        rows = TLT.read_bank_rows(bank)
        tp = TT.make_table(td, rows)
        b = _c1_build(td, tp)
        ds = b.bank_objs[0]
        ok(ds.traj_label is not None and len(b.bank_objs) >= 1)
        raw = copy.copy(ds)
        raw.traj_label, raw.traj_label_sha256, raw.traj_label_cov = None, None, None
        idxs = PG.pick_rows(ds, 2)
        ok(len(idxs) == 2)
        f = PG.isolation_facts(b.model, ds, raw, idxs)
        bad = PG.judge_isolation(f)
        ok(bad == [], (bad, f))
        ok(f["other_slots_equal"] and f["item3_differs"] == 2 and f["forward_equal"] and f["score_loss_table"] == f["score_loss_raw"] and f["score_grad_diff_max"] == 0.0)
        ok(f["traj_loss_diff"] != 0.0 and f["control_traj_loss_diff"] == 0.0 and f["control_traj_grad_diff_max"] == 0.0)
        # NEGATIVE TWINS (each must make the judge RED)
        f0 = PG.isolation_facts(b.model, raw, raw, idxs)                                     # the table never reaches item[3]: label_lookup_removed
        r = PG.judge_isolation(f0)
        ok(f0["item3_differs"] == 0 and any("did not reach item[3]" in x for x in r) and any("never reached the loss" in x for x in r), r)

        class Leaky(type(ds)):                                                                # the lever also moves the ego slot
            def __getitem__(self, i):
                it = super().__getitem__(i)
                return (it[0], it[1] + 1.0) + tuple(it[2:])
        leaky = copy.copy(ds)
        leaky.__class__ = Leaky
        fl = PG.isolation_facts(b.model, leaky, raw, idxs)
        ok(fl["other_slots_equal"] is False and any("OTHER than item[3]" in x for x in PG.judge_isolation(fl)))
        for k, v, needle in (("score_loss_table", f["score_loss_raw"] + 1e-9, "score loss"), ("score_grad_diff_max", 1e-12, "score loss"), ("control_traj_loss_diff", 1e-12, "cannot read zero"),
                             ("control_traj_grad_diff_max", 1e-12, "cannot read zero"), ("traj_loss_diff", 0.0, "never reached the loss"), ("forward_equal", False, "forward differs")):
            ok(any(needle in x for x in PG.judge_isolation(dict(f, **{k: v}))), k)
        ok(PG.judge_isolation(None) != [] and PG.judge_isolation({}) != [])
        # the gate's own function: C1 passes (f off here: no --init-from), a C0 profile on a bank WITH a table is RED, a C1 profile on a bank WITHOUT one is RED
        import gate_checks as C
        prof1 = _profile("C1", None)
        rs, det = C.p5c_live_rows(b, b, prof1)
        ok(not any(x.startswith("(h)") for x in rs) and isinstance(det["isolation"], dict), rs)
        ok(any("must not carry a label table" in x for x in C.p5c_live_rows(b, b, _profile("C0", None))[0]))
        nb = C.instrumented_run(_argv(td), mode="build", work=td / "wb_notab")
        ok(any("carries no label table" in x for x in C.p5c_live_rows(nb, nb, prof1)[0]))
        ok(not any(x.startswith("(h)") for x in C.p5c_live_rows(nb, nb, _profile("C0", None))[0]))
        # the gate's regression arm label_table_not_passed: the live bank has no table -> (h) RED
        ba = _c1_build(td, tp, arm="traj_label_not_passed")
        ok(any(x.startswith("(h)") for x in C.p5c_live_rows(ba, ba, prof1)[0]))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_no_scipy():
    import re
    src = Path(__file__).read_text(encoding="utf-8") + (HERE / "p5c_gate.py").read_text(encoding="utf-8")
    ok(re.search(r"^\s*(import|from)\s+scipy", src, re.M) is None)


CHECKS = {"warm_identity": chk_warm_identity, "argv_isolation": chk_argv_isolation, "isolation_probe": chk_isolation_probe, "no_scipy": chk_no_scipy}

LOOKUP_LINE = '        tgt_t = torch.tensor(self.traj_label[i] if (self.traj_label is not None and i in self.traj_label) else r["traj"], dtype=torch.float32)\n'
# (file, old, new, the check that must go RED): each breaks ONE emitting line of a COPY of this tree
MUTATIONS = [
    ("train.py", LOOKUP_LINE, '        tgt_t = torch.tensor(r["traj"], dtype=torch.float32)\n', "isolation_probe"),                                   # the table never reaches item[3]
    ("p5c_gate.py", "for j in range(n) if j != 3)", "for j in range(n) if False)", "isolation_probe"),                                              # the other-slots check is gone
    ("p5c_gate.py", "    lt_r, gt_r = traj_loss(tr_r, br[3])\n", "    lt_r, gt_r = traj_loss(tr_r, bt[3])\n", "isolation_probe"),                                    # the raw arm reads the table label: no difference
    ("p5c_gate.py", "    lt_c, gt_c = traj_loss(tr_t, br[3])", "    lt_c, gt_c = traj_loss(tr_t, bt[3])", "isolation_probe"),                                      # the control no longer reads zero
    ("p5c_gate.py", '    elif base["file_sha256"] != pin_sha256:\n', "    elif False:\n", "warm_identity"),                                                          # the pin is no longer compared
    ("p5c_gate.py", "    diff = sorted(k for k in ref if k in built and built[k] != ref[k])\n", "    diff = []\n", "warm_identity"),                                  # the tensor comparison is gone
    ("p5c_gate.py", '    if step0.get("opt_state_entries") != 0:\n', "    if False:\n", "warm_identity"),                                                      # a stateful optimiser passes
    ("gate_checks.py", "                    cap.step0 = PG.step0_facts(cap.model, self, a.out)\n", "                    pass\n", "warm_identity"),                           # the build capture no longer digests the model
    ("gate_checks.py", '        if arm_is("warm_not_loaded", arm):', '        if False:', "warm_identity"),                                                          # the regression arm is inert
    ("gate_checks.py", "        rs_, dt_ = p5c_live_rows(cap, build_cap, profile)         # PREREG_P5C 10.2 (f) warm-start identity and (h) lever isolation on the real model", "        rs_, dt_ = [], {}         # PREREG_P5C 10.2 (f) warm-start identity and (h) lever isolation on the real model", "wiring"),                                         # (f)/(h) not wired into G-LIVE
    ("launch_gate.py", "build_cap=b))", "build_cap=None))", "wiring"),                                                                              # G-LIVE is given no build capture
]


def chk_wiring():
    """check_live hands the build capture to p5c_live_rows for a P5-C profile (and only for one), and run_checks builds that capture even when G-DVB / G-CLOCK are not wanted and passes it on."""
    from types import SimpleNamespace as NS
    import gate_checks as C
    seen = []
    orig = C.p5c_live_rows
    C.p5c_live_rows = lambda cap, bc, prof: (seen.append(bc) or (["SENTINEL (f)"], {}))
    try:
        fake = NS(rc=0, metrics=[{"traj_L1": 0.1}], grads=[], args=NS(scorer_mode="fixed", labeller="bank", w_field=0.0, aux_drv=False), model=None, first_batch=None)
        bc = object()
        ev = C.check_live([], fake, {"live_terms": (), "p5c_rules": True}, build_cap=bc)
        ok(any("SENTINEL" in r for r in ev["reasons"]) and seen == [bc] and ev["status"] == "FAIL")
        ev2 = C.check_live([], fake, {"live_terms": ()}, build_cap=bc)
        ok(not any("SENTINEL" in r for r in ev2["reasons"]) and seen == [bc], "a non-P5-C profile never reaches the P5-C probes")
    finally:
        C.p5c_live_rows = orig
    src = (HERE / "launch_gate.py").read_text(encoding="utf-8")
    ok("build_cap=b))" in src and 'or bool("G-LIVE" in want and prof.get("p5c_rules"))' in src)


CHECKS["wiring"] = chk_wiring


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="p5clmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:60]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_p5c_live.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    print(f"ZZP5CLIVE_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
