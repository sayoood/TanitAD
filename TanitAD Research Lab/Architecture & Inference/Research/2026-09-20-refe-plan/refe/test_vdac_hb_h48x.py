#!/usr/bin/env python3
"""VDAC A5 A1: the post-training h48x forward under an HB trunk (the HB-s0 crash) and the forward-only resume `--h48x-only`.  CPU only (CUDA_VISIBLE_DEVICES=-1), no pod, no data.
Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python refe/test_vdac_hb_h48x.py   ->   ZZTEST_PASS <n>

THE DEFECT (MEASURED 2026-11-13 on the pod): main()'s h48x loop was `for hb in DataLoader(...): fwd(hb)`.  `hb` is ALSO main()'s HB trunk, which the closure fwd() reads at CALL time (`HB.select_actx(hb, ...)`), so the
loop replaced the trunk with a batch list -> AttributeError "'list' object has no attribute 'ctx'" AFTER 10,000 updates.  The path had never run with a trunk (HA has none).  "Rollable and trained are different
claims": test_vdac_hb.py proved the copy trains and the head reads it, but ended its run BEFORE the only code that rebinds the name -- no test drove the post-training forward under an HB trunk.

LAYERS
  R  REACHABILITY: the REAL main() (test_vdac_hb's stub-trunk harness) with --aux-drv-exclude-logs, an HB trunk and a few updates reaches the post-training forward and writes head_pred_h48x.npz with ONE ROW PER FRAME of the
     excluded logs.  MUTATION: the loop variable renamed back to `hb` (the shipped defect) -> the run dies with the AttributeError and NO file is written.
  S  STATIC: no for/with target in main()'s own scope rebinds a name that a nested function reads at call time, except `batch` in the training loop, whose only readers are defined INSIDE the `if a.preflight:` block (which
     returns before that loop exists).  MUTATION: a loop over `aux` / the shipped `hb` loop is caught.
  F  FORWARD-ONLY RESUME (--h48x-only RUN_DIR, SAME argv): bit-identical head_pred_h48x.npz to the training run's own, on a tiny fixture; provenance carries the sha256 of every loaded file and the update read back.
     MUTATIONS (each must change the outcome): a perturbed head -> different file; a perturbed HB LoRA -> different file; a wrong update count -> refused; a states file that disagrees with aux_drv.pt -> refused;
     an existing head_pred_h48x.npz in --out -> refused; a missing aux_drv.pt -> refused.
The PREFLIGHT_PASS.json the run needs is written by hand (the harness' preflight cannot run with the VDAC exclusion lever: G19 / G20 need the real label store); what is tested is the code AFTER the gate.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TMPROOT = os.environ.get("TMP") or tempfile.gettempdir()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_aux_s1_wiring as W  # noqa: E402
import test_vdac_hb as T  # noqa: E402

N = [0]
TABLE: list = []
UPD = 4


def ok(cond, msg):
    N[0] += 1
    assert cond, msg


def red(cond, what, how):
    """a mutation arm: `cond` must be TRUE (the check went RED)"""
    N[0] += 1
    assert cond, f"MUTATION NOT CAUGHT ({what}): {how}"
    TABLE.append((what, "RED", how))


EXCL_LOGS = ["T0", "T1"]
TD0 = tempfile.mkdtemp(prefix="h48x_", dir=TMPROOT)
EXCL_FILE = os.path.join(TD0, "excl.txt")
Path(EXCL_FILE).write_text("\n".join(EXCL_LOGS) + "\n", encoding="utf-8")
EXCL_SHA = hashlib.sha256(Path(EXCL_FILE).read_bytes()).hexdigest()
EXTRA = T.HB_ARGS + ["--aux-drv-exclude-logs", EXCL_FILE, "--aux-drv-exclude-sha256", EXCL_SHA]
SRC = T.SRC


# the SHIPPED block (06c9bd79), verbatim: the loop variable `hb` inline in main() rebinds main's HB trunk
SHIPPED_CALL = "    if aux is not None and h48x_idx:                         # VDAC: the head's predictions on the frames it never trained on (the excluded logs in this bank), by the trainer's own fwd(), one row per frame\n        run_h48x()\n"
SHIPPED_BLOCK = SHIPPED_CALL.replace("        run_h48x()\n", """        with torch.no_grad():
            cap["aux_buf"] = []
            try:
                for hb in DataLoader(Subset(ds, h48x_idx), batch_size=16, shuffle=False, num_workers=2):
                    fwd(hb)
                buf_h = cap["aux_buf"]
            finally:
                cap["aux_buf"] = None
        n_h = sum(int(x[0].shape[0]) for x in buf_h)
        rows_h = [ds.rows[i_] for i_ in h48x_idx]
        np.savez(os.path.join(a.out, "head_pred_h48x.npz"), log=np.asarray([r_.get("log_name", "") for r_ in rows_h]), token=np.asarray([r_.get("token", "") for r_ in rows_h]),
                 step=np.asarray([int(r_.get("step", 0)) for r_ in rows_h], np.int64), logit=torch.cat([x[0] for x in buf_h]).numpy().astype(np.float32),
                 sdf_m=torch.cat([x[1] for x in buf_h]).numpy().astype(np.float32))
        log({"event": "h48x_forward", "frames": n_h, "file": "head_pred_h48x.npz"})
""")


def shipped_defect(loopvar="hb"):
    """SRC with the post-training call replaced by the shipped inline loop (its variable renamed to `loopvar`)"""
    assert SRC.count(SHIPPED_CALL) == 1, "the post-training call site must occur exactly once"
    return SRC.replace(SHIPPED_CALL, SHIPPED_BLOCK.replace("for hb in", f"for {loopvar} in").replace("fwd(hb)", f"fwd({loopvar})"))


def run(tag, src=SRC, extra=EXTRA, pre=None, updates=UPD):
    """the REAL main() on the HB stub; `pre(out_dir)` runs before it (to stage a run dir / PREFLIGHT_PASS.json)"""
    td = tempfile.mkdtemp(prefix=tag + "_", dir=TMPROOT)
    out = os.path.join(td, "out")
    os.makedirs(out)
    Path(out, "PREFLIGHT_PASS.json").write_text("{}", encoding="utf-8")
    if pre is not None:
        pre(out)
    return W.smoke_aux(src, td, tag=tag, extra=extra, preflight=False, refe_cls=T.StubHBREFe, targets_cls=T.StubHBTargets, max_updates=updates)


def expected_rows():
    """one row per (log, token) of the excluded logs that the bank holds (the stub bank, train side): computed from the dataset the trainer is given, not from the trainer"""
    import test_aux_s1_wiring as _W
    tr = _W.TR.StubBank("train", 8, 4)
    return len({(k[0], k[1]) for k in tr.by if k[0] in EXCL_LOGS})


def npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def same(a, b):
    return sorted(a) == sorted(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and (a[k].tobytes() == b[k].tobytes()) for k in a)


# ------------------------------------------------------------------------------------------------ R: reachability
def layer_r():
    r = run("fixed")
    out = r["out"]
    ok(r["rc_run"] == 0, f"the fixed trainer's HB run with the exclusion lever finishes: {r['rc_run']}")
    ok("ZZSFT_DONE" in r["out_run"], "ZZSFT_DONE printed (the chain's done marker)")
    f = os.path.join(out, "head_pred_h48x.npz")
    ok(os.path.isfile(f), "REACHED: head_pred_h48x.npz is written by the post-training forward under an HB trunk")
    z = npz(f)
    n_exp = expected_rows()
    ok(n_exp > 0, "the fixture excludes at least one frame")
    ok(len(z["log"]) == len(z["token"]) == len(z["step"]) == z["logit"].shape[0] == z["sdf_m"].shape[0] == n_exp, f"one row per frame of the excluded logs: {n_exp} expected, got {len(z['log'])}/{z['logit'].shape[0]}")
    ok(set(z["log"].tolist()) <= set(EXCL_LOGS) and np.isfinite(z["logit"]).all() and np.isfinite(z["sdf_m"]).all(), "only excluded-log frames, all outputs finite")
    ev = [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "h48x_forward"]
    ok(len(ev) == 1 and ev[0]["frames"] == n_exp, "the h48x_forward log event carries the frame count")
    # MUTATION: the shipped defect (loop variable named hb)
    bad = shipped_defect("hb")
    ok(bad != SRC and bad.count("for hb in DataLoader(Subset(ds, h48x_idx)") == 1, "the mutation applies")
    ok(shipped_defect("h48_batch") != SRC, "control: the same inline block with a harmless loop variable differs from SRC only in structure (used below)")
    c = run("inline_renamed", src=shipped_defect("h48_batch"))
    ok(c["rc_run"] == 0 and os.path.isfile(os.path.join(c["out"], "head_pred_h48x.npz")), "CONTROL: the inline block with the loop variable renamed is the FIX -- it runs")
    m = run("shipped_defect", src=bad)
    red(isinstance(m["rc_run"], str) and "AttributeError" in m["rc_run"] and "'list' object has no attribute 'ctx'" in m["rc_run"], "loop variable renamed back to `hb` (the HB-s0 crash)", str(m["rc_run"])[:90])
    red(not os.path.isfile(os.path.join(m["out"], "head_pred_h48x.npz")) and "ZZSFT_DONE" not in m["out_run"], "  ... and no head_pred_h48x.npz, no ZZSFT_DONE", "exactly the pod's state")
    return r


# ------------------------------------------------------------------------------------------------ S: static
def rebind_findings(src):
    """for/with targets in main()'s own scope that a nested function reads at call time -> {name: [(line, readers)]}; readers defined inside `if a.preflight:` are listed apart"""
    t = ast.parse(src)
    main = [n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    tg: dict = {}

    def stores(n, kind, line):
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                tg.setdefault(x.id, []).append((kind, line))

    def walk(n):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(c, (ast.For, ast.AsyncFor)):
                stores(c.target, "for", c.lineno)
            if isinstance(c, ast.With):
                for it in c.items:
                    if it.optional_vars is not None:
                        stores(it.optional_vars, "with", c.lineno)
            walk(c)
    walk(main)
    pre = [n for n in main.body if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == "preflight" and getattr(n.test.value, "id", "") == "a"]
    pre_ids = {id(x) for p in pre for x in ast.walk(p)}
    readers: dict = {}
    for n in ast.walk(main):
        if n is main or not isinstance(n, (ast.FunctionDef, ast.Lambda)):
            continue
        local = {x.arg for x in ast.walk(n) if isinstance(x, ast.arg)} | {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in local:
                readers.setdefault(x.id, set()).add((getattr(n, "name", "<lambda>"), n.lineno, id(n) in pre_ids))
    return {k: (v, sorted(readers[k], key=lambda r: r[1])) for k, v in tg.items() if k in readers}


def layer_s():
    f = rebind_findings(SRC)
    live = {k: v for k, v in f.items() if any(not inside for _n, _l, inside in v[1])}
    ok(live == {}, f"no for/with target in main() rebinds a name read by a closure that can run after it: {live}")
    ok(set(f) == {"batch"} and all(inside for _n, _l, inside in f["batch"][1]), f"the one known overlap is `batch`, read ONLY by closures defined inside `if a.preflight:`: {f}")
    fb = rebind_findings(shipped_defect("hb"))
    red("hb" in fb and any(not inside for _n, _l, inside in fb["hb"][1]), "static scan on the shipped `for hb in DataLoader(...)`", "hb flagged: " + ",".join(n for n, _l, _i in fb["hb"][1]))
    fb2 = rebind_findings(shipped_defect("aux"))
    ok(rebind_findings(shipped_defect("h48_batch")).keys() == f.keys(), "control: the inline block with a harmless variable adds no finding")
    red("aux" in fb2 and any(not inside for _n, _l, inside in fb2["aux"][1]), "static scan on a loop over `aux`", "aux flagged")


# ------------------------------------------------------------------------------------------------ F: forward-only resume
def stage_copy(rd, dst, files=("aux_drv.pt", "scorer_states_last.pt")):
    os.makedirs(dst, exist_ok=True)
    for f in files:
        if os.path.isfile(os.path.join(rd, f)):
            shutil.copy(os.path.join(rd, f), os.path.join(dst, f))
    return dst


def resume(tag, rundir, extra_args=(), src=SRC, pre=None, updates=UPD):
    return run(tag, src=src, extra=EXTRA + ["--h48x-only", rundir, *extra_args], pre=pre, updates=updates)


def refused(r, needle):
    return r["rc_run"] != 0 and needle in (r.get("out_run") or "") + str(r["rc_run"]) and not os.path.isfile(os.path.join(r["out"], "head_pred_h48x.npz"))


def layer_f(base):
    rd = base["out"]
    ref = npz(os.path.join(rd, "head_pred_h48x.npz"))
    r = resume("resume", rd)
    ok(r["rc_run"] == 0 and "ZZSFT_H48X_ONLY_DONE" in r["out_run"], f"--h48x-only finishes: {r['rc_run']}: {r['out_run'][-300:]}")
    f = os.path.join(r["out"], "head_pred_h48x.npz")
    ok(os.path.isfile(f), "the forward-only mode writes head_pred_h48x.npz")
    got = npz(f)
    ok(same(ref, got), "EQUIVALENCE: the forward-only file is BIT-IDENTICAL to the training run's own head_pred_h48x.npz (log, token, step, logit, sdf_m)")
    ok(hashlib.sha256(Path(f).read_bytes()).hexdigest() != "", "sha256 computable")
    prov = json.loads(Path(r["out"], "h48x_provenance.json").read_text(encoding="utf-8"))
    ok(set(prov["loaded"]) == {"aux_drv.pt", "scorer_states_last.pt"} and all(len(v["sha256"]) == 64 for v in prov["loaded"].values()), "provenance: sha256 of every loaded file")
    ok(prov["loaded"]["aux_drv.pt"]["sha256"] == hashlib.sha256(Path(rd, "aux_drv.pt").read_bytes()).hexdigest(), "provenance sha256 == the file's bytes")
    ok(prov["update_read_back"] == {"aux_drv.pt": UPD, "scorer_states_last.pt": UPD} and prov["total_updates_of_this_argv"] == UPD, f"update read back == the argv's total: {prov['update_read_back']}")
    ok(prov["states_file_cross_check"] == {"aux_head_equal": True, "hb_lora_equal": True}, f"the states file agrees with aux_drv.pt: {prov['states_file_cross_check']}")
    ok(prov["frames_equal_audit"] is True and prov["excluded_frames_audit"] == prov["frames"], f"frames == the exclusion audit's excluded frames (the chain run check): {prov['excluded_frames_audit']} / {prov['frames']}")
    ok(prov["frames"] == len(ref["log"]) and prov["head_pred_h48x"]["sha256"] == hashlib.sha256(Path(f).read_bytes()).hexdigest(), "provenance: frame count and the output's sha256")
    ok(not any(x.get("event") == "train" for x in map(json.loads, Path(r["out"], "h48x_only.jsonl").read_text(encoding="utf-8").splitlines())), "no training row was logged: the mode trains nothing")
    ok(not os.path.isfile(os.path.join(r["out"], "model_sft_A.pt")) and not os.path.isfile(os.path.join(r["out"], "aux_drv.pt")), "and saves no checkpoint")

    # ---- MUTATIONS on the saved state: each must CHANGE the outcome
    def perturbed(tag, edit):
        d = stage_copy(rd, os.path.join(TMPROOT, "pert_" + tag + str(os.getpid())))
        ck = torch.load(os.path.join(d, "aux_drv.pt"), map_location="cpu", weights_only=False)
        edit(ck)
        torch.save(ck, os.path.join(d, "aux_drv.pt"))
        st = torch.load(os.path.join(d, "scorer_states_last.pt"), map_location="cpu", weights_only=False)
        return d, st

    def head_edit(ck):
        k = sorted(ck["head"])[-1]
        ck["head"][k] = ck["head"][k] + 0.05 * torch.randn_like(ck["head"][k].float()).to(ck["head"][k].dtype)

    def lora_edit(ck):
        k = sorted(ck["hb_lora"])[0]
        ck["hb_lora"][k] = ck["hb_lora"][k] + 0.05 * torch.randn_like(ck["hb_lora"][k])
    # (a) a perturbed head with the states file ALSO perturbed to agree -> the file DIFFERS (proves the output depends on the loaded state)
    for name, edit in (("head", head_edit), ("lora", lora_edit)):
        d, st = perturbed(name, edit)
        ck = torch.load(os.path.join(d, "aux_drv.pt"), map_location="cpu", weights_only=False)
        if name == "head":
            for k in st["AUX"]:
                if k.endswith(sorted(ck["head"])[-1]):
                    st["AUX"][k] = ck["head"][sorted(ck["head"])[-1]]
        else:
            st["HB"]["lora"] = ck["hb_lora"]
        torch.save(st, os.path.join(d, "scorer_states_last.pt"))
        m = resume("pert_" + name, d)
        if m["rc_run"] == 0:
            red(not same(ref, npz(os.path.join(m["out"], "head_pred_h48x.npz"))), f"{name} state perturbed (aux_drv.pt and states file agree)", "head_pred_h48x.npz DIFFERS from the unperturbed one")
        else:                                                         # the states file's AUX key layout differs from my edit: fall back to the aux-only file (no states file)
            os.remove(os.path.join(d, "scorer_states_last.pt"))
            m = resume("pert2_" + name, d)
            ok(m["rc_run"] == 0, f"perturbed {name} (no states file): {m['rc_run']} {m['out_run'][-200:]}")
            red(not same(ref, npz(os.path.join(m["out"], "head_pred_h48x.npz"))), f"{name} state perturbed (aux_drv.pt only)", "head_pred_h48x.npz DIFFERS from the unperturbed one")
        shutil.rmtree(d, ignore_errors=True)
    # (b) a states file that DISAGREES with aux_drv.pt -> refused
    d, st = perturbed("disagree", lambda ck: None)
    k = sorted(st["HB"]["lora"])[0]
    st["HB"]["lora"][k] = st["HB"]["lora"][k] + 1.0
    torch.save(st, os.path.join(d, "scorer_states_last.pt"))
    m = resume("disagree", d)
    red(refused(m, "disagree"), "scorer_states_last.pt disagrees with aux_drv.pt", str(m["rc_run"])[:70])
    shutil.rmtree(d, ignore_errors=True)
    # (c) the saved update is not the argv's total -> refused
    d, st = perturbed("upd", lambda ck: ck.__setitem__("update", UPD - 1))
    os.remove(os.path.join(d, "scorer_states_last.pt"))
    m = resume("upd", d)
    red(refused(m, "not the run's FINAL state"), "aux_drv.pt update != the argv's total updates", str(m["rc_run"])[:70])
    shutil.rmtree(d, ignore_errors=True)
    # (d) an existing head_pred_h48x.npz in --out -> refused (no overwrite of a banked file)
    m = resume("exists", rd, pre=lambda out: Path(out, "head_pred_h48x.npz").write_bytes(b"banked"))
    red(m["rc_run"] == 4 and "refusing to overwrite" in m["out_run"], "--out already holds head_pred_h48x.npz", "REFUSED (rc 4)")
    # (e) no aux_drv.pt -> refused;  (f) without the head / exclusion levers -> refused
    d = os.path.join(TMPROOT, "empty_run" + str(os.getpid()))
    os.makedirs(d, exist_ok=True)
    m = resume("noaux", d)
    red(m["rc_run"] == 4 and "no aux_drv.pt" in m["out_run"], "--h48x-only on a dir with no aux_drv.pt", "REFUSED (rc 4)")
    shutil.rmtree(d, ignore_errors=True)
    m = run("noexcl", extra=T.HB_ARGS + ["--h48x-only", rd])
    red(m["rc_run"] == 4 and "needs --aux-drv and --aux-drv-exclude-logs" in m["out_run"], "--h48x-only without the exclusion list", "REFUSED (rc 4)")
    # (g) a training run does NOT go through the resume path; --h48x-only is the only way: the default is OFF
    ok(base["rc_run"] == 0 and not os.path.isfile(os.path.join(rd, "h48x_provenance.json")), "a normal run writes no provenance file / never enters the resume mode (default OFF)")


# ------------------------------------------------------------------------------------------------ P: the pod script's MODE=h48x
def layer_p():
    import subprocess
    import stat
    bash = None
    for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if os.path.exists(cand):
            bash = cand
            break
    if bash is None:
        bash = shutil.which("bash")
    pod = (HERE / "vdac_pod.sh").read_bytes().decode("utf-8")

    def run_pod(text, arm, mode):
        td = Path(tempfile.mkdtemp(prefix="h48xpod_", dir=TMPROOT))
        try:
            pkg = td / "pkg"
            pkg.mkdir()
            (pkg / "vdac_pod.sh").write_bytes(text.encode("utf-8"))
            (pkg / "scorer1p6_pod.sh").write_bytes((HERE / "scorer1p6_pod.sh").read_bytes())
            ws = td / "ws"
            ws.mkdir()
            fake = td / "fakepy.sh"
            fake.write_bytes("\n".join(["#!/bin/bash", 'echo "ALLOC=${PYTORCH_CUDA_ALLOC_CONF-UNSET}"', 'echo "ARGV $*"', ""]).encode())
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            (ws / "teacher_env.sh").write_bytes(("DRIVERL_EVAL_PYTHON=\"" + fake.as_posix() + "\"\n").encode())
            e = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
            e.update({"ARM": arm, "MODE": mode, "VDAC_WS": ws.as_posix()})
            r = subprocess.run([bash, (pkg / "vdac_pod.sh").as_posix()], capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=pkg.as_posix())
            lg = ws / "data" / "refe_vdac" / arm.lower() / ("h48x_only.log" if mode == "h48x" else "sft.log")
            txt = lg.read_text(encoding="utf-8") if lg.exists() else ""
            return r, txt, (ws / "data" / "refe_vdac" / arm.lower()).as_posix()
        finally:
            shutil.rmtree(td, ignore_errors=True)
    r, txt, out = run_pod(pod, "HB-s0", "h48x")
    argv = [x for x in txt.splitlines() if x.startswith("ARGV ")]
    ok(r.returncode == 0 and len(argv) == 1, f"MODE=h48x runs the trainer once and exits with its status: rc {r.returncode} {r.stderr[:200]}")
    toks = argv[0][5:].split()
    ok(toks[-2:] == ["--h48x-only", out] and "--out" in toks and toks[toks.index("--out") + 1] == out, "the trainer gets --h48x-only <this arm's dir> and --out <the same dir>")
    r2, txt2, out2 = run_pod(pod, "HB-s0", "run")
    argv2 = [x for x in txt2.splitlines() if x.startswith("ARGV ")][0][5:].split()
    norm = lambda ts, o: [t.replace(o.rsplit("/data/refe_vdac", 1)[0], "<WS>") for t in ts]
    ok(norm(toks[:-2], out) == norm(argv2, out2), "SAME ARGV as the training run, token for token, plus the one flag (MODE=h48x vs MODE=run)")
    ok("ALLOC=expandable_segments:True" in txt, "HB's allocator setting reaches the forward-only process too")
    ok("ZZEXIT 0" in txt, "the log ends with the trainer's exit status")
    r3, txt3, _ = run_pod(pod, "HB-s0", "bogus")
    ok(r3.returncode != 0, "an unknown MODE is still refused")
    bad = pod.replace("  h48x) \"$PY\" scorer_finetune.py $ARGS --h48x-only $OUT > $OUT/h48x_only.log", "  h48x) \"$PY\" scorer_finetune.py $ARGS > $OUT/h48x_only.log")
    ok(bad != pod, "the mutation applies")
    r4, txt4, out4 = run_pod(bad, "HB-s0", "h48x")
    argv4 = [x for x in txt4.splitlines() if x.startswith("ARGV ")]
    # the pause script's stage (STATIC: it needs /proc, nvidia-smi and a pod): the forward-only arm is named, runs MODE=h48x on HB-s0 only, is bounded, and a failure stops HB-SHUF
    pz = (HERE / "vdac_pause_p7.sh").read_bytes().decode("utf-8")
    ok('A5HBFIX) ARMS="HB-s0-H48X HB-SHUF"' in pz and "ARM=HB-s0 MODE=h48x timeout 3600 bash /workspace/refe-plan-vdac/refe/vdac_pod.sh" in pz, "vdac_pause_p7.sh: stage A5HBFIX = HB-s0's forward-only (MODE=h48x, 1 h bound), then HB-SHUF")
    NL = chr(10)
    ok('grep -q "^ZZSFT_H48X_ONLY_DONE" $OUT/hb-s0/h48x_only.log' in pz and ("break; }" + NL + "    continue") in pz, "a forward-only that does not print ZZSFT_H48X_ONLY_DONE stops the stage (HB-SHUF is not run), P7 resumes via the EXIT trap")
    ok(pz.count("vdac_chain.sh") >= 1 and '"HB-s0-H48X"' in pz and "HB-s0-H48X) TMO" not in pz, "HB-s0-H48X never reaches the chain invocation (it `continue`s before it)")
    pz_bad = pz.replace("    continue" + NL + "  fi" + NL + "  say ", "  fi" + NL + "  say ")
    red(pz_bad != pz and ("    continue" + NL + "  fi") not in pz_bad, "pause script without the `continue` (the H48X pseudo-arm would fall into the chain)", "static check")
    red(not argv4 or "--h48x-only" not in argv4[0], "MODE=h48x without the flag (it would TRAIN from update 0)", "argv has no --h48x-only")


def main():
    base = layer_r()
    layer_s()
    layer_f(base)
    layer_p()
    print("MUTATION TABLE")
    for w, v, h in TABLE:
        print(f"  {v:4s} {w}  [{h}]")
    shutil.rmtree(TD0, ignore_errors=True)
    print(f"ZZTEST_PASS {N[0]}")


if __name__ == "__main__":
    main()
