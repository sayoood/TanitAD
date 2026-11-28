"""Unit tests of the OBD seed-3 package (dev box; CPU only; no network, no box 2). Prints "UNIT ok=<n> bad=<m>" and exits non-zero on a failure.
    python -B -I tests/test_obd3_check.py       (run with -I from any folder; the test inserts the package folder itself)
Scratch: F:/TanitAD-scratch/obd_seed3_build/unit_<pid> (created and deleted by this test, by literal path)."""
import contextlib
import hashlib
import io
import json
import ntpath
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import dry_helpers as DH  # noqa: E402  (pins and loads the OBD chain's stand-in trainer)
import make_argvs_seed3 as M  # noqa: E402
import obd3_check as C  # noqa: E402

B, PT = C.B, DH.PT
OK = BAD = 0
SCRATCH = f"F:/TanitAD-scratch/obd_seed3_build/unit_{os.getpid()}"
TAGS = ["OBD-CTL_s10", "OBD-BAND_s10", "OBD-CTL_s11", "OBD-BAND_s11"]
SRC_OF = {"OBD-CTL_s10": "OBD-CTL_s0", "OBD-BAND_s10": "OBD-BAND_s0", "OBD-CTL_s11": "OBD-CTL_s1", "OBD-BAND_s11": "OBD-BAND_s1"}
SEED_OF = {"OBD-CTL_s10": "10", "OBD-BAND_s10": "10", "OBD-CTL_s11": "11", "OBD-BAND_s11": "11"}


def ok(name, cond):
    global OK, BAD
    if cond:
        OK += 1
    else:
        BAD += 1
        print("FAIL:", name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def raises_key(tag):
    try:
        B.tokens(tag)
    except KeyError:
        return True
    return False


def run_main(argv):
    """-> (rc, stdout) of obd3_check.main(argv)"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = C.main(argv)
    return rc, buf.getvalue()


def setenv(**kw):
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def treehash(root):
    h = hashlib.sha256()
    for a, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            p = os.path.join(a, f)
            h.update(os.path.relpath(p, root).encode() + sha(p).encode())
    return h.hexdigest()


os.makedirs(SCRATCH, exist_ok=True)
try:
    # ---- 1 pins ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    base = C.BASE_DIR
    ok("the base files are read from theory/obd/box2", os.path.normpath(base) == os.path.normpath(os.path.join(PKG, "..", "..", "box2")))
    for n, want in C.BASE_SHA256.items():
        ok(f"base {n} carries its pinned sha256", sha(os.path.join(base, n)) == want)
    ok("base files equal the OBD chain's PACKAGE_MANIFEST.sha256 lines", all(f"{w} *./{n}" in open(os.path.join(base, "PACKAGE_MANIFEST.sha256"), encoding="utf-8").read() for n, w in C.BASE_SHA256.items()))
    ok("the frozen argv file carries the pinned sha256", sha(os.path.join(PKG, "frozen", "argvs_obd_seed3.json")) == C.ARGVS_SHA256 == B.ARGVS_SHA256)
    ok("the checker reads the seed-3 file, not the OBD one", os.path.basename(B.ARGVS_FILE) == "argvs_obd_seed3.json")
    ok("the source argv file carries its pinned sha256", sha(M.SOURCE) == M.SOURCE_SHA256 == "346a48ed7b070fffacd22bf280e72c42831126fb906c231a7ac24301e624a424")
    ok("the amendment file is the one named in the brief (sha256 a8a5848f...)", sha(os.path.join(PKG, "..", "..", "AMEND_OBD_SEED3.md")) == "a8a5848f22a3f164e4069bd7b302e7201cb05de5995494d05cf92dfde426e314")
    ok("the code of record is cdbff971fa6dad4d20b937fe8abda6cb9f622adb", C.CODE_COMMIT == "cdbff971fa6dad4d20b937fe8abda6cb9f622adb" and len(C.CODE_COMMIT) == 40)
    # a changed base file is refused at import (the pin is real): a copy with one byte more, found through OBD3_BASE
    cp = os.path.join(SCRATCH, "basecopy")
    os.makedirs(cp)
    for n in C.BASE_SHA256:
        shutil.copyfile(os.path.join(base, n), os.path.join(cp, n))
    with open(os.path.join(cp, "obd_check.py"), "ab") as f:
        f.write(b"\n# changed\n")
    r = subprocess.run([sys.executable, "-B", "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import obd3_check", PKG], capture_output=True, encoding="utf-8", env=dict(os.environ, OBD3_BASE=cp))
    ok("a changed base file is refused at import", r.returncode != 0 and "differs" not in r.stderr and "pinned" in (r.stderr + r.stdout))
    # ---- 2 the four argvs: the banked argv, token for token, only --seed and --out changed -----------------------------------------------------------------------------------------
    src = json.load(open(M.SOURCE, encoding="utf-8"))
    new = json.load(open(os.path.join(PKG, "frozen", "argvs_obd_seed3.json"), encoding="utf-8"))
    ok("exactly the four seed-3 tags, in order", list(new) == TAGS == list(M.DERIVE) == C.RUN_ORDER)
    for t in TAGS:
        a, b = src[SRC_OF[t]]["argv"], new[t]["argv"]
        diff = [i for i in range(len(a)) if a[i] != b[i]] if len(a) == len(b) else None
        si, oi = a.index("--seed") + 1, a.index("--out") + 1
        ok(f"{t}: same token count as {SRC_OF[t]}", len(a) == len(b))
        ok(f"{t}: the only differing tokens are the --seed value and the --out value", diff == [si, oi])
        ok(f"{t}: --seed {SEED_OF[t]}", b[si] == SEED_OF[t] and b.index("--seed") == a.index("--seed"))
        ok(f"{t}: --out is the source's runs root with the new tag", b[oi] == ntpath.join(ntpath.dirname(a[oi]), t) and ntpath.dirname(b[oi]) == r"D:\tanitad\runs\v8f_obd")
        ok(f"{t}: --dagger-seed and --init-ckpt stay the init's ({new[t]['init_seed']})", b[b.index("--dagger-seed") + 1] == str(new[t]["init_seed"]) and b[b.index("--init-ckpt") + 1] == a[a.index("--init-ckpt") + 1])
        ok(f"{t}: --steps 3000 and the band weights are the source's", b[b.index("--steps") + 1] == "3000" and ("--w-op" in b) == ("--w-op" in a) and ("--detach-upper" in b))
        ok(f"{t}: record fields", new[t]["source_tag"] == SRC_OF[t] and new[t]["init_seed"] == src[SRC_OF[t]]["init_seed"])
    ok("the banked arms are not touched by the derivation (CTL_s0 still seed 0)", src["OBD-CTL_s0"]["argv"][src["OBD-CTL_s0"]["argv"].index("--seed") + 1] == "0")
    out_path = os.path.join(SCRATCH, "regen.json")
    rc = M.main(["--out", out_path])
    ok("make_argvs_seed3.py regenerates the frozen file byte for byte", rc == 0 and open(out_path, "rb").read() == open(os.path.join(PKG, "frozen", "argvs_obd_seed3.json"), "rb").read())
    ok("the derivation is a function of the source only (derive() equals the frozen records)", M.derive(src) == new)
    # ---- 3 the checker's table and pins ---------------------------------------------------------------------------------------------------------------------------------------------
    ok("B.TABLE is the seed-3 table; run order CTL_s10, BAND_s10, CTL_s11, BAND_s11", B.TABLE is C.TABLE and B.RUN_ORDER == TAGS)
    ok("seeds 10, 10, 11, 11 and inits 0, 0, 1, 1", [C.TABLE[t]["seed"] for t in TAGS] == [10, 10, 11, 11] and [C.TABLE[t]["init"] for t in TAGS] == [0, 0, 1, 1] and [C.TABLE[t]["kind"] for t in TAGS] == ["ctl", "band", "ctl", "band"])
    for t in TAGS:
        ok(f"{t}: tokens() equals the frozen argv", B.tokens(t) == new[t]["argv"])
        ok(f"{t}: audit_line accepts the frozen line", B.audit_line(t, " ".join(new[t]["argv"])) == [])
        bad = list(new[t]["argv"])
        bad[bad.index("--seed") + 1] = "0"
        ok(f"{t}: audit_line refuses a changed --seed", len(B.audit_line(t, " ".join(bad))) == 1 and "token" in B.audit_line(t, " ".join(bad))[0])
        ok(f"{t}: audit_line refuses the SOURCE arm's argv", len(B.audit_line(t, " ".join(src[SRC_OF[t]]["argv"]))) == 1)
        ok(f"{t}: run folder and gate out are under obd3", B.run_dir(t).replace("\\", "/").endswith("/obd3/runs/" + t) and B.gate_out_decl(t) == "D:\\tanitad\\obd3\\gate_out\\" + t)
        ok(f"{t}: the out dir is the runs root's {t}", B.out_decl(t) == new[t]["argv"][new[t]["argv"].index("--out") + 1])
    ok("a banked tag has no tokens here (KeyError)", all(raises_key(x) for x in ("OBD-CTL_s0", "OBD-BAND_s1", "OBD-NULL_s0", "PILOT_OBD-BAND_s0")))
    rc, _ = run_main(["argv", "--tag", "OBD-CTL_s0"])
    ok("argv refuses a banked OBD tag (rc 2)", rc == 2)
    rc, _ = run_main(["argv", "--tag", "PILOT_OBD-BAND_s0"])
    ok("argv refuses the pilot tag (rc 2)", rc == 2)
    rc, _ = run_main(["argv", "--tag", "OBD-CTL_s10", "--pilot"])
    ok("argv refuses --pilot (rc 2)", rc == 2)
    for c in C.REFUSED_CMDS:
        rc, _ = run_main([c])
        ok(f"`{c}` is refused (rc 2)", rc == 2)
    rc, out = run_main(["argv", "--tag", "OBD-BAND_s11", "--out-txt", os.path.join(SCRATCH, "argv.txt")])
    ok("argv --out-txt writes one line equal to the frozen argv", rc == 0 and open(os.path.join(SCRATCH, "argv.txt"), encoding="utf-8").read() == " ".join(new["OBD-BAND_s11"]["argv"]) + "\n")
    ok("data paths are read from a seed-3 argv", B.data_paths()["v9_labels_train"].endswith("v9_labels_train.npz") and len(B.data_paths()) == 6)
    ok("the box-side test hook OBD_TEST_ARGVS is refused outside a fake box", subprocess.run([sys.executable, "-B", "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import obd3_check", PKG], capture_output=True, encoding="utf-8", env=dict(os.environ, OBD_TEST_ARGVS="1", OBD_ROOT="")).returncode != 0)
    # ---- 4 the BUILT-lever gate on planted selfchecks (the base assertions, applied to the seed-3 tags) ------------------------------------------------------------------------
    json.dump({str(k): v for k, v in B.INIT_SHA.items()}, open(os.path.join(SCRATCH, "init.json"), "w"))
    json.dump({str(k): v for k, v in B.BANK_SHA.items()}, open(os.path.join(SCRATCH, "bank.json"), "w"))
    setenv(OBD_FAKE_INIT_SHA_FILE=os.path.join(SCRATCH, "init.json"), OBD_FAKE_BANK_SHA_FILE=os.path.join(SCRATCH, "bank.json"))
    for t in TAGS:
        toks = B.tokens(t)
        good = PT.good_selfcheck(B.gate_argv(t, toks), "ok")
        ok(f"{t}: a correct selfcheck passes the gate", B.gate_problems(t, toks, good) == [])
        for mode in ("bad_params", "bad_steps", "bad_bank", "no_acc_probe", "bad_detach", "bad_acc_leak"):
            bad_sc = PT.good_selfcheck(B.gate_argv(t, toks), mode)
            ok(f"{t}: the planted defect {mode} is refused", len(B.gate_problems(t, toks, bad_sc)) >= 1)
        wrong = list(toks)
        wrong[wrong.index("--seed") + 1] = "0"
        ok(f"{t}: a gate line with the wrong --seed is refused", any("--seed" in p for p in B.gate_problems(t, wrong, PT.good_selfcheck(B.gate_argv(t, wrong), "ok"))))
        wrong = list(toks)
        wrong[wrong.index("--dagger-seed") + 1] = str(1 - new[t]["init_seed"])
        ok(f"{t}: a gate line with the other init's --dagger-seed is refused", any("dagger-seed" in p for p in B.gate_problems(t, wrong, PT.good_selfcheck(B.gate_argv(t, wrong), "ok"))))
    setenv(OBD_FAKE_INIT_SHA_FILE=None, OBD_FAKE_BANK_SHA_FILE=None)
    # ---- 5 render, token, job templates ---------------------------------------------------------------------------------------------------------------------------------------------
    for jn in ("obd3_stage_job.cmd", "obd3_ship_job.cmd"):
        txt = open(os.path.join(PKG, "jobs", jn), "rb").read().decode("utf-8")
        ok(f"{jn}: no labels, no goto, no sentinel text", "goto" not in txt.lower().replace("no goto", "") and not any(l.startswith(":") for l in txt.splitlines()) and "ZZ " not in txt)
        ok(f"{jn}: sets PYTHONDONTWRITEBYTECODE (the shared tree stays byte-identical)", "PYTHONDONTWRITEBYTECODE=1" in txt)
    rc, _ = run_main(["render", "--template", os.path.join(PKG, "jobs", "obd3_stage_job.cmd"), "--out", os.path.join(SCRATCH, "stage.cmd"), "--set", "ROOT=D:\\tanitad", "--set", "COMMIT=" + C.CODE_COMMIT, "--set", "GID=g1",
                      "--set", "ARGS=stage --tag OBD-CTL_s10 --gid g1", "--set", "MARKDIR=@ROOT@\\obd3\\runs\\OBD-CTL_s10", "--set", "MARK=STAGE"])
    rendered = open(os.path.join(SCRATCH, "stage.cmd"), "rb").read().decode()
    ok("render fills every placeholder, CRLF line ends, the marker dir under obd3", rc == 0 and "@" not in rendered.replace("@echo", "") and "\r\n" in rendered and "\n" not in rendered.replace("\r\n", "") and "D:\\tanitad\\obd3\\runs\\OBD-CTL_s10" in rendered and "obd3\\check\\obd3_check.py" in rendered)
    rc, _ = run_main(["render", "--template", os.path.join(PKG, "jobs", "obd3_stage_job.cmd"), "--out", os.path.join(SCRATCH, "stage2.cmd"), "--set", "ROOT=D:\\tanitad"])
    ok("render refuses unfilled placeholders (rc 2)", rc == 2)
    line = " ".join(new["OBD-CTL_s10"]["argv"])
    open(os.path.join(SCRATCH, "argv_ctl10.txt"), "w", encoding="utf-8").write(line + "\n")
    tok = {"written_by": "obd_check.py gate", "tag": "OBD-CTL_s10", "commit": C.CODE_COMMIT, "argv_sha256": B.tokens_sha(line.split(" "))}
    json.dump(tok, open(os.path.join(SCRATCH, "tok.json"), "w"))
    vt = lambda tag, commit, argv_file: run_main(["verify-token", "--token", os.path.join(SCRATCH, "tok.json"), "--argv-file", argv_file, "--tag", tag, "--commit", commit])[0]      # noqa: E731
    ok("verify-token accepts a token that binds tag, commit and the frozen argv", vt("OBD-CTL_s10", C.CODE_COMMIT, os.path.join(SCRATCH, "argv_ctl10.txt")) == 0)
    ok("verify-token refuses another commit", vt("OBD-CTL_s10", "1" * 40, os.path.join(SCRATCH, "argv_ctl10.txt")) == 1)
    open(os.path.join(SCRATCH, "argv_src.txt"), "w", encoding="utf-8").write(" ".join(src["OBD-CTL_s0"]["argv"]) + "\n")
    ok("verify-token refuses the source arm's argv under the new tag", vt("OBD-CTL_s10", C.CODE_COMMIT, os.path.join(SCRATCH, "argv_src.txt")) == 1)
    ok("verify-token refuses a tag this package does not know", vt("OBD-CTL_s0", C.CODE_COMMIT, os.path.join(SCRATCH, "argv_src.txt")) == 2)
    # ---- 6 shipcheck: a read-only proof of the tree (fake box root) --------------------------------------------------------------------------------------------------------------------
    commit = "a" * 40
    root = os.path.join(SCRATCH, "box", "tanitad")
    repo = os.path.join(root, "obd", "repo")
    needles = {B.TRAINER_REL: "\n".join(B.NEEDLES[B.TRAINER_REL]), B.MODEL_REL: "\n".join(B.NEEDLES[B.MODEL_REL]), B.DATA_REL: "\n".join(B.NEEDLES[B.DATA_REL]), "stack/tanitad/__init__.py": ""}

    def mktree(skip=None):
        shutil.rmtree(os.path.join(root, "obd"), ignore_errors=True)
        mem = {}
        for rel, txt in needles.items():
            p = os.path.join(repo, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", encoding="utf-8", newline="\n").write(txt)
            mem[rel] = sha(p)
        json.dump({"commit": commit, "members": mem}, open(os.path.join(root, "obd", "tree_manifest.json"), "w"))
        open(os.path.join(repo, "TIP_OBD.txt"), "w", newline="\n").write(commit + "\n")
    setenv(OBD_ROOT=root, OBD_COMMIT=commit, OBD3_CODE_COMMIT=commit)
    mktree()
    h0 = treehash(os.path.join(root, "obd"))
    rc, out = run_main(["shipcheck", "--gid", "g1"])
    res = json.load(open(os.path.join(root, "obd3", "ship", "SHIPCHECK_RESULT.json"), encoding="utf-8"))
    ok("shipcheck proves a correct tree (rc 0, done marker, ok result)", rc == 0 and os.path.exists(os.path.join(root, "obd3", "ship", "SHIPCHECK_g1.done")) and res["ok"] is True and res["n_members"] == 4 and res["gid"] == "g1" and "OBD3-SHIPCHECK-OK" in out)
    ok("shipcheck wrote nothing into the tree (the OBD tree is byte-identical)", treehash(os.path.join(root, "obd")) == h0)
    ok("shipcheck wrote only under obd3\\ship", sorted(os.listdir(os.path.join(root, "obd3"))) == ["ship"])
    rc, out = run_main(["shipcheck", "--gid", "g2"])
    ok("a second shipcheck again proves it", rc == 0)
    setenv(OBD_COMMIT="b" * 40)
    rc, out = run_main(["shipcheck", "--gid", "g3"])
    ok("an OBD_COMMIT that is not the code of record: not proven (rc 3, fail marker)", rc == 3 and os.path.exists(os.path.join(root, "obd3", "ship", "SHIPCHECK_g3.fail")) and "NOT-PROVEN" in out)
    setenv(OBD_COMMIT=commit)
    with open(os.path.join(repo, B.TRAINER_REL.replace("/", os.sep)), "a", encoding="utf-8") as f:
        f.write("# one byte\n")
    rc, out = run_main(["shipcheck", "--gid", "g4"])
    ok("a changed tree member: not proven, the problem names it", rc == 3 and "differ from the manifest" in out)
    mktree()
    open(os.path.join(repo, "TIP_OBD.txt"), "w", newline="\n").write("c" * 40 + "\n")
    rc, out = run_main(["shipcheck", "--gid", "g5"])
    ok("a wrong TIP_OBD.txt: not proven", rc == 3 and "TIP_OBD.txt" in out)
    mktree()
    os.remove(os.path.join(root, "obd", "tree_manifest.json"))
    rc, out = run_main(["shipcheck", "--gid", "g6"])
    ok("no tree_manifest.json: not proven", rc == 3 and "unreadable" in out)
    mktree()
    needles2 = dict(needles)
    p = os.path.join(repo, *B.TRAINER_REL.split("/"))
    txt = open(p, encoding="utf-8").read().replace("def acc_probe", "def xxx_probe")
    open(p, "w", encoding="utf-8", newline="\n").write(txt)
    man = json.load(open(os.path.join(root, "obd", "tree_manifest.json")))
    man["members"][B.TRAINER_REL] = sha(p)
    json.dump(man, open(os.path.join(root, "obd", "tree_manifest.json"), "w"))
    rc, out = run_main(["shipcheck", "--gid", "g7"])
    ok("a trainer without its needle strings (hashes consistent): not proven", rc == 3 and "lacks" in out)
    mktree()
    os.remove(os.path.join(repo, "stack", "tanitad", "__init__.py"))
    rc, out = run_main(["shipcheck", "--gid", "g8"])
    ok("a missing member: not proven", rc == 3)
    setenv(OBD_ROOT=None, OBD3_CODE_COMMIT=commit)
    ok("without a fake box the pinned commit cannot be overridden", C.code_commit() == C.CODE_COMMIT)
    setenv(OBD_ROOT=root)
    ok("with a fake box the override works (dry run only)", C.code_commit() == commit)
    # ---- 7 stage refusals and the cleanup rule --------------------------------------------------------------------------------------------------------------------------------------
    setenv(OBD_COMMIT="b" * 40)
    rc, out = run_main(["stage", "--tag", "OBD-CTL_s10", "--gid", "g1"])
    ok("stage refuses a commit that is not the code of record (rc 3)", rc == 3 and "code of record" in out)
    rc, out = run_main(["stage", "--tag", "OBD-CTL_s0", "--gid", "g1"])
    ok("stage refuses a tag of the OBD chain (rc 2)", rc == 2)
    rc, out = run_main(["stage", "--tag", "..\\x", "--gid", "g1"])
    ok("stage refuses a path-like tag (rc 2), nothing is created outside obd3\\runs", rc == 2 and not os.path.exists(os.path.join(root, "x")))
    setenv(OBD_COMMIT=commit)
    rd = B.run_dir("OBD-BAND_s11")
    os.makedirs(rd, exist_ok=True)
    gout = B.loc(B.gate_out_decl("OBD-BAND_s11"))
    os.makedirs(gout)
    open(os.path.join(gout, "selfcheck.json"), "w").write("{}")
    ckdir = os.path.join(root, "runs", "v8f_obd", "OBD-BAND_s11")
    os.makedirs(ckdir)
    open(os.path.join(ckdir, "ckpt_last.pt"), "wb").write(b"ckpt")
    rc, out = run_main(["cleanup", "--tag", "OBD-BAND_s11", "--gid", "c1"])
    ok("cleanup without BANK_VERIFIED deletes nothing (rc 6)", rc == 6 and os.path.exists(gout) and os.path.exists(os.path.join(rd, "CLEAN_c1.fail")))
    open(os.path.join(rd, "BANK_VERIFIED_c2"), "w").write("x")
    rc, out = run_main(["cleanup", "--tag", "OBD-BAND_s11", "--gid", "c2"])
    cj = json.load(open(os.path.join(rd, "CLEANUP.json")))
    ok("cleanup with the bank proof deletes the gate out dir only, logs path and bytes, keeps the checkpoint", rc == 0 and not os.path.exists(gout) and os.path.exists(os.path.join(ckdir, "ckpt_last.pt")) and len(cj["deleted_own_temp"]) == 1 and cj["deleted_own_temp"][0]["bytes"] == 2)
    setenv(OBD_ROOT=None, OBD_COMMIT=None, OBD3_CODE_COMMIT=None)
    # ---- 8 scripts: syntax, env names, hygiene ----------------------------------------------------------------------------------------------------------------------------------------
    sh = ["obd3_chain.sh", "stages/obd3_common.sh", "stages/obd3_ship.sh", "stages/obd3_run.sh", "tests/test_obd3_dryrun.sh", "tests/ssh_stub.sh", "tests/sftp_stub.sh"]
    bash = shutil.which("bash") or "bash"
    for s in sh:
        r = subprocess.run([bash, "-n", os.path.join(PKG, s)], capture_output=True, encoding="utf-8")
        ok(f"bash -n {s}", r.returncode == 0)
    allsh = "".join(open(os.path.join(PKG, s), encoding="utf-8").read() for s in sh[:4])
    import re
    ok("the dev-box scripts use OBD3_ names only (no OBD_SSH, OBD_STATE, OBD_COMMIT ... of the OBD chain)", re.search(r"\bOBD_[A-Z]+", allsh) is None)
    for v in ("OBD3_COMMIT", "OBD3_SSH", "OBD3_SFTP", "OBD3_DRY", "OBD3_STATE", "OBD3_PULLS", "OBD3_STOPBAL_STATE", "OBD3_WAIT_POLLS", "OBD3_WAIT_POLL_S", "OBD3_GPU_IDLE_MIB", "OBD3_RESUME"):
        ok(f"{v} is used", v in allsh)
    ok("GPU idle default is 3500", 'OBD3_GPU_IDLE_MIB:-3500' in allsh)
    ok("state dir default is C:/Users/Admin/obd_seed3_state; STOPBAL state C:/Users/Admin/stopbal_state; bank root C:/Users/Admin/t2v8f_pulls", "C:/Users/Admin/obd_seed3_state" in allsh and "C:/Users/Admin/stopbal_state" in allsh and "C:/Users/Admin/t2v8f_pulls" in allsh)
    ok("the shared busy marker is the one the other chains claim", "C:/Users/Admin/box2_ship/BOX2_BUSY.txt" in allsh)
    ok("no ZZ sentinel is printed to a log (every line carrying one writes to its own file)", all(("> \"$STATE/" in l) or l.lstrip().startswith("#") for l in allsh.splitlines() if "ZZ OBD3" in l))
    ok("the run stage knows exactly the four tags", "OBD-CTL_s10|OBD-BAND_s10|OBD-CTL_s11|OBD-BAND_s11" in open(os.path.join(PKG, "stages", "obd3_run.sh"), encoding="utf-8").read())
    ok("the run stage's box paths agree with the argv out dirs (D:/tanitad/runs/v8f_obd/<tag>)", "D:/tanitad/runs/v8f_obd/$TAG" in open(os.path.join(PKG, "stages", "obd3_run.sh"), encoding="utf-8").read())
    for fn in os.listdir(PKG):
        if fn.endswith(".py"):
            t = open(os.path.join(PKG, fn), encoding="utf-8").read()
            ok(f"{fn}: no subprocess text=True without encoding=", "text=True" not in t)
    ok("no package file uses a Windows reserved device name", not any(re.fullmatch(r"(aux|con|prn|nul)(\..*)?", f, re.I) for _, _, fs in os.walk(PKG) for f in fs))
finally:
    shutil.rmtree(SCRATCH, ignore_errors=True)
print(f"UNIT ok={OK} bad={BAD}")
sys.exit(1 if BAD else 0)
