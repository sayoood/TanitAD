"""Self-test of soup_chain.sh / soup_chain_lib.py / soup_chain.cmd / wake_soup.sh (PREREG_V8F_SOUP section 13a), end to end in THROW-AWAY WORLDS with a STUB tool and a STUB reader.  It runs no model, no rollout, no
eval, no soup read and touches nothing outside the world it builds (a temp directory carrying SOUP_TESTWORLD.marker); the real plan / tree / checkpoints are only READ (hashes, parsing) in the last test.

    SOUP_CHAIN_DIR=<dir holding soup_chain.sh + soup_chain_lib.py>   (default: this package; the mutation pass points it at a mutant copy)
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SOUP = os.path.dirname(HERE)
CHAIN = os.environ.get("SOUP_CHAIN_DIR", SOUP).replace("\\", "/")
BASH = r"C:\Program Files\Git\bin\bash.exe"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = load("soup_chain_lib_under_test", f"{CHAIN}/soup_chain_lib.py")
RP = load("render_plan_for_test", f"{SOUP}/tools/render_plan.py")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def lp(path):
    """A long-path-safe spelling: a moved-aside output sits ~120 characters deeper than the world, and pytest's basetemp under a long scratch directory pushes it past MAX_PATH (python.exe here is not long-path aware;
    the first mutation pass 'killed' 14 mutants by that alone).  Production paths are far shorter; the chain itself (git bash) is not affected."""
    p = os.path.abspath(path)
    return "\\\\?\\" + p if os.name == "nt" and len(p) > 240 and not p.startswith("\\\\?\\") else p


def w8(path, text, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding=None if "b" in mode else "utf-8", newline=None if "b" in mode else "\n") as f:
        f.write(text)


class World:
    """A throw-away world: stub tree, stub tools, three checkpoints (+ soup.json), the spec, a 6-step plan rendered from the REAL templates, stub reader, PREREG."""
    TAGS = ["SOUP_s0_m10_causal_heads", "SOUP_s0_m10_logged_heads", "M2b-soup_s0-fresh", "M2b-soup_s0", "M2b-dagoacc_0-dev", "M2b-a_0-dev"]
    KINDS = ["causal", "logged", "t0fresh", "t0std", "t0dev", "t0dev"]

    def __init__(self, root, out=None):
        self.r = str(root).replace("\\", "/") + "/soupworld"
        r = self.r
        os.makedirs(r)
        w8(f"{r}/SOUP_TESTWORLD.marker", "SOUP-THROWAWAY-WORLD\n")
        for t in ("v8f_t1_eval.py", "v8f_mini_eval.py"):
            os.makedirs(f"{r}/tree/stack/scripts", exist_ok=True)
            shutil.copy(f"{HERE}/soup_chain_stub_tool.py", f"{r}/tree/stack/scripts/{t}")
        w8(f"{r}/tree/stack/tanitad/__init__.py", "# stub\n")
        w8(f"{r}/tree/taniteval/__init__.py", "# stub\n")
        self.ck = {"soup": f"{r}/ckpt/M2b_s0_soup/ckpt_last.pt", "dag": f"{r}/ckpt/dagoacc0/ckpt_last.pt", "a": f"{r}/ckpt/a0/ckpt_last.pt", "rep": f"{r}/ckpt/rep0/ckpt_last.pt", "arep": f"{r}/ckpt/arep0/ckpt_last.pt"}
        for k, p in self.ck.items():
            w8(p, f"checkpoint {k} " * 50)
        w8(f"{r}/ckpt/M2b_s0_soup/soup.json", json.dumps({"name": "SOUP_s0", "output_sha256": sha(self.ck["soup"]), "K-BASIN": {"PASS": True, "B1": True, "B2": True}}))
        mem = lambda k, seed: {"path": self.ck[k], "sha256": sha(self.ck[k]), "init_seed": seed}      # noqa: E731
        spec = {"prereg": "stub", "inits": {}, "members": {"DAGOACC_0": mem("dag", 0), "REP_0": mem("rep", 0), "A_0": mem("a", 0), "AREP_0": mem("arep", 0)},
                "soups": {"SOUP_s0": {"seed": 0, "donor": "DAGOACC_0", "run_dir": "M2b_s0_soup", "role": "SOUP", "decisive": True, "members": ["DAGOACC_0", "REP_0"]}}}
        w8(f"{r}/soups.json", json.dumps(spec))
        os.makedirs(f"{r}/argv", exist_ok=True)
        for t in os.listdir(f"{SOUP}/argv"):
            if t.endswith(".tmpl"):
                shutil.copy(f"{SOUP}/argv/{t}", f"{r}/argv/{t}")
        self.out = out or f"{r}/out"
        P = RP.P
        keep = dict(P)
        P.update(TREE=f"{r}/tree")
        try:
            def rend(t, ck, tag, **kw):
                return " ".join(f"'{x}'" if re.search(r"[^A-Za-z0-9_./:=,@%+-]", x) else x for x in RP.render(t, CKPT=ck, TAG=tag, **kw))
            o = self.out
            lines = ["# world plan", rend("rollout_causal.tmpl", self.ck["soup"], self.TAGS[0], OUT=o), rend("rollout_logged.tmpl", self.ck["soup"], self.TAGS[1], OUT=o),
                     rend("t0fresh.tmpl", self.ck["soup"], self.TAGS[2], T0DIR=f"{o}/t0fresh"), rend("t0std.tmpl", self.ck["soup"], self.TAGS[3], T0DIR=f"{o}/t0std"),
                     rend("t0std.tmpl", self.ck["dag"], self.TAGS[4], T0DIR=f"{o}/t0dev"), rend("t0std.tmpl", self.ck["a"], self.TAGS[5], T0DIR=f"{o}/t0dev")]
        finally:
            P.clear()
            P.update(keep)
        w8(f"{r}/argv/PLAN.txt", "\n".join(lines) + "\n")
        w8(f"{r}/li_runs.json", json.dumps({"seeds": [0], "arms": {k: {"0": {"fake": k}} for k in ("CTRL", "REP", "A", "Arep")}, "build": {"b": 1}}))
        w8(f"{r}/banked.json", "{}")
        w8(f"{r}/li_reader.py", "# stub li reader\n")
        w8(f"{r}/builder.py", "# stub builder\n")
        os.makedirs(f"{r}/orc")
        shutil.copy(f"{HERE}/soup_chain_stub_reader.py", f"{r}/reader_stub.py")
        os.makedirs(f"{r}/reader")
        shutil.copy(f"{HERE}/soup_chain_stub_reader.py", f"{r}/reader/soup_read.py")
        w8(f"{r}/prereg.md", "# promoted prereg (stub)\n")
        self.bin = f"{r}/bin"
        for c in ("ssh", "scp", "rsync", "sftp"):
            w8(f"{self.bin}/{c}", f'#!/bin/bash\necho "{c} $*" >> "{r}/SSH_CALLED"\nexit 255\n')
        w8(f"{r}/trap_env.sh", "".join(f'{c}() {{ echo "{c} $*" >> "{r}/SSH_CALLED"; return 255; }}\n' for c in ("ssh", "scp", "rsync", "sftp"))
           + f'command() {{ if [ "${{1:-}}" = -v ] || [ "${{1:-}}" = -V ]; then case "${{2:-}}" in ssh|scp|rsync|sftp) echo "command -v $2" >> "{r}/CMDV_CALLED" ;; esac; fi; builtin command "$@"; }}\n'
           + f'type() {{ case "${{1:-}}" in ssh|scp|rsync|sftp) echo "type $1" >> "{r}/CMDV_CALLED" ;; esac; builtin type "$@"; }}\n'
           + f'which() {{ case "${{1:-}}" in ssh|scp|rsync|sftp) echo "which $1" >> "{r}/CMDV_CALLED" ;; esac; /usr/bin/which "$@"; }}\n')
        self.prereg_sha = sha(f"{r}/prereg.md")

    def env(self):
        r = self.r
        e = {"SOUP_WORLD": r, "SOUP_TREE": f"{r}/tree", "SOUP_OUT": self.out, "SOUP_LOG": f"{r}/logs", "SOUP_PLAN": f"{r}/argv/PLAN.txt", "SOUP_ARGV_DIR": f"{r}/argv", "SOUP_PINS": f"{r}/PINS.txt", "SOUP_SPEC": f"{r}/soups.json",
             "SOUP_BUILDER": f"{r}/builder.py", "SOUP_READER": f"{r}/reader/soup_read.py", "SOUP_LI_READER": f"{r}/li_reader.py", "SOUP_LI_READER_SHA": sha(f"{r}/li_reader.py") if os.path.isfile(f"{r}/li_reader.py") else "0" * 64, "SOUP_LI_RUNS": f"{r}/li_runs.json",
             "SOUP_BANKED_LI_READ": f"{r}/banked.json", "SOUP_ORC_DIR": f"{r}/orc", "SOUP_PREREG": f"{r}/prereg.md", "SOUP_PREREG_SHA": self.prereg_sha, "SOUP_RAM_MIN_MB": "1", "SOUP_F_MIN_GB": "0.0",
             "SOUP_C_MIN_MB": "0", "SOUP_GUARD_WAIT_S": "3", "SOUP_POLL_S": "1"}
        return e

    def chain(self, *args, env=None, trap=False, timeout=420):
        e = {k: v for k, v in os.environ.items() if not k.startswith("SOUP_")}
        e.update(self.env())
        e.update(env or {})
        e = {k: v for k, v in e.items() if v is not None}
        if trap:
            e["BASH_ENV"] = f"{self.r}/trap_env.sh"
            e["PATH"] = self.bin.replace("/", "\\") + os.pathsep + e["PATH"]
        p = subprocess.run([BASH, f"{CHAIN}/soup_chain.sh", *args], env=e, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr

    def pins(self):
        rc, out, err = self.chain("--write-pins")
        assert rc == 0, out + err

    def log(self, name="chain.out"):
        p = f"{self.r}/logs/{name}"
        return open(p, encoding="utf-8", errors="replace").read() if os.path.isfile(p) else ""

    def calls(self):
        p = f"{self.r}/stub_calls.log"
        return open(p, encoding="utf-8").read().split() if os.path.isfile(p) else []

    def hook(self, name, text="x"):
        w8(f"{self.r}/{name}", text)

    def done(self, tag):
        t0 = tag.startswith("M2b-")
        return json.load(open(f"{self.out}/done_{'t0_' if t0 else ''}{tag}.txt", encoding="utf-8"))

    def waker(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith("SOUP_")}
        e.update({"SOUP_LOG": f"{self.r}/logs", "SOUP_WAKE_POLL_S": "1", "SOUP_WAKE_MAX_POLLS": "3"})
        e.update(env or {})
        p = subprocess.run([BASH, f"{CHAIN}/wake_soup.sh", *args], env=e, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


@pytest.fixture
def pw(tmp_path):
    x = World(tmp_path)
    x.pins()
    return x


def markers(text):
    return [ln for ln in text.split("\n") if ln.startswith("ZZSOUP-")]


# ================================================================================================ static facts about the shipped files (cheap, first)
FILES = ["soup_chain.sh", "soup_chain_lib.py", "soup_chain.cmd", "wake_soup.sh"]


def test_no_remote_shell_command_is_named_anywhere_in_the_runner():
    pat = re.compile(r"\b(ssh|scp|sftp|rsync|plink)\b|box-?2|tanitad-box2", re.I)
    for f in FILES:
        for i, ln in enumerate(open(f"{CHAIN}/{f}", encoding="utf-8").read().split("\n"), 1):
            assert not pat.search(ln), (f, i, ln[:120])


def test_the_launcher_is_crlf_refuses_without_the_prereg_lines_and_launches_the_chain_detached():
    b = open(f"{CHAIN}/soup_chain.cmd", "rb").read()
    assert b.count(b"\r\n") == b.count(b"\n") == b.count(b"\r") and b.endswith(b"\r\n")
    t = b.decode("utf-8")
    assert re.search(r"^rem {3}set SOUP_PREREG=", t, re.M) and re.search(r"^rem {3}set SOUP_PREREG_SHA=", t, re.M)
    assert not re.search(r"^\s*set SOUP_PREREG", t, re.M | re.I)                                            # shipped UNSET: the chain refuses
    assert "set SOUP_RAM_MIN_MB=2048" in t and "soup_chain.sh %*" in t and "< NUL" in t and "> /e/Projects/TanitAD-artifacts/dagland/soup/logs/soup_chain.launch.out 2>&1" in t and ">>" not in t


def test_shell_files_are_lf_only_and_pass_bash_syntax_check():
    for f in ("soup_chain.sh", "wake_soup.sh", "soup_chain_lib.py"):
        assert b"\r" not in open(f"{CHAIN}/{f}", "rb").read(), f
    for f in ("soup_chain.sh", "wake_soup.sh"):
        assert subprocess.run([BASH, "-n", f"{CHAIN}/{f}"], capture_output=True).returncode == 0, f


# ================================================================================================ the lib, unit by unit
def good_line(tag="SOUP_s0_m10_causal_heads", kind="causal", ck="E:/x/ckpt_last.pt", tool=L.T1_TOOL, out="F:/o", tree="F:/t"):
    base = f"{PY} {tree}/stack/scripts/{tool} run --ckpt {ck} --tag {tag}"
    if kind in ("causal", "logged"):
        return f"{base} --out-dir {out} --limit {kind} --dump-ticks {out}/ticks --dump-heads"
    return f"{base} --out-dir {out}/t0std" + (" --ages fresh" if kind == "t0fresh" else "")


def plan_file(tmp_path, lines):
    p = tmp_path / "PLAN.txt"
    p.write_bytes(("\n".join(lines) + "\n").encode())
    return str(p)


def test_parse_plan_kinds_and_fields(tmp_path):
    lines = ["# c", good_line(), good_line("T2", "logged"), good_line("M2b-x-fresh", "t0fresh", tool=L.T0_TOOL), good_line("M2b-x", "t0std", tool=L.T0_TOOL), good_line("M2b-y-dev", "t0std", tool=L.T0_TOOL)]
    st = L.parse_plan(plan_file(tmp_path, lines), "F:/t", "F:/o")
    assert [s["kind"] for s in st] == ["causal", "logged", "t0fresh", "t0std", "t0dev"] and [s["idx"] for s in st] == list(range(5))
    assert st[0]["ckpt"] == "E:/x/ckpt_last.pt" and st[0]["tag"] == "SOUP_s0_m10_causal_heads" and st[2]["out_dir"] == "F:/o/t0std"


@pytest.mark.parametrize("mut", [
    lambda s: s.replace(PY, "python", 1), lambda s: s.replace("/stack/scripts/", "/scripts/", 1), lambda s: s.replace(" run ", " go ", 1), lambda s: s.replace("--ckpt E:/x/ckpt_last.pt", "--ckpt E:/x/model.pt", 1),
    lambda s: s.replace("--limit causal", "--limit other", 1), lambda s: s.replace("--out-dir F:/o ", "--out-dir F:/elsewhere ", 1), lambda s: s.replace("--dump-ticks F:/o/ticks", "--dump-ticks F:/o/tix", 1),
    lambda s: s.replace(" --dump-heads", "", 1), lambda s: s + " --tag dup", lambda s: s.replace("--tag SOUP_s0_m10_causal_heads", "--tag 'bad tag'", 1), lambda s: s.replace("--ckpt E:/x/ckpt_last.pt ", "", 1),
])
def test_parse_plan_refuses_each_malformed_t1_line(tmp_path, mut):
    with pytest.raises(L.PlanError):
        L.parse_plan(plan_file(tmp_path, [mut(good_line())]), "F:/t", "F:/o")


def test_parse_plan_refuses_the_other_malformed_plans(tmp_path):
    for lines in ([], ["# only a comment"], [good_line(), good_line()],
                  [good_line("M2b-x", "t0std", tool=L.T0_TOOL).replace("--out-dir F:/o/t0std", "--out-dir F:/o/elsewhere")],
                  [good_line("M2b-x", "t0fresh", tool=L.T0_TOOL).replace("--ages fresh", "--ages cycle")]):
        with pytest.raises(L.PlanError):
            L.parse_plan(plan_file(tmp_path, lines), "F:/t", "F:/o")
    p = tmp_path / "crlf.txt"
    p.write_bytes((good_line() + "\r\n").encode())
    with pytest.raises(L.PlanError, match="CR"):
        L.parse_plan(str(p), "F:/t", "F:/o")


def test_parse_plan_names_the_rule_by_which_it_refuses_a_short_line_and_a_foreign_tool(tmp_path):
    short = f"{PY} F:/t/stack/scripts/{L.T1_TOOL} run --ckpt E:/x/ckpt_last.pt"                             # 5 tokens: no real command is that short
    with pytest.raises(L.PlanError, match="first token must be"):
        L.parse_plan(plan_file(tmp_path, [short]), "F:/t", "F:/o")
    foreign = good_line("M2b-x", "t0std", tool=L.T0_TOOL).replace(L.T0_TOOL, "evil_tool.py", 1)           # under the tree's scripts dir, subcommand run, every flag valid: only the tool NAME is wrong
    assert "evil_tool.py" in foreign and L.T0_TOOL not in foreign
    with pytest.raises(L.PlanError, match="the tool must be"):
        L.parse_plan(plan_file(tmp_path, [foreign]), "F:/t", "F:/o")
    L.parse_plan(plan_file(tmp_path, [good_line("M2b-x", "t0std", tool=L.T0_TOOL)]), "F:/t", "F:/o")        # control: the same line with the real tool parses


def test_cmd_sha_is_the_nul_joined_sha256_and_fmt_quotes_only_what_needs_it():
    assert L.cmd_sha(["a", "b"]) == hashlib.sha256(b"a\0b\0").hexdigest() and L.cmd_sha(["ab"]) != L.cmd_sha(["a", "b"])
    assert L.fmt(["a", "b c", "d&e", "x/y:z=1,2@3%4+5-6.7_8"]) == "a 'b c' 'd&e' x/y:z=1,2@3%4+5-6.7_8"


def test_expected_outputs_and_step_problems(tmp_path):
    o = str(tmp_path).replace("\\", "/")
    st = L.parse_plan(plan_file(tmp_path, [good_line(out=o, tree="F:/t"), good_line("M2b-x", "t0std", tool=L.T0_TOOL, out=o)]), "F:/t", o)
    e0, e1 = L.expected(st[0]), L.expected(st[1])
    assert e0["files"] == [f"{o}/t1v8f_SOUP_s0_m10_causal_heads.json", f"{o}/ticks/ticks_SOUP_s0_m10_causal_heads.npz", f"{o}/argv/SOUP_s0_m10_causal_heads.argv"]
    assert e0["dumps"] == f"{o}/dump_SOUP_s0_m10_causal_heads" and e0["done"] == f"{o}/done_SOUP_s0_m10_causal_heads.txt"
    assert e1["files"] == [f"{o}/t0std/eval_M2b-x.json", f"{o}/t0std/eval_M2b-x.npz"] and e1["dumps"] is None and e1["done"] == f"{o}/done_t0_M2b-x.txt" and e1["extra"] == [f"{o}/t0std/eval_M2b-x_proj.npz"]
    assert len(L.step_problems(st[0])) == 4 and len(L.step_problems(st[1])) == 2
    for f in e0["files"]:
        w8(f, "x")
    os.makedirs(e0["dumps"])
    for i in range(140):
        w8(f"{e0['dumps']}/ep{i:03d}.npz", "d")
    assert L.step_problems(st[0]) == [f"{e0['dumps']}: 140 dumps < 141"]
    w8(f"{e0['dumps']}/ep140.npz", "d")
    w8(f"{e0['dumps']}/other.npz", "d")
    assert L.step_problems(st[0]) == [] and L.n_dumps(e0["dumps"]) == 141
    open(e0["files"][1], "w").close()
    assert "empty" in L.step_problems(st[0])[0]


def test_step_done_needs_every_field_and_all_outputs(tmp_path):
    o = str(tmp_path).replace("\\", "/")
    st = L.parse_plan(plan_file(tmp_path, [good_line(out=o), good_line("M2b-x", "t0std", tool=L.T0_TOOL, out=o)]), "F:/t", o)
    for s in st:
        for f in L.expected(s)["files"]:
            w8(f, "x")
        if L.expected(s)["dumps"]:
            for i in range(141):
                w8(f"{L.expected(s)['dumps']}/ep{i:03d}.npz", "d")
    S, T, C = "a" * 64, "t" * 64, "c" * 64
    assert L.step_done(st[0], S, T, C)[0] is False
    rec = L.done_record(st[0], S, T, C, 5)
    w8(L.expected(st[0])["done"], json.dumps(rec))
    assert L.step_done(st[0], S, T, C) == (True, "complete")
    assert rec["ckpt_sha256_before"] == rec["ckpt_sha256_after"] == S and rec["smoke"] is False and rec["max_windows"] is None and rec["n_dumps"] == 141 and rec["limit"] == "causal"
    for k, bad in (("ckpt_sha256", "b" * 64), ("ckpt_sha256_before", "b" * 64), ("ckpt_sha256_after", "b" * 64), ("tool_digest", "u" * 64), ("cmd_sha256", "d" * 64), ("tag", "other"), ("smoke", True), ("max_windows", 10)):
        w8(L.expected(st[0])["done"], json.dumps({**rec, k: bad}))
        assert L.step_done(st[0], S, T, C)[0] is False, k
    w8(L.expected(st[0])["done"], "not json")
    assert "unreadable" in L.step_done(st[0], S, T, C)[1]
    w8(L.expected(st[0])["done"], json.dumps(rec))
    os.remove(L.expected(st[0])["files"][0])
    assert L.step_done(st[0], S, T, C)[0] is False and "missing" in L.step_done(st[0], S, T, C)[1]
    assert L.done_record(st[1], S, T, C, 1, before="x", after="y")["ckpt_sha256_before"] == "x" and L.done_record(st[1], S, T, C, 1)["n_dumps"] is None and L.done_record(st[1], S, T, C, 1)["limit"] is None


def test_world_ok_needs_the_marker_and_a_place_outside_production(tmp_path, monkeypatch):
    w = tmp_path / "wd"
    w.mkdir()
    monkeypatch.setenv("SOUP_WORLD", str(w))
    assert L.world_ok() is False                                                                           # no marker
    (w / L.WORLD_MARKER).write_text("something else\n")
    assert L.world_ok() is False                                                                           # wrong token
    (w / L.WORLD_MARKER).write_text(L.WORLD_TOKEN + "\n")
    assert L.world_ok() is True
    monkeypatch.setenv("SOUP_WORLD", str(w / "nope"))
    assert L.world_ok() is False                                                                           # not a directory
    monkeypatch.delenv("SOUP_WORLD")
    assert L.world_ok() is False
    for attr in ("OUT_PROD", "HERE", "TREE_PROD"):                           # a valid marker inside / at / above a production place is still refused (the places are swapped for temp dirs)
        base = tmp_path / f"prod_{attr}"
        (base / "inside").mkdir(parents=True)
        for d in (base, base / "inside", tmp_path):
            (d / L.WORLD_MARKER).write_text(L.WORLD_TOKEN + "\n")
        with monkeypatch.context() as mp:
            mp.setattr(L, attr, str(base).replace("\\", "/"))
            for d in (base / "inside", base, tmp_path):
                mp.setenv("SOUP_WORLD", str(d))
                assert L.world_ok() is False, (attr, str(d))
    monkeypatch.setenv("SOUP_WORLD", str(w))
    assert L.world_ok() is True


def test_guard_hooks_are_honoured_only_in_a_valid_world_and_each_guard_reads_its_own_hook(tmp_path, monkeypatch):
    w = tmp_path / "wd"
    w.mkdir()
    (w / L.WORLD_MARKER).write_text(L.WORLD_TOKEN + "\n")
    (w / "hook_ram_mb.txt").write_text("100")
    monkeypatch.setenv("SOUP_WORLD", str(w))
    rc, txt = L.guards(256, "C:/", 0.1, 0)
    assert rc == 1 and "RAM available 100 MB < 256 MB" in txt and "free" not in txt.split("|")[0]
    (w / "hook_ram_mb.txt").unlink()
    (w / "hook_disk_gb.txt").write_text("0.2")
    rc, txt = L.guards(1, "C:/", 5.0, 0)
    assert rc == 1 and "C:/ free 0.2 GB < 5.0 GB" in txt and "RAM" not in txt.split("|")[0]
    (w / "hook_disk_gb.txt").unlink()
    (w / "hook_c_mb.txt").write_text("50")
    rc, txt = L.guards(1, "C:/", 0.0, 100)
    assert rc == 1 and "C: free 50 MB < 100 MB" in txt
    (w / "hook_c_mb.txt").unlink()
    assert L.guards(1, "C:/", 0.0, 0)[0] == 0
    (w / "hook_ram_mb.txt").write_text("1")
    monkeypatch.delenv("SOUP_WORLD")
    assert L.guards(2, "C:/", 0.0, 0)[0] == 0                                                              # hooks are ignored outside a world
    (w / L.WORLD_MARKER).unlink()
    monkeypatch.setenv("SOUP_WORLD", str(w))                                                               # ... and in a directory that holds the hooks but is not a world (no marker): the hook (1 MB < 2 MB) is not honoured
    assert L.guards(2, "C:/", 0.0, 0)[0] == 0


def mini_state(tmp_path):
    spec = {"members": {"M": {"path": f"{tmp_path}/m/ckpt_last.pt", "sha256": ""}}, "soups": {}}
    w8(f"{tmp_path}/m/ckpt_last.pt", "member")
    spec["members"]["M"]["sha256"] = sha(f"{tmp_path}/m/ckpt_last.pt")
    w8(f"{tmp_path}/s/ckpt_last.pt", "soup")
    return spec, [{"ckpt": f"{tmp_path}/m/ckpt_last.pt".replace("\\", "/")}, {"ckpt": f"{tmp_path}/s/ckpt_last.pt".replace("\\", "/")}]


def test_ready_checks_soups_against_their_record_and_controls_against_the_spec(tmp_path):
    spec, steps = mini_state(tmp_path)
    steps = [{**steps[0]}, {**steps[0]}, steps[1]]                                                         # a checkpoint used by two steps is checked once
    rec = {"output_sha256": sha(steps[2]["ckpt"]), "K-BASIN": {"PASS": True}}
    spec["members"]["M"]["path"] = spec["members"]["M"]["path"].replace("/", "\\")                         # normalised on comparison
    w8(f"{tmp_path}/s/soup.json", json.dumps(rec))
    assert L.ready(steps, spec) == []
    for bad, why in (({**rec, "output_sha256": "0" * 64}, "output_sha256"), ({**rec, "K-BASIN": {"PASS": False}}, "K-BASIN"), ({**rec, "K-BASIN": {}}, "K-BASIN"), ({"output_sha256": rec["output_sha256"]}, "K-BASIN"),
                    ({**rec, "K-BASIN": {"PASS": 1}}, "K-BASIN"), ({**rec, "K-BASIN": {"PASS": "true"}}, "K-BASIN")):         # truthy is not true: only a JSON true is a PASS
        w8(f"{tmp_path}/s/soup.json", json.dumps(bad))
        assert any(why in p for p in L.ready(steps, spec)), bad
    w8(f"{tmp_path}/s/soup.json", json.dumps(rec))
    spec["members"]["M"]["sha256"] = "0" * 64
    assert any("spec's M" in p for p in L.ready(steps, spec))
    spec["members"]["M"]["sha256"] = sha(steps[0]["ckpt"])
    os.remove(steps[0]["ckpt"])
    assert any("not a file" in p for p in L.ready(steps, spec))
    w8(f"{tmp_path}/o/ckpt_last.pt", "other")
    assert any("neither a soup" in p for p in L.ready([{"ckpt": f"{tmp_path}/o/ckpt_last.pt".replace("\\", "/")}], spec))


def test_pins_check_reports_every_kind_of_difference(tmp_path):
    spec, steps = mini_state(tmp_path)
    w8(f"{tmp_path}/t/stack/tanitad/a.py", "x")
    w8(f"{tmp_path}/t/stack/scripts/v8f_t1_eval.py", "x")
    w8(f"{tmp_path}/t/stack/scripts/v8f_mini_eval.py", "t0 tool")
    w8(f"{tmp_path}/t/taniteval/b.py", "x")
    f1, f2 = f"{tmp_path}/f1.txt".replace("\\", "/"), f"{tmp_path}/f2.txt".replace("\\", "/")
    w8(f1, "one")
    w8(f2, "two")
    files, tree = {"f1": f1, "f2": f2}, f"{tmp_path}/t".replace("\\", "/")
    ents = L.pin_entries(files, steps, tree)
    assert len(ents) == 6 and ents[0].startswith("file=f1|") and ents[2].startswith("ckpt=") and ents[4] == f"tool={tree}/stack/scripts/v8f_mini_eval.py {sha(tree + '/stack/scripts/v8f_mini_eval.py')}"
    assert ents[5] == f"tree={tree} {L.tree_digest(tree)}"
    pins = f"{tmp_path}/PINS.txt"
    w8(pins, "# header\n" + "\n".join(ents) + "\n")
    assert L.pins_check(pins, files, steps, tree, None) == []
    assert L.pins_check(pins, files, steps, tree, L.tree_digest(tree)) == []
    assert any("tool tree digest" in d for d in L.pins_check(pins, files, steps, tree, "0" * 64))
    w8(f1, "changed")
    assert len(L.pins_check(pins, files, steps, tree, None)) == 1
    w8(f1, "one")
    w8(steps[1]["ckpt"], "soup!")
    assert any(d.startswith("pin ckpt=") for d in L.pins_check(pins, files, steps, tree, None))
    w8(steps[1]["ckpt"], "soup")
    w8(f"{tmp_path}/t/taniteval/b.py", "y")
    assert any(d.startswith("pin tree=") for d in L.pins_check(pins, files, steps, tree, None))
    w8(f"{tmp_path}/t/taniteval/b.py", "x")
    w8(f"{tmp_path}/t/stack/scripts/v8f_mini_eval.py", "t0 tool changed")
    assert any(d.startswith("pin tool=") for d in L.pins_check(pins, files, steps, tree, None))              # the T0 tool is pinned by itself (it is not in the tree digest)
    w8(f"{tmp_path}/t/stack/scripts/v8f_mini_eval.py", "t0 tool")
    assert any("NO PIN" in d for d in L.pins_check(pins, {**files, "f3": f1}, steps, tree, None))          # an input with no pin line
    assert any("ABSENT" in d for d in L.pins_check(pins, {"f1": f1}, steps, tree, None))                    # a pin line that names no input
    assert L.pins_check(f"{tmp_path}/none.txt", files, steps, tree, None)[0].startswith("no PINS file")


def test_the_tree_digest_covers_the_python_files_and_the_t1_tool_and_nothing_else(tmp_path):
    t = f"{tmp_path}/t"
    for p in ("stack/tanitad/a.py", "stack/tanitad/sub/b.py", "taniteval/c.py", "taniteval/deep/d.py", "stack/scripts/v8f_t1_eval.py"):
        w8(f"{t}/{p}", p)
    d0 = L.tree_digest(t)
    assert L.digest_set(t) == sorted(["stack/tanitad/a.py", "stack/tanitad/sub/b.py", "taniteval/c.py", "taniteval/deep/d.py", "stack/scripts/v8f_t1_eval.py"], key=lambda s: s.encode())
    w8(f"{t}/stack/scripts/other.py", "x")
    w8(f"{t}/stack/tanitad/notes.txt", "x")
    w8(f"{t}/elsewhere/e.py", "x")
    assert L.tree_digest(t) == d0
    for p in ("stack/tanitad/sub/b.py", "taniteval/deep/d.py", "stack/scripts/v8f_t1_eval.py"):
        w8(f"{t}/{p}", "changed")
        assert L.tree_digest(t) != d0, p
        w8(f"{t}/{p}", p)
    assert L.tree_digest(t) == d0
    lines = "".join(f"{hashlib.sha256(p.encode()).hexdigest()} *{p}\n" for p in L.digest_set(t))
    assert d0 == hashlib.sha256(lines.encode()).hexdigest()


def test_manifest_from_the_plan_and_the_li_runs(tmp_path):
    o = str(tmp_path).replace("\\", "/") + "/out"
    spec = {"soups": {"SOUP_s0": {"seed": 0, "role": "SOUP", "run_dir": "M2b_s0_soup"}, "SOUPR_s0": {"seed": 0, "role": "SOUPR", "run_dir": "M2b_s0_soupr"}, "XINIT_s0": {"seed": 0, "role": "XINIT", "run_dir": "X", "expect": "K-BASIN FAIL"}}}
    lines = []
    for soup in ("SOUP_s0", "SOUPR_s0"):
        ck = f"E:/c/{'M2b_s0_soup' if soup == 'SOUP_s0' else 'M2b_s0_soupr'}/ckpt_last.pt"
        lines += [good_line(f"{soup}_m10_causal_heads", "causal", ck, out=o), good_line(f"{soup}_m10_logged_heads", "logged", ck, out=o), good_line(f"M2b-{soup.lower()}-fresh", "t0fresh", ck, L.T0_TOOL, o),
                  good_line(f"M2b-{soup.lower()}", "t0std", ck, L.T0_TOOL, o)]
    lines += [good_line("M2b-dagoacc_0-dev", "t0std", "E:/d/ckpt_last.pt", L.T0_TOOL, o), good_line("M2b-a_0-dev", "t0std", "E:/a/ckpt_last.pt", L.T0_TOOL, o)]
    steps = L.parse_plan(plan_file(tmp_path, lines), "F:/t", o)
    runs = {"seeds": [0], "arms": {k: {"0": {"fake": k}} for k in ("CTRL", "REP", "A", "Arep", "B")}, "build": {"b": 1}}
    with pytest.raises(L.PlanError):
        L.build_manifest(runs, steps, spec, o)                                                              # nothing ran yet
    for s in steps:
        for f in L.expected(s)["files"]:
            w8(f, "x")
        if L.expected(s)["dumps"]:
            for i in range(141):
                w8(f"{L.expected(s)['dumps']}/ep{i:03d}.npz", "d")
        w8(L.expected(s)["done"], "{}")
    m = L.build_manifest(runs, steps, spec, o)
    assert m["seeds"] == [0] and set(m["arms"]) == {"CTRL", "REP", "A", "Arep", "SOUP", "SOUPR"} and m["build"] == {"b": 1}
    e = m["arms"]["SOUP"]["0"]
    assert e["ckpt_tail"] == "M2b_s0_soup/ckpt_last.pt" and e["soup_dir"] == "E:/c/M2b_s0_soup"
    assert e["t1"] == {"tag": "SOUP_s0_m10_causal_heads", "ticks": f"{o}/ticks/ticks_SOUP_s0_m10_causal_heads.npz", "dump_dir": f"{o}/dump_SOUP_s0_m10_causal_heads", "json": f"{o}/t1v8f_SOUP_s0_m10_causal_heads.json",
                       "argv": f"{o}/argv/SOUP_s0_m10_causal_heads.argv", "done": f"{o}/done_SOUP_s0_m10_causal_heads.txt"}
    assert e["t1_logged"]["tag"] == "SOUP_s0_m10_logged_heads" and e["t1_logged"]["json"].endswith("t1v8f_SOUP_s0_m10_logged_heads.json")
    assert e["t0"] == {"json": f"{o}/t0std/eval_M2b-soup_s0.json", "npz": f"{o}/t0std/eval_M2b-soup_s0.npz"} and e["t0_fresh"]["json"] == f"{o}/t0std/eval_M2b-soup_s0-fresh.json"
    assert m["arms"]["SOUPR"]["0"]["ckpt_tail"] == "M2b_s0_soupr/ckpt_last.pt" and m["arms"]["SOUPR"]["0"]["t0"]["json"].endswith("eval_M2b-soupr_s0.json")
    assert m["arms"]["CTRL"]["0"]["t0_dev"] == {"json": f"{o}/t0std/eval_M2b-dagoacc_0-dev.json", "npz": f"{o}/t0std/eval_M2b-dagoacc_0-dev.npz"} and m["arms"]["A"]["0"]["t0_dev"]["json"].endswith("eval_M2b-a_0-dev.json")
    assert "t0_dev" not in m["arms"]["REP"]["0"] and "t0_dev" not in m["arms"]["Arep"]["0"] and runs["arms"]["CTRL"]["0"] == {"fake": "CTRL"}      # the LAUNCH-INTENT runs.json is not modified
    os.remove(L.expected(steps[3])["done"])
    with pytest.raises(L.PlanError, match="no done record"):
        L.build_manifest(runs, steps, spec, o)
    w8(L.expected(steps[3])["done"], "{}")
    f2 = L.expected(steps[2])["files"][0]                                                                      # a step whose done record stands but whose output file is gone
    os.rename(f2, f2 + ".gone")
    with pytest.raises(L.PlanError, match=re.escape(f"step {steps[2]['tag']}:")):
        L.build_manifest(runs, steps, spec, o)
    os.rename(f2 + ".gone", f2)
    d0 = L.expected(steps[0])["dumps"]
    os.remove(f"{d0}/ep140.npz")                                                                              # ... or whose dump directory is one episode short
    with pytest.raises(L.PlanError, match=re.escape(f"step {steps[0]['tag']}:")):
        L.build_manifest(runs, steps, spec, o)
    w8(f"{d0}/ep140.npz", "d")
    L.build_manifest(runs, steps, spec, o)
    lines2 = [ln for ln in lines if "M2b-a_0-dev" not in ln]
    with pytest.raises(L.PlanError, match="no step with tag"):
        L.build_manifest(runs, L.parse_plan(plan_file(tmp_path, lines2), "F:/t", o), spec, o)


def test_parsecheck_cuts_the_run_off_before_anything_executes(tmp_path):
    tool = f"{tmp_path}/tool.py"
    ran = f"{tmp_path}/RAN.flag"
    body = "import argparse, sys\ndef main(argv):\n    ap = argparse.ArgumentParser()\n    ap.add_argument('--x')\n    ap.parse_args(argv)\n    open(" + repr(ran) + ", 'w').close()\n"     # the parse result is DISCARDED: nothing but the cut-off stops the write
    w8(tool, body)
    ok, ns = L.parsecheck(tool, ["--x", "1"])
    assert ok and ns == {"x": "1"} and not os.path.exists(ran)                                              # parsed, and the run after the parse never happened
    w8(f"{tmp_path}/ctl.py", body + "\nif __name__ == '__main__':\n    main(sys.argv[1:])\n")
    subprocess.run([sys.executable, "-I", f"{tmp_path}/ctl.py", "--x", "1"], check=True)
    assert os.path.exists(ran)                                                                               # control: the same tool DOES write it when it is really run, so the absence above means something
    os.remove(ran)
    ok, err = L.parsecheck(tool, ["--nope"])
    assert not ok and "unrecognized" in err


# ================================================================================================ the chain, end to end (stub tool, stub reader)
def test_dry_run_prints_every_command_parses_each_with_the_tools_own_argparse_and_writes_nothing(w):
    w.pins()
    rc, out, err = w.chain("--dry-run")
    assert rc == 0 and out.rstrip().endswith("ZZSOUP-DRY-OK"), out + err
    assert len(re.findall(r"parse: PARSE-OK", out)) == 6 and "== reader:" in out and "--prereg-sha " + w.prereg_sha in out and "PINS.txt verified" in out and "readiness: 3 checkpoints ready" in out
    assert not os.path.exists(w.out) and not os.path.exists(f"{w.r}/logs") and w.calls() == []              # nothing created, nothing run


def test_dry_run_reports_a_tool_that_does_not_parse_and_an_unset_prereg_as_not_ready(w):
    w.pins()
    rc, out, _ = w.chain("--dry-run", env={"SOUP_PREREG": "", "SOUP_PREREG_SHA": ""})
    assert rc == 1 and "ZZSOUP-DRY-NOT-READY PREREG:" in out and "a real run would STOP here" in out and "ZZSOUP-DRY-OK" not in out
    w8(f"{w.r}/tree/stack/scripts/v8f_mini_eval.py", "def main(argv):\n    raise SystemExit(2)\n")
    rc, out, _ = w.chain("--dry-run")
    assert rc == 1 and "ZZSOUP-STOP PINS.txt" in out and "v8f_mini_eval.py" in out                           # a changed T0 tool is a pin difference, caught before any parse
    w.pins()
    rc, out, _ = w.chain("--dry-run")
    assert rc == 1 and "ZZSOUP-DRY-NOT-READY step M2b-soup_s0-fresh does not parse" in out and "ZZSOUP-DRY-OK" not in out and out.count("PARSE-OK") == 2


def test_write_pins_covers_every_input_every_checkpoint_and_the_tree(w):
    rc, out, err = w.chain("--write-pins")
    assert rc == 0 and re.search(r"wrote .*PINS.txt \(18 pins\)", out), out + err
    t = open(f"{w.r}/PINS.txt", encoding="utf-8").read()
    lines = [ln for ln in t.split("\n") if ln and not ln.startswith("#")]
    labs = sorted(ln.split("|")[0].split("=")[1] for ln in lines if ln.startswith("file="))
    assert labs == sorted(["plan", "spec", "builder", "lib", "chain", "reader", "li_reader", "banked_li_read", "li_runs"] + [f"tmpl:{x}" for x in os.listdir(f"{w.r}/argv") if x.endswith(".tmpl")])
    assert sorted(ln.split(" ")[0] for ln in lines if ln.startswith("ckpt=")) == sorted(f"ckpt={w.ck[k]}" for k in ("soup", "dag", "a"))
    assert [ln for ln in lines if ln.startswith("tree=")] == [f"tree={w.r}/tree {L.tree_digest(w.r + '/tree')}"]
    assert [ln for ln in lines if ln.startswith("tool=")] == [f"tool={w.r}/tree/stack/scripts/v8f_mini_eval.py {sha(w.r + '/tree/stack/scripts/v8f_mini_eval.py')}"]
    assert f"file=reader|{w.r}/reader/soup_read.py {sha(w.r + '/reader/soup_read.py')}" in lines and f"file=chain|{CHAIN}/soup_chain.sh {sha(CHAIN + '/soup_chain.sh')}" in lines and f"file=lib|{CHAIN}/soup_chain_lib.py {sha(CHAIN + '/soup_chain_lib.py')}" in lines


def test_write_pins_refuses_an_unready_state(w):
    w8(f"{w.r}/ckpt/dagoacc0/ckpt_last.pt", "tampered")
    rc, out, _ = w.chain("--write-pins")
    assert rc == 1 and "ZZSOUP-STOP a checkpoint is not ready" in out and not os.path.exists(f"{w.r}/PINS.txt")


def test_the_real_run_executes_every_step_once_in_plan_order_writes_done_records_and_calls_the_reader(pw):
    w = pw
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 0, out + err + log
    assert w.calls() == World.TAGS                                                                          # once each, in plan order
    assert [m.split(" ")[0] for m in markers(log)] == [f"ZZSOUP-STEP-DONE"] * 6 + ["ZZSOUP-DONE"] and "ZZSOUP-STEP-DONE 6/6 M2b-a_0-dev (t0dev)" in log
    assert re.search(r"^ZZSOUP-DONE 6/6 steps, reader rc 0 \(outcome written\): ", log, re.M)
    tool_digest = L.tree_digest(w.r + "/tree")
    for tag, ck in zip(World.TAGS, ["soup", "soup", "soup", "soup", "dag", "a"]):
        d = w.done(tag)
        assert d["ckpt_sha256"] == d["ckpt_sha256_before"] == d["ckpt_sha256_after"] == sha(w.ck[ck]) and d["tool_digest"] == tool_digest and d["smoke"] is False and d["max_windows"] is None and d["tag"] == tag
        argv = open(f"{w.out}/argv/{tag}.argv", encoding="utf-8").read().split("\n")[:-1] if tag.startswith("SOUP") else shlex_split(d["command"])
        assert d["cmd_sha256"] == hashlib.sha256(b"".join(a.encode() + b"\0" for a in argv)).hexdigest()
    assert w.done(World.TAGS[0])["n_dumps"] == 141 and w.done(World.TAGS[2])["n_dumps"] is None
    rc_ = json.load(open(f"{w.r}/reader_call.json"))
    a = rc_["args"]
    assert a["prereg"] == w.r + "/prereg.md" and a["prereg_sha"] == w.prereg_sha and a["li_reader_sha"] == sha(w.r + "/li_reader.py") and a["soup_build_sha"] == sha(w.r + "/builder.py")
    assert a["manifest"] == w.out + "/runs_soup.json" and a["out"] == w.out + "/read/SOUP_READ.json" and a["spec"] == w.r + "/soups.json" and a["orc_dir"] == w.r + "/orc" and a["work_dir"] == w.out + "/_tmp" and rc_["prereg_exists"]
    assert rc_["TANITEVAL_STACK_OVERRIDE"] is None and rc_["PYTHONPATH"] is None and rc_["omp"] == "2" and rc_["mkl"] == "2" and rc_["cuda"] == "-1"   # the reader runs in the environment its dry run was tested in
    assert os.path.normpath(rc_["TMP"]).lower() == os.path.normpath(w.out + "/_tmp").lower()                 # temp files stay off C:
    assert rc_["slots"]["SOUP"] == ["0"] and rc_["slots"]["CTRL"] == ["0"] and os.path.isfile(a["out"])
    man = json.load(open(a["manifest"], encoding="utf-8"))
    assert man["arms"]["CTRL"]["0"]["t0_dev"]["json"] == f"{w.out}/t0dev/eval_M2b-dagoacc_0-dev.json" and man["arms"]["SOUP"]["0"]["t0_fresh"]["npz"] == f"{w.out}/t0fresh/eval_M2b-soup_s0-fresh.npz"
    assert not os.path.exists(f"{w.r}/logs/chain.lock") and w.waker()[0] == 0


def shlex_split(s):
    import shlex
    return shlex.split(s)


def test_resume_skips_finished_steps_and_rotates_the_log(pw):
    w = pw
    assert w.chain()[0] == 0
    first = w.log()
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 0 and w.calls() == World.TAGS and log.count("skip ") == 6 and "ZZSOUP-DONE 6/6" in log and "ZZSOUP-STEP-DONE 6/6" in log
    prev = [f for f in os.listdir(f"{w.r}/logs") if f.startswith("chain.out.prev_")]
    assert len(prev) == 1 and open(f"{w.r}/logs/{prev[0]}", encoding="utf-8").read() == first
    assert not [d for d in os.listdir(w.out) if d.startswith("_incomplete_")]


def test_a_step_whose_done_record_is_wrong_is_run_again_and_its_old_output_is_moved_aside_never_deleted(pw):
    w = pw
    assert w.chain()[0] == 0
    tag = World.TAGS[3]
    d = w.done(tag)
    for k, bad in (("tool_digest", "0" * 64), ("cmd_sha256", "0" * 64), ("ckpt_sha256_after", "0" * 64)):
        before = w.calls().count(tag)
        w8(f"{w.out}/done_t0_{tag}.txt", json.dumps({**d, k: bad}))
        old = open(f"{w.out}/t0std/eval_{tag}.json", encoding="utf-8").read()
        rc, out, err = w.chain()
        assert rc == 0 and w.calls().count(tag) == before + 1, k
        asides = sorted(x for x in os.listdir(w.out) if x.startswith("_incomplete_"))
        assert asides and any(open(lp(f"{w.out}/{x}/eval_{tag}.json"), encoding="utf-8").read() == old for x in asides if os.path.exists(lp(f"{w.out}/{x}/eval_{tag}.json"))), k
        assert w.done(tag)["cmd_sha256"] == d["cmd_sha256"]
    assert w.calls().count(World.TAGS[0]) == 1


def test_an_incomplete_earlier_attempt_is_moved_aside_before_the_step_runs(pw):
    w = pw
    tag = World.TAGS[0]
    w8(f"{w.out}/t1v8f_{tag}.json", "PARTIAL")
    os.makedirs(f"{w.out}/dump_{tag}")
    w8(f"{w.out}/dump_{tag}/ep000.npz", "PARTIAL-DUMP")
    assert w.chain()[0] == 0
    asides = [x for x in os.listdir(w.out) if x.startswith("_incomplete_")]
    assert len(asides) == 1 and open(lp(f"{w.out}/{asides[0]}/t1v8f_{tag}.json")).read() == "PARTIAL" and open(lp(f"{w.out}/{asides[0]}/dump_{tag}/ep000.npz")).read() == "PARTIAL-DUMP"
    assert "moved the output of an earlier attempt" in w.log() and w.calls().count(tag) == 1 and json.load(open(f"{w.out}/t1v8f_{tag}.json"))["tag"] == tag


def test_a_failed_step_stops_the_chain_and_a_resume_finishes_it(pw):
    w = pw
    w.hook(f"stub_fail_{World.TAGS[2]}.txt")
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 1 and w.calls() == World.TAGS[:3]                                                         # nothing after the failed step ran
    assert re.search(r"^ZZSOUP-STOP step M2b-soup_s0-fresh failed \(rc 3\)", log, re.M) and "ZZSOUP-DONE" not in log and os.path.exists(f"{w.r}/logs/STOP")
    assert os.path.isfile(f"{w.out}/done_{World.TAGS[0]}.txt") and os.path.isfile(f"{w.out}/done_{World.TAGS[1]}.txt") and not os.path.exists(f"{w.out}/done_t0_{World.TAGS[2]}.txt")
    assert w.waker()[0] == 4
    os.remove(f"{w.r}/stub_fail_{World.TAGS[2]}.txt")
    rc, out, err = w.chain()
    assert rc == 0 and w.calls() == World.TAGS[:3] + World.TAGS[2:] and "ZZSOUP-DONE 6/6" in w.log() and not os.path.exists(f"{w.r}/logs/STOP")
    assert any(x.startswith("_incomplete_") for x in os.listdir(w.out)) and w.log().count("skip ") == 2


def test_a_step_that_exits_zero_without_its_outputs_is_a_failure(pw):
    w = pw
    shutil.copy(f"{HERE}/soup_chain_stub_tool.py", f"{w.r}/stub_orig.py")
    w8(f"{w.r}/tree/stack/scripts/v8f_mini_eval.py", "import sys\ndef main(argv):\n    print('claims success')\n\nif __name__ == '__main__':\n    main(sys.argv[1:])\n")
    w.pins()                                                                                                # the tree changed: re-pin
    rc, out, err = w.chain()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP step M2b-soup_s0-fresh: missing or empty", w.log(), re.M) and not os.path.exists(f"{w.out}/done_t0_M2b-soup_s0-fresh.txt")


def test_a_native_crash_is_retried_once_and_the_crashed_attempt_is_never_kept(pw):
    w = pw
    tag = World.TAGS[0]
    w.hook(f"stub_segv_{tag}.txt")
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 0 and w.calls().count(tag) == 2 and "native crash (rc 139)" in log and os.path.isfile(f"{w.r}/logs/{tag}.segv.out")
    assert any(x.startswith("_incomplete_") for x in os.listdir(w.out)) and os.path.isfile(f"{w.out}/argv/{tag}.argv") and w.done(tag)["n_dumps"] == 141
    w.hook(f"stub_segv_{World.TAGS[1]}.txt")
    w.hook(f"stub_fail_{World.TAGS[1]}.txt")
    shutil.rmtree(w.out)
    rc, out, err = w.chain()
    assert rc == 1 and "ZZSOUP-STOP step SOUP_s0_m10_logged_heads failed (rc 3)" in w.log()              # a second failure is not retried again


def test_the_stop_file_ends_the_chain_before_the_next_step_and_never_loses_a_finished_step(pw):
    w = pw
    w.hook(f"stub_stop_{World.TAGS[1]}.txt", f"{w.r}/logs/STOP")
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 1 and w.calls() == World.TAGS[:2] and re.search(r"^ZZSOUP-STOP the STOP file ", log, re.M)
    assert os.path.isfile(f"{w.out}/done_{World.TAGS[1]}.txt")                                              # the step that was running finished and was recorded
    os.remove(f"{w.r}/stub_stop_{World.TAGS[1]}.txt")
    rc, out, err = w.chain()                                                                                # a restart removes a stale STOP and resumes
    assert rc == 0 and w.calls() == World.TAGS and "ZZSOUP-DONE 6/6" in w.log()


@pytest.mark.parametrize("env,why", [({"SOUP_PREREG": ""}, "must BOTH be set"), ({"SOUP_PREREG_SHA": ""}, "must BOTH be set"), ({"SOUP_PREREG_SHA": "abc"}, "not a 64-hex"), ({"SOUP_PREREG_SHA": "G" * 64}, "not a 64-hex"),
                                      ({"SOUP_PREREG": "{W}/nope.md"}, "is not a file"), ({"SOUP_PREREG_SHA": "0" * 64}, "hashes to")])
def test_the_prereg_and_its_sha_must_both_come_from_the_environment_and_match(pw, env, why):
    w = pw
    rc, out, err = w.chain(env={k: v.replace("{W}", w.r) for k, v in env.items()})
    log = w.log()
    assert rc == 1 and why in out and why in log and re.search(r"^ZZSOUP-STOP ", log, re.M) and w.calls() == [] and not os.path.exists(w.out)
    assert w.waker()[0] == 4


def test_unset_prereg_variables_are_refused_when_absent_from_the_environment_altogether(pw):
    w = pw
    rc, out, err = w.chain(env={"SOUP_PREREG": None, "SOUP_PREREG_SHA": None})
    assert rc == 1 and "must BOTH be set" in w.log() and w.calls() == [] and not os.path.exists(w.out)


@pytest.mark.parametrize("what", ["plan", "tmpl", "spec", "builder", "li_reader", "li_reader_pins", "banked", "li_runs", "reader", "ckpt_dag", "tree"])
def test_a_changed_input_stops_the_chain_before_anything_runs(pw, what):
    w = pw
    chain_env = {}
    if what == "plan":
        w8(f"{w.r}/argv/PLAN.txt", open(f"{w.r}/argv/PLAN.txt", encoding="utf-8").read() + "# a comment changes the pin\n")
    elif what == "tmpl":
        t = [x for x in os.listdir(f"{w.r}/argv") if x.endswith(".tmpl")][0]
        w8(f"{w.r}/argv/{t}", open(f"{w.r}/argv/{t}", encoding="utf-8").read() + "# x\n")
    elif what == "ckpt_dag":
        w8(f"{w.r}/ckpt/dagoacc0/ckpt_last.pt", "other bytes")
    elif what == "tree":
        w8(f"{w.r}/tree/stack/tanitad/__init__.py", "# changed\n")
    else:
        f = {"spec": "soups.json", "builder": "builder.py", "li_reader": "li_reader.py", "li_reader_pins": "li_reader.py", "banked": "banked.json", "li_runs": "li_runs.json", "reader": "reader/soup_read.py"}[what]
        if what == "li_reader":
            chain_env = {"SOUP_LI_READER_SHA": sha(f"{w.r}/{f}")}                       # the environment still names the pinned LAUNCH-INTENT reader sha: the chain's own pin check fires first
        w8(f"{w.r}/{f}", open(f"{w.r}/{f}", encoding="utf-8").read() + " ")             # li_reader_pins: the environment follows the file, only PINS.txt can catch it
    rc, out, err = w.chain(env=chain_env)
    log = w.log()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP ", log, re.M) and w.calls() == [] and not os.path.exists(w.out), (what, log[-400:])
    assert {"li_reader": "the LAUNCH-INTENT reader", "ckpt_dag": "a checkpoint is not ready"}.get(what, "PINS.txt: pin ") in log, (what, log[-400:])
    if what == "li_reader":
        assert re.search(r"^ZZSOUP-STOP the LAUNCH-INTENT reader .* hashes to ", log, re.M), log[-400:]                # the sha check itself stops the chain; PINS.txt is only the second line of defence


@pytest.mark.parametrize("label", ["lib", "chain"])
def test_a_pinned_chain_script_or_lib_that_differs_from_its_pin_stops_the_chain(pw, label):
    w = pw
    lines = open(f"{w.r}/PINS.txt", encoding="utf-8").read().split("\n")
    i = [k for k, ln in enumerate(lines) if ln.startswith(f"file={label}|")][0]
    lines[i] = lines[i].rpartition(" ")[0] + " " + "0" * 64                                                 # the pin no longer matches the file on disk
    w8(f"{w.r}/PINS.txt", "\n".join(lines))
    rc, out, err = w.chain()
    assert rc == 1 and re.search(rf"^ZZSOUP-STOP PINS.txt: pin file={label}\|", w.log(), re.M) and w.calls() == []


@pytest.mark.parametrize("which", ["soup_basin", "soup_sha", "member_sha", "missing_ckpt"])
def test_readiness_stops_the_chain_when_a_checkpoint_is_not_what_it_should_be(pw, which):
    w = pw
    if which == "soup_basin":
        w8(f"{w.r}/ckpt/M2b_s0_soup/soup.json", json.dumps({"output_sha256": sha(w.ck["soup"]), "K-BASIN": {"PASS": False}}))
    elif which == "soup_sha":
        w8(f"{w.r}/ckpt/M2b_s0_soup/soup.json", json.dumps({"output_sha256": "0" * 64, "K-BASIN": {"PASS": True}}))
    elif which == "member_sha":
        spec = json.load(open(f"{w.r}/soups.json"))
        spec["members"]["A_0"]["sha256"] = "0" * 64
        w8(f"{w.r}/soups.json", json.dumps(spec))
    else:
        os.remove(w.ck["a"])
    rc, out, err = w.chain()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP a checkpoint is not ready", w.log(), re.M) and w.calls() == []


def test_a_low_guard_waits_then_proceeds_when_it_recovers(pw):
    w = pw
    w.hook("hook_ram_mb.txt", "100")

    def release():                                                                                          # the guard recovers once the chain has said it is waiting
        t0 = time.time()
        while time.time() - t0 < 120:
            if "waiting, a guard is LOW" in w.log():
                os.remove(f"{w.r}/hook_ram_mb.txt")
                return
            time.sleep(0.2)
    th = threading.Thread(target=release, daemon=True)
    th.start()
    rc, out, err = w.chain(env={"SOUP_GUARD_WAIT_S": "60", "SOUP_RAM_MIN_MB": "256"})
    th.join(5)
    log = w.log()
    assert rc == 0 and "waiting, a guard is LOW: LOW RAM available 100 MB < 256 MB" in log and "guards hold again" in log and w.calls() == World.TAGS


@pytest.mark.parametrize("hook,txt", [("hook_ram_mb.txt", "RAM available 100 MB < 256 MB"), ("hook_disk_gb.txt", "free 0.0 GB < 0.1 GB"), ("hook_c_mb.txt", "C: free 10 MB < 32 MB")])
def test_a_guard_that_stays_low_stops_the_chain_and_nothing_is_started(pw, hook, txt):
    w = pw
    w.hook(hook, {"hook_ram_mb.txt": "100", "hook_disk_gb.txt": "0.0", "hook_c_mb.txt": "10"}[hook])
    t0 = time.time()
    rc, out, err = w.chain(env={"SOUP_GUARD_WAIT_S": "2", **{"SOUP_RAM_MIN_MB": "256", "SOUP_F_MIN_GB": "0.1", "SOUP_C_MIN_MB": "32"}})
    log = w.log()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP a guard stayed LOW for 2 s before SOUP_s0_m10_causal_heads: LOW .*" + re.escape(txt), log, re.M) and w.calls() == [] and time.time() - t0 < 60


def test_the_c_drive_gate_at_the_start_stops_the_chain(pw):
    w = pw
    rc, out, err = w.chain(env={"SOUP_C_MIN_MB": "10000000"})
    assert rc == 1 and re.search(r"^ZZSOUP-STOP C: has \d+ MB free \(< 10000000 MB\)", w.log(), re.M) and w.calls() == []


@pytest.mark.parametrize("victim", ["own_ckpt", "tree", "next_ckpt"])
def test_a_checkpoint_or_the_tree_changing_during_the_chain_is_caught_by_the_re_hash(pw, victim):
    w = pw
    first = World.TAGS[0]
    path = {"own_ckpt": w.ck["soup"], "tree": f"{w.r}/tree/stack/tanitad/__init__.py", "next_ckpt": w.ck["dag"]}[victim]
    w.hook(f"stub_tamper_{first}.txt", path)
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP ", log, re.M) and "ZZSOUP-DONE" not in log
    if victim == "own_ckpt":
        assert "changed during SOUP_s0_m10_causal_heads" in log and w.calls() == [first] and not os.path.exists(f"{w.out}/done_{first}.txt")
    elif victim == "tree":
        assert "the tool tree changed (after SOUP_s0_m10_causal_heads)" in log and w.calls() == [first] and not os.path.exists(f"{w.out}/done_{first}.txt")
    else:
        assert "step M2b-dagoacc_0-dev: " in log and "PINS.txt has" in log and w.calls() == World.TAGS[:4]


@pytest.mark.parametrize("rc_in,desc", [(0, "outcome written"), (2, "REFUSED, nothing written"), (3, "UNPLACED, written"), (1, "reader crashed"), (7, "reader crashed")])
def test_the_reader_exit_code_is_reported_in_the_final_marker(pw, rc_in, desc):
    w = pw
    w.hook("stub_reader_rc.txt", str(rc_in))
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 0 and re.search(rf"^ZZSOUP-DONE 6/6 steps, reader rc {rc_in} \({re.escape(desc)}\): {re.escape(w.out)}/read/SOUP_READ.json", log, re.M), log[-600:]
    assert os.path.isfile(f"{w.out}/read/SOUP_READ.json") == (rc_in in (0, 3)) and w.waker()[0] == 0


def test_only_one_instance_runs_at_a_time_and_a_stale_lock_is_taken_over(pw):
    w = pw
    w8(f"{w.r}/logs/chain.lock/winpid", str(os.getpid()))
    rc, out, err = w.chain()
    assert rc == 1 and "another instance of this chain is running" in out and w.calls() == [] and not os.path.exists(w.out)
    w8(f"{w.r}/logs/chain.lock/winpid", "99999999")
    rc, out, err = w.chain()
    assert rc == 0 and w.calls() == World.TAGS and not os.path.exists(f"{w.r}/logs/chain.lock")


def test_a_world_marker_is_required_and_a_bad_world_is_refused_before_anything_is_read(pw):
    w = pw
    os.remove(f"{w.r}/SOUP_TESTWORLD.marker")
    rc, out, err = w.chain()
    assert rc == 1 and out.startswith("ZZSOUP-STOP SOUP_WORLD=") and w.calls() == [] and not os.path.exists(f"{w.r}/logs")


def test_the_whole_chain_runs_with_every_remote_shell_command_trapped_and_none_is_ever_called(pw):
    w = pw
    rc, out, err = w.chain("--dry-run", trap=True)
    assert rc == 0 and "ZZSOUP-DRY-OK" in out, out + err
    rc, out, err = w.chain(trap=True)
    assert rc == 0 and "ZZSOUP-DONE 6/6" in w.log(), out + err + w.log()[-800:]
    assert not os.path.exists(f"{w.r}/SSH_CALLED") and not os.path.exists(f"{w.r}/CMDV_CALLED")
    open(f"{w.r}/probe.sh", "w").write("ssh x\ncommand -v ssh\n")
    e = {"BASH_ENV": f"{w.r}/trap_env.sh"}
    subprocess.run([BASH, f"{w.r}/probe.sh"], env={**os.environ, **e}, capture_output=True)
    assert os.path.exists(f"{w.r}/SSH_CALLED") and os.path.exists(f"{w.r}/CMDV_CALLED")                       # the traps themselves work


def test_a_prereg_that_changes_during_the_steps_stops_the_chain_before_the_manifest_and_the_reader(pw):
    w = pw
    w.hook(f"stub_tamper_{World.TAGS[5]}.txt", f"{w.r}/prereg.md")                                          # the last step appends a byte to the promoted PREREG
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP the PREREG changed \(end of the steps\)", log, re.M) and "ZZSOUP-DONE" not in log
    assert w.calls() == World.TAGS and not os.path.exists(f"{w.r}/reader_call.json") and not os.path.exists(f"{w.out}/runs_soup.json")


def test_a_prereg_that_changes_while_the_reader_runs_is_caught_and_no_done_marker_is_written(pw):
    w = pw
    w.hook("stub_reader_tamper.txt", f"{w.r}/prereg.md")                                                    # the stub reader appends a byte to the PREREG
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP the PREREG changed \(after the reader\)", log, re.M) and "ZZSOUP-DONE" not in log and os.path.exists(f"{w.r}/reader_call.json")


@pytest.mark.parametrize("path,what", [("reader/soup_read.py", "the soup reader changed"), ("builder.py", "the soup builder changed")])
def test_a_reader_or_builder_that_changes_during_the_steps_stops_the_chain_before_the_reader_runs(pw, path, what):
    w = pw
    w.hook(f"stub_tamper_{World.TAGS[5]}.txt", f"{w.r}/{path}")                                             # the last step appends a byte to the file
    rc, out, err = w.chain()
    log = w.log()
    assert rc == 1 and re.search(rf"^ZZSOUP-STOP {what} \(end of the steps\)", log, re.M) and "ZZSOUP-DONE" not in log
    assert w.calls() == World.TAGS and not os.path.exists(f"{w.r}/reader_call.json")


def test_a_pins_file_that_cannot_be_written_is_a_failure_not_a_success(pw):
    w = pw
    os.makedirs(f"{w.r}/pins_is_a_directory")
    rc, out, err = w.chain("--write-pins", env={"SOUP_PINS": f"{w.r}/pins_is_a_directory"})
    assert rc == 1 and "cannot write" in w.log() + out, out + err + w.log()                                  # the shell must STOP on the lib's failure
    rc, out, err = w.chain("--write-pins")
    assert rc == 0 and os.path.isfile(f"{w.r}/PINS.txt")


def test_the_disk_guard_looks_at_the_output_drive_not_at_c(tmp_path):
    drv = os.path.splitdrive(os.path.abspath(HERE))[0].upper()
    if not drv or drv == "C:":
        pytest.skip("needs the tests directory on a drive other than C:")
    out = f"{HERE}/_guard_drive_out_{os.getpid()}"
    try:
        w = World(tmp_path, out=out)
        w.pins()
        rc, out_, err = w.chain(env={"SOUP_F_MIN_GB": "100000", "SOUP_GUARD_WAIT_S": "2"})                  # the REAL free space of the output drive (no hook) is below 100000 GB
        log = w.log()
        assert rc == 1 and re.search(rf"^ZZSOUP-STOP a guard stayed LOW for 2 s before SOUP_s0_m10_causal_heads: LOW {re.escape(drv)}/ free [0-9.]+ GB < 100000", log, re.M), log[-500:]
        assert w.calls() == []
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_waker_reports_the_chains_state_from_anchored_markers_only(tmp_path):
    L_ = tmp_path / "logs"
    L_.mkdir()

    def waker(*a):
        p = subprocess.run([BASH, f"{CHAIN}/wake_soup.sh", *a], env={**{k: v for k, v in os.environ.items() if not k.startswith("SOUP_")}, "SOUP_LOG": str(L_), "SOUP_WAKE_POLL_S": "1", "SOUP_WAKE_MAX_POLLS": "2"},
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout
    assert waker("--status")[0] == 1                                                                         # nothing yet
    w8(str(L_ / "chain.out"), "10-28 12:00:00 STOP: ZZSOUP-STOP in prose\n10-28 12:00:01 ZZSOUP-DONE in prose\n")
    assert waker("--status")[0] == 1                                                                         # prose lines do not count
    w8(str(L_ / "chain.out"), "x\nZZSOUP-STEP-DONE 1/6 a (causal)\n")
    assert waker("--status")[0] == 1
    w8(str(L_ / "chain.out"), "x\nZZSOUP-DONE 6/6 steps, reader rc 0 (outcome written): f\n")
    rc, out = waker("--status")
    assert rc == 0 and "CHAIN DONE: ZZSOUP-DONE 6/6" in out
    w8(str(L_ / "chain.out"), "x\nZZSOUP-STOP because\n")
    rc, out = waker("--status")
    assert rc == 4 and "CHAIN STOP: ZZSOUP-STOP because" in out
    w8(str(L_ / "chain.out"), "x\n")
    w8(str(L_ / "chain.out.prev_1"), "ZZSOUP-DONE old\n")
    assert waker("--status")[0] == 1                                                                         # a rotated earlier log is never read
    w8(str(L_ / "chain.winpid"), "99999999")
    rc, out = waker("--status")
    assert rc == 6 and "DEAD without an end marker" in out
    os.remove(L_ / "chain.winpid")
    w8(str(L_ / "soup_chain.launch.out"), "ZZSOUP-STOP refused before the log existed\n")
    rc, out = waker("--status")
    assert rc == 4 and "launcher log" in out
    os.makedirs(L_ / "chain.lock")
    assert waker("--status")[0] == 1                                                                         # a running chain: its launcher log is history
    os.rmdir(L_ / "chain.lock")
    w8(str(L_ / "chain.out"), "ZZSOUP-DONE 1/1 steps, reader rc 0 (x): y\n")
    assert waker()[0] == 0                                                                                   # waiting returns at once on a marker
    w8(str(L_ / "chain.out"), "x\n")
    os.remove(L_ / "soup_chain.launch.out")
    t0 = time.time()
    rc, out = waker()
    assert rc == 5 and "WAKER TIMEOUT" in out and time.time() - t0 < 30


# ================================================================================================ production facts (read-only)
REAL_TREE, REAL_OUT = "F:/TanitAD-scratch/launch_intent_t1/tree_li", "F:/TanitAD-scratch/soup_t1_out"


def test_the_shipped_plan_parses_and_re_renders_from_the_templates():
    steps = L.parse_plan(f"{SOUP}/argv/PLAN.txt", REAL_TREE, REAL_OUT)
    assert len(steps) == 28 and [s["kind"] for s in steps].count("causal") == 6 and [s["kind"] for s in steps].count("logged") == 6 and [s["kind"] for s in steps].count("t0fresh") == 6
    assert [s["kind"] for s in steps].count("t0std") == 6 and [s["kind"] for s in steps].count("t0dev") == 4 and [s["kind"] for s in steps][24:] == ["t0dev"] * 4
    spec = json.load(open(f"{SOUP}/soups.json", encoding="utf-8"))
    import shlex
    want = []
    for name, s in spec["soups"].items():
        if s.get("expect"):
            continue
        ck = f"{SOUP}/ckpt/{s['run_dir']}/ckpt_last.pt".replace("\\", "/")
        for t, kw in (("rollout_causal.tmpl", {"TAG": f"{name}_m10_causal_heads", "OUT": REAL_OUT}), ("rollout_logged.tmpl", {"TAG": f"{name}_m10_logged_heads", "OUT": REAL_OUT}),
                      ("t0fresh.tmpl", {"TAG": f"M2b-{name.lower()}-fresh", "T0DIR": f"{REAL_OUT}/t0fresh"}), ("t0std.tmpl", {"TAG": f"M2b-{name.lower()}", "T0DIR": f"{REAL_OUT}/t0std"})):
            want.append(shlex.join(RP.render(t, CKPT=ck, **kw)))
    got = [ln for ln in open(f"{SOUP}/argv/PLAN.txt", encoding="utf-8").read().split("\n") if ln and not ln.startswith("#")]
    assert got[:24] == want
    assert [s["tag"] for s in steps[24:]] == ["M2b-dagoacc_0-dev", "M2b-dagoacc_1-dev", "M2b-a_0-dev", "M2b-a_1-dev"]


def test_the_shipped_tool_digest_constant_is_the_one_the_reader_checks():
    src = open(f"{SOUP}/read/soup_read.py", encoding="utf-8").read()
    assert f'TOOL_DIGEST = "{L.TOOL_DIGEST}"' in src


def test_the_real_tree_hashes_to_the_pinned_digest_and_the_real_checkpoints_are_ready():
    if not os.path.isdir(REAL_TREE) or not os.path.isfile(f"{SOUP}/ckpt/M2b_s0_soup/ckpt_last.pt"):
        pytest.skip("production inputs not present")
    assert L.tree_digest(REAL_TREE) == L.TOOL_DIGEST
    steps = L.parse_plan(f"{SOUP}/argv/PLAN.txt", REAL_TREE, REAL_OUT)
    assert L.ready(steps, json.load(open(f"{SOUP}/soups.json", encoding="utf-8"))) == []


def test_an_invalid_plan_stops_the_chain_with_the_plans_own_error(pw):
    w = pw
    w8(f"{w.r}/argv/PLAN.txt", open(f"{w.r}/argv/PLAN.txt", encoding="utf-8").read().replace("--dump-heads", "", 1))
    rc, out, err = w.chain()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP the plan .* is invalid: PlanError: .*--dump-heads", w.log(), re.M) and w.calls() == [] and not os.path.exists(w.out)


@pytest.mark.parametrize("f", ["builder.py", "li_reader.py", "reader/soup_read.py"])
def test_a_missing_reader_or_builder_file_is_named_before_any_pin_is_compared(pw, f):
    w = pw
    os.remove(f"{w.r}/{f}")
    rc, out, err = w.chain()
    assert rc == 1 and re.search(r"^ZZSOUP-STOP a reader file is missing", w.log(), re.M) and w.calls() == []


def test_a_done_record_that_stops_verifying_before_the_manifest_is_caught(pw):
    w = pw
    w.hook(f"stub_tamper_{World.TAGS[5]}.txt", f"{w.out}/done_{World.TAGS[0]}.txt")                         # the last step corrupts the first step's done record
    rc, out, err = w.chain()
    assert rc == 1 and w.calls() == World.TAGS and re.search(r"^ZZSOUP-STOP 5 of 6 steps have a verifying done record", w.log(), re.M) and not os.path.exists(f"{w.r}/reader_call.json")


def test_a_manifest_that_cannot_be_built_stops_before_the_reader(w):
    runs = json.load(open(f"{w.r}/li_runs.json"))
    del runs["arms"]["REP"]
    w8(f"{w.r}/li_runs.json", json.dumps(runs))
    w.pins()
    rc, out, err = w.chain()
    assert rc == 1 and w.calls() == World.TAGS and re.search(r"^ZZSOUP-STOP manifest: KeyError", w.log(), re.M) and not os.path.exists(f"{w.r}/reader_call.json") and "ZZSOUP-DONE" not in w.log()


def test_the_t0_tool_changing_during_the_chain_is_caught_although_it_is_not_in_the_tree_digest(pw):
    w = pw
    w.hook(f"stub_tamper_{World.TAGS[0]}.txt", f"{w.r}/tree/stack/scripts/v8f_mini_eval.py")
    rc, out, err = w.chain()
    assert rc == 1 and "the T0 tool v8f_mini_eval.py changed (after SOUP_s0_m10_causal_heads)" in w.log() and w.calls() == [World.TAGS[0]]


def test_pins_write_refuses_a_tree_that_is_not_the_pinned_one_unless_told_there_is_no_pin(w, tmp_path):
    base = ["--plan", f"{w.r}/argv/PLAN.txt", "--tree", f"{w.r}/tree", "--out", w.out, "--spec", f"{w.r}/soups.json", "--builder", f"{w.r}/builder.py", "--write"]
    target = tmp_path / "PINS_a.txt"
    assert L.main(["pins-write", *base, str(target)]) == 2 and not target.exists()                          # the stub tree is not tree_li: no PINS are written
    assert L.main(["pins-write", *base, str(target), "--expect-tree-digest", "none"]) == 0 and target.exists()
    assert L.main(["pins-write", *base, str(tmp_path / "PINS_b.txt"), "--expect-tree-digest", L.tree_digest(f"{w.r}/tree")]) == 0
    assert L.main(["pins-verify", *base[:-1], "--pins", str(target), "--expect-tree-digest", "none"]) == 0
    assert L.main(["pins-verify", *base[:-1], "--pins", str(target)]) == 1                                  # verification defaults to the pinned production digest too
