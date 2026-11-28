"""Local tests of the ENCODE kit of card ADD108 (synthetic data only; no pod, no GPU, no network, no file above 5 MB).

Run the whole file:
  cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest lookgap/enc/test_lookgap_enc.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/lookgap/enc/tmp_pt/<new name>
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ctl_m1 as C   # noqa: E402
import gen_window_encode_p as G   # noqa: E402

PILOT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "cropread", "enc")
PILOT_WE = os.path.join(PILOT_DIR, "window_encode.py")
QUEUE = os.path.join(HERE, "pod_lookgap_enc_queue.sh")
PUSH = os.path.join(HERE, "push_lookgap_enc.sh")
have_pilot = pytest.mark.skipif(not os.path.exists(PILOT_WE), reason="the pilot's window_encode.py is not on this box")


# ------------------------------------------------------------------------------------------------ generator
@have_pilot
def test_generator_makes_a_one_arm_tool_from_the_pilot_file():
    text = G.generate(open(PILOT_WE, "rb").read())
    compile(text, "window_encode_p.py", "exec")
    assert text.count("(a.arm,)") == 6
    assert 'ap.add_argument("--arm", default="P"' in text
    assert "tD, tB" not in text and '(("D", tD)' not in text and "return tD.to" not in text
    # the pilot's B path is kept: the box read, the grids, the sampler, the encoder call on the lead frames, the cast
    for keep in ("def load_box_dirs(", "window_grid(intr, h, w, frames[i], az, el)", "window_sample(vid, gb, mb)",
                 "enc.window_tokens(Bw).reshape(idx.size, 2, 65, -1)", "tB.to(torch.float16).numpy()", "make_windows, c, L, a.box_dir, a.torch_threads"):
        assert keep in text, keep


@have_pilot
def test_generator_refuses_a_source_with_a_different_sha256_and_a_wrong_edit_count():
    src = open(PILOT_WE, "rb").read()
    with pytest.raises(SystemExit):
        G.generate(src + b"\n# x\n")                                      # source hash
    old = list(G.EDITS)
    try:
        i = [k for k, e in enumerate(G.EDITS) if e[0] == '("D", "B")'][0]
        G.EDITS[i] = (G.EDITS[i][0], G.EDITS[i][1], 5)                    # a wrong count
        with pytest.raises(SystemExit):
            G.generate(src)
    finally:
        G.EDITS[:] = old
    assert G.generate(src)                                                # restored: fine


@have_pilot
def test_generator_accepts_crlf_source_and_refuses_an_edit_text_that_is_absent():
    src = open(PILOT_WE, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert G.generate(src)                                                # line ends are normalised before the hash
    changed = open(PILOT_WE, "rb").read().replace(b"    tB = tD.clone()\n", b"    tB = tD.copy()\n")
    assert changed != open(PILOT_WE, "rb").read()
    with pytest.raises(SystemExit):
        G.generate(changed, check_sha=False)                              # the text of an edit is not found


@have_pilot
def test_generator_never_overwrites_a_different_file(tmp_path):
    out = str(tmp_path / "window_encode_p.py")
    assert G.main(["--src", PILOT_WE, "--out", out]) == 0
    assert G.main(["--src", PILOT_WE, "--out", out]) == 0                 # same bytes: fine
    with open(out, "ab") as fh:
        fh.write(b"# changed\n")
    with pytest.raises(SystemExit):
        G.main(["--src", PILOT_WE, "--out", out])


# ------------------------------------------------------------------------------------------------ the generated variant's pure functions
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeEnc:
    """Per-window deterministic 'encoder' (the mean pixel value times a fixed ramp); records the size of every call."""

    def __init__(self):
        self.calls = []

    def window_tokens(self, x):
        import torch
        self.calls.append((int(x.shape[0]), int(x.long().sum())))
        s = x.float().mean(dim=(1, 2, 3))
        return s[:, None, None] * torch.linspace(0.0, 1.0, 1024)[None, None, :].expand(1, 65, 1024)


@pytest.fixture()
def mods(tmp_path):
    if not os.path.exists(PILOT_WE):
        pytest.skip("no pilot file")
    pytest.importorskip("torch")
    out = str(tmp_path / "window_encode_p.py")
    G.main(["--src", PILOT_WE, "--out", out])
    sys.path.insert(0, PILOT_DIR)
    try:
        yield load(PILOT_WE, "pilot_we"), load(out, "variant_we")
    finally:
        sys.path.remove(PILOT_DIR)


def tiny_windows(n, seed=1):
    return np.random.RandomState(seed).randint(0, 256, size=(n, 4, 3, 256, 256), dtype=np.uint8)


def test_box_folder_in_arm_b_bytes_out_when_every_frame_has_a_lead(mods):
    pilot, var = mods
    wins, has = tiny_windows(3), np.ones(3, bool)
    _, tB_pilot = pilot.encode_clip_tokens(FakeEnc(), wins, has)
    w2 = wins.copy()
    w2[:, 0:2] = 0                                                        # the variant does not sample arm D when every frame has a direction
    e = FakeEnc()
    tB_var = var.encode_clip_tokens(e, w2, has)
    assert tB_var.dtype == np.float16 and tB_var.shape == (3, 2, 65, 1024)
    assert tB_var.tobytes() == tB_pilot.tobytes()
    assert len(e.calls) == 1 and e.calls[0][0] == 6                       # one encoder call on the 2n window images


def test_box_folder_in_arm_b_bytes_out_when_some_frames_have_no_lead(mods):
    pilot, var = mods
    wins, has = tiny_windows(4, 2), np.array([True, False, True, False])
    e1, e2 = FakeEnc(), FakeEnc()
    tD_pilot, tB_pilot = pilot.encode_clip_tokens(e1, wins, has)
    tB_var = var.encode_clip_tokens(e2, wins, has)
    assert tB_var.tobytes() == tB_pilot.tobytes()
    assert e1.calls == e2.calls                                           # the same encoder calls on the same batches, in the same order
    assert tB_var[1].tobytes() == tD_pilot[1].tobytes()                   # a frame with has False carries arm D's tokens: the pilot's rule, kept


def test_variant_encoder_call_equals_the_pilots_b_call_on_the_same_pixels(mods):
    pilot, var = mods
    wins, has = tiny_windows(3, 3), np.ones(3, bool)
    e1, e2 = FakeEnc(), FakeEnc()
    pilot.encode_clip_tokens(e1, wins, has)
    var.encode_clip_tokens(e2, wins, has)
    assert e1.calls[-1] == e2.calls[-1]                                   # the pilot's B call (last) and the variant's only call: same size, same pixels


def test_load_box_dirs_reads_a_direction_file_with_a_direction_on_every_frame(mods, tmp_path):
    _, var = mods
    n = 5
    az, el = np.linspace(-20, 20, n), np.linspace(-4, 4, n)
    np.savez(str(tmp_path / "clipA.npz"), has_lead=np.ones(n, bool), ctr_az=az, ctr_el=el)
    has, baz, bel = var.load_box_dirs(str(tmp_path), "clipA", n)
    assert has.all() and np.array_equal(baz, az) and np.array_equal(bel, el)
    with pytest.raises(ValueError):
        var.load_box_dirs(str(tmp_path), "clipA", n + 1)                  # frame count differs from the plan
    with pytest.raises(FileNotFoundError):
        var.load_box_dirs(str(tmp_path), "no_such_clip", n)
    # the pilot's has_lead False path, kept: a frame with has False or a non-finite direction is a no-lead frame (direction 0, 0)
    az2 = az.copy()
    az2[2] = np.nan
    hl = np.ones(n, bool)
    hl[0] = False
    np.savez(str(tmp_path / "clipB.npz"), has_lead=hl, ctr_az=az2, ctr_el=el)
    has, baz, bel = var.load_box_dirs(str(tmp_path), "clipB", n)
    assert list(has) == [False, True, False, True, True] and baz[0] == 0.0 and baz[2] == 0.0


def test_variant_refuses_the_pilots_arm_names(mods):
    _, var = mods
    for arm in ("D", "B"):
        with pytest.raises(AssertionError):
            var.main(["--list", "x", "--root", "x", "--box-dir", "x", "--out", "x", "--arm", arm])


# ------------------------------------------------------------------------------------------------ control m1
def test_m1_selftest_flips_every_clause(tmp_path):
    res = C.selftest(str(tmp_path / "w"))
    bad = [r for r in res if not r[2]]
    assert not bad, bad
    assert {r[1] for r in res} >= {"i", "ii", "iii"}


def build_world(w, ids, n):
    man = {"D": {}, "B": {}, "P": {}}
    for i, c in enumerate(ids):
        man["D"][c] = C._put(os.path.join(w, "pilot", "tok", "D"), c, C._tok(10 + i, n))
        man["B"][c] = C._put(os.path.join(w, "pilot", "tok", "B"), c, C._tok(20 + i, n))
        man["P"][c] = C._put(os.path.join(w, "tok", "P"), c, C._tok(30 + i, n))
        C._put(os.path.join(w, "check_tok", "tok", "B_ident"), c, C._tok(20 + i, n))
    for arm, p in (("D", "pilot/tok/D"), ("B", "pilot/tok/B"), ("P", "tok/P")):
        json.dump({"clips": man[arm]}, open(os.path.join(w, p, "MANIFEST.json"), "w"))
    os.makedirs(os.path.join(w, "pilot", "lists"), exist_ok=True)
    open(os.path.join(w, "pilot", "lists", "train_final_ids.txt"), "w", newline="\n").write("t0\nt1\n")
    open(os.path.join(w, "pilot", "lists", "eval141.txt"), "w", newline="\n").write("e0\ne1\ne2\n")


def test_m1_cli_modes_write_the_record_with_pass_m1(tmp_path):
    w = str(tmp_path / "w")
    build_world(w, ["t0", "t1", "e0", "e1", "e2"], 1)
    base = ["--root", w, "--pilot-run", os.path.join(w, "pilot"), "--eval", os.path.join(w, "pilot", "lists", "eval141.txt"),
            "--train-ids", os.path.join(w, "pilot", "lists", "train_final_ids.txt"), "--n-expected", "5", "--workers", "2"]
    assert C.main(["identity"] + base) == 0
    rec = json.load(open(os.path.join(w, "ENC_M1_IDENTITY.json")))
    assert rec["pass_m1"] is None and rec["clauses"]["i_identity_three_clips"]["pass"] is True
    assert C.main(["all"] + base) == 0
    rec = json.load(open(os.path.join(w, "ENC_M1.json")))
    assert rec["pass_m1"] is True and set(rec["clauses"]) == {"i_identity_three_clips", "ii_arm_p_files", "iii_pilot_arms_unchanged"}
    C._put(os.path.join(w, "tok", "P"), "t0", C._tok(99, 1))                # an arm P file that differs from the one in the manifest
    assert C.main(["all"] + base) == 1
    rec = json.load(open(os.path.join(w, "ENC_M1.json")))
    assert rec["pass_m1"] is False and rec["clauses"]["ii_arm_p_files"]["pass"] is False and rec["clauses"]["i_identity_three_clips"]["pass"] is True


# ------------------------------------------------------------------------------------------------ queue and push script
def bash():
    b = shutil.which("bash")
    if not b:
        pytest.skip("no bash")
    return b


def run_bash(script_text, cwd):
    return subprocess.run([bash()], input=script_text, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(cwd), timeout=120)


def test_shell_files_parse_and_have_lf_line_ends():
    for p in (QUEUE, PUSH):
        raw = open(p, "rb").read()
        assert b"\r" not in raw
        r = subprocess.run([bash(), "-n"], input=raw.decode("utf-8"), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        assert r.returncode == 0, r.stderr


def test_queue_dry_run_prints_the_steps_in_order_and_makes_nothing(tmp_path):
    text = open(QUEUE, encoding="utf-8").read()
    before = sorted(os.listdir(str(tmp_path)))
    r = run_bash("export DRY=1\n" + text, tmp_path)
    assert r.returncode == 0, r.stderr[-500:]
    assert sorted(os.listdir(str(tmp_path))) == before
    out = r.stdout
    order = ["DRY gate", "gen_window_encode_p.py", "--arm B_ident", "ctl_m1.py identity", "--arm P", "ctl_m1.py all", "would write DONE_ENC.txt"]
    pos = -1
    for key in order:
        i = out.find(key, pos + 1)
        assert i > pos, (key, out[:1500])
        pos = i
    assert out.count("DRY wait_slot") == 2 and "DRY limits" in out
    assert "--workers 5" in out and "--box-dir /workspace/v8proof/lookgap/dirs" in out and "--out /workspace/v8proof/lookgap " in out


def test_queue_refuses_without_the_go_file_and_without_done_dirs_and_makes_no_marker(tmp_path):
    text = open(QUEUE, encoding="utf-8").read().replace("ROOT=/workspace/v8proof/lookgap\n", "ROOT=./lg_root\n", 1)
    assert "ROOT=./lg_root" in text
    os.makedirs(str(tmp_path / "lg_root"))
    r = run_bash(text, tmp_path)
    assert r.returncode == 4 and "GO_STAGE_B.txt is missing" in r.stdout, (r.returncode, r.stdout, r.stderr[-300:])
    (tmp_path / "lg_root" / "GO_STAGE_B.txt").write_text("go\n")
    r = run_bash(text, tmp_path)
    assert r.returncode == 4 and "DONE_DIRS.txt is missing" in r.stdout, (r.returncode, r.stdout)
    assert sorted(os.listdir(str(tmp_path / "lg_root"))) == ["GO_STAGE_B.txt"]       # no START marker, no lock, no log: nothing was started


def test_queue_text_never_deletes_moves_or_prints_secrets_and_has_the_stop_rules():
    text = open(QUEUE, encoding="utf-8").read()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for pat in (r"\brm\b", r"\bmv\b", r"\bpkill\b", r"\bkillall\b", r"\bunlink\b", r"set -x", r"printenv", r"hf_[A-Za-z0-9]{6}", r"\bssh\b"):
        assert not re.search(pat, code), pat
    for need in ("flock -n 9", "START_ENC.txt", "STOP_ENC.txt", "DONE_ENC.txt", "GO_STAGE_B.txt", "DONE_DIRS.txt", "RESUME=${RESUME:-0}", "DRY=${DRY:-0}",
                 "GPU_LIMIT_S=${GPU_LIMIT_S:-7200}", "TOK_LIMIT_GB=${TOK_LIMIT_GB:-20}", "du -sb", "memory.max", "memory.stat", "-lt 3 ]", "-gt 28 ]",
                 "WORKERS=${WORKERS:-5}", "ROOT=/workspace/v8proof/lookgap", "kit_enc", "--arm P", "--arm B_ident"):
        assert need in text, need
    for env in ("export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/workspace/v8proof/hf PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
                "export TANITAD_STACK=/workspace/v8proof/tree/stack", "PY=/workspace/venv/bin/python",
                'CPU="env CUDA_VISIBLE_DEVICES=-1"', 'GPU="env -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"'):
        assert env in text, env


def test_identity_runs_before_the_641_encode_in_the_queue_text():
    text = open(QUEUE, encoding="utf-8").read()
    assert text.index("gen_window_encode_p.py --src") < text.index("--arm B_ident --out") < text.index("ctl_m1.py identity") < text.index("--arm P --out")
    assert text.index("--arm P --out") < text.index("ctl_m1.py all --root")


def test_push_script_refuses_without_the_lead_pytest_output():
    text = open(PUSH, encoding="utf-8").read()
    assert "lead_pytest.out is missing" in text and "set -C; cat >" in text and "GP=/workspace/v8proof/lookgap" in text and "kit_enc" in text
    assert text.index("REFUSED") < text.index("for f in ")


# ------------------------------------------------------------------------------------------------ wording and sizes
def test_wording_rules_in_every_file_of_the_kit():
    banned = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]
    files = [f for f in os.listdir(HERE) if f.endswith((".py", ".sh", ".md"))]
    assert files
    for f in files:
        t = open(os.path.join(HERE, f), encoding="utf-8").read().lower()
        for w in banned:
            assert w not in t, (f, w)
        assert not any(n in f.lower() for n in ("aux", "con.", "prn", "nul"))
        assert len("lookgap/enc/" + f) <= 200


def test_no_file_of_the_kit_is_large():
    for dp, dn, fn in os.walk(HERE):
        if "tmp_pt" in dp:
            continue
        for f in fn:
            assert os.path.getsize(os.path.join(dp, f)) < 5 * 1024 * 1024, f
