"""Local tests of the ENCODE kit of card ADD106 (synthetic data only; no pod, no GPU, no network).

Run the whole file:
  cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest cropscale/enc/test_cropscale_enc.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/cropscale/enc/tmp_pt/<new name>
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ctl_g0g1 as C   # noqa: E402
import gen_window_encode_d as G   # noqa: E402
import make_large_list as M   # noqa: E402

PILOT_WE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "cropread", "enc", "window_encode.py")
QUEUE = os.path.join(HERE, "pod_cropscale_enc_queue.sh")
PUSH = os.path.join(HERE, "push_cropscale_enc.sh")


def write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ------------------------------------------------------------------------------------------------ lists
def lists_world(tmp):
    elig = ["c%d" % i for i in range(10)]
    write(str(tmp / "elig.txt"), "".join(c + "\n" for c in elig))
    write(str(tmp / "eval.txt"), "e0\ne1\n")
    prow = [("c0", "fit"), ("c2", "fit"), ("c3", "fit"), ("c5", "sel")]
    write(str(tmp / "pilot.txt"), "".join("%s %s\n" % x for x in prow))
    return elig, prow


def args(tmp, mode, *more):
    return [mode, "--no-strict", "--eligible", str(tmp / "elig.txt"), "--eval", str(tmp / "eval.txt"), "--pilot-final", str(tmp / "pilot.txt"),
            "--out", str(tmp / "lists"), "--stage-ledger", str(tmp / "ledger.jsonl"), "--tok-dir", str(tmp / "enc_out" / "tok" / "D")] + list(more)


def test_new_order_is_eligible_minus_pilot_in_eligible_order(tmp_path):
    lists_world(tmp_path)
    assert M.main(args(tmp_path, "new-order")) == 0
    got = M.read_ids(str(tmp_path / "lists" / "train_new_order.txt"))
    assert got == ["c1", "c4", "c6", "c7", "c8", "c9"]
    assert M.main(args(tmp_path, "new-order")) == 0          # same bytes: fine


def test_new_order_refuses_an_evaluation_clip_among_the_eligible(tmp_path):
    lists_world(tmp_path)
    write(str(tmp_path / "elig.txt"), "c0\nc1\nc2\nc3\ne0\nc5\n")
    with pytest.raises(SystemExit):
        M.main(args(tmp_path, "new-order"))


def test_staged_list_is_content_addressed(tmp_path, capsys):
    lists_world(tmp_path)
    M.main(args(tmp_path, "new-order"))
    write(str(tmp_path / "ledger.jsonl"), json.dumps({"clip": "c1", "ok": True}) + "\n" + json.dumps({"clip": "c4", "ok": False, "reason": "x"}) + "\n")
    M.main(args(tmp_path, "staged", "--write"))
    out = capsys.readouterr().out
    path = [ln[5:] for ln in out.splitlines() if ln.startswith("LIST=")][-1]
    assert M.read_ids(path) == ["c1"]
    write(str(tmp_path / "ledger.jsonl"), json.dumps({"clip": "c1", "ok": True}) + "\n" + json.dumps({"clip": "c4", "ok": True}) + "\n")
    M.main(args(tmp_path, "staged", "--write"))
    out2 = capsys.readouterr().out
    path2 = [ln[5:] for ln in out2.splitlines() if ln.startswith("LIST=")][-1]
    assert path2 != path and M.read_ids(path2) == ["c1", "c4"]


def test_final_list_keeps_pilot_lines_and_lists_every_clip_left_out_with_a_reason(tmp_path):
    elig, prow = lists_world(tmp_path)
    M.main(args(tmp_path, "new-order"))
    led = [{"clip": "c1", "ok": True}, {"clip": "c4", "ok": True}, {"clip": "c6", "ok": False, "reason": "no obstacle source for this clip"},
           {"clip": "c7", "ok": True}, {"clip": "c8", "ok": True}]       # c9 has no ledger line
    write(str(tmp_path / "ledger.jsonl"), "".join(json.dumps(r) + "\n" for r in led))
    tok = tmp_path / "enc_out" / "tok" / "D"
    man = {}
    for c in ("c1", "c4", "c8"):                                         # c7 staged but not encoded
        write(str(tok / (c + ".npy")), "x" * 10)
        man[c] = {"n_frames": 1, "sha256": "0" * 64, "bytes": 10}
    write(str(tok / "MANIFEST.json"), json.dumps({"clips": man}))
    write(str(tmp_path / "enc_out" / "tok" / "ENCODE_FAILS.json"), json.dumps([["c7", "RuntimeError: decoded 3 of 5 wanted frames"]]))
    assert M.main(args(tmp_path, "final")) == 0
    rows = M.read_rows(str(tmp_path / "lists" / "train_final.txt"))
    assert rows[:4] == prow                                              # the pilot's lines, same order and roles
    assert rows[4:] == [("c1", "fit"), ("c4", "fit"), ("c8", "fit")]
    rec = json.load(open(str(tmp_path / "lists" / "LISTS_LARGE_RECORD.json")))
    left = {x["clip"]: x["reason"] for x in rec["left_out"]}
    assert set(left) == {"c6", "c7", "c9"}
    assert "did not stage" in left["c6"] and "obstacle" in left["c6"]
    assert "did not encode" in left["c7"] and "decoded 3 of 5" in left["c7"]
    assert "did not stage" in left["c9"]
    assert rec["n_large_list"] == 7 and rec["n_left_out"] == 3 and rec["left_out_counts"] == {"did not stage": 2, "did not encode": 1}
    # a second call with the same inputs is accepted, a changed list is refused
    assert M.main(args(tmp_path, "final")) == 0
    write(str(tok / "c4.npy"), "x" * 3)
    with pytest.raises(SystemExit):
        M.main(args(tmp_path, "final"))


def test_g0_on_the_made_list_passes_with_a_synthetic_selection_size(tmp_path):
    elig, prow = lists_world(tmp_path)
    M.main(args(tmp_path, "new-order"))
    write(str(tmp_path / "ledger.jsonl"), "".join(json.dumps({"clip": c, "ok": True}) + "\n" for c in ("c1", "c4", "c6", "c7", "c8", "c9")))
    tok = tmp_path / "enc_out" / "tok" / "D"
    man = {}
    for c in ("c1", "c4", "c6", "c7", "c8"):
        write(str(tok / (c + ".npy")), "x" * 10)
        man[c] = {"n_frames": 1, "sha256": "0" * 64, "bytes": 10}
    write(str(tok / "MANIFEST.json"), json.dumps({"clips": man}))
    M.main(args(tmp_path, "final"))
    r = C.g0(str(tmp_path / "elig.txt"), str(tmp_path / "eval.txt"), str(tmp_path / "pilot.txt"), str(tmp_path / "lists" / "train_final.txt"),
             str(tmp_path / "lists" / "LISTS_LARGE_RECORD.json"), n_sel=1)
    assert r["pass"], r
    assert r["n_fit"] == 8 and r["n_sel"] == 1


# ------------------------------------------------------------------------------------------------ controls
def test_controls_selftest_flips_every_clause(tmp_path):
    res = C.selftest(str(tmp_path / "w"))
    bad = [r for r in res if not r[2]]
    assert not bad, bad
    clauses = {r[1] for r in res}
    for must in ("g0_all_in_eligible", "g0_none_is_evaluation", "g0_pilot_500_unchanged", "g0_selection_is_pilot_50", "g0_left_out_listed",
                 "g0_no_duplicates", "g0_record_matches_file", "g1_new_files_shape_type", "g1_new_hash_in_manifest", "g1_pilot_unchanged",
                 "g1_identity_three_clips"):
        assert must in clauses, must


def test_identity_tolerance_is_one_in_a_thousand(tmp_path):
    a = np.random.RandomState(1).randn(1, 2, 65, 1024).astype(np.float16)
    os.makedirs(str(tmp_path / "p")), os.makedirs(str(tmp_path / "c"))
    ids = ["x", "y", "z"]
    for c in ids:
        np.save(str(tmp_path / "p" / (c + ".npy")), a)
        np.save(str(tmp_path / "c" / (c + ".npy")), a)
    assert C.g1_identity(ids, str(tmp_path / "p"), str(tmp_path / "c"))["pass"]
    b = a.copy()
    b[0, 0, 0, 0] = np.float16(float(b[0, 0, 0, 0]) + 0.002)
    np.save(str(tmp_path / "c" / "y.npy"), b)
    assert not C.g1_identity(ids, str(tmp_path / "p"), str(tmp_path / "c"))["pass"]
    assert not C.g1_identity(ids[:2], str(tmp_path / "p"), str(tmp_path / "c"))["pass"]      # exactly three clips are required


# ------------------------------------------------------------------------------------------------ generator
@pytest.mark.skipif(not os.path.exists(PILOT_WE), reason="the pilot's window_encode.py is not on this box")
def test_generator_makes_an_arm_d_only_tool_from_the_pilot_file():
    text = G.generate(open(PILOT_WE, "rb").read())
    compile(text, "window_encode_d.py", "exec")
    assert "--box-dir" not in text and "tB" not in text and '"B"' not in text
    assert "tok/<arm>" in text or "tok" in text
    assert text.count('("D",)') == 6
    assert "def encode_clip_tokens(enc, wins, has):" in text and "enc.window_tokens(D)" in text
    # the D path is the pilot's: same grids call, same sampler, same cast
    for keep in ("window_grid(intr, h, w, frames[i], ARM_D_WINDOWS[i][1], ARM_D_WINDOWS[i][2])", "window_sample(vid, g.expand(B, -1, -1, -1), m.expand(B, -1, -1))",
                 "tD.to(torch.float16).numpy()", "enc.window_tokens(D).reshape(n, 2, 65, -1)"):
        assert keep in text, keep


@pytest.mark.skipif(not os.path.exists(PILOT_WE), reason="the pilot's window_encode.py is not on this box")
def test_generator_refuses_a_changed_source_and_a_wrong_edit_count():
    src = open(PILOT_WE, "rb").read()
    with pytest.raises(SystemExit):
        G.generate(src + b"\n# x\n")                                      # source hash
    old = list(G.EDITS)
    try:
        G.EDITS[5] = (G.EDITS[5][0], G.EDITS[5][1], 2)                    # a wrong count for the box-dir argument
        with pytest.raises(SystemExit):
            G.generate(src)
    finally:
        G.EDITS[:] = old


@pytest.mark.skipif(not os.path.exists(PILOT_WE), reason="the pilot's window_encode.py is not on this box")
def test_generator_never_overwrites_a_different_file(tmp_path):
    out = str(tmp_path / "window_encode_d.py")
    assert G.main(["--src", PILOT_WE, "--out", out]) == 0
    assert G.main(["--src", PILOT_WE, "--out", out]) == 0                 # same bytes: fine
    with open(out, "ab") as fh:
        fh.write(b"# changed\n")
    with pytest.raises(SystemExit):
        G.main(["--src", PILOT_WE, "--out", out])


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
    assert sorted(os.listdir(str(tmp_path))) == before                    # nothing made in the working folder
    out = r.stdout
    order = ["make_large_list.py new-order", "gen_window_encode_d.py", "window_encode_d.py --list", "ctl_g0g1.py identity", "stage_clips.py",
             "make_large_list.py staged", "--workers 5", "make_large_list.py final", "ctl_g0g1.py run"]
    pos = -1
    for key in order:
        i = out.find(key, pos + 1)
        assert i > pos, (key, out[:1500])
        pos = i
    assert "DRY memcheck: keep 28 GB" in out
    assert out.count("stage_clips.py") >= 8                               # chunks of 150 over 1,179 clips
    assert "hf_" not in out.replace("HF_HUB", "").replace("hf_hub", "").replace("/workspace/v8proof/hf", "")


def test_queue_text_never_deletes_moves_or_prints_secrets():
    text = open(QUEUE, encoding="utf-8").read()
    import re
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for pat in (r"\brm\b", r"\bmv\b", r"\bpkill\b", r"\bkillall\b", r"\bunlink\b", r"set -x", r"printenv", r"\benv\s*\|", r"cat\s+/root/\.hf_token", r"echo\s+\$HF", r"hf_[A-Za-z0-9]{6}"):
        assert not re.search(pat, code), pat
    for need in ("flock -n 9", "START_ENC.txt", "STOP_ENC.txt", "DONE_ENC.txt", "ENC_TIMES.txt", "enc_queue.log", "LIMIT_GB:-60", "MIN_FREE_GB:-100", "memory.max", "memory.stat",
                 "WORKERS=${WORKERS:-5}", "STAGE_WORKERS=${STAGE_WORKERS:-4}", "du -sb", "DRY"):
        assert need in text, need
    # the pilot queue's environment lines, word for word
    for env in ("export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/workspace/v8proof/hf PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
                "export TANITAD_STACK=/workspace/v8proof/tree/stack",
                "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_TORCH_THREADS=2 PAI_DECODE_THREADS=1",
                "export DINO_SAFETENSORS=$(cat /workspace/v8proof/hierrep/dino_safetensors.path 2>/dev/null)",
                'CPU="env CUDA_VISIBLE_DEVICES=-1"', 'GPU="env -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"'):
        assert env in text, env


def test_identity_runs_before_any_stage_call_in_the_queue_text():
    text = open(QUEUE, encoding="utf-8").read()
    assert text.index("ctl_g0g1.py identity") < text.index("stage_clips.py --list $LISTS/train_new_order.txt")
    assert text.index("gen_window_encode_d.py") < text.index("ctl_g0g1.py identity")


def test_push_script_refuses_without_the_lead_pytest_output():
    text = open(PUSH, encoding="utf-8").read()
    assert "lead_pytest.out is missing" in text and "set -C; cat >" in text and "GP=/workspace/v8proof/cropscale" in text and "kit_enc" in text
    assert text.index("REFUSED") < text.index("for f in ")
    # (the lead removed one line here: it asserted that lead_pytest.out does not exist, which fails by construction in the lead's run, whose output is that file)


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
        rel = "cropscale/enc/" + f
        assert len(rel) <= 200


def test_no_file_of_the_kit_is_large():
    for dp, dn, fn in os.walk(HERE):
        if "tmp_pt" in dp:
            continue
        for f in fn:
            assert os.path.getsize(os.path.join(dp, f)) < 5 * 1024 * 1024, f
