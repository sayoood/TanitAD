"""CPU tests of run_dumps_s0l.py (seconds).  Run: python -m pytest -q test_run_dumps_s0l.py  (or python test_run_dumps_s0l.py)."""
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_dumps_s0l as D  # noqa: E402

PINNED = os.path.join(os.path.dirname(HERE), "..", "stopbal", "read", "run_dumps_stopbal.py")


def baseline_argv(tag_full, ckpt, out):
    """The argv of run_dumps_stopbal.py:run() for a BAL arm, written out literally (independent of run_dumps_s0l.build_argv)."""
    art = "E:/Projects/TanitAD-artifacts"
    raw = art + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
    return ["C:/Users/Admin/venvs/tanitad/Scripts/python.exe", D.TREE + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", ckpt, "--tag", tag_full,
            "--tracks", art + "/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", art + "/sfresh_inputs/cache",
            "--eval-grid", raw + "/eval_grid_windows.json", "--eval-join", raw + "/join_eval141.npz",
            "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic",
            "--limit", "causal", "--dump-ticks", out + "/ticks", "--dump-heads", "--device", "cpu"]


def diff_positions(a, b):
    assert len(a) == len(b)
    return [i for i in range(len(a)) if a[i] != b[i]]


def test_pinned_source_still_carries_the_baseline_literals():
    src = open(PINNED, encoding="utf-8").read()
    for lit in ('"--tracks": ART + "/sfresh_inputs/inputs/native_tracks.pt"', '"--eval-cache": ART + "/sfresh_inputs/cache"',
                '"/eval_grid_windows.json"', '"/join_eval141.npz"', '"--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic"',
                '"--limit", "causal", "--dump-ticks", OUT + "/ticks", "--dump-heads", "--device", "cpu"',
                'argv = [PY, TREE + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", ck, "--tag", full]'):
        assert lit in src, lit


def test_bprime_argv_differs_from_baseline_in_exactly_the_four_flags():
    out = "F:/x/t1"
    got = D.build_argv("BPRIME_s0", D.BPRIME_CKPT, D.BPRIME_CACHE, out=out)
    base = baseline_argv("BAL-NULL_s0_m10_causal_heads", "C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/ckpt_last.pt", "F:/TanitAD-scratch/stopbal_read/t1")
    idx = diff_positions(got, base)
    flags = {got[i - 1] for i in idx}
    # the four replaced flags, plus --dump-ticks which the pinned driver DERIVES from --out-dir (OUT + "/ticks"); nothing else differs
    assert flags == set(D.REPLACED) | {"--dump-ticks"}, flags
    assert len(idx) == 5, idx
    assert got[got.index("--dump-ticks") + 1] == out + "/ticks"
    g = dict(zip(got[3::2], got[4::2]))
    assert g["--eval-cache"] == "E:/Projects/TanitAD-artifacts/lorapod_eval" and g["--tag"] == "BPRIME_s0_m10_causal_heads"
    assert g["--ckpt"].endswith("bprime/M2b_s0/ckpt_last.pt") and g["--out-dir"] == out
    assert D.build_argv("BPRIME_s0", D.BPRIME_CKPT, D.BPRIME_CACHE, out=out) == got   # pure / deterministic


def test_unknown_tag_refused():
    for t in ("BAL-NULL_s0", "L_s1", "BPRIME_s1", ""):
        why, _ = D.refusal(t)
        assert why and why.startswith("REFUSED"), (t, why)
    try:
        D.arm_paths("BAL-G_s0")
        assert False
    except ValueError:
        pass


def test_L_refused_without_its_cache_or_sha(monkeypatch=None):
    keep = {k: os.environ.pop(k, None) for k in ("S0L_L_CKPT", "S0L_L_CACHE", "S0L_L_SHA256")}
    try:
        why, _ = D.refusal("L_s0")
        assert why and "L_s0 needs env" in why, why
        os.environ.update(S0L_L_CKPT=D.BPRIME_CKPT, S0L_L_SHA256=D.EXPECTED_SHA256["BPRIME_s0"], S0L_L_CACHE="E:/nonexistent/cache_xyz")
        why, _ = D.refusal("L_s0")
        assert why and "eval cache missing" in why, why
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_wrong_sha_refused():
    old = dict(D.EXPECTED_SHA256)
    try:
        D.EXPECTED_SHA256["BPRIME_s0"] = "0" * 64
        why, _ = D.refusal("BPRIME_s0")
        assert why and "differs" in why, why
    finally:
        D.EXPECTED_SHA256.clear()
        D.EXPECTED_SHA256.update(old)
    why, want = D.refusal("BPRIME_s0")                            # control: the true sha passes
    assert why is None and want == old["BPRIME_s0"], why


def test_wrong_inherited_prefix_refused():
    old = D.INHERITED_PREFIX["BPRIME_s0"]
    try:
        D.INHERITED_PREFIX["BPRIME_s0"] = "deadbeef"
        why, _ = D.refusal("BPRIME_s0")
        assert why and "inherited prefix" in why, why
    finally:
        D.INHERITED_PREFIX["BPRIME_s0"] = old


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", n)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
