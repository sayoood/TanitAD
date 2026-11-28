"""Control-flow tests for the FINAL-checkpoint chain: pull_final_r8.sh (fake Thor), xcheck_model_ema_r8.py (tiny torch tensors), chain_final_r8.sh (STUB heavy steps, no GPU / Thor / real lock),
yield48852_r8.sh (stub lock), final_index_r8.py.  NEW file (2026-10-19, EvalFlyWheel).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider test_chain_final_r8.py      (needs Git-for-Windows `sh`; torch for the xcheck / real-extract arms)

Faking, as test_snap40k_r8.py / test_chain_40k_r8.py do:  the fake "Thor" is a directory (ckpt.pt, model_ema.pt, summary.json, train.log); the fake ssh runs the remote command LOCALLY with `sh -c`, so the real probe /
stat / md5sum pipeline is exercised; the fake scp is a `cp` that can corrupt the copy, flip a byte, or touch the source mid-copy.  Every heavy chain program is a STUB that records its argv into calls.log and writes the
artifacts the chain asserts on (the stub battery refuses to run unless the stub lock is held; the stub compliant read WITHHOLDS without the clamp_posthoc tree -- measured on the real tool; the stub forward refuses without the compliant tree).
THE REAL PROGRAMS inside the stubbed chain: extract_ema_r8.py (its own --expect-step guard, 48,500 vs 48,852) and xcheck_model_ema_r8.py (a perturbed tensor).  EVERY guard has a MUTATION arm that must go RED: a copy of the script
with the guard deleted behaves differently, and the assertion that catches it is shown to discriminate (the paired control reaches the next stage).  Assertions are on ARTIFACTS (marker logs, calls.log, files), never an exit code.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SH = shutil.which("sh") or "C:/Program Files/Git/bin/sh.exe"
PYV = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
pytestmark = pytest.mark.skipif(not Path(SH).exists() and not shutil.which("sh"), reason="no sh")
CHAIN = HERE / "chain_final_r8.sh"
PULL = HERE / "pull_final_r8.sh"
XCHECK = HERE / "xcheck_model_ema_r8.py"
YIELD = HERE / "yield48852_r8.sh"
STEP = 48852
TE, TR = f"step{STEP}ema_unc", f"step{STEP}_unc"


def msys(p) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def win(p) -> str:
    return str(p).replace("\\", "/")


def have_torch() -> bool:
    return subprocess.run([PYV, "-c", "import torch"], capture_output=True).returncode == 0


# ====================================================================================================================== pull_final_r8.sh against a fake Thor
PAYLOAD = b"final-ckpt-payload-" * 1000
EMA_PAYLOAD = b"model-ema-payload-" * 500
FAKE_SSH = '#!/bin/sh\nif [ "$FAKE_SSH_DEAD" = "1" ]; then exit 255; fi\nexec sh -c "$*"\n'
FAKE_SCP = r'''#!/bin/sh
# usage: fake_scp.sh <host:/path> <dest>
src="${1#*:}"; dest="$2"
echo "scp $src" >> "$FAKE_SCP_COUNT"
cp "$src" "$dest"
[ "$FAKE_SCP_CORRUPT" = "1" ] && printf 'X' >> "$dest"
[ "$FAKE_SCP_FLIP" = "1" ] && printf 'Z' | dd of="$dest" bs=1 seek=7 count=1 conv=notrunc 2>/dev/null
[ "$FAKE_SCP_TOUCH_REMOTE" = "1" ] && touch -d "@$(( $(stat -c %Y "$src") + 7 ))" "$src"
exit 0
'''
TRAINER_SUMMARY = {"done": True, "step": STEP, "arm": "hier", "seed": 0, "wallclock_s": 1.0, "ema_decay": 0.9998, "ema_num_updates": STEP, "b1_guard": {"armed": True, "decided": True, "guard_step": 1000}}
SUP_SUMMARY = {"done": True, "arm": "refcv8-r101-s0", "final_step": STEP, "steps_requested": STEP, "out_dir": "/x", "gate_token": "t", "written_utc": "2026-10-20T19:00:00Z", "note": "supervisor"}


@pytest.fixture()
def fake(tmp_path):
    root = tmp_path / "fk"
    thor = root / "thor"
    thor.mkdir(parents=True)
    (thor / "ckpt.pt").write_bytes(PAYLOAD)
    (thor / "model_ema.pt").write_bytes(EMA_PAYLOAD)
    (thor / "train.log").write_text(f"[v3:hier] ckpt step 48500 -> ckpt.pt\n", encoding="utf-8")
    (root / "fake_ssh.sh").write_text(FAKE_SSH, encoding="utf-8", newline="\n")
    (root / "fake_scp.sh").write_text(FAKE_SCP, encoding="utf-8", newline="\n")
    (root / "scp_count.txt").write_text("", encoding="utf-8")
    (root / "kit/ckpt").mkdir(parents=True)
    return root, thor, root / "kit"


def finish_thor(thor: Path, summary=None, ckpt_line=True):
    if ckpt_line:
        with open(thor / "train.log", "a", encoding="utf-8") as f:
            f.write(f"[v3:hier] ckpt step {STEP} -> ckpt.pt\n")
    (thor / "summary.json").write_text(json.dumps(TRAINER_SUMMARY if summary is None else summary), encoding="utf-8")


def pull_env(fake, **extra):
    root, thor, kit = fake
    env = dict(os.environ, FIN_SSH=f"sh {msys(root / 'fake_ssh.sh')}", FIN_SCP=f"sh {msys(root / 'fake_scp.sh')}", FIN_SCP_HOST="fakehost", FIN_RDIR=msys(thor), FIN_KIT=msys(kit),
               FIN_CKPT_DIR=msys(kit / "ckpt"), FIN_LOG=msys(kit / "chain/pull_final.log"), FIN_POLL_S="1", FIN_SETTLE_S="1", FIN_HOLE_RECHECK_S="1", FIN_MAX_WAIT_S="25", FIN_MAX_SSH_FAIL="3",
               FIN_MIN_FREE_KB="1", FIN_CKPT_MIN="1000", FIN_CKPT_MAX=str(len(PAYLOAD) * 2), FIN_EMA_MIN="1000", FIN_EMA_MAX=str(len(EMA_PAYLOAD) * 2), FAKE_SCP_COUNT=msys(root / "scp_count.txt"))
    env.update(extra)
    return env


def run_pull(fake, script: Path = PULL, wait=True, **extra):
    env = pull_env(fake, **extra)
    cmd = [SH, msys(script)]
    if wait:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def pmarkers(kit: Path):
    p = kit / "chain/pull_final.log"
    return [ln.split()[0] for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("ZZPFN")] if p.exists() else []


def plog(kit: Path):
    p = kit / "chain/pull_final.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_pull_waits_for_done_and_the_ckpt_line_then_pulls_and_proves(fake):
    root, thor, kit = fake
    pr = run_pull(fake, wait=False)
    time.sleep(2.5)
    assert pmarkers(kit)[:1] == ["ZZPFNSTARTZZ"] and "ZZPFNSEENZZ" not in pmarkers(kit), "pulled before the trainer was done"
    assert not (kit / "ckpt/ckpt_full_48852.pt").exists()
    finish_thor(thor)
    pr.wait(timeout=60)
    dest = kit / "ckpt/ckpt_full_48852.pt"
    md = hashlib.md5(PAYLOAD).hexdigest()
    assert dest.read_bytes() == PAYLOAD and (kit / "ckpt/ckpt_full_48852.pt.md5").read_text().split()[0] == md
    prov = json.loads((kit / "ckpt/ckpt_full_48852.pt.provenance.json").read_text())
    assert prov["ok"] is True and prov["step"] == STEP and prov["md5"] == md and prov["model_ema_thor_md5"] == hashlib.md5(EMA_PAYLOAD).hexdigest()
    assert (kit / "ckpt/model_ema_48852_thor.pt").read_bytes() == EMA_PAYLOAD and json.loads((kit / "ckpt/summary_thor_48852.json").read_text())["done"] is True
    assert "ZZPFNDONEZZ" in pmarkers(kit) and f"ckpt_md5 {md} " in plog(kit)
    assert not list((kit / "ckpt").glob("*.part")), "a .part file was left behind"


def test_pull_accepts_the_supervisor_summary_schema(fake):
    root, thor, kit = fake
    finish_thor(thor, summary=SUP_SUMMARY)
    run_pull(fake)
    assert "ZZPFNDONEZZ" in pmarkers(kit) and (kit / "ckpt/ckpt_full_48852.pt").exists()


def test_pull_HOLE_summary_done_but_the_final_save_never_happened(fake):
    """The 2026-10-19 06:18 class: metrics reached 48852, the kill came before the ckpt.pt save, the SUPERVISOR wrote a done-marker anyway.  summary.json says done@48852; ckpt.pt is still 48500."""
    root, thor, kit = fake
    finish_thor(thor, summary=SUP_SUMMARY, ckpt_line=False)
    run_pull(fake)
    assert "ZZPFNHOLEZZ" in pmarkers(kit) and "ZZPFNDONEZZ" not in pmarkers(kit)
    assert not (kit / "ckpt/ckpt_full_48852.pt").exists() and not (kit / "ckpt/model_ema_48852_thor.pt").exists(), "a stale ckpt.pt was banked as the final one"


def test_pull_HOLE_mutation_without_the_ckpt_line_guard_banks_the_stale_file(fake, tmp_path):
    """Regression arm: delete the HOLE guard -> the stale 48,500 ckpt.pt is banked and a DONE marker is written.  The assertion above therefore discriminates."""
    root, thor, kit = fake
    finish_thor(thor, summary=SUP_SUMMARY, ckpt_line=False)
    s = PULL.read_text(encoding="utf-8")
    old = 'elif [ "$CKL" -lt 1 ]; then'
    assert old in s
    mut = tmp_path / "pull_mut_hole.sh"
    mut.write_text(s.replace(old, 'elif false; then'), encoding="utf-8", newline="\n")
    run_pull(fake, script=mut)
    assert "ZZPFNDONEZZ" in pmarkers(kit) and (kit / "ckpt/ckpt_full_48852.pt").exists(), "the mutant should have banked the stale file"


def test_pull_wrong_step_in_summary_is_refused(fake):
    root, thor, kit = fake
    finish_thor(thor, summary=dict(TRAINER_SUMMARY, step=48500))
    run_pull(fake)
    assert "ZZPFNWRONGSTEPZZ" in pmarkers(kit) and not (kit / "ckpt/ckpt_full_48852.pt").exists()


@pytest.mark.parametrize("flip", ["FAKE_SCP_FLIP", "FAKE_SCP_CORRUPT"])
def test_pull_md5_or_size_mismatch_banks_nothing(fake, tmp_path, flip):
    root, thor, kit = fake
    finish_thor(thor)
    run_pull(fake, **{flip: "1"})
    mk = pmarkers(kit)
    assert ("ZZPFNMD5MISMATCHZZ" in mk or "ZZPFNSIZEMISMATCHZZ" in mk) and "ZZPFNDONEZZ" not in mk
    assert not (kit / "ckpt/ckpt_full_48852.pt").exists() and not (kit / "ckpt/ckpt_full_48852.pt.part").exists()


def test_pull_md5_mutation_accepts_a_corrupted_copy(fake, tmp_path):
    """Regression arm: delete the md5 comparison -> the byte-flipped copy is banked (same size, so the size check passes).  Proves the md5 check, not another check, is what stops it above."""
    root, thor, kit = fake
    finish_thor(thor)
    s = PULL.read_text(encoding="utf-8")
    old = 'if [ "$LMD" != "$RMD" ]; then rm -f "$PART"; mark MD5MISMATCH "$PULL_REC"; return 12; fi'
    assert old in s
    mut = tmp_path / "pull_mut_md5.sh"
    mut.write_text(s.replace(old, ""), encoding="utf-8", newline="\n")
    run_pull(fake, script=mut, FAKE_SCP_FLIP="1")
    assert (kit / "ckpt/ckpt_full_48852.pt").exists() and (kit / "ckpt/ckpt_full_48852.pt").read_bytes() != PAYLOAD, "the mutant should have banked the flipped bytes"


def test_pull_remote_file_changing_during_the_copy_is_refused(fake):
    root, thor, kit = fake
    finish_thor(thor)
    run_pull(fake, FAKE_SCP_TOUCH_REMOTE="1")
    assert "ZZPFNCHANGEDZZ" in pmarkers(kit) and not (kit / "ckpt/ckpt_full_48852.pt").exists()


def test_pull_size_band_rejects_a_tiny_file_with_the_default_band(fake):
    root, thor, kit = fake
    finish_thor(thor)
    env = pull_env(fake)
    for k in ("FIN_CKPT_MIN", "FIN_CKPT_MAX", "FIN_EMA_MIN", "FIN_EMA_MAX"):
        env.pop(k)
    subprocess.run([SH, msys(PULL)], env=env, capture_output=True, timeout=120)
    assert "ZZPFNBADSIZEZZ" in pmarkers(kit) and not (kit / "ckpt/ckpt_full_48852.pt").exists()


def test_pull_without_model_ema_is_not_a_failure(fake):
    root, thor, kit = fake
    (thor / "model_ema.pt").unlink()
    finish_thor(thor)
    run_pull(fake)
    mk = pmarkers(kit)
    assert "ZZPFNNOMODELEMAZZ" in mk and "ZZPFNDONEZZ" in mk and "model_ema_md5 NONE" in plog(kit) and not (kit / "ckpt/model_ema_48852_thor.pt").exists()


def test_pull_is_idempotent_and_detects_a_damaged_bank(fake):
    root, thor, kit = fake
    finish_thor(thor)
    run_pull(fake)
    n1 = (root / "scp_count.txt").read_text().count("scp ")
    run_pull(fake)
    assert "ZZPFNALREADYZZ" in pmarkers(kit) and (root / "scp_count.txt").read_text().count("scp ") == n1, "a consistent bank was copied again"
    with open(kit / "ckpt/ckpt_full_48852.pt", "ab") as f:
        f.write(b"damage")
    run_pull(fake)
    assert "ZZPFNREBUILDZZ" in pmarkers(kit) and (kit / "ckpt/ckpt_full_48852.pt").read_bytes() == PAYLOAD


def test_pull_dead_ssh_is_loud(fake):
    root, thor, kit = fake
    run_pull(fake, FAKE_SSH_DEAD="1")
    assert "ZZPFNSSHDEADZZ" in pmarkers(kit) and not (kit / "ckpt/ckpt_full_48852.pt").exists()


def test_pull_times_out_without_a_done_marker(fake):
    root, thor, kit = fake
    run_pull(fake, FIN_MAX_WAIT_S="3")
    assert "ZZPFNTIMEOUTZZ" in pmarkers(kit) and not (kit / "ckpt/ckpt_full_48852.pt").exists()


def test_pull_supervisor_gave_up_is_loud_not_a_four_day_wait_and_a_stale_give_up_line_does_not_block_a_finished_run(fake):
    root, thor, kit = fake
    (thor / "sup_refcv8-r101-s0.log").write_text("[sup:x] exhausted 4 launches without a complete artifact. NO done-marker written.\n", encoding="utf-8")
    run_pull(fake)
    assert "ZZPFNSUPDEADZZ" in pmarkers(kit) and not (kit / "ckpt/ckpt_full_48852.pt").exists()
    # paired control: the same log line next to a FINISHED run (a done marker exists) must not stop the pull
    root2 = kit.parent.parent / "fk2"
    for sub in ("thor", "kit/ckpt"):
        (root2 / sub).mkdir(parents=True)
    for f in ("fake_ssh.sh", "fake_scp.sh", "scp_count.txt"):
        shutil.copy(root / f, root2 / f)
    (root2 / "thor/ckpt.pt").write_bytes(PAYLOAD)
    (root2 / "thor/train.log").write_text("", encoding="utf-8")
    (root2 / "thor/sup_refcv8-r101-s0.log").write_text("[sup:x] exhausted 4 launches without a complete artifact.\n", encoding="utf-8")
    finish_thor(root2 / "thor")
    run_pull((root2, root2 / "thor", root2 / "kit"))
    assert "ZZPFNDONEZZ" in pmarkers(root2 / "kit")


# ====================================================================================================================== snap40k_fix_r8.sh (the live snap40k aborts on a size EQUALITY)
SNAP_OLD, SNAP_FIX = HERE / "snap40k_r8.sh", HERE / "snap40k_fix_r8.sh"
BIG = b"rolling-ckpt-payload-" * 100000          # ~2.1 MB: large enough that 9,024 B is < 1 %, as at real scale (1.56 GB, +9,024 B between step 10,500 and 36,000)


def _snap(tmp, script, expect, payload=BIG):
    thor = tmp / "thor"
    kit = tmp / "kit"
    for d in (thor, kit / "ckpt"):
        d.mkdir(parents=True, exist_ok=True)
    (thor / "ckpt.pt").write_bytes(payload)
    (thor / "train.log").write_text("[v3:hier] ckpt step 39500 -> ckpt.pt\n[v3:hier] ckpt step 40000 -> ckpt.pt\n", encoding="utf-8")
    (tmp / "fake_ssh.sh").write_text(FAKE_SSH, encoding="utf-8", newline="\n")
    (tmp / "fake_scp.sh").write_text(FAKE_SCP, encoding="utf-8", newline="\n")
    (tmp / "cnt.txt").write_text("", encoding="utf-8")
    env = dict(os.environ, SNAP_SSH=f"sh {msys(tmp / 'fake_ssh.sh')}", SNAP_SCP=f"sh {msys(tmp / 'fake_scp.sh')}", SNAP_RDIR=msys(thor), SNAP_SRC_SPEC=f"fakehost:{msys(thor)}/ckpt.pt",
               SNAP_KIT=msys(kit), SNAP_POLL_S="1", SNAP_MAX_WAIT_S="30", SNAP_MIN_FREE_KB="1", SNAP_MAX_SSH_FAIL="3", SNAP_EXPECT_SIZE=str(expect), FAKE_SCP_COUNT=msys(tmp / "cnt.txt"))
    subprocess.run([SH, msys(script)], env=env, capture_output=True, timeout=120)
    log = (kit / "chain/snap40k.log").read_text(encoding="utf-8") if (kit / "chain/snap40k.log").exists() else ""
    return log, kit / "ckpt/ckpt_full_40000.pt"


def test_snap40k_original_aborts_when_the_live_size_differs_from_the_stale_expectation_and_the_fix_does_not(tmp_path):
    """MEASURED on Thor 2026-10-19: ckpt.pt is 1,560,412,761 B at step 36,000 vs SNAP_EXPECT_SIZE 1,560,403,737 (step 10,500).  Same situation, scaled: expectation off by 9,024 B."""
    exp = len(BIG) + 9024
    log0, dest0 = _snap(tmp_path / "orig", SNAP_OLD, exp)
    assert "ZZS40BADSIZEZZ" in log0 and not dest0.exists(), "the original should refuse a snapshot whose size merely differs from the stale expectation"
    log1, dest1 = _snap(tmp_path / "fix", SNAP_FIX, exp)
    assert "ZZS40DONEZZ" in log1 and dest1.read_bytes() == BIG and (dest1.parent / "ckpt_full_40000.pt.md5").exists()


def test_snap40k_fix_still_refuses_a_grossly_wrong_size_and_accepts_the_exact_one(tmp_path):
    log, dest = _snap(tmp_path / "gross", SNAP_FIX, len(BIG) * 2)
    assert "ZZS40BADSIZEZZ" in log and not dest.exists(), "the size gate must still exist (a truncated or foreign file is far outside 1 %)"
    log2, dest2 = _snap(tmp_path / "exact", SNAP_FIX, len(BIG))
    assert "ZZS40DONEZZ" in log2 and dest2.exists()


def test_snap40k_fix_differs_from_the_original_in_the_size_gate_only():
    a = SNAP_OLD.read_text(encoding="utf-8").replace("\r", "").split("\n")
    b = SNAP_FIX.read_text(encoding="utf-8").replace("\r", "").split("\n")
    only_new = [l for l in b if l not in a]
    only_old = [l for l in a if l not in b]
    assert len(only_old) == 1 and "EXPECT_SIZE" in only_old[0] and '!=' in only_old[0], only_old       # exactly the equality test is gone
    assert not any("scp" in l.lower() or "md5sum" in l for l in only_new), "the fix must not touch the copy or the md5 logic"


# ====================================================================================================================== xcheck_model_ema_r8.py (real, tiny tensors)
def _ema_payload(torch, step=STEP, nu=None, w=1.0, extra_key=False, kind="ema_model_only"):
    m = {"a.weight": torch.full((4, 3), w), "a.bias": torch.arange(3, dtype=torch.float32), "bn.num_batches_tracked": torch.tensor(7)}
    if extra_key:
        m["zz"] = torch.zeros(1)
    return {"model": m, "step": step, "ema_decay": 0.9998, "num_updates": step if nu is None else nu, "kind": kind}


def _xrun(tmp, a, b):
    import torch
    pa, pb, out = tmp / "thor.pt", tmp / "der.pt", tmp / "x.json"
    torch.save(a, pa)
    torch.save(b, pb)
    subprocess.run([PYV, "-I", str(XCHECK), "--thor-ema", win(pa), "--derived", win(pb), "--step", str(STEP), "--out", win(out)], capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.skipif(not have_torch(), reason="torch not importable in the venv")
def test_xcheck_equal_files_pass_and_pin_the_md5s(tmp_path):
    import torch
    j = _xrun(tmp_path, _ema_payload(torch), _ema_payload(torch))
    assert j["ok"] is True and j["n_tensors"] == 3 and j["n_mismatch"] == 0 and len(j["md5"]["thor"]) == 32 and len(j["md5"]["derived"]) == 32


@pytest.mark.skipif(not have_torch(), reason="torch not importable in the venv")
def test_xcheck_one_perturbed_tensor_fails_the_check_and_names_it(tmp_path):
    import torch
    j = _xrun(tmp_path, _ema_payload(torch, w=1.0), _ema_payload(torch, w=1.0000001))      # a float32 1-ulp-class difference in ONE tensor
    assert j["ok"] is False and j["n_mismatch"] == 1 and j["mismatch_examples"] == ["a.weight"]


@pytest.mark.skipif(not have_torch(), reason="torch not importable in the venv")
@pytest.mark.parametrize("bad", [dict(step=48500), dict(nu=48500), dict(extra_key=True), dict(kind="full")])
def test_xcheck_meta_and_key_defects_fail(tmp_path, bad):
    import torch
    j = _xrun(tmp_path, _ema_payload(torch), _ema_payload(torch, **bad))
    assert j["ok"] is False and j["problems"], bad


@pytest.mark.skipif(not have_torch(), reason="torch not importable in the venv")
def test_xcheck_unreadable_file_is_a_fail_with_a_reason_never_a_pass(tmp_path):
    out = tmp_path / "x.json"
    (tmp_path / "bad.pt").write_bytes(b"not a torch file")
    import torch
    torch.save(_ema_payload(torch), tmp_path / "ok.pt")
    subprocess.run([PYV, "-I", str(XCHECK), "--thor-ema", win(tmp_path / "bad.pt"), "--derived", win(tmp_path / "ok.pt"), "--step", str(STEP), "--out", win(out)], capture_output=True)
    j = json.loads(out.read_text(encoding="utf-8"))
    assert j["ok"] is False and j["problems"]


# ====================================================================================================================== chain_final_r8.sh with STUB heavy steps
STUBS = r'''
import hashlib, json, os, sys, time
from pathlib import Path
CALLS = os.environ["STUB_CALLS"]
def log(name, argv):
    with open(CALLS, "a", encoding="utf-8") as f:
        f.write(name + " " + " ".join(argv) + "\n")
def opt(argv, k, d=None):
    return argv[argv.index(k) + 1] if k in argv else d
def md5(p):
    h = hashlib.md5(); h.update(Path(p).read_bytes()); return h.hexdigest()
def put(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")
PKG = Path(os.environ.get("R8_PKGW", "."))
BRW = os.environ.get("R8_BATT_ROOT_W", ".")
KITW = os.environ.get("R8_KITW", ".")
LOCK = Path(os.environ["STUB_LOCK"])
STEP = 48852
def main(name, argv):
    log(name, argv)
    if name == "ema_check":
        raw, ema = opt(argv, "--raw"), opt(argv, "--ema")
        put(opt(argv, "--out"), {"ok": True, "md5": {"raw": md5(raw), "ema": md5(ema)}, "tensors": {"n_float_differing": int(os.environ.get("STUB_EMA_DIFF", "5"))},
                                 "strict_load": {"missing": 0, "unexpected": 0}})
    elif name == "run_battery":
        tag = opt(argv, "--tag")
        held = LOCK.exists() and "battery" in LOCK.read_text()
        log("run_battery_lock_held" if held else "run_battery_LOCK_NOT_HELD", [tag])
        if held and tag != os.environ.get("STUB_BATTERY_FAIL_TAG"):
            d = Path(opt(argv, "--out-root")) / tag
            put(d / "battery_summary.json", {"bars": {"x": 1}, "finished": "t", "r8": {}, "ckpt_md5": md5(opt(argv, "--ckpt"))})
            put(d / "windows_s0.json", {"e0": [1, 2]}); (d / "dump_s0").mkdir(parents=True, exist_ok=True)
    elif name == "pair_read":
        sf = {s: {"seeds": {k: {"rows": []} for k in ("0", "1")}} for s in ("S2", "S6")}
        put(opt(argv, "--out-json"), {"finished": "t", "bars": {"BAR-R8-E1": {"verdict": "x"}}, "surfaces": sf}); put(opt(argv, "--out-md"), "x")
    elif name == "bank":
        raw, tag, st = argv[1], opt(argv, "--tag"), opt(argv, "--status", "")
        put(Path(raw) / f"BATTERY_SUMMARY_{tag}.json", {"passes": {"unc": {"status": "INCOMPLETE" if st.startswith("FAILED") else "COMPLETE"}}})
    elif name == "pick_metrics":
        c = sorted(Path(argv[0]).glob("metrics_*.jsonl")); print(win(c[-1]) if c else "")
    elif name == "thor_pull":
        put(Path(os.environ["R8_KITW"]) / "thor_reads" / "metrics_20261020T200000.jsonl", "{}"); print("pull:ok:have")
    elif name == "g0chain":
        put(Path(KITW) / "battery" / f"step{STEP}" / "g0.json", {"ckpt_md5": "x", "verdict": {"G0": "FAIL", "mutation_detection": {"m1": {"detected": True}}}})
    elif name == "battchain":
        put(Path(BRW) / f"step{STEP}_unc" / "battery_summary.json", {"bars": {"x": 1}, "finished": "t", "r8": {}})
        sf = {s: {"seeds": {k: {"rows": []} for k in ("0", "1")}} for s in ("S2", "S6")}
        put(Path(BRW) / f"step{STEP}_unc" / f"PAIRED_READ_step{STEP}_unc.json", {"finished": "t", "bars": {"BAR-R8-E1": {"verdict": "x"}}, "surfaces": sf})
        put(PKG / "raw" / f"step{STEP}" / f"BATTERY_SUMMARY_step{STEP}.json", {"passes": {"unc": {"status": "COMPLETE"}}})
    elif name == "wrapfix":
        put(opt(argv, "--out"), {"ok": True, "CLAUSE": "PASS", "ckpt_md5": md5(opt(argv, "--ckpt"))})
    elif name == "clamp":
        put(PKG / "raw/clamp_posthoc/trees" / f"{opt(argv, '--tree')}.json", {"seeds": {}, "controls": {"ALL_PASS": True}})
    elif name == "compliant":
        t = opt(argv, "--tree"); ok = (PKG / "raw/clamp_posthoc/trees" / f"{t}.json").exists()
        put(PKG / "raw/compliant_reread/trees" / f"{t}.json", {"controls": {"ALL_PASS": ok}, "bars": {"status": "WRITTEN (all controls passed)" if ok else "WITHHELD"}})
    elif name == "forward":
        t = opt(argv, "--tree"); fn = {"a0k0_headroom_r8.py": ("a0k0_headroom", "FORWARD_"), "a0k0_headroom_lon_r8.py": ("a0k0_lon", "FORWARD_LON_"), "a0k0_headroom_retime_r8.py": ("a0k0_retime", "FORWARD_RETIME_")}[os.environ["STUB_SCRIPT"]]
        cj = PKG / "raw/compliant_reread/trees" / f"{t}.json"
        have = cj.exists() and json.loads(cj.read_text())["controls"]["ALL_PASS"] is True
        put(PKG / "raw" / fn[0] / f"{fn[1]}{t}.json", {"controls": {"ALL_PASS": have}, "bars": {}})
    elif name == "tf_forward":
        t = opt(argv, "--tree"); g = PKG / "raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json"
        if g.exists():
            put(PKG / "raw/retime_trainfit" / f"FORWARD_TRAINFIT_{t}.json", {"controls": {"ALL_PASS": True}, "const": {"ALL_PASS": True}, "role": "BESIDE (not in the verdict)"})
    elif name == "render":
        pass
    elif name == "heading":
        put(Path(opt(argv, "--out-dir")) / f"{opt(argv, '--tag')}.json", {}); put(Path(opt(argv, "--out-dir")) / f"{opt(argv, '--tag')}.md", "x")
    elif name == "xcheck":
        a, b = opt(argv, "--thor-ema"), opt(argv, "--derived")
        put(opt(argv, "--out"), {"ok": os.environ.get("STUB_XCHECK_FAIL") != "1", "n_tensors": 3, "md5": {"thor": md5(a), "derived": md5(b)}})
    elif name == "extract":
        src = Path(opt(argv, "--src")); internal = int(src.read_text().split("step=")[1].split()[0]); exp = opt(argv, "--expect-step")
        if exp is not None and int(exp) != internal:
            print(json.dumps({"ok": False, "reason": f"ValueError: source step is {internal}, expected {exp}"})); return 1
        out = Path(opt(argv, "--out-dir")); out.mkdir(parents=True, exist_ok=True)
        (out / f"ckpt_{internal}.pt").write_bytes(b"raw%d" % internal)
        if os.environ.get("STUB_TORCH") == "1":
            import torch
            torch.save({"model": {"w": torch.ones(4)}, "step": internal, "ema_decay": 0.9998, "num_updates": internal, "kind": "ema_model_only"}, out / f"ckpt_ema_{internal}.pt")
        else:
            (out / f"ckpt_ema_{internal}.pt").write_bytes(b"ema%d" % internal)
        put(out / f"ckpt_{internal}.provenance.extract_ema.json", {"ok": True, "step": internal, "src_md5": md5(src)})
        print(json.dumps({"ok": True, "step": internal}))
'''
KDQ_STUB = r'''#!/bin/sh
echo "kdq KD_STEP=$KD_STEP KD_NO_RELEASE=$KD_NO_RELEASE KD_SEG_MAX=$KD_SEG_MAX KD_LOG=$KD_LOG" >> "$STUB_CALLS"
[ -e "$STUB_LOCK" ] && echo "kdq_LOCK_STILL_HELD_BY_THE_CHAIN" >> "$STUB_CALLS"
[ "$STUB_KD_FAIL" = 1 ] && exit 0
R="$R8_KIT/kdraw/step${KD_STEP}ema_unc"
for s in 2 3 4 5 6 7; do mkdir -p "$R/dump_s$s"; echo '{"ok": true}' > "$R/dump_s$s/KDRAW_VERIFY.json"; done
echo '{"verdict": {}, "cases": {"a": 1, "b": 2, "c": 3}}' > "$R/KDRAW_CONTROL.json"
'''


def _sh_stub(body: str) -> str:
    return "#!/bin/sh\n" + body + "\n"


def make_sandbox(tmp: Path, internal_step: int = STEP) -> dict:
    sb = tmp / "sb"
    pkg, code, kit, arch, bat = sb / "pkg", sb / "pkg/code", sb / "kit", sb / "arch", sb / "bat"
    for d in (code, kit / "chain", kit / "ckpt", kit / "thor_reads", kit / "battery", arch, bat, pkg / "raw/heading_drift"):
        d.mkdir(parents=True, exist_ok=True)
    (code / "_stubs.py").write_text(STUBS, encoding="utf-8")
    calls, lock = sb / "calls.log", sb / "lock.cur"
    calls.write_text("", encoding="utf-8")
    pys = {"ema_check_r8.py": "ema_check", "run_battery_r8.py": "run_battery", "pair_read_r8.py": "pair_read", "bank_battery_r8.py": "bank", "pick_metrics.py": "pick_metrics",
           "wrapper_clause_r8fix_gpu.py": "wrapfix", "clamp_posthoc_r8.py": "clamp", "compliant_reread_r8.py": "compliant", "a0k0_headroom_r8.py": "forward",
           "a0k0_headroom_lon_r8.py": "forward", "a0k0_headroom_retime_r8.py": "forward", "retime_trainfit_r8.py": "tf_forward", "xcheck_stub.py": "xcheck"}
    for fn, nm in pys.items():
        body = ("import os, sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport _stubs\n"
                f"os.environ['STUB_SCRIPT'] = {fn!r}\nnm = {nm!r}\nif nm == 'forward' and sys.argv[1:2] == ['render']: nm = 'render'\nsys.exit(_stubs.main(nm, sys.argv[1:]) or 0)\n")
        (code / fn).write_text(body, encoding="utf-8")
    (pkg / "raw/heading_drift/heading_drift.py").write_text("import os, sys\nsys.path.insert(0, os.environ['STUB_CODE'])\nimport _stubs\nsys.exit(_stubs.main('heading', sys.argv[1:]) or 0)\n", encoding="utf-8")
    (code / "extract_stub.py").write_text("import os, sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport _stubs\nsys.exit(_stubs.main('extract', sys.argv[2:]) or 0)\n", encoding="utf-8")
    (code / "thor_pull_r8.sh").write_text(_sh_stub('exec "$STUBPY" "$(dirname "$0")/stubcall.py" thor_pull "$@"'), encoding="utf-8", newline="\n")
    (code / "chain_g0_r8.sh").write_text(_sh_stub('exec "$STUBPY" "$(dirname "$0")/stubcall.py" g0chain "$@"'), encoding="utf-8", newline="\n")
    (code / "chain_battery_r8.sh").write_text(_sh_stub('exec "$STUBPY" "$(dirname "$0")/stubcall.py" battchain "$@"'), encoding="utf-8", newline="\n")
    (code / "pull_noop.sh").write_text(_sh_stub('echo "pull_noop" >> "$STUB_CALLS"'), encoding="utf-8", newline="\n")
    (code / "kdraw_queue_stub.sh").write_text(KDQ_STUB, encoding="utf-8", newline="\n")
    (code / "run_kdraw_analysis_r8.sh").write_text(_sh_stub(
        'echo "kd_analysis $*" >> "$STUB_CALLS"; mkdir -p "$R8_PKGL/raw/kdraw/$1"; '
        'echo \'{"overall": {"overall": "PASS"}, "void_controls_failed": [], "status": "x"}\' > "$R8_PKGL/raw/kdraw/$1/KDRAW_E1.json"'), encoding="utf-8", newline="\n")
    (code / "run_combo_ens_retime_r8.sh").write_text(_sh_stub(
        'echo "combo $*" >> "$STUB_CALLS"; mkdir -p "$R8_PKGL/raw/combo_ens_retime/$1_$2"; '
        'echo \'{"status": "STACKS", "void_controls_failed": []}\' > "$R8_PKGL/raw/combo_ens_retime/$1_$2/COMBO_$1_$2.json"'), encoding="utf-8", newline="\n")
    (code / "stubcall.py").write_text("import os, sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport _stubs\nsys.exit(_stubs.main(sys.argv[1], sys.argv[2:]) or 0)\n", encoding="utf-8")
    for f in ("summarize_40k_r8.py", "final_index_r8.py"):
        shutil.copy(HERE / f, code / f)
    (bat / "gpu_lock.py").write_text(
        "import json, os, sys\nsys.path.insert(0, os.environ['STUB_CODE'])\nimport _stubs\nfrom pathlib import Path\n"
        "a = sys.argv[1:]; cmd = a[0]; job = a[a.index('--job') + 1]\n_stubs.log('lock_' + cmd, [job])\n"
        "if cmd == 'acquire':\n    if _stubs.LOCK.exists(): _stubs.put(a[a.index('--out') + 1], {'ok': False})\n    else:\n"
        "        _stubs.LOCK.write_text(job); _stubs.put(a[a.index('--out') + 1], {'ok': True})\n"
        "elif cmd == 'release' and _stubs.LOCK.exists() and _stubs.LOCK.read_text() == job: _stubs.LOCK.unlink()\n", encoding="utf-8")
    (bat / "wait_commit.py").write_text("import sys, json\na = sys.argv[1:]\nopen(a[a.index('--out') + 1], 'w').write(json.dumps({'met': True}))\n", encoding="utf-8")
    (bat / "chain_marker.py").write_text("pass\n", encoding="utf-8")
    (bat / "gpu_watch.py").write_text("pass\n", encoding="utf-8")
    (kit / "ckpt/config.json").write_text("{}", encoding="utf-8")
    full = kit / "ckpt" / f"ckpt_full_{STEP}.pt"
    plog_ = kit / "chain/pull_final.log"
    env = dict(os.environ)
    env.update({
        "R8_PKGW": win(pkg), "R8_PKGL": msys(pkg), "R8_KITW": win(kit), "R8_KIT": msys(kit), "R8_BATT_ROOT_W": win(arch), "R8_BATT_ROOT_L": msys(arch), "R8_BAT": win(bat), "R8_PY": PYV,
        "CFN_EXTRACT_PY": win(code / "extract_stub.py"), "CFN_XCHECK_PY": win(code / "xcheck_stub.py"), "CFN_PULL_SH": msys(code / "pull_noop.sh"), "CFN_KDQ": msys(code / "kdraw_queue_stub.sh"),
        "CFN_PULL_LOG": msys(plog_), "CFN_FULL": win(full), "CFN_FULL_L": msys(full),
        "CFN_POLL_S": "1", "CFN_LANE_POLL_S": "1", "CFN_GATE_GAP_S": "0", "CFN_MEM_POLL_S": "1", "CFN_FREEGB_CMD": "echo 99", "CFN_MIN_FREE_KB": "1", "CFN_TF_WAIT_S": "0", "CFN_TF_POLL_S": "1",
        "STUB_CALLS": win(calls), "STUB_LOCK": win(lock), "STUB_CODE": win(code), "STUBPY": PYV, "PYTHONDONTWRITEBYTECODE": "1",
    })
    for k in ("STUB_EMA_DIFF", "STUB_BATTERY_FAIL_TAG", "STUB_XCHECK_FAIL", "STUB_KD_FAIL", "STUB_TORCH", "CFN_XCHECK"):
        env.pop(k, None)
    S = {"sb": sb, "pkg": pkg, "code": code, "kit": kit, "arch": arch, "calls": calls, "lock": lock, "plog": plog_, "full": full, "env": env, "log": kit / "chain/chain_final.log"}
    S["internal"] = internal_step
    return S


def write_pull(S: dict, kind: str = "done", internal_step: int | None = None, with_ema: bool = True, summary: dict | None = None, torch_files: bool = False) -> None:
    """Fake the artifacts pull_final_r8.sh leaves behind (its own behaviour is tested above against the fake Thor)."""
    kit, full = S["kit"], S["full"]
    st = S["internal"] if internal_step is None else internal_step
    if torch_files:
        import torch
        torch.save({"model": {}, "step": st, "ema": {}}, full)
    else:
        full.write_text(f"step={st} payload", encoding="utf-8")
    md = hashlib.md5(full.read_bytes()).hexdigest()
    S["md5"] = md
    (kit / "ckpt" / f"ckpt_full_{STEP}.pt.md5").write_text(f"{md}  ckpt_full_{STEP}.pt\n", encoding="utf-8")
    (kit / "ckpt" / f"ckpt_full_{STEP}.pt.provenance.json").write_text(json.dumps({"step": STEP, "md5": md, "size": full.stat().st_size, "ok": True}), encoding="utf-8")
    (kit / "ckpt" / f"summary_thor_{STEP}.json").write_text(json.dumps(TRAINER_SUMMARY if summary is None else summary), encoding="utf-8")
    th = kit / "ckpt" / f"model_ema_{STEP}_thor.pt"
    if with_ema and not torch_files:
        th.write_bytes(b"trainer-ema")
    L = [f"ZZPFNSTARTZZ 2026-10-20T18:00:00+0200 pid 1 step {STEP} dest {msys(full)}", "ZZPFNWAITZZ 2026-10-20T18:05:00+0200 done=0 summary-step=0 ckpt-lines-for-48852=0"]
    if kind == "done":
        L.append(f"ZZPFNDONEZZ 2026-10-20T19:05:00+0200 ckpt_md5 {md} size 17 model_ema_md5 {'0' * 32 if with_ema else 'NONE'} (file=ckpt.pt)")
    elif kind:
        L.append(f"{kind} 2026-10-20T19:05:00+0200 simulated")
    S["plog"].write_text("\n".join(L) + "\n", encoding="utf-8")


def run_chain(S: dict, script: Path | None = None, timeout: int = 170, extra_env: dict | None = None) -> str:
    env = dict(S["env"])
    env.update(extra_env or {})
    subprocess.run([SH, msys(script or CHAIN)], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    return S["log"].read_text(encoding="utf-8") if S["log"].exists() else ""


def calls(S) -> list[str]:
    return S["calls"].read_text(encoding="utf-8").splitlines()


def has(L, prefix: str) -> bool:
    return any(x.startswith(prefix) for x in L)


def gains(S):
    d = S["pkg"] / "raw/retime_trainfit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "FROZEN_GAINS_RETIME_train.json").write_text("{}", encoding="utf-8")
    (d / "FROZEN_GAINS_RETIME_train.sha256").write_text("0" * 64 + "\n", encoding="utf-8")


def fake_earlier_tree(S, step: int, ema: bool = True):
    d = S["arch"] / f"step{step}{'ema' if ema else ''}_unc"
    d.mkdir(parents=True, exist_ok=True)
    (d / "battery_summary.json").write_text("{}", encoding="utf-8")


@pytest.fixture()
def S(tmp_path):
    return make_sandbox(tmp_path)


HEAVY = ("extract", "run_battery", "g0chain", "battchain", "clamp", "compliant", "forward", "heading", "xcheck", "kdq", "tf_forward", "combo", "kd_analysis")


def test_happy_path_order_artifacts_and_the_registered_reads(S):
    gains(S)
    fake_earlier_tree(S, 30000)
    fake_earlier_tree(S, 5000)
    write_pull(S)
    txt = run_chain(S)
    c = calls(S)
    for m in ("PULLSEEN", "MD5OK", "SUMJSONOK", "EXTRACTOK", "EMAGATEOK", "XCHECKOK", "EMADONE", "KDDONE", "KDANALYSIS", "RAWDONE", "LANEEND", "TFOK", "COMBODONE", "SUMMARYOK", "INDEXOK", "END"):
        assert f"ZZCFN{m}ZZ" in txt, (m, txt[-1800:])
    assert "ema=ok kd=ok raw=ok combo=rc 0" in txt
    names = [x.split()[0] for x in c]
    seq = ["extract", "ema_check", "xcheck", "lock_acquire", "run_battery_lock_held", "pair_read", "kdq", "kd_analysis", "g0chain", "wrapfix", "battchain", "combo"]
    idx = [names.index(n) for n in seq]
    assert idx == sorted(idx), (seq, idx, names)
    assert not has(c, "run_battery_LOCK_NOT_HELD") and not has(c, "kdq_LOCK_STILL_HELD") and not S["lock"].exists(), "a battery ran without the lock, the K-draw started while the chain held the lock, or the lock leaked"
    rb = [x for x in c if x.startswith("run_battery ")][0]
    assert "ckpt_ema_48852.pt" in rb and "--no-g0" in rb and "--infer-seeds 0,1" in rb and f"--tag {TE}" in rb and "--lock-held" in rb
    kq = [x for x in c if x.startswith("kdq ")][0]
    assert "KD_STEP=48852" in kq and "KD_NO_RELEASE=1" in kq and "KD_SEG_MAX=8" in kq, kq                    # the final tree; RELEASE_R7 not rewritten; one lock hold
    assert [x for x in c if x.startswith("kd_analysis")] == [f"kd_analysis {TE}"] and [x for x in c if x.startswith("combo")] == [f"combo {TE} train"]
    lines = txt.splitlines()

    def idx_of(marker, needle):
        return [i for i, l in enumerate(lines) if f"ZZCFN{marker}ZZ" in l and needle in l]
    for tree in (TE, TR):
        cl, co, fw = idx_of("CLAMPOK", tree), idx_of("COMPLOK", tree), idx_of("FWDOK", tree + " ->")
        assert cl and co and len(fw) == 3 and cl[0] < co[0] < min(fw), (tree, cl, co, fw)
        assert idx_of("TFOK", tree), tree
    assert txt.count("ZZCFNHEADDONEZZ") == 2 and txt.count("ZZCFNFWDOKZZ") == 6 and txt.count("ZZCFNCOMPLOKZZ") == 2 and txt.count("ZZCFNTFOKZZ") == 2
    # the heading's earlier list holds ONLY the trees that exist (30k and 5k here), most recent first, never a 40k/20k/15k/10.5k that is not banked
    hd = [x for x in c if x.startswith("heading ") and "--label-new EMA-final" in x][0]
    assert "EMA-30k=" in hd and "EMA-5k=" in hd and "EMA-40k" not in hd and "EMA-20k" not in hd and hd.index("EMA-30k") < hd.index("EMA-5k")
    assert (S["pkg"] / f"raw/step{STEP}/READ_FINAL.json").exists() and (S["pkg"] / f"raw/step{STEP}/FINAL_INDEX.json").exists()
    idxj = json.loads((S["pkg"] / f"raw/step{STEP}/FINAL_INDEX.json").read_text(encoding="utf-8"))
    assert idxj["weights"]["step_inside_file_per_extractor"] == STEP and idxj["weights"]["extract_ok"] is True and idxj["stages_last_marker"]["CHAIN"].startswith("ZZCFNENDZZ")
    assert (S["kit"] / f"ckpt/ckpt_{STEP}.md5").read_text().count("\n") == 3
    src = CHAIN.read_text(encoding="utf-8")
    body = src.split("STEP=")[1]
    assert "git " not in body.replace("# ", "") and "ssh " not in body and "scp " not in body and "G:/" not in src and "/g/" not in src, "the chain must touch neither git, nor Thor directly, nor G:"


def test_resumable_second_run_redoes_nothing(S):
    gains(S)
    write_pull(S)
    run_chain(S)
    n1 = len([x for x in calls(S) if x.split()[0] in HEAVY])
    txt = run_chain(S)
    n2 = len([x for x in calls(S) if x.split()[0] in HEAVY])
    assert n1 == n2, "a second run repeated heavy work"
    for m in ("EXTRACTHAVE", "XCHECKHAVE", "EMAHAVE", "KDHAVE", "KDANALYSISHAVE", "RAWHAVE", "COMPLHAVE", "FWDHAVE", "HEADHAVE", "TFHAVE", "COMBOHAVE"):
        assert f"ZZCFN{m}ZZ" in txt, m


@pytest.mark.parametrize("marker", ["ZZPFNMD5MISMATCHZZ", "ZZPFNHOLEZZ", "ZZPFNWRONGSTEPZZ", "ZZPFNSSHDEADZZ", "ZZPFNSUPDEADZZ", "ZZPFNCHANGEDZZ", "ZZPFNTIMEOUTZZ", "ZZPFNBADSIZEZZ"])
def test_a_pull_failure_marker_stops_at_a(S, marker):
    write_pull(S, marker)
    txt = run_chain(S)
    assert "ZZCFNPULLFAILZZ" in txt and "ZZCFNABORTZZ" in txt and "stage a" in txt
    assert not any(x.split()[0] in HEAVY + ("ema_check", "lock_acquire") for x in calls(S))


def test_a_pull_that_ends_without_any_terminal_marker_is_loud(S):
    write_pull(S, "")
    txt = run_chain(S)
    assert "ZZCFNPULLFAILZZ" in txt and "no terminal marker" in txt and not has(calls(S), "extract")


def test_a_md5_sidecar_disagreement_stops_at_a(S):
    write_pull(S)
    (S["kit"] / f"ckpt/ckpt_full_{STEP}.pt.md5").write_text("0" * 32 + f"  ckpt_full_{STEP}.pt\n", encoding="utf-8")
    txt = run_chain(S)
    assert "ZZCFNMD5BADZZ" in txt and not has(calls(S), "extract")


def test_a_empty_sidecar_is_inconclusive_never_a_match(S):
    write_pull(S)
    (S["kit"] / f"ckpt/ckpt_full_{STEP}.pt.md5").write_text("", encoding="utf-8")
    (S["kit"] / f"ckpt/ckpt_full_{STEP}.pt.provenance.json").write_text("{}", encoding="utf-8")
    txt = run_chain(S)
    assert "ZZCFNMD5INCONCLUSIVEZZ" in txt and "ZZCFNMD5OKZZ" not in txt and not has(calls(S), "extract")


def test_a_summary_copy_that_does_not_say_done_at_the_final_step_stops_at_a(S):
    write_pull(S, summary=dict(TRAINER_SUMMARY, step=48500))
    txt = run_chain(S)
    assert "ZZCFNSUMJSONBADZZ" in txt and not has(calls(S), "extract")
    S2 = make_sandbox(S["sb"].parent / "ok")
    write_pull(S2, summary=SUP_SUMMARY)                       # paired control: the supervisor's schema (final_step) passes
    assert "ZZCFNSUMJSONOKZZ" in run_chain(S2)


def test_b_internal_step_48500_stops_at_b_with_the_REAL_extract_guard(S, tmp_path):
    """The pull's md5 is genuine and summary.json says done@48852 (the supervisor's hole), but the step INSIDE ckpt.pt is 48,500.  The real extract_ema_r8.py refuses it before any model build."""
    if not have_torch():
        pytest.skip("torch not importable in the venv")
    S2 = make_sandbox(tmp_path / "b", internal_step=48500)
    write_pull(S2, summary=SUP_SUMMARY, torch_files=True)
    txt = run_chain(S2, extra_env={"CFN_EXTRACT_PY": win(HERE / "extract_ema_r8.py"), "CFN_MIN_COMMIT_GB": "0.1", "PYTHONPATH": "C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval"}, timeout=240)
    out = (S2["kit"] / "chain/cfn_extract.out").read_text(encoding="utf-8", errors="replace")
    assert "ZZCFNMD5OKZZ" in txt and "ZZCFNSUMJSONOKZZ" in txt, txt                  # the pull itself and its summary.json look perfect ...
    assert "ZZCFNEXTRACTFAILZZ" in txt and "ZZCFNABORTZZ" in txt                      # ... and the chain still stops at b
    assert "source step is 48500, expected 48852" in out, out[-800:]                  # by the REAL guard, not by some other failure
    assert not (S2["kit"] / "ckpt/ckpt_ema_48852.pt").exists() and not (S2["kit"] / "ckpt/ckpt_ema_48500.pt").exists()
    assert not any(x.split()[0] in ("ema_check", "run_battery", "lock_acquire", "g0chain", "kdq") for x in calls(S2)), "work ran past stage b"


def _mutant(S, drop: list[str], name: str) -> Path:
    s = CHAIN.read_text(encoding="utf-8")
    for d in drop:
        assert d in s, d
        s = s.replace(d, "")
    p = S["code"] / name
    p.write_text(s, encoding="utf-8", newline="\n")
    return p


def test_b_mutation_layer1_removed_layer2_still_stops_and_the_paired_control_reaches_the_battery(tmp_path):
    """Layer 1 = `--expect-step`, layer 2 = the chain's own re-assertion on the provenance ARTIFACT (it names the file by the step the chain expects).  Delete layer 1 on a 48,500 pull -> the stub extractor accepts
    it, and layer 2 still stops the chain.  The PAIRED CONTROL (same sandbox, internal step 48,852) reaches the EMA battery: the observable the stop-assertion reads is demonstrably able to go non-empty."""
    A = make_sandbox(tmp_path / "m1", internal_step=48500)
    write_pull(A)
    txt = run_chain(A, _mutant(A, ['--expect-step "$STEP" '], "chain_mut1.sh"))
    assert "ZZCFNEXTRACTFAILZZ" in txt and not has(calls(A), "run_battery") and not has(calls(A), "ema_check"), "layer 2 must hold with layer 1 removed"
    assert (A["kit"] / "ckpt/ckpt_48500.pt").exists(), "the mutant stub did write a 48,500 pair (so the guard, not an absent file, is what stopped it)"
    B = make_sandbox(tmp_path / "m2", internal_step=STEP)
    write_pull(B)
    run_chain(B)
    assert has(calls(B), "run_battery"), "paired control: a valid 48,852 checkpoint must reach the battery"


def test_b_mutation_md5_recheck_removed_lets_a_sidecar_lie_through(tmp_path):
    A = make_sandbox(tmp_path / "m3")
    write_pull(A)
    (A["kit"] / f"ckpt/ckpt_full_{STEP}.pt.md5").write_text("0" * 32 + f"  ckpt_full_{STEP}.pt\n", encoding="utf-8")
    mut = _mutant(A, ['if [ "$LMD" = "$SIDE" ] && [ "$LMD" = "$PROV" ] && [ "$LMD" = "$DONEMD" ]; then'], "chain_mut_md5.sh")
    s = mut.read_text(encoding="utf-8").replace('SNAPMD5="$LMD"; mark MD5OK', 'if true; then SNAPMD5="$LMD"; mark MD5OK', 1)       # keep the else-branch syntactically valid
    mut.write_text(s, encoding="utf-8", newline="\n")
    run_chain(A, mut)
    assert has(calls(A), "extract"), "the mutant (md5 agreement deleted) should have gone on to extract"


def test_c_ema_gate_identical_weights_blocks_the_battery(S):
    write_pull(S)
    txt = run_chain(S, extra_env={"STUB_EMA_DIFF": "0"})
    assert "ZZCFNEMAGATEFAILZZ" in txt and not has(calls(S), "run_battery") and not has(calls(S), "lock_acquire") and not has(calls(S), "kdq")
    S2 = make_sandbox(S["sb"].parent / "g2")
    write_pull(S2)
    run_chain(S2, _mutant(S2, ["and j['tensors']['n_float_differing']>0 "], "chain_mut_gate.sh"), extra_env={"STUB_EMA_DIFF": "0"})
    assert has(calls(S2), "run_battery"), "mutant without the differ-clause should reach the battery"


def test_c_xcheck_failure_stops_at_b_and_warn_mode_continues(S):
    write_pull(S)
    txt = run_chain(S, extra_env={"STUB_XCHECK_FAIL": "1"})
    assert "ZZCFNXCHECKFAILZZ" in txt and "ZZCFNABORTZZ" in txt and "stage b" in txt and not has(calls(S), "run_battery") and not has(calls(S), "lock_acquire")
    S2 = make_sandbox(S["sb"].parent / "w")
    write_pull(S2)
    txt2 = run_chain(S2, extra_env={"STUB_XCHECK_FAIL": "1", "CFN_XCHECK": "warn"})
    assert "ZZCFNXCHECKWARNZZ" in txt2 and has(calls(S2), "run_battery"), "CFN_XCHECK=warn must continue (and say so)"


def test_c_xcheck_not_applicable_without_the_trainer_ema_file(S):
    write_pull(S, with_ema=False)
    txt = run_chain(S)
    assert "ZZCFNXCHECKNAZZ" in txt and not has(calls(S), "xcheck") and "ZZCFNEMADONEZZ" in txt, "a resume AT step==steps writes no model_ema.pt: not a failure"


@pytest.mark.skipif(not have_torch(), reason="torch not importable in the venv")
def test_c_REAL_xcheck_inside_the_chain_one_perturbed_tensor_stops_it_and_the_paired_control_passes(tmp_path):
    import torch

    def build(tag, w):
        X = make_sandbox(tmp_path / tag)
        write_pull(X, with_ema=False)
        torch.save(_ema_payload(torch, w=w), X["kit"] / f"ckpt/model_ema_{STEP}_thor.pt")     # the trainer's file: ones(4,3)*w
        return X
    env = {"STUB_TORCH": "1", "CFN_XCHECK_PY": win(XCHECK)}
    # the stub extractor's derived EMA is {'w': ones(4)} -> rebuild the trainer's file in the SAME layout so the pair is comparable, then perturb one element
    ok = build("x_ok", 1.0)
    torch.save({"model": {"w": torch.ones(4)}, "step": STEP, "ema_decay": 0.9998, "num_updates": STEP, "kind": "ema_model_only"}, ok["kit"] / f"ckpt/model_ema_{STEP}_thor.pt")
    t_ok = run_chain(ok, extra_env=env)
    assert "ZZCFNXCHECKOKZZ" in t_ok and has(calls(ok), "run_battery"), t_ok[-800:]
    bad = build("x_bad", 1.0)
    w = torch.ones(4)
    w[2] = 1.0000001
    torch.save({"model": {"w": w}, "step": STEP, "ema_decay": 0.9998, "num_updates": STEP, "kind": "ema_model_only"}, bad["kit"] / f"ckpt/model_ema_{STEP}_thor.pt")
    t_bad = run_chain(bad, extra_env=env)
    assert "ZZCFNXCHECKFAILZZ" in t_bad and "ZZCFNABORTZZ" in t_bad and not has(calls(bad), "run_battery")


def test_c_failed_ema_battery_releases_the_lock_skips_kdraw_and_does_not_block_the_raw_stage(S):
    gains(S)
    write_pull(S)
    txt = run_chain(S, extra_env={"STUB_BATTERY_FAIL_TAG": TE})
    c = calls(S)
    assert "ZZCFNEMARETRYZZ" in txt and "ZZCFNEMAFAILZZ" in txt and "ZZCFNSTAGEFAILEDZZ" in txt
    assert "ZZCFNKDSKIPZZ" in txt and not has(c, "kdq"), "no banked EMA tree: the K-draw must SKIP, naming why"
    assert "ZZCFNRAWDONEZZ" in txt, "each stage is independent: the raw stage must still run"
    assert "ema=fail kd=fail raw=ok" in txt and not S["lock"].exists()
    assert "ZZCFNLANESKIPZZ" in txt and TE in txt.split("ZZCFNLANESKIPZZ")[1].split("\n")[0], "the EMA lane must be SKIPPED, not run on missing inputs"
    assert [x for x in c if x.startswith(f"lock_release refcv8-battery-step{STEP}ema")], "the EMA stage's EXIT trap must release the lock"
    assert "ZZCFNCOMBOSKIPZZ" in txt and "kdraw-rolls" in txt.split("ZZCFNCOMBOSKIPZZ")[1].split("\n")[0]


def test_k_a_failed_kdraw_is_named_does_not_block_raw_and_blocks_the_combo(S):
    gains(S)
    write_pull(S)
    txt = run_chain(S, extra_env={"STUB_KD_FAIL": "1"})
    assert "ZZCFNKDFAILZZ" in txt and "ZZCFNRAWDONEZZ" in txt and "ZZCFNCOMBOSKIPZZ" in txt and "kdraw-rolls" in txt.split("ZZCFNCOMBOSKIPZZ")[1].split("\n")[0]
    assert not has(calls(S), "combo"), "the combo must not run on missing K-draw rolls"


def test_k_kdraw_waits_for_a_live_kdraw_queue_and_never_runs_two(S, tmp_path):
    write_pull(S)
    p = subprocess.Popen([SH, "-c", "echo $$; exec sleep 100"], stdout=subprocess.PIPE, text=True)
    pid = p.stdout.readline().strip()
    try:
        (S["kit"] / "chain/kdraw_queue.pid").write_text(pid, encoding="utf-8")
        q = subprocess.Popen([SH, msys(CHAIN)], env=dict(S["env"]), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 150 and not (S["log"].exists() and "ZZCFNKDBUSYZZ" in S["log"].read_text(encoding="utf-8")):      # synchronise on the marker, not on a sleep: the stubbed chain's speed varies
            time.sleep(1)
        assert "ZZCFNKDBUSYZZ" in S["log"].read_text(encoding="utf-8"), "the chain never reached the K-draw stage / never noticed the live queue"
        time.sleep(4)
        assert not has(calls(S), "kdq"), "the K-draw queue started while another one was alive"
    finally:
        p.kill()
    q.wait(timeout=150)
    assert has(calls(S), "kdq")


def test_tf_absent_gains_skip_with_a_reason_and_the_chain_still_ends(S):
    write_pull(S)
    txt = run_chain(S)
    assert "ZZCFNTFSKIPZZ" in txt and "FROZEN_GAINS_RETIME_train.json" in txt.split("ZZCFNTFSKIPZZ")[1].split("\n")[0]
    assert "ZZCFNCOMBOSKIPZZ" in txt and "train-fit-gains" in txt.split("ZZCFNCOMBOSKIPZZ")[1].split("\n")[0] and "ZZCFNENDZZ" in txt
    assert not has(calls(S), "tf_forward") and not has(calls(S), "combo")


def test_d_compliant_needs_the_clamp_posthoc_tree_first(S):
    write_pull(S)
    txt = run_chain(S)
    assert txt.count("ZZCFNCOMPLOKZZ") == 2
    s = CHAIN.read_text(encoding="utf-8")
    old = 'cpu_run clamp_posthoc "$PKGW/code" "$LG" "$PY" -X utf8 clamp_posthoc_r8.py read --tree "$T"'
    assert old in s
    S2 = make_sandbox(S["sb"].parent / "d2")
    write_pull(S2)
    p = S2["code"] / "chain_mut_clamp.sh"
    p.write_text(s.replace(old, "true"), encoding="utf-8", newline="\n")
    txt2 = run_chain(S2, p)
    assert "ZZCFNCOMPLFAILZZ" in txt2 and "ZZCFNCOMPLOKZZ" not in txt2
    assert txt2.count("ZZCFNFWDCTRLFAILZZ") >= 1, "a forward without the compliant tree must not read as ALL_PASS"


def test_single_instance_guard(S):
    write_pull(S)
    p = subprocess.Popen([SH, "-c", "echo $$; exec sleep 60"], stdout=subprocess.PIPE, text=True)
    live = p.stdout.readline().strip()
    try:
        (S["kit"] / "chain/chain_final.pid").write_text(live, encoding="utf-8")
        txt = run_chain(S, timeout=30)
    finally:
        p.kill()
    assert "ZZCFNDUPZZ" in txt and not has(calls(S), "extract")


def test_the_chain_uses_the_standard_lock_and_no_priority_poller():
    src = CHAIN.read_text(encoding="utf-8")
    assert '"$BAT/gpu_lock.py" acquire' in src and "gpu_lock_fast" not in src and "FAST_POLL" not in src
    assert not (HERE / "gpu_lock_fast_r8.py").exists()


def test_end_to_end_the_REAL_pull_against_a_fake_thor_feeds_the_chain(tmp_path):
    """The whole flow against fakes: Thor unfinished -> chain waits INSIDE stage a -> Thor finishes -> pull (real script, fake ssh/scp) -> chain re-verifies -> extract ... END.
    Then the MUTATION: the same flow with a byte-flipped copy stops at stage a and nothing downstream runs."""
    root = tmp_path / "e2e"
    F = (root, root / "thor", root / "kit")
    (root / "thor").mkdir(parents=True)
    (root / "fake_ssh.sh").write_text(FAKE_SSH, encoding="utf-8", newline="\n")
    (root / "fake_scp.sh").write_text(FAKE_SCP, encoding="utf-8", newline="\n")
    (root / "scp_count.txt").write_text("", encoding="utf-8")
    S = make_sandbox(root / "chainbox")
    thor = root / "thor"
    (thor / "ckpt.pt").write_bytes(b"step=48852 e2e-payload " * 200)
    (thor / "model_ema.pt").write_bytes(b"trainer-ema" * 50)
    (thor / "train.log").write_text("[v3:hier] ckpt step 48500 -> ckpt.pt\n", encoding="utf-8")
    pe = pull_env((root, thor, S["kit"]), FIN_CKPT_MIN="100", FIN_CKPT_MAX="100000", FIN_EMA_MIN="100", FIN_EMA_MAX="100000", FIN_MAX_WAIT_S="120")
    env = dict(S["env"])
    env.update({k: v for k, v in pe.items() if k.startswith("FIN_") or k == "FAKE_SCP_COUNT"})
    env["CFN_PULL_SH"] = msys(PULL)
    p = subprocess.Popen([SH, msys(CHAIN)], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    assert not has(calls(S), "extract") and not (S["kit"] / f"ckpt/ckpt_full_{STEP}.pt").exists(), "the chain went past stage a before the trainer was done"
    finish_thor(thor)
    p.wait(timeout=170)
    txt = S["log"].read_text(encoding="utf-8")
    for m in ("PULLSEEN", "MD5OK", "SUMJSONOK", "EXTRACTOK", "EMAGATEOK", "XCHECKOK", "EMADONE", "KDDONE", "RAWDONE", "END"):
        assert f"ZZCFN{m}ZZ" in txt, (m, txt[-1500:])
    # mutation: a flipped byte in transit
    root2 = tmp_path / "e2e_bad"
    (root2 / "thor").mkdir(parents=True)
    for f in ("fake_ssh.sh", "fake_scp.sh"):
        shutil.copy(root / f, root2 / f)
    (root2 / "scp_count.txt").write_text("", encoding="utf-8")
    S2 = make_sandbox(root2 / "chainbox")
    (root2 / "thor/ckpt.pt").write_bytes(b"step=48852 e2e-payload " * 200)
    (root2 / "thor/train.log").write_text("", encoding="utf-8")
    finish_thor(root2 / "thor")
    pe2 = pull_env((root2, root2 / "thor", S2["kit"]), FIN_CKPT_MIN="100", FIN_CKPT_MAX="100000", FIN_MAX_WAIT_S="20", FAKE_SCP_FLIP="1")
    env2 = dict(S2["env"])
    env2.update({k: v for k, v in pe2.items() if k.startswith("FIN_") or k.startswith("FAKE_")})
    env2["CFN_PULL_SH"] = msys(PULL)
    txt2 = run_chain(S2, extra_env={k: v for k, v in env2.items() if k.startswith("FIN_") or k.startswith("FAKE_") or k == "CFN_PULL_SH"})
    assert "ZZCFNPULLFAILZZ" in txt2 and "ZZCFNABORTZZ" in txt2 and not any(x.split()[0] in HEAVY for x in calls(S2))
    assert not (S2["kit"] / f"ckpt/ckpt_full_{STEP}.pt").exists()


# ====================================================================================================================== yield48852_r8.sh (stub lock)
YSTUB = "\n".join([
    "import json, os, sys, time",
    "a = sys.argv; out = a[a.index('--out') + 1]; d = os.path.dirname(__file__)",
    "ok = os.path.exists(os.path.join(d, 'free'))",
    "if not ok:",
    "    time.sleep(float(a[a.index('--max-wait-s') + 1]) + 0.2)",
    "json.dump({'ok': ok}, open(out, 'w'))",
    "",
])


def _yrun(tmp, *, bank, alive, log, summary=False, append_after=None, cap=30, mutate=False, max_wait=30):
    """Source yield48852_r8.sh against a stub lock; `append_after` = (seconds, text) appended to the chain log in the background."""
    t = msys(tmp)
    os.makedirs(tmp / "bat", exist_ok=True)
    (tmp / "bat" / "gpu_lock.py").write_text(YSTUB)
    (tmp / "bat" / "free").write_text("x")
    for f, on in (("bank", bank), ("alive", alive), ("sum", summary)):
        p = tmp / f
        if on:
            p.write_text("x")
        elif p.exists():
            p.unlink()
    (tmp / "w.log").write_text("")
    (tmp / "cf.log").write_text(log)
    bg = f"( sleep {append_after[0]}; printf '{append_after[1]}' >> {t}/cf.log ) &" if append_after else ""
    script = (f"PY='{msys(sys.executable)}'; BAT={t}/bat; Y48_BANK={t}/bank; Y48_SUM={t}/sum; Y48_LOG={t}/cf.log; Y48_SLEEP=1; Y48_TRY_S=1; Y48_CAP={cap}; . '{msys(YIELD)}'; "
              f"ps() {{ [ -f {t}/alive ] && echo 'x chain_final_r8.sh'; }}; " + ("bfinal_pending() { return 1; }; " if mutate else "") + bg + f" acquire_y48 job1 {t}/lock.json {t}/w.log {max_wait}; echo RC=$?; wait")
    out = subprocess.run([SH, "-c", script], capture_output=True, text=True, timeout=120).stdout
    w = (tmp / "w.log").read_text()
    return out.strip().splitlines()[-1], w.count("ZZY48YIELDZZ"), w.count("ZZY48RESUMEZZ")


def test_yield_no_bank_acquires_at_once(tmp_path):
    assert _yrun(tmp_path, bank=False, alive=True, log="ZZCFNSTARTZZ\n") == ("RC=0", 0, 0)


def test_yield_pending_ema_battery_yields_until_the_chain_logs_a_terminal_state(tmp_path):
    assert _yrun(tmp_path, bank=True, alive=True, log="ZZCFNSTARTZZ\n", append_after=(3, "ZZCFNEMAFAILZZ\\nZZCFNKDSKIPZZ\\n")) == ("RC=0", 1, 1)


def test_yield_the_ema_summary_alone_does_not_release_it_the_kdraw_rolls_still_hold_the_card(tmp_path):
    # summary exists, no terminal K-draw marker: pending until the cap ends it (cap 3 s here)
    assert _yrun(tmp_path / "a", bank=True, alive=True, log="ZZCFNSTARTZZ\nZZCFNEMADONEZZ\n", summary=True, cap=3) == ("RC=0", 1, 1)
    # ... and a terminal K-draw marker releases it at once
    assert _yrun(tmp_path / "b", bank=True, alive=True, log="ZZCFNSTARTZZ\nZZCFNEMADONEZZ\nZZCFNKDDONEZZ\n", summary=True) == ("RC=0", 0, 0)


def test_yield_released_by_ema_failure_with_kdraw_skip_chain_end_or_dead_chain(tmp_path):
    assert _yrun(tmp_path / "b", bank=True, alive=True, log="ZZCFNSTARTZZ\nZZCFNEMAFAILZZ\nZZCFNKDSKIPZZ\n") == ("RC=0", 0, 0)
    assert _yrun(tmp_path / "c", bank=True, alive=True, log="ZZCFNSTARTZZ\nZZCFNENDZZ\n") == ("RC=0", 0, 0)
    assert _yrun(tmp_path / "d", bank=True, alive=False, log="ZZCFNSTARTZZ\n") == ("RC=0", 0, 0)


def test_yield_only_the_LAST_start_segment_counts(tmp_path):
    # an earlier run ended; the resumed run is open again -> pending (cap 3 s ends the wait)
    assert _yrun(tmp_path, bank=True, alive=True, log="ZZCFNSTARTZZ\nZZCFNENDZZ\nZZCFNSTARTZZ\n", cap=3) == ("RC=0", 1, 1)


def test_yield_mutation_without_the_predicate_never_yields(tmp_path):
    assert _yrun(tmp_path, bank=True, alive=True, log="ZZCFNSTARTZZ\n", mutate=True) == ("RC=0", 0, 0)



# ====================================================================================================================== the index reads honest status
def test_index_marks_unreadable_distinct_from_absent(tmp_path):
    S = make_sandbox(tmp_path)
    write_pull(S)
    run_chain(S)
    (S["pkg"] / "raw/compliant_reread/trees" / f"{TE}.json").write_text("{not json", encoding="utf-8")
    (S["pkg"] / "raw/kdraw" / TE / "KDRAW_E1.json").unlink()
    out = tmp_path / "idx"
    subprocess.run([PYV, "-I", str(HERE / "final_index_r8.py"), "--pkg", win(S["pkg"]), "--kit", win(S["kit"]), "--arch", win(S["arch"]), "--step", str(STEP), "--chain-log", win(S["log"]), "--out-dir", win(out)], check=True, capture_output=True)
    j = json.loads((out / "FINAL_INDEX.json").read_text(encoding="utf-8"))
    assert j["trees"][TE]["compliant_reread"].startswith("UNREADABLE") and j["kdraw_E1"] == "ABSENT"
    assert j["trees"][TR]["compliant_reread"]["controls_ALL_PASS"] is True
