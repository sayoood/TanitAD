"""The TOY test reads A17's DECAYED final snapshot and gets two red arms (the Master Mind 2026-09-27: NEW-2 found the
toy failing on the dev box at 4 threads while it passed on Thor). usage: patch_toy_test.py <test file> (in place; LF)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_bytes().decode("utf-8")
assert "\r\n" not in s
OLD = '''def test_TOY_the_loop_memorises_and_the_memory_zeros_arm_cannot():
    toy = _Toy()
    main = G.run_arm(toy, "main", steps=1200, log_every=600, lr=1e-3)
    zero = G.run_arm(toy, "memory_zeros", steps=1200, log_every=600, lr=1e-3)
    m, z = main["rows"][-1], zero["rows"][-1]
    assert m["ap2m"] > 0.8 and m["rec"] > 0.8, m
    assert z["ap2m"] < 0.5, z
    assert main["rows"][-1]["presence"] < main["rows"][0]["presence"]
    assert zero["verdict"] == "failed as required"
'''
NEW = '''class _LeakyToy(_Toy):
    """RED arm (a deliberate regression): the must-fail arm is NOT blinded -- ``memory_zeros`` still reads each
    frame's own memory, so it trains exactly as MAIN does (same seed, data, schedule and code path)."""

    def _mem(self, idx):
        return self.mem[idx]


TOY_STEPS, TOY_LR = 1200, 1e-3


def _toy_run(arm, toy=None, steps=TOY_STEPS):
    return G.run_arm(toy if toy is not None else _Toy(), arm, steps=steps, log_every=600, lr=TOY_LR)


def _memorised_failures(res) -> list:
    """Why a MAIN toy run does not show memorisation, read at A17's DECAYED final snapshot ([] = it memorised).

    ⚠️ MEASURED 2026-09-27 (the box-head package's ``raw/toy/``): at a CONSTANT lr the step-1,200 snapshot is one
    draw from a trajectory whose AP@2 m swings between 0.07 and 1.00 across the 100-step readings of steps 700-1,200
    (median centre error 0.55-2.57 m against the 2 m threshold), and the intra-op THREAD COUNT picks the draw -- tip
    2ac0bfb, seed 0, main: dev box 16 threads 1.000, 4 threads 0.652 (NEW-2's failure), 1 thread 0.172; Thor
    4 threads 1.000, 1 thread 0.145. The loop memorised every time; the read-out never settled. A17's decay (the
    harness's own rule) makes the final snapshot a settled reading, so this check REQUIRES it."""
    f, f0 = res["rows"][-1], res["rows"][0]
    out = []
    if f.get("lr_factor") != 0.0:
        out.append(f"A17: the final snapshot was read at lr factor {f.get('lr_factor')}, not after the decay (0.0)")
    if not (f["ap2m"] > 0.8 and f["rec"] > 0.8):
        out.append(f"ap2m {f['ap2m']:.4f} / rec {f['rec']:.4f}: not both > 0.8")
    if not f["presence"] < f0["presence"]:
        out.append(f"presence {f['presence']:.4f} did not fall below step 0's {f0['presence']:.4f}")
    return out


def _blinded_arm_failures(res) -> list:
    """Why a MUST-FAIL ``memory_zeros`` toy run did NOT fail as required ([] = it failed as required)."""
    f = res["rows"][-1]
    out = []
    if not f["ap2m"] < 0.5:
        out.append(f"ap2m {f['ap2m']:.4f} >= 0.5")
    if res["verdict"] != "failed as required":
        out.append(f"verdict {res['verdict']!r}")
    return out


def test_TOY_the_loop_memorises_and_the_memory_zeros_arm_cannot():
    """The loop memorises the toy's geometry and the blinded arm cannot -- both read at A17's decayed final
    snapshot, so the answer does not hang on the platform's reduction order (MEASURED: 1 / 4 / 16 threads x seeds
    0-2 on the dev box AND on Thor, ``raw/toy/``)."""
    main, zero = _toy_run("main"), _toy_run("memory_zeros")
    assert _memorised_failures(main) == [], main["rows"][-1]
    assert _blinded_arm_failures(zero) == [], zero["rows"][-1]


def test_TOY_RED_a_memory_zeros_arm_that_still_sees_the_memory_is_caught():
    """RED arm, the defect this toy exists for: a must-fail arm that is NOT blinded memorises like MAIN, both checks
    on it fire, and the harness calls the arm VOID -- never 'failed as required'."""
    leak = _toy_run("memory_zeros", toy=_LeakyToy())
    assert [r.split(" ")[0] for r in _blinded_arm_failures(leak)] == ["ap2m", "verdict"]
    assert leak["verdict"] == "VOID (a must-fail arm passed its criteria)"


def test_TOY_RED_a_snapshot_read_without_the_decay_is_refused(monkeypatch):
    """RED arm, the read-out this toy relies on: a loop whose lr never decays leaves the final snapshot at factor
    1.0, and the memorisation check names A17 first (20 steps suffice: only the snapshot's factor is at stake)."""
    ok = _toy_run("main", steps=20)
    assert ok["rows"][-1]["lr_factor"] == 0.0
    assert not [r for r in _memorised_failures(ok) if r.startswith("A17")]
    monkeypatch.setattr(G, "lr_factor", lambda s, steps=2000: 1.0)
    const = _toy_run("main", steps=20)
    assert const["rows"][-1]["lr_factor"] == 1.0
    assert _memorised_failures(const)[0].startswith("A17: the final snapshot was read at lr factor 1.0,")
'''
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)
p.write_bytes(s.encode("utf-8"))
print("patched", p, "CRLF", s.count("\r\n"))
