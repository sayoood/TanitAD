"""The rewrite guard — refuse a MOD that silently DROPS evidence.

⛔ **The defect this was written for is my own, MEASURED 2026-09-17.** A research
package's ``RESULT.md`` had already landed in its 1-arm form. The 2-arm rewrite added
a control, a paired table and a budget — and **dropped seven measured values the
landed version carried** (``human_pdms``, ``fan_pdms_mean``, ``fan_nc_fail_frac``,
``sel_ep``, ``sel_comfort``, ``traj_matches_fan_sel``, ``is_release``). It was longer,
it was better in every visible way, and it was a partial revert.

⭐ **Why the lander's existing MOD guard could not see it.** That guard is
append-only: ``new_lines > tip_lines`` and the tip's last non-blank line survives. It
is a PROXY for "not a revert", and it fails in both directions here — it refuses an
honest rewrite whose headline legitimately changed, and it passes a rewrite that keeps
the last line, adds forty, and deletes every number in between.

⇒ This checks what ``CLAUDE.md`` actually says: *your blob must contain HEAD's content
PLUS your change*. For a research artifact the admissible unit of "content" is its
**numbers** and its **code identifiers**.

⛔ Every refusal below is proven by MUTATION, and the two arms that earn the module are
``_MUT_altering_a_TIP_number_is_REFUSED`` (a changed value is a lost value) and
``_a_NEW_number_may_change_freely`` (the guard must not freeze the parts the rewrite
exists to write — a guard that refuses every edit gets switched off).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from superset_check import check, norm_num, tokens  # noqa: E402

TIP = """# Arm 1 of 3

| metric | mean |
|---|---|
| `sel_pdms` | 0.8712 |
| `human_pdms` | 0.9860 |
| `fan_minade_m` | 0.8175 |

Ran 600/600 in 1,883 s. `grad_norm_max` 216.78 with 3 / 600 clipped.
"""


def _check(new: str, allow=()):
    return check(TIP, new, set(allow))


def test_an_honest_superset_passes():
    """The real case: the headline changes, the control arrives, nothing is lost."""
    new = TIP.replace("# Arm 1 of 3", "# Arms 1 and 2 of 3") + """
The control read `sel_pdms` 0.9161 and `human_pdms` 0.9860, and ran 2,061 s.
"""
    miss_n, miss_c, _, _ = _check(new)
    assert (miss_n, miss_c) == ([], [])


def test_MUT_dropping_a_number_is_REFUSED():
    """⭐ The defect this module exists for. The rewrite is LONGER and still lossy."""
    new = "\n".join(l for l in TIP.splitlines() if "human_pdms" not in l)
    new += "\n" + "\n".join(f"a new paragraph, line {i}" for i in range(40))
    miss_n, miss_c, _, _ = _check(new)
    assert "0.986" in miss_n           # normalised: trailing zero stripped
    assert "human_pdms" in miss_c


def test_MUT_altering_a_TIP_number_is_REFUSED():
    """⛔ A changed value is a lost value. 'differs from the tip' is not the test —
    a revert also differs — so the test is that the tip's own value SURVIVES."""
    miss_n, _, _, _ = _check(TIP.replace("216.78", "999.99"))
    assert miss_n == ["216.78"]


def test_a_NEW_number_may_change_freely():
    """⭐ The half that keeps the guard usable. It is a no-LOSS check, not a freeze:
    a guard that refuses every edit to new material gets switched off within a day."""
    new = TIP + "\nThe control read 0.9161.\n"
    assert _check(new)[:2] == ([], [])
    assert _check(new.replace("0.9161", "0.9999"))[:2] == ([], [])


def test_MUT_dropping_a_code_span_is_REFUSED():
    miss_n, miss_c, _, _ = _check(TIP.replace("`fan_minade_m`", "the minimum"))
    assert miss_c == ["fan_minade_m"] and miss_n == []


def test_a_code_span_survives_OUTSIDE_backticks():
    """⚠️ Prose is a legal place for an identifier to survive. Requiring the backticks
    too would refuse a rewrite that merely restyled a table into a sentence."""
    assert _check(TIP.replace("`fan_minade_m`", "fan_minade_m"))[:2] == ([], [])


def test_a_waiver_admits_ONE_named_token_and_nothing_else():
    new = TIP.replace("| `human_pdms` | 0.9860 |\n", "")
    assert _check(new)[0] == ["0.986"]
    miss_n, miss_c, _, _ = _check(new, allow={"0.986", "human_pdms"})
    assert (miss_n, miss_c) == ([], [])
    # ⛔ and it admits ONLY what it names
    miss_n, _, _, _ = _check(TIP.replace("216.78", "x"), allow={"0.986"})
    assert miss_n == ["216.78"]


def test_thousands_separators_normalise_both_ways():
    """⚠️ MEASURED in this repo: a retraction took three revisions because `45,456`
    lives in prose comma-formatted and every `\\b45456\\b` probe missed it."""
    assert norm_num("1,883") == "1883"
    assert _check(TIP.replace("1,883 s", "1883 s"))[:2] == ([], [])


def test_trailing_zeros_do_not_manufacture_a_drop():
    """0.9860 and 0.986 are the same reading; refusing on that would be noise, and a
    noisy guard is a guard nobody runs."""
    assert _check(TIP.replace("0.9860", "0.986"))[:2] == ([], [])


def test_bare_single_digits_are_not_tracked():
    """⛔ Table pipes, list markers and `of 3` would otherwise dominate the token set
    and make every refusal unreadable. Multi-digit and decimal values are tracked."""
    nums, _ = tokens("we ran 3 arms over 600 steps at 0.5 coefficient")
    assert nums == {"600", "0.5"}


def test_the_tip_being_UNREADABLE_is_INCONCLUSIVE_not_a_pass(tmp_path, capsys):
    """⛔ The empty-string hole: a probe that fails must never read as agreement.
    This repo has been bitten by exactly that on a flaky mount."""
    import superset_check as sc
    new = tmp_path / "new.md"
    new.write_text(TIP, encoding="utf-8")
    rc = sc.main(["refs/heads/no-such-branch:no/such/file.md", str(new)])
    assert rc == 2
    assert "INCONCLUSIVE" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ⛔⛔ THE AUDIT CAUGHT `SS4-a-drop-refuses` ESCAPED, AND IT IS THE ONE THAT MATTERS.
# Replacing `if mn or mc: return 1` with `if False:` left all eleven tests above GREEN,
# because every one of them calls `check()` and reads its return value — while the
# LANDER reads the **exit code** of `main()`. So the guard's only consumer had no test,
# and the module would have shipped able to report OK on a lossy rewrite.
#
# ⭐ Same defect class as the clipid guard's `_unreadable` branch a few hours earlier:
# the tests all constructed the intermediate by hand and never drove the real path.
# These two run `main()` against a REAL git object store built in the fixture.
# ---------------------------------------------------------------------------

def _repo(tmp_path, body: str):
    """A throwaway repo whose HEAD carries ``doc.md``; returns (gitdir, ref)."""
    import subprocess
    wt = tmp_path / "wt"
    wt.mkdir()
    env = {"GIT_CONFIG_GLOBAL": str(tmp_path / "nogit"), "GIT_CONFIG_SYSTEM": str(tmp_path / "nogit"),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ.get("PATH", "")}
    run = lambda *a: subprocess.run(["git", "-C", str(wt), *a], env=env,
                                    capture_output=True, check=True)
    run("init", "-q", "-b", "main")
    (wt / "doc.md").write_text(body, encoding="utf-8", newline="\n")
    run("add", "doc.md")
    run("commit", "-q", "-m", "tip")
    return str(wt / ".git"), "HEAD:doc.md"


def test_MAIN_exits_1_on_a_lossy_rewrite(tmp_path):
    """⭐ The arm that kills `SS4`. Drives the exit code, not the return value."""
    import superset_check as sc
    gitdir, ref = _repo(tmp_path, TIP)
    lossy = tmp_path / "lossy.md"
    lossy.write_text("\n".join(l for l in TIP.splitlines() if "human_pdms" not in l),
                     encoding="utf-8", newline="\n")
    assert sc.main([ref, str(lossy), "", gitdir]) == 1


def test_MAIN_exits_0_on_an_honest_superset(tmp_path):
    """⛔ The other direction, in the same breath: without this the arm above is also
    satisfied by a `main()` that returns 1 unconditionally."""
    import superset_check as sc
    gitdir, ref = _repo(tmp_path, TIP)
    good = tmp_path / "good.md"
    good.write_text(TIP + "\nThe control read `sel_pdms` 0.9161.\n",
                    encoding="utf-8", newline="\n")
    assert sc.main([ref, str(good), "", gitdir]) == 0
