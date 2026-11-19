p="HANDOFF.md"
s=open(p,encoding="utf-8").read()
for a in ("no test has a window with some","the end to end tests use ticks","the head-source test does not separate","no test has a present tick","`test_cli_refusals` does not cover"):
    assert a in s, a
s=s.replace("no test has a window with some","probably no test has a window with some").replace("the end to end tests use ticks","probably the end to end tests use ticks").replace("the head-source test does not separate","probably the head-source test does not separate").replace("no test has a present tick","probably no test has a present tick").replace("`test_cli_refusals` does not cover","`test_cli_refusals` apparently does not cover")
s=s.replace("## STILL-GREEN mutations (what each means)","## STILL-GREEN mutations (what each means; the reasons are readings of the tests, not separately proven)")
open(p,"w",encoding="utf-8").write(s)
