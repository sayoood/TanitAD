"""Tests of ``v8f_speedsub_t1_pin.py`` -- the pin step, the argv renderer and the resume records of the SPEEDSUB-T1 chain (PREREG_V8F_SPEEDSUB_T1 section 9).

The argv renderer is the ONE place where a rollout's command line is made, so these tests read the four BANKED argv files (E:/Projects/TanitAD-artifacts, read-only) and require that a rendered argv differs
from its banked original in the registered tokens ONLY. Controls that must read as stated:
  * RENDER    for every (run, mode, smoke): only --tag / --out-dir / --dump-ticks (+ --limit on the logged lane, + --max-windows on the smoke) differ from the banked tokens.
  * DRIFT     a banked argv with ONE registered value changed (replan steps, checkpoint, nav, limit, device, missing --dump-heads, not starting with `run`) is REFUSED; the unchanged copy renders.
  * LANE      the beside logged lane exists for CTRL only.
  * DONE      a done record is accepted only for its own tag, checkpoint sha (before AND after) and the pinned tool digest; `write-done` writes none unless every file is there and the checkpoint is unchanged.
  * PIN       a checkpoint whose sha256 does not start with the PREREG's printed prefix is REFUSED; an existing pins.json is never overwritten by different pins.

Run: python -m pytest -q test_v8f_speedsub_t1_pin.py     (SPEEDSUB_PIN=<mutant.py> points the whole file at a mutated copy; see mutation_speedsub_pin.py)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PIN = os.environ.get("SPEEDSUB_PIN") or str(HERE.parent / "scripts" / "v8f_speedsub_t1_pin.py")
spec = importlib.util.spec_from_file_location("speedsub_pin_under_test", PIN)
P = importlib.util.module_from_spec(spec)
sys.modules["speedsub_pin_under_test"] = P
spec.loader.exec_module(P)

HAVE_BANKED = all(os.path.isfile(p) for p in P.BANKED_ARGV.values())
needs_banked = pytest.mark.skipif(not HAVE_BANKED, reason="the four banked argv files are not on this machine")
RUN_MODES = [(r, m) for r in sorted(P.CKPT) for m in P.MODES if not (m in P.LOGGED and not r.startswith("CTRL"))]


def parse(tokens):
    d, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--") and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            d[t] = tokens[i + 1]
            i += 2
        else:
            d[t] = None
            i += 1
    return d


# ------------------------------------------------------------------------------------------------ RENDER
@needs_banked
@pytest.mark.parametrize("run,mode", RUN_MODES)
@pytest.mark.parametrize("smoke", [0, 200])
def test_a_rendered_argv_differs_from_the_banked_one_in_the_registered_tokens_only(run, mode, smoke):
    got = P.render_argv(run, mode, "X:/out", smoke)
    banked = P.banked_tokens(run)[2:]
    assert got[0] == "run" and len(got) == len(banked) + (2 if smoke else 0)
    g, b = parse(got), parse(banked)
    allowed = {"--tag", "--out-dir", "--dump-ticks"} | ({"--limit"} if mode in P.LOGGED else set())
    changed = {k for k in set(g) | set(b) if g.get(k) != b.get(k)}
    assert changed == allowed | ({"--max-windows"} if smoke else set()), changed
    d = f"X:/out/{'SMOKE/' if smoke else ''}{run}/{mode}"
    tag = f"SST1_{'SMOKE_' if smoke else ''}{run}_{mode}"
    assert (g["--tag"], g["--out-dir"], g["--dump-ticks"]) == (tag, d, d + "/ticks") and tag == P.tag_of(run, mode, smoke)
    assert g["--limit"] == ("logged" if mode in P.LOGGED else "causal") and g["--ckpt"] == P.CKPT[run]
    assert (g.get("--max-windows") == "200") is bool(smoke)


@needs_banked
def test_the_tags_and_directories_never_collide():
    seen = set()
    for r, m in RUN_MODES:
        for sm in (0, 200):
            a = P.render_argv(r, m, "X:/out", sm)
            key = (a[a.index("--tag") + 1], a[a.index("--out-dir") + 1])
            assert key not in seen
            seen.add(key)
    assert len({k[0] for k in seen}) == len(seen) == len({k[1] for k in seen})


# ------------------------------------------------------------------------------------------------ DRIFT / LANE
@needs_banked
@pytest.mark.parametrize("what", ["replan", "ckpt", "nav", "limit", "device", "no_dump_heads", "not_run", "scene", "schedule", "tracks"])
def test_a_banked_argv_that_drifted_is_refused(tmp_path, monkeypatch, what):
    toks = list(P.banked_tokens("CTRL_s0"))

    def setv(flag, val):
        toks[toks.index(flag) + 1] = val

    # the unchanged copy renders: the refusals below are not an artefact of the copy
    ok = tmp_path / "ok.argv"
    ok.write_text("\n".join(toks) + "\n")
    monkeypatch.setitem(P.BANKED_ARGV, "CTRL_s0", str(ok))
    assert P.render_argv("CTRL_s0", "egoc", "X:/o")[0] == "run"
    if what == "replan":
        setv("--replan-steps", "5")
    elif what == "ckpt":
        setv("--ckpt", P.CKPT["CTRL_s1"])
    elif what == "nav":
        setv("--nav", "gt")
    elif what == "limit":
        setv("--limit", "logged")
    elif what == "device":
        setv("--device", "cuda")
    elif what == "no_dump_heads":
        toks.remove("--dump-heads")
    elif what == "not_run":
        toks[toks.index("run")] = "analyze"
    elif what == "scene":
        setv("--scene", "rendered")
    elif what == "schedule":
        setv("--schedule", "event")
    elif what == "tracks":
        setv("--tracks", "E:/other/native_tracks.pt")
    bad = tmp_path / "bad.argv"
    bad.write_text("\n".join(toks) + "\n")
    monkeypatch.setitem(P.BANKED_ARGV, "CTRL_s0", str(bad))
    with pytest.raises(SystemExit) as e:
        P.render_argv("CTRL_s0", "egoc", "X:/o")
    assert "REFUSED" in str(e.value)


@needs_banked
def test_the_beside_logged_lane_is_ctrl_only():
    for r in ("REP_s0", "REP_s1"):
        for m in P.LOGGED:
            with pytest.raises(SystemExit):
                P.render_argv(r, m, "X:/o")
    a = P.render_argv("CTRL_s1", "egol", "X:/o")
    assert a[a.index("--limit") + 1] == "logged"
    with pytest.raises(SystemExit):
        P.render_argv("CTRL_s0", "nonsense", "X:/o")


# ------------------------------------------------------------------------------------------------ DONE
def done_rec(tag="T", sha="a" * 64, **over):
    d = {"tag": tag, "ckpt_sha256": sha, "ckpt_sha256_before": sha, "ckpt_sha256_after": sha, "tool_digest": P.TOOL_DIGEST}
    d.update(over)
    return d


def test_done_ok_accepts_only_its_own_tag_checkpoint_and_tool(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text(json.dumps(done_rec()))
    assert P.done_ok(str(f), "T", "a" * 64) is True
    assert P.done_ok(str(f), "U", "a" * 64) is False and P.done_ok(str(f), "T", "b" * 64) is False
    for k, v in (("ckpt_sha256", "b" * 64), ("ckpt_sha256_before", "b" * 64), ("ckpt_sha256_after", "b" * 64), ("tool_digest", "0" * 64), ("tag", "U")):
        f.write_text(json.dumps(done_rec(**{k: v})))
        assert P.done_ok(str(f), "T", "a" * 64) is False, k
    f.write_text("{not json")
    assert P.done_ok(str(f), "T", "a" * 64) is False
    assert P.done_ok(str(tmp_path / "missing.txt"), "T", "a" * 64) is False


def world(tmp_path, monkeypatch, run="CTRL_s0", mode="egoc", smoke=0, n_dumps=141):
    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"checkpoint-bytes")
    monkeypatch.setitem(P.CKPT, run, str(ck).replace("\\", "/"))
    root = str(tmp_path / "out").replace("\\", "/")
    tag, d = P.tag_of(run, mode, smoke), P.out_dir_of(root, run, mode, smoke)
    os.makedirs(f"{d}/ticks")
    os.makedirs(f"{d}/dump_{tag}")
    for p in (f"{d}/t1v8f_{tag}.json", f"{d}/ticks/ticks_{tag}.npz", f"{d}/ticks/speedsub_{tag}.npz"):
        open(p, "wb").close()
    for i in range(n_dumps):
        open(f"{d}/dump_{tag}/ep{i:04d}.npz", "wb").close()
    return {"root": root, "tag": tag, "dir": d, "ck": ck, "sha": P.sha256_file(str(ck)), "run": run, "mode": mode, "smoke": smoke}


def write_done_args(w, sha=None, wall="300"):
    a = ["write-done", "--run", w["run"], "--mode", w["mode"], "--out-root", w["root"], "--sha-before", sha or w["sha"], "--wall-s", wall]
    return a + (["--smoke", str(w["smoke"])] if w["smoke"] else [])


def test_write_done_writes_the_banked_shape_and_check_done_accepts_it(tmp_path, monkeypatch, capsys):
    w = world(tmp_path, monkeypatch)
    assert P.main(write_done_args(w)) == 0
    f = f"{w['dir']}/done_{w['tag']}.txt"
    rec = json.load(open(f, encoding="utf-8"))
    assert rec["tag"] == w["tag"] and rec["limit"] == "causal" and rec["n_dumps"] == 141 and rec["smoke"] is False and rec["wall_s"] == 300 and rec["tool_digest"] == P.TOOL_DIGEST
    assert rec["ckpt_sha256"] == rec["ckpt_sha256_before"] == rec["ckpt_sha256_after"] == w["sha"]
    assert P.main(["check-done", "--done", f, "--tag", w["tag"], "--ckpt-sha", w["sha"]]) == 0
    assert P.main(["check-done", "--done", f, "--tag", w["tag"], "--ckpt-sha", "0" * 64]) == 1
    assert P.main(["check-done", "--done", f, "--tag", "SST1_other", "--ckpt-sha", w["sha"]]) == 1
    assert "ZZSPEEDSUB-T1-DONE-WRITTEN" in capsys.readouterr().out


def test_write_done_is_refused_unless_everything_is_there(tmp_path, monkeypatch, capsys):
    for name in ("dump", "json", "ticks", "rec", "ckpt"):
        sub = tmp_path / name
        sub.mkdir()
        w = world(sub, monkeypatch)
        if name == "dump":
            os.remove(f"{w['dir']}/dump_{w['tag']}/ep0070.npz")
        elif name == "json":
            os.remove(f"{w['dir']}/t1v8f_{w['tag']}.json")
        elif name == "ticks":
            os.remove(f"{w['dir']}/ticks/ticks_{w['tag']}.npz")
        elif name == "rec":
            os.remove(f"{w['dir']}/ticks/speedsub_{w['tag']}.npz")
        elif name == "ckpt":
            w["ck"].write_bytes(b"changed during the rollout")
        assert P.main(write_done_args(w)) == 1, name
        assert not os.path.exists(f"{w['dir']}/done_{w['tag']}.txt"), name
        assert "ZZSPEEDSUB-T1-DONE-STOP" in capsys.readouterr().out
    sub = tmp_path / "wrongsha"
    sub.mkdir()
    w = world(sub, monkeypatch)
    assert P.main(write_done_args(w, sha="0" * 64)) == 1


def test_write_done_for_the_smoke_and_the_logged_lane(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    w = world(a, monkeypatch, smoke=200, n_dumps=3)
    assert P.main(write_done_args(w)) == 0
    rec = json.load(open(f"{w['dir']}/done_{w['tag']}.txt", encoding="utf-8"))
    assert rec["smoke"] is True and rec["n_dumps"] == 3 and w["tag"].startswith("SST1_SMOKE_")
    b = tmp_path / "b"
    b.mkdir()
    w = world(b, monkeypatch, smoke=200, n_dumps=0)
    assert P.main(write_done_args(w)) == 1                                                         # a smoke that dumped nothing is not done
    c = tmp_path / "c"
    c.mkdir()
    w = world(c, monkeypatch, mode="egol")
    assert P.main(write_done_args(w)) == 0
    assert json.load(open(f"{w['dir']}/done_{w['tag']}.txt", encoding="utf-8"))["limit"] == "logged"


def test_get_prints_the_value_at_a_dotted_path(tmp_path, capsys):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"ckpts": {"CTRL_s0": {"sha256": "abc"}}, "tool_sha256": "def"}))
    assert P.main(["get", "--pins", str(f), "--key", "ckpts.CTRL_s0.sha256"]) == 0
    assert capsys.readouterr().out.strip() == "abc"
    P.main(["get", "--pins", str(f), "--key", "tool_sha256"])
    assert capsys.readouterr().out.strip() == "def"
    with pytest.raises(KeyError):
        P.main(["get", "--pins", str(f), "--key", "ckpts.REP_s0.sha256"])


# ------------------------------------------------------------------------------------------------ PIN
def full_pins(tag="x"):
    p = {"schema": "v8f-speedsub-t1-pins-v1", "ckpts": {r: {"path": f"/c/{r}", "sha256": tag * 64} for r in P.CKPT}, "readout_c": {"path": "/r/c", "sha256": tag * 64},
         "readout_banked": {"path": "/r/b", "sha256": tag * 64}}
    for k in ("tool", "wrapper", "instrument", "fit_script", "pin_script", "reader"):
        p[f"{k}_sha256"], p[f"{k}_path"] = tag * 64, f"/x/{k}"
    return p


def test_an_existing_pins_file_is_never_overwritten_by_different_pins(tmp_path, monkeypatch, capsys):
    pins_path = str(tmp_path / "pins.json")
    a = type("A", (), {"pins": pins_path})()
    monkeypatch.setattr(P, "build_pins", lambda a: full_pins("x"))
    assert P.cmd_pin(a) == 0 and os.path.isfile(pins_path) and os.path.isfile(str(tmp_path / "pins.txt"))
    first = open(pins_path, encoding="utf-8").read()
    assert P.cmd_pin(a) == 0 and "unchanged" in capsys.readouterr().out
    monkeypatch.setattr(P, "build_pins", lambda a: full_pins("y"))
    assert P.cmd_pin(a) == 2 and open(pins_path, encoding="utf-8").read() == first
    assert "ZZSPEEDSUB-T1-PIN-STOP" in capsys.readouterr().out


def test_a_checkpoint_off_the_prereg_prefix_is_refused(tmp_path, monkeypatch):
    ck = tmp_path / "ck.pt"
    ck.write_bytes(b"not the checkpoint")
    first = sorted(P.CKPT)[0]
    monkeypatch.setitem(P.CKPT, first, str(ck).replace("\\", "/"))
    a = type("A", (), {"code_dir": str(tmp_path), "reader": str(ck), "fit_dir": str(tmp_path)})()
    with pytest.raises(SystemExit) as e:
        P.build_pins(a)
    assert "REFUSED" in str(e.value) and P.CKPT_PREFIX[first] in str(e.value)


def test_the_registered_constants_are_the_preregs():
    assert P.CKPT_PREFIX == {"CTRL_s0": "67265ec1", "CTRL_s1": "90b01474", "REP_s0": "b03fd6ef", "REP_s1": "77853e56"}
    assert P.TOOL_SHA256 == "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397" and P.REG_PERM_SHA == "64785f8ad0797b02bdf9845cedf6cf425f8759a2993da889cf33146a0e37ab64"
    assert P.EXPECT["--replan-steps"] == "10" and P.EXPECT["--nav"] == "deadreckon" and P.EXPECT["--scene"] == "logged" and P.EXPECT["--schedule"] == "periodic"
    assert P.EXPECT["--limit"] == "causal" and P.EXPECT["--device"] == "cpu" and P.MODES == ("identity", "egoc", "egoshufc", "identity_l", "egol") and P.N_EPISODES == 141
    assert P.BANKED_READOUT_SHA256.startswith("a205609a")
