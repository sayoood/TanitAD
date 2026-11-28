"""Tests for refe/hf_egress.py against a fake HF API (no network).

Every failure branch is reached by a fault in the fake, and the listing check itself is MUTATED: a copy of the module
whose remote listing is replaced by the local sizes must then pass the size-mismatch and missing-file arms. If it did
not, the check would not be what makes those arms fail.

Run: `python test_hf_egress.py` (prints ZZTEST_PASS / ZZTEST_FAIL) or pytest.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hf_egress  # noqa: E402

FAKE_TOKEN = "hf_" + "Z" * 12                       # built at runtime: no token-shaped literal in the repo
TARGET_LINE = "        remote = _remote_sizes(api, repo, repo_type)"
REG_REPO = "google/tanitad-egress-regression-must-fail"


def _op(path_in_repo, local):
    return SimpleNamespace(path_in_repo=path_in_repo, path_or_fileobj=str(local))


class FakeApi:
    def __init__(self, deny=(), private=True, drop=False, size_delta=0, used=10**9, list_raises=False,
                 used_none=False):
        self.deny = set(deny)
        self.private, self.drop, self.size_delta = private, drop, size_delta
        self.used, self.list_raises, self.used_none = used, list_raises, used_none
        self.repos: dict[str, dict[str, int]] = {}
        self.calls: list[str] = []

    def whoami(self):
        return {"name": "Sayood"}

    def create_repo(self, repo, private, repo_type, exist_ok):
        self.calls.append("create_repo")
        if "*" in self.deny or repo in self.deny:
            raise PermissionError(f"403 Forbidden: token {FAKE_TOKEN} cannot write {repo}")
        self.repos.setdefault(repo, {})

    def repo_info(self, repo, repo_type, expand=None):
        if expand:                                                  # the live API serves usedStorage only here
            used = None if (self.used_none and repo == "Sayood/private-one") else self.used
            return SimpleNamespace(private=True, used_storage=used)
        return SimpleNamespace(private=self.private)

    def create_commit(self, repo, operations, commit_message, repo_type):
        self.calls.append("create_commit")
        for op in operations:
            self.repos[repo][op.path_in_repo] = os.path.getsize(op.path_or_fileobj)

    def list_repo_tree(self, repo, recursive, repo_type):
        files = dict(self.repos.get(repo, {}))
        if self.drop and files:
            files.pop(sorted(files)[0])
        folders = {p.rsplit("/", 1)[0] for p in files if "/" in p}
        for f in sorted(folders):
            yield SimpleNamespace(path=f)                           # a folder has no size
        for p, s in sorted(files.items()):
            real = s + self.size_delta
            if p.endswith(".pt"):                                   # LFS object: bare size is the pointer's
                yield SimpleNamespace(path=p, size=133, lfs=SimpleNamespace(size=real, pointer_size=133))
            elif p.endswith(".json"):                               # LFS info as a dict
                yield SimpleNamespace(path=p, size=133, lfs={"size": real, "pointer_size": 133})
            else:
                yield SimpleNamespace(path=p, size=real, lfs=None)

    @staticmethod
    def _no_used_storage_in_lists(expand):
        if "usedStorage" in expand:                                 # MEASURED: the list endpoints answer HTTP 400
            raise RuntimeError('400 Bad request: Invalid option: expected one of ... at expand[0]')

    def list_models(self, author, expand):
        self._no_used_storage_in_lists(expand)
        if self.list_raises:
            raise RuntimeError("listing unavailable")
        return [SimpleNamespace(id="Sayood/private-one", private=True),
                SimpleNamespace(id="Sayood/public-one", private=False)]

    def list_datasets(self, author, expand):
        self._no_used_storage_in_lists(expand)
        return []

    def list_spaces(self, author, expand):
        self._no_used_storage_in_lists(expand)
        return []


def _patch(mod):
    mod._add_op = _op
    return mod


def _files(tmp: Path):
    a = tmp / "scorer_states_u400.pt"
    a.write_bytes(b"x" * 1000)
    b = tmp / "sft.log"
    b.write_text("step 1\n", encoding="utf-8")
    c = tmp / "gate.json"
    c.write_text('{"ok": 1}', encoding="utf-8")
    return [a, b, c]


def _push(mod, api, tmp, **kw):
    kw.setdefault("chain_log", tmp / "chain.log")
    kw.setdefault("egress_log", tmp / "egress.jsonl")
    return mod.push(api, "Sayood/tanitad-refe-sft4r", "model", _files(tmp), "u400", prefix="p", **kw)


def test_identity_ok(tmp_path):
    _patch(hf_egress)
    ok, reason = _push(hf_egress, FakeApi(), tmp_path)
    assert ok, reason
    assert "ZZEGRESS_OK u400 3 files" in (tmp_path / "chain.log").read_text(encoding="utf-8")


def test_size_mismatch(tmp_path):
    _patch(hf_egress)
    ok, reason = _push(hf_egress, FakeApi(size_delta=1), tmp_path)
    assert not ok and reason.startswith("size_mismatch:"), reason
    assert "ZZEGRESS_FAIL u400 size_mismatch" in (tmp_path / "chain.log").read_text(encoding="utf-8")


def test_missing_remote(tmp_path):
    _patch(hf_egress)
    ok, reason = _push(hf_egress, FakeApi(drop=True), tmp_path)
    assert not ok and reason.startswith("missing_remote:"), reason


def test_mutant_listing_check(tmp_path):
    src = Path(hf_egress.__file__).read_text(encoding="utf-8")
    assert src.count(TARGET_LINE) == 1, "the mutation target must appear exactly once"
    mpath = tmp_path / "hf_egress_mutant.py"
    mpath.write_text(src.replace(TARGET_LINE, "        remote = dict(sizes)  # MUTANT"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("hf_egress_mutant", mpath)
    mut = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mut)
    _patch(mut)
    for fault in ({"size_delta": 1}, {"drop": True}):
        d = tmp_path / ("m_" + next(iter(fault)))
        d.mkdir()
        ok, reason = _push(mut, FakeApi(**fault), d)
        assert ok, f"the mutant still failed under {fault}: {reason} -- the listing check is not what decides"


def test_unwritable_refused_and_sanitized(tmp_path):
    _patch(hf_egress)
    api = FakeApi(deny={"*"})
    ok, reason = _push(hf_egress, api, tmp_path)
    assert not ok and reason.startswith("exception:PermissionError"), reason
    assert "create_commit" not in api.calls
    for f in ("chain.log", "egress.jsonl"):
        text = (tmp_path / f).read_text(encoding="utf-8")
        assert FAKE_TOKEN not in text and "hf_***" in text, f


def test_not_private(tmp_path):
    _patch(hf_egress)
    api = FakeApi(private=False)
    ok, reason = _push(hf_egress, api, tmp_path)
    assert not ok and reason == "not_private", reason
    assert "create_commit" not in api.calls


def test_over_budget_never_touches_the_server(tmp_path):
    _patch(hf_egress)
    api = FakeApi()
    ok, reason = _push(hf_egress, api, tmp_path, max_push_bytes=100)
    assert not ok and reason.startswith("over_budget:"), reason
    assert api.calls == []


def test_allowance_and_unknown_usage_fail_closed(tmp_path):
    _patch(hf_egress)
    for kw, want in (({"used": 10**12 - 500}, "over_allowance:"), ({"list_raises": True}, "usage_unknown:"),
                     ({"used_none": True}, "usage_unknown:")):
        d = tmp_path / want.rstrip(":")
        d.mkdir(exist_ok=True)
        api = FakeApi(**kw)
        ok, reason = _push(hf_egress, api, d)
        assert not ok and reason.startswith(want), (kw, reason)
        assert "create_repo" not in api.calls, kw


def test_real_add_op_builds_the_commit_operation(tmp_path):
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return                                                     # nothing to check without the library
    spec = importlib.util.spec_from_file_location("hf_egress_fresh", hf_egress.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    f = tmp_path / "x.log"
    f.write_text("abc", encoding="utf-8")
    op = fresh._add_op("logs/u400/x.log", f)
    assert op.path_in_repo == "logs/u400/x.log"


def test_gate(tmp_path):
    _patch(hf_egress)
    rc = hf_egress.gate(FakeApi(deny={REG_REPO}), "Sayood/tanitad-refe-sft4r", "model", tmp_path / "g1",
                        chain_log=tmp_path / "chain.log")
    res = json.loads((tmp_path / "g1" / "g_egr_result.json").read_text(encoding="utf-8"))
    assert rc == 0 and res["pass"] and res["regression"]["refused_by_server"], res
    assert "ZZEGR_PASS" in (tmp_path / "chain.log").read_text(encoding="utf-8")
    # regression arm: a token that can write the "must fail" repo proves nothing -> RED
    rc2 = hf_egress.gate(FakeApi(), "Sayood/tanitad-refe-sft4r", "model", tmp_path / "g2")
    assert rc2 == 1
    # a refusal by our own pre-checks is not a server refusal -> RED
    rc3 = hf_egress.gate(FakeApi(deny={REG_REPO}, private=False), "Sayood/tanitad-refe-sft4r", "model",
                         tmp_path / "g3")
    assert rc3 == 1


class _Rig:
    """A trainer stand-in: writes `scorer_states_last.pt` (JSON here) with controlled stats."""

    def __init__(self, tmp: Path, push_ok=True):
        self.out = tmp / "out"
        self.out.mkdir()
        self.sft = self.out / "sft.log"
        self.sft.write_text("start\n", encoding="utf-8")
        self.chain = tmp / "chain.log"
        self.chain.write_text("armed\n", encoding="utf-8")
        self.pushes: list[tuple[str, list[str]]] = []
        self.killed: list[int] = []
        self.alive = True
        self.push_ok = push_ok
        self.t = 1_700_000_000_000_000_000

    def write(self, upd):
        p = self.out / "scorer_states_last.pt"
        p.write_text(json.dumps({"update": upd}), encoding="utf-8")
        self.t += 10**9
        os.utime(p, ns=(self.t, self.t))

    def push_fn(self, files, names, tag):
        for f in files:
            assert Path(f).is_file(), f
        self.pushes.append((tag, list(names)))
        return (True, "ok") if self.push_ok else (False, "exception:Fake:down")

    def watcher(self, tmp):
        return hf_egress.Watcher(self.out, tmp / "stage", [self.sft, self.chain], self.push_fn, halt_pid=4242,
                                 poll=0, chain_log=self.chain,
                                 read_update=lambda p: json.loads(Path(p).read_text(encoding="utf-8"))["update"],
                                 pid_alive=lambda pid: self.alive, kill=self.killed.append, sleep=lambda s: None)


def test_watch_pushes_each_checkpoint_once(tmp_path):
    rig = _Rig(tmp_path)
    w = rig.watcher(tmp_path)
    rig.write(400)
    assert w.poll_once() is None                 # first sighting: not settled yet
    assert w.poll_once() == 400                  # settled -> pushed
    assert w.poll_once() is None and w.poll_once() is None   # never re-pushed
    assert [t for t, _ in rig.pushes] == ["u400"]
    assert "scorer_states/scorer_states_u400.pt" in rig.pushes[0][1]
    assert "logs/u400/sft.log" in rig.pushes[0][1] and "logs/u400/chain.log" in rig.pushes[0][1]
    rig.write(800)
    w.poll_once()
    assert w.poll_once() == 800
    assert [t for t, _ in rig.pushes] == ["u400", "u800"]
    assert rig.pushes[1][1].count("scorer_states/scorer_states_u800.pt") == 1
    assert not any("u400.pt" in n for n in rig.pushes[1][1])
    # final update: picked up WITHOUT the settle, then the exit push
    rig.write(1200)
    with open(rig.sft, "a", encoding="utf-8") as fh:
        fh.write("ZZSFT_DONE\n")
    assert w.run() == 0
    tag, names = rig.pushes[-1]
    assert tag == "exit" and "scorer_states/scorer_states_u1200.pt" in names and "logs/exit/sft.log" in names
    assert rig.killed == []


def test_watch_halts_after_the_checkpoint_following_two_failures(tmp_path):
    rig = _Rig(tmp_path, push_ok=False)
    w = rig.watcher(tmp_path)
    for upd in (400, 800):
        rig.write(upd)
        w.poll_once()
        assert w.poll_once() == upd
        assert rig.killed == [], f"killed at u{upd}: too early"
    assert w.halt_armed
    rig.write(1200)
    w.poll_once()
    assert w.poll_once() == 1200
    assert rig.killed == [4242]                  # once, after the u1200 push attempt
    rig.write(1600)
    w.poll_once()
    w.poll_once()
    assert rig.killed == [4242]                  # never twice
    assert "ZZEGRESS_HALT u1200" in rig.chain.read_text(encoding="utf-8")
    rig.alive = False
    assert w.run() == 1                          # the exit push still fails
    tag, names = rig.pushes[-1]
    assert tag == "exit"
    for u in (400, 800, 1200, 1600):             # every unpushed state rides the final push
        assert f"scorer_states/scorer_states_u{u}.pt" in names, u


def test_retry_carries_pending_states(tmp_path):
    rig = _Rig(tmp_path, push_ok=False)
    w = rig.watcher(tmp_path)
    rig.write(400)
    w.poll_once()
    w.poll_once()
    rig.push_ok = True
    rig.write(800)
    w.poll_once()
    w.poll_once()
    names = rig.pushes[-1][1]
    assert "scorer_states/scorer_states_u400.pt" in names and "scorer_states/scorer_states_u800.pt" in names
    assert w.fails == 0 and not w.halt_armed and w.pending == []


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"ZZTEST_PASS {len(tests)}" if fails == 0 else f"ZZTEST_FAIL {fails}/{len(tests)}")
    sys.exit(1 if fails else 0)
