"""OPTION B (H-NT-DECIDE-1, 2026-10-22): G-BOX-OVERFIT's per-profile FROZEN frame sets.

* ``refcv8`` keeps its frame set, its md5 and its C3 literals BYTE-FOR-BYTE (values written here as literals);
* ``refcv8-nt`` reads the NT frame set the SAME frozen selector chose labels-only on the NT TRAIN token-episodes
  (``stack/ops/refcv8nt_box_audit/raw``), pinned here by md5 and counts, and re-judged by the launch gate;
* MUTATION ARMS: the NT profile handed refcv8's frame set is REFUSED by the harness (C3, and the CLI cannot read a
  file it was not given) and by the gate (frame-set md5, profile, addendum, n_pos / n_ign literals, C3 totals);
* the harness's NT data-path repairs (``--window-starts first`` at dataset construction; the all-UNKNOWN speed sidecar
  attached the way ``train()`` attaches it) run on the REAL ``TrainerAdapter._build_items`` against a fake loader, and
  are inert for a refcv8-shaped argv.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import refcv8_assemble_argv as A  # noqa: E402


def _gbo():
    spec = importlib.util.spec_from_file_location("g_box_overfit_nt_t", str(ROOT / "scripts" / "g_box_overfit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _gbo()

# ---- the LITERALS, written independently of the code under test -------------------------------------------------- #
R8_FS_MD5 = "b291404c36f83c3e397e3b90367e8e7b"
NT_FS_MD5 = "bb60278ce8d763cc5151cbcc1cb32735"
NT_ADD_MD5 = "a9515574d356a359e08fc39dbac46d71"
NT_INPUT_MD5 = "84eafdd79f84ff4d022c4fb69b06f6e3"
PREREG_MD5 = "594c71196cc5bbd527fee40b2cb0e3f1"
SELECTOR_MD5 = "f9f93f92b60a979b581b8f78be2ec6db"
REFCV8 = {"file": "gbo_frameset.json", "md5": R8_FS_MD5, "n_frames": 16, "n_pos": 113, "n_ign": 77,
          "addendum": None, "addendum_md5": None}
NT = {"file": "gbo_frameset_nt.json", "md5": NT_FS_MD5, "n_frames": 16, "n_pos": 122, "n_ign": 108,
      "addendum": "PREREG_G_BOX_OVERFIT_ADDENDUM_NT.md", "addendum_md5": NT_ADD_MD5}
NT_RAW = ROOT / "ops" / "refcv8nt_box_audit" / "raw"
SELECTOR = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-09-26-box-head-audit"
            / "code" / "gbo_select.py")


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------- refcv8: unchanged -------- #
def test_refcv8_literals_are_unchanged_in_the_harness_and_the_gate():
    assert G.FRAMESETS["refcv8"] == REFCV8
    assert (G.FRAMESET_MD5, G.N_FRAMES, G.N_POS, G.N_IGN) == (R8_FS_MD5, 16, 113, 77)
    assert (G.PREREG_MD5, G.SELECTOR_MD5) == (PREREG_MD5, SELECTOR_MD5)
    bo = LG.PROFILES["refcv8"]["box_overfit"]
    assert (bo["prereg_md5"], bo["frameset_md5"], bo["n_frames"], bo["n_pos"], bo["n_ignore"]) == (
        PREREG_MD5, R8_FS_MD5, 16, 113, 77)
    assert "frameset_profile" not in bo and "addendum_md5" not in bo       # refcv8's judge path takes no new branch
    # the DEFAULTS still count against 113: criterion 3's window and its text, C1, C3
    ok = {"ap2m": 0.95, "prec": 0.95, "rec": 0.95, "n_conf": 124.0, "centre_p50_m": 0.1, "size_p50_m": 0.1,
          "z_p50_m": 0.05, "cls_acc": 0.95}
    c = G.criteria(ok, presence_step0=1.0, presence_end=0.2, finite=True)["3"]
    assert c["pass"] and c["what"] == "|sum confident - 113| / 113 <= 0.10"
    assert not G.criteria(dict(ok, n_conf=125.0), presence_step0=1.0, presence_end=0.2, finite=True)["3"]["pass"]
    fs = {"frames": [{"sha12": f"{i:012x}", "t": 21, "positive": 7, "ignore": 1, "dropped": 3} for i in range(15)]
          + [{"sha12": "f" * 12, "t": 64, "positive": 8, "ignore": 16, "dropped": 1}]}
    per = [(f["sha12"], f["t"], f["positive"], f["ignore"] + f["dropped"]) for f in fs["frames"]]
    c3 = G.control_c3(per, fs, R8_FS_MD5)
    assert c3["pass"] and c3["want_totals"] == [113, 77]
    # refcv8's binding template and kit never name a frame-set profile (the harness default is refcv8)
    tpl = (ROOT / "scripts" / "refcv8_bind_box.args.tpl").read_text(encoding="utf-8")
    assert "--frameset-profile" not in tpl
    kit = (ROOT / "scripts" / "refcv8_prelaunch.sh").read_text(encoding="utf-8")
    assert f'echo "{R8_FS_MD5}  $BAUD/visibility/gbo_frameset.json"' in kit and "refcv8nt" not in kit


# ------------------------------------------------------------------------------------- refcv8-nt: pinned -------- #
def test_nt_literals_are_pinned_and_the_in_tree_frame_set_reads_them():
    assert G.FRAMESETS["refcv8-nt"] == NT
    bo = LG.PROFILES["refcv8-nt"]["box_overfit"]
    assert (bo["frameset_profile"], bo["frameset_md5"], bo["addendum_md5"]) == ("refcv8-nt", NT_FS_MD5, NT_ADD_MD5)
    assert (bo["n_frames"], bo["n_pos"], bo["n_ignore"]) == (16, 122, 108)
    r8 = LG.PROFILES["refcv8"]["box_overfit"]               # every OTHER literal (bars, arms, optimiser, schedule) is refcv8's
    assert {k: v for k, v in bo.items() if k in r8 and k not in ("frameset_md5", "n_pos", "n_ignore")} == {
        k: v for k, v in r8.items() if k not in ("frameset_md5", "n_pos", "n_ignore")}
    fsb = (NT_RAW / "visibility" / "gbo_frameset_nt.json").read_bytes()
    assert _md5(fsb) == NT_FS_MD5
    fs = json.loads(fsb.decode("utf-8"))
    assert len(fs["frames"]) == 16 and {f["t"] for f in fs["frames"]} == {0}          # --window-starts first
    assert sum(f["positive"] for f in fs["frames"]) == 122
    assert sum(f["ignore"] + f["dropped"] for f in fs["frames"]) == 108                # A10: DROPPED is IGNORE
    assert (sum(f["ignore"] for f in fs["frames"]), sum(f["dropped"] for f in fs["frames"])) == (58, 50)
    assert fs["source_md5"] == NT_INPUT_MD5 == _md5((NT_RAW / "visibility" / "vis_train_boxes_nt.json").read_bytes())
    assert all(len(f["sha12"]) == 12 for f in fs["frames"])
    add = (NT_RAW / NT["addendum"]).read_bytes()
    assert _md5(add) == NT_ADD_MD5
    text = add.decode("utf-8")
    for lit in (NT_FS_MD5, SELECTOR_MD5, PREREG_MD5, NT_INPUT_MD5, "**122", "108 (58 IGNORE + 50 DROPPED)"):
        assert lit in text, lit


def test_the_frozen_selector_reproduces_the_nt_frame_set_from_the_banked_input(tmp_path):
    """An INDEPENDENT re-derivation: the frozen selector (where the tree carries it), run on the banked input,
    must print the registered md5. A stack/-only tree has no Research Lab package and skips.
    ⚠️ The selector stamps ``"source": src.split("/")[-1]``, so the input is passed as a POSIX path: a Windows
    path keeps its whole spelling in ``source`` and the md5 moves (MEASURED 2026-10-22: d98a1ab6... vs bb60278c...)."""
    if not SELECTOR.is_file():
        pytest.skip("stack/-only tree: the box-head audit's selector is not present")
    assert _md5(SELECTOR.read_bytes()) == SELECTOR_MD5
    out = tmp_path / "fs.json"
    r = subprocess.run([sys.executable, SELECTOR.as_posix(), (NT_RAW / "visibility" / "vis_train_boxes_nt.json")
                        .as_posix(), str(out)], capture_output=True, text=True, cwd=str(SELECTOR.parent),
                       env={**__import__("os").environ, "PYTHONPATH": str(SELECTOR.parent),
                            "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip().splitlines()[-1] == f"md5 {NT_FS_MD5}"


def test_nt_count_bar_and_c1_use_122():
    ok = {"ap2m": 0.95, "prec": 0.95, "rec": 0.95, "centre_p50_m": 0.1, "size_p50_m": 0.1, "z_p50_m": 0.05,
          "cls_acc": 0.95}
    crit = lambda n: G.criteria(dict(ok, n_conf=n), presence_step0=1.0, presence_end=0.2, finite=True, n_pos=122)["3"]
    assert crit(110.0)["pass"] and crit(134.0)["pass"]                   # |12| / 122 = 0.098
    assert not crit(109.0)["pass"] and not crit(135.0)["pass"]           # |13| / 122 = 0.107
    assert crit(122.0)["what"] == "|sum confident - 122| / 122 <= 0.10"


# ------------------------------------------------------------------------------------- MUTATION ARMS ------------- #
def test_mutation_harness_nt_profile_handed_the_refcv8_frame_set_is_refused(tmp_path, monkeypatch):
    # (a) C3 against the NT literals on refcv8's frame set: wrong md5 AND wrong totals
    fs8 = {"frames": [{"sha12": f"{i:012x}", "t": 21, "positive": 7, "ignore": 1, "dropped": 3} for i in range(15)]
           + [{"sha12": "f" * 12, "t": 64, "positive": 8, "ignore": 16, "dropped": 1}]}
    per8 = [(f["sha12"], f["t"], f["positive"], f["ignore"] + f["dropped"]) for f in fs8["frames"]]
    c3 = G.control_c3(per8, fs8, R8_FS_MD5, G.FRAMESETS["refcv8-nt"])
    assert not c3["pass"] and not c3["frameset_md5_ok"] and c3["want_totals"] == [122, 108]
    # (b) the CLI: an audit dir holding refcv8's frame set (and the prereg) cannot be run under the NT profile -- the
    # harness reads the NT file BY NAME before anything is built
    raw = tmp_path / "audit" / "raw"
    (raw / "visibility").mkdir(parents=True)
    (raw / "PREREG_G_BOX_OVERFIT.md").write_text("prereg", encoding="utf-8")
    (raw / "visibility" / "gbo_frameset.json").write_text(json.dumps(fs8), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="gbo_frameset_nt.json"):
        G.main(["--launch-argv", str(tmp_path / "argv.json"), "--audit-dir", str(tmp_path / "audit"),
                "--out", str(tmp_path / "o.json"), "--candidate", "t", "--frameset-profile", "refcv8-nt"])
    # (c) ... and with the NT frame set but WITHOUT its addendum, also refused
    (raw / "visibility" / "gbo_frameset_nt.json").write_bytes((NT_RAW / "visibility" / "gbo_frameset_nt.json")
                                                              .read_bytes())
    with pytest.raises(FileNotFoundError, match="PREREG_G_BOX_OVERFIT_ADDENDUM_NT.md"):
        G.main(["--launch-argv", str(tmp_path / "argv.json"), "--audit-dir", str(tmp_path / "audit"),
                "--out", str(tmp_path / "o.json"), "--candidate", "t", "--frameset-profile", "refcv8-nt"])


_SCHED = {"rule": "A17", "hold_peak_through_step": 1799, "decay": "cosine", "decay_from_step": 1800,
          "decay_to_step": 2000, "final_factor": 0.0}
_OPT = {"class": "AdamW", "clip": 10.0,
        "groups": [{"name": None, "lr": 5e-05, "weight_decay": 1e-04}, {"name": None, "lr": 1e-04, "weight_decay": 1e-04}]}
_MAIN = {"ap2m": 0.95, "prec": 0.93, "rec": 0.94, "n_conf": 117, "centre_p50_m": 0.21, "size_p50_m": 0.18,
         "z_p50_m": 0.07, "cls_acc": 0.97, "presence": 0.12}


def _arm(final, verdict="PASS"):
    r0 = {"step": 0, "ap2m": 0.01, "prec": 0.0, "rec": 0.0, "n_conf": 0, "centre_p50_m": 5.0, "size_p50_m": 2.0,
          "z_p50_m": 1.0, "cls_acc": 0.1, "presence": 1.0}
    return {"rows": [r0, {"step": 2000, **final}], "verdict": verdict,
            "criteria": {"6": {"value": [True, 1.0, final["presence"], final["presence"]]}},
            "optimizer": json.loads(json.dumps(_OPT)), "lr_schedule": dict(_SCHED)}


def _record(*, fs_md5, profile, add_md5, n_pos, n_ign, c3_totals, c3_md5, main=None):
    return {"tool": "g_box_overfit.py", "binding": True, "commit": "ab" * 20, "RESULT": "PASS",
            "prereg_md5": PREREG_MD5, "frameset_md5": fs_md5, "frameset_profile": profile, "addendum_md5": add_md5,
            "steps": 2000, "literals": {"batch": 4, "seed": 0, "n_pos": n_pos, "n_ign": n_ign, "clip": 10.0,
                                        "lr_schedule": dict(_SCHED)},
            "launch_argv_sha256": "", "dvb_mismatches": [],
            "C1_C2": {"C1": {"pass": True}, "C2": {"pass": True}},
            "C3": {"pass": True, "totals": list(c3_totals), "frameset_md5": c3_md5}, "C4_step0": {"step": 0},
            "arms": {"main": _arm(main or _MAIN),
                     "memory_zeros": _arm(dict(_MAIN, ap2m=0.12), "failed as required"),
                     "presence_w0": _arm(dict(_MAIN, prec=0.31, rec=0.52, n_conf=0), "failed as required")}}


def _nt_rec(**kw):
    base = dict(fs_md5=NT_FS_MD5, profile="refcv8-nt", add_md5=NT_ADD_MD5, n_pos=122, n_ign=108,
                c3_totals=(122, 108), c3_md5=NT_FS_MD5)
    base.update(kw)
    return _record(**base)


def _r8_rec(**kw):
    base = dict(fs_md5=R8_FS_MD5, profile="refcv8", add_md5=None, n_pos=113, n_ign=77, c3_totals=(113, 77),
                c3_md5=R8_FS_MD5)
    base.update(kw)
    return _record(**base)


def _judge(prof_name, rec, tmp_path):
    f = tmp_path / f"rec_{prof_name}.json"
    f.write_text(json.dumps(rec), encoding="utf-8")
    return LG.judge_box_overfit(LG.PROFILES[prof_name], str(f), "ab" * 20)[0]


def test_gate_accepts_each_profile_on_its_own_frame_set(tmp_path):
    assert _judge("refcv8-nt", _nt_rec(), tmp_path) == []
    assert _judge("refcv8", _r8_rec(), tmp_path) == []


def test_mutation_gate_nt_profile_handed_the_refcv8_frame_set_is_refused(tmp_path):
    reasons = _judge("refcv8-nt", _r8_rec(), tmp_path)
    tag = "G-BOX-OVERFIT: "
    want = (f"{tag}frameset_md5 '{R8_FS_MD5}' != the registered '{NT_FS_MD5}'",
            f"{tag}frameset_profile 'refcv8' != the registered 'refcv8-nt'",
            f"{tag}addendum_md5 None != the registered '{NT_ADD_MD5}'",
            f"{tag}literal n_pos = 113",
            f"{tag}literal n_ign = 77",
            f"{tag}C3 recount totals [113, 77]")
    for w in want:
        assert any(r.startswith(w) for r in reasons), (w, reasons)
    # and the converse: refcv8's profile refuses the NT record (its own literals are unmoved)
    r8 = _judge("refcv8", _nt_rec(), tmp_path)
    assert any(r.startswith(f"{tag}frameset_md5 '{NT_FS_MD5}'") for r in r8)
    assert any(r.startswith(f"{tag}literal n_pos = 122") for r in r8)


def test_mutation_gate_rejudges_c3_totals_not_the_c3_flag(tmp_path):
    """The NT record's own literals and md5 are right, but its C3 recount read refcv8's totals: refused, although
    the record's C3 flag says pass."""
    reasons = _judge("refcv8-nt", _nt_rec(c3_totals=(113, 77)), tmp_path)
    assert reasons == [f"G-BOX-OVERFIT: C3 recount totals [113, 77] on frame set {NT_FS_MD5[:8]}... != the "
                       f"registered [122, 108] on {NT_FS_MD5[:8]}..."]


def test_gate_count_bar_uses_the_profiles_n_pos(tmp_path):
    """n_conf 109: |109 - 113| / 113 = 0.035 passes refcv8's criterion 3; |109 - 122| / 122 = 0.107 fails NT's."""
    assert _judge("refcv8", _r8_rec(main=dict(_MAIN, n_conf=109)), tmp_path) == []
    reasons = _judge("refcv8-nt", _nt_rec(main=dict(_MAIN, n_conf=109)), tmp_path)
    assert len(reasons) == 1 and reasons[0].startswith("G-BOX-OVERFIT: main fails prereg criterion 3 ['count_rel']")


# ------------------------------------------------------------------------------------- the kit + template -------- #
def test_nt_box_template_names_the_profile_and_the_nt_roots():
    nt = (ROOT / "scripts" / "refcv8nt_bind_box.args.tpl").read_text(encoding="utf-8")
    r8 = (ROOT / "scripts" / "refcv8_bind_box.args.tpl").read_text(encoding="utf-8")
    subs = {"ARGV": "/a.json", "OUT": "/o", "COMMIT": "c" * 40}
    nt_args, nt_roots = A.render_template(nt, subs)
    r8_args, r8_roots = A.render_template(r8, subs)
    i = nt_args.index("--frameset-profile")
    assert nt_args[i + 1] == "refcv8-nt"
    # the same harness argv as refcv8's, but for the candidate text and the two profile tokens
    strip = lambda xs: [x for x in xs if not x.startswith("BINDING:")]
    assert strip(nt_args[:i] + nt_args[i + 2:]) == strip(r8_args)
    assert set(r8_roots) < set(nt_roots)
    assert {"/home/nvidia/r8ns/nt/labels/a1", "/home/nvidia/r8ns/nt/v2ep_a1"} <= set(nt_roots)


def test_nt_kit_bind_copies_and_pins_the_nt_audit_inputs():
    s = (ROOT / "scripts" / "nt_prelaunch.sh").read_text(encoding="utf-8")
    b = s.index("\nbind)")
    blk = s[b:s.index(";;", b)]
    assert 'NSRC="$CODE/stack/ops/refcv8nt_box_audit/raw"' in blk
    for rel, md5, base in (("PREREG_G_BOX_OVERFIT.md", PREREG_MD5, ROOT / "ops" / "refcv8_box_audit" / "raw"),
                           ("PREREG_G_BOX_OVERFIT_ADDENDUM_NT.md", NT_ADD_MD5, NT_RAW),
                           ("visibility/gbo_frameset_nt.json", NT_FS_MD5, NT_RAW)):
        assert f'echo "{md5}  $BAUD/{rel}"' in blk, rel
        assert _md5((base / rel).read_bytes()) == md5, rel           # the kit's literal IS the in-tree file
    assert "gbo_frameset.json" not in blk and R8_FS_MD5 not in blk  # the NT kit never hands over refcv8's frames
    assert 'TPL="$HERE/refcv8nt_bind_box.args.tpl"' in blk


# ------------------------------------------------------------------- the harness's NT data path (REAL code) ---- #
class _FakeDS:
    """The loader's V3Dataset stand-in: every episode has windows t = 0, 1, 2 (an NT token-episode's raw index)."""

    def __init__(self, eps, **kw):
        self.episodes = eps
        self.index = [(e_i, t) for e_i in range(len(eps)) for t in (0, 1, 2)]
        self.calls = []

    def enable_r8_speed_label(self, path, man, *, all_unknown=False):
        self.calls.append(("r8_speed_label", path, bool(all_unknown), len(self.index)))
        return {"n_windows": len(self.index)}

    def enable_vis1(self, sc, *, split):
        self.calls.append(("vis1", split, len(self.index)))
        self._vis1_sha12 = {int(e.episode_id): _sha12(e.clip) for e in self.episodes}
        return {"split": split}

    def __getitem__(self, i):
        # rows: POS (0.90, 900 px), IGNORE (0.10), POS (0.75, 300 px), padding
        return {"agent_box": torch.tensor([[10.0, 0.0, 4.0, 2.0], [20.0, 1.0, 4.0, 2.0], [30.0, -2.0, 0.6, 0.6],
                                           [0.0, 0.0, 0.0, 0.0]]),
                "agent_valid": torch.tensor([True, True, True, False]),
                "agent_vis_full": torch.tensor([1000, 1000, 400, 0], dtype=torch.int32),
                "agent_vis_px": torch.tensor([900, 100, 300, 0], dtype=torch.int32),
                "agent_vis_known": torch.tensor([True, True, True, False])}


class _FakeLoader:
    def __init__(self):
        self.restrict_calls, self.seen, self.ds = [], {}, None

        def restrict(ds, mode):
            self.restrict_calls.append((mode, len(ds.index)))
            ds.index = [(e, t) for (e, t) in ds.index if t == 0]
            return {"mode": mode, "n_windows": len(ds.index)}
        self._tr = SimpleNamespace(V3Dataset=_FakeDS, lan_dataset_class=lambda c: c, restrict_window_starts=restrict)

    def trainer(self):
        return self._tr

    def build_eval_dataset(self, model, cfg, args, config, *, with_perception_targets=True, dataset_cls=None):
        from tanitad.data import v2_dataset as v2d
        self.seen = {"max_speed_input_v6": bool(getattr(args, "max_speed_input_v6", False)), "dataset_cls": dataset_cls}
        if dataset_cls is not None and not issubclass(dataset_cls, self._tr.V3Dataset):
            raise SystemExit("[refcv6_loader] dataset_cls must subclass the trainer's V3Dataset")   # the loader's rule
        eps = v2d.build_v2_providers([args.eval_cache], lru_size=None)
        self.ds = (dataset_cls or self._tr.V3Dataset)(eps)
        return self.ds, eps, {}


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--slot-vis1", action="store_true")
    p.add_argument("--vis1-sidecar")
    p.add_argument("--v2-cache", nargs="+")
    p.add_argument("--v7-labels")
    p.add_argument("--speed-max-sidecar-v6")
    p.add_argument("--size", default="small")
    p.add_argument("--arm", default="hier")
    p.add_argument("--window-starts", choices=("all", "first"), default="all")
    p.add_argument("--r8-speed-unknown-always", action="store_true")
    p.add_argument("--max-speed-input-v6", action="store_true")
    p.add_argument("--r8-speed-input", default=None)
    p.add_argument("--r8-v9-labels", default=None)
    return p


def _ep(i, clip):
    return SimpleNamespace(episode_id=100 + i, clip=clip,
                           frames=SimpleNamespace(_cache=SimpleNamespace(files={0: f"/c/{clip}.v2ep.pt"}), _clip=0))


@pytest.fixture
def harness_fakes(monkeypatch):
    from tanitad.data import v2_dataset as v2d
    from tanitad.data import v7_labels as v7l
    from tanitad.data import vis1 as V
    clips = ["clipA", "clipB"]
    monkeypatch.setattr(v2d, "load_or_build_manifest", lambda cache, verbose=False: {"clip_id": list(clips)})
    monkeypatch.setattr(v2d, "build_v2_providers", lambda caches, lru_size=None: [_ep(i, c) for i, c in enumerate(clips)])
    monkeypatch.setattr(v7l, "load_v7_labels", lambda path, allow_oracle_nav=False: ([], SimpleNamespace(md5="m")))
    monkeypatch.setattr(V, "VIS1Sidecar", lambda path: SimpleNamespace(stamp=lambda: {"path": path}))
    tr = SimpleNamespace(build_parser=_parser, v3=SimpleNamespace(refc_v3_sized_config=lambda size, hier=False: {}),
                         _pin_trainer_cfg=lambda cfg, a: cfg, r8train=SimpleNamespace(R8_SPEED_LABEL="label"))
    return tr, [_sha12(c) for c in clips]


_BASE = ["--slot-vis1", "--vis1-sidecar", "s.npz", "--v2-cache", "c", "--v7-labels", "l.gz",
         "--speed-max-sidecar-v6", "spd.jsonl", "--max-speed-input-v6", "--r8-speed-input", "label"]


def test_harness_nt_argv_restricts_windows_at_construction_and_attaches_the_all_unknown_speed_sidecar(harness_fakes):
    tr, s12 = harness_fakes
    L = _FakeLoader()
    fs = {"frames": [{"sha12": s, "t": 0} for s in s12]}
    ad = G.TrainerAdapter(tr, L, _BASE + ["--r8-speed-unknown-always", "--window-starts", "first"], fs, "cpu")
    assert L.seen["max_speed_input_v6"] is False                      # the loader's refusing attach is skipped
    assert L.seen["dataset_cls"] is not None and issubclass(L.seen["dataset_cls"], _FakeDS)
    assert L.restrict_calls == [("first", 6)]                           # at CONSTRUCTION, over all 6 raw windows
    assert ("r8_speed_label", "spd.jsonl", True, 2) in L.ds.calls      # train()'s attach, on the 2 launch windows
    assert ("vis1", "train", 2) in L.ds.calls
    assert ad.dataset_record["window_starts"] == {"mode": "first", "n_windows": 2}
    assert ad.per_frame == [(s12[0], 0, 2, 1), (s12[1], 0, 2, 1)]


def test_harness_refcv8_shaped_argv_takes_the_loader_path_unchanged(harness_fakes):
    tr, s12 = harness_fakes
    L = _FakeLoader()
    fs = {"frames": [{"sha12": s12[0], "t": 1}]}                        # a t >= 1 window exists: nothing restricted
    ad = G.TrainerAdapter(tr, L, list(_BASE), fs, "cpu")
    assert L.seen == {"max_speed_input_v6": True, "dataset_cls": None}
    assert L.restrict_calls == [] and not any(c[0] == "r8_speed_label" for c in L.ds.calls)
    assert ("vis1", "train", 3) in L.ds.calls and "window_starts" not in ad.dataset_record   # 1 episode x 3
    assert ad.per_frame == [(s12[0], 1, 2, 1)]


def test_harness_refuses_the_all_unknown_flag_without_the_label_speed_input(harness_fakes):
    tr, s12 = harness_fakes
    argv = [x for x in _BASE if x not in ("--r8-speed-input", "label")] + ["--r8-speed-unknown-always"]
    with pytest.raises(SystemExit, match="without --r8-speed-input label"):
        G.TrainerAdapter(tr, _FakeLoader(), argv, {"frames": [{"sha12": s12[0], "t": 0}]}, "cpu")
