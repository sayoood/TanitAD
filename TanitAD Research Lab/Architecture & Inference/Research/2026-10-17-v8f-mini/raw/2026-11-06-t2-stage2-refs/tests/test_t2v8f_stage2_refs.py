"""Stage-2 reference frames (AGENT J): the preflight FAILS CLOSED on a scene without its `ref_frames.npz`, and the REAL reader and the REAL caller run on a scenes2-style folder.

Why this exists: stage 2 ran on box 2 and every cell of the 41 new scenes died with `FileNotFoundError ...\\camera_front_wide_120fov.mp4`. The harness's actor falsifier reads camera frames
0, 60 and 120; `t2v8f_nocv2.read_ref_frame` looks for `ref_frames.npz` beside the mp4 and otherwise falls through to PyAV. The overlay shipped neither. The rehearsal missed it because the stub
installs a fake `actor_map` (`t2v8f_run.py`: `m.attach_actors_verified = lambda ...`), so `read_ref_frame` never ran. "Built, tested, and unreachable from the real caller", stub-bypass form.

Every `check_*(mut=None)` is one arm; `mut="<name>"` makes the SAME assertion fail so the arm can fail (`code/run_mutations_stage2.py` runs them).
  real_read       REAL `t2v8f_nocv2.read_ref_frame` on a folder holding only `ref_frames.npz`: frames 0, 60, 120 bit-equal, `av.open` never called.      RED: the npz is absent
  real_caller     REAL `actor_map.falsify_actors` (the kit's), the REAL reader installed as `t2v8f_nocv2.install()` does, fake renderer: frames 0/60/120 come from the npz.   RED: no npz
  refused_missing the stage-2 preflight REFUSES a copied scene without its npz and NAMES it (a decoy mp4 does not excuse it).                            RED: nothing removed
  refused_kit     the same for the kit scene (7c72937c, installed by stage 1).                                                                            RED: nothing removed
  refused_changed one byte of an npz: `HASH_MISMATCH`.                                                                                                    RED: nothing changed
  refused_nolist  a MANIFEST2 without `ref_frames` entries is refused, every scene named.                                                                 RED: the entries are left in
  refused_skip    `--scene-policy skip` does NOT turn a missing npz into a skipped scene.                                                                 RED: nothing removed
  need_scales     `pass_need_mib`: a relaunch pays only for the cells with no DONE.                                                                      RED: the figure is not scaled
  artifact        the REAL overlay-2 tar: 41 npz = its MANIFEST2 sha256, 42 scenes listed, the driver equals `code/t2v8f_stage2.py`.                        RED: a tar entry dropped
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_stage2_refs.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
CODE = f"{T2}/code"
for _p in (CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import make_stage2 as MK  # noqa: E402
import t2v8f_nocv2 as N  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402
import test_t2v8f_stage2 as T  # noqa: E402

KIT_HARNESS = f"{T1.REAL_KIT}/tree/stack/experiments/alpasim-gsplat"
OVERLAY2 = f"{T2}/stage2/overlay2/t2v8f_stage2_overlay2.tar"
HW = (36, 64)                                                                          # tiny frames: the reader and the caller do not care about the size


def scratch(tag):
    d = f"{T.SCRATCH}/refs_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    assert not os.path.exists(d)
    os.makedirs(d)
    return d


def scene_only_npz(tag, seed=0):
    """A scenes2-style folder: `ref_frames.npz` and nothing else (no mp4, no volume). -> (dir, {f0, f60, f120} arrays)."""
    d = f"{scratch(tag)}/scenes2/e0000000-0000-0000-0000-000000000000"
    os.makedirs(d)
    rng = np.random.default_rng(seed)
    fr = {f"f{i}": (rng.random((*HW, 3)) * 255).astype(np.uint8) for i in D.REF_IDX}
    np.savez_compressed(f"{d}/ref_frames.npz", **fr)
    return d, fr


# ============================================================================================================================ the REAL reader
def check_real_read(mut=None):
    """`t2v8f_nocv2.read_ref_frame` (not the stub) on a scenes2-style folder returns frames 0, 60 and 120 bit for bit and never opens the mp4: `av.open` is wrapped and must not be called."""
    import av
    d, fr = scene_only_npz("read")
    if mut == "no_npz":
        os.replace(f"{d}/ref_frames.npz", f"{d}/ref_frames.npz.aside_by_test")
    mp4 = f"{d}/camera_front_wide_120fov.mp4"
    assert not os.path.exists(mp4)
    opened, real_open = [], av.open

    def spy(*a, **k):
        opened.append(a)
        return real_open(*a, **k)
    av.open = spy
    try:
        got = {}
        for i in D.REF_IDX:
            try:
                got[i] = N.read_ref_frame(mp4, i, (HW[1], HW[0]))
            except Exception as e:                                                      # noqa: BLE001  (this IS the stage-2 failure: FileNotFoundError on the mp4)
                raise AssertionError(f"read_ref_frame({i}) needed the mp4: {type(e).__name__}: {e}"[:200]) from e
    finally:
        av.open = real_open
    assert opened == [], f"the mp4 was opened: {opened}"
    for i in D.REF_IDX:
        assert got[i].dtype == np.uint8 and got[i].shape == (*HW, 3) and np.array_equal(got[i], fr[f"f{i}"]), f"frame {i} is not the banked one"
    assert not np.array_equal(got[0], got[60]) and not np.array_equal(got[60], got[120]), "the three frames are not distinct (the fixture cannot tell them apart)"
    return {"frames": sorted(got)}


class FakeRenderer:
    """Just enough of `NuRecGsplatRenderer` for `actor_map.falsify_actors`: the `on` render IS the banked frame of the asked camera frame, `off` is a dimmed copy, the wrong-time render is another frame."""

    def __init__(self, fr):
        self.fr, self.width, self.height, self.t_lo, self.t_hi, self.enabled = fr, HW[1], HW[0], 0.0, 1000.0, True
        self.frame = 0

    def gt_cam_to_nre(self, frame):
        self.frame = int(frame)
        return self.frame

    def frame_timestamps_us(self, frame):
        return (0, 100 * int(frame))

    def set_actors_enabled(self, on):
        self.enabled = bool(on)

    def render(self, c2n, actor_time_us=None):
        img = self.fr[f"f{self.frame}"]
        if not self.enabled:
            img = (img // 3).astype(np.uint8)
        elif actor_time_us is not None and actor_time_us != 100 * self.frame:        # the wrong-time control
            img = np.roll(img, 9, axis=1)
        return img, None, None


def check_real_caller(mut=None):
    """The kit's REAL `actor_map.falsify_actors`, with the reader installed exactly as `t2v8f_nocv2.install()` installs it, on a folder with only `ref_frames.npz`: it scores the three frames
    against the npz (grad_ncc of the banked frame with itself is 1.0) and reaches no mp4."""
    d, fr = scene_only_npz("caller", seed=3)
    if mut == "no_npz":
        os.replace(f"{d}/ref_frames.npz", f"{d}/ref_frames.npz.aside_by_test")
    # In a full-suite run another module has already put the STUB `gsplat_renderer` (t2a/stub_gsplat_renderer.py) into sys.modules; this arm needs the kit's REAL files, so the cached
    # entries are set aside, the real ones imported fresh from KIT_HARNESS, and the cached entries put back afterwards.
    saved = {n: sys.modules.pop(n, None) for n in ("gsplat_renderer", "actor_map")}
    sys.path.insert(0, KIT_HARNESS)
    try:
        import actor_map as A
        import gsplat_renderer as G
        assert os.path.dirname(os.path.abspath(G.__file__)).replace("\\", "/") == KIT_HARNESS, f"not the kit's real gsplat_renderer: {G.__file__}"
        old = (G.grad_ncc, G.read_ref_frame)
        info = N.install()
        try:
            assert G.read_ref_frame is N.read_ref_frame and info["replaced"] == ["grad_ncc", "read_ref_frame"], "the real harness is not using the nocv2 reader"
            out = []
            for i in D.REF_IDX:
                try:
                    out.append(A.falsify_actors(FakeRenderer(fr), d, i))
                except Exception as e:                                                  # noqa: BLE001
                    raise AssertionError(f"falsify_actors(frame {i}) failed: {type(e).__name__}: {e}"[:200]) from e
        finally:
            G.grad_ncc, G.read_ref_frame = old
    finally:
        sys.path.remove(KIT_HARNESS)
        for n in ("gsplat_renderer", "actor_map"):
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]
    assert [o["frame"] for o in out] == list(D.REF_IDX)
    assert all(abs(o["grad_ncc_actors_on"] - 1.0) < 1e-6 for o in out), [o["grad_ncc_actors_on"] for o in out]
    assert all(o["pass_placement"] for o in out)
    return {"grad_ncc_on": [o["grad_ncc_actors_on"] for o in out]}


# ============================================================================================================================ the preflight, through the full stub rehearsal
def refuse_run(tag, mutate, extra=(), manifest2=None):
    """A fresh scenes2 copy, `mutate(fx, scenes2)` applied, the whole stub driver run. -> (pull, rc, STAGE2_RESULT)."""
    fx = T.fixture()
    s2 = T.new_scenes2(fx, tag)
    undo = mutate(fx, s2)
    try:
        args = list(extra) + (["--manifest2", manifest2] if manifest2 else [])
        pull, rc, r = T.refused(fx, tag, s2, args)
    finally:
        if undo:
            undo()
    return pull, rc, r


def only_preflight_refusal(pull, rc, r, want):
    assert rc == 1 and os.path.exists(f"{pull}/STAGE2.fail") and not os.path.exists(f"{pull}/STAGE2.done"), (rc, os.listdir(pull))
    assert list(r["steps"]) == ["preflight"] and r["overall"] == "INFRASTRUCTURE_FAILURE", list(r["steps"])
    got = r["steps"]["preflight"]["failures"]
    assert got == want, f"failures {got} != {want}"
    assert not os.path.exists(f"{pull}/rollouts") and not os.path.exists(f"{pull}/GATE_SCENES.json"), "a cell or the gate ran after the refusal"
    assert all(w in open(f"{pull}/STAGE2.fail", encoding="utf-8").read() for w in want)


def check_refused_missing(mut=None):
    """A copied scene has no `ref_frames.npz` (a decoy mp4 beside it): STAGE2.fail names `MISSING_REF_FRAMES scenes2/<id>/ref_frames.npz`, nothing else, no gate, no cell."""
    fx = T.fixture()
    miss = fx["two"]

    def mutate(_fx, s2):
        open(f"{s2}/{miss}/camera_front_wide_120fov.mp4", "wb").write(b"not an mp4, and not what the package ships\n")
        if mut != "not_applied":
            os.replace(f"{s2}/{miss}/ref_frames.npz", f"{s2}/{miss}/ref_frames.npz.aside_by_test")
    man2, drv, orig = None, f"{fx['pkg']}/code/t2v8f_stage2.py", None
    if mut == "driver_unchecked":                                                   # MUTATE THE IMPLEMENTATION: delete the one line that turns a missing npz into a refusal (MANIFEST2 follows, so the driver passes its own sha check)
        orig = open(drv, "rb").read()
        new = orig.replace(b"    bad.extend(ref_bad)", b"    pass")
        assert new != orig, "the mutation did not apply: the line is not in the driver"
        m = json.load(open(f"{fx['pkg']}/MANIFEST2.json", encoding="utf-8"))
        for e in m["files"]:
            if e["path"] == "code/t2v8f_stage2.py":
                e["size"], e["sha256"] = len(new), hashlib.sha256(new).hexdigest()
        man2 = f"{fx['root']}/MANIFEST2_unchecked_{time.strftime('%H%M%S')}.json"
        json.dump(m, open(man2, "w", encoding="utf-8", newline="\n"), indent=1)
        open(drv, "wb").write(new)
    try:
        pull, rc, r = refuse_run("nonpz", mutate, manifest2=man2)
    finally:
        if orig is not None:
            open(drv, "wb").write(orig)
    only_preflight_refusal(pull, rc, r, [f"MISSING_REF_FRAMES scenes2/{miss}/ref_frames.npz"])
    assert "scene_hash_skipped" in r["steps"]["preflight"]["info"], "the 24 GB hash pass ran before the cheap refusal"
    return {"rc": rc}


def check_refused_kit(mut=None):
    """The KIT scene's npz (installed by stage 1) is gone: refused, named `kit:scenes/<id>/ref_frames.npz`. The file is moved back afterwards (the shadow kit is shared)."""
    fx = T.fixture()
    p = f"{fx['kit']}/scenes/{fx['kit_id']}/ref_frames.npz"

    def mutate(_fx, _s2):
        if mut == "not_applied":
            return None
        os.replace(p, p + ".aside_by_test")
        return lambda: os.replace(p + ".aside_by_test", p)
    pull, rc, r = refuse_run("nokit", mutate)
    only_preflight_refusal(pull, rc, r, [f"MISSING_REF_FRAMES kit:scenes/{fx['kit_id']}/ref_frames.npz"])
    assert os.path.isfile(p), "the shadow kit npz was not restored"
    return {"rc": rc}


def check_refused_changed(mut=None):
    """One byte of an npz flipped (same size): `HASH_MISMATCH scenes2/<id>/ref_frames.npz`, and only that."""
    fx = T.fixture()
    one = fx["one"]

    def mutate(_fx, s2):
        if mut != "not_applied":
            p = f"{s2}/{one}/ref_frames.npz"
            b = bytearray(open(p, "rb").read())
            b[len(b) // 2] ^= 0x01
            open(p, "wb").write(bytes(b))
    pull, rc, r = refuse_run("flipnpz", mutate)
    only_preflight_refusal(pull, rc, r, [f"HASH_MISMATCH scenes2/{one}/ref_frames.npz"])
    return {"rc": rc}


def check_refused_nolist(mut=None):
    """A MANIFEST2 with no `ref_frames` entries (overlay 1's shape) is refused and names every scene: the preflight does not assume the npz is there because nothing says otherwise."""
    fx = T.fixture()
    m = json.load(open(f"{fx['pkg']}/MANIFEST2.json", encoding="utf-8"))
    if mut != "left_in":
        for sc in m["scenes"]:
            sc.pop("ref_frames", None)
    p = f"{fx['root']}/MANIFEST2_nolist_{time.strftime('%H%M%S')}.json"
    json.dump(m, open(p, "w", encoding="utf-8", newline="\n"), indent=1)
    pull, rc, r = refuse_run("nolist", lambda *_: None, manifest2=p)
    only_preflight_refusal(pull, rc, r, [f"NO_REF_FRAMES_IN_MANIFEST2 {'kit:scenes' if sc['where'] == 'kit' else 'scenes2'}/{sc['scene']}/ref_frames.npz "
                                         "(this MANIFEST2 does not list the scene's reference frames: it is not the overlay that ships them)" for sc in m["scenes"]] if mut != "left_in" else ["x"])
    return {"rc": rc}


def check_refused_skip(mut=None):
    """`--scene-policy skip` skips a scene with damaged scene DATA; it must NOT turn a missing npz into a skipped scene: still refused, still named."""
    fx = T.fixture()
    miss = fx["two"]

    def mutate(_fx, s2):
        if mut != "not_applied":
            os.replace(f"{s2}/{miss}/ref_frames.npz", f"{s2}/{miss}/ref_frames.npz.aside_by_test")
    pull, rc, r = refuse_run("noskip", mutate, extra=["--scene-policy", "skip"])
    only_preflight_refusal(pull, rc, r, [f"MISSING_REF_FRAMES scenes2/{miss}/ref_frames.npz"])
    return {"rc": rc}


def check_positive_refs(mut=None):
    """The positive rehearsal reports every scene's frames checked: `ref_frames_ok == ref_frames == 3`."""
    fx = T.fixture()
    pull, _, rc, _ = T.positive()
    r = T.res2(pull)
    info = r["steps"]["preflight"]["info"]
    assert rc == 0 and r["overall"] == "COMPLETE", (rc, r["overall"])
    assert info["ref_frames_ok"] == info["ref_frames"] == (3 if mut != "count" else 4), info
    assert len(fx["build2"]["tar"]) > 0
    return {"ref_frames_ok": info["ref_frames_ok"]}


# ============================================================================================================================ the relaunch-aware disk figure
def check_need_scales(mut=None):
    """`pass_need_mib` is the manifest's figure for a pass with nothing done, and a proportional share when cells are DONE (a relaunch only pays for what is left)."""
    fx = T.fixture()
    m2 = json.load(open(f"{fx['pkg']}/MANIFEST2.json", encoding="utf-8"))
    pull = scratch("need")
    ctx = types.SimpleNamespace(pull=pull, man2=m2)
    ctx.live_scenes = lambda: m2["scenes"]
    cells = D.cell_order(m2["scenes"], ("deciding",))
    full = m2["footprint"]["pass_mib"]["deciding"]
    assert abs(D.pass_need_mib(ctx, "deciding") - full) < 1e-9, "a first run must pay the manifest's figure unchanged"
    k = len(cells) // 3
    for c in cells[:k]:
        d = D.cell_dir(pull, c["scene"], c["tc"], c["arm"])
        os.makedirs(d)
        open(f"{d}/DONE", "w").write("rc=0 records=1/1 executor=base\n")
    want = full * (len(cells) - k) / len(cells)
    got = D.pass_need_mib(ctx, "deciding") if mut != "unscaled" else full
    assert abs(got - want) < 1e-9 and got < full, f"{got} != {want}"
    for c in cells:                                                                     # everything DONE: nothing left to write
        d = D.cell_dir(pull, c["scene"], c["tc"], c["arm"])
        os.makedirs(d, exist_ok=True)
        open(f"{d}/DONE", "w").write("rc=0 records=1/1 executor=base\n")
    assert D.pass_need_mib(ctx, "deciding") == 0.0
    return {"full": full, "after_third": want}


# ============================================================================================================================ the real artifact
def check_artifact(mut=None):
    """The shipped overlay-2 tar, opened: MANIFEST2 v2 lists a `ref_frames` entry for all 42 scenes, the 41 `scenes2/<id>/ref_frames.npz` members equal those entries by sha256 and carry
    f0 / f60 / f120 as uint8 HxWx3, the kit scene's npz is NOT shipped, the driver equals `code/t2v8f_stage2.py`, no mp4 is in the tar."""
    assert os.path.isfile(OVERLAY2), f"{OVERLAY2} does not exist: build it with make_overlay2.py"
    assert MK.sha256_file(OVERLAY2) == open(OVERLAY2 + ".sha256").read().split()[0], "the overlay-2 tar does not match its .sha256"
    with tarfile.open(OVERLAY2) as tf:
        names = tf.getnames()
        m = json.loads(tf.extractfile("pkg/MANIFEST2.json").read())
        assert m["overlay_version"] == 2 and len(m["scenes"]) == 42
        shipped = {sc["scene"]: sc for sc in m["scenes"] if sc["where"] == "scenes2"}
        kit = [sc for sc in m["scenes"] if sc["where"] == "kit"]
        assert len(shipped) == 41 and len(kit) == 1 and not kit[0]["ref_frames"]["shipped_by_overlay2"]
        want = {f"scenes2/{sid}/ref_frames.npz" for sid in shipped} | {"pkg/MANIFEST2.json", "pkg/code/t2v8f_stage2.py"}
        if mut == "dropped":
            names = [n for n in names if n != sorted(want)[5]]
        assert sorted(names) == sorted(want), f"tar members differ: extra {sorted(set(names) - want)}, missing {sorted(want - set(names))}"
        assert not any(n.endswith(".mp4") for n in names)
        for sid, sc in shipped.items():
            b = tf.extractfile(f"scenes2/{sid}/ref_frames.npz").read()
            e = sc["ref_frames"]
            assert len(b) == e["bytes"] and hashlib.sha256(b).hexdigest() == e["sha256"], f"{sid[:8]}: npz != its MANIFEST2 entry"
        drv = tf.extractfile("pkg/code/t2v8f_stage2.py").read()
    assert drv == open(f"{CODE}/t2v8f_stage2.py", "rb").read(), "the tar's driver is not code/t2v8f_stage2.py"
    assert any(e["path"] == "code/t2v8f_stage2.py" and e["sha256"] == hashlib.sha256(drv).hexdigest() for e in m["files"]), "MANIFEST2 lists another driver"
    return {"scenes": len(m["scenes"]), "shipped": len(shipped)}


# ============================================================================================================================ pytest entry points
def test_real_read():
    check_real_read()


def test_real_read_red():
    try:
        check_real_read("no_npz")
    except AssertionError:
        return
    raise AssertionError("the arm is not RED without the npz")


def test_real_caller():
    check_real_caller()


def test_real_caller_red():
    try:
        check_real_caller("no_npz")
    except AssertionError:
        return
    raise AssertionError("the arm is not RED without the npz")


def test_refused_missing():
    check_refused_missing()


def test_refused_kit():
    check_refused_kit()


def test_refused_changed():
    check_refused_changed()


def test_refused_nolist():
    check_refused_nolist()


def test_refused_skip():
    check_refused_skip()


def test_positive_refs():
    check_positive_refs()


def test_need_scales():
    check_need_scales()


def test_artifact():
    check_artifact()
