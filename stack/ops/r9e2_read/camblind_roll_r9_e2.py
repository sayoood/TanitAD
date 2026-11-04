"""camblind_roll_r9_e2.py -- launcher: camblind_roll_r9.py (UNCHANGED) with the refcv9 staging proxy installed, for a K2 that holds B4 / T1.  NEW file (E2-ROLL); K2SAFE edit (H4).

    python -X utf8 camblind_roll_r9_e2.py --camblind-script <.../2026-11-03-refcv9-camblind/code/camblind_roll_r9.py> --camera derange --derange-seed 0 \
        --ztac-out <f.npz> --require-seal -- <battery child argv>
The hook fires on the first import of ``r7_roll`` (inside camblind's ``roll_seed_r8_pf`` import), then camblind runs as ``__main__`` with its own argv.
HAZARD (UNVERIFIED; cannot be fixed without editing it): camblind_roll_r9.py hard-codes ``EVAL_REMAP = C:/Users/Admin/refcv8_pf/refcv9_l1/code/refcv8_eval_remap_devbox_l1.json``
(a RELATIVE path on Thor => run with cwd = a dir holding that relative tree) and ``PKG = <incoming>/2026-10-27-refcv9-anticheat`` (must be shipped);
its ``L1_PF_CODE`` / ``L1_R8_CODE`` env pins are required here.

K2SAFE (H4) -- a K2 built with ``--r9-bev-warp`` (E1 arm T1 / T1s) adds FOUR dataset-item keys the camera-blind census does not know
(``bw_frames, bw_pose_now, bw_pose_past, bw_has``; ``V3Dataset.__getitem__`` -> ``bev_warp_flag.item_fields``, refc_v3_train.py:4905-4907): the stock instrument
REFUSES such a K2 at window 0 (``camblind_r9.UnclassifiedKey``).  The fix is NOT to class ``bw_frames`` as non-camera: it is the OWN past camera pixels (the
newest raw frame at rows ``row - lag``), so a "blind" arm that kept it would read its own scene through the warp (REFCV9_DESIGN W4: "every camera-derived input from the
donor").  ``patch_camblind`` therefore extends the census for the lifetime of the roll WITHOUT editing ``stack/tanitad/eval/camblind_r9.py`` or the FlyWheels package:
    bw_frames    CAMERA (pixel-derived) -> taken from the DONOR episode at the donor window's NOW row ``t_d + w - 1`` and the SAME lag rows (``bw_donor_item_fn``)
    bw_pose_now  EGO   -> the window's OWN poses (IMU-class, <= NOW)          bw_pose_past EGO -> own
    bw_has       META  -> the window's OWN validity (a function of the row index inside the clip, not of pixels)
and rebinds the four ``CB.*`` entry points the FlyWheels script calls (``make_camblind_dataset_cls``, ``other_inputs_digest``, ``camera_keys``, ``census_sha256``) so that
the dataset class, control (c)'s digest and the bank's meta all use the extended census.  A K2 WITHOUT --r9-bev-warp (``ds._r9_bw is None``): the donor fn returns ``{}``
and no ``bw_*`` key exists, so nothing is swapped (only the census sha / camera-key list stamped in the bank's meta differ).  The patch is undone when the roll ends."""
from __future__ import annotations

import inspect
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_battery_r9_e2 as R9  # noqa: E402

BW_SRC = "tanitad/train/bev_warp_flag.py"
BW_FRAMES = "bw_frames"
BW_KEYS = ("bw_frames", "bw_pose_now", "bw_pose_past", "bw_has")
#: the four CB.* entry points camblind_roll_r9.py calls (camblind_roll_r9.py:120 / :158 / :213); each takes ``census=`` as a keyword
PATCHED = ("make_camblind_dataset_cls", "other_inputs_digest", "camera_keys", "census_sha256")


def _load_cb():
    try:
        from tanitad.eval import camblind_r9 as CB
    except ImportError as e:                                                           # PYTHONPATH must carry <tree>/stack
        raise SystemExit(f"[r9e2-cb] cannot import tanitad.eval.camblind_r9 ({e}): PYTHONPATH must carry <tree>/stack") from None
    return CB


def bw_entries(CB) -> tuple:
    """The census rows of the T1 history keys, each pinned to the line of ``bev_warp_flag.item_fields`` that emits it (``CB.verify_census`` resolves them)."""
    why = "refcv9 T1 (--r9-bev-warp) history, added in V3Dataset.__getitem__ (refc_v3_train.py:4905-4907) via bev_warp_flag.item_fields: "
    return (
        CB.Entry("bw_frames", "item", CB.CAMERA, True, True, BW_SRC, 'return {"bw_frames": torch.stack(frames),', "",
                 why + "the OWN past camera frames (newest raw frame at rows row - lag): pixel-derived, read by bev_warp_flag.stage -> the trunk's stride-8 tap -> the lift -> the fused BEV"),
        CB.Entry("bw_pose_now", "item", CB.EGO, False, True, BW_SRC, '"bw_pose_now": pose_now,', "", why + "the NOW ego pose (ep.poses[row][:4]): egomotion <= NOW, IMU-class"),
        CB.Entry("bw_pose_past", "item", CB.EGO, False, True, BW_SRC, '"bw_pose_past": torch.stack(poses),', "", why + "the PAST ego poses at rows row - lag: egomotion <= NOW"),
        CB.Entry("bw_has", "item", CB.META, False, True, BW_SRC, '"bw_has": torch.tensor(has, dtype=torch.bool)}', "",
                 why + "validity of each lag (row - lag >= 0): a function of the row index inside the clip, not of pixels; masks the warped BEV (bev_warp.py lift_hook)"),
    )


def k2safe_census(CB) -> tuple:
    """``CB.CENSUS`` + the four ``bw_*`` rows (idempotent)."""
    have = {(e.kind, e.name) for e in CB.CENSUS}
    return tuple(CB.CENSUS) + tuple(e for e in bw_entries(CB) if ("item", e.name) not in have)


def bw_donor_item_fn(ds, j: int) -> dict:
    """The DONOR window's camera-derived T1 history: ``{"bw_frames": [P, 3, H, W]}`` = the DONOR episode's newest raw frame at rows ``row_d - lag`` (``row_d`` = the donor
    window's NOW row ``t_d + w - 1``, the lags of THIS dataset's ``_r9_bw`` source).  Past rows only; a row before the donor clip starts is CLAMPED to its first frame (a real
    donor frame, never zeros -- the item keeps its OWN ``bw_has``, so a zero frame under ``has = True`` would hand the blind arm an out-of-distribution black image and
    bias W4 towards 'separated').  NEVER shuffled, whatever ``ds._r9_bw.shuffle`` says (the donor is the instrument's, not the T1 control's).  No ``_r9_bw`` -> ``{}``.
    It does NOT call ``_window_u8`` (camblind's stock ``donor_item_fn`` -- ``V3Dataset.__getitem__(ds, j)`` -- re-enters the BLIND ``_window_u8`` and overwrites ``DIG`` of the donor row)."""
    import torch
    src = getattr(ds, "_r9_bw", None)
    if src is None:
        return {}
    e_d, t_d = (int(x) for x in ds.index[j])
    row_d = t_d + int(ds.window) - 1
    ep = ds.episodes[e_d]
    return {BW_FRAMES: torch.stack([ep.frames[max(row_d - int(r), 0)][-3:] for r in src.lag_rows])}


def patch_camblind(CB, census=None):
    """Rebind ``CB``'s four entry points (``PATCHED``) to the K2SAFE census; returns ``(census, undo)``.  REFUSES a census whose extra CAMERA keys are not exactly {bw_frames}
    (the donor fn knows no other).  ``undo()`` restores the originals (idempotent)."""
    census = k2safe_census(CB) if census is None else tuple(census)
    extra = tuple(k for k in CB.camera_keys("item", census) if k not in CB.WINDOW_CAMERA_KEYS)
    if extra != (BW_FRAMES,):
        raise SystemExit(f"[r9e2-cb] the extended census has CAMERA keys {extra} beyond the window pixels; this launcher can donor-swap only {(BW_FRAMES,)}")
    orig = {n: getattr(CB, n) for n in PATCHED}

    def _bind(fn):
        sig = inspect.signature(fn)

        def bound(*a, **kw):                                                           # CB's OWN internal calls (camera_keys("item", census)) pass census positionally
            if "census" not in sig.bind_partial(*a, **kw).arguments:
                kw["census"] = census
            return fn(*a, **kw)
        bound.__wrapped__, bound.__name__ = fn, fn.__name__
        return bound

    def make_cls(base, **kw):
        kw["census"] = kw.get("census") or census
        kw["donor_item_fn"] = bw_donor_item_fn                                         # the script's lambda re-enters the blind _window_u8; see bw_donor_item_fn
        return orig["make_camblind_dataset_cls"](base, **kw)

    make_cls.__wrapped__, make_cls.__name__ = orig["make_camblind_dataset_cls"], "make_camblind_dataset_cls"
    for n in PATCHED[1:]:
        setattr(CB, n, _bind(orig[n]))
    CB.make_camblind_dataset_cls = make_cls

    def undo():
        for n, f in orig.items():
            setattr(CB, n, f)
    return census, undo


def main() -> int:
    argv = sys.argv[1:]
    if "--camblind-script" not in argv:
        raise SystemExit("[r9e2-cb] --camblind-script <camblind_roll_r9.py> is required")
    i = argv.index("--camblind-script")
    script = Path(argv[i + 1])
    del argv[i:i + 2]
    R9.require_env(pins=("R8_LAUNCH_TREE", "PF_R8_CODE", "R8_EVAL_KIT", "L1_PF_CODE", "L1_R8_CODE"))
    R9.child_stage_hook()
    census, undo = patch_camblind(_load_cb())
    print(f"[r9e2-cb] K2SAFE census: {len(census)} entries (+{len(BW_KEYS)} bw_*; bw_frames CAMERA -> donor)", flush=True)
    sys.argv = [str(script)] + argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        st = getattr(sys.modules.get("r7_roll"), "R9E2_STAGER", None)
        print(f"[r9e2-cb] staged forwards: {None if st is None else st.census()}", flush=True)
        if (e.code in (0, None)) and (st is None or not st.n_forwards):
            raise SystemExit("[r9e2-cb] the roll ended OK but nothing was staged")
        raise
    finally:
        undo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
