"""Stage-5 driver for the RALIGN arm (PREREG_T2_RALIGN.md section 5), as a WRAPPER of the banked driver: shipped to box 2 (pkg/code/), `t2v8f_stage5.py` is imported and not edited.

    python -I ralign_driver.py all  --pkg PKG --kit KIT --hf HF --pull PULL5 --scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --manifest5 PKG/MANIFEST5.json [--stub] [--inproc]
    python -I ralign_driver.py plan --manifest5 MANIFEST5.json

The arguments, the cell order, the CPU gate, the preflight, the markers and the result are the banked driver's. Four module-level names of `t2v8f_stage5` are rebound before its `main` runs:
  CELL5       -> ralign_cell.py (the cell with the map);
  check_arm5  -> the banked check on a copy whose encoder kind reads frozen_dinov3, PLUS `check_ralign_block` (named refusals RALIGN_*);
  preflight5  -> `preflight_ra` (the maps re-hashed by content, identity iff status IDMAP, written to the result as `ralign`) and then the banked preflight;
  rollouts5   -> the banked one, with `install_ra` first for the --inproc rehearsal.
The banked driver keeps refusing this kind by name (ENCODER_UNSUPPORTED); only this wrapper runs it.
"""
from __future__ import annotations

import copy
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ralign_cell as RC  # noqa: E402
import ralign_lib as L  # noqa: E402
import t2v8f_stage5 as D5  # noqa: E402

KIND = RC.KIND
STATUSES = ("GATED", "IDMAP", "REHEARSAL")
ALLOW_REHEARSAL_ENV = "RALIGN_ALLOW_REHEARSAL"
BANKED_CHECK, BANKED_PREFLIGHT, BANKED_ROLLOUTS = D5.check_arm5, D5.preflight5, D5.rollouts5
CELL_RA = f"{HERE}/ralign_cell.py"


def check_ralign_block(m):
    """The RALIGN block of a MANIFEST5. -> [reason]."""
    a = D5.arm_of(m)
    blk, name, bad = a.get("ralign"), a.get("name"), []
    if not isinstance(blk, dict):
        return ["RALIGN_BLOCK_MISSING arm5.ralign"]
    if blk.get("card_sha256") != L.CARD_SHA256:
        bad.append(f"RALIGN_CARD_SHA {blk.get('card_sha256')!r} is not the frozen card's {L.CARD_SHA256}")
    st = blk.get("status")
    if st not in STATUSES:
        bad.append(f"RALIGN_STATUS {st!r} not in {STATUSES}")
    if st == "REHEARSAL" and os.environ.get(ALLOW_REHEARSAL_ENV) != "1":
        bad.append(f"RALIGN_REHEARSAL_NOT_ALLOWED this package carries rehearsal maps (not gated by M-ALIGN); set {ALLOW_REHEARSAL_ENV}=1 for a stub rehearsal only")
    gate = blk.get("gate") or {}
    if st == "GATED" and not (gate.get("m_align_verdict") == "M-ALIGN-PASS" and gate.get("g_idmap_verdict") == "G-IDMAP-PASS"):
        bad.append("RALIGN_NOT_GATED status GATED needs gate.m_align_verdict == M-ALIGN-PASS and gate.g_idmap_verdict == G-IDMAP-PASS")
    if blk.get("fold_digit") not in (0, 1):
        bad.append(f"RALIGN_FOLD_DIGIT {blk.get('fold_digit')!r}")
    files = {e["path"]: e for e in m.get("files", [])}
    for f in L.FOLDS:
        e = (blk.get("maps") or {}).get(f) or {}
        if e.get("file") not in files or files[e["file"]]["sha256"] != e.get("file_sha256"):
            bad.append(f"RALIGN_MAP_NOT_IN_OVERLAY map {f}: {e.get('file')!r} is not an overlay file with sha256 {str(e.get('file_sha256'))[:12]}")
    if isinstance(name, str):
        want = {f"{name}_s0": "C0_s0.pt", f"{name}_s1": "C0_s1.pt"}
        if a.get("ckpt_file") != want:
            bad.append(f"RALIGN_NOT_C0C_WEIGHTS ckpt_file {a.get('ckpt_file')!r}, expected C0's own checkpoints {want} (this wrapper builds C0C-RA)")
    table = blk.get("scene_fold") or {}
    if blk.get("fold_digit") in (0, 1):
        for sc in m.get("scenes", []):
            want_f = L.fold_of(sc["scene"], int(blk["fold_digit"]))
            if table.get(sc["scene"]) != want_f:
                bad.append(f"RALIGN_SCENE_FOLD {sc['scene'][:8]}: manifest {table.get(sc['scene'])!r}, sha256 {want_f!r}")
                break
    return bad


def check_arm5_ra(m):
    enc = (D5.arm_of(m).get("encoder") or {}).get("kind")
    if enc != KIND:
        return [f"ENCODER_UNSUPPORTED {enc!r}: the RALIGN driver builds arm5.encoder.kind == {KIND!r} only"]
    m2 = copy.deepcopy(m)
    m2["arm5"]["encoder"] = {"kind": D5.ENCODER_FROZEN}
    return BANKED_CHECK(m2) + check_ralign_block(m)


def preflight_ra(ctx):
    a = D5.arm_of(ctx.man2)
    blk, bad, info = a.get("ralign") or {}, [], {}
    try:
        maps = {f: L.load_map(f"{ctx.pkg}/{blk['maps'][f]['file']}", blk["maps"][f]["file_sha256"], blk["maps"][f]["tensor_sha256"]) for f in L.FOLDS}
        ident = {f: maps[f].is_identity for f in L.FOLDS}
        if blk.get("status") == "IDMAP" and not all(ident.values()):
            bad.append("RALIGN_IDMAP_NOT_IDENTITY status IDMAP but a map is not a := 1, b := 0")
        if blk.get("status") != "IDMAP" and any(ident.values()):
            bad.append(f"RALIGN_MAP_IS_IDENTITY status {blk.get('status')!r} but a map is the identity (the arm would equal C0C while claiming not to)")
        info = {"status": blk.get("status"), "fold_digit": blk.get("fold_digit"),
                "maps": {f: {"file_sha256": maps[f].file_sha256, "tensor_sha256": maps[f].tensor_sha256, "identity": ident[f]} for f in L.FOLDS},
                "scene_fold_counts": {f: sum(1 for v in (blk.get("scene_fold") or {}).values() if v == f) for f in L.FOLDS}, "gate": blk.get("gate")}
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
        bad.append(f"RALIGN_MAPS_UNREADABLE {e}")
    if bad:
        return {"verdict": "REFUSED", "failures": bad, "info": {"refused_before": "the stage-5 preflight (nothing was hashed)"}}
    ctx.res["ralign"] = info
    out = BANKED_PREFLIGHT(ctx)
    out.setdefault("info", {})["ralign"] = info
    return out


def rollouts_ra(ctx):
    if ctx.a.inproc:                                                       # rehearsal only: the cells run in this process, so the arm and the map are registered here
        import t2v8f_run as R
        RC.install_ra(R, ctx.man2_path)
        return D5.D.rollouts(ctx)
    return BANKED_ROLLOUTS(ctx)


def patch():
    D5.CELL5, D5.check_arm5, D5.preflight5, D5.rollouts5 = CELL_RA, check_arm5_ra, preflight_ra, rollouts_ra
    return D5


def main(argv=None):
    patch()
    return D5.main(argv)


if __name__ == "__main__":
    sys.exit(main())
