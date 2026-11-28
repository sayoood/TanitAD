"""Builds the eval ``ubasis`` file of one basis for the SLOWWM read (``slowwm_hook.build_eval_basis_frames``): u = the basis applied to the recomputed token mean ``nm`` of every eval clip, in the frames file's clip order.
Usage: python make_ubasis.py <basis_npz> <frames_npz> <root> <tracks_pt> <out_npz>
  root / tracks_pt: the cache root and native-tracks file the basis builder (``slow_basis.py cache``) was given for the eval split."""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lp_slowwm as S  # noqa: E402

if __name__ == "__main__":
    basis, frames, root, tracks, out = sys.argv[1:6]
    B = S.SH.SB.load_basis(basis)
    fr = S.LR.load_eval_frames(frames)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(S.SH.build_eval_basis_frames(B, fr, root, tracks, out), B["kind"], B["sha"][:12])
