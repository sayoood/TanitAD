"""RALIGN at T2: `AlignPolicy`, a wrapper beside `t2v8f_kit.BlindPolicy` (PREREG_T2_RALIGN.md section 3). Shipped to box 2 in the overlay (pkg/code/).

The map sits BETWEEN the in-loop encoder and the policy: the wrapper replaces the base policy's `encoder` with `MappedEncoder`, a proxy whose `tokens()` returns `TokenMap.apply(encoder.tokens(...))`
(fp16). The base policy's own loop (`V8FMiniT2Policy.plan`) calls `self.encoder.tokens(frame, intr)` and `push_tokens`, so EVERY scene-token row the model reads is mapped by construction -- the wrapper
does not re-implement that loop. Ego state, nav, the speed bucket, the composite (sg72 + kbox), the executor and every other input pass through untouched; the wrapper adds one key to `extra`
(`ralign`: scene fold, applied map, running count of mapped token sets) and changes nothing else.
Call contract = the kit's / BlindPolicy's: `plan(frames, intr, v0, nav_cmd, nav_known=None)`; everything else is delegated by `__getattr__`.

`install_align(R, block, root, arms)` is the stage-5 cell wiring: it wraps `t2v8f_run.build_arm` (AFTER `t2v8f_stage5_cell.install5` has put the composite on the arm) and chooses, per cell, the map of
the OTHER fold from the cell's `--t2-scene-id`.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ralign_lib as L  # noqa: E402

ADDED_KEYS = ("ralign",)


class MappedEncoder:
    """Proxy for an in-loop encoder: `tokens()` / `__call__` return the mapped fp16 rows; every other attribute (`snapshot`, `canon`, ...) is the wrapped encoder's."""

    def __init__(self, enc, tok_map: "L.TokenMap"):
        self._enc, self._map = enc, tok_map
        self.n_calls = 0

    def __getattr__(self, k):
        return getattr(self._enc, k)

    def tokens(self, frame, intr=None, **kw):
        t = self._enc.tokens(frame, intr, **kw)
        self.n_calls += 1
        return self._map.apply(t)

    def __call__(self, frame, intr=None):
        return self.tokens(frame, intr)


def base_policy(p):
    """Follows `.inner` links (CompositePolicy, ZeroPolicy, ...) to the object that owns `encoder` (a `V8FMiniT2Policy`). Reads `vars`, not `__getattr__`, so a delegating wrapper is not mistaken."""
    for _ in range(8):
        d = vars(p)
        if "encoder" in d and "push_tokens" in dir(type(p)):
            return p
        if "inner" not in d:
            break
        p = d["inner"]
    raise TypeError("RALIGN_NO_BASE_POLICY: no object with an encoder under this policy")


class AlignPolicy:
    name, ARM_ID = "v8f_align", "RA"

    def __init__(self, inner, tok_map: "L.TokenMap", *, scene_id=None, scene_fold=None):
        self.inner, self.tok_map, self.scene_id, self.scene_fold = inner, tok_map, scene_id, scene_fold
        b = base_policy(inner)
        if b.encoder is None:
            raise ValueError("RALIGN_NO_ENCODER: the wrapped policy has no encoder, so there are no render tokens to map (BLIND-style policies are not wrapped)")
        if isinstance(b.encoder, MappedEncoder):
            raise ValueError("RALIGN_ALREADY_WRAPPED: this policy's encoder is already a MappedEncoder")
        self._base, self._orig_encoder = b, b.encoder
        self._enc = MappedEncoder(b.encoder, tok_map)
        b.encoder = self._enc

    def __getattr__(self, k):
        if k in ("inner", "_base", "_enc", "_orig_encoder", "tok_map"):          # not yet set (copy / pickle): do not recurse
            raise AttributeError(k)
        return getattr(self.inner, k)

    def unwrap(self):
        """Puts the original encoder back and returns the wrapped policy."""
        self._base.encoder = self._orig_encoder
        return self.inner

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        if self._base.encoder is not self._enc:
            raise RuntimeError("RALIGN_ENCODER_REPLACED: the base policy's encoder is not this wrapper's MappedEncoder any more (the map would silently not be applied)")
        traj, extra = self.inner.plan(frames, intr, v0, nav_cmd, nav_known)
        extra = dict(extra, ralign={"scene_fold": self.scene_fold, "map_fold": self.tok_map.fold, "map_tensor_sha256": self.tok_map.tensor_sha256[:16], "n_token_sets_mapped": self._enc.n_calls})
        return traj, extra


def load_maps_from_block(block, root):
    """The two maps named by a manifest `ralign` block, each re-hashed (file and content) against the block. -> {'A': TokenMap, 'B': TokenMap}"""
    return {f: L.load_map(f"{root}/{block['maps'][f]['file']}", block["maps"][f]["file_sha256"], block["maps"][f]["tensor_sha256"]) for f in L.FOLDS}


def install_align(R, block, root, arms):
    """Wrap `R.build_arm` (module `t2v8f_run`) so that an arm in `arms` gets an AlignPolicy around its composite, with the other fold's map for the cell's scene. Idempotent for the same arms."""
    have = getattr(R, "_RA_ARMS", None)
    if have is not None:
        if tuple(have) != tuple(arms):
            raise SystemExit(f"[ralign] a different RALIGN arm {list(have)} is already installed in this process; refusing {list(arms)}")
        return R
    maps = load_maps_from_block(block, root)
    digit, table = int(block.get("fold_digit", 0)), block.get("scene_fold") or {}
    orig = R.build_arm

    def build_arm(args):
        pol = orig(args)
        if args.arm in arms:
            sid = getattr(args, "t2_scene_id", None)
            if not sid:
                raise SystemExit("[ralign] RALIGN_NO_SCENE_ID: the cell has no --t2-scene-id, so the fold (and the map) cannot be chosen")
            fold = L.fold_of(sid, digit)
            if table and table.get(sid) != fold:
                raise SystemExit(f"[ralign] RALIGN_FOLD_MISMATCH scene {sid[:8]}: the manifest says {table.get(sid)!r}, sha256 says {fold!r}")
            pol.inner = AlignPolicy(pol.inner, maps[L.other_fold(fold)], scene_id=sid, scene_fold=fold)
        return pol
    R.build_arm = build_arm
    R._RA_ARMS = tuple(arms)
    return R
