"""``--u8-batches`` on ``scripts/refc_v3_train.py`` — uint8 in flight, the
contract's ``/255`` on the device. D-REFCV3-EVAL-DEATH-DIAGNOSED (2026-09-02).

THE PROBLEM (MEASURED on the pod, 2026-09-02). refcv3's 50 GB memory cgroup
sits at its cap and the kernel OOM-kills the trainer at eval steps; 19.8 GB of
the budget is UNRECLAIMABLE shmem — the six DataLoader workers' in-flight
batches in ``/dev/shm`` as fp32: per worker one ``torch_<pid>_*`` segment of
2,359,296,064 B (= 20 x 20 x 9 x 256 x 640 x 4 B + 64 B header: the
future-frames tensor) and one of 943,718,464 B (= 20 x 8 x 9 x 256 x 640 x 4 B
+ 64: the window). The ``*.v2ep.pt`` episodes store frames as UINT8.

THE DTYPE PATH, traced from source at HEAD ``1f9d08d`` (file:line quoted):

  1. STORAGE. A ``*.v2ep.pt`` payload holds PNG/JPEG bytes (``jpeg_buf``);
     ``tanitad/data/v2_dataset.py:143-149`` (``_decode_stacked``) decodes
         raw = [dec(jpeg_buf[int(offs[i]):int(offs[i + 1])],
                    mode=tvio.ImageReadMode.RGB) ...]    # [3, H, W] u8 each
         return stack_frames(torch.stack(raw), n_stack)  # [b-a, 3*n_stack, h, w] u8
     reached through ``_V2FramesProxy.__getitem__`` (:187-199) ->
     ``V2CompressedCache.decode_stacked_range`` (:352-355). The proxy's own
     docstring (:160-162) says what happens next: *"the result is a real owned
     uint8 tensor, so the downstream ``to_float_frames`` (``.float()/255``)
     works unchanged"*. The synthetic rig stores the same way
     (``refc_v3_train.py:378-379``):
         frames=(torch.rand(T, C, h, wpx, generator=g) * 255).to(torch.uint8)
  2. THE CAST. ``tanitad/data/_contract.py:126-139``
     (``EpisodeWindowDataset.__getitem__``):
         "frames": to_float_frames(ep.frames[t:t + w]),
         "future_frames": to_float_frames(
             ep.frames[t + w:t + w + self.max_horizon]),
     with ``to_float_frames`` at ``_contract.py:51-56``:
         return x.float().div(255.0) if x.dtype == torch.uint8 else x
     i.e. uint8 [0,255] -> float32 [0,1] by ``/255`` ONLY — no mean/std, no
     channel reorder, no resize. BOTH tensors take this ONE path (the future
     frames are not special-cased anywhere). It runs INSIDE the DataLoader
     worker, so the fp32 tensors are what ``default_collate`` stacks into the
     shared-memory segment. ⚠️ This file is NOT owned by the u8 change and is
     untouched (pinned below): the switch lives in the subclass.
  3. THE CHAIN above it never touches frames: ``FailLoudWindowDataset``
     ``.__getitem__`` (``refb_train.py``) calls the base, adds nav keys and
     CLONES every tensor (``for _k, _v in item.items(): ... _v.clone()``) — a
     SECOND full-size fp32 materialisation per item, transient in the worker;
     ``RouteV21Dataset.__getitem__`` (``refc_train.py:147-155``) adds route
     keys; ``V3Dataset.__getitem__`` (``refc_v3_train.py:302-347``) adds the
     6 s future + goals; ``lan_dataset_class`` (``refc_train.py:181-236``)
     adds ``lan``. So the fp32 bytes exist THREE times per item before the
     batch leaves the worker: the cast output, its clone, and the collated
     stack (only the last one lands in ``/dev/shm``; the first two are
     transient worker RSS).
  4. COLLATE. torch's ``default_collate`` — no custom ``collate_fn`` anywhere
     on this path (``refc_v3_train.py`` ``DataLoader(ds, batch_size=..,
     shuffle=True, num_workers=.., prefetch_factor=.., drop_last=True,
     persistent_workers=..)``); in a worker it allocates the stacked storage
     in shared memory (``/dev/shm`` under the ``file_system`` strategy the
     trainer sets at ``train()``'s top). ``pin_memory`` is NOT set on this
     loader (default False). uint8 through ``default_collate`` + shared memory
     is supported (``torch.stack`` into a ``_new_shared`` storage of the
     element dtype) — test 8 below drives it through a real worker.
  5. CONSUMER. ``compute_losses_v3`` (``refc_v3_train.py``, the top of the
     function — at HEAD ``frames = batch["frames"].to(device)`` /
     ``fut_frames = batch["future_frames"].to(device)``) -> ``model(frames,
     ...)`` -> ``RefCModel.forward`` (``refc.py:1990-1991``)
         fmap_all, pooled_all = self.encoder(
             frames.reshape(b * w, *frames.shape[2:]))
     -> ``ResNetEncoder.forward`` (``refc.py:994-998``): ``x = self.stem(x)``
     — a bare Conv2d on the tensor AS RECEIVED. There is NO normalisation
     inside the model; it expects float32 in [0,1] (a uint8 tensor raises a
     dtype error in the conv — test 7 pins that a missed conversion is LOUD,
     never a silent scale error). ``fut_frames`` is consumed EXACTLY ONCE, at
         law_tgt = model.core.encode_pooled(fut_frames[:, LAW_AHEAD - 1]
                                            .to(device))
     with ``LAW_AHEAD = 5`` (``refc_train.py:114``) — ONE of its 20 frames.
     (That 19/20 of the 2.36 GB future tensor is dead weight in flight is the
     next lever after uint8; it is a batch-CONTRACT change and not this one.)

THE FIX (default OFF; the live run resumes byte-identical):
  * ``FailLoudWindowDataset.u8_frames`` (``refb_train.py``) — when True the
    window is assembled by ``_window_u8``, the base contract's dict with the
    two ``to_float_frames`` calls removed; every other key is untouched.
  * ``refc_v3_train.frames_to_device(x, device)`` — the ONE ingest point:
    ``x.to(device)`` then, for uint8, ``x.float().div_(<0-dim tensor 255.0 on
    x.device>)`` — the contract's map, division in place (one fp32
    allocation fewer on the device) and by a TENSOR divisor. ⛔ THE TENSOR
    DIVISOR IS LOAD-BEARING, MEASURED 2026-09-02 (torch 2.11.0+cu128, RTX
    4060, EXHAUSTIVE — the map is elementwise, so the 256 uint8 values are
    the whole domain): with a Python-scalar divisor CUDA's div kernel takes
    the multiply-by-reciprocal fast path and **126/256 values land 1 ulp
    (5.96e-8) off the CPU contract** — i.e. ``to_float_frames`` itself, run
    on CUDA, is NOT the contract; with a 0-dim device tensor the kernel is a
    true IEEE division and **0/256** differ. On CPU every form is exact.
    Applied to ``frames`` and ``future_frames`` at the top of
    ``compute_losses_v3`` — training AND the held-out eval (both datasets get
    the switch). ``--u8-batches`` sets it and stamps ``u8_batches`` +
    ``frame_batch_bytes_est`` into ``config.json``.

PINS (all on the synthetic rig — smoke hier config, ``_synth_episodes(2,
seed=0)``, 1 x 64 x 64, window 4, 40 windows; CPU, nothing touches a pod):
  1. (c) FLAG OFF == HEAD: every key of windows 0/1/7/13 and of the collated
     [0,1] batch has the dtype/shape/sha256 computed at HEAD BEFORE the edit;
     the OFF path is also key-for-key ``torch.equal`` to the base contract's
     own ``__getitem__``; and ``frames_to_device`` on a float batch is the
     SAME object (a pure pass-through).
  2. (a) THE LOAD-BEARING ONE: with the flag ON the dataset emits uint8, and
     ``frames_to_device`` on that batch is ``torch.equal`` — exact, not
     allclose: the same integers through the same IEEE division give the same
     bits — to the flag-OFF batch for BOTH frames and future frames, per
     window and collated; every other key is bit-identical between ON and OFF.
     ⚠️ CONTROL: the ON batch is asserted uint8 and NOT equal to OFF before
     conversion, so the identity cannot pass vacuously.
  3. The uint8 -> float map is EXHAUSTIVELY identical to ``to_float_frames``
     over all 256 values (CPU, and CUDA when a GPU is present with >= 256 MB
     free — the run converts on CUDA; ``torch.equal`` there too). CONTROL:
     the scalar-divisor form on CUDA is asserted to DIFFER (126/256, 1 ulp)
     — the probe can fail, and the mechanism is pinned so nobody simplifies
     the divisor back to ``255.0``.
  4. (b) BYTES: the collated ON batch is < 0.3 of the OFF batch (printed);
     frames-only exactly 0.25.
  5. (d) ``--u8-batches`` parses (default False) and lands in ``config.json``
     in both states, with ``frame_batch_bytes_est``.
  6. (e) A 1-step ``train()`` ON has a loss ``torch.equal`` to the OFF loss —
     exact: on one device the only difference is WHERE ``/255`` runs, and the
     OFF loss equals the HEAD-computed pin bit for bit (``9ab25643``).
     ⚠️ CONTROL: the loss fn is asserted to RECEIVE uint8 in the ON run.
     And the same on CUDA (the run's device): MEASURED run-to-run
     bit-deterministic on this rig (OFF twice -> one hex), which is asserted
     as the admissibility control before the ON/OFF ``torch.equal``.
  7. A uint8 batch fed to the model WITHOUT conversion raises (the conv's
     dtype check) — a missed conversion can never train silently.
  8. A real ``num_workers=1`` DataLoader ships the uint8 batch bit-exactly
     (the worker -> shared memory -> parent path the fix is about).
  9. LIVE-SHAPE ARITHMETIC: the two measured ``/dev/shm`` segments equal the
     fp32 future / window tensors + 64 B each; six workers x prefetch 1 =
     19.82 GB fp32 -> 4.95 GB uint8.
 10. The cast's home (``_contract.py``) is untouched by this change (textual).
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refb_train as RB                                      # noqa: E402
import refc_v3_train as T                                    # noqa: E402
from tanitad.data._contract import (EpisodeWindowDataset,    # noqa: E402
                                    to_float_frames)
from tanitad.refs import refc_v3 as v3                       # noqa: E402

WINDOWS = (0, 1, 7, 13)
FRAME_KEYS = ("frames", "future_frames")

# ---------------------------------------------------------------- HEAD pins
# Computed at HEAD 1f9d08d BEFORE the edit (scratch script u8_head_pin.py):
# smoke hier config through _pin_trainer_cfg (kin3), _synth_episodes(2, seed=0),
# V3Dataset(window 4, max_horizon 20, channels 1). (dtype, shape, sha256 of
# the contiguous bytes). Non-tensor keys (episode_id) hold their repr.
_F, _L, _B = "torch.float32", "torch.int64", "torch.bool"
HEAD_ITEMS = {
    0: {"frames": (_F, (4, 1, 64, 64), "cc75279933ed0bcf6bfb1b7c96e9036efa364cab75655373005d9a7bd4b3817f"),
        "actions": (_F, (4, 2), "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"),
        "future_frames": (_F, (20, 1, 64, 64), "06ca73bc3a8953348e013091a4352858ee6646454aa36c44daf33aca93214c48"),
        "future_actions": (_F, (20, 2), "b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4"),
        "future_poses": (_F, (20, 4), "f23a74d04bf70237b46c12e1a2cb1bb082898c4f224bfedab2351dc6a7612260"),
        "pose_last": (_F, (4,), "cf595016ad2498c7d0a14e1b1f10cc214aed28d03461cdd0d023fa6d7c71d589"),
        "episode_id": "'synth-000'",
        "nav_cmd": (_L, (), "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"),
        "nav_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        "route_target": (_L, (), "35be322d094f9d154a8aba4733b8497f180353bd7ae7b0a15f90b586b549f28b"),
        "route_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        "future_poses_ext": (_F, (60, 4), "65fb42c0451397640a7501c91a88581e5ea0050cd6749e7f8a19ac8fd5aa13cb"),
        "future_valid_ext": (_B, (60,), "153f91002ac5b7b0ac7802ac03aabf31f95cf0abc9d0584caf26278a77481c5e"),
        "goal_tac": (_F, (3, 4), "eb5b24fa5af9fb3432775fa4313acae9810ff6261c57d28e650a3c49b8d0b8e7"),
        "goal_tac_valid": (_B, (3,), "fb50dc0717ff266cf9baf82b1ce7a1c2ef6d9247859680b11a19fb7077f5f222")},
    1: {"frames": (_F, (4, 1, 64, 64), "cf98b9a6901b86e1eb3cf41642c0fd3e584bb6044bc56aeeedbc0f629e0a69b7"),
        "actions": (_F, (4, 2), "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"),
        "future_frames": (_F, (20, 1, 64, 64), "9d6722dc28f3a87afe2416e0682676f563d6f1b3ec04c1365d719e0a94c68a02"),
        "future_actions": (_F, (20, 2), "b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4"),
        "future_poses": (_F, (20, 4), "9d126e00847e61d43b2be587dc0946772db2ce6b8164ddedc1212b5e3f7d18c7"),
        "pose_last": (_F, (4,), "80dd7aecb0ac6b72a6cc57fef54c46dbb2506611be32dcef21b617427358c9f7"),
        "episode_id": "'synth-000'",
        "nav_cmd": (_L, (), "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"),
        "nav_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        "route_target": (_L, (), "35be322d094f9d154a8aba4733b8497f180353bd7ae7b0a15f90b586b549f28b"),
        "route_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        "future_poses_ext": (_F, (60, 4), "7460f46de7d52096369cff47a0208ba91b2f89c8b9eb9d401a5d21d29314fa54"),
        "future_valid_ext": (_B, (60,), "d08604ba4d263a10a78ce8b033cf4f29c938fbfe33cefb38b333aa006e81778a"),
        "goal_tac": (_F, (3, 4), "74ea603c2353df3dc9322454271a1568def749be4f921f6daea916bc4aae9b7d"),
        "goal_tac_valid": (_B, (3,), "fb50dc0717ff266cf9baf82b1ce7a1c2ef6d9247859680b11a19fb7077f5f222")},
    7: {"frames": (_F, (4, 1, 64, 64), "e6ff75428b31dca22eca5e23596912db2e71e6ca9109d82f8ee005d59de767c0"),
        "actions": (_F, (4, 2), "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"),
        "future_frames": (_F, (20, 1, 64, 64), "b4abb60d3a4b411ffa98151ec6a5a1c69dd8955b9f4e7f883d2dce8d56446840"),
        "future_actions": (_F, (20, 2), "b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4"),
        "future_poses": (_F, (20, 4), "504849912063b8385a8205b3e485ca19e654eb05cd719ac8c1ff7e19d0e950b5"),
        "pose_last": (_F, (4,), "4ba8ee56916efd98e90ca3a1aa27d735875501fba0ba0c9fb7d7d2ecef24eb03"),
        "episode_id": "'synth-000'",
        "nav_cmd": (_L, (), "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"),
        "nav_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        "route_target": (_L, (), "35be322d094f9d154a8aba4733b8497f180353bd7ae7b0a15f90b586b549f28b"),
        "route_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        "future_poses_ext": (_F, (60, 4), "27bf39f7903c61c6954efb49e2810aea4746ae51cf8ce5d01be1c579f77e26fe"),
        "future_valid_ext": (_B, (60,), "a2b892ea88dbab3b4e1f2e3feb3f7e0a709d337fa2a6731de931eff93c1c1b24"),
        "goal_tac": (_F, (3, 4), "707ecb8d7a461231dcd2ff977e094e8f266dc737e263c972f548f2c8194cce1d"),
        "goal_tac_valid": (_B, (3,), "fb50dc0717ff266cf9baf82b1ce7a1c2ef6d9247859680b11a19fb7077f5f222")},
    13: {"frames": (_F, (4, 1, 64, 64), "a7f8d84768e4fbe9de594d5b30e498a0798c968a112480d375b5549b43b959cb"),
         "actions": (_F, (4, 2), "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"),
         "future_frames": (_F, (20, 1, 64, 64), "a3c924db6e3f59b427147dfb21c70a4c5ab7277b2b4b58e1ed389a0eb28b1ea3"),
         "future_actions": (_F, (20, 2), "b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4"),
         "future_poses": (_F, (20, 4), "fe58263f8bfcf53f7416f1ad40fd7ca349f6b66f6edaa4c27f38d0622f788bd9"),
         "pose_last": (_F, (4,), "38ddaa6b32d94e1ffca39b9c25af5aa0ce57a3cefb19f42d9ed9f0b114f543b4"),
         "episode_id": "'synth-000'",
         "nav_cmd": (_L, (), "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"),
         "nav_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
         "route_target": (_L, (), "35be322d094f9d154a8aba4733b8497f180353bd7ae7b0a15f90b586b549f28b"),
         "route_valid": (_B, (), "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
         "future_poses_ext": (_F, (60, 4), "ac61cb45f4c5b232692eb2ae7f348efb0c20dad3bc34c39969c712691d141016"),
         "future_valid_ext": (_B, (60,), "8c7b81e24552e1b183abe56d3081674c29e84057fd948b0d22c63ea2217649ca"),
         "goal_tac": (_F, (3, 4), "1a876bc2e6a5ea4a70fbd2ade0c42ea93276a0320f81caabe370025b4af97ee4"),
         "goal_tac_valid": (_B, (3,), "fb50dc0717ff266cf9baf82b1ce7a1c2ef6d9247859680b11a19fb7077f5f222")},
}
HEAD_BATCH01 = {
    "frames": (_F, (2, 4, 1, 64, 64), "ff44c5385067d45500fd64763dd559d126fc29ce0704c08e5c9ac7b41971ec3b"),
    "actions": (_F, (2, 4, 2), "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"),
    "future_frames": (_F, (2, 20, 1, 64, 64), "63b5aa0a703b978c4b72b162787b3db07245ea4a55904bbc0828c29a2186f179"),
    "future_actions": (_F, (2, 20, 2), "7b6436b0c98f62380866d9432c2af0ee08ce16a171bda6951aecd95ee1307d61"),
    "future_poses": (_F, (2, 20, 4), "39ddbe904bab1b83d69daf2b3502b35af91376d122bfdcb7322c066dd545b9e0"),
    "pose_last": (_F, (2, 4), "c5902d9eadf907edce5bfa66c60e3372a3c4682acced1bc72d70d0a219789759"),
    "episode_id": "['synth-000', 'synth-000']",
    "nav_cmd": (_L, (2,), "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"),
    "nav_valid": (_B, (2,), "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"),
    "route_target": (_L, (2,), "0fec49e5cb80c80f848a00237963650b3d777cd7a75e6fadc16db41e7b6b92e1"),
    "route_valid": (_B, (2,), "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"),
    "future_poses_ext": (_F, (2, 60, 4), "c69cb3dde585cfb12e0fd581139323cb443022b64dfe4a04db110ca3317d5eea"),
    "future_valid_ext": (_B, (2, 60), "6338e8979be7f880f9e687b2f1897f12cdff05bf58b82ca69c0f827297f7aaf9"),
    "goal_tac": (_F, (2, 3, 4), "0a3f4ec127cbf5806193b05ed94d8dab94d971dbf112cbf2715a337d144b066a"),
    "goal_tac_valid": (_B, (2, 3), "917140cbddb025737b7411f40c01f2e8c37ea61e77fa6bfaafafc680a60cc117"),
}
#: the 1-step train() at HEAD (flag-less trainer): what compute_losses_v3
#: received (the collated float batch, moved to "cpu") and its raw loss.
HEAD_TRAIN1 = {
    "frames": (_F, (2, 4, 1, 64, 64), "ea9932d9bfeee4ba7c67ca675f0f369ed7fe7d00ab8a2e4fda52af5d276efb8c"),
    "future_frames": (_F, (2, 20, 1, 64, 64), "a57a5796f4f668b07599d9ef6b1034eee9db3dfcd44a7765a30546e1b5d1a474"),
    # ⭐ RE-MINTED 2026-09-04 (refcv4). The previous pin was
    # 214.69766235351562 / "9ab25643", recorded before the TACTICAL AUX BUDGET
    # FIX: `compute_losses_v3` spent `LAT_WEIGHT*(loss_lat + loss_lat_tac) +
    # LON_WEIGHT*(loss_lon + loss_lon_tac)` = 0.05*2 + 0.05*2 = **0.20**,
    # against the invariant `refc_train.py:83-91` states IN WRITING -- that the
    # total tactical aux pressure is held at EXACTLY `MANEUVER_WEIGHT` (0.10)
    # so an arm differs in STRUCTURE, not in loss budget. It is now halved to
    # 0.025*4 = 0.10.
    # ⛔ WHY RE-MINTING IS ADMISSIBLE HERE, AND WHAT WAS CHECKED FIRST. This pin
    # exists to catch an UNINTENDED numeric change, so it may only be moved with
    # the change ACCOUNTED FOR, never merely because it fired:
    #   * the property the test actually protects -- `torch.equal(on, off)`,
    #     u8-path vs float-path -- still PASSES untouched;
    #   * the delta is EXACTLY the aux term and nothing else:
    #     214.69766235 - 214.58363342 = 0.11402893 = 0.025 x 4.5611572,
    #     i.e. 0.025 x (loss_lat + loss_lat_tac + loss_lon + loss_lon_tac),
    #     which is the halving's arithmetic to 8 significant figures.
    # Any FUTURE movement of this constant needs the same two lines of evidence.
    "loss": 214.58363342285156,
    "loss_hex": "69955643",                 # struct.pack("<f", loss).hex()
}

# -------------------------------------------------------- the live shape
#: refcv3's launch (2026-09-02): --batch 20 --workers 6 --prefetch-factor 1
#: --image-hw 256 640, window 8, max_horizon 20, 9 channels.
LIVE = dict(batch=20, window=8, max_horizon=20, ch=9, h=256, w=640,
            workers=6, prefetch=1)
#: MEASURED on the pod 2026-09-02: the two /dev/shm torch_<pid>_* segments
#: per worker (future-frames tensor, window tensor), bytes.
MEASURED_SHM_SEGMENTS_B = {"future_frames": 2_359_296_064,
                           "frames": 943_718_464}
SHM_SEGMENT_HEADER_B = 64


# ---------------------------------------------------------------- helpers
def _digest(t):
    if not torch.is_tensor(t):
        return repr(t)
    a = t.detach().cpu().contiguous().numpy()
    return (str(t.dtype), tuple(t.shape), hashlib.sha256(a.tobytes()).hexdigest())


def _rig(seed: int = 0):
    """The trainer's own synthetic rig: smoke hier config pinned the way
    train() pins it (kin3, no --image-hw), 2 episodes, ONE shared episode
    list feeding two datasets that differ ONLY in the switch."""
    cfg = T._pin_trainer_cfg(v3.refc_v3_smoke_config(True),
                             SimpleNamespace(image_hw=None, v7_labels=None))
    eps = T._synth_episodes(2, cfg.core, seed=seed)
    assert all(ep.frames.dtype == torch.uint8 for ep in eps)   # stored u8
    kw = dict(window=cfg.core.window, max_horizon=20,
              channels=cfg.core.encoder.in_channels)
    ds_off, ds_on = T.V3Dataset(eps, **kw), T.V3Dataset(eps, **kw)
    assert ds_off.u8_frames is False                    # class default
    ds_on.u8_frames = True
    assert len(ds_off) == len(ds_on) == 40
    return cfg, eps, ds_off, ds_on


def _tensor_bytes(batch: dict, keys=None) -> int:
    return sum(v.numel() * v.element_size() for k, v in batch.items()
               if torch.is_tensor(v) and (keys is None or k in keys))


def _cuda_free_or_skip(min_free_b: int = 256 << 20) -> str:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this box")
    free, _tot = torch.cuda.mem_get_info()
    if free < min_free_b:
        pytest.skip(f"CUDA busy: {free / 2**20:.0f} MB free < 256 MB")
    return "cuda"


def _argv(out, *extra):
    return ["--arm", "hier", "--out", str(out), "--smoke",
            "--synth-episodes", "2", "--steps", "1", "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", "1",
            *extra]


def _train_once(tmp_path, monkeypatch, name, *extra):
    """One 1-step train(); returns (raw loss tensor, dtype of the frames the
    loss fn RECEIVED, digests of what it forwarded, config.json)."""
    real = T.compute_losses_v3
    seen = {}

    def spy(model, batch, device, **kw):
        if model.training and "loss" not in seen:
            seen["recv_dtype"] = (batch["frames"].dtype,
                                  batch["future_frames"].dtype)
            seen["fwd"] = {k: _digest(T.frames_to_device(batch[k], device))
                           for k in FRAME_KEYS}
        r = real(model, batch, device, **kw)
        if model.training and "loss" not in seen:
            seen["loss"] = r["loss"].detach().clone()
        return r

    monkeypatch.setattr(T, "compute_losses_v3", spy)
    out = tmp_path / name
    args = T.build_parser().parse_args(_argv(out, *extra))
    assert T.train(args) == {"step": 1}
    monkeypatch.setattr(T, "compute_losses_v3", real)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return seen, cfg


# ------------------------------------------------- 1. (c) flag OFF == HEAD
def test_flag_off_is_bit_identical_to_head_and_to_the_base_contract():
    _cfg, _eps, ds_off, _ = _rig()
    for i in WINDOWS:
        item = ds_off[i]
        assert set(item) == set(HEAD_ITEMS[i]), i
        for k, want in HEAD_ITEMS[i].items():
            assert _digest(item[k]) == want, (i, k)
        # the functional twin: the base contract's own __getitem__, untouched
        base = EpisodeWindowDataset.__getitem__(ds_off, i)
        for k, vb in base.items():
            if torch.is_tensor(vb):
                assert vb.dtype == item[k].dtype and torch.equal(vb, item[k]), k
            else:
                assert vb == item[k], k
        assert item["frames"].dtype == torch.float32
    batch = torch.utils.data.default_collate([ds_off[0], ds_off[1]])
    assert set(batch) == set(HEAD_BATCH01)
    for k, want in HEAD_BATCH01.items():
        assert _digest(batch[k]) == want, k
    # the trainer's ingest on a float batch is a pure pass-through: the SAME
    # object, so the flag-OFF device tensor is exactly HEAD's `.to(device)`
    for k in FRAME_KEYS:
        assert T.frames_to_device(batch[k], "cpu") is batch[k], k


# ------------------------------------- 2. (a) ON + device conversion == OFF
def test_flag_on_emits_uint8_and_the_device_conversion_is_torch_equal_to_off():
    _cfg, _eps, ds_off, ds_on = _rig()
    for i in WINDOWS:
        off, on = ds_off[i], ds_on[i]
        assert set(off) == set(on), i
        for k in FRAME_KEYS:
            # CONTROL — the switch is live: uint8, and not the float tensor
            assert on[k].dtype == torch.uint8 and off[k].dtype == torch.float32
            assert on[k].shape == off[k].shape
            assert not torch.equal(on[k].float(), off[k])       # /255 missing
            conv = T.frames_to_device(on[k], "cpu")
            assert conv.dtype == torch.float32
            assert torch.equal(conv, off[k]), (i, k)            # EXACT
            assert torch.equal(to_float_frames(on[k]), off[k]), (i, k)
            assert float(conv.min()) >= 0.0 and float(conv.max()) <= 1.0
        for k in set(off) - set(FRAME_KEYS):                    # untouched
            if torch.is_tensor(off[k]):
                assert off[k].dtype == on[k].dtype
                assert torch.equal(off[k], on[k]), (i, k)
            else:
                assert off[k] == on[k], (i, k)
    b_off = torch.utils.data.default_collate([ds_off[0], ds_off[1]])
    b_on = torch.utils.data.default_collate([ds_on[0], ds_on[1]])
    for k in FRAME_KEYS:
        assert b_on[k].dtype == torch.uint8
        assert torch.equal(T.frames_to_device(b_on[k], "cpu"), b_off[k]), k
    for k in set(b_off) - set(FRAME_KEYS):
        if torch.is_tensor(b_off[k]):
            assert torch.equal(b_off[k], b_on[k]), k
        else:
            assert b_off[k] == b_on[k], k


def test_flag_on_device_conversion_on_cuda_is_torch_equal_to_off():
    dev = _cuda_free_or_skip()
    _cfg, _eps, ds_off, ds_on = _rig()
    b_off = torch.utils.data.default_collate([ds_off[0], ds_off[1]])
    b_on = torch.utils.data.default_collate([ds_on[0], ds_on[1]])
    for k in FRAME_KEYS:
        conv = T.frames_to_device(b_on[k], dev)
        assert conv.device.type == "cuda" and conv.dtype == torch.float32
        assert torch.equal(conv.cpu(), b_off[k]), k             # EXACT on CUDA
        # the float path on CUDA is the plain move
        assert torch.equal(T.frames_to_device(b_off[k], dev).cpu(), b_off[k])


# --------------------------------- 3. the map == to_float_frames, all 256
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_uint8_map_is_exhaustively_identical_to_the_contract(dev):
    if dev == "cuda":
        _cuda_free_or_skip()
    u8 = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    want = to_float_frames(u8)                                  # the contract
    assert want.dtype == torch.float32 and float(want.max()) == 1.0
    got = T.frames_to_device(u8, dev)
    assert got.device.type == dev
    assert torch.equal(got.cpu(), want)
    assert u8.dtype == torch.uint8                              # input intact


def test_scalar_divisor_form_on_cuda_is_NOT_the_contract_the_control():
    """MEASURED 2026-09-02 (torch 2.11.0+cu128, RTX 4060): ``x.float()
    .div(255.0)`` — ``to_float_frames`` itself — run on CUDA differs from the
    CPU contract on 126 of the 256 uint8 values by exactly 1 ulp (5.96e-8):
    CUDA's scalar-divisor kernel multiplies by the float32 reciprocal. This
    is WHY ``frames_to_device`` divides by a device TENSOR, and it is the
    control proving the exhaustive identity above can fail."""
    dev = _cuda_free_or_skip()
    u8 = torch.arange(256, dtype=torch.uint8)
    want = to_float_frames(u8)
    scalar_form = to_float_frames(u8.to(dev)).cpu()       # the contract fn on CUDA
    bad = scalar_form != want
    n_bad = int(bad.sum())
    ulp = (scalar_form.view(torch.int32) - want.view(torch.int32)).abs().max()
    print(f"\n[u8] to_float_frames on CUDA vs CPU: {n_bad}/256 values differ, "
          f"max {int(ulp)} ulp, max|diff| {float((scalar_form - want).abs().max()):.3e}")
    assert n_bad > 0 and int(ulp) == 1, (n_bad, int(ulp))   # the drift is real
    assert torch.equal(T.frames_to_device(u8, dev).cpu(), want)   # ours is not


# ------------------------------------------------- 4. (b) collated bytes
def test_collated_batch_bytes_drop_about_four_times():
    _cfg, _eps, ds_off, ds_on = _rig()
    idx = [0, 1, 7, 13]
    b_off = torch.utils.data.default_collate([ds_off[i] for i in idx])
    b_on = torch.utils.data.default_collate([ds_on[i] for i in idx])
    off_b, on_b = _tensor_bytes(b_off), _tensor_bytes(b_on)
    off_f, on_f = _tensor_bytes(b_off, FRAME_KEYS), _tensor_bytes(b_on, FRAME_KEYS)
    print(f"\n[u8] collated batch (4 windows, 1x64x64, window 4 + 20 future): "
          f"OFF {off_b:,} B / ON {on_b:,} B = {on_b / off_b:.4f}; "
          f"frames-only OFF {off_f:,} / ON {on_f:,} = {on_f / off_f:.4f}")
    assert off_f == 4 * 24 * 1 * 64 * 64 * 4 and on_f == off_f // 4
    assert on_f / off_f == 0.25
    assert on_b / off_b < 0.3
    assert off_b - on_b == off_f - on_f            # ONLY the frames shrank


# --------------------------------------------- 5. (d) argparse + config
def test_flag_parses_defaults_off_and_lands_in_config_json(tmp_path, monkeypatch):
    ap = T.build_parser()
    assert ap.parse_args(_argv(tmp_path)).u8_batches is False
    assert ap.parse_args(_argv(tmp_path, "--u8-batches")).u8_batches is True
    _, cfg_off = _train_once(tmp_path, monkeypatch, "off")
    _, cfg_on = _train_once(tmp_path, monkeypatch, "on", "--u8-batches")
    assert cfg_off["u8_batches"] is False and cfg_on["u8_batches"] is True
    # 2 x (4 + 20) x 1 x 64 x 64 x {4 | 1}
    assert cfg_off["frame_batch_bytes_est"] == 2 * 24 * 64 * 64 * 4
    assert cfg_on["frame_batch_bytes_est"] == 2 * 24 * 64 * 64
    # (config.json's `argv` is sys.argv[1:] of the LAUNCHING process — under
    # pytest that is pytest's own line, so the stamp is the key, not argv.)


# ------------------------------------------- 6. (e) 1-step loss identical
def test_one_step_train_loss_is_torch_equal_on_and_off_and_off_equals_head(
        tmp_path, monkeypatch):
    off, _ = _train_once(tmp_path, monkeypatch, "off")
    on, _ = _train_once(tmp_path, monkeypatch, "on", "--u8-batches")
    # CONTROLS: the loss fn received float32 OFF and uint8 ON (the flag was
    # live inside the real DataLoader path), and forwarded identical floats
    assert off["recv_dtype"] == (torch.float32, torch.float32)
    assert on["recv_dtype"] == (torch.uint8, torch.uint8)
    assert on["fwd"] == off["fwd"]
    assert off["fwd"] == {k: HEAD_TRAIN1[k] for k in FRAME_KEYS}   # == HEAD
    # the loss: EXACT. Same device, same bytes, the only difference is where
    # /255 ran — there is no tolerance to state.
    assert torch.equal(on["loss"], off["loss"]), (float(on["loss"]),
                                                  float(off["loss"]))
    assert struct.pack("<f", float(off["loss"])).hex() == HEAD_TRAIN1["loss_hex"]
    assert float(off["loss"]) == HEAD_TRAIN1["loss"]


def test_one_step_train_on_cuda_loss_is_torch_equal_on_and_off(tmp_path,
                                                                monkeypatch):
    """The run converts on CUDA. ADMISSIBILITY CONTROL FIRST: two OFF runs
    must agree bit for bit (MEASURED so on this rig, 2026-09-02); if a torch
    build ever makes the CUDA smoke train non-deterministic, the loss
    comparison is skipped WITH THAT REASON rather than failing on noise —
    the forwarded-input identity (the tensor-divisor requirement in action)
    is asserted regardless."""
    _cuda_free_or_skip()
    off, _ = _train_once(tmp_path, monkeypatch, "off", "--device", "cuda")
    on, _ = _train_once(tmp_path, monkeypatch, "on", "--device", "cuda",
                        "--u8-batches")
    assert off["recv_dtype"] == (torch.float32, torch.float32)
    assert on["recv_dtype"] == (torch.uint8, torch.uint8)
    assert on["fwd"] == off["fwd"]              # bit-identical inputs on CUDA
    off2, _ = _train_once(tmp_path, monkeypatch, "off2", "--device", "cuda")
    if not torch.equal(off["loss"], off2["loss"]):
        pytest.skip("CUDA 1-step train is not run-to-run deterministic on "
                    "this box; a loss equality is inadmissible here (the "
                    "converted inputs WERE asserted bit-identical)")
    assert torch.equal(on["loss"], off["loss"]), (float(on["loss"]),
                                                  float(off["loss"]))


# ---------------------------------- 7. a missed conversion is LOUD, never silent
def test_uint8_batch_without_conversion_fails_loud_in_the_encoder():
    cfg, _eps, _ds_off, ds_on = _rig()
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg).eval()
    b_on = torch.utils.data.default_collate([ds_on[0], ds_on[1]])
    with torch.no_grad():
        with pytest.raises(RuntimeError):            # conv dtype mismatch
            model(b_on["frames"], v0=b_on["pose_last"][:, 3])
        with pytest.raises(RuntimeError):
            model.core.encode_pooled(b_on["future_frames"][:, 4])
        # and the converted batch runs — the same call, dtype fixed
        out = model(T.frames_to_device(b_on["frames"], "cpu"),
                    v0=b_on["pose_last"][:, 3])
    assert out["traj"].shape[0] == 2


# ------------------------------- 8. a real worker ships uint8 bit-exactly
def test_worker_collate_ships_uint8_bit_exact_through_shared_memory():
    _cfg, _eps, ds_off, ds_on = _rig()
    b_off = torch.utils.data.default_collate([ds_off[0], ds_off[1]])
    dl = torch.utils.data.DataLoader(ds_on, batch_size=2, shuffle=False,
                                     num_workers=1, timeout=300)
    it = iter(dl)
    try:
        b = next(it)
    finally:
        del it
    for k in FRAME_KEYS:
        assert b[k].dtype == torch.uint8, k
        assert torch.equal(T.frames_to_device(b[k], "cpu"), b_off[k]), k
    for k in set(b_off) - set(FRAME_KEYS):
        if torch.is_tensor(b_off[k]):
            assert torch.equal(b[k], b_off[k]), k


# ----------------------------------------- 9. the live shape, from arithmetic
def test_live_shape_bytes_match_the_measured_shm_segments():
    px = LIVE["ch"] * LIVE["h"] * LIVE["w"]
    fp32 = {"frames": LIVE["batch"] * LIVE["window"] * px * 4,
            "future_frames": LIVE["batch"] * LIVE["max_horizon"] * px * 4}
    u8 = {k: v // 4 for k, v in fp32.items()}
    for k in FRAME_KEYS:      # the diagnosis: the segments ARE these tensors
        assert fp32[k] + SHM_SEGMENT_HEADER_B == MEASURED_SHM_SEGMENTS_B[k], k
    per_batch_fp32, per_batch_u8 = sum(fp32.values()), sum(u8.values())
    n = LIVE["workers"] * LIVE["prefetch"]
    print(f"\n[u8] live shape per collated batch: fp32 {per_batch_fp32:,} B "
          f"({per_batch_fp32 / 1e9:.3f} GB) -> uint8 {per_batch_u8:,} B "
          f"({per_batch_u8 / 1e9:.3f} GB); x{n} in flight: "
          f"{n * per_batch_fp32 / 1e9:.2f} GB -> {n * per_batch_u8 / 1e9:.2f} GB")
    assert per_batch_fp32 == 3_303_014_400 and per_batch_u8 == 825_753_600
    assert n * per_batch_fp32 == 19_818_086_400        # the measured 19.8 GB
    assert n * per_batch_u8 == 4_954_521_600


# ------------------------------------------ 10. the cast's home is untouched
def test_the_base_contract_still_owns_the_cast_and_the_twin_matches_it():
    src = (ROOT / "tanitad" / "data" / "_contract.py").read_text(encoding="utf-8")
    body = src[src.index("class EpisodeWindowDataset"):]
    assert body.count("to_float_frames(") == 2, \
        "the cast moved: revisit FailLoudWindowDataset._window_u8 (its twin)"
    assert "return x.float().div(255.0) if x.dtype == torch.uint8 else x" in src
    # and REF-B's own trainer never flips the switch (its path is unchanged)
    rb = (ROOT / "scripts" / "refb_train.py").read_text(encoding="utf-8")
    assert rb.count("u8_frames = True") == 0
    assert RB.FailLoudWindowDataset.u8_frames is False
