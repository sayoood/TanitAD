"""H-RANK-8 data path: newest-frame-only decode keeps the row<->pose alignment
the manifest already encodes, while dropping 9 channels to 3.

⛔ The invariant that could break SILENTLY: with n_stack=3 the manifest slices
poses[2:], so row j must pair with RAW frame j+2 in BOTH modes. If the single-
frame path used raw frame j instead, every latent would be two ticks (0.2 s)
behind its pose and every probe downstream would read "no dynamics" — a false
negative manufactured by the data path, not the model.
"""
import torch
from tanitad.data.v2_dataset import _decode_stacked, stack_frames
import torchvision.io as tvio


def _png_frames(n: int, h: int = 8, w: int = 8):
    """n distinct PNG frames whose pixel value encodes the frame index."""
    bufs, lens = [], []
    for i in range(n):
        img = torch.full((3, h, w), fill_value=i * 10, dtype=torch.uint8)
        b = tvio.encode_png(img)
        bufs.append(b); lens.append(int(b.numel()))
    buf = torch.cat(bufs)
    offs = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()))
    return buf, offs


def test_newest_only_row_j_is_raw_frame_j_plus_k():
    buf, offs = _png_frames(8)
    n_stack, a, b = 3, 1, 4
    stacked = _decode_stacked(buf, offs, n_stack, a, b, codec="png")
    single = _decode_stacked(buf, offs, n_stack, a, b, codec="png", newest_only=True)
    assert stacked.shape == (3, 9, 8, 8) and single.shape == (3, 3, 8, 8)
    # the NEWEST frame of the stack is channels [6:9]; single must equal it
    assert torch.equal(single, stacked[:, 6:9]), \
        "newest_only must return exactly the last frame of each stack (alignment)"
    # and that frame's pixel value encodes raw index j + k
    for r in range(b - a):
        j = a + r
        assert int(single[r, 0, 0, 0]) == (j + n_stack - 1) * 10, \
            f"row {r} (j={j}) paired with the wrong raw frame"


def test_NEGATIVE_CONTROL_the_oldest_frame_would_fail_this_check():
    buf, offs = _png_frames(8)
    stacked = _decode_stacked(buf, offs, 3, 1, 4, codec="png")
    oldest = stacked[:, 0:3]
    single = _decode_stacked(buf, offs, 3, 1, 4, codec="png", newest_only=True)
    assert not torch.equal(single, oldest), \
        "the test cannot distinguish newest from oldest — fixture is degenerate"
