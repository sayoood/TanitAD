"""O2-pre — the readout's non-tiling pooling must be EXACT and ONNX-EXPORTABLE.

⛔ THE BLOCKER (MEASURED 2026-08-03 on Thor, `thor_b1b_fastpath_probe.json`): the encoder does not
export to ONNX at the DEPLOYED 176x624 geometry at all —

    SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
    output size that are not factor of input size

at both MHA-fastpath settings and both opsets. At 176x624 with patch 16 the token grid is 11x39 and
the readout grid is 4x4, which does not tile, so `SpatialGridReadout` fell back to
`nn.AdaptiveAvgPool2d`. That blocked backlog item **O2** (a TensorRT engine for the encoder) — the
designated fallback for the largest lever in the whole Thor result.

The replacement is a pair of constant averaging matrices. These tests exist to prove it is a
**re-expression, not an approximation**, because an "export fix" that quietly changes what the
trained weights see is the more expensive failure:

  1. it reproduces `F.adaptive_avg_pool2d` at the deployed grid and several others;
  2. the DEPLOYED tiling path (v1 @ 256px, 16x16 -> 4x4) is byte-for-byte untouched;
  3. the derived matrices never enter the state_dict, so every existing checkpoint still loads
     STRICT;
  4. the module actually exports to ONNX at the deployed geometry, with parity.
"""
import io

import pytest
import torch
import torch.nn.functional as F

from tanitad.models.readout import SpatialGridReadout, _adaptive_avg_matrix


# ------------------------------------------------------------------ 1. exactness
@pytest.mark.parametrize("n_in,n_out", [(11, 4), (39, 4), (16, 4), (24, 4), (42, 4), (7, 3)])
def test_matrix_reproduces_adaptive_pool_1d(n_in, n_out):
    torch.manual_seed(0)
    x = torch.randn(3, 5, n_in)
    ref = F.adaptive_avg_pool1d(x, n_out)
    got = torch.matmul(x, _adaptive_avg_matrix(n_in, n_out).transpose(0, 1))
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)


def test_matrix_bins_are_the_pytorch_bins_and_they_OVERLAP():
    """⚠️ Adaptive bins are NOT a partition — ``start`` floors and ``end`` ceils, so neighbours
    SHARE input cells. 11 -> 4 gives 3/4/4/3 = 14 cell-uses over 11 cells.

    This test earned its place: the first draft of it asserted a tidy 3/3/3/2 partition, and the
    measurement said otherwise. An 'export fix' written from the partition intuition would have
    silently changed what the trained readout sees.
    """
    m = _adaptive_avg_matrix(11, 4)
    counts = (m > 0).sum(1).tolist()
    assert counts == [3, 4, 4, 3]
    assert sum(counts) > 11, "bins must overlap — this is PyTorch's definition, not a bug"
    assert torch.allclose(m.sum(1), torch.ones(4))          # every row is a proper average
    # and the deployed width: 39 -> 4 is 10/11/11/10, also overlapping
    assert (_adaptive_avg_matrix(39, 4) > 0).sum(1).tolist() == [10, 11, 11, 10]


@pytest.mark.parametrize("th,tw", [(11, 39), (11, 42), (13, 41)])
def test_readout_matches_adaptive_pool_at_non_tiling_grids(th, tw):
    torch.manual_seed(0)
    d_model, grid, d_r = 32, 4, 8
    ro = SpatialGridReadout(th * tw, d_model, grid=grid, d_readout=d_r,
                            token_grid=(th, tw)).eval()
    assert not ro.exact_pool, "this test is meaningless on a tiling grid"
    tok = torch.randn(2, th * tw, d_model)

    x = tok.transpose(1, 2).reshape(2, d_model, th, tw)
    ref_pooled = F.adaptive_avg_pool2d(x, (grid, grid))
    ref = ro.proj(ref_pooled.flatten(2).transpose(1, 2)).flatten(1)

    got = ro(tok)
    assert got.shape == ref.shape == (2, grid * grid * d_r)
    # measured, not asserted: report the residual so a regression is legible
    rel = (got - ref).norm() / ref.norm()
    assert rel < 1e-6, f"pooling re-expression drifted: rel-err {rel:.3e}"


# ------------------------------------------------------------------ 2. deployed path untouched
def test_tiling_path_is_bit_identical_to_avgpool():
    """v1 @ 256px is 16x16 tokens onto 4x4 — it tiles, and must not be touched at all."""
    torch.manual_seed(0)
    ro = SpatialGridReadout(256, 32, grid=4, d_readout=8).eval()
    assert ro.exact_pool
    assert not hasattr(ro, "pool_mh")            # no matrices built on the exact path
    tok = torch.randn(2, 256, 32)
    x = tok.transpose(1, 2).reshape(2, 32, 16, 16)
    ref = ro.proj(torch.nn.AvgPool2d((4, 4))(x).flatten(2).transpose(1, 2)).flatten(1)
    assert torch.equal(ro(tok), ref)             # BIT-identical, not just close


# ------------------------------------------------------------------ 3. checkpoints still load
def test_derived_matrices_are_not_in_the_state_dict():
    """A persistent buffer here would break a STRICT load of every existing checkpoint."""
    ro = SpatialGridReadout(11 * 39, 32, grid=4, d_readout=8, token_grid=(11, 39))
    keys = set(ro.state_dict())
    assert not {k for k in keys if "pool_m" in k}, keys
    assert keys == {"proj.weight", "proj.bias"}
    # and a state_dict from the module loads back STRICT
    ro2 = SpatialGridReadout(11 * 39, 32, grid=4, d_readout=8, token_grid=(11, 39))
    ro2.load_state_dict(ro.state_dict())         # would raise on an unexpected key


# ------------------------------------------------------------------ 4. it actually exports
def test_readout_exports_to_onnx_at_the_deployed_geometry():
    """THE regression this file exists for: 11x39 -> 4x4 must reach an ONNX graph."""
    th, tw, d_model = 11, 39, 32
    ro = SpatialGridReadout(th * tw, d_model, grid=4, d_readout=8,
                            token_grid=(th, tw)).eval()
    tok = torch.randn(1, th * tw, d_model)
    buf = io.BytesIO()
    torch.onnx.export(ro, (tok,), buf, input_names=["tokens"], output_names=["state"],
                      opset_version=17, dynamo=False)
    assert buf.tell() > 0
    onnx = pytest.importorskip("onnx")
    g = onnx.load_from_string(buf.getvalue())
    ops = {n.op_type for n in g.graph.node}
    assert not any("AveragePool" in o or "Adaptive" in o for o in ops), ops
    assert "MatMul" in ops or "Gemm" in ops, ops

    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    y = torch.as_tensor(sess.run(["state"], {"tokens": tok.numpy()})[0])
    with torch.no_grad():
        ref = ro(tok)
    rel = float((y - ref).norm() / ref.norm())
    assert rel < 1e-5, f"ORT-vs-eager parity {rel:.3e}"
