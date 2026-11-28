"""refcv9 B3 ``--r9-box-dn`` x the rig warm start (``refcv9_perc_rig.warm_start_perc_rig``): the DN embeddings are built BEFORE the
warm start (``box_b3_flag.attach``) and refcv8 has none, so they must be NAMED new keys -- and nothing else may go missing."""
import pytest
import torch
from torch import nn

from tanitad.models.box_b3 import BoxB3Config
from tanitad.refs import refcv9_perc_rig as rig


class _Box(nn.Module):
    def __init__(self, dn: bool, extra: bool = False):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        if dn:
            self.dn_embed = nn.Linear(6, 4)
            self.dn_cls_embed = nn.Embedding(3, 4)
        if extra:
            self.other = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self, dn: bool, cfg, extra: bool = False):
        super().__init__()
        self.box3d = _Box(dn, extra)
        if cfg is not None:
            self._r9_box_b3 = cfg


def _source(tmp_path):
    src = _Model(dn=False, cfg=None)
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "step": 7}, str(p))
    return src, str(p)


def test_off_model_has_no_exemption():
    assert rig.b3_exemptions(_Model(dn=False, cfg=None)) == ()
    assert rig.b3_exemptions(_Model(dn=False, cfg=BoxB3Config(quality=True))) == ()


def test_dn_on_loads_and_names_only_the_dn_modules(tmp_path):
    src, p = _source(tmp_path)
    m = _Model(dn=True, cfg=BoxB3Config(dn=True))
    assert rig.b3_exemptions(m) == ("box3d.dn_cls_embed.", "box3d.dn_embed.")
    stamp = rig.warm_start_perc_rig(m, p)
    assert stamp["n_loaded"] == 2
    assert torch.equal(m.box3d.proj.weight, src.box3d.proj.weight)


def test_red_dn_modules_without_the_exemption_refuse(tmp_path):
    _, p = _source(tmp_path)
    m = _Model(dn=True, cfg=BoxB3Config(dn=False))          # the modules exist, the exemption is off: the tip's refusal
    with pytest.raises(SystemExit, match=r"rig key\(s\) the source lacks"):
        rig.warm_start_perc_rig(m, p)


def test_a_key_beyond_the_named_modules_still_refuses(tmp_path):
    _, p = _source(tmp_path)
    m = _Model(dn=True, cfg=BoxB3Config(dn=True), extra=True)
    with pytest.raises(SystemExit):
        rig.warm_start_perc_rig(m, p)


def test_dn_on_without_modules_refuses():
    with pytest.raises(SystemExit, match="no dn_embed"):
        rig.b3_exemptions(_Model(dn=False, cfg=BoxB3Config(dn=True)))
