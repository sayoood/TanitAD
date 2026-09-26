"""⛔⛔ G-HYG — config dataclasses REFUSE undeclared attributes at assignment (SPEC_REFCV7 §2).

The mechanism behind D-REFCV6-EQUALIZE-DROPPED, and twice before it (the `--image-hw` rebuild
2026-09-17, the trunk memory levers 2026-09-19): a non-frozen dataclass accepts
``cfg.x.lever = v`` for ANY name, the value lives only in ``__dict__``, and the first
``dataclasses.replace`` drops it while ``getattr(cfg, name, default)`` readers take the default.

Pinned here, with literals: the exact set of strict classes; an undeclared assignment RAISES and
names the attribute; declared fields, ``replace``, ``__init__`` and ``__post_init__`` still work;
the walker finds a ``__dict__`` injection the decorator cannot see; ``_pin_trainer_cfg`` refuses
one; and the deliberate-regression arm (an un-decorated class) accepts the assignment -- i.e. the
guard, not the class, is what refuses.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import config_hygiene as hyg  # noqa: E402

#: every config dataclass of the two model modules the trainer pins -- LITERAL, so a new config
#: class added without the guard fails here by name.
STRICT = {
    refc: ("CNNEncoderConfig", "MeasurementConfig", "TrajectoryConfig", "AnchorConfig",
           "DecoderConfig", "LawConfig", "StrategicCtxConfig", "ImaginationConfig",
           "LanConfig", "SelectionConfig", "RefCConfig"),
    v3: ("RefCV3Config",),
}


@pytest.mark.parametrize("mod,name", [(m, n) for m, ns in STRICT.items() for n in ns])
def test_every_refc_and_refc_v3_config_class_is_STRICT(mod, name):
    assert hyg.is_strict(getattr(mod, name)), f"{mod.__name__}.{name} accepts ad-hoc attributes"


def test_no_OTHER_dataclass_in_those_modules_escaped_the_list():
    for mod, names in STRICT.items():
        found = {n for n, o in vars(mod).items()
                 if isinstance(o, type) and dataclasses.is_dataclass(o) and o.__module__ == mod.__name__}
        assert found == set(names), (mod.__name__, sorted(found ^ set(names)))


@pytest.mark.parametrize("target", ["encoder", "decoder", "anchors", "core", "v3"])
def test_MUTATION_an_undeclared_attribute_RAISES_and_names_itself(target):
    cfg = v3.refc_v3_smoke_config(True)
    obj = {"encoder": cfg.core.encoder, "decoder": cfg.core.decoder,
           "anchors": cfg.core.anchors, "core": cfg.core, "v3": cfg}[target]
    with pytest.raises(hyg.UndeclaredConfigAttribute, match="trunk_equalize_bottom_rowz"):
        obj.trunk_equalize_bottom_rowz = 43          # one letter off a real field
    assert "trunk_equalize_bottom_rowz" not in vars(obj)


def test_the_error_is_an_AttributeError_so_hasattr_style_code_keeps_its_meaning():
    assert issubclass(hyg.UndeclaredConfigAttribute, AttributeError)


def test_declared_fields_replace_init_and_post_init_are_UNCHANGED():
    c = refc.CNNEncoderConfig()
    c.trunk_equalize_bottom_rows = 43                # a declared field: assignable
    c.image_width = 1024
    r = dataclasses.replace(c, image_size=416)
    assert (r.trunk_equalize_bottom_rows, r.image_size, r.image_width) == (43, 416, 1024)
    cfg = v3.refc_v3_smoke_config(True)              # __init__ + default factories + pins
    cfg.core.decoder.sampler = "ddim"
    assert cfg.core.decoder.sampler == "ddim"


def test_the_DELIBERATE_REGRESSION_an_undecorated_class_ACCEPTS_the_attribute():
    """Without the guard the same assignment succeeds silently -- the historical state. This is
    what proves the refusal above comes from G-HYG and not from something else."""
    @dataclasses.dataclass
    class Loose:
        a: int = 0
    x = Loose()
    x.b = 1
    assert vars(x)["b"] == 1
    Strict = hyg.strict_fields(dataclasses.dataclass(type("Strict", (), {"__annotations__": {"a": int}, "a": 0})))
    with pytest.raises(hyg.UndeclaredConfigAttribute):
        Strict().b = 1


def test_a_frozen_dataclass_is_left_alone_and_counts_as_strict():
    @dataclasses.dataclass(frozen=True)
    class F:
        a: int = 0
    assert hyg.strict_fields(F) is F and hyg.is_strict(F)


def test_a_non_dataclass_is_refused_by_the_decorator():
    with pytest.raises(TypeError, match="OUTSIDE @dataclass"):
        hyg.strict_fields(type("NotADataclass", (), {}))


def test_the_WALKER_finds_a_dict_injection_the_decorator_cannot_see():
    cfg = v3.refc_v3_smoke_config(True)
    assert hyg.undeclared_attributes(cfg) == []
    vars(cfg.core.encoder)["trunk_synthetic_lever"] = 7          # pickle / copy / __dict__ route
    assert hyg.undeclared_attributes(cfg) == [("cfg.core.encoder", "CNNEncoderConfig",
                                               "trunk_synthetic_lever")]
    with pytest.raises(SystemExit, match="trunk_synthetic_lever"):
        hyg.assert_config_hygiene(cfg, where="test")


def test_pin_trainer_cfg_REFUSES_a_surviving_dict_injection(monkeypatch):
    """The walk at the end of `_pin_trainer_cfg`: an injection that the rebuild does not drop
    (no --image-hw) reaches the end of the pin and must be refused there."""
    import refc_v3_train as T
    real = T._pin_refcv6_tactical          # runs AFTER the --image-hw rebuild

    def inject(cfg, args):
        vars(cfg.core.decoder)["sel_synthetic_lever"] = 1
        return real(cfg, args)
    monkeypatch.setattr(T, "_pin_refcv6_tactical", inject)
    a = T.build_parser().parse_args(["--arm", "hier", "--out", "z"])
    with pytest.raises(SystemExit, match="sel_synthetic_lever"):
        T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), a)


def test_every_pin_assignment_targets_a_DECLARED_field():
    """The static census of the pin helpers (the audit's `audit_adhoc_attrs.py`, as a test):
    every `cfg.<...>.<attr> = ...` in every `_pin_*` helper names a declared field. The runtime
    guard only sees the branches a given argv reaches; this sees every branch."""
    import ast
    import inspect
    import refc_v3_train as T
    cfg = v3.refc_v3_smoke_config(True)
    # ⚠️ DrivoR-T's `hc` (its head config) is out of scope (SPEC_REFCV7 §6.1, rename pending):
    # its assignments are skipped here, never guessed.
    roots = {"cfg": cfg, "core": cfg.core}
    bad, n = [], 0
    helpers = sorted(k for k, v in vars(T).items() if k.startswith("_pin_") and callable(v))
    assert {"_pin_trainer_cfg", "_pin_refcv5_seams", "_pin_refcv6_tactical"} <= set(helpers)
    for name in helpers:
        src = inspect.getsource(getattr(T, name))
        for node in ast.walk(ast.parse(src)):
            tgts = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
            for t in tgts:
                chain = []
                while isinstance(t, ast.Attribute):
                    chain.append(t.attr)
                    t = t.value
                if not isinstance(t, ast.Name) or t.id not in roots or not chain:
                    continue
                chain.reverse()
                obj = roots[t.id]
                for part in chain[:-1]:
                    obj = getattr(obj, part)
                n += 1
                if dataclasses.is_dataclass(obj) and chain[-1] not in {
                        f.name for f in dataclasses.fields(obj)}:
                    bad.append(f"{name}: {t.id}.{'.'.join(chain)}")
    assert n >= 80, f"the census saw only {n} assignments -- it is not reading the pins"
    assert bad == [], bad
