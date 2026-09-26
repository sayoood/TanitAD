"""G-HYG — config dataclasses REFUSE undeclared attributes at assignment (SPEC_REFCV7 §2).

⛔⛔ THE DEFECT CLASS THIS CLOSES, MEASURED THREE TIMES ON ONE TRAINER.

A config dataclass is not frozen, so ``cfg.core.encoder.some_lever = value`` SUCCEEDS whether or
not ``some_lever`` is a declared field. An undeclared value lives only in the instance ``__dict__``,
and it dies at the first rebuild that goes through the dataclass's own fields —
``dataclasses.replace``, ``asdict``, a ``cfg_from_dict`` round trip — while every reader that uses
``getattr(cfg, name, default)`` silently falls back to the default. The run's argv and its
``config.json`` state the lever; the model never has it.

* 2026-09-17 — the ``--image-hw`` rebuild dropped four trunk levers (D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS).
* 2026-09-19 — the trunk memory levers were first wired as ad-hoc attributes and the trunk
  OOM'd at 22.34 GB (``test_trunk_memory_levers.py::test_the_levers_SURVIVE_a_config_round_trip``).
* 2026-09-26 — ``--equalize-bottom-rows 43`` never reached the trunk of refcv6-r101-s0
  (D-REFCV6-EQUALIZE-DROPPED): ``_pin_trainer_cfg`` set ``trunk_equalize_bottom_rows`` as an
  undeclared attribute and the same ``dataclasses.replace`` dropped it.

Each time the fix declared ONE field and the class stayed open. This module closes the class:

* :func:`strict_fields` — a class decorator that makes ``setattr`` of an undeclared name RAISE
  :class:`UndeclaredConfigAttribute`, naming the class, the attribute and the declared fields.
  The mechanism becomes impossible at the moment of assignment, not detectable weeks later.
* :func:`undeclared_attributes` / :func:`assert_config_hygiene` — a walker for the one route
  the decorator cannot see: objects populated through ``__dict__`` directly (``pickle``,
  ``copy.deepcopy``, ``vars(obj).update``) and dataclasses of other modules that are not
  decorated. It walks a whole config tree and reports every undeclared key, by path.

⭐ Import cost: standard library only, so the model modules (``tanitad.refs.refc`` / ``refc_v3``)
can import it without a cycle. The launch gate imports it too; the names below are its API.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterable

__all__ = ["UndeclaredConfigAttribute", "strict_fields", "is_strict",
           "undeclared_attributes", "assert_config_hygiene"]


class UndeclaredConfigAttribute(AttributeError):
    """``setattr`` of a name that is not a declared field of a strict config dataclass."""


def _declared(cls) -> frozenset:
    return frozenset(getattr(cls, "__dataclass_fields__", {}))


def _is_data_descriptor(cls, name: str) -> bool:
    """A ``property`` with a setter (or any data descriptor) is a declared write path."""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            obj = klass.__dict__[name]
            return hasattr(obj, "__set__")
    return False


def strict_fields(cls):
    """Class decorator: the dataclass REFUSES ``setattr`` of an undeclared attribute.

    Apply it OUTSIDE ``@dataclass``::

        @strict_fields
        @dataclass
        class CNNEncoderConfig:
            ...

    Declared fields (including ``init=False`` ones), dunder names and data descriptors stay
    assignable; ``dataclasses.replace``, the generated ``__init__`` and ``__post_init__`` work
    unchanged because they only assign declared fields. A frozen dataclass already refuses every
    assignment and is returned untouched. Subclasses inherit the guard and their own fields.

    ⚠️ ``pickle`` / ``copy`` restore through ``__dict__`` and bypass ``__setattr__`` by design;
    :func:`assert_config_hygiene` is the check for that route.
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"strict_fields expects a dataclass type, got {cls!r}. Apply it OUTSIDE "
                        f"@dataclass (the decorator order matters).")
    params = getattr(cls, "__dataclass_params__", None)
    if params is not None and params.frozen:
        return cls
    base_setattr = cls.__setattr__

    def __setattr__(self, name: str, value: Any, _base=base_setattr) -> None:
        klass = type(self)
        if (name not in _declared(klass)
                and not (name.startswith("__") and name.endswith("__"))
                and not _is_data_descriptor(klass, name)):
            fields = sorted(_declared(klass))
            raise UndeclaredConfigAttribute(
                f"{klass.__module__}.{klass.__qualname__}.{name} is NOT a declared field. "
                f"An undeclared attribute lives only in this instance's __dict__ and is dropped "
                f"by the first dataclasses.replace / asdict / rebuild while every "
                f"getattr(cfg, {name!r}, default) reader silently takes the default -- the "
                f"D-REFCV6-EQUALIZE-DROPPED mechanism (argv said 43, the trunk built 0). Declare "
                f"`{name}` as a field of {klass.__qualname__} (with the default that means OFF), "
                f"or do not set it. Declared fields: {fields}")
        _base(self, name, value)

    __setattr__.__qualname__ = f"{cls.__qualname__}.__setattr__"
    cls.__setattr__ = __setattr__
    cls.__strict_fields__ = True
    return cls


def is_strict(cls_or_obj) -> bool:
    """True when the (class of the) object refuses undeclared attributes: strict or frozen."""
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    if not dataclasses.is_dataclass(cls):
        return False
    params = getattr(cls, "__dataclass_params__", None)
    return bool(getattr(cls, "__strict_fields__", False) or (params is not None and params.frozen))


def _children(obj) -> Iterable[tuple[str, Any]]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for k, v in getattr(obj, "__dict__", {}).items():
            yield f".{k}", v
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield f"[{i}]", v
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield f"[{k!r}]", v


def undeclared_attributes(root: Any, path: str = "cfg") -> list[tuple[str, str, str]]:
    """Every ``(path, class, attribute)`` in the tree under ``root`` whose attribute is in a
    dataclass instance's ``__dict__`` without being a declared field. ``[]`` means clean.

    Walks dataclass instances, lists, tuples and dicts; never recurses into modules, tensors or
    other objects (a config tree holds plain values). Cycles are cut by identity.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[int] = set()
    stack: list[tuple[str, Any]] = [(path, root)]
    while stack:
        p, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            declared = _declared(type(obj))
            for k in getattr(obj, "__dict__", {}):   # a slotted dataclass has none
                if k not in declared:
                    out.append((p, type(obj).__qualname__, k))
        for suffix, child in _children(obj):
            if dataclasses.is_dataclass(child) or isinstance(child, (list, tuple, dict)):
                stack.append((p + suffix, child))
    return sorted(out)


def assert_config_hygiene(root: Any, where: str = "config") -> None:
    """REFUSE (``SystemExit``) when the config tree carries any undeclared attribute."""
    bad = undeclared_attributes(root)
    if bad:
        lines = "\n  - ".join(f"{p}.{a}  (on {c})" for p, c, a in bad)
        raise SystemExit(
            f"[G-HYG] ⛔ {where}: {len(bad)} UNDECLARED attribute(s) on config dataclasses. Each "
            f"one is a lever the next rebuild drops while argv and config.json still state it "
            f"(D-REFCV6-EQUALIZE-DROPPED). Declare it as a field or remove it:\n  - {lines}")
