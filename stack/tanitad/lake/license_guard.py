"""License-scope enforcement (spec §6, layer 3: the export guard).

Physical partitioning (shards/catalog by ``license_class``) and ShareAlike shard
segregation are layers 1-2. This module is layer 3: a HARD assertion that runs
before ANY egress / HF export and refuses if a single row falls outside the
view's declared scope. A scope violation is a failure, not a warning — the AV
firewall and the commercial-clean gate become invariants of the topology, not a
discipline to remember.
"""

from __future__ import annotations

from typing import Iterable

from tanitad.lake.schema import LICENSE_CLASSES


class LicenseScopeError(PermissionError):
    """Raised when a view/export contains a row outside its declared scope."""


def verify_license_scope(rows: Iterable[dict], allowed_classes: set[str],
                         require_commercial_ok: bool = False,
                         forbid_share_alike: bool = False,
                         context: str = "export") -> int:
    """Assert every row's license is within scope; return the row count.

    - ``allowed_classes``: the only ``license_class`` values permitted (e.g.
      ``{'owned-safe'}`` for a public export; NEVER include gated-confidential).
    - ``require_commercial_ok``: also demand ``commercial_ok`` (excludes ZOD /
      any share-alike) — the commercial-clean gate.
    - ``forbid_share_alike``: reject any ShareAlike row (keeps a proprietary view
      SA-free even inside owned-safe).

    Raises :class:`LicenseScopeError` on the FIRST violation with the offending
    episode id + reason.
    """
    bad = allowed_classes - set(LICENSE_CLASSES)
    if bad:
        raise ValueError(f"unknown license classes in scope: {bad}")
    if "gated-confidential" in allowed_classes:
        raise LicenseScopeError(
            "gated-confidential can NEVER be in an export scope — it is not in "
            "the lake at all (spec §3.3). Refusing to build such a guard.")

    n = 0
    for r in rows:
        n += 1
        eid = r.get("episode_id")
        lc = r.get("license_class")
        if lc not in allowed_classes:
            raise LicenseScopeError(
                f"[{context}] ep {eid}: license_class {lc!r} outside scope "
                f"{sorted(allowed_classes)}")
        if require_commercial_ok and not r.get("commercial_ok", False):
            raise LicenseScopeError(
                f"[{context}] ep {eid}: not commercial_ok (source {r.get('source')!r},"
                f" share_alike={r.get('share_alike')}) — excluded from the "
                f"commercial-clean export")
        if forbid_share_alike and r.get("share_alike", False):
            raise LicenseScopeError(
                f"[{context}] ep {eid}: share_alike row rejected by SA-free scope")
    return n
