"""``E-DDA-2b`` piece 5 — SELECTOR-SIDE FAN AUGMENTATION AND THE FOREIGN BANK.

The fifth of the five DiffusionDriveV2 mechanisms the 2026-09-05 analysis
(`…/Research/2026-09-05-diffusiondrive-v2-analysis/RESULT.md` §3.2) marks
**usable now**, and the only one that was still absent from the tree on
2026-09-10 (the other four had landed on 2026-09-05/07; see this package's
`RESULT.md` §1).

## What V2 actually does — read from the BANKED source, not from a summary

`PUBLISHED-CODE`: `hustvl/DiffusionDriveV2@1cd12a1`,
`navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`, banked in-repo at
`…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/diffusiondrivev2_model_sel.py`,
sha256 `2218e4ee79f5952f4edf474c3c2796f91f0cdaf593b3ce987cec5729e1c310b6`
(re-verified against `raw/ddv2_fetched_sha256.txt` on 2026-09-10 — content, not
presence).

`add_mul_noise` (`:1270-1287`), per augmentation round:

1. **ONE std scalar for the entire round**, `sigma ~ U(std_min, std_max)`, drawn as
   `torch.empty(1).uniform_(...).item()` — ⚠️ **not per candidate and not per
   batch row.** Every candidate in the round shares it.
2. **TWO Gaussian scalars per (row, candidate)** — `[B, N, 1, 1]` "horizon" and
   `[B, N, 1, 1]` "vert", each `randn * sigma + 1.0`, concatenated on the last axis
   and repeated over the waypoint axis.
3. `aug = cand * mul` — purely multiplicative, and the ORIGINAL fan is kept as
   copy 0 (`diffusion_output_aug_list = [diffusion_output]`).

⇒ the augmented family is the SAME two-scalar (stretch-along, stretch-lateral)
object the RL stage explores in (`D-DDV2-CODE-2`,
`rl/refcv3_adapter.sample_offsets(noise_mode="two_scalar")`) — one mechanism, two
sites. Recipe values: **train `n_aug=2, U(0.1, 0.2)`** (`:1346`), **test
`n_aug=3, U(0.1, 0.3)`** (`:1444`, the defaults).

`get_vocab_pdm_subscores` (`:1202-1268`) + its call site (`:1352-1357`):

4. `keep_num = int(G_bank * (1 - dropout_ratio))`, an **independent random
   permutation per batch row**, and the kept foreign trajectories are
   **concatenated onto the candidate axis** after the augmented fan.
   ⭐ Cross-check that does not re-run V2's own arithmetic: the analysis reports
   *"`16384.npy[:, ::5]` = 3,277 trajectories … `dropout_ratio = 0.99` ≈ 32 per
   scene"*, and `int(3277 * 0.01) = 32` — the formula reproduces a number
   published from the other direction.
5. The bank's `comfort` sub-score is **filled with −1** and masked, because
   NAVSIM's precomputed vocabulary does not carry it.

⛔⛔ **AND THE FACT THE ANALYSIS DOES NOT STATE, WHICH DECIDES THE PORT'S
DEFAULT: THE FOREIGN BANK IS TRAIN-ONLY.** MEASURED 2026-09-10 by enumerating
every `vocab` reference in that file — `:930`, `:932`, `:1202-1206`, `:1352`,
`:1354` — **all of them inside `forward_train_rl` (`:1288-1389`) or the helper it
calls.** `forward_test_rl` (`:1390-`) applies `add_mul_noise` (`:1444`) and
mixes **no vocabulary at all**. ⇒ :data:`AugmentConfig.foreign_train_only`
defaults to ``True``, and :func:`augment_fan` refuses to mix a bank at eval
unless an operator turns that off **by name**.

⛔ **WHY THAT IS NOT A DETAIL.** A foreign candidate is BY CONSTRUCTION not
something the generator can emit. If the selector's argmax may land on one at
inference, the arm's "selected trajectory" is a trajectory the model does not
produce — the reported number would belong to the bank, not to REF-C. That is
the same class as scoring a staler object than the one emitted
(`D-REFC-DDAUDIT-3`), pointed the other way. :func:`assert_pick_emittable` is the
door lock and :func:`augment_fan` returns ``emittable`` so a caller cannot claim
it did not know.

## Everything is OFF at the defaults, and OFF draws no random numbers

``AugmentConfig()`` is ``n_aug=0, foreign_frac=0.0`` ⇒ :func:`augment_fan`
returns **the input tensor object itself** and consumes **zero** RNG draws, so a
downstream stream is bit-identical to a run in which this module was never
called. That is stronger than "the output is equal" and it is the property
`test_refc_selector_aug.py` pins, because an augmentation that quietly advanced
the global generator would change every arm that came after it.

Evidence class: the V2 mechanism is `PUBLISHED-CODE` (above, content-verified);
this port and every number in its docstring are `MEASURED` (ours), pinned by
``stack/tests/test_refc_selector_aug.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

__all__ = [
    "NATIVE", "FOREIGN", "AugmentConfig", "add_mul_noise", "mix_foreign_bank",
    "augment_fan", "assert_pick_emittable", "origin_summary",
]


#: Origin code of a candidate the generator actually emitted. Augmentation
#: rounds are ``1 .. n_aug``; the foreign bank is :data:`FOREIGN`.
NATIVE: int = 0
#: ⛔ Origin code of a candidate that came from the foreign bank. NEGATIVE on
#: purpose: an augmentation round index can never collide with it, so
#: ``origin < 0`` is the whole "not emittable" predicate and there is no second
#: spelling of it to drift.
FOREIGN: int = -1


@dataclass(frozen=True)
class AugmentConfig:
    """Every knob that changes what the selector SEES. Serialised whole by
    :meth:`as_dict` into ``config.json['seams']['selector_aug']``.

    ⛔ Every field defaults to the CURRENT behaviour (no augmentation, no bank),
    so constructing this config and calling :func:`augment_fan` on an existing
    arm changes nothing — not the tensor, and not the RNG stream.
    """

    #: Augmented copies APPENDED to the native fan. ``0`` = OFF. V2: 2 at train,
    #: 3 at test (`_model_sel.py:1346`, `:1444`).
    n_aug: int = 0
    #: The round's shared multiplicative std is drawn ``U(std_min, std_max)``.
    #: V2: train ``(0.1, 0.2)``, test ``(0.1, 0.3)``.
    std_min: float = 0.1
    std_max: float = 0.2

    #: ⚠️ V2 draws ONE std per round for the WHOLE BATCH
    #: (`torch.empty(1).uniform_(...).item()`, `:1272`). ``True`` draws one per
    #: batch row instead — a DEVIATION from the released code, offered because a
    #: shared scalar makes the augmented set of a whole batch correlated, and
    #: refused as a default because the port would then not be the port.
    std_per_row: bool = False

    #: Fraction of the foreign bank kept per row: V2's ``1 - dropout_ratio``.
    #: ``0.0`` = OFF. V2 uses ``0.01`` (dropout 0.99).
    foreign_frac: float = 0.0
    #: ⛔ The bank is TRAIN-ONLY in the released code (see the module docstring).
    #: Turning this off is how a foreign trajectory reaches an inference pick.
    foreign_train_only: bool = True

    def __post_init__(self) -> None:
        if self.n_aug < 0:
            raise ValueError(f"n_aug must be >= 0, got {self.n_aug}")
        if self.std_min < 0.0:
            raise ValueError(f"std_min must be >= 0, got {self.std_min}")
        if self.std_max < self.std_min:
            raise ValueError(
                f"std_max ({self.std_max}) < std_min ({self.std_min}) — "
                "`Tensor.uniform_` does not refuse an inverted range, it "
                "returns values outside BOTH bounds, so this is refused here")
        if not (0.0 <= self.foreign_frac <= 1.0):
            raise ValueError(
                f"foreign_frac is a KEPT fraction in [0, 1], got "
                f"{self.foreign_frac}. V2's knob is `dropout_ratio` and this is "
                "its complement — passing 0.99 here keeps 99 %, not 1 %.")

    @property
    def enabled(self) -> bool:
        """True when this config can change anything at all."""
        return self.n_aug > 0 or self.foreign_frac > 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["enabled"] = self.enabled
        d["native_origin_code"] = NATIVE
        d["foreign_origin_code"] = FOREIGN
        d["v2_train_recipe"] = {"n_aug": 2, "std": [0.1, 0.2],
                                "dropout_ratio": 0.99}
        d["v2_test_recipe"] = {"n_aug": 3, "std": [0.1, 0.3],
                               "foreign_bank": "ABSENT — train-only in "
                                               "forward_test_rl (:1390-)"}
        return d


def _check_fan(cand: Tensor, name: str = "cand") -> tuple[int, int, int, int]:
    if cand.dim() != 4 or cand.shape[-1] != 2:
        raise ValueError(f"{name} must be [B, N, S, 2], got {tuple(cand.shape)}")
    b, n, s, _ = cand.shape
    if s < 2:
        raise ValueError(f"{name} needs >= 2 waypoints, got S={s}")
    return b, n, s, 2


def add_mul_noise(cand: Tensor, cfg: AugmentConfig | None = None, *,
                  generator: torch.Generator | None = None
                  ) -> tuple[Tensor, Tensor]:
    """V2's ``add_mul_noise``. ``[B, N, S, 2]`` -> ``([B, N*(1+n_aug), S, 2], origin)``.

    ``origin`` is ``[B, N*(1+n_aug)]`` int64: ``0`` for the native block,
    ``r`` for augmentation round ``r``.

    ⛔ ``n_aug == 0`` returns ``cand`` ITSELF (identity, not a copy) and draws no
    random numbers — see the module docstring.

    ⚠️ The multiplier is **two scalars per (row, candidate)** broadcast over the
    waypoints, so the ratio ``aug / native`` is CONSTANT along the trajectory for
    each axis. That is the property that makes an augmented candidate a smooth,
    drivable stretch of its parent rather than a jittered path, and it is the
    same law the RL stage explores in.
    """
    cfg = cfg or AugmentConfig()
    b, n, s, _ = _check_fan(cand)
    if cfg.n_aug == 0:
        return cand, torch.zeros((b, n), dtype=torch.long, device=cand.device)

    blocks = [cand]
    origins = [torch.full((b, n), NATIVE, dtype=torch.long, device=cand.device)]
    for r in range(1, int(cfg.n_aug) + 1):
        if cfg.std_per_row:
            sigma = torch.empty((b, 1, 1, 1), device=cand.device,
                                dtype=cand.dtype).uniform_(
                cfg.std_min, cfg.std_max, generator=generator)
        else:
            # ⚠️ V2's exact shape: ONE scalar for the whole round.
            sigma = torch.empty((1,), device=cand.device,
                                dtype=cand.dtype).uniform_(
                cfg.std_min, cfg.std_max, generator=generator).reshape(1, 1, 1, 1)
        eps = torch.randn((b, n, 1, 2), generator=generator,
                          device=cand.device, dtype=cand.dtype)
        mul = eps * sigma + 1.0                       # [B, N, 1, 2]
        blocks.append(cand * mul)                     # broadcast over S
        origins.append(torch.full((b, n), r, dtype=torch.long,
                                  device=cand.device))
    return torch.cat(blocks, dim=1), torch.cat(origins, dim=1)


def mix_foreign_bank(cand: Tensor, bank: Tensor, cfg: AugmentConfig | None = None,
                     *, origin: Tensor | None = None,
                     generator: torch.Generator | None = None
                     ) -> tuple[Tensor, Tensor]:
    """Append ``int(G * foreign_frac)`` bank trajectories per row.

    ``bank`` is ``[G, S, 2]`` (shared across the batch) or ``[B, G, S, 2]``
    (per row). Each row draws its OWN permutation, exactly as V2 does
    (`:1249-1253`), so two rows in a batch see different foreign candidates.

    Returns ``(cand_out, origin_out)`` with the appended block marked
    :data:`FOREIGN`.

    ⛔ ``foreign_frac > 0`` with ``keep_num == 0`` RAISES. A knob that is parsed,
    stamped and inert is the ``--wp-index`` failure (3 of 6 knobs did nothing and
    the run record advertised all six); here it would additionally read as
    *"the foreign bank made no difference"*.
    """
    cfg = cfg or AugmentConfig()
    b, n, s, _ = _check_fan(cand)
    if origin is None:
        origin = torch.zeros((b, n), dtype=torch.long, device=cand.device)
    if origin.shape != (b, n):
        raise ValueError(f"origin {tuple(origin.shape)} must be [B, N] = "
                         f"{(b, n)} against cand {tuple(cand.shape)}")
    if cfg.foreign_frac <= 0.0:
        return cand, origin

    if bank.dim() == 3:
        bank = bank.unsqueeze(0).expand(b, *bank.shape)
    if bank.dim() != 4 or bank.shape[0] != b or bank.shape[-1] != 2:
        raise ValueError(
            f"bank must be [G, S, 2] or [B, G, S, 2] with B={b}, got "
            f"{tuple(bank.shape)}")
    if bank.shape[-2] != s:
        # ⛔ THE GEOMETRY TRAP, REFUSED RATHER THAN INTERPOLATED. A bank at a
        # different horizon or stride is a DIFFERENT quantity (CLAUDE.md's
        # derived-constant rule: HORIZON 7 -> 8 turned a "reproduction" into a
        # different experiment). Re-derive the bank at the fan's geometry.
        raise ValueError(
            f"bank has S={bank.shape[-2]} waypoints, the fan has S={s}. A "
            "foreign vocabulary at a different horizon/stride is not the same "
            "object; rebuild it at the fan's geometry rather than resampling "
            "here, where the choice would be invisible in the run record.")

    g = int(bank.shape[1])
    keep = int(g * float(cfg.foreign_frac))
    if keep <= 0:
        raise ValueError(
            f"foreign_frac={cfg.foreign_frac} over a bank of G={g} keeps "
            f"int({g} * {cfg.foreign_frac}) = 0 trajectories — the knob would "
            "be ON in the run record and INERT in the run. Raise the fraction "
            "or supply a larger bank.")

    idx = torch.stack(
        [torch.randperm(g, generator=generator, device=bank.device)[:keep]
         for _ in range(b)], dim=0)                                   # [B, keep]
    gath = idx.unsqueeze(-1).unsqueeze(-1).expand(b, keep, s, 2)
    picked = torch.gather(bank, 1, gath)
    out = torch.cat([cand, picked.to(cand.dtype)], dim=1)
    origin_out = torch.cat(
        [origin, torch.full((b, keep), FOREIGN, dtype=torch.long,
                            device=cand.device)], dim=1)
    return out, origin_out


def augment_fan(cand: Tensor, cfg: AugmentConfig | None = None, *,
                bank: Tensor | None = None, training: bool = True,
                generator: torch.Generator | None = None) -> dict:
    """The whole stage-II candidate set, in V2's order: native -> augmented -> foreign.

    Returns a dict::

        cand        [B, N_total, S, 2]   what the selector scores
        origin      [B, N_total]         NATIVE / round index / FOREIGN
        native      [B, N_total] bool    origin == NATIVE
        emittable   [B, N_total] bool    origin >= NATIVE (⛔ the pick guard)
        n_native / n_aug_cands / n_foreign / n_total
        provenance  dict                 what ran, what did not, and WHY

    ⛔ ``foreign_frac > 0`` with ``bank=None`` RAISES — an ON knob with no data
    is the inert-knob class again, and here it would silently become "the bank
    did nothing".

    ⚠️ ``training=False`` with ``foreign_train_only=True`` (the defaults) does
    NOT mix the bank, and says so in ``provenance['foreign_skipped']`` rather
    than dropping it silently. That is the released behaviour
    (`forward_test_rl` mixes no vocabulary), not a safety opinion.
    """
    cfg = cfg or AugmentConfig()
    b, n, _, _ = _check_fan(cand)

    prov: dict = {"cfg": cfg.as_dict(), "training": bool(training),
                  "foreign_skipped": None, "aug_applied": cfg.n_aug > 0}

    if cfg.foreign_frac > 0.0 and bank is None:
        raise ValueError(
            f"foreign_frac={cfg.foreign_frac} but no `bank` was supplied. A "
            "declared-and-inert knob is exactly the failure this library "
            "refuses; pass the bank or set foreign_frac=0.0.")

    out, origin = add_mul_noise(cand, cfg, generator=generator)

    if cfg.foreign_frac > 0.0:
        if cfg.foreign_train_only and not training:
            prov["foreign_skipped"] = (
                "foreign_train_only=True and training=False — the released "
                "code mixes no vocabulary in forward_test_rl (:1390-); every "
                "vocab reference is inside forward_train_rl")
        else:
            out, origin = mix_foreign_bank(out, bank, cfg, origin=origin,
                                           generator=generator)

    native = origin == NATIVE
    emittable = origin >= NATIVE
    n_total = int(out.shape[1])
    n_foreign = int((~emittable).sum(dim=1).max()) if n_total else 0
    return {
        "cand": out,
        "origin": origin,
        "native": native,
        "emittable": emittable,
        "n_native": int(n),
        "n_aug_cands": int(n * cfg.n_aug),
        "n_foreign": n_foreign,
        "n_total": n_total,
        "provenance": prov,
    }


def assert_pick_emittable(sel_idx: Tensor, origin: Tensor, *,
                          where: str = "selector pick") -> None:
    """⛔ Refuse a pick that landed on a candidate the generator cannot emit.

    ``sel_idx`` ``[B]`` into the candidate axis; ``origin`` ``[B, N_total]``.

    A foreign candidate is a trajectory from someone else's vocabulary. Emitting
    it means the reported path is not REF-C's output, and no downstream metric
    can tell — the number would simply be better and belong to the bank. Call
    this wherever a pick becomes an emitted trajectory.
    """
    if sel_idx.dim() != 1 or origin.dim() != 2 or sel_idx.shape[0] != origin.shape[0]:
        raise ValueError(f"sel_idx {tuple(sel_idx.shape)} must be [B] against "
                         f"origin {tuple(origin.shape)} = [B, N]")
    picked = torch.gather(origin, 1, sel_idx.reshape(-1, 1).to(torch.long))
    bad = (picked.reshape(-1) < NATIVE)
    if bool(bad.any()):
        rows = torch.nonzero(bad).reshape(-1).tolist()
        raise ValueError(
            f"{where}: rows {rows} selected a FOREIGN candidate. That "
            "trajectory came from the augmentation bank and the generator "
            "cannot emit it — reporting it as the arm's plan attributes the "
            "bank's quality to REF-C. Restrict the argmax to `emittable`, or "
            "run with foreign_frac=0.0 at eval (the released default).")


def origin_summary(origin: Tensor) -> dict:
    """Counts per origin class — a readout, not telemetry.

    A selector panel that does not publish how many of its candidates were
    foreign cannot be read: the same score means different things at 0 % and at
    20 % foreign.
    """
    if origin.dim() != 2:
        raise ValueError(f"origin must be [B, N], got {tuple(origin.shape)}")
    tot = int(origin.numel())
    n_for = int((origin < NATIVE).sum())
    n_nat = int((origin == NATIVE).sum())
    return {
        "n_total": tot,
        "n_native": n_nat,
        "n_augmented": tot - n_nat - n_for,
        "n_foreign": n_for,
        "frac_native": (n_nat / tot) if tot else 0.0,
        "frac_foreign": (n_for / tot) if tot else 0.0,
        "rounds": sorted({int(v) for v in origin.reshape(-1).tolist()}),
    }
