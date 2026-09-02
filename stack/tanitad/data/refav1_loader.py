"""Stage-2 loader for REF-A v1 — the piece that was deliberately missing.

WHAT "THE DATALOADER GAP" MEANT (PI question 2026-09-01): `refa_v1_train.py`
could only train on `SmokeData` (random tensors); the module binding the
stage-1 DINOv3 cache to real windows did not exist, BY DESIGN — "the trainer
must not invent a loader for a cache that does not exist yet". The cache
design is now settled (parity corpus, 0.2 s grid, 295.9 GiB), so this is it.

INPUTS
  cache_dir    <episode_id>.pt   fp16 [T_c, n_tokens, d_enc] on the 0.2 s grid
  episode_dir  <episode_id>.v2ep.pt — actions/poses at 10 Hz (frames NOT read)

GRID: cache index j <-> v2ep frame 2j (0.2 s = every 2nd frame at 10 Hz).
⛔ An episode whose cache length disagrees with ceil(T_ep/2) is REFUSED — a
mis-gridded cache is a silent time-warp, not a smaller dataset.

ACTIONS — (a, kappa), and the channel order is MEASURED, not assumed:
  v2ep `actions[:, 0]` correlates r = 0.995 with pose-derived curvature and
  `actions[:, 1]` only r = 0.47 with pose-derived accel (6 episodes,
  2026-09-01) ⇒ the STORED order is (kappa, accel-like) — the REVERSE of
  RefAV1Config's `a_dim: (a, kappa)`. This loader emits
      a      = (v[2(j+1)] - v[2j]) / 0.2      # poses ch3, exact on the grid
      kappa  = actions[2j, 0]                  # the measured true-kappa channel
  so a silent channel swap cannot reach the model.

SPEED — ``v0`` = v[2t], the ego speed MEASURED at the window anchor (the
  last observed cache index t), in m/s. Always emitted. ⛔ NO future speed is
  emitted: under the PI ruling of 2026-09-02 (*"velocity as initial measured
  state at its cycle time"* is allowed, *"future dynamic information from the
  ground truth"* is not) the MODEL integrates v_k = v0 + Σ_{j<k} a_j·dt
  (`RefAV1.augment_actions`, config-gated by ``speed_channel``); with the
  ``a`` above that telescopes to v[2(t+k)] under teacher forcing, and the same
  code runs at T1 on the model's own actions.

LABELS/NAV — the v7.2 s2 join (optional ``labels_path`` / ``nav_path``).
The parser is IMPORTED from `tanitad.data.v7_labels` (the ONE B1 label
consumer); this module only decides WHICH window gets WHICH record's value.

⭐ THE WINDOW→LABEL RULE, derived and sourced (JOIN_RULE.md carries the
measurements; every line here has a code citation):

* One record per clip — MEASURED on the canonical blobs: 4,572 records /
  4,572 unique clip_ids (train, md5 0ff902130ce76886b8a925eceed9e3a5) and
  147/147 (eval, md5 aa12c948f062181c3297265b51526ec5). Join key: the v2ep
  file's ``clip_id`` (= the cache filename stem on this corpus). The record
  is anchored at ``t0_s`` (8.0 on every record, RAW clip timeline) and its
  bands are FORWARD horizons from that anchor (`s2_geom_emit_v7.py:50-52`,
  times built as ``key + band*hz`` from the t0 index): ``a_tac`` describes
  raw seconds [t0+2, t0+6].
* THIS loader's windows live on the RAW timeline with NO offset: the stage-1
  cache encodes every 2nd RAW v2ep frame (``range(0, T, 2)``), and this
  loader reads the v2ep dict's RAW ``poses`` directly — the (n_stack−1)
  shift belongs to ``load_compressed`` (`v2_compressed.py:195-197`), which
  we bypass. (The S2 join adds that shift because it consumes the PROVIDER
  view — `s2_labels.py:730-734`. Copying the offset here would be the
  `step_s` trap: a correction applied where its cause does not exist.)
  ⇒ a window's NOW is ``t * 0.2`` s, t = last observed cache index.
* ``lat_label`` / ``lon_label`` / ``route_label``: a window at t_w is
  supervised by its clip's record iff  |t_w − t0| ≤ (tac_hi − tac_lo)/2
  (± 2.0 s at the shipped bands, computed from the RECORD, never hardcoded —
  the derived-constant trap). Justification: the window's own forward
  tactical band [t_w+2, t_w+6] then overlaps the labeled band [t0+2, t0+6]
  by ≥ half its width, so the record's action is the majority description of
  what this window's tactical horizon contains. This REPRODUCES the shipped
  v6-trainer choice — `s2_labels.S2WindowSupervision._in_band`
  (`s2_labels.py:773-776`) with its default ``valid_window_s`` (−2, 2)
  (`s2_labels.py:587`) — as a derivation instead of a default.
* Out-of-band windows and unlabeled episodes emit ``-100`` — torch
  ``cross_entropy``'s default ``ignore_index``, the same discipline as
  `s2_labels.py:194-197`. ✅ **THE MODEL SIDE NOW MASKS — this paragraph used
  to say it did not, and that warning was STALE (corrected 2026-09-02).** The
  guard it once described as "proposed, not edited here" has landed verbatim:
  ``RefAV1.forward`` validates only ``lbl[lbl != -100]`` (`refa_v1.py:991`),
  range-checks the labeled rows alone, and **skips an all-ignored family
  rather than averaging it** (`:997`) — because an all-ignored CE is NaN and a
  NaN there would poison every weight while reading as a batch hiccup. The
  route path (`:1010`) is identical, and ``F.cross_entropy`` supplies
  ``ignore_index=-100`` by default, so the -100 rows this loader emits are
  dropped from the loss rather than refused. MEASURED: 169 refa/refav1 tests
  pass, pinned in `tests/test_refav1_loader_labels.py`.
  ⚠️ The lesson is why this line existed for as long as it did: a blocker
  note is not revisited when the thing it blocks on lands, so it keeps
  reading as a live gap. State the CHECK, not the verdict — the verdict rots.
* ``route_label`` (3-class, `refb.ROUTE_CLASSES` order L/S/R) is derived
  from the record's ``nav_command`` token: NAV_TURN_L→route_left,
  NAV_FOLLOW_ROAD→route_straight, NAV_TURN_R→route_right — the same
  future-heading source REF-B's route/nav pseudo-labels shared
  (`refb.py:60-68`). ⚠️ ECHO HAZARD, by construction: ``route_label`` and
  ``nav_cmd`` come from the SAME field, so on a nav-conditioned arm
  (``nav_inject=True``) route accuracy measures the flagship-v1 nav echo
  (a bijection of its own input scored 1.0000), not skill. Route accuracy
  is only evidence on a nav-shuffle control or a nav-less arm.
* ``nav_cmd`` COPIES the v6 trainer's join wholesale: records loaded with
  ``allow_oracle_nav=True`` (the STAMP — `train_v6_staged.py:5510`; it lands
  in this loader's ``join_report`` as it lands in config.json there), ids
  minted by `v7_labels.NavEmitter` under ``t0_constant`` semantics
  (`train_v6_staged.py:5519-5522`), under which token AND args are
  window-independent (`v7_labels.py:483-488`) — so the per-clip constant is
  precomputed at init and ``batch()`` is a lookup. Args are NOT emitted:
  `RefAV1._run_brains` consumes only the token index.
* ⚠️ UNLABELED episodes with ``nav_path`` set emit ``nav_cmd = 0``
  ("follow") + ``nav_valid = False``. Index 0 is the codebase's sanctioned
  no-information command — `refb.py:60` ("index 0 is the unlabeled
  default") and `refa_v1.py:748` (nav_cmd=None ⇒ zeros) — NOT an invented
  default. The v6 trainer instead REFUSES a missing record
  (`train_v6_staged.py:5512-5518`), but there full coverage was a build
  guarantee; on THIS corpus non-coverage is a measured property (1/24 probe
  episodes labeled, 2026-09-01), so the loader reports loudly (count and
  names in ``join_report``) and marks the rows instead of dying. A trainer
  wanting v6-strictness asserts ``join_report['nav']['n_missing'] == 0``.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

__all__ = ["RefAV1Windows", "split_episodes", "IGNORE_ID", "TAC_VOCAB_VERSION"]

#: torch ``cross_entropy``'s default ``ignore_index`` — the no-label marker for
#: lat/lon/route rows (same discipline as ``s2_labels.py:194-197``).
IGNORE_ID = -100
#: The v7 mandate vocabulary version the labels are indexed against — must
#: match the model's ``RefAV1Config.tac_vocab_version`` default (v7.0).
TAC_VOCAB_VERSION = "v7.0"

#: v7 nav token ↔ legacy ``refb.NAV_COMMANDS`` semantic, position-pinned at
#: join init: ``NavEmitter`` ids enumerate ``vocab_v7.NAV_COMMAND_TOKENS``
#: while the model's ``nav_emb`` rows were sized/ordered by ``NAV_COMMANDS``
#: (`config.py:139`) — the two agree TODAY by construction, and this pin turns
#: that from a coincidence into a checked invariant.
_NAV_TOKEN_TO_LEGACY = {"NAV_FOLLOW_ROAD": "follow", "NAV_TURN_L": "left",
                        "NAV_TURN_R": "right"}
#: nav token → the 3-class route-heading label (`refb.ROUTE_CLASSES` order).
_NAV_TOKEN_TO_ROUTE = {"NAV_TURN_L": "route_left",
                       "NAV_FOLLOW_ROAD": "route_straight",
                       "NAV_TURN_R": "route_right"}


def split_episodes(names: list[str], val_frac: float = 0.15,
                   seed: int = 0) -> tuple[list[str], list[str]]:
    """Episode-disjoint split, deterministic in (names, seed)."""
    order = sorted(names)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(order), generator=g).tolist()
    n_val = max(1, int(round(len(order) * val_frac)))
    val = sorted(order[i] for i in perm[:n_val])
    train = sorted(set(order) - set(val))
    return train, val


class RefAV1Windows:
    """Deterministic window sampler over the stage-1 cache + v2ep kinematics."""

    def __init__(self, cache_dir: str | Path, episode_dir: str | Path, *,
                 op_window: int, op_steps: int, op_dt: float = 0.2,
                 str_dt: float = 3.0, str_ext_steps: int = 2,
                 episodes: list[str] | None = None,
                 lru: int = 32, seed: int = 0,
                 labels_path: str | Path | None = None,
                 nav_path: str | Path | None = None,
                 labels_expected_records: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.episode_dir = Path(episode_dir)
        self.W, self.K, self.dt = int(op_window), int(op_steps), float(op_dt)
        self.str_dt, self.k_ext = float(str_dt), int(str_ext_steps)
        if abs(self.dt - 0.2) > 1e-9:
            raise ValueError(f"op_dt {op_dt}: this loader's grid mapping "
                             "(cache j <-> frame 2j) is derived for 0.2 s")
        # ext tick k (1-based) closes at 6.0 + k*str_dt after the window start
        self.ext_close = [int(round((6.0 + k * self.str_dt) / self.dt))
                          for k in range(1, self.k_ext + 1)]
        self.ext_open = [int(round((6.0 + (k - 1) * self.str_dt) / self.dt))
                         for k in range(1, self.k_ext + 1)]
        self.reach = self.ext_close[-1] if self.ext_close else self.K

        names = episodes if episodes is not None else sorted(
            p.stem for p in self.cache_dir.glob("*.pt") if p.stem != "index")
        if not names:
            raise FileNotFoundError(f"no cached episodes under {self.cache_dir}")
        want_join = labels_path is not None or nav_path is not None
        self.names: list[str] = []
        self.windows: list[tuple[int, int]] = []      # (episode idx, t)
        self._T: dict[str, int] = {}
        self.clip_id: dict[str, str] = {}             # episode name -> clip_id
        for nm in names:
            ep = self.episode_dir / f"{nm}.v2ep.pt"
            if not ep.exists():
                raise FileNotFoundError(f"cache episode {nm} has no v2ep at {ep}")
            t_c = self._cache_len(nm)
            t_ep, cid = self._ep_meta(nm)
            want = math.ceil(t_ep / 2)
            if t_c != want:
                raise ValueError(
                    f"{nm}: cache length {t_c} != ceil(T_ep/2) = {want} — a "
                    "mis-gridded cache is a silent time-warp, refused")
            if cid is not None:
                self.clip_id[nm] = str(cid)
            elif want_join:
                raise ValueError(
                    f"[refav1-labels] episode {nm}: v2ep carries no `clip_id` "
                    f"key, so the v7.2 label/nav join has no key to join on — "
                    f"refusing rather than silently leaving it unlabeled")
            ei = len(self.names)
            self.names.append(nm)
            self._T[nm] = t_c
            # t is the LAST observed cache index; +1 below it because the
            # kinematic action at the final future step reads v at j+1.
            for t in range(self.W - 1, t_c - self.reach - 1):
                self.windows.append((ei, t))
        if not self.windows:
            raise ValueError(
                f"0 windows: episodes too short for W={self.W} + reach "
                f"{self.reach} (need T_c > {self.W - 1 + self.reach + 1})")
        g = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(len(self.windows), generator=g).tolist()
        self._cursor = 0
        self._lru_n = int(lru)
        self._lru: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

        # ---- the v7.2 label/nav join (module docstring carries the rule) ----
        self._labels_on = labels_path is not None
        self._nav_on = nav_path is not None
        #: clip_id -> (lat_id, lon_id, route_id, now_lo_s, now_hi_s)
        self._lab: dict[str, tuple[int, int, int, float, float]] = {}
        self._nav_id: dict[str, int] = {}
        self.join_report: dict = {}
        if want_join:
            self._init_v72_join(labels_path, nav_path, labels_expected_records)

    # ------------------------------------------------------------- internals
    def _cache_len(self, nm: str) -> int:
        return torch.load(self.cache_dir / f"{nm}.pt", map_location="cpu",
                          weights_only=True, mmap=True).shape[0]

    def _ep_meta(self, nm: str) -> tuple[int, str | None]:
        """(T_ep, clip_id) from the v2ep dict. RAW timeline — see docstring."""
        o = torch.load(self.episode_dir / f"{nm}.v2ep.pt",
                       map_location="cpu", weights_only=False)
        return int(o["poses"].shape[0]), o.get("clip_id")

    def _episode(self, nm: str):
        """(features fp16 [T_c,N,d], v [T_ep], kappa [T_ep]) — LRU-cached."""
        hit = self._lru.pop(nm, None)
        if hit is None:
            feats = torch.load(self.cache_dir / f"{nm}.pt",
                               map_location="cpu", weights_only=True)
            o = torch.load(self.episode_dir / f"{nm}.v2ep.pt",
                           map_location="cpu", weights_only=False)
            hit = (feats, o["poses"][:, 3].float(), o["actions"][:, 0].float())
        self._lru[nm] = hit                          # re-insert as most recent
        while len(self._lru) > self._lru_n:
            self._lru.pop(next(iter(self._lru)))
        return hit

    def _kin_actions(self, v: Tensor, kap: Tensor, j0: int, n: int) -> Tensor:
        """(a, kappa) for cache steps j0 .. j0+n-1 (each spans 0.2 s)."""
        idx = torch.arange(j0, j0 + n)
        f = idx * 2                                   # 10 Hz frame of step open
        f_next = torch.clamp((idx + 1) * 2, max=v.shape[0] - 1)
        a = (v[f_next] - v[f]) / self.dt
        return torch.stack([a, kap[f]], dim=-1)       # ⭐ (a, kappa) — swapped

    # ----------------------------------------------------- v7.2 label join
    def _init_v72_join(self, labels_path, nav_path, expected) -> None:
        """Build the per-clip lookup tables. Lazy imports on purpose — a
        label-less loader (today's only production use) must not acquire the
        v7 label stack as a mandatory dependency (the `train_v6_staged.py:
        153-161` lesson: a module-level import of the label module made it
        mandatory for EVERY launch)."""
        from tanitad.data.v7_labels import NavEmitter, load_v7_labels, oracle_nav
        from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
        from tanitad.models.vocab_v7 import NAV_COMMAND_TOKENS
        from tanitad.refs.refb import NAV_COMMANDS, ROUTE_CLASSES

        # ⛔ position pin (see _NAV_TOKEN_TO_LEGACY): NavEmitter ids index the
        # model's NAV_COMMANDS-ordered embedding — verified, not assumed.
        for i, tok in enumerate(NAV_COMMAND_TOKENS):
            if _NAV_TOKEN_TO_LEGACY.get(tok) != NAV_COMMANDS[i]:
                raise AssertionError(
                    f"[refav1-labels] nav index alignment broken: "
                    f"NAV_COMMAND_TOKENS[{i}]={tok!r} vs NAV_COMMANDS[{i}]="
                    f"{NAV_COMMANDS[i]!r} — NavEmitter ids would land on the "
                    f"wrong nav_emb rows")
        lat_vocab = tactical_lat_actions(TAC_VOCAB_VERSION)
        lon_vocab = tactical_lon_actions_v(TAC_VOCAB_VERSION)

        loaded: dict[str, tuple] = {}

        def _load(p):
            key = str(p)
            if key not in loaded:
                # ⭐ allow_oracle_nav=True IS the v6 trainer's choice, copied
                # verbatim (`train_v6_staged.py:5510`): nav here is a TRAINING
                # input/label (labels may use ego — PI 2026-08-03), and the
                # flag STAMPS the manifest so the arm carries its provenance.
                loaded[key] = load_v7_labels(p, allow_oracle_nav=True,
                                             require_records=expected)
            return loaded[key]

        def _tok_id(tok, vocab, what, clip):
            if tok is None:
                return IGNORE_ID                       # record declined -> mask
            try:
                return vocab.index(tok)
            except ValueError:
                raise ValueError(
                    f"[refav1-labels] ⛔ clip {clip}: {what} token {tok!r} is "
                    f"not in the {TAC_VOCAB_VERSION} vocabulary {vocab} — "
                    f"vocabulary drift is a different experiment, refused"
                ) from None

        rep: dict = {"rule": "in-band iff |t_now - t0| <= (tac_hi - tac_lo)/2; "
                             "t_now = t * 0.2 s (RAW timeline, no n_stack "
                             "offset — see module docstring)"}
        tols: set[float] = set()
        if self._labels_on:
            labels, man = _load(labels_path)
            by_clip = {x.clip_id: x for x in labels}
            for x in labels:
                if x.clip_id not in {self.clip_id.get(nm) for nm in self.names}:
                    continue
                if not math.isfinite(x.t0_s):
                    raise ValueError(f"[refav1-labels] ⛔ clip {x.clip_id}: "
                                     f"t0_s is not finite — the anchor cannot "
                                     f"be placed on the window grid")
                band = (x.bands or {}).get("tactical_s")
                if (not isinstance(band, (list, tuple)) or len(band) != 2
                        or band[1] <= band[0]):
                    raise ValueError(
                        f"[refav1-labels] ⛔ clip {x.clip_id}: bands.tactical_s "
                        f"{band!r} is not a [lo, hi] interval — schema drift")
                tol = (float(band[1]) - float(band[0])) / 2.0
                tols.add(tol)
                nav = oracle_nav(x, man) or {}
                r_tok = nav.get("token")
                if r_tok is None:
                    route = IGNORE_ID                  # record declined -> mask
                elif r_tok not in _NAV_TOKEN_TO_ROUTE:
                    raise ValueError(
                        f"[refav1-labels] ⛔ clip {x.clip_id}: nav token "
                        f"{r_tok!r} has no route mapping (known: "
                        f"{sorted(_NAV_TOKEN_TO_ROUTE)}) — vocabulary drift "
                        f"is a different experiment, refused")
                else:
                    route = ROUTE_CLASSES.index(_NAV_TOKEN_TO_ROUTE[r_tok])
                self._lab[x.clip_id] = (
                    _tok_id(x.tac_lat, lat_vocab, "a_tac.lat", x.clip_id),
                    _tok_id(x.tac_lon, lon_vocab, "a_tac.lon", x.clip_id),
                    route,
                    x.t0_s - tol, x.t0_s + tol)
            missing = sorted(nm for nm in self.names
                             if self.clip_id[nm] not in by_clip)
            if len(missing) == len(self.names):
                raise ValueError(
                    f"[refav1-labels] ⛔ {labels_path} joined ZERO of "
                    f"{len(self.names)} episodes — wrong blob for this corpus "
                    f"(md5={man.md5})")
            n_in_band = sum(
                1 for ei, t in self.windows
                if (row := self._lab.get(self.clip_id[self.names[ei]]))
                is not None and row[3] <= t * self.dt <= row[4])
            rep["labels"] = {
                "n_episodes": len(self.names),
                "n_labeled": len(self.names) - len(missing),
                "n_missing": len(missing), "missing_episodes": missing,
                "n_windows": len(self.windows), "n_windows_in_band": n_in_band,
                "band_tol_s": sorted(tols),
                "tac_vocab_version": TAC_VOCAB_VERSION,
                **man.to_dict()}
        if self._nav_on:
            labels, man = _load(nav_path)
            by_clip = {x.clip_id: x for x in labels}
            mapped = {ei: self.clip_id[nm] for ei, nm in enumerate(self.names)
                      if self.clip_id[nm] in by_clip}
            missing = sorted(nm for nm in self.names
                             if self.clip_id[nm] not in by_clip)
            if not mapped:
                raise ValueError(
                    f"[refav1-labels] ⛔ nav blob {nav_path} joined ZERO of "
                    f"{len(self.names)} episodes — wrong blob for this corpus "
                    f"(md5={man.md5})")
            em = NavEmitter(labels, man, mapped, semantics="t0_constant")
            eis = sorted(mapped)
            ids, _args = em(eis)          # t0_constant: window-independent
            self._nav_id = {mapped[e]: int(i)
                            for e, i in zip(eis, ids.tolist())}
            rep["nav"] = {
                "n_episodes": len(self.names), "n_labeled": len(mapped),
                "n_missing": len(missing), "missing_episodes": missing,
                "unlabeled_default": "nav_cmd=0 ('follow', refb.py:60) + "
                                     "nav_valid=False",
                **em.provenance()}
        self.join_report = rep
        # ⚠️ ASCII-only by design: this line prints at loader INIT on a cp1252
        # dev box — a fancy glyph here would crash the constructor without
        # PYTHONIOENCODING=utf-8 (the "crashed on its own checkmark" trap).
        parts = []
        if "labels" in rep:
            L = rep["labels"]
            parts.append(f"labels {L['n_labeled']}/{L['n_episodes']} eps "
                         f"({L['n_missing']} missing"
                         + (f", e.g. {L['missing_episodes'][:3]}" if L["n_missing"]
                            else "")
                         + f"), {L['n_windows_in_band']}/{L['n_windows']} "
                           f"windows in-band +-{L['band_tol_s']}s, "
                           f"md5={L['md5']}")
        if "nav" in rep:
            N = rep["nav"]
            parts.append(f"nav {N['n_labeled']}/{N['n_episodes']} eps "
                         f"({N['n_missing']} missing -> nav_cmd=0+"
                         f"nav_valid=False), md5={N['md5']}, "
                         f"allow_oracle_nav={N['allow_oracle_nav']}")
        print("[refav1-labels] v7.2 join: " + " | ".join(parts), flush=True)

    # ------------------------------------------------------------------ api
    def __len__(self) -> int:
        return len(self.windows)

    def batch(self, bs: int) -> dict:
        feats, fut, act, ext_t, ext_a, v0s = [], [], [], [], [], []
        lat, lon, route = [], [], []
        nav, nav_ok = [], []
        for _ in range(bs):
            ei, t = self.windows[self._order[self._cursor]]
            self._cursor = (self._cursor + 1) % len(self._order)
            nm = self.names[ei]
            F, v, kap = self._episode(nm)
            feats.append(F[t - self.W + 1:t + 1].float())
            fut.append(F[t + 1:t + 1 + self.K].float())
            act.append(self._kin_actions(v, kap, t, self.K))
            v0s.append(v[2 * t])             # measured at the anchor frame 2t
            if self.k_ext:
                ext_t.append(torch.stack([F[t + c] for c in self.ext_close]
                                         ).float())
                ext_a.append(torch.stack(
                    [self._kin_actions(v, kap, t + o, 1)[0]
                     for o in self.ext_open]))
            if self._labels_on:
                row = self._lab.get(self.clip_id[nm])
                if row is not None and row[3] <= t * self.dt <= row[4]:
                    lat.append(row[0]), lon.append(row[1]), route.append(row[2])
                else:                                  # out-of-band / unlabeled
                    lat.append(IGNORE_ID), lon.append(IGNORE_ID)
                    route.append(IGNORE_ID)
            if self._nav_on:
                nid = self._nav_id.get(self.clip_id[nm])
                nav.append(0 if nid is None else nid)  # 0 = unlabeled default
                nav_ok.append(nid is not None)
        _long = (lambda xs: torch.tensor(xs, dtype=torch.long))
        out = {"feats": torch.stack(feats), "future_feats": torch.stack(fut),
               "actions": torch.stack(act),
               # v7.2 join fields — Long tensors when the join is on, None
               # otherwise (the pre-join contract, kept for existing callers).
               # ⛔ -100 rows: the MODEL currently refuses them (range check,
               # refa_v1.py:978/:993) — see the module docstring.
               "lat_label": _long(lat) if self._labels_on else None,
               "lon_label": _long(lon) if self._labels_on else None,
               "route_label": _long(route) if self._labels_on else None,
               "nav_cmd": _long(nav) if self._nav_on else None}
        if self._nav_on:
            #: diagnostic — True where the clip HAD a record; False rows carry
            #: the index-0 default and must be excluded from any nav-echo /
            #: route-accuracy readout.
            out["nav_valid"] = torch.tensor(nav_ok, dtype=torch.bool)
        # the anchor speed (module docstring, SPEED) — a scalar per window;
        # the model derives every later speed from it and the actions.
        out["v0"] = torch.stack(v0s)
        if self.k_ext:
            out["str_ext_targets"] = torch.stack(ext_t)
            out["str_ext_actions"] = torch.stack(ext_a)
        return out
