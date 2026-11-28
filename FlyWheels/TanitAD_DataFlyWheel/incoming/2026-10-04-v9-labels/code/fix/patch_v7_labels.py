"""Apply the WP-A v7_labels policy-isolation fix to a copy of tanitad/data/v7_labels.py (path = argv[1])."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
crlf = "\r\n" in s
s = s.replace("\r\n", "\n")


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


# 1. V7Label carries its blob's measured policy
rep('''    audit: dict[str, Any] = field(default_factory=dict)
    _oracle: dict[str, Any] = field(default_factory=dict, repr=False)
''', '''    audit: dict[str, Any] = field(default_factory=dict)
    _oracle: dict[str, Any] = field(default_factory=dict, repr=False)
    #: ⭐⭐ THE NEGATIVE POLICY OF THE BLOB THIS LABEL WAS LOADED FROM (WP-A fix, 2026-10-04). The geometry-
    #: exhaustive tokens (absence = a supervised negative) and the vlm-cot tokens, MEASURED by
    #: :func:`load_v7_labels` over the blob and attached to every label of that blob. ⛔ It used to live ONLY in
    #: module globals refilled by every load, so the eval-label load overwrote the train policy before the
    #: DataLoader workers started (refcv8 audit D1 F2: ``LANE_CHANGE_L`` supervised as a negative on 100 % of
    #: tactical windows). Carried on the object, a later load cannot reach it and a pickled worker copy keeps it.
    #: ``None`` only for labels built outside :func:`load_v7_labels` (unit tests), which use the legacy mirror.
    goal_geometry_tokens: frozenset[str] | None = None
    goal_cot_tokens: frozenset[str] | None = None
''')
# 2. LabelManifest records it
rep('''    cot_absence_negative: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "md5": self.md5, "n_records": self.n_records,
                "schema_version": self.schema_version, "vocab": self.vocab,
                "allow_oracle_nav": self.allow_oracle_nav,
                "cot_absence_negative": self.cot_absence_negative,''', '''    cot_absence_negative: dict[str, Any] | None = None
    #: ⭐ the measured goal-negative policy of THIS blob (sorted tuples, JSON-safe) — the same sets every label of
    #: the blob carries; a run's config.json now states the policy the workers actually used.
    goal_geometry_tokens: tuple[str, ...] = ()
    goal_cot_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "md5": self.md5, "n_records": self.n_records,
                "schema_version": self.schema_version, "vocab": self.vocab,
                "allow_oracle_nav": self.allow_oracle_nav,
                "cot_absence_negative": self.cot_absence_negative,
                "goal_geometry_tokens": list(self.goal_geometry_tokens),
                "goal_cot_tokens": list(self.goal_cot_tokens),''')
# 3. load: measure FIRST, attach to every label, record on the manifest
rep('''    labels: list[V7Label] = []
    for r in recs:
        a_tac = r.get("a_tac") or {}''', '''    # ⭐⭐ THE NEGATIVE POLICY IS DERIVED FROM THE BLOB, NOT DECLARED (see the paragraph after the loop), and since
    # 2026-10-04 it is computed BEFORE the labels so every label carries its own blob's policy. Same arithmetic
    # as before (a token is vlm-cot-backed iff any annotation says so), read from the same dicts.
    _cot_backed = {t for r in recs
                   for t, m in ((r.get("g_tac") or {}).get("goals") or {}).items()
                   if isinstance(m, dict) and m.get("provenance") == "vlm-cot"}
    cot_tokens = frozenset(_cot_backed & set(TACTICAL_GOAL_TOKENS_V7))
    geometry_tokens = frozenset(
        t for t in TACTICAL_GOAL_TOKENS_V7
        if t not in _cot_backed and t not in TACTICAL_GOAL_NEEDS_PERCEPTION)

    labels: list[V7Label] = []
    for r in recs:
        a_tac = r.get("a_tac") or {}''')
rep('''                     "speed_max_input": r.get("speed_max_input")},
        ))
''', '''                     "speed_max_input": r.get("speed_max_input")},
            goal_geometry_tokens=geometry_tokens,
            goal_cot_tokens=cot_tokens,
        ))
''')
rep('''    global _MEASURED_GEOMETRY_TOKENS, _MEASURED_COT_TOKENS
    _cot_backed = {t for lb in labels for t, m in lb.tac_goal_meta.items()
                   if m.get("provenance") == "vlm-cot"}
    _MEASURED_COT_TOKENS = frozenset(_cot_backed & set(TACTICAL_GOAL_TOKENS_V7))
    _MEASURED_GEOMETRY_TOKENS = frozenset(
        t for t in TACTICAL_GOAL_TOKENS_V7
        if t not in _cot_backed and t not in TACTICAL_GOAL_NEEDS_PERCEPTION)

    return labels, LabelManifest(
        path=str(p), md5=md5, n_records=len(recs),
        schema_version=next(iter(schemas)), vocab=next(iter(vocabs)),
        allow_oracle_nav=bool(allow_oracle_nav),
        divergences=tuple(divergences))''', '''    #
    # ⚠️ LEGACY MIRROR ONLY (2026-10-04): the module globals are still written, last load wins, so code that
    # predates the fix and builds labels by hand keeps working. ⛔ No function reads them for a label that
    # carries its own policy — i.e. for every label this loader returns.
    global _MEASURED_GEOMETRY_TOKENS, _MEASURED_COT_TOKENS
    _MEASURED_COT_TOKENS = cot_tokens
    _MEASURED_GEOMETRY_TOKENS = geometry_tokens

    return labels, LabelManifest(
        path=str(p), md5=md5, n_records=len(recs),
        schema_version=next(iter(schemas)), vocab=next(iter(vocabs)),
        allow_oracle_nav=bool(allow_oracle_nav),
        divergences=tuple(divergences),
        goal_geometry_tokens=tuple(sorted(geometry_tokens)),
        goal_cot_tokens=tuple(sorted(cot_tokens)))


def _geometry_tokens_for(label: "V7Label") -> frozenset[str]:
    """The geometry-exhaustive token set that decides ``label``'s negatives: its OWN blob's policy.
    ⛔ The legacy module mirror is read ONLY for a label built outside :func:`load_v7_labels`."""
    g = label.goal_geometry_tokens
    return _MEASURED_GEOMETRY_TOKENS if g is None else g


def _policy_of(labels: Sequence["V7Label"], attr: str) -> frozenset[str]:
    """The single policy a set of labels shares; REFUSES labels from blobs with different policies."""
    sets = {getattr(lb, attr) for lb in labels if getattr(lb, attr) is not None}
    if len(sets) > 1:
        raise ValueError(f"[v7_labels] ⛔ labels from blobs with DIFFERENT {attr} policies were mixed "
                         f"({len(sets)} distinct sets) — supervise each split with its own policy")
    if sets:
        return next(iter(sets))
    return _MEASURED_GEOMETRY_TOKENS if attr == "goal_geometry_tokens" else _MEASURED_COT_TOKENS''')
rep('''def cot_backed_tokens() -> frozenset[str]:
    """The tokens this loaded split emits from ``vlm-cot``. MEASURED, not
    declared — the same read that decides the negative policy."""
    return _MEASURED_COT_TOKENS''', '''def cot_backed_tokens(labels: Sequence["V7Label"] | None = None) -> frozenset[str]:
    """The tokens a loaded split emits from ``vlm-cot``. MEASURED, not declared — the same read that decides the
    negative policy. Pass ``labels`` (2026-10-04): the answer is then THAT split's, whatever was loaded since;
    without it the legacy last-load mirror is returned."""
    if labels is not None:
        return _policy_of(labels, "goal_cot_tokens")
    return _MEASURED_COT_TOKENS''')
rep('''    cot_now = cot_backed_tokens()
    if not cot_now:''', '''    cot_now = cot_backed_tokens(labels)
    if not cot_now:''')
# 4. readers
rep('''            w.append(1.0 if tok in _MEASURED_GEOMETRY_TOKENS else IGNORE_W)''',
    '''            w.append(1.0 if tok in geom else IGNORE_W)''')
rep('''            w.append(1.0 if (tok in sidecar.tokens
                             or tok in _MEASURED_GEOMETRY_TOKENS) else IGNORE_W)''',
    '''            w.append(1.0 if (tok in sidecar.tokens
                             or tok in geom) else IGNORE_W)''')
rep('''    entailed = entailed_false(present)
    y, w = [], []
    for tok in TAC_GOAL_TOKENS:''', '''    entailed = entailed_false(present)
    geom = _geometry_tokens_for(label)          # THIS label's blob policy (never another load's)
    y, w = [], []
    for tok in TAC_GOAL_TOKENS:''')
rep('''    ws = [tactical_goal_targets(lb, lb.t0_s, negatives=negatives,
                                sidecar=sidecar)[1] for lb in labels]''', '''    ws = [tactical_goal_targets(lb, lb.t0_s, negatives=negatives,
                                sidecar=sidecar)[1] for lb in labels]
    geom = _policy_of(labels, "goal_geometry_tokens")''')
rep('''        sup = (tok in _MEASURED_GEOMETRY_TOKENS
               or (sidecar is not None and tok in sidecar.tokens''', '''        sup = (tok in geom
               or (sidecar is not None and tok in sidecar.tokens''')
rep('''        self._by_clip = {x.clip_id: x for x in labels}
        self.n_tokens = len(TAC_GOAL_TOKENS)''', '''        self._by_clip = {x.clip_id: x for x in labels}
        self.n_tokens = len(TAC_GOAL_TOKENS)
        #: the labels' own measured policy, frozen at construction (never re-read from module state)
        self.geometry_tokens = _policy_of(labels, "goal_geometry_tokens") if labels else frozenset()''')
rep('''        sup = set(_MEASURED_GEOMETRY_TOKENS)
        if self.negatives == "cot-absence-negative" and self.sidecar:''', '''        sup = set(self.geometry_tokens)
        if self.negatives == "cot-absence-negative" and self.sidecar:''')
if crlf:
    s = s.replace("\n", "\r\n")
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p, "crlf" if crlf else "lf")
