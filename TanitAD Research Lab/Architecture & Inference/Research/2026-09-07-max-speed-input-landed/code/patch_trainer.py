# -*- coding: utf-8 -*-
"""Apply --max-speed-input to LOCAL copies of refc_v3_train.py + v7_labels.py."""
from __future__ import annotations

import io
import sys

TRAIN, LABELS = sys.argv[1], sys.argv[2]
_state = {}


def load(p):
    _state["p"] = p
    _state["src"] = io.open(p, encoding="utf-8").read()
    _state["n"] = 0


def rep(anchor, new, label):
    src = _state["src"]
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"ANCHOR {label!r}: found {n} times, need exactly 1")
    _state["src"] = src.replace(anchor, new, 1)
    _state["n"] += 1


def save():
    io.open(_state["p"], "w", encoding="utf-8", newline="").write(_state["src"])
    print(f"OK: {_state['n']} hunks -> {_state['p']}")


# =========================================================================== #
# v7_labels.py — the oracle accessor (the field is gated exactly like nav)     #
# =========================================================================== #
load(LABELS)

rep(
    '            _oracle={"nav_command": r.get("nav_command")},\n',
    '            _oracle={"nav_command": r.get("nav_command"),\n'
    '                     # ⭐ E16 — the max-speed INPUT block, carried through\n'
    '                     # the SAME private channel as `nav_command` and read\n'
    '                     # only through `oracle_max_speed`. ⛔ It belongs here\n'
    '                     # and not in a public field because the record itself\n'
    '                     # declares `oracle: true` / `provenance: "ego-future"`\n'
    '                     # -- identical to nav_command -- so it must be behind\n'
    '                     # the SAME manifest permission rather than beside it.\n'
    '                     "speed_max_input": r.get("speed_max_input")},\n',
    "labels-oracle")

rep(
    'def is_oracle_nav(label: "V7Label") -> bool:\n',
    'def oracle_max_speed(label: V7Label, manifest: LabelManifest\n'
    '                     ) -> dict[str, Any] | None:\n'
    '    """The ``speed_max_input`` block — refused unless the manifest carries\n'
    '    the oracle stamp, exactly like :func:`oracle_nav`.\n'
    '\n'
    '    ⭐ WHAT IT IS. ``v_max_ms`` (= ``g_tac.goals.SPEED_BAND.v_hi_ms``) plus\n'
    '    its ``units``, its pinned posted-limit ladder and its bucket. It STANDS\n'
    '    IN FOR A MAP/NAV SPEED-LIMIT SERVICE the way ``nav_command`` stands in\n'
    '    for the nav system — an INPUT, never a training signal.\n'
    '\n'
    '    ⚠️ AND WHAT IT IS NOT. Its provenance is ``ego-future``: the training\n'
    '    value is max of the ego\'s OWN REALISED speed over [anchor+2 s, +6 s],\n'
    '    while deployment supplies a LIMIT the driver may not reach. That is a\n'
    '    TRAIN/DEPLOY MISMATCH, not a leak — and it is why the read goes through\n'
    '    the oracle gate: an arm that used it is identifiable from its own\n'
    '    artifacts, which is the whole point of the stamp.\n'
    '\n'
    '    Returns ``None`` for a record that carries no block (the v7.2 release\n'
    '    carries none on 0/4,572 — MEASURED), never a fabricated default.\n'
    '    """\n'
    '    if not manifest.allow_oracle_nav:\n'
    '        raise OracleNavRefused(\n'
    '            "[v7_labels] ⛔ speed_max_input is an ORACLE — provenance "\n'
    '            "\'ego-future\', computed from the ego\'s own future speed. Pass "\n'
    '            "allow_oracle_nav=True to load_v7_labels if this is a "\n'
    '            "deliberate max-speed arm; the flag stamps the manifest so no "\n'
    '            "eval can quote the arm without it being visible.")\n'
    '    return label._oracle.get("speed_max_input")\n'
    '\n'
    '\n'
    'def is_oracle_nav(label: "V7Label") -> bool:\n',
    "labels-accessor")
save()

# =========================================================================== #
# refc_v3_train.py                                                            #
# =========================================================================== #
load(TRAIN)

# ---------------------------------------------------------------- T1 import
rep(
    "from tanitad.refs import goal_point as gpm  # noqa: E402  — E15 (GP-2)\n",
    "from tanitad.refs import goal_point as gpm  # noqa: E402  — E15 (GP-2)\n"
    "from tanitad.refs import max_speed_input as msi  # noqa: E402  — E16\n",
    "T1-import")

# ---------------------------------------------------------------- T2 pin
rep(
    "        cfg.nav_args_inject = True\n"
    "    # ---- ⛔ D-TACGOAL-1 / D-ROLL-1h: the tactical-goal SET head ----- #\n",
    "        cfg.nav_args_inject = True\n"
    "    # ---- ⭐⭐ E16: THE MAX-SPEED (map/nav posted-limit) INPUT ---------- #\n"
    "    # PI 2026-09-01, reaffirmed 2026-09-06. OPT-IN, RECORDED IN ARGV.\n"
    "    # ⛔ THE MODE IS PINNED ONTO THE CONFIG, NOT ONLY ONTO `args`, because\n"
    "    # `refcv3_arm.rebuild_config` rebuilds through THIS helper: a mode that\n"
    "    # lived only on the Namespace would be lost on every roll.\n"
    "    # ⛔ A DEAD KNOB IS REFUSED, NOT IGNORED -- the class this trainer\n"
    "    # already refuses four times (`--w-agent` under `--agents off`,\n"
    "    # `--nav-args` without `--nav-from-v7`, both halves of the P14\n"
    "    # selection split, `--tac-goal-tok-head` under kin3).\n"
    "    _msi_on = bool(getattr(args, \"max_speed_input\", False))\n"
    "    _msi_mode = str(getattr(args, \"max_speed_mode\", msi.DEFAULT_MODE))\n"
    "    if not _msi_on and _msi_mode != msi.DEFAULT_MODE:\n"
    "        raise SystemExit(\n"
    "            f\"[v3] ⛔ --max-speed-mode {_msi_mode!r} WITHOUT \"\n"
    "            f\"--max-speed-input is INERT: no ceiling is fed, so the mode \"\n"
    "            f\"selects the encoding of a channel that does not exist, and \"\n"
    "            f\"config.json would stamp a mode the run never used. Pass \"\n"
    "            f\"--max-speed-input too, or drop the mode.\")\n"
    "    if _msi_on:\n"
    "        if args.arm != \"hier\":\n"
    "            raise SystemExit(\n"
    "                \"[v3] ⛔ --max-speed-input on a FLAT arm: the ceiling's two \"\n"
    "                \"injection sites (z_tac, ctx) exist only in the hierarchy, \"\n"
    "                \"so the conditioner would be built and never read. Pass \"\n"
    "                \"--arm hier, or drop --max-speed-input.\")\n"
    "        cfg.max_speed_input = True\n"
    "        cfg.max_speed_cfg = msi.MaxSpeedConfig(enabled=True, mode=_msi_mode)\n"
    "    # ---- ⛔ D-TACGOAL-1 / D-ROLL-1h: the tactical-goal SET head ----- #\n",
    "T2-pin")

# ---------------------------------------------------------------- T3 check
rep(
    "        raise SystemExit(\"[v3] ⛔ --nav-from-v7 with an in-training eval needs \"\n"
    "                         \"--eval-labels: the eval dataset must see the SAME \"\n"
    "                         \"nav source as training, or every eval row compares \"\n"
    "                         \"against the v1 derivation.\")\n",
    "        raise SystemExit(\"[v3] ⛔ --nav-from-v7 with an in-training eval needs \"\n"
    "                         \"--eval-labels: the eval dataset must see the SAME \"\n"
    "                         \"nav source as training, or every eval row compares \"\n"
    "                         \"against the v1 derivation.\")\n"
    "\n"
    "\n"
    "def _check_max_speed_args(args) -> None:\n"
    "    \"\"\"Refuse AT START a ``--max-speed-input`` launch that would mislead.\n"
    "\n"
    "    ⛔ A DEAD FLAG IS A REFUSAL, NOT A NO-OP. ``speed_max_input`` is a v8\n"
    "    field: the v7.2 release carries it on 0 of 4,572 train records and the\n"
    "    v8 release on 4,572/4,572 and 147/147 (MEASURED 2026-09-06). Pointed at\n"
    "    a v7.2 blob the flag would look switched, feed nothing, and the arm\n"
    "    would report as +max-speed while running without a ceiling. The\n"
    "    LOADER's own refusal names the split and the field; this one fires\n"
    "    before any GPU work, on both launch paths.\n"
    "    \"\"\"\n"
    "    if not getattr(args, \"max_speed_input\", False):\n"
    "        return\n"
    "    if not getattr(args, \"v7_labels\", None):\n"
    "        raise SystemExit(\n"
    "            \"[v3] ⛔ --max-speed-input needs --v7-labels: the ceiling is \"\n"
    "            \"read from the label record's `speed_max_input` block (v8 \"\n"
    "            \"schema `speed_max_input/1`), and without the labels the flag \"\n"
    "            \"would be a dead switch that looks switched.\")\n"
    "    if (getattr(args, \"eval_cache\", None) and getattr(args, \"eval_every\", 0)\n"
    "            and not getattr(args, \"eval_labels\", None)):\n"
    "        raise SystemExit(\n"
    "            \"[v3] ⛔ --max-speed-input with an in-training eval needs \"\n"
    "            \"--eval-labels: the eval dataset must be fed the SAME ceiling \"\n"
    "            \"channel as training, or every eval row scores a \"\n"
    "            \"max-speed-conditioned model on windows that carry none.\")\n",
    "T3-check")

# ---------------------------------------------------------------- T4a attrs
rep(
    "    nav_args_enabled: bool = False\n"
    "    _nav_args_by_sid: dict | None = None\n"
    "    nav_arg_stats = None\n"
    "    nav_args_report: dict | None = None\n",
    "    nav_args_enabled: bool = False\n"
    "    _nav_args_by_sid: dict | None = None\n"
    "    nav_arg_stats = None\n"
    "    nav_args_report: dict | None = None\n"
    "    #: --max-speed-input (E16): the map/nav posted-limit ceiling.\n"
    "    #: ``_max_speed_by_sid[sid] = (v_max_ms, valid)`` in the RAW SHIPPED\n"
    "    #: value's units (m/s). ⛔ RAW, NOT PRE-QUANTIZED -- see\n"
    "    #: :meth:`enable_max_speed`. Default False keeps every banked arm's\n"
    "    #: recipe byte-identical.\n"
    "    max_speed_enabled: bool = False\n"
    "    _max_speed_by_sid: dict | None = None\n"
    "    max_speed_report: dict | None = None\n",
    "T4a-attrs")

# ---------------------------------------------------------------- T4b enable
rep(
    "    # ---- D-GSTR-1 P3: the nav command's CONTINUOUS ARGS ------------------\n"
    "\n"
    "    def enable_nav_args(self, stats=None) -> dict:\n",
    "    # ---- ⭐⭐ E16: the MAX-SPEED ceiling ---------------------------------\n"
    "\n"
    "    def enable_max_speed(self, manifest, mode: str = msi.DEFAULT_MODE\n"
    "                         ) -> dict:\n"
    "        \"\"\"Turn the map/nav posted-limit ceiling ON for this dataset.\n"
    "\n"
    "        ⛔⛔ THE VALUE THIS SHIPS IS THE RECORD'S OWN ``v_max_ms``, RAW, AND\n"
    "        THE LADDER IS APPLIED EXACTLY ONCE -- inside\n"
    "        ``max_speed_input.encode_block``, on the model side, under the\n"
    "        arm's ``mode``. Quantizing here as well is not a harmless\n"
    "        belt-and-braces: the record's ``v_max_bucket_ms`` is rounded to\n"
    "        4 dp (13.8889) while the ladder's 50 km/h step is 13.888888..., so\n"
    "        the shipped bucket is strictly GREATER than the step it names and\n"
    "        snaps UP to the next one. MEASURED on the v8 train blob: feeding\n"
    "        the shipped bucket back through the ladder moves **2,631 of 4,572\n"
    "        clips (57.5 %) one step up** (50->70 on 1,593, 20->30 on 809,\n"
    "        100->120 on 229). That is the same mechanism that manufactured\n"
    "        2,203 phantom violations from a re-derived ``v_hi``: a rounding\n"
    "        difference in the third decimal, read as a different quantity.\n"
    "\n"
    "        ⛔ UNITS ARE REQUIRED, NEVER ASSUMED. ``read_max_speed_field``\n"
    "        raises on a payload that declares none; that raise is re-raised\n"
    "        here as a ``SystemExit`` NAMING THE FIELD and the clip, because\n"
    "        m/s vs km/h vs mph is a 1.61x spread and this programme published\n"
    "        a 396 g anchor table from exactly this error.\n"
    "\n"
    "        ⭐ THE CONTENT ASSERTION. The module's pinned ladder and the\n"
    "        record's own ``bucket_steps_kmh`` must agree, and the module's\n"
    "        snap of the shipped ``v_max_ms`` must reproduce the record's\n"
    "        ``v_max_bucket_kmh`` on EVERY clip. MEASURED 2026-09-06: 4,572/4,572\n"
    "        train and 147/147 eval, 0 mismatches. A disagreement means the\n"
    "        DataFlyWheel re-pinned the ladder under us, and the run REFUSES\n"
    "        rather than training on two different quantizations at once.\n"
    "\n"
    "        Returns the census that goes into ``config.json``.\n"
    "        \"\"\"\n"
    "        if mode not in msi.MODES:\n"
    "            raise SystemExit(f\"[v3] ⛔ --max-speed-mode must be one of \"\n"
    "                             f\"{msi.MODES}, got {mode!r}\")\n"
    "        if self.v7_by_sid is None:\n"
    "            raise SystemExit(\n"
    "                \"[v3] ⛔ --max-speed-input needs the label join \"\n"
    "                \"(v7_by_sid is None) — the trainer sets it from \"\n"
    "                \"--v7-labels / --eval-labels first\")\n"
    "        by_sid: dict[int, tuple] = {}\n"
    "        n_rec = n_block = n_over = 0\n"
    "        for sid, lab in self.v7_by_sid.items():\n"
    "            n_rec += 1\n"
    "            smi = v7l.oracle_max_speed(lab, manifest)   # ⭐ the oracle gate\n"
    "            if not smi:\n"
    "                by_sid[sid] = (0.0, 0.0)\n"
    "                continue\n"
    "            n_block += 1\n"
    "            # ⛔ THE LADDER THE RECORD SHIPPED MUST BE THE LADDER WE PIN.\n"
    "            steps = tuple(smi.get(\"bucket_steps_kmh\") or ())\n"
    "            if steps and steps != msi.POSTED_LIMIT_STEPS_KMH:\n"
    "                raise SystemExit(\n"
    "                    f\"[v3] ⛔ clip {lab.clip_id!r}: the record's \"\n"
    "                    f\"`speed_max_input.bucket_steps_kmh` is {list(steps)} \"\n"
    "                    f\"but this build pins \"\n"
    "                    f\"{list(msi.POSTED_LIMIT_STEPS_KMH)}. Two ladders is \"\n"
    "                    f\"two experiments; refusing rather than quantizing \"\n"
    "                    f\"the corpus two different ways.\")\n"
    "            try:\n"
    "                got = msi.read_max_speed_field(smi, mode=\"quantized\")\n"
    "            except msi.MaxSpeedUnitsError as exc:\n"
    "                raise SystemExit(\n"
    "                    f\"[v3] ⛔ --max-speed-input: clip {lab.clip_id!r}'s \"\n"
    "                    f\"`speed_max_input` block carries a max speed and its \"\n"
    "                    f\"UNITS cannot be established. Refusing to guess — m/s \"\n"
    "                    f\"vs km/h vs mph is a 1.61x spread. {exc}\") from None\n"
    "            if not got[\"valid\"]:\n"
    "                by_sid[sid] = (0.0, 0.0)\n"
    "                continue\n"
    "            # ⭐ THE CONTENT ASSERTION, against the SHIPPED bucket.\n"
    "            want = smi.get(\"v_max_bucket_kmh\")\n"
    "            if want is not None:\n"
    "                have = round(float(got[\"quantized_ms\"]) * 3.6)\n"
    "                if have != int(want):\n"
    "                    raise SystemExit(\n"
    "                        f\"[v3] ⛔ clip {lab.clip_id!r}: this build snaps \"\n"
    "                        f\"v_max_ms={smi.get('v_max_ms')} to {have} km/h but \"\n"
    "                        f\"the record shipped {int(want)} km/h. The channel \"\n"
    "                        f\"is scored against the SHIPPED value; a \"\n"
    "                        f\"disagreement is a re-derivation, which is what \"\n"
    "                        f\"manufactured 2,203 phantom violations.\")\n"
    "            n_over += int(bool(got[\"over_ceiling\"]))\n"
    "            # ⛔ RAW m/s, in the record's declared units. The mode lives on\n"
    "            # the MODEL (`cfg.max_speed_cfg.mode`); see the docstring.\n"
    "            by_sid[sid] = (float(got[\"raw_ms\"]), 1.0)\n"
    "        n_win = n_win_valid = 0\n"
    "        for (e_i, _t) in self.index:\n"
    "            v = by_sid.get(int(self.episodes[e_i].episode_id))\n"
    "            n_win += 1\n"
    "            if v is not None and v[1] > 0.5:\n"
    "                n_win_valid += 1\n"
    "        if n_win and n_win_valid == 0:\n"
    "            raise SystemExit(\n"
    "                f\"[v3] ⛔ --max-speed-input: NOT ONE of this split's \"\n"
    "                f\"{n_win} windows receives a `speed_max_input` value. The \"\n"
    "                f\"field is a v8 addition — the v7.2 release carries it on \"\n"
    "                f\"0/4,572 records — so this is almost certainly a v7.2 \"\n"
    "                f\"label blob (md5={manifest.md5}). The channel would be a \"\n"
    "                f\"constant invalid pad and the arm would measure it as \"\n"
    "                f\"noise. Refusing rather than feeding nothing.\")\n"
    "        self._max_speed_by_sid = by_sid\n"
    "        self.max_speed_enabled = True\n"
    "        self.max_speed_report = {\n"
    "            **msi.artifact_meta(mode),\n"
    "            \"label_md5\": manifest.md5,\n"
    "            \"allow_oracle_nav\": bool(manifest.allow_oracle_nav),\n"
    "            \"quantized_by\": \"tanitad.refs.max_speed_input.encode_block \"\n"
    "                            \"(model side, ONCE); the loader ships the RAW \"\n"
    "                            \"shipped v_max_ms\",\n"
    "            \"bucket_cross_check\": \"module snap == record v_max_bucket_kmh \"\n"
    "                                  \"on every clip (refused otherwise)\",\n"
    "            \"n_clips\": n_rec, \"n_clips_with_block\": n_block,\n"
    "            \"n_clips_over_ceiling\": n_over,\n"
    "            \"n_windows\": n_win, \"n_windows_with_ceiling\": n_win_valid,\n"
    "            #: ⭐ THE NUMBER THAT DECIDES THE CLAIM. Quote this before\n"
    "            #: saying the channel carries information.\n"
    "            \"window_ceiling_frac\": round(n_win_valid / max(n_win, 1), 4),\n"
    "        }\n"
    "        print(f\"[v3] max_speed_input ({mode}): {n_block}/{n_rec} clips \"\n"
    "              f\"carry a ceiling, {n_win_valid}/{n_win} windows fed, \"\n"
    "              f\"{n_over} over the 130 km/h top step (md5={manifest.md5})\",\n"
    "              flush=True)\n"
    "        return self.max_speed_report\n"
    "\n"
    "    # ---- D-GSTR-1 P3: the nav command's CONTINUOUS ARGS ------------------\n"
    "\n"
    "    def enable_nav_args(self, stats=None) -> dict:\n",
    "T4b-enable")

# ---------------------------------------------------------------- T4c item
rep(
    "        # ---- refcv5 WP-6: the obstacle.offline target block ---------------\n",
    "        # ---- --max-speed-input: the map/nav posted-limit ceiling ----------\n"
    "        # ⛔ ALWAYS EMITTED WHEN THE CHANNEL IS ON, including for a clip\n"
    "        # with no block — as an EXPLICITLY INVALID row, never a missing key.\n"
    "        # The `nav_args` rule verbatim: a batch that sometimes carries the\n"
    "        # key and sometimes does not would make the model's own\n"
    "        # \"supplied but no seam / seam but not supplied\" refusals fire at\n"
    "        # random. A silent 0.0 in the VALUE slot reads as \"the limit here is\n"
    "        # 0 m/s — stop\", which is a LIE; the validity slot is what says\n"
    "        # \"no limit known\".\n"
    "        if self.max_speed_enabled:\n"
    "            raw = (self._max_speed_by_sid or {}).get(int(ep.episode_id))\n"
    "            v_ms, ok = (0.0, 0.0) if raw is None else raw\n"
    "            item[\"v_max_ms\"] = torch.tensor(v_ms, dtype=torch.float32)\n"
    "            item[\"v_max_valid\"] = torch.tensor(ok, dtype=torch.float32)\n"
    "        # ---- refcv5 WP-6: the obstacle.offline target block ---------------\n",
    "T4c-item")

# ---------------------------------------------------------------- T5 losses
rep(
    "    nav_args = (batch[\"nav_args\"].to(device) if \"nav_args\" in batch else None)\n",
    "    nav_args = (batch[\"nav_args\"].to(device) if \"nav_args\" in batch else None)\n"
    "    # ⭐⭐ E16 — the map/nav posted-limit ceiling, if the loader emitted it.\n"
    "    # ⛔ THE REVERSE REFUSAL LIVES HERE, where the BATCH is the fact. A build\n"
    "    # that asked for the seam and is handed a batch without the key would\n"
    "    # train the conditioner on nothing while `config.json` stamps the edge —\n"
    "    # the false-provenance class `assert_seams_are_built` exists to close.\n"
    "    v_max_ms = v_max_valid = None\n"
    "    if getattr(cfg, \"max_speed_input\", False):\n"
    "        if \"v_max_ms\" not in batch:\n"
    "            raise SystemExit(\n"
    "                \"[v3] ⛔ this build is --max-speed-input but the batch \"\n"
    "                \"carries no `v_max_ms`: the loader's `enable_max_speed` was \"\n"
    "                \"never called on this dataset. The seam would be stamped \"\n"
    "                \"and fed nothing.\")\n"
    "        v_max_ms = batch[\"v_max_ms\"].to(device)\n"
    "        v_max_valid = batch[\"v_max_valid\"].to(device)\n",
    "T5a-read")

rep(
    "    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan,\n"
    "                ego_state=ego_state, nav_args=nav_args)\n",
    "    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan,\n"
    "                ego_state=ego_state, nav_args=nav_args,\n"
    "                v_max_ms=v_max_ms, v_max_valid=v_max_valid)\n",
    "T5b-call")

# ---------------------------------------------------------------- T6 stamp
rep(
    "            \"built\": None,      # ← the MODEL fills this; see `train`\n"
    "        },\n",
    "            \"built\": None,      # ← the MODEL fills this; see `train`\n"
    "        },\n"
    "        # ⭐⭐ E16 — THE MAX-SPEED CEILING, STAMPED AS INTENT + FACT.\n"
    "        # Same three-fact shape as `tac_goal_tok_head` above and for the\n"
    "        # same reason: `requested` is what ARGV asked for, `cfg` is what the\n"
    "        # pin put on the config, `built` is what the MODEL has, and\n"
    "        # `assert_seams_are_built` refuses when the record and the weights\n"
    "        # disagree. ⛔ `provenance` is written into the RUN RECORD rather\n"
    "        # than left in a docstring, because a reader months later must be\n"
    "        # able to answer \"what was this channel actually trained on?\" from\n"
    "        # the artifact alone — and the answer is `ego-future`, a\n"
    "        # TRAIN/DEPLOY MISMATCH that any result must be read against.\n"
    "        \"max_speed_input\": {\n"
    "            \"requested\": bool(getattr(args, \"max_speed_input\", False)),\n"
    "            \"cfg\": bool(getattr(cfg, \"max_speed_input\", False)),\n"
    "            \"mode\": str(getattr(getattr(cfg, \"max_speed_cfg\", None),\n"
    "                                \"mode\", msi.DEFAULT_MODE)),\n"
    "            \"d_speed\": int(getattr(getattr(cfg, \"max_speed_cfg\", None),\n"
    "                                   \"d_speed\", 0)),\n"
    "            \"meta\": (msi.artifact_meta(\n"
    "                str(getattr(getattr(cfg, \"max_speed_cfg\", None), \"mode\",\n"
    "                            msi.DEFAULT_MODE)))\n"
    "                if getattr(cfg, \"max_speed_input\", False) else None),\n"
    "            \"required_controls\": [\"shuffled\", \"withheld\"],\n"
    "            \"controls_note\": (\n"
    "                \"⛔ EVAL OBLIGATION, inseparable from this edge and \"\n"
    "                \"identical to E13's: a max-speed-conditioned result carries \"\n"
    "                \"a SHUFFLE control (serve another clip's ceiling) and a \"\n"
    "                \"WITHHOLD control (valid = 0), or 'the arm improved' cannot \"\n"
    "                \"be separated from 'the arm gained a parameter'.\"),\n"
    "            \"built\": None,      # ← the MODEL fills this; see `train`\n"
    "        },\n",
    "T6-stamp")

# ---------------------------------------------------------------- T7 assert
rep(
    "    elif tg_built:\n"
    "        bad.append(\n"
    "            \"the seam stamp carries no `tac_goal_tok_head` block but the \"\n"
    "            \"head WAS BUILT -- a live seam absent from the run record\")\n"
    "\n"
    "    if bad:\n",
    "    elif tg_built:\n"
    "        bad.append(\n"
    "            \"the seam stamp carries no `tac_goal_tok_head` block but the \"\n"
    "            \"head WAS BUILT -- a live seam absent from the run record\")\n"
    "\n"
    "    # --- E16: the max-speed conditioner ---------------------------------- #\n"
    "    # ⛔ BIDIRECTIONAL, exactly like the block above. A conditioner in the\n"
    "    # weights that no recorded `param_breakdown` names is the D-ROLL-1\n"
    "    # rollability defect; a conditioner in the record that the weights lack\n"
    "    # is its mirror image, and both make the checkpoint unloadable through\n"
    "    # `refcv3_arm.cross_check_config`.\n"
    "    ms = stamp.get(\"max_speed_input\")\n"
    "    ms_built = _mod(model, \"max_speed_cond\") is not None\n"
    "    if isinstance(ms, dict):\n"
    "        if ms.get(\"built\") is not None and bool(ms[\"built\"]) != ms_built:\n"
    "            bad.append(\n"
    "                f\"stamp says max_speed_input.built={ms['built']!r} but \"\n"
    "                f\"model.max_speed_cond is \"\n"
    "                f\"{'BUILT' if ms_built else 'None'} -- the record and the \"\n"
    "                f\"weights disagree about the E16 conditioner\")\n"
    "        if bool(ms.get(\"cfg\", False)) and not ms_built:\n"
    "            bad.append(\n"
    "                \"stamp says max_speed_input was pinned onto the config but \"\n"
    "                \"model.max_speed_cond is None -- the record would claim a \"\n"
    "                \"ceiling channel the weights do not contain\")\n"
    "        if not bool(ms.get(\"cfg\", False)) and ms_built:\n"
    "            bad.append(\n"
    "                \"model.max_speed_cond WAS BUILT but the run record does not \"\n"
    "                \"ask for it -- parameters absent from the record, which is \"\n"
    "                \"what makes a checkpoint unrollable\")\n"
    "    elif ms_built:\n"
    "        bad.append(\n"
    "            \"the seam stamp carries no `max_speed_input` block but the \"\n"
    "            \"conditioner WAS BUILT -- a live seam absent from the record\")\n"
    "\n"
    "    if bad:\n",
    "T7-assert")

# ---------------------------------------------------------------- T8 train
rep(
    "    nav_args_stats = eval_nav_args_stats = None\n",
    "    nav_args_stats = eval_nav_args_stats = None\n"
    "    max_speed_stats = eval_max_speed_stats = None\n",
    "T8a-vars")

rep(
    "        # ---- --nav-from-v7 (E-ARCH-NAVSRC-1, PI 2026-09-02): the nav INPUT\n",
    "        # ---- ⭐⭐ E16: the map/nav posted-limit ceiling, from the SAME\n"
    "        # label join. ⛔ Independent of --nav-from-v7: the ceiling is its own\n"
    "        # channel and tying it to the nav source would make two levers one.\n"
    "        if getattr(args, \"max_speed_input\", False):\n"
    "            max_speed_stats = ds.enable_max_speed(\n"
    "                manifest, str(getattr(args, \"max_speed_mode\",\n"
    "                                      msi.DEFAULT_MODE)))\n"
    "        # ---- --nav-from-v7 (E-ARCH-NAVSRC-1, PI 2026-09-02): the nav INPUT\n",
    "T8b-train-enable")

rep(
    "                if getattr(args, \"nav_args\", False):\n"
    "                    eval_nav_args_stats = e_ds.enable_nav_args(\n"
    "                        stats=ds.nav_arg_stats)\n",
    "                if getattr(args, \"nav_args\", False):\n"
    "                    eval_nav_args_stats = e_ds.enable_nav_args(\n"
    "                        stats=ds.nav_arg_stats)\n"
    "            # ⭐ E16 — the eval dataset is fed the SAME ceiling channel.\n"
    "            # ⚠️ No normaliser is handed down and none is fitted: the ladder\n"
    "            # is PINNED road law, not a statistic of the split, which is\n"
    "            # exactly why quantizing to a fitted quantile was rejected.\n"
    "            if getattr(args, \"max_speed_input\", False):\n"
    "                eval_max_speed_stats = e_ds.enable_max_speed(\n"
    "                    e_man, str(getattr(args, \"max_speed_mode\",\n"
    "                                       msi.DEFAULT_MODE)))\n",
    "T8c-eval-enable")

rep(
    "    _seams[\"tac_goal_tok_head\"][\"built\"] = (\n"
    "        getattr(model, \"tac_goal_tok_head\", None) is not None)\n",
    "    _seams[\"tac_goal_tok_head\"][\"built\"] = (\n"
    "        getattr(model, \"tac_goal_tok_head\", None) is not None)\n"
    "    _seams[\"max_speed_input\"][\"built\"] = (\n"
    "        getattr(model, \"max_speed_cond\", None) is not None)\n",
    "T8d-built")

rep(
    "        \"nav_args\": bool(getattr(args, \"nav_args\", False)),\n"
    "        \"nav_args_stats\": ({\"train\": nav_args_stats,\n"
    "                            \"eval\": eval_nav_args_stats}\n"
    "                           if getattr(args, \"nav_args\", False) else None),\n",
    "        \"nav_args\": bool(getattr(args, \"nav_args\", False)),\n"
    "        \"nav_args_stats\": ({\"train\": nav_args_stats,\n"
    "                            \"eval\": eval_nav_args_stats}\n"
    "                           if getattr(args, \"nav_args\", False) else None),\n"
    "        # ⭐⭐ E16 — the max-speed channel's own census, so an arm is\n"
    "        # identifiable from its own artifacts. ⛔ `window_ceiling_frac` is\n"
    "        # THE number that decides whether the channel carried information;\n"
    "        # `provenance: ego-future` inside `meta` is the caveat any result\n"
    "        # must be read against.\n"
    "        \"max_speed_input\": bool(getattr(args, \"max_speed_input\", False)),\n"
    "        \"max_speed_mode\": (str(getattr(args, \"max_speed_mode\",\n"
    "                                       msi.DEFAULT_MODE))\n"
    "                           if getattr(args, \"max_speed_input\", False)\n"
    "                           else None),\n"
    "        \"max_speed_stats\": ({\"train\": max_speed_stats,\n"
    "                             \"eval\": eval_max_speed_stats}\n"
    "                            if getattr(args, \"max_speed_input\", False)\n"
    "                            else None),\n",
    "T8e-configjson")

# ---------------------------------------------------------------- T9 parser
rep(
    "    ap.add_argument(\"--nav-args\", action=\"store_true\",\n",
    "    # ---- ⭐⭐ E16 (PI 2026-09-01, reaffirmed 2026-09-06): THE MAX-SPEED\n"
    "    # (map/nav posted-limit) INPUT. Both flags default to the OFF/inert\n"
    "    # value, because the live 40 k refcv5 run resumes through this file.\n"
    "    ap.add_argument(\"--max-speed-input\", action=\"store_true\",\n"
    "                    help=\"⭐⭐ E16 — FEED the map/nav POSTED-LIMIT CEILING \"\n"
    "                         \"as a model INPUT, beside `v0` and the nav token \"\n"
    "                         \"and nowhere near a loss. Read from the v8 label \"\n"
    "                         \"record's `speed_max_input` block (`v_max_ms`, \"\n"
    "                         \"units DECLARED on the wire as m/s; the v7.2 \"\n"
    "                         \"release carries it on 0/4,572 records and the \"\n"
    "                         \"flag REFUSES there rather than looking switched). \"\n"
    "                         \"It stands in for a speed-limit service the way \"\n"
    "                         \"`nav_command` stands in for the nav system. \"\n"
    "                         \"⚠️ TRAIN/DEPLOY MISMATCH, STATED: the training \"\n"
    "                         \"value's provenance is `ego-future` — max of the \"\n"
    "                         \"ego's OWN REALISED speed over [t0+2 s, +6 s] — \"\n"
    "                         \"while deployment supplies a limit the driver may \"\n"
    "                         \"not reach. ⚠️ AND ITS RESIDUAL DEFECT: snapping \"\n"
    "                         \"UP from a STOPPED ego reports the LOWEST limit \"\n"
    "                         \"(75 %% of intersection clips get <= 30 km/h where \"\n"
    "                         \"a map would say 50), so the channel can teach \"\n"
    "                         \"'slow ego => low limit'. ⛔ Any result carries a \"\n"
    "                         \"SHUFFLE and a WITHHOLD control or it is not \"\n"
    "                         \"evidence. Requires --arm hier and --v7-labels. \"\n"
    "                         \"Default OFF: no banked arm's recipe changes.\")\n"
    "    ap.add_argument(\"--max-speed-mode\", choices=list(msi.MODES),\n"
    "                    default=msi.DEFAULT_MODE,\n"
    "                    help=\"how the ceiling is encoded for the model. \"\n"
    "                         \"`quantized` (DEFAULT) snaps UP to the pinned \"\n"
    "                         \"posted-limit ladder (20/30/50/70/80/100/120/130 \"\n"
    "                         \"km/h — VALUES from road law, MEMBERSHIP from the \"\n"
    "                         \"corpus, NO step is a quantile). `raw` feeds the \"\n"
    "                         \"unquantized float, FOR COMPARISON ONLY. \"\n"
    "                         \"⚠️ Quantization is COSMETIC as a leak fix — the \"\n"
    "                         \"bin plus v0 still recovers the raw value at R^2 \"\n"
    "                         \"0.9702 vs 0.8789 for v0 alone, i.e. 75.4 %% of \"\n"
    "                         \"the future survives — and REAL as a SEMANTICS \"\n"
    "                         \"fix: the raw ceiling sits BELOW the ego's own \"\n"
    "                         \"current speed on 34.8 %% of clips (incoherent for \"\n"
    "                         \"a ceiling), quantized 8.4 %%. ⛔ INERT without \"\n"
    "                         \"--max-speed-input, and refused as such.\")\n"
    "    ap.add_argument(\"--nav-args\", action=\"store_true\",\n",
    "T9-parser")

# ---------------------------------------------------------------- T10 entrypoints
rep(
    "    _check_nav_from_v7_args(args)          # no-op unless --nav-from-v7\n"
    "    _check_goal_point_args(args)           # no-op unless --goal-point-*\n",
    "    _check_nav_from_v7_args(args)          # no-op unless --nav-from-v7\n"
    "    _check_max_speed_args(args)            # no-op unless --max-speed-input\n"
    "    _check_goal_point_args(args)           # no-op unless --goal-point-*\n",
    "T10a-preflight")

rep(
    "    _check_nav_from_v7_args(args)\n"
    "    _check_goal_point_args(args)           # E15 (GP-2); no-op with the flags off\n",
    "    _check_nav_from_v7_args(args)\n"
    "    _check_max_speed_args(args)            # E16; no-op with the flag off\n"
    "    _check_goal_point_args(args)           # E15 (GP-2); no-op with the flags off\n",
    "T10b-train")

save()
