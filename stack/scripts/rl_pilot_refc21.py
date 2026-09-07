#!/usr/bin/env python3
"""P-RC21 — RL post-training of the REF-C v2.1 base planner (the pilot).

Pre-registered in ``…/2026-08-29-rl-posttrain-library/PREREG_P_RC21.md`` —
both outcomes committed BEFORE any step ran. PI directive 2026-08-29: apply the
RL training to the existing REF-C until refcv3's cold start lands.

WIRING (nothing copied — refc imported, library imported)
---------------------------------------------------------
model      ``refc.RefCModel(refc.refc_config())`` + strict ``ck["model"]`` load,
           param count asserted against the registry's 104,191,577, and the
           CONFIG CONTRACT (2026-09-07) compared against the checkpoint's own
           config: agreement is STAMPED to ``config_contract.json``, a
           behavioural disagreement REFUSES, and an absent config becomes a
           stated assumption instead of an invisible one.
trainable  ``decoder.*`` EXCEPT ``decoder.conf_head`` (the v2.1 selector surface
           lives INSIDE the decoder — training it would be TRAIN-C1's inversion
           again). Enforced by ``exclude_prefixes`` + the forbidden tripwire.
sampler    the library's Gaussian-on-offset surrogate at ``decoder_steps=2``
           (the deploy path). STATED LIMIT: not the true diffusion density.
reward     ``DEFAULT_WEIGHTS`` on real scene context from the pilot join
           (A0 PASS on this corpus: ``raw/a0_pilot.json``), dt = 0.5 s.
readout    R1 fan reward · R2 fan collision rate · R3 conf-argmax ADE — the
           selector output is used for READOUT ONLY, never in the reward.

⚠️ BATCH-SHAPED CONTEXT. The reward broadcasts ctx against ``traj [B,N,G,S,2]``;
per-window facts must therefore be ``[B,1,1,…]``. No-lead windows carry a
FAR-LEAD SENTINEL (x = 1e6): constant across the window's candidates, so it
cancels in BOTH group-relative advantages — it only distorts the absolute
logged reward, which is why R1 is reported per-component too.

Tier: T0 training-side. NON-PARITY corpus. No driving claim lives here.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tanitad.refs import refc  # noqa: E402
from tanitad.rl import (PostTrainConfig, RewardSpec, rewards as RW)  # noqa: E402
from tanitad.rl.posttrain import run_posttrain, select_trainable  # noqa: E402
from tanitad.rl.refcv3_adapter import make_refcv3_sample_fn  # noqa: E402

HORIZONS = (5, 10, 15, 20)          # frame offsets @10 Hz -> 0.5/1.0/1.5/2.0 s
DT_TRAJ = 0.5                        # waypoint spacing in seconds
WINDOW = 8                           # stacked frames the encoder consumes
EXPECT_PARAMS = 104_191_577          # registry, measured at instantiation
FAR_LEAD_X = 1.0e6                   # no-lead sentinel (constant per window)
LEAD_LAT_M, LEAD_MAX_GAP_M = 2.0, 80.0


def ego_frame_np(px, py, x0, y0, yaw0):
    dx, dy = px - x0, py - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    return dx * c + dy * s, -dx * s + dy * c


def load_agents(path):
    per = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                per.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r["agents"]
    return per


class WindowSource:
    """Windows over joined episodes with a tiny episode LRU (frames are 117 MB/ep)."""

    def __init__(self, epdir, agents_path, seed=0, lru=60, k_max=32):
        # lru=60 holds the WHOLE 54-episode pilot corpus (~6.3 GB uint8) in RAM.
        # The measured risk was data-boundness: 117 MB torch.load per miss with
        # 2 random episodes/step would have dominated the 0.2-0.35 s compute
        # floor. One warm pass through the corpus, then zero disk reads.
        self.epdir, self.agents = epdir, load_agents(agents_path)
        self.stems = sorted(self.agents)
        self.rng = np.random.default_rng(seed)
        self.lru, self.cache, self.k_max = lru, {}, k_max
        if not self.stems:
            raise SystemExit("[pilot] ⛔ zero joined episodes — nothing to train on")

    def _ep(self, stem):
        if stem not in self.cache:
            if len(self.cache) >= self.lru:
                self.cache.pop(next(iter(self.cache)))
            self.cache[stem] = torch.load(os.path.join(self.epdir, stem + ".pt"),
                                          map_location="cpu", weights_only=False)
        return self.cache[stem]

    def window(self, stem=None, t0=None):
        stem = stem or self.stems[int(self.rng.integers(len(self.stems)))]
        d = self._ep(stem)
        T = int(d["poses"].shape[0])
        lo, hi = WINDOW - 1, T - HORIZONS[-1] - 1
        if hi <= lo:
            return None
        t0 = int(t0 if t0 is not None else self.rng.integers(lo, hi))
        poses = d["poses"].numpy()
        frames = d["frames_u8"][t0 - WINDOW + 1: t0 + 1].float() / 255.0  # [W,9,H,W]

        x0, y0, yaw0 = poses[t0, 0], poses[t0, 1], poses[t0, 2]
        fut = poses[[t0 + h for h in HORIZONS]]
        gx, gy = ego_frame_np(fut[:, 0], fut[:, 1], x0, y0, yaw0)
        gt = torch.tensor(np.stack([gx, gy], -1), dtype=torch.float32)  # [S,2]

        obs = [(float(a["cx"]), float(a["cy"]))
               for a in self.agents[stem].get(t0, [])
               if 0.0 < a["cx"] < LEAD_MAX_GAP_M][: self.k_max]
        inlane = [(x, y) for x, y in obs if abs(y) <= LEAD_LAT_M]
        lead_xy = min(inlane, key=lambda p: p[0]) if inlane else (FAR_LEAD_X, 0.0)

        return {"stem": stem, "t0": t0, "frames": frames,
                "v0": float(poses[t0, 3]), "gt": gt,
                "obs": obs, "lead_xy": lead_xy, "has_lead": bool(inlane)}


def collate(wins, device):
    """Windows -> a model batch + BATCH-SHAPED reward context ([B,1,1,…])."""
    b = len(wins)
    k = max(1, max(len(w["obs"]) for w in wins))
    obs = torch.full((b, k, 2), FAR_LEAD_X)
    for i, w in enumerate(wins):
        for j, (x, y) in enumerate(w["obs"]):
            obs[i, j, 0], obs[i, j, 1] = x, y
    lead = torch.stack([
        torch.tensor(w["lead_xy"], dtype=torch.float32).expand(len(HORIZONS), 2)
        for w in wins])                                          # [B,S,2] (static)
    return {
        "frames": torch.stack([w["frames"] for w in wins]).to(device),
        "v0": torch.tensor([w["v0"] for w in wins], device=device),
        "gt_traj": torch.stack([w["gt"] for w in wins]).to(device),   # [B,S,2]
        "obstacles": obs.unsqueeze(1).unsqueeze(1).to(device),        # [B,1,1,K,2]
        "lead_path": lead.unsqueeze(1).unsqueeze(1).to(device),       # [B,1,1,S,2]
        "dt": DT_TRAJ,
        "has_lead": [w["has_lead"] for w in wins],
    }


#: ⛔ THE CROSS-ARM RULER FOR R5, FIXED FOR THE WHOLE CAMPAIGN. D-SAFE-CAL
#: varies what arms TRAIN against; if the metric moved with it, a "safer" arm
#: would just be one graded more leniently.
R5_FIXED_SAFE_M = 5.0


def build_ctx(batch, out=None, extras=None):
    """Reward context for one batch.

    ⚠️ ``extras`` carries THRESHOLD OVERRIDES (D-SAFE-CAL). It is deliberately
    explicit rather than a module global, because the TRAINING context and the
    READOUT context must be allowed to DIFFER: an arm trains against its own
    ``proximity_safe_m`` while every reported R1/R2/R3 is scored on the DEFAULT
    spec, so arms stay comparable across the whole campaign. A global would
    silently make the readout follow the arm and destroy that comparability —
    the arms would each be graded by their own ruler.
    """
    ctx = {k: batch[k] for k in ("gt_traj", "obstacles", "lead_path", "v0", "dt")}
    if extras:
        ctx.update(extras)
    return ctx


# --------------------------------------------------------------------------- #
# THE CONFIG CONTRACT (2026-09-07)                                             #
# --------------------------------------------------------------------------- #
# ⛔ WHY THIS EXISTS, AND WHY THE TWO GUARDS ABOVE IT CANNOT DO ITS JOB.
# `EXPECT_PARAMS` and `load_state_dict(strict=True)` are both **SHAPE** guards:
# one counts parameters, the other matches tensor NAMES. The divergence that
# actually destroys an RL run is **BEHAVIOURAL** — `anchors.v0_conditioned` is a
# plain bool (`refc.py:383`, consumed at `refc.py:1391`) that adds no parameter
# and renames no tensor, so a state dict built under one value loads perfectly
# under the other and the model then runs a DIFFERENT ACTION SPACE. A shape
# guard cannot see a behaviour flag. That is the whole gap.
#
# ⚠️ AND THE FIX IS NOT "REBUILD FROM `ck['cfg']`". A stored config can itself
# be wrong or partial — the shipped cold start's own record carries 38 of this
# config's 96 leaves and NOT `v0_conditioned` (MEASURED, below) — so trusting it
# blindly swaps one unverified source of truth for another. What is built here
# is a COMPARISON THAT REFUSES, plus a written record of what was compared:
# a contract checked but not written down is not evidence.
#
# THREE CASES, DELIBERATELY NOT COLLAPSED:
#   AGREE   every leaf the checkpoint carries matches the rebuilt config
#           -> proceed, and STAMP what was compared.
#   DISAGREE any behaviour-affecting leaf differs
#           -> ⛔ SystemExit naming the field, BOTH values, and the consequence.
#   ABSENT  the checkpoint carries no config for a leaf (or none at all)
#           -> ⛔ neither proceed silently NOR refuse: state the assumed default,
#           mark it UNVERIFIED, and stamp it. An absent config must become a
#           STATED assumption, not an invisible one.

# ⭐ WHAT A WRONG ASSUMPTION ACTUALLY DOES, IN THE OPERATOR'S LANGUAGE. An
# operator who reads "v0_conditioned differs" shrugs; one who reads "every
# anchor would be offered at 10 m/s instead of the measured speed" does not.
# Every entry below is MEASURED FROM `refc.py` AT THE CITED LINE, not inferred.
CFG_CONSEQUENCE = {
    "anchors.v0_conditioned":
        "THE ANCHOR VOCABULARY ITSELF — the action space RL would optimise "
        "over. FALSE makes `RefCDecoder.roll_bank` (refc.py:1715) return the "
        "STORED `decoder.anchors` expanded unchanged for every row: one bank "
        "rolled once at `anchors.ref_speed_ms` = 10.0 m/s, so EVERY anchor is "
        "offered at the 10 m/s reference INSTEAD OF the window's own measured "
        "speed — identically at 5 m/s and at 25 m/s. TRUE rolls the bank per "
        "window from that measured speed through `rollout_unicycle`. MEASURED "
        "(refc.py:366-374, banked 4,823-window surface): the fixed bank reads "
        "0.3773 m oracle-in-vocabulary vs 0.2610 m for the v0-conditioned "
        "family — the best path the vocabulary CAN express is 0.1163 m worse "
        "before a single weight is trained. It also forces `prior_bank = None` "
        "(refc.py:2131), so `_lan_anchor_prior` / `_goal_along_prior` score the "
        "shared 10 m/s anchors instead of the per-window bank (refc.py:1977). "
        "=> the pilot would post-train, converge and report R1/R2/R3 on an "
        "ACTION SPACE THE CHECKPOINT WAS NEVER TRAINED IN.",
    "anchors.ref_speed_ms":
        "the reference speed the fixed bank is rolled at, and the speed a "
        "WITHHELD (ego-dropout) row is rolled at even when v0-conditioned "
        "(refc.py:1719-1729). Wrong here silently re-scales every anchor's "
        "along-track reach.",
    "anchors.control_units":
        "what channel 1 of `anchor_controls` MEANS — 'kappa' is curvature 1/m "
        "integrated literally; 'alat' is lateral acceleration, with curvature "
        "DERIVED per window as a_lat / max(v0, alat_v_floor)^2 (refc.py:1735). "
        "MEASURED (refc.py:387-395): under 'kappa' 104 of 117 anchors break a "
        "mu = 0.7 friction circle at 27 m/s; under 'alat' 0 of 117 do. Same "
        "tensor names, same parameter count, different geometry.",
    "anchors.alat_v_floor_ms":
        "the speed floor in the alat->kappa conversion; too low lets a "
        "standing-start window ask for near-infinite curvature.",
    "anchors.kappa_cap":
        "the geometric curvature bound applied after the alat conversion.",
    "ego_dropout":
        "the per-sample Bernoulli zeroing of v0 during TRAINING. The pilot's "
        "readout forces eval mode, but the RL forward passes do not — a wrong "
        "value changes how often the policy sees its own speed.",
    "route_dropout":
        "the per-sample masking of the LAN route input during training.",
    "hierarchy":
        "whether the strategic context reaches the decoder condition at all.",
    "no_strategic":
        "the PI 2026-09-06 strategic BYPASS. Adds no parameter and renames no "
        "tensor (a pure forward gate) — invisible to both shape guards — yet it "
        "cuts ctx from the decoder and forces the S5 route prior off.",
    "graft_maneuver":
        "whether maneuver logits reweight the anchor priors (H19).",
    "factored_maneuver":
        "D-TAC1 F2: lat(3) x lon(3) heads in place of one 5-way softmax. This "
        "one DOES move the count (+897 params) so EXPECT_PARAMS catches it — "
        "listed for completeness, not because the shape guards are blind here.",
    "tactical_speed_input":
        "D-TAC1 F1: whether the tactical head reads the measured speed.",
    "man_prior_tau":
        "D-TAC1 F3: logit-adjustment strength. A float — zero parameter "
        "change — that changes the DECISION RULE the maneuver head applies.",
    "graft_prior_center":
        "whether the anchor grafts are fed the log-likelihood centre.",
    "decoder.sampler":
        "the refcv5 control sampler. 'none' vs anything else changes what the "
        "decoder's state IS; refc.py:2091 already refuses a control-head "
        "sampler on a non-v0-conditioned vocabulary, which is the same class of "
        "contradiction this contract is here to catch one layer earlier.",
    "decoder.sampler_space":
        "whether the sampler operates in control space or metre space.",
    "decoder.sampler_groups":
        "fan width; > 1 emits [B, G*N, ...] and refc.py:2098 refuses it because "
        "three consumers would be silently MIS-INDEXED.",
    "decoder.feasible_decode":
        "the Kamm-circle feasibility projection during decode.",
    "decoder.feasible_entry":
        "feasibility applied at the entry prefix slots.",
    "sel_reach_clamp":
        "S2: the bounded-acceleration reachability band that filters the "
        "ARGMAX (refc.py:2358). Off, an unreachable candidate can win the pick.",
    "sel_refined":
        "S1: whether the REFINED fan is ranked by the refined confidence.",
    "sel_anchor_prefilter":
        "S2b: the same band moved BEFORE the decode, so pruned anchors are "
        "never decoded and come back conf = -inf.",
    "sel_accel_max":
        "the acceleration bound the S2 band is computed with.",
    "graft_route": "S5: the route prior on the selection surface.",
    "graft_goal": "S6: the goal-bearing prior on the selection surface.",
    "graft_gp_point": "S7: the goal-point prior on the selection surface.",
    "graft_lan": "LAN lane-anchored route conditioning.",
    "graft_cons": "the consequence-score graft on the selection surface.",
    "grounded_selector": "progress/collision proxy in place of top-1 conf.",
    "ego_valid_channel":
        "the explicit ego-validity channel (X15) — what stops the reachability "
        "band from reading a dropped-out speed as a real one (refc.py:2974).",
    "nav_known_channel": "the explicit nav-known channel.",
    "refc1": "the refc1 target-speed head variant.",
    "speed_max": "the upper edge of the target-speed class bins.",
    "speed_bins": "how many target-speed classes the head emits.",
}

# ⚠️ WHAT A FIELD WITH NO CURATED CONSEQUENCE GETS. Not silence, and not a
# waiver: an unexplained configuration difference between the model that was
# trained and the model being rebuilt is exactly the thing this contract exists
# to stop, so the default is to refuse and say that the consequence is unknown.
CFG_CONSEQUENCE_UNKNOWN = (
    "no consequence is recorded for this field in CFG_CONSEQUENCE. It is still "
    "a configuration difference between the model the checkpoint was trained "
    "under and the one this run rebuilt, and an UNEXPLAINED difference is not a "
    "safe one — refusing rather than guessing.")

# ⭐ The fields whose ASSUMED value gets printed loudly when the checkpoint does
# not carry it. Everything in CFG_CONSEQUENCE is behaviour-affecting by
# construction, so this is exactly that key set — kept as one name so the
# ABSENT banner and the DISAGREE message cannot drift apart.
BEHAVIOUR_CRITICAL = frozenset(CFG_CONSEQUENCE)


def _cfg_leaves(obj, prefix=""):
    """Flatten a RefCConfig (or a JSON dump of one) to dotted leaf -> value.

    ⚠️ Tuples become lists ON BOTH SIDES. A stored config has been through JSON,
    where `(5, 10, 15, 20)` comes back as `[5, 10, 15, 20]`; comparing those raw
    would report a DIFFERENCE ON EVERY TUPLE FIELD and the guard would refuse
    every checkpoint — the failure mode that gets a guard deleted rather than
    fixed. Normalising here is what makes the AGREE half reachable.
    """
    out = {}
    if dc.is_dataclass(obj) and not isinstance(obj, type):
        items = [(f.name, getattr(obj, f.name)) for f in dc.fields(obj)]
    elif isinstance(obj, dict):
        items = list(obj.items())
    else:
        return {prefix: obj}
    for k, v in items:
        key = f"{prefix}.{k}" if prefix else str(k)
        if (dc.is_dataclass(v) and not isinstance(v, type)) or isinstance(v, dict):
            out.update(_cfg_leaves(v, key))
        elif isinstance(v, (tuple, list)):
            out[key] = list(v)
        else:
            out[key] = v
    return out


def _stored_cfg(ck, ckpt_path, built_leaves):
    """Find the checkpoint's OWN config. Returns (leaves | None, provenance).

    Two places, in order of authority:
      1. inside the checkpoint  — `ck['cfg']` / `ck['config']`
      2. beside it              — `config.json` in the checkpoint's directory,
                                   either `{"cfg": {...}}` or a bare cfg dict.
    ⛔ TYPE-CONFUSION GUARD. A `cfg` key is NOT proof of a MODEL config. This
    pilot's own `ckpt_after.pt` saves `{"model": ..., "cfg": PostTrainConfig
    .to_dict()}`, whose keys (`method`, `group_size`, `lr`, ...) share NOTHING
    with RefCConfig's. Chain a pilot output back in as `--ckpt` and a naive
    reader would compare an RL schedule against a network architecture and call
    the result a contract. A candidate is accepted only if it actually looks
    like a RefCConfig; otherwise it is reported as absent, WITH the reason.
    """
    tried = []
    cands = [(ck.get(k), f"ckpt[{k!r}]") for k in ("cfg", "config")
             if isinstance(ck, dict) and k in ck]
    side = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
    if os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as fh:
                blob = json.load(fh)
            inner = blob.get("cfg", blob) if isinstance(blob, dict) else blob
            cands.append((inner, f"sidecar {side}"))
        except Exception as exc:                     # a broken sidecar is not
            tried.append(f"{side}: unreadable ({exc})")   # a reason to proceed
    for raw, prov in cands:
        if not isinstance(raw, dict) or not raw:
            tried.append(f"{prov}: not a non-empty dict")
            continue
        leaves = _cfg_leaves(raw)
        shared = set(leaves) & set(built_leaves)
        if not shared:
            tried.append(f"{prov}: {len(leaves)} leaves, NONE of them RefCConfig "
                         f"fields — this is some OTHER config (a PostTrainConfig "
                         f"looks exactly like this), not the model's")
            continue
        return leaves, prov, tried
    return None, None, tried


def _v0_corroboration(ck, effective_v0):
    """Independent, MEASURED evidence about `anchors.v0_conditioned` — read off
    the WEIGHTS rather than off anybody's config.

    ⭐ `decoder.anchor_controls` [N, 2] is a PERSISTENT buffer registered
    UNCONDITIONALLY (refc.py:1400), so it travels inside `ck["model"]` and the
    state dict itself carries evidence:
      absent    -> the checkpoint predates the buffer, which the same change
                   introduced as `v0_conditioned` (2026-09-04). It CANNOT have
                   been trained v0-conditioned.
      all zero  -> the fixed-path bank's signature, said in those words at
                   refc.py:2093-2096: "a fixed-path bank carries
                   `anchor_controls` of all zeros".
      non-zero  -> a real (accel, curvature) vocabulary was installed.

    ⚠️ THE INFERENCE IS ONE-DIRECTIONAL ON PURPOSE, AND THIS IS THE PART THAT
    KEEPS THE GUARD ALIVE. `load_anchors` (refc.py:1643-1649) copies `controls`
    whenever the anchor file has them — the flag is NOT consulted — so a
    genuinely FIXED decoder may legally carry non-zero controls. "non-zero while
    running fixed" is therefore NOTED, never refused; refusing it would be a
    false positive on a legal build, and a guard that cries wolf gets deleted
    rather than fixed. Only the other direction is a hard contradiction: rolling
    a bank from controls that are absent or all zero makes EVERY anchor "do
    nothing", which refc.py:2091 already calls "a plausible-looking WRONG
    experiment".

    ⛔ AND IT IS CORROBORATION, NOT A SOURCE OF TRUTH: it never sets the flag.
    """
    sd = ck.get("model") if isinstance(ck, dict) else None
    ev = {"field": "anchors.v0_conditioned", "effective_value": bool(effective_v0)}
    if not isinstance(sd, dict):
        ev.update(state="NO_STATE_DICT", verdict="NO_EVIDENCE")
        return ev
    if "decoder.anchor_controls" not in sd:
        ev["state"] = "BUFFER_ABSENT"
        ev["_reads"] = ("`decoder.anchor_controls` is not in the checkpoint's "
                        "state dict at all, so this checkpoint predates the "
                        "buffer (refc.py:1400) and therefore predates the "
                        "v0-conditioned vocabulary itself")
    else:
        t = sd["decoder.anchor_controls"]
        nz = int(torch.count_nonzero(t))
        ev["state"] = "ALL_ZERO" if nz == 0 else "NON_ZERO"
        ev["shape"] = list(t.shape)
        ev["nonzero_entries"] = nz
        ev["_reads"] = (f"`decoder.anchor_controls` is present with shape "
                        f"{list(t.shape)} and {nz} non-zero entries")
    fixedish = ev["state"] in ("BUFFER_ABSENT", "ALL_ZERO")
    if effective_v0 and fixedish:
        ev["verdict"] = "CONTRADICTED"
    elif not effective_v0 and fixedish:
        ev["verdict"] = "CORROBORATED"
    elif effective_v0:
        ev["verdict"] = "CORROBORATED"
    else:
        ev["verdict"] = "NOTED_NOT_REFUSED"
        ev["_note"] = ("the checkpoint carries a non-zero control vocabulary "
                       "while this run will use the FIXED bank. That is legal — "
                       "`load_anchors` copies controls without consulting the "
                       "flag — but it is also exactly what a mis-set flag would "
                       "look like, so it is recorded rather than waved through")
    return ev


def assert_config_contract(built_cfg, ck, ckpt_path, out_dir):
    """Compare the checkpoint's own config against the rebuilt one, and REFUSE
    a behavioural disagreement. Always writes `config_contract.json`."""
    built = _cfg_leaves(built_cfg)
    stored, prov, tried = _stored_cfg(ck, ckpt_path, built)
    corro = _v0_corroboration(ck, built.get("anchors.v0_conditioned", False))
    stamp = {
        "_what": "the pilot rebuilds the model from refc_config() DEFAULTS; this "
                 "records what the checkpoint's own config said about that, "
                 "field by field",
        "ckpt": os.path.abspath(ckpt_path),
        "ckpt_step": ck.get("step", "?") if isinstance(ck, dict) else "?",
        "built_from": "refc.refc_config()",
        "cfg_provenance": prov,
        "sources_rejected": tried,
        "n_leaves_built": len(built),
        "v0_conditioned_weight_evidence": corro,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_contract.json")

    def _bank(extra):
        stamp.update(extra)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stamp, fh, indent=1, sort_keys=True, default=str)
        return stamp

    def _corroboration_gate(case, banked):
        """⛔ Refuse a value the WEIGHTS contradict — whether that value came
        from the checkpoint's config or from an assumed default. Runs AFTER the
        config comparison so a config disagreement keeps precedence."""
        print(f"[pilot]      v0 weight-evidence: {corro['state']} -> "
              f"{corro['verdict']} (running with "
              f"anchors.v0_conditioned = {corro['effective_value']!r})",
              flush=True)
        if corro["verdict"] != "CONTRADICTED":
            return
        _bank({"case": case, "verdict": "REFUSED (v0 weight evidence)",
               **banked})
        raise SystemExit(
            f"[pilot] ⛔ CONFIG CONTRACT REFUSED — this run would set "
            f"anchors.v0_conditioned = True, and the CHECKPOINT'S OWN WEIGHTS "
            f"say that is impossible.\n"
            f"[pilot] ⛔   {corro['_reads']}.\n"
            f"[pilot] ⛔   what that silently changes: the bank would be rolled "
            f"per window from an (accel, curvature) vocabulary that is absent "
            f"or all zeros, so EVERY anchor collapses to 'do nothing' and the "
            f"anchored Gaussian is centred on it — refc.py:2091 calls exactly "
            f"this 'a plausible-looking WRONG experiment'. The run would train, "
            f"converge, and mean nothing.\n"
            f"[pilot] ⛔   {CFG_CONSEQUENCE['anchors.v0_conditioned']}\n"
            f"[pilot] ⛔ Contract written to {path}")

    # ---- CASE 3: no config at all --------------------------------------- #
    if stored is None:
        assumed = {k: built[k] for k in sorted(built)}
        crit = {k: v for k, v in assumed.items() if k in BEHAVIOUR_CRITICAL}
        payload = {"compared": {}, "disagreements": {},
                   "assumed_unverified": assumed,
                   "assumed_unverified_behaviour_critical": crit,
                   "_caveat": "the checkpoint carries no config the pilot could "
                              "read, so EVERY value above is an ASSUMPTION about "
                              "a model someone else trained — not a verified "
                              "fact"}
        print(f"[pilot] ⚠️  CONFIG CONTRACT: case ABSENT — this checkpoint "
              f"carries NO config the pilot can read.", flush=True)
        for t in tried:
            print(f"[pilot]      rejected source: {t}", flush=True)
        _corroboration_gate("ABSENT", payload)
        _bank({"case": "ABSENT",
               "verdict": "PROCEEDING ON UNVERIFIED DEFAULTS", **payload})
        print(f"[pilot] ⚠️  PROCEEDING ON {len(assumed)} UNVERIFIED DEFAULTS. "
              f"The {len(crit)} behaviour-affecting ones are ASSUMED to be:",
              flush=True)
        for k in sorted(crit):
            print(f"[pilot]        ASSUMED {k} = {crit[k]!r}  (UNVERIFIED)",
                  flush=True)
        print(f"[pilot] ⚠️  If any of those is wrong, this run post-trains on a "
              f"model configuration it never confirmed — see "
              f"{os.path.basename(path)}.", flush=True)
        return stamp

    # ---- CASES 1 and 2: compare leaf by leaf ---------------------------- #
    shared = sorted(set(built) & set(stored))
    disagree = {k: {"ckpt": stored[k], "built": built[k]} for k in shared
                if stored[k] != built[k]}
    absent = sorted(set(built) - set(stored))
    unknown = sorted(set(stored) - set(built))

    if disagree:
        lines = [f"[pilot] ⛔ CONFIG CONTRACT REFUSED — the checkpoint's own "
                 f"config ({prov}) DISAGREES with the configuration this pilot "
                 f"rebuilt from refc_config() defaults on "
                 f"{len(disagree)} field(s).",
                 "[pilot] ⛔ The parameter count and strict= load CANNOT see "
                 "this: a behaviour flag changes no tensor name and no "
                 "parameter count, so the weights would have loaded cleanly "
                 "into a model running a DIFFERENT policy."]
        for k in sorted(disagree):
            d = disagree[k]
            lines.append(f"[pilot] ⛔   {k}: checkpoint says {d['ckpt']!r}, "
                         f"this run would use {d['built']!r}")
            lines.append(f"[pilot] ⛔     what that silently changes: "
                         f"{CFG_CONSEQUENCE.get(k, CFG_CONSEQUENCE_UNKNOWN)}")
        lines.append(f"[pilot] ⛔ Fix the CONFIGURATION, not this guard: build "
                     f"the model the checkpoint was trained under, or use a "
                     f"checkpoint trained under this one. Contract written to "
                     f"{path}")
        _bank({"case": "DISAGREE", "verdict": "REFUSED",
               "compared": {k: stored[k] for k in shared},
               "disagreements": disagree,
               "assumed_unverified": {k: built[k] for k in absent},
               "unknown_in_ckpt": unknown})
        raise SystemExit("\n".join(lines))

    case = "AGREE" if not absent else "PARTIAL"
    crit_absent = {k: built[k] for k in absent if k in BEHAVIOUR_CRITICAL}
    payload = {"compared": {k: stored[k] for k in shared},
               "disagreements": {},
               "assumed_unverified": {k: built[k] for k in absent},
               "assumed_unverified_behaviour_critical": crit_absent,
               "unknown_in_ckpt": unknown}
    print(f"[pilot] ✅ CONFIG CONTRACT: case {case} via {prov} — "
          f"{len(shared)}/{len(built)} config fields COMPARED and AGREEING, "
          f"0 disagreements.", flush=True)
    _corroboration_gate(case, payload)
    _bank({"case": case, "verdict": "PROCEEDING", **payload})
    if absent:
        print(f"[pilot] ⚠️  {len(absent)} field(s) are NOT in the checkpoint's "
              f"config and are therefore ASSUMED, not verified — "
              f"{len(crit_absent)} of them behaviour-affecting:", flush=True)
        for k in sorted(crit_absent):
            print(f"[pilot]        ASSUMED {k} = {crit_absent[k]!r}  (UNVERIFIED)",
                  flush=True)
    if unknown:
        print(f"[pilot] ⚠️  {len(unknown)} field(s) in the checkpoint's config "
              f"have no counterpart in refc_config() and were NOT checked: "
              f"{', '.join(unknown[:8])}", flush=True)
    print(f"[pilot]      contract written to {path}", flush=True)
    return stamp


def load_model(ckpt_path, device, out_dir):
    m = refc.RefCModel(refc.refc_config())
    total = sum(p.numel() for p in m.parameters())
    if total != EXPECT_PARAMS:
        raise SystemExit(f"[pilot] ⛔ built {total:,} params, registry says "
                         f"{EXPECT_PARAMS:,} — wrong config, refusing")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # ⭐ BEFORE THE WEIGHTS LAND, NOT AFTER. A behavioural disagreement must
    # refuse while the model is still the one this run built — refusing after
    # `load_state_dict` would leave a mismatched model in memory and would read
    # to the operator as a loading failure rather than a contract failure.
    assert_config_contract(refc.refc_config(), ck, ckpt_path, out_dir)
    m.load_state_dict(ck["model"], strict=True)
    step = ck.get("step", "?")
    print(f"[pilot] cold start loaded: {total:,} params @ step {step}", flush=True)
    return m.to(device)


@torch.no_grad()
def readout(model, src: WindowSource, spec: RewardSpec, cfg, device, n=120):
    """R1/R2/R3 on a FIXED, seed-derived window set (identical before/after).

    ⛔ EVAL MODE IS MANDATORY AND IS SET HERE (TRAIN-C5, 2026-08-29).
    The first P-RC21 readout ran with ``model.training == True`` — nn.Module's
    default, never overridden — so EVERY reported number was measured with
    ``ego_dropout=0.5`` and ``route_dropout=0.5`` zeroing half the ego/route
    inputs, plus stochastic truncated-diffusion noise. MEASURED impact on the
    cold start, 3 seeds each (`raw/p_rc21/eval_mode_impact.json`):

        train mode  R3 2.000 m (run-to-run spread 0.099 m)
        eval  mode  R3 0.726 m (run-to-run spread **0.000 m**)

    ⇒ the readout was reporting **+175.7 %** against the deployed configuration,
    and its 17 % run-to-run wobble was dropout, not diffusion noise. ⭐ In eval
    mode the decoder is deterministic BY CONSTRUCTION (`refc.py:1396` — noise is
    ``zeros_like`` when not training), so before/after differences are EXACT and
    the seeding I had pre-registered is unnecessary.
    """
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(1234)
    picks = []
    for stem in src.stems:
        T = int(src._ep(stem)["poses"].shape[0])
        lo, hi = WINDOW - 1, T - HORIZONS[-1] - 1
        for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(8, hi - lo),
                                    replace=False)):
            picks.append((stem, int(t0)))
    picks = picks[:n]

    per_ep: dict[str, dict] = {}
    comp_sums: dict[str, float] = {}
    n_win = 0
    for stem, t0 in picks:
        w = src.window(stem, t0)
        if w is None:
            continue
        batch = collate([w], device)
        out = model(batch["frames"], None, batch["v0"],
                    steps=int(cfg.decoder_steps))
        fan = out["anchor_traj"]                                  # [1,N,S,2]
        ctx = build_ctx(batch)
        parts = spec.per_component(fan, ctx)
        coll = RW.COMPONENTS["collision"](fan, ctx)               # [1,N]
        sel = int(out["sel_score"].argmax(dim=1))                 # readout only
        ade = float((fan[0, sel] - batch["gt_traj"][0]).norm(dim=-1).mean())

        # ⭐ R5 — the SELECTED candidate's clearance violation. Added for
        # PREREG_D_SAFE_CAL exit 3, which is otherwise unmeasurable (TRAIN-C6:
        # a pre-registered exit must name a field the readout actually emits).
        #
        # ⚠️ WHY IT IS SEPARATE FROM R2. R2 is the FAN collision rate — a
        # property of all 128 candidates. R5 is a property of the ONE candidate
        # the selector picks, i.e. the trajectory that would actually be driven.
        # A policy can improve the fan while degrading what is driven, and that
        # split is exactly the reward-hacking signature the prereg's exit 3
        # rejects on. Reporting only R2 cannot see it.
        # ⭐ R5 is reported on TWO rulers and they answer different questions:
        #   viol      @ R5_FIXED_SAFE_M (5.0) — the CROSS-ARM ruler. Never the
        #             arm's own threshold, or each arm is graded by the target it
        #             was trained on and no comparison means anything.
        #   viol_arm  @ the arm's threshold — did the TRAINED objective move?
        # Exit 3 (reward hacking) is read on the FIXED one.
        viol = viol_arm = float("nan")
        obs_ = ctx.get("obstacles")
        if obs_ is not None and obs_.numel():
            r_ = (float(ctx.get("ego_radius_m", 1.0))
                  + float(ctx.get("obs_radius_m", 1.0)))
            d_ = (fan[0, sel].unsqueeze(-2) - obs_.reshape(-1, 2)
                  .unsqueeze(-3)).norm(dim=-1)
            clr_ = float((d_ - r_).clamp_min(0.0).amin())
            viol = float(clr_ < R5_FIXED_SAFE_M)
            viol_arm = float(clr_ < float(ctx.get("proximity_safe_m",
                                                  R5_FIXED_SAFE_M)))

        e = per_ep.setdefault(stem, {"r": [], "cr": [], "ade": [],
                                    "viol": [], "viol_arm": []})
        e["r"].append(float(spec(fan, ctx).mean()))
        e["cr"].append(float((coll < 0).float().mean()))
        e["ade"].append(ade)
        if viol == viol:                                # NaN-safe: skip no-obstacle
            e["viol"].append(viol)                      # windows rather than
            e["viol_arm"].append(viol_arm)              # scoring them as clean
        for k, v in parts.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.mean())
        n_win += 1

    model.train(was_training)
    # ⚠️ an episode whose every window lacked obstacles has viol == [] — np.mean
    # of an empty list is nan WITH a RuntimeWarning, and a silent nan here would
    # propagate into R5 as a number-shaped absence. Emit nan deliberately.
    ep_means = {k: {m: (float(np.mean(v[m])) if v[m] else float("nan"))
                    for m in v} for k, v in per_ep.items()}
    arr = lambda m: np.array([e[m] for e in ep_means.values()])
    boot = []
    ids = list(ep_means)
    brng = np.random.default_rng(7)
    for _ in range(2000):
        pick = brng.choice(len(ids), size=len(ids), replace=True)
        boot.append([arr("r")[pick].mean(), arr("cr")[pick].mean(),
                     arr("ade")[pick].mean()])
    lo_, hi_ = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "n_windows": n_win, "n_episodes": len(ep_means),
        "R1_fan_reward": {"mean": float(arr("r").mean()),
                          "ci95": [float(lo_[0]), float(hi_[0])]},
        "R2_fan_collision_rate": {"mean": float(arr("cr").mean()),
                                  "ci95": [float(lo_[1]), float(hi_[1])]},
        "R3_sel_ade_m": {"mean": float(arr("ade").mean()),
                         "ci95": [float(lo_[2]), float(hi_[2])]},
        "R4_component_means": {k: v / max(n_win, 1) for k, v in comp_sums.items()},
        # ⭐ R5 — see the loop above. ``n`` is the episode count with ANY
        # obstacle-bearing window; it is NOT n_episodes, and a run where the two
        # differ is telling you the obstacle join is thin.
        "R5_gt_clearance_violation_frac": {
            "mean": float(np.nanmean(arr("viol"))) if len(ep_means) else float("nan"),
            "n_episodes_with_obstacles": int(np.isfinite(arr("viol")).sum()),
            "proximity_safe_m": R5_FIXED_SAFE_M,
            "mean_at_arm_threshold": (float(np.nanmean(arr("viol_arm")))
                                      if len(ep_means) else float("nan")),
            "_what": "fraction of scored windows whose SELECTED candidate is "
                     "inside proximity_safe_m — the driven path, not the fan",
        },
        "per_episode": ep_means,      # ⭐ stored so the PAIRED bootstrap the house
                                      # rule requires is computable post-hoc —
                                      # the first readout kept aggregates only
        "eval_mode": True,
        "_estimator": "episode-cluster bootstrap, 2000 reps (UNPAIRED here; "
                      "per_episode is stored so a PAIRED test is computable) — "
                      "⚠️ few clusters, claims only when CIs separate",
        "_tier": "T0 training-side; NON-PARITY corpus; readout-only selector use",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-epdir", required=True)
    ap.add_argument("--train-agents", required=True)
    ap.add_argument("--val-epdir", required=True)
    ap.add_argument("--val-agents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reward",
                    choices=("default", "hackable", "proximity", "proximity0"),
                    default="default",
                    help="'proximity' = DEFAULT + the graded barrier at 0.5 "
                         "(AMENDMENT 2, weight pre-registered before the run); "
                         "'proximity0' = the D-SAFE-CAL FLOOR ARM — the barrier "
                         "PRESENT but at weight 0, so it isolates the threshold "
                         "from the mere presence of the term. ⛔ If this arm "
                         "moves anything, the harness is broken, not the "
                         "science (D-SAFE-CAL exit 5).")
    ap.add_argument("--proximity-safe-m", type=float, default=5.0,
                    help="the clearance threshold this arm TRAINS against. "
                         "⛔ Reported R1/R2/R3 and R5 stay on the DEFAULT spec "
                         "and the FIXED 5.0 ruler so arms remain comparable.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-anchor", type=float, default=0.0,
                    help="reference-policy trust region (0 = the unanchored "
                         "P-RC21 configuration, kept as the sweep's own null)")
    ap.add_argument("--lru", type=int, default=60,
                    help="episode cache size; 60 = whole pilot corpus in RAM")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.reward == "hackable":
        weights = dict(RW.HACKABLE_WEIGHTS)
    elif a.reward == "proximity":
        weights = dict(RW.DEFAULT_WEIGHTS); weights["proximity"] = 0.5
    elif a.reward == "proximity0":
        weights = dict(RW.DEFAULT_WEIGHTS); weights["proximity"] = 0.0
    else:
        weights = dict(RW.DEFAULT_WEIGHTS)
    cfg = PostTrainConfig(
        method="grpo", group_size=4, steps=a.steps, batch=a.batch, lr=1e-5,
        seed=a.seed, dt=DT_TRAJ, decoder_steps=2, reward_weights=weights,
        w_anchor=a.w_anchor,
        trainable_prefixes=("decoder",),
        exclude_prefixes=("decoder.conf_head",),
        forbidden_prefixes=("decoder.conf_head", "scorer"),
        w_imitation=0.0,   # ⚠️ stated: gt_similarity is OUT of the reward AND no
                           # IL loss is wired for v2.1 in this pilot — the
                           # imitation anchor is the FROZEN 96 % of the model.
                           # R3's +10 % guard is the drift alarm instead.
        out_dir=a.out, run_name=f"p-rc21-{a.reward}-dsafe{a.proximity_safe_m:g}")
    cfg.validate()

    # ⭐ EMIT THE RESOLVED WEIGHTS BEFORE TRAINING (D-SAFE-CAL-2 §1c).
    # D-SAFE-CAL's arm D was bit-identical to the default arm because
    # `--reward proximity0` set a weight DEFAULT_WEIGHTS never contained — a flag
    # name trusted without measuring what it does. Printing the resolved dict
    # makes an arm that cannot vary its own objective visible BEFORE it runs
    # instead of after the readout. Operating-standard rule 1: a claim that
    # decides a GPU-day must be MEASURED, and this is the cheapest measurement.
    print(f"[pilot] RESOLVED reward_weights = {json.dumps(weights, sort_keys=True)}",
          flush=True)
    # ⛔ COMPARE THE *EFFECTIVE* OBJECTIVE, NOT THE RAW DICT. A term at weight
    # 0.0 contributes nothing, so {"proximity": 0.0, ...} and {...} are DIFFERENT
    # dicts and the SAME objective. The first version of this guard compared
    # json.dumps() and would therefore have MISSED the very defect it was written
    # for (D-SAFE-CAL arm D). Dropping zero-weight entries first is what makes
    # the comparison mean "can this arm vary anything".
    _eff = lambda d: {k: v for k, v in d.items() if v}
    if _eff(weights) == _eff(dict(RW.DEFAULT_WEIGHTS)):
        print("[pilot] ⚠️  NOTE: this arm's EFFECTIVE objective is "
              "IDENTICAL to DEFAULT_WEIGHTS (zero-weight terms dropped) — it "
              "cannot vary any term relative to the default arm. Correct for a "
              "reproduction control, a DEFECT for anything else.", flush=True)

    model = load_model(a.ckpt, device, a.out)
    spec = RewardSpec(weights=weights, dt=DT_TRAJ)
    train_src = WindowSource(a.train_epdir, a.train_agents, seed=a.seed, lru=a.lru)
    val_src = WindowSource(a.val_epdir, a.val_agents, seed=a.seed)
    print(f"[pilot] train eps {len(train_src.stems)} · val eps "
          f"{len(val_src.stems)} · device {device} · reward {a.reward}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    before = readout(model, val_src, RewardSpec(dt=DT_TRAJ), cfg, device)
    json.dump(before, open(os.path.join(a.out, "readout_before.json"), "w"),
              indent=1)
    print(f"[pilot] BEFORE  R1 {before['R1_fan_reward']['mean']:+.4f}  "
          f"R2 {before['R2_fan_collision_rate']['mean']:.3%}  "
          f"R3 {before['R3_sel_ade_m']['mean']:.3f} m", flush=True)

    def batches():
        step = 0
        while step < a.steps:
            wins = []
            while len(wins) < a.batch:
                w = train_src.window()
                if w is not None:
                    wins.append(w)
            yield collate(wins, device)
            step += 1

    # ⭐ THE REFERENCE POLICY: a frozen deepcopy of the COLD START, taken
    # before a single optimiser step. Only built when the trust region is on, so
    # w_anchor=0 reproduces the original unanchored arm exactly.
    reference = None
    if cfg.w_anchor > 0.0:
        import copy
        from tanitad.rl.anchor import ReferencePolicy
        reference = ReferencePolicy(model).to(device)
        n_frozen = reference.assert_frozen()
        print(f"[pilot] reference policy frozen: {n_frozen:,} params, "
              f"w_anchor={cfg.w_anchor} form={cfg.anchor_form}", flush=True)
    # ⭐ THE ARM'S THRESHOLD ENTERS HERE AND ONLY HERE (D-SAFE-CAL). The
    # readout above/below calls plain build_ctx, so every reported number is on
    # the campaign's fixed ruler while the policy optimises against the arm's.
    train_extras = {"proximity_safe_m": float(a.proximity_safe_m)}
    print(f"[pilot] TRAINING against proximity_safe_m="
          f"{a.proximity_safe_m:g} m · READOUT fixed at {R5_FIXED_SAFE_M:g} m",
          flush=True)
    sample_fn = make_refcv3_sample_fn(
        model, cfg,
        build_ctx=lambda b, out=None: build_ctx(b, out, extras=train_extras),
        reference=reference)
    summary = run_posttrain(model, sample_fn, cfg, batches=batches())

    # ⛔ SAVE THE TRAINED DECODER. The first pilot saved none, so when the
    # eval-mode defect surfaced there was no way to re-measure P1's outcome
    # without re-training. A run whose result cannot be re-measured is a run
    # that has to be repeated.
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()},
               os.path.join(a.out, "ckpt_after.pt"))
    after = readout(model, val_src, RewardSpec(dt=DT_TRAJ), cfg, device)
    json.dump(after, open(os.path.join(a.out, "readout_after.json"), "w"),
              indent=1)
    print(f"[pilot] AFTER   R1 {after['R1_fan_reward']['mean']:+.4f}  "
          f"R2 {after['R2_fan_collision_rate']['mean']:.3%}  "
          f"R3 {after['R3_sel_ade_m']['mean']:.3f} m", flush=True)

    delta = {
        "R1": after["R1_fan_reward"]["mean"] - before["R1_fan_reward"]["mean"],
        "R2": (after["R2_fan_collision_rate"]["mean"]
               - before["R2_fan_collision_rate"]["mean"]),
        "R3_rel": (after["R3_sel_ade_m"]["mean"]
                   / max(before["R3_sel_ade_m"]["mean"], 1e-9) - 1.0),
    }
    summary["pilot_delta"] = delta
    summary["w_anchor"] = cfg.w_anchor
    json.dump({k: v for k, v in summary.items() if k != "history"},
              open(os.path.join(a.out, "pilot_summary.json"), "w"), indent=1)
    print(f"[pilot] DELTA   R1 {delta['R1']:+.4f}  R2 {delta['R2']:+.3%}  "
          f"R3 {delta['R3_rel']:+.2%}  -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
