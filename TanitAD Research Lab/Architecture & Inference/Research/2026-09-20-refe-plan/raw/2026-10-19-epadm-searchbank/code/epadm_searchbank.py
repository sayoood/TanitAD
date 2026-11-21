"""EP-ADMISSION SEARCH-BANK ADAPTER (SEARCH-P2-CONFIRM DQ6, MM ruling 2026-10-19).  CPU only; numpy + the FROZEN gate `ep_admission.py` (sha256-pinned) and, through it, the pinned EP-RELIAB libs.

WHAT IT DOES.  The frozen gate (`ep_admission.py`, schema `ep-admission-v1`) reads a candidate as a BANK OF SETS: one set = a fan of 64 proposals with six-head logits [64, 6] and six-head labels [64, 6];
a pick is `pickrule_lib.rule_pick` of a set's logits; its C-KEYS check needs the candidate's keys to be a bijection onto P2's bank.  A SEARCH output is not a pick of the fan, so this adapter lays each
token's search outputs INSIDE the 64-slot fan, leaving the gate's rules, loaders, estimator and verdict functions UNTOUCHED:

  * one P2 bank: one set per TOKEN (the banked fan, P2's rows, P2's pick slot);
  * three candidate banks, one per SEARCH SEED (the analogue of R1's two EP seeds; the gate's own "on ANY seed" quantifier is the seed rule): one set per (token, search seed);
  * the search output of (token, seed) lives in a HOST slot: the k-th lowest-aggregate non-pick slot of the token's fan (k = seed index; ties to the lowest index).  The label array is SHARED by all four
    banks (the gate's C-LABELS needs it bit-equal): host slot k carries the search output's OWN label (the labeller's), every other slot the banked fan label.
  * a candidate set whose output MOVED carries the search pass's own scorer rows (`pred_logit_pick` at the pick slot, `pred_logit_out` at its host slot); an UNMOVED set is P2's set verbatim (so the gate's
    "P2 as the candidate reads exactly 0" holds per set).  Host slots that are not live in a bank carry the dropped proposal's ORIGINAL row (a filler): it is the lowest-aggregate row of the fan, so it is
    never picked, and no decision statistic reads a non-pick row (A1 / A2 read rows at the two picks only; the shuffle null reads other slots, see ADAPTER_NOTES in the amendment).

GUARDS (each has a RED test and a mutation arm in test_searchbank.py): G1 keys bijective / files aligned; G2 every search output has a finite in-range label and the labeller's identity holds; G3 the search config,
checkpoint, head (null), arm, seeds and token-list sha256 equal the pins; G4 the frozen pick rule reads the intended pick in EVERY set of every candidate bank; G5 the frozen pick rule reads P2's banked pick in the P2 bank.

    python epadm_searchbank.py build  --bundle B --search S --labels L --out-dir D [--expect-config-sha .. --expect-ckpt-sha .. --expect-token-sha ..]
    python epadm_searchbank.py verify --out-dir D --bundle B --search S --labels L --out identity.json      # bit-for-bit identity of every unit value to the source arrays; NO admission decision is computed
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "ep-admission-searchbank-v1"
GATE_SHA256 = "f0193372085022ab7bfc6241f2ca5638fca710cbd4170a19ce0ae95dde6637c4"      # the frozen ep_admission.py (tip fb7ad9a2)
PIN_CONFIG_SHA = "299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441"    # stepB_config.json (SEARCH-P2-CONFIRM section 1)
PIN_CKPT_SHA = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"      # P2-s0 model_final.pt
PIN_SEEDS = (20260927, 20260928, 20260929)
LABEL_ORDER = ["nc", "dac", "ep", "ttc", "c", "ddc"]
N_PROP, N_HEADS, N_SEEDS = 64, 6, 3
GATE_DIRS = (os.environ.get("EP_ADMISSION_DIR"), os.path.join(HERE, "..", "..", "2026-10-19-p6-read-prep", "code"))


class BankError(RuntimeError):
    """A guard of the adapter refused (no bank is written)."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def token_list_sha256(tokens) -> str:
    """The SEARCH-P2-CONFIRM token-list pin: sha256 of the SORTED tokens joined by newlines."""
    return hashlib.sha256("\n".join(sorted(str(t) for t in tokens)).encode()).hexdigest()


_GATE = None


def gate():
    """The frozen `ep_admission` module, imported by path after its sha256 is checked."""
    global _GATE
    if _GATE is None:
        for d in GATE_DIRS:
            p = os.path.join(d, "ep_admission.py") if d else None
            if p and os.path.exists(p):
                sha = sha256_file(p)
                if sha != GATE_SHA256:
                    raise BankError(f"ep_admission.py at {p} has sha256 {sha} != the frozen {GATE_SHA256}")
                if os.path.dirname(os.path.abspath(p)) not in sys.path:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
                spec = importlib.util.spec_from_file_location("ep_admission", p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["ep_admission"] = mod
                spec.loader.exec_module(mod)
                _GATE = mod
                break
        else:
            raise BankError(f"ep_admission.py not found; probed {GATE_DIRS}")
    return _GATE


def pickrule():
    return gate().libs()[1]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sources and guards
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_sources(bundle_npz: str, search_npz: str, labels_npz: str) -> dict:
    """The three source files as plain arrays (float64 kept as float64; nothing is cast here).  Reads no ctx.npy."""
    b, s, lb = np.load(bundle_npz, allow_pickle=False), np.load(search_npz, allow_pickle=False), np.load(labels_npz, allow_pickle=False)
    for name, z, keys in (("bundle", b, ("token", "log", "logits", "sub", "pick")),
                          ("search", s, ("token", "bundle_index", "pick", "seeds", "moved", "pred_logit_out", "pred_logit_pick", "meta")),
                          ("labels", lb, ("token", "bundle_index", "sub", "pdms", "id_sub", "id_pdms", "sub_names", "search_sha256"))):
        miss = [k for k in keys if k not in z.files]
        if miss:
            raise BankError(f"{name} file lacks arrays {miss}")
    src = {"bundle": {k: b[k] for k in b.files}, "search": {k: s[k] for k in s.files}, "labels": {k: lb[k] for k in lb.files},
           "search_file_sha256": sha256_file(search_npz), "bundle_file_sha256": sha256_file(bundle_npz), "labels_file_sha256": sha256_file(labels_npz)}
    src["meta"] = json.loads(str(src["search"]["meta"]))
    return src


def check_alignment(src: dict) -> int:
    """G1: the three files describe the SAME tokens in the SAME order, tokens are unique, every token has a log; the keys are therefore a bijection onto the bundle's units."""
    B, S, L = src["bundle"], src["search"], src["labels"]
    tok = [str(t) for t in B["token"]]
    n = len(tok)
    if n == 0:
        raise BankError("G1: the bundle has no tokens")
    if len(set(tok)) != n:
        raise BankError("G1: the bundle's tokens are not unique (keys would not be a bijection)")
    if [str(t) for t in S["token"]] != tok or [str(t) for t in L["token"]] != tok:
        raise BankError("G1: the search / labels token lists are not the bundle's, in the bundle's order (keys not bijective onto the bundle's units)")
    if not (np.array_equal(S["bundle_index"], np.arange(n)) and np.array_equal(L["bundle_index"], np.arange(n))):
        raise BankError("G1: bundle_index is not arange(n) in the search / labels files")
    if not np.array_equal(S["pick"], B["pick"]):
        raise BankError("G1: the search file's pick differs from the bundle's pick")
    if B["logits"].shape != (n, N_PROP, N_HEADS) or B["sub"].shape != (n, N_PROP, N_HEADS):
        raise BankError(f"G1: bundle logits {B['logits'].shape} / sub {B['sub'].shape} are not [n, {N_PROP}, {N_HEADS}]")
    if len(B["log"]) != n or any(str(x) == "" for x in B["log"]):
        raise BankError("G1: a token has no log")
    return n


def check_pins(src: dict, expect: dict) -> dict:
    """G3: the search file was produced by the pinned arm: P2HEAD (head null), the pinned config / checkpoint, the pinned seeds, on a bundle whose token list hashes to the pinned value."""
    m, S = src["meta"], src["search"]
    out = {}
    def need(name, got, want):
        if want is None:
            raise BankError(f"G3: no expected {name} was given (a bank is never built against an unnamed pin)")
        if got != want:
            raise BankError(f"G3: search {name} {got!r} != the expected {want!r}")
        out[name] = got
    need("arm", m.get("arm"), "P2HEAD")
    if m.get("head_sha256") is not None:
        raise BankError(f"G3: head_sha256 is {m.get('head_sha256')!r}, not null (a head substitution is not the P2HEAD arm)")
    need("config_sha256", m.get("config_sha256"), expect.get("config_sha256"))
    need("ckpt_sha256", m.get("ckpt_sha256"), expect.get("ckpt_sha256"))
    seeds = tuple(int(x) for x in S["seeds"])
    need("seeds", seeds, tuple(expect.get("seeds", PIN_SEEDS)))
    tsha = token_list_sha256(src["bundle"]["token"])
    need("token_list_sha256", tsha, expect.get("token_sha256"))
    dc = m.get("data_clause") or {}
    if dc.get("carve_sha256") not in (None, tsha) or dc.get("bundle_tokens_sha256") not in (None, tsha):
        raise BankError(f"G3: the search file's data clause names a different token list than the bundle's {tsha}")
    if m.get("bundle_sha256") != src["bundle_file_sha256"]:
        raise BankError(f"G3: the search file was made on bundle {str(m.get('bundle_sha256'))[:12]}, not on this bundle file {src['bundle_file_sha256'][:12]}")
    lsha = str(src["labels"]["search_sha256"])
    if lsha != src["search_file_sha256"]:
        raise BankError(f"G3: the labels were made from a different search file ({lsha[:12]} != {src['search_file_sha256'][:12]})")
    return out


def check_labels(src: dict) -> None:
    """G2: every (token, seed) search output has a finite label in [0, 1] in the gate's head order, and the labeller's identity holds (id_sub = the pick's banked label; an UNMOVED output has the pick's label)."""
    B, S, L = src["bundle"], src["search"], src["labels"]
    n = len(B["token"])
    if [str(x) for x in L["sub_names"]] != LABEL_ORDER:
        raise BankError(f"G2: label head order {[str(x) for x in L['sub_names']]} != {LABEL_ORDER}")
    sub = L["sub"]
    if sub.shape != (n, N_SEEDS, N_HEADS):
        raise BankError(f"G2: the search-output labels are {sub.shape}, not [{n}, {N_SEEDS}, {N_HEADS}] (a unit's label is missing)")
    if not np.isfinite(sub).all():
        bad = np.argwhere(~np.isfinite(sub).all(axis=2))
        raise BankError(f"G2: {len(bad)} search outputs have a missing / non-finite label, e.g. (token, seed) {bad[:3].tolist()}")
    if sub.min() < 0.0 or sub.max() > 1.0:
        raise BankError("G2: a search-output label is outside [0, 1]")
    pick_lab = B["sub"][np.arange(n), B["pick"]]
    if not np.array_equal(L["id_sub"], pick_lab):
        raise BankError("G2: the labeller's identity sub-scores differ from the pick's banked label (the labels are not on this bundle)")
    um = ~S["moved"].astype(bool)
    if not np.array_equal(sub[um], np.broadcast_to(pick_lab[:, None, :], sub.shape)[um]):
        raise BankError("G2: an UNMOVED search output carries a label different from the pick's")


def host_slots(logits_f32: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """[n, 3]: the k-th lowest-aggregate NON-pick slot of each fan (aggregate = the frozen pick rule's; ties go to the lowest slot index)."""
    P = pickrule()
    agg = P.rule_aggregate(logits_f32.astype(np.float64))
    agg[np.arange(len(pick)), pick] = np.inf
    order = np.argsort(agg, axis=1, kind="stable")
    return order[:, :N_SEEDS].astype(np.int64)


def check_p2_bundle(src: dict) -> None:
    """G5: the bundle's logits / labels are finite and in range, and the frozen pick rule reads the BANKED pick of every token (so P2's bank is P2)."""
    P = pickrule()
    B = src["bundle"]
    Z, lab, pick = B["logits"].astype(np.float32), B["sub"].astype(np.float32), B["pick"].astype(np.int64)
    if not np.isfinite(Z).all() or not np.isfinite(lab).all() or lab.min() < 0.0 or lab.max() > 1.0:
        raise BankError("G5: the bundle has a non-finite logit or a label outside [0, 1]")
    if not np.array_equal(P.rule_pick(Z), pick):
        raise BankError(f"G5: the frozen pick rule does not read the bundle's banked pick on {int((P.rule_pick(Z) != pick).sum())} tokens")


def assemble(src: dict) -> dict:
    """The arrays of the four banks (float32 = the gate loader's dtype), with guards G4 / G5."""
    P = pickrule()
    B, S, L = src["bundle"], src["search"], src["labels"]
    n = check_alignment(src)
    Z = B["logits"].astype(np.float32)
    lab = B["sub"].astype(np.float32).copy()
    pick = B["pick"].astype(np.int64)
    host = host_slots(Z, pick)
    ar = np.arange(n)
    moved = S["moved"].astype(bool)
    po, pl = S["pred_logit_out"], S["pred_logit_pick"]
    if po.shape != (n, N_SEEDS, N_HEADS) or pl.shape != (n, N_HEADS) or moved.shape != (n, N_SEEDS):
        raise BankError("G1: the search file's logits / moved flags are not [n, 3, 6] / [n, 6] / [n, 3]")
    if not (np.isfinite(po).all() and np.isfinite(pl).all()):
        raise BankError("G1: a non-finite search logit")
    if not np.array_equal(po[~moved], np.broadcast_to(pl[:, None, :], po.shape)[~moved]):
        raise BankError("G4: an UNMOVED output's logits are not the pick's logits")
    for k in range(N_SEEDS):
        lab[ar, host[:, k]] = L["sub"][:, k, :].astype(np.float32)
    cand = []
    for k in range(N_SEEDS):
        Zc = Z.copy()
        mv = moved[:, k]
        Zc[ar[mv], pick[mv]] = pl[mv].astype(np.float32)
        Zc[ar[mv], host[mv, k]] = po[mv, k].astype(np.float32)
        want = np.where(mv, host[:, k], pick)
        got = P.rule_pick(Zc)
        if not np.array_equal(got, want):
            bad = np.nonzero(got != want)[0]
            raise BankError(f"G4: the frozen pick rule does not read the search output as the pick on {len(bad)} sets of seed {k} (moved={bool(mv[bad[0]])}, token {str(B['token'][bad[0]])})")
        cand.append(Zc)
    return {"n": n, "log": np.array([str(x) for x in B["log"]]), "token": np.array([str(x) for x in B["token"]]), "p2_z": Z, "lab": lab, "cand_z": cand, "pick": pick, "host": host, "moved": moved}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_bank(bank_dir: str, name: str, a: dict, z: np.ndarray, ckpt_id: str, extra: dict) -> dict:
    """One `<log>.sets.npz` unit per log (sets sorted by token), plus MANIFEST.sha256.json in the format `ep_admission.read_manifest` reads."""
    os.makedirs(bank_dir, exist_ok=True)
    files = {}
    for lg in sorted(set(a["log"].tolist())):
        idx = np.nonzero(a["log"] == lg)[0]
        idx = idx[np.argsort(a["token"][idx], kind="stable")]
        fn = f"{lg}.sets.npz"
        np.savez(os.path.join(bank_dir, fn), log=a["log"][idx], token=a["token"][idx], step=np.zeros(len(idx), np.int64), rank=np.zeros(len(idx), np.int64),
                 ckpt_step=np.full(len(idx), -1, np.int64), logits=z[idx], labels6=a["lab"][idx], label_order=np.array(LABEL_ORDER), rule=np.array("navsim_v1"), ckpt=np.array(ckpt_id),
                 pick_slot=a["pick"][idx], host_slot=a["host"][idx], layout=np.array(SCHEMA), bank_name=np.array(name), **{k: np.array(v) for k, v in extra.items()})
        p = os.path.join(bank_dir, fn)
        files[fn] = {"path": fn, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    doc = {"what": "EP-ADMISSION search-bank adapter output: sha256 of every unit file", "schema": SCHEMA, "bank": name, "ckpt": ckpt_id, "n_files": len(files), "files": files, **extra}
    with open(os.path.join(bank_dir, "MANIFEST.sha256.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True))
    return {"manifest_sha256": sha256_file(os.path.join(bank_dir, "MANIFEST.sha256.json")), "n_units": len(files)}


def write_ranked(path: str, logs) -> str:
    """The C1 bank's own ranked-log table in the gate's format (all logs selected: first_fraction 0.1 <= the S0 10 % filter; fold = rank mod 5)."""
    lines = ["rank\tlog\tfold\tfirst_fraction"] + [f"{i}\t{lg}\t{i % 5}\t0.1" for i, lg in enumerate(sorted(set(logs)))]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return sha256_file(path)


def build(bundle: str, search: str, labels: str, out_dir: str, expect: dict | None = None) -> dict:
    expect = dict(expect or {})
    expect.setdefault("config_sha256", PIN_CONFIG_SHA)
    expect.setdefault("ckpt_sha256", PIN_CKPT_SHA)
    gate()
    src = read_sources(bundle, search, labels)
    check_alignment(src)
    pins = check_pins(src, expect)
    check_p2_bundle(src)
    check_labels(src)
    a = assemble(src)
    ckpt_id = f"P2-s0 model_final.pt sha256 {pins['ckpt_sha256']}"
    extra = {"search_config_sha256": pins["config_sha256"], "search_file_sha256": src["search_file_sha256"], "labels_file_sha256": src["labels_file_sha256"], "bundle_file_sha256": src["bundle_file_sha256"],
             "token_list_sha256": pins["token_list_sha256"], "gate_sha256": GATE_SHA256}
    rep = {"schema": SCHEMA, "pins": pins, "n_tokens": a["n"], "n_logs": len(set(a["log"].tolist())), "n_candidate_sets_total": a["n"] * N_SEEDS, "moved_by_seed": a["moved"].sum(axis=0).tolist(),
           "moved_share_by_seed": a["moved"].mean(axis=0).tolist(), "tokens_with_any_moved": int(a["moved"].any(axis=1).sum()), "banks": {}, "extra": extra,
           "max_abs_pred_logit_pick_minus_bundle": float(np.abs(src["search"]["pred_logit_pick"] - src["bundle"]["logits"][np.arange(a["n"]), a["pick"]]).max()),
           "layout": "P2 bank: 1 set per token. Candidate bank k (search seed k): 1 set per token. Host slot k = k-th lowest-aggregate non-pick slot; moved sets carry pred_logit_pick at the pick slot and pred_logit_out at the host slot."}
    rep["banks"]["p2"] = write_bank(os.path.join(out_dir, "p2"), "p2", a, a["p2_z"], ckpt_id, {**extra, "seed_index": -1})
    for k in range(N_SEEDS):
        rep["banks"][f"s{k}"] = write_bank(os.path.join(out_dir, f"s{k}"), f"s{k}", a, a["cand_z"][k], ckpt_id, {**extra, "seed_index": k, "search_seed": int(src["search"]["seeds"][k])})
    rep["ranked_logs_sha256"] = write_ranked(os.path.join(out_dir, "ranked_logs.tsv"), a["log"].tolist())
    with open(os.path.join(out_dir, "ADAPTER_REPORT.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1, sort_keys=True))
    return rep


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# verification: the gate's OWN loaders read the banks; every unit value equals the source array bit for bit.  NO admission statistic is computed.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_with_gate(out_dir: str) -> tuple:
    A = gate()
    rows, rsha = A.read_ranked(os.path.join(out_dir, "ranked_logs.tsv"), expected_sha256=None)
    p2 = A.load_bank_units(os.path.join(out_dir, "p2"), rows)
    cands = [A.align_candidate(p2, A.load_bank_units(os.path.join(out_dir, f"s{k}"), rows)) for k in range(N_SEEDS)]
    return p2, cands, rows


def identity_check(out_dir: str, src: dict) -> dict:
    """Recompute the expected value of EVERY unit entry from the source arrays by an independent per-token loop, compare with what the gate's loader returns (np.array_equal, dtype float32).
    -> counts of compared values and mismatches.  P2's pick, the candidate picks and the C-KEYS / C-LABELS alignment are read through the gate's own code."""
    P = pickrule()
    p2, cands, rows = load_with_gate(out_dir)
    B, S, L = src["bundle"], src["search"], src["labels"]
    n = len(B["token"])
    pos = {(str(B["log"][i]), str(B["token"][i])): i for i in range(n)}
    res = {"n_sets_p2": p2["n_sets"], "n_sets_by_candidate": [c["n_sets"] for c in cands], "n_logs": p2["n_logs"], "mismatches": {}, "compared": {}}
    ok = True
    def rowmap(bank):
        return [pos[(k[0], k[1])] for k in bank["keys"]]
    mp = rowmap(p2)
    res["keys_cover_every_bundle_token_once"] = bool(sorted(mp) == list(range(n)))
    pk = B["pick"].astype(np.int64)
    res["p2_rule_pick_equals_bundle_pick"] = bool(np.array_equal(P.rule_pick(p2["z6"]), pk[mp]))
    ok &= res["keys_cover_every_bundle_token_once"] and res["p2_rule_pick_equals_bundle_pick"]
    hosts = {}
    for i in range(n):                                                             # independent host recomputation: full sort of the aggregate, pick removed
        agg = P.rule_aggregate(B["logits"][i].astype(np.float32).astype(np.float64)[None])[0]
        order = [j for j in np.lexsort((np.arange(N_PROP), agg)) if j != pk[i]]
        hosts[i] = order[:N_SEEDS]
    def expect_lab(i):
        e = B["sub"][i].astype(np.float32).copy()
        for k in range(N_SEEDS):
            e[hosts[i][k]] = L["sub"][i, k].astype(np.float32)
        return e
    mism = {"p2_logits": 0, "p2_labels": 0}
    for r, i in enumerate(mp):
        mism["p2_logits"] += int(not np.array_equal(p2["z6"][r], B["logits"][i].astype(np.float32)))
        mism["p2_labels"] += int(not np.array_equal(p2["lab"][r], expect_lab(i)))
    res["compared"]["p2_sets"] = len(mp)
    for k, c in enumerate(cands):
        ck = f"s{k}"
        mi = rowmap(c)
        mism[f"{ck}_logits"] = mism[f"{ck}_labels"] = mism[f"{ck}_pick_rule"] = 0
        pr = P.rule_pick(c["z6"])
        for r, i in enumerate(mi):
            e = B["logits"][i].astype(np.float32).copy()
            mv = bool(S["moved"][i, k])
            if mv:
                e[pk[i]] = S["pred_logit_pick"][i].astype(np.float32)
                e[hosts[i][k]] = S["pred_logit_out"][i, k].astype(np.float32)
            mism[f"{ck}_logits"] += int(not np.array_equal(c["z6"][r], e))
            mism[f"{ck}_labels"] += int(not np.array_equal(c["lab"][r], expect_lab(i)))
            mism[f"{ck}_pick_rule"] += int(pr[r] != (hosts[i][k] if mv else pk[i]))
        res["compared"][f"{ck}_sets"] = len(mi)
        res.setdefault("n_changed_picks_by_seed", []).append(int((P.rule_pick(c["z6"]) != P.rule_pick(p2["z6"])).sum()))
        res.setdefault("moved_by_seed", []).append(int(S["moved"][:, k].sum()))
        res[f"{ck}_alignment_checks"] = c["checks"]
    res["mismatches"] = mism
    A = gate()
    A_stats = A.make_stats(p2["log_idx"], p2["n_logs"])
    cid = A.control_id(p2["z6"], p2["lab"], A_stats, A.a1_verdict_r1)                  # P2 as the candidate: reads P2's bank only (no search output is a candidate here)
    res["gate_C-ID_P2_as_candidate"] = {"pass": bool(cid["pass"]), "n_changed": cid["n_changed"], "n_intervals_checked": cid["n_intervals_checked"]}
    ok &= bool(cid["pass"])
    res["labels_at_search_output_equal_labeller_bit_for_bit_after_float32"] = bool(mism["p2_labels"] == 0)
    res["float64_to_float32_rounding_max_abs"] = {"pred_logit_out": float(np.abs(S["pred_logit_out"] - S["pred_logit_out"].astype(np.float32).astype(np.float64)).max()),
                                                  "labels_sub": float(np.abs(L["sub"] - L["sub"].astype(np.float32).astype(np.float64)).max())}
    res["changed_equals_moved"] = bool(res["n_changed_picks_by_seed"] == res["moved_by_seed"])
    ok &= all(v == 0 for v in mism.values()) and res["changed_equals_moved"]
    res["pass"] = bool(ok)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bundle", required=True); b.add_argument("--search", required=True); b.add_argument("--labels", required=True); b.add_argument("--out-dir", required=True)
    b.add_argument("--expect-config-sha", default=PIN_CONFIG_SHA); b.add_argument("--expect-ckpt-sha", default=PIN_CKPT_SHA); b.add_argument("--expect-token-sha", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--out-dir", required=True); v.add_argument("--bundle", required=True); v.add_argument("--search", required=True); v.add_argument("--labels", required=True); v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        rep = build(a.bundle, a.search, a.labels, a.out_dir, {"config_sha256": a.expect_config_sha, "ckpt_sha256": a.expect_ckpt_sha, "token_sha256": a.expect_token_sha})
        print(json.dumps({k: rep[k] for k in ("n_tokens", "n_logs", "moved_by_seed", "banks", "ranked_logs_sha256")}, indent=1))
        print("The amendment must NAME the four manifest_sha256 values above BEFORE `ep_admission.py run` reads these banks.")
        return 0
    src = read_sources(a.bundle, a.search, a.labels)
    res = identity_check(a.out_dir, src)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, sort_keys=True))
    print("IDENTITY pass=%s mismatches=%s" % (res["pass"], res["mismatches"]))
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
