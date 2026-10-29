#!/usr/bin/env python3
"""soup_build.py -- PREREG_V8F_SOUP sections 2 and 6: a same-init uniform weight average ("soup") of v8F checkpoints.
WEIGHTS ONLY: this file never rolls out, never evaluates and never reads an eval artifact.

  build   --spec soups.json --name SOUP_s0 --out-root DIR     K-BASIN, average, K-MID, write DIR/<run_dir>/ckpt_last.pt + soup.json, re-verify by content
  basin   --spec soups.json --out basin.json                  K-BASIN on every soup of the spec (XINIT must FAIL; each clause load-bearing)
  verify  --spec soups.json --name SOUP_s0 --out-root DIR     K-SOUP-REC + K-MID re-check of a written soup, from the files
  kzero   --ckpt X [--out kzero.json]                         K-ZERO-W: a soup of X with itself (N = 2, 3, 4) equals X bit for bit
  loadcheck --ckpt X --tree TREE                              the T1 tool's own loader path (V8FMiniConfig(**cfg), V8FMini(...).load_state_dict(model)), CPU, no rollout

Exit codes: 0 PASS, 2 REFUSED (a control failed; nothing written), 3 usage / input error.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

PREREG = "PREREG_V8F_SOUP (DRAFT, 2026-10-28)"
CFG_ALLOW = frozenset({"launch_intent_w"})               # training-only cfg keys that may differ between members (section 2)
MUST_EQUAL = ("cfg_tuple_fields", "grid", "arm", "p10_path", "cons_mu", "cons_sd")    # non-weight entries the members must share (cfg is checked separately)
MEMBER_META = ("step", "epoch", "batch_in_epoch", "argv_sha", "seed", "init_ckpt", "ego_perturb", "ego_totals", "dagger", "dagger_totals", "launch_totals")
B1_FACTOR = 2.0                                          # r_i <= r'_i / B1_FACTOR
B2_FACTOR = 3.0                                          # max d_ij <= B2_FACTOR * R_ft(s)
CLAUSES = ("B1", "B2")
MID_REL_TOL = 1e-6


class SoupRefusal(Exception):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ckpt(path, want_sha=None):
    """Hash, load, hash again: the bytes torch read are the bytes that were pinned."""
    h0 = sha256_file(path)
    if want_sha is not None and h0 != want_sha:
        raise SoupRefusal(f"{path}: sha256 {h0} != pinned {want_sha}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h1 = sha256_file(path)
    if h1 != h0:
        raise SoupRefusal(f"{path}: changed while it was loaded ({h0} -> {h1})")
    return ck, h0


# ---------------------------------------------------------------- pure layer (state dicts in, numbers out)
def check_compatible(sds):
    """Identical key order, shapes and dtypes; non-float tensors must be equal.  -> list of non-float keys (kept from member 0)."""
    if len(sds) < 2:
        raise SoupRefusal("a soup needs at least two state dicts")
    keys = list(sds[0].keys())
    nonfloat = []
    for i, sd in enumerate(sds[1:], 1):
        if list(sd.keys()) != keys:
            raise SoupRefusal(f"member {i}: state-dict keys differ from member 0 (set or order)")
    for k in keys:
        t0 = sds[0][k]
        for i, sd in enumerate(sds[1:], 1):
            t = sd[k]
            if tuple(t.shape) != tuple(t0.shape):
                raise SoupRefusal(f"{k}: shape {tuple(t.shape)} (member {i}) != {tuple(t0.shape)}")
            if t.dtype != t0.dtype:
                raise SoupRefusal(f"{k}: dtype {t.dtype} (member {i}) != {t0.dtype}")
        if not torch.is_floating_point(t0):
            for i, sd in enumerate(sds[1:], 1):
                if not torch.equal(sd[k], t0):
                    raise SoupRefusal(f"{k}: non-float tensor differs between member 0 and member {i}; a soup cannot average it")
            nonfloat.append(k)
        else:
            for i, sd in enumerate(sds[1:], 1):
                nonfinite_problem(k, t0, sd[k], f"member 0 / member {i}")
    return nonfloat


def nonfinite_problem(k, ta, tb, who):
    """Amendment A0: a non-finite element (the +-inf bound buffers wm.stack.prior_lo_k / prior_hi_k) must sit at the same position with the
    same bits in both tensors; it is then copied unchanged and contributes exactly 0 to every distance.  Anything else is REFUSED."""
    fa, fb = torch.isfinite(ta), torch.isfinite(tb)
    if not torch.equal(fa, fb):
        raise SoupRefusal(f"{k}: non-finite pattern differs ({who})")
    if not bool(fa.all()) and not torch.equal(ta[~fa], tb[~fb]):
        raise SoupRefusal(f"{k}: non-finite values differ ({who})")
    return fa


def average_state(sds):
    """Uniform mean of every float tensor, accumulated in float64 and rounded ONCE to the member dtype; non-float tensors copied from member 0 (checked equal)."""
    check_compatible(sds)
    n = len(sds)
    out = collections.OrderedDict()
    for k, t0 in sds[0].items():
        if torch.is_floating_point(t0):
            acc = torch.zeros(t0.shape, dtype=torch.float64)
            for sd in sds:
                acc += sd[k].to(torch.float64)
            out[k] = (acc / n).to(t0.dtype)
        else:
            out[k] = t0.clone()
    return out


def _canon(v):
    if isinstance(v, np.ndarray):
        return ("ndarray", str(v.dtype), tuple(v.shape), v.tobytes().hex())
    if isinstance(v, torch.Tensor):
        return ("tensor", str(v.dtype), tuple(v.shape), v.numpy().tobytes().hex())
    return json.dumps(v, sort_keys=True, default=str)


def cfg_problems(cfgs, allow=CFG_ALLOW):
    keys = sorted(set().union(*[set(c) for c in cfgs]))
    bad = []
    for k in keys:
        if k in allow:
            continue
        vals = [_canon(c.get(k, "<ABSENT>")) for c in cfgs]
        if len(set(vals)) > 1:
            bad.append(k)
    return bad


def check_meta(cks, allow=CFG_ALLOW, must=MUST_EQUAL):
    bad = cfg_problems([c["cfg"] for c in cks], allow)
    if bad:
        raise SoupRefusal(f"cfg differs between members on {bad} (only {sorted(allow)} may differ)")
    for k in must:
        vals = [_canon(c.get(k, "<ABSENT>")) for c in cks]
        if len(set(vals)) > 1:
            raise SoupRefusal(f"non-weight entry {k!r} differs between members")


def sqdist(sa, sb):
    """Squared L2 distance over the float tensors, float64."""
    s = 0.0
    for k, ta in sa.items():
        if torch.is_floating_point(ta):
            fin = nonfinite_problem(k, ta, sb[k], "distance")
            s += float(((ta.to(torch.float64)[fin] - sb[k].to(torch.float64)[fin]) ** 2).sum())
    if not np.isfinite(s):
        raise SoupRefusal(f"distance is not finite ({s})")
    return s


def dist(sa, sb):
    return float(np.sqrt(sqdist(sa, sb)))


def basin(members, init, other_init, refs, b1=B1_FACTOR, b2=B2_FACTOR, clauses=CLAUSES):
    """K-BASIN (section 6).  members / refs: {name: state dict}; init: the declared shared init; other_init: the other seed's init.
    B1: every member r_i <= r'_i / b1.  B2: max_{i<j} d_ij <= b2 * R_ft, R_ft = median r over the reference fine-tunes."""
    r = {n: dist(sd, init) for n, sd in members.items()}
    rp = {n: dist(sd, other_init) for n, sd in members.items()}
    r_ref = {n: dist(sd, init) for n, sd in refs.items()}
    r_ft = float(np.median(list(r_ref.values())))
    names = list(members)
    d = {f"{a}|{b}": dist(members[a], members[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    dmax = max(d.values()) if d else 0.0
    b1_rows = {n: bool(r[n] <= rp[n] / b1) for n in names}
    b1_ok = all(b1_rows.values())
    b2_ok = bool(dmax <= b2 * r_ft)
    ok = (b1_ok or "B1" not in clauses) and (b2_ok or "B2" not in clauses)
    return {"PASS": bool(ok), "B1": b1_ok, "B2": b2_ok, "clauses": list(clauses), "B1_factor": b1, "B2_factor": b2,
            "r_to_declared_init": r, "r_to_other_init": rp, "B1_rows": b1_rows, "R_ft": r_ft, "R_ft_members": r_ref,
            "d_members": d, "d_max": dmax, "spread_ratio": (dmax / r_ft if r_ft > 0 else float("inf"))}


def k_mid(soup_sd, sds):
    """K-MID: the written tensors equal an independent numpy float64 mean rounded to the dtype (max |delta| 0.0); every element lies within
    its members' [min, max]; for N = 2, d(soup, a) and d(soup, b) equal d(a, b) / 2 within MID_REL_TOL."""
    worst, outside = 0.0, 0
    for k, t in soup_sd.items():
        if not torch.is_floating_point(t):
            if not torch.equal(t, sds[0][k]):
                return {"PASS": False, "why": f"non-float {k} changed"}
            continue
        arr = np.stack([sd[k].numpy().astype(np.float64) for sd in sds])
        got = t.numpy()
        fin = np.isfinite(arr).all(axis=0)
        if not np.array_equal(np.isfinite(got), fin) or not all(np.array_equal(got[~fin], sd[k].numpy()[~fin]) for sd in sds):
            return {"PASS": False, "why": f"{k}: a non-finite element is not the members' (bit for bit)"}
        ref = arr[:, fin].mean(axis=0).astype(got.dtype)
        g = got[fin].astype(np.float64)
        if g.size:
            dev = np.abs(g - ref.astype(np.float64))
            if not np.isfinite(dev).all():
                return {"PASS": False, "why": f"{k}: non-finite deviation"}
            worst = max(worst, float(dev.max()))
        outside += int(np.sum((g < arr[:, fin].min(axis=0)) | (g > arr[:, fin].max(axis=0))))
    out = {"max_abs_vs_numpy_mean": worst, "elements_outside_member_range": outside}
    ok = worst == 0.0 and outside == 0
    if len(sds) == 2:
        dab = dist(sds[0], sds[1])
        da, db = dist(soup_sd, sds[0]), dist(soup_sd, sds[1])
        out.update({"d_ab": dab, "d_soup_a": da, "d_soup_b": db})
        ok = ok and abs(da - dab / 2) <= MID_REL_TOL * dab and abs(db - dab / 2) <= MID_REL_TOL * dab
    out["PASS"] = bool(ok)
    return out


def make_soup_ckpt(cks, donor, soup_sd, record):
    """The donor's checkpoint with `model` replaced by the soup and `opt` set to None; each member's own run entries recorded."""
    out = dict(cks[donor])
    out["model"] = soup_sd
    out["opt"] = None
    out["soup"] = dict(record)
    out["soup"]["members_meta"] = [{k: c.get(k) for k in MEMBER_META if k in c} for c in cks]
    out["soup"]["donor_index"] = donor
    return out


def k_zero(ck, ns=(2, 3, 4)):
    sd = ck["model"]
    res = {}
    for n in ns:
        s = average_state([sd] * n)
        same = all(torch.equal(s[k], sd[k]) and s[k].dtype == sd[k].dtype for k in sd)
        res[str(n)] = bool(same and list(s.keys()) == list(sd.keys()))
    return {"PASS": all(res.values()), "per_n": res}


# ---------------------------------------------------------------- file layer
def read_spec(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def soup_paths(spec, name, out_root):
    s = spec["soups"][name]
    d = os.path.join(out_root, s["run_dir"]).replace("\\", "/")
    return d, f"{d}/ckpt_last.pt", f"{d}/soup.json"


def load_members(spec, names):
    cks, shas = {}, {}
    for n in names:
        m = spec["members"][n]
        cks[n], shas[n] = load_ckpt(m["path"], m["sha256"])
    return cks, shas


def basin_for(spec, name, cache=None, clauses=CLAUSES):
    s = spec["soups"][name]
    seed = str(s["seed"])
    other = str(1 - int(seed))
    need = set(s["members"]) | set(spec["reference"][seed])
    cache = {} if cache is None else cache
    for n in need:
        if n not in cache:
            cache[n] = load_members(spec, [n])[0][n]["model"]
    for k in (f"INIT_{seed}", f"INIT_{other}"):
        if k not in cache:
            i = spec["inits"][k[5:]]
            cache[k] = load_ckpt(i["path"], i["sha256"])[0]["model"]
    return basin({n: cache[n] for n in s["members"]}, cache[f"INIT_{seed}"], cache[f"INIT_{other}"],
                 {n: cache[n] for n in spec["reference"][seed]}, clauses=clauses)


def builder_sha():
    return sha256_file(os.path.abspath(__file__))


def build(spec, name, out_root, log=print):
    s = spec["soups"][name]
    if s.get("expect") == "K-BASIN FAIL":
        raise SoupRefusal(f"{name} is a deliberate-regression soup: it is checked by `basin`, never written")
    d, ck_path, rec_path = soup_paths(spec, name, out_root)
    if os.path.exists(ck_path) or os.path.exists(rec_path):
        raise SoupRefusal(f"{d} already holds a soup: refusing to overwrite (move it aside first)")
    t0 = time.time()
    cks, shas = load_members(spec, s["members"])
    order = list(s["members"])
    donor = order.index(s["donor"])
    lst = [cks[n] for n in order]
    check_meta(lst)
    kb = basin_for(spec, name, cache={n: cks[n]["model"] for n in order})
    if not kb["PASS"]:
        raise SoupRefusal(f"{name}: K-BASIN FAILED (B1 {kb['B1']}, B2 {kb['B2']}, spread ratio {kb['spread_ratio']:.3f}): not a same-init soup")
    sds = [c["model"] for c in lst]
    soup_sd = average_state(sds)
    km = k_mid(soup_sd, sds)
    if not km["PASS"]:
        raise SoupRefusal(f"{name}: K-MID FAILED {km}")
    rec = {"kind": "v8f-soup", "prereg": PREREG, "name": name, "seed": s["seed"], "members": order,
           "member_sha256": {n: shas[n] for n in order}, "donor": s["donor"], "weights": [1.0 / len(order)] * len(order),
           "builder_sha256": builder_sha(), "K-BASIN": kb, "K-MID": km, "opt": "None: an optimizer state belongs to one training run; a soup has none",
           "evidence_class": "MEASURED (weights only; no rollout, no eval)"}
    out = make_soup_ckpt(lst, donor, soup_sd, rec)
    os.makedirs(d, exist_ok=True)
    tmp = ck_path + ".partial"
    torch.save(out, tmp)
    os.replace(tmp, ck_path)
    rec["output_sha256"] = sha256_file(ck_path)
    rec["output_bytes"] = os.path.getsize(ck_path)
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(rec_path + ".partial", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(rec_path + ".partial", rec_path)
    v = verify(spec, name, out_root)
    if not v["PASS"]:
        raise SoupRefusal(f"{name}: written but K-SOUP-REC FAILED on re-read: {v}")
    log(f"{name}: PASS  out {ck_path}  sha256 {rec['output_sha256']}  spread {kb['spread_ratio']:.4f}  wall {rec['wall_s']} s")
    return rec


def verify(spec, name, out_root):
    """K-SOUP-REC + K-MID from the files: output hash, member pins, donor, cfg byte-equal to the donor's, opt None, builder hash, the average."""
    s = spec["soups"][name]
    d, ck_path, rec_path = soup_paths(spec, name, out_root)
    probs = []
    with open(rec_path, encoding="utf-8") as f:
        rec = json.load(f)
    if sha256_file(ck_path) != rec.get("output_sha256"):
        probs.append("output sha256 != record")
    if rec.get("members") != list(s["members"]) or rec.get("donor") != s["donor"]:
        probs.append("members / donor != spec")
    for n in s["members"]:
        if rec.get("member_sha256", {}).get(n) != spec["members"][n]["sha256"]:
            probs.append(f"member {n} sha256 != pin")
    if rec.get("builder_sha256") != builder_sha():
        probs.append("builder sha256 != this file")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cks, _ = load_members(spec, s["members"])
    donor = cks[s["donor"]]
    if _canon(ck.get("cfg")) != _canon(donor.get("cfg")):
        probs.append("cfg != donor cfg")
    if ck.get("opt") is not None:
        probs.append("opt is not None")
    for k in donor:
        if k in ("model", "opt"):
            continue
        if _canon(ck.get(k)) != _canon(donor.get(k)):
            probs.append(f"entry {k!r} != donor's")
    km = k_mid(ck["model"], [cks[n]["model"] for n in s["members"]])
    if not km["PASS"]:
        probs.append(f"K-MID {km}")
    return {"PASS": not probs, "problems": probs, "K-MID": km}


def loadcheck(ckpt, tree):
    """The T1 tool's loader (v8f_t1_eval.py:505-511), CPU, nothing rolled out."""
    sys.path.insert(0, os.path.join(tree, "stack"))
    from tanitad.models import v8f_mini as M  # noqa: E402
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    res = model.load_state_dict(ck["model"])
    return {"PASS": not res.missing_keys and not res.unexpected_keys and cfg.kin_mode == "yaw", "kin_mode": cfg.kin_mode,
            "missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys), "model_module": os.path.abspath(M.__file__)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "verify"):
        p = sub.add_parser(c)
        p.add_argument("--spec", required=True)
        p.add_argument("--name", required=True, nargs="+")
        p.add_argument("--out-root", required=True)
    p = sub.add_parser("basin")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("kzero")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out")
    p = sub.add_parser("loadcheck")
    p.add_argument("--ckpt", required=True, nargs="+")
    p.add_argument("--tree", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            spec = read_spec(a.spec)
            for n in a.name:
                build(spec, n, a.out_root)
            return 0
        if a.cmd == "verify":
            spec = read_spec(a.spec)
            ok = True
            for n in a.name:
                v = verify(spec, n, a.out_root)
                print(n, "K-SOUP-REC", "PASS" if v["PASS"] else f"FAIL {v['problems']}", json.dumps(v["K-MID"]))
                ok &= v["PASS"]
            return 0 if ok else 2
        if a.cmd == "basin":
            spec = read_spec(a.spec)
            cache, res, ok = {}, {}, True
            for n, s in spec["soups"].items():
                r = {"full": basin_for(spec, n, cache)}
                if s.get("expect") == "K-BASIN FAIL":
                    r["B2_only"] = basin_for(spec, n, cache, clauses=("B2",))
                    r["B1_only"] = basin_for(spec, n, cache, clauses=("B1",))
                    r["regression_as_required"] = bool(not r["full"]["PASS"] and not r["B2_only"]["PASS"] and not r["B1_only"]["PASS"])
                    ok &= r["regression_as_required"]
                else:
                    ok &= r["full"]["PASS"]
                res[n] = r
                print(n, "PASS" if r["full"]["PASS"] else "FAIL", f"B1 {r['full']['B1']} B2 {r['full']['B2']} spread_ratio {r['full']['spread_ratio']:.4f} R_ft {r['full']['R_ft']:.4f} d_max {r['full']['d_max']:.4f}",
                      ("regression_as_required " + str(r["regression_as_required"])) if "regression_as_required" in r else "")
            out = {"prereg": PREREG, "control": "K-BASIN + K-XINIT (section 6)", "builder_sha256": builder_sha(), "spec_sha256": sha256_file(a.spec),
                   "ALL_AS_REQUIRED": bool(ok), "soups": res, "evidence_class": "MEASURED (weights only)"}
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, sort_keys=True)
            print("K-BASIN / K-XINIT", "ALL AS REQUIRED" if ok else "NOT AS REQUIRED")
            return 0 if ok else 2
        if a.cmd == "kzero":
            ck, h = load_ckpt(a.ckpt)
            r = {"ckpt": a.ckpt, "sha256": h, **k_zero(ck)}
            print("K-ZERO-W", "PASS" if r["PASS"] else "FAIL", json.dumps(r["per_n"]))
            if a.out:
                with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(r, f, indent=1, sort_keys=True)
            return 0 if r["PASS"] else 2
        if a.cmd == "loadcheck":
            ok = True
            for c in a.ckpt:
                r = loadcheck(c, a.tree)
                print(c, "LOADCHECK", "PASS" if r["PASS"] else "FAIL", json.dumps(r))
                ok &= r["PASS"]
            return 0 if ok else 2
    except SoupRefusal as e:
        print(f"REFUSED: {e}")
        return 2
    return 3


if __name__ == "__main__":
    sys.exit(main())
