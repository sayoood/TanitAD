#!/usr/bin/env python3
"""mm_s3_records.py -- MM 2026-11-23. Read-only summary of the SEED3 arm records on the LoRA pod (B0_s2, RH_s2, KI_s2 and the s2 bank).
Prints one JSON object; writes nothing."""
import glob, hashlib, json, os

R = "/root/rollhist"
out = {"arms": {}, "bank": {}}


def j(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def tail(p, n=300):
    try:
        return open(p, encoding="utf-8", errors="replace").read()[-n:]
    except Exception as e:  # noqa: BLE001
        return f"_error {e}"


def find(o, k, depth=0):
    if depth > 6:
        return []
    r = []
    if isinstance(o, dict):
        for kk, v in o.items():
            if kk == k:
                r.append(v)
            r += find(v, k, depth + 1)
    elif isinstance(o, list):
        for v in o[:8]:
            r += find(v, k, depth + 1)
    return r


for arm in ("B0_s2", "RH_s2", "KI_s2"):
    D = f"{R}/runs/{arm}"
    O = f"{D}/train"
    a = {"ARM_DONE": os.path.exists(f"{D}/ARM_DONE")}
    a["mint_tail"] = tail(f"{D}/mint.log", 200)
    a["verify_tail"] = tail(f"{D}/verify.log", 120)
    z = j(f"{D}/zerograd_{arm}.json")
    a["zerograd"] = {k: z.get(k) for k in ("verdict",)} | {"optimizer": z.get("optimizer", {})}
    a["post"] = j(f"{O}/post.json")
    a["summary_check"] = j(f"{O}/summary_check.json")
    a["lever_acts"] = j(f"{D}/lever_acts_{arm}.json") if os.path.exists(f"{D}/lever_acts_{arm}.json") else None
    sc = j(f"{D}/selfcheck/selfcheck.json")
    a["selfcheck_dagger_base"] = find(sc, "dagger_base")[:2]
    a["selfcheck_dagger_p"] = find(sc, "dagger_p")[:2]
    a["selfcheck_seed"] = find(sc, "seed")[:2]
    a["selfcheck_pass"] = {k: (find(sc, k) or [None])[0] for k in ("pass",)}
    a["selfcheck_result"] = j(f"{D}/selfcheck.result.json")
    a["argv_sha256"] = hashlib.sha256(open(f"{D}/argv.json", "rb").read()).hexdigest() if os.path.exists(f"{D}/argv.json") else None
    ck = f"{O}/ckpt_last.pt"
    a["ckpt_sha256"] = hashlib.sha256(open(ck, "rb").read()).hexdigest() if os.path.exists(ck) else None
    out["arms"][arm] = a

B = f"{R}/bank/s2"
out["bank"]["files"] = sorted(os.path.basename(p) for p in glob.glob(f"{B}/*"))[:20]
for name in ("BANK_CHECK.json", "bank.json", "GEN_RECORD.json", "generator.json"):
    p = f"{B}/{name}"
    if os.path.exists(p):
        out["bank"][name] = j(p)
for p in glob.glob(f"{R}/runs/*s2*/bank_check*.json") + glob.glob(f"{R}/kit_s3/*bank_check*.json") + glob.glob(f"{R}/bank_s3*/*.json"):
    out["bank"][p] = j(p)
print(json.dumps(out, indent=1, default=str)[:60000])
