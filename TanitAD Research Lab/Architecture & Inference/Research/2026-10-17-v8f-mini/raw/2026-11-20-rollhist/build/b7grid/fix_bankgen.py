"""MM 2026-11-21 (ROLLHIST Amendment 5): C-RH-BANKGEN read bank.json's TOP-LEVEL `generator`; the merge writes the record only in `common`, which is
what the trainer reads (DaggerBank header = {**bank.json["common"], ...}, stack/tanitad/data/v8f_dagger.py:366). Patch the study check to read common.generator,
then run the arms on the REAL s0 bank and on mutated copies. Keeps study/dscale10k_check_rh.py.bgen. Temp copies under /root/rollhist/tmp_bankgen are deleted at the end (logged)."""
import hashlib, importlib.util, json, os, shutil, sys

R = "/root/rollhist"
F = f"{R}/study/dscale10k_check_rh.py"
OLD_SHA = "94e6e1473aa5d3e0867b816a9b1d602ecacf3785d6d5e2345c63e53374e6fc8f"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
mode = sys.argv[1]          # patch | arms

if mode == "patch":
    assert sha(F) == OLD_SHA, ("study check sha", sha(F))
    assert not os.path.exists(F + ".bgen"), "a .bgen copy exists already"
    src = open(F, "rb").read().decode("utf-8")
    nl = "\r\n" if "\r\n" in src else "\n"
    a1 = '    tk, kp = built(b.get("generator"))'
    b1 = ('    gen_b = (b.get("common") or {}).get("generator")         # ROLLHIST Amendment 5: the record the trainer reads (DaggerBank header = bank.json["common"], v8f_dagger.py:366)'
          + nl + '    tk, kp = built(gen_b)')
    a2 = 'if os.path.isfile(sp) and json.load(open(sp, encoding="utf-8")).get("generator") != b.get("generator"):'
    b2 = 'if os.path.isfile(sp) and json.load(open(sp, encoding="utf-8")).get("generator") != gen_b:'
    a3 = '            bad.append(f"shard {sh.get(\'k\')}: its provenance generator record differs from bank.json\'s")' + nl + '    return bad'
    b3 = ('            bad.append(f"shard {sh.get(\'k\')}: its provenance generator record differs from bank.json\'s")' + nl
          + '    if "generator" in b and b["generator"] != gen_b:' + nl
          + '        bad.append("bank.json carries a top-level generator record that differs from common.generator (the trainer reads only common)")' + nl
          + '    return bad')
    for a in (a1, a2, a3):
        assert src.count(a) == 1, ("anchor count", a[:60], src.count(a))
    shutil.copy2(F, F + ".bgen")
    new = src.replace(a1, b1).replace(a2, b2).replace(a3, b3)
    open(F, "wb").write(new.encode("utf-8"))
    print("patched", F, "old", OLD_SHA[:12], "new", sha(F)[:12], "full", sha(F))
    sys.exit(0)

# ---- arms
def load(path, name):
    import importlib.machinery
    s = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, path))     # a .bgen suffix has no default loader
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m

sys.path[:0] = [f"{R}/study", f"{R}/tree_cdb/stack"]
NEW = load(F, "chk_new")
OLD = load(F + ".bgen", "chk_old")
argv = json.load(open(f"{R}/runs/RH_s0/argv.json"))
toks = argv.get("argv", argv)
DSHA = toks[toks.index("--dagger-bank-sha256") + 1]
BANK = f"{R}/bank/s0"
TOK, KAP = "causal_g1x", NEW.RH_KAPPA
T = f"{R}/tmp_bankgen"
assert not os.path.exists(T)
os.makedirs(T)

def copy_bank(tag, edit_bank=None, edit_shard=None):
    d = f"{T}/{tag}"
    os.makedirs(d)
    for f in os.listdir(BANK):
        if f == "bank.json" or f.endswith(".provenance.json"):
            shutil.copy2(f"{BANK}/{f}", f"{d}/{f}")
    if edit_bank:
        b = json.load(open(f"{d}/bank.json")); edit_bank(b); json.dump(b, open(f"{d}/bank.json", "w"))
    if edit_shard:
        p = f"{d}/shard_001of004.provenance.json"; s = json.load(open(p)); edit_shard(s); json.dump(s, open(p, "w"))
    return d, sha(f"{d}/bank.json")

rows = {}
def arm(name, fn, d, s, want_pass, must_contain=None):
    p = fn.rh_bank_generator_problems(d, s, TOK, KAP)
    ok = (not p) if want_pass else (bool(p) and (must_contain is None or any(must_contain in x for x in p)))
    rows[name] = {"problems": p, "want": "PASS" if want_pass else "REFUSE", "as_expected": ok}
    print(("OK  " if ok else "BAD ") + name, p)

arm("K0_new_real_bank", NEW, BANK, DSHA, True)
arm("K0_old_real_bank_reproduces_refusal", OLD, BANK, DSHA, False, "reads readout 'pinned'")
d, s = copy_bank("R1_no_common_gen", edit_bank=lambda b: b["common"].pop("generator"))
arm("R1_common_generator_absent", NEW, d, s, False, "reads readout 'pinned'")
d, s = copy_bank("R2_kappa", edit_bank=lambda b: b["common"]["generator"].__setitem__("kappa_max", "none"))
arm("R2_common_kappa_none", NEW, d, s, False, "reads kappa 'none'")
d, s = copy_bank("R3_readout", edit_bank=lambda b: b["common"]["generator"].__setitem__("readout", "pinned"))
arm("R3_common_readout_pinned", NEW, d, s, False, "reads readout")
d, s = copy_bank("R4_shard", edit_shard=lambda x: x["generator"].__setitem__("acc_gain", 0.5))
arm("R4_one_shard_differs", NEW, d, s, False, "shard 1")
d, s = copy_bank("R5_toplevel_only", edit_bank=lambda b: b.__setitem__("generator", b["common"].pop("generator")))
arm("R5_toplevel_only_trainer_reads_pinned", NEW, d, s, False, "reads readout 'pinned'")
d, s = copy_bank("R6_toplevel_disagrees", edit_bank=lambda b: b.__setitem__("generator", {"readout": "pinned"}))
arm("R6_toplevel_disagrees_with_common", NEW, d, s, False, "top-level generator")
arm("R7_wrong_declared_sha", NEW, BANK, "0" * 64, False, "sha256")
d, s = copy_bank("R8_toplevel_equal", edit_bank=lambda b: b.__setitem__("generator", dict(b["common"]["generator"])))
arm("R8_toplevel_equal_copy_passes", NEW, d, s, True)

# the trainer's own read of the REAL bank header (the reference the check must match)
sys.path.insert(0, f"{R}/tree_cdb/stack")
from tanitad.data import v8f_dagger as DG
man = json.load(open(f"{BANK}/bank.json"))
hdr_gen = {**man["common"]}.get("generator")
rows["trainer_header_generator"] = hdr_gen
all_ok = all(v["as_expected"] for k, v in rows.items() if isinstance(v, dict) and "as_expected" in v)
out = {"kind": "rollhist-amendment5-bankgen-arms", "evidence": "MEASURED (pod)", "check_file_sha256": sha(F), "old_check_file_sha256": sha(F + ".bgen"),
       "bank_json_sha256": sha(f"{BANK}/bank.json"), "declared_sha256": DSHA, "token": TOK, "kappa": KAP, "rows": rows, "ALL_AS_EXPECTED": all_ok}
json.dump(out, open(f"{R}/logs/bankgen_arms.json", "w"), indent=1)
kb = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(T) for f in fs) // 1024
shutil.rmtree(T)
open(f"{R}/deletions.log", "a").write(f"{__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())} {T} {kb}KiB (own temp: mutated bank.json/provenance copies of the Amendment 5 arms)\n")
print("ALL_AS_EXPECTED", all_ok)
