#!/usr/bin/env python3
"""verify_paper_numbers.py -- two-way check of the inserted text of Paper v1.5 against claims_values.json.

  (1) every claim in the 'REQUIRED' set must appear, formatted exactly as claims_values.json prints it, in the inserted text;
  (2) every numeric token in the inserted text must be (a) a component of some claim's printed string, or (b) on the
      explicit WHITELIST of non-measurement numbers (section numbers, dates, thresholds quoted from registered rules ...).
Anything left over is printed as an ORPHAN for a human to resolve. Exit code 1 if anything is missing or orphaned.

Usage: python -I verify_paper_numbers.py [inserted_text.txt] [claims_values.json]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TXT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "inserted_text.txt")
CLM = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "claims_values.json")
sys.stdout.reconfigure(encoding="utf-8")

text = open(TXT, encoding="utf-8").read()
norm = re.sub(r"\s+", " ", text)
claims = json.load(open(CLM, encoding="utf-8"))
by_id = {c["id"]: c for c in claims}

# claims the text is built on (every one must be printed verbatim in the text)
REQUIRED = """a01 a05 a06 a07 a08 a09 a10 a11 a12 a14 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a30 a31 a34 a35
b01 b02 b03 b04 b05 b06 b07 b08 b08a b08b b08c b10 b11 b11a b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31
b33 b34 b35 b36 b39 b40 b41 b42 b43 b49 b50 b51 b52
b_C0A_NC b_C0A_TTC b_M25_NC b_M25_TTC
cB cR dB dR eB eR fB fR gB gR hB hR iB iR jB jR kB kR
t_lat_acc_B t_lat_acc_R t_lat_kap_B t_lat_kap_R t_lon_acc_B t_lon_acc_R t_lon_kap_B t_lon_kap_R t_man_acc_B t_man_acc_R t_man_kap_B t_man_kap_R
r_u250_ttc r_u250_nc r_u250_pdms r_u500_ttc r_u500_nc r_u500_pdms r_u750_ttc r_u750_nc r_u750_pdms r_u1000_ttc r_u1000_nc r_u1000_pdms
r_u1000_ep r_u1000_ddc r_gttc r_gnc r_lamt r_lamn r_pr1 r_pr2 r_u1000_loss r_lc_n r_lc_base r_lc_prec r_lc_rec
s_Ltr s_Lc s_Lva s_Gtv s_rz_g100 s_rz_p1loss s_ph_g s_ph_p s_pl_top s_pl_rest s_pl_pool
w_u250_p2 w_u250_p1 w_u250_G w_u500_p2 w_u500_p1 w_u500_G w_u1000_p2 w_u1000_p1 w_u1000_G
v_0.1_u250 v_0.1_u500 v_0.1_u1000 v_0.05_u250 v_0.05_u500 v_0.05_u1000 v_0.02_u250 v_0.02_u500 v_0.02_u1000 v_0.01_u250 v_0.01_u500 v_0.01_u1000
v_c3_u250 v_c3_u500 v_c3_u1000 v_c3_n
d01 d02 d04 d05 d19 d20 d22 d23 d28 d30 d31 d32 d33 d34 d35 d36
p01 p02 p03 p04 p05 p06 p07 p08 p09
r_g3_min r_g3_max r_sb_min r_sb_max r_train_sets s_ntr_logs s_ntr_sets s_nva_logs s_nva_sets s_pl_rest_n a32 a33 d25 d26""".split()

missing = []
for cid in REQUIRED:
    c = by_id.get(cid)
    if c is None:
        missing.append((cid, "NO SUCH CLAIM"))
        continue
    pr = c["printed"]
    if pr not in norm:
        missing.append((cid, pr))

# components: every numeric token of every claim's printed string
NUM = re.compile(r"[−+]?\d[\d,]*(?:\.\d+)?(?:e[−+]?\d+)?")
SKIP_CTX = [re.compile(p) for p in (r"\d{4}\.\d{5}", r"v1\.1", r"sha256", r"2026-\d\d-\d\d", r"\bA40\b", r"\bu400\b", r"\bP7\.1\b", r"LAB-RUN-019", r"line 24")]


def toks(s):
    out = set()
    for m in NUM.finditer(s):
        t = m.group(0).replace(",", "")
        out.add(t)
        out.add(t.lstrip("+−"))
    return out


known = set()
for c in claims:
    known |= toks(c["printed"])
    if isinstance(c["value"], (int, float)) and not isinstance(c["value"], bool):
        known |= toks(f"{c['value']:g}")

# numbers that are not measurements: section / table / item labels, dates, registered thresholds and design constants
WL = set("""7.20 7.21 7.19 5.8 3.12 13.1 1.5 1.4 1.4.1 10 100 2026 2025 16 18 19 20 21 25 03 04 09 10
0.5 0.25 0.1 0.05 0.02 0.01 4.0 4 3 2 1 0 64 12146 136 3000 75 2000 10000 20260927 0.001 0.55 0.894 0.97 94.55
-6 -10 6 10 10 4 2 8 0.50 0.2 1000 250 500 750 651 226 550 576
1.7 1.9 62 38 3 5 1 0 11 1 4 6 0 2 3 15 19 20 21 25 1000 250 500 750 3 four three 95 0.8 0.9""".split())
WL |= {"0.5", "0.25", "0.1", "0.05", "0.02", "0.01"}

# numbers CARRIED from text already in the paper (v1.4 section 7.20) or quoted from the register; stated as such in SOURCES.md
CARRIED = {"0.33", "0.04", "0.64", "88.25", "19", "4.5", "1.4"}
orph = []
for m in NUM.finditer(text):
    span = text[max(0, m.start() - 12): m.end() + 12]
    if any(p.search(span) for p in SKIP_CTX):
        continue
    t = m.group(0).replace(",", "")
    t0 = t.lstrip("+−")
    if t in known or t0 in known or t0 in WL or t in WL or t0 in CARRIED:
        continue
    ctx = text[max(0, m.start() - 40): m.end() + 30].replace("\n", " ")
    orph.append((t, ctx))

print(f"required claims: {len(REQUIRED)}; missing: {len(missing)}; orphan numeric tokens: {len(orph)}")
for cid, pr in missing:
    print("  MISSING", cid, "->", pr)
seen = set()
for t, ctx in orph:
    if (t, ctx) in seen:
        continue
    seen.add((t, ctx))
    print("  ORPHAN", t, "|", ctx)
sys.exit(1 if (missing or orph) else 0)
