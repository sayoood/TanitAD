"""K2DC dose-controller DIAGNOSTIC (post-hoc, MM 4a3d05 2026-11-15 23:45Z; NOT the pre-registered rule, which is reported beside it unchanged).
Reads metrics.jsonl rows (stdin) carrying dc_share_read / dc_w_before / dc_w_eff and reports:
  RULE  the pre-registered validity rule (PREREG_K2DC 'In-training check'): share inside [0.05, 0.15] at >= 2 firings at steps >= 3,000;
  DIAG  the geometric-mean share of the readings AFTER the first correction (log-mean, firing-bootstrap 95 % CI), and how discriminating RULE
        is: P(RULE passes) for a lognormal reading with the MEASURED log-sd, centred on the measured log-mean and on 2x / 4x the target.
usage: ssh thor 'grep dc_share_read <metrics.jsonl>' | python -I dose_diag.py [n_future_firings]"""
import json
import math
import random
import sys

rows = [json.loads(l) for l in sys.stdin if l.strip()]
rows = [r for r in rows if "dc_share_read" in r]
n_future = int(sys.argv[1]) if len(sys.argv) > 1 else 13          # firings at 3000..6000 every 250
LO, HI, TGT = 0.05, 0.15, 0.10
print("firings", len(rows), "steps", [r["step"] for r in rows])
late = [r for r in rows if r["step"] >= 3000]
inb = [r["step"] for r in late if LO <= r["dc_share_read"] <= HI]
print("RULE (pre-registered): firings >= 3000:", len(late), "in band:", len(inb), inb, "->", "VALID so far" if len(inb) >= 2 else "not yet")
post_all = [r["dc_share_read"] for r in rows[1:]]                   # readings after the controller's first correction (step 250)
post = [s for s in post_all if s > 0]                              # a share <= 0 (map gradient opposing the rest) has no log
print("DIAG: non-positive readings excluded from the log statistics: %d of %d" % (len(post_all) - len(post), len(post_all)))
lg = [math.log(s) for s in post]
m = sum(lg) / len(lg)
sd = (sum((x - m) ** 2 for x in lg) / (len(lg) - 1)) ** 0.5
rng = random.Random(0)
boots = sorted(sum(rng.choice(lg) for _ in lg) / len(lg) for _ in range(20000))
lo_ci, hi_ci = boots[500], boots[19499]
print("DIAG: n %d readings after the first correction; geometric-mean share %.4f, firing-bootstrap 95%% CI [%.4f, %.4f]; log-sd %.3f (x%.2f)"
      % (len(lg), math.exp(m), math.exp(lo_ci), math.exp(hi_ci), sd, math.exp(sd)))
print("DIAG: in band overall %d / %d" % (sum(LO <= s <= HI for s in post), len(post)))


def p_band(centre):
    phi = lambda z: 0.5 * (1 + math.erf(z / 2 ** 0.5))
    return phi((math.log(HI) - math.log(centre)) / sd) - phi((math.log(LO) - math.log(centre)) / sd)


def p_rule(p, n):   # P(>= 2 of n independent firings in band)
    return 1 - (1 - p) ** n - n * p * (1 - p) ** (n - 1)


for lab, c in (("measured centre", math.exp(m)), ("centre = target 0.10", TGT), ("centre 2x target 0.20", 0.2), ("centre 4x target 0.40", 0.4),
               ("centre 8x target 0.80", 0.8)):
    p = p_band(c)
    print("DIAG (ESTIMATED, lognormal, measured log-sd): %-22s P(one reading in band) %.3f -> P(RULE passes, %d firings) %.3f" % (lab, p, n_future, p_rule(p, n_future)))
print("w trajectory:", [(r["step"], round(r["dc_w_before"], 2), round(r["dc_w_eff"], 2)) for r in rows])
