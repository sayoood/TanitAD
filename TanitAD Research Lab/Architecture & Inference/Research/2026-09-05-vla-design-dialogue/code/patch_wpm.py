"""Apply the outcome-screen patch to wpm_port_influence.py.

⛔ WHY THIS IS A FILE AND NOT A HEREDOC. The first attempt was
`python - <<'PY' ... PY` inside a compound command launched with
run_in_background. The heredoc's stdin never reached the interpreter, so python
read EOF, did nothing, exited 0, and the launch that followed used the UNPATCHED
script -- producing a full 1,360-window run whose verdict rested on the very KL
screen the patch existed to replace. No error anywhere: the assert never ran
because the script never ran.

Same family as the documented `ssh` inside a pipe eating the rest of the script.
⇒ A patch that must apply gets its own FILE, is run in the FOREGROUND, and is
VERIFIED BY CONTENT before anything consumes it.
"""
import ast
import io

P = (r"C:/Users/Admin/AppData/Local/Temp/claude/"
     r"G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/"
     r"ff19d6ac-620c-4636-a8e4-d5402c796859/scratchpad/wpm_port_influence.py")
s = io.open(P, encoding="utf-8").read()

OLD = '''cons = consistency(energy, sel1)
print("\\n=== 5. CONSISTENCY of the port with the selection it produced ===")
print("  mean rank %.3f of %d (chance %.1f) - top1 %.4f - gap %.6f"
      % (cons.mean_rank, cons.k, cons.chance_rank, cons.top1_rate, cons.gap))'''

NEW = '''# ⛔⛔ 4b. THE SCREEN IN SECTION 2 MEASURES MAGNITUDE, AND MAGNITUDE IS THE WRONG
# AXIS. `semantic_null_screen` compares KL(S1 || S0) -- HOW FAR the distribution
# moved. The published failure mode is about the BEHAVIOURAL EFFECT, and at
# matched spread a RANDOM direction generically diverges MORE than a structured
# one, so a genuinely informative prior can lose the KL contest while being the
# only one that improves the outcome. MEASURED here: real KL 0.188 against a
# norm-matched null at 0.407 -- the null "wins" by moving further, at random.
# ⇒ the decision-grade form of the same question, on the metric that matters:
print("\\n=== 4b. THE SAME SCREEN ON THE OUTCOME, WHICH IS THE AXIS THAT DECIDES ===")
from tanitad.instruments.cot_influence import outcome_null_screen        # noqa: E402

oscr = outcome_null_screen(S0, energy, beta=1.0, outcome=ADE,
                           generator=torch.Generator().manual_seed(0),
                           lower_is_better=True)
print("  real prior          ADE %.4f m" % oscr.real)
print("  norm-matched null   ADE %.4f m   <- intervenes as hard, says nothing"
      % oscr.norm_matched)
print("  permuted prior      ADE %.4f m   <- a REAL prior, wrong window"
      % oscr.permuted)
print("  base (no port)      ADE %.4f m" % float(ade0.mean()))
print("  real beats BOTH nulls on the outcome: %s" % oscr.passes)

boot2 = {}
for tag, en in (("norm-matched", None), ("permuted", None)):
    pass
for tag, val in (("vs norm-matched", oscr.norm_matched - oscr.real),
                 ("vs permuted", oscr.permuted - oscr.real)):
    print("  margin %-16s %+.4f m   (positive = the real prior lands better)"
          % (tag, val))

cons = consistency(energy, sel1)
print("\\n=== 5. CONSISTENCY — reported as a CALIBRATION, not a verdict ===")
print("  mean rank %.3f of %d (chance %.1f) - top1 %.4f - gap %.6f"
      % (cons.mean_rank, cons.k, cons.chance_rank, cons.top1_rate, cons.gap))
print("  SCOPE: `consistency()` asks whether the ENERGY ALONE ranks the chosen")
print("     anchor first. REF-C's port is a SMALL ADDITIVE prior and the choice is")
print("     dominated by the base scorer, so a near-chance rank is EXPECTED and is")
print("     NOT evidence that REF-C is inconsistent. What it calibrates is the")
print("     DESIGN CONSTRAINT: sd_a[Delta] %.3g against an S0 spread of %.3g --"
      % (float(var_a.median()) ** 0.5, float(S0.std(dim=1).median())))
print("     the port runs at ~%.0f %% of the scorer's scale, and at that ratio"
      % (100 * float(var_a.median()) ** 0.5 / float(S0.std(dim=1).median())))
print("     consistency is UNMEASURABLE. The Energy Bridge needs beta*E comparable")
print("     to S0's spread or CON_rank reads chance whatever the reasoner knows.")'''

assert s.count(OLD) == 1, "anchor not unique: %d" % s.count(OLD)
s = s.replace(OLD, NEW)

OLDV = '''verdict = ("PORT IS INERT — no reasoner attached here can move behaviour"
           if real.kl < 1e-9 else
           "PORT IS WIRED BUT CONTENT-FREE — it does not beat a null that "
           "intervenes as hard and says nothing; the Energy Bridge would inherit "
           "magnitude, not meaning" if not scr.passes else
           "PORT IS WIRED AND CONTENT-BEARING — a trained prior beats both nulls, "
           "so the re-ranking channel carries information and is worth building on")'''
NEWV = '''# ⛔ THE VERDICT IS TAKEN ON THE OUTCOME SCREEN, NOT THE KL SCREEN — see 4b.
verdict = ("PORT IS INERT — no reasoner attached here can move behaviour"
           if real.kl < 1e-9 else
           "PORT IS WIRED AND ITS EFFECT BEATS BOTH NULLS ON THE OUTCOME — the "
           "re-ranking channel carries information and is worth building on"
           if oscr.passes else
           "PORT IS WIRED BUT DOES NOT BEAT A CONTENT-FREE PRIOR ON THE OUTCOME — "
           "the channel carries magnitude, not meaning, and a reasoner attached "
           "here would inherit that")'''
assert s.count(OLDV) == 1, "verdict anchor not unique: %d" % s.count(OLDV)
s = s.replace(OLDV, NEWV)

s = s.replace('''           "semantic_null": {"real_kl": scr.real_kl,''',
              '''           "outcome_screen": {"real": oscr.real,
                              "norm_matched": oscr.norm_matched,
                              "permuted": oscr.permuted, "passes": oscr.passes},
           "semantic_null_kl_only": {"real_kl": scr.real_kl,''')

io.open(P, "w", encoding="utf-8").write(s)
ast.parse(s)
for marker in ("=== 4b.", "outcome_null_screen", "oscr.passes"):
    assert marker in s, "marker missing after write: %r" % marker
print("PATCH APPLIED and verified by content: 4b present, verdict on the outcome axis")
