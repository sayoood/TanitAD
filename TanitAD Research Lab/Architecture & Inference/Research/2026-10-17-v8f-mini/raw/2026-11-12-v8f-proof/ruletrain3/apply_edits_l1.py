import sys
R = "F:/TanitAD-scratch/v8proof/ruletrain3/work/tree_patch/stack/"
def edit(path, pairs):
    s = open(R + path, "rb").read().decode("utf-8")
    assert "\r" not in s
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    open(R + path, "wb").write(s.encode("utf-8"))

edit("tanitad/data/rule_own.py", [
("""    def __init__(self, w_own: float, w_weight: float, J: dict | None = None, store=None, share: "RRL.RuleRelabel | None" = None):
        self.w_own, self.w_weight = float(w_own), float(w_weight)
""",
"""    def __init__(self, w_own: float, w_weight: float, J: dict | None = None, store=None, share: "RRL.RuleRelabel | None" = None, form: str = "sq"):
        self.w_own, self.w_weight = float(w_own), float(w_weight)
        if form not in FORMS:
            raise ValueError(f"rule_own_form must be one of {FORMS}, got {form!r}")
        self.form = form
        if form != "sq" and not self.w_own > 0.0:
            raise ValueError(f"rule_own_form {form} needs w_rule_own > 0 (got {w_own!r})")
"""),
("""RULE_M = RRL.RULE_M
""",
"""RULE_M = RRL.RULE_M

#: the forms of the own-plan term (``--rule-own-form``): ``sq`` = mean over the knots of the SQUARED distance (m^2, the card's form), ``l1`` = mean over the knots of the distance (m)
FORMS = ("sq", "l1")
"""),
("""def own_term(traj30: torch.Tensor, inp: dict) -> tuple:
""",
"""def knot_distance(diff: torch.Tensor) -> torch.Tensor:
    \"\"\"Euclidean norm over the last axis, in the units of ``diff``, with an exact zero value AND an exact zero gradient where the norm is 0 (a plain sqrt has an infinite slope there and gives NaN):
    the sqrt is taken on a safe input of 1.0 at the knots with a zero squared norm and the result is selected away from them. No epsilon: a knot that is not on its cap keeps the exact norm.\"\"\"
    sq = (diff ** 2).sum(-1)
    pos = sq > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, sq, torch.ones_like(sq))), torch.zeros_like(sq))


def own_term(traj30: torch.Tensor, inp: dict, form: str = "sq") -> tuple:
"""),
("""    per = ((pl - cap) ** 2).sum(-1).mean(1)                       # [n] mean over the knots of the squared distance (m^2)
""",
"""    if form == "l1":
        per = knot_distance(pl - cap).mean(1)                     # [n] mean over the knots of the distance (m); exact 0 value and gradient at a knot on its cap
    elif form == "sq":
        per = ((pl - cap) ** 2).sum(-1).mean(1)                   # [n] mean over the knots of the squared distance (m^2)
    else:
        raise ValueError(f"rule_own_form must be one of {FORMS}, got {form!r}")
"""),
("""def add_own_term(T: dict, traj30: torch.Tensor, inp: dict, W: float) -> dict:""",
 """def add_own_term(T: dict, traj30: torch.Tensor, inp: dict, W: float, form: str = "sq") -> dict:"""),
("""    term, info = own_term(traj30, inp)
    T["total"] = T["total"] + W * term""",
"""    term, info = own_term(traj30, inp, form)
    T["total"] = T["total"] + W * term"""),
])

edit("scripts/v8f_mini_train.py", [
("""    "rule_relabel_weight": "data:rule_relabel_weight",""",
 """    "rule_own_form": "data:rule_own_form",                    # RULEOWNL1 (note PREREG_V8F_PROOF_ADD34_N1): argv -> RuleOwn.form -> info["rule_own"]["form"] (l1 only; bad_rule_own) -> rule_own_probe (reports the form) -> `ROW.add_own_term(..., form)` in the loop
    "rule_relabel_weight": "data:rule_relabel_weight","""),
("""    ap.add_argument("--rule-relabel-weight", dest="rule_relabel_weight",""",
 """    ap.add_argument("--rule-own-form", dest="rule_own_form", choices=["sq", "l1"], default="sq",
                    help="RULEOWNL1 (PREREG_V8F_PROOF_ADD34_N1): the form of the --w-rule-own term. sq (default) = mean over the knots of the squared distance (m^2), bit for bit the earlier arms. l1 = mean over the knots of the Euclidean distance in metres between "
                         "the predicted knot and its cap (0 and a zero gradient at a knot on its cap). Needs --w-rule-own > 0 when l1. Everything else (mask, inputs, detached plan, float64 rule call) unchanged")
    ap.add_argument("--rule-relabel-weight", dest="rule_relabel_weight","""),
("""    if a.w_rule_own == 0.0:
        if rec and rec.get("w_rule_own") != 0.0:""",
 """    if a.rule_own_form not in ("sq", "l1"):
        bad.append(f"rule_own_form {a.rule_own_form!r} must be sq or l1")
    if a.rule_own_form != "sq" and not a.w_rule_own > 0.0:
        bad.append(f"rule_own_form {a.rule_own_form} needs --w-rule-own > 0 (the form is a form of that term)")
    if a.w_rule_own == 0.0:
        if rec and rec.get("w_rule_own") != 0.0:"""),
("""        if rec.get("w_rule_own") != a.w_rule_own:
            bad.append(f"w_rule_own: declared {a.w_rule_own} but built {rec.get('w_rule_own')}")
""",
 """        if rec.get("w_rule_own") != a.w_rule_own:
            bad.append(f"w_rule_own: declared {a.w_rule_own} but built {rec.get('w_rule_own')}")
        if rec.get("form", "sq") != a.rule_own_form:
            bad.append(f"rule_own_form: declared {a.rule_own_form} but built {rec.get('form', 'sq')}")
"""),
("""    term, info = ROW.own_term(tr, inp)
    term2, _ = ROW.own_term(tr, inp)""",
 """    term, info = ROW.own_term(tr, inp, RO.form)
    term2, _ = ROW.own_term(tr, inp, RO.form)"""),
("""    info2 = ROW.add_own_term(Tc, tr, inp, RO.w_own)
    none_inp = {**inp, "mask": np.zeros_like(inp["mask"])}
    term0, info0 = ROW.own_term(tr, none_inp)""",
 """    info2 = ROW.add_own_term(Tc, tr, inp, RO.w_own, RO.form)
    none_inp = {**inp, "mask": np.zeros_like(inp["mask"])}
    term0, info0 = ROW.own_term(tr, none_inp, RO.form)"""),
("""    rec = {"rule_own": "on", "w_rule_own": RO.w_own, "term":""",
 """    rec = {"rule_own": "on", "w_rule_own": RO.w_own, "form": RO.form, "term":"""),
("""    RO = ROW.RuleOwn(a.w_rule_own, a.rule_relabel_weight, J, store, share=RR)""",
 """    RO = ROW.RuleOwn(a.w_rule_own, a.rule_relabel_weight, J, store, share=RR, form=a.rule_own_form)"""),
("""                            "constants": {"D_M": RRL.R.D_M, "TAU_S": RRL.R.TAU_S, "B_MAX": RRL.R.B_MAX, "DT": RRL.R.DT}}
    if RO.weight_on:""",
 """                            "constants": {"D_M": RRL.R.D_M, "TAU_S": RRL.R.TAU_S, "B_MAX": RRL.R.B_MAX, "DT": RRL.R.DT}}
        if RO.form != "sq":
            info["rule_own"]["form"] = RO.form                         # RULEOWNL1: only the l1 record names its form (the sq record keeps its key set)
    if RO.weight_on:"""),
("""        levers["w_rule_own"] = RO.w_own
        levers["rule_own_rule"]""",
 """        levers["w_rule_own"] = RO.w_own
        if RO.form != "sq":
            levers["rule_own_form"] = RO.form                          # RULEOWNL1: absent for the default form
        levers["rule_own_rule"]"""),
("""ROW.add_own_term(T, out["traj30"], ro_in, RO.w_own)""",
 """ROW.add_own_term(T, out["traj30"], ro_in, RO.w_own, RO.form)"""),
])
