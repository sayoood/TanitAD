"""Does the TACTICAL planner read a collapsed class too? The second head, and why it matters.

`2adadda` measured a total class collapse on `perception.box_dec` — the 3-D head, 100 slots, the one
`s1_pass` scores. `RETR-2026-09-21-ALIGNMENT-MEASURED-ON-THE-WRONG-HEAD` established that there is a
SECOND, separately supervised agent head in this model: `core.agent_head`, the 2-D
:class:`AgentSlotDecoder` with 16 queries. That head was characterised only by accident, and never
on its own terms.

⭐ IT IS NOT A CURIOSITY — IT FEEDS THE HIERARCHY. Traced in source, not assumed:

  `refc.py:4185`   `agent_slots = self.agent_head(fmap.flatten(2).transpose(1, 2))`
                   ⬅ the NON-oracle branch, which is what `--agents head` selects, and what A8 ran
  `refc.py:4188`   `agent_tokens, agent_pad = self.agent_embed(agent_slots)`
  `refcv6_tactical.py:524`  `t = self.agent_in(agent_tokens) + self.source_code.weight[0]`

⇒ the tactical decoder's keys and values are built from **this head's PREDICTED slots**. ⛔ AND
THE CLASS REACHING IT IS VERIFIED IN SOURCE, NOT INFERRED FROM A LAYER WIDTH: `refc_agents.py:272`
computes `cls_p = torch.softmax(slots["cls_logits"], dim=-1)` and `:275` concatenates it straight
into the token -- `self.geo(torch.cat([feats, cls_p], dim=-1))`. So if this head's class has
collapsed the same way, the tactical planner attends over agent tokens whose class field is a
**constant**, and the hierarchy's scene input is weaker than any table has said.

⛔ THE QUESTION HERE IS CATEGORICAL, SO IT NEEDS NO MATCHER AND NO INTERVAL. "How many distinct
classes does this head EVER emit, over every slot of every window?" is answered by counting, and a
count of 1 is not a noisy estimate — it is the same form as the 2,000/2,000 reading that settled the
3-D head. Accuracy statistics are deliberately NOT computed: they would need a matcher, and the
matcher's cost includes `cls` (`agent_slots.py:475-477`), which this file does not need to touch.

⛔ THE GUARD THAT THE WRONG-HEAD RETRACTION PAID FOR: the module is selected BY NAME
(`core.agent_head.head`), and the emitted slot count is asserted to equal this head's own declared
query count — never a width match, which is exactly what bound the earlier probe to the wrong
module.
⛔ CPU only, forward hook, read-only, no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES, SLOT_SLICES   # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 20


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    root = getattr(mp, "model", None) or getattr(mp, "net", None)
    if root is None:
        for a in dir(mp):
            o = getattr(mp, a, None)
            if isinstance(o, torch.nn.Module):
                root = o
                break
    assert root is not None, "ZZABORT no model on ModelPass"

    named = dict(root.named_modules())
    cand = [(n, m) for n, m in named.items()
            if isinstance(m, torch.nn.Linear) and n.endswith("agent_head.head")]
    assert cand, f"ZZABORT no agent_head.head; agent modules: " \
                 f"{[n for n in named if 'agent_head' in n][:5]}"
    hname, hmod = cand[0]
    holder = named[hname.rsplit(".", 1)[0]]
    n_q = int(getattr(holder, "n_queries", 0) or
              getattr(getattr(holder, "query", None), "num_embeddings", 0) or 0)
    print(f"hooking {hname}  in={hmod.in_features} out={hmod.out_features}  "
          f"declared queries={n_q or 'unknown'}")

    outs = []
    h = hmod.register_forward_hook(lambda m, i, o: outs.append(o.detach()))
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]
    cls_cnt = collections.Counter()
    margins, used, seen_slots, pres = [], 0, set(), []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        outs.clear()
        mp.forward(item, str(corp.clip_ids[e_i]))
        if not outs:
            continue
        raw = outs[-1].float().reshape(-1, outs[-1].shape[-1])
        # ⛔ the identity guard the retraction paid for: this head's own query count, by name
        if n_q:
            assert raw.shape[0] == n_q, \
                f"ZZABORT hooked {raw.shape[0]} slots but {hname} declares {n_q} queries"
        seen_slots.add(int(raw.shape[0]))
        used += 1
        lg = raw[..., SLOT_SLICES["cls"]]
        cls_cnt.update(lg.argmax(-1).tolist())
        s = lg.softmax(-1)
        t2 = s.topk(2, dim=-1).values
        margins.extend((t2[:, 0] - t2[:, 1]).tolist())
        pres.extend(torch.sigmoid(raw[..., SLOT_SLICES["presence"]].squeeze(-1)).tolist())
    h.remove()
    assert used, "ZZABORT no windows produced output"

    n_pred = sum(cls_cnt.values())
    res = {"_what": "does the head that FEEDS THE TACTICAL DECODER also emit a collapsed class?",
           "_evidence_class": "MEASURED (ours), CPU, forward hook, A8 ckpt_5000",
           "_why_it_matters": [
               "refc.py:4185 — with --agents head (A8's config) agent_slots come from THIS head",
               "refc.py:4188 — agent_embed turns those slots into the agent TOKENS",
               "refcv6_tactical.py:524 — agent_in(agent_tokens) are the tactical decoder's keys/values",
               "refc_agents.py:272,275 — cls_p = softmax(cls_logits) is CONCATENATED into the "
               "token (verified in source, not inferred from the Linear(25,256) width)"],
           "_module": hname, "declared_queries": n_q or None,
           "slots_seen_per_window": sorted(seen_slots),
           "windows_used": used, "n_slot_predictions": n_pred,
           "distinct_classes_predicted": len(cls_cnt),
           "n_classes_in_vocab": len(AGENT_CLASSES),
           "distribution": {AGENT_CLASSES[k]: v for k, v in sorted(cls_cnt.items())},
           "mean_top1_minus_top2_softmax_margin": round(
               float(sum(margins) / len(margins)), 5) if margins else None,
           "presence_multiplier_on_tactical_tokens": {
               "_what": ("refc_agents.py:284 `tok = tok * presence` runs on the DEFAULT soft path "
                         "(presence_hard=False, :175) and lands AFTER self.norm (:275), so nothing "
                         "renormalises it -- this IS the scale the tactical decoder receives"),
               "mean": round(float(sum(pres) / len(pres)), 5),
               "min": round(min(pres), 5), "max": round(max(pres), 5),
               "sd": round((sum((x - sum(pres) / len(pres)) ** 2 for x in pres)
                            / len(pres)) ** 0.5, 5),
               "n": len(pres),
               "slots_above_hard_gate_0.5": int(sum(1 for x in pres if x >= 0.5)),
               "_hard_gate_note": ("presence_gate defaults to 0.5 (:171); if anyone flips "
                                   "presence_hard=True, every slot below it becomes PADDING")}}
    collapsed = len(cls_cnt) == 1
    res["_VERDICT"] = (
        "⛔⛔ THE TACTICAL PATH READS A COLLAPSED CLASS TOO — the 2-D agent head emits ONE class of "
        f"{len(AGENT_CLASSES)} on {n_pred}/{n_pred} slots, and its slots ARE the tactical decoder's "
        "keys and values. ⇒ the hierarchy's agent input carries a CONSTANT class field, so the "
        "collapse is not confined to the scored perception head."
        if collapsed else
        f"⭐ the 2-D agent head emits {len(cls_cnt)} distinct classes — it has NOT collapsed the way "
        "`perception.box_dec` has, so the two heads differ and the tactical path is not obviously "
        "reading a constant. Read the distribution before concluding anything stronger.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    pathlib.Path("tac_agent_cls.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
