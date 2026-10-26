"""Reasoning-text channel tests (v10 spec §3.3, stream T). CPU only; no VLM. They check the one-way contract, rate/duty accounting, the
consistency monitor and the CoC parser -- not explanation quality."""
import json
import os
import threading
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from tanitad.hicap.knowledge import hashed_text_embedding
from tanitad.hicap.reasoning import (ExplanationPrompt, DecisionView, ReasoningChannel, ChannelCfg, ConsistencyMonitor, MonitorCfg,
                                     Explanation, AuditRecord, parse_coc, parse_meta_action, meta_to_v9, TextChannelLoRAConfig,
                                     adapter_disabled, decode_cost_estimate, META_LON_TO_LON9, META_LANE_TO_LAT9, META_LATERAL_TO_LAT9)
from tanitad.hicap.vocab import LAT_V9, LON_V9

PROBE = "/tmp/claude-0/probe"

# verbatim records (records.parquet clip 923e09b1-787c-4ef2-956f-7bc7cf906870, t0_us 5,100,000; ood_reasoning.parquet clip 000548db-...)
META_RAW = ('{"raw_outputs": ["Slow down due to the lead vehicle ahead.Longitudinal: Gentle Deceleration.\\nLateral: Steer Right.\\nLane: Lane '
            'Keep.<|traj_future_start|><|endoftext|>"], "cot": ["Slow down due to the lead vehicle ahead."], "meta_action": ["Longitudinal: '
            'Gentle Deceleration.\\nLateral: Steer Right.\\nLane: Lane Keep."], "answer": ["Slow down due to the lead vehicle ahead."], '
            '"box": [""], "cot_auto_labeling": [""]}')
AUTO_RAW = ('{"raw_outputs": ["{\\"critical_components_analysis\\": null, \\"ego_vehicle_motion_analysis\\": null, \\"trajectory_analysis\\": '
            'null, \\"chain_of_causation\\": \\"Slow down due to queued vehicles ahead\\"}<|im_end|>"], "cot": ["{\\"critical_components_'
            'analysis\\": null, \\"ego_vehicle_motion_analysis\\": null, \\"trajectory_analysis\\": null, \\"chain_of_causation\\": \\"Slow '
            'down due to queued vehicles ahead\\"}"], "meta_action": [""], "answer": [""], "box": [""], "cot_auto_labeling": ["{\\"critical_'
            'components_analysis\\": null, \\"ego_vehicle_motion_analysis\\": null, \\"trajectory_analysis\\": null, \\"chain_of_causation\\":'
            ' \\"Slow down due to queued vehicles ahead\\"}"]}')
OOD_RAW = ('[{"event_start_frame": 16, "event_start_timestamp": 1448872, "coc": "Slow down for the one-way traffic control while navigating '
           'through the construction zone."}, {"event_start_frame": 36, "event_start_timestamp": 3448872, "coc": "Steer right to exit the '
           'construction zone."}]')


class FakeClock:
    def __init__(self, t=0.0): self.t = float(t)
    def __call__(self): return self.t


def make_gen(clock, call_s=0.34, text="Braking because the lead vehicle slows."):
    calls = []
    def gen(handle, prompt, max_new_tokens):
        calls.append((handle, prompt, max_new_tokens)); clock.t += call_s; return text
    gen.calls = calls
    return gen


# ------------------------------------------------------------------ prompt
def test_prompt_rendering_template_and_rule_cap():
    d = DecisionView("LANE_KEEP", "STOP", v_target=0.0)
    p = ExplanationPrompt().render(d, ["Yield to emergency vehicles.", {"text": "Stop  at the\nstop line;"}, "third rule"])
    assert p == ("Scene (cached). Decision: lat=LANE_KEEP, lon=STOP, target 0 m/s. Applicable rules: Yield to emergency vehicles; "
                 "Stop at the stop line. In one sentence, explain why.")
    assert ExplanationPrompt().render(DecisionView("TURN_L", "KEEP"), []).endswith("Applicable rules: none. In one sentence, explain why.")
    assert "(turn left at a junction; keep the current speed)" in ExplanationPrompt(verbalise=True).render(DecisionView("TURN_L", "KEEP", 8.0))
    assert ExplanationPrompt(max_rules=4).n_rules_used(["a", "", "b"]) == 2


# ------------------------------------------------------------------ rate limiting + duty
def test_rate_limit_half_hz():
    clk = FakeClock(); ch = ReasoningChannel(make_gen(clk, call_s=0.0), ChannelCfg(), clock=clk)
    d = DecisionView("LANE_KEEP", "KEEP", 10.0)
    got = []
    for t in np.arange(0, 10.0, 0.1):                    # a 10 Hz command clock submits every tick
        clk.t = float(t); got.append(ch.submit(d).admitted)
    assert sum(got) == 5 and ch.counts["rate"] == 95     # <= 0.5 Hz: t = 0, 2, 4, 6, 8
    assert len(ch.drain()) == 5


def test_duty_budget_rejects_and_long_run_bound():
    clk = FakeClock()
    ch = ReasoningChannel(make_gen(clk, call_s=1.22), ChannelCfg(est_call_s=0.34, window_s=10.0, d_max=0.5), clock=clk)
    d = DecisionView("LANE_KEEP", "DECELERATE", 5.0)
    for k in range(20): ch.note_refresh(k * 0.5, 0.2)    # 2 Hz refresh, 0.2 s each -> D_r = 0.4
    clk.t = 10.0
    assert abs(ch.duty()["refresh"] - 0.4) < 1e-9
    r = ch.submit(d); assert r.admitted                   # 0.4 + 0.34/10 <= 0.5
    for k in range(20, 25): ch.note_refresh(k * 0.5, 0.2)
    clk.t = 12.0
    r2 = ch.submit(d); assert not r2.admitted and r2.reason == "duty"     # 0.4 + 0.122 measured > 0.5
    assert ch.stats()["predicted_call_s"] > 0.34          # EMA moved toward the measured 1.22 s

    # long-run: refresh D_r = 0.13, explanations 1.22 s (bf16 estimate) -> admitted rate must fall below 0.5 Hz, total duty <= 0.5
    clk = FakeClock(); gen = make_gen(clk, call_s=1.22)
    ch = ReasoningChannel(gen, ChannelCfg(est_call_s=1.22, window_s=10.0), clock=clk)
    t, busy_r, T = 0.0, 0.0, 300.0
    next_refresh = 0.0
    while t < T:
        while next_refresh <= t:
            ch.note_refresh(next_refresh, 0.065); busy_r += 0.065; next_refresh += 0.5
        clk.t = t; ch.submit(d); t = max(t + 0.1, clk.t)
    n = len(ch.drain()); busy_e = n * 1.22
    assert (busy_r + busy_e) / T <= 0.5 + 0.02
    assert n / T < 0.5 and n / T > 0.2                    # ~0.30 Hz predicted by (0.5 - 0.13) / 1.22


def test_disabled_busy_and_errors_never_raise():
    clk = FakeClock(); d = DecisionView("LANE_KEEP", "KEEP", 10.0)
    assert ReasoningChannel(make_gen(clk), clock=clk, enabled=False).submit(d).reason == "disabled"
    with pytest.raises(TypeError):
        ReasoningChannel(make_gen(clk), clock=clk).submit({"lat": "LANE_KEEP"})      # never the raw decide() dict

    def boom(h, p, n): raise RuntimeError("CUDA OOM")
    ch = ReasoningChannel(boom, clock=clk); assert ch.submit(d).admitted
    ex = ch.drain()[0]; assert ex.text == "" and "CUDA OOM" in ex.audit.error and ch.counts["error"] == 1

    clk.t += 5; ch = ReasoningChannel(lambda h, p, n: torch.zeros(3), clock=clk); ch.submit(d)
    ex = ch.drain()[0]; assert ex.text == "" and "expected str" in ex.audit.error

    # thread mode: a second request while the worker decodes is rejected as busy, not queued
    gate = threading.Event()
    def slow(h, p, n): gate.wait(5); return "ok <|im_end|>"
    clk = FakeClock(); ch = ReasoningChannel(slow, ChannelCfg(mode="thread"), clock=clk)
    assert ch.submit(d, prefix_handle="kv0", refresh_id=7, t_capture=-0.3).admitted
    clk.t = 3.0; r = ch.submit(d); assert r.reason == "busy"
    gate.set(); assert ch.wait_idle(5)
    out = ch.drain(); assert len(out) == 1 and out[0].text == "ok"
    assert out[0].audit.refresh_id == 7 and abs(out[0].audit.token_age_s - 0.3) < 1e-9


def test_generate_fn_receives_prefix_handle_and_token_cap():
    clk = FakeClock(); gen = make_gen(clk)
    ch = ReasoningChannel(gen, ChannelCfg(max_new_tokens=48), clock=clk)
    ch.submit(DecisionView("TURN_R", "DECELERATE", 4.0), ["Yield to pedestrians on the crossing"], prefix_handle="KV@refresh12")
    h, p, n = gen.calls[0]
    assert h == "KV@refresh12" and n == 48 and "lat=TURN_R, lon=DECELERATE, target 4 m/s" in p and "Yield to pedestrians" in p


def test_decode_cost_estimate_matches_docstring():
    assert abs(decode_cost_estimate(bits=16) - 1.22) < 0.01 and abs(decode_cost_estimate(bits=4.5) - 0.34) < 0.01


# ------------------------------------------------------------------ one-way contract: decide() is byte-identical with the channel on/off
def _bytes(out: dict) -> bytes:
    b = []
    for k in sorted(out):
        v = out[k]
        b.append(k.encode() + (np.ascontiguousarray(v).tobytes() + str((v.dtype, v.shape)).encode() if isinstance(v, np.ndarray)
                               else repr(v).encode()))
    return b"|".join(b)


def _primitive(x) -> bool:
    if is_dataclass(x): return all(_primitive(getattr(x, f.name)) for f in fields(x))
    if isinstance(x, (tuple, list)): return all(_primitive(y) for y in x)
    return x is None or isinstance(x, (str, int, float, bool))


@pytest.fixture(scope="module")
def decide_setup():
    from test_hicap_architecture import synth_world, build_vt, make_batch, SMALL
    from tanitad.hicap import vocab as V
    from tanitad.hicap.model import HiCAP, HiCAPConfig
    p, lat, dv, lon, v0 = synth_world()
    vocab = V.build_factored_vocab(p, lat, dv, lon, v0, K_P=48, K_V=24, floor_P=4, floor_V=2)
    torch.manual_seed(0)
    c = HiCAPConfig(**SMALL); m = HiCAP(c)
    return m, build_vt(c, vocab), vocab, make_batch(1, c)


def test_decide_output_byte_identical_with_channel_on_vs_off(decide_setup):
    from tanitad.hicap.decode import decide, CascadeCfg
    m, Vt, vocab, b = decide_setup
    cfg = CascadeCfg(k_p=6, k_v=4, shortlist=12, deploy_k=8)
    off = decide(m, Vt, vocab, b, v0=8.0, a0=0.0, yaw=0.0, cfg=cfg); ref = _bytes(off)
    sd = {k: v.clone() for k, v in m.state_dict().items()}

    on = decide(m, Vt, vocab, b, v0=8.0, a0=0.0, yaw=0.0, cfg=cfg)
    clk = FakeClock(); ch = ReasoningChannel(make_gen(clk), clock=clk)
    mon = ConsistencyMonitor(lambda s: hashed_text_embedding(s, 128))
    dv = DecisionView.from_decide(on)
    assert dv.lat in LAT_V9 and dv.lon in LON_V9 and dv.cell == tuple(int(x) for x in on["cell"]) and dv.v_target >= 0
    rec = ch.submit(dv, ["Keep a safe time gap"], prefix_handle=object())
    ex = ch.drain()[0]; mrec = mon.check_explanation(ex, dv)
    assert _bytes(on) == ref                                         # channel use did not mutate the control output
    again = decide(m, Vt, vocab, b, v0=8.0, a0=0.0, yaw=0.0, cfg=cfg)
    assert _bytes(again) == ref                                      # nor the model / its state
    assert all(torch.equal(sd[k], v) for k, v in m.state_dict().items())
    # what the channel hands out is text + primitive audit only (no tensor, no alias of control state)
    assert isinstance(ex, Explanation) and isinstance(ex.text, str) and isinstance(ex.audit, AuditRecord)
    assert _primitive(ex.audit) and _primitive(rec) and _primitive(mrec) and _primitive(dv)


# ------------------------------------------------------------------ consistency monitor
def test_monitor_flags_contradicting_explanation():
    mon = ConsistencyMonitor(lambda s: hashed_text_embedding(s, 256))
    bad = mon.check("Speed up to a higher speed since the road ahead is clear.", "LANE_KEEP", "STOP")
    assert bad.lon_agree is False and not bad.agree and bad.lon_text == "ACCELERATE"
    good = mon.check("Brake to a stop for the red light ahead.", "LANE_KEEP", "STOP")
    assert good.lon_agree is True and good.lon_text == "STOP"
    r = mon.rate(); assert r["lon"]["n"] == 2 and r["lon"]["rate"] == 0.5
    assert mon.verifier_trigger(bad) is False                        # default system: logging only
    assert ConsistencyMonitor(lambda s: hashed_text_embedding(s, 256), MonitorCfg(verifier_trigger_ablation=True)).verifier_trigger(bad)


def test_monitor_silent_axis_not_counted():
    mon = ConsistencyMonitor(lambda s: hashed_text_embedding(s, 256), MonitorCfg(silent_margin=10.0))
    rec = mon.check("anything at all", "LANE_KEEP", "KEEP")
    assert rec.lat_agree is None and rec.lon_agree is None and rec.agree and mon.rate()["lon"]["n"] == 0


# ------------------------------------------------------------------ CoC parsing
def test_parse_coc_verbatim_both_sources():
    it = parse_coc(META_RAW, "meta_action")
    assert [i.kind for i in it] == ["cot", "meta_action"] and it[0].text == "Slow down due to the lead vehicle ahead."
    assert it[1].meta == {"Longitudinal": "Gentle Deceleration", "Lateral": "Steer Right", "Lane": "Lane Keep"}
    assert "\n" not in it[1].text and "<|" not in it[1].text
    assert parse_coc(META_RAW)[1].kind == "meta_action"                          # format autodetect
    a = parse_coc(AUTO_RAW, "auto_labeling"); assert len(a) == 1 and a[0].kind == "coc" and a[0].text == "Slow down due to queued vehicles ahead"
    assert parse_coc(AUTO_RAW)[0].kind == "coc"
    t = parse_coc(json.dumps({"cot": "Slow down due to the lead vehicle ahead.", "pred_xyz": [], "ade_vs_gt_m": 1.0})); assert t[0].kind == "cot"
    o = parse_coc(OOD_RAW)
    assert [i.kind for i in o] == ["ood_event", "ood_event"] and o[1].text == "Steer right to exit the construction zone."
    assert o[0].meta == {"event_start_frame": 16, "event_start_timestamp_us": 1448872}
    assert parse_coc(None) == [] and parse_coc(float("nan")) == [] and parse_coc('{"answer": ["x"], "meta_action": [""]}') == []


def test_meta_to_v9_keyword_table():
    m = meta_to_v9(parse_meta_action("Longitudinal: Stop.\nLateral: Go Straight.\nLane: Turn Left."))
    assert m["lon"][0] == "STOP" and m["lat"][:1] == ("TURN_L",) and m["lat"][2] == "Lane"
    m = meta_to_v9({"Longitudinal": "Reverse", "Lateral": "Sharp Steer Right"})
    assert m["lon"][0] is None and m["lat"][0] == "TURN_R" and m["lat"][2] == "Lateral"
    valid = set(LON_V9) | {None}
    assert all(p in valid and c <= set(LON_V9) and (p is None or p in c) for p, c in META_LON_TO_LON9.values())
    lat_all = {"LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R"}
    assert all((p is None or p in c) and c <= lat_all for p, c in list(META_LANE_TO_LAT9.values()) + list(META_LATERAL_TO_LAT9.values()))


@pytest.mark.skipif(not (os.path.exists(f"{PROBE}/records.parquet") and os.path.exists(f"{PROBE}/ood_reasoning.parquet")),
                    reason="probe parquet files not present (scratch data, /tmp/claude-0/probe)")
def test_parse_coc_real_parquet_rows():
    pd = pytest.importorskip("pandas")
    r = pd.read_parquet(f"{PROBE}/records.parquet")
    for task, kinds in [("meta_action", ["cot", "meta_action"]), ("auto_labeling", ["coc"]), ("trajectory", ["cot"])]:
        sub = r[r.task == task]
        got = [[i.kind for i in parse_coc(j, task)] for j in sub.raw_json]
        assert sum(g == kinds for g in got) == len(sub)
        assert all([i.kind for i in parse_coc(j)] == kinds for j in sub.raw_json.head(200))
    o = pd.read_parquet(f"{PROBE}/ood_reasoning.parquet")
    items = [parse_coc(e) for e in o.events]
    assert sum(len(x) for x in items) > 0 and all(i.kind == "ood_event" and i.text for x in items for i in x)
    assert sum(1 for e in o.events if not isinstance(e, str)) == sum(1 for x, e in zip(items, o.events) if not isinstance(e, str) and x == [])


# ------------------------------------------------------------------ text-channel LoRA config + adapter guard
def test_lora_config_scope():
    c = TextChannelLoRAConfig()
    assert c.disabled_during_tap_extraction and c.condition_on == "executed_decision" and c.max_new_tokens == 48
    assert c.module_ok("model.language_model.layers.3.self_attn.q_proj")
    assert not c.module_ok("model.visual.blocks.3.attn.q_proj") and not c.module_ok("model.visual.merger.mlp.0")
    assert not c.module_ok("model.language_model.embed_tokens") and not c.module_ok("lm_head")


def test_adapter_disabled_guard():
    class Peft:
        def __init__(self): self.on = True
        def disable_adapter(self):
            outer = self
            class Ctx:
                def __enter__(s): outer.on = False
                def __exit__(s, *a): outer.on = True
            return Ctx()
    p = Peft()
    with adapter_disabled(p): assert p.on is False
    assert p.on is True

    class HFInt:
        _hf_peft_config_loaded = True
        on = True
        def disable_adapters(self): self.on = False
        def enable_adapters(self): self.on = True
    h = HFInt()
    with pytest.raises(ValueError):
        with adapter_disabled(h):
            assert h.on is False; raise ValueError
    assert h.on is True                                                          # re-enabled even when extraction raises

    plain = object()
    with adapter_disabled(plain) as x: assert x is plain
