"""Defect A resolution (a) + the v0-conditioned anchor plumbing, in refc_v3.py."""
import io
import sys

P = sys.argv[1]
raw = io.open(P, "rb").read()
CRLF = raw.count(b"\r\n")
BARE = raw.count(b"\n") - CRLF
assert BARE == 0 or CRLF == 0, "mixed line endings"
NL = "\r\n" if CRLF else "\n"
src = raw.decode("utf-8").replace("\r\n", "\n")
orig = src
EDITS = []
print("line endings: %s" % ("CRLF" if CRLF else "LF"))


def sub(old, new, tag):
    global src
    n = src.count(old)
    assert n == 1, "edit %r matched %d times (expected 1)" % (tag, n)
    src = src.replace(old, new)
    EDITS.append(tag)


sub("""            lat = self.lat_head_tac(z_tac)
            lon = self.lon_head_tac(z_tac)
            man5 = tac.derive_man5_logprobs(lat, lon)             # exact push-fwd""",
    """            lat = self.lat_head_tac(z_tac)
            lon = self.lon_head_tac(z_tac)
            # ⛔⛔ DEFECT A — RESOLVED 2026-09-04 (option (a), pre-registered).
            # `derive_man5_logprobs` is DEFINED on [B, 3] x [B, 3] and indexes
            # the LAT_/LON_ constants POSITIONALLY. Fed the 8-wide v7 heads it
            # did not raise: it silently read the WRONG classes — `turn_left`
            # <- LANE_CHANGE_L, `turn_right` <- LANE_CHANGE_R, `accelerate` <-
            # YIELD_MERGE, `brake_stop` <- FOLLOW, and 10 of the 16 classes were
            # never read at all — and still returned a valid-looking
            # distribution, which then reweighted `refc.py:1405`, the LIVE H19
            # anchor prior. It ran that way for all of refcv3 and for refcv4's
            # first 6,400 steps.
            #
            # ⇒ The push-forward is now called ONLY on the kin3 vocabulary its
            # positional contract is defined on. Under a v7 vocabulary the hook
            # supplies NO `maneuver_logits`, so `refc.py` falls back to the
            # CORE's own 3-wide kin3-derived 5-way (`refc.py:2115-2117`, heads
            # sized `N_LAT_MAN`/`N_LON_MAN` = 3) — the documented intent, and no
            # invented 8->5 mapping is shipped.
            #
            # ⚠️ STATED HONESTLY, because it is a real reduction: under a v7
            # vocabulary the TACTICAL BRAIN no longer drives the anchor prior.
            # H19 stays live but is fed by the core's aux head, so the tactical
            # level reaches the decoder through E7 (`target_latent`) and E9
            # (goal selection) only. That is the price of removing a scrambled
            # input rather than adding an unvalidated mapping.
            man5 = (tac.derive_man5_logprobs(lat, lon)
                    if self.tac_vocab_version == "kin3" else None)""",
    "defect-a")

sub("""            return {"maneuver_logits": man5,
                    "target_latent": self.tac_latent_proj(z_up)}""",
    """            hook_out = {"target_latent": self.tac_latent_proj(z_up)}
            if man5 is not None:
                hook_out["maneuver_logits"] = man5
            return hook_out""",
    "defect-a-return")

io.open(P, "wb").write(src.replace("\n", NL).encode("utf-8"))
print("applied %d edits: %s" % (len(EDITS), ", ".join(EDITS)))
print("delta bytes: %+d" % (len(src) - len(orig)))
