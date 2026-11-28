"""Write PREREG_V8F_LEVER3_V2.md from the frozen v1 (sha256 4c019b01...) by asserted edits, then hash it."""
import hashlib

D = "F:/TanitAD-scratch/rollhist/lever3/"
v1 = open(D + "PREREG_V8F_LEVER3.md", "rb").read()
assert hashlib.sha256(v1).hexdigest().startswith("4c019b0134167ef5"), "v1 sha"
s = v1.decode("utf-8")

old_title = "# PREREG_V8F_LEVER3 (FROZEN): does a longer stage B cut RH's rollout-age cost? Rebuilt on the dev box"
new_title = "# PREREG_V8F_LEVER3_V2 (FROZEN): does a longer stage B cut RH's rollout-age cost? Rebuilt on the dev box"
v2_note = """
## Card v2 (frozen 2026-11-28, before any bank, mint, training or read of this experiment)
* **Card v1 (`PREREG_V8F_LEVER3.md`, sha256 4c019b01...) stops at its control b2 for one file.** The re-merged `native_tracks_union.pt` has sha256 938bc572... against the pod's 90229524..., at the same size (51,494,151 B). Every other b2 file is equal by md5 and sha256: v9 9881b170, v9 manifest d10e033c, lead 73693175 / e96c2d5e, speed fd808a26.
* **Cause (MEASURED, `build/TRACKS_RESAVE_PROBE2.json`):** `torch.save` writes an environment-dependent `.data/serialization_id`.
  * The pod-made 10k tracks input, re-saved here, has the same size and every zip member equal except that one.
  * The Windows-made corpus input re-saves byte-identical.
  * Two saves here, in different processes, are byte-identical (`build/TRACKS_SERIALIZATION_ID.json`).
  * So a sha256 cannot test this file's identity across the two machines.
* **The only change in v2:** control b2 for `native_tracks_union.pt` takes the CONTENT form of `build/tracks_content_check.py`:
  * the union equals the object built from the two sha-verified inputs: the same tracks in the same order, every tensor equal in dtype, shape and bytes, and the meta equal;
  * its size equals the pod record;
  * four planted arms must FAIL the check: a flipped byte, a dropped track, a swapped order and a changed meta value;
  * a positive control must PASS: a pod-made file against its own re-save (different sha256, same content).
  * Read 2026-11-28: PASS on all of these (`build/B2_TRACKS_CONTENT.json`).
* Everything else in v1 is unchanged: arms, reads, names and bars.
"""
assert s.count(old_title) == 1
s = s.replace(old_title, new_title + "\n" + v2_note, 1)
old_b2 = "   * Control b2 also covers the other merged files: lead_union sha256 e96c2d5e…, native_tracks_union sha256 90229524… (14,653 tracks), and speed_union.jsonl (from the records tar) sha256 fd808a26…, each equal to `merge_record.json`."
new_b2 = "   * Control b2 also covers the other merged files: lead_union sha256 e96c2d5e… and speed_union.jsonl (from the records tar) sha256 fd808a26…, each equal to `merge_record.json`; native_tracks_union (14,653 tracks) by the content form of card v2 (`build/tracks_content_check.py`)."
assert s.count(old_b2) == 1, "b2 line"
s = s.replace(old_b2, new_b2, 1)
out = s.encode("utf-8")
open(D + "PREREG_V8F_LEVER3_V2.md", "wb").write(out)
h = hashlib.sha256(out).hexdigest()
open(D + "PREREG_V8F_LEVER3_V2.sha256", "w").write(f"{h} *PREREG_V8F_LEVER3_V2.md\n")
print(h)
