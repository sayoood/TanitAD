"""Uniform frame access for an MP4 or a folder of extracted frames.

The 2025-08-11 session exists only as 2247 per-frame directories, not as a video
file, so anything that reaches for ``cv2.VideoCapture`` silently cannot run on
it.  That is how the lane and scale calibrations ended up working on the iPhone
recordings and not on the Android one.

Both access patterns matter and they have opposite costs: seeking is cheap for a
frame folder and expensive for an MP4, while sequential reading is the reverse.
This keeps the fast path for each -- ``seek`` then repeated ``read`` -- instead
of forcing one to emulate the other.
"""
from __future__ import annotations


class FrameSource:
    """``seek(i)`` then ``read()`` repeatedly; ``read`` advances by one frame."""

    def __init__(self, video):
        self.n = int(getattr(video, "nb_frames", 0))
        self.fps = float(getattr(video, "avg_fps", 0.0) or 0.0)
        self.width = int(getattr(video, "width", 0))
        self.height = int(getattr(video, "height", 0))
        self._folder = hasattr(video, "read_bgr")
        if self._folder:
            self._v = video
            self._i = 0
        else:
            import cv2
            self._cap = cv2.VideoCapture(str(video.path))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self.fps = float(fps)

    def seek(self, i: int) -> None:
        i = int(i)
        if self._folder:
            self._i = i
        else:
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def read(self):
        """Frame at the current position as BGR, or None; advances by one."""
        if self._folder:
            if not (0 <= self._i < self.n):
                return None
            img = self._v.read_bgr(self._i)
            self._i += 1
            return img
        ok, fr = self._cap.read()
        return fr if ok else None

    def release(self) -> None:
        if not self._folder:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
