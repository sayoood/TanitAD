"""TanitResim — TanitAD's branded replay-visualization web app.

TanitResim *consumes* the existing replay engine (:mod:`tanitad.replay`): it
does not re-run models. The replay app's ``--mode export`` streams
:class:`~tanitad.replay.engine.TimestepRecord` objects into a self-contained
*session bundle* (:func:`tanitad.resim.export.export_bundle`) — a directory of
one ``session.json`` plus one shared camera frame per step — which the
single-port FastAPI server (``stack/scripts/resim_app.py``) serves to a
vanilla-JS single-page app under :mod:`tanitad.resim.static`.

The bundle is fully portable (relative paths only) so it can be zipped,
downloaded off a pod, and served from anywhere. Design language, view
descriptions and the pod deploy command live in ``tanitad/resim/README.md``.
"""

from tanitad.resim.export import (RESIM_COLORS, export_bundle, resim_color,
                                  static_dir)

__all__ = ["RESIM_COLORS", "export_bundle", "resim_color", "static_dir"]
