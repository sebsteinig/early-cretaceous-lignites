"""Make a bold Helvetica face available to matplotlib.

macOS ships Helvetica as a ``.ttc`` collection from which matplotlib only
registers the regular (weight-400) face, so ``fontweight="bold"`` / ``"semibold"``
silently fall back to regular. :func:`ensure_helvetica_bold` extracts the bold
sub-face from the system collection into matplotlib's font cache and registers
it, so bold weights render correctly.

The extracted glyphs are embedded (subset) into ``pdf.fonttype = 42`` PDFs, so
exported figures look correct for anyone opening/editing them. Only *re-running*
these scripts on a machine without the source collection (e.g. Linux/CI) falls
back to the next font in ``font.sans-serif`` (Arial, then DejaVu Sans).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager as fm

# Candidate system locations of the Helvetica collection on macOS.
_HELVETICA_TTCS = (
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/opt/X11/share/system_fonts/Helvetica.ttc"),
)


def _bold_helvetica_registered() -> bool:
    """True if matplotlib already knows a bold Helvetica face."""
    for f in fm.fontManager.ttflist:
        if f.name == "Helvetica":
            weight = f.weight
            if weight == "bold" or (isinstance(weight, (int, float)) and weight >= 600):
                return True
    return False


def ensure_helvetica_bold() -> bool:
    """Register a bold Helvetica face if one isn't already available.

    Idempotent and safe to call on non-macOS machines (it simply returns
    ``False`` if no Helvetica collection or ``fontTools`` is found).

    Returns
    -------
    bool
        ``True`` if a bold Helvetica face is available afterwards.
    """
    if _bold_helvetica_registered():
        return True

    try:
        from fontTools.ttLib import TTCollection
    except Exception:
        return False

    source = next((p for p in _HELVETICA_TTCS if p.exists()), None)
    if source is None:
        return False

    cache_dir = Path(matplotlib.get_cachedir()) / "extracted_fonts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / "Helvetica-Bold.ttf"

    if not out_file.exists():
        collection = TTCollection(str(source))
        bold_face = None
        for face in collection.fonts:
            subfamily = (face["name"].getDebugName(2) or "").lower()
            if subfamily == "bold":
                bold_face = face
                break
        if bold_face is None:
            return False
        bold_face.save(str(out_file))

    fm.fontManager.addfont(str(out_file))
    return _bold_helvetica_registered()
