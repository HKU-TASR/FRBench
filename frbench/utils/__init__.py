"""Utility submodules; the stable public surface is re-exported from ``frbench``.

Attribute access is lazy so that importing this package does not create
circular imports (``frbench.detector`` imports ``frbench.utils.geometry``
while ``frbench.utils.preprocess`` imports ``frbench.detector``).
"""

_GEOMETRY_EXPORTS = {
    "ARCFACE_112_TEMPLATE",
    "align",
    "arcface_template",
    "crop",
    "estimate_similarity_transform",
    "invert_similarity",
    "square_boxes",
}

__all__ = sorted(_GEOMETRY_EXPORTS | {"Preprocessor", "Postprocessor"})


def __getattr__(name: str):
    if name in _GEOMETRY_EXPORTS:
        from . import geometry

        return getattr(geometry, name)
    if name == "Preprocessor":
        from .preprocess import Preprocessor

        return Preprocessor
    if name == "Postprocessor":
        from .postprocess import Postprocessor

        return Postprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
