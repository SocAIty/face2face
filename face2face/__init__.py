from face2face.core.face2face import Face2Face

try:
    import importlib.metadata as metadata
except ImportError:
    # For Python < 3.8
    import importlib_metadata as metadata

try:
    __version__ = metadata.version("socaity-face2face")
except Exception:
    __version__ = "0.0.0"

__all__ = ["Face2Face"]
