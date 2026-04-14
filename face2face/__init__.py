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


def __getattr__(name: str):
    """Lazy-load Face2Face so importing submodules does not always load ONNX."""
    if name == "Face2Face":
        from face2face.core.face2face import Face2Face as _Face2Face

        return _Face2Face
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
