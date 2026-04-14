__all__ = ["Face2Face"]


def __getattr__(name: str):
    if name == "Face2Face":
        from face2face.core.face2face import Face2Face as _Face2Face

        return _Face2Face
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
