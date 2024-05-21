import numpy as np
from insightface.app.common import Face

class FileWriteableFace(dict):
    """
    Enables the insightface Face class to be saved.
    To recast the object simply use new_face = Face(pickleSafeFace)
    The problem was that Face overwrites __getattr__ which all the pickle methods use.
    In addition, some np types are not pickleable.
    """

    def __init__(self, face: Face):
        super().__init__()
        # copy all attributes
        for attr in dir(face):
            if not attr.startswith("__"):
                try:
                    self[attr] = face[attr]
                except:
                    try:
                        if type(face.attr) in [
                            np.float32,
                            np.float64,
                            np.int32,
                            np.int64,
                        ]:
                            self.attr = (float)(face[attr])
                        else:
                            self.attr = face.attr
                    except:
                        pass

    @staticmethod
    def to_face(pickle_safe_face):
        if type(pickle_safe_face) == Face:
            return pickle_safe_face
        f = Face(pickle_safe_face)
        return f
