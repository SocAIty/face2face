"""
HTTP smoke tests for face2face + APIPod.

Uses a mocked Face2Face so models are not loaded. ``/swap_video`` still hits the
route handler but ``_swap_video`` is stubbed to return a minimal VideoFile.
"""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# APIPod backend selection is read at import time — force local FastAPI, no job queue.
os.environ.setdefault("APIPOD_ORCHESTRATOR", "local")
os.environ.setdefault("APIPOD_COMPUTE", "dedicated")
os.environ.setdefault("APIPOD_PROVIDER", "localhost")


def _face_class():
    """Load Face without importing ``face2face`` package (avoids pulling Face2Face / ONNX)."""
    path = (
        Path(__file__).resolve().parents[1]
        / "face2face"
        / "core"
        / "compatibility"
        / "Face.py"
    )
    spec = importlib.util.spec_from_file_location("_face2face_face_cls", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Face

# Tiny valid PNG (1x1 transparent)
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    Face = _face_class()
    mock_f2f = MagicMock()
    mock_f2f.swap_img_to_img.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    mock_f2f.add_face.return_value = (
        "testface",
        Face({"embedding": np.zeros(512, dtype=np.float32)}),
    )
    mock_f2f.swap.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    mock_f2f.get_faces.return_value = {
        "a": Face({"embedding": np.zeros(512, dtype=np.float32)}),
    }
    mock_f2f.swap_to_face_generator.return_value = iter(
        [np.zeros((8, 8, 3), dtype=np.uint8)]
    )
    mock_f2f.enhance_faces.return_value = np.zeros((16, 16, 3), dtype=np.uint8)

    # Patch the class in core so ``f2f = Face2Face()`` in server never touches ONNX/models.
    with patch("face2face.core.face2face.Face2Face", return_value=mock_f2f):
        import face2face.server as srv

        from media_toolkit import VideoFile

        def _fake_swap_video(**kwargs):
            return VideoFile(file_name="stub.mp4").from_np_array(
                np.zeros((8, 8, 3), dtype=np.uint8)
            )

        with patch.object(srv, "_swap_video", side_effect=_fake_swap_video):
            fastapi_app = srv.app.app
            fastapi_app.include_router(srv.app)
            with TestClient(fastapi_app, raise_server_exceptions=True) as tc:
                yield tc


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200


def test_openapi_lists_face2face_endpoints(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    # APIPod normalizes path segments to kebab-case (see ``normalize_name``).
    for path in (
        "/swap-img-to-img",
        "/add-face",
        "/swap",
        "/swap-video",
        "/enhance-face",
    ):
        assert path in paths, f"missing route {path}"


def test_swap_img_to_img(client: TestClient) -> None:
    files = [
        ("source_img", ("s.png", io.BytesIO(PNG_1X1), "image/png")),
        ("target_img", ("t.png", io.BytesIO(PNG_1X1), "image/png")),
    ]
    r = client.post("/swap-img-to-img", files=files)
    assert r.status_code == 200, r.text


def test_add_face(client: TestClient) -> None:
    files = {"image": ("face.png", io.BytesIO(PNG_1X1), "image/png")}
    data = {"face_name": "alice"}
    r = client.post("/add-face", files=files, data=data)
    assert r.status_code == 200, r.text


def test_swap_image(client: TestClient) -> None:
    files = {"media": ("m.png", io.BytesIO(PNG_1X1), "image/png")}
    data = {"faces": "alice"}
    r = client.post("/swap", files=files, data=data)
    assert r.status_code == 200, r.text


def test_swap_video(client: TestClient) -> None:
    files = {
        "target_video": ("v.mp4", io.BytesIO(PNG_1X1), "application/octet-stream"),
    }
    data = {"faces": "alice"}
    r = client.post("/swap-video", files=files, data=data)
    assert r.status_code == 200, r.text


def test_enhance_face(client: TestClient) -> None:
    files = {"face_image": ("f.png", io.BytesIO(PNG_1X1), "image/png")}
    r = client.post("/enhance-face", files=files)
    assert r.status_code == 200, r.text
