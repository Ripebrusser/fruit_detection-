"""Flask endpoint and integration tests."""
from __future__ import annotations

import io
import json

import pytest
from PIL import Image


def test_home_page_renders(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "AI-Based Fruit Detection and Freshness Analysis" in body
    assert 'id="dropzone"' in body
    assert 'id="results"' in body


def test_static_assets_exist(client) -> None:
    for path in ("/static/css/style.css", "/static/js/app.js", "/static/images/favicon.svg"):
        assert client.get(path).status_code == 200, path


def test_health_endpoint(client) -> None:
    response = client.get("/api/health")
    payload = response.get_json()
    assert "model_loaded" in payload
    assert response.status_code in (200, 503)


def test_model_info_endpoint(client) -> None:
    response = client.get("/api/model-info")
    if response.status_code == 503:
        pytest.skip("evaluation metrics not generated yet")
    payload = response.get_json()
    assert payload["ok"] is True
    assert len(payload["supported_fruits"]) == 8
    assert payload["shelf_life_supported"] is False
    assert payload["cross_section_validated"] is False
    assert 0.0 <= payload["test_accuracy"] <= 1.0


def test_analyze_valid_image(client, sample_png_bytes: bytes) -> None:
    response = client.post(
        "/api/analyze", data={"image": (io.BytesIO(sample_png_bytes), "fruit.png")}, content_type="multipart/form-data"
    )
    if response.status_code == 503:
        pytest.skip("model not trained yet")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["image_url"].startswith("/static/uploads/")
    assert payload["fresh_probability"] + payload["rotten_probability"] == pytest.approx(1.0, abs=1e-5)
    assert payload["shelf_life"]["available"] is False
    assert "cannot guarantee food safety" in payload["edibility"]["disclaimer"]
    assert 0 <= payload["identification_confidence"] <= 1

    # Clean up the stored upload through the documented endpoint.
    name = payload["image_url"].rsplit("/", 1)[-1]
    assert client.delete(f"/api/uploads/{name}").status_code == 200
    assert client.delete(f"/api/uploads/{name}").status_code == 404


def test_analyze_without_file(client) -> None:
    response = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_analyze_rejects_wrong_extension(client) -> None:
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_analyze_rejects_corrupt_image(client) -> None:
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(b"\x89PNG\r\n\x1a\n broken"), "broken.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_analyze_rejects_tiny_image(client) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(buffer, format="PNG")
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(buffer.getvalue()), "tiny.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "too small" in response.get_json()["error"]


def test_upload_size_limit(client) -> None:
    oversized = b"\x00" * (9 * 1024 * 1024)
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(oversized), "huge.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "too large" in response.get_json()["error"].lower()


def test_unknown_api_route_returns_json(client) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_delete_upload_rejects_traversal(client) -> None:
    response = client.delete("/api/uploads/..%2F..%2Fapp.py")
    assert response.status_code in (400, 404)


def test_analyze_endpoint_is_post_only(client) -> None:
    assert client.get("/api/analyze").status_code == 405


def test_home_page_never_leaks_server_paths(client) -> None:
    body = client.get("/").get_data(as_text=True)
    assert "C:\\" not in body
    assert "/home/" not in body
    assert "dataset_manifest" not in body
