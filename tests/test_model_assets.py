import io

import pytest

from playlab import model_assets


def test_model_download_is_verified_and_reused(tmp_path, monkeypatch):
    import hashlib

    payload = b"synthetic model asset; never loaded as a model"
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_assets, "DEFAULT_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []
    monkeypatch.setattr(model_assets.urllib.request, "urlopen",
                        lambda url, timeout: calls.append(url) or io.BytesIO(payload))
    path = model_assets.ensure_ball_rim_model()
    assert path.read_bytes() == payload
    assert model_assets.ensure_ball_rim_model() == path
    assert len(calls) == 1
    assert calls[0].startswith("https://raw.githubusercontent.com/")
    assert not list(tmp_path.glob(".basketball-model-*"))


def test_changed_existing_model_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(tmp_path))
    path = model_assets.default_ball_rim_path()
    path.write_bytes(b"user's different model")
    with pytest.raises(ValueError, match="does not match"):
        model_assets.ensure_ball_rim_model()
    assert path.read_bytes() == b"user's different model"


def test_checksum_mismatch_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_assets.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"wrong weights"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        model_assets.ensure_ball_rim_model()
    assert not model_assets.default_ball_rim_path().exists()
    assert not list(tmp_path.glob(".basketball-model-*"))


def test_concurrent_user_model_is_not_overwritten(tmp_path, monkeypatch):
    import hashlib

    payload = b"synthetic verified model"
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_assets, "DEFAULT_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(model_assets.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(payload))

    def concurrent_link(source, destination):
        destination.write_bytes(b"user model created during installation")
        raise FileExistsError

    monkeypatch.setattr(model_assets.os, "link", concurrent_link)
    with pytest.raises(ValueError, match="left unchanged"):
        model_assets.ensure_ball_rim_model()
    assert model_assets.default_ball_rim_path().read_bytes() == b"user model created during installation"
    assert not list(tmp_path.glob(".basketball-model-*"))
