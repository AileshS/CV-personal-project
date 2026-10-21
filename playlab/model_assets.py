"""Pinned basketball weights; download once with TLS and content verification."""
from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from playlab.dataset import file_hash

MODEL_REVISION = "4eb004437919f3667dce3d0a95575a510d763485"
MODEL_REPOSITORY = "https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker"
MODEL_URL = f"https://raw.githubusercontent.com/avishah3/AI-Basketball-Shot-Detection-Tracker/{MODEL_REVISION}/best.pt"
DEFAULT_MODEL_SHA256 = "72c1e7c64cc6054db3dc5429385630f81c2136f5935fed546e6887bfe158d306"
MODEL_FILENAME = "basketball-ball-rim.pt"
MAX_MODEL_BYTES = 32 * 1024 * 1024


def default_ball_rim_path() -> Path:
    directory = Path(os.environ.get("PLAYLAB_MODEL_DIR", Path(__file__).resolve().parents[1] / "artifacts/models"))
    return (directory / MODEL_FILENAME).resolve()


def model_provenance() -> dict:
    return {"name": "Basketball and hoop detector", "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION, "url": MODEL_URL, "sha256": DEFAULT_MODEL_SHA256,
            "classes": {"0": "Basketball", "1": "Basketball Hoop"},
            "score_definition": "Raw detector score, not a statistical confidence interval or calibrated probability.",
            "limitations": "Public pretrained weights; no accuracy estimate on the user's game. "
                           "Hoop boxes can include the rim and net. Upstream does not supply a model license file."}


def ensure_ball_rim_model() -> Path:
    """Return the exact pinned asset; never silently replace a different file."""
    destination = default_ball_rim_path()
    if destination.exists():
        if not destination.is_file() or file_hash(destination) != DEFAULT_MODEL_SHA256:
            raise ValueError("Basketball detector file does not match the pinned model. "
                             "Move that file aside before downloading the model again.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".basketball-model-", suffix=".pt", delete=False,
                                         dir=destination.parent) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(MODEL_URL, timeout=45) as response:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > MAX_MODEL_BYTES:
                        raise ValueError("Basketball model download exceeded the expected size limit.")
                    stream.write(chunk)
        if file_hash(temporary) != DEFAULT_MODEL_SHA256:
            raise ValueError("Basketball model checksum mismatch; the download was not installed.")
        # Publish without replacing anything a concurrent process/user created.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or file_hash(destination) != DEFAULT_MODEL_SHA256:
                raise ValueError("A different detector appeared during download; it was left unchanged.")
        return destination
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError("Could not download the basketball detector. Allow raw.githubusercontent.com "
                         "and retry with Internet access; the pinned model is about 6 MB.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
