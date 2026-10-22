"""Import local clip metadata without changing existing dataset assignments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from playlab.media import probe_video, source_fingerprint


def import_quick_source(store, source, *, name=None, offset=None, provenance_path=None):
    path = Path(source).expanduser().resolve()
    meta = probe_video(path)
    provenance = {}
    if provenance_path:
        provenance = json.loads(Path(provenance_path).expanduser().read_text())
        if not isinstance(provenance, dict):
            raise ValueError("Provenance must be a JSON object")
        expected = provenance.get("clip_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Provenance requires clip_sha256")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Source SHA-256 does not match provenance")
        declared = float(provenance["clip_duration_seconds"])
        origin = float(provenance["original_source_start_seconds"])
        end = float(provenance["original_source_end_seconds"])
        if not all(math.isfinite(v) for v in (declared, origin, end)) or origin < 0:
            raise ValueError("Provenance times must be finite and nonnegative")
        if declared <= 0 or abs(declared-meta["duration"]) > .25 or abs(end-origin-declared) > .01:
            raise ValueError("Provenance duration and original interval must match the source")
        if offset is not None and float(offset) != origin:
            raise ValueError("Offset conflicts with provenance")
        offset = origin
        if provenance.get("split") not in {"train", "val", "test", "unassigned"}:
            raise ValueError("Provenance has an invalid dataset split")
    offset = 0.0 if offset is None else float(offset)
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Source offset must be finite and nonnegative")
    fingerprint = source_fingerprint(path)
    for existing in store.list("game"):
        if existing.get("source_fingerprint") == fingerprint or Path(existing["source_path"]).resolve() == path:
            if (float(existing.get("source_offset_seconds", 0)) != offset
                    or (provenance and existing.get("provenance") != provenance)
                    or (name and existing.get("name") != name)
                    or (existing.get("source_fingerprint") and existing["source_fingerprint"] != fingerprint)
                    or abs(float(existing["duration"])-meta["duration"]) > .25):
                raise ValueError("This source already exists with different metadata; existing records were preserved")
            return existing
    return store.save("game", {"name": name or path.stem, "source_path": str(path),
                              "source_fingerprint": fingerprint, "source_offset_seconds": offset,
                              "provenance": provenance, "split": provenance.get("split", "unassigned"),
                              "split_mode": "game", **meta})
