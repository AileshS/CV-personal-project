"""Background queue executor, independent of Streamlit page reruns."""
from __future__ import annotations

import contextlib
import os
import time

from playlab.attention import request_attention
from playlab.media import clip_video, import_game
from playlab.store import Store


def perform(store: Store, job: dict):
    job_id, payload = job["id"], job["payload"]

    def cancelled():
        return store.cancelled(job_id)

    def progress(fraction, checkpoint=None):
        store.update_job(job_id, progress=fraction, checkpoint=checkpoint)
        if cancelled():
            raise InterruptedError("Job cancelled; completed stages are retained")

    if cancelled():
        raise InterruptedError("Job cancelled")
    task = job["task"]
    if task == "import":
        result = import_game(store, payload["source"], payload.get("name"), cancelled=cancelled)
        request_attention(store, f"setup:{result['id']}",
                          "A game is ready. Confirm its baskets/team colors, review possession proposals, "
                          "and assign dataset splits before labeling.")
    elif task == "index":
        from playlab.segmentation import propose_possessions
        result = propose_possessions(store, payload["game_id"], progress=progress, cancelled=cancelled)
        request_attention(store, f"boundaries:{payload['game_id']}",
                          "Possession proposals are ready. Review boundaries and split/merge them before labeling.")
    elif task == "clip":
        possession = store.get("possession", payload["possession_id"])
        if not possession:
            raise ValueError("Possession no longer exists")
        game = store.get("game", possession["game_id"])
        old = store.get("clip", possession["id"])
        if old and not old.get("stale") and old.get("possession_revision") == possession["revision"]:
            from pathlib import Path
            if Path(old["path"]).is_file():
                return old
        path = clip_video(game["source_path"], store.root / "clips" / f"{possession['id']}.mp4",
                          possession["start"], possession["end"], cancelled=cancelled)
        result = store.save("clip", {"id": possession["id"], "possession_id": possession["id"],
                                    "path": str(path), "possession_revision": possession["revision"],
                                    "start": possession["start"], "end": possession["end"], "stale": False})
    elif task == "review_clip":
        from playlab.review_clips import perform_review_clip
        result = perform_review_clip(store, payload, cancelled=cancelled, progress=progress)
    elif task == "analyze":
        from playlab.analysis import analyze_possession
        result = analyze_possession(store, payload["possession_id"], device=payload.get("device", "cpu"),
                                    weights=payload.get("weights") or None,
                                    model_bundle=payload.get("model_bundle") or None,
                                    mode=payload.get("mode", "assisted"),
                                    progress=progress, cancelled=cancelled)
        request_attention(store, f"analysis:{payload['possession_id']}:{result.get('id')}",
                          "Shot-analysis evidence is ready. Review uncertain outcomes and defender highlights.")
    elif task == "export":
        from playlab.dataset import export_dataset
        result = str(export_dataset(store, payload["destination"]))
    else:
        raise ValueError(f"Unknown job task: {task}")
    progress(1.0)
    return result


def work_once(store: Store) -> bool:
    job = store.claim_job()
    if job is None:
        return False
    try:
        perform(store, job)
        store.update_job(job["id"], status="cancelled" if store.cancelled(job["id"]) else "done", progress=1)
    except InterruptedError as error:
        store.update_job(job["id"], status="cancelled", error=str(error))
    except Exception as error:
        store.update_job(job["id"], status="failed", error=f"{type(error).__name__}: {error}")
        request_attention(store, f"job_failed:{job['id']}",
                          "A processing job failed. Open the Jobs view for the reason and resume after fixing it.")
    return True


def run_worker(store: Store, once: bool = False, poll_seconds: float = 2):
    # A filesystem lock protects explicit recovery from an accidentally concurrent worker.
    lock_path = store.root / ".worker.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("A worker lock exists. Stop the previous worker, then use worker --recover.") from error
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        while True:
            worked = work_once(store)
            if once:
                return worked
            if not worked:
                time.sleep(poll_seconds)
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()

