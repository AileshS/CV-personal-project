"""Background queue executor, independent of Streamlit page reruns."""
from __future__ import annotations

import contextlib
import os
import time

from playlab.attention import request_attention
from playlab.media import clip_video, import_game
from playlab.store import Store


@contextlib.contextmanager
def _lifecycle_guard(store: Store):
    """Serialize startup/recovery; OS releases the guard even if its owner crashes."""
    with (store.root / ".worker.guard").open("a+b") as guard:
        guard.seek(0, os.SEEK_END)
        if guard.tell() == 0:
            guard.write(b"0")
            guard.flush()
        guard.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Worker startup or recovery is already in progress; retry shortly.") from error
        try:
            yield
        finally:
            guard.seek(0)
            if os.name == "nt":
                msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


def _process_alive(pid: int) -> bool | None:
    """Check without sending a signal on Windows; None means access is uncertain."""
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return True


def worker_status(store: Store) -> dict:
    """Expose stale locks so the UI never mistakes a queued job for active processing."""
    lock = store.root / ".worker.lock"
    try:
        pid = int(lock.read_text().strip())
    except FileNotFoundError:
        return {"state": "stopped", "pid": None}
    except (OSError, ValueError):
        return {"state": "unknown", "pid": None}
    alive = _process_alive(pid)
    return {"state": "running" if alive else "stopped" if alive is False else "unknown", "pid": pid}


def recover_worker(store: Store):
    """Explicit recovery must never remove the lock of a potentially live worker."""
    with _lifecycle_guard(store):
        status = worker_status(store)
        if status["state"] != "stopped":
            raise ValueError("Stop the worker before recovery; its process is still running or cannot be verified.")
        with contextlib.suppress(FileNotFoundError):
            (store.root / ".worker.lock").unlink()
        store.recover_interrupted()


def perform(store: Store, job: dict, on_progress=None):
    job_id, payload = job["id"], job["payload"]

    def cancelled():
        return store.cancelled(job_id)

    def progress(fraction, checkpoint=None):
        store.update_job(job_id, progress=fraction, checkpoint=checkpoint)
        if on_progress:
            on_progress(fraction, checkpoint or {})
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
    elif task == "analyze_clip":
        from playlab.quick_analysis import perform_analysis_clip
        result = perform_analysis_clip(store, payload, cancelled=cancelled, progress=progress)
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


def _run_claimed(store: Store, job: dict, on_progress=None):
    try:
        perform(store, job, on_progress)
        store.update_job(job["id"], status="cancelled" if store.cancelled(job["id"]) else "done", progress=1)
    except InterruptedError as error:
        store.update_job(job["id"], status="cancelled", error=str(error))
    except Exception as error:
        store.update_job(job["id"], status="failed", error=f"{type(error).__name__}: {error}")
        request_attention(store, f"job_failed:{job['id']}",
                          "A processing job failed. Open the Jobs view for the reason and resume after fixing it.")
    except BaseException:
        # A Streamlit rerun/stop interrupts in-process work; keep it resumable
        # instead of leaving a job marked running with no process behind it.
        store.update_job(job["id"], status="cancelled", error="Interrupted by leaving the page; resume to continue.")
        raise


def work_once(store: Store) -> bool:
    job = store.claim_job()
    if job is None:
        return False
    _run_claimed(store, job)
    return True


def run_job_now(store: Store, job_id: str, on_progress=None) -> bool:
    """Run one queued job in this process (the viewer); False if a worker already claimed it."""
    job = store.claim_job_by_id(job_id)
    if job is None:
        return False
    _run_claimed(store, job, on_progress)
    return True


def run_worker(store: Store, once: bool = False, poll_seconds: float = 2):
    # A filesystem lock protects explicit recovery from an accidentally concurrent worker.
    lock_path = store.root / ".worker.lock"
    with _lifecycle_guard(store):
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

