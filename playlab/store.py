"""Durable JSON records and a single-claim SQLite job queue.

The same workspace can be opened by the viewer and a separate worker. No game
or human annotation is stored solely in Streamlit's session state.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "playlab.sqlite"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(kind,id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, task TEXT NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    checkpoint TEXT NOT NULL DEFAULT '{}', error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def save(self, kind: str, record: dict, *, expected_revision: int | None = None) -> dict:
        record = dict(record)
        record.setdefault("id", uuid.uuid4().hex)
        # Version is database-owned and is not duplicated inside the JSON payload.
        record.pop("revision", None)
        encoded = json.dumps(record, allow_nan=False)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT revision FROM records WHERE kind=? AND id=?",
                             (kind, record["id"])).fetchone()
            current = old["revision"] if old else 0
            if expected_revision is not None and current != expected_revision:
                raise ValueError("This record changed elsewhere. Reload before saving.")
            revision = current + 1
            db.execute("INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE "
                       "SET payload=excluded.payload, revision=excluded.revision",
                       (kind, record["id"], encoded, revision))
        return {**record, "revision": revision}

    def get(self, kind: str, record_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload,revision FROM records WHERE kind=? AND id=?",
                             (kind, record_id)).fetchone()
        return {**json.loads(row["payload"]), "revision": row["revision"]} if row else None

    def save_many(self, changes, *, checks=(), deletes=(), collections=()):
        """Atomically save (kind, record, expected_revision) tuples with read checks."""
        prepared = []
        for kind, record, expected in changes:
            record = dict(record)
            record.setdefault("id", uuid.uuid4().hex)
            record.pop("revision", None)
            prepared.append((kind, record, expected, json.dumps(record, allow_nan=False)))
        if len({(kind, row["id"]) for kind, row, _, _ in prepared}) != len(prepared):
            raise ValueError("Duplicate record in transaction")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for kind, filters, expected in collections:
                rows = db.execute("SELECT payload,revision FROM records WHERE kind=?", (kind,)).fetchall()
                actual = {}
                for entry in rows:
                    payload = json.loads(entry["payload"])
                    if all(payload.get(key) == value for key, value in filters.items()):
                        actual[payload["id"]] = entry["revision"]
                if actual != expected:
                    raise ValueError("These records changed elsewhere. Reload before saving.")
            for kind, identifier, expected in list(checks) + list(deletes) + [
                    (kind, row["id"], expected) for kind, row, expected, _ in prepared]:
                old = db.execute("SELECT revision FROM records WHERE kind=? AND id=?",
                                 (kind, identifier)).fetchone()
                if expected is not None and (old["revision"] if old else 0) != expected:
                    raise ValueError("This record changed elsewhere. Reload before saving.")
            saved = []
            for kind, row, _, encoded in prepared:
                old = db.execute("SELECT revision FROM records WHERE kind=? AND id=?",
                                 (kind, row["id"])).fetchone()
                revision = (old["revision"] if old else 0) + 1
                db.execute("INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE "
                           "SET payload=excluded.payload, revision=excluded.revision",
                           (kind, row["id"], encoded, revision))
                saved.append({**row, "revision": revision})
            for kind, identifier, _ in deletes:
                db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identifier))
        return saved

    def list(self, kind: str, **filters) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload,revision FROM records WHERE kind=? ORDER BY rowid",
                              (kind,)).fetchall()
        records = [{**json.loads(r["payload"]), "revision": r["revision"]} for r in rows]
        return [r for r in records if all(r.get(k) == v for k, v in filters.items())]

    def delete(self, kind: str, record_id: str):
        with self.connect() as db:
            db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, record_id))

    def enqueue(self, task: str, payload: dict) -> str:
        job_id = uuid.uuid4().hex
        now = utcnow()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # An ordinary UI rerun must not start a second identical unfinished job.
            encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
            old = db.execute("SELECT id FROM jobs WHERE task=? AND payload=? "
                             "AND status IN ('queued','running')", (task, encoded)).fetchone()
            if old:
                return old["id"]
            db.execute("INSERT INTO jobs(id,task,payload,status,created_at,updated_at) "
                       "VALUES (?,?,?,'queued',?,?)", (job_id, task, encoded, now, now))
        return job_id

    def jobs(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._job(r) for r in rows]

    @staticmethod
    def _job(row) -> dict:
        return {**dict(row), "payload": json.loads(row["payload"]),
                "checkpoint": json.loads(row["checkpoint"])}

    def claim_job(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 "
                             "ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (utcnow(), row["id"]))
        return {**self._job(row), "status": "running"}

    def claim_job_by_id(self, job_id: str) -> dict | None:
        """Claim one specific queued job; None if a worker already took it."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=? AND status='queued' AND cancel_requested=0",
                             (job_id,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (utcnow(), job_id))
        return {**self._job(row), "status": "running"}

    def update_job(self, job_id: str, *, progress=None, checkpoint=None, status=None, error=None):
        fields = {"updated_at": utcnow()}
        if progress is not None:
            fields["progress"] = max(0.0, min(1.0, float(progress)))
        if checkpoint is not None:
            fields["checkpoint"] = json.dumps(checkpoint, allow_nan=False)
        if status is not None:
            if status not in {"queued", "running", "done", "failed", "cancelled"}:
                raise ValueError("Unsupported job status")
            fields["status"] = status
        fields["error"] = error
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                       (*fields.values(), job_id))

    def cancel_job(self, job_id: str):
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel_requested=1, status=CASE WHEN status='queued' "
                       "THEN 'cancelled' ELSE status END WHERE id=?", (job_id,))

    def cancelled(self, job_id: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def resume_job(self, job_id: str):
        with self.connect() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] == "done":
                raise ValueError("Only unfinished jobs can be resumed.")
            if row["status"] == "running":
                raise ValueError("Stop the worker and recover interrupted jobs before resuming.")
            db.execute("UPDATE jobs SET status='queued',cancel_requested=0,error=NULL,updated_at=? WHERE id=?",
                       (utcnow(), job_id))

    def recover_interrupted(self):
        """Explicit operation: call only after the previous worker has stopped."""
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='failed',error='Worker interrupted; resume this job', "
                       "updated_at=? WHERE status='running'", (utcnow(),))
