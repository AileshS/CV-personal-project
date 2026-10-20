"""Explicit human-action requests. Optional ntfy delivery is disabled by default."""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request

from playlab.store import utcnow


def notify(message: str, *, title: str = "Play Lab needs attention") -> dict:
    url = os.environ.get("PLAYLAB_NTFY_URL", "")
    if not url:
        return {"delivered": False, "reason": "Phone notifications are not configured"}
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.path.strip("/"):
        return {"delivered": False, "reason": "Notification topic is invalid; use an HTTPS topic URL"}
    headers = {"Title": title, "Priority": "default", "Content-Type": "text/plain; charset=utf-8"}
    token = os.environ.get("PLAYLAB_NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=message.encode(), headers=headers, method="POST")
    try:
        # Never follow redirects with the Authorization token.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        with urllib.request.build_opener(NoRedirect).open(request, timeout=15) as response:
            return {"delivered": 200 <= response.status < 300, "accepted_at": utcnow()}
    except (urllib.error.URLError, TimeoutError, ValueError, UnicodeError):
        return {"delivered": False, "reason": "Notification delivery failed; check the worker's network access"}


def request_attention(store, key: str, message: str, *, deliver: bool = True) -> dict:
    record_id = hashlib.sha256(key.encode()).hexdigest()[:24]
    old = store.get("attention", record_id)
    if old and not old.get("resolved"):
        # Retry unsent alerts, but do not repeatedly notify about unchanged requests.
        if deliver and not old.get("notification", {}).get("delivered"):
            old["notification"] = notify(message)
            return store.save("attention", old)
        return old
    record = store.save("attention", {"id": record_id, "key": key, "message": message,
                                      "created_at": utcnow(), "resolved": False,
                                      "notification": {"delivered": False, "reason": "Delivery pending"}})
    return store.save("attention", {**record, "notification": notify(message) if deliver else {
        "delivered": False, "reason": "Delivery disabled"}})


def resolve_attention(store, record_id: str):
    record = store.get("attention", record_id)
    if record:
        store.save("attention", {**record, "resolved": True, "resolved_at": utcnow()})

