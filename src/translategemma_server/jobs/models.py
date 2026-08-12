from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Job:
    id: str
    text: str
    src: str
    tgt: str
    max_tokens: int
    request_id: Optional[str] = None
    image: Optional[Any] = None
    src_code: Optional[str] = None
    tgt_code: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"
    worker_id: Optional[str] = None
    result: Optional[str] = None
    public_error: Optional[str] = None
    internal_error: Optional[str] = None
    inference_seconds: Optional[float] = None
    metrics: Optional[dict] = None
    runtime: Optional[dict] = None
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    def public_dict(self, include_result: bool = True) -> dict:
        payload = {
            "job_id": self.id,
            "status": self.status,
            "request_id": self.request_id,
            "source_lang": self.src,
            "target_lang": self.tgt,
            "worker_id": self.worker_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "inference_seconds": self.inference_seconds,
        }
        if include_result and self.status == "completed":
            payload["translation"] = self.result
            payload["runtime"] = self.runtime or {}
            payload["metrics"] = self.metrics or {}
        if self.status == "failed":
            payload["error"] = self.public_error or "Translation failed"
        return payload
