"""The only module that writes SQL. Moving to Postgres touches this file."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass(frozen=True)
class StageResult:
    id: int
    stage: str
    output: dict[str, Any]
    versions: dict[str, Any]
    created_at: str


class Repository(Protocol):
    def get_stage_result(self, stage: str, fp: str, sig: str) -> StageResult | None: ...

    def put_stage_result(
        self,
        stage: str,
        fp: str,
        sig: str,
        output: dict[str, Any],
        versions: dict[str, Any],
        *,
        image_hash: str | None = None,
        candidate_id: str | None = None,
    ) -> int: ...

    def record_attempt(
        self,
        stage: str,
        fp: str | None,
        sig: str | None,
        *,
        error_kind: str,
        error_detail: str = "",
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        image_hash: str | None = None,
        candidate_id: str | None = None,
    ) -> int: ...


class SqliteRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- candidates and images ------------------------------------------

    def save_candidate(
        self,
        *,
        id: str,
        source: str,
        title: str = "",
        listing_url: str | None = None,
        listing_id: str | None = None,
        asking_price: str | None = None,
        supplied_card_type: str | None = None,
        supplied_set: str | None = None,
    ) -> str:
        self._conn.execute(
            "INSERT OR IGNORE INTO candidate(id, source, listing_url, listing_id,"
            " title, asking_price, supplied_card_type, supplied_set, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (id, source, listing_url, listing_id, title, asking_price,
             supplied_card_type, supplied_set, _now()),
        )
        self._conn.commit()
        return id

    def save_image(
        self,
        image_hash: str,
        path: Path | str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO image(image_hash, path, width, height,"
            " created_at) VALUES(?,?,?,?,?)",
            (image_hash, str(path), width, height, _now()),
        )
        self._conn.commit()

    def link_image(
        self,
        candidate_id: str,
        image_hash: str,
        *,
        supplied_role: str | None = None,
        source_url: str | None = None,
        ordering: int = 0,
    ) -> None:
        """The many-to-many join. The same photograph across two listings is
        stored and analyzed once, and linked twice."""
        self._conn.execute(
            "INSERT OR IGNORE INTO candidate_image(candidate_id, image_hash,"
            " supplied_role, source_url, ordering) VALUES(?,?,?,?,?)",
            (candidate_id, image_hash, supplied_role, source_url, ordering),
        )
        self._conn.commit()

    # -- stage cache ----------------------------------------------------

    def get_stage_result(self, stage: str, fp: str, sig: str) -> StageResult | None:
        row = self._conn.execute(
            "SELECT id, stage, output_json, versions_json, created_at"
            " FROM stage_result WHERE stage=? AND input_fingerprint=?"
            " AND producer_signature=?",
            (stage, fp, sig),
        ).fetchone()
        if row is None:
            return None
        return StageResult(
            id=row[0], stage=row[1], output=json.loads(row[2]),
            versions=json.loads(row[3]), created_at=row[4],
        )

    def put_stage_result(
        self,
        stage: str,
        fp: str,
        sig: str,
        output: dict[str, Any],
        versions: dict[str, Any],
        *,
        image_hash: str | None = None,
        candidate_id: str | None = None,
    ) -> int:
        """Returns the row id, so `review` can reference the exact result that
        produced it rather than re-deriving the cache key."""
        self._conn.execute(
            "INSERT OR IGNORE INTO stage_result(stage, input_fingerprint,"
            " producer_signature, output_json, versions_json, created_at,"
            " image_hash, candidate_id) VALUES(?,?,?,?,?,?,?,?)",
            (stage, fp, sig, json.dumps(output), json.dumps(versions), _now(),
             image_hash, candidate_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM stage_result WHERE stage=? AND input_fingerprint=?"
            " AND producer_signature=?",
            (stage, fp, sig),
        ).fetchone()
        return int(row[0])

    def record_attempt(
        self,
        stage: str,
        fp: str | None,
        sig: str | None,
        *,
        error_kind: str,
        error_detail: str = "",
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        image_hash: str | None = None,
        candidate_id: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO stage_attempt(stage, input_fingerprint,"
            " producer_signature, error_kind, error_detail, cost_usd,"
            " latency_ms, created_at, image_hash, candidate_id)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (stage, fp, sig, error_kind, error_detail, cost_usd, latency_ms,
             _now(), image_hash, candidate_id),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # -- decisions and reviews ------------------------------------------

    def save_routing_decision(
        self,
        *,
        candidate_id: str,
        policy_version: str,
        mode: str,
        call_vision: bool,
        trigger_reasons: list[str],
        input_fingerprint: str,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO routing_decision(candidate_id, policy_version, mode,"
            " call_vision, trigger_reasons, input_fingerprint, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (candidate_id, policy_version, mode, int(call_vision),
             json.dumps(trigger_reasons), input_fingerprint, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def save_review(self, **kw: Any) -> int:
        kw = dict(kw)
        kw["output_json"] = json.dumps(kw.pop("output", {}))
        kw["rankable"] = int(kw["rankable"])
        kw.setdefault("created_at", _now())
        cols = ", ".join(kw)
        marks = ", ".join("?" for _ in kw)
        cur = self._conn.execute(
            f"INSERT INTO review({cols}) VALUES({marks})", tuple(kw.values())
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def reviews_for(self, candidate_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM review WHERE candidate_id=? ORDER BY id",
                (candidate_id,),
            )
        )
