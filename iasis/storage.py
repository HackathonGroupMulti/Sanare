from __future__ import annotations

import json
import os
from typing import Protocol

from iasis.schemas import RunRecord


class RunStore(Protocol):
    def save(self, record: RunRecord) -> None:
        ...

    def get(self, run_id: str) -> RunRecord | None:
        ...


class MemoryRunStore:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}

    def save(self, record: RunRecord) -> None:
        self._records[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        return self._records.get(run_id)


class PostgresRunStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for PostgresRunStore")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] to use PostgresRunStore") from exc
        self._psycopg = psycopg
        self._ensure_table()

    def save(self, record: RunRecord) -> None:
        payload = record.model_dump(mode="json")
        with self._psycopg.connect(self.database_url) as conn:
            conn.execute(
                """
                insert into analysis_runs (run_id, status, payload)
                values (%s, %s, %s::jsonb)
                on conflict (run_id) do update
                set status = excluded.status, payload = excluded.payload
                """,
                (record.run_id, record.status, json.dumps(payload)),
            )

    def get(self, run_id: str) -> RunRecord | None:
        with self._psycopg.connect(self.database_url) as conn:
            row = conn.execute("select payload from analysis_runs where run_id = %s", (run_id,)).fetchone()
        if not row:
            return None
        return RunRecord.model_validate(row[0])

    def _ensure_table(self) -> None:
        with self._psycopg.connect(self.database_url) as conn:
            conn.execute(
                """
                create table if not exists analysis_runs (
                    run_id text primary key,
                    status text not null,
                    payload jsonb not null,
                    created_at timestamptz not null default now()
                )
                """
            )


def build_run_store() -> RunStore:
    if os.getenv("DATABASE_URL"):
        return PostgresRunStore()
    return MemoryRunStore()
