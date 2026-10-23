"""What the SQL checkpoint stores share: the table layout, the statements, and one write.

`SqliteCheckpointStore` and `PostgresCheckpointStore` run the same statements; only the placeholder and the
clock differ. The statements keep a `{table}` field for the store's validated (or composed) table name.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, LiteralString

from .core.errors import ConfigError
from .core.state import RunState

if TYPE_CHECKING:
    from .encryption import StateCodec

DEFAULT_TABLE: Final = "ai_agent_harness_checkpoints"
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}")

COLUMNS: Final[tuple[LiteralString, ...]] = ("tenant_id", "run_id", "revision", "status", "schema_version", "state")
"""The row values of a write, in the order of `WriteRecord.params` (`updated_at` is set by the database)."""

STATUS_LIST: Final[LiteralString] = "'blocked', 'cancelled', 'done', 'failed', 'limit_reached', 'refused'"

FINISHED_STATUSES: Final[tuple[str, ...]] = (
    "blocked",
    "cancelled",
    "done",
    "failed",
    "limit_reached",
    "refused",
)


def sql_identifier(value: str, what: str) -> str:
    """Table and schema names go into DDL text, so only plain lower-case identifiers are accepted."""
    if not _IDENTIFIER.fullmatch(value):
        raise ConfigError(f"Invalid {what} identifier {value!r}: use 1-63 lower-case letters, digits, or underscores")
    return value


def status_index_name(table: str) -> str:
    """Names are cut at 63 characters: a short hash keeps long tables with a shared prefix from sharing an index."""
    return f"{table[:30]}_{hashlib.sha256(table.encode()).hexdigest()[:8]}_status_idx"


@dataclass(frozen=True)
class Statements:
    """The statements of one SQL dialect. Each keeps a `{table}` field."""

    upsert: LiteralString
    """One atomic write: stores the row, or changes nothing (no row returned) if the stored revision moved on."""
    load: LiteralString
    list_all: LiteralString
    list_status: LiteralString
    delete_run: LiteralString
    delete_tenant: LiteralString
    purge_all: LiteralString
    purge_tenant: LiteralString


def statements(placeholder: LiteralString, now: LiteralString, cutoff: LiteralString) -> Statements:
    """The statements with the dialect's parameter `placeholder`, current-time expression `now`, and cutoff `cutoff`."""
    p = placeholder
    updates = ", ".join(f"{column} = excluded.{column}" for column in COLUMNS[2:])
    listing = "SELECT run_id, status, revision, updated_at FROM {table} WHERE tenant_id = " + p
    order = f" ORDER BY updated_at DESC, run_id LIMIT {p}"
    return Statements(
        upsert=(
            f"INSERT INTO {{table}} AS c ({', '.join(COLUMNS)}, updated_at) "
            f"VALUES ({', '.join(p for _ in COLUMNS)}, {now}) "
            f"ON CONFLICT (tenant_id, run_id) DO UPDATE SET {updates}, updated_at = excluded.updated_at "
            f"WHERE c.revision = {p} RETURNING revision"
        ),
        load=f"SELECT state FROM {{table}} WHERE tenant_id = {p} AND run_id = {p}",
        list_all=listing + order,
        list_status=f"{listing} AND status = {p}{order}",
        delete_run=f"DELETE FROM {{table}} WHERE tenant_id = {p} AND run_id = {p}",
        delete_tenant=f"DELETE FROM {{table}} WHERE tenant_id = {p}",
        purge_all=f"DELETE FROM {{table}} WHERE status IN ({STATUS_LIST}) AND updated_at < {cutoff}",
        purge_tenant=(
            f"DELETE FROM {{table}} WHERE tenant_id = {p} AND status IN ({STATUS_LIST}) AND updated_at < {cutoff}"
        ),
    )


@dataclass(frozen=True)
class WriteRecord:
    """One checkpoint write: the new row values plus the revision the row must still have."""

    tenant_id: str
    run_id: str
    expected: int
    new_revision: int
    payload: str
    status: str
    schema_version: int

    @classmethod
    def from_state(cls, state: RunState, codec: StateCodec | None = None) -> WriteRecord:
        new_revision = state.revision + 1
        plaintext = state.model_copy(update={"revision": new_revision}).model_dump_json()
        if codec is not None:
            payload = codec.encode(
                plaintext,
                tenant_id=state.tenant_id,
                run_id=state.run_id,
                revision=new_revision,
            )
        else:
            payload = plaintext
        return cls(
            tenant_id=state.tenant_id,
            run_id=state.run_id,
            expected=state.revision,
            new_revision=new_revision,
            payload=payload,
            status=str(state.status),
            schema_version=state.schema_version,
        )

    @property
    def params(self) -> tuple[str, str, int, str, int, str, int]:
        """The values for `Statements.upsert`: the row in `COLUMNS` order, then the expected revision."""
        return (
            self.tenant_id,
            self.run_id,
            self.new_revision,
            self.status,
            self.schema_version,
            self.payload,
            self.expected,
        )
