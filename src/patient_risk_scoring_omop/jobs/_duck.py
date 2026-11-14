"""Run one DuckDB query over Arrow tables and return an Arrow table."""

from __future__ import annotations

import duckdb
import pyarrow as pa


def run_query(
    sql: str, tables: dict[str, pa.Table], params: dict[str, object] | None = None
) -> pa.Table:
    """Run the SQL against named Arrow tables; every `$name` placeholder binds to params."""
    connection = duckdb.connect()
    try:
        for name, table in tables.items():
            connection.register(name, table)
        return connection.sql(sql, params=params or {}).to_arrow_table()
    finally:
        connection.close()


def align(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Select a query result into a schema's column order and cast it there."""
    return table.select(schema.names).cast(schema)
