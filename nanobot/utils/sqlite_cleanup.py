"""Reclaim deleted SQLite content after the caller commits its cleanup."""

import sqlite3


def vacuum_database(connection: sqlite3.Connection) -> None:
    connection.execute("VACUUM")
    if connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
        busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise sqlite3.OperationalError("数据库仍被占用，WAL 清理未完成")
