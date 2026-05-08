"""
Row Store
SQLite-backed source-of-truth storage for full dataset rows.
"""

from __future__ import annotations
import os 
import sqlite3
import re
from typing import List, Dict, Any, Optional
import pandas as pd


class RowStore:
    """
    Stores the full dataset rows in SQLite.
    Each dataset gets its own table.
    """

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

        # 👇 keep both names (IMPORTANT FIX)
        self._raw_table: str = ""
        self._table: str = ""   # safe sql table name

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sanitize_table_name(self, name: str) -> str:
        """
        Convert any filename into safe SQLite table name
        """
        name = name.lower()
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        return f"t_{name}"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, df: pd.DataFrame, table_name: str = "dataset") -> None:
        """
        Write the full DataFrame into SQLite.
        """

        if self._conn is None:
            self.connect()

        # store raw name (for debugging / tracing)
        self._raw_table = table_name

        # safe name for SQLite only
        self._table = self._sanitize_table_name(table_name)

        df = df.copy()
        df.insert(0, "_row_id", range(len(df)))

        df.to_sql(self._table, self._conn, if_exists="replace", index=False)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def fetch_rows(self, row_ids: List[int]) -> List[Dict[str, Any]]:
        if not row_ids:
            return []

        placeholders = ",".join("?" * len(row_ids))

        cur = self._conn.execute(
            f"SELECT * FROM {self._table} WHERE _row_id IN ({placeholders})",
            row_ids,
        )

        cols = [d[0] for d in cur.description]

        rows = []
        for r in cur.fetchall():
            row_dict = dict(zip(cols, r))
            row_dict.pop("_row_id", None)
            rows.append(row_dict)

        return rows

    def fetch_all_ids(self) -> List[int]:
        cur = self._conn.execute(f"SELECT _row_id FROM {self._table}")
        return [r[0] for r in cur.fetchall()]

    def get_column_values(self, col: str) -> List[tuple]:
        cur = self._conn.execute(
            f"SELECT _row_id, [{col}] FROM {self._table}"
        )
        return cur.fetchall()

    def total_rows(self) -> int:
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {self._table}")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Numeric ops
    # ------------------------------------------------------------------

    def numeric_op(
        self,
        col: str,
        op: str,
        row_ids: Optional[List[int]] = None,
    ) -> Any:

        where = ""
        params: List[Any] = []

        if row_ids is not None:
            placeholders = ",".join("?" * len(row_ids))
            where = f"WHERE _row_id IN ({placeholders})"
            params = row_ids

        sql_op = {
            "min": "MIN",
            "max": "MAX",
            "avg": "AVG",
            "sum": "SUM",
            "count": "COUNT",
        }.get(op, "MIN")

        cur = self._conn.execute(
            f"SELECT {sql_op}([{col}]) FROM {self._table} {where}",
            params,
        )

        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def rows_sorted_by(
        self,
        col: str,
        ascending: bool = True,
        row_ids: Optional[List[int]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:

        direction = "ASC" if ascending else "DESC"

        where = ""
        params: List[Any] = []

        if row_ids is not None:
            placeholders = ",".join("?" * len(row_ids))
            where = f"WHERE _row_id IN ({placeholders})"
            params = row_ids

        params.append(limit)

        cur = self._conn.execute(
            f"""
            SELECT * FROM {self._table}
            {where}
            ORDER BY [{col}] {direction}
            LIMIT ?
            """,
            params,
        )

        cols = [d[0] for d in cur.description]

        rows = []
        for r in cur.fetchall():
            row_dict = dict(zip(cols, r))
            row_dict.pop("_row_id", None)
            rows.append(row_dict)

        return rows

    # ------------------------------------------------------------------
    # Range query
    # ------------------------------------------------------------------

    def rows_in_range(
        self,
        col: str,
        lo: float,
        hi: float,
        row_ids: Optional[List[int]] = None,
    ) -> List[int]:

        where_parts = [f"[{col}] BETWEEN ? AND ?"]
        params: List[Any] = [lo, hi]

        if row_ids is not None:
            placeholders = ",".join("?" * len(row_ids))
            where_parts.append(f"_row_id IN ({placeholders})")
            params.extend(row_ids)

        where = "WHERE " + " AND ".join(where_parts)

        cur = self._conn.execute(
            f"SELECT _row_id FROM {self._table} {where}",
            params,
        )

        return [r[0] for r in cur.fetchall()]
    
    
print("DB EXISTS:", os.path.exists("data.db"))     
    
    
    
    
print("DB EXISTS:", os.path.exists("data.db"))      