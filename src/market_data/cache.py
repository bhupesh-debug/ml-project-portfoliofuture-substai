"""
Local SQLite Cache with TTL & Provenance Tracking.
Stores on-demand market quotes and historical OHLCV bars.
Ensures source timestamps are kept separate from ingestion timestamps.
"""

import sqlite3
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

DEFAULT_DB_PATH = "data/market_cache.sqlite"

class MarketCache:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, default_quote_ttl_sec: int = 300, default_history_ttl_sec: int = 3600):
        self.db_path = db_path
        self.quote_ttl = default_quote_ttl_sec       # 5 minutes for quotes
        self.history_ttl = default_history_ttl_sec   # 1 hour for historical bars
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quotes (
                    symbol TEXT PRIMARY KEY,
                    price REAL NOT NULL,
                    change REAL,
                    change_percent REAL,
                    volume INTEGER,
                    currency TEXT,
                    market_data_timestamp TEXT,
                    fetched_at TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    freshness TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    cache_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    total_bars INTEGER NOT NULL,
                    fetched_at TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_expires ON quotes(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_expires ON history(expires_at)")
            conn.commit()

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM quotes WHERE symbol = ? AND expires_at > ?", (symbol.upper(), now))
            row = cur.fetchone()
            if row:
                data = json.loads(row["payload_json"])
                data["cache_status"] = "HIT"
                return data
        return None

    def set_quote(self, symbol: str, quote_data: Dict[str, Any]):
        now = time.time()
        expires_at = now + self.quote_ttl
        symbol_clean = symbol.upper()
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quotes 
                (symbol, price, change, change_percent, volume, currency, market_data_timestamp, fetched_at, source_provider, freshness, payload_json, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol_clean,
                quote_data.get("price", 0.0),
                quote_data.get("change", 0.0),
                quote_data.get("change_percent", 0.0),
                quote_data.get("volume", 0),
                quote_data.get("currency", "USD"),
                quote_data.get("market_data_timestamp", datetime.now(timezone.utc).isoformat()),
                quote_data.get("fetched_at", datetime.now(timezone.utc).isoformat()),
                quote_data.get("source_provider", "UNKNOWN"),
                quote_data.get("freshness", "CACHED"),
                json.dumps(quote_data),
                expires_at
            ))
            conn.commit()

    def get_history(self, symbol: str, period: str, interval: str) -> Optional[Dict[str, Any]]:
        key = f"{symbol.upper()}_{period}_{interval}"
        now = time.time()
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM history WHERE cache_key = ? AND expires_at > ?", (key, now))
            row = cur.fetchone()
            if row:
                data = json.loads(row["payload_json"])
                data["cache_status"] = "HIT"
                return data
        return None

    def set_history(self, symbol: str, period: str, interval: str, history_data: Dict[str, Any]):
        key = f"{symbol.upper()}_{period}_{interval}"
        now = time.time()
        expires_at = now + self.history_ttl
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO history
                (cache_key, symbol, period, interval, total_bars, fetched_at, source_provider, payload_json, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                symbol.upper(),
                period,
                interval,
                history_data.get("total_bars", len(history_data.get("bars", []))),
                history_data.get("fetched_at", datetime.now(timezone.utc).isoformat()),
                history_data.get("source_provider", "UNKNOWN"),
                json.dumps(history_data),
                expires_at
            ))
            conn.commit()

    def clear(self):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM quotes")
            conn.execute("DELETE FROM history")
            conn.commit()

    def count(self) -> int:
        with self._get_conn() as conn:
            q = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
            h = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            return q + h
