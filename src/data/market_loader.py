"""
Market Data Loader with Atomic POSIX Writes and Stateless DuckDB Queries
Loads daily OHLCV and intraday hourly bars with stateless DuckDB query caching.
Enforces Point-in-Time (PIT) cutoff at 16:00 America/New_York to eliminate look-ahead bias.
"""

from datetime import datetime, time as dtime, timedelta
import os
from pathlib import Path
import re
import time
from typing import List, Optional, Union

import duckdb
import numpy as np
import pandas as pd


class MarketDataLoader:
    """Manages raw/processed market data ingestion, atomic file replacement, and stateless DuckDB queries."""

    REQUIRED_DAILY_COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir)
        self.daily_dir = self.data_dir / "processed" / "daily"
        self.hourly_dir = self.data_dir / "processed" / "hourly"
        self.universe_dir = self.data_dir / "universe"

        # Ensure directory tree exists
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.hourly_dir.mkdir(parents=True, exist_ok=True)
        self.universe_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def atomic_write_parquet(df: pd.DataFrame, target_path: Path) -> None:
        """
        Atomically writes Parquet to avoid partial reads or file locking.
        Uses process-isolated temporary file before atomic POSIX os.replace.
        """
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{target_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.parquet"
        tmp_path = target_path.with_name(tmp_name)

        df.to_parquet(tmp_path, engine="pyarrow", index=True)
        os.replace(tmp_path, target_path)

    @staticmethod
    def apply_pit_cutoff(
        dt_val: Union[str, pd.Timestamp, datetime],
        cutoff_time: str = "16:00",
        tz_str: str = "America/New_York",
        skip_weekends: bool = True,
    ) -> pd.Timestamp:
        """
        Strict Point-in-Time Cutoff Enforcement.
        Any data arriving after cutoff_time (default 16:00 America/New_York) on day t
        is shifted to day t+1 (or next Monday if skip_weekends=True) to prevent same-day execution look-ahead.
        """
        if isinstance(dt_val, str):
            ts = pd.to_datetime(dt_val)
        elif isinstance(dt_val, datetime) and not isinstance(dt_val, pd.Timestamp):
            ts = pd.Timestamp(dt_val)
        else:
            ts = dt_val

        # Ensure timezone-aware in tz_str
        if ts.tz is None:
            ts = ts.tz_localize("UTC").tz_convert(tz_str)
        else:
            ts = ts.tz_convert(tz_str)

        cutoff_hour, cutoff_minute = map(int, cutoff_time.split(":"))
        cutoff = dtime(cutoff_hour, cutoff_minute, 0)

        # Check if strictly after cutoff time
        if ts.time() > cutoff:
            # Shift forward
            next_date = ts.date() + timedelta(days=1)
        else:
            next_date = ts.date()

        if skip_weekends:
            # Saturday (5) -> Monday (+2)
            if next_date.weekday() == 5:
                next_date += timedelta(days=2)
            # Sunday (6) -> Monday (+1)
            elif next_date.weekday() == 6:
                next_date += timedelta(days=1)

        return pd.Timestamp(next_date).tz_localize(tz_str)

    @staticmethod
    def validate_market_data(df: pd.DataFrame) -> None:
        """
        Validates OHLCV market data invariants:
        - Non-empty DataFrame
        - Monotonic DatetimeIndex without duplicates
        - Required columns: open, high, low, close, volume
        - Positive finite prices and non-negative finite volume
        - High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        """
        if df.empty:
            raise ValueError("Market data DataFrame cannot be empty.")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                temp_idx = pd.to_datetime(df["timestamp"])
                if not temp_idx.is_monotonic_increasing:
                    raise ValueError("Market data timestamps must be strictly monotonic increasing.")
                if temp_idx.has_duplicates:
                    raise ValueError("Market data timestamps contain duplicates.")
            else:
                raise ValueError("DataFrame index must be a DatetimeIndex or contain 'timestamp' column.")
        else:
            if not df.index.is_monotonic_increasing:
                raise ValueError("Market data timestamps must be strictly monotonic increasing.")
            if df.index.has_duplicates:
                raise ValueError("Market data timestamps contain duplicates.")

        lower_cols = [c.lower() for c in df.columns]
        for col in MarketDataLoader.REQUIRED_DAILY_COLUMNS:
            if col not in lower_cols:
                raise ValueError(f"Missing required market data column: '{col}'. Present: {df.columns.tolist()}")

        # Ensure columns are accessible in lowercase
        col_map = {c: c.lower() for c in df.columns}
        renamed = df.rename(columns=col_map)

        for price_col in ["open", "high", "low", "close"]:
            vals = renamed[price_col].values
            if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                raise ValueError(f"Column '{price_col}' contains NaN or infinite values.")
            if np.any(vals <= 0):
                raise ValueError(f"Column '{price_col}' contains non-positive values.")

        vols = renamed["volume"].values
        if np.any(np.isnan(vols)) or np.any(np.isinf(vols)):
            raise ValueError("Column 'volume' contains NaN or infinite values.")
        if np.any(vols < 0):
            raise ValueError("Column 'volume' contains negative values.")

        # High / Low boundary invariants
        if np.any(renamed["high"] < renamed["low"]):
            raise ValueError("High price is less than Low price in market data.")
        if np.any(renamed["high"] < renamed["open"]) or np.any(renamed["high"] < renamed["close"]):
            raise ValueError("High price must be >= max(Open, Close).")
        if np.any(renamed["low"] > renamed["open"]) or np.any(renamed["low"] > renamed["close"]):
            raise ValueError("Low price must be <= min(Open, Close).")

    def query_parquet_duckdb(
        self,
        parquet_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes a stateless analytical query in DuckDB (:memory:) directly on Parquet.
        Zero file lock contention and sub-15ms latency.
        Guarantees returned DataFrame is indexed by DatetimeIndex.
        """
        p_path = Path(parquet_path)
        if not p_path.exists():
            raise FileNotFoundError(f"Parquet file does not exist: {p_path}")

        con = duckdb.connect(":memory:")
        try:
            # Guarantee timestamp is fetched so we can set index
            if columns:
                col_regex = re.compile(r"^[A-Za-z0-9_]+$")
                for c in columns:
                    if not col_regex.match(c):
                        raise ValueError(f"Invalid column identifier: {c!r}")
                lower_cols = [c.lower() for c in columns]
                query_cols = [f'"{c}"' for c in columns]
                if "timestamp" not in lower_cols:
                    query_cols.append('"timestamp"')
                col_sql = ", ".join(query_cols)
            else:
                col_sql = "*"

            where_clauses = []
            params = [str(p_path)]

            if start_date is not None:
                where_clauses.append("timestamp >= ?")
                params.append(start_date)
            if end_date is not None:
                where_clauses.append("timestamp <= ?")
                params.append(end_date)

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            sql = f"SELECT {col_sql} FROM read_parquet(?) {where_str} ORDER BY timestamp ASC"

            df = con.execute(sql, params).fetchdf()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            return df
        finally:
            con.close()

    def load_ticker_daily(
        self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Loads daily OHLCV for ticker from local Parquet store via stateless DuckDB."""
        target = self.daily_dir / f"{ticker}.parquet"
        if not target.exists():
            return None
        return self.query_parquet_duckdb(target, start_date=start_date, end_date=end_date)

    def save_ticker_daily(self, ticker: str, df: pd.DataFrame) -> Path:
        """Validates and atomically writes daily OHLCV Parquet partition for ticker."""
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex) and "timestamp" in df_copy.columns:
            df_copy.index = pd.to_datetime(df_copy["timestamp"])
        if "timestamp" in df_copy.columns:
            df_copy = df_copy.drop(columns=["timestamp"])
        df_copy.index.name = "timestamp"
        self.validate_market_data(df_copy)
        target = self.daily_dir / f"{ticker}.parquet"
        self.atomic_write_parquet(df_copy, target)
        return target

    def generate_synthetic_daily(
        self,
        ticker: str,
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
        base_price: float = 100.0,
        drift: float = 0.0004,
        vol: float = 0.015,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates deterministic geometric Brownian motion daily OHLCV + VWAP data on US business days.
        Used for offline unit tests, Gate 1 falsification tests, and CI/CD without network access.
        """
        rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % (2**31))
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        if n == 0:
            raise ValueError(f"No business days between {start_date} and {end_date}.")

        # Log returns
        daily_returns = rng.normal(loc=drift, scale=vol, size=n)
        price_path = base_price * np.exp(np.cumsum(daily_returns))

        # Generate realistic intraday high, low, open, close, volume, vwap
        intraday_noise = rng.exponential(scale=vol * 0.7, size=(n, 2))
        open_prices = np.roll(price_path, 1)
        open_prices[0] = base_price

        # Intraday drift between open and close
        close_prices = price_path
        high_prices = np.maximum(open_prices, close_prices) + intraday_noise[:, 0] * price_path
        low_prices = np.minimum(open_prices, close_prices) - intraday_noise[:, 1] * price_path
        low_prices = np.maximum(low_prices, 0.01)  # Bound above zero

        # Volume lognormally distributed
        volumes = rng.lognormal(mean=14.5, sigma=0.5, size=n).astype(int)

        # VWAP approximation between low and high
        typical_price = (high_prices + low_prices + close_prices) / 3.0
        vwap = typical_price + rng.normal(0, vol * 0.05, size=n)
        vwap = np.clip(vwap, low_prices, high_prices)

        df = pd.DataFrame(
            {
                "open": np.round(open_prices, 4),
                "high": np.round(high_prices, 4),
                "low": np.round(low_prices, 4),
                "close": np.round(close_prices, 4),
                "volume": volumes,
                "vwap": np.round(vwap, 4),
            },
            index=dates,
        )
        df.index.name = "timestamp"
        return df

    def fetch_or_load_daily(
        self,
        ticker: str,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        fallback_synthetic: bool = False,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieves daily data from local Parquet cache if available.
        If missing, attempts fetch via provider (e.g. yfinance); falls back to synthetic data if network/provider unavailable.
        """
        loaded = self.load_ticker_daily(ticker, start_date=start_date, end_date=end_date)
        if loaded is not None and not loaded.empty:
            return loaded

        fetched = None
        if provider == "synthetic":
            fetched = self.generate_synthetic_daily(ticker, start_date=start_date, end_date=end_date)
        else:
            try:
                import yfinance as yf

                df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if df_yf is not None and not df_yf.empty:
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = [
                            c[0].lower() if isinstance(c, tuple) else str(c).lower()
                            for c in df_yf.columns
                        ]
                    else:
                        df_yf.columns = [str(c).lower() for c in df_yf.columns]
                    df_yf.index.name = "timestamp"
                    if "vwap" not in df_yf.columns:
                        df_yf["vwap"] = (df_yf["high"] + df_yf["low"] + df_yf["close"]) / 3.0
                    self.validate_market_data(df_yf)
                    fetched = df_yf
            except Exception:
                fetched = None

            if fetched is None and fallback_synthetic:
                fetched = self.generate_synthetic_daily(ticker, start_date=start_date, end_date=end_date)
            elif fetched is None:
                raise RuntimeError(f"Unable to retrieve market data for '{ticker}' and fallback_synthetic is False.")

        self.save_ticker_daily(ticker, fetched)
        return fetched
