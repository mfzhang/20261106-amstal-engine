"""
Unit tests for Universe Selection, Point-in-Time Stamping, and MarketDataLoader.
Verifies:
1. 250-stock universe target size and GICS sector ETF mappings
2. Point-in-Time annual snapshots and survivorship-bias mitigation (delisting returns)
3. Stateless DuckDB Parquet querying and atomic POSIX writes
4. 16:00 America/New_York PIT cutoff enforcement
5. Market data invariant validations
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.market_loader import MarketDataLoader
from src.data.universe import UniverseSelector, SecurityMasterRecord


def test_universe_selector_target_size_and_sectors() -> None:
    selector = UniverseSelector(target_size=250)
    snapshot = selector.get_annual_snapshot(2024)

    assert len(snapshot) == 250
    assert len(set(snapshot)) == 250  # Unique tickers

    # Verify sector mappings for key representatives across all 9 GICS sectors
    sector_checks = {
        "NVDA": "XLK",
        "UNH": "XLV",
        "AMZN": "XLY",
        "GOOGL": "XLC",
        "GE": "XLI",
        "PG": "XLP",
        "XOM": "XLE",
        "LIN": "XLB",
        "PLD": "XLRE",
    }
    for ticker, expected_etf in sector_checks.items():
        assert selector.get_sector_etf(ticker) == expected_etf

    # SPY and Sector ETFs map to themselves
    assert selector.get_sector_etf("SPY") == "SPY"
    assert selector.get_sector_etf("XLK") == "XLK"

    with pytest.raises(KeyError):
        selector.get_sector_etf("UNKNOWN_TICKER_XYZ")


def test_universe_pit_snapshots_and_delistings() -> None:
    selector = UniverseSelector(target_size=250)

    # 1. Twitter (TWTR) acquired 2022-10-27
    snapshot_2021 = selector.get_annual_snapshot(2021)
    snapshot_2023 = selector.get_annual_snapshot(2023)
    assert "TWTR" in snapshot_2021
    assert "TWTR" not in snapshot_2023

    # On terminal day, TWTR is active for final return calculation; excluded the next session
    assert selector.is_active("TWTR", "2022-10-27") is True
    assert selector.is_active("TWTR", "2022-10-28") is False

    # 2. Silicon Valley Bank (SIVB) failed 2023-03-28
    snapshot_2023 = selector.get_annual_snapshot(2023)
    snapshot_2024 = selector.get_annual_snapshot(2024)
    assert "SIVB" in snapshot_2023
    assert "SIVB" not in snapshot_2024

    assert selector.is_active("SIVB", "2023-03-28") is True
    assert selector.is_active("SIVB", "2023-03-29") is False

    # 3. Delisting metadata audit
    sivb_rec = selector.get_delisting_record("SIVB")
    assert sivb_rec is not None
    assert sivb_rec.delisting_return == -1.00
    assert sivb_rec.delisting_date == "2023-03-28"
    assert "Bank failure" in sivb_rec.delisting_reason

    twtr_rec = selector.get_delisting_record("TWTR")
    assert twtr_rec is not None
    assert twtr_rec.delisting_return == 0.005
    assert twtr_rec.delisting_date == "2022-10-27"

    # Active security has no delisting record
    aapl_rec = selector.get_delisting_record("AAPL")
    assert aapl_rec is None


def test_export_annual_snapshots(tmp_path: Path) -> None:
    selector = UniverseSelector(target_size=250)
    selector.export_annual_snapshots(tmp_path)

    snap_file = tmp_path / "constituents_2023.parquet"
    assert snap_file.exists()

    df = pd.read_parquet(snap_file)
    assert len(df) == 250
    expected_cols = [
        "ticker",
        "name",
        "cik",
        "figi",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "delisting_date",
        "delisting_return",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_duckdb_stateless_query_and_atomic_write(tmp_path: Path) -> None:
    loader = MarketDataLoader(data_dir=tmp_path)
    synth_df = loader.generate_synthetic_daily("NVDA", start_date="2022-01-01", end_date="2022-12-31", seed=42)

    target_file = tmp_path / "processed" / "daily" / "NVDA.parquet"
    loader.atomic_write_parquet(synth_df, target_file)

    assert target_file.exists()
    # Ensure no lingering temporary files
    tmp_files = list(target_file.parent.glob("*.tmp.parquet"))
    assert len(tmp_files) == 0

    # Test stateless DuckDB query with date filtering
    queried = loader.query_parquet_duckdb(
        target_file,
        columns=["open", "close", "volume"],
        start_date="2022-03-01",
        end_date="2022-03-31",
    )
    assert not queried.empty
    assert queried.index.min() >= pd.Timestamp("2022-03-01")
    assert queried.index.max() <= pd.Timestamp("2022-03-31")
    assert list(queried.columns) == ["open", "close", "volume"]
    assert queried.index.is_monotonic_increasing


def test_pit_cutoff_1600_rule() -> None:
    # 1. 15:59:59 America/New_York (before close) -> remains current date
    t1 = pd.Timestamp("2023-05-10 15:59:59", tz="America/New_York")
    cutoff1 = MarketDataLoader.apply_pit_cutoff(t1)
    assert cutoff1.date() == pd.Timestamp("2023-05-10").date()

    # 2. 16:00:00 America/New_York (exact close) -> remains current date
    t2 = pd.Timestamp("2023-05-10 16:00:00", tz="America/New_York")
    cutoff2 = MarketDataLoader.apply_pit_cutoff(t2)
    assert cutoff2.date() == pd.Timestamp("2023-05-10").date()

    # 3. 16:00:01 America/New_York (after close) -> shifted to next day
    t3 = pd.Timestamp("2023-05-10 16:00:01", tz="America/New_York")
    cutoff3 = MarketDataLoader.apply_pit_cutoff(t3)
    assert cutoff3.date() == pd.Timestamp("2023-05-11").date()

    # 4. Friday 16:05:00 America/New_York -> shifted to Monday (skipping Saturday & Sunday)
    friday_pm = pd.Timestamp("2023-05-12 16:05:00", tz="America/New_York")
    cutoff_fri = MarketDataLoader.apply_pit_cutoff(friday_pm, skip_weekends=True)
    assert cutoff_fri.date() == pd.Timestamp("2023-05-15").date()
    assert cutoff_fri.weekday() == 0  # Monday

    # 5. Saturday event -> shifted to Monday
    sat = pd.Timestamp("2023-05-13 12:00:00", tz="America/New_York")
    cutoff_sat = MarketDataLoader.apply_pit_cutoff(sat, skip_weekends=True)
    assert cutoff_sat.date() == pd.Timestamp("2023-05-15").date()


def test_market_data_validation(tmp_path: Path) -> None:
    loader = MarketDataLoader(data_dir=tmp_path)

    # Empty DataFrame
    with pytest.raises(ValueError, match="cannot be empty"):
        loader.validate_market_data(pd.DataFrame())

    # Missing column
    dates = pd.bdate_range("2023-01-01", periods=5)
    df_missing = pd.DataFrame({"open": [10.0] * 5, "high": [11.0] * 5, "low": [9.0] * 5, "close": [10.5] * 5}, index=dates)
    with pytest.raises(ValueError, match="Missing required market data column: 'volume'"):
        loader.validate_market_data(df_missing)

    # Non-monotonic timestamps
    non_mono_dates = [dates[1], dates[0], dates[2], dates[3], dates[4]]
    df_non_mono = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100},
        index=pd.DatetimeIndex(non_mono_dates),
    )
    with pytest.raises(ValueError, match="monotonic increasing"):
        loader.validate_market_data(df_non_mono)

    # Negative price
    df_neg = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": -1.0, "close": 10.0, "volume": 100},
        index=dates,
    )
    with pytest.raises(ValueError, match="contains non-positive values"):
        loader.validate_market_data(df_neg)

    # High < Low invariant violation
    df_high_low = pd.DataFrame(
        {"open": 10.0, "high": 8.0, "low": 9.0, "close": 8.5, "volume": 100},
        index=dates,
    )
    with pytest.raises(ValueError, match="High price is less than Low price"):
        loader.validate_market_data(df_high_low)


def test_fetch_or_load_daily_synthetic_cache(tmp_path: Path) -> None:
    loader = MarketDataLoader(data_dir=tmp_path)
    df = loader.fetch_or_load_daily(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-03-31",
        fallback_synthetic=True,
        provider="synthetic",
    )

    assert not df.empty
    assert len(df) > 50
    assert "close" in df.columns
    assert "vwap" in df.columns

    # Verify Parquet file was cached on disk
    cached_file = tmp_path / "processed" / "daily" / "AAPL.parquet"
    assert cached_file.exists()

    # Second call should load directly from cache
    df_cached = loader.fetch_or_load_daily("AAPL", start_date="2023-01-01", end_date="2023-03-31")
    pd.testing.assert_frame_equal(df, df_cached, check_freq=False)
