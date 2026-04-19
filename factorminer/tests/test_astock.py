"""Tests for AShareDataLoader."""

import pytest


def test_astock_loader_basic():
    """Test basic loading of A-share data."""
    from factorminer.data import AShareDataLoader

    loader = AShareDataLoader(
        ts_codes=["000001.SZ", "600519.SH"],
        count=5,
        adj="hfq",
    )
    df = loader.load()

    assert not df.empty
    assert len(df) == 10  # 2 stocks * 5 days
    assert "asset_id" in df.columns
    assert "datetime" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "net_mf_vol" in df.columns

    loader.close()


def test_astock_loader_moneyflow_features():
    """Test that moneyflow features are loaded."""
    from factorminer.data import AShareDataLoader

    loader = AShareDataLoader(
        ts_codes=["000001.SZ"],
        count=10,
        include_moneyflow=True,
    )
    df = loader.load()

    expected_features = [
        "$net_mf_vol",
        "$net_mf_amount",
        "$lg_buy_vol",
        "$lg_sell_vol",
    ]
    col_mapping = {
        "$lg_buy_vol": "buy_lg_vol",
        "$lg_sell_vol": "sell_lg_vol",
    }
    for feat in expected_features:
        col_name = col_mapping.get(feat, feat.lstrip("$"))
        assert col_name in df.columns, f"Missing column: {col_name}"

    loader.close()


def test_astock_loader_context_manager():
    """Test context manager usage."""
    from factorminer.data import AShareDataLoader

    with AShareDataLoader(ts_codes=["000001.SZ"], count=3) as loader:
        df = loader.load()
        assert not df.empty

    # Connection should be closed automatically
    # (Can't easily test this, but context manager is good practice)


def test_astock_loader_returns():
    """Test that returns column is computed."""
    from factorminer.data import AShareDataLoader

    loader = AShareDataLoader(ts_codes=["000001.SZ"], count=10)
    df = loader.load()

    assert "returns" in df.columns
    # First row per asset should have NaN returns
    first_rows = df.groupby("asset_id").head(1)
    assert first_rows["returns"].isna().all()

    loader.close()


def test_afeatures_list():
    """Test that AFEATURES includes moneyflow features."""
    from factorminer.data import AFEATURES

    assert "$net_mf_vol" in AFEATURES
    assert "$lg_buy_vol" in AFEATURES
    assert "$elg_buy_vol" in AFEATURES
