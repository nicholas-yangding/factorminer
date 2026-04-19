"""A股数据加载器 - 对接 stock-data SDK.

从 tushare pro 数据源获取A股日线数据，支持：
- 股票、ETF、指数日线行情（含前复权/后复权）
- 个股资金流向（大单/中单/小单净流入）

使用 DuckDB 本地存储，支持增量同步。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AdjType = Literal["qfq", "hfq", None]

# stock_data 的 moneyflow 字段到 FactorMiner 特征名的映射
MONEYFLOW_FEATURES = {
    "$net_mf_vol": "net_mf_vol",
    "$net_mf_amount": "net_mf_amount",
    "$lg_buy_vol": "buy_lg_vol",
    "$lg_sell_vol": "sell_lg_vol",
    "$elg_buy_vol": "buy_elg_vol",
    "$elg_sell_vol": "sell_elg_vol",
    "$md_buy_vol": "buy_md_vol",
    "$md_sell_vol": "sell_md_vol",
    "$sm_buy_vol": "buy_sm_vol",
    "$sm_sell_vol": "sell_sm_vol",
}

# 全部 A股特有特征名
AFEATURES: List[str] = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amt",
    "$vwap",
    "$returns",
    "$net_mf_vol",
    "$net_mf_amount",
    "$lg_buy_vol",
    "$lg_sell_vol",
    "$elg_buy_vol",
    "$elg_sell_vol",
    "$md_buy_vol",
    "$md_sell_vol",
    "$sm_buy_vol",
    "$sm_sell_vol",
]

AFUTURE_SET = frozenset(AFEATURES)


@dataclass
class AShareDataLoader:
    """A股数据加载器.

    Parameters
    ----------
    ts_codes : list of str, optional
        股票代码列表，如 ["000001.SZ", "600519.SH"]。
        默认 None 表示加载全部可交易股票。
    adj : str, optional
        复权类型："qfq"（前复权，默认）, "hfq"（后复权）, None（不复权）。
    start : str, optional
        开始日期，ISO格式，如 "2020-01-01"。
    end : str, optional
        结束日期，ISO格式，如 "2024-12-31"。
    count : int, optional
        最近交易日数量，会覆盖 start。
    include_moneyflow : bool
        是否加载资金流向数据（默认 True）。
        资金流向数据仅保留约 2 年历史。
    """

    ts_codes: Optional[List[str]] = None
    adj: AdjType = "qfq"
    start: Optional[str] = None
    end: Optional[str] = None
    count: Optional[int] = None
    include_moneyflow: bool = True

    def __post_init__(self):
        try:
            from stock_data import StockData
            self._db = StockData()
        except ImportError:
            raise ImportError(
                "stock-data SDK 未安装。请运行: pip install stock-data"
            )

    def load(self) -> pd.DataFrame:
        """加载数据并合并为长格式 DataFrame。

        Returns
        -------
        pd.DataFrame
            列: asset_id, datetime, open, high, low, close, volume, amount,
                vwap, returns, net_mf_vol, ...
        """
        if self.count is not None:
            df_daily = self._db.stock_daily(
                self.ts_codes,
                count=self.count,
                adj=self.adj,
            )
        else:
            df_daily = self._db.stock_daily(
                self.ts_codes,
                start=self.start,
                end=self.end,
                adj=self.adj,
            )

        if df_daily.empty:
            logger.warning("日线数据为空")
            return df_daily

        df_daily = df_daily.rename(columns={
            "ts_code": "asset_id",
            "vol": "volume",
        })
        df_daily["datetime"] = pd.to_datetime(df_daily["trade_date"])
        df_daily = df_daily.sort_values(["asset_id", "datetime"])

        df_daily["vwap"] = df_daily["amount"] / df_daily["volume"].replace(0, np.nan)

        df_daily["returns"] = df_daily.groupby("asset_id")["close"].pct_change()

        if self.include_moneyflow:
            df_mf = self._load_moneyflow()
            if not df_mf.empty:
                df_daily = df_daily.merge(df_mf, on=["asset_id", "datetime"], how="left")

        df_daily = df_daily.drop(columns=["trade_date"], errors="ignore")

        logger.info(
            "加载 %d 行, %d 只股票, 时间范围 %s ~ %s",
            len(df_daily),
            df_daily["asset_id"].nunique(),
            df_daily["datetime"].min(),
            df_daily["datetime"].max(),
        )
        return df_daily

    def _load_moneyflow(self) -> pd.DataFrame:
        """加载资金流向数据."""
        try:
            if self.count is not None:
                df_mf = self._db.moneyflow(self.ts_codes, count=self.count)
            else:
                df_mf = self._db.moneyflow(
                    self.ts_codes,
                    start=self.start,
                    end=self.end,
                )
        except Exception as e:
            logger.warning("资金流向数据加载失败: %s", e)
            return pd.DataFrame()

        if df_mf.empty:
            return pd.DataFrame()

        df_mf = df_mf.rename(columns={"ts_code": "asset_id"})
        df_mf["datetime"] = pd.to_datetime(df_mf["trade_date"])
        df_mf = df_mf.drop(columns=["trade_date"], errors="ignore")

        return df_mf

    def get_stock_list(self, industry: Optional[str] = None, market: Optional[str] = None) -> pd.DataFrame:
        """获取股票列表.

        Parameters
        ----------
        industry : str, optional
            行业名称筛选。
        market : str, optional
            市场筛选，如 "主板", "创业板", "科创板"。

        Returns
        -------
        pd.DataFrame
            股票列表。
        """
        df = self._db.stock_list(industry=industry, market=market)
        return df

    def close(self):
        """关闭数据库连接."""
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
