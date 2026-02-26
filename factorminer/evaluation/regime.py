"""Regime-aware factor validation.

Classifies market periods into BULL / BEAR / SIDEWAYS regimes using
rolling return and volatility statistics, then evaluates factor IC
within each regime to ensure robustness across market conditions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class MarketRegime(enum.Enum):
    """Market regime labels."""

    BULL = 0
    BEAR = 1
    SIDEWAYS = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """Parameters controlling regime detection and per-regime IC validation.

    Attributes
    ----------
    enabled : bool
        Whether regime-aware evaluation is active.
    lookback_window : int
        Rolling window length (periods) for mean-return and volatility
        estimation.
    bull_return_threshold : float
        Minimum rolling-mean return to qualify as BULL (when volatility is
        also below the threshold).
    bear_return_threshold : float
        Maximum rolling-mean return to qualify as BEAR.
    volatility_percentile : float
        Percentile (0-1) of rolling volatility used to compute the
        vol_threshold separating low-vol (BULL) from high-vol environments.
    min_regime_ic : float
        Minimum mean |IC| required within a single regime for it to "pass".
    min_regimes_passing : int
        How many regimes must pass for the factor to be accepted.
    """

    enabled: bool = True
    lookback_window: int = 60
    bull_return_threshold: float = 0.0
    bear_return_threshold: float = 0.0
    volatility_percentile: float = 0.7
    min_regime_ic: float = 0.03
    min_regimes_passing: int = 2


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class RegimeClassification:
    """Output of :class:`RegimeDetector.classify`.

    Attributes
    ----------
    labels : np.ndarray, shape (T,)
        Integer regime codes per period (0=BULL, 1=BEAR, 2=SIDEWAYS).
    periods : Dict[MarketRegime, np.ndarray]
        Boolean masks of shape (T,) indicating which periods belong to
        each regime.
    stats : Dict[MarketRegime, Dict[str, float]]
        Descriptive statistics per regime: ``mean_return``, ``volatility``,
        ``n_periods``.
    """

    labels: np.ndarray
    periods: Dict[MarketRegime, np.ndarray]
    stats: Dict[MarketRegime, Dict[str, float]]


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Classify time periods into market regimes.

    Parameters
    ----------
    config : RegimeConfig
        Regime detection parameters.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    # ----- public API -----

    def classify(self, returns: np.ndarray) -> RegimeClassification:
        """Classify each period into a market regime.

        Parameters
        ----------
        returns : np.ndarray, shape (M, T)
            Forward returns for *M* assets over *T* periods.

        Returns
        -------
        RegimeClassification
        """
        cfg = self.config
        M, T = returns.shape

        # Cross-sectional average return per period (handles NaN)
        market_return = np.nanmean(returns, axis=0)  # (T,)

        # Rolling statistics
        rolling_mean = self._rolling_nanmean(market_return, cfg.lookback_window)
        rolling_vol = self._rolling_nanstd(market_return, cfg.lookback_window)

        # Volatility threshold from valid (non-NaN) rolling vol values
        valid_vol = rolling_vol[~np.isnan(rolling_vol)]
        if len(valid_vol) > 0:
            vol_threshold = float(
                np.percentile(valid_vol, cfg.volatility_percentile * 100)
            )
        else:
            vol_threshold = np.inf  # fallback: nothing qualifies as low-vol

        # Assign labels
        labels = np.full(T, MarketRegime.SIDEWAYS.value, dtype=np.int64)

        # BEAR: rolling_return < bear_threshold  (checked first)
        bear_mask = rolling_mean < cfg.bear_return_threshold
        labels[bear_mask] = MarketRegime.BEAR.value

        # BULL: rolling_return > bull_threshold AND rolling_vol < vol_threshold
        bull_mask = (rolling_mean > cfg.bull_return_threshold) & (
            rolling_vol < vol_threshold
        )
        labels[bull_mask] = MarketRegime.BULL.value

        # First lookback_window periods default to SIDEWAYS
        labels[: cfg.lookback_window] = MarketRegime.SIDEWAYS.value

        # Build boolean masks & stats
        periods: Dict[MarketRegime, np.ndarray] = {}
        stats: Dict[MarketRegime, Dict[str, float]] = {}

        for regime in MarketRegime:
            mask = labels == regime.value
            periods[regime] = mask
            regime_returns = market_return[mask]
            valid = regime_returns[~np.isnan(regime_returns)]
            stats[regime] = {
                "mean_return": float(np.mean(valid)) if len(valid) > 0 else 0.0,
                "volatility": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "n_periods": int(mask.sum()),
            }

        return RegimeClassification(labels=labels, periods=periods, stats=stats)

    # ----- helpers -----

    @staticmethod
    def _rolling_nanmean(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean that ignores NaN, returning NaN for the first *window-1* entries."""
        T = len(arr)
        out = np.full(T, np.nan, dtype=np.float64)
        for t in range(window - 1, T):
            chunk = arr[t - window + 1 : t + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 0:
                out[t] = float(np.mean(valid))
        return out

    @staticmethod
    def _rolling_nanstd(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling std (ddof=1) that ignores NaN."""
        T = len(arr)
        out = np.full(T, np.nan, dtype=np.float64)
        for t in range(window - 1, T):
            chunk = arr[t - window + 1 : t + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 1:
                out[t] = float(np.std(valid, ddof=1))
        return out


# ---------------------------------------------------------------------------
# Per-regime IC result
# ---------------------------------------------------------------------------

@dataclass
class RegimeICResult:
    """Evaluation result for a single factor across market regimes.

    Attributes
    ----------
    factor_name : str
        Human-readable factor identifier.
    regime_ic : Dict[MarketRegime, float]
        Mean |IC| per regime.
    regime_icir : Dict[MarketRegime, float]
        ICIR per regime.
    regime_n_periods : Dict[MarketRegime, int]
        Number of valid IC periods per regime.
    n_regimes_passing : int
        How many regimes met the ``min_regime_ic`` threshold.
    passes : bool
        Whether ``n_regimes_passing >= config.min_regimes_passing``.
    overall_regime_score : float
        Weighted average |IC| across regimes (weighted by n_periods).
    """

    factor_name: str
    regime_ic: Dict[MarketRegime, float]
    regime_icir: Dict[MarketRegime, float]
    regime_n_periods: Dict[MarketRegime, int]
    n_regimes_passing: int
    passes: bool
    overall_regime_score: float


# ---------------------------------------------------------------------------
# Regime-aware evaluator
# ---------------------------------------------------------------------------

class RegimeAwareEvaluator:
    """Evaluate factor IC within each market regime.

    Parameters
    ----------
    returns : np.ndarray, shape (M, T)
        Forward returns.
    regime : RegimeClassification
        Pre-computed regime classification.
    config : RegimeConfig
        Thresholds and evaluation parameters.
    """

    def __init__(
        self,
        returns: np.ndarray,
        regime: RegimeClassification,
        config: RegimeConfig | None = None,
    ) -> None:
        self.returns = returns
        self.regime = regime
        self.config = config or RegimeConfig()

    # ----- public API -----

    def evaluate(self, factor_name: str, signals: np.ndarray) -> RegimeICResult:
        """Evaluate a single factor across regimes.

        Parameters
        ----------
        factor_name : str
            Identifier for reporting.
        signals : np.ndarray, shape (M, T)
            Factor signal matrix.

        Returns
        -------
        RegimeICResult
        """
        cfg = self.config
        min_periods = cfg.lookback_window * 2

        regime_ic: Dict[MarketRegime, float] = {}
        regime_icir: Dict[MarketRegime, float] = {}
        regime_n_periods: Dict[MarketRegime, int] = {}

        for regime in MarketRegime:
            mask = self.regime.periods[regime]
            n_regime = int(mask.sum())

            if n_regime < min_periods:
                regime_ic[regime] = 0.0
                regime_icir[regime] = 0.0
                regime_n_periods[regime] = n_regime
                continue

            # Extract time-sliced sub-arrays
            indices = np.where(mask)[0]
            sig_sub = signals[:, indices]
            ret_sub = self.returns[:, indices]

            ic_series = self._compute_ic(sig_sub, ret_sub)
            valid_ic = ic_series[~np.isnan(ic_series)]

            mean_abs_ic = float(np.mean(np.abs(valid_ic))) if len(valid_ic) > 0 else 0.0
            icir = self._compute_icir(valid_ic)

            regime_ic[regime] = mean_abs_ic
            regime_icir[regime] = icir
            regime_n_periods[regime] = int(len(valid_ic))

        # Count passing regimes
        n_passing = sum(
            1
            for r in MarketRegime
            if regime_n_periods[r] >= min_periods and regime_ic[r] >= cfg.min_regime_ic
        )

        # Weighted average IC
        total_weight = sum(regime_n_periods[r] for r in MarketRegime)
        if total_weight > 0:
            overall_score = sum(
                regime_ic[r] * regime_n_periods[r] for r in MarketRegime
            ) / total_weight
        else:
            overall_score = 0.0

        return RegimeICResult(
            factor_name=factor_name,
            regime_ic=regime_ic,
            regime_icir=regime_icir,
            regime_n_periods=regime_n_periods,
            n_regimes_passing=n_passing,
            passes=n_passing >= cfg.min_regimes_passing,
            overall_regime_score=overall_score,
        )

    def evaluate_batch(
        self,
        candidates: Dict[str, np.ndarray],
    ) -> Dict[str, RegimeICResult]:
        """Evaluate multiple factors.

        Parameters
        ----------
        candidates : Dict[str, np.ndarray]
            Mapping of factor name to signal matrix (M, T).

        Returns
        -------
        Dict[str, RegimeICResult]
        """
        return {name: self.evaluate(name, sig) for name, sig in candidates.items()}

    # ----- IC helpers (mirrors metrics.py conventions) -----

    @staticmethod
    def _compute_ic(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Cross-sectional Spearman IC per period.

        Replicates the logic in ``metrics.compute_ic`` to keep this module
        self-contained while matching the project convention.
        """
        M, T = signals.shape
        ic_series = np.full(T, np.nan, dtype=np.float64)

        for t in range(T):
            s = signals[:, t]
            r = returns[:, t]
            valid = ~(np.isnan(s) | np.isnan(r))
            n = valid.sum()
            if n < 5:
                continue
            rs = rankdata(s[valid])
            rr = rankdata(r[valid])
            rs_m = rs - rs.mean()
            rr_m = rr - rr.mean()
            denom = np.sqrt((rs_m ** 2).sum() * (rr_m ** 2).sum())
            if denom < 1e-12:
                ic_series[t] = 0.0
            else:
                ic_series[t] = (rs_m * rr_m).sum() / denom

        return ic_series

    @staticmethod
    def _compute_icir(valid_ic: np.ndarray) -> float:
        """ICIR = mean(IC) / std(IC)."""
        if len(valid_ic) < 3:
            return 0.0
        std = float(np.std(valid_ic, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.mean(valid_ic)) / std
