"""TDD Tests for IC Metrics.

RED phase: These tests define expected behavior.
They should FAIL before the fix is applied.
"""

import numpy as np
import pytest


class TestComputeIcMean:
    """Test compute_ic_mean returns signed (not absolute) IC."""

    def test_ic_mean_returns_signed_value(self):
        """IC mean should return signed mean, not absolute."""
        from factorminer.evaluation.metrics import compute_ic_mean

        # IC series with mean = -0.02
        ic_series = np.array([-0.03, -0.02, -0.01, 0.00])
        result = compute_ic_mean(ic_series)
        
        # Should be negative, not positive
        assert result < 0, f"IC mean should be negative for negative IC series, got {result}"
        assert abs(result - (-0.015)) < 0.001

    def test_ic_mean_positive_series(self):
        """IC mean should return positive for positive IC series."""
        from factorminer.evaluation.metrics import compute_ic_mean

        ic_series = np.array([0.01, 0.02, 0.03, 0.00])
        result = compute_ic_mean(ic_series)
        
        assert result > 0, f"IC mean should be positive for positive IC series, got {result}"


class TestComputeIcAbsMean:
    """Test compute_ic_abs_mean function exists and returns absolute value."""

    def test_compute_ic_abs_mean_exists(self):
        """compute_ic_abs_mean function should exist."""
        from factorminer.evaluation.metrics import compute_ic_abs_mean
        
        # Function should be importable
        assert callable(compute_ic_abs_mean)

    def test_ic_abs_mean_returns_positive(self):
        """IC abs mean should always return positive value."""
        from factorminer.evaluation.metrics import compute_ic_abs_mean

        # IC series with negative mean
        ic_series = np.array([-0.03, -0.02, -0.01, 0.00])
        result = compute_ic_abs_mean(ic_series)
        
        # Should always be positive
        assert result >= 0, f"IC abs mean should always be >= 0, got {result}"
        assert abs(result - 0.015) < 0.001

    def test_ic_abs_mean_same_for_opposite_signs(self):
        """Absolute IC mean should be same for opposite sign series."""
        from factorminer.evaluation.metrics import compute_ic_abs_mean

        ic_positive = np.array([0.01, 0.02, 0.03, 0.00])
        ic_negative = np.array([-0.01, -0.02, -0.03, 0.00])
        
        result_pos = compute_ic_abs_mean(ic_positive)
        result_neg = compute_ic_abs_mean(ic_negative)
        
        assert abs(result_pos - result_neg) < 0.001


class TestComputeFactorStats:
    """Test compute_factor_stats returns both ic_mean and ic_abs_mean."""

    def test_factor_stats_has_both_ic_fields(self):
        """compute_factor_stats should return both ic_mean (signed) and ic_abs_mean."""
        from factorminer.evaluation.metrics import compute_factor_stats

        # Create dummy signals and returns
        signals = np.random.randn(10, 100)
        returns = np.random.randn(10, 100)
        
        stats = compute_factor_stats(signals, returns)
        
        assert "ic_mean" in stats, "Should have ic_mean field"
        assert "ic_abs_mean" in stats, "Should have ic_abs_mean field"

    def test_factor_stats_ic_mean_is_signed(self):
        """ic_mean in stats should be signed."""
        from factorminer.evaluation.metrics import compute_factor_stats

        # Create signals with built-in negative bias
        np.random.seed(42)
        returns = np.random.randn(10, 100) * 0.01
        # Signals are positively correlated with returns
        signals = returns + np.random.randn(10, 100) * 0.005
        
        stats = compute_factor_stats(signals, returns)
        
        # ic_mean should be primarily determined by correlation
        # With positive correlation, ic_mean should be positive
        assert stats["ic_mean"] > 0, f"ic_mean should be positive for positively correlated signals, got {stats['ic_mean']}"

    def test_factor_stats_ic_abs_mean_always_positive(self):
        """ic_abs_mean in stats should always be positive."""
        from factorminer.evaluation.metrics import compute_factor_stats

        signals = np.random.randn(10, 100)
        returns = np.random.randn(10, 100)
        
        stats = compute_factor_stats(signals, returns)
        
        assert stats["ic_abs_mean"] >= 0
