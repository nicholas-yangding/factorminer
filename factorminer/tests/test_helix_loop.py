"""Tests for the Helix Loop (core/helix_loop.py)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from factorminer.core.helix_loop import HelixLoop
    HAS_HELIX = True
except ImportError:
    HAS_HELIX = False

from factorminer.agent.llm_interface import MockProvider
from factorminer.core.config import MiningConfig


pytestmark = pytest.mark.skipif(not HAS_HELIX, reason="helix_loop not yet built")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_tensor(rng):
    """Small data tensor and returns for HelixLoop tests."""
    M, T, F = 10, 50, 3
    data = rng.normal(0, 1, (M, T, F))
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (M, T)), axis=1)
    returns = np.zeros((M, T))
    returns[:, 1:] = np.diff(close, axis=1) / close[:, :-1]
    return data, returns


# -----------------------------------------------------------------------
# HelixLoop can be instantiated with all defaults
# -----------------------------------------------------------------------

def test_helix_loop_instantiates_with_defaults(small_tensor):
    """HelixLoop with all features off should be instantiable."""
    data, returns = small_tensor
    config = MiningConfig(target_library_size=5, max_iterations=1)
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )
    assert loop is not None


# -----------------------------------------------------------------------
# HelixLoop with canonicalize=True
# -----------------------------------------------------------------------

def test_helix_loop_canonicalize_flag(small_tensor):
    """HelixLoop with canonicalize=True should initialize the canonicalizer."""
    data, returns = small_tensor
    config = MiningConfig(target_library_size=5, max_iterations=1)
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=True,
    )
    assert loop._canonicalize is True


# -----------------------------------------------------------------------
# HelixLoop with MockProvider runs 1 iteration
# -----------------------------------------------------------------------

def test_helix_loop_runs_one_iteration(small_tensor):
    """HelixLoop should complete 1 iteration without error using MockProvider."""
    data, returns = small_tensor
    config = MiningConfig(
        target_library_size=3,
        max_iterations=1,
        batch_size=5,
    )
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )
    # Run the loop -- should not raise
    loop.run()
    assert loop.library is not None
