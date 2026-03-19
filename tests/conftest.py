"""Shared pytest configuration and fixtures for CorridorKeyOpenVINO tests."""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Basic frame/mask fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_frame_rgb():
    """Small 64x64 RGB frame as float32 in [0, 1] (sRGB)."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 3), dtype=np.float32)


@pytest.fixture
def sample_mask():
    """Matching 64x64 single-channel alpha mask as float32 in [0, 1]."""
    rng = np.random.default_rng(42)
    mask = rng.random((64, 64), dtype=np.float32)
    # Make it more mask-like: threshold to create distinct FG/BG regions
    return (mask > 0.5).astype(np.float32)

