"""Unit tests for clinical segmentation validation metrics (Dice, HD95, ASD)."""

import numpy as np
import pytest
from src.metrics import compute_dice, compute_hd95, compute_asd


def test_identical_masks():
    grid = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)
    mask = np.zeros(grid, dtype=np.uint8)
    mask[20:44, 20:44, 20:44] = 1

    dsc = compute_dice(mask, mask)
    hd95 = compute_hd95(mask, mask, spacing)
    asd = compute_asd(mask, mask, spacing)

    assert np.isclose(dsc, 1.0)
    assert np.isclose(hd95, 0.0)
    assert np.isclose(asd, 0.0)


def test_shifted_masks():
    grid = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)
    mask1 = np.zeros(grid, dtype=np.uint8)
    mask1[20:44, 20:44, 20:44] = 1

    mask2 = np.zeros(grid, dtype=np.uint8)
    mask2[25:49, 20:44, 20:44] = 1

    dsc = compute_dice(mask1, mask2)
    hd95 = compute_hd95(mask1, mask2, spacing)
    asd = compute_asd(mask1, mask2, spacing)

    assert dsc < 1.0
    assert hd95 > 0.0
    assert asd > 0.0


def test_disjoint_masks():
    grid = (64, 64, 64)
    mask1 = np.zeros(grid, dtype=np.uint8)
    mask1[20:44, 20:44, 20:44] = 1

    mask2 = np.zeros(grid, dtype=np.uint8)
    mask2[0:10, 0:10, 0:10] = 1

    dsc = compute_dice(mask1, mask2)
    assert np.isclose(dsc, 0.0)
