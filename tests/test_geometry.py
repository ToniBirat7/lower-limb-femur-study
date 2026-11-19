"""Tests for sphere fit, line fit and angles."""

import numpy as np
import pytest
from src.geometry import (
    fit_sphere_least_squares,
    fit_line_3d,
    angle_between_vectors_3d,
    project_vector_on_plane,
    angle_projected_on_plane,
)


def test_sphere_fit():
    true_center = np.array([15.0, -25.0, 110.0])
    true_radius = 24.5

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = true_center[0] + true_radius * np.outer(np.cos(u), np.sin(v)).flatten()
    y = true_center[1] + true_radius * np.outer(np.sin(u), np.sin(v)).flatten()
    z = true_center[2] + true_radius * np.outer(np.ones_like(u), np.cos(v)).flatten()
    sphere_pts = np.column_stack([x, y, z])

    est_center, est_radius = fit_sphere_least_squares(sphere_pts)
    assert np.allclose(est_center, true_center, atol=1e-3)
    assert np.isclose(est_radius, true_radius, atol=1e-3)


def test_line_fit():
    t = np.linspace(-100, 100, 200)
    line_pts = np.column_stack([
        np.zeros_like(t),
        np.zeros_like(t),
        t + 50.0
    ])
    centroid, direction = fit_line_3d(line_pts)
    expected_dir = np.array([0.0, 0.0, 1.0])
    angle_err = angle_between_vectors_3d(direction, expected_dir)

    assert np.isclose(angle_err, 0.0, atol=1e-3) or np.isclose(angle_err, 180.0, atol=1e-3)


def test_vector_angle_and_projection():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    ang = angle_between_vectors_3d(v1, v2)
    assert np.isclose(ang, 90.0)

    # Projection onto XY plane (normal = [0, 0, 1])
    normal = np.array([0.0, 0.0, 1.0])
    v3 = np.array([1.0, 1.0, 5.0])
    proj = project_vector_on_plane(v3, normal)
    assert np.isclose(proj[2], 0.0)
