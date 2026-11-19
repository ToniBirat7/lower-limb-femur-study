"""
Geometry helpers: least-squares sphere fit, SVD line fit, 3D and projected angles.
"""

import numpy as np


def fit_sphere_least_squares(points):
    """Linear least-squares sphere through (N, 3) points; returns (center, radius).

    Solves 2 x.c + k = |x|^2 for the centre c and k = r^2 - |c|^2.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        raise ValueError("At least 4 non-coplanar points required to fit a sphere.")

    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = np.sum(pts ** 2, axis=1)

    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    c_param = sol[3]
    radius_sq = c_param + np.sum(center ** 2)

    if radius_sq < 0:
        radius = 0.0
    else:
        radius = np.sqrt(radius_sq)

    return center, float(radius)


def fit_line_3d(points):
    """Line through (N, 3) points by SVD; returns (centroid, unit direction)."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction /= np.linalg.norm(direction)

    return centroid, direction


def angle_between_vectors_3d(v1, v2, degrees=True):
    """
    Computes angle between two 3D vectors.
    """
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)

    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    angle = np.arccos(dot)
    return np.degrees(angle) if degrees else angle


def project_vector_on_plane(vector, plane_normal):
    """
    Projects a 3D vector onto a plane defined by its normal vector.
    """
    n = plane_normal / np.linalg.norm(plane_normal)
    v_proj = vector - np.dot(vector, n) * n
    norm = np.linalg.norm(v_proj)
    if norm > 0:
        v_proj /= norm
    return v_proj


def angle_projected_on_plane(v1, v2, plane_normal, degrees=True):
    """
    Projects two 3D vectors onto a plane and calculates their 2D angle.
    """
    p1 = project_vector_on_plane(v1, plane_normal)
    p2 = project_vector_on_plane(v2, plane_normal)
    return angle_between_vectors_3d(p1, p2, degrees)
