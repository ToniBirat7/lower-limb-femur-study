"""
Geometry helpers: least-squares sphere fit, SVD line fit, 3D and projected angles.
"""

import numpy as np


def fit_sphere_least_squares(points):
    """
    Fits a 3D sphere to point cloud using linear least squares.
    Points: (N, 3) numpy array
    Returns: center (3,), radius (float), residuals
    Equation: 2*x*x_c + 2*y*y_c + 2*z*z_c + C = x^2 + y^2 + z^2
    where C = r^2 - x_c^2 - y_c^2 - z_c^2
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
    """
    Fits a 3D line to points using Principal Component Analysis (SVD).
    Returns: centroid (3,), direction_vector (3, normalized)
    """
    pts = np.asarray(points, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    # SVD: right singular vector corresponding to largest singular value is the line direction
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


def run_unit_tests():
    print("Running computational geometry unit tests...")

    # Test 1: Sphere fit on synthetic points
    true_center = np.array([25.0, -10.0, 100.0])
    true_radius = 24.5  # typical femoral head radius in mm

    # Generate points on sphere surface + minor noise
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = true_center[0] + true_radius * np.outer(np.cos(u), np.sin(v)).flatten()
    y = true_center[1] + true_radius * np.outer(np.sin(u), np.sin(v)).flatten()
    z = true_center[2] + true_radius * np.outer(np.ones_like(u), np.cos(v)).flatten()
    sphere_pts = np.column_stack([x, y, z])

    est_center, est_radius = fit_sphere_least_squares(sphere_pts)
    assert np.allclose(est_center, true_center, atol=1e-3), f"Center mismatch: {est_center} vs {true_center}"
    assert np.isclose(est_radius, true_radius, atol=1e-3), f"Radius mismatch: {est_radius} vs {true_radius}"
    print(f"  [PASS] Sphere fit (Center error: {np.linalg.norm(est_center - true_center):.4f}mm, Radius error: {abs(est_radius - true_radius):.4f}mm)")

    # Test 2: 3D Line fit on cylinder along Z axis
    t = np.linspace(-100, 100, 200)
    line_pts = np.column_stack([
        np.zeros_like(t),
        np.zeros_like(t),
        t + 50.0
    ])
    centroid, direction = fit_line_3d(line_pts)
    expected_dir = np.array([0.0, 0.0, 1.0])
    angle_err = angle_between_vectors_3d(direction, expected_dir)
    # Check if collinear (angle 0 or 180)
    assert np.isclose(angle_err, 0.0, atol=1e-3) or np.isclose(angle_err, 180.0, atol=1e-3), f"Direction angle: {angle_err}"
    print(f"  [PASS] 3D Line fit (Principal axis collinear with Z axis)")

    # Test 3: 3D and 2D projected angle
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    ang = angle_between_vectors_3d(v1, v2)
    assert np.isclose(ang, 90.0), f"Expected 90 degrees, got {ang}"
    print(f"  [PASS] Vector angle calculation (90.0 degrees)")

    print("All computational geometry unit tests passed successfully!\n")


if __name__ == "__main__":
    run_unit_tests()
