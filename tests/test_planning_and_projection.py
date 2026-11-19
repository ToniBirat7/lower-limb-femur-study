"""Tests for landmark parsing, projection geometry and the classical baseline."""

import numpy as np
import pyvista as pv
import SimpleITK as sitk

from src.classical_baseline import segment_femur_classical
from src.evaluate import bootstrap_ci
from src.planning import fit_femoral_head, narrowest_section, parse_palp_landmarks
from src.xray_projection import femoral_frame, projected_nsa, rotate_vector_around_axis


def test_parse_palp_landmarks(tmp_path):
    f = tmp_path / "Pat000_palp.inp"
    f.write_text(
        "femurHeadX = -100.5\nfemurHeadY = 25.2\nfemurHeadZ = -50.0\n"
        "headDirX = -105.0\nheadDirY = 20.0\nheadDirZ = -40.0\n"
        "greatTrochX = -150.0\ngreatTrochY = -10.0\ngreatTrochZ = -55.0\n"
        "kneeCenterX = -95.0\nkneeCenterY = 5.0\nkneeCenterZ = -400.0\n"
    )
    lm = parse_palp_landmarks(f)
    assert set(lm) == {"femurHead", "headDir", "greatTroch", "kneeCenter"}
    assert np.allclose(lm["kneeCenter"], [-95.0, 5.0, -400.0])


def test_rotate_vector_around_axis():
    v, z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotate_vector_around_axis(v, z, 90.0), [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(rotate_vector_around_axis(v, z, 180.0), [-1.0, 0.0, 0.0], atol=1e-9)


def test_projected_nsa_equals_3d_when_neck_in_detector_plane():
    nsa = np.radians(130.0)
    shaft = np.array([0.0, 0.0, 1.0])
    neck = np.array([np.sin(nsa), 0.0, np.cos(nsa)])  # in the lateral-up plane
    assert np.isclose(projected_nsa(neck, shaft, 0.0), 130.0)


def test_anteverted_neck_looks_more_valgus_until_rotated_back():
    nsa, ante = np.radians(125.0), np.radians(15.0)
    shaft = np.array([0.0, 0.0, 1.0])
    # neck from head to lateral side tilts posteriorly by the anteversion angle
    neck = np.array([np.sin(nsa) * np.cos(ante), -np.sin(nsa) * np.sin(ante), np.cos(nsa)])
    assert projected_nsa(neck, shaft, 0.0) > 125.5
    assert np.isclose(projected_nsa(neck, shaft, 15.0), 125.0, atol=1e-6)


def test_femoral_frame_is_orthonormal_and_anterior():
    frame = femoral_frame(np.array([0.05, 0.0, 1.0]), np.array([-1.0, 0.1, 0.0]))
    assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-9)
    assert frame[1] @ np.array([0.0, -1.0, 0.0]) > 0  # y points anteriorly (LPS)


def test_head_fit_on_ball_and_stick():
    ball = pv.Sphere(radius=22.0, center=(0.0, 0.0, 0.0), theta_resolution=80, phi_resolution=80)
    stick = pv.Cylinder(center=(0.0, 0.0, -150.0), direction=(0, 0, 1), radius=12.0, height=260.0, resolution=60)
    pts = np.vstack([ball.points, stick.points])
    center, radius = fit_femoral_head(pts, "R")
    assert np.linalg.norm(center) < 0.5
    assert abs(radius - 22.0) < 0.5


def test_narrowest_section_on_waisted_tube():
    # a tube along +x whose radius is smallest at x = 30
    x = np.linspace(0, 60, 61)
    r = 10 + 0.01 * (x - 30) ** 2
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    pts = np.array([[xi, ri * np.cos(ti), ri * np.sin(ti)] for xi, ri in zip(x, r) for ti in t])
    mesh = pv.PolyData(pts).delaunay_3d().extract_surface(algorithm="dataset_surface")
    c = narrowest_section(mesh, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), head_radius=25.0)
    assert c is not None and abs(c[0] - 30.0) < 5.0 and np.linalg.norm(c[1:]) < 1.0


def test_classical_baseline_fills_hollow_bone():
    vol = np.full((40, 40, 40), -100, dtype=np.int16)
    zz, yy, xx = np.mgrid[:40, :40, :40]
    rr = np.hypot(yy - 20, xx - 20)
    vol[(rr < 10) & (zz > 5) & (zz < 35)] = 150  # trabecular bone below the threshold
    vol[(rr >= 7) & (rr < 10) & (zz > 5) & (zz < 35)] = 1200  # cortical shell
    ct = sitk.GetImageFromArray(vol)
    seg = sitk.GetArrayFromImage(segment_femur_classical(ct)).astype(bool)
    truth = (rr < 10) & (zz > 5) & (zz < 35)
    dice = 2 * (seg & truth).sum() / (seg.sum() + truth.sum())
    assert dice > 0.95


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(1).normal(0.9, 0.02, 30)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_planning_agreement_bias_and_limits():
    import pandas as pd

    from src.planning_agreement import agreement

    ref = pd.DataFrame({"subject_id": ["a", "b", "c"], "head_diameter_mm": [44.0, 46.0, 45.0],
                        "nsa_deg": [125.0, 130.0, 120.0], "anteversion_deg": [10.0, 5.0, 0.0], "ama_deg": [8.0, 7.0, 9.0]})
    pred = ref.copy()
    pred["nsa_deg"] += [1.0, 2.0, 3.0]
    out = agreement(ref, pred).set_index("measure")
    assert out.loc["nsa_deg", "bias"] == 2.0
    assert out.loc["nsa_deg", "loa_low"] == round(2.0 - 1.96, 2)
    assert out.loc["head_diameter_mm", "mean_abs_diff"] == 0.0
