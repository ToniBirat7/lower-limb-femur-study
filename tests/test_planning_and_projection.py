"""Unit tests for surgical planning landmarks and radiographic projection geometry."""

import tempfile
import numpy as np
import pytest
from src.planning import parse_palp_landmarks
from src.xray_projection import rotate_vector_around_axis


def test_parse_palp_landmarks():
    sample_content = """femurHeadX = -100.5
femurHeadY = 25.2
femurHeadZ = -50.0
headDirX = -105.0
headDirY = 20.0
headDirZ = -40.0
greatTrochX = -150.0
greatTrochY = -10.0
greatTrochZ = -55.0
kneeCenterX = -95.0
kneeCenterY = 5.0
kneeCenterZ = -400.0
"""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".inp") as tmp:
        tmp.write(sample_content)
        tmp_path = tmp.name

    try:
        landmarks = parse_palp_landmarks(tmp_path)
        assert "femurHead" in landmarks
        assert "headDir" in landmarks
        assert "greatTroch" in landmarks
        assert "kneeCenter" in landmarks

        assert np.allclose(landmarks["femurHead"], np.array([-100.5, 25.2, -50.0]))
        assert np.allclose(landmarks["kneeCenter"], np.array([-95.0, 5.0, -400.0]))
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_rotate_vector_around_axis():
    v = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])

    # 90 degrees around Z should map [1, 0, 0] to [0, 1, 0]
    v_rot_90 = rotate_vector_around_axis(v, axis, 90.0)
    assert np.allclose(v_rot_90, np.array([0.0, 1.0, 0.0]), atol=1e-5)

    # 180 degrees around Z should map [1, 0, 0] to [-1, 0, 0]
    v_rot_180 = rotate_vector_around_axis(v, axis, 180.0)
    assert np.allclose(v_rot_180, np.array([-1.0, 0.0, 0.0]), atol=1e-5)
