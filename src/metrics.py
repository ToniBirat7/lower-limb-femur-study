"""
Evaluation Metrics for Medical Image Segmentation
Implements clinically standard metrics:
- Dice Similarity Coefficient (DSC)
- 95th Percentile Hausdorff Distance (HD95 in mm)
- Average Surface Distance (ASD in mm)
Includes unit tests on synthetic geometries.
"""

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import SimpleITK as sitk


def compute_dice(pred_mask, gt_mask):
    """
    Computes Dice Similarity Coefficient between two binary masks.
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()

    if total == 0:
        return 1.0 if not np.any(gt_bool) else 0.0
    return 2.0 * float(intersection) / float(total)


def get_surface_points(mask, spacing=(1.0, 1.0, 1.0)):
    """
    Extracts the physical coordinates (in mm) of the surface boundary voxels.
    """
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return np.empty((0, 3), dtype=np.float64)

    # Surface = mask - eroded(mask)
    eroded = binary_erosion(mask_bool)
    border = np.logical_xor(mask_bool, eroded)

    # Border voxel indices (z, y, x)
    indices = np.argwhere(border)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Convert to physical mm: (x, y, z) = (index_x * sx, index_y * sy, index_z * sz)
    # Note: indices are in order (z, y, x) -> swap to (x, y, z)
    coords_mm = np.zeros_like(indices, dtype=np.float64)
    coords_mm[:, 0] = indices[:, 2] * spacing[0]  # X
    coords_mm[:, 1] = indices[:, 1] * spacing[1]  # Y
    coords_mm[:, 2] = indices[:, 0] * spacing[2]  # Z

    return coords_mm


def compute_surface_distances(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Computes directed nearest-neighbor surface distances (mm) from Pred->GT and GT->Pred.
    """
    pts_pred = get_surface_points(pred_mask, spacing)
    pts_gt = get_surface_points(gt_mask, spacing)

    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return None, None

    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    dist_pred_to_gt, _ = tree_gt.query(pts_pred)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt)

    return dist_pred_to_gt, dist_gt_to_pred


def compute_hd95(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Computes the 95th Percentile Hausdorff Distance (HD95 in mm).
    """
    d_p2g, d_g2p = compute_surface_distances(pred_mask, gt_mask, spacing)
    if d_p2g is None or d_g2p is None:
        return np.nan

    all_distances = np.concatenate([d_p2g, d_g2p])
    return float(np.percentile(all_distances, 95))


def compute_asd(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Computes Average Surface Distance (ASD in mm).
    """
    d_p2g, d_g2p = compute_surface_distances(pred_mask, gt_mask, spacing)
    if d_p2g is None or d_g2p is None:
        return np.nan

    all_distances = np.concatenate([d_p2g, d_g2p])
    return float(np.mean(all_distances))


def evaluate_case(pred_sitk, gt_sitk):
    """
    Evaluates Dice, HD95, and ASD between two SimpleITK binary images.
    """
    spacing = pred_sitk.GetSpacing()
    pred_arr = sitk.GetArrayFromImage(pred_sitk)
    gt_arr = sitk.GetArrayFromImage(gt_sitk)

    dsc = compute_dice(pred_arr, gt_arr)
    hd95 = compute_hd95(pred_arr, gt_arr, spacing)
    asd = compute_asd(pred_arr, gt_arr, spacing)

    return {
        "Dice": round(dsc, 4),
        "HD95_mm": round(hd95, 3) if not np.isnan(hd95) else None,
        "ASD_mm": round(asd, 3) if not np.isnan(asd) else None,
    }


def run_unit_tests():
    print("Running metrics unit tests...")
    grid = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Test 1: Identical volumes
    mask1 = np.zeros(grid, dtype=np.uint8)
    mask1[20:44, 20:44, 20:44] = 1
    
    dsc_ident = compute_dice(mask1, mask1)
    hd95_ident = compute_hd95(mask1, mask1, spacing)
    asd_ident = compute_asd(mask1, mask1, spacing)

    assert np.isclose(dsc_ident, 1.0), f"Expected Dice 1.0, got {dsc_ident}"
    assert np.isclose(hd95_ident, 0.0), f"Expected HD95 0.0, got {hd95_ident}"
    assert np.isclose(asd_ident, 0.0), f"Expected ASD 0.0, got {asd_ident}"
    print("  [PASS] Identical volumes (DSC=1.0, HD95=0.0mm, ASD=0.0mm)")

    # Test 2: Shifted volume by 5 mm along Z
    mask2 = np.zeros(grid, dtype=np.uint8)
    mask2[25:49, 20:44, 20:44] = 1

    dsc_shift = compute_dice(mask1, mask2)
    hd95_shift = compute_hd95(mask1, mask2, spacing)
    asd_shift = compute_asd(mask1, mask2, spacing)

    assert dsc_shift < 1.0, "Shifted volume should have Dice < 1.0"
    assert hd95_shift > 0.0, "Shifted volume should have HD95 > 0.0"
    print(f"  [PASS] Shifted volume (DSC={dsc_shift:.3f}, HD95={hd95_shift:.2f}mm, ASD={asd_shift:.2f}mm)")

    # Test 3: Completely disjoint
    mask3 = np.zeros(grid, dtype=np.uint8)
    mask3[0:10, 0:10, 0:10] = 1
    dsc_disjoint = compute_dice(mask1, mask3)
    assert np.isclose(dsc_disjoint, 0.0), f"Expected 0.0 for disjoint, got {dsc_disjoint}"
    print("  [PASS] Disjoint volumes (DSC=0.0)")

    print("All metric unit tests passed successfully!\n")


if __name__ == "__main__":
    run_unit_tests()
