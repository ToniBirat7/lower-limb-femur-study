"""
STL to Binary Voxel Mask Converter for HFValid Dataset
Voxelizes closed surface STL meshes into binary 3D masks exactly aligned
to the reference CT volumes using VTK and SimpleITK.
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import vtk
from vtkmodules.util import numpy_support
import SimpleITK as sitk
from tqdm import tqdm


def voxelize_single_subject(args):
    sdir, out_dir, overwrite = args
    pat_id = sdir.name
    nrrd_path = sdir / f"{pat_id}.nrrd"
    stl_path = sdir / f"{pat_id}.stl"
    out_mask_path = out_dir / f"{pat_id}_mask.nii.gz"

    if out_mask_path.exists() and not overwrite:
        return pat_id, "skipped"

    if not nrrd_path.exists() or not stl_path.exists():
        return pat_id, "missing_files"

    try:
        ref_img = sitk.ReadImage(str(nrrd_path))
        size = ref_img.GetSize()
        spacing = ref_img.GetSpacing()
        origin = ref_img.GetOrigin()
        direction = ref_img.GetDirection()

        # Read STL with VTK
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_path))
        reader.Update()
        polydata = reader.GetOutput()

        # Target extent
        extent = (0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

        # PolyData to Image Stencil
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(origin[0], origin[1], origin[2])
        pol2stenc.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
        pol2stenc.SetOutputWholeExtent(extent)
        pol2stenc.Update()

        # Convert stencil to binary image
        stencil_to_image = vtk.vtkImageStencilToImage()
        stencil_to_image.SetInputConnection(pol2stenc.GetOutputPort())
        stencil_to_image.SetInsideValue(1)
        stencil_to_image.SetOutsideValue(0)
        stencil_to_image.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
        stencil_to_image.Update()

        output_vtk = stencil_to_image.GetOutput()
        dims = output_vtk.GetDimensions()
        scalars = output_vtk.GetPointData().GetScalars()
        np_arr = numpy_support.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])

        mask_sitk = sitk.GetImageFromArray(np_arr)
        mask_sitk.SetSpacing(spacing)
        mask_sitk.SetOrigin(origin)
        mask_sitk.SetDirection(direction)

        sitk.WriteImage(mask_sitk, str(out_mask_path), True)
        return pat_id, "success"
    except Exception as e:
        return pat_id, f"error: {e}"


def run_voxelization(data_dir, out_dir=None, num_workers=4, overwrite=False):
    data_path = Path(data_dir)
    subj_dir = data_path / "Subjects" if (data_path / "Subjects").exists() else data_path
    subject_dirs = sorted([d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith("Pat")])

    if out_dir is None:
        out_path = data_path / "masks"
    else:
        out_path = Path(out_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(subject_dirs)} subjects to voxelize.")
    print(f"Saving masks to: {out_path}")

    tasks = [(sdir, out_path, overwrite) for sdir in subject_dirs]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(voxelize_single_subject, tasks), total=len(tasks)):
            results.append(res)

    success_count = sum(1 for _, status in results if status == "success")
    skip_count = sum(1 for _, status in results if status == "skipped")
    err_count = sum(1 for _, status in results if status.startswith("error"))

    print(f"\nVoxelization finished!")
    print(f"  Success: {success_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Errors: {err_count}")

    if err_count > 0:
        for pat_id, status in results:
            if status.startswith("error"):
                print(f"    {pat_id}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize STL meshes to CT masks")
    parser.add_argument("--data_dir", default="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid", help="HFValid dataset path")
    parser.add_argument("--out_dir", default="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid/masks", help="Output directory for binary masks")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    args = parser.parse_args()

    run_voxelization(args.data_dir, args.out_dir, args.workers, args.overwrite)
