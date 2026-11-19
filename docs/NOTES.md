# Notes

Working notes on the choices made in this study and what each problem taught me.
Results are in the [README](../README.md); data details in the [data card](DATA_CARD.md).

## Why this setup

- HFValid has what planning work needs and most public CT sets lack: an expert surface and anatomical landmarks for every femur, so segmentation and measurements can both be scored.
- Reference masks are the STL surfaces voxelised on each CT grid. I checked CT, STL, mask and landmarks together in 3D Slicer before training anything.
- The split is by patient and balanced by side. All choices were made on the 10 validation cases; the 20 test cases were scored once at the end.
- Four segmentation methods cover the range a lab would consider: a threshold baseline, a 3D U-Net I configure myself (MONAI), a self-configuring one (nnU-Net), and a pretrained model used without training (TotalSegmentator).
- Planning measurements are scored on predicted masks against the reference surfaces, because a good Dice is only useful if the measurements taken from the mask hold up.

## Problems and what they taught me

**Reading the landmark file.**
My first neck-shaft angles were about 171 degrees.
I had read the `headDir` landmark as the neck direction; it is a point on the head-to-knee axis that defines the dataset's femur frame.
The neck axis is now found from the bone itself: from the head centre to the narrowest section of the neck.
Lesson: check what each annotation means before building on it.

**Choosing the TotalSegmentator side.**
My first version kept whichever femur (left or right) overlapped the reference mask more, which uses the answer to make a prediction.
The side now comes from the dataset metadata.

**The threshold baseline.**
In older women much of the trabecular bone is below 200 HU, so a threshold gives a hollow shell; hole filling fixes that.
It still merges with the pelvis at the hip and the tibia at the knee, which no threshold can separate.

**NaN losses in MONAI training.**
With float16 mixed precision the loss became NaN on some batches, yet validation Dice kept rising because the gradient scaler silently skipped those steps.
Computing the loss in float32 did not help; the overflow was in the forward pass.
bfloat16 has the range of float32 and fixed it.
Lesson: a rising validation curve does not prove that every training step is healthy.

**nnU-Net on a 6 GB GPU.**
The default plan chose a 112 x 160 x 112 patch; capping GPU memory at 5 GB gave 96 x 128 x 96.
Because the slice spacing (2.0 mm) is 2.7 times the pixel spacing, the planner keeps the first stage in-plane only (1 x 3 x 3 kernels).

**One stray region (Pat027).**
nnU-Net added a separate 22 ml region near the midline on one test scan, which alone pushes its mean HD95 from under 1 mm to 7 mm.
Keeping the largest connected component removes it, but no validation case needed that step, so I report the raw output rather than tune on the test set.

**Meshes from voxel masks.**
Marching cubes on a mask gives a stepped surface, and the neck-section search failed on at least 5 of 20 reference masks.
Taubin smoothing removes the steps without shrinking the head, which plain Laplacian smoothing would do.
Running the reference masks through the same path gives a control row: the error added by the voxel step alone.

**Silent failures in the planning code.**
Two bugs made single femurs fail while the run carried on: a slice lookup that read cell data instead of point data, and a 40 mm guard that rejected the shaft of a femur with a large offset (Pat023).
Both are fixed, and the agreement table reports how many femurs were measured, so a skipped case is visible.

**A Dice of 0.99 and an 11 degree error (Pat058).**
On one femur all three models give near-perfect masks, yet the neck-shaft angle moves 11 degrees and anteversion 33 degrees.
The narrowest-section search jumps to a different section after a small surface change.
Overlap scores do not guarantee stable measurements; the neck axis needs a stability check, such as fitting over several sections.

**Registration.**
Restricting the mutual-information metric to a femur mask made rigid registration diverge in the known-transform test, so the metric runs on the whole CT.
Testing on a known transform first (0.03 mm error) separates bugs in the code from real differences between patients (about 14 mm).

## Next steps

- A stability check for the neck axis, then repeat the agreement study.
- An external test set from another centre and scanner.
- Whole-leg alignment (hip-knee-ankle angle) needs full-leg CT, which HFValid does not have.
- Replace the projected axes with landmarks detected on simulated or real radiographs.
