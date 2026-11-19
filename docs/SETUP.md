# Setup and reproduction

## Environment

Tested on Linux with an RTX 3060 Laptop GPU (6 GB), 12 CPU cores and 30 GB RAM.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu132
uv pip install -e ".[models,dev]"
pytest
```

The unit tests need no dataset or GPU; they also run in Docker:

```bash
docker build -t femur-study . && docker run --rm femur-study
```

Versions used: Python 3.12.13, torch 2.15.0 (CUDA 13.2), MONAI 1.6.1, nnU-Net v2 2.8.1, TotalSegmentator 2.18.0, SimpleITK 2.5.6, PyVista 0.49.2, VTK 9.7.1.

## Data

Download HFValid v3 (see [DATA_CARD.md](DATA_CARD.md)) and point `FEMUR_DATA` at a folder holding it:

```
$FEMUR_DATA/
  HFValid/Subjects/PatXXX/{PatXXX.nrrd, PatXXX.stl, PatXXX_palp.inp}
  HFValid/HFValid_Metadata.csv
  HFValid/masks/          written by voxelize.py
  nnunet/                 nnU-Net raw, preprocessed and results
  predictions/<method>/   one PatXXX.nii.gz per case
  weights/                MONAI checkpoint
```

```bash
export FEMUR_DATA=/path/to/data
export nnUNet_raw=$FEMUR_DATA/nnunet/nnUNet_raw
export nnUNet_preprocessed=$FEMUR_DATA/nnunet/nnUNet_preprocessed
export nnUNet_results=$FEMUR_DATA/nnunet/nnUNet_results
```

## Steps

```bash
# 1. Audit, masks, split, quality checks
python src/audit.py                     # writes results/data/hfvalid_audit.csv
python src/voxelize.py --workers 4
python src/split_data.py                # writes splits.json
python src/data_quality.py

# 2. nnU-Net: train 71, validate 10 (fold 0 = splits.json), test 20 in imagesTs
python src/convert_nnunet.py
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres -gpu_memory_target 5 -overwrite_plans_name nnUNetPlans_5G
python -c "import json; s=json.load(open('splits.json')); json.dump([{'train': s['train'], 'val': s['val']}], open('$nnUNet_preprocessed/Dataset001_Femur/splits_final.json', 'w'))"
nnUNetv2_train 1 3d_fullres 0 -p nnUNetPlans_5G -tr nnUNetTrainer_100epochs
mkdir -p $FEMUR_DATA/predictions/nnunet
cp $nnUNet_results/Dataset001_Femur/nnUNetTrainer_100epochs__nnUNetPlans_5G__3d_fullres/fold_0/validation/Pat*.nii.gz $FEMUR_DATA/predictions/nnunet/
nnUNetv2_predict -i $nnUNet_raw/Dataset001_Femur/imagesTs -o $FEMUR_DATA/predictions/nnunet \
  -d 1 -c 3d_fullres -p nnUNetPlans_5G -tr nnUNetTrainer_100epochs -f 0

# 3. Other methods
python src/classical_baseline.py --split val && python src/classical_baseline.py --split test
python src/totalseg_pipeline.py --split val && python src/totalseg_pipeline.py --split test
python src/monai_pipeline.py --mode sanity    # overfits two cases as a check
python src/monai_pipeline.py --mode train --hours 2 --epochs 1000
python src/monai_pipeline.py --mode predict --split val && python src/monai_pipeline.py --mode predict --split test

# 4. Scoring
python src/evaluate.py --split val
python src/evaluate.py --split test

# 5. Planning measurements and agreement
python src/planning.py --split all
python src/planning.py --split test
for m in reference_mask nnunet monai totalseg classical; do python src/planning.py --split test --method $m; done
python src/planning_agreement.py --split test --methods reference_mask nnunet monai totalseg classical

# 6. Projection experiment, registration and summary plots
python src/xray_projection.py --split all
python src/registration.py --synthetic 10
python src/registration.py --pairs 10
python src/plot_results.py
```

Outputs go to `results/<topic>/` and `figures/`.
TotalSegmentator downloads its weights on first use; set `TOTALSEG_HOME_DIR` to keep them off the system drive.
