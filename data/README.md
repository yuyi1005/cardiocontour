# Data layout

All dataset classes read from `data/` relative to the repository root, so run the scripts from the repo root. Download each dataset from its official source and place it as shown below.

```
data
├── ACDC
│   └── database
│       ├── training
│       │   ├── patient001
│       │   │   ├── Info.cfg
│       │   │   ├── MANDATORY_CITATION.md
│       │   │   ├── patient001_4d.nii.gz
│       │   │   ├── patient001_frame01.nii.gz
│       │   │   ├── patient001_frame01_gt.nii.gz
│       │   │   ├── patient001_frame12.nii.gz
│       │   │   └── patient001_frame12_gt.nii.gz
│       │   └── ...                  # patient001 - patient100
│       └── testing
│           ├── patient101
│           │   └── ...              # same files as training
│           └── ...                  # patient101 - patient150
├── MnMs
│   ├── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
│   ├── Training
│   │   ├── Labeled
│   │   │   ├── A0S9V9
│   │   │   │   ├── A0S9V9_sa.nii.gz
│   │   │   │   └── A0S9V9_sa_gt.nii.gz
│   │   │   └── ...
│   │   └── Unlabeled
│   │       ├── A2E3W4
│   │       │   └── A2E3W4_sa.nii.gz
│   │       └── ...
│   ├── Validation
│   │   └── A5C2D2/ ...              # same files as Labeled
│   └── Testing
│       └── A1K2P5/ ...              # same files as Labeled
└── MnMs2
    ├── dataset_information.csv
    ├── readme.txt
    └── dataset
        ├── 001
        │   ├── 001_SA_CINE.nii.gz
        │   ├── 001_SA_ED.nii.gz
        │   ├── 001_SA_ED_gt.nii.gz
        │   ├── 001_SA_ES.nii.gz
        │   ├── 001_SA_ES_gt.nii.gz
        │   └── 001_LA_*.nii.gz      # long-axis files, not used
        └── ...                      # 001 - 360
```

Only short-axis data is used. The `_gt` files are label masks (1 = RV, 2 = MYO, 3 = LV).

## How each dataset is split

| Dataset | Train | Validation / test |
|---|---|---|
| ACDC | `training` (100 patients) | `testing` (50 patients). `--split val` and `--split test` both use it. |
| M&Ms | `Training/Labeled`, `Training/Unlabeled` and `Validation` | `Testing` (ED/ES frames only) |
| M&Ms-2 | First 200 cases (001-200) | Last 160 cases (201-360). `--split val` and `--split test` both use them. |

## Labeled and unlabeled samples

A sample is labeled when it has a ground-truth mask, and unlabeled otherwise. Training uses both (see the main README).

| Dataset | Labeled samples | Unlabeled samples |
|---|---|---|
| ACDC | `*_frameXX.nii.gz` with a matching `*_frameXX_gt.nii.gz` (ED and ES) | Every frame of `patientXXX_4d.nii.gz` (training patients only) |
| M&Ms | Annotated frames of `*_sa.nii.gz` in `Training/Labeled` and `Validation` | Every frame of `*_sa.nii.gz` in `Training/Unlabeled`. Any `_sa_gt` file there is ignored. |
| M&Ms-2 | `SA_ED` and `SA_ES` with their `_gt` files (training cases) | Every frame of `SA_CINE` (training cases only) |
