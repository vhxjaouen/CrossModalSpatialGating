# Cross-Modal Spatial Gating for CBCT / MRI → CT Synthesis

This repository provides the code, model definitions and trained weights for the
**Cross-Modal Spatial Gating** experiments: synthesising a **planning CT** from up
to **three co-registered CBCTs** and a **co-registered MRI** using an RRDB
generator gated by **Large Separable Kernel Attention (LSKA)**.

It is fully self-contained: no external (private) data is distributed, and the
model code has been factored down to exactly what the experiments use (no
unrelated networks are included).

---

## 1. What is included

```
CrossModalSpatialGating/
├── crossmodal/
│   ├── __init__.py
│   ├── models.py         # LSKA + RRDB generators, discriminators, adversarial loss
│   ├── dataset.py        # 2D slice + 3D volume loaders (MONAI)
│   ├── prepare_data.py   # build the 2D slice cache from the 3D raw data
│   ├── config.py         # default hyper-parameters / paths
│   └── trainer.py        # training + 2D/3D validation loops
├── weights/              # trained checkpoints (one folder per experiment)
│   ├── NECSR_1_CBCT_0_MRI_to_CT_LSKA_e15/latest.h5
│   ├── NECSR_1_CBCT_1_MRI_to_CT_LSKA_e15/latest.h5
│   ├── NECSR_3_CBCT_0_MRI_to_CT_LSKA_e15/latest.h5
│   ├── NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15/latest.h5
│   ├── NECSR_1_CBCT_0_MRI_to_CT_MultiModalLSKA_e15/latest.h5
│   ├── NECSR_1_CBCT_1_MRI_to_CT_MultiModalLSKA_e15/latest.h5
│   ├── NECSR_3_CBCT_0_MRI_to_CT_MultiModalLSKA_e15/latest.h5
│   └── NECSR_3_CBCT_1_MRI_to_CT_MultiModalLSKA_e15/latest.h5
├── train.py              # training entry point
├── inference.py          # native-space 3D inference entry point
├── requirements.txt
└── README.md
```

**Two model families are provided:**

| Model class | Description |
|---|---|
| `Pix2PixRRDB_LSKA` | A single LSKA gate applied to the concatenated input channels before a standard RRDB trunk. |
| `Pix2PixRRDB_MultiModalLSKA` | Independent stem per modality, feature fusion, then a **reliability-aware fusion gate** (LSKA spatial + residual SE channel) before the RRDB trunk. |

**Model naming convention (experiment folders):**
`NECSR_{nCBCT}_CBCT_{mMRI}_MRI_to_CT_{variant}_e{epochs}`

* `1_CBCT_0_MRI` → input is `[CBCT1]` only (1 channel).
* `1_CBCT_1_MRI` → input is `[CBCT1, MR]` (2 channels).
* `3_CBCT_0_MRI` → input is `[CBCT1, CBCT2, CBCT3]` (3 channels).
* `3_CBCT_1_MRI` → input is `[CBCT1, CBCT2, CBCT3, MR]` (4 channels).

The `_e15` suffix means the model was trained for 15 epochs.

---

## 2. Environment

```bash
pip install -r requirements.txt
```

Requires a GPU with at least ~16 GB VRAM for the full 4-channel model. Python 3.8+.

---

## 3. Data format

> The original patient data is **private** and is **not** distributed here. Only the
> expected format is documented below so you can prepare your own (or an
> anonymised) dataset.

### 3.1 Raw 3D data (input to `prepare_data.py`)

Place the co-registered volumes in one folder per patient, all **rigidly aligned
to the CT grid**:

```
<where_is_3d>/
├── <pid>/
│   ├── <pid>_CT.nii.gz        # target planning CT  (mandatory)
│   ├── <pid>_body.nii.gz      # CT body mask        (optional, recommended)
│   ├── <pid>_CBCT1.nii.gz     # CBCT #1             (optional)
│   ├── <pid>_CBCT2.nii.gz     # CBCT #2             (optional)
│   ├── <pid>_CBCT3.nii.gz     # CBCT #3             (optional)
│   ├── <pid>_CBCT1_body.nii.gz
│   ├── <pid>_CBCT2_body.nii.gz
│   ├── <pid>_CBCT3_body.nii.gz
│   └── <pid>_MR.nii.gz        # MRI                 (optional)
```

Notes:
* `<pid>` is the patient folder name; the CT filename must equal the folder name
  plus `_CT.nii.gz`. The MRI filename uses the **first token** of `<pid>` before
  `_` (e.g. for folder `BrainMets-CHRU-106_1`, the MRI is `BrainMets-CHRU-106_MR.nii.gz`).
* Missing CBCT/MR channels are padded to `-1` and simply ignored by the selected
  channel subset; any number of the three CBCTs / MRI may be present.
* CBCT and CT values are **HU**, MRI values are non-negative intensities
  (clamped `0..4000`).

### 3.2 Intensity normalisation

Applied automatically by the loaders / `prepare_data.py`:

* **CT & CBCT:** linear scaling from `[HU_min, HU_max] = [-1000, 3000]` to `[-1, 1]`.
* **MRI:** linear scaling from `[0, 4000]` to `[-1, 1]`.
* **Background:** voxels outside the body mask are set to `-1000` (CT/CBCT) or `0` (MR).
* Volumes are resized in-plane to `512×512` (depth kept).

### 3.3 2D slice cache (output of `prepare_data.py`)

The training loop consumes 2D axial slices. After running `prepare_data.py` the
following is produced (80% of patients for training, 20% held out for validation):

```
<where_is_2d>/
├── SRC/{pid}_s{s:03d}.nii.gz      # 4-channel stack [CBCT1, CBCT2, CBCT3, MR]
├── TGT/{pid}_s{s:03d}.nii.gz      # corresponding CT slice
├── SRC_val/...
└── TGT_val/...
```

Each `SRC` file stores a 4-channel image (channels ordered `[CBCT1, CBCT2, CBCT3,
MR]`); the loaders select the subset used by each experiment via the `--channels`
argument (index into this stack).

---

## 4. Preparing the 2D slice cache

```bash
python -m crossmodal.prepare_data --where_is_3d /data/MRXCBCT \
    --where_is_2d /data/MRXCBCT_2D
```

---

## 5. Training

```bash
# Single-gate LSKA, 3 CBCT + MRI -> CT, 15 epochs
python train.py --model lska \
    --where_is_3d /data/MRXCBCT --where_is_2d /data/MRXCBCT_2D \
    --results_dir /data/RESULTS \
    --experiment NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15 \
    --channels 0 1 2 3

# Multi-modal LSKA, 1 CBCT + MRI -> CT
python train.py --model multimodallska \
    --where_is_3d /data/MRXCBCT --where_is_2d /data/MRXCBCT_2D \
    --results_dir /data/RESULTS \
    --experiment NECSR_1_CBCT_1_MRI_to_CT_MultiModalLSKA_e15 \
    --channels 0 3 --modalities cbct mri
```

`--channels` indexes into the 4-channel SRC stack `[CBCT1(0), CBCT2(1), CBCT3(2), MR(3)]`.

**Key hyper-parameters (matching the shipped weights):**

| Parameter | Value |
|---|---|
| Epochs | 15 |
| RRDB blocks (`num_rrdb_G`) | 9 |
| Dense layers per RRDB | 2 |
| Growth rate | 32 |
| Feature channels | 64 |
| Stem feature channels (multi-modal) | 16 |
| `λ_NGF` (normalised gradient loss) | 100 |
| `λ_L1-MSSSIM` | 100 |
| `λ_GAN` | 1 |
| `alpha_NGF` | 0.25 |

During training the best 2D and 3D checkpoints are saved as
`{experiment}_best_e{epoch}_{psnr}dB.h5` and the latest as `latest.h5`.

---

## 6. Inference (native-space 3D)

```bash
python inference.py --model lska \
    --experiment NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15 \
    --weights weights/NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15/latest.h5 \
    --where_is_3d /data/MRXCBCT --results_dir /data/RESULTS \
    --channels 0 1 2 3

python inference.py --model multimodallska \
    --experiment NECSR_3_CBCT_1_MRI_to_CT_MultiModalLSKA_e15 \
    --weights weights/NECSR_3_CBCT_1_MRI_to_CT_MultiModalLSKA_e15/latest.h5 \
    --where_is_3d /data/MRXCBCT --results_dir /data/RESULTS \
    --channels 0 1 2 3 --modalities cbct1 cbct2 cbct3 mri
```

For each patient, inference runs slice-by-slice, re-scales back to true HU and
resamples to the native CT grid, writing:
`<results_dir>/<experiment>/3D_{pid}_inference_native.nii.gz`.

> Make sure `--channels` (and `--modalities`) exactly match the experiment used to
> train the checkpoint you load.

---

## 7. License & data notice

The trained weights embed no patient data; the source patient images are **not**
included and must be obtained separately. Please respect any institutional review
board / data-sharing restrictions that apply to the original dataset.