"""Data loading for CBCT/MRI -> CT synthesis.

Two loaders are provided:

* `get_dataloaders` -- 2D slice training/validation loader (the model is trained
  on 2D axial slices of shape (C, 256, 256)).
* `get_val_dataloader_3d` -- full 3D volumes for native-space clinical evaluation.
"""

import os
import glob

import numpy as np
import torch

from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, RandSpatialCropd,
    MapTransform, ResampleToMatchd, Resized, ScaleIntensityRanged,
)


class SelectChannelsd(MapTransform):
    """Select a subset of the stacked SRC channels (e.g. [0] for CBCT1 only)."""

    def __init__(self, keys, channel_indices):
        super().__init__(keys)
        self.channel_indices = channel_indices

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if self.channel_indices is not None and key == 'SRC':
                d[key] = d[key][self.channel_indices, ...]
        return d


class MaskBackgroundd(MapTransform):
    """Set voxels outside a body mask to a background value."""

    def __init__(self, keys, mask_keys, bg_values, allow_missing_keys=True):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.mask_keys = mask_keys
        self.bg_values = bg_values

    def __call__(self, data):
        d = dict(data)
        for key, mask_key, bg_val in zip(self.keys, self.mask_keys, self.bg_values):
            if key in d and d[key] is not None:
                mask = d.get(mask_key) or d.get('CT_body')
                if mask is not None:
                    d[key] = torch.where(mask > 0, d[key], torch.tensor(bg_val, dtype=d[key].dtype, device=d[key].device))
        return d


class CombineAndPadChannelsd(MapTransform):
    """Stack CBCT1..3 and MR into a single SRC tensor (pad missing channels with -1)."""

    def __init__(self):
        super().__init__(keys=['CT'])

    def __call__(self, data):
        d = dict(data)
        ct = d['CT']
        d['TGT'] = ct

        src_channels = []
        for i in range(1, 4):
            cbct_key = f'CBCT{i}'
            src_channels.append(d[cbct_key] if (cbct_key in d and d[cbct_key] is not None)
                                else torch.full_like(ct, fill_value=-1.0))
        src_channels.append(d['MR'] if ('MR' in d and d['MR'] is not None)
                            else torch.full_like(ct, fill_value=-1.0))
        d['SRC'] = torch.cat(src_channels, dim=0)

        out_dict = {'SRC': d['SRC'], 'TGT': d['TGT']}
        if 'CT_body' in d:
            out_dict['CT_body'] = d['CT_body']
        if d.get('CT_meta_dict') and 'filename_or_obj' in d.get('CT_meta_dict', {}):
            out_dict['filename'] = d['CT_meta_dict']['filename_or_obj']
        elif isinstance(d.get('CT'), str):
            out_dict['filename'] = d['CT']
        else:
            out_dict['filename'] = "unknown"
        return out_dict


def _stacked_channel_order(channel_indices):
    """Return the modality label per SRC channel for MultiModalLSKA models.

    The 2D/3D SRC stack is always ordered as [CBCT1, CBCT2, CBCT3, MR].
    """
    all_mods = ['cbct1', 'cbct2', 'cbct3', 'mri']
    return [all_mods[i] for i in channel_indices]


# ---------------------------------------------------------------------------
# 2D training / validation slices
# ---------------------------------------------------------------------------
def get_dataloaders(args, channel_indices=None):
    transforms_2d = Compose([
        LoadImaged(keys=['SRC', 'TGT'], image_only=True),
        EnsureChannelFirstd(keys=['SRC', 'TGT'], channel_dim=-1),
        SelectChannelsd(keys=['SRC'], channel_indices=channel_indices),
        EnsureTyped(keys=['SRC', 'TGT']),
        RandSpatialCropd(keys=['SRC', 'TGT'], roi_size=(256, 256), random_center=True, random_size=False),
    ])
    transforms_2d_val = Compose([
        LoadImaged(keys=['SRC', 'TGT'], image_only=True),
        EnsureChannelFirstd(keys=['SRC', 'TGT'], channel_dim=-1),
        SelectChannelsd(keys=['SRC'], channel_indices=channel_indices),
        EnsureTyped(keys=['SRC', 'TGT']),
    ])

    fnames_train_A = sorted(glob.glob(os.path.join(args.where_is_2d, 'SRC', '*.nii.gz')))
    fnames_train_B = sorted(glob.glob(os.path.join(args.where_is_2d, 'TGT', '*.nii.gz')))
    train_dic = [{'SRC': a, 'TGT': b} for a, b in zip(fnames_train_A, fnames_train_B)]
    train_loader = DataLoader(Dataset(train_dic, transforms_2d), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    fnames_val_A = sorted(glob.glob(os.path.join(args.where_is_2d, 'SRC_val', '*.nii.gz')))
    fnames_val_B = sorted(glob.glob(os.path.join(args.where_is_2d, 'TGT_val', '*.nii.gz')))
    val_dic = [{'SRC': a, 'TGT': b} for a, b in zip(fnames_val_A, fnames_val_B)]
    val_loader = DataLoader(CacheDataset(val_dic, transforms_2d_val), batch_size=1,
                            shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, fnames_val_A


# ---------------------------------------------------------------------------
# 3D validation volumes (native-space clinical evaluation)
# ---------------------------------------------------------------------------
def get_val_dataloader_3d(args, channel_indices=None, num_samples=5):
    patient_dirs = sorted(glob.glob(os.path.join(args.where_is_3d, '*')))
    data_dicts = []
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        ct_path = os.path.join(pdir, f"{pid}_CT.nii.gz")
        if not os.path.exists(ct_path):
            continue
        item = {'CT': ct_path}
        ct_body = os.path.join(pdir, f"{pid}_body.nii.gz")
        if os.path.exists(ct_body):
            item['CT_body'] = ct_body
        mr_path = os.path.join(pdir, f"{pid.split('_')[0]}_MR.nii.gz")
        if os.path.exists(mr_path):
            item['MR'] = mr_path
        for i in range(1, 4):
            cbct_path = os.path.join(pdir, f"{pid}_CBCT{i}.nii.gz")
            if os.path.exists(cbct_path):
                item[f'CBCT{i}'] = cbct_path
                body_path = os.path.join(pdir, f"{pid}_CBCT{i}_body.nii.gz")
                if os.path.exists(body_path):
                    item[f'CBCT{i}_body'] = body_path
        data_dicts.append(item)

    np.random.seed(29100)
    np.random.shuffle(data_dicts)
    N_train = int(len(data_dicts) * 0.8)
    val_dic_3d = data_dicts[N_train:]
    if num_samples is not None:
        val_dic_3d = val_dic_3d[:num_samples]

    existing_keys = ['CT', 'MR', 'CBCT1', 'CBCT2', 'CBCT3', 'CT_body', 'CBCT1_body', 'CBCT2_body', 'CBCT3_body']

    transforms_3d_val = Compose([
        LoadImaged(keys=existing_keys, allow_missing_keys=True, image_only=False),
        EnsureChannelFirstd(keys=existing_keys, allow_missing_keys=True),
        ResampleToMatchd(keys=['MR', 'CBCT1', 'CBCT2', 'CBCT3'], key_dst='CT',
                         allow_missing_keys=True, mode='bilinear'),
        ResampleToMatchd(keys=['CBCT1_body', 'CBCT2_body', 'CBCT3_body'], key_dst='CT',
                         allow_missing_keys=True, mode='nearest'),
        Resized(keys=['CT', 'MR', 'CBCT1', 'CBCT2', 'CBCT3'], spatial_size=(512, 512, -1),
                allow_missing_keys=True, mode='bilinear'),
        Resized(keys=['CT_body', 'CBCT1_body', 'CBCT2_body', 'CBCT3_body'], spatial_size=(512, 512, -1),
                allow_missing_keys=True, mode='nearest'),
        MaskBackgroundd(keys=['CT', 'CBCT1', 'CBCT2', 'CBCT3', 'MR'],
                        mask_keys=['CT_body', 'CBCT1_body', 'CBCT2_body', 'CBCT3_body', 'CT_body'],
                        bg_values=[-1000] * 4 + [0]),
        ScaleIntensityRanged(keys=['CT', 'CBCT1', 'CBCT2', 'CBCT3'], a_min=args.HU_min,
                             a_max=args.HU_max, b_min=-1, b_max=1, clip=True, allow_missing_keys=True),
        ScaleIntensityRanged(keys=['MR'], a_min=0, a_max=4000, b_min=-1, b_max=1, clip=True, allow_missing_keys=True),
        CombineAndPadChannelsd(),
        SelectChannelsd(keys=['SRC'], channel_indices=channel_indices),
    ])

    val_ds_3d = Dataset(val_dic_3d, transforms_3d_val)
    return DataLoader(val_ds_3d, batch_size=1, shuffle=False, num_workers=0)