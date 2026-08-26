"""Extract 2D axial training/validation slices from the 3D raw dataset.

Given a set of patient folders each containing co-registered `*_CT.nii.gz`,
`*_CBCT{1,2,3}.nii.gz` and `*_MR.nii.gz` volumes (optionally with `*_body.nii.gz`
body masks), this builds the 4-channel stacked `SRC` / `TGT` slice cache used by
the training loop.

Input  (per patient, all co-registered to the CT):
    {pid}_CT.nii.gz          target (mandatory)
    {pid}_body.nii.gz        CT body mask (optional, recommended)
    {pid}_CBCT1.nii.gz       input CBCT #1 (optional)
    {pid}_CBCT2.nii.gz       input CBCT #2 (optional)
    {pid}_CBCT3.nii.gz       input CBCT #3 (optional)
    {pid}_MR.nii.gz          input MRI (optional)

Output (2D cache, 4-channel SRC stack = [CBCT1, CBCT2, CBCT3, MR]):
    <where_is_2d>/SRC/{pid}_s{slice:03d}.nii.gz
    <where_is_2d>/TGT/{pid}_s{slice:03d}.nii.gz
    <where_is_2d>/SRC_val/...   and TGT_val/... (held-out 20% patients)
"""

import os
import glob

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, ResampleToMatchd, Resized,
    ScaleIntensityRanged, MapTransform,
)

from .dataset import MaskBackgroundd, CombineAndPadChannelsd


def _load_patient_items(args):
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
    return data_dicts


def prepare_2d_slices(args):
    data_dicts = _load_patient_items(args)
    np.random.seed(29100)
    np.random.shuffle(data_dicts)
    N_train = int(len(data_dicts) * 0.8)
    train_dic_3d, val_dic_3d = data_dicts[:N_train], data_dicts[N_train:]

    existing_keys = ['CT', 'MR', 'CBCT1', 'CBCT2', 'CBCT3', 'CT_body', 'CBCT1_body', 'CBCT2_body', 'CBCT3_body']

    transforms_3d = Compose([
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
    ])

    for d in ['SRC', 'TGT', 'SRC_val', 'TGT_val']:
        os.makedirs(os.path.join(args.where_is_2d, d), exist_ok=True)

    def _extract(items, src_dir, tgt_dir):
        for item in tqdm(items):
            transformed = transforms_3d(item)
            src_3d, tgt_3d = transformed['SRC'], transformed['TGT']
            mask_3d = transformed.get('CT_body')
            pid = item['CT'].split('/')[-2]
            for s in range(src_3d.shape[-1]):
                if mask_3d is not None:
                    if mask_3d[0, ..., s].sum() < (512 * 512 * 0.01):
                        continue
                elif torch.max(tgt_3d[0, ..., s]) <= -0.99:
                    continue
                src_np = np.transpose(src_3d[..., s].numpy(), (1, 2, 0))
                tgt_np = np.transpose(tgt_3d[..., s].numpy(), (1, 2, 0))
                nib.save(nib.Nifti1Image(src_np, np.eye(4)),
                         os.path.join(src_dir, f'{pid}_s{s:03d}.nii.gz'))
                nib.save(nib.Nifti1Image(tgt_np, np.eye(4)),
                         os.path.join(tgt_dir, f'{pid}_s{s:03d}.nii.gz'))

    print('Extracting training slices (2D tensors)...')
    _extract(train_dic_3d, os.path.join(args.where_is_2d, 'SRC'), os.path.join(args.where_is_2d, 'TGT'))
    print('Extracting validation slices (2D tensors)...')
    _extract(val_dic_3d, os.path.join(args.where_is_2d, 'SRC_val'), os.path.join(args.where_is_2d, 'TGT_val'))
    print('Done.')


if __name__ == '__main__':
    from .config import get_config
    prepare_2d_slices(get_config())