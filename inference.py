"""Run native-space 3D inference with a trained LSKA model and its weights.

Example:
    python inference.py --model lska --experiment NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15 \
        --weights weights/NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15/latest.h5 \
        --where_is_3d /data/MRXCBCT --results_dir /data/RESULTS \
        --channels 0 1 2 3

    python inference.py --model multimodallska --experiment NECSR_3_CBCT_1_MRI_to_CT_MultiModalLSKA_e15 \
        --weights weights/NECSR_3_CBCT_1_MRI_to_CT_MultiModalLSKA_e15/latest.h5 \
        --where_is_3d /data/MRXCBCT --results_dir /data/RESULTS \
        --channels 0 1 2 3 --modalities cbct1 cbct2 cbct3 mri

Outputs a native-space synthetic CT `3D_{pid}_inference_native.nii.gz` per patient.
"""

import argparse
import os
from types import SimpleNamespace

import torch
import nibabel as nib
import torch.nn.functional as F

from crossmodal.dataset import get_val_dataloader_3d
from crossmodal.models import Pix2PixRRDB_LSKA, Pix2PixRRDB_MultiModalLSKA


def main():
    parser = argparse.ArgumentParser(description='Inference with LSKA CBCT/MRI -> CT')
    parser.add_argument('--model', choices=['lska', 'multimodallska'], default='lska')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment/output folder name.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the .h5 checkpoint (e.g. latest.h5).')
    parser.add_argument('--channels', type=int, nargs='+', required=True,
                        help='Indices of SRC channels to use (0..3 in [CBCT1,CBCT2,CBCT3,MR]).')
    parser.add_argument('--modalities', type=str, nargs='+', default=None,
                        help='Modality names for MultiModalLSKA.')
    parser.add_argument('--where_is_3d', type=str, default='/data/MRXCBCT')
    parser.add_argument('--results_dir', type=str, default='/data/RESULTS')
    parser.add_argument('--num_rrdb_G', type=int, default=9)
    parser.add_argument('--num_dense_layers_G', type=int, default=2)
    parser.add_argument('--growth_rate_G', type=int, default=32)
    parser.add_argument('--feature_channels_G', type=int, default=64)
    parser.add_argument('--stem_feature_channels_G', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cfg = SimpleNamespace(where_is_3d=args.where_is_3d, results_dir=args.results_dir,
                          HU_min=-1000, HU_max=3000)

    channels = args.channels
    if args.model == 'multimodallska':
        modalities = args.modalities
        model = Pix2PixRRDB_MultiModalLSKA(
            modalities=modalities, out_channels=1,
            num_rrdb_G=args.num_rrdb_G, num_dense_layers_G=args.num_dense_layers_G,
            growth_rate_G=args.growth_rate_G, feature_channels_G=args.feature_channels_G,
            stem_feature_channels_G=args.stem_feature_channels_G,
        ).to(device)
    else:
        model = Pix2PixRRDB_LSKA(
            in_channels=len(channels), out_channels=1,
            num_rrdb_G=args.num_rrdb_G, num_dense_layers_G=args.num_dense_layers_G,
            growth_rate_G=args.growth_rate_G, feature_channels_G=args.feature_channels_G,
        ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    test_loader = get_val_dataloader_3d(cfg, channel_indices=channels, num_samples=None)
    hu_range = cfg.HU_max - cfg.HU_min

    out_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src_3d = batch['SRC'][0].to(device)
            tgt_name = batch.get('filename', [f'case_{i}'])[0]
            pid = os.path.basename(str(tgt_name)).split('_CT')[0].replace('3D_', '')

            D = src_3d.shape[-1]
            pred_scaled = torch.zeros((1, 1, 512, 512, D), device=device)
            def predictor(patch):
                return model(patch, None, is_training=False)[0]
            for d in range(D):
                pred_scaled[..., d] = predictor(src_3d[..., d].unsqueeze(0))

            pred_hu = ((pred_scaled + 1.0) / 2.0) * hu_range + cfg.HU_min

            native_ct = os.path.join(args.where_is_3d, pid, f"{pid}_CT.nii.gz")
            if not os.path.exists(native_ct):
                print(f"  -> WARNING: native `{pid}_CT.nii.gz` not found, skipping.")
                continue
            tgt_nii = nib.load(native_ct)
            pred_native = F.interpolate(pred_hu.float().cpu(), size=tgt_nii.get_fdata().shape,
                                        mode='trilinear', align_corners=False).squeeze().numpy()
            corrected = nib.Nifti1Image(pred_native, affine=tgt_nii.affine, header=tgt_nii.header)
            out_path = os.path.join(out_dir, f"3D_{pid}_inference_native.nii.gz")
            nib.save(corrected, out_path)
            print(f"  -> Saved {out_path}  {tgt_nii.get_fdata().shape}")


if __name__ == '__main__':
    main()