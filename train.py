"""Train the LSKA (single-gate or multi-modal) CBCT/MRI -> CT model.

Example (single-gate LSKA, 3 CBCT + 1 MRI -> CT):
    python train.py --model lska \
        --where_is_3d /data/MRXCBCT --where_is_2d /data/MRXCBCT_2D \
        --results_dir /data/RESULTS \
        --experiment NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15 \
        --channels 0 1 2 3

Example (multi-modal LSKA):
    python train.py --model multimodallska \
        --where_is_3d /data/MRXCBCT --where_is_2d /data/MRXCBCT_2D \
        --results_dir /data/RESULTS \
        --experiment NECSR_3_CBCT_1_MRI_to_CT_MultiModalLSKA_e15 \
        --channels 0 1 2 3 --modalities cbct1 cbct2 cbct3 mri
"""

import argparse
from types import SimpleNamespace

import torch

from crossmodal.dataset import get_dataloaders, get_val_dataloader_3d, _stacked_channel_order
from crossmodal.trainer import Trainer
from crossmodal.models import Pix2PixRRDB_LSKA, Pix2PixRRDB_MultiModalLSKA


def main():
    parser = argparse.ArgumentParser(description='Train LSKA CBCT/MRI -> CT')
    parser.add_argument('--model', choices=['lska', 'multimodallska'], default='lska')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment prefix (output folder name), e.g. NECSR_3_CBCT_1_MRI_to_CT_LSKA_e15.')
    parser.add_argument('--channels', type=int, nargs='+', required=True,
                        help='Indices of SRC channels to use (0..3 in [CBCT1,CBCT2,CBCT3,MR]).')
    parser.add_argument('--modalities', type=str, nargs='+', default=None,
                        help='Modality names for MultiModalLSKA (e.g. cbct1 cbct2 cbct3 mri).')
    parser.add_argument('--where_is_3d', type=str, default='/data/MRXCBCT')
    parser.add_argument('--where_is_2d', type=str, default='/data/MRXCBCT_2D')
    parser.add_argument('--results_dir', type=str, default='/data/RESULTS')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--num_rrdb_G', type=int, default=9)
    parser.add_argument('--num_dense_layers_G', type=int, default=2)
    parser.add_argument('--growth_rate_G', type=int, default=32)
    parser.add_argument('--feature_channels_G', type=int, default=64)
    parser.add_argument('--stem_feature_channels_G', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_val_3d', type=int, default=5, help='Number of 3D volumes used for native-space validation.')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Experiment: {args.experiment}')

    # Build model.
    channels = args.channels
    if args.model == 'multimodallska':
        modalities = args.modalities or _stacked_channel_order(channels)
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

    optimizer_G = torch.optim.Adam(model.generator_A_to_B.parameters(), lr=args.learning_rate)
    optimizer_D = torch.optim.Adam(model.discriminator_B.parameters(), lr=args.learning_rate)

    # Runtime config consumed by the loaders and the trainer.
    train_args = SimpleNamespace(
        where_is_3d=args.where_is_3d,
        where_is_2d=args.where_is_2d,
        results_dir=args.results_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        experiment_prefix=args.experiment,
        continue_training=False,
        HU_min=-1000,
        HU_max=3000,
        # Loss weights (defaults matching the shipped models).
        lambda_identity=0, lambda_gan=1, lambda_NGF=100, lambda_GF=0, lambda_GDL=0,
        lambda_sobel=0, alpha_NGF=0.25, lambda_l1=0, lambda_l1_ssim=0, lambda_l1_msssim=100,
    )

    # Loaders (need HU_min/HU_max, so use the config object).
    train_loader, val_loader, fnames_val_A = get_dataloaders(train_args, channel_indices=channels)
    val_loader_3d = get_val_dataloader_3d(train_args, channel_indices=channels, num_samples=args.num_val_3d)

    trainer = Trainer(model, optimizer_G, optimizer_D, device, train_args)
    for epoch in range(1, args.num_epochs + 1):
        trainer.train_epoch(train_loader, epoch)
        trainer.validate(val_loader, epoch, fnames_val_A)
        trainer.validate_3d(val_loader_3d, epoch)


if __name__ == '__main__':
    main()