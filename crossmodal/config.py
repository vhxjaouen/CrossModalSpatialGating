import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description="CrossModalSpatialGating (LSKA) CBCT/MRI -> CT Training Configuration")

    # Paths
    parser.add_argument('--where_is_3d', type=str, default='/data/BrainMets/MRXCBCT',
                        help='Path to the 3D raw dataset (one subfolder per patient).')
    parser.add_argument('--where_is_2d', type=str, default='/data/BrainMets/MRXCBCT_2D_4CH',
                        help='Path where pre-extracted 2D slices are saved/loaded.')
    parser.add_argument('--results_dir', type=str, default='/data/BrainMets/RESULTS',
                        help='Path to save results and weights.')

    # Model architecture
    parser.add_argument('--use_se', action='store_true', help='Unused for LSKA models (kept for CLI compatibility).')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels (e.g. CBCT1+2+3+MR).')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels (e.g. CT).')
    parser.add_argument('--num_rrdb_G', type=int, default=9, help='Number of RRDB blocks.')
    parser.add_argument('--num_dense_layers_G', type=int, default=2, help='Number of dense layers in each RRDB.')
    parser.add_argument('--growth_rate_G', type=int, default=32, help='Growth rate of the dense blocks.')
    parser.add_argument('--feature_channels_G', type=int, default=64, help='Base feature channels of the generator.')
    parser.add_argument('--stem_feature_channels_G', type=int, default=16,
                        help='Feature channels of each independent modality stem (MultiModalLSKA only).')

    # Training params
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--resume_weights', type=str, default='', help='Path to weights file if continuing.')

    # Losses
    parser.add_argument('--lambda_identity', type=float, default=0)
    parser.add_argument('--lambda_gan', type=float, default=1)
    parser.add_argument('--lambda_NGF', type=float, default=100)
    parser.add_argument('--lambda_GF', type=float, default=0)
    parser.add_argument('--lambda_GDL', type=float, default=0)
    parser.add_argument('--lambda_sobel', type=float, default=0)
    parser.add_argument('--alpha_NGF', type=float, default=0.25)
    parser.add_argument('--lambda_l1', type=float, default=0)
    parser.add_argument('--lambda_l1_ssim', type=float, default=0)
    parser.add_argument('--lambda_l1_msssim', type=float, default=100)

    # Data limits
    parser.add_argument('--HU_min', type=float, default=-1000)
    parser.add_argument('--HU_max', type=float, default=3000)

    args = parser.parse_args()

    # Generated experiment prefix (override in train.py for the LSKA experiments).
    args.experiment_prefix = (
        f'CrossModalSpatialGating'
        f'_{args.in_channels}CH_to_{args.out_channels}CH'
        f'_HU{abs(int(args.HU_min))}to{int(args.HU_max)}'
        f'_ep{args.num_epochs}'
    )
    return args