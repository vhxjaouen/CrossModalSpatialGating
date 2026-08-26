"""Training / evaluation logic for the LSKA CBCT/MRI -> CT models."""

import os

import numpy as np
import torch
import imageio.v2 as imageio
import nibabel as nib
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from monai.inferers import sliding_window_inference


class Trainer:
    def __init__(self, model, optimizer_G, optimizer_D, device, args):
        self.model = model
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.device = device
        self.args = args
        self.weights_dir = os.path.join(args.results_dir, args.experiment_prefix)
        os.makedirs(self.weights_dir, exist_ok=True)
        self.best_psnr = 0.0
        self.best_mae = 1e5

    # -- prediction helpers ------------------------------------------------
    def predict_val(self, input_image, target_image=None, patch_size=(256, 256),
                    batch_size=4, overlap=0.25):
        self.model.eval()
        with torch.no_grad():
            def predictor(patch):
                tgt = target_image.expand(patch.shape[0], -1, -1, -1) if target_image is not None else None
                return self.model(patch, tgt, is_training=False)[0]
            return sliding_window_inference(
                inputs=input_image, roi_size=patch_size, sw_batch_size=batch_size,
                predictor=predictor, overlap=overlap, mode='gaussian',
            )

    def _hu(self, x):
        hu_range = self.args.HU_max - self.args.HU_min
        return ((x + 1.0) / 2.0) * hu_range + self.args.HU_min

    # -- training ----------------------------------------------------------
    def train_epoch(self, loader, epoch):
        self.model.train()
        loop = tqdm(loader, desc=f'Epoch [{epoch}/{self.args.num_epochs}]')
        for batch_idx, batch in enumerate(loop):
            real_A = batch['SRC'].float().to(self.device)
            real_B = batch['TGT'].float().to(self.device)
            fake_B, identity_B, pred_fake_B = self.model(real_A, real_B, is_training=True)

            generator_loss = 0.0
            if self.args.lambda_l1:
                generator_loss += self.args.lambda_l1 * self.model.compute_l1_loss(fake_B, real_B)
            if self.args.lambda_l1_ssim:
                generator_loss += self.args.lambda_l1_ssim * self.model.compute_l1_ssim_loss(fake_B, real_B)
            if self.args.lambda_NGF:
                generator_loss += self.args.lambda_NGF * self.model.compute_NGF_loss(fake_B, real_A[:, 0:1, ...], self.args.alpha_NGF)
            if self.args.lambda_GF:
                generator_loss += self.args.lambda_GF * self.model.compute_GF_loss(fake_B, real_B)
            if self.args.lambda_GDL:
                generator_loss += self.args.lambda_GDL * self.model.compute_NGF_loss(fake_B, real_B, self.args.alpha_NGF)
            if self.args.lambda_gan:
                generator_loss += self.args.lambda_gan * self.model.compute_adv_loss(pred_fake_B)
            if self.args.lambda_identity:
                generator_loss += self.args.lambda_identity * self.model.compute_identity_loss(real_B)
            if self.args.lambda_l1_msssim:
                generator_loss += self.args.lambda_l1_msssim * self.model.compute_l1_mssim_loss(fake_B, real_B)
            if self.args.lambda_sobel:
                generator_loss += self.args.lambda_sobel * self.model.compute_sobel_loss(fake_B, real_B)

            discriminator_loss = self.model.compute_discriminator_loss(real_B, fake_B)

            self.optimizer_G.zero_grad()
            generator_loss.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            discriminator_loss.backward()
            self.optimizer_D.step()

    # -- 2D validation ------------------------------------------------------
    def validate(self, val_loader, epoch, fnames_val_A):
        current_psnrs, current_MAEs = [], []
        for i, real_val in enumerate(val_loader):
            real_A_val = real_val['SRC'].float().to(self.device)
            real_B_val = real_val['TGT'].float().to(self.device)
            fake_B_val = self.predict_val(real_A_val, real_B_val)

            fake_0255 = np.uint8(np.clip(255 * (fake_B_val[0, 0].cpu().numpy().squeeze() + 1) / 2, 0, 255))
            real_0255 = np.uint8(np.clip(255 * (real_B_val[0, 0].cpu().numpy().squeeze() + 1) / 2, 0, 255))
            current_psnrs.append(psnr(fake_0255, np.float32(real_0255), data_range=255))
            current_MAEs.append(mean_absolute_error(fake_0255, real_0255))

            fprefix = os.path.basename(fnames_val_A[i]).split('.')[0]
            imageio.imwrite(os.path.join(self.weights_dir, f'{fprefix}_fakeCT_e{epoch:03d}.png'), fake_0255)

        avg_psnr = np.mean(current_psnrs)
        avg_mae = np.mean(current_MAEs)
        print(f'Epoch {epoch} | 2D PSNR = {avg_psnr:.2f} | 2D MAE = {avg_mae:.2f}')

        if avg_psnr > self.best_psnr or avg_mae < self.best_mae:
            self.best_psnr = avg_psnr
            self.best_mae = avg_mae
            best_fname = os.path.join(self.weights_dir,
                                      f'{self.args.experiment_prefix}_best_e{epoch:04d}_{avg_psnr:.2f}dB.h5')
            torch.save({'model': self.model.state_dict()}, best_fname)

        torch.save({'model': self.model.state_dict()}, os.path.join(self.weights_dir, 'latest.h5'))

    # -- 3D native-space validation ----------------------------------------
    def validate_3d(self, val_loader_3d, epoch):
        self.model.eval()
        current_psnrs, current_MAEs = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader_3d):
                src_3d = batch['SRC'][0].to(self.device)
                tgt_3d = batch['TGT'][0].to(self.device)
                mask_3d = batch.get('CT_body', None)
                if mask_3d is not None:
                    mask_3d = mask_3d[0].to(self.device)

                D = src_3d.shape[-1]
                pred_3d = torch.zeros_like(tgt_3d)
                def predictor(patch):
                    return self.model(patch, None, is_training=False)[0]
                for d in range(D):
                    pred_3d[..., d] = predictor(src_3d[..., d].unsqueeze(0)).squeeze(0)

                hu_range = self.args.HU_max - self.args.HU_min
                pred_hu = self._hu(pred_3d).clamp(min=self.args.HU_min, max=self.args.HU_max)
                tgt_hu = self._hu(tgt_3d)

                if mask_3d is not None:
                    valid_mask = mask_3d > 0
                    for key in ['CBCT1_body', 'CBCT2_body', 'CBCT3_body', 'MR_body']:
                        if key in batch:
                            valid_mask = valid_mask & (batch[key][0].to(self.device) > 0)
                else:
                    valid_mask = tgt_hu > (self.args.HU_min + 1)

                if valid_mask.sum() == 0:
                    continue
                true_mae = torch.abs(pred_hu[valid_mask] - tgt_hu[valid_mask]).mean().item()
                mse = torch.nn.functional.mse_loss(pred_hu[valid_mask], tgt_hu[valid_mask])
                true_psnr = 10 * torch.log10(torch.tensor(hu_range ** 2, dtype=torch.float32,
                                                          device=self.device) / mse).item()
                current_psnrs.append(true_psnr)
                current_MAEs.append(true_mae)

                mid = D // 2
                pred_mid_0255 = np.uint8(np.clip(255 * (pred_3d[0, :, :, mid].cpu().numpy() + 1) / 2, 0, 255))
                pname = os.path.basename(str(batch.get('filename', [f'case_{i}'])[0])).split('.')[0]
                imageio.imwrite(os.path.join(self.weights_dir, f'3D_{pname}_fakeCT_e{epoch:03d}.png'), pred_mid_0255)

                pred_hu_np = pred_hu.squeeze(0).cpu().numpy()
                nib.save(nib.Nifti1Image(pred_hu_np, affine=np.eye(4)),
                         os.path.join(self.weights_dir, f'3D_{pname}_fakeCT_e{epoch:03d}.nii.gz'))

        avg_psnr = np.mean(current_psnrs)
        avg_mae = np.mean(current_MAEs)
        print(f'-> Epoch {epoch} 3D True PSNR = {avg_psnr:.2f} dB | 3D True MAE = {avg_mae:.2f} HU')

        if avg_psnr > self.best_psnr or avg_mae < self.best_mae:
            self.best_psnr = avg_psnr
            self.best_mae = avg_mae
            best_fname = os.path.join(self.weights_dir,
                                      f'{self.args.experiment_prefix}_best_e{epoch:04d}_3D_{avg_psnr:.2f}dB.h5')
            torch.save({'model': self.model.state_dict()}, best_fname)

        torch.save({'model': self.model.state_dict()}, os.path.join(self.weights_dir, 'latest.h5'))