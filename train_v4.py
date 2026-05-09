import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from efficient_kan import KAN
from piq import psnr, ssim

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# DATASET với Data Augmentation
# =========================
class VideoDataset(Dataset):
    def __init__(self, root_dir, target_size=(160, 90), augment=True):
        self.hr_dir  = os.path.join(root_dir, "frames")
        self.lr_dir  = os.path.join(root_dir, "lr_frames")
        self.augment = augment
        self.samples = []

        self.hr_size = (target_size[1] * 2, target_size[0] * 2)   # (H, W)
        self.lr_size = (target_size[1],     target_size[0])

        for clip in sorted(os.listdir(self.hr_dir)):
            if clip.startswith("."): continue
            hr_p = os.path.join(self.hr_dir, clip)
            lr_p = os.path.join(self.lr_dir, clip)
            if os.path.isdir(hr_p):
                for f in sorted(os.listdir(hr_p)):
                    if f.startswith("."): continue
                    self.samples.append(
                        (os.path.join(lr_p, f), os.path.join(hr_p, f))
                    )

        # Resize về đúng size rồi mới augment
        self.to_tensor = transforms.ToTensor()
        self.resize_lr = transforms.Resize(self.lr_size)
        self.resize_hr = transforms.Resize(self.hr_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr = Image.open(self.samples[idx][0]).convert("RGB")
        hr = Image.open(self.samples[idx][1]).convert("RGB")

        lr = self.resize_lr(lr)
        hr = self.resize_hr(hr)

        # Augmentation nhất quán trên cả LR và HR
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if random.random() > 0.5:
                lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotate 90°
            if random.random() > 0.5:
                k = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                lr = lr.transpose(k)
                hr = hr.transpose(k)

        return self.to_tensor(lr), self.to_tensor(hr)


# =========================
# BUILDING BLOCKS
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
    def forward(self, x):
        return x + self.block(x)


# Channel Attention (Squeeze-and-Excitation)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.se(x).view(b, c, 1, 1)


# Dense Residual Block — mỗi layer nhận input từ tất cả layer trước
class DenseResBlock(nn.Module):
    def __init__(self, channels=64, growth=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = channels
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, growth, 3, padding=1),
                nn.PReLU(),
            ))
            in_ch += growth
        # Project về channels gốc
        self.project = nn.Conv2d(in_ch, channels, 1)
        self.se      = SEBlock(channels)

    def forward(self, x):
        feat = x
        for layer in self.layers:
            out  = layer(feat)
            feat = torch.cat([feat, out], dim=1)   # dense connection
        feat = self.project(feat)
        feat = self.se(feat)
        return x + feat * 0.2   # residual scaling tránh exploding


# =========================
# SPATIAL ATTENTION
# =========================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x,  dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


# =========================
# KAN_SR_V4
# Cải tiến chính:
#   - Dense Residual Blocks thay ResBlocks thường
#   - Spatial Attention sau Dense blocks
#   - KAN ở 2 scale (full + compressed) → fusion
#   - Sub-pixel + ICNR init (tránh checkerboard artifact)
#   - Loss = L1 + SSIM + Perceptual + Frequency
# =========================
class KAN_SR_V4(nn.Module):
    def __init__(self, upscale_factor=2, num_dense_blocks=6):
        super().__init__()
        self.upscale_factor = upscale_factor
        C = 64

        # ① Entry
        self.entry = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # ② Dense Residual Blocks
        self.dense_blocks = nn.Sequential(
            *[DenseResBlock(C, growth=32, num_layers=4)
              for _ in range(num_dense_blocks)]
        )
        self.post_dense = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C),
        )

        # ③ Spatial Attention
        self.spatial_attn = SpatialAttention()

        # ④-A KAN branch tại full spatial (pixel-level detail)
        #     Chỉ dùng 32 channels để nhẹ
        self.pre_kan_full = nn.Conv2d(C, 32, 1)
        self.kan_full     = KAN([32, 64, 32], grid_size=5)
        self.post_kan_full = nn.Conv2d(32, C, 1)

        # ④-B KAN branch tại compressed spatial (global context)
        self.pre_kan_comp = nn.Sequential(
            nn.Conv2d(C, 128, 3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.PReLU(),
        )
        self.kan_comp = KAN([128, 256, 128], grid_size=5)
        self.post_kan_comp = nn.Sequential(
            nn.Conv2d(128, C, 1),
            nn.PReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(C, C, 3, padding=1),
        )

        # ④-C Fusion gate: học tỷ lệ trộn 2 KAN branch
        self.fusion = nn.Sequential(
            nn.Conv2d(C * 2, C, 1),
            nn.PReLU(),
            nn.Conv2d(C, C, 3, padding=1),
        )

        # ⑤ Upsample với ICNR init
        self.upsample = nn.Sequential(
            nn.Conv2d(C, C * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
            nn.Conv2d(C, 3, 3, padding=1),
        )
        self._icnr_init()

    def _icnr_init(self):
        """ICNR initialization cho PixelShuffle — giảm checkerboard artifact"""
        conv = self.upsample[0]
        r = self.upscale_factor
        c_out, c_in, kh, kw = conv.weight.shape
        c_out_sub = c_out // (r * r)
        w = torch.zeros(c_out_sub, c_in, kh, kw)
        nn.init.kaiming_normal_(w)
        w = w.repeat(r * r, 1, 1, 1)
        conv.weight.data.copy_(w)

    def forward(self, x):
        b, c, h, w = x.shape

        # Global skip
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                             mode='bicubic', align_corners=False)

        # Feature extraction
        feat = self.entry(x)
        feat = feat + self.post_dense(self.dense_blocks(feat))
        feat = self.spatial_attn(feat)

        # KAN branch A — full spatial, pixel-level
        fa   = self.pre_kan_full(feat)              # [B, 32, H, W]
        flat = fa.permute(0,2,3,1).reshape(-1, 32)
        fa   = self.kan_full(flat).view(b, h, w, 32).permute(0,3,1,2)
        fa   = self.post_kan_full(fa)               # [B, C, H, W]

        # KAN branch B — compressed spatial, global context
        comp = self.pre_kan_comp(feat)              # [B, 128, H/4, W/4]
        bk, ck, hk, wk = comp.shape
        flat_c = comp.permute(0,2,3,1).reshape(-1, 128)
        fb     = self.kan_comp(flat_c).view(bk, hk, wk, 128).permute(0,3,1,2)
        fb     = self.post_kan_comp(fb)             # [B, C, H*4, W*4] → crop
        fb     = fb[:, :, :h, :w]                   # fix size mismatch

        # Fusion
        fused = self.fusion(torch.cat([fa, fb], dim=1))
        out   = feat + fused

        # Upsample
        res = self.upsample(out)
        return torch.clamp(base + res, 0.0, 1.0)


# =========================
# LOSSES
# =========================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:4]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.slice(pred), self.slice(target))


def frequency_loss(pred, target):
    g = lambda t: 0.299*t[:,0] + 0.587*t[:,1] + 0.114*t[:,2]
    return F.l1_loss(
        torch.abs(torch.fft.rfft2(g(pred).float())),
        torch.abs(torch.fft.rfft2(g(target).float()))
    )


# SSIM Loss — tối ưu trực tiếp SSIM metric
def ssim_loss(pred, target):
    return 1.0 - ssim(pred.float().clamp(0,1),
                      target.float(), data_range=1.0)


# =========================
# MIXED PRECISION LOSS WEIGHTS
# Tối ưu cho PSNR cao:
#   L1 chiếm phần lớn (PSNR = -10*log10(MSE) → cần pixel accuracy)
#   SSIM bổ trợ cấu trúc
#   Perceptual + Freq giữ nhỏ (chúng trade-off với PSNR)
# =========================
LOSS_WEIGHTS = {
    "l1":   1.0,
    "ssim": 0.1,    # tăng so với V3
    "perc": 0.005,  # giảm so với V3 (perceptual trade-off với PSNR)
    "freq": 0.01,   # giảm nhẹ
}


# =========================
# TRAINING
# =========================
def train():
    dataset = VideoDataset("data", augment=True)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    print(f"📦 {len(dataset)} samples | {len(dataloader)} batches/epoch")

    model = KAN_SR_V4(upscale_factor=2, num_dense_blocks=6).to(device)

    # Warm-start từ V3 nếu có (transfer learning)
    v3_path = "kan_upscale_v3.pth"
    model_path = "kan_upscale_v4.pth"

    if os.path.exists(model_path):
        print(f"♻️  Resuming từ {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif os.path.exists(v3_path):
        print(f"♻️  Transfer learning từ V3 (partial weights)...")
        v3_state  = torch.load(v3_path, map_location=device)
        v4_state  = model.state_dict()
        # Chỉ load các key khớp (entry, res_blocks → không còn nhưng KAN khớp)
        matched = {k: v for k, v in v3_state.items()
                   if k in v4_state and v4_state[k].shape == v.shape}
        v4_state.update(matched)
        model.load_state_dict(v4_state)
        print(f"   Loaded {len(matched)}/{len(v4_state)} layers từ V3")

    criterion    = nn.L1Loss()
    perc_loss_fn = PerceptualLoss().to(device)
    scaler       = torch.amp.GradScaler("cuda")

    # AdamW + OneCycleLR: tốt nhất để đạt PSNR cao
    NUM_EPOCHS = 30
    optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,          # 10% warmup
        anneal_strategy='cos',
        div_factor=10,          # start lr = max_lr/10
        final_div_factor=1000,  # end lr = max_lr/1000
    )

    best_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = total_psnr = total_ssim = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                output = model(lr_img)

                l1   = criterion(output, hr_img)
                s    = ssim_loss(output, hr_img)
                perc = perc_loss_fn(output, hr_img)
                freq = frequency_loss(output, hr_img)

                loss = (LOSS_WEIGHTS["l1"]   * l1 +
                        LOSS_WEIGHTS["ssim"] * s  +
                        LOSS_WEIGHTS["perc"] * perc +
                        LOSS_WEIGHTS["freq"] * freq)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            with torch.no_grad():
                p = psnr(output.float().clamp(0,1), hr_img.float(), data_range=1.0)
                sv = ssim(output.float().clamp(0,1), hr_img.float(), data_range=1.0)
                total_psnr += p.item()
                total_ssim += sv.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{p.item():.2f}",
                ssim=f"{sv.item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )

        n = len(dataloader)
        avg_psnr = total_psnr / n
        avg_ssim = total_ssim / n
        print(f"\n✨ Epoch {epoch+1}: "
              f"Loss {total_loss/n:.4f} | "
              f"PSNR {avg_psnr:.2f} | "
              f"SSIM {avg_ssim:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

        # Lưu model tốt nhất theo PSNR
        torch.save(model.state_dict(), model_path)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), "kan_upscale_v4_best.pth")
            print(f"   🏆 Best PSNR: {best_psnr:.2f} — saved v4_best.pth")

    print(f"\n🎉 Training xong! Best PSNR đạt: {best_psnr:.2f} dB")


if __name__ == "__main__":
    train()