import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import os
from tqdm import tqdm
import torch.nn.functional as F

from efficient_kan import KAN
from piq import psnr, ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# DATASET (giữ nguyên)
# =========================
class VideoDataset(Dataset):
    def __init__(self, root_dir, target_size=(160, 90)):
        self.hr_dir = os.path.join(root_dir, "frames")
        self.lr_dir = os.path.join(root_dir, "lr_frames")
        self.samples = []
        self.hr_size = (target_size[1] * 2, target_size[0] * 2)
        self.lr_size = (target_size[1], target_size[0])

        for clip in sorted(os.listdir(self.hr_dir)):
            if clip.startswith("."): continue
            hr_p = os.path.join(self.hr_dir, clip)
            lr_p = os.path.join(self.lr_dir, clip)
            if os.path.isdir(hr_p):
                for f in sorted(os.listdir(hr_p)):
                    if f.startswith("."): continue
                    self.samples.append((os.path.join(lr_p, f), os.path.join(hr_p, f)))

        self.transform_lr = transforms.Compose([transforms.Resize(self.lr_size), transforms.ToTensor()])
        self.transform_hr = transforms.Compose([transforms.Resize(self.hr_size), transforms.ToTensor()])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        lr = Image.open(self.samples[idx][0]).convert("RGB")
        hr = Image.open(self.samples[idx][1]).convert("RGB")
        return self.transform_lr(lr), self.transform_hr(hr)

# =========================
# RESIDUAL BLOCK (thêm mới)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection

# =========================
# MODEL CẢI TIẾN
# =========================
class KAN_SR_V2(nn.Module):
    def __init__(self, upscale_factor=2, num_res_blocks=8):
        super().__init__()
        self.upscale_factor = upscale_factor

        # ① Feature extraction sâu hơn
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # ② Stack Residual Blocks để học đặc trưng phong phú
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        # ③ KAN: học ánh xạ phi tuyến từ feature → enhanced feature
        #    Tăng grid_size=8 để xấp xỉ hàm phức tạp hơn
        self.kan = KAN([64, 256, 128, 64], grid_size=8)

        # ④ Upsampling bằng PixelShuffle (sub-pixel convolution)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),        # 256 → 64 channels, H×W → 2H×2W
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
        )

    def forward(self, x):
        # Bicubic base (skip connection toàn cục)
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                             mode='bicubic', align_corners=False)

        b, c, h, w = x.shape

        # Feature path
        feat = self.entry(x)                        # [B, 64, H, W]
        res  = self.res_blocks(feat)
        feat = feat + self.post_res(res)            # long skip

        # KAN refinement (pixel-wise)
        flat = feat.permute(0,2,3,1).reshape(-1, 64)
        kan_out = self.kan(flat)                    # [-1, 64]
        feat2 = kan_out.view(b, h, w, 64).permute(0,3,1,2)

        # Upsample residual
        residual = self.upsample(feat2)             # [B, 3, 2H, 2W]

        return torch.clamp(base + residual, 0.0, 1.0)


# =========================
# PERCEPTUAL LOSS (VGG)
# =========================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # Lấy đến relu2_2 (không quá sâu, tránh mất spatial info)
        self.slice = nn.Sequential(*list(vgg.children())[:9]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.slice(pred), self.slice(target))


# =========================
# FREQUENCY LOSS (FFT) - thêm mới
# =========================
def frequency_loss(pred, target):
    """Phạt sự khác biệt trong miền tần số - phục hồi cạnh sắc nét"""
    pred_fft   = torch.fft.fft2(pred.float())
    target_fft = torch.fft.fft2(target.float())
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


# =========================
# TRAINING
# =========================
def train():
    dataset    = VideoDataset("data")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ✅ FIX: dùng đúng class KAN_SR_V2
    model = KAN_SR_V2(upscale_factor=2, num_res_blocks=8).to(device)

    model_path = "kan_upscale_v2.pth"
    if os.path.exists(model_path):
        print(f"♻️ Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Load v1 nếu muốn fine-tune tiếp (optional)
    # model.load_state_dict(torch.load("kan_upscale_v1.pth", ...), strict=False)

    perc_loss_fn = PerceptualLoss().to(device)

    criterion  = nn.L1Loss()
    optimizer  = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

    # ✅ Cosine Annealing: LR giảm dần theo chu kỳ
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')

    NUM_EPOCHS = 20

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = total_psnr = total_ssim = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                output = model(lr_img)

                # ── Tổ hợp loss ──────────────────────────────
                l1   = criterion(output, hr_img)
                perc = perc_loss_fn(output, hr_img)
                freq = frequency_loss(output, hr_img)

                # Trọng số: L1 chính, perceptual và freq bổ trợ
                loss = l1 + 0.01 * perc + 0.05 * freq

            scaler.scale(loss).backward()
            # Gradient clipping tránh exploding gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            with torch.no_grad():
                out_fp32 = output.float().clamp(0, 1)
                hr_fp32  = hr_img.float()
                p = psnr(out_fp32, hr_fp32, data_range=1.0)
                s = ssim(out_fp32, hr_fp32, data_range=1.0)
                total_psnr += p.item()
                total_ssim += s.item()

            loop.set_postfix(loss=f"{loss.item():.4f}",
                             psnr=f"{p.item():.2f}",
                             ssim=f"{s.item():.3f}",
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")

        scheduler.step()

        n = len(dataloader)
        print(f"\n✨ Epoch {epoch+1}: "
              f"Loss {total_loss/n:.4f} | "
              f"PSNR {total_psnr/n:.2f} | "
              f"SSIM {total_ssim/n:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.1e}")

        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train()