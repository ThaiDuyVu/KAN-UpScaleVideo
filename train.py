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
# DATASET
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
# RESIDUAL BLOCK (nhẹ hơn - bỏ BatchNorm → GroupNorm)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),   # GroupNorm nhẹ hơn BatchNorm
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
    def forward(self, x):
        return x + self.block(x)

# =========================
# MODEL TỐI ƯU: KAN_SR_V3
# KEY IDEA: KAN chỉ hoạt động trên feature map đã được
# spatial pooling (H/4 × W/4) thay vì full resolution
# → giảm ~16x số vectors đưa vào KAN
# =========================
class KAN_SR_V3(nn.Module):
    def __init__(self, upscale_factor=2, num_res_blocks=6):
        super().__init__()
        self.upscale_factor = upscale_factor

        # ① Shallow feature extraction
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # ② Residual blocks (giảm xuống 6 thay vì 8)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.post_res   = nn.Conv2d(64, 64, 3, padding=1)

        # ③ Compress spatial trước KAN: 90×160 → 23×40 (÷4)
        #    KAN xử lý ~920 vectors thay vì 14,400 vectors/sample
        self.pre_kan = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # /2
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # /2
            nn.PReLU(),
        )

        # ④ KAN hoạt động ở không gian nén — grid_size=5 đủ dùng
        self.kan = KAN([128, 256, 128], grid_size=5)

        # ⑤ Expand trở lại sau KAN
        self.post_kan = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        # ⑥ Upsample × upscale_factor
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

        # ⑦ Channel attention (SE block) — học xem kênh nào quan trọng
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Global skip: bicubic
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                             mode='bicubic', align_corners=False)

        # Feature extraction
        feat = self.entry(x)
        feat = feat + self.post_res(self.res_blocks(feat))  # residual learning

        # SE channel attention
        se_w = self.se(feat).view(b, 64, 1, 1)
        feat = feat * se_w

        # KAN trên spatial nén
        compressed = self.pre_kan(feat)          # [B, 128, H/4, W/4]
        bk, ck, hk, wk = compressed.shape
        flat = compressed.permute(0,2,3,1).reshape(-1, 128)
        kan_out = self.kan(flat)
        refined = kan_out.view(bk, hk, wk, 128).permute(0,3,1,2)

        # Expand + merge
        expanded = self.post_kan(refined)        # [B, 64, H, W]
        expanded = expanded[:, :, :h, :w]
        fused = feat + expanded                  # skip connection

        # Upsample residual
        residual = self.upsample(fused)          # [B, 3, 2H, 2W]

        return torch.clamp(base + residual, 0.0, 1.0)


# =========================
# PERCEPTUAL LOSS (nhẹ: chỉ relu1_2 thay vì relu2_2)
# =========================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:4]).eval()  # chỉ lấy relu1_2
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.slice(pred), self.slice(target))


# =========================
# FREQUENCY LOSS (chỉ tính trên 1 channel để nhẹ hơn)
# =========================
def frequency_loss(pred, target):
    # Chuyển sang grayscale trước FFT → 3x nhẹ hơn
    pred_gray   = 0.299*pred[:,0] + 0.587*pred[:,1] + 0.114*pred[:,2]
    target_gray = 0.299*target[:,0] + 0.587*target[:,1] + 0.114*target[:,2]
    pred_fft    = torch.fft.rfft2(pred_gray.float())
    target_fft  = torch.fft.rfft2(target_gray.float())
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


# =========================
# TRAINING
# =========================
def train():
    dataset    = VideoDataset("data")
    dataloader = DataLoader(
        dataset, batch_size=16,     # tăng batch size vì model nhẹ hơn
        shuffle=True, num_workers=4, pin_memory=True,
        prefetch_factor=2           # prefetch để GPU không bị idle
    )

    model = KAN_SR_V3(upscale_factor=2, num_res_blocks=6).to(device)

    model_path = "kan_upscale_v3.pth"
    if os.path.exists(model_path):
        print(f"♻️ Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    perc_loss_fn = PerceptualLoss().to(device)
    criterion    = nn.L1Loss()
    optimizer    = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # Warm-up 2 epoch rồi cosine decay
    def lr_lambda(epoch):
        if epoch < 2: return epoch / 2       # warm up
        return 0.5 * (1 + torch.cos(torch.tensor((epoch-2) / 18 * 3.14159)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')

    NUM_EPOCHS = 20
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = total_psnr = total_ssim = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)   # nhanh hơn zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                output = model(lr_img)
                l1     = criterion(output, hr_img)
                perc   = perc_loss_fn(output, hr_img)
                freq   = frequency_loss(output, hr_img)
                loss   = l1 + 0.01 * perc + 0.05 * freq

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{p.item():.2f}",
                ssim=f"{s.item():.3f}",
            )

        scheduler.step()
        n = len(dataloader)
        print(f"\n✨ Epoch {epoch+1}: "
              f"Loss {total_loss/n:.4f} | "
              f"PSNR {total_psnr/n:.2f} | "
              f"SSIM {total_ssim/n:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train()