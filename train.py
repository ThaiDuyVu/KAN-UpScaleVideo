import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from efficient_kan import KAN
from piq import psnr, ssim

# =========================
# 1. DEVICE SETUP (V100 OPTIMIZED)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# 2. DATASET
# =========================
class VideoDataset(Dataset):
    def __init__(self, root_dir, target_size=(160, 90)):
        self.hr_dir = os.path.join(root_dir, "frames")
        self.lr_dir = os.path.join(root_dir, "lr_frames")
        self.samples = []

        self.hr_size = (target_size[1] * 2, target_size[0] * 2)
        self.lr_size = (target_size[1], target_size[0])

        for clip in sorted(os.listdir(self.hr_dir)):
            if clip.startswith("."):
                continue

            hr_p = os.path.join(self.hr_dir, clip)
            lr_p = os.path.join(self.lr_dir, clip)

            if os.path.isdir(hr_p):
                frames = sorted(os.listdir(hr_p))
                for f in frames:
                    if f.startswith("."):
                        continue
                    self.samples.append(
                        (os.path.join(lr_p, f), os.path.join(hr_p, f))
                    )

        self.transform_lr = transforms.Compose([
            transforms.Resize(self.lr_size),
            transforms.ToTensor()
        ])

        self.transform_hr = transforms.Compose([
            transforms.Resize(self.hr_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr = Image.open(self.samples[idx][0]).convert("RGB")
        hr = Image.open(self.samples[idx][1]).convert("RGB")

        return self.transform_lr(lr), self.transform_hr(hr)

# =========================
# 3. MODEL
# =========================
class KAN_SR(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 32, 3, padding=1)
        self.kan = KAN([32, 64, 32], grid_size=3)

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        feat = torch.relu(self.conv_in(x))

        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, 32)

        out = self.kan(feat_flat)

        feat = out.view(b, h, w, 32).permute(0, 3, 1, 2)

        return self.upsample(feat)

# =========================
# 4. TRAINING
# =========================
def train():
    dataset = VideoDataset("data")

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = KAN_SR().to(device)

    model_path = "kan_upscale_v1.pth"
    if os.path.exists(model_path):
        print(f"♻️ Loading model: {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ✅ FIX AMP (NEW API)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(10):
        model.train()

        total_loss = 0
        total_psnr = 0
        total_ssim = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")

        for lr, hr in loop:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad()

            # ✅ FIX AUTOCAST
            with torch.amp.autocast(device_type="cuda"):
                output = model(lr)
                loss = criterion(output, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # =========================
            # ✅ FIX SSIM + PSNR (FP32 ONLY)
            # =========================
            with torch.no_grad():
                out_fp32 = output.float()
                hr_fp32 = hr.float()

                p = psnr(out_fp32, hr_fp32, data_range=1.0)
                s = ssim(out_fp32, hr_fp32, data_range=1.0)

                total_psnr += p.item()
                total_ssim += s.item()

            loop.set_postfix(
                loss=loss.item(),
                psnr=p.item(),
                ssim=s.item()
            )

        print(
            f"\n✨ Epoch {epoch+1}: "
            f"Loss {total_loss/len(dataloader):.4f} | "
            f"PSNR {total_psnr/len(dataloader):.2f} | "
            f"SSIM {total_ssim/len(dataloader):.4f}"
        )

        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train()