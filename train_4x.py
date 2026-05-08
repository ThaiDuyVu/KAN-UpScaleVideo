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

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# PATHS — chỉnh ở đây nếu cần
# =========================
V3_WEIGHTS     = "kan_upscale_v3.pth"          # weights từ V3 đã train
MODEL_4X_PATH  = "kan_upscale_4x.pth"          # lưu sau phase 2
MODEL_4X_FT    = "kan_upscale_4x_ft.pth"       # lưu sau phase 3 (fine-tune)
DATA_ROOT      = "data"

# =========================
# DATASET 4x
# LR : data/lr_frames_4x  (1/4 resolution so với frames gốc)
# HR : data/frames         (resolution gốc)
# =========================
class VideoDataset4x(Dataset):
    def __init__(self, root_dir, hr_size=(160, 90)):
        """
        hr_size = (W, H) của frames gốc
        lr_size = (W//4, H//4)
        """
        self.hr_dir = os.path.join(root_dir, "frames")
        self.lr_dir = os.path.join(root_dir, "lr_frames_4x")
        self.samples = []

        w, h = hr_size
        self.hr_resize = (h, w)          # (H, W) cho transforms.Resize
        self.lr_resize = (h // 4, w // 4)

        for clip in sorted(os.listdir(self.hr_dir)):
            if clip.startswith("."):
                continue
            hr_p = os.path.join(self.hr_dir, clip)
            lr_p = os.path.join(self.lr_dir, clip)
            if os.path.isdir(hr_p):
                for f in sorted(os.listdir(hr_p)):
                    if f.startswith("."):
                        continue
                    self.samples.append(
                        (os.path.join(lr_p, f), os.path.join(hr_p, f))
                    )

        self.transform_lr = transforms.Compose([
            transforms.Resize(self.lr_resize),
            transforms.ToTensor()
        ])
        self.transform_hr = transforms.Compose([
            transforms.Resize(self.hr_resize),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr = Image.open(self.samples[idx][0]).convert("RGB")
        hr = Image.open(self.samples[idx][1]).convert("RGB")
        return self.transform_lr(lr), self.transform_hr(hr)


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


# =========================
# KAN_SR_V3 — giữ nguyên kiến trúc để load weights
# =========================
class KAN_SR_V3(nn.Module):
    def __init__(self, upscale_factor=2, num_res_blocks=6):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )
        self.post_res = nn.Conv2d(64, 64, 3, padding=1)

        # Compress spatial trước KAN: ÷4
        self.pre_kan = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
        )

        self.kan = KAN([128, 256, 128], grid_size=5)

        self.post_kan = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

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

        base = F.interpolate(x, scale_factor=self.upscale_factor,
                             mode='bicubic', align_corners=False)

        feat = self.entry(x)
        feat = feat + self.post_res(self.res_blocks(feat))

        se_w = self.se(feat).view(b, 64, 1, 1)
        feat = feat * se_w

        compressed = self.pre_kan(feat)
        bk, ck, hk, wk = compressed.shape
        flat    = compressed.permute(0, 2, 3, 1).reshape(-1, 128)
        kan_out = self.kan(flat)
        refined = kan_out.view(bk, hk, wk, 128).permute(0, 3, 1, 2)

        expanded = self.post_kan(refined)
        expanded = expanded[:, :, :h, :w]       # fix size mismatch
        fused    = feat + expanded

        residual = self.upsample(fused)
        return torch.clamp(base + residual, 0.0, 1.0)


# =========================
# KAN_SR_4x — 2 stage progressive
# Stage1: LR(H/4, W/4) → Mid(H/2, W/2)
# Stage2: Mid(H/2, W/2) → HR(H,   W  )
# =========================
class KAN_SR_4x(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = KAN_SR_V3(upscale_factor=2, num_res_blocks=6)
        self.stage2 = KAN_SR_V3(upscale_factor=2, num_res_blocks=6)

        # Nhẹ nhàng refine output cuối
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x):
        mid = self.stage1(x)                            # 2x
        out = self.stage2(mid)                          # 4x
        return torch.clamp(out + self.refine(out), 0.0, 1.0)


# =========================
# PERCEPTUAL LOSS (relu1_2 — nhẹ)
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


# =========================
# FREQUENCY LOSS (grayscale FFT)
# =========================
def frequency_loss(pred, target):
    pred_gray   = 0.299*pred[:,0]   + 0.587*pred[:,1]   + 0.114*pred[:,2]
    target_gray = 0.299*target[:,0] + 0.587*target[:,1] + 0.114*target[:,2]
    pred_fft    = torch.fft.rfft2(pred_gray.float())
    target_fft  = torch.fft.rfft2(target_gray.float())
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


# =========================
# HELPER: một epoch train
# =========================
def run_epoch(model, dataloader, optimizer, criterion, perc_loss_fn,
              scaler, desc, scheduler=None):
    model.train()
    total_loss = total_psnr = total_ssim = 0
    loop = tqdm(dataloader, desc=desc)

    for lr_img, hr_img in loop:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            output = model(lr_img)
            l1   = criterion(output, hr_img)
            perc = perc_loss_fn(output, hr_img)
            freq = frequency_loss(output, hr_img)
            loss = l1 + 0.01 * perc + 0.05 * freq

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            p = psnr(output.float().clamp(0, 1), hr_img.float(), data_range=1.0)
            s = ssim(output.float().clamp(0, 1), hr_img.float(), data_range=1.0)
            total_psnr += p.item()
            total_ssim += s.item()

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            psnr=f"{p.item():.2f}",
            ssim=f"{s.item():.3f}",
        )

    if scheduler:
        scheduler.step()

    n = len(dataloader)
    return total_loss / n, total_psnr / n, total_ssim / n


# =========================
# MAIN TRAINING
# =========================
def train_4x():
    # ── Kiểm tra data ──────────────────────────────────────────────
    lr4x_dir = os.path.join(DATA_ROOT, "lr_frames_4x")
    if not os.path.exists(lr4x_dir):
        print("❌ Không tìm thấy data/lr_frames_4x/")
        print("   Hãy chạy prepare_4x_data.py trước!")
        return

    dataset    = VideoDataset4x(DATA_ROOT)
    dataloader = DataLoader(
        dataset, batch_size=12, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    print(f"📦 Dataset: {len(dataset)} samples | {len(dataloader)} batches/epoch")

    # ── Build model ─────────────────────────────────────────────────
    model = KAN_SR_4x().to(device)

    # Load V3 weights vào Stage1
    if os.path.exists(V3_WEIGHTS):
        print(f"♻️  Loading Stage1 từ {V3_WEIGHTS}")
        model.stage1.load_state_dict(
            torch.load(V3_WEIGHTS, map_location=device)
        )
        # Dùng lại V3 weights cho Stage2 (warm start — tốt hơn random)
        print(f"♻️  Warm-start Stage2 từ {V3_WEIGHTS}")
        model.stage2.load_state_dict(
            torch.load(V3_WEIGHTS, map_location=device)
        )
    else:
        print(f"⚠️  Không tìm thấy {V3_WEIGHTS} — train từ đầu (chậm hơn)")

    # Resume 4x nếu đã có
    if os.path.exists(MODEL_4X_PATH):
        print(f"♻️  Resuming từ {MODEL_4X_PATH}")
        model.load_state_dict(torch.load(MODEL_4X_PATH, map_location=device))

    criterion    = nn.L1Loss()
    perc_loss_fn = PerceptualLoss().to(device)
    scaler       = torch.amp.GradScaler("cuda")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Freeze Stage1 — chỉ train Stage2 + refine
    # Stage2 học cách upscale từ mid-resolution lên HR
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🔒 PHASE 2: Freeze Stage1 — Train Stage2 + Refine")
    print("="*60)

    for p in model.stage1.parameters():
        p.requires_grad = False

    optimizer_p2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=1e-4
    )
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=10, eta_min=1e-6
    )

    PHASE2_EPOCHS = 10
    for epoch in range(PHASE2_EPOCHS):
        # Stage1 luôn eval khi bị freeze
        model.stage1.eval()

        loss_avg, psnr_avg, ssim_avg = run_epoch(
            model, dataloader, optimizer_p2, criterion,
            perc_loss_fn, scaler,
            desc=f"[Phase2] Epoch {epoch+1}/{PHASE2_EPOCHS}",
            scheduler=scheduler_p2
        )
        print(f"\n✨ [Phase2] Epoch {epoch+1}: "
              f"Loss {loss_avg:.4f} | PSNR {psnr_avg:.2f} | SSIM {ssim_avg:.4f} | "
              f"LR {scheduler_p2.get_last_lr()[0]:.2e}")
        torch.save(model.state_dict(), MODEL_4X_PATH)

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Unfreeze tất cả — Fine-tune end-to-end với LR nhỏ
    # Cả Stage1 + Stage2 cùng tối ưu cho task 4x
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🔓 PHASE 3: Unfreeze All — End-to-End Fine-tuning")
    print("="*60)

    for p in model.parameters():
        p.requires_grad = True

    optimizer_p3 = optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=1e-4
    )
    scheduler_p3 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p3, T_max=10, eta_min=1e-7
    )

    PHASE3_EPOCHS = 10
    for epoch in range(PHASE3_EPOCHS):
        loss_avg, psnr_avg, ssim_avg = run_epoch(
            model, dataloader, optimizer_p3, criterion,
            perc_loss_fn, scaler,
            desc=f"[Phase3] Epoch {epoch+1}/{PHASE3_EPOCHS}",
            scheduler=scheduler_p3
        )
        print(f"\n✨ [Phase3] Epoch {epoch+1}: "
              f"Loss {loss_avg:.4f} | PSNR {psnr_avg:.2f} | SSIM {ssim_avg:.4f} | "
              f"LR {scheduler_p3.get_last_lr()[0]:.2e}")
        torch.save(model.state_dict(), MODEL_4X_FT)

    print(f"\n🎉 Training xong! Model cuối: {MODEL_4X_FT}")


if __name__ == "__main__":
    train_4x()