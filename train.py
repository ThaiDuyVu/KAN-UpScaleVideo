import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from efficient_kan import KAN
# Thư viện tính toán chỉ số
from piq import psnr, ssim 

# 1. Cấu hình thiết bị
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using MPS (Metal GPU) on Mac")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

# 2. Dataset (Giữ nguyên logic Resize để tránh lỗi Size)
class VideoDataset(Dataset):
    def __init__(self, root_dir, target_size=(160, 90)):
        self.hr_dir = os.path.join(root_dir, 'frames')
        self.lr_dir = os.path.join(root_dir, 'lr_frames')
        self.samples = []
        self.hr_size = (target_size[1] * 2, target_size[0] * 2) # (H, W) cho PIL
        self.lr_size = (target_size[1], target_size[0])

        for clip in sorted(os.listdir(self.hr_dir)):
            if clip.startswith('.'): continue
            hr_p = os.path.join(self.hr_dir, clip)
            lr_p = os.path.join(self.lr_dir, clip)
            if os.path.isdir(hr_p):
                frames = sorted(os.listdir(hr_p))
                for f in frames:
                    if f.startswith('.'): continue
                    self.samples.append((os.path.join(lr_p, f), os.path.join(hr_p, f)))
        
        self.transform_lr = transforms.Compose([transforms.Resize(self.lr_size), transforms.ToTensor()])
        self.transform_hr = transforms.Compose([transforms.Resize(self.hr_size), transforms.ToTensor()])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        lr_img = self.transform_lr(Image.open(self.samples[idx][0]).convert('RGB'))
        hr_img = self.transform_hr(Image.open(self.samples[idx][1]).convert('RGB'))
        return lr_img, hr_img

# 3. Kiến trúc KAN-SR
class KAN_SR(nn.Module):
    def __init__(self):
        super(KAN_SR, self).__init__()
        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.kan = KAN([32, 64, 32], grid_size=3) 
        self.upsample = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = torch.relu(self.conv_in(x))
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, 32)
        out_kan = self.kan(feat_flat)
        feat_res = out_kan.view(b, h, w, 32).permute(0, 3, 1, 2)
        return self.upsample(feat_res)

# 4. Huấn luyện và Đánh giá
def train():
    dataset = VideoDataset('data')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = KAN_SR().to(device)
    model_path = 'kan_upscale_v1.pth'
    if os.path.exists(model_path):
        print(f"♻️ Loading existing model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        epoch_loss, total_psnr, total_ssim = 0, 0, 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")

        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)
            
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()

            # Tính toán chỉ số (không tính gradient để tiết kiệm bộ nhớ)
            with torch.no_grad():
                # PSNR & SSIM yêu cầu input trong dải [0, 1]
                p_val = psnr(output, hr, data_range=1.0).item()
                s_val = ssim(output, hr, data_range=1.0).item()
                
                total_psnr += p_val
                total_ssim += s_val
                epoch_loss += loss.item()

            loop.set_postfix(Loss=f"{loss.item():.4f}", PSNR=f"{p_val:.2f}", SSIM=f"{s_val:.3f}")

        # Tính trung bình cho cả epoch
        avg_psnr = total_psnr / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        print(f"\n✨ Epoch [{epoch+1}] Summary: Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f}")

        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train()