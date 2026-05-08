import torch
import torch.nn as nn
from efficient_kan import KAN # Thư viện tối ưu hóa KAN cho PyTorch

class KANUpscaler(nn.Module):
    def __init__(self, upscale_factor=2):
        super(KANUpscaler, self).__init__()
        
        # 1. Feature Extraction (CNN)
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        
        # 2. KAN Nonlinear Mapping 
        # Chúng ta dùng KAN để tinh chỉnh đặc trưng. 
        # Vì KAN nhận vector phẳng, ta sẽ áp dụng KAN lên từng "pixel" đặc trưng (1x1 Conv style)
        self.kan_refine = KAN([64, 128, 64]) 
        
        # 3. Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, 3, H, W]
        feat = self.conv_in(x) 
        
        # Chuyển đổi để đưa vào KAN: [B, C, H, W] -> [B*H*W, C]
        b, c, h, w = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Qua KAN
        refined_flat = self.kan_refine(feat_flat)
        
        # Đưa về lại hình dạng ban đầu
        refined = refined_flat.view(b, h, w, c).permute(0, 3, 1, 2)
        
        # Cộng residual (tùy chọn) và Upscale
        out = self.upsample(refined + feat) 
        return out