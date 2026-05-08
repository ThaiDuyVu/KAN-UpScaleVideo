import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VideoKANData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.hr_dir = os.path.join(root_dir, 'frames')
        self.lr_dir = os.path.join(root_dir, 'lr_frames')
        self.clips = os.listdir(self.hr_dir)
        self.transform = transform
        
        # Tạo danh sách các cặp (path_lr, path_hr)
        self.all_frames = []
        for clip in self.clips:
            frames = sorted(os.listdir(os.path.join(self.hr_dir, clip)))
            for frame in frames:
                self.all_frames.append((
                    os.path.join(self.lr_dir, clip, frame),
                    os.path.join(self.hr_dir, clip, frame)
                ))

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        lr_path, hr_path = self.all_frames[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
            
        return lr_img, hr_img