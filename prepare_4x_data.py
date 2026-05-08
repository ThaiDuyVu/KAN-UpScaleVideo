# prepare_4x_data.py
from PIL import Image
import os

src = "data/frames"
dst = "data/lr_frames_4x"

for clip in os.listdir(src):
    clip_path = os.path.join(src, clip)
    if not os.path.isdir(clip_path): continue
    out_clip = os.path.join(dst, clip)
    os.makedirs(out_clip, exist_ok=True)
    for f in os.listdir(clip_path):
        img = Image.open(os.path.join(clip_path, f))
        w, h = img.size
        lr = img.resize((w//4, h//4), Image.BICUBIC)
        lr.save(os.path.join(out_clip, f))

print("✅ Done")