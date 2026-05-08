"""
prepare_4x_data.py
Chạy 1 lần để tạo thư mục data/lr_frames_4x/
từ data/frames/ (ảnh gốc full resolution)
"""
from PIL import Image
import os

SRC = "data/frames"
DST = "data/lr_frames_4x"

def main():
    if not os.path.exists(SRC):
        print(f"❌ Không tìm thấy {SRC}")
        return

    clips = [c for c in sorted(os.listdir(SRC)) if not c.startswith(".")]
    print(f"📂 Tìm thấy {len(clips)} clips")

    total = 0
    for clip in clips:
        src_clip = os.path.join(SRC, clip)
        dst_clip = os.path.join(DST, clip)
        if not os.path.isdir(src_clip):
            continue
        os.makedirs(dst_clip, exist_ok=True)

        frames = [f for f in sorted(os.listdir(src_clip)) if not f.startswith(".")]
        for f in frames:
            src_path = os.path.join(src_clip, f)
            dst_path = os.path.join(dst_clip, f)
            if os.path.exists(dst_path):
                continue
            img = Image.open(src_path).convert("RGB")
            w, h = img.size
            lr = img.resize((w // 4, h // 4), Image.BICUBIC)
            lr.save(dst_path)
            total += 1

        print(f"  ✅ {clip}: {len(frames)} frames")

    print(f"\n🎉 Xong! Đã tạo {total} ảnh LR 4x tại {DST}")

if __name__ == "__main__":
    main()