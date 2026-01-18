import os
import shutil

def flatten(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    for video in os.listdir(src_root):
        video_path = os.path.join(src_root, video)
        if not os.path.isdir(video_path):
            continue

        for fname in os.listdir(video_path):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            src = os.path.join(video_path, fname)
            dst = os.path.join(dst_root, f"{video}_{fname}")
            shutil.copy(src, dst)

    print(f"Flattened {src_root} → {dst_root}")

flatten("train/original", "train_flat/original")
flatten("train/fake", "train_flat/fake")
