import os
import torch
import cv2
import argparse
import torchvision.transforms as transforms
from torchvision import models
from collections import defaultdict
from preprocess import extract_frames, detect_and_crop_faces
import csv
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.2  # confidence <= 0.2 => DEEPFAKE
scores_list = []

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- MODEL ---------------- #
def load_model(weights_path):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ---------------- DATA LOADING ---------------- #
def load_faces_by_frame(faces_dir):
    frames = defaultdict(list)
    for fname in os.listdir(faces_dir):
        if fname.endswith(".jpg"):
            frame_id = fname.split("_face")[0]  # frame_00012
            frames[frame_id].append(os.path.join(faces_dir, fname))
    return frames

# ---------------- INFERENCE ---------------- #
def infer_frames(model, faces_dir, fps):
    frames = load_faces_by_frame(faces_dir)
    results = []

    for frame_id in sorted(frames.keys()):
        face_paths = frames[frame_id]
        faces = []

        for path in face_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            faces.append(img)

        faces = torch.stack(faces).to(device)

        with torch.no_grad():
            logits = model(faces)
            probs = torch.softmax(logits, dim=1)
            fake_probs = probs[:, 1]  # class 1 = FAKE

        frame_confidence = fake_probs.max().item()
        scores_list.append(frame_confidence)
        is_fake = frame_confidence <= THRESHOLD
        label = "DEEPFAKE" if is_fake else "REAL"

        frame_num = int(frame_id.split("_")[1])
        timestamp = frame_num / fps

        results.append({
            "frame": frame_id,
            "frame_num": frame_num,
            "confidence": frame_confidence,
            "is_fake": is_fake,
            "label": label,
            "timestamp": timestamp
        })

        print(
            f"{frame_id} | {label} | "
            f"confidence={frame_confidence:.3f} | "
            f"frame_num={frame_num}"
        )

    return results

# ---------------- SEGMENT BUILDING ---------------- #
def build_fake_segments(results, fps):
    segments = []
    current_start = None
    prev_frame = None

    for r in results:
        frame_num = r["frame_num"]
        is_fake = r["is_fake"]

        if is_fake:
            if current_start is None:
                current_start = frame_num
            elif prev_frame is not None and frame_num != prev_frame + 1:
                segments.append((current_start / fps, prev_frame / fps))
                current_start = frame_num
        else:
            if current_start is not None:
                segments.append((current_start / fps, prev_frame / fps))
                current_start = None

        prev_frame = frame_num

    if current_start is not None:
        segments.append((current_start / fps, prev_frame / fps))

    return segments

# ---------------- PLOT & CSV ---------------- #
def save_scores_csv_and_plot(scores_list):
    # Save CSV
    csv_file_path = "frame_scores.csv"
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "score"])
        for idx, score in enumerate(scores_list):
            writer.writerow([idx, score])
    print(f"[INFO] Saved {len(scores_list)} scores to {csv_file_path}")

    # Plot confidence vs frame
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(scores_list)), scores_list, marker='o')
    plt.title("Fake Confidence per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Fake Confidence")
    plt.grid(True)
    plt.savefig("confidence_vs_frame.png")
    plt.show()
    print("[INFO] Saved plot to confidence_vs_frame.png")

# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_dir", type=str, required=True, help="Folder of cropped faces or input video")
    parser.add_argument("--weights", type=str, required=True, help="Model weights .pth file")
    parser.add_argument("--fps", type=int, default=5, help="FPS for frame extraction")
    args = parser.parse_args()

    model = load_model(args.weights)

    # If input is a video, preprocess first
    if args.faces_dir.endswith(".mp4"):
        extract_frames(args.faces_dir, frames_dir="temp_infer_frames", fps=args.fps)
        detect_and_crop_faces("temp_infer_frames", "temp_infer_faces", face_size=224)
        results = infer_frames(model, "temp_infer_faces", args.fps)
    else:
        results = infer_frames(model, args.faces_dir, args.fps)

    # Build fake segments
    segments = build_fake_segments(results, args.fps)
    is_video_fake = any(r["is_fake"] for r in results)

    print("\n================ SUMMARY ================")
    print(f"Video/Image is fake: {is_video_fake}")
    if segments:
        print("⚠️ Deepfake detected in segments:")
        for start, end in segments:
            print(f"- From {start:.2f}s to {end:.2f}s")
    else:
        print("✅ No deepfake segments detected")

    # Save CSV and plot
    save_scores_csv_and_plot(scores_list)

if __name__ == "__main__":
    main()
