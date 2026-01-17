import os
import cv2
import torch
from facenet_pytorch import MTCNN
import argparse
import torchvision.transforms as transforms


normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)


def extract_frames(video_path, frames_dir="frames", fps=5):
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps / fps), 1)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{saved:05d}.jpg"),
                frame
            )
            saved += 1

        count += 1

    cap.release()
    print(f"[INFO] Saved {saved} frames to {frames_dir}")


def detect_and_crop_faces(frames_dir, face_size=380):
    faces_dir = "cropped_faces"
    os.makedirs(faces_dir, exist_ok=True)

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    )

    total_faces = 0

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        boxes, _ = mtcnn.detect(frame)
        faces_batch = []

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (face_size, face_size))

            face_filename = f"{frame_file[:-4]}_face{total_faces}.jpg"
            cv2.imwrite(os.path.join(faces_dir, face_filename), crop)

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = normalize_transform(crop)

            faces_batch.append(crop)
            total_faces += 1

        if faces_batch:
            faces_batch = torch.stack(faces_batch)
            print(f"{frame_file}: {faces_batch.shape}")

    print(f"[INFO] Total faces saved: {total_faces}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, default="frames")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--face_size", type=int, default=380)
    args = parser.parse_args()

    extract_frames(args.video, args.frames_dir, fps=args.fps)
    detect_and_crop_faces(args.frames_dir, face_size=args.face_size)


if __name__ == "__main__":
    main()
