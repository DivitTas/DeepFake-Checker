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

def detect_and_crop_faces(frames_dir, output_dir, face_size=380):
    
    os.makedirs(output_dir, exist_ok=True)
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    frame_files.sort()

    total_faces = 0
    valid_frames = 0

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

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
            cv2.imwrite(os.path.join(output_dir, face_filename), crop)

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = normalize_transform(crop)

            faces_batch.append(crop)
            total_faces += 1

        if faces_batch:
            faces_batch = torch.stack(faces_batch)
            print(f"{frame_file}: {faces_batch.shape}")

    print(f"[INFO] Total faces saved: {total_faces}")

def get_mp4s(dir_path):
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".mp4")
    ]

def process_faces_from_parent(frames_parent, faces_parent, face_size):
    os.makedirs(faces_parent, exist_ok=True)

    for video_folder in os.listdir(frames_parent):
        frames_dir = os.path.join(frames_parent, video_folder)
        output_dir = os.path.join(faces_parent, video_folder)

        if not os.path.isdir(frames_dir):
            continue

        detect_and_crop_faces(
            frames_dir,
            output_dir,
            face_size=face_size
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=False)
    parser.add_argument("--frames_dir", type=str, default="frames")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="faces")
    parser.add_argument("--face_size", type=int, default=380)
    args = parser.parse_args()


    #sorry python gods, I am hardcoding this rn"
    for video_path in get_mp4s("training_files/train/original"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("training_files/frames/original", video_name)

        extract_frames(
            video_path,
            out_dir,
            fps=args.fps
        )


    for video_path in get_mp4s("training_files/train/fake"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("training_files/frames/fake", video_name)

        extract_frames(
            video_path,
            out_dir,
            fps=args.fps
        )


    process_faces_from_parent(
        "training_files/frames/original",
        "training_files/faces/original",
        face_size=args.face_size
    )

    process_faces_from_parent(
        "training_files/frames/fake",
        "training_files/faces/fake",
        face_size=args.face_size
    )


if __name__ == "__main__":
    main()
