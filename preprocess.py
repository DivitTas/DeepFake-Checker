import os
import cv2
import torch
from facenet_pytorch import MTCNN
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def extract_frames(video_path, frames_dir="SampleTest", fps=5):
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps / fps), 1)
    count = 0
    saved_count = 0

    while True:
        success, frame = vidcap.read()
        if not success or frame is None:
            print("[WARNING] Skipping invalid frame")
            break

        if count % step == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        count += 1

    vidcap.release()
    print(f"[INFO] Saved {saved_count} frames to {frames_dir}")

extract_frames("sample_vid.mp4")

def detect_and_crop_faces(frames_dir, output_dir, face_size=380):
    """
    Detect faces in frames and save crops of size face_size x face_size.
    Skips invalid frames and ensures crops are valid for EfficientNet-B4.
    """
    import os
    import cv2
    import torch
    from facenet_pytorch import MTCNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    os.makedirs(output_dir, exist_ok=True)
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    frame_files.sort()

    total_faces = 0
    valid_frames = 0

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARNING] Invalid frame: {frame_file}")
            continue

        boxes, probs = mtcnn.detect(frame)

        if boxes is None or len(boxes) == 0:
            print(f"[INFO] No faces detected in {frame_file}")
            continue

        faces_in_frame = 0
        h, w = frame.shape[:2]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] Invalid crop in {frame_file}, skipping")
                continue

            crop = frame[y1:y2, x1:x2]

            ch, cw = crop.shape[:2]

            if ch >= face_size and cw >= face_size:
                crop = cv2.resize(crop, (face_size, face_size))

            else:
                top = max(0, (face_size - ch) // 2)
                bottom = max(0, face_size - ch - top)
                left = max(0, (face_size - cw) // 2)
                right = max(0, face_size - cw - left)

                crop = cv2.copyMakeBorder(
                    crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

                crop = cv2.resize(crop, (face_size, face_size))

            crop = cv2.resize(crop, (face_size, face_size))

            save_path = os.path.join(output_dir, f"{frame_file[:-4]}_face{i}.jpg")
            cv2.imwrite(save_path, crop)
            total_faces += 1
            faces_in_frame += 1
        print(f"[FRAME] {frame_file} → {faces_in_frame} face(s)")


        if faces_in_frame > 0:
            valid_frames += 1

    print(f"[INFO] Processed {len(frame_files)} frames")
    print(f"[INFO] Saved {total_faces} face crops from {valid_frames} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--frames_dir", type=str, default="frames", help="Folder to save frames")
    parser.add_argument("--faces_dir", type=str, default="faces", help="Folder to save cropped faces")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract")
    parser.add_argument("--face_size", type=int, default=380, help="Face crop size")
    args = parser.parse_args()

    extract_frames(args.video, args.frames_dir, fps=args.fps)

    detect_and_crop_faces(args.frames_dir, args.faces_dir, face_size=args.face_size)

if __name__ == "__main__" :
    main() 