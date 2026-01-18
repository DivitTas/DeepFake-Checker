import os
import torch
import cv2
import argparse
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(weights_path):
    model = EfficientNet.from_name("efficientnet-b4")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_faces_by_frame(faces_dir):
    frames = defaultdict(list)

    for fname in os.listdir(faces_dir):
        if not fname.endswith(".jpg"):
            continue

        frame_id = fname.split("_face")[0]  # frame_00012
        frames[frame_id].append(os.path.join(faces_dir, fname))

    return frames

def infer_frames(model, faces_dir):
    frames = load_faces_by_frame(faces_dir)
    frame_confidences = {}

    for frame_id, face_paths in frames.items():
        faces = []

        for path in face_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            faces.append(img)

        faces = torch.stack(faces).to(device)

        with torch.no_grad():
            preds = model(faces)
            preds = torch.sigmoid(preds)

        frame_confidence = preds.max().item()
        frame_confidences[frame_id] = frame_confidence

        print(f"{frame_id} → fake confidence: {frame_confidence:.3f}")

    return frame_confidences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.weights)
    infer_frames(model, args.faces_dir)

if __name__ == "__main__":
    main()
