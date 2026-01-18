import os
import torch
import cv2
import argparse
import torchvision.transforms as transforms
from torchvision import models
from collections import defaultdict
from preprocess import extract_frames, detect_and_crop_faces
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(weights_path):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 2
    )

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

        frame_id = fname.split("_face")[0]
        frames[frame_id].append(os.path.join(faces_dir, fname))

    return frames

def infer_frames(model, faces_dir):
    frames = load_faces_by_frame(faces_dir)
    frame_confidences = {}
    confidence_sum = 0.0

    for frame_id, face_paths in frames.items():
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
            fake_probs = probs[:, 1]

        frame_confidence = fake_probs.max().item()
        frame_confidences[frame_id] = frame_confidence
        confidence_sum += frame_confidence
        print(f"{frame_id} → real confidence: {frame_confidence:.3f}")
    avg_confidence = confidence_sum / len(frame_confidences)
    print(f"Average real confidence across frames: {avg_confidence:.3f}")
    return frame_confidences



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.weights)
    extract_frames(args.faces_dir, frames_dir="temp_infer_frames")
    detect_and_crop_faces(
        "temp_infer_frames",
        "temp_infer_faces",
        face_size=224
    )
    infer_frames(model, "temp_infer_faces")

if __name__ == "__main__":
    main()
