import os
import cv2
import torch
from facenet_pytorch import MTCNN
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def extract_frames(video_path, frames_dir, fps=5):
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps / fps), 1)
    count = 0
    saved_count = 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
print("meow")
