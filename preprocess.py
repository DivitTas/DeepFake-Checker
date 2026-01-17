import os
import cv2
import torch
from facenet_pytorch import MTCNN
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

