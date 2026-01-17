from torchvision import models
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

model = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(root='training_files/train', transform=transform)


model.train()
print("Meow we be training")
img, label = train_dataset[0]
print(img.shape, label)

epochs = 10
#for epoch in range(epochs):
    #print(f"Epoch {epoch+1}/{epochs} running...")
    #blah blah training code
    