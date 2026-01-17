from torchvision import models
import torch.nn as nn
from torchvison import transforms

model = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

model.train()
print("Meow we be training")

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs} running...")
    #blah blah training code
    