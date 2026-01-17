from torchvision import models
import torch.nn as nn

model = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)