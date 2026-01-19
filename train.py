from torchvision import models
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt





def main():
    model = models.efficientnet_b2(weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root='train_flat/', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)


    model.train()
    print("Meow we be training")
    img, label = train_dataset[0]
    print(img.shape, label)


    criterion = nn.CrossEntropyLoss()

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    epochs = 3
    train_losses = []
    print("Starting training loop")
    for epoch in range(epochs):
        print(f"👉 starting epoch {epoch+1}")
        running_loss = 0.0

        for images, labels in train_loader:
            print(f"Processing batch of size {images.size(0)}")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)     
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        train_losses.append(avg_loss)

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.show()




if __name__ == "__main__":
    main()