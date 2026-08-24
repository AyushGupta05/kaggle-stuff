import torch
import torch.nn as nn
import os

from pipeline import train_loader, val_loader
from train import MNISTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists("best_model.pth"):
    checkpoint = torch.load("best_model.pth",map_location=device)
    best_accuracy = checkpoint["accuracy"]
    print(f"Existing best accuracy: {best_accuracy:.2f}%")
else:
    best_accuracy = 0

for run in range(20):
    print(f"\nRun {run + 1}/20")

    model = MNISTClassifier(
        n_layers=4,
        n_filters=[64,128,256,256],
        kernel_size=3,
        dropout_rate=0.2,
        fc_size=256,
        use_batchnorm=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()

        for image,target in train_loader:
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(image)
            loss = loss_function(output,target)

            loss.backward()
            optimizer.step()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs,targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct/total * 100

        print(f"Run {run + 1} | Epoch {epoch + 1}/10 | Validation accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "n_layers": 4,
                "n_filters": [64,128,256,256],
                "kernel_size": 3,
                "dropout_rate": 0.2,
                "fc_size": 256,
                "use_batchnorm": True,
                "accuracy": accuracy,
                "epoch": epoch + 1,
                "run": run + 1
            },"best_model.pth")

            print(f"Saved new best model: {accuracy:.2f}%")

print(f"\nHighest accuracy: {best_accuracy:.2f}%")