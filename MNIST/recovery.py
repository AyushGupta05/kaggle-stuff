import torch
import torch.nn as nn
import numpy as np
import random
import copy

from pipeline import train_loader, val_loader
from train import MNISTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
top_models = []

for i in range(1,4):
    checkpoint = torch.load(f"best_model_noavg{i}.pth",map_location="cpu")
    top_models.append(checkpoint)

top_models = sorted(
    top_models,
    key=lambda x: x["accuracy"],
    reverse=True
)[:3]

print("Starting top 3 models:")

for i,checkpoint in enumerate(top_models):
    print(
        f"{i + 1}: {checkpoint['accuracy']:.2f}% | "
        f"Run: {checkpoint['run']} | "
        f"Seed: {checkpoint['seed']}"
    )

for run in range(20):
    print(f"\nRun {run + 1}/20")

    run_seed = seed + 20 + run
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)

    model = MNISTClassifier(
        n_layers=4,
        n_filters=[64,128,256,256],
        kernel_size=5,
        dropout_rate=0.2,
        fc_size=256,
        use_batchnorm=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    loss_function = nn.CrossEntropyLoss()

    run_best_accuracy = 0
    run_best_checkpoint = None

    for epoch in range(20):
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

        print(f"Run {run + 1} | Epoch {epoch + 1}/20 | Validation accuracy: {accuracy:.2f}%")

        if accuracy > run_best_accuracy:
            run_best_accuracy = accuracy

            run_best_checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "n_layers": 4,
                "n_filters": [64,128,256,256],
                "kernel_size": 5,
                "dropout_rate": 0.2,
                "fc_size": 256,
                "use_batchnorm": True,
                "accuracy": accuracy,
                "epoch": epoch + 1,
                "run": 21 + run,
                "seed": run_seed
            }

        scheduler.step()

    top_models.append(run_best_checkpoint)

    top_models = sorted(
        top_models,
        key=lambda x: x["accuracy"],
        reverse=True
    )[:3]

    for i,checkpoint in enumerate(top_models):
        torch.save(
            checkpoint,
            f"best_model_extended{i + 1}.pth"
        )

    print(f"Best accuracy for run {run + 1}: {run_best_accuracy:.2f}%")
    print("Current top models:")

    for i,checkpoint in enumerate(top_models):
        print(f"{i + 1}: {checkpoint['accuracy']:.2f}%")

print("\nFinal top 3 models:")

for i,checkpoint in enumerate(top_models):
    print(
        f"best_model_extended{i + 1}.pth | "
        f"Accuracy: {checkpoint['accuracy']:.2f}% | "
        f"Run: {checkpoint['run']} | "
        f"Epoch: {checkpoint['epoch']} | "
        f"Seed: {checkpoint['seed']}"
    )

    