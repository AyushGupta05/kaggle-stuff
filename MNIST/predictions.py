import torch
import numpy as np
import pandas as pd

from train import MNISTClassifier
from pipeline import val_loader, test_loader

def i_guess():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    checkpoint = torch.load("best_model.pth",map_location=device)

    model = MNISTClassifier(
        n_layers=checkpoint["n_layers"],
        n_filters=checkpoint["n_filters"],
        kernel_size=checkpoint["kernel_size"],
        dropout_rate=checkpoint["dropout_rate"],
        fc_size=checkpoint["fc_size"],
        use_batchnorm=checkpoint["use_batchnorm"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images,targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct/total * 100

    print(f"Dev accuracy: {accuracy:.2f}%")
    print(f"Saved checkpoint accuracy: {checkpoint['accuracy']:.2f}%")

    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            predictions.extend(predicted.cpu().numpy())

    predictions = np.array(predictions)

    submission = pd.DataFrame({
        "ImageId": np.arange(1,len(predictions) + 1),
        "Label": predictions
    })

    submission.to_csv("submissionfixed.csv",index=False)

    print(predictions.shape)
    print("Saved submissionfixed.csv")

i_guess()