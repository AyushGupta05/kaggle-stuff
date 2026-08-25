import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import MNISTClassifier
from pipeline import val_loader, test_loader

import torch
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    return model, checkpoint


def load_ensemble(device):
    model1, checkpoint1 = load_model("best_model_extended1.pth", device)
    model2, checkpoint2 = load_model("best_model_extended2.pth", device)
    model3, checkpoint3 = load_model("best_model_extended3.pth", device)
    model4, checkpoint4 = load_model("best_model_noavg1.pth", device)
    model5, checkpoint5 = load_model("best_model_noavg.pth", device)

    models = (model1, model2, model3, model4, model5)

    print(checkpoint1["accuracy"])
    print(checkpoint2["accuracy"])
    print(checkpoint3["accuracy"])
    print(checkpoint4["accuracy"])
    print(checkpoint5["accuracy"])

    return models


def view_wrong_images(num_images=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    models = load_ensemble(device)

    wrong_images = []
    wrong_preds = []
    wrong_targets = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            ensemble_probs = 0

            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                ensemble_probs += probs

            ensemble_probs /= len(models)

            predicted = ensemble_probs.argmax(dim=1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            wrong = predicted != targets

            wrong_images.extend(images[wrong].cpu())
            wrong_preds.extend(predicted[wrong].cpu().numpy())
            wrong_targets.extend(targets[wrong].cpu().numpy())

    accuracy = correct / total * 100

    print(f"Ensemble dev accuracy: {accuracy:.4f}%")
    print(f"Total wrong: {len(wrong_images)}")

    for i in range(min(num_images, len(wrong_images))):
        image = wrong_images[i].squeeze()

        plt.imshow(image, cmap="gray")
        plt.title(f"Predicted: {wrong_preds[i]} | Actual: {wrong_targets[i]}")
        plt.axis("off")
        plt.show()


def make_prediction_file(filename="submission_ensemble.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    models = load_ensemble(device)

    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            ensemble_probs = 0

            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                ensemble_probs += probs

            ensemble_probs /= len(models)

            predicted = ensemble_probs.argmax(dim=1)

            predictions.extend(predicted.cpu().numpy())

    predictions = np.array(predictions)

    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions
    })

    submission.to_csv(filename, index=False)

    print(predictions.shape)
    print(f"Saved {filename}")


if __name__ == "__main__":
    view_wrong_images()
   