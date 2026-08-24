import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pipeline import train_loader, val_loader, test_loader
import optuna
import os
import pandas as pd
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class MNISTClassifier(nn.Module):
    def __init__(self,n_filters, n_layers, kernel_size, dropout_rate, fc_size, use_batchnorm):
        super(MNISTClassifier, self).__init__() 
        blocks = []
        in_channels = 1 
        for i in range(n_layers):
            out_channels = n_filters[i]
            padding = (kernel_size - 1) // 2
            layers = [ 
                nn.Conv2d( 
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    padding=padding 
                ) 
            ] 
            if use_batchnorm: 
                layers.append( 
                    nn.BatchNorm2d(out_channels) 
                ) 
            layers.append(nn.ReLU()) 
            layers.append(nn.MaxPool2d(2))
            block = nn.Sequential(*layers)
            blocks.append(block) 
            in_channels = out_channels  
        self.features = nn.Sequential(*blocks) 
        spatial_size = 28 // (2 ** n_layers) 
        self.classifier = nn.Sequential( 
            nn.Flatten(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(n_filters[-1] * spatial_size * spatial_size, fc_size), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(fc_size, 10) 
        ) 

    def forward(self, x): 
        x = self.features(x) 
         
        
        x = self.classifier(x) 
        return x 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    for idx, (image,target) in enumerate(train_loader):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True) 
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()

def evaluate(model,val_loader,device):
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
    return accuracy 

if os.path.exists("best_model_noavg.pth"):
    checkpoint = torch.load("best_model_noavg.pth",map_location=device)
    best_accuracy = checkpoint["accuracy"]
    print(f"Existing best accuracy: {best_accuracy:.2f}%")
else:
    best_accuracy = 0

def objective(trial):
    global best_accuracy
    trial_seed = seed + trial.number
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)
    n_layers = trial.suggest_int("n_layers",2,4) 
    initial_filters = trial.suggest_categorical(
        "initial_filters",
        [16,32,64]
    )
    dropout_rate = 0.2
    n_filters = []
    kernel_size = trial.suggest_categorical("kernel_size",[3,5,7])
    for i in range(n_layers): 
        n_filters.append(min(initial_filters * (2 ** i),256))
    use_batchnorm = trial.suggest_categorical("use_batchnorm",[True,False]) 
    fc_size = trial.suggest_categorical("fc_size",[32,64,128,256]) 
    print(f"\n{'='*60}") 
    print(f"Trial {trial.number}") 
    print(f"{'='*60}") 
    print(f"Number of layers: {n_layers}") 
    print(f"Filters: {n_filters}") 
    print(f"Kernel sizes: {kernel_size}") 
    print(f"BatchNorm: {use_batchnorm}") 
    print(f"Dropout: {dropout_rate}") 
    print(f"FC size: {fc_size}") 
    print(f"Pooling: MaxPool2d") 
    print(f"{'='*60}") 
    model = MNISTClassifier( 
        n_layers=n_layers, 
        n_filters=n_filters, 
        kernel_size=kernel_size, 
        dropout_rate=dropout_rate, 
        fc_size=fc_size, 
        use_batchnorm=use_batchnorm 
    ).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=13)
    loss_function = nn.CrossEntropyLoss() 
    num_epochs = 13
    trial_best_accuracy = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_epoch(model,train_loader,loss_function,optimizer,device)
        scheduler.step()
        val_accuracy = evaluate(model,val_loader,device)
        print(f"Validation accuracy: {val_accuracy:.2f}%")
        if val_accuracy > trial_best_accuracy:
            trial_best_accuracy = val_accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_layers": n_layers,
                "n_filters": n_filters,
                "kernel_size": kernel_size,
                "dropout_rate": dropout_rate,
                "fc_size": fc_size,
                "use_batchnorm": use_batchnorm,
                "accuracy": val_accuracy,
                "epoch": epoch + 1,
                "trial": trial.number,
                "seed": trial_seed
            },"best_model_noavg.pth")
            print(f"Saved new best model | Trial {trial.number} | Epoch {epoch + 1} | Accuracy: {val_accuracy:.2f}%")
    return trial_best_accuracy

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(objective,n_trials=30)
    df = study.trials_dataframe()
    df.to_csv("mnist_study.csv",index=False)
    print(f"\nBest accuracy: {study.best_value:.2f}%")
    print(f"Best parameters: {study.best_params}")
