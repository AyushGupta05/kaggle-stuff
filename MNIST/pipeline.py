from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from torchvision import transforms


X_train = np.load("X_train.npy")
X_dev = np.load("X_dev.npy")
X_test = np.load("X_test.npy")
y_dev = np.load("y_dev.npy")
y_train = np.load("y_train.npy")
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1)
    )
])

class MNISTdataset(Dataset):
    def __init__(self, images, labels = None, transform = None):
        self.labels = labels
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx],dtype=torch.float32)
        if self.labels is None:
            return image
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

generator = torch.Generator()
generator.manual_seed(42)
train_dataset = MNISTdataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True, generator = generator)


val_dataset = MNISTdataset(X_dev, y_dev)
val_loader = DataLoader(val_dataset,batch_size=64,shuffle=False)

test_dataset = MNISTdataset(X_test)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)


images, labels = next(iter(train_loader))
print(X_train.shape)
print(y_train.shape)

print(X_dev.shape)
print(y_dev.shape)

print(X_test.shape)