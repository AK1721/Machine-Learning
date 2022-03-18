from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


dataset = ImageFolder("dataset")

validation_split = 0.1
random_seed = 11
dataset_size = len(dataset)
classes = len(set(dataset.targets))
indices = list(range(dataset_size))
split = int(np.floor(dataset_size * validation_split))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)


class map_transform(Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map_fn = map_fn

    def __getitem__(self, index):
        return self.map_fn(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


transformer = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = map_transform(train_dataset, transformer)
val_dataset = map_transform(val_dataset, transformer)

print("dataset loaded")

batch_size = 128
train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size)
val_loader = DataLoader(dataset= val_dataset, batch_size=batch_size)

