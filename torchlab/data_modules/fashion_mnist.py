import lightning as L
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class FashionMNISTDM(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.ToTensor()

        dataset = FashionMNIST(root=".", train=True, download=True, transform=transform)
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])

        self.test_set = FashionMNIST(root=".", train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)