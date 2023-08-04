import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from model import CustomResNet
import config




class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers,model):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = model
        

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [45000, 5000])
        VALIDATIONDS = self.val_ds
        self.test_ds = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def get_misclassified_data(self):
        """
        Function to run the model on test set and return misclassified images
        :param model: Network Architecture
        :param device: CPU/GPU
        :param test_loader: DataLoader for test set
        """
        # Prepare the model for evaluation i.e. drop the dropout layer

        from model import CustomResNet


        self.model = self.model.to("cuda")
        self.model.eval()

        # List to store misclassified Images
        misclassified_data = []

        test_loader = self.val_dataloader()

        # Reset the gradients
        with torch.no_grad():
            # Extract images, labels in a batch
            device = "cuda"
            for data, target in test_loader:

                # Migrate the data to the device
                data, target = data.to(device), target.to(device)

                # Extract single image, label from the batch
                for image, label in zip(data, target):

                    # Add batch dimension to the image
                    image = image.unsqueeze(0)

                    # Get the model prediction on the image
                    output = self.model(image)

                    # Convert the output from one-hot encoding to a value
                    pred = output.argmax(dim=1, keepdim=True)

                    # If prediction is incorrect, append the data
                    if pred != label:
                        misclassified_data.append((image, label, pred))
        return misclassified_data    
