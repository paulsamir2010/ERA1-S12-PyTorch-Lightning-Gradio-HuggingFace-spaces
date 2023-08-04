
import math
from typing import NoReturn


import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import transforms
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image

def display_cifar_data_samples(data_set, number_of_samples: int, classes: list):
    """
    Function to display samples for data_set
    :param data_set: Train or Test data_set transformed to Tensor
    :param number_of_samples: Number of samples to be displayed
    :param classes: Name of classes to be displayed
    """
    # Get batch from the data_set
    batch_data = []
    batch_label = []
    for count, item in enumerate(data_set):
        if not count <= number_of_samples:
            break
        batch_data.append(item[0])
        batch_label.append(item[1])
    batch_data = torch.stack(batch_data, dim=0).numpy()

    # Plot the samples from the batch
    fig = plt.figure()
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        plt.tight_layout()
        plt.imshow(np.transpose(batch_data[i].squeeze(), (1, 2, 0)))
        plt.title(classes[batch_label[i]])
        plt.xticks([])
        plt.yticks([])

def display_cifar_misclassified_data(data: list,
                                     classes: list[str],
                                     inv_normalize: transforms.Normalize,
                                     number_of_samples: int = 10):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(10, 10))

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = data[i][0].squeeze().to('cpu')
        img = inv_normalize(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        #plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.title(r"Correct: " + classes[data[i][1]] + '\n' + 'Output: ' + classes[data[i][2]])
        plt.xticks([])
        plt.yticks([])       

def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data



