import json
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram
from skimage.color import rgb2gray
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def process_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def convert_to_grayscale(image):
    image = image.squeeze().numpy().transpose((1, 2, 0))
    image = process_image(image)
    grayscale_image = rgb2gray(image)
    return grayscale_image

def generate_cubical_persistence(grayscale_image):
    cubical_persistence = CubicalPersistence(homology_dimensions=[0, 1, 2])
    image_for_gtda = grayscale_image.reshape((1, *grayscale_image.shape))
    diagrams = cubical_persistence.fit_transform(image_for_gtda)
    return diagrams

def generate_betti_curves(diagrams):
    bc = BettiCurve()
    betti_curves = bc.fit_transform(diagrams)
    return betti_curves

class BloodCellDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 5]
        if self.transform:
            image = self.transform(image)
        lbl = 0 if label == 'rbc' else 1
        return image, torch.tensor(lbl)

if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    num_classes = 2
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BloodCellDataset(root_dir='./data/images', csv_file='./archive/annotations.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, labels in dataloader:
        image = images.squeeze().numpy().transpose((1, 2, 0))
        image = process_image(image)
        plt.imshow(image)
        plt.show()

        grayscale_image = convert_to_grayscale(images)
        plt.imshow(grayscale_image, cmap='gray')
        plt.show()

        diagrams = generate_cubical_persistence(grayscale_image)
        for diagram in diagrams:
            plot = plot_diagram(diagram)
            plot.show()

        bc = BettiCurve()
        betti_curves = bc.fit_transform(diagrams)
        for dim in range(betti_curves.shape[1]):
            plt.plot(betti_curves[0, dim], label=f'Betti curve - Dimension {dim}')
        plt.show()

    with open('./labels.json', 'r') as file:
        resnet_labels = json.load(file)

    random_image, random_label = next(iter(dataloader))
    plt.imshow(process_image(random_image.squeeze().numpy().transpose((1, 2, 0))))
    plt.show()

    with torch.no_grad():
        output = model(random_image)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = resnet_labels[predicted_idx.item()]

    print(f"Predicted label: {predicted_label}")

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for images, labels in dataloader:
    #         grayscale_image = convert_to_grayscale(images)
    #         diagrams = generate_cubical_persistence(grayscale_image)
    #         betti_curves = generate_betti_curves(diagrams)
    #         betti_curves_flattened = torch.tensor(betti_curves.flatten(), dtype=torch.float32).unsqueeze(0)
    #
    #         outputs = model(images)
    #         outputs = torch.cat((outputs, betti_curves_flattened), dim=1)
    #
    #         loss = criterion(outputs, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')
    #
    # torch.save(model.state_dict(), 'blood_cell_model.pth')