import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
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

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        yes_dir = os.path.join(root_dir, 'yes')
        no_dir = os.path.join(root_dir, 'no')

        for img_name in os.listdir(yes_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(yes_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(1)

        for img_name in os.listdir(no_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(no_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(0)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    num_classes = 2

    sample_image = Image.open('./images/yes/Y1.jpg').convert("RGB")
    # plt.imshow(sample_image)
    # plt.show()

    sample_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    sample_image = sample_transform(sample_image)
    sample_grayscale = convert_to_grayscale(sample_image)
    sample_diagrams = generate_cubical_persistence(sample_grayscale)
    sample_betti_curves = generate_betti_curves(sample_diagrams)
    betti_curve_size = sample_betti_curves.flatten().shape[0]

    # for dim in range(sample_betti_curves.shape[1]):
    #     plt.plot(sample_betti_curves[0, dim], label=f'Betti curve - Dimension {dim}')
    # plt.show()

    class CombinedModel(torch.nn.Module):
        def __init__(self, resnet, betti_curve_size, num_classes):
            super(CombinedModel, self).__init__()
            self.resnet = resnet
            self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 512)
            self.betti_fc = torch.nn.Linear(betti_curve_size, 512)
            self.classifier = torch.nn.Linear(1024, num_classes)

        def forward(self, image, betti_curves):
            resnet_features = self.resnet(image)
            betti_features = self.betti_fc(betti_curves)
            combined_features = torch.cat((resnet_features, betti_features), dim=1)
            output = self.classifier(combined_features)
            return output

    model = CombinedModel(model, betti_curve_size, num_classes)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BrainTumorDataset(root_dir='./images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # num_epochs = 5
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for images, labels in dataloader:
    #         grayscale_image = convert_to_grayscale(images)
    #         diagrams = generate_cubical_persistence(grayscale_image)
    #         betti_curves = generate_betti_curves(diagrams)
    #
    #         betti_curves_tensor = torch.tensor(betti_curves.flatten(), dtype=torch.float32).unsqueeze(0)
    #
    #         optimizer.zero_grad()
    #         outputs = model(images, betti_curves_tensor)  # Pass both image and Betti curve
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')
    #
    # torch.save(model.state_dict(), 'brain_tumor.pth')
    # print("Model saved to brain_tumor.pth")

    model.load_state_dict(torch.load('brain_tumor.pth'))
    model.eval()

    def get_random_image_from_dataloader(dataloader):
        while True:
            try:
                random_index = random.randint(0, len(dataloader.dataset) - 1)
                for i, (images, labels) in enumerate(dataloader):
                    if i == random_index:
                        return images, labels

            except Exception as e:
                print(f"Error loading image: {e}")
                continue

    images, labels = get_random_image_from_dataloader(dataloader)
    grayscale_image = convert_to_grayscale(images)
    diagrams = generate_cubical_persistence(grayscale_image)
    betti_curves = generate_betti_curves(diagrams)
    betti_curves_tensor = torch.tensor(betti_curves.flatten(), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(images, betti_curves_tensor)

    _, predicted_idx = torch.max(output, 1)
    classifications = [0, 1]
    predicted_label = classifications[predicted_idx.item()]
    real_label = labels[0]

    print(f"Predicted label: {predicted_label}, Real label: {real_label}, matching: {predicted_label == real_label}")


