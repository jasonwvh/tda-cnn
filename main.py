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
    def __init__(self, root_dir, transform=None, split='train', test_size=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.test_size = test_size

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

        combined_data = list(zip(self.image_paths, self.labels))
        random.shuffle(combined_data)
        self.image_paths, self.labels = zip(*combined_data)

        split_index = int(len(self.image_paths) * (1 - test_size))
        train_image_paths = self.image_paths[:split_index]
        train_labels = self.labels[:split_index]
        test_image_paths = self.image_paths[split_index:]
        test_labels = self.labels[split_index:]

        if self.split == 'train':
            self.image_paths = train_image_paths
            self.labels = train_labels
        elif self.split == 'test':
            self.image_paths = test_image_paths
            self.labels = test_labels
        else:
            raise ValueError("Split must be 'train' or 'test'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

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

class ImageOnlyModel(torch.nn.Module):
    def __init__(self, resnet, num_classes):
        super(ImageOnlyModel, self).__init__()
        self.resnet = resnet
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, image):
        output = self.resnet(image)
        return output

if __name__ == '__main__':
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

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BrainTumorDataset(root_dir='./images', transform=transform, split='train')
    test_dataset = BrainTumorDataset(root_dir='./images', transform=transform, split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_classes = 2
    num_epochs = 5
    resnet50 = models.resnet50(pretrained=True)

    # ----- training combined model ----- #
    combined_model = CombinedModel(resnet50, betti_curve_size, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        combined_model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            grayscale_image = convert_to_grayscale(images)
            diagrams = generate_cubical_persistence(grayscale_image)
            betti_curves = generate_betti_curves(diagrams)
            betti_curves_tensor = torch.tensor(betti_curves.flatten(), dtype=torch.float32).unsqueeze(0)

            optimizer.zero_grad()
            outputs = combined_model(images, betti_curves_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader)}')

    torch.save(combined_model.state_dict(), 'brain_tumor_ph.pth')
    print("Combined Model saved to brain_tumor_ph.pth")

    # ----- training image-only model ----- #
    image_only_model = ImageOnlyModel(resnet50, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(image_only_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        image_only_model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = image_only_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            epoch_loss = running_loss / len(train_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    torch.save(image_only_model.state_dict(), 'brain_tumor_image_only.pth')
    print("Image-only model saved to brain_tumor_image_only.pth")

    # ----- testing combined model ----- #
    combined_model = CombinedModel(resnet50, betti_curve_size, num_classes)
    combined_model.load_state_dict(torch.load('brain_tumor_ph.pth'))
    combined_model.eval()

    correct_predictions_combined = 0
    total_samples_combined = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            grayscale_image = convert_to_grayscale(images)
            diagrams = generate_cubical_persistence(grayscale_image)
            betti_curves = generate_betti_curves(diagrams)
            betti_curves_tensor = torch.tensor(betti_curves.flatten(), dtype=torch.float32).unsqueeze(0)

            outputs = combined_model(images, betti_curves_tensor)
            _, predicted_labels = torch.max(outputs, 1)

            total_samples_combined += labels.size(0)
            correct_predictions_combined += (predicted_labels == labels).sum().item()

    accuracy_combined = correct_predictions_combined / total_samples_combined
    print(f'Accuracy of the Combined Model on the test images: {accuracy_combined * 100:.2f}%')

    # ----- testing image-only model ----- #
    image_only_model = ImageOnlyModel(resnet50, num_classes)
    image_only_model.load_state_dict(torch.load('brain_tumor_image_only.pth'))
    image_only_model.eval()

    correct_predictions_image_only = 0
    total_samples_image_only = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = image_only_model(images)
            _, predicted_labels = torch.max(outputs, 1)

            total_samples_image_only += labels.size(0)
            correct_predictions_image_only += (predicted_labels == labels).sum().item()

    accuracy_image_only = correct_predictions_image_only / total_samples_image_only
    print(f'Accuracy of the Image-Only model on the test images: {accuracy_image_only * 100:.2f}%')