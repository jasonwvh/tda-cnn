import json

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_random_cifar_image():
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    index = np.random.randint(0, len(cifar_dataset))
    image, label = cifar_dataset[index]
    return image

def classify_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return classes[predicted.item()]

if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_path = 'deer.jpeg'
    input_image = Image.open(img_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    with open('./labels.json', 'r') as file:
        resnet_labels = json.load(file)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = resnet_labels[predicted_idx.item()]

    print(f"Predicted label: {predicted_label}")