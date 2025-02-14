import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram
from skimage.color import rgb2gray

def load_random_cifar_image():
    # cifar_dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True)
    # index = np.random.randint(0, len(cifar_dataset))
    # image, label = cifar_dataset[index]
    # return image
    return Image.open("deer.jpeg")


def convert_to_grayscale(image):
    grayscale_image = rgb2gray(image)
    return grayscale_image


def generate_circle_points(radius=1.0, num_points=100, center=(0, 0)):
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    points = np.column_stack([x, y])
    return points


def generate_cubical_persistence(grayscale_image):
    cubical_persistence = CubicalPersistence(homology_dimensions=[0, 1, 2])
    image_for_gtda = grayscale_image.reshape((1, *grayscale_image.shape))
    diagrams = cubical_persistence.fit_transform(image_for_gtda)

    return diagrams


def generate_betti_curves(diagrams):
    bc = BettiCurve()
    betti_curves = bc.fit_transform(diagrams)

    return betti_curves


image = load_random_cifar_image()
plt.imshow(image)
plt.show()

grayscale_image = convert_to_grayscale(image)
plt.imshow(grayscale_image, cmap='gray')
plt.show()

diagrams = generate_cubical_persistence(grayscale_image)
for diagram in diagrams:
    plot = plot_diagram(diagram)
    plot.show()

betti_curves = generate_betti_curves(diagrams)
for dim in range(betti_curves.shape[1]):
    plt.plot(betti_curves[0, dim], label=f'Betti curve - Dimension {dim}')

plt.xlabel('Filtration value')
plt.ylabel('Betti number')
plt.legend()
plt.title('Betti Curves')
plt.show()