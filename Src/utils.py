import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image
from torchvision.transforms import ToTensor, Resize

def load_images(directory, as_tensor=False, size=(256, 256)):
    images = []
    resize = Resize(size)
    to_tensor = ToTensor()

    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(directory, filename)).convert("L")  # Grayscale
            img_resized = resize(img)  # Resize to uniform size
            images.append(to_tensor(img_resized) if as_tensor else img_resized)

    return torch.stack(images) if as_tensor else images


def save_image(tensor, path):
    img = ToPILImage()(tensor)
    img.save(path)

def log_time(mode, runtime):
    with open("benchmarks/results.csv", "a") as f:
        f.write(f"{mode},{runtime:.4f}\n")

