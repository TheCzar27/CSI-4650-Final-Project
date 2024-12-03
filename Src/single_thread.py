from filters import SobelFilter
from utils import load_images, save_image, log_time
from torchvision.transforms import ToTensor
import time
import os

def process_images_single():
    input_dir = "data/raw"
    output_dir = "data/processed/single"
    os.makedirs(output_dir, exist_ok=True)

    sobel_filter = SobelFilter(device="cpu")  # Use CPU for single-threaded processing
    images = load_images(input_dir)  # Load PIL images
    to_tensor = ToTensor()  # Transformation to convert PIL to Tensor

    start_time = time.time()
    for i, image in enumerate(images):
        image_tensor = to_tensor(image)  # Convert PIL image to tensor
        processed_image = sobel_filter.apply_single(image_tensor)
        save_image(processed_image, os.path.join(output_dir, f"processed_{i}.png"))
    end_time = time.time()

    runtime = end_time - start_time
    log_time("single-threaded", runtime)
    print(f"Single-threaded processing complete. Runtime: {runtime:.4f} seconds")

