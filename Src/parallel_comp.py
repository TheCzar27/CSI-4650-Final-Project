import os
import time
import torch
from filters import SobelFilter
from utils import load_images, save_image, log_time

def process_images_parallel():
    input_dir = "data/raw"
    output_dir = "data/processed/parallel"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sobel_filter = SobelFilter(device=device)  # Use GPU for parallel processing
    images = load_images(input_dir, as_tensor=True, size=(256, 256)).to(device)

    start_time = time.time()
    processed_images = sobel_filter.apply_parallel(images)
    end_time = time.time()

    for i, image in enumerate(processed_images):
        save_image(image.cpu(), os.path.join(output_dir, f"processed_{i}.png"))

    runtime = end_time - start_time
    log_time("parallel", runtime)
    print(f"Parallel processing complete. Runtime: {runtime:.4f} seconds")


