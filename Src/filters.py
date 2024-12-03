import torch
import torch.nn.functional as F

class SobelFilter:
    def __init__(self, device="cpu"):
        """Initialize Sobel filters."""
        self.device = torch.device(device)
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                     [-2., 0., 2.],
                                     [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0).to(self.device)

        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                     [ 0.,  0.,  0.],
                                     [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0).to(self.device)

    def apply_single(self, image):
        """Apply Sobel filter to a single image (3D tensor: [1, H, W])."""
        if image.dim() == 2:  # If 2D, add channel dimension
            image = image.unsqueeze(0)
        image = F.pad(image.unsqueeze(0), (1, 1, 1, 1), mode="reflect")  # Add batch and pad
        edges_x = F.conv2d(image, self.sobel_x)
        edges_y = F.conv2d(image, self.sobel_y)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    def apply_parallel(self, images):
        """Apply Sobel filter to a batch of images (4D tensor: [N, 1, H, W])."""
        if images.dim() == 3:  # Add batch dimension if missing
            images = images.unsqueeze(0)
        images = F.pad(images, (1, 1, 1, 1), mode="reflect")  # Pad for convolution
        edges_x = F.conv2d(images, self.sobel_x)
        edges_y = F.conv2d(images, self.sobel_y)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges


