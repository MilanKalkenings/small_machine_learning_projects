import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "milankalkenings==0.1.42"])

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, GaussianBlur
from milankalkenings.visualization import images_subplot


def grad_to_cam(grad: torch.Tensor) -> torch.Tensor:
    return torch.max(grad.abs(), dim=0).values


def min_max_scale(tensor: torch.Tensor) -> torch.Tensor:
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)


def calc_grad(module: torch.nn.Module, input: torch.Tensor, class_id: int) -> torch.Tensor:
    input.requires_grad_(True)
    input.grad = None  # avoid gradient accumulation
    class_probabilities = module(input)  # forward pass
    class_probabilities[0][class_id].backward()  # populate gradients
    return input.grad.data


def smooth_grad(
        module: torch.nn.Module,
        image_tensor: torch.Tensor,
        class_id: int,
        n: int,
        noise_level: float) \
        -> torch.Tensor:
    sum_grad = torch.zeros(size=image_tensor.shape)
    for _ in range(n):
        image_tensor_noisy = image_tensor + noise_level * torch.rand(size=image_tensor.shape)
        sum_grad += calc_grad(module=module, input=image_tensor_noisy, class_id=class_id)
    grad = torch.squeeze(sum_grad / n)
    return grad_to_cam(grad=torch.squeeze(grad))


def mask(tensor: torch.Tensor, thresh: float) -> torch.Tensor:
    return tensor > thresh


def clip(tensor: torch.Tensor, thresh: float) -> torch.Tensor:
    return (mask(tensor=tensor, thresh=thresh)).int() * 255


data_path = "../../data/imagenet"
image_path = data_path + "/dog_cat.jpeg"
monitoring_path = "../../monitoring/smoothgrad"
image_size = 256
smoothgrad_n = 10  # up to 50 is recommended by the authors
noise_level = 1
thresh = 0.4
enhance = 2


# load data
to_tensor = Compose([Resize(size=image_size), ToTensor()])
image = Image.open(image_path)
image_tensor = to_tensor(image)
image_batch = torch.unsqueeze(image_tensor, dim=0)

# load model
resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2).eval()

# determine top softmax output
top_ids = torch.topk(input=resnet(image_batch), k=2).indices[0]
print(f"top 2 class ids:", top_ids)

smoothgrad_cams = []
for class_id in top_ids:
    cam = smooth_grad(
        module=resnet,
        image_tensor=image_batch,
        n=smoothgrad_n,
        noise_level=noise_level,
        class_id=class_id)
    smoothgrad_cams.append(torch.unsqueeze(cam, dim=0))

smoothgrad_cams.append(torch.zeros_like(smoothgrad_cams[-1]))
smoothgrad_cams = min_max_scale(torch.cat(smoothgrad_cams)) * enhance
smoothgrad_cams_clipped = clip(smoothgrad_cams, thresh=thresh)

images_subplot(
    images=[image_tensor, smoothgrad_cams, smoothgrad_cams_clipped],
    title="SmoothGrad for the two most dominant classes",
    subplot_titles=["image", "SmoothGrad CAM", "SmoothGrad CAM clipped"],
    save_file=monitoring_path + "/mixed_smoothgrad")


