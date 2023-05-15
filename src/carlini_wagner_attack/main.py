import torch
from torchvision.transforms import ToTensor
from PIL import Image
from modules import CarliniWagnerResNet
import matplotlib.pyplot as plt


# hyperparameters
kappa = 0.01
lr = 0.4
t = 130  # imagenet flamingo
lamb = 0.1

# load data
trans = ToTensor()
fish_img = Image.open("../../data/carlini_wagner/fish.JPEG")
x0 = torch.unsqueeze(trans(fish_img), dim=0)

# create learnable input
x = torch.clone(x0)

# load model
resnet = CarliniWagnerResNet(kappa=kappa, x=x, t=t, lamb=lamb)

pred = t + 1
preds = []
while True:
    if pred == t:
        break
    loss, x, scores = resnet()
    loss.backward()
    resnet.x.data = resnet.x.data - lr * resnet.x.grad
    pred = torch.argmax(scores, dim=1).item()
    preds.append(pred)

numpy_array = torch.squeeze(x.data).permute(1, 2, 0).numpy()
plt.imshow(numpy_array)
plt.title(f"misclassified as flamingo, {len(preds)} carlini wagner iterations")

plt.savefig("../../monitoring/carlini_wagner_attack/result.png")
torch.save(torch.tensor(preds), "../../monitoring/carlini_wagner/preds")
