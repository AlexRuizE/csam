import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from src.func import get_k_img



# Paths
in_dir = '/home/monorhesus/Data/oidv6/' # Path to Open Image Dataset root. Use toolkit.
out_dir = '/home/monorhesus/Pictures'

if out_dir[-1] == '/':
    out_dir=out_dir[:-1]


# Load images and transforms
img_dir = 'validation'
img_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.Resize((512,512)),
                                     transforms.ToTensor()])
img_dataset = datasets.ImageFolder(torch.os.path.join(in_dir, img_dir),
                                     transform=img_transforms)


# Make grid
paddings = [0,100]
pad_values = [0,1]
for p in paddings:
    for v in pad_values:
        imgs = get_k_img(25, img_dataset)
        g = make_grid(imgs,
                      nrow=5,
                      padding=p,           # Size in pixels of padding
                      pad_value=v,          # 0 black, 1 white
                      normalize=False
                      )
        plt.imshow(g.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'{out_dir}/tiles_{p}pad_{v}bg.png', bbox_inches='tight', pad_inches=0)
        plt.close()
