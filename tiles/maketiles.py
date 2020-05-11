import argparse
import os
from src.func import *
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from shutil import rmtree, copyfile
import numpy as np

# Args parse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', dest='oid_dir', type=str, required=True, help='Path to Open Image Dataset root. Use toolkit.')
parser.add_argument('-o', '--outdir', dest='out_dir', type=str, required=True, help='Output directory for tiles.')
parser.add_argument('-n', '--ntiles', dest='num_tiles', type=int, required=True, help='Number of tiles to generate.')
parser.add_argument('-x', '--nnotiles', dest='num_notiles', type=int, required=True, help='Number of non-tiles (images) to generate.')
parser.add_argument('-c', '--cleanup', dest='cleanup', type=int, required=False, default='1', help='Remove temp dirs.')
parser.add_argument('-s', '--separate', dest='separate', type=int, required=False, default='0', help='Separate dir for different backgrounds?')
args = parser.parse_args()

oid_dir = args.oid_dir
out_dir = args.out_dir
num_tiles = args.num_tiles
cleanup = args.cleanup
separate = args.separate
num_notiles = args.num_notiles

# Clean-up paths
if out_dir[-1] == '/':
    out_dir=out_dir[:-1]
if oid_dir[-1] == '/':
    oid_dir=oid_dir[:-1]

# Image set type with images (train or validation)
img_set = 'validation'


#########
# Tiles #
#########

# Define pad colors
pad_colors = ('white', 'black')

# Generate images with unequal paddings
input_img_dir = f'{oid_dir}/{img_set}'
pads = (0, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 1500)
temp_dir = out_dir + f'/transformed_images'

for bg in pad_colors:
    k = 32  # Images per combos
    n = 32  # Number of pad combos to run.
    for pad_combo in range(n):
        pad = torch._np.random.choice(pads, 2, replace=False)

        pad_dir = f'{temp_dir}/{bg}/{pad[0]}_{pad[1]}'
        if not os.path.exists(pad_dir):
            os.makedirs(pad_dir)  # Clunky but make_grid operates on all classes afaics

        print(f'Padding ({pad_combo} out of {n}): {pad}, Background: {bg}')

        img_transforms = transforms.Compose([
            transforms.Pad(
                padding=tuple(pad),
                fill=bg,
                padding_mode='constant'),
            transforms.Resize((512,512)),
            transforms.ToTensor()])
        img_dataset = datasets.ImageFolder(
            input_img_dir,
            transform=img_transforms)

        imgs = get_k_img(k, img_dataset)

        for img in imgs:
            filename = gen_str(7)
            save_image(img, f'{pad_dir}/{filename}.jpg')

    # Make tiles
    img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((512,512)),
        transforms.ToTensor()])
    img_dataset = datasets.ImageFolder(
        torch.os.path.join(temp_dir, bg),
        transform=img_transforms)

    paddings = [0,1,2,3,4,5,6,7,8,9,10,25,50,100]
    pad_values = {'white':1,'black':0}

    assert separate in (0,1,'0','1'), "Separate can only be 0 or 1."
    if separate==1:
        tiles_dir = out_dir+f'/tiles/{bg}'
    else:
        tiles_dir = out_dir + f'/tiles'

    if not os.path.exists(tiles_dir):
        os.makedirs(tiles_dir)

    for i in range(num_tiles//2): # Half white/half black backgrounds
        n_rows = int(torch._np.random.choice(range(3, 6), 1)[0])  # Rows in tile
        n_cols = int(torch._np.random.choice(range(1, 6), 1)[0])
        p = int(torch._np.random.choice(paddings, 1)[0])

        imgs = get_k_img(n_rows*n_cols, img_dataset)
        g = make_grid(imgs,
                      nrow=n_rows,
                      padding=p,                    # Size in pixels of padding
                      pad_value=pad_values[bg],     # 0 black, 1 white
                      normalize=False
                      )
        # plt.imshow(g.permute(1, 2, 0))
        # plt.axis('off')
        # plt.close()
        filename = gen_str(7)
        save_image(g, f'{tiles_dir}/{filename}.jpg')

# Cleanup
if cleanup==1:
    rmtree(temp_dir)


############
# No tiles #
############
assert num_notiles>=0, "You cannot specify negative number of tiles."

class_dir = os.path.join(oid_dir, img_set)
img_classes = os.listdir(class_dir)
num_classes = len(img_classes)
img_per_class = int(np.round(num_notiles/num_classes))

for c in img_classes:
    print(f"Selecting {img_per_class} images from {c}")

    in_path = os.path.join(class_dir, c)
    out_path = os.path.join(out_dir, 'no_tiles')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    class_examples = np.random.choice(os.listdir(in_path), img_per_class, replace=True)

    for i in class_examples:
        src = os.path.join(in_path,i)
        dst = os.path.join(out_path,i)
        copyfile(src, dst)
