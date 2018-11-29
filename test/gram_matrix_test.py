import torch

from skimage import io
from torchvision.transforms import ToTensor
import os
from SuperStyleTransfer.Network.CroppedVGGWithoutCuda import CroppedVGGWithoutCuda
from SuperStyleTransfer.Utils.Utils import calc_gram_matrix

vgg = CroppedVGGWithoutCuda()
# load style
style_dir = "/Users/rl/PycharmProjects/SuperStyleTransfer/data/images/style-images/rsz_udnie.jpg"
style = ToTensor()(io.imread(style_dir)).unsqueeze(0)
style_res = vgg(style)

# load image
images_dir = "/Users/rl/PycharmProjects/SuperStyleTransfer/data/images/content-images/TrainingData/"
files = os.listdir(images_dir)
res = None

for f in files:
    image_path = "{}{}".format(images_dir, f)
    image = ToTensor()(io.imread(image_path)).unsqueeze(0)
    image_res = vgg(image)
    # compute gram_diff for 4 layers
    print(f)
    for i in range(4):
        loss = torch.sum((calc_gram_matrix(image_res[i]) - calc_gram_matrix(style_res[i])) ** 2)
        print(loss)
    print('\n')