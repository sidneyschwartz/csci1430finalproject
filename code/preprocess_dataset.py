import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm as CM
from image import *

# root is the path to ShanghaiTech dataset
root='data/ShanghaiTech/'

# Create the paths
part_B_train = os.path.join(root, 'part_A/train_data','images')
part_B_test = os.path.join(root, 'part_A/test_data','images')
path_sets = [part_B_train,part_B_test]

# Add all the individual images to a list
img_paths  = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

# For each image, turn it into an h5 file
# print("image Paths: ",img_paths,'\n')
for  img_path  in img_paths:
    print(img_path)

    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path) # Read image
    k = np.zeros((img.shape[0],img.shape[1])) # Create matrix with same size as image
    gt = mat["image_info"][0,0][0,0][0]

    # Set values in matrix
    for i in range(0,len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])] = 1

    k = gaussian_filter(k,15) # Filter the matrix

    # Create the h5 file
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'), 'w') as hf:
        hf['density'] = k
