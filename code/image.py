from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.metrics import mean_squared_error
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from matplotlib import cm as CM
import matplotlib.pyplot as plt
import tensorflow as tf


from tqdm import tqdm
import scipy.io as io
from PIL import Image
import PIL
import h5py
import os
import glob
import cv2
import random
import math
import numpy as np


def create_img(path):
    #Function to load,normalize and return image
    #print(path)
    im = Image.open(path).convert('RGB')

    im = np.array(im)

    im = im/255.0

    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225

    #print(im.shape)
    #im = np.expand_dims(im,axis  = 0)
    return im

def get_input(path):
    img = create_img(path)
    return(img)

def get_output(path):

    gt_file = h5py.File(path,'r')
    
    target = np.asarray(gt_file['density'])
    #print(target.shape)
    
    #print(img.shape)
    img = np.expand_dims(target, axis = 2) # makes every cell in a 2D array its own matrix.

    #print(img.shape)

    return img

def preprocess_input(image,target):
    #crop image
    #crop target
    #resize target
    crop_size = (int(image.shape[0]/2),int(image.shape[1]/2))


    if random.randint(0,9)<= -1:
            dx = int(random.randint(0,1)*image.shape[0]*1./2)
            dy = int(random.randint(0,1)*image.shape[1]*1./2)
    else:
            dx = int(random.random()*image.shape[0]*1./2)
            dy = int(random.random()*image.shape[1]*1./2)

    #print(crop_size , dx , dy)
    img = image[dx : crop_size[0] + dx , dy:crop_size[1] + dy]

    target_aug = target[dx:crop_size[0] + dx,dy:crop_size[1] + dy]
    #print(img.shape)

    return(img,target_aug)


#Image data generator
def image_generator(files, batch_size = 64, resize_dim = 1024, grayscale = 0):
    while True:
        batch_paths = np.random.choice(a = files, size = batch_size)
        batch_input = []
        batch_output = []
        
        max_x, max_y = 0, 0
        
        for input_path in batch_paths:
            inputt = get_input(input_path)
            output = get_output(input_path.replace('.jpg','.h5').replace('images','ground-truth'))# .replace('IMG','GT_IMG') )

            # print(inputt.shape, output.shape)
            max_y = max_y if inputt.shape[0] <= max_y else inputt.shape[0]
            max_x = max_x if inputt.shape[1] <= max_x else inputt.shape[1]
            batch_input += [inputt]
            batch_output += [output]
        #Austen's edit to avoid weird upsampling errors!
        max_y = 1024
        max_x = 1024
            
        for index, input_image in enumerate(batch_input):
            #pls forgive me scott, I have commented out thy bidding to do my own :(
            img = np.array(input_image)
            gt = np.array(batch_output[index])
            #print("Image: ", img.shape, gt.shape, resize_dim - img.shape[0], resize_dim - img.shape[1])
            assert(img.shape[0] == gt.shape[0])
            assert(img.shape[1] == gt.shape[1])
            
            # Crop the images to the resize dimensions. If the images are smaller than the dimensions, pad them up to this size.
            crop_y, crop_x = 0, 0
            
            if img.shape[0] < resize_dim:
                img = cv2.copyMakeBorder(img, 0, resize_dim - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                gt = cv2.copyMakeBorder(gt, 0, resize_dim - gt.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
            else:
                crop_y = random.randint(0, img.shape[0] - resize_dim)
            
            if img.shape[1] < resize_dim:
                img = cv2.copyMakeBorder(img, 0, 0, 0, resize_dim - img.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])
                gt = cv2.copyMakeBorder(gt, 0, 0, 0, resize_dim - gt.shape[1], cv2.BORDER_CONSTANT, value=0)
            else:
                crop_x = random.randint(0, img.shape[1] - resize_dim)
            
            
            img = img[crop_y:crop_y + resize_dim, crop_x:crop_x + resize_dim,:]
            gt = gt[crop_y:crop_y + resize_dim, crop_x:crop_x + resize_dim]
            
            # img = cv2.resize(img,(resize_dim, resize_dim))
            # gt = cv2.resize(gt, (resize_dim, resize_dim))
            #print(img.shape, gt.shape, crop_y, crop_x)
            assert(img.shape[0] == gt.shape[0])
            gt = cv2.resize(gt, (gt.shape[1]//8, gt.shape[0]//8), interpolation = cv2.INTER_CUBIC)*64
            batch_input[index] = img
            batch_output[index] = gt
        
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        print("Batch sizes: ", batch_x.shape, batch_y.shape)
        
        yield(batch_x, batch_y)
