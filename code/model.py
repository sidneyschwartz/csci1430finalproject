import glob
import math
import os
import random
import sys
from time import ctime, time
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import PIL
import scipy.io as io
import tensorflow as tf
import tensorflow.keras as keras
from austens_model import *
from austens_model2 import *
from image import *
from matplotlib import cm as CM
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import argparse



def main():
    print("Starting train of ASS model...")

    # Arguments parsing: (put this in later)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Not implemented but should be able to choose from A or B')
    parser.add_argument('--data_subset',type=int, help='Subset size of data to use')
    parser.add_argument('--validate_size', type=int, help='Size of dataset we use to validate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', help='Number of epochs (times we go through the data)')
    parser.add_argument('--resize_size', type=int, help = 'Dimension (square) to resize image with')
    parser.add_argument('--grayscale', type=int, help = '1 for converting to grayscale, 0 for normal')
    args = parser.parse_args()


    dataset = args.dataset if args.dataset else 'A'
    data_subset = int(args.data_subset) if args.data_subset else -1
    test_size = int(args.validate_size) if args.validate_size else 16 # Number of images we test on
    batch_size = int(args.batch_size) if args.batch_size else 16 # sections of data we use
    epochs = int(args.epochs) if args.epochs else 4 # default epochs
    resize_size = int(args.resize_size) if args.resize_size else 1024
    grayscale = 1 if args.grayscale else 0
    # Directories
    root = 'data/ShanghaiTech/'
    model_dir = 'model/'
    weights_dir = 'model/weights/'
    part_A_train = os.path.join(root,'part_A/train_data','images') # need to generate h5 for Part A
    part_A_test = os.path.join(root,'part_A/test_data','images')
    part_B_train = os.path.join(root,'part_B/train_data','images') # Already generated
    part_B_test = os.path.join(root,'part_B/test_data','images')
    temp = 'test_images'
    train_sets = [part_A_train]
    test_sets = [part_A_test]

    img_paths = []
    for path in train_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(str(img_path))
    print("Number of Images found: ", len(img_paths),"\n")
    # Get the images for evaluation
    test_paths = []
    for path in test_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            test_paths.append(str(img_path))

    test_paths = test_paths[0:data_subset]
    print("Number of test images found: ", len(test_paths))
    test_gen = image_generator(test_paths, test_size, resize_size, grayscale)

    img_paths = img_paths[0:data_subset] # Takes only a portion of the data
    len_images = len(img_paths)
    print("Total images: ", len_images)

    model = contextaware() # Model being used

    # Training Parameters
    print(model.summary())
    net = 'ASSNet' # Naming
    time_start = time()
    #epochs = 2
    #batch_size = 64
    validate_rate = 0.5 # Evaluate the model twice every epoch
    losses = [[1e5, 1e5, 1e5, 1e5]]
    best_values = {'mae': 1e5, 'rmse': 1e5, 'sfn': 1e5, 'mape': 1e5} # Best loss values
    val_rate_dec = {'A': [80, 70], 'B': [9, 8.5]}
    num_iter = int((len_images - 0.1) // batch_size + 1) # Avoid overflow
    test_x, test_y = next(test_gen)
    
    train_gen = image_generator(img_paths, batch_size, resize_size, grayscale)


    print("Beginning training:")
    #losses file write
    returnlosses = []
    # Training iterations
    for epoch in range(epochs):
        print("Epoch: ", epoch,'\n')
        for data_part in range(0, len_images, batch_size):
            x, y = next(train_gen) # Get the next images

            model.fit(x, y, batch_size, validation_data = (test_x, test_y), verbose=1)

            idx_val = (data_part / batch_size + 1)
            # Eval losses and save models
            if idx_val % (num_iter * validate_rate) == 0:
                print("Evaluating..\n")
                loss = eval_loss(model, test_x, test_y, quality=False) # get the loss
                if loss[0] < val_rate_dec[dataset][0]:
                    validate_rate = min(validate_rate, 0.25)
                if loss[0] < val_rate_dec[dataset][1]:
                    validate_rate = min(validate_rate, 0.1)
                losses.append(loss)
                if (loss[0] < best_values['mae']) or (loss[0] == best_values['mae'] and loss[1] < best_values['rmse']):
                    model.save_weights(os.path.join(weights_dir, '{}_best.hdf5'.format(net)))
                to_save = False
                for idx_best in range(len(loss)):
                    if loss[idx_best] < best_values[list(best_values.keys())[idx_best]]:
                        best_values[list(best_values.keys())[idx_best]] = loss[idx_best]
                        to_save = True
                if to_save:
                    path_save = os.path.join(weights_dir, ''.join([
                        net,
                        '_MAE', str(round(loss[0], 3)), '_RMSE', str(round(loss[1], 3)),
                        '_SFN', str(round(loss[2], 3)), '_MAPE', str(round(loss[3], 3)),
                        '_epoch', str(epoch + 1), '-', str(idx_val), '.hdf5'
                    ]))
                    # model.save_weights(path_save)
                    to_save = False
                returnlosses.append([epoch, data_part, loss[0]])

        # Progress panel
        time_consuming = time() - time_start
        print('In epoch {}, with MAE-RMSE-SFN-MAPE={}, time consuming={}m-{}s\r'.format(
           epoch, np.round(np.array(losses)[-1, :], 2),
           int(time_consuming/60), int(time_consuming-int(time_consuming/60)*60)
        ))
        if epoch % 10 == 0:
            tf.keras.models.save_model(model, model_dir+"modelbackup.h5")
            print("temporary model backup saved")

    print("FINISHED TRAINING")
    #save_mod(model, model_dir + "weights/model_A_weights.h5", model_dir + "/model.json")
    #store_dir = "bestmodel.h5"
    tf.keras.models.save_model(model, model_dir + "bestmodel.h5")
    print("Saved model to ", model_dir+"bestmodel.h5")
    lossesarray = np.array(returnlosses)
    np.savetxt('losses.txt', lossesarray, delimiter = ',')
    print("CHRISTIAN YOU CAN EXIT THE THING NOW")


    #def save_mod(model , str1):# , str2):
    #austen edit to try to get model to save differently
    #model.save_weights(str1)

    #model_json = model.to_json()

    #with open(str2, "w") as json_file:
    #    json_file.write(model_json)

def init_weights_vgg(model, model_dir):
    #vgg =  VGG16(weights='imagenet', include_top=False)
    # print(model_dir)
    # EDIT BY AUSTEN--the following block is replaced by...
    #     json_file = open(model_dir + 'VGG_16.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     loaded_model = model_from_json(loaded_model_json)
    #     loaded_model.load_weights(model_dir + "weights/VGG_16.h5")
    #
    # vgg = loaded_model

    # THIS!
    vgg =  VGG16(weights='imagenet', include_top=False)


    vgg_weights=[]
    for layer in vgg.layers:
        if('conv' in layer.name):
            vgg_weights.append(layer.get_weights())

    #AUSTEN EDIT TO FREEZE VGG
    # vgg.trainable = False

    offset = 0
    i = 0
    while(i < 10):
        if('conv' in model.layers[i + offset].name):
            model.layers[i + offset].set_weights(vgg_weights[i])
            i = i + 1
        else:
            offset=offset+1
        # # Following line is an Austen Edit
        # model.layers[i+offset].trainable = False

    return (model)

def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# Neural network model : VGG + Conv
def austensmodel2(model_dir):
    #Variable Input Size
    rows = None
    cols = None

    #Batch Normalisation option

    batch_norm = 0
    kernel = (3, 3)
    init = RandomNormal(stddev=0.01)
    model = Sequential()

    # Custom VGG:
    if(batch_norm):
        model.add(Conv2D(64, kernel_size = kernel, input_shape = (rows,cols,3),activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        # model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        # model.add(BatchNormalization())

    else:
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same',input_shape = (rows, cols, 3), kernel_initializer = init))
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        # model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))

    last = Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')
    model.add(last)
    # pool1 = tf.keras.layers.AveragePooling2D(pool_size=(1,1), padding='same')(last)
    pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(last)
    pool3 = tf.keras.layers.AveragePooling2D(pool_size=(3,3), padding='same')(last)
    pool4 = tf.keras.layers.AveragePooling2D(pool_size=(6,6), padding='same')(last)
    # conv1 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(pool1)
    conv1 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(last)
    conv2 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(pool2)
    conv3 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(pool3)
    conv4 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(pool4)

    up2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(conv2)
    up3 = tf.keras.layers.UpSampling2D(size=(3, 3), interpolation='nearest')(conv3)
    up4 = tf.keras.layers.UpSampling2D(size=(6, 6), interpolation='nearest')(conv4)

    sub1 = keras.layers.Subtract()([last, conv1])
    sub2 = keras.layers.Subtract()([last, up2])
    sub3 = keras.layers.Subtract()([last, up3])
    sub4 = keras.layers.Subtract()([last, up4])

    second_conv1 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(sub1)
    second_conv2 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(sub2)
    second_conv3 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(sub3)
    second_conv4 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same', kernel_initializer = init)(sub4)

    mult1 = keras.layers.Multiply()([conv1, second_conv1])
    mult2 = keras.layers.Multiply()([up2,   second_conv2])
    mult3 = keras.layers.Multiply()([up3,   second_conv3])
    mult4 = keras.layers.Multiply()([up4,   second_conv4])

    weighted = keras.layers.Add()([mult1, mult2, mult3, mult4])
    conc = tf.keras.layers.concatenate(inputs = [weighted, last])

    #Conv2D
    backend = Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same')(conc)
    model.add(backend)
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate = 1, kernel_initializer = init, padding = 'same'))

    adam_optimizer = Adam(lr=1e-5)
    sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
    model.compile(optimizer=adam_optimizer, loss="MSE", metrics=['mse'])

    model = init_weights_vgg(model, model_dir)

    return model

def eval_loss(model, x, y, quality=False):
    preds, DM, GT = [], [], []
    losses_SFN, losses_MAE, losses_MAPE, losses_RMSE = [], [], [], []
    for idx_pd in range(x.shape[0]):
        pred = model.predict(np.array([x[idx_pd]]))
        preds.append(np.squeeze(pred))
        DM.append(np.squeeze(np.array([y[idx_pd]])))
        GT.append(round(np.sum(np.array([y[idx_pd]]))))    # To make sure the GT is an integral value
    print(len(preds), len(DM), len(GT))
    for idx_pd in range(len(preds)):
        losses_SFN.append(np.mean(np.square(preds[idx_pd] - DM[idx_pd])))     # mean of Frobenius norm
        losses_MAE.append(np.abs(np.sum(preds[idx_pd]) - GT[idx_pd]))
        losses_MAPE.append(np.abs(np.sum(preds[idx_pd]) - GT[idx_pd]) / GT[idx_pd])
        losses_RMSE.append(np.square(np.sum(preds[idx_pd]) - GT[idx_pd]))

    loss_SFN = np.sum(losses_SFN)
    loss_MAE = np.mean(losses_MAE)
    loss_MAPE = np.mean(losses_MAPE)
    loss_RMSE = np.sqrt(np.mean(losses_RMSE))
    if quality:
        psnr, ssim = [], []
        for idx_pd in range(len(preds)):
            data_range = max([np.max(preds[idx_pd]), np.max(DM[idx_pd])]) - min([np.min(preds[idx_pd]), np.min(DM[idx_pd])])
            psnr_ = compare_psnr(preds[idx_pd], DM[idx_pd], data_range=data_range)
            ssim_ = compare_ssim(preds[idx_pd], DM[idx_pd], data_range=data_range)
            psnr.append(psnr_)
            ssim.append(ssim_)
        return loss_MAE, loss_RMSE, loss_SFN, loss_MAPE, np.mean(psnr), np.mean(ssim)
    return loss_MAE, loss_RMSE, loss_SFN, loss_MAPE

if __name__ == "__main__":
    main()
