import tensorflow as tf
#import keras
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
# from tf import keras.layers.BatchNormalization as BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.metrics import mean_squared_error
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D, UpSampling2D, Subtract, concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

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
def contextaware():
    #Variable Input Size
    rows = None
    cols = None

    #Batch Normalisation option

    batch_norm = 0
    kernel = (3, 3)
    init = RandomNormal(stddev=0.01)
    # model = Sequential()

    # # Custom VGG front end
    input = Input(shape = (rows,cols,3))
    x1 = Conv2D(64, kernel_size = kernel, activation = 'relu', padding='same')(input)
    x2 = Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(x1)
    x3 = MaxPooling2D(strides=2)(x2)
    x4 = Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same')(x3)
    x5 = Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same')(x4)
    x6 = MaxPooling2D(strides=2)(x5)
    x7 = Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(x6)
    x8 = Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(x7)
    x9 = Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(x8)
    x10 = MaxPooling2D(strides=2)(x9)
    x11 = Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')(x10)
    x12 = Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')(x11)
    last= Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')(x12)

    #context aware stuff
    pool2 = AveragePooling2D(pool_size = (2,2), padding = 'same')(last)
    pool3 = AveragePooling2D(pool_size = (4,4), padding = 'same')(last)
    pool4 = AveragePooling2D(pool_size = (8,8), padding = 'same')(last)

    #first convolution
    conv1 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(last)
    conv2 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(pool2)
    conv3 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(pool3)
    conv4 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(pool4)
    
    #conv1 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same')(last)
    #conv2 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same')(pool2)
    #conv3 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same')(pool3)
    #conv4 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same')(pool4)

    #Upsampling
    up2 = UpSampling2D(size = (2,2))(conv2)
    up3 = UpSampling2D(size = (4,4))(conv3)
    up4 = UpSampling2D(size = (8,8))(conv4)

    #subtraction layer
    sub1 = Subtract()([conv1, last])
    sub2 = Subtract()([up2, last])
    sub3 = Subtract()([up3, last])
    sub4 = Subtract()([up4, last])
    
    #second conv2
    conv12 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(sub1)
    conv22 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(sub2)
    conv32 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(sub3)
    conv42 = Conv2D(512, kernel_size = (1,1), activation = 'relu', padding='same')(sub4)

    #conv32 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same')(sub3)
    #conv42 = Conv2D(1, kernel_size = (1,1), activation = 'relu', padding='same')(sub4)





    #multiplication
    mult1 = Multiply()([conv12,conv1])
    mult2 = Multiply()([conv22,up2])
    mult3 = Multiply()([conv32,up3])
    mult4 = Multiply()([conv42,up4])

    # conc = concatenate()([mult1, mult2, mult3, mult4, last])
    #or
    conc = concatenate(inputs = [mult1, mult2, mult3, mult4, last])


    #Back end network
    b1 = Conv2D(512, dilation_rate = 2, kernel_size = kernel, activation = 'relu', padding='same')(conc)#formerly conc
    b2 = Conv2D(512, dilation_rate = 2, kernel_size = kernel, activation = 'relu', padding='same')(b1)
    b3 = Conv2D(512, dilation_rate = 2, kernel_size = kernel, activation = 'relu', padding='same')(b2)
    b4 = Conv2D(256, dilation_rate = 2, kernel_size = kernel, activation = 'relu', padding='same')(b3)
    b5 = Conv2D(128, dilation_rate = 2, kernel_size = kernel, activation = 'relu', padding='same')(b4)
    b6 = Conv2D(64, dilation_rate = 2, kernel_size = kernel, activation = 'relu', padding='same')(b5)
    out = Conv2D(1, dilation_rate = 1, kernel_size = (1, 1), activation = 'relu', padding='same')(b6)

    #adam_optimizer = Adam(lr=1e-5)
    #sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
    model = Model(input, out)
    #opt = tf.keras.optimizers.Adam(lr = 0.0001)
    opt = tf.keras.optimizers.Adam(lr = 0.00001)
    model.compile(optimizer = opt, loss="MSE", metrics=['mse'])

    # model = init_weights_vgg(model, model_dir)

    return model
def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
