import tensorflow as tf
import tensorflow.keras as keras
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
# from tf import keras.layers.BatchNormalization as BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.metrics import mean_squared_error
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

def austensmodel():
    #Variable Input Size
    rows = None
    cols = None

    kernel = (3, 3)
    init = RandomNormal(stddev=0.01)
    model = Sequential()

    # Custom VGG:

    model.add(Conv2D(1, kernel_size = kernel,activation = 'relu', padding='same',input_shape = (rows, cols, 3), kernel_initializer = init))
    model.add(Conv2D(1, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(1,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
    model.add(Conv2D(1,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(1,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
    model.add(Conv2D(1,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
    model.add(Conv2D(1,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(1, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
    model.add(Conv2D(1, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
    model.add(Conv2D(1, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
    #

    #Conv2D
    model.add(Conv2D(1, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate = 1, kernel_initializer = init, padding = 'same'))

    sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
    model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse'])

    # model = init_weights_vgg(model, model_dir)

    return model
def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
