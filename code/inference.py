import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from tensorflow.keras.models import model_from_json
from tensorflow.keras import models
import tensorflow as tf

model_dir = 'model/'
root_data = 'data/ShanghaiTech/'
image_path = root_data + 'part_A/train_data/images/IMG_28.jpg'
#image_path = 'test_cases/trump.jpg'
has_gt = True

def load_model():
    # Function to load and return neural network model
    json_file = open(model_dir + 'original.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dir + "weights/original_B_weights.h5")
    #loaded_model = models.load_model("model/newmodel.h5")
    return loaded_model


def create_img(path):
    # Function to load,normalize and return image
    print("Getting image: ", path)
    im = Image.open(path).convert('RGB')

    im = np.array(im)

    im = im / 255.0

    # Normalize
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    #im = np.expand_dims(im, axis=0)
    return im


def predict(path):
    # Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = load_model()
    # model = tf.keras.models.load_model("model/MAE152.h5")
    image = create_img(path)
    print(image.shape)
    #image = np.array([cv2.resize(image[0], dsize=(512,512), interpolation=cv2.INTER_CUBIC)])#image.shape[1]//2, image.shape[2]//2
    # if image.shape[0]>image.shape[1]:
    #     #more rows than columns, add columns
    #     image = cv2.copyMakeBorder(image, 0, 0, 0, image.shape[0]-image.shape[1], cv2.BORDER_CONSTANT, value = [0,0,0])
    # elif image.shape[1] > image.shape[0]:
    #     image = cv2.copyMakeBorder(image, 0, image.shape[1]-image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
    # image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis = 0)
    print(image.shape)
    ans = model.predict(image)
    count = np.sum(ans)
        
    return count, image, ans

def get_ground_truth(path):
    gt_path = path.replace('.jpg','.h5').replace('images','ground-truth') # Get the path to the ground truth
    gt = h5py.File(gt_path, 'r') # Read the h5 file
    gt_arr = np.asarray(gt['density'])
    actual = int(np.sum(gt_arr)) + 1 # Integrate the whole density map to get the number of people
    
    return actual, gt_arr


ans, img, hmap = predict(image_path)
num_figs = 2
if (has_gt):
    num_figs = 3
    actual, gt_img = get_ground_truth(image_path)
    print("Predicted:", ans, " Actual: ", actual)
else:
    print("Predicted:", ans)
    
fig = plt.figure()
fig.add_subplot(1, num_figs, 1)
plt.imshow(img.reshape(img.shape[1], img.shape[2], img.shape[3]), label="Image")
#plt.show()
fig.add_subplot(1, num_figs, 2)
plt.imshow(hmap.reshape(hmap.shape[1], hmap.shape[2]), cmap=c.jet, label="Predicted")
#plt.colorbar()
if(has_gt):
    fig.add_subplot(1, 3, 3)
    plt.imshow(gt_img.reshape(gt_img.shape[0], gt_img.shape[1]), cmap=c.jet, label="Ground Truth")
plt.show()
