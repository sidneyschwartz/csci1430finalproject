from tensorflow.keras.models import model_from_json
import os
import cv2
import glob
import h5py
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.io as io
from PIL import Image
import numpy as np
import tensorflow as tf



def load_model():
    # Function to load and return neural network model
    # json_file = open(model_dir + 'model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # loaded_model.load_weights(model_dir + "weights/model_A_weights.h5")
    loaded_model = models.load_model("model/MAE140.h5")
    return loaded_model

def create_img(path):
    im = Image.open(path).convert('RGB')

    im = np.array(im)

    im = im / 255.0

    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    
    if im.shape[0] > im.shape[1]:
        im = cv2.copyMakeBorder(im, 0, 0, 0, im.shape[0]-im.shape[1], cv2.BORDER_CONSTANT, value = [0,0,0])
    elif im.shape[1] > im.shape[0]:
        im = cv2.copyMakeBorder(im, 0, im.shape[1] - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
    im = cv2.resize(im, (512,512))
    im = np.expand_dims(im, axis=0)
    return im


root = 'data/ShanghaiTech'
part_A_train = os.path.join(root, 'part_A/train_data', 'images')
part_A_test = os.path.join(root, 'part_A/test_data', 'images')
part_B_train = os.path.join(root, 'part_B/train_data', 'images')
part_B_test = os.path.join(root, 'part_B/test_data', 'images')
path_sets = [part_B_test]
img_paths = []
# print("PATH: ",path_sets[0])
# print("PRINTING\n")

for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

#model = load_model()
model = tf.keras.models.load_model("./model/MAE152.h5")
name = []
y_true = []
y_pred = []
count = 0
print(len(img_paths))
for image in img_paths:
    print("Num Images: ", count)
    count = count+1
    name.append(image)
    gt = h5py.File(image.replace('.jpg', '.h5').replace('images', 'ground-truth'),'r')
    groundtruth = np.asarray(gt['density'])
    num1 = np.sum(groundtruth)
    y_true.append(np.sum(num1))
    img = create_img(image)
    num = np.sum(model.predict(tf.convert_to_tensor(img)))
    y_pred.append(np.sum(num))
    #print(counter, "/", num_imgs)
    #counter += 1
    if (count == 100):
        break

data = pd.DataFrame({'name': name, 'y_pred': y_pred, 'y_true': y_true})
data.to_csv('CSV/A_on_B_test.csv', sep=',')
data = pd.read_csv('CSV/A_on_B_test.csv')
y_true = data['y_true']
y_pred = data['y_pred']

ans = mean_absolute_error(np.array(y_true), np.array(y_pred))
rmse = mean_squared_error(np.array(y_true), np.array(y_pred), squared=True)
print("MAE : ", ans)
print("RMSE : ", rmse)
#data = pd.read_csv('CSV/B_on_B_test.csv', sep='\t')
#y_true = data['y_true']
#y_pred = data['y_pred']
#ans = mean_absolute_error(np.array(y_true), np.array(y_pred))
#print("MAE : ", ans)
