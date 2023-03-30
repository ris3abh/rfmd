# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# %%
df_eval = pd.read_csv('../data/Evaluation_Set/RFMiD_Validation_Labels.csv')

# %%
df_train = pd.read_csv('../data/Training_Set/RFMiD_Training_Labels.csv')

# %%
df_test = pd.read_csv('../data/Test_Set/RFMiD_Testing_Labels.csv')

# %%
df_train.head()

# %% [markdown]
# Creating a different df to store disease risk column for the data

# %%
df_disease_risk_train = df_train['Disease_Risk']
df_disease_risk_eval = df_eval['Disease_Risk']
df_disease_risk_test = df_test['Disease_Risk']

# %%
df_train.drop(['Disease_Risk'], axis=1, inplace=True)
df_eval.drop(['Disease_Risk'], axis=1, inplace=True)
df_test.drop(['Disease_Risk'], axis=1, inplace=True)

# %%
df_train.head()

# %%
df_eval.head()

# %%
df_test.head()

# %%
## making ID column as index
df_train.set_index('ID', inplace=True)
df_eval.set_index('ID', inplace=True)
df_test.set_index('ID', inplace=True)

# %%
df_train = df_train[['DR', 'MH', 'TSLN', 'ODC']]
df_eval = df_eval[['DR', 'MH', 'TSLN', 'ODC']]
df_test = df_test[['DR', 'MH', 'TSLN', 'ODC']]
print("shape of train data: ", df_train.shape)
print("shape of eval data: ", df_eval.shape)
print("shape of test data: ", df_test.shape)

# %%
## removing the rows which have no diseases
df_train = df_train[df_train.sum(axis=1) != 0]
df_eval = df_eval[df_eval.sum(axis=1) != 0]
df_test = df_test[df_test.sum(axis=1) != 0]

# %%
print("shape of train data: ", df_train.shape)
print("shape of eval data: ", df_eval.shape)
print("shape of eval data: ", df_test.shape)

# %%
df_train.head()

# %%
df_eval.head()

# %%
## adding the disease risk column to the train and eval data with ID as index
df_train['Disease_Risk'] = df_disease_risk_train
df_eval['Disease_Risk'] = df_disease_risk_eval
df_test['Disease_Risk'] = df_disease_risk_test

# %%
df_train.head()

# %% [markdown]
# Im encode the image data,
# Subsampling the data. 

# %%
## opening an image and converting it to numpy array
import cv2
import matplotlib.pyplot as plt

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
    
def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , 30) ,-4 ,128)
    return img

img = cv2.imread('../data/Training_Set/Training/103.png')
img = circle_crop(img)
img = crop_image_from_gray(img)
img = cv2.resize(img, (64, 64))
plt.imshow(img)

# %%
img
print(img.shape)

# %%
## size of the image in mb after preprocessing
import sys
print(sys.getsizeof(img)/1000000)

# %%
import os
import cv2
import numpy as np

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = circle_crop(img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (64, 64))
    plt.imshow(img)
    return img

# %%
import os
## preprocessing the train images
output_path = '../data/Training_Set/preprocessed_numpy_64'
input_path = '../data/Training_Set/Training'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(os.path.abspath(''))))
for file in os.listdir(input_path):
    if file.endswith('.png'):
        img = preprocess_image(os.path.join(input_path, file))  
        np.save(os.path.join(__location__ + "\data\Training_Set\preprocessed_numpy_64", file.split('.')[0]), img)

# %%
## preprocessing the valdaition images
output_path = '../data/Evaluation_Set/preprocessed_eval_64'
input_path = '../data/Evaluation_Set/Validation'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(os.path.abspath(''))))
for file in os.listdir(input_path):
    if file.endswith('.png'):
        img = preprocess_image(os.path.join(input_path, file))
        np.save(os.path.join(__location__ + "\data\Evaluation_Set\preprocessed_eval_64", file.split('.')[0]), img)

# %%
## preprocessing the valdaition images
output_path = '../data/Test_Set/preprocessed_test_64'
input_path = '../data/Test_Set/Test'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(os.path.abspath(''))))
for file in os.listdir(input_path):
    if file.endswith('.png'):
        img = preprocess_image(os.path.join(input_path, file))
        np.save(os.path.join(__location__ + "\data\Test_Set\preprocessed_test_64", file.split('.')[0]), img)

# %%
file_loc = '../data/Training_Set/preprocessed_numpy_64/'

array = []
for i in df_train.index:
    array.append(str(i)+'.npy')

## only keeping the images in the folder which are in the index of df_train
for file in os.listdir(file_loc):
    if file not in array:
        os.remove(file_loc+file)

# %%
## checking if the number of images in the folder is same as the number of rows in the dataframe
print(len(os.listdir(file_loc)))

# %%
df_train.shape

# %%
file_loc = '../data/Evaluation_Set/preprocessed_eval_64/'

array = []
for i in df_eval.index:
    array.append(str(i)+'.npy')

## only keeping the images in the folder which are in the index of df_train
for file in os.listdir(file_loc):
    if file not in array:
        os.remove(file_loc+file)

# %%
print(len(os.listdir(file_loc)))

# %%
file_loc = '../data/Test_Set/preprocessed_test_64/'

array = []
for i in df_test.index:
    array.append(str(i)+'.npy')

## only keeping the images in the folder which are in the index of df_train
for file in os.listdir(file_loc):
    if file not in array:
        os.remove(file_loc+file)
print(len(os.listdir(file_loc)))

# %%
df_test.shape


