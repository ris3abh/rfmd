from src.layers import *

## implementing multi-layer perceptron model for classification

# load data
import src.util as util
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def oneencoding(Yhat):
    Y_unique, Y_inverse = np.unique(Yhat, return_inverse=True)
    return np.eye(Y_unique.shape[0])[Y_inverse]

def load_rfmd_npy(file_loc):
    X = []
    for file in os.listdir(file_loc):
            img = np.load(file_loc+file)
            X.append(np.transpose(img))
    X = np.array(X)
    return X

row_size = 100
data_location = '/Users/rishabhsharma/Documents/GitHub/rfmd'
X_train_img = load_rfmd_npy( data_location + '/data/Training_Set/preprocessed_numpy/')
X_train_img = X_train_img[0:row_size]

X_test_img = load_rfmd_npy(data_location + '/data/Evaluation_Set/preprocessed_eval/') #taking eval as a test right now
X_test_img = X_test_img[0:row_size]

Y_train = pd.read_csv( data_location + '/data/Training_Set/RFMiD_Training_Labels.csv')
Y_train = np.array(Y_train[['DR', 'MH', 'TSLN', 'ODC']]) #currently doing to for disease_risk - only binary classification
#Y_train = np.array(Y_train['Disease_Risk'])
Y_train = np.array(Y_train[0:row_size])
#Y_train = util.one_hot_array(Y_train, 4)

Y_test = pd.read_csv(data_location + '/data/Evaluation_Set/RFMiD_Validation_Labels.csv')
Y_test = np.array(Y_test[['DR', 'MH', 'TSLN', 'ODC']])#currently doing to for disease_risk - only binary classification
#Y_test = np.array(Y_test['Disease_Risk'])
Y_test = np.array(Y_test[0:row_size])

print("shape of X_train_img: ", X_train_img.shape)
print("shape of Y_train: ", Y_train.shape)
print("shape of X_test_img: ", X_test_img.shape)
print("shape of Y_test: ", Y_test.shape)


X_train_img = X_train_img.reshape(X_train_img.shape[0], X_train_img.shape[1]*X_train_img.shape[2]*X_train_img.shape[3])
X_test_img = X_test_img.reshape(X_test_img.shape[0], X_test_img.shape[1]*X_test_img.shape[2]*X_test_img.shape[3])

## creating layers for the MLP model
L1 = InputLayer(X_train_img)
L2 = FullyConnectedLayer(X_train_img.shape[1], 512)
L3 = ReluLayer()
Ld0 = DropoutLayer(0.5)
L4 = FullyConnectedLayer(512, 256)
L5 = ReluLayer()
Ld1 = DropoutLayer(0.5)
L6 = FullyConnectedLayer(256, 128)
L7 = ReluLayer()
Ld2 = DropoutLayer(0.5)
L8 = FullyConnectedLayer(128, 64)
L9 = ReluLayer()
Ld3 = DropoutLayer(0.5)
L10 = FullyConnectedLayer(64, 4)
L11 = LogisticSigmoidLayer()
L12 = BinaryCrossEntropy()

Ld = [L1, L2, L3, Ld0, L4, L5, Ld1, L6, L7, Ld2, L8, L9, Ld3, L10, L11, L12]
L = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]

util.train_model(Ld, L, X_train_img, Y_train, X_test_img, Y_test, learning_rate=0.001, max_epochs=30, batch_size=25, condition=10e-10, skip_first_layer=True)

